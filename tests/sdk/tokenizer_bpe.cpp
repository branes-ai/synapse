// Unit tests for bpe_tokenizer: loading (every rejection rule), encode
// options, and the round trip T2 on random text.

#include <branes/synapse/gguf/mapped_file.hpp>
#include <branes/synapse/gguf/reader.hpp>
#include <branes/synapse/tokenizer/bpe.hpp>

#include "../support/gguf_writer.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace branes::synapse;
using namespace synapse_test::gguf_writer;
using tokenizer::bpe_tokenizer;
using tokenizer::tokenizer_errc;

namespace {

const gguf::gguf_file& synthetic() {
    static const gguf::mapped_file file = [] {
        auto f = gguf::mapped_file::open(std::string(SYNAPSE_TEST_DATA_DIR) + "/bpe_vocab.gguf");
        REQUIRE(f);
        return std::move(*f);
    }();
    static const gguf::gguf_file parsed = [] {
        auto p = gguf::parse(file.bytes());
        REQUIRE(p);
        return std::move(*p);
    }();
    return parsed;
}

const bpe_tokenizer& tok() {
    static const bpe_tokenizer t = [] {
        auto r = bpe_tokenizer::from_gguf(synthetic());
        REQUIRE(r);
        return std::move(*r);
    }();
    return t;
}

kv_fn string_array(std::string_view key, const std::vector<std::string>& items) {
    return [=](writer& w) {
        w.put_string(key).put(static_cast<std::uint32_t>(value_type::array));
        w.put(static_cast<std::uint32_t>(value_type::string)).put<std::uint64_t>(items.size());
        for (const auto& s : items) {
            w.put_string(s);
        }
    };
}

kv_fn int32_array(std::string_view key, const std::vector<std::int32_t>& items) {
    return [=](writer& w) {
        w.put_string(key).put(static_cast<std::uint32_t>(value_type::array));
        w.put(static_cast<std::uint32_t>(value_type::int32)).put<std::uint64_t>(items.size());
        for (auto v : items) {
            w.put(v);
        }
    };
}

// A minimal valid vocabulary: the 256 byte tokens, one merge, a BOS.
struct vocab_spec {
    std::map<std::string, kv_fn> keys;
    std::vector<std::string> tokens;

    vocab_spec() {
        for (std::uint32_t b = 0; b < 256; ++b) {
            std::string s;
            tokenizer::append_utf8(tokenizer::byte_to_unicode[b], s);
            tokens.push_back(s);
        }
        tokens.push_back("ab");
        tokens.push_back("<s>");
        keys["tokenizer.ggml.model"] = kv_string("tokenizer.ggml.model", "gpt2");
        keys["tokenizer.ggml.pre"] = kv_string("tokenizer.ggml.pre", "llama-bpe");
        keys["tokenizer.ggml.merges"] = string_array("tokenizer.ggml.merges", {"a b"});
        keys["tokenizer.ggml.bos_token_id"] = kv("tokenizer.ggml.bos_token_id", value_type::uint32, std::uint32_t{257});
        std::vector<std::int32_t> types(tokens.size(), 1);
        types.back() = 3;
        keys["tokenizer.ggml.token_type"] = int32_array("tokenizer.ggml.token_type", types);
    }

    std::vector<std::byte> image() const {
        std::vector<kv_fn> kvs;
        for (const auto& [k, f] : keys) {
            kvs.push_back(f);
        }
        kvs.push_back(string_array("tokenizer.ggml.tokens", tokens));
        return build(3, kvs, {}, 0);
    }
};

result<bpe_tokenizer, tokenizer::tokenizer_error> load(const vocab_spec& spec, std::vector<std::byte>& keep) {
    keep = spec.image();
    auto f = gguf::parse(keep);
    REQUIRE(f);
    return bpe_tokenizer::from_gguf(*f);
}

void require_error(const vocab_spec& spec, tokenizer_errc code, const std::string& name) {
    std::vector<std::byte> keep;
    const auto t = load(spec, keep);
    REQUIRE_FALSE(t);
    CHECK(t.error().code == code);
    CHECK(t.error().name == name);
}

}  // namespace

TEST_CASE("tokenizer_bpe: a minimal vocabulary loads and merges", "[tokenizer][bpe]") {
    vocab_spec spec;
    std::vector<std::byte> keep;
    auto t = load(spec, keep);
    REQUIRE(t);
    CHECK(t->size() == 258);
    CHECK(t->bos() == 257);
    CHECK(t->adds_bos());  // llama.cpp's default for llama-bpe
    // Byte tokens are listed in byte order, so "c" is token 99.
    CHECK(t->encode("abc") == std::vector<std::uint32_t>{257, 256, 'c'});
    CHECK(t->encode("abc", {.add_bos = false}) == std::vector<std::uint32_t>{256, 'c'});
    CHECK(t->encode("<s>", {.add_bos = false, .parse_special = true}) == std::vector<std::uint32_t>{257});
    CHECK(t->decode(257) == "");
    CHECK(t->decode(257, true) == "<s>");
}

TEST_CASE("tokenizer_bpe: rejects vocabularies it cannot tokenize correctly", "[tokenizer][bpe]") {
    {
        vocab_spec spec;
        spec.keys["tokenizer.ggml.model"] = kv_string("tokenizer.ggml.model", "llama");  // SentencePiece
        require_error(spec, tokenizer_errc::unsupported_model, "tokenizer.ggml.model");
    }
    {
        vocab_spec spec;
        spec.keys.erase("tokenizer.ggml.model");
        require_error(spec, tokenizer_errc::missing_key, "tokenizer.ggml.model");
    }
    {
        vocab_spec spec;  // GPT-2's own pre-tokenizer splits differently
        spec.keys["tokenizer.ggml.pre"] = kv_string("tokenizer.ggml.pre", "gpt-2");
        require_error(spec, tokenizer_errc::unsupported_pre, "tokenizer.ggml.pre");
    }
    {
        vocab_spec spec;
        spec.keys.erase("tokenizer.ggml.merges");
        require_error(spec, tokenizer_errc::missing_key, "tokenizer.ggml.merges");
    }
    {
        vocab_spec spec;  // T1
        spec.keys["tokenizer.ggml.bos_token_id"] =
            kv("tokenizer.ggml.bos_token_id", value_type::uint32, std::uint32_t{258});
        require_error(spec, tokenizer_errc::bad_value, "tokenizer.ggml.bos_token_id");
    }
    {
        vocab_spec spec;  // BOS is required when the vocabulary adds one
        spec.keys.erase("tokenizer.ggml.bos_token_id");
        require_error(spec, tokenizer_errc::missing_key, "tokenizer.ggml.bos_token_id");
    }
    {
        vocab_spec spec;
        spec.keys["tokenizer.ggml.token_type"] = int32_array("tokenizer.ggml.token_type", {1, 1, 1});
        require_error(spec, tokenizer_errc::bad_value, "tokenizer.ggml.token_type");
    }
    {
        vocab_spec spec;  // T2: a merge whose result is no token would drop characters
        spec.keys["tokenizer.ggml.merges"] = string_array("tokenizer.ggml.merges", {"a b", "b c"});
        require_error(spec, tokenizer_errc::bad_value, "tokenizer.ggml.merges");
    }
    {
        vocab_spec spec;  // a merge rule must name two operands
        spec.keys["tokenizer.ggml.merges"] = string_array("tokenizer.ggml.merges", {"ab"});
        require_error(spec, tokenizer_errc::bad_value, "tokenizer.ggml.merges");
    }
    {
        vocab_spec spec;  // T2: a byte with no token cannot be encoded
        spec.tokens[0] = "x!";
        require_error(spec, tokenizer_errc::bad_value, "tokenizer.ggml.tokens");
    }
}

TEST_CASE("tokenizer_bpe: decode(encode(s)) == s for random valid UTF-8 (T2)", "[tokenizer][bpe]") {
    std::mt19937 rng(31);
    // Draw from ASCII, Latin-1, CJK, emoji and whitespace, weighted to ASCII.
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> pools = {
        {0x20, 0x7E}, {0x20, 0x7E}, {0x20, 0x7E}, {0x09, 0x0D}, {0xA0, 0x17F}, {0x4E00, 0x4FFF}, {0x1F300, 0x1F64F}};
    for (int trial = 0; trial < 300; ++trial) {
        std::string s;
        const int n = std::uniform_int_distribution<int>(0, 40)(rng);
        for (int i = 0; i < n; ++i) {
            const auto [lo, hi] = pools[std::uniform_int_distribution<std::size_t>(0, pools.size() - 1)(rng)];
            tokenizer::append_utf8(std::uniform_int_distribution<std::uint32_t>(lo, hi)(rng), s);
        }
        INFO("text: \"" << s << "\"");
        CHECK(tok().decode(tok().encode(s, {.add_bos = false})) == s);
    }
}

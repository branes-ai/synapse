// Unit tests for spm_tokenizer and any_tokenizer: score-ordered merging,
// the space prefix, byte fallback, decoding, every loader rejection rule,
// and the round trip T2 on arbitrary bytes. (Agreement with llama.cpp on
// the real Llama-2 vocabulary is checked by tokenizer_golden.)

#include <branes/synapse/gguf/mapped_file.hpp>
#include <branes/synapse/gguf/reader.hpp>
#include <branes/synapse/tokenizer/spm.hpp>
#include <branes/synapse/tokenizer/tokenizer.hpp>
#include <branes/synapse/tokenizer/unicode.hpp>

#include "../support/gguf_writer.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace branes::synapse;
using namespace synapse_test::gguf_writer;
using tokenizer::spm_tokenizer;
using tokenizer::tokenizer_errc;

namespace {

kv_fn string_array(std::string_view key, const std::vector<std::string>& items) {
    return [=](writer& w) {
        w.put_string(key).put(static_cast<std::uint32_t>(value_type::array));
        w.put(static_cast<std::uint32_t>(value_type::string)).put<std::uint64_t>(items.size());
        for (const auto& s : items) {
            w.put_string(s);
        }
    };
}

template <typename T>
kv_fn scalar_array(std::string_view key, value_type type, const std::vector<T>& items) {
    return [=](writer& w) {
        w.put_string(key).put(static_cast<std::uint32_t>(value_type::array));
        w.put(static_cast<std::uint32_t>(type)).put<std::uint64_t>(items.size());
        for (auto v : items) {
            w.put(v);
        }
    };
}

const std::string sp = "\xe2\x96\x81";  // U+2581

// A minimal SentencePiece vocabulary in Llama-2 layout: <unk>, <s>, </s>,
// the 256 byte tokens, then a few pieces with scores chosen to test merge
// order. "ab" (-1) outscores "bc" (-2), so "abc" must become "ab" + "c" even
// though "bc" also exists; "▁x" and "▁" let the space prefix show.
struct vocab_spec {
    std::map<std::string, kv_fn> keys;
    std::vector<std::string> tokens{"<unk>", "<s>", "</s>"};
    std::vector<float> scores{0, 0, 0};
    std::vector<std::int32_t> types{2, 3, 3};

    vocab_spec() {
        for (unsigned b = 0; b < 256; ++b) {
            char hex[7];
            std::snprintf(hex, sizeof hex, "<0x%02X>", b);
            add(hex, 0, 6);
        }
        add(sp, -0.5f);
        for (const char* c : {"a", "b", "c", "x"}) {
            add(c, -3);
        }
        add("ab", -1);
        add("bc", -2);
        add(sp + "x", -1.5f);
        keys["tokenizer.ggml.model"] = kv_string("tokenizer.ggml.model", "llama");
    }

    void add(const std::string& text, float score, std::int32_t type = 1) {
        tokens.push_back(text);
        scores.push_back(score);
        types.push_back(type);
    }

    std::uint32_t id(const std::string& text) const {
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            if (tokens[i] == text) {
                return static_cast<std::uint32_t>(i);
            }
        }
        return tokenizer::no_token;
    }

    std::vector<std::byte> image() const {
        std::vector<kv_fn> kvs;
        for (const auto& [k, f] : keys) {
            kvs.push_back(f);
        }
        kvs.push_back(string_array("tokenizer.ggml.tokens", tokens));
        kvs.push_back(scalar_array("tokenizer.ggml.scores", value_type::float32, scores));
        kvs.push_back(scalar_array("tokenizer.ggml.token_type", value_type::int32, types));
        return build(3, kvs, {}, 0);
    }
};

result<spm_tokenizer, tokenizer::tokenizer_error> load(const vocab_spec& spec, std::vector<std::byte>& keep) {
    keep = spec.image();
    auto f = gguf::parse(keep);
    REQUIRE(f);
    return spm_tokenizer::from_gguf(*f);
}

void require_error(const vocab_spec& spec, tokenizer_errc code, const std::string& name) {
    std::vector<std::byte> keep;
    const auto t = load(spec, keep);
    REQUIRE_FALSE(t);
    CHECK(t.error().code == code);
    CHECK(t.error().name == name);
}

}  // namespace

TEST_CASE("tokenizer_spm: merges by score, prefixes a space, falls back to bytes", "[tokenizer][spm]") {
    vocab_spec spec;
    std::vector<std::byte> keep;
    auto t = load(spec, keep);
    REQUIRE(t);
    CHECK(t->bos() == 1);  // Llama-2 defaults
    CHECK(t->eos() == 2);
    CHECK(t->adds_bos());
    CHECK(t->adds_space_prefix());
    const auto id = [&](const std::string& s) { return spec.id(s); };
    using V = std::vector<std::uint32_t>;

    // "abc" -> "▁abc": "ab" (score -1) beats "bc" (-2), leaving "▁", "ab", "c".
    CHECK(t->encode("abc", {.add_special = false}) == V{id(sp), id("ab"), id("c")});
    // BOS first by default.
    CHECK(t->encode("abc") == V{1, id(sp), id("ab"), id("c")});
    // "▁x" is a token: the prefixed space merges into it.
    CHECK(t->encode("x", {.add_special = false}) == V{id(sp + "x")});
    // "z" is not in the vocabulary: its byte, 0x7A, is the <0x7A> token.
    CHECK(t->encode("xz", {.add_special = false}) == V{id(sp + "x"), id("<0x7A>")});
    // A multi-byte character outside the vocabulary: one byte token per byte.
    CHECK(t->encode("\xc3\xa9", {.add_special = false}) == V{id(sp), id("<0xC3>"), id("<0xA9>")});
    // A special token resets the prefix: the text after it gets one too.
    CHECK(t->encode("x<s>x", {.add_special = false, .parse_special = true}) == V{id(sp + "x"), 1, id(sp + "x")});
    // Without parse_special, "<s>" is plain text (bytes: '<', 's', '>').
    CHECK(t->encode("<s>", {.add_special = false}).size() == 4);

    // Decoding turns "▁" back into spaces and byte tokens into their bytes.
    CHECK(t->decode(t->encode("abc xz", {.add_special = false})) == " abc xz");
    CHECK(t->decode(1) == "");  // control tokens only with special
    CHECK(t->decode(1, true) == "<s>");
}

TEST_CASE("tokenizer_spm: a malformed byte token decodes without throwing, as llama.cpp", "[tokenizer][spm]") {
    vocab_spec spec;
    spec.add("<0xZZ>", 0, 6);  // typed byte, but not hex: llama.cpp's strtol gives 0
    std::vector<std::byte> keep;
    auto t = load(spec, keep);
    REQUIRE(t);
    CHECK(t->decode(spec.id("<0xZZ>")) == std::string(1, '\0'));
}

TEST_CASE("tokenizer_spm: honours add_space_prefix, add_bos_token and add_eos_token", "[tokenizer][spm]") {
    vocab_spec spec;
    spec.keys["tokenizer.ggml.add_space_prefix"] =
        kv("tokenizer.ggml.add_space_prefix", value_type::boolean, std::uint8_t{0});
    spec.keys["tokenizer.ggml.add_bos_token"] =
        kv("tokenizer.ggml.add_bos_token", value_type::boolean, std::uint8_t{0});
    spec.keys["tokenizer.ggml.add_eos_token"] =
        kv("tokenizer.ggml.add_eos_token", value_type::boolean, std::uint8_t{1});
    std::vector<std::byte> keep;
    auto t = load(spec, keep);
    REQUIRE(t);
    CHECK(t->encode("ab") == std::vector<std::uint32_t>{spec.id("ab"), 2});
}

TEST_CASE("tokenizer_spm: rejects vocabularies it cannot tokenize correctly", "[tokenizer][spm]") {
    {
        vocab_spec spec;
        spec.keys["tokenizer.ggml.model"] = kv_string("tokenizer.ggml.model", "gpt2");
        require_error(spec, tokenizer_errc::unsupported_model, "tokenizer.ggml.model");
    }
    {
        vocab_spec spec;  // scores must match the tokens one for one
        spec.scores.pop_back();
        require_error(spec, tokenizer_errc::bad_value, "tokenizer.ggml.scores");
    }
    {
        vocab_spec spec;  // T2: byte 0x41 has neither <0x41> nor "A"
        spec.tokens[3 + 0x41] = "<not-a-byte>";
        require_error(spec, tokenizer_errc::bad_value, "tokenizer.ggml.tokens");
    }
    {
        vocab_spec spec;  // T1: a BOS the vocabulary does not have, but must add
        spec.keys["tokenizer.ggml.bos_token_id"] =
            kv("tokenizer.ggml.bos_token_id", value_type::uint32, std::uint32_t{1u << 20});
        require_error(spec, tokenizer_errc::bad_value, "tokenizer.ggml.bos_token_id");
    }
    {
        vocab_spec spec;  // T2: <0xXX> tokens typed `normal` would decode as their text
        for (std::size_t b = 0; b < 256; ++b) {
            spec.types[3 + b] = 1;
        }
        require_error(spec, tokenizer_errc::bad_value, "tokenizer.ggml.token_type");
    }
    {
        vocab_spec spec;
        spec.keys["tokenizer.ggml.add_space_prefix"] = kv_string("tokenizer.ggml.add_space_prefix", "yes");
        require_error(spec, tokenizer_errc::bad_value, "tokenizer.ggml.add_space_prefix");
    }
}

TEST_CASE("tokenizer_spm: decode(encode(s)) is s with the space prefix (T2)", "[tokenizer][spm]") {
    auto file = gguf::mapped_file::open(std::string(SYNAPSE_TEST_DATA_DIR) + "/spm_vocab.gguf");
    REQUIRE(file);
    auto parsed = gguf::parse(file->bytes());
    REQUIRE(parsed);
    auto t = spm_tokenizer::from_gguf(*parsed);
    REQUIRE(t);
    // Random non-empty valid UTF-8 from ASCII, Latin-1, CJK, emoji (much of
    // it outside the vocabulary, so byte fallback is exercised) and
    // whitespace, never U+2581.
    std::mt19937 rng(41);
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> pools = {
        {0x20, 0x7E}, {0x20, 0x7E}, {0x09, 0x0D}, {0xA0, 0x17F}, {0x4E00, 0x9FFF}, {0x1F300, 0x1FAFF}};
    for (int trial = 0; trial < 400; ++trial) {
        std::string s;
        const int n = std::uniform_int_distribution<int>(1, 40)(rng);
        for (int i = 0; i < n; ++i) {
            const auto [lo, hi] = pools[std::uniform_int_distribution<std::size_t>(0, pools.size() - 1)(rng)];
            tokenizer::append_utf8(std::uniform_int_distribution<std::uint32_t>(lo, hi)(rng), s);
        }
        INFO("text: \"" << s << "\"");
        CHECK(t->decode(t->encode(s, {.add_special = false})) == " " + s);
    }
    // Where T2 stops, as in llama.cpp. An empty input encodes to nothing (no
    // prefix); an input U+2581 is indistinguishable from an escaped space.
    CHECK(t->encode("", {.add_special = false}).empty());
    CHECK(t->decode(t->encode("a\xe2\x96\x81"
                              "b",
                              {.add_special = false})) == " a b");
}

TEST_CASE("tokenizer_spm: any_tokenizer picks the algorithm from tokenizer.ggml.model", "[tokenizer]") {
    for (const auto& [name, kind] : {std::pair{"spm_vocab.gguf", "spm"}, std::pair{"bpe_vocab.gguf", "bpe"}}) {
        auto file = gguf::mapped_file::open(std::string(SYNAPSE_TEST_DATA_DIR) + "/" + name);
        REQUIRE(file);
        auto parsed = gguf::parse(file->bytes());
        REQUIRE(parsed);
        auto t = tokenizer::any_tokenizer::from_gguf(*parsed);
        REQUIRE(t);
        CHECK(t->kind() == kind);
        CHECK(t->decode(t->encode("Hello world", {.add_special = false})).ends_with("Hello world"));
    }
    vocab_spec spec;
    spec.keys["tokenizer.ggml.model"] = kv_string("tokenizer.ggml.model", "t5");
    const auto img = spec.image();
    auto f = gguf::parse(img);
    REQUIRE(f);
    const auto t = tokenizer::any_tokenizer::from_gguf(*f);
    REQUIRE_FALSE(t);
    CHECK(t.error().code == tokenizer_errc::unsupported_model);
}

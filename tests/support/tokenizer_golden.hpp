// Checks synapse's BPE tokenizer against llama.cpp's tokenizations of the
// cases in a golden file (written by scripts/gen_tokenizer_fixture.py).
#pragma once

#include <branes/synapse/gguf/mapped_file.hpp>
#include <branes/synapse/gguf/reader.hpp>
#include <branes/synapse/tokenizer/bpe.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace synapse_test {

inline std::vector<std::uint32_t> u32_array(const branes::synapse::gguf::gguf_file& f, const std::string& key) {
    const auto* a = f.get_array(key);
    REQUIRE(a != nullptr);
    std::vector<std::uint32_t> v;
    for (std::uint64_t i = 0; i < a->size(); ++i) {
        v.push_back(*a->at<std::uint32_t>(i));
    }
    return v;
}

/// Number of cases checked.
inline std::size_t check_tokenizer_golden(const std::filesystem::path& vocab_path,
                                          const std::filesystem::path& golden_path) {
    using namespace branes::synapse;
    auto vf = gguf::mapped_file::open(vocab_path);
    REQUIRE(vf);
    auto gf = gguf::mapped_file::open(golden_path);
    REQUIRE(gf);
    auto vocab = gguf::parse(vf->bytes());
    REQUIRE(vocab);
    auto golden = gguf::parse(gf->bytes());
    REQUIRE(golden);
    auto tok = tokenizer::bpe_tokenizer::from_gguf(*vocab);
    if (!tok) {
        FAIL(to_string(tok.error().code) << ": " << tok.error().name);
    }
    const auto count = golden->get<std::uint32_t>("synapse.tok.count");
    REQUIRE(count);
    for (std::uint32_t i = 0; i < *count; ++i) {
        const std::string key = "synapse.tok." + std::to_string(i) + ".";
        const auto text = golden->get<std::string_view>(key + "text");
        REQUIRE(text);
        INFO("case " << i << ": \"" << *text << "\"");
        CHECK(tok->encode(*text, {.add_bos = true, .parse_special = false}) == u32_array(*golden, key + "plain"));
        const auto special = u32_array(*golden, key + "special");
        CHECK(tok->encode(*text, {.add_bos = true, .parse_special = true}) == special);
        CHECK(tok->decode(special, true) == *golden->get<std::string_view>(key + "detok"));
    }
    return *count;
}

}  // namespace synapse_test

// The BPE tokenizer against llama.cpp's llama_tokenize and token_to_piece.
//
// tests/data/bpe_vocab.gguf is a small synthetic Llama-3-style vocabulary
// (256 byte tokens, merges trained on a multilingual corpus, control and
// user-defined special tokens) and tests/data/bpe_golden.gguf holds
// llama.cpp's tokenizations of llama.cpp's own tokenizer test vectors plus
// synapse's edge cases (contractions, digit runs, scripts, emoji, whitespace
// and newline runs, special tokens, invalid UTF-8), with and without
// special-token parsing, and their detokenizations. The pre-tokenizer does
// not depend on the vocabulary, so the small vocabulary covers it fully.
//
// The real 128K Llama-3 vocabulary is checked by the opt-in case below.

#include "../support/tokenizer_golden.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <optional>
#include <string>

namespace {

std::optional<std::string> env(const char* name) {
#if defined(_MSC_VER)
    char* value = nullptr;
    std::size_t len = 0;
    if (_dupenv_s(&value, &len, name) != 0 || value == nullptr) {
        return std::nullopt;
    }
    std::string s(value);
    std::free(value);
    return s;
#else
    const char* value = std::getenv(name);
    return value == nullptr ? std::nullopt : std::optional<std::string>(value);
#endif
}

}  // namespace

TEST_CASE("tokenizer_golden: encode and decode match llama.cpp on the synthetic vocabulary", "[tokenizer][golden]") {
    const std::string dir = SYNAPSE_TEST_DATA_DIR;
    CHECK(synapse_test::check_tokenizer_golden(dir + "/bpe_vocab.gguf", dir + "/bpe_golden.gguf") >= 60);
}

// Opt-in: SYNAPSE_BPE_VOCAB (e.g. llama.cpp's models/ggml-vocab-llama-bpe.gguf
// or a Llama-3 model) and SYNAPSE_BPE_GOLDEN (scripts/gen_tokenizer_fixture.py
// golden <vocab> <golden>).
TEST_CASE("tokenizer_golden: encode and decode match llama.cpp on a real vocabulary", "[tokenizer][golden][model]") {
    const auto vocab = env("SYNAPSE_BPE_VOCAB");
    const auto golden = env("SYNAPSE_BPE_GOLDEN");
    if (!vocab || !golden) {
        SKIP("set SYNAPSE_BPE_VOCAB and SYNAPSE_BPE_GOLDEN to run");
    }
    CHECK(synapse_test::check_tokenizer_golden(*vocab, *golden) >= 60);
}

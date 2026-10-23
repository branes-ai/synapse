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

#include "../support/env.hpp"

#include <catch2/catch_test_macros.hpp>

#include <optional>
#include <string>

using synapse_test::get_env;

TEST_CASE("tokenizer_golden: encode and decode match llama.cpp on the synthetic vocabulary", "[tokenizer][golden]") {
    const std::string dir = SYNAPSE_TEST_DATA_DIR;
    CHECK(synapse_test::check_tokenizer_golden(dir + "/bpe_vocab.gguf", dir + "/bpe_golden.gguf") >= 60);
}

// Opt-in: SYNAPSE_BPE_VOCAB (e.g. llama.cpp's models/ggml-vocab-llama-bpe.gguf
// or a Llama-3 model) and SYNAPSE_BPE_GOLDEN (scripts/gen_tokenizer_fixture.py
// golden <vocab> <golden>).
TEST_CASE("tokenizer_golden: encode and decode match llama.cpp on a real vocabulary", "[tokenizer][golden][model]") {
    const auto vocab = get_env("SYNAPSE_BPE_VOCAB");
    const auto golden = get_env("SYNAPSE_BPE_GOLDEN");
    if (!vocab || !golden) {
        SKIP("set SYNAPSE_BPE_VOCAB and SYNAPSE_BPE_GOLDEN to run");
    }
    CHECK(synapse_test::check_tokenizer_golden(*vocab, *golden) >= 50);  // BPE or SPM case set
}

// tests/data/spm_vocab.gguf is llama.cpp's own Llama-2 SentencePiece
// vocabulary (models/ggml-vocab-llama-spm.gguf: 32,000 tokens with scores,
// 256 byte tokens), and spm_golden.gguf holds llama.cpp's tokenizations of
// its SPM test vectors plus synapse's SPM edge cases (space prefix around
// special tokens, byte fallback, invalid UTF-8).
TEST_CASE("tokenizer_golden: SentencePiece encode and decode match llama.cpp on the Llama-2 vocabulary",
          "[tokenizer][golden][spm]") {
    const std::string dir = SYNAPSE_TEST_DATA_DIR;
    CHECK(synapse_test::check_tokenizer_golden(dir + "/spm_vocab.gguf", dir + "/spm_golden.gguf") >= 50);
}

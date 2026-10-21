// The Llama decode graph against llama.cpp itself.
//
// Two small random-weight Llama models (two layers, grouped-query attention,
// llama3-style rope_freqs, tied output embedding), each with the logits
// llama.cpp computes for a 16-token sequence decoded one token at a time;
// all written by scripts/gen_llama_fixture.py. synapse decodes the same
// tokens and must reproduce llama.cpp's logits at every position.
//
//   - llama_tiny_f32: every weight F32, so neither side quantizes an
//     activation and the logits must agree to float rounding. This is the
//     exact check of the graph's semantics (RoPE with frequency factors,
//     GQA, the F16 KV cache, softmax, residuals, tied output).
//   - llama_tiny: Q4_K/Q6_K/Q8_0/Q4_0/F32 weights. On the generating machine
//     it also agrees to ~1e-7, but a transformer amplifies any one flipped
//     8-bit activation rounding (an ulp of exp or cos differs between C
//     runtimes) to ~1e-2 in the logits, so the bound is set there; an error
//     in a quantized format would be O(1).
//
// F16 weights are left out of the fixture on purpose: ggml rounds the
// activations of an F16 matmul to F16, synapse keeps them float (see
// quant/scheme.hpp), and that difference alone is amplified to ~1e-2.

#include "../support/llama_golden.hpp"

#include <catch2/catch_test_macros.hpp>

#include <string>

TEST_CASE("llama_golden: decode logits match llama.cpp on the tiny model", "[llama][golden]") {
    const std::string dir = SYNAPSE_TEST_DATA_DIR;
    const auto run = synapse_test::decode_against_golden(dir + "/llama_tiny.gguf", dir + "/llama_tiny_golden.gguf");
    REQUIRE(run.positions.size() == 16);
    for (std::size_t i = 0; i < run.positions.size(); ++i) {
        const auto& m = run.positions[i];
        INFO("position " << i << ": rel max err " << m.rel_max_err << ", rel rms err " << m.rel_rms_err);
        CHECK(m.rel_rms_err < 1e-2);
        CHECK(m.same_argmax);
    }
}

TEST_CASE("llama_golden: decode logits match llama.cpp on the all-F32 tiny model", "[llama][golden]") {
    const std::string dir = SYNAPSE_TEST_DATA_DIR;
    const auto run =
        synapse_test::decode_against_golden(dir + "/llama_tiny_f32.gguf", dir + "/llama_tiny_f32_golden.gguf");
    REQUIRE(run.positions.size() == 16);
    for (std::size_t i = 0; i < run.positions.size(); ++i) {
        const auto& m = run.positions[i];
        INFO("position " << i << ": rel max err " << m.rel_max_err << ", rel rms err " << m.rel_rms_err);
        CHECK(m.rel_max_err < 1e-5);
        CHECK(m.same_argmax);
    }
}

// The Llama-family decode graph against llama.cpp itself.
//
// Small random-weight models (two layers, grouped-query attention), each
// with the logits
// llama.cpp computes for a 16-token sequence decoded one token at a time;
// all written by scripts/gen_llama_fixture.py. synapse decodes the same
// tokens and must reproduce llama.cpp's logits at every position.
//
//   - llama_tiny_f32: every weight F32, so neither side quantizes an
//     activation and the logits must agree to float rounding. This is the
//     exact check of the graph's semantics (RoPE with frequency factors,
//     GQA, the F16 KV cache, softmax, residuals, tied output). Both sides
//     still round q, K, V and the attention probabilities to F16, and a
//     different float summation order (synapse's Highway build, or ggml's
//     SIMD) occasionally moves one of those values across an F16 rounding
//     boundary: the logits then move by up to a few 1e-5 (seen on
//     qwen2_tiny_f32, whose biases sharpen attention). Most positions agree
//     to ~1e-7; a semantic error is 1e-2 or more. The bound is 1e-4.
//   - llama_tiny: Q4_K/Q6_K/Q8_0/Q5_0/Q4_0/F32 weights. On the generating machine
//     it also agrees to ~1e-7, but a transformer amplifies any one flipped
//     8-bit activation rounding (an ulp of exp or cos differs between C
//     runtimes, and Highway's lane order differs from ggml's) to ~1e-2 in the
//     logits, and the KV cache carries it to every later position; an error
//     in a quantized format is O(1) (dropping or swapping Q5_0's 5th bits
//     gives ~1.6). The bound is 2e-2 RMS: in the Highway build llama_tiny
//     flips at position 3 and qwen2_tiny at 11, up to 1.15e-2, while the
//     scalar build stays ~1e-7. It is 5e-2 for mistral3_tiny, which
//     shows two such flips, 2e-2 at position 15 and 7e-3 at position 7
//     (every other position ~1e-7), in both the scalar and the Highway
//     build. They trace to RoPE:
//     ggml's CPU build (GCC, -march=native, which contracts by default) fuses
//     the rotation x0*cos - x1*sin into FMAs, so its rotated Q and K differ
//     from synapse's (contraction off) by an ulp in about a fifth of the
//     elements, with or without YaRN. mistral3_tiny_f32 runs the same graph
//     to ~4e-7.
//
// Each comes as a Llama (llama3-style rope_freqs, tied output embedding) and
// as a Qwen2 (NEOX RoPE, Q/K/V and output biases; the F32 one with its own
// output matrix): qwen2_tiny and qwen2_tiny_f32; and as a Mistral 3 (YaRN
// RoPE scaling with the ramp across the middle pairs, the attention
// temperature rescaling Q from position 4 on, and every bias llama.cpp's
// mistral3 loader accepts): mistral3_tiny and mistral3_tiny_f32.
//
// F16 weights are left out of the fixture on purpose: ggml rounds the
// activations of an F16 matmul to F16, synapse keeps them float (see
// quant/scheme.hpp), and that difference alone is amplified to ~1e-2.

#include "../support/llama_golden.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <string>

TEST_CASE("llama_golden: decode logits match llama.cpp on the tiny model", "[llama][golden]") {
    const std::string name =
        GENERATE(std::string("llama_tiny"), std::string("qwen2_tiny"), std::string("mistral3_tiny"));
    INFO(name);
    const std::string dir = SYNAPSE_TEST_DATA_DIR;
    const auto run = synapse_test::decode_against_golden(dir + "/" + name + ".gguf", dir + "/" + name + "_golden.gguf");
    REQUIRE(run.positions.size() == 16);
    for (std::size_t i = 0; i < run.positions.size(); ++i) {
        const auto& m = run.positions[i];
        INFO("position " << i << ": rel max err " << m.rel_max_err << ", rel rms err " << m.rel_rms_err);
        CHECK(m.rel_rms_err < (name == "mistral3_tiny" ? 5e-2 : 2e-2));
        CHECK(m.same_argmax);
    }
}

TEST_CASE("llama_golden: decode logits match llama.cpp on the all-F32 tiny model", "[llama][golden]") {
    const std::string name =
        GENERATE(std::string("llama_tiny_f32"), std::string("qwen2_tiny_f32"), std::string("mistral3_tiny_f32"));
    INFO(name);
    const std::string dir = SYNAPSE_TEST_DATA_DIR;
    const auto run = synapse_test::decode_against_golden(dir + "/" + name + ".gguf", dir + "/" + name + "_golden.gguf");
    REQUIRE(run.positions.size() == 16);
    for (std::size_t i = 0; i < run.positions.size(); ++i) {
        const auto& m = run.positions[i];
        INFO("position " << i << ": rel max err " << m.rel_max_err << ", rel rms err " << m.rel_rms_err);
        CHECK(m.rel_max_err < 1e-4);
        CHECK(m.same_argmax);
    }
}

// Golden tests for the mmproj image encoder: tests/data/clip_tiny*.gguf are
// tiny random-weight idefics3 mmproj files (SigLIP ViT, pixel shuffle, linear
// projector) and clip_tiny*_golden.gguf the embeddings llama.cpp's clip.cpp
// computes for a fixed image (scripts/gen_clip_fixture.py tiny | tiny-q8).
//
//   - clip_tiny: every matrix F32. Both sides then compute in float and
//     differ in summation order (ggml's SIMD dots, its vectorized exp in
//     softmax); where that moves a GELU input across a binary16 rounding
//     boundary, the output moves by a half-precision ulp. The embeddings agree
//     to ~1e-5. The bound is 1e-4.
//   - clip_tiny_q8: Q8_0 matrices, and the FFN stored with ffn_up and
//     ffn_down swapped as SmolVLM's files are. Both sides quantize the
//     activations to Q8_0, and here they agree to ~1e-7; one activation
//     rounding the other way would move the embeddings by ~1e-3. The bound is
//     1e-2; a wrong kernel or graph is O(1).

#include "../support/clip_golden.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <string>

TEST_CASE("clip_golden: embeddings match llama.cpp on the tiny mmproj", "[clip][golden]") {
    const std::string name = GENERATE(std::string("clip_tiny"), std::string("clip_tiny_q8"));
    INFO(name);
    const std::string dir = SYNAPSE_TEST_DATA_DIR;
    const auto run = synapse_test::encode_against_golden(dir + "/" + name + ".gguf", dir + "/" + name + "_golden.gguf");
    REQUIRE(run.tokens.size() == 16);
    for (std::size_t t = 0; t < run.tokens.size(); ++t) {
        const auto& m = run.tokens[t];
        INFO("token " << t << ": rel max err " << m.rel_max_err << ", rel rms err " << m.rel_rms_err);
        CHECK(m.rel_rms_err < (name == "clip_tiny" ? 1e-4 : 1e-2));
    }
}

TEST_CASE("clip_golden: results do not depend on the thread count", "[clip][golden]") {
    const std::string dir = SYNAPSE_TEST_DATA_DIR;
    const auto one =
        synapse_test::encode_against_golden(dir + "/clip_tiny_q8.gguf", dir + "/clip_tiny_q8_golden.gguf", 1);
    const auto all =
        synapse_test::encode_against_golden(dir + "/clip_tiny_q8.gguf", dir + "/clip_tiny_q8_golden.gguf", 0);
    CHECK(one.embeddings == all.embeddings);
}

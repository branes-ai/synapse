// A real mmproj encoder against llama.cpp: opt-in, since the model is not in
// the repository. Set
//
//   SYNAPSE_MMPROJ_MODEL   an idefics3 mmproj GGUF (e.g. SmolVLM-256M-Instruct's)
//   SYNAPSE_MMPROJ_GOLDEN  its golden embeddings, written by
//                          scripts/gen_clip_fixture.py ... encode <mmproj> <golden>
//
// and synapse encodes the golden file's image and compares every token's
// embedding with llama.cpp's. Skipped when either variable is unset.
//
// SmolVLM-256M-Instruct's F16 file agrees to 1.7e-3 rms (cosine 0.999999):
// ggml rounds the activations of an F16 matmul to F16 and synapse keeps them
// float, and 12 layers carry that. The Q8_0 file agrees to 4.8e-2 (cosine
// 0.9992): both sides quantize activations to Q8_0, and once one block's
// scale rounds the other way, SigLIP's outlier activations spread it. That is
// less than the 7% llama.cpp's own Q8_0 and F16 encoders differ by. A wrong
// kernel or graph is O(1), with cosine far from 1.

#include "../support/clip_golden.hpp"
#include "../support/env.hpp"

#include <catch2/catch_test_macros.hpp>

using synapse_test::get_env;

TEST_CASE("clip_model: embeddings match llama.cpp on a real mmproj", "[clip][model]") {
    const auto model = get_env("SYNAPSE_MMPROJ_MODEL");
    const auto golden = get_env("SYNAPSE_MMPROJ_GOLDEN");
    if (!model || !golden) {
        SKIP("set SYNAPSE_MMPROJ_MODEL and SYNAPSE_MMPROJ_GOLDEN to run");
    }
    const auto run = synapse_test::encode_against_golden(*model, *golden, 0);
    REQUIRE(!run.tokens.empty());
    for (std::size_t t = 0; t < run.tokens.size(); ++t) {
        const auto& m = run.tokens[t];
        INFO("token " << t << ": rel max err " << m.rel_max_err << ", rel rms err " << m.rel_rms_err << ", cosine "
                      << m.cosine);
        CHECK(m.rel_rms_err < 0.1);
        CHECK(m.cosine > 0.998);
    }
}

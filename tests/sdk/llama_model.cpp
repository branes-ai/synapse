// A real Llama model against llama.cpp: opt-in, since the model is not in
// the repository. Set
//
//   SYNAPSE_LLAMA_MODEL   a Llama-architecture GGUF (e.g. Llama-3.2-1B Q4_K_M)
//   SYNAPSE_LLAMA_GOLDEN  its golden logits, written by
//                         scripts/gen_llama_fixture.py ... logits <model> <golden> <tokens>
//
// and synapse decodes the golden file's tokens and compares its logits with
// llama.cpp's at every position. Skipped when either variable is unset.
//
// The bounds are loose by necessity: both sides agree to ~1e-7 until the
// first 8-bit activation rounding that falls differently, after which a
// trained model amplifies it to a few percent of the logits (synapse
// perturbed against itself, by skipping the F16 rounding of q, moves just as
// far). A wrong kernel or graph moves them by far more and loses the top token.

#include "../support/llama_golden.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>

TEST_CASE("llama_model: decode logits match llama.cpp on a real model", "[llama][model]") {
    const char* model = std::getenv("SYNAPSE_LLAMA_MODEL");
    const char* golden = std::getenv("SYNAPSE_LLAMA_GOLDEN");
    if (model == nullptr || golden == nullptr) {
        SKIP("set SYNAPSE_LLAMA_MODEL and SYNAPSE_LLAMA_GOLDEN to run");
    }
    const auto run = synapse_test::decode_against_golden(model, golden);
    REQUIRE(!run.positions.empty());
    std::size_t same = 0;
    for (std::size_t i = 0; i < run.positions.size(); ++i) {
        const auto& m = run.positions[i];
        INFO("position " << i << ": rel max err " << m.rel_max_err << ", rel rms err " << m.rel_rms_err
                         << ", llama.cpp's top token is our #" << m.ref_rank + 1);
        CHECK(m.rel_rms_err < 0.15);
        CHECK(m.ref_rank < 5);
        same += m.same_argmax ? 1 : 0;
    }
    INFO(same << " of " << run.positions.size() << " positions agree on the top token");
    CHECK(same * 5 >= run.positions.size() * 4);
}

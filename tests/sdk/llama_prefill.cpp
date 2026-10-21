// Batched prefill of the Llama graph. By G1 (gemm row t == gemv row t, bit
// for bit) and the token-by-token order of the per-row kernels, running a
// prompt in batches must give exactly the logits, and leave exactly the KV
// cache, of decoding it one token at a time. Checked bit for bit on the two
// llama.cpp-generated tiny models, for one batch, uneven chunks, and decoding
// on after a prefill; and the batched logits are checked against llama.cpp.

#include "../support/llama_golden.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cstring>
#include <string>
#include <vector>

using synapse_test::bit_equal;
using synapse_test::decode_in_chunks;

TEST_CASE("llama_prefill: batched runs reproduce token-by-token decoding bit for bit", "[llama][prefill]") {
    const std::string name = GENERATE(std::string("llama_tiny"), std::string("llama_tiny_f32"));
    const std::string dir = SYNAPSE_TEST_DATA_DIR;
    const auto files = synapse_test::open_golden(dir + "/" + name + ".gguf", dir + "/" + name + "_golden.gguf");
    const auto& tokens = files->tokens;
    REQUIRE(tokens.size() == 16);
    INFO(name);

    // Reference: one token per run, logits at every position.
    const auto sequential = decode_in_chunks(*files->model, tokens, std::vector<std::size_t>(16, 1), 1);

    SECTION("the whole prompt in one run") {
        const auto batched = decode_in_chunks(*files->model, tokens, {16}, 16);
        CHECK(bit_equal(batched[0], sequential[15]));
    }
    SECTION("uneven chunks through a larger graph") {
        const auto batched = decode_in_chunks(*files->model, tokens, {5, 7, 4}, 8);
        CHECK(bit_equal(batched[0], sequential[4]));
        CHECK(bit_equal(batched[1], sequential[11]));
        CHECK(bit_equal(batched[2], sequential[15]));
    }
    SECTION("decoding on after a prefill (the KV cache matches too)") {
        const auto batched = decode_in_chunks(*files->model, tokens, {12, 1, 1, 1, 1}, 12);
        for (std::size_t i = 0; i < 5; ++i) {
            CHECK(bit_equal(batched[i], sequential[11 + i]));
        }
    }
}

TEST_CASE("llama_prefill: batched logits match llama.cpp", "[llama][prefill][golden]") {
    const std::string dir = SYNAPSE_TEST_DATA_DIR;
    const auto files = synapse_test::open_golden(dir + "/llama_tiny_f32.gguf", dir + "/llama_tiny_f32_golden.gguf");
    const auto batched = decode_in_chunks(*files->model, files->tokens, {16}, 16);
    const auto* info = files->golden->find_tensor("logits");
    REQUIRE(info != nullptr);
    const std::size_t n_vocab = batched[0].size();
    std::vector<float> ref(n_vocab);
    std::memcpy(
        ref.data(), files->golden->tensor_data(*info).data() + 15 * n_vocab * sizeof(float), n_vocab * sizeof(float));
    double max_err = 0.0, max_ref = 0.0;
    for (std::size_t v = 0; v < n_vocab; ++v) {
        max_err = std::max(max_err, std::fabs(static_cast<double>(batched[0][v]) - ref[v]));
        max_ref = std::max(max_ref, std::fabs(static_cast<double>(ref[v])));
    }
    CHECK(max_err / max_ref < 1e-5);
}

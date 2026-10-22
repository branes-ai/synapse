// Multithreaded execution. A session splits matmuls across weight rows and
// attention across KV head groups on MTL5's process-wide pool; every output
// row and head is computed by exactly the operations of a single-threaded
// run, so results must be bit-identical for any thread count (D1), for
// token-by-token decoding and for batched prefill alike.
//
// The pool is sized once per process from MTL5_NUM_THREADS (clamped to the
// hardware), so each test sets it before the first session asks for threads.

#include "../support/env.hpp"
#include "../support/llama_golden.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <mtl/detail/thread_pool.hpp>

#include <string>
#include <vector>

using synapse_test::bit_equal;
using synapse_test::decode_in_chunks;

namespace {

unsigned pool_threads() {
    synapse_test::set_env("MTL5_NUM_THREADS", "8");  // before the pool's first use
    return mtl::detail::thread_pool::instance().size();
}

}  // namespace

TEST_CASE("llama_threads: decoding and prefill are bit-identical for any thread count (D1)", "[llama][threads]") {
    const unsigned pool = pool_threads();
    INFO("pool of " << pool << " threads");
    const std::string name = GENERATE(std::string("llama_tiny"), std::string("llama_tiny_f32"));
    const std::string dir = SYNAPSE_TEST_DATA_DIR;
    const auto files = synapse_test::open_golden(dir + "/" + name + ".gguf", dir + "/" + name + "_golden.gguf");
    const auto& tokens = files->tokens;
    const std::vector<std::size_t> one_by_one(tokens.size(), 1);
    const auto reference = decode_in_chunks(*files->model, tokens, one_by_one, 1, 1);
    const auto reference_prefill = decode_in_chunks(*files->model, tokens, {5, 7, 4}, 8, 1);

    for (unsigned threads : {2u, 3u, 8u, 0u}) {  // 0: the whole pool
        INFO(name << ", " << threads << " threads");
        const auto decoded = decode_in_chunks(*files->model, tokens, one_by_one, 1, threads);
        REQUIRE(decoded.size() == reference.size());
        for (std::size_t i = 0; i < decoded.size(); ++i) {
            CHECK(bit_equal(decoded[i], reference[i]));
        }
        const auto prefilled = decode_in_chunks(*files->model, tokens, {5, 7, 4}, 8, threads);
        for (std::size_t i = 0; i < prefilled.size(); ++i) {
            CHECK(bit_equal(prefilled[i], reference_prefill[i]));
        }
    }
}

TEST_CASE("llama_threads: a session uses at most the pool's threads", "[llama][threads]") {
    const unsigned pool = pool_threads();
    const std::string dir = SYNAPSE_TEST_DATA_DIR;
    const auto files = synapse_test::open_golden(dir + "/llama_tiny.gguf", dir + "/llama_tiny_golden.gguf");
    auto model = branes::synapse::models::load_llama(*files->model, 4);
    REQUIRE(model);
    CHECK(branes::synapse::graph::session{model->graph}.threads() == 1);
    CHECK(branes::synapse::graph::session{model->graph, 0}.threads() == pool);
    CHECK(branes::synapse::graph::session{model->graph, 1000}.threads() == pool);
}

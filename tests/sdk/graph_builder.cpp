// Unit tests for graph_builder: well-formed graphs and every rejection rule.

#include <branes/synapse/graph/graph.hpp>

#include "../support/random_weights.hpp"

#include <catch2/catch_test_macros.hpp>

#include <random>
#include <variant>

using namespace branes::synapse::graph;
using synapse_test::random_matrix;
using synapse_test::tensor_type;

TEST_CASE("graph_builder: builds a valid graph in topological order", "[graph]") {
    std::mt19937 rng(1);
    const random_matrix<tensor_type::q4_k> w(512, 256, rng);
    const random_matrix<tensor_type::f32> gamma(1, 256, rng);

    graph_builder b;
    const tensor_id x = b.input(256);
    const tensor_id h = b.rms_norm(x, gamma.view(), 1e-5f);
    const tensor_id y = b.matmul(w.view(), h);
    const tensor_id z = b.silu(y);
    b.output(z);
    auto g = std::move(b).finalize();
    REQUIRE(g);

    CHECK(g->inputs().size() == 1);
    CHECK(g->outputs().size() == 1);
    CHECK(g->nodes().size() == 3);
    CHECK(g->tensors()[y].size == 512);
    CHECK(g->tensors()[z].role == tensor_role::output);
    // The matmul's scratch tensor holds the gemv activation buffers.
    const auto& mm = std::get<op_matmul>(g->nodes()[1]);
    CHECK(g->tensors()[mm.scratch].role == tensor_role::scratch);
    CHECK(g->tensors()[mm.scratch].bytes ==
          branes::synapse::kernels::gemv_scratch_bytes<branes::synapse::quant::scheme<tensor_type::q4_k>>(256));

    // I1: every operand of node i is an input or a result of an earlier node.
    for (std::size_t i = 0; i < g->nodes().size(); ++i) {
        for (tensor_id t : operands(g->nodes()[i])) {
            bool earlier = g->tensors()[t].role == tensor_role::input;
            for (std::size_t j = 0; j < i; ++j) {
                for (tensor_id r : results(g->nodes()[j])) {
                    earlier = earlier || r == t;
                }
            }
            CHECK(earlier);
        }
    }
}

TEST_CASE("graph_builder: rejects mismatched sizes and reports the node", "[graph]") {
    std::mt19937 rng(2);
    const random_matrix<tensor_type::q8_0> w(64, 128, rng);

    graph_builder b;
    const tensor_id x = b.input(256);
    const tensor_id y = b.silu(x);
    CHECK(b.matmul(w.view(), y) == invalid_tensor);  // 128 columns vs 256 elements
    b.output(y);
    const auto g = std::move(b).finalize();
    REQUIRE_FALSE(g);
    CHECK(g.error().code == graph_errc::size_mismatch);
    CHECK(g.error().node == 1);
}

TEST_CASE("graph_builder: the first error wins and later calls are no-ops", "[graph]") {
    graph_builder b;
    const tensor_id a = b.input(8);
    const tensor_id c = b.input(16);
    CHECK(b.add(a, c) == invalid_tensor);  // size mismatch
    CHECK(b.silu(a) == invalid_tensor);    // no-op after the error
    CHECK(b.input(4) == invalid_tensor);
    const auto g = std::move(b).finalize();
    REQUIRE_FALSE(g);
    CHECK(g.error().code == graph_errc::size_mismatch);
}

TEST_CASE("graph_builder: rejects invalid operands, empty inputs, and bad outputs", "[graph]") {
    {
        graph_builder b;
        CHECK(b.silu(42) == invalid_tensor);
        CHECK(std::move(b).finalize().error().code == graph_errc::invalid_operand);
    }
    {
        graph_builder b;
        CHECK(b.input(0) == invalid_tensor);
        CHECK(std::move(b).finalize().error().code == graph_errc::empty_tensor);
    }
    {
        graph_builder b;
        b.input(8);
        CHECK(std::move(b).finalize().error().code == graph_errc::no_outputs);
    }
    {
        graph_builder b;
        const tensor_id x = b.input(8);
        const tensor_id y = b.silu(x);
        b.output(y);
        b.output(y);  // listed twice
        CHECK(std::move(b).finalize().error().code == graph_errc::bad_output);
    }
    {
        std::mt19937 rng(3);
        const random_matrix<tensor_type::q8_0> w(32, 32, rng);
        graph_builder b;
        const tensor_id x = b.input(32);
        const tensor_id y = b.matmul(w.view(), x);
        b.output(y + 1);  // the matmul's scratch tensor
        CHECK(std::move(b).finalize().error().code == graph_errc::bad_output);
    }
    {
        std::mt19937 rng(4);
        const random_matrix<tensor_type::f32> gamma(2, 8, rng);  // not a row vector
        graph_builder b;
        const tensor_id x = b.input(8);
        CHECK(b.rms_norm(x, gamma.view(), 1e-5f) == invalid_tensor);
        CHECK(std::move(b).finalize().error().code == graph_errc::size_mismatch);
    }
}

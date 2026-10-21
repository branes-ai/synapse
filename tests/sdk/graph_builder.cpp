// Unit tests for graph_builder: well-formed graphs and every rejection rule.

#include <branes/synapse/graph/graph.hpp>

#include "../support/random_weights.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <variant>
#include <vector>

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
    // The matmul's scratch tensor holds the gemm activation buffers for one row.
    const auto& mm = std::get<op_matmul>(g->nodes()[1]);
    CHECK(g->tensors()[mm.scratch].role == tensor_role::scratch);
    CHECK(g->tensors()[mm.scratch].bytes ==
          branes::synapse::kernels::gemm_scratch_bytes<branes::synapse::quant::scheme<tensor_type::q4_k>>(256, 1));

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

TEST_CASE("graph_builder: rejects sizes whose byte counts would overflow", "[graph]") {
    {
        graph_builder b;
        CHECK(b.input(std::numeric_limits<std::size_t>::max() / 2) == invalid_tensor);  // * sizeof(float) wraps
        CHECK(std::move(b).finalize().error().code == graph_errc::too_large);
    }
    {
        // Each input fits; together they exceed max_graph_bytes.
        graph_builder b;
        // Leave room for the alignment padding the bound also counts.
        const std::size_t quarter = (max_graph_bytes / 4 - plan_alignment) / sizeof(float);
        CHECK(b.input(quarter) != invalid_tensor);
        CHECK(b.input(quarter) != invalid_tensor);
        CHECK(b.input(quarter) != invalid_tensor);
        CHECK(b.input(quarter) != invalid_tensor);
        CHECK(b.input(quarter) == invalid_tensor);
        CHECK(std::move(b).finalize().error().code == graph_errc::too_large);
    }
    {
        // An op whose result would cross the bound fails too.
        graph_builder b;
        const tensor_id x = b.input(max_graph_bytes / sizeof(float) / 2);
        CHECK(x != invalid_tensor);
        CHECK(b.silu(x) == invalid_tensor);
        CHECK(std::move(b).finalize().error().code == graph_errc::too_large);
    }
}

TEST_CASE("graph_builder: embed, rope, and attention over a KV cache", "[graph][attention]") {
    std::mt19937 rng(5);
    const random_matrix<tensor_type::q8_0> table(100, 64, rng);
    graph_builder b;
    const tensor_id x = b.embed(table.view());
    const tensor_id q = b.rope(x, 16, 10000.0f);          // 4 heads
    const tensor_id k = b.rope(b.silu(x), 16, 10000.0f);  // stands in for Wk x
    const tensor_id v = b.silu(x);
    const state_id small = b.kv_cache(32, 4, 16);
    const state_id large = b.kv_cache(64, 4, 16);
    const tensor_id a = b.attention(q, k, v, small, 4, 0.25f);
    const tensor_id a2 = b.attention(q, k, v, large, 4, 0.25f);
    b.output(b.add(a, a2));
    auto g = std::move(b).finalize();
    REQUIRE(g);

    CHECK(g->inputs().empty());  // the token arrives with the run
    CHECK(g->token_limit() == 100);
    CHECK(g->position_limit() == 32);  // the smaller cache bounds the positions
    REQUIRE(g->states().size() == 2);
    CHECK(g->states()[small].bytes == 2 * 32 * 4 * 16 * sizeof(std::uint16_t));
    const auto& op = std::get<op_attention>(g->nodes()[5]);
    CHECK(operands(g->nodes()[5]) == std::vector<tensor_id>{q, k, v});
    CHECK(results(g->nodes()[5]) == std::vector<tensor_id>{a, op.scratch});
    CHECK(g->tensors()[op.scratch].role == tensor_role::scratch);
    CHECK(g->tensors()[op.scratch].bytes ==
          branes::synapse::kernels::attention_scratch_floats(4, 4, 16, 32) * sizeof(float));
}

TEST_CASE("graph_builder: rejects head layouts and parameters that do not fit", "[graph][attention]") {
    const auto error_of = [](auto build) {
        graph_builder b;
        build(b);
        return std::move(b).finalize().error().code;
    };
    // rope (I6)
    CHECK(error_of([](graph_builder& b) { b.rope(b.input(30), 15, 1e4f); }) == graph_errc::bad_shape);  // odd head
    CHECK(error_of([](graph_builder& b) { b.rope(b.input(30), 4, 1e4f); }) == graph_errc::bad_shape);   // 7.5 heads
    CHECK(error_of([](graph_builder& b) {
              const float f[3] = {1, 1, 1};
              b.rope(b.input(32), 8, 1e4f, f);  // needs 4 factors
          }) == graph_errc::bad_shape);
    CHECK(error_of([](graph_builder& b) { b.rope(b.input(32), 8, std::nanf("")); }) == graph_errc::bad_parameter);
    CHECK(error_of([](graph_builder& b) {
              const float f[4] = {1, 0, 1, 1};
              b.rope(b.input(32), 8, 1e4f, f);
          }) == graph_errc::bad_parameter);
    // kv_cache
    CHECK(error_of([](graph_builder& b) { b.kv_cache(0, 2, 8); }) == graph_errc::bad_shape);
    CHECK(error_of([](graph_builder& b) { b.kv_cache(8, 2, branes::synapse::kernels::max_head_dim + 2); }) ==
          graph_errc::bad_shape);
    CHECK(error_of([](graph_builder& b) { b.kv_cache(std::size_t{1} << 30, std::size_t{1} << 28, 64); }) ==
          graph_errc::too_large);
    // attention
    const auto attend = [](std::size_t q_size, std::size_t n_heads, float scale) {
        return [=](graph_builder& b) {
            const tensor_id q = b.input(q_size);
            const tensor_id kv = b.input(16);  // 2 KV heads of 8
            b.attention(q, kv, kv, b.kv_cache(8, 2, 8), n_heads, scale);
        };
    };
    CHECK(error_of(attend(24, 3, 1.0f)) == graph_errc::bad_shape);  // 3 heads on 2 KV heads
    CHECK(error_of(attend(24, 4, 1.0f)) == graph_errc::bad_shape);  // q is not 4 heads of 8
    CHECK(error_of(attend(32, 4, 0.0f)) == graph_errc::bad_parameter);
    CHECK(error_of([](graph_builder& b) {
              const tensor_id t = b.input(16);
              b.attention(t, t, t, 0, 2, 1.0f);  // no such state
          }) == graph_errc::bad_state);
    CHECK(error_of([](graph_builder& b) {
              const tensor_id t = b.input(16);
              const state_id c = b.kv_cache(8, 2, 8);
              b.attention(t, t, t, c, 2, 1.0f);
              b.attention(t, t, t, c, 2, 1.0f);  // I7: one writer per cache
          }) == graph_errc::bad_state);
    // embed
    CHECK(error_of([](graph_builder& b) { b.embed(matrix_ref{}); }) == graph_errc::empty_tensor);
}

TEST_CASE("graph_builder: batched graphs size tensors for max_batch rows", "[graph][batch]") {
    std::mt19937 rng(6);
    const random_matrix<tensor_type::q8_0> w(64, 32, rng);
    graph_builder b(8);
    const tensor_id x = b.input(32);
    const tensor_id y = b.matmul(w.view(), x);
    const tensor_id last = b.last_row(y);
    const tensor_id z = b.silu(last);
    b.output(z);
    auto g = std::move(b).finalize();
    REQUIRE(g);
    CHECK(g->max_batch() == 8);
    CHECK(g->rows(x) == 8);
    CHECK(g->tensors()[x].bytes == 8 * 32 * sizeof(float));
    CHECK(g->rows(y) == 8);
    CHECK(g->rows(last) == 1);  // I8: last_row is single-row, and so is what follows
    CHECK(g->rows(z) == 1);
    const auto& mm = std::get<op_matmul>(g->nodes()[0]);
    CHECK(g->tensors()[mm.scratch].bytes ==
          branes::synapse::kernels::gemm_scratch_bytes<branes::synapse::quant::scheme<tensor_type::q8_0>>(32, 8));
}

TEST_CASE("graph_builder: rejects ops that mix batched and single-row tensors (I8)", "[graph][batch]") {
    const auto error_of = [](auto build) {
        graph_builder b(4);
        build(b);
        return std::move(b).finalize().error().code;
    };
    CHECK(error_of([](graph_builder& b) {
              const tensor_id x = b.input(16);
              b.add(x, b.last_row(x));
          }) == graph_errc::size_mismatch);
    CHECK(error_of([](graph_builder& b) { b.rope(b.last_row(b.input(16)), 8, 1e4f); }) == graph_errc::bad_shape);
    CHECK(error_of([](graph_builder& b) { b.last_row(b.last_row(b.input(16))); }) == graph_errc::bad_shape);
    CHECK(error_of([](graph_builder& b) {
              const tensor_id x = b.input(16);
              b.attention(b.last_row(x), x, x, b.kv_cache(8, 2, 8), 2, 1.0f);
          }) == graph_errc::bad_shape);
    {
        graph_builder b(0);
        CHECK(b.input(4) == invalid_tensor);
        CHECK(std::move(b).finalize().error().code == graph_errc::bad_shape);
    }
    {
        graph_builder b(std::numeric_limits<std::size_t>::max() / 8);  // max_batch * size * 4 wraps
        CHECK(b.input(64) == invalid_tensor);
        CHECK(std::move(b).finalize().error().code == graph_errc::too_large);
    }
}

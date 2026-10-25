// Unit tests for the image-encoder ops of the graph: layer_norm, gelu,
// add_rows, self_attention and pixel_shuffle, their checks (I6, I8, I9), and
// how a session runs a graph of patches.

#include <branes/synapse/graph/graph.hpp>
#include <branes/synapse/graph/session.hpp>
#include <branes/synapse/kernels/elementwise.hpp>

#include "../support/random_weights.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

using namespace branes::synapse;
using namespace branes::synapse::graph;
using synapse_test::random_matrix;
using synapse_test::tensor_type;

namespace {

graph_errc error_of(graph_builder&& b) {
    auto g = std::move(b).finalize();
    REQUIRE_FALSE(g);
    return g.error().code;
}

}  // namespace

TEST_CASE("graph_vision: pixel_shuffle divides the rows and fixes the batch (I9)", "[graph][vision]") {
    std::mt19937 rng(1);
    const random_matrix<tensor_type::q8_0> fc(48, 4 * 32, rng);
    graph_builder b(16);  // a 4 x 4 patch grid
    const tensor_id x = b.input(32);
    const tensor_id merged = b.pixel_shuffle(x, 4, 2);
    const tensor_id y = b.matmul(fc.view(), merged);
    b.output(y);
    auto g = std::move(b).finalize();
    REQUIRE(g);
    CHECK(g->full_batch_only());
    CHECK_FALSE(g->embeds());
    CHECK(g->tensors()[merged].size == 128);
    CHECK(g->tensors()[merged].row_div == 4);
    CHECK(g->rows(merged) == 4);
    CHECK(g->rows(y) == 4);  // matmul keeps its operand's rows
    CHECK(g->tensors()[y].bytes == 4 * 48 * sizeof(float));
}

TEST_CASE("graph_vision: rejects grids, batching and parameters that do not fit", "[graph][vision]") {
    std::mt19937 rng(2);
    const random_matrix<tensor_type::f32> g32(1, 32, rng);
    const random_matrix<tensor_type::f32> g16(1, 16, rng);
    const random_matrix<tensor_type::f32> pos(16, 32, rng);
    const random_matrix<tensor_type::f32> short_pos(15, 32, rng);
    {
        graph_builder b(16);
        b.pixel_shuffle(b.input(32), 4, 3);  // 3 does not divide the grid
        CHECK(error_of(std::move(b)) == graph_errc::bad_shape);
    }
    {
        graph_builder b(16);
        b.pixel_shuffle(b.input(32), 5, 2);  // 5 does not divide max_batch
        CHECK(error_of(std::move(b)) == graph_errc::bad_shape);
    }
    {
        graph_builder b(16);
        b.pixel_shuffle(b.input(32), 4, 1);  // nothing to merge
        CHECK(error_of(std::move(b)) == graph_errc::bad_shape);
    }
    {
        graph_builder b(16);  // per-patch ops reject a merged tensor (I8)
        const tensor_id m = b.pixel_shuffle(b.input(8), 4, 2);
        b.add_rows(m, pos.view());
        CHECK(error_of(std::move(b)) == graph_errc::bad_shape);
    }
    {
        graph_builder b(16);
        const tensor_id m = b.pixel_shuffle(b.input(8), 4, 2);
        b.self_attention(m, m, m, 4, 0.5f);
        CHECK(error_of(std::move(b)) == graph_errc::bad_shape);
    }
    {
        graph_builder b(16);
        const tensor_id m = b.pixel_shuffle(b.input(8), 4, 2);
        b.last_row(m);
        CHECK(error_of(std::move(b)) == graph_errc::bad_shape);
    }
    {
        graph_builder b(16);  // element-wise ops need equal row counts (I8)
        const tensor_id x = b.input(32);
        b.add(b.pixel_shuffle(b.input(8), 4, 2), x);
        CHECK(error_of(std::move(b)) == graph_errc::size_mismatch);
    }
    {
        graph_builder b(16);
        b.add_rows(b.input(32), short_pos.view());  // fewer table rows than patches
        CHECK(error_of(std::move(b)) == graph_errc::size_mismatch);
    }
    {
        graph_builder b(16);
        b.layer_norm(b.input(32), g16.view(), {}, 1e-6f);
        CHECK(error_of(std::move(b)) == graph_errc::size_mismatch);
    }
    {
        graph_builder b(16);
        b.layer_norm(b.input(32), g32.view(), g16.view(), 1e-6f);
        CHECK(error_of(std::move(b)) == graph_errc::size_mismatch);
    }
    {
        graph_builder b(16);
        b.layer_norm(b.input(32), g32.view(), {}, -1.0f);
        CHECK(error_of(std::move(b)) == graph_errc::bad_parameter);
    }
    {
        graph_builder b(16);
        const tensor_id x = b.input(32);
        b.self_attention(x, x, x, 5, 0.5f);  // 5 heads do not divide 32
        CHECK(error_of(std::move(b)) == graph_errc::bad_shape);
    }
    {
        graph_builder b(16);
        const tensor_id x = b.input(32);
        b.self_attention(x, x, x, 4, 0.0f);
        CHECK(error_of(std::move(b)) == graph_errc::bad_parameter);
    }
}

TEST_CASE("graph_vision: a session runs a patch graph only in full and without tokens", "[graph][vision][session]") {
    std::mt19937 rng(3);
    const random_matrix<tensor_type::f32> gamma(1, 8, rng, 0.5f, 1.5f);
    const random_matrix<tensor_type::f32> beta(1, 8, rng, -0.5f, 0.5f);
    const random_matrix<tensor_type::f32> pos(16, 8, rng, -1.0f, 1.0f);
    graph_builder b(16);
    const tensor_id x = b.input(8);
    const tensor_id p = b.add_rows(x, pos.view());
    const tensor_id h = b.gelu(b.layer_norm(p, gamma.view(), beta.view(), 1e-6f));
    const tensor_id a = b.self_attention(h, h, h, 2, 0.5f);
    const tensor_id m = b.pixel_shuffle(a, 4, 2);
    b.output(m);
    auto g = std::move(b).finalize();
    REQUIRE(g);

    session s{*g};
    std::vector<float> in(16 * 8);
    for (std::size_t i = 0; i < in.size(); ++i) {
        in[i] = static_cast<float>(i % 7) * 0.25f - 0.75f;
    }
    std::copy(in.begin(), in.end(), s.tensor(x).begin());
    CHECK_FALSE(s.run_rows(15));  // I9: the grid is fixed
    std::vector<std::uint32_t> tokens(16, 0);
    CHECK_FALSE(s.run(batch{std::span<const std::uint32_t>(tokens).first(8), 0}));
    REQUIRE(s.run_rows(16));
    CHECK(s.tensor(m).size() == 4 * 32);

    // The same math, op by op with the kernels.
    std::vector<float> want_p(in.size());
    std::vector<float> row(8);
    for (std::size_t i = 0; i < 16; ++i) {
        pos.view().decode_row(i, row.data());
        for (std::size_t e = 0; e < 8; ++e) {
            want_p[i * 8 + e] = in[i * 8 + e] + row[e];
        }
    }
    std::vector<float> want_h(in.size());
    for (std::size_t i = 0; i < 16; ++i) {
        kernels::layer_norm(std::span<const float>(want_p).subspan(i * 8, 8),
                            gamma.view(),
                            beta.view(),
                            1e-6f,
                            std::span<float>(want_h).subspan(i * 8, 8));
    }
    std::vector<float> want_g(in.size());
    kernels::gelu(want_h, want_g);
    std::vector<float> want_a(in.size());
    std::vector<float> scores(16);
    for (std::size_t hd = 0; hd < 2; ++hd) {
        kernels::self_attend_head(want_g, want_g, want_g, 16, 8, 4, hd, 0, 16, 0.5f, want_a, scores);
    }
    std::vector<float> want(in.size());
    kernels::pixel_shuffle(want_a, 8, 4, 4, 2, want);
    const auto got = s.tensor(m);
    CHECK(std::vector<float>(got.begin(), got.end()) == want);

    // A graph that embeds needs tokens: run_rows refuses it.
    const random_matrix<tensor_type::f32> table(4, 8, rng);
    graph_builder eb(2);
    eb.output(eb.embed(table.view()));
    auto eg = std::move(eb).finalize();
    REQUIRE(eg);
    CHECK(eg->embeds());
    session es{*eg};
    CHECK_FALSE(es.run_rows(1));
}

TEST_CASE("graph_vision: gelu over many rows does not depend on the thread count", "[graph][vision][session]") {
    // 37 rows of 300: pieces of a multiple of 8 rows, and a short last one (D1).
    constexpr std::size_t n = 37;
    constexpr std::size_t size = 300;
    graph_builder b(n);
    const tensor_id x = b.input(size);
    const tensor_id y = b.gelu(x);
    b.output(y);
    auto g = std::move(b).finalize();
    REQUIRE(g);

    std::mt19937 rng(5);
    std::normal_distribution<float> normal(0.0f, 4.0f);
    std::vector<float> in(n * size);
    for (auto& v : in) {
        v = normal(rng);
    }
    std::vector<float> want(in.size());
    kernels::gelu(in, want);
    for (unsigned threads : {1u, 3u, 8u}) {
        INFO("threads = " << threads);
        session s{*g, threads};
        std::copy(in.begin(), in.end(), s.tensor(x).begin());
        REQUIRE(s.run_rows(n));
        const auto got = s.tensor(y);
        CHECK(std::vector<float>(got.begin(), got.end()) == want);
    }
}

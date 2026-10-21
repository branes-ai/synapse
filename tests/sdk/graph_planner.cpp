// Tests for the static memory planner: liveness on a known graph, the plan
// invariants (P1-P3) on many random graphs, and the memory saved by aliasing.

#include <branes/synapse/graph/planner.hpp>

#include "../support/random_weights.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <random>
#include <vector>

using namespace branes::synapse::graph;
using synapse_test::random_matrix;
using synapse_test::tensor_type;

TEST_CASE("graph_planner: liveness of a chain", "[graph][planner]") {
    // x -> a = silu(x) -> b = silu(a) -> c = add(a, b) (output)
    graph_builder bld;
    const tensor_id x = bld.input(64);
    const tensor_id a = bld.silu(x);
    const tensor_id b = bld.silu(a);
    const tensor_id c = bld.add(a, b);
    bld.output(c);
    const auto g = std::move(bld).finalize();
    REQUIRE(g);
    const auto live = liveness(*g);
    CHECK(live[x].first == 0);
    CHECK(live[x].last == 0);  // last read by node 0
    CHECK(live[a].first == 0);
    CHECK(live[a].last == 2);  // read again by the add
    CHECK(live[b].first == 1);
    CHECK(live[c].last == 2);  // output: to the end
}

TEST_CASE("graph_planner: a long chain runs in a constant footprint", "[graph][planner]") {
    // 100 element-wise ops on 1024 floats: each value dies as soon as the next
    // is computed, so two buffers suffice (an op's input and output).
    graph_builder bld;
    tensor_id t = bld.input(1024);
    for (int i = 0; i < 100; ++i) {
        t = bld.silu(t);
    }
    bld.output(t);
    const auto g = std::move(bld).finalize();
    REQUIRE(g);
    const auto p = plan(*g);
    CHECK(plan_is_valid(*g, p));
    CHECK(p.arena_bytes == 2 * 1024 * sizeof(float));
    CHECK(p.naive_bytes == 101 * 1024 * sizeof(float));
}

TEST_CASE("graph_planner: plans of random graphs satisfy P1-P3", "[graph][planner]") {
    std::mt19937 rng(20261021);
    const std::size_t widths[] = {32, 64, 96};
    std::uniform_int_distribution<int> pick_width(0, 2);
    std::uniform_int_distribution<int> pick_op(0, 5);

    for (int trial = 0; trial < 200; ++trial) {
        // Weights must outlive the graph; one per (in, out) width pair used.
        std::vector<random_matrix<tensor_type::q8_0>> mats;
        std::vector<random_matrix<tensor_type::f32>> gammas;
        mats.reserve(64);
        gammas.reserve(64);

        graph_builder b;
        std::vector<std::pair<tensor_id, std::size_t>> pool;  // tensor, width
        for (int i = 0; i < 2; ++i) {
            const std::size_t w = widths[pick_width(rng)];
            pool.emplace_back(b.input(w), w);
        }
        const int n_ops = std::uniform_int_distribution<int>(1, 40)(rng);
        for (int k = 0; k < n_ops; ++k) {
            const auto [x, wx] = pool[std::uniform_int_distribution<std::size_t>(0, pool.size() - 1)(rng)];
            switch (pick_op(rng)) {
            case 0: {  // matmul to a random width
                const std::size_t wo = widths[pick_width(rng)];
                mats.emplace_back(wo, wx, rng);
                pool.emplace_back(b.matmul(mats.back().view(), x), wo);
                break;
            }
            case 1:
                pool.emplace_back(b.silu(x), wx);
                break;
            case 2:
                pool.emplace_back(b.softmax(x), wx);
                break;
            case 3: {
                gammas.emplace_back(1, wx, rng);
                pool.emplace_back(b.rms_norm(x, gammas.back().view(), 1e-5f), wx);
                break;
            }
            default: {  // add / mul with a same-width partner, if any
                for (const auto& [y, wy] : pool) {
                    if (wy == wx && y != x) {
                        pool.emplace_back(k % 2 == 0 ? b.add(x, y) : b.mul(x, y), wx);
                        break;
                    }
                }
                break;
            }
            }
        }
        // Two outputs: the newest tensor and a random older one.
        b.output(pool.back().first);
        const tensor_id older = pool[std::uniform_int_distribution<std::size_t>(0, pool.size() - 2)(rng)].first;
        if (older != pool.back().first) {
            b.output(older);
        }
        const auto g = std::move(b).finalize();
        REQUIRE(g);
        const auto p = plan(*g);
        INFO("trial " << trial << ": " << g->nodes().size() << " nodes, " << g->tensors().size() << " tensors");
        REQUIRE(plan_is_valid(*g, p));
        CHECK(p.arena_bytes <= p.naive_bytes);
    }
}

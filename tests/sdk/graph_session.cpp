// End-to-end tests for session: a transformer-shaped decode step (RMSNorm,
// a SwiGLU feed-forward block with a residual, a final norm, logits, and a
// softmax) run through the planned arena.
//
//   - Aliasing: the session's outputs must equal, bit for bit, an op-by-op
//     execution that gives every tensor its own buffer and calls the same
//     kernels. Any planner bug that overlaps live tensors breaks this.
//   - Semantics: with F32/F16 weights (whose kernels keep activations in
//     float), the session must match a double-precision reference to float
//     rounding, which checks that the graph computes the intended math.
//     Accuracy of the quantized kernels themselves is covered by
//     kernels_golden (against ggml) and needs no re-test here: compounded
//     through four 8-bit-activation matmuls and a softmax, its error is
//     data-dependent and would only blur this check.

#include <branes/synapse/graph/session.hpp>
#include <branes/synapse/kernels/elementwise.hpp>
#include <branes/synapse/kernels/gemv.hpp>

#include "../support/random_weights.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

using namespace branes::synapse;
using namespace branes::synapse::graph;
using synapse_test::random_matrix;
using synapse_test::tensor_type;

namespace {

constexpr std::size_t d = 256;     // model width
constexpr std::size_t ff = 512;    // feed-forward width
constexpr std::size_t vocab = 64;  // logits
constexpr float eps = 1e-5f;

// Quantized weights, in the formats of a Q4_K_M model.
struct weights {
    std::mt19937 rng{20261021};
    random_matrix<tensor_type::f32> g1{1, d, rng, 0.5f, 1.5f};
    random_matrix<tensor_type::q4_k> w_gate{ff, d, rng, 1e-4f, 1e-3f};
    random_matrix<tensor_type::q6_k> w_up{ff, d, rng, 1e-4f, 1e-3f};
    random_matrix<tensor_type::q4_0> w_down{d, ff, rng, 1e-3f, 5e-3f};
    random_matrix<tensor_type::f32> g2{1, d, rng, 0.5f, 1.5f};
    random_matrix<tensor_type::q8_0> w_out{vocab, d, rng, 1e-3f, 5e-3f};
};

// Plain-format weights: the kernels keep activations in float.
struct plain_weights {
    std::mt19937 rng{20261022};
    random_matrix<tensor_type::f32> g1{1, d, rng, 0.5f, 1.5f};
    random_matrix<tensor_type::f32> w_gate{ff, d, rng, 0.01f, 0.1f};
    random_matrix<tensor_type::f16> w_up{ff, d, rng, 0.01f, 0.1f};
    random_matrix<tensor_type::f32> w_down{d, ff, rng, 0.005f, 0.05f};
    random_matrix<tensor_type::f32> g2{1, d, rng, 0.5f, 1.5f};
    random_matrix<tensor_type::f16> w_out{vocab, d, rng, 0.01f, 0.1f};
};

struct decode_graph {
    compute_graph g;
    tensor_id x, residual, probs;
};

template <typename Weights>
decode_graph build(const Weights& w) {
    graph_builder b;
    const tensor_id x = b.input(d);
    const tensor_id h = b.rms_norm(x, w.g1.view(), eps);
    const tensor_id gate = b.matmul(w.w_gate.view(), h);
    const tensor_id up = b.matmul(w.w_up.view(), h);
    const tensor_id act = b.mul(b.silu(gate), up);
    const tensor_id down = b.matmul(w.w_down.view(), act);
    const tensor_id residual = b.add(x, down);
    const tensor_id h2 = b.rms_norm(residual, w.g2.view(), eps);
    const tensor_id logits = b.matmul(w.w_out.view(), h2);
    const tensor_id probs = b.softmax(logits);
    b.output(residual);
    b.output(probs);
    auto g = std::move(b).finalize();
    REQUIRE(g);
    return {std::move(*g), x, residual, probs};
}

std::vector<float> input_vector() {
    std::mt19937 rng(7);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> x(d);
    for (auto& v : x) {
        v = normal(rng);
    }
    return x;
}

template <typename View>
std::vector<float> gemv_unaliased(const View& w, const std::vector<float>& x) {
    using S = typename View::scheme_type;
    std::vector<float> y(w.rows());
    std::vector<std::byte> scratch(kernels::gemv_scratch_bytes<S>(w.cols()));
    kernels::gemv(w, x, y, scratch);
    return y;
}

bool bit_equal(std::span<const float> a, std::span<const float> b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](float u, float v) {
               return std::bit_cast<std::uint32_t>(u) == std::bit_cast<std::uint32_t>(v);
           });
}

// Double-precision reference on the dequantized weights.
template <typename View>
std::vector<double> matvec_ref(const View& w, const std::vector<double>& x) {
    std::vector<double> y(w.rows(), 0.0);
    std::vector<float> row(w.cols());
    for (std::size_t r = 0; r < w.rows(); ++r) {
        w.dequantize_row(r, row);
        for (std::size_t c = 0; c < w.cols(); ++c) {
            y[r] += static_cast<double>(row[c]) * x[c];
        }
    }
    return y;
}

template <typename View>
std::vector<double> rms_norm_ref(const std::vector<double>& x, const View& gamma) {
    double ss = 0.0;
    for (double v : x) {
        ss += v * v;
    }
    const double scale = 1.0 / std::sqrt(ss / static_cast<double>(x.size()) + eps);
    std::vector<float> g(x.size());
    gamma.dequantize_row(0, g);
    std::vector<double> y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        y[i] = x[i] * scale * g[i];
    }
    return y;
}

double max_abs(const std::vector<double>& v) {
    double m = 0.0;
    for (double e : v) {
        m = std::max(m, std::fabs(e));
    }
    return m;
}

}  // namespace

TEST_CASE("graph_session: arena execution equals unaliased op-by-op execution bit for bit", "[graph][session]") {
    const weights w;
    const auto dg = build(w);
    const auto x = input_vector();

    session s{dg.g};
    std::copy(x.begin(), x.end(), s.tensor(dg.x).begin());
    s.run();

    // Op by op, every tensor in its own buffer, same kernels.
    std::vector<float> h(d), act(ff), silu_gate(ff), residual(d), h2(d), probs(vocab);
    kernels::rms_norm(x, w.g1.view(), eps, h);
    const auto gate = gemv_unaliased(w.w_gate.view(), h);
    const auto up = gemv_unaliased(w.w_up.view(), h);
    kernels::silu(gate, silu_gate);
    kernels::mul(silu_gate, up, act);
    const auto down = gemv_unaliased(w.w_down.view(), act);
    kernels::add(x, down, residual);
    kernels::rms_norm(residual, w.g2.view(), eps, h2);
    const auto logits = gemv_unaliased(w.w_out.view(), h2);
    kernels::softmax(logits, probs);

    CHECK(bit_equal(s.tensor(dg.residual), residual));
    CHECK(bit_equal(s.tensor(dg.probs), probs));

    // Aliasing really happened.
    CHECK(s.memory().arena_bytes < s.memory().naive_bytes);
    INFO("arena " << s.memory().arena_bytes << " B vs " << s.memory().naive_bytes << " B unaliased");
}

TEST_CASE("graph_session: computes the intended math (double-precision reference)", "[graph][session]") {
    const plain_weights w;
    const auto dg = build(w);
    const auto xf = input_vector();
    session s{dg.g};
    std::copy(xf.begin(), xf.end(), s.tensor(dg.x).begin());
    s.run();

    const std::vector<double> x(xf.begin(), xf.end());
    const auto h = rms_norm_ref(x, w.g1.view());
    const auto gate = matvec_ref(w.w_gate.view(), h);
    const auto up = matvec_ref(w.w_up.view(), h);
    std::vector<double> act(ff);
    for (std::size_t i = 0; i < ff; ++i) {
        act[i] = gate[i] / (1.0 + std::exp(-gate[i])) * up[i];
    }
    const auto down = matvec_ref(w.w_down.view(), act);
    std::vector<double> residual(d);
    for (std::size_t i = 0; i < d; ++i) {
        residual[i] = x[i] + down[i];
    }
    const auto logits = matvec_ref(w.w_out.view(), rms_norm_ref(residual, w.g2.view()));
    std::vector<double> probs(vocab);
    const double mx = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for (std::size_t i = 0; i < vocab; ++i) {
        probs[i] = std::exp(logits[i] - mx);
        sum += probs[i];
    }
    for (auto& p : probs) {
        p /= sum;
    }

    // Float vs double rounding only.
    const auto r = s.tensor(dg.residual);
    const auto p = s.tensor(dg.probs);
    for (std::size_t i = 0; i < d; ++i) {
        CHECK(std::fabs(r[i] - residual[i]) <= 1e-4 * max_abs(residual));
    }
    double psum = 0.0;
    for (std::size_t i = 0; i < vocab; ++i) {
        CHECK(std::fabs(p[i] - probs[i]) <= 1e-4 * max_abs(probs));
        psum += p[i];
    }
    CHECK(std::fabs(psum - 1.0) < 1e-5);
}

TEST_CASE("graph_session: runs are repeatable with fresh inputs", "[graph][session]") {
    const weights w;
    const auto dg = build(w);
    const auto x = input_vector();
    session s{dg.g};

    std::copy(x.begin(), x.end(), s.tensor(dg.x).begin());
    s.run();
    const std::vector<float> first(s.tensor(dg.probs).begin(), s.tensor(dg.probs).end());

    // The planner may reuse the input's memory during a run, so inputs are
    // rewritten before each run.
    std::copy(x.begin(), x.end(), s.tensor(dg.x).begin());
    s.run();
    CHECK(bit_equal(s.tensor(dg.probs), first));
}

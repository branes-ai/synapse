// Unit tests for kernels::rope and kernels::attention against double-precision
// references. (Their agreement with llama.cpp is checked end to end by
// llama_golden.)

#include <branes/synapse/kernels/attention.hpp>
#include <branes/synapse/numeric/half.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

using namespace branes::synapse;

namespace {

std::vector<float> normal_vector(std::size_t n, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) {
        x = normal(rng);
    }
    return v;
}

double rounded(float x) {
    return static_cast<double>(fp16_to_float(float_to_fp16(x)));
}

}  // namespace

TEST_CASE("kernels_attention: rope rotates each pair by pos * base^(-2i/d) / factor", "[kernels][rope]") {
    constexpr std::size_t head_dim = 64, n_heads = 3;
    constexpr float base = 10000.0f;
    std::mt19937 rng(5);
    const auto x = normal_vector(n_heads * head_dim, rng);
    std::vector<float> factors(head_dim / 2);
    for (std::size_t i = 0; i < factors.size(); ++i) {
        factors[i] = 1.0f + static_cast<float>(i % 4);
    }
    const float theta_scale = kernels::rope_theta_scale(base, head_dim);

    for (bool with_factors : {false, true}) {
        for (std::uint32_t pos : {0u, 1u, 7u, 100u}) {
            std::vector<float> out(x.size());
            kernels::rope(x,
                          out,
                          head_dim,
                          pos,
                          theta_scale,
                          with_factors ? std::span<const float>(factors) : std::span<const float>{});
            for (std::size_t h = 0; h < n_heads; ++h) {
                for (std::size_t i = 0; i < head_dim / 2; ++i) {
                    const double ff = with_factors ? factors[i] : 1.0;
                    const double theta =
                        pos * std::pow(static_cast<double>(base), -2.0 * static_cast<double>(i) / head_dim) / ff;
                    const double x0 = x[h * head_dim + 2 * i], x1 = x[h * head_dim + 2 * i + 1];
                    // ggml builds theta by repeated float multiplication, so
                    // its relative error grows with the pair index.
                    const double tol = 1e-5 * (1.0 + pos);
                    INFO("pos " << pos << " head " << h << " pair " << i);
                    CHECK(std::fabs(out[h * head_dim + 2 * i] - (x0 * std::cos(theta) - x1 * std::sin(theta))) < tol);
                    CHECK(std::fabs(out[h * head_dim + 2 * i + 1] - (x0 * std::sin(theta) + x1 * std::cos(theta))) <
                          tol);
                }
            }
            if (pos == 0) {
                CHECK(out == x);  // theta = 0: the identity, exactly
            }
        }
    }
}

TEST_CASE("kernels_attention: neox rope rotates (x[i], x[i + d/2]) by pos * base^(-2i/d)", "[kernels][rope]") {
    constexpr std::size_t head_dim = 64, n_heads = 2, half = head_dim / 2;
    constexpr float base = 1000000.0f;  // Qwen2's
    std::mt19937 rng(6);
    const auto x = normal_vector(n_heads * head_dim, rng);
    const float theta_scale = kernels::rope_theta_scale(base, head_dim);
    for (std::uint32_t pos : {0u, 1u, 9u, 300u}) {
        std::vector<float> out(x.size());
        kernels::rope(x, out, head_dim, pos, theta_scale, {}, kernels::rope_mode::neox);
        for (std::size_t h = 0; h < n_heads; ++h) {
            for (std::size_t i = 0; i < half; ++i) {
                const double theta =
                    pos * std::pow(static_cast<double>(base), -2.0 * static_cast<double>(i) / head_dim);
                const double x0 = x[h * head_dim + i], x1 = x[h * head_dim + i + half];
                const double tol = 1e-5 * (1.0 + pos);
                INFO("pos " << pos << " head " << h << " pair " << i);
                CHECK(std::fabs(out[h * head_dim + i] - (x0 * std::cos(theta) - x1 * std::sin(theta))) < tol);
                CHECK(std::fabs(out[h * head_dim + i + half] - (x0 * std::sin(theta) + x1 * std::cos(theta))) < tol);
            }
        }
        if (pos == 0) {
            CHECK(out == x);
        }
    }
    // The same rotations as normal mode, applied to the de-interleaved head.
    std::vector<float> interleaved(head_dim), normal(head_dim), neox(head_dim);
    for (std::size_t i = 0; i < half; ++i) {
        interleaved[2 * i] = x[i];
        interleaved[2 * i + 1] = x[i + half];
    }
    kernels::rope(
        std::span<const float>(x).first(head_dim), neox, head_dim, 17, theta_scale, {}, kernels::rope_mode::neox);
    kernels::rope(interleaved, normal, head_dim, 17, theta_scale, {});
    for (std::size_t i = 0; i < half; ++i) {
        CHECK(neox[i] == normal[2 * i]);
        CHECK(neox[i + half] == normal[2 * i + 1]);
    }
}

TEST_CASE("kernels_attention: grouped-query attention over a growing F16 cache", "[kernels][attention]") {
    constexpr std::size_t n_heads = 4, n_kv = 2, hd = 16, max_ctx = 12;
    constexpr std::size_t row = n_kv * hd;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    std::mt19937 rng(9);

    std::vector<std::uint16_t> k_cache(max_ctx * row), v_cache(max_ctx * row);
    const kernels::kv_layer cache{k_cache.data(), v_cache.data(), max_ctx, n_kv, hd};
    std::vector<float> scratch(kernels::attention_scratch_floats(n_heads, n_kv, hd, max_ctx));
    std::vector<std::vector<float>> ks, vs;

    for (std::uint32_t pos = 0; pos < max_ctx; ++pos) {
        const auto q = normal_vector(n_heads * hd, rng);
        ks.push_back(normal_vector(row, rng));
        vs.push_back(normal_vector(row, rng));
        std::vector<float> out(n_heads * hd);
        kernels::attention(q, ks.back(), vs.back(), cache, pos, n_heads, scale, out, scratch);

        // The new row is stored as F16.
        for (std::size_t i = 0; i < row; ++i) {
            CHECK(k_cache[pos * row + i] == float_to_fp16(ks.back()[i]));
            CHECK(v_cache[pos * row + i] == float_to_fp16(vs.back()[i]));
        }

        // Reference: on the F16-rounded q, K, V (what the cache holds), with
        // query head h reading KV head h / (n_heads / n_kv), in double.
        for (std::size_t h = 0; h < n_heads; ++h) {
            const std::size_t g = h / (n_heads / n_kv);
            std::vector<double> p(pos + 1);
            double mx = -INFINITY;
            for (std::size_t t = 0; t <= pos; ++t) {
                double dot = 0.0;
                for (std::size_t d = 0; d < hd; ++d) {
                    dot += rounded(q[h * hd + d]) * rounded(ks[t][g * hd + d]);
                }
                p[t] = dot * scale;
                mx = std::max(mx, p[t]);
            }
            double sum = 0.0;
            for (auto& e : p) {
                e = std::exp(e - mx);
                sum += e;
            }
            for (std::size_t d = 0; d < hd; ++d) {
                double ref = 0.0;
                for (std::size_t t = 0; t <= pos; ++t) {
                    ref += p[t] / sum * rounded(vs[t][g * hd + d]);
                }
                // The kernel also rounds the probabilities to F16 (relative 2^-11).
                INFO("pos " << pos << " head " << h << " dim " << d);
                CHECK(std::fabs(out[h * hd + d] - ref) < 2e-3);
            }
        }
        if (pos == 0) {
            // One position: softmax is exactly 1, so out is V's F16 value.
            for (std::size_t h = 0; h < n_heads; ++h) {
                for (std::size_t d = 0; d < hd; ++d) {
                    CHECK(out[h * hd + d] == static_cast<float>(rounded(vs[0][h / 2 * hd + d])));
                }
            }
        }
    }
}

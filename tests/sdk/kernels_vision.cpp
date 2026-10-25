// Unit tests for the image-encoder kernels: layer_norm, gelu, pixel_shuffle
// and bidirectional self-attention. kernels_golden checks gelu and layer_norm
// against ggml itself; these check the math and the layouts.

#include <branes/synapse/kernels/attention.hpp>
#include <branes/synapse/kernels/elementwise.hpp>
#include <branes/synapse/numeric/half.hpp>

#include "../support/random_weights.hpp"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <random>
#include <vector>

using namespace branes::synapse;
using synapse_test::random_matrix;
using synapse_test::tensor_type;

TEST_CASE("kernels_vision: layer_norm normalizes, then scales and shifts", "[kernels][vision]") {
    std::mt19937 rng(3);
    std::normal_distribution<float> normal(2.0f, 3.0f);
    constexpr std::size_t n = 300;  // a partial 256-element chunk
    std::vector<float> x(n);
    for (auto& v : x) {
        v = normal(rng);
    }
    const random_matrix<tensor_type::f32> gamma(1, n, rng, 0.5f, 1.5f);
    const random_matrix<tensor_type::f32> beta(1, n, rng, -0.5f, 0.5f);
    std::vector<float> y(n);
    std::vector<float> yb(n);
    kernels::layer_norm(x, gamma.view(), 1e-6f, y);
    kernels::layer_norm(x, gamma.view(), beta.view(), 1e-6f, yb);

    double mean = 0.0;
    for (float v : x) {
        mean += v;
    }
    mean /= n;
    double var = 0.0;
    for (float v : x) {
        var += (v - mean) * (v - mean);
    }
    var /= n;
    std::vector<float> g(n);
    std::vector<float> b(n);
    gamma.view().decode_row(0, g.data());
    beta.view().decode_row(0, b.data());
    for (std::size_t i = 0; i < n; ++i) {
        const double ref = (x[i] - mean) / std::sqrt(var + 1e-6) * g[i];
        CHECK(std::fabs(y[i] - ref) <= 1e-5 * std::fabs(ref) + 1e-6);
        // The bias is a separate add after the multiply, as ggml_add after ggml_mul.
        CHECK(std::bit_cast<std::uint32_t>(yb[i]) == std::bit_cast<std::uint32_t>(y[i] + b[i]));
    }
}

TEST_CASE("kernels_vision: gelu passes through binary16 and cuts off at +-10", "[kernels][vision]") {
    const std::vector<float> x{-11.0f, -10.0f, -9.99f, -1.0f, 0.0f, 0.3333f, 1.0f, 9.99f, 10.0f, 11.0f};
    std::vector<float> y(x.size());
    kernels::gelu(x, y);
    CHECK(y[0] == 0.0f);
    CHECK(y[1] == 0.0f);
    CHECK(y[4] == 0.0f);
    CHECK(y[8] == 10.0f);
    CHECK(y[9] == 11.0f);
    for (std::size_t i : {2u, 3u, 5u, 6u, 7u}) {
        INFO("x = " << x[i]);
        // Every value inside the cutoffs is a binary16 value...
        CHECK(fp16_to_float(float_to_fp16(y[i])) == y[i]);
        // ...within half-precision rounding of the exact tanh GELU.
        const double h = fp16_to_float(float_to_fp16(x[i]));
        const double ref = 0.5 * h * (1.0 + std::tanh(std::sqrt(2.0 / std::numbers::pi) * (h + 0.044715 * h * h * h)));
        CHECK(std::fabs(y[i] - ref) <= 1e-3 * std::fabs(ref) + 1e-4);
    }
}

TEST_CASE("kernels_vision: round_to_fp16 equals float_to_fp16 at every rounding boundary", "[kernels][vision][half]") {
    // Each binary16 value, its float neighbours, and the floats on and either
    // side of the halfway point to the next value: every place where rounding
    // can go wrong. Plus the whole of [-2^-24, 2^-24] at float resolution
    // around the subnormal cutoff, and infinities.
    std::size_t mismatches = 0;
    const auto check = [&](float f) { mismatches += kernels::detail::round_to_fp16(f) != float_to_fp16(f) ? 1 : 0; };
    for (std::uint32_t bits = 0; bits <= 0xffff; ++bits) {
        const float f = fp16_to_float(static_cast<std::uint16_t>(bits));
        if (std::isnan(f)) {
            continue;
        }
        const float next = fp16_to_float(static_cast<std::uint16_t>(bits + 1));
        const float mid = std::isfinite(next) && !std::isnan(next) ? (f + next) / 2 : f;
        for (float g : {f, mid}) {
            check(g);
            check(std::nextafter(g, 0.0f));
            check(std::nextafter(g, std::copysign(INFINITY, g)));
        }
    }
    for (float f = 0x1p-26f; f <= 0x1p-23f; f = std::nextafter(f, 1.0f)) {
        check(f);
        check(-f);
    }
    check(65520.0f);
    check(std::nextafter(65520.0f, 0.0f));
    check(INFINITY);
    check(-INFINITY);
    CHECK(mismatches == 0);
    CHECK((kernels::detail::round_to_fp16(NAN) & 0x7c00u) == 0x7c00u);
}

TEST_CASE("kernels_vision: gelu's table equals evaluating ggml_gelu_f32 per element", "[kernels][vision]") {
    // Every binary16 value inside the cutoffs, and the floats either side of
    // it, against the formula gelu evaluated before it had a table.
    constexpr float coef_a = 0.044715f;
    constexpr float sqrt_2_over_pi = 0.79788456080286535587989211986876f;
    std::vector<float> x;
    for (std::uint32_t bits = 0; bits <= 0xffff; ++bits) {
        const float f = fp16_to_float(static_cast<std::uint16_t>(bits));
        if (std::isfinite(f) && std::fabs(f) <= 10.0f) {
            x.insert(x.end(), {f, std::nextafter(f, -11.0f), std::nextafter(f, 11.0f)});
        }
    }
    std::vector<float> y(x.size());
    kernels::gelu(x, y);  // x.size() is not a multiple of the chunk
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        float want = x[i];
        if (x[i] <= -10.0f) {
            want = 0.0f;
        } else if (x[i] < 10.0f) {
            const float h = fp16_to_float(float_to_fp16(x[i]));
            const float g = 0.5f * h * (1.0f + std::tanh(sqrt_2_over_pi * h * (1.0f + coef_a * h * h)));
            want = fp16_to_float(float_to_fp16(g));
        }
        mismatches += std::bit_cast<std::uint32_t>(y[i]) != std::bit_cast<std::uint32_t>(want) ? 1 : 0;
    }
    CHECK(mismatches == 0);
}

namespace {

// ggml's build_patch_merge_permute, emulated op by op on logical indices:
// reshape_3d(n*s, w/s, h), permute(0,2,1,3), cont_3d(n*s*s, h/s, w/s),
// permute(0,2,1,3), cont_2d. Element (i0, i1, i2) of a contiguous tensor with
// ne = {a, b, c} is at i0 + a * (i1 + b * i2).
std::vector<float>
ggml_patch_merge(const std::vector<float>& in, std::size_t n, std::size_t w, std::size_t h, std::size_t s) {
    // After reshape_3d: ne {n*s, w/s, h}, same memory. permute(0,2,1,3) swaps
    // dims 1 and 2: logical (i0, i1, i2) of the view reads (i0, i2, i1).
    // cont_3d materializes the view in its logical order {n*s, h, w/s} and
    // reshapes to {n*s*s, h/s, w/s}.
    std::vector<float> a(in.size());
    std::size_t k = 0;
    for (std::size_t x2 = 0; x2 < w / s; ++x2) {
        for (std::size_t y = 0; y < h; ++y) {
            for (std::size_t e = 0; e < n * s; ++e) {
                a[k++] = in[e + n * s * (x2 + (w / s) * y)];
            }
        }
    }
    // a has ne {n*s*s, h/s, w/s}; permute(0,2,1,3) and cont_2d give
    // ne {n*s*s, (w/s) * (h/s)} in the logical order (e, x2, y2).
    std::vector<float> out(in.size());
    k = 0;
    for (std::size_t y2 = 0; y2 < h / s; ++y2) {
        for (std::size_t x2 = 0; x2 < w / s; ++x2) {
            for (std::size_t e = 0; e < n * s * s; ++e) {
                out[k++] = a[e + n * s * s * (y2 + (h / s) * x2)];
            }
        }
    }
    return out;
}

}  // namespace

TEST_CASE("kernels_vision: pixel_shuffle is llama.cpp's patch merge", "[kernels][vision]") {
    for (auto [w, h, s, n] : {std::array<std::size_t, 4>{6, 4, 2, 3}, {4, 4, 2, 1}, {6, 9, 3, 2}, {8, 4, 4, 5}}) {
        INFO("grid " << w << " x " << h << ", s " << s << ", n " << n);
        std::vector<float> in(w * h * n);
        for (std::size_t i = 0; i < in.size(); ++i) {
            in[i] = static_cast<float>(i);
        }
        std::vector<float> out(in.size());
        kernels::pixel_shuffle(in, n, w, h, s, out);
        CHECK(out == ggml_patch_merge(in, n, w, h, s));
    }
}

TEST_CASE("kernels_vision: self-attention attends every row to every row", "[kernels][vision]") {
    std::mt19937 rng(5);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    constexpr std::size_t n = 37;
    constexpr std::size_t heads = 3;
    constexpr std::size_t hd = 8;
    constexpr std::size_t row = heads * hd;
    std::vector<float> q(n * row);
    std::vector<float> k(n * row);
    std::vector<float> v(n * row);
    for (auto* t : {&q, &k, &v}) {
        for (auto& x : *t) {
            x = normal(rng);
        }
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    std::vector<float> out(n * row);
    std::vector<float> scores(n);
    for (std::size_t h = 0; h < heads; ++h) {
        kernels::self_attend_head(q, k, v, n, row, hd, h, 0, n, scale, out, scores);
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t h = 0; h < heads; ++h) {
            std::vector<double> p(n);
            double max = -1e300;
            for (std::size_t j = 0; j < n; ++j) {
                double d = 0.0;
                for (std::size_t e = 0; e < hd; ++e) {
                    d += static_cast<double>(q[i * row + h * hd + e]) * k[j * row + h * hd + e];
                }
                p[j] = d * scale;
                max = std::max(max, p[j]);
            }
            double sum = 0.0;
            for (auto& x : p) {
                x = std::exp(x - max);
                sum += x;
            }
            for (std::size_t e = 0; e < hd; ++e) {
                double ref = 0.0;
                for (std::size_t j = 0; j < n; ++j) {
                    ref += p[j] / sum * v[j * row + h * hd + e];
                }
                CHECK(std::fabs(out[i * row + h * hd + e] - ref) <= 1e-5);
            }
        }
    }

    // Splitting the queries into ranges changes nothing, bit for bit (D1).
    std::vector<float> split(n * row);
    for (std::size_t h = 0; h < heads; ++h) {
        kernels::self_attend_head(q, k, v, n, row, hd, h, 0, 10, scale, split, scores);
        kernels::self_attend_head(q, k, v, n, row, hd, h, 10, 11, scale, split, scores);
        kernels::self_attend_head(q, k, v, n, row, hd, h, 11, n, scale, split, scores);
    }
    CHECK(split == out);
}

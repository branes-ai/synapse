#pragma once
/// @file elementwise.hpp
/// @brief The vector operations of a transformer decode step, besides matmul.
///
/// All kernels take float spans, never allocate, and never throw. Outputs may
/// not alias inputs (the memory planner guarantees this), except where noted. Numerics follow
/// ggml's CPU ops: rms_norm and layer_norm accumulate in double, softmax
/// subtracts the maximum before exponentiating, and gelu goes through binary16
/// as ggml's lookup table does.

#include <branes/synapse/numeric/half.hpp>
#include <branes/synapse/quant/scheme.hpp>

#include <mtl/mat/block_matrix_view.hpp>
#include <mtl/simd/batch.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

namespace branes::synapse::kernels {

/// out = a + b.  Pre: equal sizes.  out may alias a or b.
inline void add(std::span<const float> a, std::span<const float> b, std::span<float> out) noexcept {
    assert(a.size() == b.size() && a.size() == out.size());  // Pre
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = a[i] + b[i];
    }
}

/// out = x + bias, with bias a 1 x n weight row in any decodable scheme
/// (usually F32), as ggml_add adds a bias vector to each row. The bias is
/// decoded straight into out, so no scratch is needed.
///   Pre: x.size() == out.size() == bias.num_cols(), bias.num_rows() == 1; out does not alias x.
template <quant::Scheme S>
void add_bias(std::span<const float> x, const mtl::mat::block_matrix_view<S>& bias, std::span<float> out) noexcept {
    assert(bias.num_rows() == 1 && x.size() == bias.num_cols() && x.size() == out.size());  // Pre
    bias.decode_row(0, out.data());
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = x[i] + out[i];
    }
}

/// out = a * b (element-wise).  Pre: equal sizes.
inline void mul(std::span<const float> a, std::span<const float> b, std::span<float> out) noexcept {
    assert(a.size() == b.size() && a.size() == out.size());  // Pre
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = a[i] * b[i];
    }
}

/// out = x * s.  Pre: equal sizes.
inline void scale(std::span<const float> x, float s, std::span<float> out) noexcept {
    assert(x.size() == out.size());  // Pre
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = x[i] * s;
    }
}

/// The attention temperature of position pos: 1 + scale * ln(floor(pos /
/// floor_len) + 1), computed as llama.cpp's llm_graph_input_attn_temp does
/// (a float quotient, the logarithm in double, rounded to float).
///   Pre: floor_len >= 1.
inline float attn_temperature(std::uint32_t pos, float scale, std::size_t floor_len) noexcept {
    assert(floor_len >= 1);  // Pre
    const float q = std::floor(static_cast<float>(pos) / static_cast<float>(floor_len));
    return static_cast<float>(std::log(q + 1.0) * scale + 1.0);
}

/// out = x * sigmoid(x).  Pre: equal sizes.
inline void silu(std::span<const float> x, std::span<float> out) noexcept {
    assert(x.size() == out.size());  // Pre
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

namespace detail {

/// float to binary16, rounding to nearest even, without branches: the
/// arithmetic of ggml's fallback GGML_FP32_TO_FP16 (after Maratyszcza's FP16
/// library). Scaling |f| by 2^112 and then 2^-110 sends what overflows to
/// infinity; adding a power of two at the binary16 exponent of f leaves
/// exactly the bits binary16 keeps, rounded by the float add. It equals
/// float_to_fp16 for every float, NaN payloads included, and loops of it
/// vectorize.
///   Pre: the default floating-point environment (round to nearest, no
///   flush-to-zero).
inline std::uint16_t round_to_fp16(float f) noexcept {
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
    float base = (std::fabs(f) * scale_to_inf) * scale_to_zero;
    const std::uint32_t w = std::bit_cast<std::uint32_t>(f);
    const std::uint32_t shl1_w = w + w;
    const std::uint32_t sign = w & 0x80000000u;
    const std::uint32_t bias = std::max(shl1_w & 0xff000000u, 0x71000000u);
    base = std::bit_cast<float>((bias >> 1) + 0x07800000u) + base;
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(base);
    const std::uint32_t exp_bits = (bits >> 13) & 0x00007c00u;
    const std::uint32_t mantissa_bits = bits & 0x00000fffu;
    const std::uint32_t nonsign = exp_bits + mantissa_bits;
    // NaN keeps its top payload bits, quiet, as float_to_fp16 does (ggml's
    // fallback returns 0x7e00 instead). A mask, not a ?: between the two
    // results, which GCC will not vectorize.
    const std::uint32_t nan = 0u - static_cast<std::uint32_t>(shl1_w > 0xff000000u);
    const std::uint32_t nan_bits = 0x7e00u | ((w >> 13) & 0x3ffu);
    return static_cast<std::uint16_t>((sign >> 16) | (nonsign & ~nan) | (nan_bits & nan));
}

/// ggml_gelu_f32 of every binary16 value h, rounded to binary16 and widened
/// back: what gelu returns for any x inside the cutoffs that rounds to h.
/// ggml_table_gelu_f16 holds the same values as halves; holding floats saves
/// the widening per element. Built on first use, in static storage, so a
/// session that runs gelu still allocates nothing.
struct gelu_table_t {
    std::array<float, 65536> values;
    // Fills values in place, keeping 256 KiB off the stack of whichever
    // thread runs gelu first.
    gelu_table_t() noexcept {
        constexpr float coef_a = 0.044715f;
        constexpr float sqrt_2_over_pi = 0.79788456080286535587989211986876f;
        for (std::size_t i = 0; i < values.size(); ++i) {
            const float h = fp16_to_float(static_cast<std::uint16_t>(i));
            const float g = 0.5f * h * (1.0f + std::tanh(sqrt_2_over_pi * h * (1.0f + coef_a * h * h)));
            values[i] = fp16_to_float(float_to_fp16(g));
        }
    }
};

inline const std::array<float, 65536>& gelu_table() noexcept {
    static const gelu_table_t table;
    return table.values;
}

/// Elements gelu rounds at a time: the binary16 patterns of a chunk go to a
/// stack buffer first, so the rounding loop vectorizes apart from the lookups.
inline constexpr std::size_t gelu_chunk = 256;

}  // namespace detail

/// out = gelu(x) with the tanh approximation, computed as ggml's CPU
/// ggml_vec_gelu_f32 does: x <= -10 gives 0, x >= 10 gives x, and otherwise x
/// is rounded to binary16 and the result is ggml_gelu_f32 of that half,
/// rounded to binary16, read from a 65536-entry table as ggml reads it.
///   Pre: equal sizes; the default floating-point environment.
inline void gelu(std::span<const float> x, std::span<float> out) noexcept {
    assert(x.size() == out.size());  // Pre
    const std::array<float, 65536>& table = detail::gelu_table();
    std::array<std::uint16_t, detail::gelu_chunk> h{};
    for (std::size_t i = 0; i < x.size(); i += detail::gelu_chunk) {
        const std::size_t len = std::min(detail::gelu_chunk, x.size() - i);
        for (std::size_t j = 0; j < len; ++j) {
            h[j] = detail::round_to_fp16(x[i + j]);
        }
        for (std::size_t j = 0; j < len; ++j) {
            const float v = x[i + j];
            const float t = table[h[j]];
            const float above = v >= 10.0f ? v : t;  // selects, not branches: the loop vectorizes
            out[i + j] = v <= -10.0f ? 0.0f : above;
        }
    }
}

/// out = softmax(x).  Pre: equal, non-zero sizes.  May run in place (out == x).
///
/// The maximum is taken a vector at a time (exact in any order), and the
/// exponentials are summed in a pass of their own, in element order, so the
/// double sum's latency does not hold up the exponentials.
inline void softmax(std::span<const float> x, std::span<float> out) noexcept {
    assert(x.size() == out.size() && !x.empty());  // Pre
    using lanes = mtl::simd::batch<float>;
    constexpr std::size_t w = lanes::size;
    const std::size_t n = x.size();
    float max = x[0];
    std::size_t i = 0;
    for (; i + w <= n; i += w) {
        max = std::max(max, reduce_max(lanes::load_unaligned(x.data() + i)));
    }
    for (; i < n; ++i) {
        max = std::max(max, x[i]);
    }
    for (i = 0; i < n; ++i) {
        out[i] = std::exp(x[i] - max);
    }
    double sum = 0.0;
    for (const float v : out) {
        sum += v;
    }
    const auto scale = static_cast<float>(1.0 / sum);
    for (i = 0; i + w <= n; i += w) {
        (lanes::load_unaligned(out.data() + i) * lanes(scale)).store_unaligned(out.data() + i);
    }
    for (; i < n; ++i) {
        out[i] *= scale;
    }
}

/// out = x / rms(x) * gamma, rms(x) = sqrt(mean(x^2) + eps), as llama.cpp
/// computes RMSNorm followed by the multiply by its weight. gamma is a 1 x n
/// weight row in any decodable scheme (usually F32), decoded 256 at a time.
///   Pre: x.size() == out.size() == gamma.num_cols(), gamma.num_rows() == 1, x non-empty.
template <quant::Scheme S>
void rms_norm(std::span<const float> x,
              const mtl::mat::block_matrix_view<S>& gamma,
              float eps,
              std::span<float> out) noexcept {
    assert(gamma.num_rows() == 1 && x.size() == gamma.num_cols() && x.size() == out.size() && !x.empty());  // Pre
    double sum = 0.0;
    for (float v : x) {
        sum += static_cast<double>(v * v);
    }
    const auto mean = static_cast<float>(sum / static_cast<double>(x.size()));
    const float scale = 1.0f / std::sqrt(mean + eps);

    const std::span<const std::byte> g = gamma.row(0);
    constexpr std::size_t chunk = 256;
    std::array<float, chunk> gbuf{};
    for (std::size_t i = 0; i < x.size(); i += chunk) {
        const std::size_t len = std::min(chunk, x.size() - i);
        for (std::size_t j = 0; j < len; j += S::block_size) {
            S::dequantize(g.data() + (i + j) / S::block_size * S::type_size, gbuf.data() + j);
        }
        for (std::size_t j = 0; j < len; ++j) {
            out[i + j] = (x[i + j] * scale) * gbuf[j];
        }
    }
}

namespace detail {

/// Decode elements [i, i + len) of the 1 x n weight row w into buf.
///   Pre: i and len are multiples of S::block_size (any multiple of 256 is,
///   and so is a row's tail); len <= buf.size().
template <quant::Scheme S>
void decode_span(const mtl::mat::block_matrix_view<S>& w, std::size_t i, std::size_t len, float* buf) noexcept {
    const std::span<const std::byte> row = w.row(0);
    for (std::size_t j = 0; j < len; j += S::block_size) {
        S::dequantize(row.data() + (i + j) / S::block_size * S::type_size, buf + j);
    }
}

inline constexpr std::size_t norm_chunk = 256;

}  // namespace detail

/// out = (x - mean(x)) / sqrt(var(x) + eps) * gamma, as llama.cpp's ggml_norm
/// followed by the multiply by its weight: the sum and the sum of squared
/// deviations in double, the mean and the variance rounded to float. gamma
/// is a 1 x n weight row in any decodable scheme (usually F32).
///   Pre: x.size() == out.size() == gamma.num_cols(), gamma.num_rows() == 1,
///        x non-empty; out does not alias x.
template <quant::Scheme G>
void layer_norm(std::span<const float> x,
                const mtl::mat::block_matrix_view<G>& gamma,
                float eps,
                std::span<float> out) noexcept {
    assert(gamma.num_rows() == 1 && x.size() == gamma.num_cols() && x.size() == out.size() && !x.empty());  // Pre
    double sum = 0.0;
    for (float v : x) {
        sum += static_cast<double>(v);
    }
    const float mean = static_cast<float>(sum) / static_cast<float>(x.size());
    double sq = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        out[i] = x[i] - mean;
        sq += static_cast<double>(out[i] * out[i]);
    }
    const auto variance = static_cast<float>(sq / static_cast<double>(x.size()));
    const float scale = 1.0f / std::sqrt(variance + eps);

    std::array<float, detail::norm_chunk> buf{};
    for (std::size_t i = 0; i < x.size(); i += detail::norm_chunk) {
        const std::size_t len = std::min(detail::norm_chunk, x.size() - i);
        detail::decode_span(gamma, i, len, buf.data());
        for (std::size_t j = 0; j < len; ++j) {
            out[i + j] = (out[i + j] * scale) * buf[j];
        }
    }
}

/// layer_norm(x, gamma, eps, out) followed by the add of the 1 x n bias row
/// beta, as ggml_norm, ggml_mul and ggml_add compute a LayerNorm with bias.
///   Pre: as above, and beta.num_rows() == 1, beta.num_cols() == x.size().
template <quant::Scheme G, quant::Scheme B>
void layer_norm(std::span<const float> x,
                const mtl::mat::block_matrix_view<G>& gamma,
                const mtl::mat::block_matrix_view<B>& beta,
                float eps,
                std::span<float> out) noexcept {
    assert(beta.num_rows() == 1 && beta.num_cols() == x.size());  // Pre
    layer_norm(x, gamma, eps, out);
    std::array<float, detail::norm_chunk> buf{};
    for (std::size_t i = 0; i < x.size(); i += detail::norm_chunk) {
        const std::size_t len = std::min(detail::norm_chunk, x.size() - i);
        detail::decode_span(beta, i, len, buf.data());
        for (std::size_t j = 0; j < len; ++j) {
            out[i + j] = out[i + j] + buf[j];
        }
    }
}

/// Merge each s x s block of a grid_w x grid_h grid of patch rows (n floats
/// each, patch x + grid_w * y) into one row of s^2 * n floats, as llama.cpp's
/// build_patch_merge_permute: out row bx + (grid_w / s) * by holds patch
/// (bx * s + dx, by * s + dy) at offset (dy * s + dx) * n.
///   Pre: grid_w and grid_h multiples of s >= 1; in.size() == grid_w * grid_h * n
///        == out.size(); out does not alias in.
inline void pixel_shuffle(std::span<const float> in,
                          std::size_t n,
                          std::size_t grid_w,
                          std::size_t grid_h,
                          std::size_t s,
                          std::span<float> out) noexcept {
    assert(s >= 1 && grid_w % s == 0 && grid_h % s == 0);  // Pre
    assert(in.size() == grid_w * grid_h * n && out.size() == in.size());
    const std::size_t bw = grid_w / s;
    for (std::size_t by = 0; by < grid_h / s; ++by) {
        for (std::size_t bx = 0; bx < bw; ++bx) {
            float* dst = out.data() + (bx + bw * by) * s * s * n;
            for (std::size_t dy = 0; dy < s; ++dy) {
                for (std::size_t dx = 0; dx < s; ++dx) {
                    const float* src = in.data() + ((bx * s + dx) + grid_w * (by * s + dy)) * n;
                    std::copy_n(src, n, dst + (dy * s + dx) * n);
                }
            }
        }
    }
}

}  // namespace branes::synapse::kernels

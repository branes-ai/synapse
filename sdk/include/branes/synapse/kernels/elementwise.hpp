#pragma once
/// @file elementwise.hpp
/// @brief The vector operations of a transformer decode step, besides matmul.
///
/// All kernels take float spans, never allocate, and never throw. Outputs may
/// not alias inputs (the memory planner guarantees this), except where noted. Numerics follow
/// ggml's CPU ops: rms_norm accumulates the sum of squares in double, and
/// softmax subtracts the maximum before exponentiating.

#include <branes/synapse/quant/scheme.hpp>

#include <mtl/mat/block_matrix_view.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

namespace branes::synapse::kernels {

/// out = a + b.  Pre: equal sizes.
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

/// out = softmax(x).  Pre: equal, non-zero sizes.  May run in place (out == x).
inline void softmax(std::span<const float> x, std::span<float> out) noexcept {
    assert(x.size() == out.size() && !x.empty());  // Pre
    const float max = *std::max_element(x.begin(), x.end());
    double sum = 0.0;
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = std::exp(x[i] - max);
        sum += out[i];
    }
    const auto scale = static_cast<float>(1.0 / sum);
    for (auto& v : out) {
        v *= scale;
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

}  // namespace branes::synapse::kernels

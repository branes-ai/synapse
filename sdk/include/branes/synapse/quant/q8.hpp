#pragma once
// Ported from ggml (MIT License, The ggml authors); see THIRD_PARTY_NOTICES.md.
/// @file q8.hpp
/// @brief 8-bit activation formats and the quantizers that produce them.
///
/// The fused kernels never expand weights to float. Instead the activation
/// vector is quantized once per matrix-vector product into 8-bit blocks, and
/// each weight block's dot product runs in exact integer arithmetic:
///
///   Q8_0  32 values: x ~= d * q,  d = binary16, q int8          (for Q4_0, Q8_0)
///   Q8_K 256 values: x ~= d * q,  d = float, q int8, plus the sum of each
///        16 consecutive q (bsums), which lets the K-quant kernels fold
///        their offsets and mins out of the inner loop             (for Q4_K, Q6_K)
///
/// The quantizers mirror ggml's quantize_row_q8_0_ref / quantize_row_q8_K_ref
/// exactly (same scale, same rounding), so the activations are bit-identical
/// to the ones llama.cpp's reference path feeds its kernels.

#include <branes/synapse/numeric/half.hpp>
#include <branes/synapse/quant/blocks.hpp>

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace branes::synapse::quant {

struct block_q8_k {
    float d;                        ///< scale
    std::int8_t qs[qk_k];           ///< quants
    std::int16_t bsums[qk_k / 16];  ///< sum of each 16 consecutive quants
};
static_assert(sizeof(block_q8_k) == 292);

namespace detail {

/// Round to nearest, ties to even, for |v| <= 2^22 (ggml's nearest_int).
/// Adding 1.5 * 2^23 places the integer in the low mantissa bits.
inline int nearest_int(float v) noexcept {
    assert(std::fabs(v) <= 4194303.0f);
    const auto bits = std::bit_cast<std::int32_t>(v + 12582912.0f);
    return (bits & 0x007fffff) - 0x00400000;
}

}  // namespace detail

/// Quantize x (x.size() == 32 * y.size()) into Q8_0 blocks.
///   Pre: x.size() == y.size() * 32.
inline void quantize_row_q8_0(std::span<const float> x, std::span<block_q8_0> y) noexcept {
    assert(x.size() == y.size() * qk_legacy);  // Pre
    for (std::size_t i = 0; i < y.size(); ++i) {
        const float* xb = x.data() + i * qk_legacy;
        float amax = 0.0f;
        for (std::size_t j = 0; j < qk_legacy; ++j) {
            amax = std::max(amax, std::fabs(xb[j]));
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        y[i].d = float_to_fp16(d);
        for (std::size_t j = 0; j < qk_legacy; ++j) {
            y[i].qs[j] = static_cast<std::int8_t>(std::round(xb[j] * id));
        }
    }
}

/// Quantize x into Q8_K blocks.
///   Pre: x.size() == y.size() * 256.
inline void quantize_row_q8_k(std::span<const float> x, std::span<block_q8_k> y) noexcept {
    assert(x.size() == y.size() * qk_k);  // Pre
    for (std::size_t i = 0; i < y.size(); ++i) {
        const float* xb = x.data() + i * qk_k;
        float max = 0.0f;  // the signed value of largest magnitude
        float amax = 0.0f;
        for (std::size_t j = 0; j < qk_k; ++j) {
            const float ax = std::fabs(xb[j]);
            if (ax > amax) {
                amax = ax;
                max = xb[j];
            }
        }
        if (amax == 0.0f) {
            y[i] = block_q8_k{};  // ggml leaves bsums stale here; zero them
            continue;
        }
        // -127 rather than -128: keeps every quant in [-127, 127].
        const float iscale = -127.0f / max;
        for (std::size_t j = 0; j < qk_k; ++j) {
            y[i].qs[j] = static_cast<std::int8_t>(std::min(127, detail::nearest_int(iscale * xb[j])));
        }
        for (std::size_t j = 0; j < qk_k / 16; ++j) {
            int sum = 0;
            for (std::size_t l = 0; l < 16; ++l) {
                sum += y[i].qs[j * 16 + l];
            }
            y[i].bsums[j] = static_cast<std::int16_t>(sum);
        }
        y[i].d = 1.0f / iscale;
    }
}

}  // namespace branes::synapse::quant

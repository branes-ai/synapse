#pragma once
/// @file dot.hpp
/// @brief Fused dot products of one encoded weight row with quantized activations.
///
/// Each kernel walks a row of weight blocks next to the matching activation
/// blocks. A block's dot product is computed exactly in int32, then scaled
/// into a float accumulator once per block, so weights are never expanded to
/// float. The integer arithmetic follows ggml's ggml_vec_dot_*_q8_*_generic:
///
///   Q4_0 x Q8_0  sum (q - 8) a
///   Q8_0 x Q8_0  sum q a
///   Q4_K x Q8_K  sum_j sc_j dot(q_j, a_j) - dmin/d sum_j m_j sum(a_j)
///                (sub-block sums come from Q8_K's bsums)
///   Q6_K x Q8_K  sum_j sc_j dot(q_j - 32, a_j)
///
/// Every block dot is one 16-bit widening dot product through
/// mtl::simd::reduce_dot_widen, which lowers to vpmaddwd (x86) / SMLAL (NEON)
/// when MTL5's Highway backend is on and to a scalar loop otherwise. Weights
/// are unpacked to int16 with their offset (and, for the K formats, their
/// sub-block scale) folded in, so each 256-weight K block is one dot instead
/// of 8 or 16 short ones; activations are widened to int16 once per gemv (see
/// activations<>). No overflow is possible: |w| <= 4096 and |a| <= 127, so a
/// block sums to at most 256 * 4096 * 127 < 2^31.
///
/// The per-block integer sums are exact and so independent of the SIMD path;
/// only the float accumulation across blocks is order-dependent.

#include <branes/synapse/numeric/half.hpp>
#include <branes/synapse/quant/blocks.hpp>
#include <branes/synapse/quant/q8.hpp>
#include <branes/synapse/quant/scheme.hpp>

#include <mtl/simd/algorithm.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace branes::synapse::kernels {

/// Activations prepared once per matrix-vector product: the quantized blocks
/// (in ggml's format) plus their quants widened to int16, the operand type of
/// the 16-bit widening dot. Widening once per gemv, not once per row, keeps
/// the per-row work to unpacking weights and one dot per block.
template <typename A>
struct activations;

template <>
struct activations<quant::block_q8_0> {
    const quant::block_q8_0* blocks;
    const std::int16_t* wide;  ///< 32 per block
};

template <>
struct activations<quant::block_q8_k> {
    const quant::block_q8_k* blocks;
    const std::int16_t* wide;  ///< 256 per block
};

template <>
struct activations<float> {
    const float* x;
};

/// Q4_0 weights (nb blocks at x) . Q8_0 activations.
inline float vec_dot_q4_0_q8_0(const std::byte* x, activations<quant::block_q8_0> a, std::size_t nb) noexcept {
    float sumf = 0.0f;
    for (std::size_t i = 0; i < nb; ++i) {
        const auto b = quant::detail::load_block<quant::block_q4_0>(x + i * sizeof(quant::block_q4_0));
        std::array<std::int16_t, quant::qk_legacy> w{};  // q - 8
        for (std::size_t j = 0; j < quant::qk_legacy / 2; ++j) {
            w[j] = static_cast<std::int16_t>((b.qs[j] & 0x0f) - 8);
            w[j + quant::qk_legacy / 2] = static_cast<std::int16_t>((b.qs[j] >> 4) - 8);
        }
        const std::int32_t sumi =
            mtl::simd::reduce_dot_widen<std::int32_t>(w.data(), a.wide + i * quant::qk_legacy, quant::qk_legacy);
        sumf += static_cast<float>(sumi) * fp16_to_float(b.d) * fp16_to_float(a.blocks[i].d);
    }
    return sumf;
}

/// Q8_0 weights . Q8_0 activations.
inline float vec_dot_q8_0_q8_0(const std::byte* x, activations<quant::block_q8_0> a, std::size_t nb) noexcept {
    float sumf = 0.0f;
    for (std::size_t i = 0; i < nb; ++i) {
        const auto b = quant::detail::load_block<quant::block_q8_0>(x + i * sizeof(quant::block_q8_0));
        std::array<std::int16_t, quant::qk_legacy> w{};
        for (std::size_t j = 0; j < quant::qk_legacy; ++j) {
            w[j] = b.qs[j];
        }
        const std::int32_t sumi =
            mtl::simd::reduce_dot_widen<std::int32_t>(w.data(), a.wide + i * quant::qk_legacy, quant::qk_legacy);
        sumf += static_cast<float>(sumi) * (fp16_to_float(b.d) * fp16_to_float(a.blocks[i].d));
    }
    return sumf;
}

/// Q4_K weights . Q8_K activations.
inline float vec_dot_q4_k_q8_k(const std::byte* x, activations<quant::block_q8_k> a, std::size_t nb) noexcept {
    float sumf = 0.0f;
    for (std::size_t i = 0; i < nb; ++i) {
        const auto b = quant::detail::load_block<quant::block_q4_k>(x + i * sizeof(quant::block_q4_k));
        const quant::block_q8_k& y = a.blocks[i];
        std::array<std::uint8_t, 8> sc{};
        std::array<std::uint8_t, 8> m{};
        for (std::size_t j = 0; j < 8; ++j) {
            quant::q4_k_scale_min(j, b.scales, sc[j], m[j]);
        }
        // Weight operand sc_j * q (<= 63 * 15): sub-block 2k is the low
        // nibbles of qs[32k..32k+31], sub-block 2k+1 the high nibbles.
        std::array<std::int16_t, quant::qk_k> w{};
        for (std::size_t k = 0; k < 4; ++k) {
            const std::uint8_t* q = b.qs + 32 * k;
            for (std::size_t l = 0; l < 32; ++l) {
                w[64 * k + l] = static_cast<std::int16_t>(sc[2 * k] * (q[l] & 0x0f));
                w[64 * k + 32 + l] = static_cast<std::int16_t>(sc[2 * k + 1] * (q[l] >> 4));
            }
        }
        const std::int32_t sumi =
            mtl::simd::reduce_dot_widen<std::int32_t>(w.data(), a.wide + i * quant::qk_k, quant::qk_k);
        std::int32_t summ = 0;  // sum_j m_j * sum(a_j), from the 16-wide bsums
        for (std::size_t j = 0; j < quant::qk_k / 16; ++j) {
            summ += y.bsums[j] * m[j / 2];
        }
        const float d = fp16_to_float(b.d) * y.d;
        const float dmin = fp16_to_float(b.dmin) * y.d;
        sumf += d * static_cast<float>(sumi) - dmin * static_cast<float>(summ);
    }
    return sumf;
}

/// Q6_K weights . Q8_K activations.
inline float vec_dot_q6_k_q8_k(const std::byte* x, activations<quant::block_q8_k> a, std::size_t nb) noexcept {
    float sumf = 0.0f;
    for (std::size_t i = 0; i < nb; ++i) {
        const auto b = quant::detail::load_block<quant::block_q6_k>(x + i * sizeof(quant::block_q6_k));
        // Weight operand sc_{e/16} * (q_e - 32) (|.| <= 128 * 32), in element
        // order; the bit layout is the one dequantize(block_q6_k) decodes.
        std::array<std::int16_t, quant::qk_k> w{};
        for (std::size_t n = 0; n < 2; ++n) {
            const std::uint8_t* ql = b.ql + 64 * n;
            const std::uint8_t* qh = b.qh + 32 * n;
            const std::int8_t* sc = b.scales + 8 * n;
            std::int16_t* out = w.data() + 128 * n;
            for (std::size_t l = 0; l < 32; ++l) {
                const std::size_t is = l / 16;
                const int q1 = ((ql[l] & 0x0f) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = ((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                out[l] = static_cast<std::int16_t>(sc[is + 0] * q1);
                out[l + 32] = static_cast<std::int16_t>(sc[is + 2] * q2);
                out[l + 64] = static_cast<std::int16_t>(sc[is + 4] * q3);
                out[l + 96] = static_cast<std::int16_t>(sc[is + 6] * q4);
            }
        }
        const std::int32_t sumi =
            mtl::simd::reduce_dot_widen<std::int32_t>(w.data(), a.wide + i * quant::qk_k, quant::qk_k);
        sumf += fp16_to_float(b.d) * a.blocks[i].d * static_cast<float>(sumi);
    }
    return sumf;
}

/// A plain-format row (F32 / F16 / BF16, n elements at x) . float activations.
/// Decodes 256 elements at a time into a stack buffer and dots each chunk.
template <quant::Scheme S>
    requires(S::block_size == 1)
float vec_dot_plain(const std::byte* x, const float* a, std::size_t n) noexcept {
    constexpr std::size_t chunk = 256;
    std::array<float, chunk> buf{};
    float sumf = 0.0f;
    for (std::size_t i = 0; i < n; i += chunk) {
        const std::size_t len = n - i < chunk ? n - i : chunk;
        for (std::size_t j = 0; j < len; ++j) {
            S::dequantize(x + (i + j) * S::type_size, &buf[j]);
        }
        sumf += mtl::simd::reduce_dot(buf.data(), a + i, len);
    }
    return sumf;
}

/// Dot of one encoded row (`blocks` weight blocks at x) with the activations
/// in the scheme's vec_dot_block format.
template <quant::Scheme S>
float row_dot(const std::byte* x, activations<typename S::vec_dot_block> a, std::size_t blocks) noexcept {
    using quant::tensor_type;
    if constexpr (S::type == tensor_type::q4_0) {
        return vec_dot_q4_0_q8_0(x, a, blocks);
    } else if constexpr (S::type == tensor_type::q8_0) {
        return vec_dot_q8_0_q8_0(x, a, blocks);
    } else if constexpr (S::type == tensor_type::q4_k) {
        return vec_dot_q4_k_q8_k(x, a, blocks);
    } else if constexpr (S::type == tensor_type::q6_k) {
        return vec_dot_q6_k_q8_k(x, a, blocks);
    } else {
        return vec_dot_plain<S>(x, a.x, blocks);
    }
}

}  // namespace branes::synapse::kernels

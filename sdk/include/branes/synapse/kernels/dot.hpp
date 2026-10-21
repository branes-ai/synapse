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

#include <algorithm>
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

/// The fused dot of one quantized format, split at the block: unpack() turns
/// weight block i of a row into its int16 operand and scales, and dot()
/// returns that block's contribution against one activation row. row_dot
/// (gemv) and gemm both accumulate dot(unpack(row, i), a, i) over i in block
/// order, so for any activation row they perform the same float operations
/// in the same order: gemm's results equal gemv's bit for bit (G1), while
/// gemm unpacks each weight block once for all its activation rows.
template <quant::Scheme S>
struct block_kernel;

/// Q4_0 weights . Q8_0 activations: sum (q - 8) a.
template <>
struct block_kernel<quant::scheme<quant::tensor_type::q4_0>> {
    using act = activations<quant::block_q8_0>;
    struct unpacked {
        std::array<std::int16_t, quant::qk_legacy> w;  // q - 8
        float d;
    };
    static unpacked unpack(const std::byte* row, std::size_t i) noexcept {
        const auto b = quant::detail::load_block<quant::block_q4_0>(row + i * sizeof(quant::block_q4_0));
        unpacked u{};
        for (std::size_t j = 0; j < quant::qk_legacy / 2; ++j) {
            u.w[j] = static_cast<std::int16_t>((b.qs[j] & 0x0f) - 8);
            u.w[j + quant::qk_legacy / 2] = static_cast<std::int16_t>((b.qs[j] >> 4) - 8);
        }
        u.d = fp16_to_float(b.d);
        return u;
    }
    static float dot(const unpacked& u, act a, std::size_t i) noexcept {
        const std::int32_t sumi =
            mtl::simd::reduce_dot_widen<std::int32_t>(u.w.data(), a.wide + i * quant::qk_legacy, quant::qk_legacy);
        return static_cast<float>(sumi) * u.d * fp16_to_float(a.blocks[i].d);
    }
};

/// Q8_0 weights . Q8_0 activations: sum q a.
template <>
struct block_kernel<quant::scheme<quant::tensor_type::q8_0>> {
    using act = activations<quant::block_q8_0>;
    struct unpacked {
        std::array<std::int16_t, quant::qk_legacy> w;
        float d;
    };
    static unpacked unpack(const std::byte* row, std::size_t i) noexcept {
        const auto b = quant::detail::load_block<quant::block_q8_0>(row + i * sizeof(quant::block_q8_0));
        unpacked u{};
        for (std::size_t j = 0; j < quant::qk_legacy; ++j) {
            u.w[j] = b.qs[j];
        }
        u.d = fp16_to_float(b.d);
        return u;
    }
    static float dot(const unpacked& u, act a, std::size_t i) noexcept {
        const std::int32_t sumi =
            mtl::simd::reduce_dot_widen<std::int32_t>(u.w.data(), a.wide + i * quant::qk_legacy, quant::qk_legacy);
        return static_cast<float>(sumi) * (u.d * fp16_to_float(a.blocks[i].d));
    }
};

/// Q4_K weights . Q8_K activations:
/// sum_j sc_j dot(q_j, a_j) - dmin/d sum_j m_j sum(a_j).
template <>
struct block_kernel<quant::scheme<quant::tensor_type::q4_k>> {
    using act = activations<quant::block_q8_k>;
    struct unpacked {
        std::array<std::int16_t, quant::qk_k> w;  // sc_j * q
        std::array<std::uint8_t, 8> m;            // sub-block mins
        float d, dmin;
    };
    static unpacked unpack(const std::byte* row, std::size_t i) noexcept {
        const auto b = quant::detail::load_block<quant::block_q4_k>(row + i * sizeof(quant::block_q4_k));
        unpacked u{};
        std::array<std::uint8_t, 8> sc{};
        for (std::size_t j = 0; j < 8; ++j) {
            quant::q4_k_scale_min(j, b.scales, sc[j], u.m[j]);
        }
        // Weight operand sc_j * q (<= 63 * 15): sub-block 2k is the low
        // nibbles of qs[32k..32k+31], sub-block 2k+1 the high nibbles.
        for (std::size_t k = 0; k < 4; ++k) {
            const std::uint8_t* q = b.qs + 32 * k;
            for (std::size_t l = 0; l < 32; ++l) {
                u.w[64 * k + l] = static_cast<std::int16_t>(sc[2 * k] * (q[l] & 0x0f));
                u.w[64 * k + 32 + l] = static_cast<std::int16_t>(sc[2 * k + 1] * (q[l] >> 4));
            }
        }
        u.d = fp16_to_float(b.d);
        u.dmin = fp16_to_float(b.dmin);
        return u;
    }
    static float dot(const unpacked& u, act a, std::size_t i) noexcept {
        const quant::block_q8_k& y = a.blocks[i];
        const std::int32_t sumi =
            mtl::simd::reduce_dot_widen<std::int32_t>(u.w.data(), a.wide + i * quant::qk_k, quant::qk_k);
        std::int32_t summ = 0;  // sum_j m_j * sum(a_j), from the 16-wide bsums
        for (std::size_t j = 0; j < quant::qk_k / 16; ++j) {
            summ += y.bsums[j] * u.m[j / 2];
        }
        const float d = u.d * y.d;
        const float dmin = u.dmin * y.d;
        return d * static_cast<float>(sumi) - dmin * static_cast<float>(summ);
    }
};

/// Q6_K weights . Q8_K activations: sum_j sc_j dot(q_j - 32, a_j).
template <>
struct block_kernel<quant::scheme<quant::tensor_type::q6_k>> {
    using act = activations<quant::block_q8_k>;
    struct unpacked {
        std::array<std::int16_t, quant::qk_k> w;  // sc_{e/16} * (q_e - 32)
        float d;
    };
    static unpacked unpack(const std::byte* row, std::size_t i) noexcept {
        const auto b = quant::detail::load_block<quant::block_q6_k>(row + i * sizeof(quant::block_q6_k));
        unpacked u{};
        // Weight operand sc_{e/16} * (q_e - 32) (|.| <= 128 * 32), in element
        // order; the bit layout is the one dequantize(block_q6_k) decodes.
        for (std::size_t n = 0; n < 2; ++n) {
            const std::uint8_t* ql = b.ql + 64 * n;
            const std::uint8_t* qh = b.qh + 32 * n;
            const std::int8_t* sc = b.scales + 8 * n;
            std::int16_t* out = u.w.data() + 128 * n;
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
        u.d = fp16_to_float(b.d);
        return u;
    }
    static float dot(const unpacked& u, act a, std::size_t i) noexcept {
        const std::int32_t sumi =
            mtl::simd::reduce_dot_widen<std::int32_t>(u.w.data(), a.wide + i * quant::qk_k, quant::qk_k);
        return u.d * a.blocks[i].d * static_cast<float>(sumi);
    }
};

/// Plain formats (F32 / F16 / BF16) dot float activations in chunks of
/// plain_chunk elements: the "block" is a chunk decoded to float.
inline constexpr std::size_t plain_chunk = 256;

template <quant::Scheme S>
    requires(S::block_size == 1)
struct plain_kernel {
    /// Decode chunk i (len elements) of the row at x into buf.
    static void unpack(const std::byte* row, std::size_t i, std::size_t len, float* buf) noexcept {
        for (std::size_t j = 0; j < len; ++j) {
            S::dequantize(row + (i * plain_chunk + j) * S::type_size, &buf[j]);
        }
    }
    static float dot(const float* buf, const float* a, std::size_t i, std::size_t len) noexcept {
        return mtl::simd::reduce_dot(buf, a + i * plain_chunk, len);
    }
};

/// Dot of one encoded row (`blocks` weight blocks at x; for plain formats,
/// `blocks` elements) with the activations in the scheme's vec_dot_block format.
template <quant::Scheme S>
float row_dot(const std::byte* x, activations<typename S::vec_dot_block> a, std::size_t blocks) noexcept {
    float sumf = 0.0f;
    if constexpr (S::block_size == 1) {
        std::array<float, plain_chunk> buf{};
        for (std::size_t i = 0; i * plain_chunk < blocks; ++i) {
            const std::size_t len = std::min(plain_chunk, blocks - i * plain_chunk);
            plain_kernel<S>::unpack(x, i, len, buf.data());
            sumf += plain_kernel<S>::dot(buf.data(), a.x, i, len);
        }
    } else {
        for (std::size_t i = 0; i < blocks; ++i) {
            sumf += block_kernel<S>::dot(block_kernel<S>::unpack(x, i), a, i);
        }
    }
    return sumf;
}

}  // namespace branes::synapse::kernels

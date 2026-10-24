#pragma once
// Ported from ggml (MIT License, The ggml authors); see THIRD_PARTY_NOTICES.md.
/// @file dot.hpp
/// @brief Fused dot products of one encoded weight row with quantized activations.
///
/// Each kernel walks a row of weight blocks next to the matching activation
/// blocks. A block's dot product is computed exactly in integers and scaled by
/// the block's scales, so weights are never expanded to float. The integer
/// arithmetic follows ggml's ggml_vec_dot_*_q8_*_generic:
///
///   Q4_0 x Q8_0  sum (q - 8) a
///   Q5_0 x Q8_0  sum (q - 16) a
///   Q8_0 x Q8_0  sum q a
///   Q4_K x Q8_K  sum_j sc_j dot(q_j, a_j) - dmin/d sum_j m_j sum(a_j)
///                (sub-block sums come from Q8_K's bsums)
///   Q6_K x Q8_K  sum_j sc_j dot(q_j - 32, a_j)
///
/// Weights are unpacked to int16 with their offset (and, for the K formats,
/// their sub-block scale) folded in, a chunk of up to chunk_elements weights
/// at a time; activations are widened to int16 once per product (see
/// activations<>). Each chunk is one mtl::simd::reduce_dot_block_scaled
/// (MTL5 >= 5.20): every block's exact integer sum stays in vector lanes and
/// is scaled in-register by d_w * d_a, with one horizontal reduce per chunk
/// rather than one per block. No overflow is possible: |w| <= 4096 and
/// |a| <= 127, so a block sums to at most 256 * 4096 * 127 < 2^28, inside the
/// primitive's per-block precondition of 2^31.
///
/// Reproducibility: every block's integer sum is exact, and the float order
/// is fixed for a given target, so results are deterministic run to run, and
/// gemv and gemm agree bit for bit (G1). Across SIMD targets they agree to
/// float re-association (the vector width decides how a block's sum splits
/// into lanes before conversion).

#include <branes/synapse/numeric/half.hpp>
#include <branes/synapse/quant/blocks.hpp>
#include <branes/synapse/quant/q8.hpp>
#include <branes/synapse/quant/scheme.hpp>

#include <mtl/simd/algorithm.hpp>
#include <mtl/version.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

// simd::reduce_dot_block_scaled arrived in MTL5 5.20.0. Checked here as well
// as in cmake/deps.cmake, so an older MTL5 supplied by a parent project fails
// with this message rather than deep inside a kernel.
static_assert(MTL5_VERSION_MAJOR > 5 || (MTL5_VERSION_MAJOR == 5 && MTL5_VERSION_MINOR >= 20),
              "synapse requires MTL5 >= 5.20.0 (mtl::simd::reduce_dot_block_scaled)");

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
    const std::int16_t* wide;   ///< 256 per block
    const std::int16_t* bsums;  ///< the blocks' 16 sub-block sums each, contiguous
};

template <>
struct activations<float> {
    const float* x;
};

/// Weights unpacked per chunk: up to this many int16 operands (2 KiB), on the
/// stack. Measured on AVX2 for Q4_K, Q6_K and Q8_0, gemv and gemm alike: a
/// chunk this size amortizes the horizontal reduce while one token's chunk of
/// activations, for a whole token tile, still sits in L1; 2048 is slower for
/// Q4_K's batched products (0.180 vs 0.133 ms per token at 2048 x 2048).
inline constexpr std::size_t chunk_elements = 1024;

/// The fused dot of one quantized format, split at the block. unpack() writes
/// weight block i of a row as its int16 operand and returns its scales;
/// act_scale() is an activation block's scale. The chunk functions below
/// build every quantized row dot from them: row_dot (gemv) and gemm make the
/// same calls in the same order for any activation row, so their results are
/// bit-identical (G1), while gemm unpacks each chunk once for all its rows.
template <quant::Scheme S>
struct block_kernel;

/// Q4_0 weights . Q8_0 activations: sum (q - 8) a.
template <>
struct block_kernel<quant::scheme<quant::tensor_type::q4_0>> {
    using act = activations<quant::block_q8_0>;
    static constexpr std::size_t qk = quant::qk_legacy;
    struct scales {
        float d;
    };
    static scales unpack(const std::byte* row, std::size_t i, std::int16_t* w) noexcept {
        const auto b = quant::detail::load_block<quant::block_q4_0>(row + i * sizeof(quant::block_q4_0));
        for (std::size_t j = 0; j < qk / 2; ++j) {
            w[j] = static_cast<std::int16_t>((b.qs[j] & 0x0f) - 8);
            w[j + qk / 2] = static_cast<std::int16_t>((b.qs[j] >> 4) - 8);
        }
        return {fp16_to_float(b.d)};
    }
    static float act_scale(act a, std::size_t i) noexcept {
        return fp16_to_float(a.blocks[i].d);
    }
};

/// Q5_0 weights . Q8_0 activations: sum (q - 16) a.
template <>
struct block_kernel<quant::scheme<quant::tensor_type::q5_0>> {
    using act = activations<quant::block_q8_0>;
    static constexpr std::size_t qk = quant::qk_legacy;
    struct scales {
        float d;
    };
    static scales unpack(const std::byte* row, std::size_t i, std::int16_t* w) noexcept {
        const auto b = quant::detail::load_block<quant::block_q5_0>(row + i * sizeof(quant::block_q5_0));
        const std::uint32_t qh = quant::q5_0_high_bits(b);
        const auto hi_bit = [qh](std::size_t k) { return static_cast<int>((qh >> k) & 1) << 4; };
        for (std::size_t j = 0; j < qk / 2; ++j) {
            w[j] = static_cast<std::int16_t>(((b.qs[j] & 0x0f) | hi_bit(j)) - 16);
            w[j + qk / 2] = static_cast<std::int16_t>(((b.qs[j] >> 4) | hi_bit(j + qk / 2)) - 16);
        }
        return {fp16_to_float(b.d)};
    }
    static float act_scale(act a, std::size_t i) noexcept {
        return fp16_to_float(a.blocks[i].d);
    }
};

/// Q8_0 weights . Q8_0 activations: sum q a.
template <>
struct block_kernel<quant::scheme<quant::tensor_type::q8_0>> {
    using act = activations<quant::block_q8_0>;
    static constexpr std::size_t qk = quant::qk_legacy;
    struct scales {
        float d;
    };
    static scales unpack(const std::byte* row, std::size_t i, std::int16_t* w) noexcept {
        const auto b = quant::detail::load_block<quant::block_q8_0>(row + i * sizeof(quant::block_q8_0));
        for (std::size_t j = 0; j < qk; ++j) {
            w[j] = b.qs[j];
        }
        return {fp16_to_float(b.d)};
    }
    static float act_scale(act a, std::size_t i) noexcept {
        return fp16_to_float(a.blocks[i].d);
    }
};

/// Q4_K weights . Q8_K activations:
/// sum_j sc_j dot(q_j, a_j) - dmin/d sum_j m_j sum(a_j).
template <>
struct block_kernel<quant::scheme<quant::tensor_type::q4_k>> {
    using act = activations<quant::block_q8_k>;
    static constexpr std::size_t qk = quant::qk_k;
    static constexpr bool has_mins = true;
    struct scales {
        float d, dmin;
    };
    /// Also writes the block's min operand: m_{j/2} for each of its 16
    /// 16-element sub-block sums (the Q8_K bsums it multiplies).
    static scales unpack(const std::byte* row, std::size_t i, std::int16_t* w, std::int16_t* m16) noexcept {
        const auto b = quant::detail::load_block<quant::block_q4_k>(row + i * sizeof(quant::block_q4_k));
        const scales s{fp16_to_float(b.d), fp16_to_float(b.dmin)};
        std::array<std::uint8_t, 8> sc{};
        std::array<std::uint8_t, 8> m{};
        for (std::size_t j = 0; j < 8; ++j) {
            quant::q4_k_scale_min(j, b.scales, sc[j], m[j]);
        }
        for (std::size_t j = 0; j < qk / 16; ++j) {
            m16[j] = m[j / 2];
        }
        // Weight operand sc_j * q (<= 63 * 15): sub-block 2k is the low
        // nibbles of qs[32k..32k+31], sub-block 2k+1 the high nibbles.
        for (std::size_t k = 0; k < 4; ++k) {
            const std::uint8_t* q = b.qs + 32 * k;
            for (std::size_t l = 0; l < 32; ++l) {
                w[64 * k + l] = static_cast<std::int16_t>(sc[2 * k] * (q[l] & 0x0f));
                w[64 * k + 32 + l] = static_cast<std::int16_t>(sc[2 * k + 1] * (q[l] >> 4));
            }
        }
        return s;
    }
    static float act_scale(act a, std::size_t i) noexcept {
        return a.blocks[i].d;
    }
};

/// Q6_K weights . Q8_K activations: sum_j sc_j dot(q_j - 32, a_j).
template <>
struct block_kernel<quant::scheme<quant::tensor_type::q6_k>> {
    using act = activations<quant::block_q8_k>;
    static constexpr std::size_t qk = quant::qk_k;
    struct scales {
        float d;
    };
    static scales unpack(const std::byte* row, std::size_t i, std::int16_t* w) noexcept {
        const auto b = quant::detail::load_block<quant::block_q6_k>(row + i * sizeof(quant::block_q6_k));
        // Weight operand sc_{e/16} * (q_e - 32) (|.| <= 128 * 32), in element
        // order; the bit layout is the one dequantize(block_q6_k) decodes.
        for (std::size_t n = 0; n < 2; ++n) {
            const std::uint8_t* ql = b.ql + 64 * n;
            const std::uint8_t* qh = b.qh + 32 * n;
            const std::int8_t* sc = b.scales + 8 * n;
            std::int16_t* out = w + 128 * n;
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
        return {fp16_to_float(b.d)};
    }
    static float act_scale(act a, std::size_t i) noexcept {
        return a.blocks[i].d;
    }
};

template <typename K>
concept has_min_term = requires { requires K::has_mins; };

/// Up to chunk_elements weights of one row, unpacked: blocks b0 .. b0+nb-1.
template <quant::Scheme S>
struct weight_chunk {
    using K = block_kernel<S>;
    static constexpr std::size_t max_blocks = chunk_elements / K::qk;
    /// Sub-block sums per block in Q8_K, the operand of the K formats' mins.
    static constexpr std::size_t sub_blocks = 16;
    std::array<std::int16_t, chunk_elements> w;
    std::array<std::int16_t, has_min_term<K> ? max_blocks * sub_blocks : 1> m16;
    std::array<typename K::scales, max_blocks> s;
    std::size_t b0 = 0, nb = 0;

    /// Unpack blocks [b0, b0 + nb) of the row at `row`.  Pre: nb <= max_blocks.
    void load(const std::byte* row, std::size_t first, std::size_t count) noexcept {
        b0 = first;
        nb = count;
        for (std::size_t b = 0; b < nb; ++b) {
            if constexpr (has_min_term<K>) {
                s[b] = K::unpack(row, b0 + b, w.data() + b * K::qk, m16.data() + b * sub_blocks);
            } else {
                s[b] = K::unpack(row, b0 + b, w.data() + b * K::qk);
            }
        }
    }

    /// This chunk's contribution against one activation row: one block-scaled
    /// reduction with scales d_w * d_a, less (for the K formats with mins) a
    /// second over the mins and Q8_K's sub-block sums with scales dmin_w * d_a.
    float dot(typename K::act a) const noexcept {
        std::array<float, max_blocks> scale;
        for (std::size_t b = 0; b < nb; ++b) {
            scale[b] = s[b].d * K::act_scale(a, b0 + b);
        }
        float sum = mtl::simd::reduce_dot_block_scaled(w.data(), a.wide + b0 * K::qk, scale.data(), nb, K::qk);
        if constexpr (has_min_term<K>) {
            std::array<float, max_blocks> min_scale;
            for (std::size_t b = 0; b < nb; ++b) {
                min_scale[b] = s[b].dmin * K::act_scale(a, b0 + b);
            }
            sum -= mtl::simd::reduce_dot_block_scaled(
                m16.data(), a.bsums + b0 * sub_blocks, min_scale.data(), nb, sub_blocks);
        }
        return sum;
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
        weight_chunk<S> c;
        for (std::size_t b0 = 0; b0 < blocks; b0 += weight_chunk<S>::max_blocks) {
            c.load(x, b0, std::min(weight_chunk<S>::max_blocks, blocks - b0));
            sumf += c.dot(a);
        }
    }
    return sumf;
}

}  // namespace branes::synapse::kernels

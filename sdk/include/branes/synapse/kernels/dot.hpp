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
/// activations<>). Each chunk is one block-scaled sum (detail::
/// block_scaled_dots, on MTL5 >= 5.20's widening dot and lane conversion):
/// every block's exact integer sum stays in vector lanes and is scaled
/// in-register by d_w * d_a, with one horizontal reduce per chunk rather than
/// one per block. gemm computes a register tile of weight rows x activation
/// rows at once, each pair exactly as gemv computes it. No overflow is
/// possible: |w| <= 4096 and |a| <= 127, so a block sums to at most
/// 256 * 4096 * 127 < 2^28, well inside the int32 lanes (converting a lane
/// to float rounds once it passes 2^24, the same in gemv and gemm).
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
#include <mtl/simd/batch.hpp>
#include <mtl/version.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

// batch<float>::convert, the lane conversion block_scaled_dots scales the
// integer sums with, arrived in MTL5 5.20.0. Checked here as well
// as in cmake/deps.cmake, so an older MTL5 supplied by a parent project fails
// with this message rather than deep inside a kernel.
static_assert(MTL5_VERSION_MAJOR > 5 || (MTL5_VERSION_MAJOR == 5 && MTL5_VERSION_MINOR >= 20),
              "synapse requires MTL5 >= 5.20.0 (mtl::simd::batch<float>::convert)");

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
    const float* d;            ///< the blocks' scales, decoded from binary16
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
        return a.d[i];
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
        return a.d[i];
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
        return a.d[i];
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

namespace detail {

using lanes_i32 = mtl::simd::batch<std::int32_t>;
using lanes_f32 = mtl::simd::batch<float>;

/// The block-scaled dots of R x T operand pairs at once: for weight operand
/// w[r] and activation operand x[t], each nb blocks of `len` int16,
///
///   sums[r * T + t] = sum_b scale[(r * T + t) * nb + b] * (w[r]_b . x[t]_b).
///
/// Register-blocked: each block of one operand meets T (or R) blocks of the
/// other while it is in registers. Every pair is computed the same way
/// whatever R and T are, so tiling does not change a result (G1, D1): the
/// block's exact integer sum in lanes (widen_dot_accumulate), converted, and
/// accumulated as fma(lanes, scale_b, acc) into one float accumulator in
/// block order; then one horizontal reduce. A block length the vector does
/// not divide (on wide targets, Q8_K's 16-element sub-blocks) is summed
/// exactly in a scalar and scaled in block order instead.
///
/// A short block (Q8_0's 32) is finished one pair at a time, which keeps the
/// R x T float accumulators in registers. A long one (the K formats' 256)
/// walks the block a vector at a time for all pairs, which keeps their
/// integer sums in registers instead. The integer lanes are exact either
/// way, so the two orders give the same lanes and the same result.
template <std::size_t R, std::size_t T, std::size_t len>
void block_scaled_dots(const std::array<const std::int16_t*, R>& w,
                       const std::array<const std::int16_t*, T>& x,
                       const float* scale,
                       std::size_t nb,
                       std::array<float, R * T>& sums) noexcept {
    constexpr std::size_t step = lanes_i32::widen_step;
    if constexpr (len % step == 0) {
        std::array<lanes_f32, R * T> acc{};
        for (std::size_t b = 0; b < nb; ++b) {
            if constexpr (len <= 2 * step) {
                for (std::size_t r = 0; r < R; ++r) {
                    for (std::size_t t = 0; t < T; ++t) {
                        lanes_i32 s0{}, s1{};  // s1 must start zeroed (mtl::simd::batch)
                        for (std::size_t i = 0; i < len; i += step) {
                            lanes_i32::widen_dot_accumulate(w[r] + b * len + i, x[t] + b * len + i, s0, s1);
                        }
                        const std::size_t p = r * T + t;
                        acc[p] = fma(lanes_f32::convert(s0 + s1), lanes_f32(scale[p * nb + b]), acc[p]);
                    }
                }
            } else {
                std::array<lanes_i32, R * T> s0{}, s1{};  // s1 must start zeroed (mtl::simd::batch)
                for (std::size_t i = 0; i < len; i += step) {
                    for (std::size_t r = 0; r < R; ++r) {
                        for (std::size_t t = 0; t < T; ++t) {
                            lanes_i32::widen_dot_accumulate(
                                w[r] + b * len + i, x[t] + b * len + i, s0[r * T + t], s1[r * T + t]);
                        }
                    }
                }
                for (std::size_t p = 0; p < R * T; ++p) {
                    acc[p] = fma(lanes_f32::convert(s0[p] + s1[p]), lanes_f32(scale[p * nb + b]), acc[p]);
                }
            }
        }
        for (std::size_t p = 0; p < R * T; ++p) {
            sums[p] = reduce_add(acc[p]);
        }
    } else {
        sums.fill(0.0f);
        for (std::size_t b = 0; b < nb; ++b) {
            for (std::size_t r = 0; r < R; ++r) {
                for (std::size_t t = 0; t < T; ++t) {
                    std::int32_t s = 0;
                    for (std::size_t i = 0; i < len; ++i) {
                        s += std::int32_t{w[r][b * len + i]} * x[t][b * len + i];
                    }
                    const std::size_t p = r * T + t;
                    sums[p] += static_cast<float>(s) * scale[p * nb + b];
                }
            }
        }
    }
}

}  // namespace detail

/// Up to chunk_elements weights of one row, unpacked: blocks b0 .. b0+nb-1.
template <quant::Scheme S>
struct weight_chunk {
    using K = block_kernel<S>;
    using act = typename K::act;
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

    /// The contributions of R chunks c[0..R) (the same blocks of R rows)
    /// against T activation rows a[0..T): y[t * y_stride + r] += chunk r . a[t].
    /// Each is one block-scaled sum with scales d_w * d_a, less (for the K
    /// formats with mins) a second over the mins and Q8_K's sub-block sums
    /// with scales dmin_w * d_a, the same for every R and T (G1).
    ///   Pre: the R chunks hold the same blocks (b0, nb).
    template <std::size_t R, std::size_t T>
    static void dot_tile(const weight_chunk* c, const act* a, float* y, std::size_t y_stride) noexcept {
        const std::size_t first = c[0].b0;
        const std::size_t n = c[0].nb;
        std::array<float, R * T * max_blocks> scale;
        std::array<const std::int16_t*, R> wr;
        std::array<const std::int16_t*, T> xt;
        for (std::size_t r = 0; r < R; ++r) {
            wr[r] = c[r].w.data();
            for (std::size_t t = 0; t < T; ++t) {
                for (std::size_t b = 0; b < n; ++b) {
                    scale[(r * T + t) * n + b] = c[r].s[b].d * K::act_scale(a[t], first + b);
                }
            }
        }
        for (std::size_t t = 0; t < T; ++t) {
            xt[t] = a[t].wide + first * K::qk;
        }
        std::array<float, R * T> sums;
        detail::block_scaled_dots<R, T, K::qk>(wr, xt, scale.data(), n, sums);
        if constexpr (has_min_term<K>) {
            for (std::size_t r = 0; r < R; ++r) {
                wr[r] = c[r].m16.data();
                for (std::size_t t = 0; t < T; ++t) {
                    for (std::size_t b = 0; b < n; ++b) {
                        scale[(r * T + t) * n + b] = c[r].s[b].dmin * K::act_scale(a[t], first + b);
                    }
                }
            }
            for (std::size_t t = 0; t < T; ++t) {
                xt[t] = a[t].bsums + first * sub_blocks;
            }
            std::array<float, R * T> mins;
            detail::block_scaled_dots<R, T, sub_blocks>(wr, xt, scale.data(), n, mins);
            for (std::size_t p = 0; p < R * T; ++p) {
                sums[p] -= mins[p];
            }
        }
        for (std::size_t r = 0; r < R; ++r) {
            for (std::size_t t = 0; t < T; ++t) {
                y[t * y_stride + r] += sums[r * T + t];
            }
        }
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
            weight_chunk<S>::template dot_tile<1, 1>(&c, &a, &sumf, 0);
        }
    }
    return sumf;
}

}  // namespace branes::synapse::kernels

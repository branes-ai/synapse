#pragma once
// Ported from ggml (MIT License, The ggml authors); see THIRD_PARTY_NOTICES.md.
/// @file blocks.hpp
/// @brief GGML block-quantization layouts and their reference dequantizers.
///
/// Each struct reproduces the corresponding `block_*` in ggml-common.h byte
/// for byte, so a GGUF tensor is a plain array of them. fp16 fields hold raw
/// binary16 bits. The dequantizers follow ggml's dequantize_row_* in
/// ggml-quants.c, including the order of the float operations, so results
/// match ggml and gguf-py bit for bit on hardware that does not fuse them.
///
/// These are the reference (scalar, one block at a time) definitions of each
/// format. The fused SIMD kernels of roadmap step 3 are tested against them.
///
/// Q4_0  32 weights: x = d * (q - 8),           q in [0, 15]
/// Q8_0  32 weights: x = d * q,                 q in [-128, 127]
/// Q4_K 256 weights in 8 sub-blocks of 32: x = d*sc[j] * q - dmin*m[j],
///      q in [0, 15], sc/m 6-bit, packed into 12 bytes
/// Q6_K 256 weights in 16 sub-blocks of 16: x = d*sc[j] * (q - 32),
///      q 6-bit (4 low bits in ql, 2 high bits in qh), sc int8

#include <branes/synapse/numeric/half.hpp>

#include <cstddef>
#include <cstdint>

namespace branes::synapse::quant {

/// Weights per block for the legacy formats (QK4_0, QK8_0).
inline constexpr std::size_t qk_legacy = 32;
/// Weights per super-block for the K formats (QK_K).
inline constexpr std::size_t qk_k = 256;

struct block_q4_0 {
    std::uint16_t d;                 ///< scale (binary16)
    std::uint8_t qs[qk_legacy / 2];  ///< nibbles: low = weights 0..15, high = 16..31
};
static_assert(sizeof(block_q4_0) == 18);

struct block_q8_0 {
    std::uint16_t d;            ///< scale (binary16)
    std::int8_t qs[qk_legacy];  ///< quants
};
static_assert(sizeof(block_q8_0) == 34);

struct block_q4_k {
    std::uint16_t d;            ///< super-block scale for the sub-block scales (binary16)
    std::uint16_t dmin;         ///< super-block scale for the sub-block mins (binary16)
    std::uint8_t scales[12];    ///< 8 x (6-bit scale, 6-bit min), packed
    std::uint8_t qs[qk_k / 2];  ///< nibbles
};
static_assert(sizeof(block_q4_k) == 144);

struct block_q6_k {
    std::uint8_t ql[qk_k / 2];      ///< low 4 bits of each quant
    std::uint8_t qh[qk_k / 4];      ///< high 2 bits of each quant
    std::int8_t scales[qk_k / 16];  ///< 8-bit sub-block scales
    std::uint16_t d;                ///< super-block scale (binary16)
};
static_assert(sizeof(block_q6_k) == 210);

/// Write the 32 weights of `b` to y[0..31].
inline void dequantize(const block_q4_0& b, float* y) noexcept {
    const float d = fp16_to_float(b.d);
    for (std::size_t j = 0; j < qk_legacy / 2; ++j) {
        const int lo = (b.qs[j] & 0x0f) - 8;
        const int hi = (b.qs[j] >> 4) - 8;
        y[j] = static_cast<float>(lo) * d;
        y[j + qk_legacy / 2] = static_cast<float>(hi) * d;
    }
}

/// Write the 32 weights of `b` to y[0..31].
inline void dequantize(const block_q8_0& b, float* y) noexcept {
    const float d = fp16_to_float(b.d);
    for (std::size_t j = 0; j < qk_legacy; ++j) {
        y[j] = static_cast<float>(b.qs[j]) * d;
    }
}

/// The 6-bit scale and min of sub-block j (0..7) of a Q4_K block
/// (get_scale_min_k4 in ggml-quants.c).
///
///   bytes 0-3:  scale j   (low 6 bits), high 2 bits of scale j+4
///   bytes 4-7:  min j     (low 6 bits), high 2 bits of min j+4
///   bytes 8-11: low 4 bits of scale j+4 | low 4 bits of min j+4 << 4
inline void q4_k_scale_min(std::size_t j, const std::uint8_t* q, std::uint8_t& sc, std::uint8_t& m) noexcept {
    if (j < 4) {
        sc = static_cast<std::uint8_t>(q[j] & 63);
        m = static_cast<std::uint8_t>(q[j + 4] & 63);
    } else {
        sc = static_cast<std::uint8_t>((q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4));
        m = static_cast<std::uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

/// Write the 256 weights of `b` to y[0..255].
inline void dequantize(const block_q4_k& b, float* y) noexcept {
    const float d = fp16_to_float(b.d);
    const float dmin = fp16_to_float(b.dmin);
    const std::uint8_t* q = b.qs;
    // Each 32 bytes of qs hold two sub-blocks: low nibbles, then high nibbles.
    for (std::size_t j = 0; j < 8; j += 2) {
        std::uint8_t sc = 0;
        std::uint8_t m = 0;
        q4_k_scale_min(j, b.scales, sc, m);
        const float d1 = d * static_cast<float>(sc);
        const float m1 = dmin * static_cast<float>(m);
        q4_k_scale_min(j + 1, b.scales, sc, m);
        const float d2 = d * static_cast<float>(sc);
        const float m2 = dmin * static_cast<float>(m);
        for (std::size_t l = 0; l < 32; ++l) {
            *y++ = d1 * static_cast<float>(q[l] & 0x0f) - m1;
        }
        for (std::size_t l = 0; l < 32; ++l) {
            *y++ = d2 * static_cast<float>(q[l] >> 4) - m2;
        }
        q += 32;
    }
}

/// Write the 256 weights of `b` to y[0..255].
inline void dequantize(const block_q6_k& b, float* y) noexcept {
    const float d = fp16_to_float(b.d);
    const std::uint8_t* ql = b.ql;
    const std::uint8_t* qh = b.qh;
    const std::int8_t* sc = b.scales;
    // Two halves of 128 weights; within a half, weight l + 32*k takes its low
    // bits from nibble (k / 2) of ql[l + 32*(k % 2)] and its high bits from
    // bits 2k..2k+1 of qh[l].
    for (std::size_t n = 0; n < qk_k; n += 128) {
        for (std::size_t l = 0; l < 32; ++l) {
            const std::size_t is = l / 16;
            const int q1 = ((ql[l] & 0x0f) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = ((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l] = d * static_cast<float>(sc[is + 0]) * static_cast<float>(q1);
            y[l + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
            y[l + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
            y[l + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

}  // namespace branes::synapse::quant

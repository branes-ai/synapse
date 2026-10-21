#pragma once
/// @file half.hpp
/// @brief Bit-exact widening of the 16-bit float formats GGUF stores.
///
/// GGUF keeps F16 and BF16 tensors, and the fp16 scales inside every
/// block-quantized format, as raw 16-bit patterns. These functions widen a
/// pattern to float exactly: both formats are subsets of IEEE binary32, so the
/// conversion is lossless, including subnormals, signed zeros, infinities,
/// and NaN payloads. They are constexpr and depend on no hardware F16C/NEON
/// support; the step-3 kernels can swap in vector conversions behind them.

#include <bit>
#include <cstdint>

namespace branes::synapse {

/// IEEE binary16 (1 sign, 5 exponent, 10 fraction bits) to float.
constexpr float fp16_to_float(std::uint16_t h) noexcept {
    const std::uint32_t x = h;
    const std::uint32_t sign = (x & 0x8000u) << 16;
    const std::uint32_t exponent = (x >> 10) & 0x1fu;
    std::uint32_t fraction = x & 0x3ffu;

    if (exponent == 0x1fu) {  // infinity or NaN: keep the payload
        return std::bit_cast<float>(sign | 0x7f800000u | (fraction << 13));
    }
    if (exponent != 0) {  // normal: rebias 15 -> 127
        return std::bit_cast<float>(sign | ((exponent + 112u) << 23) | (fraction << 13));
    }
    if (fraction == 0) {  // signed zero
        return std::bit_cast<float>(sign);
    }
    // Subnormal binary16 (fraction * 2^-24) is a normal binary32: shift the
    // leading one up to the implicit-bit position, lowering the exponent.
    std::uint32_t biased = 113;  // 127 - 15 + 1
    while ((fraction & 0x400u) == 0) {
        fraction <<= 1;
        --biased;
    }
    return std::bit_cast<float>(sign | (biased << 23) | ((fraction & 0x3ffu) << 13));
}

/// float to IEEE binary16, rounding to nearest, ties to even (what F16C's
/// vcvtps2ph and ggml's GGML_FP32_TO_FP16 produce). Overflow goes to
/// infinity; NaN stays NaN (quiet, top payload bits kept).
constexpr std::uint16_t float_to_fp16(float f) noexcept {
    const std::uint32_t x = std::bit_cast<std::uint32_t>(f);
    const auto sign = static_cast<std::uint16_t>((x >> 16) & 0x8000u);
    const std::uint32_t mag = x & 0x7fffffffu;

    if (mag >= 0x7f800000u) {  // infinity or NaN
        const std::uint32_t nan_bits = mag > 0x7f800000u ? (0x200u | ((mag >> 13) & 0x3ffu)) : 0u;
        return static_cast<std::uint16_t>(sign | 0x7c00u | nan_bits);
    }
    if (mag >= 0x477ff000u) {  // >= 65520: rounds past the largest finite (65504)
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    if (mag < 0x38800000u) {       // below 2^-14: binary16 subnormal (or zero)
        if (mag <= 0x33000000u) {  // <= 2^-25: rounds to zero (2^-25 itself ties to even)
            return sign;
        }
        // Value = m * 2^(e - 150); in units of 2^-24 that is m >> (126 - e).
        const std::uint32_t e = mag >> 23;
        const std::uint32_t m = (mag & 0x7fffffu) | 0x800000u;
        const std::uint32_t shift = 126u - e;  // 14..23
        std::uint32_t h = m >> shift;
        const std::uint32_t rem = m & ((1u << shift) - 1u);
        const std::uint32_t halfway = 1u << (shift - 1u);
        if (rem > halfway || (rem == halfway && (h & 1u) != 0)) {
            ++h;  // a carry into bit 10 correctly yields the smallest normal
        }
        return static_cast<std::uint16_t>(sign | h);
    }
    // Normal: rebias 127 -> 15 and round the 13 dropped fraction bits.
    const std::uint32_t r = mag - 0x38000000u;
    std::uint32_t h = r >> 13;
    const std::uint32_t rem = r & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) != 0)) {
        ++h;  // a carry out of the fraction correctly bumps the exponent
    }
    return static_cast<std::uint16_t>(sign | h);
}

/// bfloat16 (the upper half of a binary32) to float.
constexpr float bf16_to_float(std::uint16_t h) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(h) << 16);
}

}  // namespace branes::synapse

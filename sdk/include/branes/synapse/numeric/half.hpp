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

/// bfloat16 (the upper half of a binary32) to float.
constexpr float bf16_to_float(std::uint16_t h) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(h) << 16);
}

}  // namespace branes::synapse

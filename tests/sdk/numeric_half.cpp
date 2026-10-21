// Exhaustive tests for the 16-bit float widenings: every one of the 65536 bit
// patterns of binary16 and bfloat16 is checked. Finite binary16 values and all
// bfloat16 values are compared with Universal's independent implementations
// (cfloat<16,5> with subnormals, and bfloat16). The binary16 patterns with an
// all-ones exponent are checked against IEEE 754 directly: Universal's cfloat
// places its infinity and NaN encodings differently (0x7c00 is +inf in IEEE
// binary16 but a NaN in cfloat), so it is not an oracle for those.

#include <branes/synapse/numeric/half.hpp>

#include <catch2/catch_test_macros.hpp>
#include <universal/number/bfloat16/bfloat16.hpp>
#include <universal/number/cfloat/cfloat.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

using branes::synapse::bf16_to_float;
using branes::synapse::fp16_to_float;

namespace {

using binary16 = sw::universal::cfloat<16, 5, std::uint16_t, true, false, false>;  // IEEE-754 binary16 layout

// Same value, or both NaN (payload encodings are an implementation choice).
bool same_float(float a, float b) {
    if (std::isnan(a) || std::isnan(b)) {
        return std::isnan(a) && std::isnan(b);
    }
    return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
}

}  // namespace

TEST_CASE("half: fp16_to_float matches Universal binary16 on all 63488 finite patterns", "[half]") {
    int mismatches = 0;
    for (std::uint32_t bits = 0; bits <= 0xffff; ++bits) {
        if ((bits & 0x7c00u) == 0x7c00u) {
            continue;  // infinity / NaN: see the IEEE case below
        }
        binary16 ref;
        ref.setbits(bits);
        const float expected = static_cast<float>(ref);
        const float actual = fp16_to_float(static_cast<std::uint16_t>(bits));
        if (!same_float(actual, expected)) {
            if (++mismatches <= 10) {
                UNSCOPED_INFO("bits 0x" << std::hex << bits << ": got " << actual << ", expected " << expected);
            }
        }
    }
    CHECK(mismatches == 0);
}

TEST_CASE("half: fp16_to_float follows IEEE 754 for an all-ones exponent", "[half]") {
    for (std::uint32_t fraction = 0; fraction < 0x400; ++fraction) {
        for (std::uint32_t sign : {0x0000u, 0x8000u}) {
            const float v = fp16_to_float(static_cast<std::uint16_t>(sign | 0x7c00u | fraction));
            if (fraction == 0) {
                CHECK(std::isinf(v));
            } else {
                CHECK(std::isnan(v));
            }
            CHECK(std::signbit(v) == (sign != 0));
        }
    }
}

TEST_CASE("half: bf16_to_float matches Universal bfloat16 on all 65536 patterns", "[half]") {
    int mismatches = 0;
    for (std::uint32_t bits = 0; bits <= 0xffff; ++bits) {
        sw::universal::bfloat16 ref;
        ref.setbits(bits);
        const float expected = static_cast<float>(ref);
        const float actual = bf16_to_float(static_cast<std::uint16_t>(bits));
        if (!same_float(actual, expected)) {
            if (++mismatches <= 10) {
                UNSCOPED_INFO("bits 0x" << std::hex << bits << ": got " << actual << ", expected " << expected);
            }
        }
    }
    CHECK(mismatches == 0);
}

TEST_CASE("half: landmark values", "[half]") {
    STATIC_REQUIRE(fp16_to_float(0x3c00) == 1.0f);
    STATIC_REQUIRE(fp16_to_float(0xc000) == -2.0f);
    STATIC_REQUIRE(fp16_to_float(0x7bff) == 65504.0f);  // largest finite
    STATIC_REQUIRE(fp16_to_float(0x0001) == 0x1p-24f);  // smallest subnormal
    STATIC_REQUIRE(fp16_to_float(0x0400) == 0x1p-14f);  // smallest normal
    STATIC_REQUIRE(fp16_to_float(0x7c00) == std::numeric_limits<float>::infinity());
    STATIC_REQUIRE(std::bit_cast<std::uint32_t>(fp16_to_float(0x8000)) == 0x80000000u);  // -0
    STATIC_REQUIRE(bf16_to_float(0x3f80) == 1.0f);
    CHECK(std::isnan(fp16_to_float(0x7e00)));
}

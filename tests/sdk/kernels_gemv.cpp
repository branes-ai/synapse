// Unit tests for the activation quantizers, float_to_fp16, and gemv on plain
// formats, with expected values that follow from the format definitions; and
// MTL5's mult dispatching to gemv through mtl::block_kernels.

#include <branes/synapse/kernels/gemv.hpp>
#include <branes/synapse/numeric/half.hpp>
#include <branes/synapse/quant/q8.hpp>

#include "../support/random_weights.hpp"

#include <catch2/catch_test_macros.hpp>
#include <mtl/mat/block_matrix_view.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <universal/number/cfloat/cfloat.hpp>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

using namespace branes::synapse;
using quant::tensor_type;

namespace {

using binary16 = sw::universal::cfloat<16, 5, std::uint16_t, true, false, false>;  // IEEE-754 binary16 layout

// mtl::mult on a synapse view runs kernels::gemv: same bits, with the
// caller's scratch and with the scratch mult allocates.
template <tensor_type T>
void check_mult_is_gemv(std::size_t rows, std::size_t cols) {
    using S = quant::scheme<T>;
    STATIC_REQUIRE(mtl::BlockGemvScheme<S>);
    STATIC_REQUIRE(mtl::block_gemv_scratch_alignment<S> == kernels::gemv_scratch_alignment);
    std::mt19937 rng(static_cast<unsigned>(rows * 17 + cols));
    const synapse_test::random_matrix<T> w(rows, cols, rng);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    mtl::vec::dense_vector<float> x(cols);
    for (std::size_t i = 0; i < cols; ++i) {
        x(i) = normal(rng);
    }

    std::vector<float> expected(rows);
    std::vector<std::byte> scratch(kernels::gemv_scratch_bytes<S>(cols));
    kernels::gemv(w.view(), std::span<const float>(x.data(), cols), expected, scratch);

    CHECK(mtl::block_gemv_scratch_bytes<S>(cols) == scratch.size());
    mtl::vec::dense_vector<float> y(rows);
    mtl::mult(w.view(), x, y, scratch);
    mtl::vec::dense_vector<float> y_alloc(rows);
    mtl::mult(w.view(), x, y_alloc);
    for (std::size_t r = 0; r < rows; ++r) {
        INFO("row " << r);
        CHECK(std::bit_cast<std::uint32_t>(y(r)) == std::bit_cast<std::uint32_t>(expected[r]));
        CHECK(std::bit_cast<std::uint32_t>(y_alloc(r)) == std::bit_cast<std::uint32_t>(expected[r]));
    }
}

}  // namespace

TEST_CASE("kernels: float_to_fp16 inverts fp16_to_float on every non-NaN pattern", "[kernels][half]") {
    int mismatches = 0;
    for (std::uint32_t bits = 0; bits <= 0xffff; ++bits) {
        const auto h = static_cast<std::uint16_t>(bits);
        const float f = fp16_to_float(h);
        if (std::isnan(f)) {
            CHECK(std::isnan(fp16_to_float(float_to_fp16(f))));
            continue;
        }
        mismatches += float_to_fp16(f) != h ? 1 : 0;
    }
    CHECK(mismatches == 0);
}

TEST_CASE("kernels: float_to_fp16 rounds to nearest even like Universal", "[kernels][half]") {
    // Random float bit patterns across the binary16 finite range; Universal's
    // cfloat conversion is an independent round-to-nearest-even.
    std::mt19937 rng(7);
    std::uniform_int_distribution<std::uint32_t> mag(0, 0x477fefffu);  // below the overflow threshold 65520
    int mismatches = 0;
    for (int i = 0; i < 200000; ++i) {
        const std::uint32_t bits = mag(rng) | ((i & 1) != 0 ? 0x80000000u : 0u);
        const float f = std::bit_cast<float>(bits);
        const binary16 ref(f);
        if (float_to_fp16(f) != ref.block(0)) {
            if (++mismatches <= 5) {
                UNSCOPED_INFO("float bits 0x" << std::hex << bits);
            }
        }
    }
    CHECK(mismatches == 0);

    // Ties: 1 + 2^-11 is halfway between 1 and the next half (1 + 2^-10): even is 1.
    CHECK(float_to_fp16(1.0f + 0x1p-11f) == 0x3c00);
    CHECK(float_to_fp16(1.0f + 3 * 0x1p-11f) == 0x3c02);  // halfway, rounds up to even
    CHECK(float_to_fp16(65520.0f) == 0x7c00);             // first value that overflows
    CHECK(float_to_fp16(65519.996f) == 0x7bff);
    CHECK(float_to_fp16(0x1p-25f) == 0x0000);  // tie between 0 and 2^-24
    CHECK(float_to_fp16(0x1.000002p-25f) == 0x0001);
    CHECK(float_to_fp16(std::numeric_limits<float>::infinity()) == 0x7c00);
}

TEST_CASE("kernels: Q8_0 quantizer scales by the block maximum", "[kernels]") {
    std::vector<float> x(32);
    for (std::size_t j = 0; j < 32; ++j) {
        x[j] = static_cast<float>(j) - 16.0f;  // max |x| = 16
    }
    std::vector<quant::block_q8_0> y(1);
    quant::quantize_row_q8_0(x, y);
    CHECK(y[0].d == float_to_fp16(16.0f / 127.0f));  // stored as binary16; quantized with the float scale
    CHECK(y[0].qs[0] == -127);
    CHECK(y[0].qs[16] == 0);
    CHECK(y[0].qs[24] == 64);  // 8 * 127 / 16 = 63.5 rounds away from zero
}

TEST_CASE("kernels: Q8_K quantizer fills bsums and handles zero blocks", "[kernels]") {
    std::vector<float> x(512, 0.0f);
    for (std::size_t j = 0; j < 256; ++j) {
        x[j] = std::sin(0.1f * static_cast<float>(j));
    }
    std::vector<quant::block_q8_k> y(2);
    quant::quantize_row_q8_k(x, y);
    for (std::size_t g = 0; g < 16; ++g) {
        int sum = 0;
        for (std::size_t l = 0; l < 16; ++l) {
            sum += y[0].qs[g * 16 + l];
        }
        CHECK(y[0].bsums[g] == sum);
    }
    // The signed maximum maps to -127 (scale is -127 / max).
    int qmin = 0;
    for (auto q : y[0].qs) {
        qmin = std::min<int>(qmin, q);
    }
    CHECK(qmin == -127);
    CHECK(y[1].d == 0.0f);
    CHECK(y[1].bsums[0] == 0);
}

TEST_CASE("kernels: gemv_scratch_bytes", "[kernels]") {
    // Blocks, then the int16-widened quants; for Q8_0 also each block's
    // scale decoded to float, for Q8_K each block's 16 sub-block sums,
    // contiguous (the operand of Q4_K's min terms).
    STATIC_REQUIRE(kernels::gemv_scratch_bytes<quant::scheme<tensor_type::q4_0>>(4096) ==
                   128 * 34 + 4096 * 2 + 128 * 4);
    STATIC_REQUIRE(kernels::gemv_scratch_bytes<quant::scheme<tensor_type::q4_k>>(4096) ==
                   16 * 292 + 4096 * 2 + 16 * 16 * 2);
    STATIC_REQUIRE(kernels::gemv_scratch_bytes<quant::scheme<tensor_type::f16>>(4096) == 0);
}

TEST_CASE("kernels: plain-format gemv is an exact float matvec", "[kernels]") {
    // 3 x 300 F16 matrix of small integers (exact in binary16) times small
    // integers: every product and partial sum is exact in float, so the
    // result is exact regardless of summation order. 300 exercises the
    // 256-element chunking and its tail.
    constexpr std::size_t rows = 3;
    constexpr std::size_t cols = 300;
    std::vector<std::uint16_t> w(rows * cols);
    std::vector<float> x(cols);
    std::vector<double> expected(rows, 0.0);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const auto v = static_cast<float>(static_cast<int>((r * 7 + c * 3) % 11) - 5);
            w[r * cols + c] = float_to_fp16(v);
            x[c] = static_cast<float>(static_cast<int>(c % 5) - 2);
            expected[r] += static_cast<double>(v) * x[c];
        }
    }
    std::vector<std::byte> bytes(w.size() * sizeof(std::uint16_t));
    std::memcpy(bytes.data(), w.data(), bytes.size());
    const mtl::mat::block_matrix_view<quant::scheme<tensor_type::f16>> view(rows, cols, bytes);
    std::vector<float> y(rows);
    kernels::gemv(view, x, y, {});
    for (std::size_t r = 0; r < rows; ++r) {
        CHECK(static_cast<double>(y[r]) == expected[r]);
    }
}

TEST_CASE("kernels: mtl::mult on a synapse view is gemv bit for bit", "[kernels]") {
    check_mult_is_gemv<tensor_type::q4_0>(37, 512);
    check_mult_is_gemv<tensor_type::q5_0>(37, 512);
    check_mult_is_gemv<tensor_type::q8_0>(37, 512);
    check_mult_is_gemv<tensor_type::q4_k>(19, 512);
    check_mult_is_gemv<tensor_type::q6_k>(19, 512);
    check_mult_is_gemv<tensor_type::f16>(23, 600);
}

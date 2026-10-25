// Unit tests for kernels::gemm: contract G1, row t of gemm(W, X) equals
// gemv(W, row t of X) bit for bit, for every weight format and batch size.
// (gemv's accuracy against ggml is covered by kernels_golden.)

#include <branes/synapse/kernels/gemm.hpp>
#include <branes/synapse/kernels/gemv.hpp>

#include "../support/random_weights.hpp"

#include <catch2/catch_test_macros.hpp>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

using namespace branes::synapse;
using synapse_test::random_matrix;
using synapse_test::tensor_type;

namespace {

template <tensor_type T>
void check_gemm_equals_gemv(std::size_t rows, std::size_t cols) {
    using S = quant::scheme<T>;
    std::mt19937 rng(static_cast<unsigned>(rows * 131 + cols));
    const random_matrix<T> w(rows, cols, rng);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    for (std::size_t n : {1u, 2u, 5u, 16u}) {
        std::vector<float> x(n * cols);
        for (auto& v : x) {
            v = normal(rng);
        }
        std::vector<float> y(n * rows);
        std::vector<std::byte> scratch(kernels::gemm_scratch_bytes<S>(cols, n));
        kernels::gemm(w.view(), x, n, y, scratch);

        std::vector<float> yt(rows);
        std::vector<std::byte> gemv_scratch(kernels::gemv_scratch_bytes<S>(cols));
        for (std::size_t t = 0; t < n; ++t) {
            kernels::gemv(w.view(), std::span<const float>(x).subspan(t * cols, cols), yt, gemv_scratch);
            for (std::size_t r = 0; r < rows; ++r) {
                INFO("n " << n << " row " << t << " output " << r);
                CHECK(std::bit_cast<std::uint32_t>(y[t * rows + r]) == std::bit_cast<std::uint32_t>(yt[r]));
            }
        }
    }
}

}  // namespace

TEST_CASE("kernels_gemm: each batch row equals gemv bit for bit (G1)", "[kernels][gemm]") {
    check_gemm_equals_gemv<tensor_type::q4_0>(37, 512);
    check_gemm_equals_gemv<tensor_type::q5_0>(37, 512);
    check_gemm_equals_gemv<tensor_type::q8_0>(37, 512);
    check_gemm_equals_gemv<tensor_type::q4_k>(19, 512);
    check_gemm_equals_gemv<tensor_type::q6_k>(19, 512);
    check_gemm_equals_gemv<tensor_type::f32>(23, 300);  // a partial 256-element chunk
    check_gemm_equals_gemv<tensor_type::f16>(23, 600);
    // Several chunks, the last one partial: 2.5 chunks of Q8_0, chunks of
    // 4, 4 and 2 blocks of Q4_K.
    check_gemm_equals_gemv<tensor_type::q8_0>(37, 2560);
    check_gemm_equals_gemv<tensor_type::q4_k>(19, 2560);
}

TEST_CASE("kernels_gemm: the row split does not change any output (D1)", "[kernels][gemm]") {
    using S = quant::scheme<tensor_type::q8_0>;
    constexpr std::size_t rows = 45;
    constexpr std::size_t cols = 1536;
    constexpr std::size_t n = 7;
    std::mt19937 rng(11);
    const random_matrix<tensor_type::q8_0> w(rows, cols, rng);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> x(n * cols);
    for (auto& v : x) {
        v = normal(rng);
    }
    std::vector<std::byte> scratch(kernels::gemm_scratch_bytes<S>(cols, n));
    const auto* acts = kernels::gemm_prepare<S>(cols, x, n, scratch);
    std::vector<float> whole(n * rows);
    kernels::gemm_rows(w.view(), acts, n, whole, 0, rows);
    // Splits that cut register tiles and row blocks apart.
    std::vector<float> split(n * rows);
    for (auto [r0, r1] : {std::pair<std::size_t, std::size_t>{0, 3}, {3, 21}, {21, 22}, {22, 45}}) {
        kernels::gemm_rows(w.view(), acts, n, split, r0, r1);
    }
    for (std::size_t i = 0; i < whole.size(); ++i) {
        INFO("output " << i);
        CHECK(std::bit_cast<std::uint32_t>(split[i]) == std::bit_cast<std::uint32_t>(whole[i]));
    }
}

TEST_CASE("kernels_gemm: activation rows quantized in ranges equal gemm_prepare's", "[kernels][gemm]") {
    using S = quant::scheme<tensor_type::q4_k>;
    constexpr std::size_t cols = 512;
    constexpr std::size_t n = 5;
    std::mt19937 rng(12);
    const random_matrix<tensor_type::q4_k> w(9, cols, rng);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> x(n * cols);
    for (auto& v : x) {
        v = normal(rng);
    }
    std::vector<std::byte> s1(kernels::gemm_scratch_bytes<S>(cols, n));
    std::vector<std::byte> s2(s1.size());
    std::vector<float> y1(n * 9);
    std::vector<float> y2(n * 9);
    kernels::gemm_rows(w.view(), kernels::gemm_prepare<S>(cols, x, n, s1), n, y1, 0, 9);
    auto* acts = kernels::gemm_activations<S>(cols, n, s2);
    kernels::gemm_prepare_rows<S>(cols, x, s2, acts, 3, n);
    kernels::gemm_prepare_rows<S>(cols, x, s2, acts, 0, 3);
    kernels::gemm_rows(w.view(), acts, n, y2, 0, 9);
    CHECK(y1 == y2);
}

TEST_CASE("kernels_gemm: scratch grows linearly with the batch", "[kernels][gemm]") {
    using S = quant::scheme<tensor_type::q4_k>;
    const std::size_t one = kernels::gemm_scratch_bytes<S>(2048, 1);
    CHECK(one >= kernels::gemv_scratch_bytes<S>(2048));
    CHECK(kernels::gemm_scratch_bytes<S>(2048, 64) <= 64 * one);
    using P = quant::scheme<tensor_type::f32>;  // plain formats quantize nothing
    CHECK(kernels::gemm_scratch_bytes<P>(2048, 4) < 128);
}

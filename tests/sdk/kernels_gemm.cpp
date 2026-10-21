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
    check_gemm_equals_gemv<tensor_type::q8_0>(37, 512);
    check_gemm_equals_gemv<tensor_type::q4_k>(19, 512);
    check_gemm_equals_gemv<tensor_type::q6_k>(19, 512);
    check_gemm_equals_gemv<tensor_type::f32>(23, 300);  // a partial 256-element chunk
    check_gemm_equals_gemv<tensor_type::f16>(23, 600);
}

TEST_CASE("kernels_gemm: scratch grows linearly with the batch", "[kernels][gemm]") {
    using S = quant::scheme<tensor_type::q4_k>;
    const std::size_t one = kernels::gemm_scratch_bytes<S>(2048, 1);
    CHECK(one >= kernels::gemv_scratch_bytes<S>(2048));
    CHECK(kernels::gemm_scratch_bytes<S>(2048, 64) <= 64 * one);
    using P = quant::scheme<tensor_type::f32>;  // plain formats quantize nothing
    CHECK(kernels::gemm_scratch_bytes<P>(2048, 4) < 128);
}

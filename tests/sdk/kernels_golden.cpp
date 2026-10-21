// Golden tests for the fused kernels against ggml itself.
// tests/data/kernel_golden.gguf is written by scripts/gen_kernel_fixture.py,
// which runs ggml's own code (see scripts/kernel_oracle/oracle.cpp): weights
// quantized by ggml_quantize_chunk, activations quantized by ggml's reference
// quantizers, and y from ggml's CPU vec_dot kernels.
//
//   - Activations: synapse's quantizers must reproduce ggml's bit for bit.
//   - y: synapse's gemv must match ggml's vec_dot up to float re-association
//     (the per-block integer sums are exact on both sides).
//   - Accuracy: against the float reference dequantize(W) . x, the error must
//     stay within the exact bound sum |w| |x - x^| that rounding x to the
//     8-bit activations x^ allows.

#include <branes/synapse/gguf/mapped_file.hpp>
#include <branes/synapse/gguf/reader.hpp>
#include <branes/synapse/gguf/tensor_view.hpp>
#include <branes/synapse/kernels/gemv.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

using namespace branes::synapse;

namespace {

const gguf::gguf_file& fixture() {
    static const gguf::mapped_file file = [] {
        auto f = gguf::mapped_file::open(std::string(SYNAPSE_TEST_DATA_DIR) + "/kernel_golden.gguf");
        REQUIRE(f);
        return std::move(*f);
    }();
    static const gguf::gguf_file parsed = [] {
        auto p = gguf::parse(file.bytes());
        REQUIRE(p);
        return std::move(*p);
    }();
    return parsed;
}

std::vector<float> f32_tensor(const std::string& name) {
    const auto* t = fixture().find_tensor(name);
    REQUIRE(t != nullptr);
    REQUIRE(t->type == gguf::tensor_type::f32);
    std::vector<float> v(static_cast<std::size_t>(t->n_elements()));
    std::memcpy(v.data(), fixture().tensor_data(*t).data(), v.size() * sizeof(float));
    return v;
}

// The activations as floats: x^_e = d * q_e.
std::vector<float> dequantize_activations(const std::vector<quant::block_q8_0>& a) {
    std::vector<float> v;
    for (const auto& b : a) {
        for (auto q : b.qs) {
            v.push_back(fp16_to_float(b.d) * static_cast<float>(q));
        }
    }
    return v;
}

std::vector<float> dequantize_activations(const std::vector<quant::block_q8_k>& a) {
    std::vector<float> v;
    for (const auto& b : a) {
        for (auto q : b.qs) {
            v.push_back(b.d * static_cast<float>(q));
        }
    }
    return v;
}

template <quant::Scheme S>
void check_format(const std::string& fmt) {
    using A = typename S::vec_dot_block;
    const auto& file = fixture();
    const auto x = f32_tensor("x");
    const auto y_ggml = f32_tensor(fmt + ".y");
    const auto* wt = file.find_tensor(fmt + ".w");
    const auto* at = file.find_tensor(fmt + ".act");
    REQUIRE(wt != nullptr);
    REQUIRE(at != nullptr);
    const auto w = gguf::view_as<S>(file, *wt);
    REQUIRE(w);
    REQUIRE(w->cols() == x.size());

    // Activations: bit-identical to ggml's reference quantizer.
    std::vector<A> act(at->nbytes / sizeof(A));
    REQUIRE(act.size() * sizeof(A) == at->nbytes);
    if constexpr (std::is_same_v<A, quant::block_q8_0>) {
        quant::quantize_row_q8_0(x, act);
    } else {
        quant::quantize_row_q8_k(x, act);
    }
    CHECK(std::memcmp(act.data(), file.tensor_data(*at).data(), at->nbytes) == 0);

    // y: ggml's vec_dot up to float re-association.
    std::vector<float> y(w->rows());
    std::vector<std::byte> scratch(kernels::gemv_scratch_bytes<S>(w->cols()) + kernels::gemv_scratch_alignment);
    void* p = scratch.data();
    std::size_t space = scratch.size();
    REQUIRE(std::align(kernels::gemv_scratch_alignment, kernels::gemv_scratch_bytes<S>(w->cols()), p, space));
    kernels::gemv(*w, x, y, std::span<std::byte>(static_cast<std::byte*>(p), space));
    for (std::size_t r = 0; r < y.size(); ++r) {
        INFO("row " << r << ": synapse " << y[r] << ", ggml " << y_ggml[r]);
        CHECK(std::fabs(y[r] - y_ggml[r]) <= 1e-5f * std::fmax(1.0f, std::fabs(y_ggml[r])));
    }

    // Accuracy against the float reference, within the activation-rounding bound.
    const auto xq = dequantize_activations(act);
    std::vector<float> row(w->cols());
    for (std::size_t r = 0; r < w->rows(); ++r) {
        w->dequantize_row(r, row);
        double ref = 0.0;
        double bound = 0.0;
        double magnitude = 0.0;
        for (std::size_t c = 0; c < row.size(); ++c) {
            ref += static_cast<double>(row[c]) * x[c];
            bound += std::fabs(static_cast<double>(row[c])) * std::fabs(static_cast<double>(x[c]) - xq[c]);
            magnitude += std::fabs(static_cast<double>(row[c]) * xq[c]);
        }
        const double slack = 1e-6 * magnitude;  // float accumulation in the kernel
        INFO("row " << r << ": synapse " << y[r] << ", float reference " << ref << ", bound " << bound);
        CHECK(std::fabs(y[r] - ref) <= bound + slack);
    }
}

}  // namespace

TEST_CASE("kernels_golden: Q4_0 matches ggml", "[kernels][golden]") {
    check_format<quant::scheme<gguf::tensor_type::q4_0>>("q4_0");
}

TEST_CASE("kernels_golden: Q8_0 matches ggml", "[kernels][golden]") {
    check_format<quant::scheme<gguf::tensor_type::q8_0>>("q8_0");
}

TEST_CASE("kernels_golden: Q4_K matches ggml", "[kernels][golden]") {
    check_format<quant::scheme<gguf::tensor_type::q4_k>>("q4_K");
}

TEST_CASE("kernels_golden: Q6_K matches ggml", "[kernels][golden]") {
    check_format<quant::scheme<gguf::tensor_type::q6_k>>("q6_K");
}

#pragma once
// Random block-encoded weight matrices for tests: random bytes, with every
// fp16 scale field replaced by a finite value so no block decodes to inf/NaN.

#include <branes/synapse/numeric/half.hpp>
#include <branes/synapse/quant/scheme.hpp>
#include <branes/synapse/tensor/quantized_view.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace synapse_test {

using branes::synapse::quant::tensor_type;

// Owns the bytes of a random rows x cols matrix in scheme S.
template <tensor_type T>
struct random_matrix {
    using scheme = branes::synapse::quant::scheme<T>;

    std::vector<std::byte> data;
    std::size_t rows;
    std::size_t cols;

    random_matrix(std::size_t r, std::size_t c, std::mt19937& rng, float scale_lo = 1e-3f, float scale_hi = 2e-2f)
        : data(r * c / scheme::block_size * scheme::type_size), rows(r), cols(c) {
        std::uniform_int_distribution<int> byte(0, 255);
        std::uniform_real_distribution<float> scale(scale_lo, scale_hi);
        for (auto& b : data) {
            b = static_cast<std::byte>(byte(rng));
        }
        const auto put_f16 = [&](std::size_t block, std::size_t offset, float v) {
            const std::uint16_t h = branes::synapse::float_to_fp16(v);
            std::memcpy(data.data() + block * scheme::type_size + offset, &h, sizeof h);
        };
        const std::size_t nblocks = r * c / scheme::block_size;
        for (std::size_t i = 0; i < nblocks; ++i) {
            if constexpr (T == tensor_type::q4_0 || T == tensor_type::q8_0) {
                put_f16(i, 0, scale(rng));
            } else if constexpr (T == tensor_type::q4_k) {
                put_f16(i, 0, scale(rng));
                put_f16(i, 2, scale(rng));
            } else if constexpr (T == tensor_type::q6_k) {
                put_f16(i, 208, scale(rng));
            } else if constexpr (T == tensor_type::f16) {
                put_f16(i, 0, scale(rng));
            } else if constexpr (T == tensor_type::f32) {
                const float v = scale(rng);
                std::memcpy(data.data() + i * sizeof(float), &v, sizeof v);
            }
        }
    }

    branes::synapse::quantized_matrix_view<scheme> view() const {
        return {data, rows, cols};
    }
};

}  // namespace synapse_test

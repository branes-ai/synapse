// gemv_bench — fused quantized matrix-vector product vs. dequantize-then-dot.
//
// For each weight format, builds a rows x cols matrix of random blocks (finite
// fp16 scales, random quants), then times
//
//   fused     kernels::gemv: quantize x once, integer block dots per row
//   baseline  dequantize each row to float, then a float dot (step 2's path)
//
// and reports milliseconds per matvec and the weight bandwidth achieved.
// Single-threaded. Timings depend on the build: configure the
// gcc-release-hwy preset to see the Highway SIMD path.
//
// Usage: gemv_bench [rows cols reps]   (default 4096 4096 20)

#include <branes/synapse/kernels/gemv.hpp>
#include <branes/synapse/numeric/half.hpp>
#include <branes/synapse/quant/scheme.hpp>

#include <mtl/mat/block_matrix_view.hpp>
#include <mtl/simd/algorithm.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace branes::synapse;
using quant::tensor_type;

namespace {

// Random encoded matrix: random bytes, with every fp16 scale field replaced
// by a finite value so no block decodes to inf/NaN.
template <quant::Scheme S>
std::vector<std::byte> random_matrix(std::size_t rows, std::size_t cols, std::mt19937& rng) {
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(1e-3f, 2e-2f);
    const std::size_t nblocks = rows * cols / S::block_size;
    std::vector<std::byte> data(nblocks * S::type_size);
    for (auto& b : data) {
        b = static_cast<std::byte>(byte(rng));
    }
    const auto put_scale = [&](std::size_t block, std::size_t offset) {
        const std::uint16_t h = float_to_fp16(scale(rng));
        std::memcpy(data.data() + block * S::type_size + offset, &h, sizeof h);
    };
    for (std::size_t i = 0; i < nblocks; ++i) {
        if constexpr (S::type == tensor_type::q4_0 || S::type == tensor_type::q5_0 || S::type == tensor_type::q8_0 ||
                      S::type == tensor_type::f16) {
            put_scale(i, 0);  // d (or the value itself, for F16)
        } else if constexpr (S::type == tensor_type::q4_k) {
            put_scale(i, 0);  // d
            put_scale(i, 2);  // dmin
        } else if constexpr (S::type == tensor_type::q6_k) {
            put_scale(i, 208);  // d
        }
    }
    return data;
}

template <typename F>
double median_ms(int reps, F&& f) {
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) {
        const auto start = std::chrono::steady_clock::now();
        f();
        const auto stop = std::chrono::steady_clock::now();
        t.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

template <tensor_type T>
void bench(const char* name, std::size_t rows, std::size_t cols, int reps, std::mt19937& rng) {
    using S = quant::scheme<T>;
    const auto data = random_matrix<S>(rows, cols, rng);
    const mtl::mat::block_matrix_view<S> w(rows, cols, data);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> x(cols);
    for (auto& v : x) {
        v = normal(rng);
    }

    std::vector<float> y_fused(rows);
    std::vector<float> y_base(rows);
    std::vector<quant::block_q8_k> scratch_storage(kernels::gemv_scratch_bytes<S>(cols) / sizeof(quant::block_q8_k) +
                                                   1);
    const std::span<std::byte> scratch(reinterpret_cast<std::byte*>(scratch_storage.data()),
                                       scratch_storage.size() * sizeof(quant::block_q8_k));

    const double fused = median_ms(reps, [&] { kernels::gemv(w, x, y_fused, scratch); });
    std::vector<float> row(cols);
    const double base = median_ms(reps, [&] {
        for (std::size_t r = 0; r < rows; ++r) {
            w.decode_row(r, row.data());
            y_base[r] = mtl::simd::reduce_dot(row.data(), x.data(), cols);
        }
    });

    // Keep both results observable, and report how far apart they are.
    double max_diff = 0.0;
    double max_abs = 0.0;
    for (std::size_t r = 0; r < rows; ++r) {
        max_diff = std::max(max_diff, static_cast<double>(std::abs(y_fused[r] - y_base[r])));
        max_abs = std::max(max_abs, static_cast<double>(std::abs(y_base[r])));
    }
    const double gbytes = static_cast<double>(data.size()) / 1e9;
    std::printf("%-5s %9.3f ms %7.2f GB/s   %9.3f ms %7.2f GB/s   %5.2fx   rel diff %.1e\n",
                name,
                fused,
                gbytes / (fused / 1e3),
                base,
                gbytes / (base / 1e3),
                base / fused,
                max_abs > 0 ? max_diff / max_abs : 0.0);
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t rows = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 4096;
    const std::size_t cols = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 4096;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 20;
    if (rows == 0 || cols == 0 || cols % 256 != 0 || reps <= 0) {
        std::fprintf(stderr, "rows > 0, cols a positive multiple of 256, reps > 0\n");
        return 1;
    }
#if defined(MTL5_HAS_HIGHWAY)
    const char* backend = "Highway";
#else
    const char* backend = "scalar";
#endif
    std::printf("gemv %zu x %zu, median of %d, mtl::simd backend: %s\n", rows, cols, reps, backend);
    std::printf("fmt         fused                    dequantize + dot       speedup\n");
    std::mt19937 rng(42);
    bench<tensor_type::q4_0>("q4_0", rows, cols, reps, rng);
    bench<tensor_type::q5_0>("q5_0", rows, cols, reps, rng);
    bench<tensor_type::q8_0>("q8_0", rows, cols, reps, rng);
    bench<tensor_type::q4_k>("q4_K", rows, cols, reps, rng);
    bench<tensor_type::q6_k>("q6_K", rows, cols, reps, rng);
    bench<tensor_type::f16>("f16", rows, cols, reps, rng);
    return 0;
}

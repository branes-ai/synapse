// kernel_oracle — writes tests/data/kernel_golden.gguf from ggml itself.
//
// Built and run by scripts/gen_kernel_fixture.py against a local llama.cpp
// checkout. For each weight format it records, using ggml's own code:
//
//   <fmt>.w    weights quantized by ggml_quantize_chunk (realistic data)
//   <fmt>.act  the activations x quantized by ggml's reference quantizer for
//              the format's vec_dot_type (quantize_row_q8_0_ref / _q8_K_ref)
//   <fmt>.y    y[r] = ggml's CPU vec_dot(row r of .w, .act)
//
// plus the shared float input `x`. synapse's tests check that its quantizers
// reproduce .act bit for bit and that its gemv reproduces .y to float
// re-association error. The file is written with ggml's gguf writer, so it
// also exercises synapse's parser on a ggml-produced image.

#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// Exported by libggml-base (declared in the internal ggml-quants.h).
extern "C" void quantize_row_q8_K_ref(const float* x, void* y, int64_t k);

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <output.gguf> <description>\n", argv[0]);
        return 2;
    }
    constexpr int64_t rows = 8;
    constexpr int64_t cols = 1024;  // 4 Q8_K blocks, 32 Q8_0 blocks per row

    ggml_cpu_init();
    ggml_init_params params{};
    params.mem_size = 64 * 1024 * 1024;
    ggml_context* ctx = ggml_init(params);
    gguf_context* out = gguf_init_empty();
    gguf_set_val_str(out, "general.architecture", "synapse-test");
    gguf_set_val_str(out, "synapse.fixture.generator", "scripts/gen_kernel_fixture.py");
    gguf_set_val_str(out, "synapse.fixture.reference", argv[2]);

    std::mt19937 rng(20261021);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cols);
    ggml_set_name(x, "x");
    auto* xs = static_cast<float*>(x->data);
    for (int64_t i = 0; i < cols; ++i) {
        xs[i] = normal(rng);
    }
    gguf_add_tensor(out, x);

    const ggml_type types[] = {GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K};
    for (ggml_type type : types) {
        const std::string fmt = ggml_type_name(type);

        std::vector<float> wf(static_cast<size_t>(rows * cols));
        for (auto& v : wf) {
            v = 0.05f * normal(rng);  // a typical weight scale
        }
        ggml_tensor* w = ggml_new_tensor_2d(ctx, type, cols, rows);
        ggml_set_name(w, (fmt + ".w").c_str());
        ggml_quantize_chunk(type, wf.data(), w->data, 0, rows, cols, nullptr);

        const ggml_type vt = ggml_get_type_traits_cpu(type)->vec_dot_type;
        ggml_tensor* act = ggml_new_tensor_1d(ctx, vt, cols);
        ggml_set_name(act, (fmt + ".act").c_str());
        if (vt == GGML_TYPE_Q8_K) {
            quantize_row_q8_K_ref(xs, act->data, cols);
        } else {
            ggml_get_type_traits(vt)->from_float_ref(xs, act->data, cols);
        }

        ggml_tensor* y = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, rows);
        ggml_set_name(y, (fmt + ".y").c_str());
        auto* ys = static_cast<float*>(y->data);
        const auto vec_dot = ggml_get_type_traits_cpu(type)->vec_dot;
        for (int64_t r = 0; r < rows; ++r) {
            const char* row = static_cast<const char*>(w->data) + r * w->nb[1];
            vec_dot(static_cast<int>(cols), &ys[r], 0, row, 0, act->data, 0, 1);
        }

        gguf_add_tensor(out, w);
        gguf_add_tensor(out, act);
        gguf_add_tensor(out, y);
        std::printf("%-5s vec_dot_type=%s y[0]=%g\n", fmt.c_str(), ggml_type_name(vt), static_cast<double>(ys[0]));
    }

    if (!gguf_write_to_file(out, argv[1], false)) {
        std::fprintf(stderr, "failed to write %s\n", argv[1]);
        return 1;
    }
    gguf_free(out);
    ggml_free(ctx);
    return 0;
}

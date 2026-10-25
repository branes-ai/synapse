// clip_oracle — golden image embeddings from llama.cpp's mmproj encoder (clip.cpp).
//
// Built and run by scripts/gen_clip_fixture.py against a local llama.cpp
// checkout. Modes:
//
//   oracle tiny     <mmproj.gguf> <golden.gguf>   a tiny random-weight idefics3 mmproj, F32 matrices
//   oracle tiny-q8  <mmproj.gguf> <golden.gguf>   the same with Q8_0 matrices, FFN stored swapped
//   oracle encode   <mmproj.gguf> <golden.gguf>   read an existing mmproj (e.g. SmolVLM's)
//
// The tiny modes write the mmproj first. Every mode then encodes one
// deterministic image (image_size^2 interleaved RGB floats, already
// normalized: what clip_image_f32 holds after preprocessing) with
// clip_encode_float_image on the CPU, flash attention off, and writes:
//
//   image       F32 [3 * image_size^2]       the input
//   embeddings  F32 [n_embd_text, n_tokens]  the projector output, one row per token
//
// The tiny files mirror what llama.cpp's converter writes for SmolVLM: F32
// norms, biases, position embeddings and patch kernel; the matrices F32 or
// Q8_0. tiny-q8 stores the FFN with ffn_up and ffn_down swapped, as
// SmolVLM's files do, so the loader's swap rule is exercised both ways.

#include "clip.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

std::mt19937 rng(20261024);

// A tensor of `type` with ne = {ne0, ne1, ne2, ne3} and N(mean, scale)
// entries, quantized by ggml.
void add_weight(ggml_context* ctx,
                gguf_context* out,
                const std::string& name,
                ggml_type type,
                std::vector<int64_t> ne,
                float mean,
                float scale) {
    std::normal_distribution<float> normal(mean, scale);
    int64_t n = 1;
    for (int64_t d : ne) {
        n *= d;
    }
    std::vector<float> w(static_cast<size_t>(n));
    for (auto& v : w) {
        v = normal(rng);
    }
    ggml_tensor* t = ggml_new_tensor(ctx, type, static_cast<int>(ne.size()), ne.data());
    ggml_set_name(t, name.c_str());
    ggml_quantize_chunk(type, w.data(), t->data, 0, n / ne[0], ne[0], nullptr);
    gguf_add_tensor(out, t);
}

constexpr uint32_t image_size = 64;
constexpr uint32_t patch_size = 8;
constexpr uint32_t n_embd = 64;
constexpr uint32_t n_ff = 128;
constexpr uint32_t n_head = 4;
constexpr uint32_t n_layer = 2;
constexpr uint32_t scale_factor = 2;
constexpr uint32_t n_embd_text = 48;

void write_tiny_mmproj(const char* path, ggml_type mat, bool swap_ffn) {
    ggml_init_params params{};
    params.mem_size = 64 * 1024 * 1024;
    ggml_context* ctx = ggml_init(params);
    gguf_context* out = gguf_init_empty();
    gguf_set_val_str(out, "general.architecture", "clip");
    gguf_set_val_str(out, "general.name", "synapse tiny idefics3 mmproj (random weights)");
    gguf_set_val_str(out, "clip.projector_type", "idefics3");
    gguf_set_val_bool(out, "clip.has_vision_encoder", true);
    gguf_set_val_bool(out, "clip.use_gelu", true);
    gguf_set_val_u32(out, "clip.vision.image_size", image_size);
    gguf_set_val_u32(out, "clip.vision.patch_size", patch_size);
    gguf_set_val_u32(out, "clip.vision.embedding_length", n_embd);
    gguf_set_val_u32(out, "clip.vision.feed_forward_length", n_ff);
    gguf_set_val_u32(out, "clip.vision.block_count", n_layer);
    gguf_set_val_u32(out, "clip.vision.attention.head_count", n_head);
    gguf_set_val_f32(out, "clip.vision.attention.layer_norm_epsilon", 1e-6f);
    gguf_set_val_u32(out, "clip.vision.projection_dim", n_embd_text);
    gguf_set_val_u32(out, "clip.vision.projector.scale_factor", scale_factor);
    const float half[3] = {0.5f, 0.5f, 0.5f};
    gguf_set_arr_data(out, "clip.vision.image_mean", GGUF_TYPE_FLOAT32, half, 3);
    gguf_set_arr_data(out, "clip.vision.image_std", GGUF_TYPE_FLOAT32, half, 3);

    const int64_t grid = image_size / patch_size;
    const float wscale = 1.0f / std::sqrt(static_cast<float>(n_embd));
    add_weight(ctx, out, "v.patch_embd.weight", GGML_TYPE_F32, {patch_size, patch_size, 3, n_embd}, 0.0f, 0.1f);
    add_weight(ctx, out, "v.patch_embd.bias", GGML_TYPE_F32, {n_embd}, 0.0f, 0.1f);
    add_weight(ctx, out, "v.position_embd.weight", GGML_TYPE_F32, {n_embd, grid * grid}, 0.0f, 0.5f);
    for (uint32_t l = 0; l < n_layer; ++l) {
        const std::string p = "v.blk." + std::to_string(l) + ".";
        add_weight(ctx, out, p + "ln1.weight", GGML_TYPE_F32, {n_embd}, 1.0f, 0.1f);
        add_weight(ctx, out, p + "ln1.bias", GGML_TYPE_F32, {n_embd}, 0.0f, 0.1f);
        for (const char* m : {"attn_q", "attn_k", "attn_v", "attn_out"}) {
            add_weight(ctx, out, p + m + ".weight", mat, {n_embd, n_embd}, 0.0f, wscale);
            add_weight(ctx, out, p + m + ".bias", GGML_TYPE_F32, {n_embd}, 0.0f, 0.1f);
        }
        add_weight(ctx, out, p + "ln2.weight", GGML_TYPE_F32, {n_embd}, 1.0f, 0.1f);
        add_weight(ctx, out, p + "ln2.bias", GGML_TYPE_F32, {n_embd}, 0.0f, 0.1f);
        const std::string up = swap_ffn ? "ffn_down" : "ffn_up";
        const std::string down = swap_ffn ? "ffn_up" : "ffn_down";
        add_weight(ctx, out, p + up + ".weight", mat, {n_embd, n_ff}, 0.0f, wscale);
        add_weight(ctx, out, p + up + ".bias", GGML_TYPE_F32, {n_ff}, 0.0f, 0.1f);
        add_weight(ctx, out, p + down + ".weight", mat, {n_ff, n_embd}, 0.0f, 1.0f / std::sqrt(float(n_ff)));
        add_weight(ctx, out, p + down + ".bias", GGML_TYPE_F32, {n_embd}, 0.0f, 0.1f);
    }
    add_weight(ctx, out, "v.post_ln.weight", GGML_TYPE_F32, {n_embd}, 1.0f, 0.1f);
    add_weight(ctx, out, "v.post_ln.bias", GGML_TYPE_F32, {n_embd}, 0.0f, 0.1f);
    const int64_t merged = n_embd * scale_factor * scale_factor;
    add_weight(ctx, out, "mm.model.fc.weight", mat, {merged, n_embd_text}, 0.0f, 1.0f / std::sqrt(float(merged)));

    if (!gguf_write_to_file(out, path, false)) {
        std::fprintf(stderr, "failed to write %s\n", path);
        std::exit(1);
    }
    gguf_free(out);
    ggml_free(ctx);
}

// A smooth image with detail at every scale, in [-1, 1] (normalized pixels).
std::vector<float> test_image(int size) {
    std::vector<float> img(static_cast<size_t>(3 * size * size));
    std::uniform_real_distribution<float> noise(-0.2f, 0.2f);
    std::mt19937 img_rng(7);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            for (int c = 0; c < 3; ++c) {
                const float u = static_cast<float>(x) / static_cast<float>(size);
                const float v = static_cast<float>(y) / static_cast<float>(size);
                const float s = std::sin(6.0f * u + 2.0f * static_cast<float>(c)) * std::cos(4.0f * v - u);
                img[static_cast<size_t>((y * size + x) * 3 + c)] = 0.8f * s + noise(img_rng);
            }
        }
    }
    return img;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s tiny|tiny-q8|encode <mmproj.gguf> <golden.gguf>\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    if (mode == "tiny") {
        write_tiny_mmproj(argv[2], GGML_TYPE_F32, false);
    } else if (mode == "tiny-q8") {
        write_tiny_mmproj(argv[2], GGML_TYPE_Q8_0, true);
    } else if (mode != "encode") {
        std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 2;
    }

    clip_context_params cp{};
    cp.use_gpu = false;
    cp.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
    cp.warmup = false;
    clip_ctx* ctx = clip_init(argv[2], cp).ctx_v;
    if (ctx == nullptr) {
        std::fprintf(stderr, "clip_init failed for %s\n", argv[2]);
        return 1;
    }
    const int size = clip_get_image_size(ctx);
    std::vector<float> img = test_image(size);
    const int n_embd_out = clip_n_mmproj_embd(ctx);
    std::vector<float> emb(clip_embd_nbytes_by_img(ctx, size, size) / sizeof(float));
    if (!clip_encode_float_image(ctx, 1, img.data(), size, size, emb.data())) {
        std::fprintf(stderr, "encode failed\n");
        return 1;
    }
    const int64_t n_tokens = static_cast<int64_t>(emb.size()) / n_embd_out;

    ggml_init_params params{};
    params.mem_size = (img.size() + emb.size()) * sizeof(float) + 1024 * 1024;
    ggml_context* gctx = ggml_init(params);
    gguf_context* out = gguf_init_empty();
    gguf_set_val_str(out, "general.architecture", "synapse-test");
    gguf_set_val_str(out, "synapse.fixture.generator", "scripts/gen_clip_fixture.py");
    gguf_set_val_str(out, "synapse.fixture.mode", mode.c_str());
    ggml_tensor* ti = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, static_cast<int64_t>(img.size()));
    ggml_set_name(ti, "image");
    std::memcpy(ti->data, img.data(), img.size() * sizeof(float));
    gguf_add_tensor(out, ti);
    ggml_tensor* te = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_embd_out, n_tokens);
    ggml_set_name(te, "embeddings");
    std::memcpy(te->data, emb.data(), emb.size() * sizeof(float));
    gguf_add_tensor(out, te);
    if (!gguf_write_to_file(out, argv[3], false)) {
        std::fprintf(stderr, "failed to write %s\n", argv[3]);
        return 1;
    }
    std::printf("%s: %d x %d image -> %lld tokens x %d, e[0] = %g\n",
                mode.c_str(),
                size,
                size,
                static_cast<long long>(n_tokens),
                n_embd_out,
                static_cast<double>(emb[0]));
    gguf_free(out);
    ggml_free(gctx);
    clip_free(ctx);
    return 0;
}

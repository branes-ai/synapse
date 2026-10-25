#pragma once
/// @file clip.hpp
/// @brief Build the image-encoder graph of a llama.cpp mmproj file (SigLIP ViT, idefics3 projector).
///
/// A vision-language model's GGUF release keeps its vision tower in a
/// separate "mmproj" file, read by llama.cpp's clip.cpp. This loader builds
/// the encoder of projector type "idefics3" (SmolVLM, Idefics3): a SigLIP
/// ViT followed by a pixel shuffle and one linear projection into the text
/// model's embedding space. The graph mirrors clip_graph_siglip::build:
///
///   x = Wpatch patches + bpatch + pos_embd          (one row per patch)
///   per layer:
///     x = x + Wo self_attention(Wq h + bq, Wk h + bk, Wv h + bv) + bo,  h = LN1(x)
///     x = x + Wdown gelu(Wup h + bup) + bdown,                         h = LN2(x)
///   embeddings = Wfc pixel_shuffle(LN_post(x), s)    (one row per s x s patches)
///
/// The patch embedding is the stride-p convolution with a p x p kernel
/// written as a matmul: encode() unfolds each p x p patch of the image into
/// one row, in the (channel, y, x) order of the kernel's 3 x p x p weights.
///
/// Scope: one image_size x image_size tile, given as floats already
/// normalized with the file's image_mean and image_std (what clip.cpp's
/// clip_image_f32 holds after preprocessing). Resizing, normalization and
/// Idefics3's split of a large image into tiles are the caller's.
///
/// The weights stay in the mapped file: the graph holds views into it, so the
/// gguf_file (and the image under it) must outlive the model and its sessions.
///
/// Model
/// -----
///   clip_config — hyperparameters read from the clip.* metadata keys
///   clip_model
///     config     : clip_config
///     graph      : compute_graph — max_batch = n_patches; full runs only (I9)
///     patches    : tensor_id — input: n_patches rows of 3 * patch_size^2 floats
///     embeddings : tensor_id — output: n_tokens rows of n_embd_text floats
///
/// Contract
/// --------
///   load_clip(file)
///     Pre:  none (the file is untrusted).
///     Post: on success, a graph satisfying I1-I9 whose every weight view has
///           the shape the config implies; on failure, what was wrong and,
///           for a key or tensor, its name. Never throws for bad input.
///   encode(session, model, image, out)
///     Pre:  session runs model.graph; image.size() == 3 * image_size^2;
///           out.size() == n_tokens * n_embd_text.
///     Post: out holds the embeddings, row t for the s x s patch block t.
///
/// Not supported (rejected, never silently mis-computed): other projector
/// types, an audio encoder, activations other than GELU (clip.use_gelu),
/// pre-layernorm, fused QKV, Q/K norms, layer scales, gated FFNs, a class
/// embedding, and a projector bias.

#include <branes/synapse/gguf/reader.hpp>
#include <branes/synapse/graph/graph.hpp>
#include <branes/synapse/graph/session.hpp>
#include <branes/synapse/models/common.hpp>
#include <branes/synapse/result.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

namespace branes::synapse::models {

struct clip_config {
    std::size_t image_size = 0;    ///< pixels per side of the square input
    std::size_t patch_size = 0;    ///< pixels per side of a patch
    std::size_t n_embd = 0;        ///< ViT width
    std::size_t n_ff = 0;          ///< ViT MLP width
    std::size_t n_layer = 0;       ///< ViT blocks
    std::size_t n_head = 0;        ///< attention heads
    float eps = 0.0f;              ///< LayerNorm epsilon
    std::size_t scale_factor = 0;  ///< pixel shuffle s: s x s patches per output token
    std::size_t n_embd_text = 0;   ///< width of an output embedding (the text model's)
    std::array<float, 3> image_mean{};
    std::array<float, 3> image_std{};

    std::size_t grid() const noexcept {
        return image_size / patch_size;
    }
    std::size_t n_patches() const noexcept {
        return grid() * grid();
    }
    std::size_t n_tokens() const noexcept {
        return n_patches() / (scale_factor * scale_factor);
    }
    std::size_t patch_floats() const noexcept {
        return 3 * patch_size * patch_size;
    }
};

struct clip_model {
    clip_config config;
    graph::compute_graph graph;
    graph::tensor_id patches = graph::invalid_tensor;
    graph::tensor_id embeddings = graph::invalid_tensor;
};

namespace detail {

/// A metadata bool, or fallback if absent.
inline result<bool, model_error> get_bool(const gguf::gguf_file& f, std::string_view key, bool fallback) {
    const gguf::metadata_value* v = f.find(key);
    if (v == nullptr) {
        return fallback;
    }
    if (const auto* b = std::get_if<bool>(v)) {
        return *b;
    }
    return fail(model_errc::bad_value, key);
}

/// Three finite floats (image_mean, image_std), stored as an F32 array.
inline result<std::array<float, 3>, model_error> get_rgb(const gguf::gguf_file& f, std::string_view key) {
    const gguf::array_value* a = f.get_array(key);
    if (a == nullptr) {
        return fail(f.find(key) == nullptr ? model_errc::missing_key : model_errc::bad_value, key);
    }
    std::array<float, 3> rgb{};
    if (a->size() != rgb.size()) {
        return fail(model_errc::bad_value, key);
    }
    for (std::size_t i = 0; i < rgb.size(); ++i) {
        const auto x = a->at<float>(i);
        if (!x || !std::isfinite(*x)) {
            return fail(model_errc::bad_value, key);
        }
        rgb[i] = *x;
    }
    return rgb;
}

inline std::string vision_layer_tensor(std::size_t layer, std::string_view suffix) {
    return "v.blk." + std::to_string(layer) + "." + std::string(suffix);
}

/// The p x p x 3 x n_embd patch-embedding kernel as an n_embd x 3p^2 matrix:
/// each output channel's weights are contiguous, in (channel, y, x) order.
inline result<graph::matrix_ref, model_error>
bind_patch_kernel(const gguf::gguf_file& f, const std::string& name, std::size_t p, std::size_t n_embd) {
    const gguf::tensor_info* t = f.find_tensor(name);
    if (t == nullptr) {
        return fail(model_errc::missing_tensor, name);
    }
    if (t->ne[0] != p || t->ne[1] != p || t->ne[2] != 3 || t->ne[3] != n_embd) {
        return fail(model_errc::bad_tensor, name);
    }
    std::optional<graph::matrix_ref> m;
    const bool known = gguf::visit_matrix(f, *t, [&](const auto& view) {
        using V = std::decay_t<decltype(view)>;
        // A row of 3p^2 elements must hold whole blocks; the parser only
        // guarantees that for rows of p.
        if (3 * p * p % V::scheme_type::block_size == 0) {
            m = V(n_embd, 3 * p * p, f.tensor_data(*t));
        }
    });
    if (!known) {
        return fail(model_errc::unsupported_type, name);
    }
    if (!m) {
        return fail(model_errc::bad_tensor, name);
    }
    return *m;
}

}  // namespace detail

/// The hyperparameters of an mmproj file, read from its "clip." keys.
inline result<clip_config, model_error> read_clip_config(const gguf::gguf_file& f) {
    using detail::fail;
    if (f.get<std::string_view>("clip.projector_type") != std::string_view{"idefics3"}) {
        return fail(model_errc::wrong_architecture, "clip.projector_type");
    }
    for (auto [key, want] : {std::pair{"clip.has_vision_encoder", true}, std::pair{"clip.has_audio_encoder", false}}) {
        auto v = detail::get_bool(f, key, false);
        if (!v) {
            return v.error();
        }
        if (*v != want) {
            return fail(model_errc::unsupported, key);
        }
    }
    auto gelu = detail::get_bool(f, "clip.use_gelu", false);
    if (!gelu) {
        return gelu.error();
    }
    if (!*gelu) {
        return fail(model_errc::unsupported, "clip.use_gelu");  // gelu_quick or silu
    }

    clip_config c;
    for (auto [key, out] : {std::pair{"clip.vision.image_size", &c.image_size},
                            std::pair{"clip.vision.patch_size", &c.patch_size},
                            std::pair{"clip.vision.embedding_length", &c.n_embd},
                            std::pair{"clip.vision.feed_forward_length", &c.n_ff},
                            std::pair{"clip.vision.block_count", &c.n_layer},
                            std::pair{"clip.vision.attention.head_count", &c.n_head},
                            std::pair{"clip.vision.projector.scale_factor", &c.scale_factor},
                            std::pair{"clip.vision.projection_dim", &c.n_embd_text}}) {
        auto v = detail::get_count(f, key);
        if (!v) {
            return v.error();
        }
        if (*v == 0) {
            return fail(model_errc::bad_value, key);
        }
        *out = *v;
    }
    // Bounds that keep every product below exact and far from wrapping; real
    // encoders are orders of magnitude inside them.
    if (c.image_size > 1u << 16) {
        return fail(model_errc::bad_value, "clip.vision.image_size");
    }
    if (c.patch_size > c.image_size || c.image_size % c.patch_size != 0) {
        return fail(model_errc::bad_value, "clip.vision.patch_size");
    }
    if (c.n_embd % c.n_head != 0) {
        return fail(model_errc::bad_value, "clip.vision.attention.head_count");
    }
    if (c.scale_factor < 2 || c.grid() % c.scale_factor != 0) {
        return fail(model_errc::unsupported, "clip.vision.projector.scale_factor");  // llama.cpp would pad
    }
    if (c.n_embd > std::numeric_limits<std::uint32_t>::max() / (c.scale_factor * c.scale_factor)) {
        return fail(model_errc::bad_value, "clip.vision.embedding_length");  // keeps n_embd * s^2 exact
    }
    auto eps = detail::get_positive(f, "clip.vision.attention.layer_norm_epsilon", std::nullopt);
    if (!eps) {
        return eps.error();
    }
    c.eps = *eps;
    auto mean = detail::get_rgb(f, "clip.vision.image_mean");
    if (!mean) {
        return mean.error();
    }
    c.image_mean = *mean;
    auto sd = detail::get_rgb(f, "clip.vision.image_std");
    if (!sd) {
        return sd.error();
    }
    c.image_std = *sd;
    return c;
}

/// The encoder graph of an mmproj file.  See the file comment for the contract.
inline result<clip_model, model_error> load_clip(const gguf::gguf_file& f) {
    using detail::fail;
    auto cfg = read_clip_config(f);
    if (!cfg) {
        return cfg.error();
    }
    const clip_config& c = *cfg;
    // Rejected rather than ignored: they change the result.
    for (std::string_view extra : {"v.pre_ln.weight", "v.class_embd", "mm.model.fc.bias"}) {
        if (f.find_tensor(extra) != nullptr) {
            return fail(model_errc::unsupported, extra);
        }
    }

    graph::graph_builder b(c.n_patches());
    std::optional<model_error> error;
    const auto weight = [&](const std::string& name, std::size_t rows, std::size_t cols) -> graph::matrix_ref {
        if (error) {
            return {};
        }
        auto m = detail::bind(f, name, rows, cols);
        if (!m) {
            error = m.error();
            return {};
        }
        return *m;
    };
    // 1 x n if present, else empty (clip.cpp treats every bias as optional).
    const auto bias = [&](const std::string& name, std::size_t n) -> graph::matrix_ref {
        return f.find_tensor(name) != nullptr ? weight(name, 1, n) : graph::matrix_ref{};
    };
    const auto linear = [&](const graph::matrix_ref& w, const graph::matrix_ref& bv, graph::tensor_id h) {
        const graph::tensor_id y = b.matmul(w, h);
        return std::holds_alternative<std::monostate>(bv) ? y : b.add_bias(y, bv);
    };

    auto patch_w = detail::bind_patch_kernel(f, "v.patch_embd.weight", c.patch_size, c.n_embd);
    if (!patch_w) {
        return patch_w.error();
    }
    const graph::matrix_ref patch_b = bias("v.patch_embd.bias", c.n_embd);
    const graph::matrix_ref pos = weight("v.position_embd.weight", c.n_patches(), c.n_embd);
    if (error) {
        return std::move(*error);
    }

    const graph::tensor_id patches = b.input(c.patch_floats());
    graph::tensor_id x = b.add_rows(linear(*patch_w, patch_b, patches), pos);
    const float scale = 1.0f / std::sqrt(static_cast<float>(c.n_embd / c.n_head));
    for (std::size_t l = 0; l < c.n_layer; ++l) {
        const auto name = [&](std::string_view suffix) { return detail::vision_layer_tensor(l, suffix); };
        for (std::string_view extra : {"attn_qkv.weight",
                                       "attn_q_norm.weight",
                                       "attn_k_norm.weight",
                                       "ls1.weight",
                                       "ls2.weight",
                                       "ffn_gate.weight"}) {
            if (f.find_tensor(name(extra)) != nullptr) {
                return fail(model_errc::unsupported, name(extra));
            }
        }
        // clip.cpp swaps ffn_up and ffn_down (and their biases) when the file
        // stores them the other way round: the up projection is the one that
        // reads n_embd-wide rows.
        const gguf::tensor_info* down = f.find_tensor(name("ffn_down.weight"));
        const bool swapped = down != nullptr && down->ne[0] == c.n_embd;
        const std::string up_name = swapped ? "ffn_down" : "ffn_up";
        const std::string down_name = swapped ? "ffn_up" : "ffn_down";

        const graph::matrix_ref ln1_w = weight(name("ln1.weight"), 1, c.n_embd);
        const graph::matrix_ref ln1_b = bias(name("ln1.bias"), c.n_embd);
        const graph::matrix_ref wq = weight(name("attn_q.weight"), c.n_embd, c.n_embd);
        const graph::matrix_ref wk = weight(name("attn_k.weight"), c.n_embd, c.n_embd);
        const graph::matrix_ref wv = weight(name("attn_v.weight"), c.n_embd, c.n_embd);
        const graph::matrix_ref wo = weight(name("attn_out.weight"), c.n_embd, c.n_embd);
        const graph::matrix_ref bq = bias(name("attn_q.bias"), c.n_embd);
        const graph::matrix_ref bk = bias(name("attn_k.bias"), c.n_embd);
        const graph::matrix_ref bv = bias(name("attn_v.bias"), c.n_embd);
        const graph::matrix_ref bo = bias(name("attn_out.bias"), c.n_embd);
        const graph::matrix_ref ln2_w = weight(name("ln2.weight"), 1, c.n_embd);
        const graph::matrix_ref ln2_b = bias(name("ln2.bias"), c.n_embd);
        const graph::matrix_ref up = weight(name(up_name + ".weight"), c.n_ff, c.n_embd);
        const graph::matrix_ref up_b = bias(name(up_name + ".bias"), c.n_ff);
        const graph::matrix_ref dn = weight(name(down_name + ".weight"), c.n_embd, c.n_ff);
        const graph::matrix_ref dn_b = bias(name(down_name + ".bias"), c.n_embd);
        if (error) {
            return std::move(*error);
        }

        const graph::tensor_id h = b.layer_norm(x, ln1_w, ln1_b, c.eps);
        const graph::tensor_id attn =
            b.self_attention(linear(wq, bq, h), linear(wk, bk, h), linear(wv, bv, h), c.n_head, scale);
        x = b.add(linear(wo, bo, attn), x);
        const graph::tensor_id h2 = b.layer_norm(x, ln2_w, ln2_b, c.eps);
        x = b.add(x, linear(dn, dn_b, b.gelu(linear(up, up_b, h2))));
    }
    if (f.find_tensor("v.post_ln.weight") != nullptr) {
        const graph::matrix_ref post_w = weight("v.post_ln.weight", 1, c.n_embd);
        const graph::matrix_ref post_b = bias("v.post_ln.bias", c.n_embd);
        if (error) {
            return std::move(*error);
        }
        x = b.layer_norm(x, post_w, post_b, c.eps);
    }
    const std::size_t merged = c.n_embd * c.scale_factor * c.scale_factor;
    const graph::matrix_ref fc = weight("mm.model.fc.weight", c.n_embd_text, merged);
    if (error) {
        return std::move(*error);
    }
    const graph::tensor_id embeddings = b.matmul(fc, b.pixel_shuffle(x, c.grid(), c.scale_factor));
    b.output(embeddings);

    auto g = std::move(b).finalize();
    if (!g) {
        return model_error{model_errc::graph, {}, g.error()};
    }
    return clip_model{c, std::move(*g), patches, embeddings};
}

/// Unfold an image_size x image_size image (interleaved RGB floats, row by
/// row) into its patches: row px + grid * py holds patch (px, py) as
/// image[c][py * p + y][px * p + x] at offset (c * p + y) * p + x, the order
/// of the patch kernel's weights.
///   Pre: image.size() == 3 * image_size^2; out.size() == n_patches * 3p^2.
inline void unfold_patches(const clip_config& c, std::span<const float> image, std::span<float> out) noexcept {
    const std::size_t p = c.patch_size;
    const std::size_t w = c.image_size;
    assert(image.size() == 3 * w * w && out.size() == c.n_patches() * c.patch_floats());  // Pre
    for (std::size_t py = 0; py < c.grid(); ++py) {
        for (std::size_t px = 0; px < c.grid(); ++px) {
            float* dst = out.data() + (px + c.grid() * py) * c.patch_floats();
            for (std::size_t ch = 0; ch < 3; ++ch) {
                for (std::size_t y = 0; y < p; ++y) {
                    for (std::size_t x = 0; x < p; ++x) {
                        dst[(ch * p + y) * p + x] = image[((py * p + y) * w + px * p + x) * 3 + ch];
                    }
                }
            }
        }
    }
}

/// Encode one image: unfold it into the session's input, run the graph and
/// copy out the embeddings. Returns false only if the run is rejected, which
/// cannot happen for a session of model.graph.
///   Pre: s runs model.graph; image.size() == 3 * image_size^2;
///        out.size() == n_tokens * n_embd_text.
[[nodiscard]] inline bool
encode(graph::session& s, const clip_model& m, std::span<const float> image, std::span<float> out) noexcept {
    const clip_config& c = m.config;
    assert(out.size() == c.n_tokens() * c.n_embd_text);  // Pre
    unfold_patches(c, image, s.tensor(m.patches));
    if (!s.run_rows(c.n_patches())) {
        return false;
    }
    const std::span<const float> e = s.tensor(m.embeddings);
    std::copy(e.begin(), e.begin() + static_cast<std::ptrdiff_t>(out.size()), out.begin());
    return true;
}

}  // namespace branes::synapse::models

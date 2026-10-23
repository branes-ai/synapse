#pragma once
/// @file llama.hpp
/// @brief Build the decode graph of a Llama-family GGUF model (Llama, Qwen2, Mistral 3).
///
/// The graph mirrors llama.cpp's llm_build_llama, llm_build_qwen2 and llm_build_mistral3:
///
///   x = embed(token_embd)
///   per layer:
///     h = rms_norm(x, attn_norm)
///     q = rope(Wq h + bq) * temp(pos), k = rope(Wk h + bk), v = Wv h + bv
///     x = x + Wo attention(q, k, v, kv_cache) + bo
///     h = rms_norm(x, ffn_norm)
///     x = x + Wdown (silu(Wgate h + bgate) * (Wup h + bup)) + bdown
///   logits = Wout rms_norm(last_row(x), output_norm) + bout   (Wout = token_embd if tied)
///
/// The architectures differ only where llama.cpp's builders do:
///
///   general.architecture   metadata keys   RoPE pairs        biases (optional)
///   "llama"                llama.*         normal (2i, 2i+1)  none
///   "qwen2"                qwen2.*         neox (i, i+d/2)    bq, bk, bv, bout
///   "mistral3"             mistral3.*      normal (2i, 2i+1)  bq, bk, bv, bo, bgate, bup, bdown
///
/// Every architecture takes YaRN RoPE scaling (rope.scaling.type "yarn"),
/// derived as llama.cpp derives it; Mistral 3 also reads its YaRN beta and
/// log-multiplier keys, and its attention temperature
/// (attention.temperature_scale): temp(pos) = 1 + scale * ln(floor(pos /
/// n_ctx_orig) + 1), else 1.
///
/// A run processes up to max_batch tokens (prefill); only the last token's
/// logits are computed, as llama.cpp does for a prompt.
///
/// The weights stay in the mapped file: the graph holds views into it, so the
/// gguf_file (and the image under it) must outlive the model and its sessions.
///
/// Model
/// -----
///   llama_config — hyperparameters read from the llama.* metadata keys
///   llama_model
///     config   : llama_config
///     graph     : compute_graph — no inputs; one output, the logits
///     logits    : tensor_id     — n_vocab floats, for the run's last token
///     max_ctx   : size_t        — positions each KV cache holds
///     max_batch : size_t        — tokens a run may process
///
/// Contract
/// --------
///   load_llama(file, max_ctx, max_batch)
///     Pre:  none (the file is untrusted; max_ctx, max_batch >= 1 are checked).
///     Post: on success, a graph satisfying I1-I7 whose every weight view has
///           the shape the config implies; on failure, what was wrong and,
///           for a key or tensor, its name. Never throws for bad input.
///
/// Not supported (rejected, never silently mis-computed): biases other than
/// those above, mixture-of-experts layers, partial rotary dimensions, RoPE
/// scaling other than YaRN and frequency factors (Llama, Mistral 3), and
/// custom attention scales.

#include <branes/synapse/gguf/reader.hpp>
#include <branes/synapse/gguf/tensor_view.hpp>
#include <branes/synapse/graph/graph.hpp>
#include <branes/synapse/result.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace branes::synapse::models {

/// The Llama-family architectures load_llama builds.
enum class architecture : std::uint8_t { llama, qwen2, mistral3 };

constexpr std::string_view to_string(architecture a) noexcept {
    switch (a) {
    case architecture::qwen2:
        return "qwen2";
    case architecture::mistral3:
        return "mistral3";
    case architecture::llama:
        break;
    }
    return "llama";
}

struct llama_config {
    architecture arch = architecture::llama;
    kernels::rope_mode rope = kernels::rope_mode::normal;
    std::size_t n_layer = 0;
    std::size_t n_embd = 0;
    std::size_t n_ff = 0;
    std::size_t n_head = 0;
    std::size_t n_head_kv = 0;
    std::size_t head_dim = 0;
    std::size_t n_vocab = 0;
    std::size_t n_ctx_train = 0;
    float rms_eps = 0.0f;
    float rope_base = 0.0f;
    std::size_t n_ctx_orig = 0;          ///< context YaRN and the temperature refer to (default n_ctx_train)
    kernels::rope_scaling rope_scaling;  ///< YaRN, or none
    float attn_temp_scale = 0.0f;        ///< Q temperature (Mistral 3); 0 = none
};

enum class model_errc : std::uint32_t {
    wrong_architecture = 1,  ///< general.architecture is not "llama", "qwen2" or "mistral3"
    missing_key,             ///< a required metadata key is absent
    bad_value,               ///< a key has the wrong type or an invalid value
    unsupported,             ///< a feature this loader does not implement
    missing_tensor,          ///< a required tensor is absent
    bad_tensor,              ///< a tensor's shape does not match the config
    unsupported_type,        ///< a tensor in a format synapse cannot compute with
    graph,                   ///< the graph builder rejected the model (see graph_error)
};

constexpr std::string_view to_string(model_errc code) noexcept {
    switch (code) {
    case model_errc::wrong_architecture:
        return "unsupported architecture";
    case model_errc::missing_key:
        return "missing metadata key";
    case model_errc::bad_value:
        return "invalid metadata value";
    case model_errc::unsupported:
        return "unsupported model feature";
    case model_errc::missing_tensor:
        return "missing tensor";
    case model_errc::bad_tensor:
        return "tensor shape does not match the model";
    case model_errc::unsupported_type:
        return "unsupported tensor type";
    case model_errc::graph:
        return "graph construction failed";
    }
    return "unknown error";
}

struct model_error {
    model_errc code;
    std::string name;  ///< the key or tensor concerned, if any
    graph::graph_error graph{};
};

struct llama_model {
    llama_config config;
    graph::compute_graph graph;
    graph::tensor_id logits = graph::invalid_tensor;
    std::size_t max_ctx = 0;
    std::size_t max_batch = 1;
};

namespace detail {

inline model_error fail(model_errc code, std::string_view name = {}) {
    return model_error{code, std::string(name), {}};
}

/// An unsigned count stored as any GGUF integer type (writers differ).
inline result<std::size_t, model_error> get_count(const gguf::gguf_file& f, std::string_view key) {
    const gguf::metadata_value* v = f.find(key);
    if (v == nullptr) {
        return fail(model_errc::missing_key, key);
    }
    const std::optional<std::uint64_t> n = std::visit(
        [](const auto& x) -> std::optional<std::uint64_t> {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
                if (x < 0) {
                    return std::nullopt;
                }
                return static_cast<std::uint64_t>(x);
            } else {
                return std::nullopt;
            }
        },
        *v);
    if (!n || *n > std::numeric_limits<std::uint32_t>::max()) {
        return fail(model_errc::bad_value, key);
    }
    return static_cast<std::size_t>(*n);
}

inline result<std::size_t, model_error>
get_count(const gguf::gguf_file& f, std::string_view key, std::size_t fallback) {
    if (f.find(key) == nullptr) {
        return fallback;
    }
    return get_count(f, key);
}

/// A finite, positive float stored as F32 (or F64).
inline result<float, model_error>
get_positive(const gguf::gguf_file& f, std::string_view key, std::optional<float> fallback) {
    const gguf::metadata_value* v = f.find(key);
    if (v == nullptr) {
        if (fallback) {
            return *fallback;
        }
        return fail(model_errc::missing_key, key);
    }
    float x = 0.0f;
    if (const auto* p = std::get_if<float>(v)) {
        x = *p;
    } else if (const auto* d = std::get_if<double>(v)) {
        x = static_cast<float>(*d);
    } else {
        return fail(model_errc::bad_value, key);
    }
    if (!std::isfinite(x) || x <= 0.0f) {
        return fail(model_errc::bad_value, key);
    }
    return x;
}

/// A finite float stored as F32 (or F64), or fallback if absent.
inline result<float, model_error> get_finite(const gguf::gguf_file& f, std::string_view key, float fallback) {
    const gguf::metadata_value* v = f.find(key);
    if (v == nullptr) {
        return fallback;
    }
    float x = 0.0f;
    if (const auto* p = std::get_if<float>(v)) {
        x = *p;
    } else if (const auto* d = std::get_if<double>(v)) {
        x = static_cast<float>(*d);
    } else {
        return fail(model_errc::bad_value, key);
    }
    if (!std::isfinite(x)) {
        return fail(model_errc::bad_value, key);
    }
    return x;
}

/// llama.cpp's get_mscale: YaRN's magnitude correction for a scaling factor.
inline float yarn_mscale(float factor, float mscale) noexcept {
    return factor <= 1.0f ? 1.0f : 0.1f * mscale * std::log(factor) + 1.0f;
}

/// The RoPE context scaling of c (rope.scaling.*), derived as llama.cpp's
/// load_hparams and llama_context do. Pre: c's arch, head_dim, rope_base and
/// n_ctx_orig are set. Type "none" or an identity factor is no scaling;
/// "yarn" is YaRN; anything else (linear, longrope, ...) is unsupported.
template <typename Key>
std::optional<model_error> read_rope_scaling(const gguf::gguf_file& f, llama_config& c, const Key& key) {
    if (f.find(key("rope.scale_linear")) != nullptr) {
        return fail(model_errc::unsupported, key("rope.scale_linear"));
    }
    std::string_view type = "linear";  // llama.cpp's default
    if (f.find(key("rope.scaling.type")) != nullptr) {
        const auto t = f.get<std::string_view>(key("rope.scaling.type"));
        if (!t) {
            return fail(model_errc::bad_value, key("rope.scaling.type"));
        }
        type = *t;
    }
    // 0 (or absent) means unset: a scale of 1.
    auto factor = get_finite(f, key("rope.scaling.factor"), 0.0f);
    if (!factor || *factor < 0.0f) {
        return fail(model_errc::bad_value, key("rope.scaling.factor"));
    }
    const float freq_scale = *factor == 0.0f ? 1.0f : 1.0f / *factor;
    auto attn = get_finite(f, key("rope.scaling.attn_factor"), 1.0f);
    if (!attn || *attn <= 0.0f) {
        return fail(model_errc::bad_value, key("rope.scaling.attn_factor"));
    }
    kernels::rope_scaling s;
    if (type == "none" || (type == "linear" && freq_scale == 1.0f)) {
        s.attn_factor = *attn;
    } else if (type == "yarn") {
        if (freq_scale > 1.0f) {
            return fail(model_errc::unsupported, key("rope.scaling.factor"));  // a factor below 1
        }
        // Beta and the log multiplier are read for Mistral 3 only, as llama.cpp does.
        const bool mistral3 = c.arch == architecture::mistral3;
        const auto read = [&](std::string_view k, float fallback) {
            return mistral3 ? get_finite(f, key(k), fallback) : result<float, model_error>(fallback);
        };
        auto beta_fast = read("rope.scaling.yarn_beta_fast", 32.0f);
        auto beta_slow = read("rope.scaling.yarn_beta_slow", 1.0f);
        auto log_mul = read("rope.scaling.yarn_log_multiplier", 0.0f);
        for (const auto* r : {&beta_fast, &beta_slow, &log_mul}) {
            if (!*r) {
                return r->error();
            }
        }
        if (*beta_fast <= 0.0f || *beta_slow <= 0.0f) {
            return fail(model_errc::bad_value, key("rope.scaling.yarn_beta_fast"));
        }
        s.freq_scale = freq_scale;
        s.ext_factor = 1.0f;
        const float yarn_factor = 1.0f / freq_scale;
        float a = *log_mul != 0.0f ? yarn_mscale(yarn_factor, 1.0f) / yarn_mscale(yarn_factor, *log_mul)
                                   : yarn_mscale(yarn_factor, 1.0f);
        // Cancels the magnitude ggml's rope_yarn multiplies in (llama.cpp #17945).
        a *= 1.0f / (1.0f + 0.1f * std::log(yarn_factor));
        a *= *attn;
        s.attn_factor = a;
        const auto dims = kernels::rope_yarn_corr_dims(c.head_dim, c.n_ctx_orig, c.rope_base, *beta_fast, *beta_slow);
        s.corr_low = dims[0];
        s.corr_high = dims[1];
    } else {
        return fail(model_errc::unsupported,
                    *factor != 0.0f && type == "linear" ? key("rope.scaling.factor") : key("rope.scaling.type"));
    }
    if (!std::isfinite(s.attn_factor) || s.attn_factor <= 0.0f) {
        return fail(model_errc::bad_value, key("rope.scaling.attn_factor"));
    }
    c.rope_scaling = s;
    return std::nullopt;
}

/// Tensor `name` as a rows x cols matrix in a computable format.
inline result<graph::matrix_ref, model_error>
bind(const gguf::gguf_file& f, const std::string& name, std::size_t rows, std::size_t cols) {
    const gguf::tensor_info* t = f.find_tensor(name);
    if (t == nullptr) {
        return fail(model_errc::missing_tensor, name);
    }
    if (t->ne[0] != cols || t->ne[1] * t->ne[2] * t->ne[3] != rows) {
        return fail(model_errc::bad_tensor, name);
    }
    std::optional<graph::matrix_ref> m;
    if (!gguf::visit_matrix(f, *t, [&](const auto& view) { m = view; })) {
        return fail(model_errc::unsupported_type, name);
    }
    return *m;
}

inline std::string layer_tensor(std::size_t layer, std::string_view suffix) {
    return "blk." + std::to_string(layer) + "." + std::string(suffix);
}

}  // namespace detail

/// The hyperparameters of a Llama-family file, read from its "<arch>." keys.
inline result<llama_config, model_error> read_llama_config(const gguf::gguf_file& f) {
    using detail::fail;
    llama_config c;
    const auto arch = f.get<std::string_view>("general.architecture");
    if (arch == std::string_view{"llama"}) {
        c.arch = architecture::llama;
        c.rope = kernels::rope_mode::normal;
    } else if (arch == std::string_view{"qwen2"}) {
        c.arch = architecture::qwen2;
        c.rope = kernels::rope_mode::neox;
    } else if (arch == std::string_view{"mistral3"}) {
        c.arch = architecture::mistral3;
        c.rope = kernels::rope_mode::normal;
    } else {
        return fail(model_errc::wrong_architecture, "general.architecture");
    }
    const std::string prefix(to_string(c.arch));
    const auto key = [&](std::string_view suffix) { return prefix + "." + std::string(suffix); };
    const auto count = [&](const std::string& name, std::size_t& out) -> std::optional<model_error> {
        auto r = detail::get_count(f, name);
        if (!r) {
            return r.error();
        }
        out = *r;
        return std::nullopt;
    };
    for (auto [suffix, out] : {std::pair{"block_count", &c.n_layer},
                               std::pair{"embedding_length", &c.n_embd},
                               std::pair{"feed_forward_length", &c.n_ff},
                               std::pair{"attention.head_count", &c.n_head},
                               std::pair{"context_length", &c.n_ctx_train}}) {
        if (auto e = count(key(suffix), *out)) {
            return std::move(*e);
        }
        if (*out == 0) {
            return fail(model_errc::bad_value, key(suffix));
        }
    }
    auto kv = detail::get_count(f, key("attention.head_count_kv"), c.n_head);
    if (!kv) {
        return kv.error();
    }
    c.n_head_kv = *kv;
    if (c.n_head_kv == 0 || c.n_head % c.n_head_kv != 0) {
        return fail(model_errc::bad_value, key("attention.head_count_kv"));
    }
    auto hd = detail::get_count(f, key("attention.key_length"), c.n_embd / c.n_head);
    if (!hd) {
        return hd.error();
    }
    c.head_dim = *hd;
    if (c.head_dim == 0) {
        return fail(model_errc::bad_value, key("attention.key_length"));
    }
    auto vl = detail::get_count(f, key("attention.value_length"), c.head_dim);
    if (!vl) {
        return vl.error();
    }
    if (*vl != c.head_dim) {
        return fail(model_errc::unsupported, key("attention.value_length"));
    }
    auto rot = detail::get_count(f, key("rope.dimension_count"), c.head_dim);
    if (!rot) {
        return rot.error();
    }
    if (*rot != c.head_dim) {
        return fail(model_errc::unsupported, key("rope.dimension_count"));
    }
    auto eps = detail::get_positive(f, key("attention.layer_norm_rms_epsilon"), std::nullopt);
    if (!eps) {
        return eps.error();
    }
    c.rms_eps = *eps;
    auto base = detail::get_positive(f, key("rope.freq_base"), 10000.0f);
    if (!base) {
        return base.error();
    }
    c.rope_base = *base;

    auto orig = detail::get_count(f, key("rope.scaling.original_context_length"), c.n_ctx_train);
    if (!orig) {
        return orig.error();
    }
    c.n_ctx_orig = *orig;
    if (c.n_ctx_orig == 0) {
        return fail(model_errc::bad_value, key("rope.scaling.original_context_length"));
    }
    if (auto e = detail::read_rope_scaling(f, c, key)) {
        return std::move(*e);
    }
    if (c.arch == architecture::mistral3) {
        auto temp = detail::get_finite(f, key("attention.temperature_scale"), 0.0f);
        if (!temp) {
            return temp.error();
        }
        c.attn_temp_scale = *temp;
    }
    if (f.find(key("attention.scale")) != nullptr) {
        return fail(model_errc::unsupported, key("attention.scale"));
    }
    if (auto experts = detail::get_count(f, key("expert_count"), 0); !experts || *experts != 0) {
        return fail(model_errc::unsupported, key("expert_count"));
    }

    const gguf::tensor_info* embd = f.find_tensor("token_embd.weight");
    if (embd == nullptr) {
        return fail(model_errc::missing_tensor, "token_embd.weight");
    }
    if (embd->ne[0] != c.n_embd || embd->ne[2] != 1 || embd->ne[3] != 1 || embd->ne[1] == 0) {
        return fail(model_errc::bad_tensor, "token_embd.weight");
    }
    c.n_vocab = static_cast<std::size_t>(embd->ne[1]);
    return c;
}

/// The decode graph of a Llama-family file, with KV caches of max_ctx
/// positions, for runs of up to max_batch tokens.  See the file comment for
/// the contract.
inline result<llama_model, model_error>
load_llama(const gguf::gguf_file& f, std::size_t max_ctx, std::size_t max_batch = 1) {
    using detail::fail;
    auto cfg = read_llama_config(f);
    if (!cfg) {
        return cfg.error();
    }
    const llama_config& c = *cfg;
    if (max_ctx == 0) {
        return fail(model_errc::bad_value, "max_ctx");
    }
    if (max_batch == 0) {
        return fail(model_errc::bad_value, "max_batch");
    }
    const std::size_t q_rows = c.n_head * c.head_dim;
    const std::size_t kv_rows = c.n_head_kv * c.head_dim;

    std::vector<float> freq_factors;
    if (const gguf::tensor_info* rf = f.find_tensor("rope_freqs.weight")) {
        if (c.arch == architecture::qwen2) {
            return fail(model_errc::unsupported, "rope_freqs.weight");  // llama.cpp does not read it for Qwen2
        }
        if (rf->type != quant::tensor_type::f32) {
            return fail(model_errc::unsupported_type, "rope_freqs.weight");
        }
        if (rf->n_elements() != c.head_dim / 2) {
            return fail(model_errc::bad_tensor, "rope_freqs.weight");
        }
        freq_factors.resize(c.head_dim / 2);
        std::memcpy(freq_factors.data(), f.tensor_data(*rf).data(), freq_factors.size() * sizeof(float));
    }

    graph::graph_builder b(max_batch);
    // Every weight is bound before it is used; the first failure is returned.
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

    const graph::matrix_ref embd = weight("token_embd.weight", c.n_vocab, c.n_embd);
    if (error) {
        return std::move(*error);
    }
    // The biases an architecture may have, as llama.cpp loads them (all
    // optional; any other bias is rejected).
    const auto has_bias = [&](std::string_view suffix) {
        switch (c.arch) {
        case architecture::llama:
            return false;
        case architecture::qwen2:
            return suffix == "attn_q.bias" || suffix == "attn_k.bias" || suffix == "attn_v.bias" ||
                   suffix == "output.bias";
        case architecture::mistral3:
            return suffix == "attn_q.bias" || suffix == "attn_k.bias" || suffix == "attn_v.bias" ||
                   suffix == "attn_output.bias" || suffix == "ffn_gate.bias" || suffix == "ffn_up.bias" ||
                   suffix == "ffn_down.bias";
        }
        return false;
    };
    // bound when present: 1 x rows, or invalid
    const auto bias = [&](const std::string& name, std::size_t rows) -> std::optional<graph::matrix_ref> {
        if (f.find_tensor(name) == nullptr) {
            return std::nullopt;
        }
        return weight(name, 1, rows);
    };
    // w h, plus the bias if there is one
    const auto linear = [&](const graph::matrix_ref& w,
                            const std::optional<graph::matrix_ref>& bv,
                            graph::tensor_id h) -> graph::tensor_id {
        const graph::tensor_id y = b.matmul(w, h);
        return bv ? b.add_bias(y, *bv) : y;
    };

    graph::tensor_id x = b.embed(embd);
    const float scale = 1.0f / std::sqrt(static_cast<float>(c.head_dim));
    for (std::size_t l = 0; l < c.n_layer; ++l) {
        const auto w = [&](std::string_view suffix, std::size_t rows, std::size_t cols) {
            return weight(detail::layer_tensor(l, suffix), rows, cols);
        };
        const graph::matrix_ref attn_norm = w("attn_norm.weight", 1, c.n_embd);
        const graph::matrix_ref wq = w("attn_q.weight", q_rows, c.n_embd);
        const graph::matrix_ref wk = w("attn_k.weight", kv_rows, c.n_embd);
        const graph::matrix_ref wv = w("attn_v.weight", kv_rows, c.n_embd);
        const graph::matrix_ref wo = w("attn_output.weight", c.n_embd, q_rows);
        const graph::matrix_ref ffn_norm = w("ffn_norm.weight", 1, c.n_embd);
        const graph::matrix_ref gate = w("ffn_gate.weight", c.n_ff, c.n_embd);
        const graph::matrix_ref up = w("ffn_up.weight", c.n_ff, c.n_embd);
        const graph::matrix_ref down = w("ffn_down.weight", c.n_embd, c.n_ff);
        const auto b_opt = [&](std::string_view suffix, std::size_t rows) -> std::optional<graph::matrix_ref> {
            return has_bias(suffix) ? bias(detail::layer_tensor(l, suffix), rows) : std::nullopt;
        };
        const auto bq = b_opt("attn_q.bias", q_rows);
        const auto bk = b_opt("attn_k.bias", kv_rows);
        const auto bv = b_opt("attn_v.bias", kv_rows);
        const auto bo = b_opt("attn_output.bias", c.n_embd);
        const auto b_gate = b_opt("ffn_gate.bias", c.n_ff);
        const auto b_up = b_opt("ffn_up.bias", c.n_ff);
        const auto b_down = b_opt("ffn_down.bias", c.n_embd);
        if (error) {
            return std::move(*error);
        }
        // Rejected rather than ignored: they change the result. Checked per
        // layer, after the layer's weights bound, so a block_count larger than
        // the file's tensors fails at the first missing layer, not after
        // scanning for billions of names.
        for (std::string_view extra : {"attn_q.bias",
                                       "attn_k.bias",
                                       "attn_v.bias",
                                       "attn_output.bias",
                                       "ffn_gate.bias",
                                       "ffn_up.bias",
                                       "ffn_down.bias",
                                       "ffn_gate_inp.weight"}) {
            if (has_bias(extra)) {
                continue;  // bound above
            }
            if (const std::string name = detail::layer_tensor(l, extra); f.find_tensor(name) != nullptr) {
                return fail(model_errc::unsupported, name);
            }
        }

        const graph::tensor_id h = b.rms_norm(x, attn_norm, c.rms_eps);
        graph::tensor_id q = b.rope(linear(wq, bq, h), c.head_dim, c.rope_base, freq_factors, c.rope, c.rope_scaling);
        const graph::tensor_id k =
            b.rope(linear(wk, bk, h), c.head_dim, c.rope_base, freq_factors, c.rope, c.rope_scaling);
        const graph::tensor_id v = linear(wv, bv, h);
        if (c.attn_temp_scale != 0.0f) {  // after RoPE, as llm_build_mistral3
            q = b.attn_temperature(q, c.attn_temp_scale, c.n_ctx_orig);
        }
        const graph::state_id cache = b.kv_cache(max_ctx, c.n_head_kv, c.head_dim);
        const graph::tensor_id attn = b.attention(q, k, v, cache, c.n_head, scale);
        x = b.add(linear(wo, bo, attn), x);

        const graph::tensor_id h2 = b.rms_norm(x, ffn_norm, c.rms_eps);
        const graph::tensor_id act = b.mul(b.silu(linear(gate, b_gate, h2)), linear(up, b_up, h2));
        x = b.add(linear(down, b_down, act), x);
    }
    const graph::matrix_ref out_norm = weight("output_norm.weight", 1, c.n_embd);
    const graph::matrix_ref out =
        f.find_tensor("output.weight") != nullptr ? weight("output.weight", c.n_vocab, c.n_embd) : embd;
    if (!has_bias("output.bias") && f.find_tensor("output.bias") != nullptr) {
        return fail(model_errc::unsupported, "output.bias");
    }
    const std::optional<graph::matrix_ref> out_bias =
        has_bias("output.bias") ? bias("output.bias", c.n_vocab) : std::nullopt;
    if (error) {
        return std::move(*error);
    }
    const graph::tensor_id logits = linear(out, out_bias, b.rms_norm(b.last_row(x), out_norm, c.rms_eps));
    b.output(logits);

    auto g = std::move(b).finalize();
    if (!g) {
        return model_error{model_errc::graph, {}, g.error()};
    }
    return llama_model{c, std::move(*g), logits, max_ctx, max_batch};
}

}  // namespace branes::synapse::models

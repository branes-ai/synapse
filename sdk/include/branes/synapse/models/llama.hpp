#pragma once
/// @file llama.hpp
/// @brief Build the decode graph of a Llama-architecture GGUF model.
///
/// The graph mirrors llama.cpp's llm_build_llama for one token per run:
///
///   x = embed(token_embd)
///   per layer:
///     h = rms_norm(x, attn_norm)
///     q = rope(Wq h), k = rope(Wk h), v = Wv h
///     x = x + Wo attention(q, k, v, kv_cache)
///     h = rms_norm(x, ffn_norm)
///     x = x + Wdown (silu(Wgate h) * Wup h)
///   logits = Wout rms_norm(last_row(x), output_norm)   (Wout = token_embd if tied)
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
/// Not supported (rejected, never silently mis-computed): attention or FFN
/// biases, mixture-of-experts layers, partial rotary dimensions, RoPE
/// scaling other than llama3-style frequency factors, and custom attention
/// scales.

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

struct llama_config {
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
};

enum class model_errc : std::uint32_t {
    wrong_architecture = 1,  ///< general.architecture is not "llama"
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
        return "not a llama model";
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

/// The hyperparameters of a Llama-architecture file.
inline result<llama_config, model_error> read_llama_config(const gguf::gguf_file& f) {
    using detail::fail;
    if (f.get<std::string_view>("general.architecture") != std::string_view{"llama"}) {
        return fail(model_errc::wrong_architecture, "general.architecture");
    }
    llama_config c;
    const auto count = [&](std::string_view key, std::size_t& out) -> std::optional<model_error> {
        auto r = detail::get_count(f, key);
        if (!r) {
            return r.error();
        }
        out = *r;
        return std::nullopt;
    };
    for (auto [key, out] : {std::pair{"llama.block_count", &c.n_layer},
                            std::pair{"llama.embedding_length", &c.n_embd},
                            std::pair{"llama.feed_forward_length", &c.n_ff},
                            std::pair{"llama.attention.head_count", &c.n_head},
                            std::pair{"llama.context_length", &c.n_ctx_train}}) {
        if (auto e = count(key, *out)) {
            return std::move(*e);
        }
        if (*out == 0) {
            return fail(model_errc::bad_value, key);
        }
    }
    auto kv = detail::get_count(f, "llama.attention.head_count_kv", c.n_head);
    if (!kv) {
        return kv.error();
    }
    c.n_head_kv = *kv;
    if (c.n_head_kv == 0 || c.n_head % c.n_head_kv != 0) {
        return fail(model_errc::bad_value, "llama.attention.head_count_kv");
    }
    auto hd = detail::get_count(f, "llama.attention.key_length", c.n_embd / c.n_head);
    if (!hd) {
        return hd.error();
    }
    c.head_dim = *hd;
    if (c.head_dim == 0) {
        return fail(model_errc::bad_value, "llama.attention.key_length");
    }
    auto vl = detail::get_count(f, "llama.attention.value_length", c.head_dim);
    if (!vl) {
        return vl.error();
    }
    if (*vl != c.head_dim) {
        return fail(model_errc::unsupported, "llama.attention.value_length");
    }
    auto rot = detail::get_count(f, "llama.rope.dimension_count", c.head_dim);
    if (!rot) {
        return rot.error();
    }
    if (*rot != c.head_dim) {
        return fail(model_errc::unsupported, "llama.rope.dimension_count");
    }
    auto eps = detail::get_positive(f, "llama.attention.layer_norm_rms_epsilon", std::nullopt);
    if (!eps) {
        return eps.error();
    }
    c.rms_eps = *eps;
    auto base = detail::get_positive(f, "llama.rope.freq_base", 10000.0f);
    if (!base) {
        return base.error();
    }
    c.rope_base = *base;

    if (f.find("llama.rope.scaling.type") != nullptr &&
        f.get<std::string_view>("llama.rope.scaling.type") != std::string_view{"none"}) {
        return fail(model_errc::unsupported, "llama.rope.scaling.type");
    }
    if (f.find("llama.rope.scale_linear") != nullptr) {
        return fail(model_errc::unsupported, "llama.rope.scale_linear");
    }
    // llama.cpp scales positions by 1 / factor (linear scaling when the type
    // key is absent); 0 means unset and 1 is the identity, so only those load.
    if (const gguf::metadata_value* v = f.find("llama.rope.scaling.factor")) {
        const auto* factor = std::get_if<float>(v);
        if (factor == nullptr || (*factor != 0.0f && *factor != 1.0f)) {
            return fail(model_errc::unsupported, "llama.rope.scaling.factor");
        }
    }
    if (f.find("llama.attention.scale") != nullptr) {
        return fail(model_errc::unsupported, "llama.attention.scale");
    }
    if (auto experts = detail::get_count(f, "llama.expert_count", 0); !experts || *experts != 0) {
        return fail(model_errc::unsupported, "llama.expert_count");
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

/// The decode graph of a Llama-architecture file, with KV caches of max_ctx
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
            if (const std::string name = detail::layer_tensor(l, extra); f.find_tensor(name) != nullptr) {
                return fail(model_errc::unsupported, name);
            }
        }

        const graph::tensor_id h = b.rms_norm(x, attn_norm, c.rms_eps);
        const graph::tensor_id q = b.rope(b.matmul(wq, h), c.head_dim, c.rope_base, freq_factors);
        const graph::tensor_id k = b.rope(b.matmul(wk, h), c.head_dim, c.rope_base, freq_factors);
        const graph::tensor_id v = b.matmul(wv, h);
        const graph::state_id cache = b.kv_cache(max_ctx, c.n_head_kv, c.head_dim);
        const graph::tensor_id attn = b.attention(q, k, v, cache, c.n_head, scale);
        x = b.add(b.matmul(wo, attn), x);

        const graph::tensor_id h2 = b.rms_norm(x, ffn_norm, c.rms_eps);
        const graph::tensor_id act = b.mul(b.silu(b.matmul(gate, h2)), b.matmul(up, h2));
        x = b.add(b.matmul(down, act), x);
    }
    const graph::matrix_ref out_norm = weight("output_norm.weight", 1, c.n_embd);
    const graph::matrix_ref out =
        f.find_tensor("output.weight") != nullptr ? weight("output.weight", c.n_vocab, c.n_embd) : embd;
    if (error) {
        return std::move(*error);
    }
    const graph::tensor_id logits = b.matmul(out, b.rms_norm(b.last_row(x), out_norm, c.rms_eps));
    b.output(logits);

    auto g = std::move(b).finalize();
    if (!g) {
        return model_error{model_errc::graph, {}, g.error()};
    }
    return llama_model{c, std::move(*g), logits, max_ctx, max_batch};
}

}  // namespace branes::synapse::models

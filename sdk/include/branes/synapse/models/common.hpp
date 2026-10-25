#pragma once
/// @file common.hpp
/// @brief What every model loader shares: its error type and the readers of
/// metadata values and weight tensors, each checked as untrusted input.

#include <branes/synapse/gguf/reader.hpp>
#include <branes/synapse/gguf/tensor_view.hpp>
#include <branes/synapse/graph/graph.hpp>
#include <branes/synapse/result.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

namespace branes::synapse::models {

enum class model_errc : std::uint32_t {
    wrong_architecture = 1,  ///< not an architecture (or projector type) the loader builds
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

}  // namespace detail

}  // namespace branes::synapse::models

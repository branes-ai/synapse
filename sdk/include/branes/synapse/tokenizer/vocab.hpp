#pragma once
// Ported from llama.cpp (MIT License, The ggml authors); see THIRD_PARTY_NOTICES.md.
/// @file vocab.hpp
/// @brief What the tokenizers share: token types, errors, options, and
/// special-token splitting, as llama.cpp's llama-vocab.cpp does them.

#include <branes/synapse/gguf/reader.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace branes::synapse::tokenizer {

/// GGUF token types (tokenizer.ggml.token_type).
enum class token_type : std::int32_t {
    undefined = 0,
    normal = 1,
    unknown = 2,
    control = 3,
    user_defined = 4,
    unused = 5,
    byte = 6,
};

inline constexpr std::uint32_t no_token = std::numeric_limits<std::uint32_t>::max();

enum class tokenizer_errc : std::uint32_t {
    unsupported_model = 1,  ///< tokenizer.ggml.model is not one synapse implements
    unsupported_pre,        ///< the pre-tokenizer is not one synapse implements
    missing_key,            ///< a required tokenizer key is absent
    bad_value,              ///< a key has the wrong type or an invalid value
};

constexpr std::string_view to_string(tokenizer_errc code) noexcept {
    switch (code) {
    case tokenizer_errc::unsupported_model:
        return "unsupported tokenizer model";
    case tokenizer_errc::unsupported_pre:
        return "unsupported pre-tokenizer";
    case tokenizer_errc::missing_key:
        return "missing tokenizer key";
    case tokenizer_errc::bad_value:
        return "invalid tokenizer value";
    }
    return "unknown error";
}

struct tokenizer_error {
    tokenizer_errc code;
    std::string name;  ///< the key concerned, if any
};

struct encode_options {
    bool add_special = true;     ///< add BOS / EOS where the vocabulary asks for them
    bool parse_special = false;  ///< recognize control tokens' text (e.g. "<|eot_id|>")
};

/// Ids of the tokens split out of text before tokenization: control,
/// unknown and user-defined tokens with non-empty text, longest text first.
inline std::vector<std::uint32_t> special_tokens(std::span<const std::string> tokens,
                                                 std::span<const token_type> types) {
    std::vector<std::uint32_t> ids;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        const token_type ty = types[i];
        if ((ty == token_type::control || ty == token_type::unknown || ty == token_type::user_defined) &&
            !tokens[i].empty()) {
            ids.push_back(static_cast<std::uint32_t>(i));
        }
    }
    std::stable_sort(
        ids.begin(), ids.end(), [&](std::uint32_t a, std::uint32_t b) { return tokens[a].size() > tokens[b].size(); });
    return ids;
}

/// A piece of the input: raw text, or one special token.
struct fragment {
    std::size_t offset, length;
    std::uint32_t token;  ///< no_token for raw text
};

/// Split text at every occurrence of a special token's text, longest token
/// first (llama.cpp's tokenizer_st_partition). Control and unknown tokens
/// are recognized only with parse_special; user-defined ones always are.
inline std::vector<fragment> split_specials(std::string_view text,
                                            std::span<const std::string> tokens,
                                            std::span<const token_type> types,
                                            std::span<const std::uint32_t> specials,
                                            bool parse_special) {
    std::vector<fragment> frags;
    if (text.empty()) {
        return frags;
    }
    frags.push_back({0, text.size(), no_token});
    for (std::uint32_t id : specials) {
        if (!parse_special && types[id] != token_type::user_defined) {
            continue;
        }
        const std::string_view s = tokens[id];
        std::vector<fragment> next;
        for (const fragment& fr : frags) {
            if (fr.token != no_token) {
                next.push_back(fr);
                continue;
            }
            std::size_t at = fr.offset;
            const std::size_t stop = fr.offset + fr.length;
            while (true) {
                const std::size_t m = text.substr(0, stop).find(s, at);
                if (m == std::string_view::npos) {
                    break;
                }
                if (m > at) {
                    next.push_back({at, m - at, no_token});
                }
                next.push_back({m, s.size(), id});
                at = m + s.size();
            }
            if (at < stop) {
                next.push_back({at, stop - at, no_token});
            }
        }
        frags = std::move(next);
    }
    return frags;
}

/// Read an optional boolean key into out; false if it is present but not a bool.
inline bool read_flag(const gguf::gguf_file& f, std::string_view key, bool& out) {
    if (f.find(key) == nullptr) {
        return true;
    }
    const auto v = f.get<bool>(key);
    if (!v) {
        return false;
    }
    out = *v;
    return true;
}

}  // namespace branes::synapse::tokenizer

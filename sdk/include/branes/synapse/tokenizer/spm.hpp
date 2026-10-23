#pragma once
// Ported from llama.cpp (MIT License, The ggml authors); see THIRD_PARTY_NOTICES.md.
/// @file spm.hpp
/// @brief SentencePiece (Llama 2 family) tokenization, as llama.cpp does it.
///
/// For GGUF vocabularies with tokenizer.ggml.model = "llama" (Llama 2,
/// TinyLlama, Mistral v1, ...). A port of llama.cpp's
/// llm_tokenizer_spm_session and the SPM branch of llama_vocab::tokenize.
///
/// encode(text):
///   1. Split out special tokens (vocab.hpp's split_specials, shared with
///      the BPE tokenizer).
///   2. Each raw fragment that is the first or follows a special token gets
///      a leading space when add_space_prefix (the default); then every
///      space becomes U+2581 ("▁").
///   3. Start from one symbol per UTF-8 character (length from the lead
///      byte, as llama.cpp: an invalid sequence stays one symbol). Repeatedly
///      merge the adjacent pair whose concatenation is a vocabulary token of
///      the highest score (ties: leftmost). There is no merge list: the
///      vocabulary and its scores drive it.
///   4. Each final symbol is its token, or (a character not in the
///      vocabulary) its bytes as byte tokens <0xXX>.
///   5. BOS first and EOS last when the vocabulary asks for them (BOS by
///      default) and add_special is set.
///
/// Model
/// -----
///   spm_tokenizer
///     tokens_ : vector<string>, scores_ : vector<float>, types_ : vector<token_type>
///     ids_    : map text -> id (a text listed twice maps to its last id)
///     byte_   : the token of each byte (<0xXX>, or the one-byte text)
///     specials_, bos_, eos_, add_bos_, add_eos_, add_space_prefix_
///
/// Invariants (established by from_gguf)
/// ----------
///   T1: every id in ids_, byte_, specials_, bos_, eos_ is < tokens_.size()
///   T2: decode(encode(s)) == (add_space_prefix ? " " : "") + s for every
///       non-empty valid UTF-8 s with no special-token text and no U+2581
///       (every byte has a fallback token that decodes back to that byte,
///       checked at load, so nothing is dropped or misread).
///       As in llama.cpp it does not extend further: an empty s encodes to
///       nothing; an input U+2581 reads back as a space; and in invalid
///       UTF-8 a stray lead byte can absorb a following escaped space into a
///       byte-fallback symbol, whose byte tokens decode to the raw U+2581.
///
/// The tokenizer allocates; it belongs at the application boundary.

#include <branes/synapse/gguf/reader.hpp>
#include <branes/synapse/result.hpp>
#include <branes/synapse/tokenizer/unicode.hpp>
#include <branes/synapse/tokenizer/vocab.hpp>

#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <queue>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace branes::synapse::tokenizer {

class spm_tokenizer {
public:
    /// The tokenizer of a GGUF file.  Pre: none (the file is untrusted).
    /// Post: T1, T2, or the first problem found.
    static result<spm_tokenizer, tokenizer_error> from_gguf(const gguf::gguf_file& f) {
        const auto fail = [](tokenizer_errc code, std::string_view name) {
            return tokenizer_error{code, std::string(name)};
        };
        const auto model = f.get<std::string_view>("tokenizer.ggml.model");
        if (!model) {
            return fail(tokenizer_errc::missing_key, "tokenizer.ggml.model");
        }
        if (*model != "llama") {
            return fail(tokenizer_errc::unsupported_model, "tokenizer.ggml.model");
        }

        spm_tokenizer t;
        const gguf::array_value* tokens = f.get_array("tokenizer.ggml.tokens");
        if (tokens == nullptr) {
            return fail(tokenizer_errc::missing_key, "tokenizer.ggml.tokens");
        }
        if (tokens->element_type() != gguf::value_type::string || tokens->size() == 0 || tokens->size() >= no_token) {
            return fail(tokenizer_errc::bad_value, "tokenizer.ggml.tokens");
        }
        const auto texts = tokens->strings();
        t.tokens_.assign(texts.begin(), texts.end());
        for (std::size_t i = 0; i < t.tokens_.size(); ++i) {
            t.ids_[t.tokens_[i]] = static_cast<std::uint32_t>(i);
        }

        // Scores are optional (llama.cpp reads 0 without them).
        t.scores_.assign(t.tokens_.size(), 0.0f);
        if (const gguf::array_value* scores = f.get_array("tokenizer.ggml.scores")) {
            if (scores->element_type() != gguf::value_type::float32 || scores->size() != t.tokens_.size()) {
                return fail(tokenizer_errc::bad_value, "tokenizer.ggml.scores");
            }
            for (std::size_t i = 0; i < t.scores_.size(); ++i) {
                t.scores_[i] = *scores->at<float>(i);
            }
        }
        t.types_.assign(t.tokens_.size(), token_type::normal);
        if (const gguf::array_value* types = f.get_array("tokenizer.ggml.token_type")) {
            if (types->element_type() != gguf::value_type::int32 || types->size() != t.tokens_.size()) {
                return fail(tokenizer_errc::bad_value, "tokenizer.ggml.token_type");
            }
            for (std::size_t i = 0; i < t.types_.size(); ++i) {
                t.types_[i] = static_cast<token_type>(*types->at<std::int32_t>(i));
            }
        }

        // llama.cpp's SPM defaults: BOS 1, EOS 2; add BOS, not EOS; space prefix on.
        t.bos_ = 1;
        t.eos_ = 2;
        for (auto [key, out] :
             {std::pair{"tokenizer.ggml.bos_token_id", &t.bos_}, std::pair{"tokenizer.ggml.eos_token_id", &t.eos_}}) {
            if (f.find(key) != nullptr) {
                const auto id = f.get<std::uint32_t>(key);
                if (!id) {
                    return fail(tokenizer_errc::bad_value, key);
                }
                *out = *id;
            }
        }
        for (auto [key, out] : {std::pair{"tokenizer.ggml.add_bos_token", &t.add_bos_},
                                std::pair{"tokenizer.ggml.add_eos_token", &t.add_eos_},
                                std::pair{"tokenizer.ggml.add_space_prefix", &t.add_space_prefix_}}) {
            if (!read_flag(f, key, *out)) {
                return fail(tokenizer_errc::bad_value, key);
            }
        }
        if (t.bos_ >= t.tokens_.size()) {  // T1
            if (t.add_bos_) {
                return fail(tokenizer_errc::bad_value, "tokenizer.ggml.bos_token_id");
            }
            t.bos_ = no_token;
        }
        if (t.eos_ >= t.tokens_.size()) {  // T1
            if (t.add_eos_) {
                return fail(tokenizer_errc::bad_value, "tokenizer.ggml.eos_token_id");
            }
            t.eos_ = no_token;
        }

        // T2: every byte needs a token for the byte fallback: <0xXX>, else
        // the byte's own one-character text (llama.cpp's byte_to_token).
        for (std::size_t b = 0; b < 256; ++b) {
            std::array<char, 7> hex{};
            std::snprintf(hex.data(), hex.size(), "<0x%02X>", static_cast<unsigned>(b));
            const std::string byte(1, static_cast<char>(b));
            if (const auto id = t.id_of(hex.data())) {
                t.byte_[b] = *id;
            } else if (const auto raw = t.id_of(byte)) {
                t.byte_[b] = *raw;
            } else {
                return fail(tokenizer_errc::bad_value, "tokenizer.ggml.tokens");
            }
            // The fallback must also decode back to its byte: a <0xXX> token
            // not typed `byte` (e.g. a vocabulary without token types) would
            // decode as its literal text, as in llama.cpp. Reject rather than
            // round-trip wrongly.
            if (t.decode(t.byte_[b]) != byte) {
                return fail(tokenizer_errc::bad_value, "tokenizer.ggml.token_type");
            }
        }
        t.specials_ = special_tokens(t.tokens_, t.types_);
        return t;
    }

    std::size_t size() const noexcept {
        return tokens_.size();
    }
    std::uint32_t bos() const noexcept {
        return bos_;
    }
    std::uint32_t eos() const noexcept {
        return eos_;
    }
    bool adds_bos() const noexcept {
        return add_bos_;
    }
    bool adds_space_prefix() const noexcept {
        return add_space_prefix_;
    }
    token_type type(std::uint32_t id) const noexcept {
        return id < types_.size() ? types_[id] : token_type::undefined;
    }

    /// The token ids of `text`.
    std::vector<std::uint32_t> encode(std::string_view text, encode_options opt = {}) const {
        std::vector<std::uint32_t> out;
        if (opt.add_special && add_bos_) {
            out.push_back(bos_);
        }
        bool prev_special = true;  // the first fragment gets the space prefix too
        for (const fragment& fr : split_specials(text, tokens_, types_, specials_, opt.parse_special)) {
            if (fr.token != no_token) {
                out.push_back(fr.token);
                prev_special = true;
                continue;
            }
            std::string piece;
            if (add_space_prefix_ && prev_special) {
                piece = ' ';
            }
            piece += text.substr(fr.offset, fr.length);
            encode_piece(escape_spaces(piece), out);
            prev_special = false;
        }
        if (opt.add_special && add_eos_) {
            out.push_back(eos_);
        }
        return out;
    }

    /// The bytes token `id` stands for: normal tokens with U+2581 turned back
    /// into spaces, byte tokens as their byte, user-defined tokens as their
    /// text; control and unknown tokens as their text only when `special`;
    /// unused and undefined tokens as nothing (llama.cpp's token_to_piece).
    std::string decode(std::uint32_t id, bool special = false) const {
        std::string out;
        if (id >= tokens_.size()) {
            return out;
        }
        const std::string_view text = tokens_[id];
        switch (types_[id]) {
        case token_type::normal:
            out = unescape_spaces(text);
            break;
        case token_type::byte:
            // llama.cpp's token_to_byte: strtol of the two hex digits. Parsed
            // without exceptions (the vocabulary is untrusted); like strtol,
            // digits that do not parse give 0.
            if (text.size() == 6 && text.starts_with("<0x") && text.ends_with('>')) {
                unsigned value = 0;
                std::from_chars(text.data() + 3, text.data() + 5, value, 16);
                out.push_back(static_cast<char>(value));
            }
            break;
        case token_type::user_defined:
            out = text;
            break;
        case token_type::control:
        case token_type::unknown:
            if (special) {
                out = text;
            }
            break;
        default:
            break;
        }
        return out;
    }

    std::string decode(std::span<const std::uint32_t> ids, bool special = false) const {
        std::string out;
        for (std::uint32_t id : ids) {
            out += decode(id, special);
        }
        return out;
    }

private:
    spm_tokenizer() = default;

    static constexpr std::string_view space_marker = "\xe2\x96\x81";  // U+2581

    static std::string escape_spaces(std::string_view s) {
        std::string out;
        for (char ch : s) {
            if (ch == ' ') {
                out += space_marker;
            } else {
                out += ch;
            }
        }
        return out;
    }

    static std::string unescape_spaces(std::string_view s) {
        std::string out;
        for (std::size_t i = 0; i < s.size();) {
            if (s.substr(i, space_marker.size()) == space_marker) {
                out += ' ';
                i += space_marker.size();
            } else {
                out += s[i++];
            }
        }
        return out;
    }

    std::optional<std::uint32_t> id_of(const std::string& text) const {
        const auto it = ids_.find(text);
        return it == ids_.end() ? std::nullopt : std::optional<std::uint32_t>(it->second);
    }

    /// llama.cpp's llm_tokenizer_spm_session::tokenize for one fragment.
    void encode_piece(const std::string& text, std::vector<std::uint32_t>& out) const {
        struct symbol {
            std::size_t begin, length;
            int prev, next;
        };
        std::vector<symbol> sym;
        for (std::size_t i = 0; i < text.size();) {
            // Length from the lead byte's high nibble (llama.cpp's unicode_len_utf8).
            const auto lead = static_cast<unsigned char>(text[i]);
            const std::size_t n = lead < 0xC0 ? 1 : lead < 0xE0 ? 2 : lead < 0xF0 ? 3 : 4;
            const std::size_t len = std::min(n, text.size() - i);
            sym.push_back({i, len, static_cast<int>(sym.size()) - 1, -1});
            i += len;
        }
        for (std::size_t i = 0; i + 1 < sym.size(); ++i) {
            sym[i].next = static_cast<int>(i + 1);
        }
        struct bigram {
            int left, right;
            float score;
            std::size_t size;
        };
        // Highest score first; ties go to the leftmost pair.
        const auto lower = [](const bigram& a, const bigram& b) {
            return a.score < b.score || (a.score == b.score && a.left > b.left);
        };
        std::priority_queue<bigram, std::vector<bigram>, decltype(lower)> queue(lower);
        const auto try_add = [&](int l, int r) {
            if (l < 0 || r < 0) {
                return;
            }
            const symbol& a = sym[static_cast<std::size_t>(l)];
            const symbol& b = sym[static_cast<std::size_t>(r)];
            const std::string joined = text.substr(a.begin, a.length + b.length);
            if (const auto id = id_of(joined)) {
                queue.push({l, r, scores_[*id], joined.size()});
            }
        };
        for (std::size_t i = 1; i < sym.size(); ++i) {
            try_add(static_cast<int>(i - 1), static_cast<int>(i));
        }
        while (!queue.empty()) {
            const bigram bg = queue.top();
            queue.pop();
            symbol& l = sym[static_cast<std::size_t>(bg.left)];
            symbol& r = sym[static_cast<std::size_t>(bg.right)];
            if (l.length == 0 || r.length == 0 || l.length + r.length != bg.size) {
                continue;  // stale: one side has merged since
            }
            l.length += r.length;
            r.length = 0;
            l.next = r.next;
            if (r.next >= 0) {
                sym[static_cast<std::size_t>(r.next)].prev = bg.left;
            }
            try_add(l.prev, bg.left);
            try_add(bg.left, l.next);
        }
        // Each final symbol is a token (every merge produced one), or a single
        // character missing from the vocabulary: emit its bytes. (llama.cpp's
        // resegment also consults the merges it recorded, but only for texts
        // that are tokens, which the first lookup already found.)
        for (int i = sym.empty() ? -1 : 0; i != -1; i = sym[static_cast<std::size_t>(i)].next) {
            const symbol& s = sym[static_cast<std::size_t>(i)];
            const std::string piece = text.substr(s.begin, s.length);
            if (const auto id = id_of(piece)) {
                out.push_back(*id);
            } else {
                for (char ch : piece) {
                    out.push_back(byte_[static_cast<unsigned char>(ch)]);
                }
            }
        }
    }

    std::vector<std::string> tokens_;
    std::vector<float> scores_;
    std::vector<token_type> types_;
    std::unordered_map<std::string, std::uint32_t> ids_;
    std::array<std::uint32_t, 256> byte_{};
    std::vector<std::uint32_t> specials_;
    std::uint32_t bos_ = no_token;
    std::uint32_t eos_ = no_token;
    bool add_bos_ = true;
    bool add_eos_ = false;
    bool add_space_prefix_ = true;
};

}  // namespace branes::synapse::tokenizer

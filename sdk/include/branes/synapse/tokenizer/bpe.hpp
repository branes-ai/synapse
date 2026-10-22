#pragma once
/// @file bpe.hpp
/// @brief Byte-level BPE tokenization of Llama-3-family vocabularies, as llama.cpp does it.
///
/// encode(text):
///   1. Split out special tokens: every control, unknown or user-defined
///      token's text found in the input becomes that token directly (longest
///      first). Control and unknown tokens are only recognized with
///      parse_special; user-defined ones always are, as in Hugging Face's
///      tokenizers and llama.cpp.
///   2. Pre-tokenize each remaining fragment into words (split_llama3).
///   3. Spell each word's UTF-8 bytes with GPT-2's byte map. A word that is a
///      token as a whole is that token (ignore_merges); otherwise apply the
///      merges, lowest rank first (ties: leftmost), until none applies, and
///      look each resulting piece up (a piece with no token falls back to its
///      single-character tokens).
///   4. Prepend BOS when add_bos.
///
/// Model
/// -----
///   bpe_tokenizer
///     tokens_   : vector<string>   — token texts, by id (as stored in the file)
///     types_    : vector<token_type>
///     ids_      : map text -> id   (a text listed twice maps to its last id)
///     ranks_    : map (left, right) -> merge rank
///     specials_ : ids of control / unknown / user-defined tokens, longest text first
///     bos_, eos_: ids or no_token; add_bos_
///
/// Invariants (established by from_gguf)
/// ----------
///   T1: every id in ids_, specials_, bos_, eos_ is < tokens_.size()
///   T2: decode(encode(s)) == s for valid UTF-8 s with no special-token text.
///       From two load-time checks: each byte has a single-character token,
///       and each merge's operands and result are tokens. So every piece BPE
///       leaves is a token, and no character is ever dropped. (Every
///       byte-level vocabulary llama.cpp ships satisfies both.)
///
/// The tokenizer allocates; it belongs at the application boundary, outside
/// the allocation-free session::run.

#include <branes/synapse/gguf/reader.hpp>
#include <branes/synapse/result.hpp>
#include <branes/synapse/tokenizer/pretokenize.hpp>
#include <branes/synapse/tokenizer/unicode.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <queue>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
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
    unsupported_model = 1,  ///< tokenizer.ggml.model is not "gpt2"
    unsupported_pre,        ///< the pre-tokenizer is not Llama 3's
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
    bool add_bos = true;         ///< prepend BOS if the vocabulary asks for it
    bool parse_special = false;  ///< recognize control tokens' text (e.g. "<|eot_id|>")
};

class bpe_tokenizer {
public:
    /// The tokenizer of a GGUF file.  Pre: none (the file is untrusted).
    /// Post: T1, T2, or the first problem found.
    static result<bpe_tokenizer, tokenizer_error> from_gguf(const gguf::gguf_file& f) {
        const auto fail = [](tokenizer_errc code, std::string_view name) {
            return tokenizer_error{code, std::string(name)};
        };
        const auto model = f.get<std::string_view>("tokenizer.ggml.model");
        if (!model) {
            return fail(tokenizer_errc::missing_key, "tokenizer.ggml.model");
        }
        if (*model != "gpt2") {
            return fail(tokenizer_errc::unsupported_model, "tokenizer.ggml.model");
        }
        const auto pre = f.get<std::string_view>("tokenizer.ggml.pre");
        if (!pre || (*pre != "llama-bpe" && *pre != "llama3" && *pre != "llama-v3")) {
            return fail(tokenizer_errc::unsupported_pre, "tokenizer.ggml.pre");
        }

        bpe_tokenizer t;
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

        t.types_.assign(t.tokens_.size(), token_type::normal);
        if (const gguf::array_value* types = f.get_array("tokenizer.ggml.token_type")) {
            if (types->element_type() != gguf::value_type::int32 || types->size() != t.tokens_.size()) {
                return fail(tokenizer_errc::bad_value, "tokenizer.ggml.token_type");
            }
            for (std::size_t i = 0; i < t.types_.size(); ++i) {
                t.types_[i] = static_cast<token_type>(*types->at<std::int32_t>(i));
            }
        }

        const gguf::array_value* merges = f.get_array("tokenizer.ggml.merges");
        if (merges == nullptr) {
            return fail(tokenizer_errc::missing_key, "tokenizer.ggml.merges");
        }
        if (merges->element_type() != gguf::value_type::string) {
            return fail(tokenizer_errc::bad_value, "tokenizer.ggml.merges");
        }
        const auto rules = merges->strings();
        for (std::size_t i = 0; i < rules.size(); ++i) {
            // "left right", split at the first space after the first character
            // (a left side can itself be a space-like character).
            const std::size_t sp = rules[i].find(' ', 1);
            if (sp == std::string_view::npos) {
                return fail(tokenizer_errc::bad_value, "tokenizer.ggml.merges");
            }
            const std::string left(rules[i].substr(0, sp));
            const std::string right(rules[i].substr(sp + 1));
            if (!t.ids_.contains(left) || !t.ids_.contains(right) || !t.ids_.contains(left + right)) {
                return fail(tokenizer_errc::bad_value, "tokenizer.ggml.merges");  // T2
            }
            t.ranks_.emplace(pair_key(left, right), i);
        }

        for (auto [key, out] :
             {std::pair{"tokenizer.ggml.bos_token_id", &t.bos_}, std::pair{"tokenizer.ggml.eos_token_id", &t.eos_}}) {
            if (f.find(key) == nullptr) {
                continue;
            }
            const auto id = f.get<std::uint32_t>(key);
            if (!id || *id >= t.tokens_.size()) {
                return fail(tokenizer_errc::bad_value, key);  // T1
            }
            *out = *id;
        }
        if (f.find("tokenizer.ggml.add_bos_token") != nullptr) {
            const auto add = f.get<bool>("tokenizer.ggml.add_bos_token");
            if (!add) {
                return fail(tokenizer_errc::bad_value, "tokenizer.ggml.add_bos_token");
            }
            t.add_bos_ = *add;
        }
        if (t.add_bos_ && t.bos_ == no_token) {
            return fail(tokenizer_errc::missing_key, "tokenizer.ggml.bos_token_id");
        }

        // T2: every byte must be spellable, or some input cannot be encoded.
        for (std::uint32_t b = 0; b < 256; ++b) {
            std::string s;
            append_utf8(byte_to_unicode[b], s);
            if (!t.ids_.contains(s)) {
                return fail(tokenizer_errc::bad_value, "tokenizer.ggml.tokens");
            }
        }

        for (std::size_t i = 0; i < t.tokens_.size(); ++i) {
            const token_type ty = t.types_[i];
            if ((ty == token_type::control || ty == token_type::unknown || ty == token_type::user_defined) &&
                !t.tokens_[i].empty()) {
                t.specials_.push_back(static_cast<std::uint32_t>(i));
            }
        }
        std::stable_sort(t.specials_.begin(), t.specials_.end(), [&](std::uint32_t a, std::uint32_t b) {
            return t.tokens_[a].size() > t.tokens_[b].size();
        });
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
    token_type type(std::uint32_t id) const noexcept {
        return id < types_.size() ? types_[id] : token_type::undefined;
    }

    /// The token ids of `text`.
    std::vector<std::uint32_t> encode(std::string_view text, encode_options opt = {}) const {
        std::vector<std::uint32_t> out;
        if (opt.add_bos && add_bos_) {
            out.push_back(bos_);
        }
        // Fragments: raw text (offset, length) or a special token.
        struct fragment {
            std::size_t offset, length;
            std::uint32_t token;  // no_token for raw text
        };
        std::vector<fragment> frags{{0, text.size(), no_token}};
        for (std::uint32_t id : specials_) {
            if (!opt.parse_special && types_[id] != token_type::user_defined) {
                continue;
            }
            const std::string_view s = tokens_[id];
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
        for (const fragment& fr : frags) {
            if (fr.token != no_token) {
                out.push_back(fr.token);
            } else {
                encode_text(text.substr(fr.offset, fr.length), out);
            }
        }
        return out;
    }

    /// The bytes token `id` stands for. Control and unknown tokens decode to
    /// their text only when `special` is set; unused, byte and undefined
    /// tokens to nothing (as llama.cpp's token_to_piece).
    std::string decode(std::uint32_t id, bool special = false) const {
        std::string out;
        if (id >= tokens_.size()) {
            return out;
        }
        const std::string_view text = tokens_[id];
        switch (types_[id]) {
        case token_type::normal:
            for (std::uint32_t cp : decode_utf8(text)) {
                const int b = unicode_to_byte(cp);
                if (b >= 0) {
                    out.push_back(static_cast<char>(b));
                } else {
                    append_utf8(cp, out);  // not a byte-level character: keep it as text
                }
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
    bpe_tokenizer() = default;

    /// An unambiguous key for a merge pair: left's length, then both texts.
    static std::string pair_key(std::string_view left, std::string_view right) {
        std::string key = std::to_string(left.size());
        key += ':';
        key += left;
        key += right;
        return key;
    }

    std::optional<std::size_t> rank(std::string_view left, std::string_view right) const {
        const auto it = ranks_.find(pair_key(left, right));
        return it == ranks_.end() ? std::nullopt : std::optional<std::size_t>(it->second);
    }

    std::optional<std::uint32_t> id_of(const std::string& text) const {
        const auto it = ids_.find(text);
        return it == ids_.end() ? std::nullopt : std::optional<std::uint32_t>(it->second);
    }

    void encode_text(std::string_view text, std::vector<std::uint32_t>& out) const {
        const std::vector<std::uint32_t> cpts = decode_utf8(text);
        std::size_t start = 0;
        for (std::size_t len : split_llama3(cpts)) {
            // The word's bytes (re-encoded from its code points, so invalid
            // input reads as U+FFFD), each spelled as its byte-level character.
            std::string bytes;
            for (std::size_t i = start; i < start + len; ++i) {
                append_utf8(cpts[i], bytes);
            }
            start += len;
            std::string word;
            for (char ch : bytes) {
                append_utf8(byte_to_unicode[static_cast<unsigned char>(ch)], word);
            }
            merge_word(word, out);
        }
    }

    /// BPE within one word: repeatedly merge the adjacent pair of lowest rank.
    void merge_word(const std::string& word, std::vector<std::uint32_t>& out) const {
        if (const auto whole = id_of(word)) {  // ignore_merges
            out.push_back(*whole);
            return;
        }
        // Symbols: one per character, as a linked list over byte ranges.
        struct symbol {
            std::size_t begin, length;
            int prev, next;
        };
        std::vector<symbol> sym;
        for (std::size_t i = 0; i < word.size();) {
            // Length from the lead byte's high nibble, as llama.cpp's unicode_len_utf8.
            const auto lead = static_cast<unsigned char>(word[i]);
            const std::size_t n = lead < 0xC0 ? 1 : lead < 0xE0 ? 2 : lead < 0xF0 ? 3 : 4;
            const std::size_t len = std::min(n, word.size() - i);
            sym.push_back({i, len, static_cast<int>(sym.size()) - 1, -1});
            i += len;
        }
        for (std::size_t i = 0; i + 1 < sym.size(); ++i) {
            sym[i].next = static_cast<int>(i + 1);
        }
        struct bigram {
            int left, right;
            std::size_t rank;
            std::string text;
        };
        // Lowest rank first; ties go to the leftmost pair.
        const auto worse = [](const bigram& a, const bigram& b) {
            return a.rank > b.rank || (a.rank == b.rank && a.left > b.left);
        };
        std::priority_queue<bigram, std::vector<bigram>, decltype(worse)> queue(worse);
        const auto text_of = [&](int i) {
            return std::string_view(word).substr(sym[static_cast<std::size_t>(i)].begin,
                                                 sym[static_cast<std::size_t>(i)].length);
        };
        const auto add = [&](int l, int r) {
            if (l < 0 || r < 0) {
                return;
            }
            if (const auto rk = rank(text_of(l), text_of(r))) {
                queue.push({l, r, *rk, std::string(text_of(l)) + std::string(text_of(r))});
            }
        };
        for (std::size_t i = 1; i < sym.size(); ++i) {
            add(static_cast<int>(i - 1), static_cast<int>(i));
        }
        while (!queue.empty()) {
            const bigram b = queue.top();
            queue.pop();
            symbol& l = sym[static_cast<std::size_t>(b.left)];
            symbol& r = sym[static_cast<std::size_t>(b.right)];
            if (l.length == 0 || r.length == 0 ||
                std::string(text_of(b.left)) + std::string(text_of(b.right)) != b.text) {
                continue;  // stale: one side has merged since
            }
            l.length += r.length;
            r.length = 0;
            l.next = r.next;
            if (r.next >= 0) {
                sym[static_cast<std::size_t>(r.next)].prev = b.left;
            }
            add(l.prev, b.left);
            add(b.left, l.next);
        }
        for (const symbol& s : sym) {
            if (s.length == 0) {
                continue;
            }
            const std::string piece = word.substr(s.begin, s.length);
            if (const auto id = id_of(piece)) {
                out.push_back(*id);
            } else {
                // No token for the piece: fall back to its single bytes' tokens,
                // as llama.cpp does. Unreachable for a vocabulary that passed
                // from_gguf's checks (T2); kept for parity.
                for (char ch : piece) {
                    if (const auto b = id_of(std::string(1, ch))) {
                        out.push_back(*b);
                    }
                }
            }
        }
    }

    std::vector<std::string> tokens_;
    std::vector<token_type> types_;
    std::unordered_map<std::string, std::uint32_t> ids_;
    std::unordered_map<std::string, std::size_t> ranks_;
    std::vector<std::uint32_t> specials_;
    std::uint32_t bos_ = no_token;
    std::uint32_t eos_ = no_token;
    bool add_bos_ = true;  ///< llama.cpp's default for the llama3 pre-tokenizer
};

}  // namespace branes::synapse::tokenizer

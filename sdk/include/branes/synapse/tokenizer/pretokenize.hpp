#pragma once
/// @file pretokenize.hpp
/// @brief The Llama-3, Qwen2 and Tekken pre-tokenizers: split text into the words BPE merges within.
///
/// Implements the split regex of Llama 3's tokenizer.json,
///
///   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}{1,3}
///   | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
///
/// as a hand-written matcher, a port of llama.cpp's
/// unicode_regex_split_custom_llama3 (std::regex has no Unicode classes).
/// Qwen2's regex is the same with \p{N} in place of \p{N}{1,3}: each digit
/// is a word of its own. Alternatives are tried in order at each position, so
/// the result is the regex's leftmost-first match sequence.
///
/// Tekken (Mistral Nemo, Mistral Small 3, Ministral 3) splits on letter case.
/// llama.cpp runs it through std::regex as
///
///   p?U*L+ | p?U+L* | \p{N} | ?[^\s\p{L}\p{N}]+[\r\n/]* | \s*[\r\n]+
///   | \s+(?!\S) | \s+          with p = [^\r\n\p{L}\p{N}],
///
/// U a letter other than a-z and L a letter other than A-Z (its stand-in for
/// \p{Lu}... and \p{Ll}...: a non-ASCII letter is both), and no
/// contractions. split_words reproduces std::regex's backtracking order: U*
/// gives back letters until L+ can match.

#include <branes/synapse/tokenizer/unicode.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace branes::synapse::tokenizer {

/// Which split regex a vocabulary uses (tokenizer.ggml.pre).
enum class pre_style : std::uint8_t {
    llama3,  ///< digits in runs of up to 3
    qwen2,   ///< one digit per word
    tekken,  ///< one digit per word; letter runs split on case; no contractions
};

/// Lengths, in code points, of the consecutive words of cpts (they sum to
/// cpts.size(); none is empty).
inline std::vector<std::size_t> split_words(std::span<const std::uint32_t> cpts, pre_style style) {
    const std::size_t max_digits = style == pre_style::llama3 ? 3 : 1;
    const bool tekken = style == pre_style::tekken;
    std::vector<std::size_t> words;
    const std::size_t end = cpts.size();
    constexpr std::uint32_t out_of_range = 0xFFFFFFFF;
    const auto cpt = [&](std::size_t p) { return p < end ? cpts[p] : out_of_range; };
    const auto letter = [&](std::size_t p) { return p < end && is_letter(cpts[p]); };
    const auto number = [&](std::size_t p) { return p < end && is_number(cpts[p]); };
    const auto space = [&](std::size_t p) { return p < end && is_whitespace(cpts[p]); };
    const auto lower = [](std::uint32_t c) { return c >= 'A' && c <= 'Z' ? c + ('a' - 'A') : c; };

    std::size_t prev = 0;
    const auto emit = [&](std::size_t to) {
        if (to > prev) {
            words.push_back(to - prev);
        }
        prev = to;
    };

    // Tekken's stand-ins for upper- and lower-case letters.
    const auto upper = [&](std::size_t p) { return letter(p) && !(cpts[p] >= 'a' && cpts[p] <= 'z'); };
    const auto lower_case = [&](std::size_t p) { return letter(p) && !(cpts[p] >= 'A' && cpts[p] <= 'Z'); };

    for (std::size_t pos = 0; pos < end;) {
        const std::uint32_t c = cpts[pos];

        if (tekken) {
            // p?: a non-letter, non-number prefix other than \r \n. Without
            // it the match would start at that same non-letter and fail, so
            // the regex's retry without the prefix never succeeds.
            const bool prefix = !(c == '\r' || c == '\n' || letter(pos) || number(pos));
            const std::size_t q = prefix ? pos + 1 : pos;
            if (letter(q) && (prefix || letter(pos))) {
                std::size_t u = 0;  // the greedy U* run
                while (upper(q + u)) {
                    ++u;
                }
                // p?U*L+: U* backs off until an L follows.
                std::size_t k = u + 1;
                while (k > 0 && !lower_case(q + k - 1)) {
                    --k;
                }
                if (k > 0) {
                    std::size_t e = q + k - 1;
                    while (lower_case(e)) {
                        ++e;
                    }
                    emit(pos = e);
                    continue;
                }
                // p?U+L*
                if (u > 0) {
                    std::size_t e = q + u;
                    while (lower_case(e)) {
                        ++e;
                    }
                    emit(pos = e);
                    continue;
                }
            }
        }

        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (!tekken && c == '\'' && pos + 1 < end) {
            const std::uint32_t n1 = lower(cpt(pos + 1));
            if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') {
                emit(pos += 2);
                continue;
            }
            if (pos + 2 < end) {
                const std::uint32_t n2 = lower(cpt(pos + 2));
                if ((n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e') || (n1 == 'l' && n2 == 'l')) {
                    emit(pos += 3);
                    continue;
                }
            }
        }

        // [^\r\n\p{L}\p{N}]?\p{L}+
        if (!tekken && !(c == '\r' || c == '\n' || number(pos)) && (letter(pos) || letter(pos + 1))) {
            ++pos;
            while (letter(pos)) {
                ++pos;
            }
            emit(pos);
            continue;
        }

        // \p{N}{1,3} (Llama 3) or \p{N} (Qwen2)
        if (number(pos)) {
            std::size_t start = pos;
            while (number(pos)) {
                if (++pos - start >= max_digits) {
                    emit(pos);
                    start = pos;
                }
            }
            emit(pos);
            continue;
        }

        // <space>?[^\s\p{L}\p{N}]+[\r\n]*
        const std::size_t q = c == ' ' ? pos + 1 : pos;
        const auto other = [&](std::size_t p) { return p < end && !space(p) && !letter(p) && !number(p); };
        if (other(q)) {
            pos = q;
            while (other(pos)) {
                ++pos;
            }
            while (cpt(pos) == '\r' || cpt(pos) == '\n' || (tekken && cpt(pos) == '/')) {
                ++pos;
            }
            emit(pos);
            continue;
        }

        std::size_t n_space = 0;
        std::size_t last_newline_end = 0;
        while (space(pos + n_space)) {
            const std::uint32_t s = cpts[pos + n_space];
            if (s == '\r' || s == '\n') {
                last_newline_end = pos + n_space + 1;
            }
            ++n_space;
        }
        // \s*[\r\n]+
        if (last_newline_end > 0) {
            emit(pos = last_newline_end);
            continue;
        }
        // \s+(?!\S): leave the last space to prefix the next word
        if (n_space > 1 && pos + n_space < end) {
            emit(pos += n_space - 1);
            continue;
        }
        // \s+
        if (n_space > 0) {
            emit(pos += n_space);
            continue;
        }
        emit(++pos);  // no alternative matched: a word of its own
    }
    return words;
}

/// The Llama-3 split.
inline std::vector<std::size_t> split_llama3(std::span<const std::uint32_t> cpts) {
    return split_words(cpts, pre_style::llama3);
}

}  // namespace branes::synapse::tokenizer

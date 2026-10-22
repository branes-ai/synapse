#pragma once
/// @file unicode.hpp
/// @brief The Unicode pieces of byte-level BPE: UTF-8, code point classes, GPT-2's byte map.
///
/// Everything here mirrors llama.cpp's unicode.cpp, so that tokenization is
/// identical on any input, valid or not:
///
///   - decode_utf8 maps each byte that does not start a well-formed sequence
///     to U+FFFD and moves on by one byte. Like llama.cpp it checks only the
///     lead and continuation bit patterns, so it accepts overlong forms and
///     surrogates; sequences that decode past U+10FFFF (which llama.cpp
///     rejects with an exception) become U+FFFD here.
///   - is_letter / is_number are the general categories L* and N* of Unicode
///     15.1.0 (unicode_tables.hpp, generated; identical to llama.cpp's table
///     for every code point), is_whitespace the White_Space property.
///   - byte_to_unicode is GPT-2's bytes_to_unicode: byte-level BPE
///     vocabularies spell every byte as one printable code point.

#include <branes/synapse/tokenizer/unicode_tables.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace branes::synapse::tokenizer {

inline constexpr std::uint32_t replacement_character = 0xFFFD;

/// The code points of `s`, with llama.cpp's handling of invalid input.
inline std::vector<std::uint32_t> decode_utf8(std::string_view s) {
    std::vector<std::uint32_t> out;
    out.reserve(s.size());
    const auto byte = [&](std::size_t i) { return static_cast<std::uint32_t>(static_cast<unsigned char>(s[i])); };
    const auto cont = [&](std::size_t i) { return i < s.size() && (byte(i) & 0xC0u) == 0x80u; };
    for (std::size_t i = 0; i < s.size();) {
        const std::uint32_t b = byte(i);
        std::uint32_t cp = replacement_character;
        std::size_t len = 1;
        if ((b & 0x80u) == 0) {
            cp = b;
        } else if ((b & 0xE0u) == 0xC0u && cont(i + 1)) {
            cp = ((b & 0x1Fu) << 6) | (byte(i + 1) & 0x3Fu);
            len = 2;
        } else if ((b & 0xF0u) == 0xE0u && cont(i + 1) && cont(i + 2)) {
            cp = ((b & 0x0Fu) << 12) | ((byte(i + 1) & 0x3Fu) << 6) | (byte(i + 2) & 0x3Fu);
            len = 3;
        } else if ((b & 0xF8u) == 0xF0u && cont(i + 1) && cont(i + 2) && cont(i + 3)) {
            cp = ((b & 0x07u) << 18) | ((byte(i + 1) & 0x3Fu) << 12) | ((byte(i + 2) & 0x3Fu) << 6) |
                 (byte(i + 3) & 0x3Fu);
            len = 4;
            if (cp > 0x10FFFFu) {
                cp = replacement_character;
                len = 1;
            }
        }
        out.push_back(cp);
        i += len;
    }
    return out;
}

/// Append the UTF-8 encoding of cp (<= U+10FFFF) to out.
inline void append_utf8(std::uint32_t cp, std::string& out) {
    const auto put = [&](std::uint32_t v) { out.push_back(static_cast<char>(v)); };
    if (cp <= 0x7Fu) {
        put(cp);
    } else if (cp <= 0x7FFu) {
        put(0xC0u | (cp >> 6));
        put(0x80u | (cp & 0x3Fu));
    } else if (cp <= 0xFFFFu) {
        put(0xE0u | (cp >> 12));
        put(0x80u | ((cp >> 6) & 0x3Fu));
        put(0x80u | (cp & 0x3Fu));
    } else {
        put(0xF0u | (cp >> 18));
        put(0x80u | ((cp >> 12) & 0x3Fu));
        put(0x80u | ((cp >> 6) & 0x3Fu));
        put(0x80u | (cp & 0x3Fu));
    }
}

namespace detail {

template <std::size_t N>
constexpr bool in_ranges(const std::array<code_point_range, N>& ranges, std::uint32_t cp) noexcept {
    const auto it = std::upper_bound(
        ranges.begin(), ranges.end(), cp, [](std::uint32_t c, const code_point_range& r) { return c < r.first; });
    return it != ranges.begin() && cp <= std::prev(it)->last;
}

}  // namespace detail

/// \p{L}: general category Lu, Ll, Lt, Lm or Lo.
constexpr bool is_letter(std::uint32_t cp) noexcept {
    return detail::in_ranges(detail::letter_ranges, cp);
}

/// \p{N}: general category Nd, Nl or No.
constexpr bool is_number(std::uint32_t cp) noexcept {
    return detail::in_ranges(detail::number_ranges, cp);
}

/// \s: the White_Space property.
constexpr bool is_whitespace(std::uint32_t cp) noexcept {
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
           cp == 0x3000;
}

/// GPT-2's bytes_to_unicode: printable bytes stand for themselves; the other
/// 68 are shifted to U+0100 and up, in byte order.
inline constexpr std::array<std::uint32_t, 256> byte_to_unicode = [] {
    std::array<std::uint32_t, 256> map{};
    std::uint32_t shifted = 0;
    for (std::uint32_t b = 0; b < 256; ++b) {
        const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        map[b] = printable ? b : 256 + shifted++;
    }
    return map;
}();

/// The byte a code point of a byte-level token stands for, or -1.
constexpr int unicode_to_byte(std::uint32_t cp) noexcept {
    if (cp < 256 && byte_to_unicode[cp] == cp) {
        return static_cast<int>(cp);
    }
    if (cp >= 256 && cp < 256 + 68) {
        for (std::size_t b = 0; b < 256; ++b) {
            if (byte_to_unicode[b] == cp) {
                return static_cast<int>(b);
            }
        }
    }
    return -1;
}

}  // namespace branes::synapse::tokenizer

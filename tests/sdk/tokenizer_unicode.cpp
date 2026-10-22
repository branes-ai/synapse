// Unit tests for the tokenizer's Unicode layer and the Llama-3 pre-tokenizer.
// (Agreement with llama.cpp on whole inputs is checked by tokenizer_golden.)

#include <branes/synapse/tokenizer/pretokenize.hpp>
#include <branes/synapse/tokenizer/unicode.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <set>
#include <string>
#include <string_view>
#include <vector>

using namespace branes::synapse::tokenizer;

namespace {

std::vector<std::string> words(std::string_view text) {
    const auto cpts = decode_utf8(text);
    std::vector<std::string> out;
    std::size_t start = 0;
    for (std::size_t len : split_llama3(cpts)) {
        std::string w;
        for (std::size_t i = start; i < start + len; ++i) {
            append_utf8(cpts[i], w);
        }
        out.push_back(w);
        start += len;
    }
    return out;
}

}  // namespace

TEST_CASE("tokenizer_unicode: UTF-8 decoding follows llama.cpp on invalid input", "[tokenizer][unicode]") {
    CHECK(decode_utf8("aé€🦙") == std::vector<std::uint32_t>{'a', 0xE9, 0x20AC, 0x1F999});
    CHECK(decode_utf8("\x80") == std::vector<std::uint32_t>{replacement_character});  // lone continuation
    CHECK(decode_utf8("\xC3") == std::vector<std::uint32_t>{replacement_character});  // truncated
    CHECK(decode_utf8("\xE2\x82x") == std::vector<std::uint32_t>{replacement_character, replacement_character, 'x'});
    CHECK(decode_utf8("\xF8\x88") == std::vector<std::uint32_t>{replacement_character, replacement_character});
    CHECK(decode_utf8("\xC0\x80") == std::vector<std::uint32_t>{0});          // overlong, accepted as llama.cpp does
    CHECK(decode_utf8("\xF4\x90\x80\x80").front() == replacement_character);  // past U+10FFFF
    std::string s;
    for (std::uint32_t cp : {0x24u, 0xA2u, 0x939u, 0x20ACu, 0xD55Cu, 0x10348u}) {
        append_utf8(cp, s);
    }
    CHECK(decode_utf8(s) == std::vector<std::uint32_t>{0x24, 0xA2, 0x939, 0x20AC, 0xD55C, 0x10348});
}

TEST_CASE("tokenizer_unicode: letter, number and whitespace classes", "[tokenizer][unicode]") {
    for (std::uint32_t cp : {0x61u, 0x5Au, 0xE9u, 0x416u, 0x4E2Du, 0x5D0u, 0x2C6u}) {
        CHECK(is_letter(cp));
    }
    for (std::uint32_t cp : {0x30u, 0x39u, 0xBDu, 0x663u, 0x2167u, 0xB2u}) {
        CHECK(is_number(cp));
        CHECK_FALSE(is_letter(cp));
    }
    for (std::uint32_t cp : {0x20u, 0x09u, 0x0Au, 0x0Du, 0xA0u, 0x3000u, 0x2003u}) {
        CHECK(is_whitespace(cp));
    }
    for (std::uint32_t cp : {0x21u, 0x5Fu, 0x1F999u, 0x200Bu, 0x10FFFFu}) {
        CHECK_FALSE(is_letter(cp));
        CHECK_FALSE(is_number(cp));
        CHECK_FALSE(is_whitespace(cp));
    }
}

TEST_CASE("tokenizer_unicode: GPT-2's byte map is a bijection onto printable code points", "[tokenizer][unicode]") {
    std::set<std::uint32_t> seen;
    for (std::uint32_t b = 0; b < 256; ++b) {
        const std::uint32_t cp = byte_to_unicode[b];
        CHECK_FALSE(is_whitespace(cp));
        CHECK(unicode_to_byte(cp) == static_cast<int>(b));
        seen.insert(cp);
    }
    CHECK(seen.size() == 256);
    CHECK(byte_to_unicode[' '] == 0x120);   // the familiar "Ġ"
    CHECK(byte_to_unicode['\n'] == 0x10A);  // "Ċ"
    CHECK(unicode_to_byte(0x2581) == -1);
}

TEST_CASE("tokenizer_unicode: the Llama-3 split regex", "[tokenizer][pretokenize]") {
    using V = std::vector<std::string>;
    CHECK(words("Hello world") == V{"Hello", " world"});
    CHECK(words("it's we'RE") == V{"it", "'s", " we", "'RE"});
    CHECK(words("1234567") == V{"123", "456", "7"});
    CHECK(words("a1b22") == V{"a", "1", "b", "22"});
    CHECK(words(" !!? x") == V{" !!?", " x"});
    CHECK(words("x\n\ny") == V{"x", "\n\n", "y"});
    CHECK(words("a   b") == V{"a", "  ", " b"});  // \s+(?!\S) leaves the last space to the word
    CHECK(words("a   ") == V{"a", "   "});        // at the end, \s+ takes all
    CHECK(words("(x)") == V{"(x", ")"});          // [^\r\n\p{L}\p{N}]?\p{L}+
    CHECK(words("..\r\n") == V{"..\r\n"});        // punctuation takes trailing newlines
    CHECK(words("").empty());
    // Lengths always cover the input.
    const auto cpts = decode_utf8("Mixed 中文 and 123 ✨\t\n end ");
    std::size_t total = 0;
    for (std::size_t len : split_llama3(cpts)) {
        CHECK(len > 0);
        total += len;
    }
    CHECK(total == cpts.size());
}

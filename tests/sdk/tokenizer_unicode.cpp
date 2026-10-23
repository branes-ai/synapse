// Unit tests for the tokenizer's Unicode layer and the Llama-3 and Tekken
// pre-tokenizers.
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

std::vector<std::string> words(std::string_view text, pre_style style = pre_style::llama3) {
    const auto cpts = decode_utf8(text);
    std::vector<std::string> out;
    std::size_t start = 0;
    for (std::size_t len : split_words(cpts, style)) {
        std::string w;
        for (std::size_t i = start; i < start + len; ++i) {
            append_utf8(cpts[i], w);
        }
        out.push_back(w);
        start += len;
    }
    return out;
}

std::vector<std::string> tekken_words(std::string_view text) {
    return words(text, pre_style::tekken);
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

// Expected splits are llama.cpp's own: unicode_regex_split with Tekken's
// regex (llama-vocab.cpp, LLAMA_VOCAB_PRE_TYPE_TEKKEN) at e6f2ec01f. Many of
// these splits leave the tokens unchanged for any vocabulary trained with
// the same regex, so the golden tests cannot see them; this pins them.
TEST_CASE("tokenizer_unicode: the Tekken split regex", "[tokenizer][pretokenize][tekken]") {
    using V = std::vector<std::string>;
    // U*L+ splits CamelCase; U* gives back the last capital to a lowercase run.
    CHECK(tekken_words("HelloWorld XMLHttpRequest iPhone McDonald's") ==
          V{"Hello", "World", " XMLHttp", "Request", " i", "Phone", " Mc", "Donald", "'s"});
    // A non-ASCII letter is both upper and lower case.
    CHECK(tekken_words("\xc3\x89"
                       "COLE \xc3\x89"
                       "cole \xc3\xa9"
                       "COLE \xc3\x80"
                       "BCdef") == V{"\xc3\x89",
                                     "COLE",
                                     " \xc3\x89"
                                     "cole",
                                     " \xc3\xa9",
                                     "COLE",
                                     " \xc3\x80"
                                     "BCdef"});
    CHECK(tekken_words("\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82\xd0\x9c\xd0\xb8\xd1\x80 "
                       "\xd0\x9f\xd0\xa0\xd0\x98\xd0\x92\xd0\x95\xd0\xa2\xd0\xbc\xd0\xb8\xd1\x80") ==
          V{"\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82\xd0\x9c\xd0\xb8\xd1\x80",
            " \xd0\x9f\xd0\xa0\xd0\x98\xd0\x92\xd0\x95\xd0\xa2\xd0\xbc\xd0\xb8\xd1\x80"});
    // Punctuation runs take trailing newlines and slashes.
    CHECK(tekken_words("end.\n/next !\r\n//x") == V{"end", ".\n/", "next", " !\r\n//", "x"});
    CHECK(tekken_words("a/b ../up") == V{"a", "/b", " ../", "up"});
    // One digit per word.
    CHECK(tekken_words("x1Y22 ABC123def") == V{"x", "1", "Y", "2", "2", " ABC", "1", "2", "3", "def"});
    // A combining mark (U+0301) is not a letter: it prefixes the next run.
    CHECK(tekken_words("e\xcc\x81"
                       "cole MIXED\xcc\x81"
                       "case") == V{"e",
                                    "\xcc\x81"
                                    "cole",
                                    " MIXED",
                                    "\xcc\x81"
                                    "case"});
    CHECK(tekken_words("  two  spaces\t\ttabs ") == V{" ", " two", " ", " spaces", "\t", "\ttabs", " "});
    // No contractions.
    CHECK(tekken_words("it's 'S") == V{"it", "'s", " '", "S"});
    CHECK(tekken_words("(Paren) [x]") == V{"(Paren", ")", " [", "x", "]"});
}

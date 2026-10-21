// Unit tests for branes::synapse::gguf::parse. Images are assembled in memory
// by a small GGUF writer, so each case states exactly which bytes it feeds the
// parser: one valid model-shaped file for the round trip, and one targeted
// corruption per rejection rule.

#include <branes/synapse/gguf/reader.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <span>
#include <string_view>
#include <vector>

using namespace branes::synapse::gguf;

namespace {

// ── a minimal GGUF writer ──────────────────────────────────────────────────

class writer {
public:
    template <typename T>
    writer& put(T v) {
        append(&v, sizeof(T));
        return *this;
    }
    writer& put_string(std::string_view s) {
        put<std::uint64_t>(s.size());
        append(s.data(), s.size());
        return *this;
    }
    writer& pad_to(std::size_t alignment) {
        while (bytes_.size() % alignment != 0) {
            bytes_.push_back(std::byte{0});
        }
        return *this;
    }
    std::vector<std::byte>& bytes() {
        return bytes_;
    }

private:
    // resize + memcpy rather than vector::insert: GCC 12/13 at -O2 reports a
    // false -Wstringop-overflow on small inserted ranges.
    void append(const void* p, std::size_t n) {
        const std::size_t old = bytes_.size();
        bytes_.resize(old + n);
        if (n != 0) {
            std::memcpy(bytes_.data() + old, p, n);
        }
    }

    std::vector<std::byte> bytes_;
};

using kv_fn = std::function<void(writer&)>;

template <typename T>
kv_fn kv(std::string_view key, value_type type, T v) {
    return [=](writer& w) { w.put_string(key).put(static_cast<std::uint32_t>(type)).put(v); };
}

kv_fn kv_string(std::string_view key, std::string_view v) {
    return [=](writer& w) { w.put_string(key).put(static_cast<std::uint32_t>(value_type::string)).put_string(v); };
}

struct tensor_spec {
    std::string_view name;
    std::vector<std::uint64_t> ne;
    std::uint32_t type;
    std::uint64_t offset;
};

std::vector<std::byte> build(std::uint32_t version,
                             const std::vector<kv_fn>& kvs,
                             const std::vector<tensor_spec>& tensors,
                             std::size_t data_bytes,
                             std::size_t alignment = 32) {
    writer w;
    w.put<std::uint32_t>(0x46554747);  // "GGUF"
    w.put(version).put<std::uint64_t>(tensors.size()).put<std::uint64_t>(kvs.size());
    for (const auto& f : kvs) {
        f(w);
    }
    for (const auto& t : tensors) {
        w.put_string(t.name).put<std::uint32_t>(static_cast<std::uint32_t>(t.ne.size()));
        for (auto d : t.ne) {
            w.put(d);
        }
        w.put(t.type).put(t.offset);
    }
    w.pad_to(alignment);
    for (std::size_t i = 0; i < data_bytes; ++i) {
        w.bytes().push_back(static_cast<std::byte>(i & 0xff));
    }
    return w.bytes();
}

constexpr std::uint32_t f32_tag = 0;
constexpr std::uint32_t q4_0_tag = 2;

// A model-shaped file: every scalar type, a string, a string array and a float
// array (tokenizer style), an f32 embedding [4, 3] at offset 0 (48 bytes) and a
// Q4_0 weight [64, 2] at offset 64 (2 rows x 2 blocks x 18 bytes = 72 bytes).
std::vector<kv_fn> model_kvs() {
    return {
        kv_string("general.architecture", "llama"),
        kv("llama.context_length", value_type::uint32, std::uint32_t{2048}),
        kv("llama.rope.freq_base", value_type::float32, 10000.0f),
        kv("test.u8", value_type::uint8, std::uint8_t{200}),
        kv("test.i8", value_type::int8, std::int8_t{-5}),
        kv("test.u16", value_type::uint16, std::uint16_t{60000}),
        kv("test.i16", value_type::int16, std::int16_t{-30000}),
        kv("test.i32", value_type::int32, std::int32_t{-70000}),
        kv("test.u64", value_type::uint64, std::uint64_t{1} << 40),
        kv("test.i64", value_type::int64, std::int64_t{-1} << 40),
        kv("test.f64", value_type::float64, 2.5),
        kv("test.flag", value_type::boolean, std::uint8_t{1}),
        [](writer& w) {
            w.put_string("tokenizer.ggml.tokens").put(static_cast<std::uint32_t>(value_type::array));
            w.put(static_cast<std::uint32_t>(value_type::string)).put<std::uint64_t>(3);
            w.put_string("<s>").put_string("</s>").put_string("hi");
        },
        [](writer& w) {
            w.put_string("tokenizer.ggml.scores").put(static_cast<std::uint32_t>(value_type::array));
            w.put(static_cast<std::uint32_t>(value_type::float32)).put<std::uint64_t>(3);
            w.put(0.0f).put(0.0f).put(-1.5f);
        },
    };
}

std::vector<tensor_spec> model_tensors() {
    return {
        {"token_embd.weight", {4, 3}, f32_tag, 0},
        {"blk.0.attn_q.weight", {64, 2}, q4_0_tag, 64},
    };
}

constexpr std::size_t model_data_bytes = 64 + 72;

std::vector<std::byte> model_image() {
    return build(3, model_kvs(), model_tensors(), model_data_bytes);
}

// The single-tensor image used by the corruption cases.
std::vector<std::byte> one_tensor(tensor_spec t, std::size_t data_bytes = 4096) {
    return build(3, {}, {t}, data_bytes);
}

errc parse_error(std::span<const std::byte> image) {
    const auto r = parse(image);
    REQUIRE_FALSE(r);
    return r.error().code;
}

}  // namespace

// ── round trip ─────────────────────────────────────────────────────────────

TEST_CASE("gguf_reader: header, alignment, and data offset", "[gguf]") {
    const auto image = model_image();
    const auto r = parse(image);
    REQUIRE(r);
    CHECK(r->version() == 3);
    CHECK(r->alignment() == default_alignment);
    CHECK(r->data_offset() % 32 == 0);
    CHECK(r->data_offset() + model_data_bytes == image.size());
    CHECK(r->metadata().size() == model_kvs().size());
    CHECK(r->tensors().size() == 2);
}

TEST_CASE("gguf_reader: scalar and string metadata decode to their exact types", "[gguf]") {
    const auto image = model_image();
    const auto r = parse(image);
    REQUIRE(r);
    CHECK(r->get<std::string_view>("general.architecture") == "llama");
    CHECK(r->get<std::uint32_t>("llama.context_length") == 2048u);
    CHECK(r->get<float>("llama.rope.freq_base") == 10000.0f);
    CHECK(r->get<std::uint8_t>("test.u8") == std::uint8_t{200});
    CHECK(r->get<std::int8_t>("test.i8") == std::int8_t{-5});
    CHECK(r->get<std::uint16_t>("test.u16") == std::uint16_t{60000});
    CHECK(r->get<std::int16_t>("test.i16") == std::int16_t{-30000});
    CHECK(r->get<std::int32_t>("test.i32") == -70000);
    CHECK(r->get<std::uint64_t>("test.u64") == (std::uint64_t{1} << 40));
    CHECK(r->get<std::int64_t>("test.i64") == (std::int64_t{-1} << 40));
    CHECK(r->get<double>("test.f64") == 2.5);
    CHECK(r->get<bool>("test.flag") == true);
    CHECK(type_of(*r->find("llama.context_length")) == value_type::uint32);

    // Exact-type lookup: a uint32 does not read as uint64, absent keys are empty.
    CHECK_FALSE(r->get<std::uint64_t>("llama.context_length"));
    CHECK_FALSE(r->get<std::uint32_t>("no.such.key"));
    CHECK(r->find("no.such.key") == nullptr);
}

TEST_CASE("gguf_reader: arrays decode element by element", "[gguf]") {
    const auto image = model_image();
    const auto r = parse(image);
    REQUIRE(r);

    const auto* tokens = r->get_array("tokenizer.ggml.tokens");
    REQUIRE(tokens != nullptr);
    CHECK(tokens->element_type() == value_type::string);
    REQUIRE(tokens->size() == 3);
    CHECK(tokens->at<std::string_view>(0) == "<s>");
    CHECK(tokens->at<std::string_view>(2) == "hi");
    CHECK(tokens->strings().size() == 3);
    CHECK_FALSE(tokens->at<std::string_view>(3));  // out of range
    CHECK_FALSE(tokens->at<float>(0));             // wrong element type

    const auto* scores = r->get_array("tokenizer.ggml.scores");
    REQUIRE(scores != nullptr);
    CHECK(scores->element_type() == value_type::float32);
    CHECK(scores->raw().size() == 3 * sizeof(float));
    CHECK(scores->at<float>(2) == -1.5f);

    CHECK(r->get_array("general.architecture") == nullptr);  // not an array
}

TEST_CASE("gguf_reader: tensor table and data views", "[gguf]") {
    const auto image = model_image();
    const auto r = parse(image);
    REQUIRE(r);

    const auto* embd = r->find_tensor("token_embd.weight");
    REQUIRE(embd != nullptr);
    CHECK(embd->n_dims == 2);
    CHECK(embd->ne[0] == 4);
    CHECK(embd->ne[1] == 3);
    CHECK(embd->ne[2] == 1);
    CHECK(embd->n_elements() == 12);
    CHECK(embd->type == tensor_type::f32);
    CHECK(embd->nbytes == 48);

    const auto* wq = r->find_tensor("blk.0.attn_q.weight");
    REQUIRE(wq != nullptr);
    CHECK(wq->type == tensor_type::q4_0);
    CHECK(wq->offset == 64);
    CHECK(wq->nbytes == 72);

    // Views point into the image at data_offset + offset; the writer fills the
    // data section with its byte index.
    const auto data = r->tensor_data(*wq);
    CHECK(data.size() == 72);
    CHECK(data.data() == image.data() + r->data_offset() + 64);
    CHECK(data[0] == std::byte{64});

    CHECK(r->find_tensor("missing") == nullptr);
}

TEST_CASE("gguf_reader: general.alignment moves the data section", "[gguf]") {
    const auto image =
        build(3, {kv("general.alignment", value_type::uint32, std::uint32_t{64})}, {{"w", {16}, f32_tag, 0}}, 64, 64);
    const auto r = parse(image);
    REQUIRE(r);
    CHECK(r->alignment() == 64);
    CHECK(r->data_offset() % 64 == 0);
}

TEST_CASE("gguf_reader: version 2 and a file without tensors are accepted", "[gguf]") {
    const auto image = build(2, {kv_string("general.name", "empty")}, {}, 0);
    const auto r = parse(image);
    REQUIRE(r);
    CHECK(r->version() == 2);
    CHECK(r->tensors().empty());
}

// ── rejections ─────────────────────────────────────────────────────────────

TEST_CASE("gguf_reader: rejects bad magic and unsupported versions", "[gguf]") {
    auto image = model_image();
    image[0] = std::byte{'X'};
    CHECK(parse_error(image) == errc::bad_magic);

    CHECK(parse_error(build(1, {}, {}, 0)) == errc::unsupported_version);
    CHECK(parse_error(build(4, {}, {}, 0)) == errc::unsupported_version);
    // A big-endian v3 file reads as version 0x03000000.
    CHECK(parse_error(build(0x03000000, {}, {}, 0)) == errc::unsupported_version);
}

TEST_CASE("gguf_reader: every truncation of a valid image is rejected", "[gguf]") {
    const auto image = model_image();
    for (std::size_t n = 0; n < image.size(); ++n) {
        const auto r = parse(std::span(image).first(n));
        INFO("prefix length " << n);
        REQUIRE_FALSE(r);
        CHECK(r.error().offset <= n);
        CHECK((r.error().code == errc::truncated || r.error().code == errc::tensor_out_of_bounds));
    }
}

TEST_CASE("gguf_reader: rejects counts larger than the image can hold", "[gguf]") {
    writer w;
    w.put<std::uint32_t>(0x46554747).put<std::uint32_t>(3);
    w.put<std::uint64_t>(0).put<std::uint64_t>(std::uint64_t{1} << 60);  // 2^60 key/value pairs
    CHECK(parse_error(w.bytes()) == errc::truncated);

    const auto huge_array = [](writer& out) {
        out.put_string("a").put(static_cast<std::uint32_t>(value_type::array));
        out.put(static_cast<std::uint32_t>(value_type::uint64)).put<std::uint64_t>(std::uint64_t{1} << 61);
    };
    CHECK(parse_error(build(3, {huge_array}, {}, 0)) == errc::truncated);
}

TEST_CASE("gguf_reader: rejects malformed metadata", "[gguf]") {
    const auto dup = kv("k", value_type::uint8, std::uint8_t{1});
    CHECK(parse_error(build(3, {dup, dup}, {}, 0)) == errc::duplicate_key);

    CHECK(parse_error(build(3, {kv("k", static_cast<value_type>(13), std::uint8_t{0})}, {}, 0)) ==
          errc::bad_value_type);

    const auto nested = [](writer& w) {
        w.put_string("k").put(static_cast<std::uint32_t>(value_type::array));
        w.put(static_cast<std::uint32_t>(value_type::array)).put<std::uint64_t>(1);
    };
    CHECK(parse_error(build(3, {nested}, {}, 0)) == errc::nested_array);

    const auto bad_elem = [](writer& w) {
        w.put_string("k").put(static_cast<std::uint32_t>(value_type::array));
        w.put<std::uint32_t>(99).put<std::uint64_t>(0);
    };
    CHECK(parse_error(build(3, {bad_elem}, {}, 0)) == errc::bad_value_type);
}

TEST_CASE("gguf_reader: rejects a bad general.alignment", "[gguf]") {
    CHECK(parse_error(build(3, {kv("general.alignment", value_type::uint32, std::uint32_t{48})}, {}, 0)) ==
          errc::bad_alignment);
    CHECK(parse_error(build(3, {kv("general.alignment", value_type::uint32, std::uint32_t{0})}, {}, 0)) ==
          errc::bad_alignment);
    CHECK(parse_error(build(3, {kv("general.alignment", value_type::uint64, std::uint64_t{32})}, {}, 0)) ==
          errc::bad_alignment);
}

TEST_CASE("gguf_reader: rejects invalid tensor records", "[gguf]") {
    CHECK(parse_error(one_tensor({"t", {}, f32_tag, 0})) == errc::bad_dims);
    CHECK(parse_error(one_tensor({"t", {1, 1, 1, 1, 1}, f32_tag, 0})) == errc::bad_dims);
    CHECK(parse_error(one_tensor({"t", {4, 0}, f32_tag, 0})) == errc::bad_dims);
    CHECK(parse_error(one_tensor({"t", {std::uint64_t{1} << 32, std::uint64_t{1} << 32}, f32_tag, 0})) ==
          errc::bad_dims);
    CHECK(parse_error(one_tensor({"t", {256}, 16, 0})) == errc::unsupported_tensor_type);  // IQ2_XXS
    CHECK(parse_error(one_tensor({"t", {33}, q4_0_tag, 0})) == errc::partial_block);
    CHECK(parse_error(one_tensor({"t", {4}, f32_tag, 8})) == errc::misaligned_tensor);
    CHECK(parse_error(one_tensor({"t", {4}, f32_tag, 0}, 15)) == errc::tensor_out_of_bounds);
    CHECK(parse_error(one_tensor({"t", {4}, f32_tag, std::uint64_t{1} << 63})) == errc::tensor_out_of_bounds);

    CHECK(parse_error(build(3, {}, {{"t", {4}, f32_tag, 0}, {"t", {4}, f32_tag, 32}}, 64)) == errc::duplicate_tensor);
}

TEST_CASE("gguf_reader: errors report the offset of the rejected record", "[gguf]") {
    // Header is 24 bytes; the second tensor record starts after the first
    // (8 + 1 name, 4 n_dims, 8 dim, 4 type, 8 offset = 33 bytes).
    const auto image = build(3, {}, {{"a", {4}, f32_tag, 0}, {"b", {4}, f32_tag, 8}}, 64);
    const auto r = parse(image);
    REQUIRE_FALSE(r);
    CHECK(r.error().code == errc::misaligned_tensor);
    CHECK(r.error().offset == 24 + 33);
    CHECK(to_string(r.error().code) == "tensor offset is not aligned");
}

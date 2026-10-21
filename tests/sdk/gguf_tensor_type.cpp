// Unit tests for branes::synapse::gguf::tensor_type — the on-disk tags and
// block layouts must match ggml bit for bit, or every tensor offset computed
// from a GGUF file is wrong.

#include <branes/synapse/gguf/tensor_type.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

using branes::synapse::gguf::row_size;
using branes::synapse::gguf::tensor_type;
using branes::synapse::gguf::tensor_type_traits;

TEST_CASE("gguf: tensor_type tags match the ggml on-disk values", "[gguf]") {
    STATIC_REQUIRE(static_cast<std::uint32_t>(tensor_type::f32) == 0);
    STATIC_REQUIRE(static_cast<std::uint32_t>(tensor_type::f16) == 1);
    STATIC_REQUIRE(static_cast<std::uint32_t>(tensor_type::q4_0) == 2);
    STATIC_REQUIRE(static_cast<std::uint32_t>(tensor_type::q8_0) == 8);
    STATIC_REQUIRE(static_cast<std::uint32_t>(tensor_type::q4_k) == 12);
    STATIC_REQUIRE(static_cast<std::uint32_t>(tensor_type::q6_k) == 14);
    STATIC_REQUIRE(static_cast<std::uint32_t>(tensor_type::bf16) == 30);
}

TEST_CASE("gguf: plain types have one element per block", "[gguf]") {
    for (auto t : {tensor_type::f32, tensor_type::f16, tensor_type::bf16, tensor_type::i32, tensor_type::f64}) {
        const auto info = tensor_type_traits(t);
        REQUIRE(info);
        CHECK(info->block_size == 1);
        CHECK_FALSE(info->quantized);
    }
}

TEST_CASE("gguf: K-quants use 256-element super-blocks", "[gguf]") {
    STATIC_REQUIRE(tensor_type_traits(tensor_type::q4_k)->block_size == 256);
    STATIC_REQUIRE(tensor_type_traits(tensor_type::q4_k)->type_size == 144);
    STATIC_REQUIRE(tensor_type_traits(tensor_type::q6_k)->type_size == 210);
}

TEST_CASE("gguf: row_size counts whole blocks", "[gguf]") {
    // A 4096-wide Llama row: 128 Q4_0 blocks of 18 bytes, 16 Q4_K blocks of 144.
    CHECK(row_size(tensor_type::q4_0, 4096) == 128u * 18u);
    CHECK(row_size(tensor_type::q4_k, 4096) == 16u * 144u);
    CHECK(row_size(tensor_type::f16, 4096) == 8192u);
}

TEST_CASE("gguf: row_size rejects partial blocks and unsupported tags", "[gguf]") {
    CHECK_FALSE(row_size(tensor_type::q4_k, 100));
    CHECK_FALSE(row_size(static_cast<tensor_type>(16), 256));      // IQ2_XXS
    CHECK_FALSE(tensor_type_traits(static_cast<tensor_type>(4)));  // retired tag
}

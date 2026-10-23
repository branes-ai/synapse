#pragma once
/// @file scheme.hpp
/// @brief Compile-time descriptions of tensor storage formats.
///
/// `scheme<T>` turns a runtime GGUF tensor_type tag into a type the compiler
/// can dispatch on: its block struct, block geometry, and a dequantizer that
/// reads one encoded block from raw bytes. Plain formats (F32, F16, BF16) are
/// schemes with one element per block, so kernels and views treat every
/// format uniformly.
///
/// `vec_dot_block` names the activation format a scheme's fused dot product
/// consumes (roadmap step 3): Q8_0 blocks for the legacy formats, Q8_K blocks
/// for the K formats, and plain float for F32/F16/BF16. It matches ggml's
/// vec_dot_type, except that plain formats keep float activations where ggml
/// rounds them to the weight format.
///
/// Every scheme is also an MTL5 BlockScheme (`block_bytes`, `decode`), so
/// mtl::mat::block_matrix_view<S> is the matrix view over its bytes.
///
/// `scheme<T>` is only defined for the formats synapse can decode; `has_scheme`
/// tests that. Each specialization is checked against tensor_type_traits, so
/// the block geometry the parser validated and the geometry the decoder
/// assumes cannot drift apart.

#include <branes/synapse/gguf/tensor_type.hpp>
#include <branes/synapse/numeric/half.hpp>
#include <branes/synapse/quant/blocks.hpp>
#include <branes/synapse/quant/q8.hpp>

#include <mtl/concepts/block_scheme.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace branes::synapse::quant {

using gguf::tensor_type;

template <tensor_type T>
struct scheme;  // defined only for decodable formats

namespace detail {

/// Copy one block out of (possibly unaligned) storage. memcpy rather than a
/// pointer cast: GGUF buffers carry no C++ objects, and block alignment is
/// not guaranteed for arbitrary images.
template <typename Block>
Block load_block(const std::byte* src) noexcept {
    Block b;
    std::memcpy(&b, src, sizeof(Block));
    return b;
}

template <typename Block, tensor_type T, std::size_t BlockSize, typename VecDotBlock>
struct block_scheme {
    using block_type = Block;
    using vec_dot_block = VecDotBlock;
    static constexpr tensor_type type = T;
    static constexpr std::size_t block_size = BlockSize;
    static constexpr std::size_t type_size = sizeof(Block);
    static constexpr std::size_t block_bytes = type_size;

    static void dequantize(const std::byte* src, float* dst) noexcept {
        quant::dequantize(load_block<Block>(src), dst);
    }
    static void decode(const std::byte* src, float* dst) noexcept {
        dequantize(src, dst);
    }
};

}  // namespace detail

template <>
struct scheme<tensor_type::f32> {
    using vec_dot_block = float;
    using block_type = float;
    static constexpr tensor_type type = tensor_type::f32;
    static constexpr std::size_t block_size = 1;
    static constexpr std::size_t type_size = sizeof(float);
    static constexpr std::size_t block_bytes = type_size;

    static void dequantize(const std::byte* src, float* dst) noexcept {
        std::memcpy(dst, src, sizeof(float));
    }
    static void decode(const std::byte* src, float* dst) noexcept {
        dequantize(src, dst);
    }
};

template <>
struct scheme<tensor_type::f16> {
    using vec_dot_block = float;
    using block_type = std::uint16_t;  ///< binary16 bits
    static constexpr tensor_type type = tensor_type::f16;
    static constexpr std::size_t block_size = 1;
    static constexpr std::size_t type_size = sizeof(std::uint16_t);
    static constexpr std::size_t block_bytes = type_size;

    static void dequantize(const std::byte* src, float* dst) noexcept {
        *dst = fp16_to_float(detail::load_block<std::uint16_t>(src));
    }
    static void decode(const std::byte* src, float* dst) noexcept {
        dequantize(src, dst);
    }
};

template <>
struct scheme<tensor_type::bf16> {
    using vec_dot_block = float;
    using block_type = std::uint16_t;  ///< bfloat16 bits
    static constexpr tensor_type type = tensor_type::bf16;
    static constexpr std::size_t block_size = 1;
    static constexpr std::size_t type_size = sizeof(std::uint16_t);
    static constexpr std::size_t block_bytes = type_size;

    static void dequantize(const std::byte* src, float* dst) noexcept {
        *dst = bf16_to_float(detail::load_block<std::uint16_t>(src));
    }
    static void decode(const std::byte* src, float* dst) noexcept {
        dequantize(src, dst);
    }
};

template <>
struct scheme<tensor_type::q4_0> : detail::block_scheme<block_q4_0, tensor_type::q4_0, qk_legacy, block_q8_0> {};
template <>
struct scheme<tensor_type::q8_0> : detail::block_scheme<block_q8_0, tensor_type::q8_0, qk_legacy, block_q8_0> {};
template <>
struct scheme<tensor_type::q4_k> : detail::block_scheme<block_q4_k, tensor_type::q4_k, qk_k, block_q8_k> {};
template <>
struct scheme<tensor_type::q6_k> : detail::block_scheme<block_q6_k, tensor_type::q6_k, qk_k, block_q8_k> {};

/// A storage format: block type, geometry, and a block dequantizer. It
/// refines mtl::BlockScheme, with block_bytes == type_size and decode the
/// same function as dequantize.
template <typename S>
concept Scheme = mtl::BlockScheme<S> && requires(const std::byte* src, float* dst) {
    typename S::block_type;
    typename S::vec_dot_block;
    { S::type } -> std::convertible_to<tensor_type>;
    { S::block_size } -> std::convertible_to<std::size_t>;
    { S::type_size } -> std::convertible_to<std::size_t>;
    S::dequantize(src, dst);
} && sizeof(typename S::block_type) == S::type_size && S::block_bytes == S::type_size && S::block_size > 0;

/// True if synapse can decode tensors of type T.
template <tensor_type T>
inline constexpr bool has_scheme = requires { typename scheme<T>::block_type; };

/// The scheme's geometry must agree with the table the GGUF parser validates against.
template <tensor_type T>
constexpr bool matches_tensor_type_traits() {
    using S = scheme<T>;
    return Scheme<S> && S::type == T && gguf::tensor_type_traits(T)->block_size == S::block_size &&
           gguf::tensor_type_traits(T)->type_size == S::type_size;
}

static_assert(matches_tensor_type_traits<tensor_type::f32>());
static_assert(matches_tensor_type_traits<tensor_type::f16>());
static_assert(matches_tensor_type_traits<tensor_type::bf16>());
static_assert(matches_tensor_type_traits<tensor_type::q4_0>());
static_assert(matches_tensor_type_traits<tensor_type::q8_0>());
static_assert(matches_tensor_type_traits<tensor_type::q4_k>());
static_assert(matches_tensor_type_traits<tensor_type::q6_k>());

}  // namespace branes::synapse::quant

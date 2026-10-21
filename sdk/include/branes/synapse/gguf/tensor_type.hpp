#pragma once
/// @file tensor_type.hpp
/// @brief GGML tensor element types as they appear in GGUF tensor-info records.
///
/// Every tensor in a GGUF file carries a 32-bit `ggml_type` tag. For plain
/// types (F32, F16, I8, ...) an element is one value; for the block-quantized
/// types (Q4_0, Q4_K, ...) values are stored in fixed-size blocks of
/// `block_size` elements that occupy `type_size` bytes each, scales included.
/// A row of `n` elements therefore takes `n / block_size * type_size` bytes,
/// and `n` must be a multiple of `block_size`.
///
/// The enumerator values are the on-disk tags and must match ggml exactly;
/// gaps are tags ggml has retired. Only the types synapse plans to support are
/// listed. The importance-matrix (IQ*), ternary (TQ*), and MXFP4 types are
/// recognized as "not supported" by `tensor_type_traits` returning nullopt.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace branes::synapse::gguf {

/// On-disk GGML element-type tag.
enum class tensor_type : std::uint32_t {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    bf16 = 30,
};

/// Storage layout of one tensor_type.
struct tensor_type_info {
    std::string_view name;   ///< ggml's spelling, e.g. "q4_K"
    std::size_t block_size;  ///< elements per block (1 for plain types)
    std::size_t type_size;   ///< bytes per block
    bool quantized;          ///< block-quantized (carries per-block scales)
};

/// Layout of `t`, or nullopt for a tag synapse does not support.
constexpr std::optional<tensor_type_info> tensor_type_traits(tensor_type t) noexcept {
    // Block sizes: 32 for the legacy quants (QK4_0 .. QK8_1), 256 for the
    // K-quants (QK_K). Byte sizes are sizeof(block_*) in ggml-common.h.
    switch (t) {
    case tensor_type::f32:
        return tensor_type_info{"f32", 1, 4, false};
    case tensor_type::f16:
        return tensor_type_info{"f16", 1, 2, false};
    case tensor_type::q4_0:
        return tensor_type_info{"q4_0", 32, 18, true};
    case tensor_type::q4_1:
        return tensor_type_info{"q4_1", 32, 20, true};
    case tensor_type::q5_0:
        return tensor_type_info{"q5_0", 32, 22, true};
    case tensor_type::q5_1:
        return tensor_type_info{"q5_1", 32, 24, true};
    case tensor_type::q8_0:
        return tensor_type_info{"q8_0", 32, 34, true};
    case tensor_type::q8_1:
        return tensor_type_info{"q8_1", 32, 36, true};
    case tensor_type::q2_k:
        return tensor_type_info{"q2_K", 256, 84, true};
    case tensor_type::q3_k:
        return tensor_type_info{"q3_K", 256, 110, true};
    case tensor_type::q4_k:
        return tensor_type_info{"q4_K", 256, 144, true};
    case tensor_type::q5_k:
        return tensor_type_info{"q5_K", 256, 176, true};
    case tensor_type::q6_k:
        return tensor_type_info{"q6_K", 256, 210, true};
    case tensor_type::q8_k:
        return tensor_type_info{"q8_K", 256, 292, true};
    case tensor_type::i8:
        return tensor_type_info{"i8", 1, 1, false};
    case tensor_type::i16:
        return tensor_type_info{"i16", 1, 2, false};
    case tensor_type::i32:
        return tensor_type_info{"i32", 1, 4, false};
    case tensor_type::i64:
        return tensor_type_info{"i64", 1, 8, false};
    case tensor_type::f64:
        return tensor_type_info{"f64", 1, 8, false};
    case tensor_type::bf16:
        return tensor_type_info{"bf16", 1, 2, false};
    }
    return std::nullopt;
}

/// Bytes occupied by a row of `n` elements of type `t`, or nullopt if `t` is
/// unsupported or `n` is not a whole number of blocks.
constexpr std::optional<std::size_t> row_size(tensor_type t, std::size_t n) noexcept {
    const auto info = tensor_type_traits(t);
    if (!info || n % info->block_size != 0) {
        return std::nullopt;
    }
    return n / info->block_size * info->type_size;
}

}  // namespace branes::synapse::gguf

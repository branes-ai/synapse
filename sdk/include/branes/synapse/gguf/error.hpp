#pragma once
/// @file error.hpp
/// @brief Error reporting for the GGUF loader: error codes and a no-throw result.
///
/// Loading a model is the one place synapse touches untrusted input, and it
/// often runs on targets built with exceptions disabled. Every loader entry
/// point therefore returns a `result<T>` holding either the value or an
/// `error_info` describing what went wrong and where. Nothing in the loader
/// throws, except std::bad_alloc from the standard containers it fills.

#include <branes/synapse/result.hpp>

#include <cstdint>
#include <string_view>

namespace branes::synapse::gguf {

/// Why a GGUF image (or the file holding it) was rejected.
enum class errc : std::uint32_t {
    io_error = 1,             ///< the file could not be opened, sized, or mapped
    truncated,                ///< the image ends inside a record, or a count exceeds the bytes left
    bad_magic,                ///< the image does not start with "GGUF"
    unsupported_version,      ///< not GGUF v2 or v3 (v1 and big-endian files land here)
    bad_value_type,           ///< a metadata value-type tag outside the GGUF set
    nested_array,             ///< an array of arrays (rejected, as by ggml)
    duplicate_key,            ///< a metadata key appears twice
    bad_alignment,            ///< general.alignment is not a uint32 power of two
    bad_dims,                 ///< n_dims outside [1, 4], a zero dimension, or too many elements
    unsupported_tensor_type,  ///< a tensor type synapse does not support
    partial_block,            ///< ne[0] is not a whole number of quantization blocks
    duplicate_tensor,         ///< a tensor name appears twice
    misaligned_tensor,        ///< a tensor offset is not a multiple of the alignment
    tensor_out_of_bounds,     ///< a tensor's data extends past the end of the image
};

/// Human-readable description of an error code.
constexpr std::string_view to_string(errc code) noexcept {
    switch (code) {
    case errc::io_error:
        return "I/O error";
    case errc::truncated:
        return "truncated image";
    case errc::bad_magic:
        return "not a GGUF file (bad magic)";
    case errc::unsupported_version:
        return "unsupported GGUF version";
    case errc::bad_value_type:
        return "invalid metadata value type";
    case errc::nested_array:
        return "nested metadata arrays are not supported";
    case errc::duplicate_key:
        return "duplicate metadata key";
    case errc::bad_alignment:
        return "general.alignment must be a uint32 power of two";
    case errc::bad_dims:
        return "invalid tensor dimensions";
    case errc::unsupported_tensor_type:
        return "unsupported tensor type";
    case errc::partial_block:
        return "tensor row is not a whole number of blocks";
    case errc::duplicate_tensor:
        return "duplicate tensor name";
    case errc::misaligned_tensor:
        return "tensor offset is not aligned";
    case errc::tensor_out_of_bounds:
        return "tensor data extends past the end of the file";
    }
    return "unknown error";
}

/// What went wrong and where.
struct error_info {
    errc code;
    std::uint64_t offset = 0;  ///< byte offset in the image of the record that failed
    int os_code = 0;           ///< errno / GetLastError() for errc::io_error, else 0
};

/// A loader result: either a T or an error_info.
template <typename T>
using result = synapse::result<T, error_info>;

}  // namespace branes::synapse::gguf

#pragma once
/// @file metadata.hpp
/// @brief GGUF metadata values: the typed key/value pairs in a GGUF header.
///
/// A GGUF header carries model configuration as key/value pairs such as
/// `general.architecture = "llama"` or `llama.context_length = 8192u32`, plus
/// large arrays (the tokenizer vocabulary and scores). Values are views into
/// the GGUF image: strings are std::string_view and arrays keep a span over
/// their raw bytes, so nothing is copied and nothing outlives the image.
///
/// The alternatives of metadata_value are listed in GGUF value-type order, so
/// `value.index()` equals the on-disk `value_type` tag.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace branes::synapse::gguf {

namespace detail {
class parser;
}

/// On-disk GGUF metadata value-type tag.
enum class value_type : std::uint32_t {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    boolean = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
};

/// True if `t` is one of the GGUF value types.
constexpr bool is_valid(value_type t) noexcept {
    return static_cast<std::uint32_t>(t) <= static_cast<std::uint32_t>(value_type::float64);
}

/// Encoded size in bytes of a fixed-size value type; 0 for string, array, and
/// invalid tags.
constexpr std::size_t fixed_size(value_type t) noexcept {
    switch (t) {
    case value_type::uint8:
    case value_type::int8:
    case value_type::boolean:
        return 1;
    case value_type::uint16:
    case value_type::int16:
        return 2;
    case value_type::uint32:
    case value_type::int32:
    case value_type::float32:
        return 4;
    case value_type::uint64:
    case value_type::int64:
    case value_type::float64:
        return 8;
    case value_type::string:
    case value_type::array:
        return 0;
    }
    return 0;
}

/// The GGUF value type that a C++ element type decodes from.
template <typename T>
constexpr value_type value_type_of() noexcept {
    if constexpr (std::is_same_v<T, std::uint8_t>) {
        return value_type::uint8;
    } else if constexpr (std::is_same_v<T, std::int8_t>) {
        return value_type::int8;
    } else if constexpr (std::is_same_v<T, std::uint16_t>) {
        return value_type::uint16;
    } else if constexpr (std::is_same_v<T, std::int16_t>) {
        return value_type::int16;
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
        return value_type::uint32;
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
        return value_type::int32;
    } else if constexpr (std::is_same_v<T, float>) {
        return value_type::float32;
    } else if constexpr (std::is_same_v<T, bool>) {
        return value_type::boolean;
    } else if constexpr (std::is_same_v<T, std::string_view>) {
        return value_type::string;
    } else if constexpr (std::is_same_v<T, std::uint64_t>) {
        return value_type::uint64;
    } else if constexpr (std::is_same_v<T, std::int64_t>) {
        return value_type::int64;
    } else {
        static_assert(std::is_same_v<T, double>, "not a GGUF scalar type");
        return value_type::float64;
    }
}

/// A GGUF metadata array (one level deep: nested arrays are rejected).
///
/// Model
/// -----
///   array_value
///     element_type_ : value_type — any type but array
///     count_        : uint64 — number of elements
///     raw_          : span<const byte> [non-owning, into the image] — the
///                     encoded elements, for fixed-size element types
///     strings_      : vector<string_view> [views into the image] — the
///                     elements, for string arrays
///
/// Invariants
/// ----------
///   I1: element_type_ is valid and != array            — constrains element_type_
///   I2: fixed-size elements: raw_.size() == count_ * fixed_size(element_type_)
///       and strings_ is empty                          — constrains raw_, strings_
///   I3: string elements: strings_.size() == count_ and raw_ is empty
///                                                      — constrains raw_, strings_
///
/// Only the parser constructs arrays, and it establishes I1-I3.
class array_value {
public:
    value_type element_type() const noexcept {
        return element_type_;
    }
    std::uint64_t size() const noexcept {
        return count_;
    }

    /// Element `i` decoded as T, or nullopt if `i` is out of range or T does
    /// not match the element type exactly. Elements are copied out with
    /// memcpy: GGUF arrays are not aligned for their element type.
    template <typename T>
    std::optional<T> at(std::uint64_t i) const noexcept {
        if (element_type_ != value_type_of<T>() || i >= count_) {
            return std::nullopt;
        }
        if constexpr (std::is_same_v<T, std::string_view>) {
            return strings_[static_cast<std::size_t>(i)];  // I3
        } else if constexpr (std::is_same_v<T, bool>) {
            return raw_[static_cast<std::size_t>(i)] != std::byte{0};  // I2
        } else {
            T v;
            std::memcpy(&v, raw_.data() + static_cast<std::size_t>(i) * sizeof(T), sizeof(T));  // I2
            return v;
        }
    }

    /// The elements of a string array (empty for other element types).
    std::span<const std::string_view> strings() const noexcept {
        return strings_;
    }

    /// The encoded elements of a fixed-size array (empty for string arrays).
    std::span<const std::byte> raw() const noexcept {
        return raw_;
    }

private:
    friend class detail::parser;

    array_value(value_type element_type,
                std::uint64_t count,
                std::span<const std::byte> raw,
                std::vector<std::string_view> strings)
        : element_type_(element_type), count_(count), raw_(raw), strings_(std::move(strings)) {
        assert(is_valid(element_type_) && element_type_ != value_type::array);  // I1
        assert(element_type_ == value_type::string
                   ? strings_.size() == count_ && raw_.empty()
                   : raw_.size() == count_ * fixed_size(element_type_) && strings_.empty());  // I2, I3
    }

    value_type element_type_;
    std::uint64_t count_;
    std::span<const std::byte> raw_;
    std::vector<std::string_view> strings_;
};

/// One metadata value. Alternative i holds value_type i.
using metadata_value = std::variant<std::uint8_t,
                                    std::int8_t,
                                    std::uint16_t,
                                    std::int16_t,
                                    std::uint32_t,
                                    std::int32_t,
                                    float,
                                    bool,
                                    std::string_view,
                                    array_value,
                                    std::uint64_t,
                                    std::int64_t,
                                    double>;

/// The on-disk type of a metadata value.
inline value_type type_of(const metadata_value& v) noexcept {
    return static_cast<value_type>(v.index());
}

/// One key/value pair, in file order.
struct metadata_entry {
    std::string_view key;
    metadata_value value;
};

}  // namespace branes::synapse::gguf

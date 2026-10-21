#pragma once
/// @file reader.hpp
/// @brief GGUF parser: validates a GGUF image and exposes its metadata and tensor table.
///
/// GGUF layout (v2/v3, little-endian):
///
///   magic "GGUF" | version u32 | tensor_count u64 | metadata_kv_count u64
///   metadata_kv_count x { key: string, type: u32, value }
///   tensor_count      x { name: string, n_dims: u32, ne: u64[n_dims],
///                         type: u32, offset: u64 }
///   padding to general.alignment (default 32)
///   tensor data; each tensor at data_offset + offset
///
///   string = { len: u64, bytes: u8[len] } (UTF-8, not NUL-terminated)
///
/// `parse()` works on an in-memory image (typically a mapped_file) and copies
/// nothing: names, strings, and tensor data are views into the image.
///
/// Model
/// -----
///   gguf_file
///     image_       : span<const byte> [non-owning] — the whole GGUF image
///     version_     : uint32 — GGUF version
///     alignment_   : uint64 — data alignment (general.alignment, default 32)
///     metadata_    : vector<metadata_entry> — key/value pairs in file order
///     tensors_     : vector<tensor_info> — tensor table in file order
///     data_offset_ : uint64 — start of the tensor data section in image_
///
/// Invariants
/// ----------
///   I1: version_ is 2 or 3                                — constrains version_
///   I2: alignment_ is a power of two and
///       data_offset_ % alignment_ == 0                    — constrains alignment_, data_offset_
///   I3: metadata keys are unique; tensor names are unique — constrains metadata_, tensors_
///   I4: every tensor has 1 <= n_dims <= 4, all ne >= 1 (unused ne == 1),
///       a supported type, and ne[0] a whole number of blocks
///                                                         — constrains tensors_
///   I5: every tensor has offset % alignment_ == 0 and
///       data_offset_ + offset + nbytes <= image_.size()   — constrains tensors_, data_offset_
///   I6: every string_view and span held lies inside image_ — constrains metadata_, tensors_
///   I7: no metadata array has array elements              — constrains metadata_
///
/// gguf_file is immutable and only parse() constructs it, so I1-I7 hold for
/// its whole lifetime. Lifetime: the views are only valid while the image is;
/// keep the mapped_file (or buffer) alive as long as the gguf_file.

#include <branes/synapse/gguf/error.hpp>
#include <branes/synapse/gguf/metadata.hpp>
#include <branes/synapse/gguf/tensor_type.hpp>

#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace branes::synapse::gguf {

static_assert(std::endian::native == std::endian::little, "the GGUF reader assumes a little-endian host");

/// Maximum tensor rank (GGML_MAX_DIMS).
inline constexpr std::uint32_t max_dims = 4;

/// Default data alignment when general.alignment is absent (GGUF_DEFAULT_ALIGNMENT).
inline constexpr std::uint64_t default_alignment = 32;

/// One entry of the tensor table.
struct tensor_info {
    std::string_view name;
    std::uint32_t n_dims = 0;
    std::array<std::uint64_t, max_dims> ne{1, 1, 1, 1};  ///< ne[0] innermost; unused dims are 1
    tensor_type type = tensor_type::f32;
    std::uint64_t offset = 0;  ///< relative to the data section
    std::uint64_t nbytes = 0;  ///< size of the tensor's data

    std::uint64_t n_elements() const noexcept {
        return ne[0] * ne[1] * ne[2] * ne[3];
    }
};

class gguf_file;

/// Parse and validate a GGUF image. The image is untrusted: every read is
/// bounds-checked and every count is checked against the bytes remaining
/// before anything is allocated.
///
///   Pre:  none.
///   Post: on success, a gguf_file satisfying I1-I7 whose views point into
///         `image`; on failure, the error code and the byte offset of the
///         record that was rejected.
result<gguf_file> parse(std::span<const std::byte> image);

class gguf_file {
public:
    std::uint32_t version() const noexcept {
        return version_;
    }
    std::uint64_t alignment() const noexcept {
        return alignment_;
    }
    std::uint64_t data_offset() const noexcept {
        return data_offset_;
    }
    std::span<const std::byte> image() const noexcept {
        return image_;
    }
    std::span<const metadata_entry> metadata() const noexcept {
        return metadata_;
    }
    std::span<const tensor_info> tensors() const noexcept {
        return tensors_;
    }

    /// The value stored under `key`, or nullptr.
    const metadata_value* find(std::string_view key) const noexcept {
        for (const auto& entry : metadata_) {
            if (entry.key == key) {
                return &entry.value;
            }
        }
        return nullptr;
    }

    /// The scalar or string stored under `key`, or nullopt if absent or not
    /// stored as exactly T (a uint32 key does not read as uint64).
    template <typename T>
        requires(!std::is_same_v<T, array_value>)
    std::optional<T> get(std::string_view key) const noexcept {
        const auto* v = find(key);
        if (v == nullptr) {
            return std::nullopt;
        }
        if (const T* p = std::get_if<T>(v)) {
            return *p;
        }
        return std::nullopt;
    }

    /// The array stored under `key`, or nullptr if absent or not an array.
    const array_value* get_array(std::string_view key) const noexcept {
        const auto* v = find(key);
        return v == nullptr ? nullptr : std::get_if<array_value>(v);
    }

    /// The tensor named `name`, or nullptr.
    const tensor_info* find_tensor(std::string_view name) const noexcept {
        for (const auto& t : tensors_) {
            if (t.name == name) {
                return &t;
            }
        }
        return nullptr;
    }

    /// The bytes of tensor `t`.
    ///
    ///   Pre:  t is an element of tensors().
    ///   Post: the span lies inside image() (I5).
    std::span<const std::byte> tensor_data(const tensor_info& t) const noexcept {
        assert(&t >= tensors_.data() && &t < tensors_.data() + tensors_.size());  // Pre
        return image_.subspan(static_cast<std::size_t>(data_offset_ + t.offset), static_cast<std::size_t>(t.nbytes));
    }

private:
    friend class detail::parser;

    gguf_file() = default;

    /// Runtime check of I1, I2, I4, I5 (I3, I6, I7 are enforced while parsing).
    bool invariant_holds() const noexcept {
        if (version_ != 2 && version_ != 3) {
            return false;  // I1
        }
        if (!std::has_single_bit(alignment_) || data_offset_ % alignment_ != 0) {
            return false;  // I2
        }
        for (const auto& t : tensors_) {
            if (t.n_dims < 1 || t.n_dims > max_dims || t.offset % alignment_ != 0) {
                return false;  // I4, I5
            }
            if (data_offset_ + t.offset + t.nbytes > image_.size()) {
                return false;  // I5
            }
        }
        return true;
    }

    std::span<const std::byte> image_;
    std::uint32_t version_ = 0;
    std::uint64_t alignment_ = default_alignment;
    std::vector<metadata_entry> metadata_;
    std::vector<tensor_info> tensors_;
    std::uint64_t data_offset_ = 0;
};

namespace detail {

/// Bounds-checked little-endian reader over the image.
class cursor {
public:
    explicit cursor(std::span<const std::byte> image) noexcept : image_(image) {}

    std::size_t pos() const noexcept {
        return pos_;
    }
    std::size_t remaining() const noexcept {
        return image_.size() - pos_;
    }

    template <typename T>
    bool read(T& out) noexcept {
        static_assert(std::is_trivially_copyable_v<T>);
        if (remaining() < sizeof(T)) {
            return false;
        }
        std::memcpy(&out, image_.data() + pos_, sizeof(T));
        pos_ += sizeof(T);
        return true;
    }

    bool read_bytes(std::uint64_t n, std::span<const std::byte>& out) noexcept {
        if (n > remaining()) {
            return false;
        }
        out = image_.subspan(pos_, static_cast<std::size_t>(n));
        pos_ += static_cast<std::size_t>(n);
        return true;
    }

    bool read_string(std::string_view& out) noexcept {
        std::uint64_t len = 0;
        std::span<const std::byte> bytes;
        if (!read(len) || !read_bytes(len, bytes)) {
            return false;
        }
        out = std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        return true;
    }

private:
    std::span<const std::byte> image_;
    std::size_t pos_ = 0;
};

/// a * b, or nullopt on uint64 overflow.
constexpr std::optional<std::uint64_t> checked_mul(std::uint64_t a, std::uint64_t b) noexcept {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        return std::nullopt;
    }
    return a * b;
}

class parser {
public:
    static result<gguf_file> run(std::span<const std::byte> image) {
        // Smallest encodings, used to reject counts the image cannot hold
        // before reserving space for them.
        constexpr std::size_t min_kv_bytes = 8 + 4 + 1;              // empty key, type, 1-byte value
        constexpr std::size_t min_tensor_bytes = 8 + 4 + 8 + 4 + 8;  // empty name, n_dims, 1 dim, type, offset

        cursor in{image};
        gguf_file file;
        file.image_ = image;

        std::array<std::byte, 4> magic{};
        if (!in.read(magic)) {
            return fail(errc::truncated, 0);
        }
        if (std::memcmp(magic.data(), "GGUF", 4) != 0) {
            return fail(errc::bad_magic, 0);
        }
        if (!in.read(file.version_)) {
            return fail(errc::truncated, in.pos());
        }
        if (file.version_ != 2 && file.version_ != 3) {
            return fail(errc::unsupported_version, 4);
        }
        std::uint64_t tensor_count = 0;
        std::uint64_t kv_count = 0;
        if (!in.read(tensor_count) || !in.read(kv_count)) {
            return fail(errc::truncated, in.pos());
        }
        if (kv_count > in.remaining() / min_kv_bytes || tensor_count > in.remaining() / min_tensor_bytes) {
            return fail(errc::truncated, in.pos());
        }

        // ── metadata ────────────────────────────────────────────────
        file.metadata_.reserve(static_cast<std::size_t>(kv_count));
        std::unordered_set<std::string_view> keys;
        for (std::uint64_t k = 0; k < kv_count; ++k) {
            const std::size_t start = in.pos();
            std::string_view key;
            std::uint32_t type = 0;
            if (!in.read_string(key) || !in.read(type)) {
                return fail(errc::truncated, start);
            }
            if (!keys.insert(key).second) {
                return fail(errc::duplicate_key, start);  // I3
            }
            std::optional<metadata_value> value;
            if (auto err = read_value(in, static_cast<value_type>(type), true, value)) {
                return *err;
            }
            if (key == "general.alignment") {
                const auto* a = std::get_if<std::uint32_t>(&*value);
                if (a == nullptr || !std::has_single_bit(*a)) {
                    return fail(errc::bad_alignment, start);  // I2
                }
                file.alignment_ = *a;
            }
            file.metadata_.push_back(metadata_entry{key, std::move(*value)});
        }

        // ── tensor table ────────────────────────────────────────────
        file.tensors_.reserve(static_cast<std::size_t>(tensor_count));
        std::vector<std::size_t> record_starts;
        record_starts.reserve(static_cast<std::size_t>(tensor_count));
        std::unordered_set<std::string_view> names;
        for (std::uint64_t i = 0; i < tensor_count; ++i) {
            const std::size_t start = in.pos();
            tensor_info t;
            if (auto err = read_tensor_info(in, t)) {
                return *err;
            }
            if (!names.insert(t.name).second) {
                return fail(errc::duplicate_tensor, start);  // I3
            }
            file.tensors_.push_back(t);
            record_starts.push_back(start);
        }

        // ── data section ────────────────────────────────────────────
        // pos <= image.size() and alignment <= 2^31, so this cannot overflow.
        const std::uint64_t pos = in.pos();
        file.data_offset_ = (pos + file.alignment_ - 1) / file.alignment_ * file.alignment_;  // I2
        const std::uint64_t size = image.size();
        for (std::size_t i = 0; i < file.tensors_.size(); ++i) {
            const auto& t = file.tensors_[i];
            if (t.offset % file.alignment_ != 0) {
                return fail(errc::misaligned_tensor, record_starts[i]);  // I5
            }
            if (file.data_offset_ > size || t.offset > size - file.data_offset_ ||
                t.nbytes > size - file.data_offset_ - t.offset) {
                return fail(errc::tensor_out_of_bounds, record_starts[i]);  // I5
            }
        }

        assert(file.invariant_holds());
        return file;
    }

private:
    static error_info fail(errc code, std::uint64_t offset) noexcept {
        return error_info{code, offset, 0};
    }

    /// Read one value of `type`. Arrays are only accepted at the top level (I7).
    static std::optional<error_info>
    read_value(cursor& in, value_type type, bool top_level, std::optional<metadata_value>& out) {
        const std::size_t start = in.pos();
        switch (type) {
        case value_type::uint8:
            return read_scalar<std::uint8_t>(in, out);
        case value_type::int8:
            return read_scalar<std::int8_t>(in, out);
        case value_type::uint16:
            return read_scalar<std::uint16_t>(in, out);
        case value_type::int16:
            return read_scalar<std::int16_t>(in, out);
        case value_type::uint32:
            return read_scalar<std::uint32_t>(in, out);
        case value_type::int32:
            return read_scalar<std::int32_t>(in, out);
        case value_type::float32:
            return read_scalar<float>(in, out);
        case value_type::uint64:
            return read_scalar<std::uint64_t>(in, out);
        case value_type::int64:
            return read_scalar<std::int64_t>(in, out);
        case value_type::float64:
            return read_scalar<double>(in, out);
        case value_type::boolean: {
            std::uint8_t b = 0;
            if (!in.read(b)) {
                return fail(errc::truncated, start);
            }
            out = b != 0;
            return std::nullopt;
        }
        case value_type::string: {
            std::string_view s;
            if (!in.read_string(s)) {
                return fail(errc::truncated, start);
            }
            out = s;
            return std::nullopt;
        }
        case value_type::array: {
            if (!top_level) {
                return fail(errc::nested_array, start);  // I7
            }
            std::uint32_t raw_type = 0;
            std::uint64_t count = 0;
            if (!in.read(raw_type) || !in.read(count)) {
                return fail(errc::truncated, start);
            }
            const auto elem = static_cast<value_type>(raw_type);
            if (!is_valid(elem)) {
                return fail(errc::bad_value_type, start);
            }
            if (elem == value_type::array) {
                return fail(errc::nested_array, start);  // I7
            }
            if (elem == value_type::string) {
                if (count > in.remaining() / 8) {  // each string needs at least its length
                    return fail(errc::truncated, start);
                }
                std::vector<std::string_view> strings;
                strings.reserve(static_cast<std::size_t>(count));
                for (std::uint64_t i = 0; i < count; ++i) {
                    std::string_view s;
                    if (!in.read_string(s)) {
                        return fail(errc::truncated, start);
                    }
                    strings.push_back(s);
                }
                out = array_value{elem, count, {}, std::move(strings)};
                return std::nullopt;
            }
            const std::size_t elem_size = fixed_size(elem);
            std::span<const std::byte> raw;
            if (count > in.remaining() / elem_size || !in.read_bytes(count * elem_size, raw)) {
                return fail(errc::truncated, start);
            }
            out = array_value{elem, count, raw, {}};
            return std::nullopt;
        }
        }
        return fail(errc::bad_value_type, start);
    }

    template <typename T>
    static std::optional<error_info> read_scalar(cursor& in, std::optional<metadata_value>& out) {
        const std::size_t start = in.pos();
        T v{};
        if (!in.read(v)) {
            return fail(errc::truncated, start);
        }
        out = v;
        return std::nullopt;
    }

    /// Read one tensor-table record and check I4.
    static std::optional<error_info> read_tensor_info(cursor& in, tensor_info& t) {
        const std::size_t start = in.pos();
        if (!in.read_string(t.name) || !in.read(t.n_dims)) {
            return fail(errc::truncated, start);
        }
        if (t.n_dims < 1 || t.n_dims > max_dims) {
            return fail(errc::bad_dims, start);  // I4
        }
        std::uint64_t elements = 1;
        for (std::uint32_t d = 0; d < t.n_dims; ++d) {
            if (!in.read(t.ne[d])) {
                return fail(errc::truncated, start);
            }
            const auto product = checked_mul(elements, t.ne[d]);
            // ggml counts elements in int64_t.
            if (t.ne[d] == 0 || !product || *product > static_cast<std::uint64_t>(INT64_MAX)) {
                return fail(errc::bad_dims, start);  // I4
            }
            elements = *product;
        }
        std::uint32_t type = 0;
        if (!in.read(type) || !in.read(t.offset)) {
            return fail(errc::truncated, start);
        }
        t.type = static_cast<tensor_type>(type);
        const auto info = tensor_type_traits(t.type);
        if (!info) {
            return fail(errc::unsupported_tensor_type, start);  // I4
        }
        if (t.ne[0] % info->block_size != 0) {
            return fail(errc::partial_block, start);  // I4
        }
        // row_size <= ne[0] * 4.5 bytes for every supported type and
        // elements <= INT64_MAX, but the product can still overflow uint64.
        const auto bytes = checked_mul(t.ne[0] / info->block_size * info->type_size, elements / t.ne[0]);
        if (!bytes) {
            return fail(errc::bad_dims, start);
        }
        t.nbytes = *bytes;
        return std::nullopt;
    }
};

}  // namespace detail

inline result<gguf_file> parse(std::span<const std::byte> image) {
    return detail::parser::run(image);
}

}  // namespace branes::synapse::gguf

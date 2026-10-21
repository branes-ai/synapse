#pragma once
/// @file quantized_view.hpp
/// @brief Non-owning row-major matrix view over block-encoded weights.
///
/// A quantized_matrix_view<S> interprets a byte buffer (typically GGUF tensor
/// data inside a mapped file) as a rows x cols matrix stored row-major in the
/// blocks of scheme S. Nothing is copied or decoded until asked: block(r, b)
/// returns one encoded block, dequantize_row writes one row as float, and
/// to_dense materializes the whole matrix as an MTL5 dense2D<float> (the
/// reference the step-3 fused kernels are checked against).
///
/// Model
/// -----
///   quantized_matrix_view<S>
///     data_ : span<const byte> [non-owning] — the encoded matrix
///     rows_ : size_t
///     cols_ : size_t
///
/// Invariants
/// ----------
///   I1: cols_ % S::block_size == 0                                — constrains cols_
///   I2: data_.size() == rows_ * (cols_ / S::block_size) * S::type_size
///                                                                 — constrains data_, rows_, cols_
///
/// The view is immutable after construction, so I1-I2 hold for its lifetime.
/// Lifetime: valid only while the underlying buffer is.

#include <branes/synapse/quant/scheme.hpp>

#include <mtl/mat/dense2D.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <span>

namespace branes::synapse {

template <quant::Scheme S>
class quantized_matrix_view {
public:
    using scheme_type = S;
    using block_type = typename S::block_type;

    /// An empty 0 x 0 view.
    quantized_matrix_view() noexcept = default;

    ///   Pre:  cols % S::block_size == 0 and
    ///         data.size() == rows * (cols / S::block_size) * S::type_size.
    ///   Post: I1, I2.
    quantized_matrix_view(std::span<const std::byte> data, std::size_t rows, std::size_t cols) noexcept
        : data_(data), rows_(rows), cols_(cols) {
        assert(cols_ % S::block_size == 0);                                      // Pre: I1
        assert(data_.size() == rows_ * (cols_ / S::block_size) * S::type_size);  // Pre: I2
    }

    std::size_t rows() const noexcept {
        return rows_;
    }
    std::size_t cols() const noexcept {
        return cols_;
    }
    std::size_t blocks_per_row() const noexcept {
        return cols_ / S::block_size;
    }
    /// Bytes per encoded row.
    std::size_t row_stride() const noexcept {
        return blocks_per_row() * S::type_size;
    }
    std::span<const std::byte> data() const noexcept {
        return data_;
    }

    /// The encoded bytes of row r.  Pre: r < rows().
    std::span<const std::byte> row_bytes(std::size_t r) const noexcept {
        assert(r < rows_);  // Pre
        return data_.subspan(r * row_stride(), row_stride());
    }

    /// Encoded block b of row r.  Pre: r < rows(), b < blocks_per_row().
    block_type block(std::size_t r, std::size_t b) const noexcept {
        assert(r < rows_ && b < blocks_per_row());  // Pre
        return quant::detail::load_block<block_type>(data_.data() + r * row_stride() + b * S::type_size);
    }

    /// Element (r, c), decoding only the block that holds it.
    ///   Pre: r < rows(), c < cols().
    float at(std::size_t r, std::size_t c) const noexcept {
        assert(r < rows_ && c < cols_);  // Pre
        std::array<float, S::block_size> block{};
        S::dequantize(data_.data() + r * row_stride() + (c / S::block_size) * S::type_size, block.data());
        return block[c % S::block_size];
    }

    /// Decode row r into out.  Pre: r < rows(), out.size() == cols().
    void dequantize_row(std::size_t r, std::span<float> out) const noexcept {
        assert(r < rows_ && out.size() == cols_);  // Pre
        const std::byte* src = data_.data() + r * row_stride();
        for (std::size_t b = 0; b < blocks_per_row(); ++b) {
            S::dequantize(src + b * S::type_size, out.data() + b * S::block_size);
        }
    }

    /// The whole matrix decoded into a row-major MTL5 dense matrix.
    mtl::mat::dense2D<float> to_dense() const {
        mtl::mat::dense2D<float> m(rows_, cols_);
        for (std::size_t r = 0; r < rows_; ++r) {
            dequantize_row(r, std::span<float>(m.data() + r * cols_, cols_));
        }
        return m;
    }

private:
    std::span<const std::byte> data_;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
};

}  // namespace branes::synapse

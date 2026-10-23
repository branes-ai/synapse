#pragma once
/// @file gemm.hpp
/// @brief Y = X W^T for a batch of activation rows and a quantized weight matrix.
///
/// The batched form of gemv, for prefill: X holds n activation rows (tokens)
/// of w.num_cols() floats, Y receives n rows of w.num_rows() outputs. Each activation
/// row is quantized once, as gemv does; then every weight block is unpacked
/// once and dotted with all n rows, so the cost of reading and decoding the
/// weights is shared by the batch.
///
/// Contract G1: row t of Y equals gemv(w, row t of X) bit for bit. Both
/// accumulate weight_chunk<S>::dot(a_t) over the row's chunks in order,
/// starting from 0.0f (see dot.hpp), and quantize the activations with the
/// same function. So a prompt prefilled in batches leaves the same
/// results as decoding it one token at a time.
///
/// G1 holds when the compiler does not fuse multiply-adds (FMA contraction),
/// which could otherwise happen differently at gemm's and gemv's inlined
/// accumulation sites. synapse's own targets build with -ffp-contract=off
/// (cmake/numerics.cmake); MSVC does not contract by default, nor does any
/// compiler for targets without FMA. A consumer that needs G1 on an FMA
/// target with GCC or Clang compiles with -ffp-contract=off as well.
///
/// gemm never allocates and never throws: the activations and their
/// descriptors live in caller-provided scratch, the sums in y itself.
/// gemm_prepare + gemm_rows split it so the rows can be spread over threads.

#include <branes/synapse/kernels/dot.hpp>
#include <branes/synapse/kernels/gemv.hpp>
#include <branes/synapse/quant/scheme.hpp>

#include <mtl/mat/block_matrix_view.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace branes::synapse::kernels {

/// Alignment gemm requires of its scratch buffer (it holds pointers and
/// floats besides the activation blocks). Satisfied by ::operator new and by
/// a session's arena.
inline constexpr std::size_t gemm_scratch_alignment = alignof(std::max_align_t);

namespace detail {

constexpr std::size_t align_scratch(std::size_t n) noexcept {
    return (n + gemm_scratch_alignment - 1) / gemm_scratch_alignment * gemm_scratch_alignment;
}

/// Bytes between consecutive activation rows' quantized forms.
template <quant::Scheme S>
constexpr std::size_t gemm_row_stride(std::size_t cols) noexcept {
    return align_scratch(gemv_scratch_bytes<S>(cols));
}

}  // namespace detail

/// Weight rows gemm_rows unpacks together (one chunk each, on the stack), so
/// a token's chunk of activations is reused across the tile while it is in
/// L1. The tiling only reorders independent (row, token) sums, never the
/// order within one, so it does not affect results (G1, D1). Measured on
/// AVX2 with Llama-3.2-1B Q4_K_M prefill: 4 beats 1 by ~10% and 8 by ~5%.
inline constexpr std::size_t gemm_row_tile = 4;
/// Batches smaller than this skip the row tile: a single token (decode) has
/// no activations to reuse, and one row at a time keeps its chunk hotter.
inline constexpr std::size_t gemm_tile_min_tokens = 2;

/// Scratch bytes gemm needs for n activation rows of `cols` columns: per row,
/// the quantized activations (as gemv) and their descriptor.
///   Pre: cols % S::block_size == 0.
template <quant::Scheme S>
constexpr std::size_t gemm_scratch_bytes(std::size_t cols, std::size_t n) noexcept {
    using A = typename S::vec_dot_block;
    return n * detail::gemm_row_stride<S>(cols) + detail::align_scratch(n * sizeof(activations<A>));
}

/// Quantize the n activation rows of x into scratch, once for the whole
/// product, and return their descriptors (which live in scratch).
///   Pre: as gemm.
template <quant::Scheme S>
const activations<typename S::vec_dot_block>*
gemm_prepare(std::size_t cols, std::span<const float> x, std::size_t n, std::span<std::byte> scratch) noexcept {
    using A = typename S::vec_dot_block;
    assert(n >= 1 && x.size() == n * cols);                                                  // Pre
    assert(scratch.size() >= gemm_scratch_bytes<S>(cols, n));                                // Pre
    assert(reinterpret_cast<std::uintptr_t>(scratch.data()) % gemm_scratch_alignment == 0);  // Pre
    const std::size_t stride = detail::gemm_row_stride<S>(cols);
    auto* const acts = ::new (static_cast<void*>(scratch.data() + n * stride)) activations<A>[n];
    for (std::size_t t = 0; t < n; ++t) {
        acts[t] = detail::prepare_activations<A>(x.subspan(t * cols, cols), scratch.subspan(t * stride, stride));
    }
    return acts;
}

/// Output rows [r0, r1) of every token: y[t * w.num_rows() + r] for r in [r0, r1).
/// Reads only w and acts and writes only those outputs, so disjoint row
/// ranges can run on different threads, and the results do not depend on how
/// the rows are split (D1).
///   Pre: acts from gemm_prepare for these n rows; r0 <= r1 <= w.num_rows().
template <quant::Scheme S>
void gemm_rows(const mtl::mat::block_matrix_view<S>& w,
               const activations<typename S::vec_dot_block>* acts,
               std::size_t n,
               std::span<float> y,
               std::size_t r0,
               std::size_t r1) noexcept {
    const std::size_t cols = w.num_cols();
    const std::size_t rows = w.num_rows();
    assert(r0 <= r1 && r1 <= rows && y.size() == n * rows);  // Pre
    if constexpr (S::block_size == 1) {
        // Plain formats: each chunk of a row decoded to float once, dotted
        // with every token; the sums accumulate in y (a float store and
        // reload is exact), starting at 0, in chunk order, as gemv (G1).
        std::array<float, plain_chunk> buf{};
        for (std::size_t r = r0; r < r1; ++r) {
            const std::byte* row = w.row(r).data();
            for (std::size_t t = 0; t < n; ++t) {
                y[t * rows + r] = 0.0f;
            }
            for (std::size_t i = 0; i * plain_chunk < cols; ++i) {
                const std::size_t len = std::min(plain_chunk, cols - i * plain_chunk);
                plain_kernel<S>::unpack(row, i, len, buf.data());
                for (std::size_t t = 0; t < n; ++t) {
                    y[t * rows + r] += plain_kernel<S>::dot(buf.data(), acts[t].x, i, len);
                }
            }
        }
    } else {
        // Quantized formats: a tile of gemm_row_tile weight rows x all tokens.
        // Chunk c of every row in the tile is unpacked once (in L1), then each
        // token's chunk of activations is dotted with all of them while it is
        // hot. Every (row, token) sum still starts at 0 and adds its chunk
        // dots in chunk order, as gemv does (G1), whatever the tiling.
        using C = weight_chunk<S>;
        std::array<C, gemm_row_tile> chunks;
        const std::size_t blocks = w.blocks_per_row();
        // A single token has nothing to reuse across rows; one row at a time
        // keeps its chunk hot instead.
        const std::size_t tile = n >= gemm_tile_min_tokens ? gemm_row_tile : 1;
        for (std::size_t g = r0; g < r1; g += tile) {
            const std::size_t nr = std::min(tile, r1 - g);
            for (std::size_t t = 0; t < n; ++t) {
                for (std::size_t j = 0; j < nr; ++j) {
                    y[t * rows + g + j] = 0.0f;
                }
            }
            for (std::size_t b0 = 0; b0 < blocks; b0 += C::max_blocks) {
                const std::size_t nb = std::min(C::max_blocks, blocks - b0);
                for (std::size_t j = 0; j < nr; ++j) {
                    chunks[j].load(w.row(g + j).data(), b0, nb);
                }
                for (std::size_t t = 0; t < n; ++t) {
                    float* yt = y.data() + t * rows + g;
                    for (std::size_t j = 0; j < nr; ++j) {
                        yt[j] += chunks[j].dot(acts[t]);
                    }
                }
            }
        }
    }
}

/// Y = X W^T: y[t * w.num_rows() + r] = row r of w . row t of x.
///
///   Pre:  n >= 1, x.size() == n * w.num_cols(), y.size() == n * w.num_rows(),
///         scratch.size() >= gemm_scratch_bytes<S>(w.num_cols(), n), scratch
///         aligned to gemm_scratch_alignment; x, y and scratch do not overlap.
///   Post: G1 — row t of y equals gemv(w, row t of x) bit for bit.
template <quant::Scheme S>
void gemm(const mtl::mat::block_matrix_view<S>& w,
          std::span<const float> x,
          std::size_t n,
          std::span<float> y,
          std::span<std::byte> scratch) noexcept {
    assert(y.size() == n * w.num_rows());  // Pre
    gemm_rows(w, gemm_prepare<S>(w.num_cols(), x, n, scratch), n, y, 0, w.num_rows());
}

}  // namespace branes::synapse::kernels

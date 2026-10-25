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
/// accumulate each chunk's weight_chunk<S>::dot_tile contribution over the
/// row's chunks in order, starting from 0.0f, computing every (row, token)
/// pair the same way whatever the tile (see dot.hpp), and quantize the
/// activations with the same function. So a prompt prefilled in batches
/// leaves the same results as decoding it one token at a time.
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
/// a token's chunk of activations is reused across the block while it is in
/// L1. The tiling only reorders independent (row, token) sums, never the
/// order within one, so it does not affect results (G1, D1). Measured on
/// AVX2 (i7-12700K) with the register tile below: 16 is ~5% faster than 4
/// for SmolVLM's Q8_0 encoder and ~10% for Llama-3.2-1B Q4_K_M prefill, and
/// 8 and 32 are within noise of it.
inline constexpr std::size_t gemm_row_tile = 16;
/// Batches smaller than this skip the row tile: a single token (decode) has
/// no activations to reuse, and one row at a time keeps its chunk hotter.
inline constexpr std::size_t gemm_tile_min_tokens = 2;
/// The register tile: weight rows x tokens whose dots run together, one
/// float accumulator each (8 of AVX2's 16 vector registers). Measured on
/// AVX2: 2 x 4 and 4 x 3 are within noise of 4 x 2, 4 x 4 is ~20% slower for
/// Q4_K prefill (it spills), and 8 x 1 ~10% slower.
inline constexpr std::size_t gemm_tile_rows = 4;
inline constexpr std::size_t gemm_tile_tokens = 2;
static_assert(gemm_row_tile % gemm_tile_rows == 0, "a row block is whole register tiles");

/// Scratch bytes gemm needs for n activation rows of `cols` columns: per row,
/// the quantized activations (as gemv) and their descriptor.
///   Pre: cols % S::block_size == 0.
template <quant::Scheme S>
constexpr std::size_t gemm_scratch_bytes(std::size_t cols, std::size_t n) noexcept {
    using A = typename S::vec_dot_block;
    return n * detail::gemm_row_stride<S>(cols) + detail::align_scratch(n * sizeof(activations<A>));
}

/// The descriptors of n activation rows in scratch, created but not yet
/// filled: gemm_prepare_rows fills them, a range of rows at a time.
///   Pre: as gemm.
template <quant::Scheme S>
activations<typename S::vec_dot_block>*
gemm_activations(std::size_t cols, std::size_t n, std::span<std::byte> scratch) noexcept {
    using A = typename S::vec_dot_block;
    assert(n >= 1);                                                                          // Pre
    assert(scratch.size() >= gemm_scratch_bytes<S>(cols, n));                                // Pre
    assert(reinterpret_cast<std::uintptr_t>(scratch.data()) % gemm_scratch_alignment == 0);  // Pre
    return ::new (static_cast<void*>(scratch.data() + n * detail::gemm_row_stride<S>(cols))) activations<A>[n];
}

/// Quantize activation rows [t0, t1) of x into scratch and fill their
/// descriptors. Each row is quantized alone, so disjoint ranges can run on
/// different threads.
///   Pre: acts from gemm_activations(cols, n, scratch) with x.size() == n * cols;
///        t0 <= t1 <= n.
template <quant::Scheme S>
void gemm_prepare_rows(std::size_t cols,
                       std::span<const float> x,
                       std::span<std::byte> scratch,
                       activations<typename S::vec_dot_block>* acts,
                       std::size_t t0,
                       std::size_t t1) noexcept {
    using A = typename S::vec_dot_block;
    assert(t0 <= t1 && t1 * cols <= x.size());  // Pre
    const std::size_t stride = detail::gemm_row_stride<S>(cols);
    for (std::size_t t = t0; t < t1; ++t) {
        acts[t] = detail::prepare_activations<A>(x.subspan(t * cols, cols), scratch.subspan(t * stride, stride));
    }
}

/// Quantize the n activation rows of x into scratch, once for the whole
/// product, and return their descriptors (which live in scratch).
///   Pre: as gemm.
template <quant::Scheme S>
const activations<typename S::vec_dot_block>*
gemm_prepare(std::size_t cols, std::span<const float> x, std::size_t n, std::span<std::byte> scratch) noexcept {
    assert(x.size() == n * cols);  // Pre
    auto* const acts = gemm_activations<S>(cols, n, scratch);
    gemm_prepare_rows<S>(cols, x, scratch, acts, 0, n);
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
        // Quantized formats: a block of gemm_row_tile weight rows x all tokens.
        // Chunk c of every row in the block is unpacked once (in L1), then
        // register tiles of gemm_tile_rows rows x gemm_tile_tokens tokens
        // dot it with each token's chunk of activations while that is hot.
        // Every (row, token) sum still starts at 0 and adds its chunk dots in
        // chunk order, computed as gemv computes them (G1), whatever the tiling.
        using C = weight_chunk<S>;
        constexpr std::size_t R = gemm_tile_rows;
        constexpr std::size_t T = gemm_tile_tokens;
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
                std::size_t t = 0;
                for (; t + T <= n; t += T) {
                    float* yt = y.data() + t * rows + g;
                    std::size_t j = 0;
                    for (; j + R <= nr; j += R) {
                        C::template dot_tile<R, T>(&chunks[j], &acts[t], yt + j, rows);
                    }
                    for (; j < nr; ++j) {
                        C::template dot_tile<1, T>(&chunks[j], &acts[t], yt + j, rows);
                    }
                }
                for (; t < n; ++t) {
                    float* yt = y.data() + t * rows + g;
                    std::size_t j = 0;
                    for (; j + R <= nr; j += R) {
                        C::template dot_tile<R, 1>(&chunks[j], &acts[t], yt + j, rows);
                    }
                    for (; j < nr; ++j) {
                        C::template dot_tile<1, 1>(&chunks[j], &acts[t], yt + j, rows);
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

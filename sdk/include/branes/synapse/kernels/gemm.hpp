#pragma once
/// @file gemm.hpp
/// @brief Y = X W^T for a batch of activation rows and a quantized weight matrix.
///
/// The batched form of gemv, for prefill: X holds n activation rows (tokens)
/// of w.cols() floats, Y receives n rows of w.rows() outputs. Each activation
/// row is quantized once, as gemv does; then every weight block is unpacked
/// once and dotted with all n rows, so the cost of reading and decoding the
/// weights is shared by the batch.
///
/// Contract G1: row t of Y equals gemv(w, row t of X) bit for bit. Both
/// accumulate block_kernel<S>::dot(unpack(row, i), a_t, i) over the blocks i
/// in order, starting from 0.0f (see dot.hpp), and quantize the activations
/// with the same function. So a prompt prefilled in batches leaves the same
/// results as decoding it one token at a time.
///
/// G1 holds when the compiler does not fuse multiply-adds (FMA contraction),
/// which could otherwise happen differently at gemm's and gemv's inlined
/// accumulation sites. synapse's own targets build with -ffp-contract=off
/// (cmake/numerics.cmake); MSVC does not contract by default, nor does any
/// compiler for targets without FMA. A consumer that needs G1 on an FMA
/// target with GCC or Clang compiles with -ffp-contract=off as well.
///
/// gemm never allocates and never throws: the activations, their descriptors
/// and the per-row accumulators live in caller-provided scratch.

#include <branes/synapse/kernels/dot.hpp>
#include <branes/synapse/kernels/gemv.hpp>
#include <branes/synapse/quant/scheme.hpp>
#include <branes/synapse/tensor/quantized_view.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>

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

/// Scratch bytes gemm needs for n activation rows of `cols` columns: per row,
/// the quantized activations (as gemv) and their descriptor, plus one float
/// accumulator per row.
///   Pre: cols % S::block_size == 0.
template <quant::Scheme S>
constexpr std::size_t gemm_scratch_bytes(std::size_t cols, std::size_t n) noexcept {
    using A = typename S::vec_dot_block;
    return n * detail::gemm_row_stride<S>(cols) + detail::align_scratch(n * sizeof(activations<A>)) + n * sizeof(float);
}

/// Y = X W^T: y[t * w.rows() + r] = row r of w . row t of x.
///
///   Pre:  n >= 1, x.size() == n * w.cols(), y.size() == n * w.rows(),
///         scratch.size() >= gemm_scratch_bytes<S>(w.cols(), n), scratch
///         aligned to gemm_scratch_alignment; x, y and scratch do not overlap.
///   Post: G1 — row t of y equals gemv(w, row t of x) bit for bit.
template <quant::Scheme S>
void gemm(const quantized_matrix_view<S>& w,
          std::span<const float> x,
          std::size_t n,
          std::span<float> y,
          std::span<std::byte> scratch) noexcept {
    using A = typename S::vec_dot_block;
    const std::size_t cols = w.cols();
    const std::size_t rows = w.rows();
    assert(n >= 1 && x.size() == n * cols && y.size() == n * rows);                          // Pre
    assert(scratch.size() >= gemm_scratch_bytes<S>(cols, n));                                // Pre
    assert(reinterpret_cast<std::uintptr_t>(scratch.data()) % gemm_scratch_alignment == 0);  // Pre

    const std::size_t stride = detail::gemm_row_stride<S>(cols);
    std::byte* const act_bytes = scratch.data() + n * stride;
    auto* const acts = ::new (static_cast<void*>(act_bytes)) activations<A>[n];
    auto* const acc =
        ::new (static_cast<void*>(act_bytes + detail::align_scratch(n * sizeof(activations<A>)))) float[n];
    for (std::size_t t = 0; t < n; ++t) {
        acts[t] = detail::prepare_activations<A>(x.subspan(t * cols, cols), scratch.subspan(t * stride, stride));
    }

    for (std::size_t r = 0; r < rows; ++r) {
        const std::byte* row = w.row_bytes(r).data();
        for (std::size_t t = 0; t < n; ++t) {
            acc[t] = 0.0f;
        }
        if constexpr (S::block_size == 1) {
            std::array<float, plain_chunk> buf{};
            for (std::size_t i = 0; i * plain_chunk < cols; ++i) {
                const std::size_t len = std::min(plain_chunk, cols - i * plain_chunk);
                plain_kernel<S>::unpack(row, i, len, buf.data());
                for (std::size_t t = 0; t < n; ++t) {
                    acc[t] += plain_kernel<S>::dot(buf.data(), acts[t].x, i, len);
                }
            }
        } else {
            for (std::size_t i = 0; i < w.blocks_per_row(); ++i) {
                const auto u = block_kernel<S>::unpack(row, i);
                for (std::size_t t = 0; t < n; ++t) {
                    acc[t] += block_kernel<S>::dot(u, acts[t], i);
                }
            }
        }
        for (std::size_t t = 0; t < n; ++t) {
            y[t * rows + r] = acc[t];
        }
    }
}

}  // namespace branes::synapse::kernels

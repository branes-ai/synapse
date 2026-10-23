#pragma once
/// @file gemv.hpp
/// @brief y = W x for a quantized weight matrix, with no allocation.
///
/// gemv quantizes x once into the scheme's activation format (Q8_0 or Q8_K
/// blocks, or leaves it float for plain formats), then computes each output as
/// a fused dot of one encoded weight row with those activations (see dot.hpp).
/// The quantized activations live in caller-provided scratch memory, so a
/// static memory planner can place them; gemv itself never allocates and
/// never throws. It is also MTL5's fused kernel for synapse's schemes
/// (mtl::block_kernels, below), so mtl::mult on their views runs it.
///
/// Accuracy: against the float reference dequantize(W) . x, the only extra
/// error is the rounding of x to 8 bits (relative ~1/254 per element for
/// Q8_0/Q8_K); the integer block sums are exact. Results agree with ggml's
/// CPU kernels to float re-association error.

#include <branes/synapse/kernels/dot.hpp>
#include <branes/synapse/quant/q8.hpp>
#include <branes/synapse/quant/scheme.hpp>

#include <mtl/mat/block_matrix_view.hpp>

#include <cassert>
#include <cstddef>  // std::max_align_t
#include <cstdint>
#include <new>
#include <span>
#include <type_traits>

namespace branes::synapse::kernels {

/// Alignment gemv requires of its scratch buffer. It does not exceed the
/// fundamental alignment, so any storage from ::operator new (for example a
/// std::vector<std::byte>) or a max_align_t-aligned arena satisfies it.
inline constexpr std::size_t gemv_scratch_alignment = alignof(quant::block_q8_k);
static_assert(gemv_scratch_alignment <= alignof(std::max_align_t),
              "gemv scratch must be satisfiable by default-aligned allocations");

namespace detail {

template <typename A>
constexpr std::size_t activation_block_size() noexcept {
    if constexpr (std::is_same_v<A, quant::block_q8_0>) {
        return quant::qk_legacy;
    } else {
        static_assert(std::is_same_v<A, quant::block_q8_k>);
        return quant::qk_k;
    }
}

/// Byte offset of the widened int16 activations in the scratch buffer: after
/// the blocks, rounded up to int16 alignment.
template <typename A>
constexpr std::size_t wide_offset(std::size_t cols) noexcept {
    const std::size_t blocks = cols / activation_block_size<A>() * sizeof(A);
    return (blocks + alignof(std::int16_t) - 1) / alignof(std::int16_t) * alignof(std::int16_t);
}

/// Quantize x into the activation format A in scratch (the blocks, then
/// their quants widened to int16), or use x itself for float activations.
///   Pre: scratch holds gemv_scratch_bytes for x.size() columns, aligned to
///   gemv_scratch_alignment.
template <typename A>
activations<A> prepare_activations(std::span<const float> x, std::span<std::byte> scratch) noexcept {
    if constexpr (std::is_same_v<A, float>) {
        return {x.data()};
    } else {
        // Create the activation blocks and the widened quants in the scratch
        // storage, then fill them: once per activation row, shared by every
        // weight row.
        constexpr std::size_t qk = activation_block_size<A>();
        const std::size_t nb = x.size() / qk;
        A* blocks = ::new (static_cast<void*>(scratch.data())) A[nb];
        if constexpr (std::is_same_v<A, quant::block_q8_0>) {
            quant::quantize_row_q8_0(x, std::span<A>(blocks, nb));
        } else {
            quant::quantize_row_q8_k(x, std::span<A>(blocks, nb));
        }
        std::int16_t* wide =
            ::new (static_cast<void*>(scratch.data() + wide_offset<A>(x.size()))) std::int16_t[x.size()];
        for (std::size_t i = 0; i < nb; ++i) {
            for (std::size_t j = 0; j < qk; ++j) {
                wide[i * qk + j] = blocks[i].qs[j];
            }
        }
        if constexpr (std::is_same_v<A, quant::block_q8_k>) {
            // The sub-block sums, contiguous: the operand of Q4_K's min terms.
            constexpr std::size_t subs = sizeof(quant::block_q8_k::bsums) / sizeof(std::int16_t);
            std::int16_t* bsums =
                ::new (static_cast<void*>(scratch.data() + wide_offset<A>(x.size()) + x.size() * sizeof(std::int16_t)))
                    std::int16_t[nb * subs];
            for (std::size_t i = 0; i < nb; ++i) {
                for (std::size_t j = 0; j < subs; ++j) {
                    bsums[i * subs + j] = blocks[i].bsums[j];
                }
            }
            return {blocks, wide, bsums};
        } else {
            return {blocks, wide};
        }
    }
}

}  // namespace detail

/// Scratch bytes gemv needs for a matrix of scheme S with `cols` columns:
/// the quantized activation blocks followed by their quants widened to int16
/// (0 for plain formats, whose activations stay float).
///   Pre: cols % S::block_size == 0.
template <quant::Scheme S>
constexpr std::size_t gemv_scratch_bytes(std::size_t cols) noexcept {
    using A = typename S::vec_dot_block;
    if constexpr (std::is_same_v<A, float>) {
        return 0;
    } else {
        const std::size_t bytes = detail::wide_offset<A>(cols) + cols * sizeof(std::int16_t);
        if constexpr (std::is_same_v<A, quant::block_q8_k>) {
            return bytes + cols / quant::qk_k * sizeof(quant::block_q8_k::bsums);  // contiguous bsums
        } else {
            return bytes;
        }
    }
}

/// y = W x.
///
///   Pre:  x.size() == w.num_cols(), y.size() == w.num_rows(),
///         scratch.size() >= gemv_scratch_bytes<S>(w.num_cols()), and scratch is
///         aligned to gemv_scratch_alignment; x and y do not overlap scratch.
///   Post: y[r] == row_dot(row r of w, activations(x)) for every r; scratch
///         holds the quantized activation blocks (ggml's Q8_0 / Q8_K layout)
///         followed by their int16-widened quants.
template <quant::Scheme S>
void gemv(const mtl::mat::block_matrix_view<S>& w,
          std::span<const float> x,
          std::span<float> y,
          std::span<std::byte> scratch) noexcept {
    assert(x.size() == w.num_cols() && y.size() == w.num_rows());                            // Pre
    assert(scratch.size() >= gemv_scratch_bytes<S>(w.num_cols()));                           // Pre
    assert(reinterpret_cast<std::uintptr_t>(scratch.data()) % gemv_scratch_alignment == 0);  // Pre

    const auto act = detail::prepare_activations<typename S::vec_dot_block>(x, scratch);
    for (std::size_t r = 0; r < w.num_rows(); ++r) {
        y[r] = row_dot<S>(w.row(r).data(), act, w.blocks_per_row());
    }
}

}  // namespace branes::synapse::kernels

/// MTL5's fused mat-vec hook for synapse's schemes: mtl::mult(w, x, y,
/// scratch) on a block_matrix_view of a synapse scheme runs kernels::gemv.
///
///   Pre:  as kernels::gemv, with x of w.num_cols() and y of w.num_rows() floats
///         (mult checks the scratch size and alignment).
///   Post: y equals kernels::gemv's output bit for bit.
template <branes::synapse::quant::Scheme S>
struct mtl::block_kernels<S> {
    static void
    gemv(const mtl::mat::block_matrix_view<S>& w, const float* x, float* y, std::span<std::byte> scratch) noexcept {
        branes::synapse::kernels::gemv(
            w, std::span<const float>(x, w.num_cols()), std::span<float>(y, w.num_rows()), scratch);
    }
    static std::size_t scratch_bytes(std::size_t cols) noexcept {
        return branes::synapse::kernels::gemv_scratch_bytes<S>(cols);
    }
    static constexpr std::size_t scratch_alignment = branes::synapse::kernels::gemv_scratch_alignment;
};

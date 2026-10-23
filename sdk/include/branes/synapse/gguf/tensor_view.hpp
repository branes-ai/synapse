#pragma once
/// @file tensor_view.hpp
/// @brief Bind GGUF tensors to typed MTL5 block_matrix_views.
///
/// GGML shapes map onto a row-major matrix: ne[0] (contiguous) is the column
/// count and ne[1] * ne[2] * ne[3] the row count. A weight W[out, in] is
/// stored with ne = {in, out}, so its rows are exactly the dot products of
/// y = W x.
///
/// `visit_matrix` is the one place a runtime tensor_type tag becomes a
/// compile-time scheme; everything past it is typed, so kernels never switch
/// on ggml_type.

#include <branes/synapse/gguf/reader.hpp>
#include <branes/synapse/quant/scheme.hpp>

#include <mtl/mat/block_matrix_view.hpp>

#include <cstddef>
#include <optional>
#include <utility>

namespace branes::synapse::gguf {

/// Tensor `t` of `file` viewed as a matrix of scheme S, or nullopt if `t` is
/// not stored as S::type.
///
///   Pre:  t is an element of file.tensors().
///   Post: on success, the view's I1-I2 hold (they follow from the parser's
///         I4-I5: whole blocks per row, and nbytes = rows * row size), so
///         block_matrix_view's constructor, which checks them, cannot throw.
template <quant::Scheme S>
std::optional<mtl::mat::block_matrix_view<S>> view_as(const gguf_file& file, const tensor_info& t) noexcept {
    if (t.type != S::type) {
        return std::nullopt;
    }
    const auto cols = static_cast<std::size_t>(t.ne[0]);
    const auto rows = static_cast<std::size_t>(t.ne[1] * t.ne[2] * t.ne[3]);
    return mtl::mat::block_matrix_view<S>(rows, cols, file.tensor_data(t));
}

namespace detail {

template <tensor_type T, typename F>
bool visit_as(const gguf_file& file, const tensor_info& t, F&& fn) {
    std::forward<F>(fn)(*view_as<quant::scheme<T>>(file, t));
    return true;
}

}  // namespace detail

/// Call fn(mtl::mat::block_matrix_view<S>) with the scheme S of tensor `t`.
/// Returns false, without calling fn, if synapse has no scheme for t.type yet.
///
///   Pre:  t is an element of file.tensors().
template <typename F>
bool visit_matrix(const gguf_file& file, const tensor_info& t, F&& fn) {
    switch (t.type) {
    case tensor_type::f32:
        return detail::visit_as<tensor_type::f32>(file, t, std::forward<F>(fn));
    case tensor_type::f16:
        return detail::visit_as<tensor_type::f16>(file, t, std::forward<F>(fn));
    case tensor_type::bf16:
        return detail::visit_as<tensor_type::bf16>(file, t, std::forward<F>(fn));
    case tensor_type::q4_0:
        return detail::visit_as<tensor_type::q4_0>(file, t, std::forward<F>(fn));
    case tensor_type::q8_0:
        return detail::visit_as<tensor_type::q8_0>(file, t, std::forward<F>(fn));
    case tensor_type::q4_k:
        return detail::visit_as<tensor_type::q4_k>(file, t, std::forward<F>(fn));
    case tensor_type::q6_k:
        return detail::visit_as<tensor_type::q6_k>(file, t, std::forward<F>(fn));
    default:
        return false;
    }
}

}  // namespace branes::synapse::gguf

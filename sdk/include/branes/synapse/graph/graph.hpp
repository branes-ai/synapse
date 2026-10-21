#pragma once
/// @file graph.hpp
/// @brief The compute graph of a decode step: tensors, operations, and a checked builder.
///
/// A graph is a DAG of operations over float activation vectors. Weights are
/// not tensors of the graph: they are bound into the operations as typed
/// quantized_matrix_views (zero-copy views into the model file), so the
/// executor never switches on a storage format.
///
/// Model
/// -----
///   tensor_desc
///     bytes : size_t — storage the tensor needs in the arena
///     size  : size_t — float elements (0 for kernel scratch)
///     role  : input | intermediate | output | scratch
///
///   node = std::variant<op_matmul, op_add, op_mul, op_silu, op_softmax, op_rms_norm>
///     each op names its operand and result tensors by tensor_id; op_matmul
///     also names a scratch tensor for the gemv activation buffers
///
///   compute_graph
///     tensors_ : vector<tensor_desc>
///     nodes_   : vector<node>
///     inputs_  : vector<tensor_id>,  outputs_ : vector<tensor_id>
///
/// Invariants (established by graph_builder::finalize, immutable afterwards)
/// ----------
///   I1: DAG by construction — every operand of node i is an input or the
///       result of a node j < i, so node order is a topological order
///                                                   — constrains nodes_
///   I2: every node's operand sizes agree with its operation
///                                                   — constrains nodes_, tensors_
///   I3: every tensor is an input or the result (or scratch) of exactly one
///       node                                        — constrains nodes_, tensors_
///   I4: at least one output; outputs are distinct, non-scratch tensors
///                                                   — constrains outputs_
///
/// The builder is no-throw on bad input: graphs are built from model
/// metadata (untrusted), so it records the first error and finalize()
/// returns it, with the node index where it occurred.

#include <branes/synapse/kernels/gemv.hpp>
#include <branes/synapse/quant/scheme.hpp>
#include <branes/synapse/result.hpp>
#include <branes/synapse/tensor/quantized_view.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace branes::synapse::graph {

using quant::tensor_type;

/// Index of a tensor in its graph.
using tensor_id = std::uint32_t;

/// The id the builder returns after an error; operations on it are no-ops.
inline constexpr tensor_id invalid_tensor = std::numeric_limits<tensor_id>::max();

/// A weight matrix in any format synapse can compute with.
using matrix_ref = std::variant<quantized_matrix_view<quant::scheme<tensor_type::f32>>,
                                quantized_matrix_view<quant::scheme<tensor_type::f16>>,
                                quantized_matrix_view<quant::scheme<tensor_type::bf16>>,
                                quantized_matrix_view<quant::scheme<tensor_type::q4_0>>,
                                quantized_matrix_view<quant::scheme<tensor_type::q8_0>>,
                                quantized_matrix_view<quant::scheme<tensor_type::q4_k>>,
                                quantized_matrix_view<quant::scheme<tensor_type::q6_k>>>;

inline std::size_t rows(const matrix_ref& m) noexcept {
    return std::visit([](const auto& v) { return v.rows(); }, m);
}
inline std::size_t cols(const matrix_ref& m) noexcept {
    return std::visit([](const auto& v) { return v.cols(); }, m);
}

enum class tensor_role : std::uint8_t { input, intermediate, output, scratch };

struct tensor_desc {
    std::size_t bytes = 0;
    std::size_t size = 0;  ///< float elements; 0 for scratch
    tensor_role role = tensor_role::intermediate;
};

struct op_matmul {  ///< out = w * in (gemv)
    matrix_ref w;
    tensor_id in, out, scratch;
};
struct op_add {  ///< out = a + b
    tensor_id a, b, out;
};
struct op_mul {  ///< out = a * b, element-wise
    tensor_id a, b, out;
};
struct op_silu {  ///< out = silu(in)
    tensor_id in, out;
};
struct op_softmax {  ///< out = softmax(in)
    tensor_id in, out;
};
struct op_rms_norm {  ///< out = in / rms(in) * gamma
    tensor_id in;
    matrix_ref gamma;  ///< 1 x size
    float eps;
    tensor_id out;
};

using node = std::variant<op_matmul, op_add, op_mul, op_silu, op_softmax, op_rms_norm>;

/// Operand (read) tensors of a node.
inline std::vector<tensor_id> operands(const node& n) {
    struct visitor {
        std::vector<tensor_id> operator()(const op_matmul& o) const {
            return {o.in};
        }
        std::vector<tensor_id> operator()(const op_add& o) const {
            return {o.a, o.b};
        }
        std::vector<tensor_id> operator()(const op_mul& o) const {
            return {o.a, o.b};
        }
        std::vector<tensor_id> operator()(const op_silu& o) const {
            return {o.in};
        }
        std::vector<tensor_id> operator()(const op_softmax& o) const {
            return {o.in};
        }
        std::vector<tensor_id> operator()(const op_rms_norm& o) const {
            return {o.in};
        }
    };
    return std::visit(visitor{}, n);
}

/// Tensors a node writes: its result, and its scratch if it has one.
inline std::vector<tensor_id> results(const node& n) {
    if (const auto* m = std::get_if<op_matmul>(&n)) {
        return {m->out, m->scratch};
    }
    return {std::visit([](const auto& o) { return o.out; }, n)};
}

/// Why a graph was rejected.
enum class graph_errc : std::uint32_t {
    invalid_operand = 1,  ///< a tensor id that is not in the graph
    size_mismatch,        ///< operand sizes do not fit the operation
    empty_tensor,         ///< a zero-sized input or operand
    bad_output,           ///< output is invalid, scratch, or listed twice
    no_outputs,           ///< finalize() on a graph with no outputs
};

constexpr std::string_view to_string(graph_errc code) noexcept {
    switch (code) {
    case graph_errc::invalid_operand:
        return "invalid operand";
    case graph_errc::size_mismatch:
        return "operand sizes do not match";
    case graph_errc::empty_tensor:
        return "empty tensor";
    case graph_errc::bad_output:
        return "invalid graph output";
    case graph_errc::no_outputs:
        return "graph has no outputs";
    }
    return "unknown error";
}

struct graph_error {
    graph_errc code;
    std::size_t node = 0;  ///< index of the node being added when the error occurred
};

class graph_builder;

/// An immutable, validated compute graph (I1-I4).
class compute_graph {
public:
    std::span<const tensor_desc> tensors() const noexcept {
        return tensors_;
    }
    std::span<const node> nodes() const noexcept {
        return nodes_;
    }
    std::span<const tensor_id> inputs() const noexcept {
        return inputs_;
    }
    std::span<const tensor_id> outputs() const noexcept {
        return outputs_;
    }

private:
    friend class graph_builder;
    compute_graph() = default;

    std::vector<tensor_desc> tensors_;
    std::vector<node> nodes_;
    std::vector<tensor_id> inputs_;
    std::vector<tensor_id> outputs_;
};

/// Builds a graph one operation at a time. Each method returns the result
/// tensor; after the first error every method returns invalid_tensor and
/// finalize() reports that error.
class graph_builder {
public:
    /// A graph input of `size` floats.
    tensor_id input(std::size_t size) {
        if (failed()) {
            return invalid_tensor;
        }
        if (size == 0) {
            return fail(graph_errc::empty_tensor);
        }
        const tensor_id t = add_tensor(size * sizeof(float), size, tensor_role::input);
        g_.inputs_.push_back(t);
        return t;
    }

    /// w * x.  Pre (checked): x.size == cols(w).
    tensor_id matmul(const matrix_ref& w, tensor_id x) {
        if (!operand_ok(x)) {
            return invalid_tensor;
        }
        if (size_of(x) != cols(w)) {
            return fail(graph_errc::size_mismatch);
        }
        const std::size_t scratch_bytes = std::visit(
            [](const auto& v) {
                using S = typename std::decay_t<decltype(v)>::scheme_type;
                return kernels::gemv_scratch_bytes<S>(v.cols());
            },
            w);
        const tensor_id out = add_tensor(rows(w) * sizeof(float), rows(w), tensor_role::intermediate);
        const tensor_id scratch = add_tensor(scratch_bytes, 0, tensor_role::scratch);
        g_.nodes_.emplace_back(op_matmul{w, x, out, scratch});
        return out;
    }

    tensor_id add(tensor_id a, tensor_id b) {
        return binary<op_add>(a, b);
    }
    tensor_id mul(tensor_id a, tensor_id b) {
        return binary<op_mul>(a, b);
    }
    tensor_id silu(tensor_id x) {
        return unary<op_silu>(x);
    }
    tensor_id softmax(tensor_id x) {
        return unary<op_softmax>(x);
    }

    /// x / rms(x) * gamma.  Pre (checked): gamma is 1 x x.size.
    tensor_id rms_norm(tensor_id x, const matrix_ref& gamma, float eps) {
        if (!operand_ok(x)) {
            return invalid_tensor;
        }
        if (rows(gamma) != 1 || cols(gamma) != size_of(x)) {
            return fail(graph_errc::size_mismatch);
        }
        const tensor_id out = add_tensor(size_of(x) * sizeof(float), size_of(x), tensor_role::intermediate);
        g_.nodes_.emplace_back(op_rms_norm{x, gamma, eps, out});
        return out;
    }

    /// Mark t as a graph output: its value is kept until the end of a run.
    void output(tensor_id t) {
        if (failed()) {
            return;
        }
        if (t >= g_.tensors_.size() || g_.tensors_[t].role == tensor_role::scratch ||
            g_.tensors_[t].role == tensor_role::output) {
            fail(graph_errc::bad_output);
            return;
        }
        if (g_.tensors_[t].role == tensor_role::intermediate) {
            g_.tensors_[t].role = tensor_role::output;
        }
        g_.outputs_.push_back(t);
    }

    /// The graph, or the first error.  Post: on success, I1-I4 hold.
    result<compute_graph, graph_error> finalize() && {
        if (error_) {
            return *error_;
        }
        if (g_.outputs_.empty()) {
            return graph_error{graph_errc::no_outputs, g_.nodes_.size()};
        }
        return std::move(g_);
    }

private:
    bool failed() const noexcept {
        return error_.has_value();
    }

    tensor_id fail(graph_errc code) {
        if (!error_) {
            error_ = graph_error{code, g_.nodes_.size()};
        }
        return invalid_tensor;
    }

    bool operand_ok(tensor_id t) {
        if (failed()) {
            return false;
        }
        // I1: only tensors that already exist (inputs or earlier results).
        if (t >= g_.tensors_.size() || g_.tensors_[t].role == tensor_role::scratch) {
            fail(graph_errc::invalid_operand);
            return false;
        }
        return true;
    }

    std::size_t size_of(tensor_id t) const noexcept {
        return g_.tensors_[t].size;
    }

    tensor_id add_tensor(std::size_t bytes, std::size_t size, tensor_role role) {
        g_.tensors_.push_back(tensor_desc{bytes, size, role});
        return static_cast<tensor_id>(g_.tensors_.size() - 1);
    }

    template <typename Op>
    tensor_id binary(tensor_id a, tensor_id b) {
        if (!operand_ok(a) || !operand_ok(b)) {
            return invalid_tensor;
        }
        if (size_of(a) != size_of(b)) {
            return fail(graph_errc::size_mismatch);
        }
        const tensor_id out = add_tensor(size_of(a) * sizeof(float), size_of(a), tensor_role::intermediate);
        g_.nodes_.emplace_back(Op{a, b, out});
        return out;
    }

    template <typename Op>
    tensor_id unary(tensor_id x) {
        if (!operand_ok(x)) {
            return invalid_tensor;
        }
        const tensor_id out = add_tensor(size_of(x) * sizeof(float), size_of(x), tensor_role::intermediate);
        g_.nodes_.emplace_back(Op{x, out});
        return out;
    }

    compute_graph g_;
    std::optional<graph_error> error_;
};

}  // namespace branes::synapse::graph

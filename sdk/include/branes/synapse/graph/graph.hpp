#pragma once
/// @file graph.hpp
/// @brief The compute graph of a decode step: tensors, operations, and a checked builder.
///
/// A graph is a DAG of operations over float activation vectors. Weights are
/// not tensors of the graph: they are bound into the operations as typed
/// quantized_matrix_views (zero-copy views into the model file), so the
/// executor never switches on a storage format.
///
/// State that outlives a run (the KV cache) is not a tensor either: a graph
/// declares it as a state, the session allocates it apart from the arena, and
/// the planner never sees it. Operations that depend on the step being
/// decoded (embed reads its token; rope and attention read its position) get
/// it from the session at run time.
///
/// Model
/// -----
///   tensor_desc
///     bytes   : size_t — storage the tensor needs in the arena
///     size    : size_t — float elements per row (0 for kernel scratch)
///     batched : bool   — max_batch rows (one per token of a run) or 1 row
///     role    : input | intermediate | output | scratch
///
///   A graph is built for a max_batch (1 for decode only). A run processes
///   n <= max_batch tokens: batched tensors use their first n rows, and
///   op_last_row reduces a batched tensor to its row n-1, so a prefill
///   computes the final layers (e.g. the vocabulary projection) only once.
///
///   state_desc — a KV cache layer: max_ctx rows of n_kv_heads * head_dim
///     F16 keys and as many values
///
///   node = std::variant<op_matmul, op_add, op_add_bias, op_mul, op_silu,
///                       op_softmax, op_rms_norm, op_embed, op_rope,
///                       op_attention, op_last_row>
///     each op names its operand and result tensors by tensor_id; op_matmul
///     and op_attention also name a scratch tensor, and op_attention the
///     state it reads and appends to
///
///   compute_graph
///     tensors_ : vector<tensor_desc>
///     states_  : vector<state_desc>
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
///   I5: the sum of all tensor sizes, each rounded up to plan_alignment, is
///       at most max_graph_bytes (sizes come from untrusted metadata; this
///       keeps every byte count and arena offset free of overflow); the
///       same bound holds for the sum of all state bytes
///                                                   — constrains tensors_, states_
///   I6: every head layout fits its operation: tensor sizes are whole heads,
///       head_dim is even (rope) and at most kernels::max_head_dim
///       (attention), and the query heads divide evenly among the KV heads
///                                                   — constrains nodes_, states_
///   I7: every state is read and written by at most one attention node
///                                                   — constrains nodes_, states_
///   I8: an op's operands and result agree in batching: element-wise ops
///       take operands that are both batched or both single-row (add_bias
///       and rms_norm keep their operand's batching); embed,
///       rope and attention work on batched tensors; last_row turns a
///       batched tensor into a single-row one
///                                                   — constrains nodes_, tensors_
///
/// The builder is no-throw on bad input: graphs are built from model
/// metadata (untrusted), so it records the first error and finalize()
/// returns it, with the node index where it occurred.

#include <branes/synapse/kernels/attention.hpp>
#include <branes/synapse/kernels/gemm.hpp>
#include <branes/synapse/kernels/gemv.hpp>
#include <branes/synapse/quant/scheme.hpp>
#include <branes/synapse/result.hpp>
#include <branes/synapse/tensor/quantized_view.hpp>

#include <algorithm>
#include <cmath>
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

/// Alignment of every tensor in the arena: a cache line, which also covers
/// any SIMD load width and every scratch block type.
inline constexpr std::size_t plan_alignment = 64;

/// Upper bound on a graph's total (aligned) tensor bytes. Every arena offset
/// the planner computes is at most this total, so with the bound at half of
/// size_t no planner arithmetic can wrap.
inline constexpr std::size_t max_graph_bytes = std::numeric_limits<std::size_t>::max() / 2;

/// The id the builder returns after an error; operations on it are no-ops.
inline constexpr tensor_id invalid_tensor = std::numeric_limits<tensor_id>::max();

/// Index of a state (a KV cache layer) in its graph.
using state_id = std::uint32_t;
inline constexpr state_id invalid_state = std::numeric_limits<state_id>::max();

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
    std::size_t size = 0;  ///< float elements per row; 0 for scratch
    tensor_role role = tensor_role::intermediate;
    bool batched = true;  ///< max_batch rows, or a single row
};

/// A KV cache layer, allocated by the session and kept across runs.
struct state_desc {
    std::size_t bytes = 0;  ///< 2 * max_ctx * n_kv_heads * head_dim halves
    std::size_t max_ctx = 0;
    std::size_t n_kv_heads = 0;
    std::size_t head_dim = 0;
};

struct op_matmul {  ///< out = w * in (gemv)
    matrix_ref w;
    tensor_id in, out, scratch;
};
struct op_add {  ///< out = a + b
    tensor_id a, b, out;
};
struct op_add_bias {  ///< out = in + bias, the bias added to every row
    tensor_id in;
    matrix_ref bias;  ///< 1 x size
    tensor_id out;
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

struct op_embed {  ///< out = row `token` of table (the step's token)
    matrix_ref table;
    tensor_id out;
};
struct op_rope {  ///< out = in with each head rotated by the step's position
    tensor_id in, out;
    std::size_t head_dim;
    float theta_scale;                ///< base^(-2 / head_dim)
    std::vector<float> freq_factors;  ///< empty, or head_dim / 2 divisors of theta
    kernels::rope_mode mode;          ///< which elements form a pair
};
struct op_attention {  ///< append (k, v) to cache; out = causal attention of q over it
    tensor_id q, k, v, out, scratch;
    state_id cache;
    std::size_t n_heads;
    float scale;
};

struct op_last_row {  ///< out = row n-1 of in (the last token of the run)
    tensor_id in, out;
};

using node = std::variant<op_matmul,
                          op_add,
                          op_add_bias,
                          op_mul,
                          op_silu,
                          op_softmax,
                          op_rms_norm,
                          op_embed,
                          op_rope,
                          op_attention,
                          op_last_row>;

/// Operand (read) tensors of a node.
inline std::vector<tensor_id> operands(const node& n) {
    struct visitor {
        std::vector<tensor_id> operator()(const op_matmul& o) const {
            return {o.in};
        }
        std::vector<tensor_id> operator()(const op_add& o) const {
            return {o.a, o.b};
        }
        std::vector<tensor_id> operator()(const op_add_bias& o) const {
            return {o.in};
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
        std::vector<tensor_id> operator()(const op_embed&) const {
            return {};
        }
        std::vector<tensor_id> operator()(const op_rope& o) const {
            return {o.in};
        }
        std::vector<tensor_id> operator()(const op_attention& o) const {
            return {o.q, o.k, o.v};
        }
        std::vector<tensor_id> operator()(const op_last_row& o) const {
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
    if (const auto* a = std::get_if<op_attention>(&n)) {
        return {a->out, a->scratch};
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
    too_large,            ///< the graph's tensors would exceed max_graph_bytes
    bad_shape,            ///< a head layout that does not fit its operation (I6)
    bad_parameter,        ///< a non-finite or non-positive scalar parameter
    bad_state,            ///< a state that is not in the graph, or already in use (I7)
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
    case graph_errc::too_large:
        return "graph tensors too large";
    case graph_errc::bad_shape:
        return "head layout does not fit the operation";
    case graph_errc::bad_parameter:
        return "invalid scalar parameter";
    case graph_errc::bad_state:
        return "invalid or reused state";
    }
    return "unknown error";
}

struct graph_error {
    graph_errc code;
    std::size_t node = 0;  ///< index of the node being added when the error occurred
};

class graph_builder;

/// An immutable, validated compute graph (I1-I8).
class compute_graph {
public:
    std::span<const tensor_desc> tensors() const noexcept {
        return tensors_;
    }
    std::span<const state_desc> states() const noexcept {
        return states_;
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
    /// Tokens a run accepts are < token_limit(): the smallest embedding
    /// table's row count (unbounded if the graph embeds nothing).
    std::size_t token_limit() const noexcept {
        return token_limit_;
    }
    /// Positions a run accepts are < position_limit(): the smallest KV cache's
    /// max_ctx (unbounded if the graph has no attention).
    std::size_t position_limit() const noexcept {
        return position_limit_;
    }
    /// Tokens a run may process at once.
    std::size_t max_batch() const noexcept {
        return max_batch_;
    }
    /// Rows a tensor holds: max_batch() if batched, else 1.
    std::size_t rows(tensor_id t) const noexcept {
        return tensors_[t].batched ? max_batch_ : 1;
    }

private:
    friend class graph_builder;
    compute_graph() = default;

    std::vector<tensor_desc> tensors_;
    std::vector<state_desc> states_;
    std::size_t token_limit_ = std::numeric_limits<std::size_t>::max();
    std::size_t position_limit_ = std::numeric_limits<std::size_t>::max();
    std::size_t max_batch_ = 1;
    std::vector<node> nodes_;
    std::vector<tensor_id> inputs_;
    std::vector<tensor_id> outputs_;
};

// GCC 13 reports a false -Wstringop-overflow ("writing 1 byte into a region of
// size 0") inside std::vector's reallocation once the builder's push_back calls
// are inlined at -O3. Verified: GCC 13.5 warns; GCC 14.4 and 15.2 do not. GCC 12
// is untested but shares the analysis, so it is included. The warning would
// fail -Werror builds of every consumer, so it is silenced here, for this class
// only and only on those compilers.
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ == 12 || __GNUC__ == 13)
#define SYNAPSE_GCC_STRINGOP_WORKAROUND 1
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif

/// Builds a graph one operation at a time. Each method returns the result
/// tensor; after the first error every method returns invalid_tensor and
/// finalize() reports that error.
class graph_builder {
public:
    /// A builder for graphs that process up to max_batch tokens per run.
    /// max_batch == 0 is recorded as a bad_shape error.
    explicit graph_builder(std::size_t max_batch = 1) {
        if (max_batch == 0) {
            fail(graph_errc::bad_shape);
        } else {
            g_.max_batch_ = max_batch;
        }
    }

    /// A graph input of `size` floats per token (batched).
    tensor_id input(std::size_t size) {
        if (failed()) {
            return invalid_tensor;
        }
        if (size == 0) {
            return fail(graph_errc::empty_tensor);
        }
        const tensor_id t = add_floats(size, tensor_role::input, true);
        if (t != invalid_tensor) {
            g_.inputs_.push_back(t);
        }
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
        // gemm scratch for every row of x; n * (per-row bytes) must not wrap.
        const std::size_t n = g_.rows(x);
        const std::optional<std::size_t> scratch_bytes = std::visit(
            [n](const auto& v) -> std::optional<std::size_t> {
                using S = typename std::decay_t<decltype(v)>::scheme_type;
                using A = typename S::vec_dot_block;
                const std::size_t per_row =
                    kernels::detail::gemm_row_stride<S>(v.cols()) + sizeof(kernels::activations<A>) + sizeof(float);
                if (n > (max_graph_bytes - kernels::gemm_scratch_alignment) / per_row) {
                    return std::nullopt;
                }
                return kernels::gemm_scratch_bytes<S>(v.cols(), n);
            },
            w);
        if (!scratch_bytes) {
            return fail(graph_errc::too_large);
        }
        const tensor_id out = add_floats(rows(w), tensor_role::intermediate, batched(x));
        const tensor_id scratch = add_tensor(*scratch_bytes, 0, tensor_role::scratch, false);
        if (out == invalid_tensor || scratch == invalid_tensor) {
            return invalid_tensor;
        }
        g_.nodes_.emplace_back(op_matmul{w, x, out, scratch});
        return out;
    }

    tensor_id add(tensor_id a, tensor_id b) {
        return binary<op_add>(a, b);
    }
    tensor_id mul(tensor_id a, tensor_id b) {
        return binary<op_mul>(a, b);
    }

    /// x + bias, the bias added to each row of x.
    ///   Pre (checked): bias is 1 x x.size (I2). The result has x's batching (I8).
    tensor_id add_bias(tensor_id x, const matrix_ref& bias) {
        if (!operand_ok(x)) {
            return invalid_tensor;
        }
        if (rows(bias) != 1 || cols(bias) != size_of(x)) {
            return fail(graph_errc::size_mismatch);
        }
        const tensor_id out = add_floats(size_of(x), tensor_role::intermediate, batched(x));
        if (out != invalid_tensor) {
            g_.nodes_.emplace_back(op_add_bias{x, bias, out});
        }
        return out;
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
        const tensor_id out = add_floats(size_of(x), tensor_role::intermediate, batched(x));
        if (out != invalid_tensor) {
            g_.nodes_.emplace_back(op_rms_norm{x, gamma, eps, out});
        }
        return out;
    }

    /// Row `token` of table, for the token of each run.  Pre (checked): table non-empty.
    tensor_id embed(const matrix_ref& table) {
        if (failed()) {
            return invalid_tensor;
        }
        if (rows(table) == 0 || cols(table) == 0) {
            return fail(graph_errc::empty_tensor);
        }
        const tensor_id out = add_floats(cols(table), tensor_role::intermediate, true);
        if (out != invalid_tensor) {
            g_.token_limit_ = std::min(g_.token_limit_, rows(table));
            g_.nodes_.emplace_back(op_embed{table, out});
        }
        return out;
    }

    /// x, viewed as heads of head_dim floats, each rotated by the run's
    /// position with theta_i = pos * base^(-2i / head_dim) / freq_factors[i],
    /// pairing elements as mode says.
    ///   Pre (checked): I6 (head_dim even, x.size a multiple of it); base and
    ///   every freq factor finite and positive; freq_factors empty or of size
    ///   head_dim / 2.
    tensor_id rope(tensor_id x,
                   std::size_t head_dim,
                   float base,
                   std::span<const float> freq_factors = {},
                   kernels::rope_mode mode = kernels::rope_mode::normal) {
        if (!operand_ok(x)) {
            return invalid_tensor;
        }
        if (head_dim == 0 || head_dim % 2 != 0 || size_of(x) % head_dim != 0 ||
            (!freq_factors.empty() && freq_factors.size() != head_dim / 2) || !batched(x)) {
            return fail(graph_errc::bad_shape);  // I6, I8
        }
        const auto positive = [](float f) { return std::isfinite(f) && f > 0.0f; };
        if (!positive(base) || !std::all_of(freq_factors.begin(), freq_factors.end(), positive)) {
            return fail(graph_errc::bad_parameter);
        }
        const tensor_id out = add_floats(size_of(x), tensor_role::intermediate, true);
        if (out != invalid_tensor) {
            g_.nodes_.emplace_back(op_rope{x,
                                           out,
                                           head_dim,
                                           kernels::rope_theta_scale(base, head_dim),
                                           std::vector<float>(freq_factors.begin(), freq_factors.end()),
                                           mode});
        }
        return out;
    }

    /// A KV cache layer of max_ctx positions.
    ///   Pre (checked): all sizes non-zero, head_dim <= kernels::max_head_dim
    ///   (I6), max_ctx representable as a position, and the bytes within I5.
    state_id kv_cache(std::size_t max_ctx, std::size_t n_kv_heads, std::size_t head_dim) {
        if (failed()) {
            return invalid_state;
        }
        if (max_ctx == 0 || n_kv_heads == 0 || head_dim == 0 || head_dim > kernels::max_head_dim ||
            max_ctx > std::size_t{std::numeric_limits<std::uint32_t>::max()} + 1) {
            fail(graph_errc::bad_shape);
            return invalid_state;
        }
        // I5: 2 (K and V) * max_ctx * n_kv_heads * head_dim halves, unwrapped.
        constexpr std::size_t half = sizeof(std::uint16_t);
        const std::size_t limit = max_graph_bytes - state_bytes_;
        if (n_kv_heads > limit / head_dim || max_ctx > limit / (n_kv_heads * head_dim) / (2 * half)) {
            fail(graph_errc::too_large);
            return invalid_state;
        }
        const std::size_t bytes = 2 * half * max_ctx * n_kv_heads * head_dim;
        state_bytes_ += bytes;
        g_.states_.push_back(state_desc{bytes, max_ctx, n_kv_heads, head_dim});
        state_used_.push_back(false);
        return static_cast<state_id>(g_.states_.size() - 1);
    }

    /// Append (k, v) to cache at the run's position, then attend q, as
    /// n_heads query heads, to every cached position up to it.
    ///   Pre (checked): cache is a state of this graph not used by another
    ///   attention (I7); q.size == n_heads * head_dim, k.size == v.size ==
    ///   n_kv_heads * head_dim, n_heads a multiple of n_kv_heads (I6); scale
    ///   finite and positive.
    tensor_id attention(tensor_id q, tensor_id k, tensor_id v, state_id cache, std::size_t n_heads, float scale) {
        if (!operand_ok(q) || !operand_ok(k) || !operand_ok(v)) {
            return invalid_tensor;
        }
        if (cache >= g_.states_.size() || state_used_[cache]) {
            return fail(graph_errc::bad_state);  // I7
        }
        const state_desc& c = g_.states_[cache];
        const std::size_t row = c.n_kv_heads * c.head_dim;
        if (n_heads == 0 || n_heads % c.n_kv_heads != 0 || n_heads > max_graph_bytes / c.head_dim ||
            size_of(q) != n_heads * c.head_dim || size_of(k) != row || size_of(v) != row || !batched(q) ||
            !batched(k) || !batched(v)) {
            return fail(graph_errc::bad_shape);  // I6, I8
        }
        if (!std::isfinite(scale) || scale <= 0.0f) {
            return fail(graph_errc::bad_parameter);
        }
        // attention_scratch_floats = n_heads * (max_ctx + head_dim) (one slot
        // per KV group), unwrapped.
        const std::size_t per_head = c.max_ctx + c.head_dim;  // both bounded by kv_cache
        if (per_head > max_graph_bytes / sizeof(float) / n_heads) {
            return fail(graph_errc::too_large);
        }
        const std::size_t scratch_floats =
            kernels::attention_scratch_floats(n_heads, c.n_kv_heads, c.head_dim, c.max_ctx);
        const tensor_id out = add_floats(size_of(q), tensor_role::intermediate, true);
        const tensor_id scratch = add_tensor(scratch_floats * sizeof(float), 0, tensor_role::scratch, false);
        if (out == invalid_tensor || scratch == invalid_tensor) {
            return invalid_tensor;
        }
        state_used_[cache] = true;
        g_.position_limit_ = std::min(g_.position_limit_, c.max_ctx);
        g_.nodes_.emplace_back(op_attention{q, k, v, out, scratch, cache, n_heads, scale});
        return out;
    }

    /// Row n-1 of x for a run of n tokens: the last token's values.
    ///   Pre (checked): x is batched (I8).
    tensor_id last_row(tensor_id x) {
        if (!operand_ok(x)) {
            return invalid_tensor;
        }
        if (!batched(x)) {
            return fail(graph_errc::bad_shape);  // I8
        }
        const tensor_id out = add_floats(size_of(x), tensor_role::intermediate, false);
        if (out != invalid_tensor) {
            g_.nodes_.emplace_back(op_last_row{x, out});
        }
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

    /// The graph, or the first error.  Post: on success, I1-I8 hold.
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
    bool batched(tensor_id t) const noexcept {
        return g_.tensors_[t].batched;
    }

    /// Append a float tensor of `size` elements per row, max_batch rows if
    /// batched. The only place a float tensor's byte count is computed, so
    /// the multiplication is checked once for every op: sizes can come from
    /// untrusted metadata (an input) or from a weight view's row count, and
    /// max_batch from the caller, any of which can be large enough to wrap.
    tensor_id add_floats(std::size_t size, tensor_role role, bool is_batched) {
        const std::size_t n = is_batched ? g_.max_batch_ : 1;
        if (size > max_graph_bytes / sizeof(float) / n) {
            return fail(graph_errc::too_large);  // I5: n * size * sizeof(float) must not wrap
        }
        return add_tensor(n * size * sizeof(float), size, role, is_batched);
    }

    /// Append a tensor of `bytes` bytes, or fail with too_large if the graph
    /// would exceed max_graph_bytes (I5).
    tensor_id add_tensor(std::size_t bytes, std::size_t size, tensor_role role, bool is_batched) {
        if (bytes > max_graph_bytes - total_bytes_ ||
            (bytes + plan_alignment - 1) / plan_alignment * plan_alignment > max_graph_bytes - total_bytes_) {
            return fail(graph_errc::too_large);
        }
        total_bytes_ += (bytes + plan_alignment - 1) / plan_alignment * plan_alignment;
        g_.tensors_.push_back(tensor_desc{bytes, size, role, is_batched});
        return static_cast<tensor_id>(g_.tensors_.size() - 1);
    }

    template <typename Op>
    tensor_id binary(tensor_id a, tensor_id b) {
        if (!operand_ok(a) || !operand_ok(b)) {
            return invalid_tensor;
        }
        if (size_of(a) != size_of(b) || batched(a) != batched(b)) {
            return fail(graph_errc::size_mismatch);  // I2, I8
        }
        const tensor_id out = add_floats(size_of(a), tensor_role::intermediate, batched(a));
        if (out != invalid_tensor) {
            g_.nodes_.emplace_back(Op{a, b, out});
        }
        return out;
    }

    template <typename Op>
    tensor_id unary(tensor_id x) {
        if (!operand_ok(x)) {
            return invalid_tensor;
        }
        const tensor_id out = add_floats(size_of(x), tensor_role::intermediate, batched(x));
        if (out != invalid_tensor) {
            g_.nodes_.emplace_back(Op{x, out});
        }
        return out;
    }

    compute_graph g_;
    std::optional<graph_error> error_;
    std::size_t total_bytes_ = 0;   ///< sum of aligned tensor bytes (I5)
    std::size_t state_bytes_ = 0;   ///< sum of state bytes (I5)
    std::vector<bool> state_used_;  ///< by an attention node (I7)
};

#if defined(SYNAPSE_GCC_STRINGOP_WORKAROUND)
#pragma GCC diagnostic pop
#undef SYNAPSE_GCC_STRINGOP_WORKAROUND
#endif

}  // namespace branes::synapse::graph

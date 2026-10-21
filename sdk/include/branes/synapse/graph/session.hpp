#pragma once
/// @file session.hpp
/// @brief Executes a planned graph in one preallocated arena.
///
/// A session plans its graph and allocates the arena once, at construction.
/// run() then walks the nodes in order (a topological order, by I1) and
/// dispatches each to its kernel, with every tensor at its planned offset.
/// run() performs no allocation and cannot throw: the property that lets an
/// inference step sit inside a real-time loop.
///
///   session s{g};                       // plan + one allocation
///   copy(x, s.tensor(in));              // write the inputs
///   s.run();                            // no allocation
///   read(s.tensor(out));                // read the outputs
///
/// Model
/// -----
///   session
///     graph_ : const compute_graph* [non-owning; must outlive the session]
///     plan_  : memory_plan
///     arena_ : unique_ptr<byte[]> [owned], plan_.arena_bytes, aligned to plan_alignment
///
/// Invariants
/// ----------
///   S1: plan_ satisfies P1-P3 for *graph_
///   S2: arena_ holds at least plan_.arena_bytes bytes, aligned to plan_alignment
///
/// Inputs must be written before every run: once an input's last reader has
/// run, the planner may reuse its memory. Outputs stay valid until the next
/// run.

#include <branes/synapse/graph/graph.hpp>
#include <branes/synapse/graph/planner.hpp>
#include <branes/synapse/kernels/elementwise.hpp>
#include <branes/synapse/kernels/gemv.hpp>

#include <cassert>
#include <cstddef>
#include <memory>
#include <new>
#include <span>
#include <variant>

namespace branes::synapse::graph {

class session {
public:
    /// Plans g and allocates the arena.  Post: S1, S2.
    explicit session(const compute_graph& g) : graph_(&g), plan_(plan(g)), arena_(allocate(plan_.arena_bytes)) {
        assert(plan_is_valid(g, plan_));  // S1
    }

    session(const session&) = delete;
    session& operator=(const session&) = delete;
    session(session&&) noexcept = default;
    session& operator=(session&&) noexcept = default;

    /// The float storage of tensor t.  Pre: t is a non-scratch tensor of the graph.
    std::span<float> tensor(tensor_id t) noexcept {
        assert(t < graph_->tensors().size() && graph_->tensors()[t].role != tensor_role::scratch);  // Pre
        return {reinterpret_cast<float*>(arena_.get() + plan_.offset[t]), graph_->tensors()[t].size};
    }

    const memory_plan& memory() const noexcept {
        return plan_;
    }

    /// Execute every node in order.  Pre: the inputs have been written.
    void run() noexcept {
        for (const node& n : graph_->nodes()) {
            std::visit([this](const auto& op) { execute(op); }, n);
        }
    }

private:
    struct arena_deleter {
        void operator()(std::byte* p) const noexcept {
            ::operator delete[](p, std::align_val_t{plan_alignment});
        }
    };
    using arena_ptr = std::unique_ptr<std::byte[], arena_deleter>;

    static arena_ptr allocate(std::size_t bytes) {
        // At least one byte, so an all-scratch-free empty plan still has storage.
        auto* p = static_cast<std::byte*>(::operator new[](bytes > 0 ? bytes : 1, std::align_val_t{plan_alignment}));
        return arena_ptr(p);
    }

    std::span<std::byte> bytes(tensor_id t) noexcept {
        return {arena_.get() + plan_.offset[t], graph_->tensors()[t].bytes};
    }

    void execute(const op_matmul& o) noexcept {
        const std::span<const float> in = tensor(o.in);
        const std::span<float> out = tensor(o.out);
        const std::span<std::byte> scratch = bytes(o.scratch);
        std::visit([&](const auto& w) { kernels::gemv(w, in, out, scratch); }, o.w);
    }
    void execute(const op_add& o) noexcept {
        kernels::add(tensor(o.a), tensor(o.b), tensor(o.out));
    }
    void execute(const op_mul& o) noexcept {
        kernels::mul(tensor(o.a), tensor(o.b), tensor(o.out));
    }
    void execute(const op_silu& o) noexcept {
        kernels::silu(tensor(o.in), tensor(o.out));
    }
    void execute(const op_softmax& o) noexcept {
        kernels::softmax(tensor(o.in), tensor(o.out));
    }
    void execute(const op_rms_norm& o) noexcept {
        const std::span<const float> in = tensor(o.in);
        const std::span<float> out = tensor(o.out);
        std::visit([&](const auto& g) { kernels::rms_norm(in, g, o.eps, out); }, o.gamma);
    }

    const compute_graph* graph_;
    memory_plan plan_;
    arena_ptr arena_;
};

}  // namespace branes::synapse::graph

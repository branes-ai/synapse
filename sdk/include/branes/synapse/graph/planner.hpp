#pragma once
/// @file planner.hpp
/// @brief Static memory planning: liveness analysis and arena offsets.
///
/// Every activation and scratch tensor of a graph is placed at a fixed offset
/// in one arena, computed once before the first run. Tensors whose lifetimes
/// do not overlap share memory, so a deep network runs in roughly the
/// footprint of its widest few layers.
///
/// Liveness: tensor t lives over the closed node-index interval
/// [first(t), last(t)]. A node's result starts at that node; an input starts
/// at node 0. A tensor ends at the last node that reads it, or at the end of
/// the graph if it is an output. A node's operands and results are all live
/// at that node, so a result never aliases an operand of its own node.
///
/// Placement: greedy by size (as in TensorFlow Lite's arena planner). Tensors
/// are placed largest first, each at the lowest aligned offset that does not
/// overlap a placed tensor whose interval intersects its own.
///
/// Model
/// -----
///   memory_plan
///     offset      : vector<size_t> — byte offset of each tensor in the arena
///     arena_bytes : size_t — arena size
///     naive_bytes : size_t — sum of all tensor sizes (the no-aliasing cost)
///
/// Invariants (of plan(g), checked by plan_is_valid)
/// ----------
///   P1: offset[t] % plan_alignment == 0 for every tensor
///   P2: offset[t] + bytes(t) <= arena_bytes for every tensor
///   P3: tensors with intersecting live intervals have disjoint byte ranges

#include <branes/synapse/graph/graph.hpp>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

namespace branes::synapse::graph {

/// Alignment of every tensor in the arena: a cache line, which also covers
/// any SIMD load width and every scratch block type.
inline constexpr std::size_t plan_alignment = 64;

struct live_interval {
    std::size_t first = 0;
    std::size_t last = 0;
};

inline bool intersects(live_interval a, live_interval b) noexcept {
    return a.first <= b.last && b.first <= a.last;
}

/// Live interval of every tensor of g.
inline std::vector<live_interval> liveness(const compute_graph& g) {
    const std::size_t n_nodes = g.nodes().size();
    const std::size_t end = n_nodes == 0 ? 0 : n_nodes - 1;
    std::vector<live_interval> live(g.tensors().size());
    for (std::size_t i = 0; i < n_nodes; ++i) {
        for (tensor_id t : results(g.nodes()[i])) {
            live[t] = {i, i};
        }
    }
    for (tensor_id t : g.inputs()) {
        live[t] = {0, 0};
    }
    for (std::size_t i = 0; i < n_nodes; ++i) {
        for (tensor_id t : operands(g.nodes()[i])) {
            live[t].last = std::max(live[t].last, i);
        }
    }
    for (tensor_id t : g.outputs()) {
        live[t].last = end;
    }
    return live;
}

struct memory_plan {
    std::vector<std::size_t> offset;
    std::size_t arena_bytes = 0;
    std::size_t naive_bytes = 0;
};

namespace detail {

constexpr std::size_t align_up(std::size_t n) noexcept {
    return (n + plan_alignment - 1) / plan_alignment * plan_alignment;
}

}  // namespace detail

/// Arena offsets for every tensor of g.  Post: P1-P3.
inline memory_plan plan(const compute_graph& g) {
    const auto tensors = g.tensors();
    const auto live = liveness(g);

    std::vector<tensor_id> order(tensors.size());
    std::iota(order.begin(), order.end(), tensor_id{0});
    std::stable_sort(order.begin(), order.end(), [&](tensor_id a, tensor_id b) {
        return tensors[a].bytes > tensors[b].bytes;  // largest first; ties keep id order
    });

    memory_plan p;
    p.offset.assign(tensors.size(), 0);
    std::vector<tensor_id> placed;
    std::vector<tensor_id> conflicts;
    for (tensor_id t : order) {
        p.naive_bytes += detail::align_up(tensors[t].bytes);
        // Placed tensors that are live at the same time as t, by offset.
        conflicts.clear();
        for (tensor_id u : placed) {
            if (intersects(live[t], live[u])) {
                conflicts.push_back(u);
            }
        }
        std::sort(
            conflicts.begin(), conflicts.end(), [&](tensor_id a, tensor_id b) { return p.offset[a] < p.offset[b]; });
        // Lowest aligned gap that fits.
        std::size_t candidate = 0;
        for (tensor_id u : conflicts) {
            if (candidate + tensors[t].bytes <= p.offset[u]) {
                break;
            }
            candidate = std::max(candidate, detail::align_up(p.offset[u] + tensors[u].bytes));
        }
        p.offset[t] = candidate;
        p.arena_bytes = std::max(p.arena_bytes, detail::align_up(candidate + tensors[t].bytes));
        placed.push_back(t);
    }
    return p;
}

/// Checks P1-P3 (exhaustively, O(tensors^2)): for tests and debug asserts.
inline bool plan_is_valid(const compute_graph& g, const memory_plan& p) {
    const auto tensors = g.tensors();
    const auto live = liveness(g);
    if (p.offset.size() != tensors.size()) {
        return false;
    }
    for (std::size_t t = 0; t < tensors.size(); ++t) {
        if (p.offset[t] % plan_alignment != 0 || p.offset[t] + tensors[t].bytes > p.arena_bytes) {
            return false;  // P1, P2
        }
        for (std::size_t u = t + 1; u < tensors.size(); ++u) {
            const bool disjoint =
                p.offset[t] + tensors[t].bytes <= p.offset[u] || p.offset[u] + tensors[u].bytes <= p.offset[t];
            if (intersects(live[t], live[u]) && !disjoint && tensors[t].bytes > 0 && tensors[u].bytes > 0) {
                return false;  // P3
            }
        }
    }
    return true;
}

}  // namespace branes::synapse::graph

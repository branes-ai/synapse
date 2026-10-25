#pragma once
/// @file session.hpp
/// @brief Executes a planned graph in one preallocated arena.
///
/// A session plans its graph and allocates the arena and the graph's states
/// (KV caches) once, at construction. run() then walks the nodes in order (a
/// topological order, by I1) and dispatches each to its kernel, with every
/// tensor at its planned offset. run() performs no allocation and cannot
/// throw: the property that lets an inference step sit inside a real-time
/// loop.
///
///   session s{g};                       // plan + two allocations
///   copy(x, s.tensor(in));              // write the inputs
///   s.run({token, pos});                // one token; no allocation
///   s.run(batch{tokens, pos});          // up to max_batch tokens (prefill)
///   s.run_rows(n);                      // n rows of a graph that embeds nothing (an encoder)
///   read(s.tensor(out));                // read the outputs
///
/// Model
/// -----
///   session
///     graph_  : const compute_graph* [non-owning; must outlive the session]
///     plan_   : memory_plan
///     arena_  : unique_ptr<byte[]> [owned], plan_.arena_bytes, aligned to plan_alignment
///     states_ : unique_ptr<byte[]> [owned], every state of *graph_, each at
///               state_offset_[s], aligned to plan_alignment; zeroed at construction
///
/// Invariants
/// ----------
///   S1: plan_ satisfies P1-P3 for *graph_
///   S2: arena_ holds at least plan_.arena_bytes bytes, aligned to plan_alignment
///   S3: states_ holds every state's bytes at its offset, disjoint and aligned
///
/// Inputs must be written before every run: once an input's last reader has
/// run, the planner may reuse its memory. Outputs stay valid until the next
/// run. A sequence is decoded by running positions 0, 1, 2, ... in order, one
/// token or a batch at a time: attention at position p reads the cache rows
/// earlier runs (and earlier tokens of the same batch) wrote. A run of n
/// tokens uses the first n rows of each batched tensor (n / row_div below a
/// pixel_shuffle); by G1 and the order of the per-row kernels, its results
/// equal running the tokens one by one, except across a self_attention or
/// pixel_shuffle, which mix rows, so graphs with a pixel_shuffle accept only
/// full runs (I9).

#include <branes/synapse/graph/graph.hpp>
#include <branes/synapse/graph/planner.hpp>
#include <branes/synapse/kernels/attention.hpp>
#include <branes/synapse/kernels/elementwise.hpp>
#include <branes/synapse/kernels/gemm.hpp>

#include <mtl/detail/thread_pool.hpp>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <span>
#include <variant>
#include <vector>

namespace branes::synapse::graph {

/// What a run decodes: the token fed to embed nodes and its position in the
/// sequence (read by rope and attention nodes).
struct step {
    std::uint32_t token = 0;
    std::uint32_t pos = 0;
};

/// A run of tokens.size() consecutive tokens, the first at position pos.
/// Graphs without embed nodes ignore the token values but still process
/// tokens.size() rows of their inputs.
struct batch {
    std::span<const std::uint32_t> tokens;
    std::uint32_t pos = 0;
};

class session {
public:
    /// Plans g and allocates the arena and the states.  Post: S1, S2, S3.
    ///
    /// threads: how many of MTL5's process-wide pool threads a run may use
    /// (the pool's size is MTL5_NUM_THREADS, read once). 1, the default,
    /// never touches the pool; 0 means the whole pool. Results do not depend
    /// on it (D1).
    explicit session(const compute_graph& g, unsigned threads = 1)
        : graph_(&g), plan_(plan(g)), arena_(allocate(plan_.arena_bytes)), state_offset_(state_offsets(g)),
          states_(allocate(state_offset_.back())), threads_(team_size(threads)) {
        assert(plan_is_valid(g, plan_));  // S1
        reset();
    }
    /// The session keeps a pointer to its graph: a temporary would dangle.
    session(const compute_graph&&, unsigned = 1) = delete;

    session(const session&) = delete;
    session& operator=(const session&) = delete;
    session(session&&) noexcept = default;
    session& operator=(session&&) noexcept = default;

    /// The float storage of tensor t: rows(t) rows of size floats; a run of
    /// n tokens uses the first n rows of a batched tensor.
    ///   Pre: t is a non-scratch tensor of the graph.
    std::span<float> tensor(tensor_id t) noexcept {
        assert(t < graph_->tensors().size() && graph_->tensors()[t].role != tensor_role::scratch);  // Pre
        return {reinterpret_cast<float*>(arena_.get() + plan_.offset[t]), graph_->rows(t) * graph_->tensors()[t].size};
    }

    const memory_plan& memory() const noexcept {
        return plan_;
    }

    /// Threads a run uses (1 to the pool's size).
    unsigned threads() const noexcept {
        return threads_;
    }

    /// Bytes held by the graph's states (the KV cache).
    std::size_t state_bytes() const noexcept {
        return state_offset_.back();
    }

    /// Zero every state, as before the first run of a new sequence.
    void reset() noexcept {
        std::memset(states_.get(), 0, state_offset_.back());
    }

    /// Execute every node in order for one token.  Same as run(batch) with
    /// a single token.
    [[nodiscard]] bool run(step s = {}) noexcept {
        return run(batch{std::span<const std::uint32_t>(&s.token, 1), s.pos});
    }

    /// Execute every node in order for the tokens of b.  Returns false, and
    /// runs nothing, unless 1 <= n <= max_batch() (n == max_batch() if
    /// full_batch_only(), I9), pos + n <= position_limit() and every token <
    /// token_limit() (n = b.tokens.size()).
    ///   Pre: the inputs' first n rows have been written; positions
    ///   0..b.pos-1 of the current sequence have been run.
    [[nodiscard]] bool run(batch b) noexcept {
        const std::size_t n = b.tokens.size();
        if (!accepts(n, b.pos)) {
            return false;
        }
        for (std::uint32_t t : b.tokens) {
            if (t >= graph_->token_limit()) {
                return false;
            }
        }
        batch_ = b;
        execute_all(n);
        return true;
    }

    /// Execute every node in order for n rows at position 0, for a graph that
    /// embeds nothing (an image encoder, whose rows are patches). Returns
    /// false, and runs nothing, if the graph embeds, or unless 1 <= n <=
    /// max_batch() (n == max_batch() if full_batch_only()) and n <=
    /// position_limit().
    ///   Pre: the inputs' first n rows have been written.
    [[nodiscard]] bool run_rows(std::size_t n) noexcept {
        if (graph_->embeds() || !accepts(n, 0)) {
            return false;
        }
        batch_ = batch{{}, 0};
        execute_all(n);
        return true;
    }

private:
    bool accepts(std::size_t n, std::uint32_t pos) const noexcept {
        return n != 0 && n <= graph_->max_batch() && (!graph_->full_batch_only() || n == graph_->max_batch()) &&
               std::size_t{pos} + n <= graph_->position_limit();
    }

    void execute_all(std::size_t n) noexcept {
        n_ = n;
        for (const node& nd : graph_->nodes()) {
            std::visit([this](const auto& op) { execute(op); }, nd);
        }
    }

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

    /// Aligned offset of each state, followed by the total. I5 bounds the
    /// unaligned sum by max_graph_bytes; each state adds under plan_alignment
    /// of padding, far from wrapping.
    static std::vector<std::size_t> state_offsets(const compute_graph& g) {
        std::vector<std::size_t> offset;
        offset.reserve(g.states().size() + 1);
        std::size_t total = 0;
        for (const state_desc& d : g.states()) {
            offset.push_back(total);
            total += detail::align_up(d.bytes);
        }
        offset.push_back(total);
        return offset;
    }

    kernels::kv_layer cache(state_id c) noexcept {
        const state_desc& d = graph_->states()[c];
        auto* k = reinterpret_cast<std::uint16_t*>(states_.get() + state_offset_[c]);
        return {k, k + d.max_ctx * d.n_kv_heads * d.head_dim, d.max_ctx, d.n_kv_heads, d.head_dim};
    }

    std::span<std::byte> bytes(tensor_id t) noexcept {
        return {arena_.get() + plan_.offset[t], graph_->tensors()[t].bytes};
    }

    /// Rows of t this run uses: n / row_div if batched, else 1.
    std::size_t rows(tensor_id t) const noexcept {
        const tensor_desc& d = graph_->tensors()[t];
        return d.batched ? n_ / d.row_div : 1;
    }
    /// The rows of t this run uses.
    std::span<float> active(tensor_id t) noexcept {
        return tensor(t).first(rows(t) * graph_->tensors()[t].size);
    }
    /// Row i of t.
    std::span<float> row(tensor_id t, std::size_t i) noexcept {
        const std::size_t size = graph_->tensors()[t].size;
        return tensor(t).subspan(i * size, size);
    }

    static unsigned team_size(unsigned requested) {
        if (requested == 1) {
            return 1;
        }
        const unsigned pool = mtl::detail::thread_pool::instance().size();
        return requested == 0 ? pool : std::min(requested, pool);
    }

    /// Pieces per thread parallel() cuts its range into. The threads take
    /// pieces as they finish, so a slower core (an E-core, or a hyperthread
    /// sharing its core) takes fewer, rather than holding up the whole team.
    static constexpr std::size_t pieces_per_thread = 4;

    /// body(begin, end) over contiguous pieces of [0, n), each a multiple of
    /// `grain` long (but the last), on up to threads_ threads of MTL5's pool,
    /// which take the next piece as they finish one. Returns when every piece
    /// is done, which orders all of their writes before the caller's next
    /// read (P2). The pieces only partition the work: each output is produced
    /// by one body call whatever the split or the thread (D1).
    template <typename Body>
    void parallel(std::size_t n, std::size_t grain, Body&& body) noexcept {
        grain = std::max<std::size_t>(grain, 1);
        const std::size_t team = std::min<std::size_t>(threads_, n / grain);
        if (team <= 1) {
            body(std::size_t{0}, n);
            return;
        }
        const std::size_t pieces = std::min(n / grain, team * pieces_per_thread);
        const std::size_t step = ((n + pieces - 1) / pieces + grain - 1) / grain * grain;
        std::atomic<std::size_t> next{0};
        mtl::detail::thread_pool::instance().run(static_cast<unsigned>(team), [&](unsigned) {
            for (;;) {
                const std::size_t begin = next.fetch_add(1, std::memory_order_relaxed) * step;
                if (begin >= n) {
                    return;
                }
                body(begin, std::min(n, begin + step));
            }
        });
    }

    /// Weight rows per thread below which splitting a matmul does not pay.
    static constexpr std::size_t matmul_grain = 16;
    /// Activation rows per thread below which splitting their quantization does not pay.
    static constexpr std::size_t prepare_grain = 8;

    void execute(const op_matmul& o) noexcept {
        const std::span<std::byte> scratch = bytes(o.scratch);
        const std::size_t n = rows(o.in);
        const std::span<float> y = active(o.out);
        visit_view(o.w, [&](const auto& w) {
            using S = typename std::decay_t<decltype(w)>::scheme_type;
            // Quantize the activations once, then split the weight rows.
            auto* const acts = kernels::gemm_activations<S>(w.num_cols(), n, scratch);
            parallel(n, prepare_grain, [&](std::size_t t0, std::size_t t1) {
                kernels::gemm_prepare_rows<S>(w.num_cols(), active(o.in), scratch, acts, t0, t1);
            });
            parallel(w.num_rows(), matmul_grain, [&](std::size_t r0, std::size_t r1) {
                kernels::gemm_rows(w, acts, n, y, r0, r1);
            });
        });
    }
    void execute(const op_add& o) noexcept {
        kernels::add(active(o.a), active(o.b), active(o.out));
    }
    void execute(const op_add_bias& o) noexcept {
        for (std::size_t i = 0; i < rows(o.in); ++i) {
            const std::span<const float> in = row(o.in, i);
            const std::span<float> out = row(o.out, i);
            visit_view(o.bias, [&](const auto& b) { kernels::add_bias(in, b, out); });
        }
    }
    void execute(const op_mul& o) noexcept {
        kernels::mul(active(o.a), active(o.b), active(o.out));
    }
    void execute(const op_silu& o) noexcept {
        kernels::silu(active(o.in), active(o.out));
    }
    void execute(const op_softmax& o) noexcept {
        for (std::size_t i = 0; i < rows(o.in); ++i) {
            kernels::softmax(row(o.in, i), row(o.out, i));
        }
    }
    void execute(const op_rms_norm& o) noexcept {
        for (std::size_t i = 0; i < rows(o.in); ++i) {
            const std::span<const float> in = row(o.in, i);
            const std::span<float> out = row(o.out, i);
            visit_view(o.gamma, [&](const auto& g) { kernels::rms_norm(in, g, o.eps, out); });
        }
    }
    void execute(const op_embed& o) noexcept {
        for (std::size_t i = 0; i < n_; ++i) {
            const std::span<float> out = row(o.out, i);
            visit_view(o.table, [&](const auto& t) { t.decode_row(batch_.tokens[i], out.data()); });
        }
    }
    void execute(const op_rope& o) noexcept {
        for (std::size_t i = 0; i < n_; ++i) {
            kernels::rope(
                row(o.in, i), row(o.out, i), o.head_dim, position(i), o.theta_scale, o.freq_factors, o.mode, o.scaling);
        }
    }
    void execute(const op_attn_temperature& o) noexcept {
        for (std::size_t i = 0; i < n_; ++i) {
            kernels::scale(row(o.in, i), kernels::attn_temperature(position(i), o.scale, o.floor_len), row(o.out, i));
        }
    }
    void execute(const op_attention& o) noexcept {
        // Append every token's K and V first: token i reads only rows up to
        // pos+i, so this is what running the tokens one by one would see.
        // Then the KV groups, each over all tokens with its own scratch slot.
        const kernels::kv_layer c = cache(o.cache);
        const std::size_t n = n_;
        for (std::size_t i = 0; i < n; ++i) {
            kernels::kv_append(row(o.k, i), row(o.v, i), c, position(i));
        }
        const std::span<std::byte> scratch = bytes(o.scratch);
        const std::span<float> scores{reinterpret_cast<float*>(scratch.data()), scratch.size() / sizeof(float)};
        const std::size_t slot = kernels::attention_group_floats(o.n_heads, c.n_kv_heads, c.head_dim, c.max_ctx);
        parallel(c.n_kv_heads, 1, [&](std::size_t g0, std::size_t g1) {
            for (std::size_t g = g0; g < g1; ++g) {
                for (std::size_t i = 0; i < n; ++i) {
                    kernels::attend_group(row(o.q, i),
                                          c,
                                          position(i),
                                          o.n_heads,
                                          o.scale,
                                          row(o.out, i),
                                          scores.subspan(g * slot, slot),
                                          g);
                }
            }
        });
    }
    void execute(const op_last_row& o) noexcept {
        const std::span<const float> last = row(o.in, n_ - 1);
        std::copy(last.begin(), last.end(), tensor(o.out).begin());
    }

    void execute(const op_layer_norm& o) noexcept {
        for (std::size_t i = 0; i < rows(o.in); ++i) {
            const std::span<const float> in = row(o.in, i);
            const std::span<float> out = row(o.out, i);
            visit_view(o.gamma, [&](const auto& g) {
                if (std::holds_alternative<std::monostate>(o.beta)) {
                    kernels::layer_norm(in, g, o.eps, out);
                } else {
                    visit_view(o.beta, [&](const auto& b) { kernels::layer_norm(in, g, b, o.eps, out); });
                }
            });
        }
    }
    void execute(const op_gelu& o) noexcept {
        kernels::gelu(active(o.in), active(o.out));
    }
    void execute(const op_add_rows& o) noexcept {
        for (std::size_t i = 0; i < n_; ++i) {
            const std::span<const float> in = row(o.in, i);
            const std::span<float> out = row(o.out, i);
            visit_view(o.table, [&](const auto& t) { t.decode_row(i, out.data()); });
            kernels::add(in, out, out);
        }
    }
    void execute(const op_self_attention& o) noexcept {
        const std::size_t size = graph_->tensors()[o.q].size;
        const std::size_t head_dim = size / o.n_heads;
        const std::span<std::byte> scratch = bytes(o.scratch);
        const std::span<float> scores{reinterpret_cast<float*>(scratch.data()), scratch.size() / sizeof(float)};
        const std::size_t slot = graph_->max_batch();
        // Work units are (head, block of queries); a unit's scores live in its
        // slot. Every row's result is independent of the split (D1).
        const std::size_t blocks = self_attention_query_blocks;
        const std::size_t per_block = (n_ + blocks - 1) / blocks;
        parallel(o.n_heads * blocks, 1, [&](std::size_t u0, std::size_t u1) {
            for (std::size_t u = u0; u < u1; ++u) {
                const std::size_t h = u / blocks;
                const std::size_t i0 = std::min(n_, (u % blocks) * per_block);
                const std::size_t i1 = std::min(n_, i0 + per_block);
                kernels::self_attend_head(active(o.q),
                                          active(o.k),
                                          active(o.v),
                                          n_,
                                          size,
                                          head_dim,
                                          h,
                                          i0,
                                          i1,
                                          o.scale,
                                          active(o.out),
                                          scores.subspan(u * slot, slot));
            }
        });
    }
    void execute(const op_pixel_shuffle& o) noexcept {
        kernels::pixel_shuffle(
            active(o.in), graph_->tensors()[o.in].size, o.grid_w, n_ / o.grid_w, o.scale, active(o.out));
    }

    std::uint32_t position(std::size_t i) const noexcept {
        return batch_.pos + static_cast<std::uint32_t>(i);  // < position_limit, checked by run()
    }

    const compute_graph* graph_;
    memory_plan plan_;
    arena_ptr arena_;
    std::vector<std::size_t> state_offset_;
    arena_ptr states_;
    unsigned threads_ = 1;
    batch batch_{};
    std::size_t n_ = 0;  ///< rows of the current run
};

}  // namespace branes::synapse::graph

#pragma once
// Ported from ggml (MIT License, The ggml authors), YaRN scaling also from YaRN
// (MIT License, Jeffrey Quesnelle and Bowen Peng); see THIRD_PARTY_NOTICES.md.
/// @file attention.hpp
/// @brief Rotary position embedding and causal self-attention over a KV cache.
///
/// Both kernels serve one decode step (one token at position pos). They never
/// allocate and never throw. Numerics follow ggml's CPU path as llama.cpp uses
/// it with flash attention off and the default F16 KV cache:
///
///   rope:      theta for pair i is pos * theta_scale^i / freq_factor[i],
///              built by repeated float multiplication exactly as ggml's
///              rope cache does, then (x0, x1) rotated by theta. Pair i is
///              (x[2i], x[2i+1]) in "normal" mode (how GGUF stores Llama's
///              Q, K) and (x[i], x[i + head_dim/2]) in "neox" mode (Qwen2).
///              With YaRN scaling (rope_scaling), theta and the rotation's
///              magnitude follow ggml's rope_yarn, operation for operation.
///   attention: K and V are stored as F16; q is rounded to F16 before the
///              q.k dot (ggml's f16 mul_mat converts its f32 operand); the
///              softmax runs in float with a double sum; the probabilities
///              are rounded to F16 before weighting V.
///
/// What remains different from ggml is float re-association in the dots and
/// ggml's vectorized exp: well below the error of the quantized matmuls.

#include <branes/synapse/kernels/elementwise.hpp>
#include <branes/synapse/numeric/half.hpp>

#include <mtl/simd/algorithm.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

namespace branes::synapse::kernels {

/// Largest head dimension the attention kernel supports (it keeps one
/// head's K or V row in a stack buffer).
inline constexpr std::size_t max_head_dim = 256;

/// theta_scale = base^(-2/head_dim), computed in float as ggml does.
inline float rope_theta_scale(float base, std::size_t head_dim) noexcept {
    return std::pow(base, -2.0f / static_cast<float>(head_dim));
}

/// Which elements of a head RoPE rotates together (ggml's rope types).
enum class rope_mode : std::uint8_t {
    normal,  ///< pair i is (x[2i], x[2i+1]): Llama
    neox,    ///< pair i is (x[i], x[i + head_dim/2]): GPT-NeoX, Qwen2
};

/// YaRN context scaling of RoPE (ggml's freq_scale, ext_factor, attn_factor
/// and correction dimensions). The default is no scaling: theta and the
/// rotation are then exactly those of plain RoPE.
struct rope_scaling {
    float freq_scale = 1.0f;   ///< 1 / scaling factor: theta_interp = freq_scale * theta
    float ext_factor = 0.0f;   ///< weight of the extrapolation ramp (1 for YaRN, 0 off)
    float attn_factor = 1.0f;  ///< magnitude scale of cos and sin
    float corr_low = 0.0f;     ///< pairs below are extrapolated ...
    float corr_high = 0.0f;    ///< ... above interpolated, ramped in between
};

/// ggml_rope_yarn_corr_dims: the pair range [low, high] over which YaRN ramps
/// from extrapolation to interpolation, for a model trained on n_ctx_orig
/// positions. Computed in float, as ggml does.
inline std::array<float, 2>
rope_yarn_corr_dims(std::size_t n_dims, std::size_t n_ctx_orig, float base, float beta_fast, float beta_slow) noexcept {
    constexpr float pi = 3.14159265358979323846f;
    const auto corr_dim = [&](float n_rot) {
        return static_cast<float>(n_dims) * std::log(static_cast<float>(n_ctx_orig) / (n_rot * 2 * pi)) /
               (2 * std::log(base));
    };
    const float start = std::floor(corr_dim(beta_fast));
    const float end = std::ceil(corr_dim(beta_slow));
    return {std::max(0.0f, start), std::min(static_cast<float>(n_dims) - 1, end)};
}

/// Rotate every head of x by its position: pair i of each head by
/// theta_i = pos * theta_scale^i / freq_factors[i], YaRN-scaled when
/// `scaling` says so. x holds heads of head_dim floats; mode selects the pairs.
///   Pre: x.size() == out.size(), head_dim even and non-zero, x.size() % head_dim == 0,
///        freq_factors empty or of size head_dim / 2; out does not alias x.
///   Post: with the default scaling, exactly plain RoPE.
inline void rope(std::span<const float> x,
                 std::span<float> out,
                 std::size_t head_dim,
                 std::uint32_t pos,
                 float theta_scale,
                 std::span<const float> freq_factors,
                 rope_mode mode = rope_mode::normal,
                 const rope_scaling& scaling = {}) noexcept {
    assert(x.size() == out.size() && head_dim != 0 && head_dim % 2 == 0 && x.size() % head_dim == 0);  // Pre
    assert(freq_factors.empty() || freq_factors.size() == head_dim / 2);                               // Pre
    // Pair i is (x[a], x[a + offset]) with a = stride * i.
    const std::size_t stride = mode == rope_mode::normal ? 2 : 1;
    const std::size_t offset = mode == rope_mode::normal ? 1 : head_dim / 2;
    float theta = static_cast<float>(pos);
    for (std::size_t i = 0; i < head_dim / 2; ++i) {
        const float ff = freq_factors.empty() ? 1.0f : freq_factors[i];
        // ggml's rope_yarn. Unscaled, theta_interp == theta_extrap and
        // mscale == 1 exactly, so this is plain RoPE.
        const float theta_extrap = theta / ff;
        const float theta_interp = scaling.freq_scale * theta_extrap;
        float t = theta_interp;
        float mscale = scaling.attn_factor;
        if (scaling.ext_factor != 0.0f) {
            const float y =
                (static_cast<float>(i) - scaling.corr_low) / std::max(0.001f, scaling.corr_high - scaling.corr_low);
            const float ramp_mix = (1 - std::min(1.0f, std::max(0.0f, y))) * scaling.ext_factor;
            t = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * std::log(1.0f / scaling.freq_scale);
        }
        const float c = std::cos(t) * mscale;
        const float s = std::sin(t) * mscale;
        const std::size_t a = stride * i;
        for (std::size_t h = 0; h < x.size(); h += head_dim) {
            const float x0 = x[h + a];
            const float x1 = x[h + a + offset];
            out[h + a] = x0 * c - x1 * s;
            out[h + a + offset] = x0 * s + x1 * c;
        }
        theta *= theta_scale;
    }
}

/// One layer of an F16 KV cache: K and V each hold max_ctx rows of
/// n_kv_heads * head_dim halves, row p belonging to position p.
struct kv_layer {
    std::uint16_t* k = nullptr;
    std::uint16_t* v = nullptr;
    std::size_t max_ctx = 0;
    std::size_t n_kv_heads = 0;
    std::size_t head_dim = 0;

    std::size_t row() const noexcept {
        return n_kv_heads * head_dim;
    }
};

/// Floats of scratch one KV group needs: for each of its query heads, a
/// score row over max_ctx positions and the head rounded to F16.
constexpr std::size_t attention_group_floats(std::size_t n_heads,
                                             std::size_t n_kv_heads,
                                             std::size_t head_dim,
                                             std::size_t max_ctx) noexcept {
    return n_heads / n_kv_heads * (max_ctx + head_dim);
}

/// Floats of scratch attention() needs: one slot per KV group, so the groups
/// can be computed concurrently.
constexpr std::size_t attention_scratch_floats(std::size_t n_heads,
                                               std::size_t n_kv_heads,
                                               std::size_t head_dim,
                                               std::size_t max_ctx) noexcept {
    return n_kv_heads * attention_group_floats(n_heads, n_kv_heads, head_dim, max_ctx);
}

/// Store k, v as cache row pos, rounded to F16.
///   Pre: pos < cache.max_ctx; k.size() == v.size() == cache.row().
inline void
kv_append(std::span<const float> k, std::span<const float> v, const kv_layer& cache, std::uint32_t pos) noexcept {
    const std::size_t row = cache.row();
    assert(pos < cache.max_ctx && k.size() == row && v.size() == row);  // Pre
    std::uint16_t* const k_row = cache.k + std::size_t{pos} * row;
    std::uint16_t* const v_row = cache.v + std::size_t{pos} * row;
    for (std::size_t i = 0; i < row; ++i) {
        k_row[i] = float_to_fp16(k[i]);
        v_row[i] = float_to_fp16(v[i]);
    }
}

/// The query heads of KV group g: out = softmax(scale * Q K^T) V over cache
/// rows 0..pos, for heads g * group .. (g + 1) * group - 1 of q and out. It
/// writes only those heads of out and its own scratch slot, so the groups of
/// one token can run on different threads.
///   Pre: rows 0..pos are filled (kv_append); g < cache.n_kv_heads;
///        q.size() == out.size() == n_heads * head_dim; n_heads % n_kv_heads == 0;
///        head_dim <= max_head_dim; scores.size() >= attention_group_floats(...);
///        out does not alias q or scores.
inline void attend_group(std::span<const float> q,
                         const kv_layer& cache,
                         std::uint32_t pos,
                         std::size_t n_heads,
                         float scale,
                         std::span<float> out,
                         std::span<float> scores,
                         std::size_t g) noexcept {
    const std::size_t hd = cache.head_dim;
    const std::size_t n_kv = cache.n_kv_heads;
    const std::size_t row = cache.row();
    assert(pos < cache.max_ctx && hd <= max_head_dim && n_kv != 0 && n_heads % n_kv == 0 && g < n_kv);  // Pre
    assert(q.size() == n_heads * hd && out.size() == q.size());
    const std::size_t group = n_heads / n_kv;
    const std::size_t n_pos = std::size_t{pos} + 1;
    assert(scores.size() >= attention_group_floats(n_heads, n_kv, hd, cache.max_ctx));  // Pre
    const std::span<float> qg = scores.subspan(group * cache.max_ctx, group * hd);

    std::array<float, max_head_dim> kv{};  // one cached K or V head, widened
    for (std::size_t i = 0; i < group * hd; ++i) {
        qg[i] = fp16_to_float(float_to_fp16(q[g * group * hd + i]));
    }
    // Scores of the group's query heads against every cached position.
    for (std::size_t t = 0; t < n_pos; ++t) {
        const std::uint16_t* kt = cache.k + t * row + g * hd;
        for (std::size_t d = 0; d < hd; ++d) {
            kv[d] = fp16_to_float(kt[d]);
        }
        for (std::size_t j = 0; j < group; ++j) {
            const float* qj = qg.data() + j * hd;
            float dot = 0.0f;
            for (std::size_t d = 0; d < hd; ++d) {
                dot += qj[d] * kv[d];
            }
            scores[j * n_pos + t] = dot * scale;
        }
    }
    for (std::size_t j = 0; j < group; ++j) {
        const std::span<float> s = scores.subspan(j * n_pos, n_pos);
        softmax(s, s);  // in place
        for (float& p : s) {
            p = fp16_to_float(float_to_fp16(p));
        }
        std::fill_n(out.begin() + static_cast<std::ptrdiff_t>((g * group + j) * hd), hd, 0.0f);
    }
    // Probability-weighted sum of the cached V rows.
    for (std::size_t t = 0; t < n_pos; ++t) {
        const std::uint16_t* vt = cache.v + t * row + g * hd;
        for (std::size_t d = 0; d < hd; ++d) {
            kv[d] = fp16_to_float(vt[d]);
        }
        for (std::size_t j = 0; j < group; ++j) {
            const float p = scores[j * n_pos + t];
            float* oj = out.data() + (g * group + j) * hd;
            for (std::size_t d = 0; d < hd; ++d) {
                oj[d] += p * kv[d];
            }
        }
    }
}

/// Store k, v as cache row pos, then out = softmax(scale * Q K^T) V over rows
/// 0..pos, with query head h attending to KV head h / (n_heads / n_kv_heads):
/// kv_append followed by attend_group for every group.
///   Pre: pos < cache.max_ctx; q.size() == out.size() == n_heads * head_dim;
///        k.size() == v.size() == cache.row(); n_heads % n_kv_heads == 0;
///        head_dim <= max_head_dim; scores.size() >= attention_scratch_floats(...);
///        rows 0..pos-1 hold this sequence's earlier positions;
///        out does not alias q, k, v or scores.
inline void attention(std::span<const float> q,
                      std::span<const float> k,
                      std::span<const float> v,
                      const kv_layer& cache,
                      std::uint32_t pos,
                      std::size_t n_heads,
                      float scale,
                      std::span<float> out,
                      std::span<float> scores) noexcept {
    const std::size_t slot = attention_group_floats(n_heads, cache.n_kv_heads, cache.head_dim, cache.max_ctx);
    assert(scores.size() >= cache.n_kv_heads * slot);  // Pre
    kv_append(k, v, cache, pos);
    for (std::size_t g = 0; g < cache.n_kv_heads; ++g) {
        attend_group(q, cache, pos, n_heads, scale, out, scores.subspan(g * slot, slot), g);
    }
}

/// One head of bidirectional self-attention over n positions, with float
/// operands and no cache, as llama.cpp's vision encoders compute it without
/// flash attention (ggml_mul_mat of K and Q, ggml_soft_max_ext with the
/// scale, ggml_mul_mat of V and the probabilities): for every query position
/// i in [i0, i1), out_i = softmax_j(scale * q_i . k_j) v_j over j = 0..n-1,
/// on head h's head_dim elements of each row. It writes only those rows of
/// head h of out, and scores, so heads and query ranges can run on different
/// threads; each row's result does not depend on the range it is computed in.
///   Pre: q, k, v, out hold n rows of row_size floats; h * head_dim + head_dim
///        <= row_size; i0 <= i1 <= n; n >= 1; scores.size() >= n; out does not
///        alias q, k, v or scores.
inline void self_attend_head(std::span<const float> q,
                             std::span<const float> k,
                             std::span<const float> v,
                             std::size_t n,
                             std::size_t row_size,
                             std::size_t head_dim,
                             std::size_t h,
                             std::size_t i0,
                             std::size_t i1,
                             float scale,
                             std::span<float> out,
                             std::span<float> scores) noexcept {
    assert(n >= 1 && (h + 1) * head_dim <= row_size && i0 <= i1 && i1 <= n && scores.size() >= n);  // Pre
    assert(q.size() >= n * row_size && k.size() >= n * row_size && v.size() >= n * row_size &&
           out.size() >= n * row_size);
    const std::size_t off = h * head_dim;
    const std::span<float> s = scores.first(n);
    for (std::size_t i = i0; i < i1; ++i) {
        const float* qi = q.data() + i * row_size + off;
        for (std::size_t j = 0; j < n; ++j) {
            s[j] = mtl::simd::reduce_dot(qi, k.data() + j * row_size + off, head_dim) * scale;
        }
        softmax(s, s);  // in place
        float* oi = out.data() + i * row_size + off;
        std::fill_n(oi, head_dim, 0.0f);
        for (std::size_t j = 0; j < n; ++j) {
            const float p = s[j];
            const float* vj = v.data() + j * row_size + off;
            for (std::size_t d = 0; d < head_dim; ++d) {
                oi[d] += p * vj[d];
            }
        }
    }
}

}  // namespace branes::synapse::kernels

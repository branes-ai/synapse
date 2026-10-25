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
#include <mtl/simd/batch.hpp>

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

/// Floats in each of the head_dim rows of a head's transposed K
/// (self_attention_keys): n keys, padded with zeros to whole vectors.
inline constexpr std::size_t self_attention_key_stride(std::size_t n) noexcept {
    constexpr std::size_t w = mtl::simd::batch<float>::size;
    return (n + w - 1) / w * w;
}

/// Queries self_attend_head computes together, sharing each K and V load.
inline constexpr std::size_t self_attention_query_tile = 6;

/// Floats of scores self_attend_head needs: a padded score row per query of a tile.
inline constexpr std::size_t self_attention_score_floats(std::size_t n) noexcept {
    return self_attention_query_tile * self_attention_key_stride(n);
}

/// kt = head h of k transposed: kt[d * stride + j] = k_j[h * head_dim + d],
/// stride = self_attention_key_stride(n), zero beyond key n - 1. With it,
/// one vector operation advances the dots of a vector's worth of keys.
///   Pre: k holds n rows of row_size floats; (h + 1) * head_dim <= row_size;
///        kt.size() >= head_dim * stride.
inline void self_attention_keys(std::span<const float> k,
                                std::size_t n,
                                std::size_t row_size,
                                std::size_t head_dim,
                                std::size_t h,
                                std::span<float> kt) noexcept {
    const std::size_t stride = self_attention_key_stride(n);
    assert((h + 1) * head_dim <= row_size && k.size() >= n * row_size && kt.size() >= head_dim * stride);  // Pre
    // Blocks of keys, so each kt row is written a cache line at a time.
    constexpr std::size_t keys = 16;
    for (std::size_t j0 = 0; j0 < n; j0 += keys) {
        const std::size_t j1 = std::min(n, j0 + keys);
        for (std::size_t d = 0; d < head_dim; ++d) {
            const float* kd = k.data() + h * head_dim + d;
            for (std::size_t j = j0; j < j1; ++j) {
                kt[d * stride + j] = kd[j * row_size];
            }
        }
    }
    for (std::size_t d = 0; d < head_dim; ++d) {
        std::fill(kt.begin() + static_cast<std::ptrdiff_t>(d * stride + n),
                  kt.begin() + static_cast<std::ptrdiff_t>((d + 1) * stride),
                  0.0f);
    }
}

namespace detail {

using attention_lanes = mtl::simd::batch<float>;

/// Scores of Q queries against the KV vectors of keys from j:
/// s[t * stride + j ..] = scale * (q[t] . k_j ..), each dot one fma chain
/// in d order, the same for every Q and KV.
template <std::size_t Q, std::size_t KV>
void score_tile(const std::array<const float*, Q>& q,
                const float* kt,
                std::size_t stride,
                std::size_t head_dim,
                std::size_t j,
                float scale,
                float* s) noexcept {
    using lanes = attention_lanes;
    constexpr std::size_t w = lanes::size;
    std::array<lanes, Q * KV> acc{};
    for (std::size_t d = 0; d < head_dim; ++d) {
        std::array<lanes, KV> kd;
        for (std::size_t v = 0; v < KV; ++v) {
            kd[v] = lanes::load_unaligned(kt + d * stride + j + v * w);
        }
        for (std::size_t t = 0; t < Q; ++t) {
            const lanes qd(q[t][d]);
            for (std::size_t v = 0; v < KV; ++v) {
                acc[t * KV + v] = fma(qd, kd[v], acc[t * KV + v]);
            }
        }
    }
    for (std::size_t t = 0; t < Q; ++t) {
        for (std::size_t v = 0; v < KV; ++v) {
            (acc[t * KV + v] * lanes(scale)).store_unaligned(s + t * stride + j + v * w);
        }
    }
}

/// o[t][e0 .. e0 + DV vectors) = sum_j p[t * stride + j] v_j[e0 ..] for Q
/// queries, each element one fma chain in j order, the same for every Q and DV.
template <std::size_t Q, std::size_t DV>
void weigh_tile(const float* p,
                std::size_t stride,
                const float* v,
                std::size_t n,
                std::size_t row_size,
                const std::array<float*, Q>& o,
                std::size_t e0) noexcept {
    using lanes = attention_lanes;
    constexpr std::size_t w = lanes::size;
    std::array<lanes, Q * DV> acc{};
    for (std::size_t j = 0; j < n; ++j) {
        std::array<lanes, DV> vj;
        for (std::size_t u = 0; u < DV; ++u) {
            vj[u] = lanes::load_unaligned(v + j * row_size + e0 + u * w);
        }
        for (std::size_t t = 0; t < Q; ++t) {
            const lanes pj(p[t * stride + j]);
            for (std::size_t u = 0; u < DV; ++u) {
                acc[t * DV + u] = fma(pj, vj[u], acc[t * DV + u]);
            }
        }
    }
    for (std::size_t t = 0; t < Q; ++t) {
        for (std::size_t u = 0; u < DV; ++u) {
            acc[t * DV + u].store_unaligned(o[t] + e0 + u * w);
        }
    }
}

/// self_attend_head for the Q queries from i.
template <std::size_t Q>
void self_attend_tile(const float* q,
                      const float* kt,
                      const float* v,
                      std::size_t n,
                      std::size_t row_size,
                      std::size_t off,
                      std::size_t head_dim,
                      std::size_t i,
                      float scale,
                      float* out,
                      float* s) noexcept {
    constexpr std::size_t w = attention_lanes::size;
    constexpr std::size_t kv = 2;  // key vectors per score tile
    constexpr std::size_t dv = 2;  // output vectors per weighted-sum tile
    const std::size_t stride = self_attention_key_stride(n);
    std::array<const float*, Q> qt;
    std::array<float*, Q> ot;
    for (std::size_t t = 0; t < Q; ++t) {
        qt[t] = q + (i + t) * row_size + off;
        ot[t] = out + (i + t) * row_size + off;
    }
    std::size_t j = 0;
    for (; j + kv * w <= stride; j += kv * w) {
        score_tile<Q, kv>(qt, kt, stride, head_dim, j, scale, s);
    }
    for (; j < stride; j += w) {
        score_tile<Q, 1>(qt, kt, stride, head_dim, j, scale, s);
    }
    for (std::size_t t = 0; t < Q; ++t) {
        const std::span<float> st{s + t * stride, n};
        softmax(st, st);  // in place
    }
    const float* vh = v + off;
    std::size_t e = 0;
    for (; e + dv * w <= head_dim; e += dv * w) {
        weigh_tile<Q, dv>(s, stride, vh, n, row_size, ot, e);
    }
    for (; e + w <= head_dim; e += w) {
        weigh_tile<Q, 1>(s, stride, vh, n, row_size, ot, e);
    }
    for (e = head_dim / w * w; e < head_dim; ++e) {  // a head_dim the vector does not divide
        for (std::size_t t = 0; t < Q; ++t) {
            const float* pt = s + t * stride;
            const float* ve = vh + e;
            float acc = 0.0f;
            for (std::size_t jj = 0; jj < n; ++jj, ve += row_size) {
                acc = mtl::simd::scalar_fma(pt[jj], *ve, acc);
            }
            ot[t][e] = acc;
        }
    }
}

}  // namespace detail

/// One head of bidirectional self-attention over n positions, with float
/// operands and no cache, as llama.cpp's vision encoders compute it without
/// flash attention (ggml_mul_mat of K and Q, ggml_soft_max_ext with the
/// scale, ggml_mul_mat of V and the probabilities): for every query position
/// i in [i0, i1), out_i = softmax_j(scale * q_i . k_j) v_j over j = 0..n-1,
/// on head h's head_dim elements of each row. kt is head h of k transposed
/// by self_attention_keys. It writes only those rows of head h of out, and
/// scores, so heads and query ranges can run on different threads.
///
/// Register-tiled: self_attention_query_tile queries share every load of kt
/// and of a V row. Each score is one fma chain over d and each output
/// element one over j, whatever the tile, so a row's result does not depend
/// on the range it is computed in (D1).
///   Pre: q, v, out hold n rows of row_size floats; h * head_dim + head_dim
///        <= row_size; kt.size() >= head_dim * self_attention_key_stride(n);
///        i0 <= i1 <= n; n >= 1; scores.size() >= self_attention_score_floats(n);
///        out does not alias q, kt, v or scores.
inline void self_attend_head(std::span<const float> q,
                             std::span<const float> kt,
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
    assert(n >= 1 && (h + 1) * head_dim <= row_size && i0 <= i1 && i1 <= n);  // Pre
    assert(scores.size() >= self_attention_score_floats(n) &&
           kt.size() >= head_dim * self_attention_key_stride(n));
    assert(q.size() >= n * row_size && v.size() >= n * row_size && out.size() >= n * row_size);
    constexpr std::size_t tile = self_attention_query_tile;
    const std::size_t off = h * head_dim;
    std::size_t i = i0;
    for (; i + tile <= i1; i += tile) {
        detail::self_attend_tile<tile>(
            q.data(), kt.data(), v.data(), n, row_size, off, head_dim, i, scale, out.data(), scores.data());
    }
    for (; i < i1; ++i) {
        detail::self_attend_tile<1>(
            q.data(), kt.data(), v.data(), n, row_size, off, head_dim, i, scale, out.data(), scores.data());
    }
}

}  // namespace branes::synapse::kernels

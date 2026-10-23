# What a Llama model computes, and why

This page walks through the computation synapse runs for a Llama-family model, one step at a time. For each step it shows:
- **What:** the formula and the shapes.
- **Why:** the design reason behind it.
- **Where:** synapse's implementation and llama.cpp's.

How the surrounding ecosystem (GGUF, ggml, libllama) is organized is in [ggml-ecosystem.md](ggml-ecosystem.md).

The running example is **Llama-3.2-1B-Instruct**: 16 layers, width `d = 2048`, 32 query heads and 8 KV heads of dimension 64, a feed-forward width of 8192 and 128,256 tokens. Its costs are measured in [assessments/block-scaled-dot-performance.md §2](assessments/block-scaled-dot-performance.md#2-the-workload).

## Contents

1. [The shape of the model](#1-the-shape-of-the-model)
2. [One token, step by step](#2-one-token-step-by-step)
3. [Why it is built this way](#3-why-it-is-built-this-way)
4. [A batch of tokens: prefill](#4-a-batch-of-tokens-prefill)
5. [What it costs](#5-what-it-costs)
6. [The family: Llama, Qwen2, Mistral](#6-the-family-llama-qwen2-mistral)
7. [Text in, tokens out](#7-text-in-tokens-out)
8. [Numerics: why synapse and llama.cpp agree, and where they stop agreeing](#8-numerics-why-synapse-and-llamacpp-agree-and-where-they-stop-agreeing)

## 1. The shape of the model

A Llama model is a **decoder-only transformer**. Given the tokens so far, it produces a score (logit) for every possible next token. Generation repeats one step: run the newest token through the model, pick a next token from the logits, append it, and run again.

```text
 token id ──► embedding row x  (d = 2048 floats: the "residual stream")
              │
              │   ┌───────────────── repeated for each of the 16 layers ─────────────────┐
              ├──►│ h = RMSNorm(x) ──► Q, K, V ──► RoPE(Q, K) ──► attention over the     │
              │   │                                              KV cache (K, V appended)│
              │   │ x = x + W_o · attention                                              │
              │   │ h = RMSNorm(x) ──► W_down · ( SiLU(W_gate · h) ⊙ (W_up · h) )        │
              │   │ x = x + that                                                         │
              │   └──────────────────────────────────────────────────────────────────────┘
              ▼
       RMSNorm(x) ──► W_out · x ──► logits (128,256) ──► next token (argmax, or sampled)
```

Two things to notice:
- **The residual stream:** every layer *adds* to `x`, never overwrites it.
- **The cache:** attention reaches back to every earlier token, through the cache of their keys and values.

synapse builds exactly this graph in `sdk/include/branes/synapse/models/llama.hpp` (`load_llama`). llama.cpp's builder is `llm_build_llama` in `src/models/llama.cpp`.

## 2. One token, step by step

The token is at position `p` (0 for the first token of the sequence). Shapes are for Llama-3.2-1B.

### 2.1 Embedding

`x = token_embd[token]`: row `token` of a 128,256 × 2048 table, expanded to float. synapse: `graph_builder::embed`, run by `session::execute(op_embed)`.

### 2.2 RMSNorm

```text
RMSNorm(x) = x / sqrt(mean(x²) + ε) ⊙ g        ε = 1e-5, g a learned 2048-vector
```

This divides the residual stream by its root-mean-square, then applies the learned per-channel scale `g`, before each sub-block reads it. The result therefore has the scale `g` gives it, not unit RMS.

synapse: `kernels::rms_norm` (`kernels/elementwise.hpp`), which accumulates the sum of squares in double as ggml's `ggml_compute_forward_rms_norm_f32` does (`ggml_float`).

### 2.3 Q, K, V projections

```text
q = W_q h    2048 → 2048  = 32 heads × 64
k = W_k h    2048 →  512  =  8 heads × 64
v = W_v h    2048 →  512  =  8 heads × 64
```

These are three matrix–vector products with quantized weights. They dominate the cost together with the FFN's (§5). synapse: `graph_builder::matmul`, executed by `kernels::gemm_rows`.

### 2.4 RoPE

Rotary position embedding rotates each head's vector by an angle proportional to the position.

```text
for pair i = 0 .. 31 of each head:
    θ_i  = p · base^(−2i/64) / f_i          base = 500,000 (Llama 3), f_i a frequency factor
    (a, b) ← (a cos θ_i − b sin θ_i,  a sin θ_i + b cos θ_i)
```

A "pair" is two elements of the head:
- **normal mode** (Llama): `(x[2i], x[2i+1])`;
- **NEOX mode** (Qwen2): `(x[i], x[i+32])`.

[ggml-ecosystem.md §3](ggml-ecosystem.md#3-from-a-hugging-face-checkpoint-to-a-gguf) explains why the two layouts exist.

RoPE is applied to q and k, not v.

synapse: `kernels::rope` (`kernels/attention.hpp`). It builds θ by repeated float multiplication exactly as ggml's rope cache does (`ggml_rope_cache_init` in `ggml/src/ggml-cpu/ops.cpp`).

### 2.5 Attention over the KV cache

```text
append k, v to this layer's cache as row p        (stored as F16)
for query head h = 0 .. 31:
    g      = h / 4                                 (its KV head: grouped-query attention)
    s_t    = (q_h · K_g[t]) / sqrt(64)    for t = 0 .. p
    a      = softmax(s)                            (over the p+1 positions)
    out_h  = Σ_t a_t · V_g[t]
```

Each query head scores every cached position, turns the scores into weights that sum to 1, and takes the weighted average of those positions' values. Positions after `p` are never in the cache, which is what makes the model **causal**.

synapse: `kernels::attend_group`, one call per KV group, so groups can run on different threads. llama.cpp: `build_attn_mha` in `src/llama-graph.cpp`, with the causal mask in `ggml_soft_max_ext`.

### 2.6 Output projection and residual

`x = x + W_o · out`, where `out` is the 32 heads concatenated (2048) and `W_o` is 2048 × 2048.

### 2.7 Feed-forward (SwiGLU)

```text
h   = RMSNorm(x)
u   = SiLU(W_gate h) ⊙ (W_up h)       2048 → 8192, SiLU(z) = z · sigmoid(z)
x   = x + W_down u                    8192 → 2048
```

synapse: `matmul`, `silu`, `mul`, `matmul`, `add`. llama.cpp: `build_ffn` with `LLM_FFN_SILU` and `LLM_FFN_PAR`, which ends in `ggml_swiglu_split`.

### 2.8 Logits

`logits = W_out · RMSNorm(x)`: 128,256 scores. Llama-3.2-1B has no separate `output.weight`, so `W_out` is the embedding table itself (tied).

synapse returns the logits; `tools/llama_decode` takes their argmax (greedy decoding). llama.cpp hands them to a sampler chain.

## 3. Why it is built this way

**Pre-norm residual stream.**
- **What:** each sub-block reads a normalized copy of `x` and adds its result back.
- **Why:** with no normalization on the residual path itself, gradients and activations pass through dozens of layers unchanged. That is what makes deep stacks trainable, and why the final RMSNorm before the logits is needed.

**RMSNorm instead of LayerNorm.**
- **What:** LayerNorm subtracts the mean and adds a learned bias. RMSNorm (Zhang & Sennrich, 2019) keeps only the rescaling.
- **Why:** it costs less and trains as well, because the re-centering turned out not to matter.

**Rotary position embeddings instead of learned positions** (Su et al., 2021).
- **The property:** rotating q by `p_q·θ` and k by `p_k·θ` makes their dot product depend only on `p_q − p_k`. Attention therefore sees *relative* position.
- **What it saves:** there is no position table to learn or to outgrow.
- **Context extension:** the angle schedule can be adjusted to stretch a model beyond its training context.
  - Llama 3's frequency factors (§6) are one way;
  - YaRN (§6) is another.

**Grouped-query attention** (Ainslie et al., 2023).
- **What:** 32 query heads share 8 key/value heads.
- **Why:** the KV cache is 4× smaller, and each decode step reads 4× fewer cached bytes, at almost no quality cost against full multi-head attention.
- **For Llama-3.2-1B:** 32 KiB per position instead of 128 KiB.

**SwiGLU feed-forward** (Shazeer, 2020).
- **What:** a gated FFN multiplies one projection by a nonlinearity of another.
- **Why:** it is better per parameter than a plain `W_2 ReLU(W_1 x)`.
- **The cost:** three matrices instead of two, so the width is set to 8192 (4× `d`) to keep the parameter count comparable.

**Tied embeddings.** In a small model the 128,256 × 2048 table is a large share of the weights: 263 M of 1.24 B. Using it both to embed and to score saves that much again.

**The KV cache.**
- **What:** each decode step appends one key/value row per layer.
- **What it saves:** the token's attention costs O(p), and there is no need to recompute the earlier tokens' keys and values, which would be O(p²) per step.
- **The trade:** memory grows linearly with the context.

## 4. A batch of tokens: prefill

A prompt is processed in batches. Every activation becomes a matrix of `n` tokens × width:
- **The matrix–vector products become matrix–matrix products** (`kernels::gemm`). Each weight block is read and unpacked once per batch instead of once per token.
- **RMSNorm, RoPE and attention work per token.** Token `i` of the batch is at position `p+i`, and its attention covers cache rows `0..p+i`, so causality holds inside the batch too.
- **synapse computes only the last token's logits** (`graph_builder::last_row`). That is llama.cpp's default when the batch carries no output mask (`batch.logits`, `src/llama-batch.cpp`). llama.cpp can also return logits for every token of the batch or for selected ones.

synapse's contract G1 makes each gemm row perform exactly gemv's operations, and the per-token kernels run in token order. So by construction, prefilling a prompt gives the same logits and KV cache as decoding it token by token.

`tests/sdk/llama_prefill.cpp` checks this bit for bit on the tiny llama.cpp-generated fixtures:
- the logits after a single batch and after uneven chunks;
- tokens decoded after a prefill. Those read the cache the prefill wrote, so this checks the cache indirectly; the test does not compare the cache tensors themselves.

The opt-in real-model test (`tests/sdk/llama_model.cpp`) checks the same for each real model tested.

## 5. What it costs

Per decode token, from [assessments/block-scaled-dot-performance.md §2.2](assessments/block-scaled-dot-performance.md#22-decode-one-token):

| | Llama-3.2-1B Q4_K_M |
|---|---|
| weights per layer | 60.8 M (about 33 MiB at Q4_K) |
| multiply-accumulates in matrix products | 1.236 G (973 M in the layers + 263 M for the logits) |
| weight bytes read | 762.8 MiB: every weight once |
| attention multiply-accumulates | 65,536 · (p+1): 0.5% of the matrix products at p = 93 |
| KV cache | 32 KiB per position |
| arithmetic intensity | about 1.6 multiply-accumulates per weight byte |

**Decode is memory-bound.**
- **Why:** every token streams the whole model through the cores and does little arithmetic per byte.
- **On the test machine:** 20 threads decode only about 8× faster than one, consistent with the cores saturating memory bandwidth. Peak DRAM bandwidth was not measured.

**Prefill is compute-bound.**
- **Why:** a batch of 64 reads each weight once for 64 tokens, about 78 multiply-accumulates per weight byte. The assessment's figure of about 57 (its §2.3) is for its 93-token prompt, which runs as two batches, 64 + 29, and so reads the weights twice.
- **Where the time goes:** unpacking 4- and 6-bit weights and doing integer multiply-adds.

## 6. The family: Llama, Qwen2, Mistral

synapse builds all of these through the same graph (`models/llama.hpp`). The differences are where llama.cpp's builders differ. The values are those of the models tested; each file's own metadata is what synapse reads.

| | Llama 3 | Qwen2 / Qwen2.5 | Mistral 7B, Nemo | Ministral 3 (`mistral3`) |
|---|---|---|---|---|
| RoPE pairs | normal | NEOX | normal | normal |
| RoPE schedule | base 500,000 with **frequency factors** | base 1,000,000 | base 1,000,000 (v0.2, v0.3, Nemo); 10,000 (v0.1) | **YaRN** (factor 16) |
| biases | none | Q, K, V (and optional output) | none | optional on Q, K, V, O and the FFN |
| attention temperature | none | none | none | Q scaled by `1 + 0.1·ln(⌊p/16384⌋ + 1)` |
| tokenizer | BPE, `llama-bpe` | BPE, `qwen2` | SPM (7B); BPE `tekken` (Nemo) | BPE, `tekken` |

### Llama 3's frequency factors

Llama 3's frequency factors are the `rope_freqs.weight` tensor: one divisor per pair.
- The **high-frequency pairs** (short wavelengths, which encode local order) keep their angle: factor 1.
- The **low-frequency pairs** (long wavelengths, which no training context ever completed) are slowed by the scaling factor, 32 for Llama 3.2.
- Pairs in between are interpolated smoothly.

`convert_hf_to_gguf.py` computes them from `rope_scaling`; see `low_freq_wavelen` there.

### YaRN (Peng et al., 2023)

YaRN does the same per pair with a ramp: it extrapolates the fast pairs and interpolates the slow ones, and rescales the attention logits to compensate. synapse ports ggml's `rope_yarn` operation for operation (`kernels::rope_scaling`). It derives the parameters as llama.cpp's `llama_context` does.

### Mistral 3's attention temperature

It sharpens attention at positions beyond the original context. Below 16,384 positions it is exactly 1.

## 7. Text in, tokens out

The model sees token ids, not text. The vocabulary in the GGUF file defines the mapping, and synapse reproduces llama.cpp's tokenizers exactly (`tokenizer/`).

**Byte-level BPE** (Llama 3, Qwen2, Tekken; `tokenizer/bpe.hpp`) works in four steps:
1. Split out special tokens. User-defined tokens are always split out; control and unknown tokens such as `<|eot_id|>` only when `parse_special` is set.
2. Split the remaining text into words with the pre-tokenizer's regex (`tokenizer/pretokenize.hpp`). The regexes differ:
   - Llama 3 groups digits in threes;
   - Qwen2 splits every digit;
   - Tekken also splits letter runs at case changes.
3. Spell each word's bytes with GPT-2's byte-to-character map.
4. Apply the vocabulary's merges, lowest rank first, until none applies.

**SentencePiece** (Llama 2, TinyLlama, Mistral 7B; `tokenizer/spm.hpp`) works differently:
- it marks spaces with `▁`;
- it merges the adjacent pair whose union is the highest-scoring token, repeatedly;
- it falls back to `<0xXX>` byte tokens for anything outside the vocabulary.

Detokenization concatenates each token's text, with the byte map or the `▁` escape undone.

## 8. Numerics: why synapse and llama.cpp agree, and where they stop agreeing

### Where llama.cpp rounds

On the path synapse implements (CPU, flash attention off, F16 KV cache), llama.cpp rounds at fixed points. synapse rounds at the same ones:
- **Activations of a quantized matrix product** are rounded to 8 bits (Q8_0 or Q8_K) before the integer dot. Plain float weight formats keep float activations: ggml converts them to the weight's format, so they are rounded to F16 for F16 weights and to BF16 for BF16 weights, and stay exact for F32 weights.
- **K and V** are stored in the cache as F16.
- **q** is rounded to F16 before the q·k dot, because ggml's F16 matrix product converts its F32 operand.
- **The softmax probabilities** are rounded to F16 before they weight V.

### How closely they agree

- **No quantized activations:** on models with F32 weights, synapse and llama.cpp agree to about 1e-7 relative at every position (`*_tiny_f32` in `tests/sdk/llama_golden.cpp`).
- **Quantized models:** they also agree to about 1e-7, *until* one of the rounding points above lands on the other side of a rounding boundary. Two things can cause that:
  - a different float summation order (synapse's Highway build, ggml's vector kernels);
  - ggml's build fusing `a*b + c` into an FMA.

  From then on the logits differ by a few percent RMS (1–7% on the real models tested). That is about the distance synapse moves from itself under a comparable perturbation. llama.cpp's top token stays in synapse's top few.

The golden tests' bounds are set from these measurements, and their comments record each case.

## References

- **Llama:** Touvron et al., *LLaMA: Open and Efficient Foundation Language Models*, 2023; Llama Team, AI @ Meta, *The Llama 3 Herd of Models*, 2024.
- **RMSNorm:** Zhang & Sennrich, *Root Mean Square Layer Normalization*, 2019.
- **RoPE:** Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021.
- **YaRN:** Peng, Quesnelle, Fan & Shippole, *YaRN: Efficient Context Window Extension of Large Language Models*, 2023.
- **Grouped-query attention:** Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, 2023.
- **SwiGLU:** Shazeer, *GLU Variants Improve Transformer*, 2020.

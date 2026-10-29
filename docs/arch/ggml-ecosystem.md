# The GGML ecosystem, and where synapse fits

synapse loads a subset of the models distributed by the GGML ecosystem: GGUF files whose `general.architecture` is `llama`, `qwen2` or `mistral3`. llama.cpp builds them with separate graph builders, but they differ only in a few places, so synapse builds all three with one graph (`models/llama.hpp`). It computes what llama.cpp computes for them. It rejects other architectures and unsupported features rather than approximating them. Its tests hold it to llama.cpp's logits and tokens. This page describes that ecosystem, so that the claims in synapse's code and tests make sense: "bit-identical to llama.cpp", "follows ggml's rounding", "llama.cpp's `yarn_attn_factor`".

What the model itself computes is in [llama-model.md](llama-model.md).

References to llama.cpp use the checkout from which synapse's fixtures were generated (`e6f2ec01f`).
- **llama.cpp paths:** relative to that repository (`src/`, `ggml/`, `tools/`, `gguf-py/`, `convert_hf_to_gguf.py`).
- **synapse paths:** relative to synapse's repository root when they start with `sdk/`, `scripts/`, `tests/`, `cmake/` or `docs/`.
- **Short synapse paths** in synapse's own column, like `quant/q8.hpp` or `models/llama.hpp`: relative to `sdk/include/branes/synapse/`. `tools/llama_decode` is the one exception, at synapse's root.

## Contents

1. [The layers](#1-the-layers)
2. [GGUF: the file format](#2-gguf-the-file-format)
3. [From a Hugging Face checkpoint to a GGUF](#3-from-a-hugging-face-checkpoint-to-a-gguf)
4. [ggml: tensors, graphs, memory and backends](#4-ggml-tensors-graphs-memory-and-backends)
5. [Quantization](#5-quantization)
6. [libllama: the runtime](#6-libllama-the-runtime)
7. [Where synapse sits](#7-where-synapse-sits)

## 1. The layers

```text
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ tools/        llama-cli, llama-server, llama-bench, llama-quantize,      │  applications
 │ examples/     llama-perplexity, llama-imatrix, ...                       │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ src/          libllama (include/llama.h)                                 │  the LLM runtime
 │   llama-model-loader   llama-model    llama-vocab    llama-context       │
 │   llama-graph + src/models/*.cpp (one llm_build_* per architecture)      │
 │   llama-kv-cache  llama-batch  llama-sampler  llama-quant  llama-chat    │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ ggml/         the tensor library (ggml/include/ggml.h)                   │  the compute engine
 │   ggml.c       tensors and ops, building a graph                         │
 │   ggml-alloc   graph memory planning                                     │
 │   ggml-backend backends, buffers, the multi-backend scheduler            │
 │   ggml-cpu/  ggml-cuda/  ggml-metal/  ggml-vulkan/  ...  (kernels)       │
 │   gguf.cpp     the GGUF file format (ggml/include/gguf.h)                │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ convert_hf_to_gguf.py, gguf-py/   Hugging Face → GGUF, in Python         │  model preparation
 └──────────────────────────────────────────────────────────────────────────┘
```

- **ggml** knows nothing about language models. It provides typed n-dimensional tensors, a library of operations on them, a way to record those operations as a graph, and backends that execute the graph on CPUs and GPUs. GGUF lives in ggml because it is ggml's way of storing named tensors with metadata.
- **libllama** is the language-model runtime built on ggml:
  - reading a GGUF into a model, and tokenizing text with the model's vocabulary;
  - building each architecture's forward graph;
  - managing the KV cache across calls, and turning logits into tokens.
- **The tools** are thin programs over `llama.h`.
- **The Python side** (`convert_hf_to_gguf.py`, using `gguf-py`) turns training checkpoints into GGUF files. `llama-quantize` then re-encodes their weights.

synapse corresponds to the middle two layers: a GGUF reader, kernels for ggml's formats, a graph runtime, and the Llama-family model builder and tokenizers. It takes GGUF files as produced by the bottom layer.

## 2. GGUF: the file format

A GGUF file is one self-describing container: metadata that says what the model is, followed by its tensors (`ggml/include/gguf.h` documents the layout).

```text
 ┌─────────────────────────────────────────────────────────────┐
 │ magic "GGUF" │ version (3) │ n_tensors (u64) │ n_kv (u64)   │  header
 ├─────────────────────────────────────────────────────────────┤
 │ n_kv × { key (string) │ value type │ value }                │  metadata
 │   general.architecture = "llama"                            │
 │   llama.block_count = 16, llama.embedding_length = 2048 ... │
 │   tokenizer.ggml.model = "gpt2", tokenizer.ggml.tokens = [] │
 ├─────────────────────────────────────────────────────────────┤
 │ n_tensors × { name │ n_dims │ ne[0..n_dims) │ type │ offset}│  tensor info
 ├──────────── padding to general.alignment (default 32) ──────┤
 │ tensor data: each tensor at its offset, aligned             │  data
 └─────────────────────────────────────────────────────────────┘
```

**Metadata is typed key/value pairs.**
- **Types:** integers of each width, floats, bools, strings, and arrays of them.
- **Architecture keys:** they are namespaced by the architecture's name, which comes from `general.architecture`. A Qwen2 file therefore has `qwen2.block_count` and a Mistral 3 file `mistral3.rope.scaling.yarn_beta_fast`.
- **The key table:** `src/llama-arch.cpp` holds the canonical list, as `LLM_KV_*` enums mapped to format strings like `"%s.attention.head_count_kv"`.
- **The tokenizer:** it is stored in the metadata too, as `tokenizer.ggml.*`: the token texts, their types, scores (SentencePiece) or merges (BPE), the special token ids, and the pre-tokenizer name.

**Tensor info gives each tensor its name, shape and element type.**
- **Names** follow `src/llama-arch.cpp`'s `LLM_TENSOR_*` table: `token_embd.weight`, `blk.N.attn_q.weight`, `output_norm.weight`, and so on.
- **The shape `ne`** is listed innermost dimension first. A weight matrix applied as `y = W x` with `cols` inputs and `rows` outputs has `ne = [cols, rows]`, so each row is contiguous.
- **The type** is one of ggml's (`enum ggml_type`): F32, F16, BF16, or a block-quantized format (§5).

**The data section** is aligned to `general.alignment` (32 bytes when absent, `GGUF_DEFAULT_ALIGNMENT`). Every tensor sits at an aligned offset, so a reader can memory-map the file and use the weights in place without copying. libllama does this (`src/llama-mmap.cpp`, `use_mmap` in `src/llama-model-loader.h`), and so does synapse (`sdk/include/branes/synapse/gguf/mapped_file.hpp`).

## 3. From a Hugging Face checkpoint to a GGUF

`convert_hf_to_gguf.py` reads a checkpoint's `config.json`, tokenizer files and tensors. It writes the GGUF through `gguf-py`. Each architecture is a class registered against Hugging Face's model class names; for example `LlamaModel` handles `LlamaForCausalLM` and `MistralForCausalLM`. The conversion does four things:

1. **It writes the hyperparameters** as `<arch>.*` keys, and the vocabulary as `tokenizer.ggml.*`. For BPE vocabularies it also records which pre-tokenizer regex the tokenizer uses (`tokenizer.ggml.pre`, e.g. `llama-bpe`, `qwen2`, `tekken`). The pre-tokenizer is identified by hashing the tokenization of a probe string (`convert_hf_to_gguf_update.py`).
2. **It renames the tensors** to GGUF's names (`gguf-py/gguf/tensor_mapping.py`).
3. **It permutes Llama's Q and K rows.** Hugging Face's Llama applies RoPE to the pairs `(x[i], x[i + d/2])`, its `rotate_half`. ggml's "normal" RoPE rotates adjacent pairs `(x[2i], x[2i+1])`. So `LlamaModel.permute` (`convert_hf_to_gguf.py`, around line 2874) reorders each head's rows of `q_proj` and `k_proj` to put the two halves side by side.

   Architectures converted without the permute keep Hugging Face's layout, and ggml rotates them in "NEOX" mode instead. Qwen2 is one (`llama_model_rope_type` in `src/llama-model.cpp` returns `LLAMA_ROPE_TYPE_NEOX` for it). This is why synapse has both `rope_mode::normal` and `rope_mode::neox`.
4. **It writes the weights** in F32, F16 or BF16. Quantization is a separate step.

**`llama-quantize`** (`tools/quantize`, implemented in `src/llama-quant.cpp`) re-encodes a GGUF's weights. A file type like `Q4_K_M` is a mix chosen per tensor by `llama_tensor_get_type_impl`, not one format:
- most matrices go to Q4_K;
- `attn_v` and `ffn_down` get Q6_K in some layers (the `use_more_bits` rule: the first and last eighth of the layers, and every third one between);
- norms and other 1-D tensors stay F32;
- a tensor whose row length is not a multiple of a format's block size falls back to another format (`tensor_type_fallback`). For example, Qwen2.5-0.5B's 896-wide rows cannot hold 256-weight K-quant blocks, so they become Q5_0.

An importance matrix (`tools/imatrix`) can weight the quantization error by how much each weight matters on calibration text.

## 4. ggml: tensors, graphs, memory and backends

### Tensors and operations

A `ggml_tensor` (`ggml/include/ggml.h`) has:
- an element type;
- a shape `ne[4]` and byte strides `nb[4]`, so views, transposes and permutes cost nothing;
- a data pointer;
- an `op` and up to `GGML_MAX_SRC` source tensors.

Calling an operation such as `ggml_mul_mat(ctx, a, b)` or `ggml_rope_ext(...)` does not compute anything. It creates a result tensor that records the op and its sources.

### Graphs

`ggml_build_forward_expand(graph, result)` walks the sources back from a result and appends every reachable node to a `ggml_cgraph`, in dependency order. libllama builds one such graph per forward pass: `llama_context::process_ubatch` calls `model.build_graph`, which runs the architecture's `llm_build_*` constructor, for example `src/models/llama.cpp`. When the shapes allow it, the previous graph is reused (`can_reuse` in `src/llama-context.cpp`).

### Memory

`ggml-alloc` (`ggml/include/ggml-alloc.h`) plans the graph's intermediate tensors into one buffer:
- `ggml_gallocr_reserve` sizes the buffer for the largest graph;
- `ggml_gallocr_alloc_graph` places each tensor;
- a tensor's memory is reused once its last consumer has run.

The model's weights live in their own buffers, typically the memory-mapped file.

### Backends and the scheduler

**Backends.** A backend (`ggml/include/ggml-backend.h`) executes graphs on a device: CPU, CUDA, Metal, Vulkan, SYCL and others, each in its own `ggml/src/ggml-<name>/`.

**The scheduler.** `ggml_backend_sched` splits a graph across backends:
- it assigns each node to a backend that supports it and holds its weights;
- it inserts copies where a tensor crosses between devices;
- it runs the pieces.

libllama always goes through the scheduler, even CPU-only (`ggml_backend_sched_alloc_graph` in `src/llama-context.cpp`).

### The CPU backend

`ggml/src/ggml-cpu/` runs each node with a pool of threads. `ggml_graph_compute_thread` in `ggml-cpu.c` splits each op's work across the threads, with a barrier between nodes.

Its kernels exist in generic C and in hand-vectorized versions per instruction set (`ggml/src/ggml-cpu/arch/{x86,arm,riscv,...}`), and which one runs depends on the build:
- **A native build** (`-march=native`, the default for a local build) compiles the host's best kernels.
- **A portable build** (`GGML_CPU_ALL_VARIANTS` with `GGML_BACKEND_DL`) compiles several variants and loads the best one at run time, scored by `ggml_backend_cpu_x86_score` and its counterparts.

**Results depend on the host.**
- **Summation order:** the vector kernels sum in an order set by the vector width.
- **Contraction:** a native GCC build contracts `a*b + c` into fused multiply-adds, for example in RoPE's rotation.

So llama.cpp's logits are deterministic for a given commit, build and CPU, but not across them. synapse builds its numerics without contraction (`cmake/numerics.cmake`). Its tests therefore compare against llama.cpp with bounds that allow for this; see `tests/sdk/llama_golden.cpp`.

## 5. Quantization

### Block formats

ggml's quantized formats store a matrix row as consecutive **blocks**, each holding a fixed number of weights and its own scale (`ggml/src/ggml-common.h`):

| format | weights per block | bytes | layout |
|---|---|---|---|
| Q4_0 | 32 | 18 | fp16 scale `d`; 32 4-bit quants; `w = d·(q − 8)` |
| Q5_0 | 32 | 22 | fp16 `d`; 32-bit `qh` holding each quant's 5th bit; 32 4-bit low parts; `w = d·(q − 16)` |
| Q8_0 | 32 | 34 | fp16 `d`; 32 int8 quants; `w = d·q` |
| Q4_K | 256 | 144 | fp16 `d`, `dmin`; 8 sub-blocks of 32, each with a 6-bit scale and min; `w = d·sc·q − dmin·m` |
| Q6_K | 256 | 210 | fp16 `d`; 16 sub-blocks of 16 with int8 scales; 6-bit quants split into a 4-bit and a 2-bit plane |
| Q8_K | 256 | 292 | float `d`; 256 int8 quants; 16 int16 sub-block sums. Activations only, never weights |

The K-quants (Q2_K through Q6_K) are "super-blocks" of 256 weights whose sub-block scales are themselves quantized. That is how Q4_K reaches 4.5 bits per weight.

### Products: weights against quantized activations

The CPU backend never expands a quantized weight matrix to float for a matrix product. The flow, from `type_traits_cpu` in `ggml/src/ggml-cpu/ggml-cpu.c`:
- **`vec_dot_type`:** every weight type names one. It is Q8_0 for Q4_0, Q5_0 and Q8_0, Q8_K for the K-quants, and F16 for F16.
- **Quantizing the activations:** before a product the activations are converted to that type, once per row of activations (`from_float`).
- **The dot product:** each weight block is dotted with the matching activation block in integer arithmetic (`vec_dot`, e.g. `ggml_vec_dot_q4_K_q8_K`). Only the per-block scales are float.

Two consequences:
- **Activations are rounded.** Quantized matmuls round them to 8 bits, and an F16 weight's activations are rounded to F16. This is a large part of why a quantized model's logits differ from the original's.
- **It sets the order of arithmetic.** synapse reproduces both the activation formats and the order of the scale arithmetic:
  - `quant/q8.hpp` follows `quantize_row_q8_0_ref` and `quantize_row_q8_K_ref`;
  - `kernels/dot.hpp` follows the generic `ggml_vec_dot_*_q8_*`.

  So on the same inputs synapse's products match ggml's exactly, apart from float summation order.

### Dequantization

Embedding lookups and norm weights are expanded to float by each format's dequantizer (`dequantize_row_*` in `ggml/src/ggml-quants.c`). synapse's `quant/blocks.hpp` reproduces them operation for operation, and is checked bit-exact against `gguf-py`'s (`tests/sdk/quant_golden.cpp`).

## 6. libllama: the runtime

### Loading

`llama_model_load_from_file` goes through `llama_model_loader` (`src/llama-model-loader.cpp`), which maps the file and reads the metadata. `llama_model::load_hparams` (`src/llama-model.cpp`) then turns `<arch>.*` keys into `llama_hparams` (`src/llama-hparams.h`): layer count, widths, head counts, RoPE base, scaling and so on.

`load_tensors` binds each layer's weights by name. Tensors an architecture may lack are marked `TENSOR_NOT_REQUIRED`; Qwen2's biases and an untied `output.weight` are examples. A file with tensors the architecture does not use fails to load.

### The vocabulary

`llama_vocab` (`src/llama-vocab.cpp`) implements several tokenizer families (`LLAMA_VOCAB_TYPE_*` in `include/llama.h`):
- SentencePiece (SPM): Llama 2, TinyLlama, Mistral 7B;
- byte-level BPE: Llama 3, Qwen2, Mistral Nemo and later;
- WordPiece, Unigram and others.

For BPE, `tokenizer.ggml.pre` selects the pre-tokenizer's split regexes and flags such as `ignore_merges` and the default `add_bos`. Matching and splitting are in `src/unicode.cpp`: a few regexes have hand-written matchers (`unicode_regex_split_custom`), and the rest run through `std::regex` over a category-collapsed copy of the text. Special tokens are split out first (`tokenizer_st_partition`). Detokenization is `llama_token_to_piece`.

### Context settings

`llama_context` (`src/llama-context.cpp`) holds `llama_cparams`, the settings of this run: context size, batch and ubatch sizes, flash attention, KV cache types. It also derives values that depend on both the file and the run. YaRN's attention factor is one: `yarn_attn_factor` is computed in the `llama_context` constructor from the scaling factor and `rope.scaling.yarn_log_multiplier`.

### Decoding a batch

`llama_decode(ctx, batch)` → `llama_context::decode`:

1. **Split into ubatches.** The batch is split into micro-batches of at most `n_ubatch` tokens (`llama_batch_allocr`, `split_simple`, `src/llama-batch.cpp`).
2. **Find KV cells.** The memory module assigns each ubatch's tokens cells in the KV cache (`memory->init_batch` → `llama_kv_cache::find_slot`, `src/llama-kv-cache.cpp`).
3. **Build and run the graph.** For each ubatch the graph is built (or reused), allocated by the scheduler and computed (`process_ubatch`).

Inside it, `llm_graph_context` (`src/llama-graph.cpp`) supplies the shared pieces:
- `build_norm`;
- `build_ffn`, whose SiLU-gated FFN is `ggml_swiglu_split`;
- `build_attn`, which writes K and V into the cache (`cpy_k`, `cpy_v`) and attends over it.

Attention goes one of two ways:
- **Flash attention:** `ggml_flash_attn_ext`.
- **Otherwise:** `kq = ggml_mul_mat(k, q)` at F32 precision, then `ggml_soft_max_ext(kq, mask, scale)` with the causal mask, then a product with V.

4. **Keep only the requested outputs.** Only the tokens whose logits were asked for are carried through the last layer and the output projection (`inp_out_ids`). For a prompt that is normally just the last token.

### Sampling

The logits go to a sampler chain (`src/llama-sampler.cpp`: `llama_sampler_init_greedy`, `_top_k`, `_top_p`, `_temp`, `_dist`, ...) that picks the next token.

### Defaults that matter for numerics

- The KV cache is F16 (`type_k` and `type_v` default to `GGML_TYPE_F16`).
- Flash attention defaults to `auto`.
- synapse's oracle turns flash attention off and decodes one token per call: the path synapse implements (`scripts/llama_oracle/oracle.cpp`).

## 7. Where synapse sits

| GGML ecosystem | synapse | What differs, by design |
|---|---|---|
| `gguf.cpp`, `llama-mmap` | `gguf/` (`mapped_file`, `reader`, `metadata`, `tensor_view`) | Untrusted input is validated with no exceptions. The reader is written from the format, not ported, and checked against `gguf-py`. |
| `ggml-common.h`, `ggml-quants.c` | `quant/blocks.hpp`, `quant/q8.hpp`, `quant/scheme.hpp` | A format is a C++ type (a `Scheme`), so weights are typed views and kernels are templates, not a switch over `ggml_type`. |
| `ggml-cpu` kernels (`vec_dot_*`, rope, softmax, rms_norm) | `kernels/` (`dot`, `gemv`, `gemm`, `attention`, `elementwise`) | Built on MTL5's block-scaled integer dot. Each gemm row is bit-identical to gemv (G1). |
| `ggml_cgraph`, `ggml-alloc` | `graph/graph.hpp`, `graph/planner.hpp` | The graph is validated once at build time (invariants I1–I8). The arena is planned once. |
| `ggml_backend_sched`, CPU threadpool | `graph/session.hpp` on MTL5's thread pool | `run()` never allocates and never throws. Results are bit-identical for any thread count (D1). CPU only. |
| `llama_model`, `llm_build_llama` / `_qwen2` / `_mistral3`, `llama_hparams`, `llama_cparams` | `models/llama.hpp` | One loader for the Llama family. Unsupported features are rejected, never ignored. |
| `llama_kv_cache` | graph states (`kv_cache` in the builder) | One sequence, positions in order. No cell bookkeeping. |
| `llama_vocab`, `unicode.cpp` | `tokenizer/` (`bpe`, `spm`, `pretokenize`, `unicode`, `vocab`) | Hand-written matchers for every pre-tokenizer, reproducing llama.cpp's `std::regex` behaviour, including its case approximation for Tekken. |
| `llama-cli` (greedy) | `tools/llama_decode` | |

synapse also deliberately has no:
- GPU backends;
- multiple sequences;
- sampling beyond greedy;
- chat templates;
- KV cache shifting;
- training.

Which synapse files port llama.cpp or ggml code, and under which license, is listed in [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).

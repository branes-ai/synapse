# synapse

Tensor compute engine, graph execution framework, and spatial processing system for Embodied AI.

`synapse` is the Branes.AI ML runtime layer: a modern C++20 tensor graph engine that loads GGML/GGUF models, plans their memory statically, and executes them on **MTL5** linear-algebra kernels. It is one of three sister platform repos, alongside `cortex` and `reflex`, that mission applications are built from.

```text
+--------------+ +-------------------+ +-----------+ +------------+
| racing-drone | | interceptor-drone | | isr-drone | | nano-drone | ...  Mission
+--------------+ +-------------------+ +-----------+ +------------+
        |                  |                 |              |
        v                  v                 v              v
+---------------------+ +---------------------+ +---------------------+
|    Branes Cortex    | |   Branes Synapse    | |    Branes Reflex    |  Platform
| perception (VIO,    | | tensor graph engine | | stabilization, MPC, |
| SLAM), VLA, world   | | for GGUF models,    | | state propagation,  |
| model, planning     | | static memory plan  | | fault recovery      |
+---------------------+ +---------------------+ +---------------------+
           |                       |                       |
           v                       v                       v
+---------------------------------+ +---------------------------------+
|              MTL5               | |            Universal            |  Math
| high-performance linear algebra | |       computer arithmetic       |
+---------------------------------+ +---------------------------------+
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
+-----------+ +-------------------+ +-------------------+ +-----------+
|    CPU    | |     CPU + GPU     | |     CPU + KPU     | |    KPU    |  Hardware
+-----------+ +-------------------+ +-------------------+ +-----------+
```

## Where synapse fits

The Branes.AI stack has four tiers:

- **Mission**: one repo per vehicle and mission (`racing-drone`, `interceptor-drone`, `isr-drone`, `nano-drone`, ...). It owns the mission capability and composes platform components; algorithms do not live there.
- **Platform**: three sister repos, `cortex`, `synapse`, and `reflex`, that provide the functionality and services a mission is built from. None of them sits above the others.
- **Math**: [MTL5](https://github.com/stillwater-sc/mtl5) provides high-performance linear algebra and [Universal](https://github.com/stillwater-sc/universal) provides computer arithmetic (posits, intervals, mixed precision, hardware-aligned types).
- **Hardware**: the configurations the stack targets: CPU, CPU + GPU, CPU + KPU, and KPU. Today the stack runs on CPU; GPU and KPU support are planned, and the KPU is a stub until the silicon ships.

Arrows point from a tier to the tier below it that it builds on. They show how the tiers relate, not the dependencies of individual repos: each repo's build declares the libraries it actually uses.

The platform repos are organized after the nervous system:

| Repo | Biological analogue | Responsibility |
|---|---|---|
| [`cortex`](https://github.com/branes-ai/cortex) | Cerebral cortex | Perception (VIO, SLAM, SfM), vision-language-action models, world building, path planning |
| **`synapse`** (this repo) | The synapses: where signals are transmitted and computed | Tensor storage, graph construction, memory planning, and execution of ML models |
| [`reflex`](https://github.com/branes-ai/reflex) | Brainstem, spinal cord, autonomic system | Hard real-time control and safety: stabilization, IMU state propagation of the `cortex` estimate, MPC, fault recovery |

`cortex` decides *what* to compute (a VLA policy, a depth network, a learned motion model). `synapse` turns that model into a static, allocation-free compute graph and runs it. `reflex` consumes the results inside its bounded control loops and must never wait on anything that can allocate or stall, which is why synapse's execution path is designed to be deterministic as well.

## Design: GGML compatibility, MTL5 math

The core idea is a clean division of responsibilities:

- **GGML is an ML environment.** It is a file format (GGUF), a DAG execution scheduler, and a static memory planner. Its model ecosystem (Llama, Phi, Mistral, Qwen, LLaVA, and custom policy networks, all quantized and distributed as `.gguf`) is the reason to be compatible with it.
- **MTL5 is a math engine.** It provides C++20 concepts, expression templates, layout traits, vectorization, and number systems beyond IEEE floats through [Universal](https://github.com/stillwater-sc/universal) (posits, intervals, mixed precision, hardware-aligned types).

`synapse` reimplements the ML-environment half in modern C++20 and delegates the math to MTL5. The result reads GGML artifacts unchanged without carrying GGML's C code, `void*` dispatch, or `switch (ggml_type)` kernels.

```text
+-----------------------------------------------------------------------+
|                       Model Weights & Config                          |
|                         GGUF Parser / Loader                          |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                      C++20 Graph Builder                              |
|          (Defines Nodes, Activations, Quantization Traits)            |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                    Static Memory Allocator                            |
|        (Liveness Analysis -> Reusable MTL5 Slab Allocator)            |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                     Graph Execution Scheduler                         |
|       (Topological Sort -> Node Dispatch -> Expression Fused)         |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                       MTL5 Operator Kernels                           |
|      dense2D<T>, views, matmul, block-quantized (Q4_K, Q8_0, ...)     |
+-----------------------------------------------------------------------+
```

### Components

**GGUF frontend.** A native GGUF parser memory-maps the model file (POSIX `mmap`, Windows `CreateFileMapping`), reads the metadata key/value pairs (architecture, head counts, context length, embedding width) and the tensor-info table (shape, element type, offset), and binds tensor data to non-owning MTL5 views. Weights are never copied.

**Graph builder.** A strongly typed DAG of operations (`MatMul`, `RoPE`, `RMSNorm`, `Softmax`, ...) built from the model architecture. Element and quantization types are part of the node types, so mismatches are compile errors rather than runtime `switch` defaults.

**Static memory planner.** Before the first inference, a liveness pass computes the lifetime of every intermediate activation. Activations whose lifetimes do not overlap alias the same offset in one contiguous arena. Inference then performs no heap allocation, which is the property that makes GGML practical on embedded targets and the property `reflex` depends on.

**Execution engine.** The scheduler walks the graph in topological order and dispatches each node to an MTL5 kernel selected by concepts and traits instead of function pointers:

```cpp
template <typename WeightType, typename InputType, typename OutputType>
struct matmul_node : graph_node {
    mtl::matrix_view<WeightType> weights;
    mtl::vector_view<InputType>  input;
    mtl::vector_view<OutputType> output;

    void execute() override {
        // MTL5 handles block-scale unpacking, vectorization,
        // and expression-template fusion.
        output = mtl::matmul(weights, input);
    }
};
```

### Migration path for GGML users

**Tier 1: model compatibility.** Any standard `.gguf` file, downloaded from Hugging Face or produced by the `llama.cpp` conversion tools, loads without conversion.

```cpp
auto model  = branes::synapse::load_model("llama-3.2-1b-instruct.Q4_K_M.gguf");
auto logits = model.forward(input_tokens);
```

**Tier 2: API mirroring.** A C++20 surface that maps GGML's graph-construction concepts onto type-safe, scoped handles:

| GGML C API | synapse / MTL5 C++20 | Benefit |
|---|---|---|
| `struct ggml_tensor*` | `tensor<T, Layout>` | Type safety, bounds and layout traits |
| `ggml_add(ctx, a, b)` | `a + b` | Expression templates, lazy evaluation and fusion |
| `ggml_mul_mat(ctx, w, x)` | `matmul(w, x)` | Dispatch to vectorized / quantized MTL5 GEMM |
| `enum ggml_type` | `quantized<Scheme>` | Extensible compile-time type system (posits, custom bit widths) |

## Roadmap

1. **GGUF parser.** Header, metadata key/value pairs, and the tensor-info table; zero-copy memory mapping. *(Done: `branes/synapse/gguf/`. Validated against llama.cpp's `gguf-py` reader and writer.)*
2. **Quantized tensor views.** Block-quantized traits for the standard formats (`Q4_0`, `Q8_0`, `Q4_K`, `Q6_K`, `F16`, `BF16`) and views over raw GGUF buffers. *(Done: `branes/synapse/quant/`, `tensor/quantized_view.hpp`, `gguf/tensor_view.hpp`. Dequantizers bit-exact against llama.cpp's `gguf-py`.)*
3. **Fused quantized kernels.** Quantized matrix-vector and matrix-matrix products in MTL5 that unpack block scales in-register and accumulate in a wider type with SIMD dot products. *(Started: `kernels/gemv.hpp` and `kernels/gemm.hpp`, ggml-style integer dots for Q4_0/Q8_0/Q4_K/Q6_K, validated against ggml's own kernels; each gemm row is bit-identical to gemv. Matmul and attention run multithreaded on MTL5's thread pool, bit-identical for any thread count. Block dots use MTL5's block-scaled integer dot (one horizontal reduce per chunk, not per block) with rows x tokens tiling; see [docs/assessments/block-scaled-dot-performance.md](docs/assessments/block-scaled-dot-performance.md).)*
4. **Graph builder and memory planner.** A lightweight DAG, liveness analysis, and a single arena with no allocation during execution. *(Done: `graph/`, for decode and for batched prefill of up to `max_batch` tokens per run.)*
5. **Reference application.** Load a small GGUF model (e.g. Llama-3.2-1B or a vision-policy model) and verify logits against `llama.cpp`. *(Done for Llama decode: `models/llama.hpp`, RoPE and grouped-query attention over an F16 KV cache in `kernels/attention.hpp`, and `tools/llama_decode`. Logits are checked against llama.cpp on llama.cpp-generated fixtures and, opt-in, on Llama-3.2-1B Q4_K_M. Prompts prefill in batches, bit-identical to token-by-token decoding. Text prompts are tokenized by `tokenizer/bpe.hpp`, which matches llama.cpp's tokenizer on Llama-3 vocabularies.)*

## Building and testing

`synapse` is a header-only C++20 library (`branes::synapse`). It requires CMake >= 4.0 and Ninja, and builds with GCC, Clang, and MSVC.

```bash
cmake --preset gcc-debug              # also: gcc-release, clang-debug, clang-release, msvc, msvc-ninja, ...
cmake --build --preset gcc-debug
ctest --preset gcc-debug
```

Dependencies are pinned in [`cmake/deps.cmake`](cmake/deps.cmake) and resolved the same way as in `cortex` and `reflex`: an installed package found by `find_package` wins, otherwise the headers are fetched with FetchContent.

| Dependency | Role | When |
|---|---|---|
| [MTL5](https://github.com/stillwater-sc/mtl5) | Math engine; part of the `branes::synapse` interface | Always |
| [Universal](https://github.com/stillwater-sc/universal) | Number systems (posit, cfloat, fixpnt) for the type-generic tests | Tests only |
| [Catch2](https://github.com/catchorg/Catch2) v3 | Test framework | Tests only |
| [Highway](https://github.com/google/highway) | SIMD backend for MTL5's kernels (Apache-2.0 / BSD-3) | Opt-in: `-DSYNAPSE_WITH_HIGHWAY=ON` |

Highway selects its instruction set at compile time, so pair it with ISA flags: the `gcc-release-hwy` preset uses `-march=x86-64-v3 -maes -mpclmul` (AVX2; Highway's AVX2 target also requires AES and PCLMUL, and falls back to SSSE3 without them). Without Highway the kernels use MTL5's portable scalar path.

To develop against local checkouts of the Stillwater sister projects (no network):

```bash
cmake --preset gcc-debug \
  -DFETCHCONTENT_SOURCE_DIR_MTL5=/path/to/mtl5 \
  -DFETCHCONTENT_SOURCE_DIR_UNIVERSAL=/path/to/universal
```

Use it from another CMake project:

```cmake
FetchContent_Declare(synapse GIT_REPOSITORY https://github.com/branes-ai/synapse.git GIT_TAG <release tag>)
FetchContent_MakeAvailable(synapse)
target_link_libraries(my_target PRIVATE branes::synapse)
```

Tests, benchmarks, and tools are only built when synapse is the top-level project, so consumers fetch MTL5 and nothing else.

### Adding tests

Drop a `<name>.cpp` into `tests/sdk/` (unit tests) or `tests/examples/` (end-to-end examples). [`cmake/compile_all.cmake`](cmake/compile_all.cmake) builds one Catch2 executable per file and registers every `TEST_CASE` with CTest; no CMake edit is needed. File stems must be unique. Project targets compile with strict warnings as errors ([`cmake/warnings.cmake`](cmake/warnings.cmake)).

### CI and releases

- **CI** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)): clang-format lint, then build and test on Linux (GCC and Clang, Debug and Release) and Windows (MSVC, Debug and Release).
- **Conventional Commits** are enforced on PRs by [commitlint](commitlint.config.cjs).
- **release-please** turns merged commits into a release PR, `CHANGELOG.md`, and a `vX.Y.Z` tag.

## Repository layout

```text
synapse/
├── bench/        # micro-benchmarks (kernel throughput / latency)
├── cmake/        # dependency pins, test/warning helpers
├── docs/         # design notes, assessments, session records
├── scripts/      # developer and CI helper scripts, golden-fixture generators
├── sdk/          # the header-only library: sdk/include/branes/synapse/
├── tests/        # Catch2 regression tests, one executable per file; data/ holds golden fixtures
└── tools/        # host-side tools and reference applications (llama_decode)
```

## Status

Early stage. Build, dependency management, and CI are in place, mirroring `cortex` and `reflex`. The GGUF loader is the first component: a zero-copy, no-throw parser for GGUF v2/v3 that validates the header, metadata, and tensor table of an untrusted file, plus a read-only memory mapping on POSIX and Windows.

```cpp
#include <branes/synapse/gguf/gguf.hpp>

using namespace branes::synapse::gguf;

bool load(const std::filesystem::path& path) {
    auto file = mapped_file::open(path);  // result<mapped_file>
    if (!file) {
        return false;  // errc::io_error (os_code: errno / GetLastError()) or errc::truncated (empty file)
    }
    auto model = parse(file->bytes());  // result<gguf_file>
    if (!model) {
        return false;  // to_string(model.error().code), model.error().offset: what failed, and where
    }
    auto arch   = model->get<std::string_view>("general.architecture");  // "llama"
    auto* vocab = model->get_array("tokenizer.ggml.tokens");
    for (const auto& t : model->tensors()) {
        std::span<const std::byte> weights = model->tensor_data(t);  // view into the mapping
    }
    return true;  // keep `file` alive as long as `model` is used
}
```

Weights are then viewed in their storage format without copying. `visit_matrix` turns the tensor's runtime type tag into a compile-time scheme once, at the boundary:

```cpp
#include <branes/synapse/gguf/tensor_view.hpp>

const auto* wq = model->find_tensor("blk.0.attn_q.weight");
visit_matrix(*model, *wq, [](const auto& view) {   // quantized_matrix_view<scheme<q4_k>>, ...
    std::vector<float> row(view.cols());
    view.dequantize_row(0, row);                    // reference decode of one row
    mtl::mat::dense2D<float> w = view.to_dense();   // or the whole matrix, as MTL5
});
```

Fused kernels compute `y = W x` without expanding the weights. `x` is quantized once per call into 8-bit blocks, as llama.cpp does, and each block's dot product runs in exact integer arithmetic. The scratch memory comes from the caller, so inference never allocates:

```cpp
#include <branes/synapse/kernels/gemv.hpp>

// At load time: size the scratch once, for the most demanding matrix.
std::size_t scratch_bytes = 0;
for (const auto& t : model->tensors()) {
    visit_matrix(*model, t, [&](const auto& w) {
        using S = typename std::decay_t<decltype(w)>::scheme_type;
        scratch_bytes = std::max(scratch_bytes, kernels::gemv_scratch_bytes<S>(w.cols()));
    });
}
std::vector<std::byte> scratch(scratch_bytes);

// Per inference: no allocation.
visit_matrix(*model, *wq, [&](const auto& w) {
    kernels::gemv(w, x, y, scratch);
});
```

`bench/gemv_bench` compares the fused kernels with dequantize-then-dot.

A decode step is described once as a graph, planned into a single arena, and then run with no allocation. Weights are bound into the graph as typed views, so execution never switches on a storage format:

```cpp
#include <branes/synapse/graph/session.hpp>

using namespace branes::synapse::graph;

graph_builder b;                                   // no-throw: errors surface at finalize()
tensor_id x    = b.input(d);
tensor_id h    = b.rms_norm(x, ffn_norm, eps);
tensor_id act  = b.mul(b.silu(b.matmul(w_gate, h)), b.matmul(w_up, h));
tensor_id out  = b.add(x, b.matmul(w_down, act));
b.output(out);
auto g = std::move(b).finalize();                  // result<compute_graph, graph_error>

session s{*g};                                     // liveness plan + the one arena allocation
std::ranges::copy(input, s.tensor(x).begin());
s.run();                                           // no allocation, no throw
```

Tensors whose lifetimes don't overlap share memory. Four Llama-3.2-1B-width feed-forward blocks run in a 112 KiB arena, against 767 KiB without aliasing, and the arena stays that size however many blocks are added.

A whole Llama-architecture model is built from its GGUF metadata into one such graph, with a KV cache per layer that the session allocates beside the arena. Each run decodes one token at one position:

```cpp
#include <branes/synapse/models/llama.hpp>

auto model = models::load_llama(file, max_ctx, max_batch);  // result<llama_model, model_error>; validates every key and shape
graph::session s{model->graph, 0};                 // arena + KV cache: the only allocations; 0 = all pool threads
s.run(graph::batch{prompt, 0});                    // prefill: up to max_batch tokens per run
s.run({next_token, pos});                          // decode: one token per run
std::span<float> logits = s.tensor(model->logits); // for the run's last token
```

A batched run quantizes each token's activations once and unpacks every weight block once for the whole batch (`kernels::gemm`). Each gemm row performs exactly gemv's operations, so prefilling a prompt gives bit-for-bit the logits and KV cache of decoding it token by token. On Llama-3.2-1B, prefill runs 4.5x faster than token-by-token decoding (single thread).

Runs are multithreaded on MTL5's process-wide thread pool (sized by `MTL5_NUM_THREADS`), so synapse and MTL5's own dense kernels share one set of workers instead of oversubscribing the cores. A matmul quantizes its activations once and splits the weight rows across threads; attention splits across KV head groups. Every output is computed by exactly the operations of a single-threaded run, so results are bit-identical for any thread count, and a run still allocates nothing. On a 20-thread x86 machine, Llama-3.2-1B Q4_K_M decodes at 24 tokens/s with the portable kernels (2.65 on one thread) and 31 tokens/s with the AVX2 (Highway) build, where prompts prefill at 110 tokens/s.

Llama-3.2-1B Q4_K_M decodes in a 515 KiB arena. Its logits agree with llama.cpp's to ~1e-7 until the first 8-bit activation rounding that falls differently, then to a few percent, the same distance synapse moves from itself under a comparable perturbation. llama.cpp's top token is synapse's top token at every position of the test prompt. On an all-F32 model, where no activation is quantized, the two agree to 2e-7 at every position. `tools/llama_decode` runs greedy generation from a text prompt:

```text
$ llama_decode Llama-3.2-1B-Instruct-Q4_K_M.gguf -n 40 -p "The three primary colors are"
<|begin_of_text|>The three primary colors are red, blue, and yellow. These colors are combined in different ways to create various shades and hues of colors.
```

The tokenizer (`tokenizer/`) implements byte-level BPE with Llama 3's pre-tokenizer: special-token splitting, the split regex as a hand-written matcher over Unicode 15.1 character classes, and rank-ordered merges. It reproduces llama.cpp's `llama_tokenize` and `token_to_piece` exactly, on llama.cpp's own tokenizer test vectors and synapse's edge cases (scripts, emoji, whitespace runs, special tokens, invalid UTF-8), for a synthetic vocabulary in CI and for the real 128K Llama-3 vocabulary (opt-in).

## License

MIT. See [LICENSE](LICENSE).

# Introduction

`synapse` is the Branes.AI ML runtime layer: a modern C++20 tensor graph engine that loads GGML/GGUF models, plans their memory statically, and executes them on **MTL5** linear-algebra kernels. It is one of three sister platform repos, alongside `cortex` and `reflex`, that mission applications are built from.

## How the platform fits together

The Branes.AI stack has five tiers:

- **Mission**: one repo per vehicle and mission. `racing-drone` and `scout-drone` are the two that stand up the architecture; more follow (`interceptor-drone`, `isr-drone`, `nano-drone`, `counter-uas`). A mission repo owns its task: its world, the slots that see it, its stack and scorer, and its data. Algorithms do not live there.
- **Mission framework**: [`anatomy`](https://github.com/branes-ai/anatomy), the body plan every mission composes its vehicle with: the data contracts, the slot concepts and their composition, configs, the replay log, the harness, and the closed-loop simulator. It holds no algorithms and no mission.
- **Platform**: three sister repos, `cortex`, `synapse`, and `reflex`, that provide the functionality and services a mission is built from. None of them sits above the others.
- **Math**: [MTL5](https://github.com/stillwater-sc/mtl5) provides high-performance linear algebra and [Universal](https://github.com/stillwater-sc/universal) provides computer arithmetic (posits, intervals, mixed precision, hardware-aligned types).
- **Hardware**: CPU, CPU + GPU, CPU + KPU, and KPU. Today the stack runs on CPU; GPU and KPU support are planned, and the KPU is a stub until the silicon ships.

The platform repos are organized after the nervous system, and the mission framework, anatomy, is the body plan they are assembled into:

| Repo | Biological analogue | Responsibility |
|---|---|---|
| [`cortex`](https://github.com/branes-ai/cortex) | Cerebral cortex | Perception (VIO, SLAM, SfM), vision-language-action models, world building, path planning |
| **`synapse`** (this repo) | The synapses: where signals are transmitted and computed | Tensor storage, graph construction, memory planning, and execution of ML models |
| [`reflex`](https://github.com/branes-ai/reflex) | Brainstem, spinal cord, autonomic system | Hard real-time control and safety: stabilization, IMU state propagation of the `cortex` estimate, MPC, fault recovery |
| [`anatomy`](https://github.com/branes-ai/anatomy) | The body plan | The mission framework: data contracts, slots and composition, configs, replay log, harness, closed-loop simulator |

`cortex` decides *what* to compute (a VLA policy, a depth network, a learned motion model). `synapse` turns that model into a static, allocation-free compute graph and runs it. `reflex` consumes the results inside its bounded control loops and must never wait on anything that can allocate or stall, which is why synapse's execution path is designed to be deterministic as well. The full stack diagram is in the [README](../../README.md).

## Design: GGML compatibility, MTL5 math

The core idea is a clean division of responsibilities:

- **The GGML ecosystem is an ML environment.** ggml is a tensor library with a DAG execution scheduler and a static memory planner, GGUF is its model container format, and its model ecosystem (Llama, Phi, Mistral, Qwen, LLaVA, and custom policy networks, distributed as `.gguf` in full precision and quantized variants) is the reason to be compatible with it. [The GGML ecosystem](../arch/ggml-ecosystem.md) describes how it is put together, and [what a Llama model computes](../arch/llama-model.md) walks through the model itself.
- **MTL5 is a math engine.** It provides C++20 concepts, expression templates, layout traits, vectorization, and number systems beyond IEEE floats through Universal.

`synapse` reimplements the ML-environment half in modern C++20 and delegates the math to MTL5. The result reads GGML artifacts unchanged without carrying GGML's C code, `void*` dispatch, or `switch (ggml_type)` kernels.

## The components

- **GGUF frontend** (`gguf/`): a native parser memory-maps the model file, reads the metadata key/value pairs and the tensor-info table, and binds tensor data to non-owning MTL5 views. Weights are never copied.
- **Quantized tensor views** (`quant/`, `gguf/tensor_view.hpp`): block-quantized traits for `Q4_0`, `Q5_0`, `Q8_0`, `Q4_K`, `Q6_K`, `F16` and `BF16`, over MTL5's `block_matrix_view`.
- **Fused kernels** (`kernels/`): quantized matrix-vector and matrix-matrix products that unpack block scales in-register and accumulate each block's dot product in exact integer arithmetic, plus attention and the element-wise operations. They run multithreaded on MTL5's thread pool, bit-identical for any thread count.
- **Graph builder and memory planner** (`graph/`): a strongly typed DAG of operations, a liveness pass that aliases activations whose lifetimes do not overlap into one arena, and a session that runs the graph with no allocation.
- **Models** (`models/`): Llama-family text models (`llama`, `qwen2`, `mistral3`) and the SigLIP image encoder of SmolVLM's `mmproj` file, built from their GGUF metadata.
- **Tokenizers** (`tokenizer/`): SentencePiece and byte-level BPE, reproducing llama.cpp's `llama_tokenize` and `token_to_piece` exactly.

Paths are relative to `sdk/include/branes/synapse/`. The [C++ API reference](https://branes-ai.github.io/synapse/api/) documents each header.

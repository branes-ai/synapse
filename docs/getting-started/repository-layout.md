# Repository Layout

```text
synapse/
├── bench/        # micro-benchmarks (kernel throughput / latency)
├── cmake/        # dependency pins, test/warning helpers
├── docs/         # design notes, assessments, plans, session records: the source of this site
├── docs-site/    # the Starlight site generated from docs/, published to GitHub Pages
├── scripts/      # developer and CI helper scripts, golden-fixture generators
├── sdk/          # the header-only library: sdk/include/branes/synapse/
├── tests/        # Catch2 regression tests, one executable per file; data/ holds golden fixtures
└── tools/        # host-side tools and reference applications (llama_decode)
```

## The library

```text
sdk/include/branes/synapse/
├── gguf/         # GGUF parser, memory mapping, tensor types, typed tensor views
├── quant/        # block-quantization schemes and their 8-bit activation blocks
├── kernels/      # fused quantized gemv / gemm, block dots, attention, element-wise operations
├── graph/        # graph builder, liveness planner, session (the one arena)
├── models/       # Llama-family text models and the CLIP / SigLIP image encoder
├── tokenizer/    # SentencePiece and byte-level BPE, and their Unicode tables
└── numeric/      # bit-exact widening of F16 / BF16 to float
```

## Where the documents are

| Folder | Contents |
|---|---|
| [`getting-started/`](introduction.md) | This introduction, building and testing, and this layout |
| [`arch/`](../arch/ggml-ecosystem.md) | The GGML ecosystem, and what a Llama model computes |
| [`assessments/`](../assessments/block-scaled-dot-performance.md) | Analyses and trade studies |
| [`sessions/`](../sessions/) | Dated records of development sessions, on GitHub only |
| `plan/` | One plan per issue, published on the [architecture site](https://branes-ai.github.io/architecture/synapse/) |

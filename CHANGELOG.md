# Changelog

## [0.12.0](https://github.com/branes-ai/synapse/compare/v0.11.1...v0.12.0) (2026-10-24)


### Features

* support Q5_0 weights, enabling Qwen2.5-0.5B ([#46](https://github.com/branes-ai/synapse/issues/46)) ([f21405f](https://github.com/branes-ai/synapse/commit/f21405faa754e6011a0c3a880e6553e3a38c111d)), closes [#43](https://github.com/branes-ai/synapse/issues/43)

## [0.11.1](https://github.com/branes-ai/synapse/compare/v0.11.0...v0.11.1) (2026-10-23)


### Continuous Integration

* stop release PRs leaving expired and failed runs ([#42](https://github.com/branes-ai/synapse/issues/42)) ([67d76d1](https://github.com/branes-ai/synapse/commit/67d76d1f710d602899d0721c175f4a455c7f7638))

## [0.11.0](https://github.com/branes-ai/synapse/compare/v0.10.1...v0.11.0) (2026-10-23)


### ⚠ BREAKING CHANGES

* view weights as MTL5's block_matrix_view, replacing quantized_matrix_view

### Bug Fixes

* reject a GGUF tensor whose row byte count overflows ([505f826](https://github.com/branes-ai/synapse/commit/505f826b8da99a71f4a41834978d4e74ee2d78a4))


### Code Refactoring

* view weights as MTL5's block_matrix_view, replacing quantized_matrix_view ([505f826](https://github.com/branes-ai/synapse/commit/505f826b8da99a71f4a41834978d4e74ee2d78a4))

## [0.10.1](https://github.com/branes-ai/synapse/compare/v0.10.0...v0.10.1) (2026-10-23)


### Documentation

* explain the GGML ecosystem and what a Llama model computes ([#37](https://github.com/branes-ai/synapse/issues/37)) ([8137115](https://github.com/branes-ai/synapse/commit/813711599c50c1766dccdd02b21d832bac3121a3))

## [0.10.0](https://github.com/branes-ai/synapse/compare/v0.9.1...v0.10.0) (2026-10-23)


### Features

* **models:** mistral support: mistral3 with YaRN and attention temperature, tekken tokenizer ([#33](https://github.com/branes-ai/synapse/issues/33)) ([f8f4409](https://github.com/branes-ai/synapse/commit/f8f44099a0b84137bb6464873d5bae8deb238c84))

## [0.9.1](https://github.com/branes-ai/synapse/compare/v0.9.0...v0.9.1) (2026-10-23)


### Documentation

* **readme:** qualify hardware targets and explain the stack diagram arrows ([#27](https://github.com/branes-ai/synapse/issues/27)) ([84a062d](https://github.com/branes-ai/synapse/commit/84a062df445e16cf2a0aa36a4b4512218202c553))

## [0.9.0](https://github.com/branes-ai/synapse/compare/v0.8.1...v0.9.0) (2026-10-23)


### Features

* **tokenizer:** sentencepiece tokenizer for llama-2-family vocabularies ([#26](https://github.com/branes-ai/synapse/issues/26)) ([5f432d8](https://github.com/branes-ai/synapse/commit/5f432d8d2d74a2954af14c220016dec79cddebaf))

## [0.8.1](https://github.com/branes-ai/synapse/compare/v0.8.0...v0.8.1) (2026-10-23)


### Performance

* **kernels:** use mtl5 5.20's block-scaled integer dot for quantized products ([#24](https://github.com/branes-ai/synapse/issues/24)) ([7ea1db9](https://github.com/branes-ai/synapse/commit/7ea1db948220ccdd547e5122e4a00bcd4cbd2169))

## [0.8.0](https://github.com/branes-ai/synapse/compare/v0.7.0...v0.8.0) (2026-10-22)


### Features

* **graph:** multithreaded runs on mtl5's thread pool, bit-identical for any thread count ([#22](https://github.com/branes-ai/synapse/issues/22)) ([419f3d1](https://github.com/branes-ai/synapse/commit/419f3d1c25cac36d9961da81249a4455d716679f))

## [0.7.0](https://github.com/branes-ai/synapse/compare/v0.6.1...v0.7.0) (2026-10-22)


### Features

* **tokenizer:** llama-3 byte-level bpe tokenizer matching llama.cpp ([#20](https://github.com/branes-ai/synapse/issues/20)) ([67508f2](https://github.com/branes-ai/synapse/commit/67508f21bdc0028b558145f787e037f3ae8cec7a))

## [0.6.1](https://github.com/branes-ai/synapse/compare/v0.6.0...v0.6.1) (2026-10-21)


### Documentation

* **readme:** four-tier stack diagram with cortex, synapse, and reflex as sisters ([#17](https://github.com/branes-ai/synapse/issues/17)) ([d2fc0a6](https://github.com/branes-ai/synapse/commit/d2fc0a60fdd82912b73b78caab43a8f6c082acac))

## [0.6.0](https://github.com/branes-ai/synapse/compare/v0.5.0...v0.6.0) (2026-10-21)


### Features

* **kernels:** batched prefill with a gemm bit-identical to gemv ([#16](https://github.com/branes-ai/synapse/issues/16)) ([b75eeb4](https://github.com/branes-ai/synapse/commit/b75eeb426d5895f038c3529b1e712c4ea39dfccc))

## [0.5.0](https://github.com/branes-ai/synapse/compare/v0.4.0...v0.5.0) (2026-10-21)


### Features

* **models:** llama decode graph from gguf, verified against llama.cpp ([#14](https://github.com/branes-ai/synapse/issues/14)) ([2ca4266](https://github.com/branes-ai/synapse/commit/2ca426613d46a1ea00a0146722080046a6cf94ff))

## [0.4.0](https://github.com/branes-ai/synapse/compare/v0.3.1...v0.4.0) (2026-10-21)


### Features

* **graph:** graph builder, liveness-based arena planner, and allocation-free session ([#11](https://github.com/branes-ai/synapse/issues/11)) ([1870015](https://github.com/branes-ai/synapse/commit/1870015fad45c89012b2b8677a093ef9f2e6a197))

## [0.3.1](https://github.com/branes-ai/synapse/compare/v0.3.0...v0.3.1) (2026-10-21)


### Continuous Integration

* release every merge automatically ([#10](https://github.com/branes-ai/synapse/issues/10)) ([4ff3810](https://github.com/branes-ai/synapse/commit/4ff3810725192dd546d1d78d735fdedf489c8bbb))

## [0.3.0](https://github.com/branes-ai/synapse/compare/v0.2.0...v0.3.0) (2026-10-21)


### Features

* **kernels:** fused quantized gemv with ggml-style integer dot products ([#8](https://github.com/branes-ai/synapse/issues/8)) ([253c273](https://github.com/branes-ai/synapse/commit/253c273f8963fa8cd4f3b1622b8ba140b9bd962a))

## [0.2.0](https://github.com/branes-ai/synapse/compare/v0.1.1...v0.2.0) (2026-10-21)


### Features

* **gguf:** zero-copy, no-throw gguf v2/v3 parser and memory-mapped loader ([#5](https://github.com/branes-ai/synapse/issues/5)) ([5553c34](https://github.com/branes-ai/synapse/commit/5553c3447a07a752ce896bf13e30f012f02396c0))
* **quant:** bit-exact ggml block formats and zero-copy quantized matrix views ([#6](https://github.com/branes-ai/synapse/issues/6)) ([b942881](https://github.com/branes-ai/synapse/commit/b94288183f359222af5c6ab72da364b0750eb4b9))

## [0.1.1](https://github.com/branes-ai/synapse/compare/v0.1.0...v0.1.1) (2026-10-21)


### Continuous Integration

* move actions off the deprecated node.js 20 runtime ([#3](https://github.com/branes-ai/synapse/issues/3)) ([6adaa48](https://github.com/branes-ai/synapse/commit/6adaa48a324d4fcb7261e3e2399a6f76589c73ee))

## 0.1.0 (2026-10-21)


### Build System

* cmake, mtl5/universal deps, and ci mirroring cortex and reflex ([#1](https://github.com/branes-ai/synapse/issues/1)) ([68908f6](https://github.com/branes-ai/synapse/commit/68908f6aaede999d6437757ebceac4b134c28b23))

# Changelog

## [0.13.4](https://github.com/branes-ai/synapse/compare/v0.13.3...v0.13.4) (2026-10-26)


### Documentation

* add the 2026-10-25 session log ([#59](https://github.com/branes-ai/synapse/issues/59)) ([2f4da61](https://github.com/branes-ai/synapse/commit/2f4da617c1ef93ec3bc3baa8d647f5d328314377))

## [0.13.3](https://github.com/branes-ai/synapse/compare/v0.13.2...v0.13.3) (2026-10-25)


### Performance

* register-tiled self-attention for the image encoder ([#57](https://github.com/branes-ai/synapse/issues/57)) ([98b68a4](https://github.com/branes-ai/synapse/commit/98b68a4f6fafe1577ac9a2da8f646fbf7546c8c2))

## [0.13.2](https://github.com/branes-ai/synapse/compare/v0.13.1...v0.13.2) (2026-10-25)


### Performance

* gelu by table lookup, split across threads ([#55](https://github.com/branes-ai/synapse/issues/55)) ([abeea35](https://github.com/branes-ai/synapse/commit/abeea35e6f8fab1c149ee1f73df557fac693b66a))

## [0.13.1](https://github.com/branes-ai/synapse/compare/v0.13.0...v0.13.1) (2026-10-25)


### Performance

* register-blocked quantized gemm and dynamic thread scheduling ([#51](https://github.com/branes-ai/synapse/issues/51)) ([f6dc7f4](https://github.com/branes-ai/synapse/commit/f6dc7f4501079b520ee7c409996d890179a0bde9))

## [0.13.0](https://github.com/branes-ai/synapse/compare/v0.12.0...v0.13.0) (2026-10-25)


### Features

* run llama.cpp mmproj image encoders, SigLIP with the idefics3 projector ([#48](https://github.com/branes-ai/synapse/issues/48)) ([ac7faf7](https://github.com/branes-ai/synapse/commit/ac7faf73160427d2fb638d6651d9e0a4d67cfc38)), closes [#45](https://github.com/branes-ai/synapse/issues/45)

## [0.12.0](https://github.com/branes-ai/synapse/compare/v0.11.1...v0.12.0) (2026-10-24)


### Features

* support Q5_0 weights, enabling Qwen2.5-0.5B ([#46](https://github.com/branes-ai/synapse/issues/46)) ([abc0f9e](https://github.com/branes-ai/synapse/commit/abc0f9e6db5463fe29692da5ecdca576b6472066)), closes [#43](https://github.com/branes-ai/synapse/issues/43)

## [0.11.1](https://github.com/branes-ai/synapse/compare/v0.11.0...v0.11.1) (2026-10-23)


### Continuous Integration

* stop release PRs leaving expired and failed runs ([#42](https://github.com/branes-ai/synapse/issues/42)) ([22f7ba6](https://github.com/branes-ai/synapse/commit/22f7ba68d4809c5f4407140359f801dbb19e83bc))

## [0.11.0](https://github.com/branes-ai/synapse/compare/v0.10.1...v0.11.0) (2026-10-23)


### ⚠ BREAKING CHANGES

* view weights as MTL5's block_matrix_view, replacing quantized_matrix_view

### Bug Fixes

* reject a GGUF tensor whose row byte count overflows ([6dbb2f4](https://github.com/branes-ai/synapse/commit/6dbb2f458a30d4b57d415128748bdfdbf8f3081e))


### Code Refactoring

* view weights as MTL5's block_matrix_view, replacing quantized_matrix_view ([6dbb2f4](https://github.com/branes-ai/synapse/commit/6dbb2f458a30d4b57d415128748bdfdbf8f3081e))

## [0.10.1](https://github.com/branes-ai/synapse/compare/v0.10.0...v0.10.1) (2026-10-23)


### Documentation

* explain the GGML ecosystem and what a Llama model computes ([#37](https://github.com/branes-ai/synapse/issues/37)) ([1f9cc9f](https://github.com/branes-ai/synapse/commit/1f9cc9f8ac129abfccc19cc2911357eb46b6c26a))

## [0.10.0](https://github.com/branes-ai/synapse/compare/v0.9.1...v0.10.0) (2026-10-23)


### Features

* **models:** mistral support: mistral3 with YaRN and attention temperature, tekken tokenizer ([#33](https://github.com/branes-ai/synapse/issues/33)) ([3cbf40f](https://github.com/branes-ai/synapse/commit/3cbf40fc29b35decbacf73d53f2899cf9dcae8a6))

## [0.9.1](https://github.com/branes-ai/synapse/compare/v0.9.0...v0.9.1) (2026-10-23)


### Documentation

* **readme:** qualify hardware targets and explain the stack diagram arrows ([#27](https://github.com/branes-ai/synapse/issues/27)) ([639ec4b](https://github.com/branes-ai/synapse/commit/639ec4b5a13cb31fb5e538c9741dc466b16a9d67))

## [0.9.0](https://github.com/branes-ai/synapse/compare/v0.8.1...v0.9.0) (2026-10-23)


### Features

* **tokenizer:** sentencepiece tokenizer for llama-2-family vocabularies ([#26](https://github.com/branes-ai/synapse/issues/26)) ([15ebc47](https://github.com/branes-ai/synapse/commit/15ebc47d67d27f82e00094826c7afa1afaf7cd8d))

## [0.8.1](https://github.com/branes-ai/synapse/compare/v0.8.0...v0.8.1) (2026-10-23)


### Performance

* **kernels:** use mtl5 5.20's block-scaled integer dot for quantized products ([#24](https://github.com/branes-ai/synapse/issues/24)) ([743ad8e](https://github.com/branes-ai/synapse/commit/743ad8ed296f1b67033f13486748c3ff7336edc8))

## [0.8.0](https://github.com/branes-ai/synapse/compare/v0.7.0...v0.8.0) (2026-10-22)


### Features

* **graph:** multithreaded runs on mtl5's thread pool, bit-identical for any thread count ([#22](https://github.com/branes-ai/synapse/issues/22)) ([0a93916](https://github.com/branes-ai/synapse/commit/0a9391692ed248e8371d9d646821c468dab7fe07))

## [0.7.0](https://github.com/branes-ai/synapse/compare/v0.6.1...v0.7.0) (2026-10-22)


### Features

* **tokenizer:** llama-3 byte-level bpe tokenizer matching llama.cpp ([#20](https://github.com/branes-ai/synapse/issues/20)) ([16737ce](https://github.com/branes-ai/synapse/commit/16737ce3e6e7e033289be3ca08353be000e7d897))

## [0.6.1](https://github.com/branes-ai/synapse/compare/v0.6.0...v0.6.1) (2026-10-21)


### Documentation

* **readme:** four-tier stack diagram with cortex, synapse, and reflex as sisters ([#17](https://github.com/branes-ai/synapse/issues/17)) ([c594a59](https://github.com/branes-ai/synapse/commit/c594a598a37bb1b2d4f048efc9d0d9e3b9ad6127))

## [0.6.0](https://github.com/branes-ai/synapse/compare/v0.5.0...v0.6.0) (2026-10-21)


### Features

* **kernels:** batched prefill with a gemm bit-identical to gemv ([#16](https://github.com/branes-ai/synapse/issues/16)) ([2adbce4](https://github.com/branes-ai/synapse/commit/2adbce45e2b3453b6b84d9f9fc7eb044c10ce1ee))

## [0.5.0](https://github.com/branes-ai/synapse/compare/v0.4.0...v0.5.0) (2026-10-21)


### Features

* **models:** llama decode graph from gguf, verified against llama.cpp ([#14](https://github.com/branes-ai/synapse/issues/14)) ([c8dc436](https://github.com/branes-ai/synapse/commit/c8dc436fbf3fd6b08fe1a40974bd95c6ec0d7754))

## [0.4.0](https://github.com/branes-ai/synapse/compare/v0.3.1...v0.4.0) (2026-10-21)


### Features

* **graph:** graph builder, liveness-based arena planner, and allocation-free session ([#11](https://github.com/branes-ai/synapse/issues/11)) ([b0c9e50](https://github.com/branes-ai/synapse/commit/b0c9e5087617f1a01c74badf90c858ba4aeab4c8))

## [0.3.1](https://github.com/branes-ai/synapse/compare/v0.3.0...v0.3.1) (2026-10-21)


### Continuous Integration

* release every merge automatically ([#10](https://github.com/branes-ai/synapse/issues/10)) ([5da4342](https://github.com/branes-ai/synapse/commit/5da4342f8e20546365e64ca36b66a5fcd623c0ec))

## [0.3.0](https://github.com/branes-ai/synapse/compare/v0.2.0...v0.3.0) (2026-10-21)


### Features

* **kernels:** fused quantized gemv with ggml-style integer dot products ([#8](https://github.com/branes-ai/synapse/issues/8)) ([3385dce](https://github.com/branes-ai/synapse/commit/3385dced7caef63d90cf95d80378ad3944b737ea))

## [0.2.0](https://github.com/branes-ai/synapse/compare/v0.1.1...v0.2.0) (2026-10-21)


### Features

* **gguf:** zero-copy, no-throw gguf v2/v3 parser and memory-mapped loader ([#5](https://github.com/branes-ai/synapse/issues/5)) ([eafbf89](https://github.com/branes-ai/synapse/commit/eafbf89f9c944975b5b94d30aeb242bdf60d3fcc))
* **quant:** bit-exact ggml block formats and zero-copy quantized matrix views ([#6](https://github.com/branes-ai/synapse/issues/6)) ([fde741d](https://github.com/branes-ai/synapse/commit/fde741d52c6b1affcac84a71f52ba8900a1c95bf))

## [0.1.1](https://github.com/branes-ai/synapse/compare/v0.1.0...v0.1.1) (2026-10-21)


### Continuous Integration

* move actions off the deprecated node.js 20 runtime ([#3](https://github.com/branes-ai/synapse/issues/3)) ([fef79ce](https://github.com/branes-ai/synapse/commit/fef79cefb963e9a723a31aa942263042deb77ba6))

## 0.1.0 (2026-10-21)


### Build System

* cmake, mtl5/universal deps, and ci mirroring cortex and reflex ([#1](https://github.com/branes-ai/synapse/issues/1)) ([018b309](https://github.com/branes-ai/synapse/commit/018b3092cfc22ecd860d8e76575681725666024d))

# Changelog

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

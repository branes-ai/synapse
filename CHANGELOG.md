# Changelog

## [0.15.2](https://github.com/branes-ai/synapse/compare/v0.15.1...v0.15.2) (2026-10-31)


### Continuous Integration

* give each run on main a concurrency group of its own ([#74](https://github.com/branes-ai/synapse/issues/74)) ([a1e1e56](https://github.com/branes-ai/synapse/commit/a1e1e5635e83f2c30eff3b536df4fa431aa1a22a))
* lint the PR title as well as its commits ([#75](https://github.com/branes-ai/synapse/issues/75)) ([ad90a45](https://github.com/branes-ai/synapse/commit/ad90a45882771e24d3987596c64db935493c3a2c))

## [0.15.1](https://github.com/branes-ai/synapse/compare/v0.15.0...v0.15.1) (2026-10-30)


### Continuous Integration

* check the plans and ADRs on every PR ([#72](https://github.com/branes-ai/synapse/issues/72)) ([601860b](https://github.com/branes-ai/synapse/commit/601860b13862b181762c30fbae679bda852a1061))

## [0.15.0](https://github.com/branes-ai/synapse/compare/v0.14.0...v0.15.0) (2026-10-30)


### Features

* **docs-site:** build the site with docs-kit v0.1.0 ([#70](https://github.com/branes-ai/synapse/issues/70)) ([746c177](https://github.com/branes-ai/synapse/commit/746c177c42a8131c52078fd0c727de98542a6ae1))

## [0.14.0](https://github.com/branes-ai/synapse/compare/v0.13.7...v0.14.0) (2026-10-29)


### Features

* **docs-site:** publish docs/ as a github pages site ([#68](https://github.com/branes-ai/synapse/issues/68)) ([71dbf52](https://github.com/branes-ai/synapse/commit/71dbf52e786bc86cda9bcdbfee4b3f6b4df11584))

## [0.13.7](https://github.com/branes-ai/synapse/compare/v0.13.6...v0.13.7) (2026-10-28)


### Documentation

* the five-tier stack diagram, with anatomy and scout-drone ([#66](https://github.com/branes-ai/synapse/issues/66)) ([e463869](https://github.com/branes-ai/synapse/commit/e4638691e290aaa8dfab5c06a21387627d5d6e82))

## [0.13.6](https://github.com/branes-ai/synapse/compare/v0.13.5...v0.13.6) (2026-10-27)


### Documentation

* add the 2026-10-26 session log ([#63](https://github.com/branes-ai/synapse/issues/63)) ([3123fea](https://github.com/branes-ai/synapse/commit/3123fea980d81594660162f7a9bc9d218de275a6))

## [0.13.5](https://github.com/branes-ai/synapse/compare/v0.13.4...v0.13.5) (2026-10-27)


### Documentation

* link the mtl5 vectorized exp issue from the session log ([#61](https://github.com/branes-ai/synapse/issues/61)) ([fea1162](https://github.com/branes-ai/synapse/commit/fea1162198f420637409e2ec7c8321716467b1e2))

## [0.13.4](https://github.com/branes-ai/synapse/compare/v0.13.3...v0.13.4) (2026-10-26)


### Documentation

* add the 2026-10-25 session log ([#59](https://github.com/branes-ai/synapse/issues/59)) ([e1c54a7](https://github.com/branes-ai/synapse/commit/e1c54a74f27d22acec0c146d679b37458ea08eb6))

## [0.13.3](https://github.com/branes-ai/synapse/compare/v0.13.2...v0.13.3) (2026-10-25)


### Performance

* register-tiled self-attention for the image encoder ([#57](https://github.com/branes-ai/synapse/issues/57)) ([1aa0812](https://github.com/branes-ai/synapse/commit/1aa08129727e7ef9cc5b89cc98d2d1f47358ccaf))

## [0.13.2](https://github.com/branes-ai/synapse/compare/v0.13.1...v0.13.2) (2026-10-25)


### Performance

* gelu by table lookup, split across threads ([#55](https://github.com/branes-ai/synapse/issues/55)) ([7749f12](https://github.com/branes-ai/synapse/commit/7749f12951273db96a6fd288e4dec5807661f04c))

## [0.13.1](https://github.com/branes-ai/synapse/compare/v0.13.0...v0.13.1) (2026-10-25)


### Performance

* register-blocked quantized gemm and dynamic thread scheduling ([#51](https://github.com/branes-ai/synapse/issues/51)) ([d68f4a8](https://github.com/branes-ai/synapse/commit/d68f4a8322496134fadd2aeb341625e9085e1786))

## [0.13.0](https://github.com/branes-ai/synapse/compare/v0.12.0...v0.13.0) (2026-10-25)


### Features

* run llama.cpp mmproj image encoders, SigLIP with the idefics3 projector ([#48](https://github.com/branes-ai/synapse/issues/48)) ([2a53439](https://github.com/branes-ai/synapse/commit/2a53439ed5ad581d00b33fa22949113be42788a8)), closes [#45](https://github.com/branes-ai/synapse/issues/45)

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

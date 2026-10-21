# Changelog

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

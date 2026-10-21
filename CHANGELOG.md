# Changelog

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

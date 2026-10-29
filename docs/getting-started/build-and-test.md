# Build & Test

`synapse` is a header-only C++20 library (`branes::synapse`). The CMake build exists to compile and run the tests, benchmarks and tools. It requires CMake ≥ 4.0 and Ninja, and builds with GCC, Clang, and MSVC, using the same presets as the sibling repos.

## Build and run the tests

```bash
cmake --preset gcc-debug              # also: gcc-release, clang-debug, clang-release, msvc, msvc-ninja, ...
cmake --build --preset gcc-debug
ctest --preset gcc-debug
```

Dependencies are pinned in [`cmake/deps.cmake`](../../cmake/deps.cmake): an installed package found by `find_package` wins, otherwise the headers are fetched with FetchContent.

| Dependency | Role | When |
|---|---|---|
| [MTL5](https://github.com/stillwater-sc/mtl5) | Math engine; part of the `branes::synapse` interface | Always |
| [Universal](https://github.com/stillwater-sc/universal) | Number systems (posit, cfloat, fixpnt) for the type-generic tests | Tests only |
| [Catch2](https://github.com/catchorg/Catch2) v3 | Test framework | Tests only |
| [Highway](https://github.com/google/highway) | SIMD backend for MTL5's kernels | Opt-in: `-DSYNAPSE_WITH_HIGHWAY=ON` |

Highway selects its instruction set at compile time, so pair it with ISA flags: the `gcc-release-hwy` preset uses `-march=x86-64-v3 -maes -mpclmul` (AVX2; Highway's AVX2 target also requires AES and PCLMUL, and falls back to SSSE3 without them). Without Highway the kernels use MTL5's portable scalar path.

To develop against local checkouts of the Stillwater sister projects (no network):

```bash
cmake --preset gcc-debug \
  -DFETCHCONTENT_SOURCE_DIR_MTL5=/path/to/mtl5 \
  -DFETCHCONTENT_SOURCE_DIR_UNIVERSAL=/path/to/universal
```

## Run a model

`tools/llama_decode` runs greedy generation from a text prompt on a Llama-family GGUF file:

```text
$ llama_decode Llama-3.2-1B-Instruct-Q4_K_M.gguf -n 40 -p "The three primary colors are"
<|begin_of_text|>The three primary colors are red, blue, and yellow. These colors are combined in different ways to create various shades and hues of colors.
```

Its options are in [`tools/README.md`](../../tools/README.md). `bench/gemv_bench` compares the fused kernels with dequantize-then-dot ([`bench/README.md`](../../bench/README.md)).

## Use synapse from another CMake project

```cmake
FetchContent_Declare(synapse GIT_REPOSITORY https://github.com/branes-ai/synapse.git GIT_TAG <release tag>)
FetchContent_MakeAvailable(synapse)
target_link_libraries(my_target PRIVATE branes::synapse)
```

Tests, benchmarks, and tools are only built when synapse is the top-level project, so consumers fetch MTL5 and nothing else.

## Adding tests

Drop a `<name>.cpp` into `tests/sdk/` (unit tests) or `tests/examples/` (end-to-end examples). [`cmake/compile_all.cmake`](../../cmake/compile_all.cmake) builds one Catch2 executable per file and registers every `TEST_CASE` with CTest; no CMake edit is needed. File stems must be unique. Project targets compile with strict warnings as errors ([`cmake/warnings.cmake`](../../cmake/warnings.cmake)).

## CI and releases

- **CI** ([`ci.yml`](../../.github/workflows/ci.yml)): clang-format lint, a build of this site, then build and test on Linux (GCC and Clang, Debug and Release) and Windows (MSVC, Debug and Release).
- **Conventional Commits** are enforced on PRs by [commitlint](../../commitlint.config.cjs).
- **release-please** turns merged commits into a release PR, `CHANGELOG.md`, and a `vX.Y.Z` tag.
- **The docs site** ([`deploy-docs.yml`](../../.github/workflows/deploy-docs.yml)) is rebuilt from `docs/` and published on every merge to main that touches the documents, the site, or the public headers.

# ADR 0001 — Third-party license compatibility

**Status:** Accepted
**Date:** 2026-10-31
**Source:** the stack's license posture, set in cortex's [ADR 0001](https://github.com/branes-ai/cortex/blob/main/docs/adr/0001-third-party-license-compatibility.md); synapse's notices, [#34](https://github.com/branes-ai/synapse/issues/34)
**Related:** [ADR-0002](0002-no-gpl-port-only-permissive.md)

## Context

synapse is MIT-licensed (see the top-level [`LICENSE`](../../LICENSE)), like the rest of the Branes.AI stack. It is meant to be redistributed inside partners' products, so every dependency, and every file of someone else's work it contains, has to be compatible with MIT and carry its notices. cortex records the same posture in its ADR 0001. synapse had followed it without recording it, and differs from cortex in one way: parts of synapse are ports of llama.cpp and ggml, rather than only linked dependencies.

## Dependency and source license matrix

| Dependency or source | License | Compatible with MIT synapse? | How it is used | Obligation |
|---|---|---|---|---|
| MTL5 | MIT | Yes | The linear algebra and the block-scaled kernels | Bundle its LICENSE in a distribution |
| Universal | MIT | Yes | Tests only: the tests instantiate the runtime with posits and cfloats | Bundle its LICENSE with source |
| Highway | Apache-2.0 **or** BSD-3-Clause | Yes | Opt-in (`SYNAPSE_WITH_HIGHWAY`): MTL5's SIMD backend | We take the BSD-3-Clause branch: bundle its notice, and don't use the authors' names for endorsement |
| Catch2 v3 | BSL-1.0 | Yes | Tests only | None for object code; bundle with source |
| llama.cpp and ggml | MIT | Yes | Ported code: the tokenizers, the quantized block formats, the dot-product arithmetic, RoPE and YaRN; one vocab file copied | Notice in [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md), and each ported header points there |
| YaRN | MIT | Yes | Ported, through ggml's `rope_yarn` | Notice in `THIRD_PARTY_NOTICES.md` |
| Unicode Character Database | Unicode License v3 | Yes | `tokenizer/unicode_tables.hpp` is generated from `UnicodeData.txt` | Notice in `THIRD_PARTY_NOTICES.md` |

`cmake/deps.cmake` takes MTL5 and Universal from an installed package or a parent project's target when there is one, and Highway from MTL5's target, and fetches them with FetchContent otherwise. The license is the dependency's, however it arrives: a build that supplies its own copy must supply one under the same license.

The test fixtures that `scripts/` writes (the `*_golden.gguf` files and the tiny models) are llama.cpp's output, not its code.

**Model weights are not part of synapse.** The real-model tests (`SYNAPSE_LLAMA_MODEL`, `SYNAPSE_MMPROJ_MODEL`, …) are opt-in and read models the developer already has. The weights carry their own licenses, some of them not open source (the Llama models' community license), and synapse neither commits nor redistributes them.

No GPL, LGPL, AGPL, MPL or SSPL dependency is accepted. [ADR-0002](0002-no-gpl-port-only-permissive.md) is the wider policy: no GPL code for any purpose, and what may be ported. An exception would need an ADR of its own that supersedes this one.

## Decision

1. **synapse stays MIT-licensed.** Every dependency and every ported source must be compatible with MIT.
2. **[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md) is the ledger** of everything in synapse that is someone else's work: each ported file, its source, and the license text. A PR that ports code, copies a file or generates one from a third-party source must add it there. Each ported header must say so and point to the ledger.
3. **A PR that adds a dependency must add it to this table**, with its license and its obligation. A dependency that is dual-licensed is used under its permissive branch.
4. **Model weights stay outside the repo.** Only fixtures that synapse's scripts generate, or files whose license is in the ledger, are committed.

## Consequences

- The ledger and this table together answer a partner's license review: what synapse links, and what it contains.
- Porting from llama.cpp keeps synapse's numerics bit-comparable with it. The cost is the bookkeeping in decision 2.
- **Scope.** This covers what synapse ships, the code that builds it and the tests. The docs site's toolchain is not part of it. Astro brings in sharp and libvips (LGPL-3.0) to process images, but they only build the documentation, in CI and on developers' machines.
- **Current state:** synapse's dependencies and ported sources are those in the table, and `THIRD_PARTY_NOTICES.md` lists every ported file ([#34](https://github.com/branes-ai/synapse/issues/34)). No release artifact bundles the dependencies' license files yet, because synapse has no packaged release.

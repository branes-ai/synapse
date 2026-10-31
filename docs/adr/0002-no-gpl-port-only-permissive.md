# ADR 0002 — No GPL contact; port only from permissive sources

**Status:** Accepted
**Date:** 2026-10-31
**Source:** the stack's no-GPL policy, set in cortex's [ADR 0002](https://github.com/branes-ai/cortex/blob/main/docs/adr/0002-no-gpl-clean-room-policy.md)
**Related:** [ADR-0001](0001-third-party-license-compatibility.md)

## Context

cortex's ADR 0002 rules out GPL code anywhere in cortex, for any purpose. The reason applies to synapse unchanged: partners redistribute the stack inside their own certified products, and any GPL obligation, even one confined to a test, is something their legal reviews would have to clear.

synapse works differently from cortex in one respect. cortex implements its operators clean-room from papers. synapse's job is to run the models llama.cpp runs, with numerics that match llama.cpp's, so parts of it are deliberately ports of llama.cpp and ggml code. Those are MIT-licensed, and the port is recorded ([ADR-0001](0001-third-party-license-compatibility.md)). This ADR records where that line is.

## Decision

**No GPL-licensed code anywhere in synapse, for any purpose.** Not as a dependency, not in a test target, not as a build-time tool in what synapse builds, and not as a reference oracle.

- **Porting is allowed from permissive sources only:** MIT, BSD, Apache-2.0, or a license of the same kind. Each port is recorded in `THIRD_PARTY_NOTICES.md`, and the ported header says so (ADR-0001, decision 2).
- **Check a source's license before opening its code.** Reading GPL source and then writing the same thing "from scratch" is neither a port nor clean-room. A GPL project may be consulted for its structure only (how it splits the problem), never for code.
- **Otherwise, implement from papers, specifications and file formats.** The GGUF reader, for example, is written from the format, and checked against llama.cpp's `gguf-py`.
- **Test oracles must be permissive too.** The golden outputs come from llama.cpp (MIT), written by `scripts/`. Never compare against a GPL program's output.
- LGPL, MPL and SSPL candidates are flagged for explicit review and default to rejected.

## Consequences

- synapse can follow llama.cpp closely where matching it matters (tokenization, quantized formats, the kernels' arithmetic), at the cost of recording each port.
- A contributor who wants to port from a new project checks its license first, and adds it to the ledger in the same PR.
- Partners can redistribute synapse under MIT terms with no copyleft review.
- **Current state:** every port in synapse is from MIT sources (llama.cpp, ggml and YaRN), listed in `THIRD_PARTY_NOTICES.md`. No GPL code or oracle is used.

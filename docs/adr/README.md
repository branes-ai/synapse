# Architecture decision records

One file per load-bearing decision in synapse: what was decided, why, and what it commits us to. The format follows cortex's [`docs/adr/template.md`](https://github.com/branes-ai/cortex/blob/main/docs/adr/template.md): Status / Date / Source / Related, Context, Decision, Consequences, and a closing *Current state* line. Files are numbered from 0001 and named after the decision: `0001-short-title.md`.

The decisions are published, with their status, on the [architecture site](https://branes-ai.github.io/architecture/synapse/), next to the other repos' decisions, rather than on synapse's own site.

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-third-party-license-compatibility.md) | Third-party license compatibility (MIT; permissive dependencies and ports only) | Accepted (amended) |
| [0002](0002-no-gpl-port-only-permissive.md) | No GPL code for any purpose; port only from permissive sources | Accepted |

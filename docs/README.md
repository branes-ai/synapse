# Documentation map

These documents are published at <https://branes-ai.github.io/synapse/>. The site is generated from this folder on every merge to main ([plan](plan/docs-site/README.md)): edit the documents here, never the site. [`index.mdx`](index.mdx) is the site's landing page, and this file is its documentation map.

Start here:

- [The GGML ecosystem](arch/ggml-ecosystem.md): how the GGML ecosystem (GGUF, ggml, libllama, the conversion and quantization tools) is put together, and which part of it each synapse component corresponds to
- [What a Llama model computes](arch/llama-model.md): what a Llama-family model computes, step by step, why it is built that way, what it costs, and where synapse's numerics meet llama.cpp's

| Folder | Contents |
|---|---|
| [getting-started/](getting-started/introduction.md) | [Introduction](getting-started/introduction.md), [build & test](getting-started/build-and-test.md), [repository layout](getting-started/repository-layout.md) |
| [arch/](arch/ggml-ecosystem.md) | Architecture and design: [the GGML ecosystem](arch/ggml-ecosystem.md) and where synapse fits, [what a Llama model computes](arch/llama-model.md); later the GGUF loader, graph IR, memory planner and kernel dispatch |
| [assessments/](assessments/block-scaled-dot-performance.md) | Analyses and trade studies: [block-scaled dot performance](assessments/block-scaled-dot-performance.md), adopting MTL5's block-scaled integer dot in the quantized kernels |
| [sessions/](sessions/) | Dated records of development sessions, kept in the repo and not published on the site |
| plan/ | Per-issue plans, each with its issues and PRs: [docs site](plan/docs-site/README.md) |
| [adr/](adr/README.md) | Architecture decision records: [third-party license compatibility](adr/0001-third-party-license-compatibility.md), [no GPL contact; port only from permissive sources](adr/0002-no-gpl-port-only-permissive.md) |

The plans and the decisions are published on the [architecture site](https://branes-ai.github.io/architecture/synapse/), alongside every other repo's plans and decisions and their status, rather than here.

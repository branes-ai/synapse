# tools/

Host-side developer tools and reference applications: GGUF inspectors, graph
dumpers, model runners, and logit diffing against `llama.cpp`. Tools may use
heavier dependencies than the header-only library, but nothing here is a
dependency of `branes::synapse`.

Each `<name>.cpp` here is built as a standalone executable (see
`CMakeLists.txt`; `-DSYNAPSE_BUILD_TOOLS=OFF` skips them). They need a model
file to run, so none is registered with CTest.

| Tool | What it does |
|---|---|
| `llama_decode` | Greedy decoding of a Llama-architecture GGUF model: `llama_decode model.gguf [-n new] [-c ctx] [-b batch] (-p "text" \| token ids...)`. The prompt is tokenized (Llama-3 BPE vocabularies; special tokens like `<\|eot_id\|>` in the text are recognized) or given as ids, prefilled in runs of up to `-b` tokens (default 64), then tokens are generated one at a time until `-n` or the end-of-sequence token. Prints the text, with the prefill and decode rates. |

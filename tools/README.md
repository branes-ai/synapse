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
| `llama_decode` | Greedy decoding of a Llama-architecture GGUF model: `llama_decode model.gguf [-n new] [-c ctx] [token ids...]`. Prompt tokens are given as ids (synapse has no tokenizer yet); the output is shown as text from the model's vocabulary, with the decode rate. |

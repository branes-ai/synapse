# docs/

Engineering documentation that lives with the code: design notes, assessments,
and session records.

Start here:

- [ggml-ecosystem.md](ggml-ecosystem.md): how the GGML ecosystem (GGUF, ggml, libllama, the conversion and quantization tools) is put together, and which part of it each synapse component corresponds to
- [llama-model.md](llama-model.md): what a Llama-family model computes, step by step, why it is built that way, what it costs, and where synapse's numerics meet llama.cpp's

Suggested layout as content lands:

- `arch/`: architecture and design decisions (GGUF loader, graph IR, memory planner, kernel dispatch)
- `assessments/`: analyses and trade studies (e.g. quantization formats vs. accuracy, kernel performance vs. llama.cpp)
- `sessions/`: dated records of development sessions

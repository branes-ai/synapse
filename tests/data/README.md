# tests/data

Golden fixtures for the regression tests. Every file is regenerable with the
script named below; none is hand-edited.

| File | What it is | Produced by |
|---|---|---|
| `quant_golden.gguf` | Quantized blocks and their dequantized values | `scripts/gen_quant_fixture.py` (gguf-py) |
| `kernel_golden.gguf` | Weights, quantized activations, and ggml's `vec_dot` results | `scripts/gen_kernel_fixture.py` (ggml) |
| `llama_tiny.gguf`, `llama_tiny_golden.gguf` | Tiny random-weight Llama model (quantized) and llama.cpp's logits | `scripts/gen_llama_fixture.py tiny` |
| `llama_tiny_f32.gguf`, `llama_tiny_f32_golden.gguf` | The same with all-F32 weights | `scripts/gen_llama_fixture.py tiny-f32` |
| `qwen2_tiny.gguf`, `qwen2_tiny_golden.gguf` | The quantized tiny model as Qwen2 (NEOX RoPE, Q/K/V and output biases) and llama.cpp's logits | `scripts/gen_llama_fixture.py qwen2-tiny` |
| `qwen2_tiny_f32.gguf`, `qwen2_tiny_f32_golden.gguf` | The same with all-F32 weights and its own output matrix | `scripts/gen_llama_fixture.py qwen2-tiny-f32` |
| `bpe_vocab.gguf`, `bpe_golden.gguf` | Synthetic Llama-3-style BPE vocabulary and llama.cpp's tokenizations | `scripts/gen_tokenizer_fixture.py synthetic` |
| `qwen2_vocab.gguf`, `qwen2_golden.gguf` | The synthetic vocabulary with Qwen2's pre-tokenizer, and llama.cpp's tokenizations of its Qwen2 test vectors | `scripts/gen_tokenizer_fixture.py --pre qwen2 --cases qwen2 synthetic` |
| `spm_vocab.gguf` | llama.cpp's Llama-2 SentencePiece vocabulary, `models/ggml-vocab-llama-spm.gguf`, copied unchanged from llama.cpp (MIT License, Copyright (c) 2023-2026 The ggml authors; checkout `e6f2ec01f`) | copied from llama.cpp |
| `spm_golden.gguf` | llama.cpp's tokenizations of `spm_vocab.gguf` | `scripts/gen_tokenizer_fixture.py --cases spm golden` |

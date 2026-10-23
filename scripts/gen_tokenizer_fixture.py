#!/usr/bin/env python3
"""Generate golden tokenizations from llama.cpp for synapse's BPE tokenizer tests.

Two modes:

  # the CI fixture: a small synthetic byte-level BPE vocabulary (the 256 byte
  # tokens, merges trained on a built-in multilingual corpus, and special
  # tokens), with llama.cpp's tokenizations of the test cases
  python3 scripts/gen_tokenizer_fixture.py --llama-cpp /path/to/llama.cpp \\
      [--build-dir DIR] synthetic tests/data/bpe_vocab.gguf tests/data/bpe_golden.gguf

  # the same vocabulary with Qwen2's pre-tokenizer (single-digit numbers, no
  # ignore_merges, no BOS), checked on llama.cpp's Qwen2 test vectors
  python3 scripts/gen_tokenizer_fixture.py --llama-cpp /path/to/llama.cpp \\
      [--build-dir DIR] --pre qwen2 --cases qwen2 synthetic tests/data/qwen2_vocab.gguf tests/data/qwen2_golden.gguf

  # golden tokenizations for an existing vocabulary (e.g. llama.cpp's
  # models/ggml-vocab-llama-bpe.gguf, or a full model), for the opt-in test
  python3 scripts/gen_tokenizer_fixture.py --llama-cpp /path/to/llama.cpp \\
      [--build-dir DIR] golden vocab.gguf golden.gguf

  # the SentencePiece CI fixture: llama.cpp's Llama-2 vocabulary
  # (models/ggml-vocab-llama-spm.gguf, copied to tests/data/spm_vocab.gguf)
  python3 scripts/gen_tokenizer_fixture.py --llama-cpp /path/to/llama.cpp \\
      [--build-dir DIR] --cases spm golden tests/data/spm_vocab.gguf tests/data/spm_golden.gguf

The test cases are llama.cpp's own tokenizer test vectors
(models/ggml-vocab-{llama-bpe,llama-spm,qwen2}.gguf.inp, per --cases) followed by
synapse's additional edge cases for that tokenizer family. The pre-tokenizer's behaviour does not depend on the vocabulary, so
the small vocabulary exercises it fully; the merges exercise BPE.
"""

import argparse
import collections
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_llama_fixture import build_oracle, run  # noqa: E402

SEPARATOR = "\n__ggml_vocab_test__\n"

CORPUS = """\
The quick brown fox jumps over the lazy dog. It's a test: we're testing, they've tested, I'll test.
Hello world! Hello, World. HELLO WORLD? hello_world = 42; print(hello_world)
Numbers 0 1 12 123 1234 12345 1234567890 3.14159 2026-10-21 100,000 and 7%.
Die Äpfel sind grün und süß. Straße, Müller, Größe.
Привет, мир! Как дела? Это тест токенизатора.
你好，世界。这是一个测试。日本語のテキストも少し。
مرحبا بالعالم. שלום עולם.
Emoji: 🦙🚀✨ and symbols ← → ∑ ∞ ≠ ° ©.
def tokenize(text):\n    return [t for t in text.split() if t]\n\tindented\r\nwindows line\n\n\n
    four spaces, two  spaces,   three, and trailing   \n
"""

SPECIALS = [("<|begin_of_text|>", 3), ("<|end_of_text|>", 3), ("<|eot_id|>", 3), ("<|custom|>", 4)]

EXTRA_CASES = [
    "it's IT'S It'S we're WE'RE they've I'll I'd you'D 'll 's",
    "'s at the start, and an apostrophe alone ' in the middle '",
    "1 22 333 4444 55555 666666 7777777 ½ ² ٣٤٥ ⅷ",
    "x1y22z333 a1b2c3 3rd 21st",
    "Ünïcödé çàfé naïve résumé — “quotes” ‘single’ …",
    "mixed中文and English文字",
    "🦙🦙 🦙.cpp 👍🏽 👨‍👩‍👧",
    "tabs\tand\t\tdouble\ttabs",
    "line1\nline2\r\nline3\rline4\n\n\nline5",
    "   leading spaces",
    "trailing spaces   ",
    " \n \n ",
    "a  b   c    d",
    "!!!??? ...,,, ;;;:::",
    " (parenthesized) [bracketed] {braced} <angled>",
    "<|begin_of_text|>Hello<|eot_id|>",
    "text with <|custom|> inside and <|custom|><|custom|> twice",
    "almost special <|begin_of_tex|> and <|eot_id",
    "<|end_of_text|>",
    " non-breaking em space　ideographic",
    "invalid utf-8: \udcff\udcfe and a lone continuation \udc80 here",
    "",
]


# SentencePiece-specific cases: the space prefix around special tokens, the
# U+2581 escape, byte fallback for characters outside the vocabulary, and
# raw invalid UTF-8 (kept byte for byte by SPM's byte fallback).
SPM_EXTRA_CASES = [
    "<s>Hello</s>",
    "<s> Hello <s>world</s>",
    "text<unk>more <unk> text",
    "already\u2581escaped\u2581text",
    "rare characters: 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 ꙮ 🫠 ᚠᚢᚦ",
    "  two leading spaces and trailing  ",
    "tabs\tand\nnewlines\r\n",
    "invalid utf-8: \udcff\udcfe and a lone continuation \udc80 here",
    "\udce2AB truncated lead",
    "",
]


def bpe_train(corpus: str, n_merges: int) -> list[tuple[str, str]]:
    """Merges of a byte-level BPE trained on corpus (training only shapes the
    vocabulary; encoding correctness is checked against llama.cpp)."""
    byte_map = gpt2_bytes_to_unicode()
    words = collections.Counter(
        tuple(byte_map[b] for b in w.encode()) for w in re.findall(r" ?\w+| ?[^\w\s]+|\s+", corpus))
    merges = []
    for _ in range(n_merges):
        pairs = collections.Counter()
        for w, n in words.items():
            for a, b in zip(w, w[1:]):
                pairs[(a, b)] += n
        if not pairs:
            break
        best = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))[0]
        merges.append(best)
        merged = collections.Counter()
        for w, n in words.items():
            out, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and (w[i], w[i + 1]) == best:
                    out.append(w[i] + w[i + 1])
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            merged[tuple(out)] += n
        words = merged
    return merges


def gpt2_bytes_to_unicode() -> dict[int, str]:
    printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    shifted = 0
    out = {}
    for b in range(256):
        if b in printable:
            out[b] = chr(b)
        else:
            out[b] = chr(256 + shifted)
            shifted += 1
    return out


def write_vocab(llama: Path, path: Path, pre: str) -> None:
    sys.path.insert(0, str(llama / "gguf-py"))
    import gguf  # noqa: PLC0415 (from the llama.cpp checkout)

    byte_map = gpt2_bytes_to_unicode()
    tokens = [byte_map[b] for b in range(256)]
    merges = bpe_train(CORPUS, 400)
    for a, b in merges:
        if a + b not in tokens:
            tokens.append(a + b)
    types = [1] * len(tokens)
    bos = len(tokens)
    for text, ty in SPECIALS:
        tokens.append(text)
        types.append(ty)

    w = gguf.GGUFWriter(str(path), "llama")
    w.add_name(f"synapse synthetic {pre} vocab")
    for key, val in [("block_count", 1), ("context_length", 128), ("embedding_length", 64),
                     ("feed_forward_length", 128), ("attention.head_count", 4), ("attention.head_count_kv", 4),
                     ("rope.dimension_count", 16), ("vocab_size", len(tokens))]:
        w.add_uint32(f"llama.{key}", val)
    w.add_float32("llama.rope.freq_base", 500000.0)
    w.add_float32("llama.attention.layer_norm_rms_epsilon", 1e-5)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre(pre)
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges([f"{a} {b}" for a, b in merges])
    w.add_bos_token_id(bos)
    w.add_eos_token_id(bos + 1)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {path}: {len(tokens)} tokens, {len(merges)} merges")


UPSTREAM_CASES = {"bpe": "ggml-vocab-llama-bpe.gguf.inp", "spm": "ggml-vocab-llama-spm.gguf.inp",
                  "qwen2": "ggml-vocab-qwen2.gguf.inp"}


def cases_file(llama: Path, directory: Path, family: str) -> Path:
    upstream = (llama / "models" / UPSTREAM_CASES[family]).read_bytes()
    # Lone surrogates above stand for raw invalid bytes.
    cases = SPM_EXTRA_CASES if family == "spm" else EXTRA_CASES
    extra = b"".join(c.encode("utf-8", "surrogateescape") + SEPARATOR.encode() for c in cases)
    path = directory / "cases.txt"
    path.write_bytes(upstream + extra)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", required=True, type=Path, help="path to a llama.cpp checkout")
    ap.add_argument("--build-dir", type=Path, help="llama.cpp build directory (default: a temp dir)")
    ap.add_argument("--cases", choices=sorted(UPSTREAM_CASES), default="bpe",
                    help="tokenizer family whose test vectors to use (default: bpe)")
    ap.add_argument("--pre", choices=["llama-bpe", "qwen2"], default="llama-bpe",
                    help="pre-tokenizer of the synthetic vocabulary (default: llama-bpe)")
    ap.add_argument("mode", choices=["synthetic", "golden"])
    ap.add_argument("vocab", type=Path, help="vocabulary to write (synthetic) or read (golden)")
    ap.add_argument("golden", type=Path, help="golden tokenizations to write")
    args = ap.parse_args()

    llama = args.llama_cpp.resolve()
    oracle = build_oracle(llama, args.build_dir)
    if args.mode == "synthetic":
        args.vocab.parent.mkdir(parents=True, exist_ok=True)
        write_vocab(llama, args.vocab, args.pre)
    with tempfile.TemporaryDirectory() as tmp:
        cases = cases_file(llama, Path(tmp), args.cases)
        args.golden.parent.mkdir(parents=True, exist_ok=True)
        run([str(oracle), "tokenize", str(args.vocab), str(cases), str(args.golden)])
    print(f"wrote {args.golden} ({args.golden.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

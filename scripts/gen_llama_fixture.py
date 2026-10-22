#!/usr/bin/env python3
"""Generate golden logits from llama.cpp for synapse's Llama decode tests.

Builds libllama (with ggml) from a local llama.cpp checkout, compiles
scripts/llama_oracle/oracle.cpp against it, and runs it. See oracle.cpp for
what the files hold.

Usage:
  # the CI fixture: a tiny random-weight model and its golden logits
  python3 scripts/gen_llama_fixture.py --llama-cpp /path/to/llama.cpp \\
      [--build-dir DIR] tiny tests/data/llama_tiny.gguf tests/data/llama_tiny_golden.gguf

  # the same with all-F32 weights (graph semantics without quantization noise)
  python3 scripts/gen_llama_fixture.py --llama-cpp /path/to/llama.cpp \\
      [--build-dir DIR] tiny-f32 tests/data/llama_tiny_f32.gguf tests/data/llama_tiny_f32_golden.gguf

  # golden logits of a real model, for the opt-in llama_model test
  python3 scripts/gen_llama_fixture.py --llama-cpp /path/to/llama.cpp \\
      [--build-dir DIR] logits model.gguf golden.gguf 128000 791 6864 315 9822 374

The llama.cpp build is reused if --build-dir already holds one. Results are
deterministic for a given llama.cpp commit and CPU (ggml dispatches SIMD
kernels by host ISA, which sets the float summation order).
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_oracle(llama_cpp: Path, build_dir: Path | None) -> Path:
    """Build libllama from the checkout (reusing build_dir) and the oracle against it."""
    llama = llama_cpp.resolve()
    build = (build_dir or Path(tempfile.mkdtemp(prefix="synapse-llama-"))).resolve()
    libs = [build / "src" / "libllama.a", build / "ggml" / "src" / "libggml.a",
            build / "ggml" / "src" / "libggml-cpu.a", build / "ggml" / "src" / "libggml-base.a"]

    if not all(lib.exists() for lib in libs):
        run(["cmake", "-S", str(llama), "-B", str(build), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
             "-DBUILD_SHARED_LIBS=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
             "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_TOOLS=OFF", "-DLLAMA_CURL=OFF", "-DGGML_OPENMP=OFF"])
    run(["cmake", "--build", str(build), "--target", "llama", "ggml"])

    oracle = build / "llama_oracle"
    run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2", f"-I{llama / 'include'}",
         f"-I{llama / 'ggml' / 'include'}", str(HERE / "llama_oracle" / "oracle.cpp"), "-o", str(oracle),
         *map(str, libs), "-lpthread", "-lm"])
    return oracle


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", required=True, type=Path, help="path to a llama.cpp checkout")
    ap.add_argument("--build-dir", type=Path, help="llama.cpp build directory (default: a temp dir)")
    ap.add_argument("mode", choices=["tiny", "tiny-f32", "logits"])
    ap.add_argument("model", type=Path, help="model to write (tiny) or read (logits)")
    ap.add_argument("golden", type=Path, help="golden logits to write")
    ap.add_argument("tokens", nargs="*", type=int, help="token ids to decode (logits mode)")
    args = ap.parse_args()
    if args.mode == "logits" and not args.tokens:
        ap.error("logits mode needs at least one token")

    oracle = build_oracle(args.llama_cpp, args.build_dir)
    for path in (args.model, args.golden):
        path.parent.mkdir(parents=True, exist_ok=True)
    run([str(oracle), args.mode, str(args.model), str(args.golden), *map(str, args.tokens)])
    print(f"wrote {args.golden} ({args.golden.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

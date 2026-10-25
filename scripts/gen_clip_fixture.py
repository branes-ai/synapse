#!/usr/bin/env python3
"""Generate golden image embeddings from llama.cpp for synapse's mmproj encoder tests.

Builds libmtmd (clip.cpp, with ggml) from a local llama.cpp checkout,
compiles scripts/clip_oracle/oracle.cpp against it, and runs it. See
oracle.cpp for what the files hold.

Usage:
  # the CI fixtures: tiny random-weight idefics3 mmproj files and their embeddings
  python3 scripts/gen_clip_fixture.py --llama-cpp /path/to/llama.cpp \\
      [--build-dir DIR] tiny tests/data/clip_tiny.gguf tests/data/clip_tiny_golden.gguf
  python3 scripts/gen_clip_fixture.py --llama-cpp /path/to/llama.cpp \\
      [--build-dir DIR] tiny-q8 tests/data/clip_tiny_q8.gguf tests/data/clip_tiny_q8_golden.gguf

  # golden embeddings of a real mmproj, for the opt-in clip_model test
  python3 scripts/gen_clip_fixture.py --llama-cpp /path/to/llama.cpp \\
      [--build-dir DIR] encode mmproj.gguf golden.gguf

The llama.cpp build is reused if --build-dir already holds one; it is
configured with the tools enabled (libmtmd lives under tools/), so use a
directory of its own. Results are deterministic for a given llama.cpp commit
and CPU (ggml dispatches SIMD kernels by host ISA).
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
    """Build libmtmd from the checkout (reusing build_dir) and the oracle against it."""
    llama = llama_cpp.resolve()
    build = (build_dir or Path(tempfile.mkdtemp(prefix="synapse-mtmd-"))).resolve()
    libs = [build / "tools" / "mtmd" / "libmtmd.a", build / "src" / "libllama.a", build / "ggml" / "src" / "libggml.a",
            build / "ggml" / "src" / "libggml-cpu.a", build / "ggml" / "src" / "libggml-base.a"]

    if not all(lib.exists() for lib in libs):
        run(["cmake", "-S", str(llama), "-B", str(build), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
             "-DBUILD_SHARED_LIBS=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
             "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_TOOLS=ON", "-DLLAMA_CURL=OFF", "-DGGML_OPENMP=OFF"])
    run(["cmake", "--build", str(build), "--target", "mtmd", "llama", "ggml"])

    oracle = build / "clip_oracle"
    run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2", f"-I{llama / 'tools' / 'mtmd'}", f"-I{llama / 'include'}",
         f"-I{llama / 'ggml' / 'include'}", str(HERE / "clip_oracle" / "oracle.cpp"), "-o", str(oracle),
         *map(str, libs), "-lpthread", "-lm"])
    return oracle


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", required=True, type=Path, help="path to a llama.cpp checkout")
    ap.add_argument("--build-dir", type=Path, help="llama.cpp build directory with tools (default: a temp dir)")
    ap.add_argument("mode", choices=["tiny", "tiny-q8", "encode"])
    ap.add_argument("mmproj", type=Path, help="mmproj to write (tiny modes) or read (encode)")
    ap.add_argument("golden", type=Path, help="golden embeddings to write")
    args = ap.parse_args()

    oracle = build_oracle(args.llama_cpp, args.build_dir)
    for path in (args.mmproj, args.golden):
        path.parent.mkdir(parents=True, exist_ok=True)
    run([str(oracle), args.mode, str(args.mmproj), str(args.golden)])
    print(f"wrote {args.golden} ({args.golden.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

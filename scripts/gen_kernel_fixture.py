#!/usr/bin/env python3
"""Generate tests/data/kernel_golden.gguf, the golden fixture for synapse's fused kernels.

Builds ggml (ggml-base + ggml-cpu) from a local llama.cpp checkout, compiles
scripts/kernel_oracle/oracle.cpp against it, and runs it. The oracle records
ggml's own quantized weights, reference-quantized activations, and CPU vec_dot
results; see oracle.cpp for the tensor layout.

Usage:
  python3 scripts/gen_kernel_fixture.py --llama-cpp /path/to/llama.cpp \\
      [--build-dir DIR] tests/data/kernel_golden.gguf

The ggml build is reused if --build-dir already holds one. The output is
deterministic for a given llama.cpp commit and CPU (vec_dot is the
SIMD-dispatched kernel, so float summation order follows the host ISA).
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", required=True, type=Path, help="path to a llama.cpp checkout")
    ap.add_argument("--build-dir", type=Path, help="ggml build directory (default: a temp dir)")
    ap.add_argument("output", type=Path)
    args = ap.parse_args()

    llama = args.llama_cpp.resolve()
    build = (args.build_dir or Path(tempfile.mkdtemp(prefix="synapse-ggml-"))).resolve()
    libs = [build / "ggml" / "src" / "libggml-cpu.a", build / "ggml" / "src" / "libggml-base.a"]

    if not all(lib.exists() for lib in libs):
        run(["cmake", "-S", str(llama), "-B", str(build), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
             "-DBUILD_SHARED_LIBS=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
             "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_TOOLS=OFF", "-DLLAMA_CURL=OFF", "-DGGML_OPENMP=OFF"])
        run(["cmake", "--build", str(build), "--target", "ggml-base", "ggml-cpu"])

    oracle = build / "kernel_oracle"
    run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2", f"-I{llama / 'ggml' / 'include'}",
         str(HERE / "kernel_oracle" / "oracle.cpp"), "-o", str(oracle), *map(str, libs), "-lpthread", "-lm"])

    commit = subprocess.run(["git", "-C", str(llama), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "unknown"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run([str(oracle), str(args.output), f"ggml (llama.cpp @ {commit}) CPU kernels"])
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

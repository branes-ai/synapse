#!/usr/bin/env python3
"""Generate tests/data/quant_golden.gguf, the golden fixture for synapse's dequantizers.

For each format synapse decodes, the fixture holds an encoded tensor
`<fmt>.q` and the float32 tensor `<fmt>.expected` that llama.cpp's gguf-py
(an independent numpy implementation of the GGML formats) dequantizes it to.
The C++ test loads the fixture through synapse's own GGUF parser and checks
that its dequantizers reproduce `.expected`.

  - Q4_0, Q8_0, BF16: random weights quantized by gguf-py (realistic data).
  - F16: random weights cast to binary16.
  - Q4_K, Q6_K: gguf-py has no quantizer for these, and any bit pattern is a
    valid block, so blocks are random bytes with finite fp16 super-block
    scales. Random bits exercise every scale/min/high-bit combination.
  - q8_0.3d: a [2, 2, 64] tensor, checking that ne[1..3] collapse into rows.
  - i32.unsupported: a type with no scheme, for visit_matrix's negative path.

Usage:
  python3 scripts/gen_quant_fixture.py --gguf-py /path/to/llama.cpp/gguf-py \\
      tests/data/quant_golden.gguf

The output is deterministic for a given gguf-py version (fixed seed).
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROWS = 3
COLS = 512  # a multiple of 256 (QK_K) and 32


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf-py", required=True, type=Path, help="path to llama.cpp/gguf-py")
    ap.add_argument("output", type=Path)
    args = ap.parse_args()

    sys.path.insert(0, str(args.gguf_py))
    from gguf import GGMLQuantizationType as Q  # noqa: E402
    from gguf import GGUFWriter  # noqa: E402
    from gguf import quants  # noqa: E402

    rng = np.random.default_rng(20261021)

    def weights(shape):
        return rng.standard_normal(shape).astype(np.float32)

    def random_k_blocks(qtype, n_blocks, scale_offsets):
        _, type_size = quants.GGML_QUANT_SIZES[qtype]
        blocks = rng.integers(0, 256, size=(n_blocks, type_size), dtype=np.uint8)
        for off in scale_offsets:  # finite, modest fp16 scales
            scale = rng.uniform(1e-3, 2e-2, size=n_blocks).astype(np.float16)
            blocks[:, off:off + 2] = scale.view(np.uint8).reshape(n_blocks, 2)
        return blocks

    def k_tensor(qtype, scale_offsets):
        block_size, type_size = quants.GGML_QUANT_SIZES[qtype]
        n_blocks = ROWS * COLS // block_size
        return random_k_blocks(qtype, n_blocks, scale_offsets).reshape(ROWS, COLS // block_size * type_size)

    encoded = {
        "f16": (np.float16, weights((ROWS, COLS)).astype(np.float16)),
        "bf16": (Q.BF16, quants.quantize(weights((ROWS, COLS)), Q.BF16)),
        "q4_0": (Q.Q4_0, quants.quantize(weights((ROWS, COLS)), Q.Q4_0)),
        "q8_0": (Q.Q8_0, quants.quantize(weights((ROWS, COLS)), Q.Q8_0)),
        "q8_0.3d": (Q.Q8_0, quants.quantize(weights((2, 2, 64)), Q.Q8_0)),
        "q4_k": (Q.Q4_K, k_tensor(Q.Q4_K, [0, 2])),    # d, dmin at the block start
        "q6_k": (Q.Q6_K, k_tensor(Q.Q6_K, [208])),     # d at the block end
    }

    try:
        commit = subprocess.run(["git", "-C", str(args.gguf_py), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(args.output), "synapse-test")
    w.add_string("synapse.fixture.generator", "scripts/gen_quant_fixture.py")
    w.add_string("synapse.fixture.reference", f"llama.cpp gguf-py @ {commit}")

    for name, (qtype, data) in encoded.items():
        if qtype is np.float16:
            w.add_tensor(f"{name}.q", data)
            expected = data.astype(np.float32)
        else:
            w.add_tensor(f"{name}.q", data, raw_dtype=qtype)
            expected = quants.dequantize(data, qtype).astype(np.float32)
        w.add_tensor(f"{name}.expected", expected)

    # A tensor type synapse has no scheme for, for visit_matrix's negative path.
    w.add_tensor("i32.unsupported", np.arange(8, dtype=np.int32))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes, reference gguf-py @ {commit})")


if __name__ == "__main__":
    main()

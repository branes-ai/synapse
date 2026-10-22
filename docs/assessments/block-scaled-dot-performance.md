# Adopting MTL5's block-scaled integer dot: a performance investigation

*2026-10-22. Branch `feat/block-scaled-dot`. MTL5 v5.19.0 → v5.20.0.*

This is a record of how synapse's quantized kernels were moved onto MTL5's new
`simd::reduce_dot_block_scaled` (stillwater-sc/mtl5#628, PR #631). The first
version of the migration was **slower** end to end, even though the new
primitive is faster in isolation. This document records:
- every experiment run to find out why, with its commands;
- what each experiment ruled in or out;
- how hardware performance counters finally pointed at the two real causes.

In short: the primitive was never the problem. Two costs that the old
per-block loop had hidden became visible once each weight row was processed
in chunks:
- Q4_K's min term ran as a serial scalar loop;
- a cache-unfriendly loop order re-read activations from L2.

Fixing both in synapse turned a 21% prefill **loss** into a 23% **gain**.

## Contents

1. [Background: what changed and why](#1-background)
2. [Machine and methodology](#2-machine-and-methodology)
3. [The investigation, step by step](#3-the-investigation)
4. [Results](#4-results)
5. [Lessons](#5-lessons)
6. [Appendix: command reference](#6-appendix-command-reference)

---

## 1. Background

A quantized matrix row is a sequence of blocks: 32 weights for Q4_0/Q8_0, 256
for Q4_K/Q6_K. Each block has a float scale. The dot product of a row with
quantized activations is

    Σ_b  d_w[b] · d_a[b] · Σ_{j∈b} w_j · a_j

**The old kernel** computed each block's integer sum with
`mtl::simd::reduce_dot_widen` (`vpmaddwd` on AVX2). Every block therefore
ended in a **horizontal reduce**: the 8 vector lanes collapsed to a scalar
before the scale was applied. For a 32-element block that reduce is most of
the work, and synapse's Q4_0/Q8_0 gemv was measured at 0.81× of simply
dequantizing to float.

**The new primitive** keeps each block's exact integer sum in the vector lanes,
converts the lanes to float, and applies the scale with an fma:

    acc = fma(convert(block_lanes), broadcast(d_w·d_a), acc)

That leaves one horizontal reduce per call instead of one per block.

**Constraints the migration had to keep.**
- **G1:** each row of a batched `gemm` is bit-identical to `gemv`. This is
  what makes batched prefill exactly equal to token-by-token decode.
- **D1:** results are bit-identical for any thread count.

Both rule out choosing the chunk length per batch size. The float summation
order is set by the chunking, so the chunk length must be a fixed constant,
the same for gemv and gemm. Loop *tiling*, by contrast, only reorders
independent (row, token) sums, so it is free to tune.

---

## 2. Machine and methodology

| | |
|---|---|
| CPU | Intel Core i7-12700K (hybrid: 8 P-cores with 48 KiB L1d and 1.25 MiB L2 each, plus 4 E-cores; 25 MiB L3) |
| Compiler | GCC 15.2 |
| Build | `gcc-release-hwy` preset: `-O3 -march=x86-64-v3 -maes -mpclmul`, MTL5 with the Highway SIMD backend (AVX2, 8 × 32-bit lanes) |
| Model | Llama-3.2-1B-Instruct Q4_K_M (Q4_K for most matrices, Q6_K for half of `ffn_down` and `attn_v` and for the tied output embedding) |

### End-to-end measurement

`tools/llama_decode` reports prefill and decode rates. The first measurements
used a 6-token prompt, and that turned out to be **too short**: prefill took
about 50 ms and the 20-thread numbers were noise. All the conclusions below
use a **93-token prompt**, run three times each:

```bash
P="$(python3 -c "print(' '.join(['The history of the Roman Empire spans many centuries, \
from the founding of the city to the fall of Constantinople.']*4))")"
for t in 1 20; do for rep in 1 2 3; do
  ./build/gcc-release-hwy/tools/llama_decode ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
      -n 16 -t $t -p "$P" 2>&1 | grep -E "^prefill|^decode"
done; done
```

### Separating the two variables

Bumping MTL5 changes two things at once: the MTL5 version, which brings other
changes such as its thread pool, and synapse's kernels. To compare kernels
**on the same MTL5**, `main` was checked out into a worktree and built
against v5.20.0's sources:

```bash
git worktree add build/wt-main main
cmake -S build/wt-main -B build/wt-main-hwy -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++ -DSYNAPSE_WITH_HIGHWAY=ON \
  -DCMAKE_CXX_FLAGS="-march=x86-64-v3 -maes -mpclmul" -DSYNAPSE_BUILD_TESTS=OFF \
  -DFETCHCONTENT_SOURCE_DIR_MTL5=$PWD/build/gcc-release-hwy/_deps/mtl5-src \
  -DFETCHCONTENT_SOURCE_DIR_HIGHWAY=$PWD/build/gcc-release-hwy/_deps/highway-src
cmake --build build/wt-main-hwy --target llama_decode
```

Every "main" number below comes from this build: the old kernels on MTL5 v5.20.0.

### Kernel-level measurement

There were two microbenchmarks, both pinned to one P-core with `taskset -c 3`
so that hybrid scheduling stays out of the numbers:

- **`kbench.cpp`**: one source compiled twice, once against `main`'s
  `sdk/include` and once against the branch's. It times `kernels::gemm` on
  real model shapes for n = 1 (gemv), 64 and 93 tokens:

  ```bash
  FL="-std=c++20 -O3 -DNDEBUG -march=x86-64-v3 -maes -mpclmul -DMTL5_HAS_HIGHWAY -ffp-contract=off \
      -isystem $H/mtl5-src/include -isystem $H/mtl5-build/include -isystem $H/highway-src -Itests/support"
  g++ $FL -Ibuild/wt-main/sdk/include kbench.cpp -o old
  g++ $FL -Isdk/include            kbench.cpp -o new
  paste <(taskset -c 3 ./old) <(taskset -c 3 ./new | awk '{print $4,$5,$6,$7}') | column -t
  ```

- **`prim.cpp`**: MTL5's two primitives head to head: `reduce_dot_widen`
  per block against `reduce_dot_block_scaled` over 2048 int16 elements, at
  block lengths 32 and 256.

---

## 3. The investigation

### Step 0: Baseline (before any change)

`bench/gemv_bench` (4096 × 4096, AVX2, one core), with `llama_decode` on the
6-token prompt:

| format | fused gemv | dequantize + dot | fused speed-up |
|---|---|---|---|
| Q4_0 | 6.43 ms | 5.19 ms | **0.81×** |
| Q8_0 | 6.04 ms | 4.94 ms | **0.82×** |
| Q4_K | 1.33 ms | 2.18 ms | 1.64× |
| Q6_K | 8.58 ms | 11.59 ms | 1.35× |

The 0.81× for Q4_0/Q8_0 is the problem MTL5 #628 was filed for. Llama-1B
decode ran at 3.53 tok/s on 1 thread and 29.6 tok/s on 20.

### Step 1: First migration (2048-element chunks, 64-token tile)

**What changed:**
- each format's per-block `dot` became "unpack a chunk of up to 2048 weights
  into an int16 stack buffer, then one `reduce_dot_block_scaled` per chunk
  and token";
- `gemm` processed tokens in tiles of 64, unpacking each chunk once per tile.

| gemv (4096²) | before | after |
|---|---|---|
| Q4_0 | 6.43 ms | **3.90 ms (1.65× faster)** |
| Q8_0 | 6.04 ms | **3.91 ms** |
| Q4_K | 1.33 ms | 1.49 ms (12% *slower*) |
| Q6_K | 8.58 ms | 8.05 ms |

On the 6-token prompt, prefill at 20 threads went from 110 to 59 tok/s.

**Conclusion:** the format #628 was aimed at, Q4_0/Q8_0, got the expected
speed-up. Q4_K, this model's main format, got slower.

### Step 2: Better measurement, and isolating MTL5 from the kernels

Using the 93-token prompt and the `wt-main-hwy` build from section 2, with
both builds on MTL5 v5.20.0:

| | main kernels | block-scaled |
|---|---|---|
| prefill, 1 thread | 29.8 tok/s | **23.4 (−21%)** |
| prefill, 20 threads | ~129 | ~123 |
| decode, 1 thread | 3.51 | 3.65 (+4%) |

**Conclusion:** the loss is in synapse's kernels, not in MTL5 5.20's other
changes, and it shows mainly in **prefill**, the batched path.

### Step 3: Kernel microbenchmark, old against new headers

`kbench` on 2048 × 2048, cost per token:

| | old | new |
|---|---|---|
| Q8_0 gemm (n = 32) | 0.313 ms | 0.261 |
| Q6_K gemm (n = 32) | 0.179 | 0.144 |
| **Q4_K gemm (n = 32)** | **0.157** | **0.186 (+18%)** |
| Q4_K gemv (n = 1) | 0.377 | 0.371 |

**Conclusion:** only Q4_K regresses, and only in gemm. In gemm the weight
unpack is shared across tokens, so the per-token dot is what remains. The
regression is therefore in Q4_K's per-token work.

### Step 4: Hypothesis A, the primitive itself (rejected)

The guess was that for a 256-element block the primitive builds a serial
dependency chain of 16 `vpmaddwd`s into one accumulator, while
`reduce_dot_widen` unrolls across several.

**First attempt was invalid.** The block-scaled timing came out at 0.4 ns per
2048 elements. That is impossible: the compiler had proved the call
loop-invariant and hoisted it out of the timing loop. Fix: a compiler barrier
per iteration, which makes the inputs opaque.

```cpp
for (std::size_t r = 0; r < R; ++r) {
    asm volatile("" : : "r"(a.data()), "r"(x.data()), "r"(sc.data()) : "memory");
    s += f();
}
```

| ns per 2048 elements | block 32 | block 256 |
|---|---|---|
| `reduce_dot_widen` per block | 107.1 | 48.2 |
| `reduce_dot_block_scaled` | **81.2** | **43.0** |

**Conclusion:** the primitive is faster at both block lengths. Hypothesis A
was rejected, so the regression is in synapse's code around it.

### Step 5: Hypothesis B, the min term separated from the dot (partial)

Q4_K blocks also carry a "min" correction,
`dmin · Σ_j m_j · bsums_j`. In the old loop it was computed in the same block
loop as the vector dot, so out-of-order execution could overlap the two. In
the chunk form it had become a separate loop *after* the vector reduce.

Fusing it into the scale loop computes the same values in the same order, so
the bits are unchanged. Per token, Q4_K went from 0.186 to 0.180 ms.

**Conclusion:** a small effect, not the main cause.

### Step 6: Chunk-size sweep

Chunk length trades fewer horizontal reduces against a larger activation
footprint per chunk. Per token at n = 32, 2048 × 2048:

| chunk | Q4_K | Q6_K | Q8_0 |
|---|---|---|---|
| 256 | 0.176 | 0.187 | 0.233 |
| 512 | 0.130 | 0.159 | 0.239 |
| **1024** | **0.133** | **0.149** | 0.268 |
| 2048 | 0.180 | 0.144 | 0.261 |
| *old kernel* | *0.157* | *0.179* | *0.313* |

At 1024, every format beat the old kernel. End to end, however, prefill at 1
thread was **still 16% slower** (25.0 against 29.9 tok/s).

**Conclusion:** the 2048 × 2048 benchmark does not represent the model.

### Step 7: Benchmarking the model's real shapes

`kbench` was rerun on the actual matmul shapes at n = 93, the prompt length
(one 64-token tile plus one of 29):

| shape | old | new |
|---|---|---|
| Q6_K 2048 × 8192 (`ffn_down`), n = 64 | 78.2 ms | 70.5 |
| **Q6_K 2048 × 8192, n = 93** | **71.6** | **137.8 (1.9× slower)** |

**Characterization.** A batch above 64 tokens spans two tiles, and the 64-token
tile **re-unpacked every chunk per tile**. Q6_K's unpack is expensive: the
whole n = 1 gemv is 16–19 ms, almost all of it unpack. So the second tile
doubled the cost. The old code also tiled, but its per-block unpack was small
enough not to matter.

**Fix 1: accumulate in `y`, with no token tile.** Each chunk is unpacked once
per row, then dotted with every token, accumulating directly into
`y[t·rows + r]`. A float store and reload is exact, and each sum still starts
at 0.0f and adds chunk dots in chunk order, so G1 holds by construction.

Q6_K at n = 93 improved only to 120 ms. **Characterization, part 2:
locality.**
- The old block-major loop touched one 512 B activation block per token per
  weight block. For 93 tokens that is about 47 KB, which fits the 48 KiB L1.
- A 1024-element chunk is 2 KB of activations per token. For 93 tokens that
  is 186 KB, which spills to L2 on every row.

**Fix 2: rows × tokens tiling.** Chunk c of R consecutive weight rows is
unpacked first; each token's activation chunk is then dotted against all R
rows while it is hot. With R = 8, Q6_K at n = 93 dropped to 75 ms.

End to end, prefill at 1 thread was **still about 11% slower** (26.5 against
29.9), and decode at 1 thread got slightly worse (3.39). The row tile hurt
the single-token case, so it was limited to n ≥ 2.

**Conclusion:** timing alone had stopped converging. Each per-shape benchmark
now favoured the new code, yet the model did not. At this point I stopped
tuning blind and asked for hardware counters.

### Step 8: Enabling processor performance counters

`perf` was installed, but `kernel.perf_event_paranoid` was 4, which forbids
unprivileged hardware-counter access. You ran (temporarily, until reboot):

```bash
sudo sysctl kernel.perf_event_paranoid=1      # restore: sudo sysctl kernel.perf_event_paranoid=4
```

Level 1 allows a user to profile their own processes, including CPU events,
but not kernel-wide tracing.

The i7-12700K is a hybrid part, so perf exposes two PMUs, `cpu_core/…/` and
`cpu_atom/…/`. Pinning to CPU 3, a P-core, put all the counts on `cpu_core`.

### Step 9: Counters, `perf stat`

Measured on a prefill-dominated run (93-token prompt, 1 new token, 1 thread),
for both builds:

```bash
taskset -c 3 perf stat \
  -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,l2_rqsts.miss,LLC-load-misses -- \
  ./build/<build>/tools/llama_decode ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -n 1 -t 1 -p "$P"
```

| `cpu_core` counters | main kernels | block-scaled (step 7) | change |
|---|---|---|---|
| cycles | 17.55 G | 20.40 G | +16% |
| instructions | 63.30 G | 68.34 G | **+8%** |
| IPC | 3.61 | 3.35 | −7% |
| L1-dcache-load-misses | 1.88 G | 1.57 G | −17% |
| l2_rqsts.miss | 492 M | 575 M | **+17%** |
| LLC-load-misses | 0.85 M | 0.83 M | — |

**Characterization.** The new code does more work (+8% instructions) and runs
it less efficiently (lower IPC, more L2 misses). LLC misses are negligible in
both builds, so DRAM bandwidth is not the limit in prefill. This is a
compute and cache-locality problem.

### Step 10: Where the cycles go, `perf record` and `perf report`

```bash
taskset -c 3 perf record -q -o perf-new.data -e cpu_core/cycles/ -- \
  ./build/gcc-release-hwy/tools/llama_decode ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -n 1 -t 1 -p "$P"
perf report -i perf-new.data --stdio --no-children --percent-limit 2
```

| | main kernels | block-scaled |
|---|---|---|
| `gemm_rows<Q4_K>` | 64.5% | 38.2% |
| `gemm_rows<Q6_K>` | 26.6% | 14.8% |
| `mtl::simd::reduce_dot_block_scaled<short,short>` **[clone .constprop.1]** | — (none) | **39.0%** |
| attention | 3.9% | 3.3% |

**Finding 1.** The old `reduce_dot_widen` was inlined into `gemm_rows`. The
new primitive is **not inlined**: it runs as an out-of-line,
constant-propagated clone, so every (chunk, token) call pays a function
prologue and stack setup.

### Step 11: Inside the primitive, `perf annotate`

```bash
perf annotate -i perf-new.data --stdio \
  -s 'float mtl::simd::reduce_dot_block_scaled<short, short>(short const*, short const*, float const*, unsigned long, unsigned long) [clone .constprop.1]' \
  | awk '$1+0 >= 0.5'
```

These are the hottest instructions, with % of samples in the function:

```text
 2.29 :  vfmadd213ps 0x80(%rsp),%ymm1,%ymm0    # fma whose accumulator is a stack slot...
 3.52 :  vmovaps     %ymm0,0x80(%rsp)          # ...and written back to the stack
 1.87 :  vmovaps     %ymm0,0xa0(%rsp)          # same for another accumulator
 1.33 :  vpmaddwd    0x400(%rax),%ymm15,%ymm15 # 16 fully unrolled vpmaddwd per 256-block,
 1.02 :  vpmaddwd    0x4c0(%rax),%ymm10,%ymm10 #   each loading into a different register
 ...   :  vpaddd      %ymmN,%ymm0,%ymm0         # and summed into ymm0 serially
```

**Finding 2: register spill.**
- MTL5's source names four float accumulators, `f0..f3`, written out so that
  they stay in registers. With `block` constant-propagated to 256, GCC fully
  unrolled the 16-step widening loop and gave each load its own register.
- That exhausts AVX2's 16 `ymm` registers, so the accumulators are
  **spilled**. Every block pays a store, a reload (store-forwarding latency)
  and an fma on memory.
- The unrolled body is also the likely reason the function is too large to
  inline (finding 1). That is inferred from its size; it was not checked
  against GCC's inlining report.

**Experiment: prototype an MTL5 fix without touching the MTL5 clone.** The
MTL5 headers were copied into `build/mtl5-proto/`, and the inner loop was
capped at unroll 4:

```cpp
#if defined(__clang__)
#pragma clang loop unroll_count(4)
#elif defined(__GNUC__)
#pragma GCC unroll 4
#endif
for (std::size_t i = 0; i < body; i += S)
    BI::widen_dot_accumulate(ab + i, xb + i, s0, s1);
```

synapse was then built against the copy
(`-DFETCHCONTENT_SOURCE_DIR_MTL5=$PWD/build/mtl5-proto`).
- **`prim.cpp`:** block 256 improved from 43.0 to **34.3 ns** (−20%), and
  block 32 from 81.0 to 74.5 ns.
- **End to end:** decode at 1 thread rose to 3.68 tok/s (+5%), but prefill
  was **still 21% behind main** (23.6 against 30.0).

**Conclusion:** the spill is real, but it is not the prefill bottleneck.
Something in synapse's own `gemm_rows` still is.

### Step 12: Inside synapse's loop, `perf annotate gemm_rows<Q4_K>`

The prototype build was profiled the same way, annotating the Q4_K instantiation:

```text
 3.91 :  vmulss  %xmm5,%xmm1,%xmm1      # dmin_w * d_a, per block, scalar
 5.59 :  vmulss  %xmm6,%xmm0,%xmm0      # ... * float(summ)
 3.03 :  vaddss  %xmm0,%xmm1,%xmm1      # mins += ...   serial scalar chain across blocks
 4.36 :  vmulss  %xmm6,%xmm0,%xmm0
 3.55 :  vaddss  %xmm0,%xmm1,%xmm1
 0.98 :  vpinsrb $0x1,%esi,%xmm11,%xmm11  # byte-by-byte gather of the 8 mins into a vector
 2.28 :  vpsrldq $0x8,%xmm0,%xmm5         # horizontal reduce of the integer min sum,
 1.01 :  vpsrldq $0x4,%xmm0,%xmm5         #   once per block
 1.13 :  vextracti128 $0x1,%ymm5,%xmm0
```

**Finding 3 (the main cause).** Q4_K's min term
`Σ_b (dmin_b·d_y,b) · Σ_j m_b[j/2] · bsums_b[j]` costs, per block and per
token:
- a byte-by-byte gather of the mins;
- an integer multiply-add with a horizontal reduce;
- a scalar `vmulss`/`vaddss` pair on a serial dependency chain.

In prefill that runs 93 tokens × 8 blocks per weight row.
- **Old code:** the work was interleaved with the vector dot in the same
  block loop, so the out-of-order core hid much of it.
- **Chunked form:** it stands alone and is fully exposed. That accounts for
  most of the extra instructions in step 9.

**Fix 3: run the min term through the block-scaled primitive as well.** It
has exactly the block-scaled shape:
- **16-element "blocks":** mins against Q8_K's 16 sub-block sums;
- **scale per block:** `dmin_w · d_a`.

Two small layout changes make the operands contiguous:
- **Activations:** `prepare_activations` also writes each Q8_K block's 16
  `bsums` into scratch, contiguously. That adds 32 B per 256 activations.
- **Weights:** the Q4_K chunk unpack expands the 8 mins into 16 int16, one
  per sub-block.

The whole min term becomes one `reduce_dot_block_scaled` per chunk and
token, with no per-block horizontal reduce, no scalar chain and no gathers.
G1 and D1 still hold, because gemv and gemm share the code.

`kbench` after fix 3 (old → new):

| Q4_K | n = 1 | n = 64 | n = 93 |
|---|---|---|---|
| 2048 × 2048 | 0.370 → **0.295** | 0.299 → **0.220** | 0.289 → **0.223** ms/token |
| 8192 × 2048 | 1.48 → **1.19** | 1.20 → **0.78** | 1.09 → **0.98** |
| 2048 × 8192 | 1.45 → **1.17** | 1.42 → **1.21** | 1.52 → **1.40** |

### Step 13: Final tuning, and dropping the MTL5 change

With fix 3 in place, the row tile and the MTL5 unroll cap were re-evaluated end to end:

| build | prefill, 1 thread | decode, 1 thread |
|---|---|---|
| main kernels | 30.1 | 3.51 |
| fix 3 + prototype MTL5, R = 1 | 31.9 | 3.75 |
| fix 3 + prototype MTL5, R = 4 | 34.8 | 3.77 |
| **fix 3 + released MTL5 v5.20.0, R = 4** | **37.1** | **3.70** |

The unroll cap adds a little decode speed but *costs* prefill, about 6% at
R = 4. So it is a mixed result, not a clear MTL5 fix, and it was dropped.
**All the decisive fixes were in synapse.** R = 4 beat R = 8 by about 5% and
R = 1 by about 10% on prefill; the single-token case stays untiled.

---

## 4. Results

The final configuration:
- MTL5 v5.20.0 as released;
- 1024-element chunks;
- accumulation in `y`;
- rows × tokens tiling with R = 4 for n ≥ 2;
- the min term through the block-scaled primitive.

**End to end, Llama-3.2-1B Q4_K_M, 93-token prompt, AVX2:**

| | main (old kernels) | this branch | change |
|---|---|---|---|
| prefill, 1 thread | 29.9 tok/s | **37.1** | **+24%** |
| decode, 1 thread | 3.50 | **3.73** | **+7%** |
| prefill, 20 threads | 127–148 | 139–153 | about level (noisy) |
| decode, 20 threads | 29.5–30.3 | 29.2–30.3 | flat: memory-bound |

**`gemv_bench`, 4096 × 4096, one core:**

| format | before | after | against dequantize + dot |
|---|---|---|---|
| Q4_0 | 6.43 ms | **3.88** | 0.81× → **1.35×** |
| Q8_0 | 6.04 | **3.92** | 0.82× → **1.31×** |
| Q4_K | 1.33 | **1.12** | 1.64× → **1.96×** |
| Q6_K | 8.58 | **8.22** | 1.35× → **1.40×** |

**Correctness (unchanged throughout).** All 103 tests pass on every preset
and under ASan/UBSan, and the threaded tests pass under TSan. That includes:
- G1: prefill bit-identical to decode, on the tiny models and on Llama-3.2-1B;
- D1: results bit-identical for any thread count;
- the ggml golden at 1e-5 relative;
- the llama.cpp logit goldens.

The one contract that changed, by design and approved before the work, is
reproducibility *across SIMD targets*. Results are bit-identical run to run
on one build, and agree to float re-association between scalar, SSE, AVX2
and NEON builds.

---

## 5. Lessons

1. **A faster primitive does not guarantee a faster kernel.** Restructuring a
   loop around a new primitive moves costs around. Here the old per-block
   loop had been *hiding* Q4_K's min-term work behind the vector dot.
2. **Measure end to end, with inputs long enough to be stable.** A 6-token
   prompt made a 21% single-thread prefill regression look like a 46% one at
   20 threads, for the wrong reasons.
3. **Control the variables.** Building `main` against the new MTL5 separated
   "MTL5 changed" from "our kernel changed". Without that, the thread-pool
   and other v5.20.0 changes could have been blamed.
4. **Microbenchmarks need compiler barriers.** A pure function with constant
   inputs gets hoisted out of the timing loop, giving a reading of 0.4 ns.
5. **Benchmark the shapes you run.** 2048 × 2048 at n = 32 said "faster". The
   model's 2048 × 8192 Q6_K at n = 93 was 1.9× slower, because of a tile
   boundary at 64 tokens.
6. **Profile before tuning further.** Six timing experiments converged
   slowly. Two hardware-counter passes (`perf stat` for the shape of the
   problem, `perf record` and `perf annotate` for its location) found both
   remaining causes directly.
7. **A bit-exactness contract shapes the design.** G1 fixes the chunk
   length, the float order, while leaving loop tiling free. Knowing which
   knobs are free is what kept every experiment correctness-preserving.

---

## 6. Appendix: command reference

```bash
# --- enable / restore hardware performance counters (needs root) ---------------
cat /proc/sys/kernel/perf_event_paranoid          # was 4: no unprivileged HW counters
sudo sysctl kernel.perf_event_paranoid=1          # allow profiling own processes
sudo sysctl kernel.perf_event_paranoid=4          # restore (also reset by a reboot)

# --- counters for one run, pinned to a P-core ---------------------------------
taskset -c 3 perf stat \
  -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,l2_rqsts.miss,LLC-load-misses -- \
  ./build/gcc-release-hwy/tools/llama_decode MODEL.gguf -n 1 -t 1 -p "$P"

# --- sampling profile: which functions ------------------------------------------
taskset -c 3 perf record -q -o perf.data -e cpu_core/cycles/ -- \
  ./build/gcc-release-hwy/tools/llama_decode MODEL.gguf -n 1 -t 1 -p "$P"
perf report -i perf.data --stdio --no-children --percent-limit 2

# --- which instructions inside a function (exact symbol from perf report) -------
SYM=$(perf report -i perf.data --stdio --no-children | grep -m1 "gemm_rows<.*tensor_type)12>" | sed 's/.*\[\.\] //')
perf annotate -i perf.data --stdio -s "$SYM" | awk '$1+0 >= 0.8'

# --- the A/B harness: old kernels on the new MTL5 ----------------------------
git worktree add build/wt-main main
cmake -S build/wt-main -B build/wt-main-hwy ... -DFETCHCONTENT_SOURCE_DIR_MTL5=<new mtl5 src>
git worktree remove --force build/wt-main        # clean up afterwards

# --- prototyping an MTL5 change without touching the MTL5 repository ---------
cp -r build/gcc-release-hwy/_deps/mtl5-src/include build/mtl5-proto/   # edit the copy
cmake -S . -B build/proto-hwy ... -DFETCHCONTENT_SOURCE_DIR_MTL5=$PWD/build/mtl5-proto
```

`tensor_type)12` is Q4_K and `14` is Q6_K, as GGML type ids in the mangled
template names.

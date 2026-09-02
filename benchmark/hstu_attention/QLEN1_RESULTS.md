# HSTU single-query optimization

## Target workload

- BF16, causal HSTU attention
- `seqlen_q = 1` for every sequence
- Variable `seqlen_kv` from 1024 to 3072, averaging about 2048
- 4 heads, head dimension 128
- Batch sizes 64, 512, and 1024

`benchmark_hstu_qlen1.py` uses 10 warmup iterations and reports the median of
seven groups of 30 executions. The baseline is commit `8baf903`, which adds
only the benchmark on top of the original HSTU kernel. Baseline and candidate
were run sequentially in the same allocation.

## Initial forward and direct-path results

These controlled baseline/candidate measurements predate the small-MMA
backward path. The final backward dispatch is reported below.

### B300 (SM10.3)

| Batch | Forward baseline (ms) | Forward optimized (ms) | Speedup | Backward baseline (ms) | Backward optimized (ms) | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.0798 | 0.0569 | 1.40x | 0.2338 | 0.2108 | 1.11x |
| 512 | 0.4235 | 0.3220 | 1.32x | 1.7760 | 1.0318 | 1.72x |
| 1024 | 1.0102 | 0.6482 | 1.56x | 3.5453 | 1.8740 | 1.89x |

### Rubin (SM10.7)

| Batch | Forward baseline (ms) | Forward optimized (ms) | Speedup | Backward baseline (ms) | Backward optimized (ms) | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.0442 | 0.0441 | 1.00x | 0.1445 | 0.1293 | 1.12x |
| 512 | 0.2285 | 0.2249 | 1.02x | 1.0261 | 0.8069 | 1.27x |
| 1024 | 0.4557 | 0.4322 | 1.05x | 2.0375 | 1.4264 | 1.43x |

## Split-KV forward

The qlen=1 HSTU output is additive across KV partitions: each partition can
compute `silu(alpha * QK^T) @ V / scaling_seqlen` independently, without the
softmax renormalization needed by standard attention. The split path exposes
2 or 4 virtual heads per real batch/head tile, gives each CTA a contiguous KV
block range, and atomically combines packed BF16 output pairs. Output zeroing
is included in every timing below.

The CUDA-core forward experiment was correct but lost decisively to the
tensor-core kernel. Its B300 times were 0.2728, 0.5855, and 1.0120 ms for
BS=64/512/1024; Rubin took 0.2525, 0.5355, and 0.9471 ms. The losing kernel is
not retained.

The final tensor-core comparison uses 10 warmups and the median of seven
groups of 30 executions:

| GPU | Batch | Unsplit TC (ms) | Selected split (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| B300 | 64 | 0.0556 | 0.0488 (`split4`) | 1.14x |
| B300 | 512 | 0.3197 | 0.3154 (`split2`) | 1.01x |
| B300 | 1024 | 0.6925 | 0.6925 (unsplit) | 1.00x |
| Rubin | 64 | 0.0440 | 0.0387 (`split4`) | 1.14x |
| Rubin | 512 | 0.2238 | 0.2170 (`split2`) | 1.03x |
| Rubin | 1024 | 0.4317 | 0.4224 (`split2`) | 1.02x |

Extra batch-size sweeps set the dispatch boundaries. B300 uses `split4`
through BS=256, `split2` through BS=512, then unsplit TC. Rubin uses `split4`
through BS=128, `split2` through BS=1024, then unsplit TC. Split-KV is enabled
only for the supported causal BF16 D=128 layout when average KV length is at
least 1536; shorter or unsupported cases preserve the existing path.

## Forward tile-shape experiment

The smallest useful one-CTA MMA M dimension on these devices is 64, so a
separate experiment replaced the M128/N128 QK and PV tiles with M64/N128. It
also tested an N64 KV tile and a qlen=1 epilogue in which only the warp owning
output row zero participates. M64 needs the native 32-data-path TMEM mapping;
reusing the M128 software fragment interpretation is numerically incorrect.

All M64 variants pass the boundary oracle on B300 and Rubin. Unsplit M64 is
bitwise identical to M128 for KV lengths 1, 2, 63, 64, 65, 127, 128, 129, 255,
256, 257, 2048, 2049, and 3072. The stable interleaved Rubin comparison was:

| Batch | Selected M128 (ms) | Best M64/N128 (ms) | M64 delta | Best M64/N64 (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 0.0385 (`split4`) | 0.0411 (`split2`) | +6.7% | 0.0434 (`split4`) |
| 512 | 0.2169 (`split2`) | 0.2221 (`split2`) | +2.4% | 0.2548 (`split4`) |
| 1024 | 0.4222 (`split2`) | 0.4313 (`split2`) | +2.2% | 0.4978 (`split4`) |

On B300, M128+split4 remains about 4% faster than the best M64 schedule at
BS=64, and BS=512 slightly favors M128. At BS=1024, forward/reverse
interleaved runs changed which M tile won as shared-node frequency moved, so
there is no reproducible M64 speedup. N64 is consistently slower because it
doubles the roughly 2K-long KV loop and its TMA/barrier overhead even though
every KV token is useful. Restricting the epilogue to one warp is correct but
does not materially change latency.

The production dispatch therefore remains M128/N128. The reproducible M64,
N64, and epilogue variants live on the
`experiment/hstu-qlen1-fwd-m64` branch and are not selected automatically.

## Small-MMA Q-major backward

The benchmark can force four backward implementations with
`--backward-impl legacy|tc|tc-small|direct`. `legacy` is the original
tensor-core schedule, `tc` is the first Q-major experiment, `tc-small` is the
new small-MMA path, and `direct` is the specialized CUDA-core path. The tables
below use 10 warmups and the median of five groups of 30 executions.

### B300 (SM10.3)

| Batch | Q-major TC (ms) | Small-MMA TC (ms) | Speedup |
| ---: | ---: | ---: | ---: |
| 64 | 0.2575 | 0.1846 | 1.39x |
| 512 | 1.4395 | 1.0736 | 1.34x |
| 1024 | 2.8185 | 2.1168 | 1.33x |

### Rubin (SM10.7)

| Batch | Q-major TC (ms) | Small-MMA TC (ms) | Speedup |
| ---: | ---: | ---: | ---: |
| 64 | 0.1835 | 0.1352 | 1.36x |
| 512 | 0.9274 | 0.7761 | 1.19x |
| 1024 | 1.7963 | 1.5186 | 1.18x |

The new path keeps the FA2 `bwd_loop_opt` loop order: one CTA owns one
batch/head pair, keeps its single Q fixed, and walks the KV tiles. The one
valid dQ row remains local across all KV tiles and is written once, while the
CTA writes its disjoint dK/dV rows directly. There are no dQ atomics, global
FP32 dQ workspace, workspace-zeroing launch, or conversion launch.

The computation is still tensor-core based. S, dP, dK, and dV use an
M128N16K128 tile instead of reserving 128 Q rows. dQ uses the transposed
`K^T @ dS^T` form with M128N8K128, which is the smallest useful N shape for
this SM100-family UMMA path. K is loaded once and reinterpreted as the
transposed dQ operand in shared memory. Four compute warps replace eight for
the reduced Q tile.

## Backward automatic dispatch

The crossover was measured at extra batch sizes rather than inferred from the
three target points:

- B300: `tc-small` below BS=64, `direct` through BS=79, `tc-small` through
  BS=127, and vectorized `direct` from BS=128. The two small-grid regions are
  retained because CTA-wave boundaries make the crossover non-monotonic.
- Rubin: `legacy` below BS=128, `tc-small` through BS=191, and vectorized
  `direct` from BS=192.
- Other SM100-family devices retain the previous policy.

For the requested batches, the resulting automatic choices and final timings
are:

| GPU | BS=64 | BS=512 | BS=1024 |
| --- | ---: | ---: | ---: |
| B300 | 0.1697 ms (`direct`) | 0.9163 ms (`direct`) | 1.7364 ms (`direct`) |
| Rubin | 0.1272 ms (`legacy`) | 0.6266 ms (`direct`) | 1.1806 ms (`direct`) |

Relative to the pre-vectorization dispatch, this final pass lowers B300 by
7.9%, 2.5%, and 2.5% at BS=64/512/1024. Rubin BS=64 keeps the faster legacy
path, while BS=512/1024 improve by 13.3% and 12.3%.

## NCU pipeline and work analysis

`profile_hstu_qlen1.py` compiles and warms up outside the CUDA profiler range,
then exposes exactly one selected execution to Nsight Compute. The B300 runs
used the command-line `LaunchStats`, occupancy, compute, memory, scheduler,
warp-state, and instruction metrics.

### Useful work and compulsory traffic

Ignoring the small activation cost, forward performs one QK dot product and
one weighted-V accumulation per KV row, about `4 * D = 512` FLOPs per
KV/head. Backward recomputes QK and adds dP, dQ, dK, and dV work, about
`8 * D = 1024` FLOPs per KV/head. Its compulsory traffic is one K/V read and
one dK/dV write. The requested workload is therefore approximately one useful
FLOP per compulsory byte in both directions:

| Batch | Total KV | Forward useful FLOPs / K+V bytes | Backward useful FLOPs / read+write bytes |
| ---: | ---: | ---: | ---: |
| 64 | 130,304 | 0.267 GFLOP / 0.267 GB | 0.534 GFLOP / 0.534 GB |
| 512 | 1,047,808 | 2.146 GFLOP / 2.146 GB | 4.292 GFLOP / 4.292 GB |
| 1024 | 2,098,944 | 4.299 GFLOP / 4.299 GB | 8.597 GFLOP / 8.597 GB |

The forward M128 tensor-core tile has only one valid Q row, so its hardware
MMA work is about 128 times the useful Q-row work, plus KV-tail padding. That
does not make the target workload compute-bound: at BS=1024 NCU reports the
expected 4.30 GB of K/V reads, 88.3% of peak DRAM throughput, and 40.7% tensor
pipe activity. The redundant tensor-core work is cheaper than replacing the
three-stage TMA/MMA pipeline with scalar work.

At BS=64, split4 changes the forward launch from 256 to 1024 CTAs while keeping
the same 267 MB K/V read volume. In NCU it raises DRAM throughput from 59.6%
to 69.4%, tensor-core activity from 26.6% to 33.4%, and launch waves from 1.73
to 6.92; the main kernel falls from 59.3 to 50.9 us. Its included output-zero
launch costs about 4.2 us. This is why split-KV helps the small batch but does
little once the unsplit kernel already saturates memory.

### Vectorized direct backward and CTA tuning

The original direct kernel launched 256 threads (8 warps) per batch/head CTA.
Each warp serially loaded one KV row, reduced QK and dO-V, wrote dK/dV, and
then advanced by eight rows. NCU identified long-scoreboard stalls as the
dominant bubble, with no excess DRAM traffic and only 34 registers per thread.

The first optimization widened SM103 and SM107 to 512 threads (16 warps).
The math, Q-major loop order, and one final dQ write stayed unchanged, while
twice as many independent KV rows hid more memory latency. The dQ shared
reduction grew from about 5.1 to 9.2 KB per CTA, which is not the occupancy
limiter.

| B300 NCU metric | BS64, 8 warps | BS64, 16 warps | BS1024, 8 warps | BS1024, 16 warps |
| --- | ---: | ---: | ---: | ---: |
| Kernel duration | 383.8 us | 210.9 us | 1.86 ms | 1.78 ms |
| DRAM throughput | 16.9% | 30.6% | 59.9% | 62.8% |
| Active warps | 18.6% | 37.5% | 69.6% | 73.0% |
| Eligible warps / scheduler cycle | 0.21 | 0.50 | 0.97 | 1.04 |
| Issue-active | 18.3% | 33.9% | 47.3% | 49.2% |

Instruction profiling then exposed another bottleneck. The scalar kernel
issued eight coalesced global loads and eight stores per KV/head row. Assigning
four adjacent BF16 values to each lane allows one 64-bit K load, V load, dK
store, and dV store. At BS=1024 this reduces executed global-load instructions
from 67.95 million to 17.38 million and stores from 67.18 million to 16.81
million. The total executed instruction count falls from 999.34 million to
863.02 million without changing DRAM bytes or arithmetic.

The vector fragments increase register pressure. A second CTA sweep therefore
keeps 16 warps for small B300 grids, but uses 12 warps from BS=448 so five CTAs
can reside per SM. Rubin consistently prefers 16 warps. Final B300 BS=1024 NCU
reports 44 registers/thread, 7.2 KB shared memory, 64.4% peak DRAM throughput,
54.8% active warps, 0.74 eligible warps per scheduler cycle, and 43.5%
issue-active. Although issue occupancy is lower than the scalar 16-warp
kernel, the 13.6% smaller instruction stream and higher DRAM utilization lower
kernel time from about 1.78 to 1.73 ms.

At BS=64 the final 16-warp vector kernel keeps active warps essentially flat
at 37.7%, but raises eligible warps from 0.50 to 0.57, issue-active from 33.9%
to 35.6%, and DRAM throughput from 30.6% to 37.7%. NCU kernel time falls from
210.9 to 170.6 us.

The complete direct-path progression is:

| GPU | Batch | 8-warp scalar (ms) | Wider scalar (ms) | Final vector (ms) | Final vs scalar |
| --- | ---: | ---: | ---: | ---: | ---: |
| B300 | 64 | 0.3851 | 0.2121 | 0.1697 (16 warps) | -55.9% |
| B300 | 512 | 1.0303 | 0.9396 | 0.9163 (12 warps) | -11.1% |
| B300 | 1024 | 1.8698 | 1.7810 | 1.7364 (12 warps) | -7.1% |
| Rubin | 64 | 0.3346 | 0.1847 | 0.1512 (16 warps) | -54.8% |
| Rubin | 512 | 0.8042 | 0.7227 | 0.6266 (16 warps) | -22.1% |
| Rubin | 1024 | 1.4247 | 1.3462 | 1.1806 (16 warps) | -17.1% |

Two rejected variants confirm the resource tradeoff. Holding two KV rows per
warp improved BS=64 by about 4%, but its extra registers slowed BS=512/1024 by
about 5%. Before vectorization, a 384-thread CTA was slower than 512 threads;
after vectorization changed the register footprint, retuning made 384 threads
best for large B300 grids. A 1024-thread CTA reduces CTA residency and remains
much slower at large batch sizes. Vector loads with scalar stores were also
rejected: the scalar stores raise final B300 BS=1024 from 1.74 to 2.09 ms.

The early Rubin node runs normal kernels and CUDA-event benchmarks, but both
Nsight Compute 2025.3.1 and 2026.2.1 fail to initialize its hardware counter
library (`Failed to initialize LOP`, `LibraryNotLoaded`) with driver 615.12.
Rubin therefore uses the same kernel-structure diagnosis from B300 plus direct
before/after timing and correctness measurements on SM107.

### Rejected forward residency experiment

The qlen=1 Q and output shared-memory lifetimes can be made disjoint. Reusing
that storage and reducing the KV pipeline from three stages to two lowers the
CTA allocation from about 166 KB to 96 KB and permits two resident CTAs per
SM. It remains numerically correct, but the shallower pipeline slows B300 by
about 10% at BS=64/512 and 15% at BS=1024. A driver-level asynchronous memset
also has no measurable advantage over the existing output zeroing. Neither
experiment is retained.

## Design

The forward kernel uses one Q stage for a one-token query, removing the
second fully masked 128-row Q tile. The new split-KV schedule increases CTA
parallelism only at measured crossover points and fuses the partial-output
reduction into its epilogue.

Backward chooses among the original tensor-core kernel, the new small-MMA
Q-major kernel, and the vectorized CUDA-core `hstu_bwd_q1.py` using the
measured architecture-specific crossovers above. Unsupported layouts and
non-causal or arbitrary masks keep the existing implementation.

This is the same high-level loop-order lesson as the FA2 `bwd_loop_opt`
branch, specialized further for the exact one-query case.

## Correctness

The benchmark checks KV lengths `1, 127, 128, 2049, 3072` against a float32
PyTorch oracle. The oracle runs on CPU so it also works on early Rubin systems
whose PyTorch device-code toolchain does not yet recognize SM10.7. A separate
forced-path test checks `tc-small` against `tc` at KV lengths
`1, 127, 128, 129, 2049`; dQ, dK, and dV are bitwise identical on both B300
and Rubin. Forced `split2` and `split4` forward runs pass the boundary-heavy
forward oracle on both architectures; the largest observed forward absolute
error is below `1.6e-5`. The final automatic dispatch passes the forward and
all three gradient checks on both architectures, with maximum absolute
gradient error around `1e-5`.

# HSTU single-query optimization

## Target workload

- BF16, causal HSTU attention
- `seqlen_q = 1` for every sequence
- Variable `seqlen_kv` from 1024 to 3072, averaging about 2048
- 4 heads, head dimension 128
- Batch sizes 64, 512, and 1024

`benchmark_hstu_qlen1.py` uses 10 warmup iterations and reports the median of
seven groups of 30 executions. The baseline is commit `8baf903`, which adds
only the benchmark on top of the original HSTU kernel.

## Broad-sweep final policy

The final selection uses `sweep_hstu_qlen1.py` rather than the three target
points alone. Its 36 unique cases cover batch sizes 16 through 2048, 1/2/4/8
heads, average KV lengths 128 through 4096, and opposite low-grid/long-KV and
high-grid/short-KV corners. Every case uses variable per-sequence KV lengths.
Each case has equal weight in the geometric-mean speedup.

Baseline and candidate were measured in both process orders on the same B300
and Rubin allocations. Candidate alternatives were also rerun with their
measurement order reversed. The selection rule was: no case may regress
against the original kernel, then maximize the geometric-mean speedup. This
produces one fixed schedule per architecture and direction; batch size, head
count, average KV length, and packed total KV no longer select a schedule.

| GPU | Direction | Fixed qlen=1 schedule | Geomean speedup | Slowest case | Regressions |
| --- | --- | --- | ---: | ---: | ---: |
| B300 (SM10.3) | Forward | M128/N128 tensor core, unsplit | 1.461x | 1.196x | 0 / 36 |
| Rubin (SM10.7) | Forward | M128/N128 tensor core, split2 | 1.182x | 1.050x | 0 / 36 |
| B300 (SM10.3) | Backward | CUDA core, Q-major, split22 | 2.061x | 1.602x | 0 / 36 |
| Rubin (SM10.7) | Backward | CUDA core, Q-major, split26 | 1.729x | 1.398x | 0 / 36 |

Across both architectures, the equal-weight geometric means are 1.314x for
forward and 1.888x for backward. At the requested H=4, D=128, average-KV=2048
points, the order-balanced measurements are:

| GPU | Batch | Forward original -> fixed (ms) | Speedup | Backward original -> fixed (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| B300 | 64 | 0.0789 -> 0.0553 | 1.43x | 0.2300 -> 0.1224 | 1.88x |
| B300 | 512 | 0.4213 -> 0.3203 | 1.32x | 1.7653 -> 0.8457 | 2.09x |
| B300 | 1024 | 1.0293 -> 0.6495 | 1.58x | 3.5315 -> 1.6859 | 2.09x |
| Rubin | 64 | 0.0443 -> 0.0399 | 1.11x | 0.1435 -> 0.0898 | 1.60x |
| Rubin | 512 | 0.2331 -> 0.2170 | 1.07x | 1.0301 -> 0.6203 | 1.66x |
| Rubin | 1024 | 0.4754 -> 0.4224 | 1.13x | 2.0481 -> 1.2266 | 1.67x |

The final automatic path matches the forced selected kernel within 0.1% in
geometric mean. The specialized policy applies only to causal BF16 qlen=1,
D=128, matching Q/K/V heads, and supported direct layouts. Other dtypes,
dimensions, masks, paged KV, and unsupported layouts stay on the existing
general dispatch paths.

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

## Historical per-shape split-KV forward experiment

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

These early per-shape boundaries were useful for proving the split-KV idea,
but are superseded by the fixed broad-sweep policy above. Production now uses
unsplit TC on B300 and split2 TC on Rubin for every supported target shape.

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

### M64 critical-path counter check

A follow-up B300 NCU run forced unsplit M128/N128 and M64/N128 on identical
inputs. This isolates the M dimension while keeping the KV tile, launch grid,
and input bytes fixed. NCU metric-replay durations are higher than CUDA-event
timings, so only paired relative values are used here.

| Batch | M tile | Duration (us) | DRAM read | DRAM peak | BF16 tensor ops | UTCMMA instructions | Tensor-pipe active |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 128 | 77.82 | 267.04 MB | 45.59% | 34.16 G | 65.2 K | 35.44% |
| 64 | 64 | 81.41 | 267.05 MB | 43.38% | 17.08 G | 65.2 K | 33.14% |
| 512 | 128 | 364.90 | 2.15 GB | 76.85% | 274.68 G | 523.9 K | 58.46% |
| 512 | 64 | 374.24 | 2.15 GB | 74.91% | 137.34 G | 523.9 K | 56.30% |
| 1024 | 128 | 696.67 | 4.30 GB | 80.55% | 550.23 G | 1049.5 K | 60.30% |
| 1024 | 64 | 712.51 | 4.30 GB | 78.76% | 275.11 G | 1049.5 K | 58.75% |

M64 exactly halves the tensor math operations, so the smaller instruction is
doing what was intended. It does not, however, reduce the number of issued
UTCMMA instructions. Tensor-pipe activity falls by only 1.6--2.3 percentage
points; at BS=1024 its activity multiplied by duration is essentially
unchanged (about 420 us versus 419 us). K/V traffic is unchanged and DRAM is
the more heavily utilized pipeline. Kernel duration increases by 4.6%, 2.6%, and 2.3% at
BS=64/512/1024. At BS=1024 the M64 fragment/TMEM path also raises total
executed instructions from 118.66 M to 178.72 M.

The precise conclusion is therefore stronger than a FLOP count but narrower
than saying that all MMA latency is free: K/V movement is on the critical
path, and halving nominal MMA work does not halve MMA issue cost or tensor
active cycles. The saved operations cannot shorten the memory-dominated tile
pipeline, while the M64 software mapping adds instruction overhead.

The production dispatch therefore remains M128/N128. The reproducible M64,
N64, and epilogue variants live on the
`experiment/hstu-qlen1-fwd-m64` branch and are not selected automatically.

## Small-MMA Q-major backward

The benchmark can force the four base backward implementations with
`--backward-impl legacy|tc|tc-small|direct`; later sections also use the
`direct-split*` variants. `legacy` is the original tensor-core schedule, `tc`
is the first Q-major experiment, `tc-small` is the new small-MMA path, and
`direct` is the specialized CUDA-core path. The tables below use 10 warmups
and the median of five groups of 30 executions.

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

## Historical pre-split backward automatic dispatch

Before adding backward split-KV, the crossover was measured at extra batch
sizes rather than inferred from the three target points:

- B300: `tc-small` below BS=64, `direct` through BS=79, `tc-small` through
  BS=127, and vectorized `direct` from BS=128. The two small-grid regions are
  retained because CTA-wave boundaries make the crossover non-monotonic.
- Rubin: `legacy` below BS=128, `tc-small` through BS=191, and vectorized
  `direct` from BS=192.
- Other SM100-family devices retain the previous policy.

For the requested batches, those pre-split automatic choices and timings were:

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

### Why the CUDA-core forward loses

A second B300 experiment kept the direct kernel's scalar K/V row loads and
loop structure but removed QK, SiLU, and the weighted-V accumulation. The
loaded values were accumulated into a live result so the compiler could not
delete the traffic. With 10 warmups and five groups of 30 executions, full
direct, load-only, and tensor-core timings were:

| Batch | Direct full (ms) | Direct load-only (ms) | Tensor core (ms) |
| ---: | ---: | ---: | ---: |
| 64 | 0.2734 | 0.2174 | 0.0566 |
| 512 | 0.5864 | 0.5671 | 0.3201 |
| 1024 | 1.0123 | 0.9875 | 0.6616 |

At BS=1024, deleting nearly all arithmetic saves only 0.025 ms, while the
load-only path remains 1.49x slower than the tensor-core path. NCU confirms
that this is a copy/pipeline problem rather than useful CUDA-core math. All
three kernels read about 4.30 GB from DRAM, but load-only reaches 57.2% of
peak DRAM bandwidth versus 87.8% for the TMA/tensor-core kernel. It executes
67.30 million scalar global-load instructions and 348.80 million total
instructions; the TMA path reports 0.246 million global-load instructions
and 118.73 million total instructions. Long-scoreboard stall ratio is 26.2
for load-only versus 13.0 for TMA. The diagnostic direct kernels were removed
after profiling; production keeps the TMA/tensor-core implementation.

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

### Historical per-shape split-KV direct backward

The direct qlen=1 backward is also additive over KV partitions for dQ. Each
split CTA owns a contiguous KV range, writes its disjoint dK/dV rows normally,
and reduces its local dQ contribution in shared memory. Only the final 128
dQ values need cross-CTA combination; 64 threads issue packed BF16x2 atomic
adds after a small dQ zero-fill. There are no dK/dV atomics and no partial
gradient workspace.

The split count and CTA size were swept in the same allocation. Splitting by
2 or 4 is too little for the small grid, while excessive splitting repeats
Q/dO loads, metadata, CTA setup, local dQ reduction, and atomics. The measured
choices target roughly 32K short CTAs, cap the split at 64, and retain at least
about 32 average KV rows per CTA. B300 accepts split8 as its last useful
crossover; Rubin requires at least split16 because its unsplit direct kernel
is already faster at large grids.

The following early automatic choices and same-allocation comparisons used 10
warmups and the median of five groups of 30 executions. They are superseded by
the fixed split22/split26 policy from the 36-case sweep:

| GPU | Batch | Previous dispatch (ms) | Split selection | Final (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| B300 | 64 | 0.1727 (`direct`) | 64 | 0.1178 | 1.47x |
| B300 | 512 | 0.9204 (`direct`) | 16 | 0.8444 | 1.09x |
| B300 | 1024 | 1.7438 (`direct`) | 8 | 1.6658 | 1.05x |
| Rubin | 64 | 0.1310 (`legacy`) | 64 | 0.0904 | 1.45x |
| Rubin | 512 | 0.6653 (`direct`) | 16 | 0.6457 | 1.03x |
| Rubin | 1024 | 1.2435 (`direct`) | 1 | 1.2435 | 1.00x |

The B300 BS=64 NCU comparison makes the tradeoff explicit. The main kernel
falls from 177.7 to 114.2 us; the included dQ zero-fill is 10.3 us. Total
K/V and gradient DRAM traffic stays near 492 MB, while active warps rise from
37.6% to 58.7%, eligible warps per scheduler cycle from 0.50 to 0.85, issue
activity from 34.1% to 47.1%, and DRAM throughput from 36.1% to 56.1%.
Executed instructions increase from 51.63 to 60.58 million, so the speedup
comes from latency hiding and higher memory utilization despite extra work.

Forced split8, split16, and split64 pass the boundary-length CPU oracle on
B300 and Rubin. dK/dV maximum absolute errors remain below 1e-6. Packed BF16
dQ atomics introduce order-dependent rounding; observed dQ maximum absolute
error is 2e-5 to 6e-5. Deterministic HSTU backward was already unsupported.

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
second fully masked 128-row Q tile. B300 always uses the unsplit M128/N128
tensor-core schedule for the supported target, while Rubin always uses two KV
partitions and atomically combines their partial outputs in the epilogue.

Backward always uses the vectorized Q-major CUDA-core `hstu_bwd_q1.py` for the
supported target. B300 splits the KV range 22 ways and Rubin splits it 26
ways. Each CTA writes disjoint dK/dV rows directly; only dQ is atomically
combined. Unsupported layouts, non-causal or arbitrary masks, FP16, and other
head dimensions keep the existing implementation.

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
error is below `1.6e-5`. Forced backward split8, split16, split22, split26,
and split64 runs pass on the tested architectures. The final automatic
split22/split26 boundary oracle has maximum absolute errors below `3.8e-5`
for dQ and `1e-6` for dK/dV. BF16 dQ atomic ordering accounts for the larger
dQ error.

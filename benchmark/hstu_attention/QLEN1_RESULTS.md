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

- B300: `tc-small` through BS=384; `direct` from the measured BS=448 point.
- Rubin: `legacy` below BS=128, `tc-small` through BS=832, and `direct` from
  the measured BS=896 point.
- Other SM100-family devices retain the previous policy.

For the requested batches, the resulting automatic choices and final timings
are:

| GPU | BS=64 | BS=512 | BS=1024 |
| --- | ---: | ---: | ---: |
| B300 | 0.1860 ms (`tc-small`) | 1.0320 ms (`direct`) | 1.8724 ms (`direct`) |
| Rubin | 0.1313 ms (`legacy`) | 0.7768 ms (`tc-small`) | 1.4907 ms (`direct`) |

Against the implementation selected by the previous policy, the meaningful
target-case changes are 12.5% lower latency at B300 BS=64 and 8.7% lower
latency at Rubin BS=512. The other requested points keep their previous
implementation.

## Design

The forward kernel uses one Q stage for a one-token query, removing the
second fully masked 128-row Q tile. The new split-KV schedule increases CTA
parallelism only at measured crossover points and fuses the partial-output
reduction into its epilogue.

Backward chooses among the original tensor-core kernel, the new small-MMA
Q-major kernel, and `hstu_bwd_q1.py` using the measured architecture-specific
crossovers above. Unsupported layouts and non-causal or arbitrary masks keep
the existing implementation.

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

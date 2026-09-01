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

## Automatic dispatch

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
second fully masked 128-row Q tile. Small Rubin launches retain the existing
two-CTA path because its extra occupancy is faster at batch size 64; larger
launches and B300 use the one-CTA specialization.

Backward chooses among the original tensor-core kernel, the new small-MMA
Q-major kernel, and `hstu_bwd_q1.py` using the measured architecture-specific
crossovers above. Unsupported layouts and non-causal or arbitrary masks keep
the existing implementation.

This is the same high-level loop-order lesson as the FA2 `bwd_loop_opt`
branch, specialized further for the exact one-query case.

## Split-KV assessment

HSTU's SiLU-weighted output is additive across KV partitions, so forward
split-KV does not need the softmax renormalization used by standard attention.
It is technically straightforward, but it adds partial-output storage and a
reduction launch. At batch sizes 512 and 1024 there are already 2048 and 4096
independent batch/head CTAs, so the measured kernels have enough parallelism
without splitting. Batch size 64 is the only target likely to benefit, but its
measured forward latency is already 0.0569 ms on B300 and 0.0441 ms on Rubin.
For that reason split-KV is left as a follow-up experiment rather than added
to the main path.

## Correctness

The benchmark checks KV lengths `1, 127, 128, 2049, 3072` against a float32
PyTorch oracle. The oracle runs on CPU so it also works on early Rubin systems
whose PyTorch device-code toolchain does not yet recognize SM10.7. A separate
forced-path test checks `tc-small` against `tc` at KV lengths
`1, 127, 128, 129, 2049`; dQ, dK, and dV are bitwise identical on both B300
and Rubin, including 12 repeated launches used to check for races. The final
automatic dispatch passes the forward and all three gradient checks on both
architectures, with maximum absolute gradient error around `1e-5`.

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

## Results

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

## Tensor-core Q-major backward experiment

The benchmark can force three backward implementations with
`--backward-impl legacy|tc|direct`. `legacy` is the original tensor-core
schedule, `tc` is the Q-major tensor-core experiment, and `direct` is the
specialized CUDA-core path used by the production dispatch at batch sizes of
at least 256. `auto` retains that production dispatch.

The following measurements use the same workload and allocations as above,
with 10 warmups and the median of five groups of 30 executions.

### B300 (SM10.3)

| Batch | Legacy TC (ms) | Q-major TC (ms) | Q-major vs legacy | Scalar direct (ms) | Direct vs Q-major TC |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.2083 | 0.2548 | 0.82x | 0.3866 | 0.66x |
| 512 | 1.7138 | 1.4321 | 1.20x | 1.0298 | 1.39x |
| 1024 | 3.4460 | 2.8148 | 1.22x | 1.8693 | 1.51x |

### Rubin (SM10.7)

| Batch | Legacy TC (ms) | Q-major TC (ms) | Q-major vs legacy | Scalar direct (ms) | Direct vs Q-major TC |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.1301 | 0.1763 | 0.74x | 0.3288 | 0.54x |
| 512 | 0.9992 | 0.8766 | 1.14x | 0.7644 | 1.15x |
| 1024 | 2.0010 | 1.7084 | 1.17x | 1.3456 | 1.27x |

The Q-major tensor-core path follows the FA2 `bwd_loop_opt` loop-order idea:
one CTA owns a batch/head pair, keeps its one Q row fixed, and walks the KV
tiles. Each KV tile is still evaluated by UMMA. The one valid dQ row is copied
from TMEM and accumulated in registers across KV tiles, then written once;
dK and dV are written directly by their owning CTA. This removes the global
dQ accumulation workspace, workspace zeroing, conversion launch, and global
dQ reductions from this path.

The experiment improves the old tensor-core path once the batch supplies
enough batch/head CTAs, but it does not beat the scalar direct kernel. With
`seqlen_q=1`, an UMMA tile still reserves and evaluates a 128-row Q tile while
only one row is useful. The direct kernel instead assigns warps to disjoint KV
rows and computes only the required row. Reducing the tensor-core Q tile to 64
is not a standalone tuning change: the current dQ MMA/TMEM load mapping assumes
the 128-row layout and needs a different transposed dQ MMA design. Therefore
the Q-major tensor-core implementation remains an explicit comparison option,
and the `auto` policy remains unchanged: legacy at batch 64 and scalar direct
at batch 512/1024 for the target cases.

The forced `tc` path passes the variable-length oracle on both architectures.
Maximum absolute `(dQ, dK, dV)` errors are `(9.13e-6, 1.83e-6, 1.45e-6)` on
B300 and `(6.65e-6, 1.22e-6, 1.04e-6)` on Rubin.

## Design

The forward kernel uses one Q stage for a one-token query, removing the
second fully masked 128-row Q tile. Small Rubin launches retain the existing
two-CTA path because its extra occupancy is faster at batch size 64; larger
launches and B300 use the one-CTA specialization.

Backward uses a workload dispatch:

- Batches below 256 retain the tensor-core kernel, but use one CTA instead of
  a two-CTA cluster for D128 single-query work.
- Batches of at least 256 use `hstu_bwd_q1.py`. One CTA owns one batch/head
  pair, eight warps traverse disjoint KV rows, dK/dV are written directly,
  and dQ stays in registers until one shared-memory CTA reduction. This
  Q-major path has no dQ atomics, FP32 accumulation workspace, workspace
  zeroing, or gradient conversion kernel.
- Unsupported layouts and non-causal or arbitrary masks keep the existing
  implementation.

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
batch-256 test exercises the direct Q-major backward dispatch. B300 and Rubin
both pass the forward and all three gradient checks.

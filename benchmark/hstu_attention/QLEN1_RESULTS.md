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

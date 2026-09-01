# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the packed HSTU qlen=1, variable-KV workload.

The default cases model the target decode-like workload: H=4, D=128, every
sequence has one query, and per-sequence KV lengths vary around 2K tokens.
Both explicit APIs use preallocated public outputs. Backward timing includes
the internal dQ accumulation workspace and conversion kernel because those are
part of the current HSTU execution path.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

import cudnn
from cudnn import HSTUBwdSm100, HSTUFwdSm100


_KV_LENGTH_PATTERN = (1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072)


def _kv_lengths(batch_size: int, average_kv: int) -> list[int]:
    if average_kv <= 0:
        raise ValueError("average_kv must be positive")
    scaled = [max(1, round(length * average_kv / 2048)) for length in _KV_LENGTH_PATTERN]
    # Rotate each repetition so non-multiples of nine do not always inherit the
    # same short prefix of the symmetric pattern.
    lengths = []
    for batch_idx in range(batch_size):
        repetition = batch_idx // len(scaled)
        pattern_idx = (batch_idx + repetition * 4) % len(scaled)
        lengths.append(scaled[pattern_idx])
    return lengths


def _cu_seqlens(lengths: list[int], device: torch.device) -> torch.Tensor:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return torch.tensor(offsets, dtype=torch.int32, device=device)


def _make_inputs(
    batch_size: int,
    heads: int,
    head_dim: int,
    average_kv: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor | list[int]]:
    q_lengths = [1] * batch_size
    k_lengths = _kv_lengths(batch_size, average_kv)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    def randn(tokens: int) -> torch.Tensor:
        return torch.randn(
            (tokens, heads, head_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        ) * 0.2

    q = randn(batch_size)
    k = randn(sum(k_lengths))
    v = randn(sum(k_lengths))
    do = randn(batch_size)
    return {
        "q": q,
        "k": k,
        "v": v,
        "do": do,
        "cu_q": _cu_seqlens(q_lengths, device),
        "cu_k": _cu_seqlens(k_lengths, device),
        "k_lengths": k_lengths,
    }


def _reference_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_lengths: list[int],
    alpha: float,
    scaling_seqlen: float,
) -> torch.Tensor:
    outputs = []
    k_offset = 0
    for batch_idx, k_length in enumerate(k_lengths):
        q_i = q[batch_idx : batch_idx + 1]
        k_i = k[k_offset : k_offset + k_length]
        v_i = v[k_offset : k_offset + k_length]
        scores = alpha * torch.einsum("qhd,khd->hqk", q_i, k_i)
        weights = F.silu(scores)
        outputs.append(torch.einsum("hqk,khd->qhd", weights, v_i) / scaling_seqlen)
        k_offset += k_length
    return torch.cat(outputs, dim=0)


def _measure_ms(run: Callable[[], None], warmup: int, iterations: int, groups: int) -> dict[str, float | list[float]]:
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    samples = []
    for _ in range(groups):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            run()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def _compile_forward(
    tensors: dict[str, torch.Tensor | list[int]],
    max_k: int,
    window_size: tuple[int, int],
    alpha: float,
    scaling_seqlen: float,
) -> tuple[torch.Tensor, Callable[[], None], float]:
    q = tensors["q"]
    k = tensors["k"]
    v = tensors["v"]
    cu_q = tensors["cu_q"]
    cu_k = tensors["cu_k"]
    assert isinstance(q, torch.Tensor)
    assert isinstance(k, torch.Tensor)
    assert isinstance(v, torch.Tensor)
    assert isinstance(cu_q, torch.Tensor)
    assert isinstance(cu_k, torch.Tensor)
    out = torch.empty_like(q)
    api = HSTUFwdSm100(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=out,
        sample_cu_seqlens_q=cu_q,
        sample_cu_seqlens_k=cu_k,
        max_seqlen_q=1,
        max_seqlen_k=max_k,
        window_size=window_size,
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
    )
    api.check_support()
    start = time.perf_counter()
    api.compile()
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - start

    def run() -> None:
        api.execute(q, k, v, out, cu_q, cu_k)

    return out, run, compile_seconds


def _compile_backward(
    tensors: dict[str, torch.Tensor | list[int]],
    max_k: int,
    window_size: tuple[int, int],
    alpha: float,
    scaling_seqlen: float,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], Callable[[], None], float]:
    q = tensors["q"]
    k = tensors["k"]
    v = tensors["v"]
    do = tensors["do"]
    cu_q = tensors["cu_q"]
    cu_k = tensors["cu_k"]
    assert isinstance(q, torch.Tensor)
    assert isinstance(k, torch.Tensor)
    assert isinstance(v, torch.Tensor)
    assert isinstance(do, torch.Tensor)
    assert isinstance(cu_q, torch.Tensor)
    assert isinstance(cu_k, torch.Tensor)
    dq, dk, dv = (torch.empty_like(tensor) for tensor in (q, k, v))
    api = HSTUBwdSm100(
        sample_do=do,
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_dq=dq,
        sample_dk=dk,
        sample_dv=dv,
        sample_cu_seqlens_q=cu_q,
        sample_cu_seqlens_k=cu_k,
        max_seqlen_q=1,
        max_seqlen_k=max_k,
        window_size=window_size,
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
    )
    api.check_support()
    start = time.perf_counter()
    api.compile()
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - start

    def run() -> None:
        api.execute(do, q, k, v, dq, dk, dv, cu_q, cu_k)

    return (dq, dk, dv), run, compile_seconds


def _correctness(
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    window_size: tuple[int, int],
    alpha: float,
    scaling_seqlen: float,
) -> dict[str, float | bool]:
    # Keep enough query-head CTAs to exercise Rubin's supported clustered
    # launch while repeating boundary-heavy KV tails.
    correctness_batch = 64
    boundary_lengths = (1, 127, 128, 2049, 3072)
    tensors = _make_inputs(correctness_batch, heads, head_dim, 2048, dtype, device, seed=20260901)
    k_lengths = tensors["k_lengths"]
    assert isinstance(k_lengths, list)
    # Preserve the benchmark's packed allocation shape so correctness and
    # timing compile the same dynamic-layout specialization. Redistribute the
    # rows removed by the boundary cases over the remaining sequences.
    packed_kv = sum(k_lengths)
    k_lengths[: len(boundary_lengths)] = boundary_lengths
    rows_to_restore = packed_kv - sum(k_lengths)
    for index in range(len(boundary_lengths), len(k_lengths)):
        rows_added = min(rows_to_restore, max(boundary_lengths) - k_lengths[index])
        k_lengths[index] += rows_added
        rows_to_restore -= rows_added
    assert rows_to_restore == 0
    # Rebuild K/V and metadata for the boundary-heavy correctness lengths.
    generator = torch.Generator(device=device)
    generator.manual_seed(20260902)
    tensors["k"] = torch.randn((sum(k_lengths), heads, head_dim), dtype=dtype, device=device, generator=generator) * 0.2
    tensors["v"] = torch.randn((sum(k_lengths), heads, head_dim), dtype=dtype, device=device, generator=generator) * 0.2
    tensors["cu_k"] = _cu_seqlens(k_lengths, device)

    q = tensors["q"]
    k = tensors["k"]
    v = tensors["v"]
    do = tensors["do"]
    assert isinstance(q, torch.Tensor)
    assert isinstance(k, torch.Tensor)
    assert isinstance(v, torch.Tensor)
    assert isinstance(do, torch.Tensor)

    q_ref = q.float().detach().requires_grad_(True)
    k_ref = k.float().detach().requires_grad_(True)
    v_ref = v.float().detach().requires_grad_(True)
    expected_out = _reference_forward(q_ref, k_ref, v_ref, k_lengths, alpha, scaling_seqlen)
    expected_grads = torch.autograd.grad(expected_out, (q_ref, k_ref, v_ref), do.float())

    actual_out, fwd_run, _ = _compile_forward(tensors, max(k_lengths), window_size, alpha, scaling_seqlen)
    actual_grads, bwd_run, _ = _compile_backward(tensors, max(k_lengths), window_size, alpha, scaling_seqlen)
    fwd_run()
    bwd_run()
    torch.cuda.synchronize()

    fwd_error = (actual_out.float() - expected_out).abs()
    grad_errors = [(actual.float() - expected).abs() for actual, expected in zip(actual_grads, expected_grads)]
    forward_ok = torch.allclose(actual_out.float(), expected_out, rtol=3.0e-2, atol=3.0e-2)
    backward_ok = all(torch.allclose(actual.float(), expected, rtol=6.0e-2, atol=6.0e-2) for actual, expected in zip(actual_grads, expected_grads))
    return {
        "forward_ok": bool(forward_ok),
        "backward_ok": bool(backward_ok),
        "forward_max_abs": float(fwd_error.max().item()),
        "dq_max_abs": float(grad_errors[0].max().item()),
        "dk_max_abs": float(grad_errors[1].max().item()),
        "dv_max_abs": float(grad_errors[2].max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(64, 512, 1024))
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--average-kv", type=int, default=2048)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--mask", choices=("causal", "full"), default="causal")
    parser.add_argument("--direction", choices=("forward", "backward", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--groups", type=int, default=7)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_hstu_qlen1.py requires a CUDA GPU")
    device = torch.device("cuda:0")
    dtype = getattr(torch, args.dtype)
    window_size = (-1, 0) if args.mask == "causal" else (-1, -1)
    alpha = 0.7
    scaling_seqlen = 2048.0
    results: dict[str, object] = {
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "cudnn_module": cudnn.__file__,
        "cute_dsl_arch": os.environ.get("CUTE_DSL_ARCH"),
        "dtype": args.dtype,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "average_kv_target": args.average_kv,
        "mask": args.mask,
        "direction": args.direction,
    }

    if not args.skip_correctness:
        correctness = _correctness(args.heads, args.head_dim, dtype, device, window_size, alpha, scaling_seqlen)
        print("CORRECTNESS " + json.dumps(correctness, sort_keys=True), flush=True)
        if not correctness["forward_ok"] or not correctness["backward_ok"]:
            raise AssertionError("HSTU qlen=1 correctness check failed")
        results["correctness"] = correctness

    case_results = []
    for case_index, batch_size in enumerate(args.batch_sizes):
        tensors = _make_inputs(batch_size, args.heads, args.head_dim, args.average_kv, dtype, device, seed=1234 + case_index)
        k_lengths = tensors["k_lengths"]
        assert isinstance(k_lengths, list)
        case: dict[str, object] = {
            "batch_size": batch_size,
            "total_q": batch_size,
            "total_kv": sum(k_lengths),
            "average_kv": sum(k_lengths) / batch_size,
            "min_kv": min(k_lengths),
            "max_kv": max(k_lengths),
        }
        if args.direction in ("forward", "both"):
            _, run, compile_seconds = _compile_forward(tensors, max(k_lengths), window_size, alpha, scaling_seqlen)
            case["forward"] = {
                "compile_seconds": compile_seconds,
                **_measure_ms(run, args.warmup, args.iterations, args.groups),
            }
        if args.direction in ("backward", "both"):
            _, run, compile_seconds = _compile_backward(tensors, max(k_lengths), window_size, alpha, scaling_seqlen)
            case["backward"] = {
                "compile_seconds": compile_seconds,
                **_measure_ms(run, args.warmup, args.iterations, args.groups),
            }
        print("CASE " + json.dumps(case, sort_keys=True), flush=True)
        case_results.append(case)
        del tensors
        torch.cuda.empty_cache()

    results["cases"] = case_results
    payload = json.dumps(results, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(f"WROTE {args.output}", flush=True)
    else:
        print(payload, end="", flush=True)


if __name__ == "__main__":
    main()

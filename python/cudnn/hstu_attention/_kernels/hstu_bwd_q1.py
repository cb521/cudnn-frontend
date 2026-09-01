# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HSTU backward specialized for one query per sequence."""

import operator
from typing import Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32

from .utils import tanhf


class HSTUAttentionBackwardQlen1Sm100:
    """Q-major CUDA-core kernel for HSTU qlen=1 and D=128.

    One CTA owns a batch/head pair.  Its eight warps traverse disjoint KV
    rows, write dK/dV without atomics, and retain their dQ partials in
    registers until a single CTA reduction at the end.
    """

    num_threads = 256
    num_warps = num_threads // cute.arch.WARP_SIZE
    values_per_lane = 128 // cute.arch.WARP_SIZE
    head_dim = 128

    def __init__(self, element_dtype: Type[cutlass.Numeric]):
        self.element_dtype = element_dtype
        self.smem_bytes = self.num_warps * self.head_dim * Float32.width // 8

    @cute.kernel
    def kernel(
        self,
        Q: cute.Tensor,
        K: cute.Tensor,
        V: cute.Tensor,
        dO: cute.Tensor,
        dQ: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        alpha: Float32,
        scaling_seqlen: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        batch_idx, head_idx, _ = cute.arch.block_idx()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()

        q_begin = cu_seqlens_q[batch_idx]
        q_end = cu_seqlens_q[batch_idx + 1]
        k_begin = cu_seqlens_k[batch_idx]
        k_end = cu_seqlens_k[batch_idx + 1]
        has_query = q_begin < q_end

        rQ = cute.make_rmem_tensor(cute.make_layout((self.values_per_lane,)), Float32)
        rdO = cute.make_rmem_tensor(cute.make_layout((self.values_per_lane,)), Float32)
        rdQ = cute.make_rmem_tensor(cute.make_layout((self.values_per_lane,)), Float32)
        for value_idx in cutlass.range_constexpr(self.values_per_lane):
            dim_idx = lane_idx + value_idx * cute.arch.WARP_SIZE
            if has_query:
                rQ[value_idx] = Q[q_begin, head_idx, dim_idx].to(Float32)
                rdO[value_idx] = dO[q_begin, head_idx, dim_idx].to(Float32)
            else:
                rQ[value_idx] = Float32(0.0)
                rdO[value_idx] = Float32(0.0)
            rdQ[value_idx] = Float32(0.0)

        inv_scaling = cute.arch.rcp_approx(scaling_seqlen)
        grad_scale = alpha * inv_scaling
        k_idx = k_begin + warp_idx
        while k_idx < k_end:
            rK = cute.make_rmem_tensor(cute.make_layout((self.values_per_lane,)), Float32)
            rV = cute.make_rmem_tensor(cute.make_layout((self.values_per_lane,)), Float32)
            qk = Float32(0.0)
            dov = Float32(0.0)
            for value_idx in cutlass.range_constexpr(self.values_per_lane):
                dim_idx = lane_idx + value_idx * cute.arch.WARP_SIZE
                rK[value_idx] = K[k_idx, head_idx, dim_idx].to(Float32)
                rV[value_idx] = V[k_idx, head_idx, dim_idx].to(Float32)
                qk += rQ[value_idx] * rK[value_idx]
                dov += rdO[value_idx] * rV[value_idx]

            qk = cute.arch.warp_reduction(qk, operator.add)
            dov = cute.arch.warp_reduction(dov, operator.add)
            score = alpha * qk
            score_tanh = tanhf(score * Float32(0.5))
            sigmoid = Float32(0.5) * score_tanh + Float32(0.5)
            weight = score * sigmoid
            dsilu = sigmoid + weight * (Float32(1.0) - sigmoid)
            ds = dov * dsilu

            for value_idx in cutlass.range_constexpr(self.values_per_lane):
                dim_idx = lane_idx + value_idx * cute.arch.WARP_SIZE
                dK[k_idx, head_idx, dim_idx] = dK.element_type(ds * grad_scale * rQ[value_idx])
                dV[k_idx, head_idx, dim_idx] = dV.element_type(weight * inv_scaling * rdO[value_idx])
                rdQ[value_idx] += ds * rK[value_idx]
            k_idx += self.num_warps

        smem = utils.SmemAllocator()
        sdQ = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.head_dim, self.num_warps), stride=(1, self.head_dim)),
            byte_alignment=16,
        )
        for value_idx in cutlass.range_constexpr(self.values_per_lane):
            dim_idx = lane_idx + value_idx * cute.arch.WARP_SIZE
            sdQ[dim_idx, warp_idx] = rdQ[value_idx]
        cute.arch.barrier()

        if tidx < self.head_dim and has_query:
            dq_sum = Float32(0.0)
            for reduce_warp in cutlass.range_constexpr(self.num_warps):
                dq_sum += sdQ[tidx, reduce_warp]
            dQ[q_begin, head_idx, tidx] = dQ.element_type(dq_sum * grad_scale)

    @cute.jit
    def __call__(
        self,
        Q: cute.Tensor,
        K: cute.Tensor,
        V: cute.Tensor,
        dO: cute.Tensor,
        dQ: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        batch_size: Int32,
        num_heads: Int32,
        alpha: Float32,
        scaling_seqlen: Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            Q,
            K,
            V,
            dO,
            dQ,
            dK,
            dV,
            cu_seqlens_q,
            cu_seqlens_k,
            alpha,
            scaling_seqlen,
        ).launch(
            grid=(batch_size, num_heads, 1),
            block=(self.num_threads, 1, 1),
            smem=self.smem_bytes,
            stream=stream,
        )

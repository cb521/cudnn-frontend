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

    One CTA owns a batch/head pair. Its warps traverse disjoint KV rows,
    use contiguous 64-bit per-lane K/V and dK/dV transfers, write dK/dV
    without atomics, and retain their dQ partials in registers until a
    single CTA reduction at the end.
    """

    values_per_lane = 128 // cute.arch.WARP_SIZE
    head_dim = 128

    def __init__(self, element_dtype: Type[cutlass.Numeric], num_threads: int = 256):
        assert 0 < num_threads <= 1024 and num_threads % cute.arch.WARP_SIZE == 0
        self.element_dtype = element_dtype
        self.num_threads = num_threads
        self.num_warps = num_threads // cute.arch.WARP_SIZE
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
            dim_idx = lane_idx * self.values_per_lane + value_idx
            if has_query:
                rQ[value_idx] = Q[q_begin, head_idx, dim_idx].to(Float32)
                rdO[value_idx] = dO[q_begin, head_idx, dim_idx].to(Float32)
            else:
                rQ[value_idx] = Float32(0.0)
                rdO[value_idx] = Float32(0.0)
            rdQ[value_idx] = Float32(0.0)

        inv_scaling = cute.arch.rcp_approx(scaling_seqlen)
        grad_scale = alpha * inv_scaling
        vector_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.element_dtype,
            num_bits_per_copy=self.values_per_lane * self.element_dtype.width,
        )
        vector_copy = cute.make_tiled_copy_tv(
            vector_copy_atom,
            cute.make_layout((1, cute.arch.WARP_SIZE)),
            cute.make_layout((1, self.values_per_lane)),
        )
        thread_copy = vector_copy.get_slice(lane_idx)
        row_layout = cute.make_layout((1, self.head_dim), stride=(0, 1))
        k_idx = k_begin + warp_idx
        while k_idx < k_end:
            row_offset_k = cute.assume(
                k_idx * K.stride[0] + head_idx * K.stride[1],
                divby=128 // self.element_dtype.width,
            )
            row_offset_v = cute.assume(
                k_idx * V.stride[0] + head_idx * V.stride[1],
                divby=128 // self.element_dtype.width,
            )
            gK = cute.make_tensor(K.iterator + row_offset_k, row_layout)
            gV = cute.make_tensor(V.iterator + row_offset_v, row_layout)
            tKgK = thread_copy.partition_S(gK)
            tVgV = thread_copy.partition_S(gV)
            tKrK = cute.make_fragment_like(tKgK)
            tVrV = cute.make_fragment_like(tVgV)
            cute.copy(vector_copy_atom, tKgK, tKrK)
            cute.copy(vector_copy_atom, tVgV, tVrV)
            rK = tKrK.load().to(Float32)
            qk = Float32(0.0)
            for value_idx in cutlass.range_constexpr(self.values_per_lane):
                qk += rQ[value_idx] * rK[value_idx]
            qk = cute.arch.warp_reduction(qk, operator.add)

            rV = tVrV.load().to(Float32)
            dov = Float32(0.0)
            for value_idx in cutlass.range_constexpr(self.values_per_lane):
                dov += rdO[value_idx] * rV[value_idx]
            dov = cute.arch.warp_reduction(dov, operator.add)
            score = alpha * qk
            score_tanh = tanhf(score * Float32(0.5))
            sigmoid = Float32(0.5) * score_tanh + Float32(0.5)
            weight = score * sigmoid
            dsilu = sigmoid + weight * (Float32(1.0) - sigmoid)
            ds = dov * dsilu

            rdK = cute.make_rmem_tensor(cute.make_layout((self.values_per_lane,)), dK.element_type)
            for value_idx in cutlass.range_constexpr(self.values_per_lane):
                rdK[value_idx] = dK.element_type(ds * grad_scale * rQ[value_idx])
                rdQ[value_idx] += ds * rK[value_idx]
            row_offset_dk = cute.assume(
                k_idx * dK.stride[0] + head_idx * dK.stride[1],
                divby=128 // self.element_dtype.width,
            )
            gdK = cute.make_tensor(dK.iterator + row_offset_dk, row_layout)
            tDgdK = thread_copy.partition_D(gdK)
            tDrdK = cute.make_fragment_like(tDgdK)
            tDrdK.store(rdK.load())
            cute.copy(vector_copy_atom, tDrdK, tDgdK)

            rdV = cute.make_rmem_tensor(cute.make_layout((self.values_per_lane,)), dV.element_type)
            for value_idx in cutlass.range_constexpr(self.values_per_lane):
                rdV[value_idx] = dV.element_type(weight * inv_scaling * rdO[value_idx])
            row_offset_dv = cute.assume(
                k_idx * dV.stride[0] + head_idx * dV.stride[1],
                divby=128 // self.element_dtype.width,
            )
            gdV = cute.make_tensor(dV.iterator + row_offset_dv, row_layout)
            tDgdV = thread_copy.partition_D(gdV)
            tDrdV = cute.make_fragment_like(tDgdV)
            tDrdV.store(rdV.load())
            cute.copy(vector_copy_atom, tDrdV, tDgdV)
            k_idx += self.num_warps

        smem = utils.SmemAllocator()
        sdQ = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.head_dim, self.num_warps), stride=(1, self.head_dim)),
            byte_alignment=16,
        )
        for value_idx in cutlass.range_constexpr(self.values_per_lane):
            dim_idx = lane_idx * self.values_per_lane + value_idx
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

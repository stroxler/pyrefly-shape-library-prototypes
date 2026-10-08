# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Kernel bodies copied from Triton's numbered 06-fused-attention-ws.py (beta).
# Triton's LICENSE includes the following notice:
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Seven numbered warp-specialized attention kernels with typed parameters.

The original JIT bodies retain their statements, except that the outer backward
kernel names selected head pointers locally to preserve their lowered types.
The forward autotuner and upstream GPU wrapper are not included. The checked
host boundary below covers preprocessing only; the forward and backward tests
compile their real nested helpers without launching a GPU. The four local type
ignores cover the split/join shape rule, FP8 value layout, and float constexpr.
"""

from __future__ import annotations

import os
import unittest
from typing import Literal, cast

import torch
import triton.language as tl
from shape_extensions import Int, IntListLiteral, IntVar
from triton.backends.compiler import GPUTarget
from triton_examples.testing import compile_ttir
from triton_library import tlt
from triton_library.launch_layout import attention_preprocess_output
from triton_library.semantic_jit import ConstExpr, semantic_jit
from triton_library.torch_views import (
    checked_attention_backward_delta,
    checked_attention_backward_input,
)

BM = IntVar("BM")
BN = IntVar("BN")
D = IntVar("D")
Y = IntVar("Y")
NC = IntVar("NC")
R = IntVar("R")
C = IntVar("C")
S = IntVar("S")
BR = IntVar("BR")
BC = IntVar("BC")
ZDim = IntVar("ZDim")
HDim = IntVar("HDim")
NDim = IntVar("NDim")
Batch = IntVar("Batch")
Heads = IntVar("Heads")
Tokens = IntVar("Tokens")
Dim = IntVar("Dim")
BM1 = IntVar("BM1")
BN1 = IntVar("BN1")
BM2 = IntVar("BM2")
BN2 = IntVar("BN2")
TokenStride = IntVar("TokenStride")
FeatureStride = IntVar("FeatureStride")
SliceFactor = IntVar("SliceFactor")


@semantic_jit
def _attn_fwd_inner(
    acc: tl.tensor[[BM, D]],
    l_i: tl.tensor[[BM]],
    m_i: tl.tensor[[BM]],
    q: tl.tensor[[BM, D]],  #
    desc_k: tl.tensor_descriptor[Y, D, D, BN, D],
    desc_v: tl.tensor_descriptor[Y, D, D, BN, D],  #
    offset_y: int,
    dtype: ConstExpr[object],
    start_m: tl.ProgramId[Literal[0]],
    qk_scale: float,  #
    BLOCK_M: ConstExpr[Int[BM]],
    HEAD_DIM: ConstExpr[Int[D]],
    BLOCK_N: ConstExpr[Int[BN]],  #
    STAGE: ConstExpr[int],
    offs_m: tl.Offsets[[BM], 1, str, Literal[0]],
    offs_n: tl.Offsets[[BN]],  #
    N_CTX: ConstExpr[Int[NC]],
    warp_specialize: ConstExpr[bool],
    IS_HOPPER: ConstExpr[bool],
):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    offsetk_y = offset_y + lo
    if dtype == tl.float8e5:
        offsetv_y = offset_y * HEAD_DIM + lo
    else:
        offsetv_y = offset_y + lo
    # loop over k, v and update accumulator
    for start_n in tl.range(
        lo, hi, BLOCK_N, warp_specialize=warp_specialize, disallow_acc_multi_buffer=True
    ):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        # -- update output accumulator --
        if not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = (
                acc.reshape([BM, 2, BN // 2])  # pyrefly: ignore[no-matching-overload]
                .permute(0, 2, 1)
                .split()
            )
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = (
                tl.join(acc0, acc1)  # pyrefly: ignore[missing-attribute]
                .permute(0, 2, 1)
                .reshape([BM, BN])
            )
        else:
            acc = acc * alpha[:, None]
        # prepare p and v for the dot
        if dtype == tl.float8e5:
            v = desc_v.load([0, offsetv_y]).T
        else:
            v = desc_v.load([offsetv_y, 0])
        p = p.to(dtype)
        # note that this non transposed v for FP8 is only supported on Blackwell
        acc = tl.dot(p, v, acc)  # pyrefly: ignore[bad-argument-type]
        # update m_i and l_i
        # place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i


@semantic_jit
def _maybe_make_tensor_desc(
    desc_or_ptr: tl.tensor_descriptor[R, C, S, BR, BC] | tl.AttentionPointer[R, C, S],
    shape: IntListLiteral[[R, C]],
    strides: IntListLiteral[[S, 1]],
    block_shape: IntListLiteral[[BR, BC]],
) -> tl.tensor_descriptor[R, C, S, BR, BC]:
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


# Keep union annotations free of extra parentheses: semantic_jit erases these spans.
# fmt: off
@semantic_jit
def _attn_fwd(
    sm_scale: float,
    M: tl.AttentionStatsPointer[ZDim * HDim, NDim],  #
    Z: ConstExpr[Int[ZDim]],
    H: ConstExpr[Int[HDim]],
    desc_q: tl.AttentionPointer[ZDim * HDim * NDim, D, D]
    | tl.tensor_descriptor[ZDim * HDim * NDim, D, D, BM, D],
    desc_k: tl.AttentionPointer[ZDim * HDim * NDim, D, D]
    | tl.tensor_descriptor[ZDim * HDim * NDim, D, D, BN, D],
    desc_v: tl.AttentionPointer[ZDim * HDim * NDim, D, D]
    | tl.tensor_descriptor[ZDim * HDim * NDim, D, D, BN, D],
    desc_o: tl.AttentionPointer[ZDim * HDim * NDim, D, D]
    | tl.tensor_descriptor[ZDim * HDim * NDim, D, D, BM, D],
    N_CTX: ConstExpr[Int[NDim]],  #
    HEAD_DIM: ConstExpr[Int[D]],  #
    BLOCK_M: ConstExpr[Int[BM]],  #
    BLOCK_N: ConstExpr[Int[BN]],  #
    FP8_OUTPUT: ConstExpr[Literal[False]],  #
    STAGE: ConstExpr[int],  #
    warp_specialize: ConstExpr[bool],  #
    IS_HOPPER: ConstExpr[bool],  #
):
    # fmt: on
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(
        desc_q,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    if FP8_OUTPUT:
        desc_v = _maybe_make_tensor_desc(
            desc_v,
            shape=[HEAD_DIM, y_dim],
            strides=[N_CTX, 1],
            block_shape=[HEAD_DIM, BLOCK_N],
        )
    else:
        desc_v = _maybe_make_tensor_desc(
            desc_v,
            shape=[y_dim, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=[BLOCK_N, HEAD_DIM],
        )
    desc_k = _maybe_make_tensor_desc(
        desc_k,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_o = _maybe_make_tensor_desc(
        desc_o,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = desc_q.load([qo_offset_y, 0])
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,  #
            desc_k,
            desc_v,  #
            offset_y,
            dtype,
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            4 - STAGE,
            offs_m,
            offs_n,
            N_CTX,  #
            warp_specialize,
            IS_HOPPER,
        )
    # stage 2: on-band
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,  #
            desc_k,
            desc_v,  #
            offset_y,
            dtype,
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            2,
            offs_m,
            offs_n,
            N_CTX,  #
            warp_specialize,
            IS_HOPPER,
        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))


@semantic_jit
def _attn_bwd_preprocess(
    O: tlt.InPointer[[Batch, Heads, Tokens, Dim], [int, int, Dim, 1]],
    DO: tlt.InPointer[[Batch, Heads, Tokens, Dim], [int, int, Dim, 1]],  #
    Delta: tlt.OutPointer[[Batch, Heads, Tokens], [int, Tokens, 1]],  #
    Z: Int[Batch],
    H: Int[Heads],
    N_CTX: Int[Tokens],  #
    BLOCK_M: ConstExpr[Int[BM]],
    HEAD_DIM: ConstExpr[Int[Dim]],  #
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(
        O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]
    )
    do = tl.load(
        DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


@semantic_jit
def _attn_bwd_dkdv(
    dk: tl.tensor[[BN, Dim]],
    dv: tl.tensor[[BN, Dim]],  #
    Q: tl.AttentionHeadLocalInputPointer[Tokens, Dim, TokenStride, FeatureStride],
    k: tl.tensor[[BN, Dim]],
    v: tl.tensor[[BN, Dim]],
    sm_scale: float,  #
    DO: tl.AttentionHeadLocalInputPointer[Tokens, Dim, TokenStride, FeatureStride],  #
    M: tl.AttentionHeadLocalStatsPointer[Tokens],
    D: tl.AttentionHeadLocalStatsPointer[Tokens],  #
    stride_tok: Int[TokenStride],
    stride_d: Int[FeatureStride],  #
    H: Int[Heads],
    N_CTX: Int[Tokens],
    BLOCK_M1: ConstExpr[Int[BM]],  #
    BLOCK_N1: ConstExpr[Int[BN]],  #
    HEAD_DIM: ConstExpr[Int[Dim]],  #
    start_n: int,
    start_m: int,
    num_steps: int,  #
    MASK: ConstExpr[bool],
    warp_specialize: ConstExpr[bool] = False,
):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in tl.range(0, num_steps, warp_specialize=warp_specialize):
        qT = tl.load(qT_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = offs_m[None, :] >= offs_n[:, None]
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(tl.float16)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)
        dk += tl.dot(dsT, tl.trans(qT))
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    return dk, dv


@semantic_jit
def _attn_bwd_dq(
    dq: tl.tensor[[BM, Dim]],
    q: tl.tensor[[BM, Dim]],
    K: tl.AttentionHeadLocalInputPointer[Tokens, Dim, TokenStride, FeatureStride],
    V: tl.AttentionHeadLocalInputPointer[Tokens, Dim, TokenStride, FeatureStride],  #
    do: tl.tensor[[BM, Dim]],
    m: tl.tensor[[BM, 1]],
    D: tl.AttentionHeadLocalStatsPointer[Tokens],
    # shared by Q/K/V/DO.
    stride_tok: Int[TokenStride],
    stride_d: Int[FeatureStride],  #
    H: Int[Heads],
    N_CTX: Int[Tokens],  #
    BLOCK_M2: ConstExpr[Int[BM]],  #
    BLOCK_N2: ConstExpr[Int[BN]],  #
    HEAD_DIM: ConstExpr[Int[Dim]],
    # Filled in by the wrapper.
    start_m: int,
    start_n: int,
    num_steps: int,  #
    MASK: ConstExpr[bool],
    warp_specialize: ConstExpr[bool] = False,
):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in tl.range(0, num_steps, warp_specialize=warp_specialize):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = offs_m[:, None] >= offs_n[None, :]
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.float16)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq


@semantic_jit
def _attn_bwd(
    Q: tlt.InPointer[[Batch, Heads, Tokens, Dim], [int, int, Dim, 1]],
    K: tlt.InPointer[[Batch, Heads, Tokens, Dim], [int, int, Dim, 1]],
    V: tlt.InPointer[[Batch, Heads, Tokens, Dim], [int, int, Dim, 1]],
    sm_scale: float,
    DO: tlt.InPointer[[Batch, Heads, Tokens, Dim], [int, int, Dim, 1]],
    DQ: tlt.OutPointer[[Batch, Heads, Tokens, Dim], [int, int, Dim, 1]],
    DK: tlt.OutPointer[[Batch, Heads, Tokens, Dim], [int, int, Dim, 1]],
    DV: tlt.OutPointer[[Batch, Heads, Tokens, Dim], [int, int, Dim, 1]],
    M: tlt.InPointer[[Batch, Heads, Tokens], [int, Tokens, 1]],
    D: tlt.InPointer[[Batch, Heads, Tokens], [int, Tokens, 1]],
    stride_z: tl.AttentionBatchStride[Heads, int],
    stride_h: tl.AttentionHeadStride[Heads, int],
    stride_tok: Int[Dim],
    stride_d: Int[1],
    H: tl.AttentionHeadCount[Heads],
    N_CTX: Int[Tokens],
    BLOCK_M1: ConstExpr[Int[BM1]],
    BLOCK_N1: ConstExpr[Int[BN1]],
    BLOCK_M2: ConstExpr[Int[BM2]],
    BLOCK_N2: ConstExpr[Int[BN2]],
    BLK_SLICE_FACTOR: ConstExpr[Int[SliceFactor]],
    HEAD_DIM: ConstExpr[Int[Dim]],
    CAUSAL: ConstExpr[bool],
    warp_specialize: ConstExpr[bool] = False,
):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)  # pyrefly: ignore[bad-assignment]

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # Selecting a head changes pointer ranks; keep the original allocations unchanged.
    q_head = Q + adj
    k_head = K + adj
    v_head = V + adj
    do_head = DO + adj
    dq_head = DQ + adj
    dk_head = DK + adj
    dv_head = DV + adj
    m_head = M + off_chz
    delta_head = D + off_chz

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1
    start_m = 0

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load k_head and v_head: they stay in SRAM throughout the inner loop.
    k = tl.load(k_head + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(v_head + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    if CAUSAL:
        start_m = start_n
        num_steps = BLOCK_N1 // MASK_BLOCK_M1
        dk, dv = _attn_bwd_dkdv(
            dk,
            dv,  #
            q_head,
            k,
            v,
            sm_scale,  #
            do_head,  #
            m_head,
            delta_head,  #
            stride_tok,
            stride_d,  #
            H,
            N_CTX,  #
            MASK_BLOCK_M1,
            BLOCK_N1,
            HEAD_DIM,  #
            start_n,
            start_m,
            num_steps,  #
            MASK=True,  #
            warp_specialize=warp_specialize,
        )

        start_m += num_steps * MASK_BLOCK_M1

    # Compute dK and dV for non-masked blocks.
    num_steps = (N_CTX - start_m) // BLOCK_M1
    dk, dv = _attn_bwd_dkdv(  #
        dk,
        dv,  #
        q_head,
        k,
        v,
        sm_scale,  #
        do_head,  #
        m_head,
        delta_head,  #
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        BLOCK_M1,
        BLOCK_N1,
        HEAD_DIM,  #
        start_n,
        start_m,
        num_steps,  #
        MASK=False,  #
        warp_specialize=warp_specialize,
    )

    dv_ptrs = dv_head + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = dk_head + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # THIS BLOCK DOES dq_head:
    start_m = pid * BLOCK_M2
    start_n = 0
    num_steps = N_CTX // BLOCK_N2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(q_head + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(do_head + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(m_head + offs_m)
    m = m[:, None]

    if CAUSAL:
        # Compute dQ for masked (diagonal) blocks.
        # NOTE: This code scans each row of QK^T backward (from right to left,
        # but inside each call to _attn_bwd_dq, from left to right), but that's
        # not due to anything important.  I just wanted to reuse the loop
        # structure for dK & dV above as much as possible.
        end_n = start_m + BLOCK_M2
        num_steps = BLOCK_M2 // MASK_BLOCK_N2
        dq = _attn_bwd_dq(
            dq,
            q,
            k_head,
            v_head,  #
            do,
            m,
            delta_head,  #
            stride_tok,
            stride_d,  #
            H,
            N_CTX,  #
            BLOCK_M2,
            MASK_BLOCK_N2,
            HEAD_DIM,  #
            start_m,
            end_n - num_steps * MASK_BLOCK_N2,
            num_steps,  #
            MASK=True,  #
            warp_specialize=warp_specialize,
        )
        end_n -= num_steps * MASK_BLOCK_N2
        # stage 2
        num_steps = end_n // BLOCK_N2
        start_n = end_n - num_steps * BLOCK_N2

    dq = _attn_bwd_dq(
        dq,
        q,
        k_head,
        v_head,  #
        do,
        m,
        delta_head,  #
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        BLOCK_M2,
        BLOCK_N2,
        HEAD_DIM,  #
        start_m,
        start_n,
        num_steps,  #
        MASK=False,  #
        warp_specialize=warp_specialize,
    )
    # Write back dQ.
    dq_ptrs = dq_head + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


def attention_ws_delta[B: IntVar, H: IntVar, T: IntVar, F: IntVar](
    out: torch.Tensor[[B, H, T, F]],
    dout: torch.Tensor[[B, H, T, F]],
    *,
    block_m: int,
) -> torch.Tensor[[B, H, T]]:
    """Validate dense FP32 attention inputs and output before preprocessing."""
    if out.ndim != 4:
        raise ValueError("Attention expects [batch, heads, tokens, features]")
    batch, heads, tokens, dim = out.shape
    delta = torch.empty((batch, heads, tokens), dtype=torch.float32, device=out.device)
    layout = attention_preprocess_output(out, dout, delta, block_m=block_m)
    shape = (batch, heads, tokens, dim)
    layout.launch(
        _attn_bwd_preprocess,
        checked_attention_backward_input(out, shape),
        checked_attention_backward_input(dout, shape),
        checked_attention_backward_delta(delta, (batch, heads, tokens)),
        batch,
        heads,
        tokens,
        block_m,
        dim,
    )
    return delta


class NumberedWarpSpecializedAttentionTest(unittest.TestCase):
    """Check the numbered forward and backward kernels without a GPU."""

    def test_forward_frontend(self) -> None:
        if os.environ.get("TRITON_INTERPRET") == "1":
            self.skipTest("Frontend compilation uses normal JIT mode")
        ir = compile_ttir(
            _attn_fwd,
            signature={
                "sm_scale": "fp32",
                "M": "*fp32",
                "desc_q": "*fp16",
                "desc_k": "*fp16",
                "desc_v": "*fp16",
                "desc_o": "*fp16",
            },
            constexprs={
                "Z": 1,
                "H": 1,
                "N_CTX": 16,
                "HEAD_DIM": 16,
                "BLOCK_M": 16,
                "BLOCK_N": 16,
                "FP8_OUTPUT": 0,
                "STAGE": 1,
                "warp_specialize": 0,
                "IS_HOPPER": 0,
            },
            target=GPUTarget("cuda", 90, 32),
        )
        self.assertIn("tt.func", ir)

    def test_backward_frontend(self) -> None:
        if os.environ.get("TRITON_INTERPRET") == "1":
            self.skipTest("Frontend compilation uses normal JIT mode")
        ir = compile_ttir(
            _attn_bwd,
            signature={
                name: "*fp16" for name in ("Q", "K", "V", "DO", "DQ", "DK", "DV")
            }
            | {"M": "*fp32", "D": "*fp32", "sm_scale": "fp32"}
            | {
                name: "i32"
                for name in (
                    "stride_z",
                    "stride_h",
                    "stride_tok",
                    "stride_d",
                    "H",
                    "N_CTX",
                )
            },
            constexprs={
                "BLOCK_M1": 16,
                "BLOCK_N1": 32,
                "BLOCK_M2": 32,
                "BLOCK_N2": 16,
                "BLK_SLICE_FACTOR": 2,
                "HEAD_DIM": 16,
                "CAUSAL": 0,
                "warp_specialize": 0,
            },
        )
        self.assertIn("tt.func", ir)

    def test_preprocess_cpu(self) -> None:
        if os.environ.get("TRITON_INTERPRET") != "1":
            self.skipTest("CPU launches use interpreter mode")
        out = torch.arange(1 * 2 * 4 * 4, dtype=torch.float32).reshape(1, 2, 4, 4)
        dout = torch.flip(out, dims=(-1,)).contiguous()
        delta = attention_ws_delta(out, dout, block_m=2)
        torch.testing.assert_close(delta, (out * dout).sum(dim=-1))

    def test_preprocess_rejects_wrong_shape_and_dtype(self) -> None:
        out = torch.ones((1, 2, 4, 4), dtype=torch.float32)
        with self.assertRaises(ValueError):
            attention_ws_delta(
                out,
                cast("torch.Tensor[[1, 2, 4, 4]]", out[:, :1].contiguous()),
                block_m=2,
            )
        with self.assertRaises(ValueError):
            attention_ws_delta(out, out.half(), block_m=2)

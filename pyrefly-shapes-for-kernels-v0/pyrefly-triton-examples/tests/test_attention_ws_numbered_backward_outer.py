# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The kernel body is copied from Triton's 06-fused-attention-ws.py.
# Triton's LICENSE includes the following notice:
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

# This static-only kernel is checked by Pyrefly, not executed as a Buck test.
# @lint-ignore-every AUTODEPS2

"""Backward attention host-pointer roles with numbered warp-specialized helpers."""

import triton
import triton.language as tl
from shape_extensions import Int, IntVar
from tests.test_attention_ws_numbered_backward_dkdv import _attn_bwd_dkdv
from tests.test_attention_ws_numbered_backward_dq import _attn_bwd_dq


@triton.jit
def _attn_bwd[
    Batch: IntVar,
    Heads: IntVar,
    Tokens: IntVar,
    Dim: IntVar,
    StrideZ: IntVar,
    StrideH: IntVar,
    TokenStride: IntVar,
    FeatureStride: IntVar,
    BM1: IntVar,
    BN1: IntVar,
    BM2: IntVar,
    BN2: IntVar,
    SliceFactor: IntVar,
](
    Q: tl.Attention4DStridedInputPointer[
        Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
    ],
    K: tl.Attention4DStridedInputPointer[
        Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
    ],
    V: tl.Attention4DStridedInputPointer[
        Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
    ],
    sm_scale: float,  #
    DO: tl.Attention4DStridedInputPointer[
        Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
    ],  #
    DQ: tl.Attention4DStridedOutputPointer[
        Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
    ],
    DK: tl.Attention4DStridedOutputPointer[
        Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
    ],
    DV: tl.Attention4DStridedOutputPointer[
        Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
    ],  #
    M: tl.Attention3DStatsPointer[Batch, Heads, Tokens],
    D: tl.Attention3DStatsPointer[Batch, Heads, Tokens],
    # shared by Q/K/V/DO.
    stride_z: tl.AttentionBatchStride[Heads, StrideZ],
    stride_h: tl.AttentionHeadStride[Heads, StrideH],
    stride_tok: Int[TokenStride],
    stride_d: Int[FeatureStride],  #
    H: tl.AttentionHeadCount[Heads],
    N_CTX: Int[Tokens],  #
    BLOCK_M1: Int[BM1],  #
    BLOCK_N1: Int[BN1],  #
    BLOCK_M2: Int[BM2],  #
    BLOCK_N2: Int[BN2],  #
    BLK_SLICE_FACTOR: Int[SliceFactor],  #
    HEAD_DIM: Int[Dim],
    CAUSAL: bool,
    warp_specialize: bool = False,
):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)  # E: float

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1
    start_m = 0

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    if CAUSAL:
        start_m = start_n
        num_steps = BLOCK_N1 // MASK_BLOCK_M1
        dk, dv = _attn_bwd_dkdv(
            dk,
            dv,  #
            Q,
            k,
            v,
            sm_scale,  #
            DO,  #
            M,
            D,  #
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
        Q,
        k,
        v,
        sm_scale,  #
        DO,  #
        M,
        D,  #
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

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    start_n = 0
    num_steps = N_CTX // BLOCK_N2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(M + offs_m)
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
            K,
            V,  #
            do,
            m,
            D,  #
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
        K,
        V,  #
        do,
        m,
        D,  #
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
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)

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

# This static-only helper is checked by Pyrefly, not executed as a Buck test.
# @lint-ignore-every AUTODEPS2
# Preserve Triton's unused loop variable in the unchanged helper body.
# flake8: noqa: B007

"""Backward dQ tile checks with numbered attention's warp-specialized loop."""

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def _attn_bwd_dq[
    Heads: IntVar,
    Tokens: IntVar,
    Dim: IntVar,
    TokenStride: IntVar,
    FeatureStride: IntVar,
    BM: IntVar,
    BN: IntVar,
](
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
    BLOCK_M2: Int[BM],  #
    BLOCK_N2: Int[BN],  #
    HEAD_DIM: Int[Dim],
    # Filled in by the wrapper.
    start_m: int,
    start_n: int,
    num_steps: int,  #
    MASK: bool,
    warp_specialize: bool = False,
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


def test_wrong_softmax_rows[
    Heads: IntVar,
    Tokens: IntVar,
    Dim: IntVar,
    Other: IntVar,
    TokenStride: IntVar,
    FeatureStride: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    dq: tl.tensor[[BM, Dim]],
    q: tl.tensor[[BM, Dim]],
    k: tl.AttentionHeadLocalInputPointer[Tokens, Dim, TokenStride, FeatureStride],
    v: tl.AttentionHeadLocalInputPointer[Tokens, Dim, TokenStride, FeatureStride],
    do: tl.tensor[[BM, Dim]],
    m: tl.tensor[[Other, 1]],
    delta: tl.AttentionHeadLocalStatsPointer[Tokens],
    token_stride: Int[TokenStride],
    feature_stride: Int[FeatureStride],
    heads: Int[Heads],
    tokens: Int[Tokens],
    block_m: Int[BM],
    block_n: Int[BN],
    dim: Int[Dim],
) -> None:
    _attn_bwd_dq(
        dq,
        q,
        k,
        v,
        do,
        m,  # E: is not assignable
        delta,
        token_stride,
        feature_stride,
        heads,
        tokens,
        block_m,
        block_n,
        dim,
        0,
        0,
        1,
        False,
        True,
    )

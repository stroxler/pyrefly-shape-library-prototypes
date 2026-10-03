# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The kernel body is copied from Triton's 06-fused-attention.py.
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
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# This static-only helper is checked by Pyrefly, not executed as a Buck test.
# @lint-ignore-every AUTODEPS2
# Preserve Triton's unused loop variable in the unchanged helper body.
# flake8: noqa: B007

"""The dK/dV attention backward tile helper from Triton's tutorial 06."""

from typing import assert_type

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def _attn_bwd_dkdv[
    Heads: IntVar,
    Tokens: IntVar,
    Dim: IntVar,
    TokenStride: IntVar,
    FeatureStride: IntVar,
    BM: IntVar,
    BN: IntVar,
](
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
    BLOCK_M1: Int[BM],  #
    BLOCK_N1: Int[BN],  #
    HEAD_DIM: Int[Dim],  #
    start_n: int,
    start_m: int,
    num_steps: int,  #
    MASK: bool,
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
    for blk_idx in range(num_steps):
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


def test_wrong_q_and_do_address_axes[
    Tokens: IntVar,
    Dim: IntVar,
    TokenStride: IntVar,
    FeatureStride: IntVar,
    Other: IntVar,
    BM: IntVar,
](
    base: tl.AttentionHeadLocalInputPointer[Tokens, Dim, TokenStride, FeatureStride],
    q_columns: tl.AttentionTransposeColumns[
        Tokens, Dim, TokenStride, FeatureStride, BM
    ],
    do_rows: tl.AttentionForwardRows[Tokens, Dim, TokenStride, FeatureStride, BM],
    wrong_token_stride: tl.ColumnAddress[[BM], Other],
    wrong_feature_width: tl.RowAddress[[Other], FeatureStride],
    wrong_feature_axis: tl.ColumnAddress[[Dim], FeatureStride],
    wrong_do_axis: tl.RowAddress[[Dim], FeatureStride],
) -> None:
    base + wrong_token_stride  # E: No matching overload
    q_columns + wrong_feature_width  # E: is not assignable
    q_columns + wrong_feature_axis  # E: is not assignable
    do_rows + wrong_do_axis  # E: is not assignable


def test_wrong_pointer_step[
    Tokens: IntVar,
    Dim: IntVar,
    TokenStride: IntVar,
    Other: IntVar,
    BM: IntVar,
](
    q: tl.AttentionQTransposeTilePointers[Tokens, Dim, TokenStride, BM],
    do: tl.AttentionDOTilePointers[Tokens, Dim, TokenStride, BM],
    wrong_step: Int[Other * TokenStride],
) -> None:
    q.__iadd__(wrong_step)  # E: is not assignable
    do.__iadd__(wrong_step)  # E: is not assignable


def test_wrong_dot_and_causal_mask[
    BM: IntVar,
    BN: IntVar,
    Dim: IntVar,
    Other: IntVar,
](
    k: tl.tensor[[BN, Dim]],
    wrong_k: tl.tensor[[BN, Other]],
    qt: tl.tensor[[Dim, BM]],
    do: tl.tensor[[BM, Dim]],
    wrong_do: tl.tensor[[Other, Dim]],
    wrong_v: tl.tensor[[BN, Other]],
    pt: tl.tensor[[BN, BM]],
    wrong_causal_mask: tl.tensor[[BM, BN]],
) -> None:
    tl.dot(wrong_k, qt)  # E: is not assignable to parameter
    tl.dot(pt, wrong_do)  # E: is not assignable to parameter
    tl.dot(wrong_v, tl.trans(do))  # E: is not assignable to parameter
    tl.where(wrong_causal_mask, pt, 0.0)  # E: No matching overload
    assert_type(tl.dot(pt, do), tl.tensor[[BN, Dim]])
    assert_type(tl.dot(pt, tl.trans(qt)), tl.tensor[[BN, Dim]])


def test_wrong_boundary[
    Heads: IntVar,
    Tokens: IntVar,
    Dim: IntVar,
    TokenStride: IntVar,
    FeatureStride: IntVar,
    Other: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    dk: tl.tensor[[BN, Dim]],
    dv: tl.tensor[[BN, Dim]],
    wrong_dv: tl.tensor[[Other, Dim]],
    q: tl.AttentionHeadLocalInputPointer[Tokens, Dim, TokenStride, FeatureStride],
    do: tl.AttentionHeadLocalInputPointer[Tokens, Dim, TokenStride, FeatureStride],
    wrong_do: tl.AttentionHeadLocalInputPointer[Other, Dim, TokenStride, FeatureStride],
    k: tl.tensor[[BN, Dim]],
    v: tl.tensor[[BN, Dim]],
    m: tl.AttentionHeadLocalStatsPointer[Tokens],
    wrong_m: tl.AttentionHeadLocalStatsPointer[Other],
    delta: tl.AttentionHeadLocalStatsPointer[Tokens],
    wrong_delta: tl.AttentionHeadLocalStatsPointer[Other],
    token_stride: Int[TokenStride],
    feature_stride: Int[FeatureStride],
    heads: Int[Heads],
    tokens: Int[Tokens],
    bm: Int[BM],
    bn: Int[BN],
    dim: Int[Dim],
) -> None:
    _attn_bwd_dkdv(
        dk,
        wrong_dv,  # E: is not assignable to parameter
        q,
        k,
        v,
        1.0,
        do,
        m,
        delta,
        token_stride,
        feature_stride,
        heads,
        tokens,
        bm,
        bn,
        dim,
        0,
        0,
        1,
        False,
    )
    _attn_bwd_dkdv(
        dk,
        dv,
        q,
        k,
        v,
        1.0,
        wrong_do,  # E: is not assignable to parameter
        m,
        delta,
        token_stride,
        feature_stride,
        heads,
        tokens,
        bm,
        bn,
        dim,
        0,
        0,
        1,
        False,
    )
    _attn_bwd_dkdv(
        dk,
        dv,
        q,
        k,
        v,
        1.0,
        do,
        wrong_m,  # E: is not assignable to parameter
        delta,
        token_stride,
        feature_stride,
        heads,
        tokens,
        bm,
        bn,
        dim,
        0,
        0,
        1,
        False,
    )
    _attn_bwd_dkdv(
        dk,
        dv,
        q,
        k,
        v,
        1.0,
        do,
        m,
        wrong_delta,  # E: is not assignable to parameter
        token_stride,
        feature_stride,
        heads,
        tokens,
        bm,
        bn,
        dim,
        0,
        0,
        1,
        False,
    )


def test_unverified_stats_bounds_and_dtype[Tokens: IntVar, BM: IntVar](
    stats: tl.AttentionHeadLocalStatsPointer[Tokens],
    unrelated_indices: tl.Offsets[[BM]],
) -> None:
    # The helper has no N_CTX check or mask on these input loads.
    assert_type(
        tl.load(stats + unrelated_indices),
        tl.tensor[[BM]],
    )

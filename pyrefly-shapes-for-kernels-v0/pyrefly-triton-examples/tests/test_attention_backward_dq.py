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

"""The dQ attention backward tile helper from Triton's tutorial 06."""

from typing import assert_type

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
    for blk_idx in range(num_steps):
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


def test_wrong_k_and_v_address_axes[
    Tokens: IntVar,
    Dim: IntVar,
    TokenStride: IntVar,
    FeatureStride: IntVar,
    Other: IntVar,
    BN: IntVar,
](
    base: tl.AttentionHeadLocalInputPointer[Tokens, Dim, TokenStride, FeatureStride],
    columns: tl.AttentionTransposeColumns[Tokens, Dim, TokenStride, FeatureStride, BN],
    wrong_token_stride: tl.ColumnAddress[[BN], Other],
    wrong_feature_width: tl.RowAddress[[Other], FeatureStride],
    wrong_feature_axis: tl.ColumnAddress[[Dim], FeatureStride],
) -> None:
    base + wrong_token_stride  # E: No matching overload
    columns + wrong_feature_width  # E: is not assignable
    columns + wrong_feature_axis  # E: is not assignable


def test_wrong_pointer_step[
    Tokens: IntVar,
    Dim: IntVar,
    TokenStride: IntVar,
    Other: IntVar,
    BN: IntVar,
](
    kt: tl.AttentionQTransposeTilePointers[Tokens, Dim, TokenStride, BN],
    vt: tl.AttentionQTransposeTilePointers[Tokens, Dim, TokenStride, BN],
    wrong_step: Int[Other * TokenStride],
) -> None:
    kt.__iadd__(wrong_step)  # E: is not assignable
    vt.__iadd__(wrong_step)  # E: is not assignable


def test_wrong_dot_and_causal_mask[
    BM: IntVar,
    BN: IntVar,
    Dim: IntVar,
    Other: IntVar,
](
    q: tl.tensor[[BM, Dim]],
    wrong_q: tl.tensor[[BM, Other]],
    kt: tl.tensor[[Dim, BN]],
    wrong_vt: tl.tensor[[Other, BN]],
    ds: tl.tensor[[BM, BN]],
    wrong_causal_mask: tl.tensor[[BN, BM]],
) -> None:
    tl.dot(wrong_q, kt)  # E: is not assignable to parameter
    tl.dot(q, wrong_vt)  # E: is not assignable to parameter
    tl.dot(ds, q)  # E: is not assignable to parameter
    tl.where(wrong_causal_mask, ds, 0.0)  # E: No matching overload
    assert_type(tl.dot(q, kt), tl.tensor[[BM, BN]])
    assert_type(tl.dot(ds, tl.trans(kt)), tl.tensor[[BM, Dim]])


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
    dq: tl.tensor[[BM, Dim]],
    wrong_dq: tl.tensor[[Other, Dim]],
    q: tl.tensor[[BM, Dim]],
    wrong_q: tl.tensor[[BM, Other]],
    k: tl.AttentionHeadLocalInputPointer[Tokens, Dim, TokenStride, FeatureStride],
    wrong_k: tl.AttentionHeadLocalInputPointer[Other, Dim, TokenStride, FeatureStride],
    v: tl.AttentionHeadLocalInputPointer[Tokens, Dim, TokenStride, FeatureStride],
    wrong_v: tl.AttentionHeadLocalInputPointer[
        Tokens, Other, TokenStride, FeatureStride
    ],
    do: tl.tensor[[BM, Dim]],
    m: tl.tensor[[BM, 1]],
    wrong_m: tl.tensor[[Other, 1]],
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
    checked_dq: tl.tensor[[BM, Dim]] = wrong_dq  # E: is not assignable
    checked_k: tl.AttentionHeadLocalInputPointer[
        Tokens, Dim, TokenStride, FeatureStride
    ] = wrong_k  # E: is not assignable
    assert_type(checked_dq, tl.tensor[[BM, Dim]])
    assert_type(
        checked_k,
        tl.AttentionHeadLocalInputPointer[Tokens, Dim, TokenStride, FeatureStride],
    )
    _attn_bwd_dq(
        dq,
        wrong_q,  # E: is not assignable to parameter
        k,
        v,
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
    _attn_bwd_dq(
        dq,
        q,
        k,
        wrong_v,  # E: is not assignable to parameter
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
    _attn_bwd_dq(
        dq,
        q,
        k,
        v,
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
    _attn_bwd_dq(
        dq,
        q,
        k,
        v,
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


def test_unverified_stats_and_kv_bounds[
    Tokens: IntVar,
    Dim: IntVar,
    TokenStride: IntVar,
    BN: IntVar,
    BM: IntVar,
](
    k_tile: tl.AttentionQTransposeTilePointers[Tokens, Dim, TokenStride, BN],
    delta: tl.AttentionHeadLocalStatsPointer[Tokens],
    arbitrary_offsets: tl.Offsets[[BM]],
) -> None:
    # Accepted gaps: the pointer load checks tile geometry, not runtime bounds.
    tl.load(k_tile)
    tl.load(delta + arbitrary_offsets)

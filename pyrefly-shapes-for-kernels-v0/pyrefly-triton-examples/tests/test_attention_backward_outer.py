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

# Static-only, not an executable Triton kernel.
# @lint-ignore-every AUTODEPS2

"""Tutorial 06's outer attention backward kernel and host allocation roles."""

from typing import assert_type

import triton
import triton.language as tl
from shape_extensions import Int, IntVar
from tests.test_attention_backward_dkdv import _attn_bwd_dkdv
from tests.test_attention_backward_dq import _attn_bwd_dq


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
    )
    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


def test_only_axis_two_has_attention_batch_head_division[Heads: IntVar, Other: IntVar](
    heads: tl.AttentionHeadCount[Heads], ordinary_divisor: Int[Other]
) -> None:
    # Accepted gap: quotient head-count provenance is widened to int.
    assert_type(tl.program_id(2) // heads, tl.AttentionBatchIndex[int])
    assert_type(tl.program_id(0) // ordinary_divisor, int)


def test_wrong_batch_head_adjustment[
    Batch: IntVar,
    Heads: IntVar,
    Tokens: IntVar,
    Dim: IntVar,
    StrideZ: IntVar,
    StrideH: IntVar,
    TokenStride: IntVar,
    FeatureStride: IntVar,
    Other: IntVar,
](
    head_stride: tl.AttentionHeadStride[Heads, StrideH],
    batch_stride: tl.AttentionBatchStride[Heads, StrideZ],
    wrong_head_index: tl.GroupIndex[Other],
    wrong_batch_index: tl.AttentionBatchIndex[Other],
    q: tl.Attention4DStridedInputPointer[
        Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
    ],
    dq: tl.Attention4DStridedOutputPointer[
        Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
    ],
    m: tl.Attention3DStatsPointer[Batch, Heads, Tokens],
    wrong_head_stride: tl.AttentionBatchHeadOffset[Heads, StrideZ, Other],
    wrong_batch_stride: tl.AttentionBatchHeadOffset[Heads, Other, StrideH],
    wrong_head_count: tl.AttentionBatchHeadOffset[Other, StrideZ, StrideH],
    wrong_stats_token_extent: tl.TileStart[[Other]],
) -> None:
    head_stride * wrong_head_index  # E: is not assignable
    batch_stride * wrong_batch_index  # E: is not assignable
    q.__iadd__(wrong_head_stride)  # E: is not assignable
    q.__iadd__(wrong_batch_stride)  # E: is not assignable
    q.__iadd__(wrong_head_count)  # E: is not assignable
    dq.__iadd__(wrong_head_count)  # E: is not assignable
    m.__iadd__(wrong_stats_token_extent)  # E: is not assignable


def test_wrong_host_buffer_roles[
    Batch: IntVar,
    Heads: IntVar,
    Tokens: IntVar,
    Dim: IntVar,
    StrideZ: IntVar,
    StrideH: IntVar,
    TokenStride: IntVar,
    FeatureStride: IntVar,
    Other: IntVar,
](
    wrong_q_batch: tl.Attention4DStridedInputPointer[
        Other, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
    ],
    wrong_k_tokens: tl.Attention4DStridedInputPointer[
        Batch, Heads, Other, Dim, StrideZ, StrideH, TokenStride, FeatureStride
    ],
    wrong_v_feature: tl.Attention4DStridedInputPointer[
        Batch, Heads, Tokens, Other, StrideZ, StrideH, TokenStride, FeatureStride
    ],
    wrong_do_token_stride: tl.Attention4DStridedInputPointer[
        Batch, Heads, Tokens, Dim, StrideZ, StrideH, Other, FeatureStride
    ],
    wrong_dq_feature_stride: tl.Attention4DStridedOutputPointer[
        Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, Other
    ],
    wrong_dk_feature: tl.Attention4DStridedOutputPointer[
        Batch, Heads, Tokens, Other, StrideZ, StrideH, TokenStride, FeatureStride
    ],
    wrong_dv_batch_stride: tl.Attention4DStridedOutputPointer[
        Batch, Heads, Tokens, Dim, Other, StrideH, TokenStride, FeatureStride
    ],
    wrong_m_tokens: tl.Attention3DStatsPointer[Batch, Heads, Other],
    wrong_delta_heads: tl.Attention3DStatsPointer[Batch, Other, Tokens],
) -> None:
    q: tl.Attention4DStridedInputPointer[
        Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
    ] = wrong_q_batch  # E: is not assignable
    k: tl.Attention4DStridedInputPointer[
        Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
    ] = wrong_k_tokens  # E: is not assignable
    v: tl.Attention4DStridedInputPointer[
        Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
    ] = wrong_v_feature  # E: is not assignable
    do: tl.Attention4DStridedInputPointer[
        Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
    ] = wrong_do_token_stride  # E: is not assignable
    dq: tl.Attention4DStridedOutputPointer[
        Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
    ] = wrong_dq_feature_stride  # E: is not assignable
    dk: tl.Attention4DStridedOutputPointer[
        Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
    ] = wrong_dk_feature  # E: is not assignable
    dv: tl.Attention4DStridedOutputPointer[
        Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
    ] = wrong_dv_batch_stride  # E: is not assignable
    m: tl.Attention3DStatsPointer[Batch, Heads, Tokens] = (
        wrong_m_tokens  # E: is not assignable
    )
    delta: tl.Attention3DStatsPointer[Batch, Heads, Tokens] = (
        wrong_delta_heads  # E: is not assignable
    )
    assert_type(
        q,
        tl.Attention4DStridedInputPointer[
            Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
        ],
    )
    assert_type(
        k,
        tl.Attention4DStridedInputPointer[
            Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
        ],
    )
    assert_type(
        v,
        tl.Attention4DStridedInputPointer[
            Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
        ],
    )
    assert_type(
        do,
        tl.Attention4DStridedInputPointer[
            Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
        ],
    )
    assert_type(
        dq,
        tl.Attention4DStridedOutputPointer[
            Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
        ],
    )
    assert_type(
        dk,
        tl.Attention4DStridedOutputPointer[
            Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
        ],
    )
    assert_type(
        dv,
        tl.Attention4DStridedOutputPointer[
            Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
        ],
    )
    assert_type(m, tl.Attention3DStatsPointer[Batch, Heads, Tokens])
    assert_type(delta, tl.Attention3DStatsPointer[Batch, Heads, Tokens])


def test_wrong_gradient_store_tile[
    Tokens: IntVar,
    Dim: IntVar,
    BM: IntVar,
    Other: IntVar,
](
    gradient: tl.AttentionOutputTilePointers[Tokens, Dim, BM],
    wrong_rows: tl.tensor[[Other, Dim]],
    wrong_feature_width: tl.tensor[[BM, Other]],
) -> None:
    tl.store(gradient, wrong_rows)  # E: No matching overload
    tl.store(gradient, wrong_feature_width)  # E: No matching overload


def test_wrong_gradient_address_axes[
    Tokens: IntVar,
    Dim: IntVar,
    TokenStride: IntVar,
    FeatureStride: IntVar,
    BM: IntVar,
    Other: IntVar,
](
    gradient: tl.AttentionHeadLocalOutputPointer[
        Tokens, Dim, TokenStride, FeatureStride
    ],
    rows: tl.AttentionOutputRows[Tokens, Dim, FeatureStride, BM],
    wrong_token_stride: tl.RowAddress[[BM], Other],
    wrong_feature_width: tl.ColumnAddress[[Other], FeatureStride],
    wrong_feature_axis: tl.RowAddress[[Dim], FeatureStride],
) -> None:
    gradient + wrong_token_stride  # E: is not assignable
    rows + wrong_feature_width  # E: is not assignable
    rows + wrong_feature_axis  # E: is not assignable


def test_unchecked_order_and_runtime_bounds[
    Batch: IntVar,
    Heads: IntVar,
    Tokens: IntVar,
    Dim: IntVar,
    StrideZ: IntVar,
    StrideH: IntVar,
    TokenStride: IntVar,
    FeatureStride: IntVar,
    BM: IntVar,
](
    q_unadjusted: tl.Attention4DStridedInputPointer[
        Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
    ],
    dq_unadjusted: tl.Attention4DStridedOutputPointer[
        Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
    ],
    arbitrary_rows: tl.RowAddress[[BM], TokenStride],
    arbitrary_cols: tl.ColumnAddress[[Dim], FeatureStride],
    arbitrary_values: tl.tensor[[BM, Dim]],
) -> None:
    # Accepted gap: subtype relationships allow accesses before batch/head adjustment.
    tl.load(q_unadjusted + arbitrary_rows + arbitrary_cols)
    tl.store(dq_unadjusted + arbitrary_rows + arbitrary_cols, arbitrary_values)

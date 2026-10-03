# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The helper body is copied from Triton's fused-attention-ws-device-tma.py.
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

# Static-only parameter annotations are not accepted by Triton's runtime.
# @lint-ignore-every AUTODEPS2

"""Tile and descriptor contracts for device-TMA warp-specialized attention."""

from typing import assert_type, Literal

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


# The sibling test_attention_ws_subtile.py checks the subtile body separately.
@triton.jit
def _attn_fwd_subtile[BM: IntVar, BN: IntVar, D: IntVar](
    q: tl.tensor[[BM, D]],
    k: tl.tensor[[D, BN]],
    offs_m: tl.Offsets[[BM]],
    start_n: int,
    offs_n: tl.Offsets[[BN]],
    qk_scale: float,
    l_i0: tl.tensor[[BM]],
    l_i1: tl.tensor[[BM]] | Literal[0],
    m_i: tl.tensor[[BM]],
    acc: tl.tensor[[BM, D]],
    v: tl.tensor[[BN, D]],
    dtype: object,
    STAGE: int,
    SUBTILING: bool,
    VECT_MUL: int,
    FADD2_REDUCE: bool,
) -> tuple[
    tl.tensor[[BM]],
    tl.tensor[[BM]] | Literal[0],
    tl.tensor[[BM]],
    tl.tensor[[BM, D]],
]: ...


# Triton TR001: The outer launcher owns autotuning; this inner helper is static-only.
@triton.jit
def _attn_fwd_inner_oss_dp[  # noqa: TR001
    BM: IntVar,
    BN: IntVar,
    D: IntVar,
    Rows: IntVar,
    NCtx: IntVar,
](
    acc0: tl.tensor[[BM, D]],
    l_i0: tl.tensor[[BM]],
    l_i0_1: tl.tensor[[BM]] | Literal[0],
    m_i0: tl.tensor[[BM]],
    q0: tl.tensor[[BM, D]],
    desc_k: tl.InputMatrixDescriptor[Rows, D, D, BN, D],
    desc_v: tl.InputMatrixDescriptor[Rows, D, D, BN, D],  #
    offset_y: int,
    dtype: object,
    start_m: int | tl.ProgramId,
    qk_scale: float,  #
    BLOCK_M: Int[BM],
    HEAD_DIM: Int[D],
    BLOCK_N: Int[BN],  #
    STAGE: int,
    offs_m0: tl.Offsets[[BM]],
    offs_n: tl.Offsets[[BN]],  #
    N_CTX: Int[NCtx],
    warp_specialize: bool,
    SUBTILING: bool,
    VECT_MUL: int,
    FADD2_REDUCE: bool,
    DP_FACTOR: Literal[2],
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
    offsetkv_y = offset_y + lo

    # loop over k, v and update accumulator
    for start_n in tl.range(
        lo,
        hi,
        BLOCK_N,
        warp_specialize=warp_specialize,
        disallow_acc_multi_buffer=True,
        data_partition_factor=DP_FACTOR,
    ):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k = desc_k.load([offsetkv_y, 0]).T
        v = desc_v.load([offsetkv_y, 0])

        l_i0, l_i0_1, m_i0, acc0 = _attn_fwd_subtile(
            q0,
            k,
            offs_m0,
            start_n,
            offs_n,
            qk_scale,
            l_i0,
            l_i0_1,
            m_i0,
            acc0,
            v,
            dtype,
            STAGE,
            SUBTILING,
            VECT_MUL,
            FADD2_REDUCE,
        )

        offsetkv_y += BLOCK_N

    return acc0, l_i0, l_i0_1, m_i0


def test_descriptor_tile_shapes[
    Rows: IntVar,
    D: IntVar,
    BN: IntVar,
    Other: IntVar,
](
    k: tl.InputMatrixDescriptor[Rows, D, D, BN, D],
    v: tl.InputMatrixDescriptor[Rows, D, D, BN, D],
    wrong_k: tl.InputMatrixDescriptor[Rows, D, D, Other, D],
    wrong_v: tl.InputMatrixDescriptor[Rows, D, D, BN, Other],
) -> None:
    assert_type(k.load([0, 0]).T, tl.tensor[[D, BN]])
    assert_type(v.load([0, 0]), tl.tensor[[BN, D]])
    assert_type(wrong_k.load([0, 0]).T, tl.tensor[[D, Other]])
    assert_type(wrong_v.load([0, 0]), tl.tensor[[BN, Other]])


def test_inner_helper_boundary[
    BM: IntVar,
    BN: IntVar,
    D: IntVar,
    Rows: IntVar,
    NCtx: IntVar,
    Other: IntVar,
](
    accumulator: tl.tensor[[BM, D]],
    statistics: tl.tensor[[BM]],
    q: tl.tensor[[BM, D]],
    k: tl.InputMatrixDescriptor[Rows, D, D, BN, D],
    v: tl.InputMatrixDescriptor[Rows, D, D, BN, D],
    wrong_k_tile: tl.InputMatrixDescriptor[Rows, D, D, Other, D],
    wrong_v_rows: tl.InputMatrixDescriptor[Other, D, D, BN, D],
    wrong_q: tl.tensor[[BM, Other]],
    program: tl.ProgramId,
    offsets_m: tl.Offsets[[BM]],
    offsets_n: tl.Offsets[[BN]],
    bm: Int[BM],
    bn: Int[BN],
    dim: Int[D],
    n_ctx: Int[NCtx],
) -> None:
    assert_type(
        _attn_fwd_inner_oss_dp(
            accumulator,
            statistics,
            statistics,
            statistics,
            q,
            k,
            v,
            0,
            tl.float16,
            program,
            1.0,
            bm,
            dim,
            bn,
            2,
            offsets_m,
            offsets_n,
            n_ctx,
            True,
            False,
            0,
            False,
            2,
        ),
        tuple[
            tl.tensor[[BM, D]],
            tl.tensor[[BM]],
            tl.tensor[[BM]] | Literal[0],
            tl.tensor[[BM]],
        ],
    )
    _attn_fwd_inner_oss_dp(
        accumulator,
        statistics,
        statistics,
        statistics,
        wrong_q,  # E: is not assignable
        k,
        wrong_v_rows,  # E: is not assignable
        0,
        tl.float16,
        program,
        1.0,
        bm,
        dim,
        bn,
        2,
        offsets_m,
        offsets_n,
        n_ctx,
        True,
        False,
        0,
        False,
        2,
    )
    _attn_fwd_inner_oss_dp(
        accumulator,
        statistics,
        statistics,
        statistics,
        q,
        wrong_k_tile,
        v,  # E: is not assignable
        0,
        tl.float16,
        program,
        1.0,
        bm,
        dim,
        bn,  # E: is not assignable
        2,
        offsets_m,
        offsets_n,  # E: is not assignable
        n_ctx,
        True,
        False,
        0,
        False,
        2,
    )


def test_outer_descriptor_and_sentinel_link[
    Batch: IntVar,
    Heads: IntVar,
    Tokens: IntVar,
    BM: IntVar,
    BN: IntVar,
    D: IntVar,
](
    accumulator: tl.tensor[[BM, D]],
    statistics: tl.tensor[[BM]],
    query: tl.tensor[[BM, D]],
    k: tl.InputMatrixDescriptor[Batch * Heads * Tokens, D, D, BN, D],
    v: tl.InputMatrixDescriptor[Batch * Heads * Tokens, D, D, BN, D],
    program: tl.ProgramId,
    rows: tl.Offsets[[BM]],
    columns: tl.Offsets[[BN]],
    bm: Int[BM],
    bn: Int[BN],
    d: Int[D],
    n_ctx: Int[Tokens],
) -> None:
    # This is the outer device-TMA fixture's descriptor type and false FADD2 path.
    assert_type(
        _attn_fwd_inner_oss_dp(
            accumulator,
            statistics,
            0,
            statistics,
            query,
            k,
            v,
            0,
            tl.float16,
            program,
            1.0,
            bm,
            d,
            bn,
            1,
            rows,
            columns,
            n_ctx,
            True,
            False,
            0,
            False,
            2,
        ),
        tuple[
            tl.tensor[[BM, D]],
            tl.tensor[[BM]],
            tl.tensor[[BM]] | Literal[0],
            tl.tensor[[BM]],
        ],
    )


def test_subtile_rejects_swapped_contracts[
    BM: IntVar,
    BN: IntVar,
    D: IntVar,
    Other: IntVar,
](
    q: tl.tensor[[BM, D]],
    k: tl.tensor[[D, BN]],
    bad_k: tl.tensor[[Other, BN]],
    v: tl.tensor[[BN, D]],
    bad_v: tl.tensor[[BN, Other]],
    offsets_m: tl.Offsets[[BM]],
    bad_offsets_m: tl.Offsets[[Other]],
    offsets_n: tl.Offsets[[BN]],
    statistics: tl.tensor[[BM]],
    bad_statistics: tl.tensor[[Other]],
    acc: tl.tensor[[BM, D]],
    bad_acc: tl.tensor[[Other, D]],
) -> None:
    assert_type(
        _attn_fwd_subtile(
            q,
            k,
            offsets_m,
            0,
            offsets_n,
            1.0,
            statistics,
            statistics,
            statistics,
            acc,
            v,
            tl.float16,
            2,
            False,
            0,
            False,
        ),
        tuple[
            tl.tensor[[BM]],
            tl.tensor[[BM]] | Literal[0],
            tl.tensor[[BM]],
            tl.tensor[[BM, D]],
        ],
    )
    _attn_fwd_subtile(
        q,
        k,
        offsets_m,
        0,
        offsets_n,
        1.0,
        statistics,
        bad_statistics,  # E: is not assignable
        statistics,
        acc,
        v,
        tl.float16,
        2,
        False,
        0,
        True,
    )
    _attn_fwd_subtile(
        q,
        bad_k,  # E: is not assignable
        bad_offsets_m,  # E: is not assignable
        0,
        offsets_n,
        1.0,
        bad_statistics,  # E: is not assignable
        statistics,
        statistics,
        bad_acc,  # E: is not assignable
        bad_v,  # E: is not assignable
        tl.float16,
        2,
        False,
        0,
        False,
    )


def test_nonpartitioned_range[DP: IntVar](factor: Int[DP]) -> None:
    tl.range(
        0,
        256,
        128,
        warp_specialize=True,
        disallow_acc_multi_buffer=True,
        data_partition_factor=factor,  # E: is not assignable
    )

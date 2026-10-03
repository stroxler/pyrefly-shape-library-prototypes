# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The kernel body is copied from Triton's fused-attention-ws.py.
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

"""Descriptor and statistics output geometry in warp-specialized attention."""

from typing import Literal

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


def _attn_fwd_inner_oss_dp[Half: IntVar, BN: IntVar, Dim: IntVar, Flat: IntVar](
    acc0: tl.tensor[[Half, Dim]],
    acc1: tl.tensor[[Half, Dim]],
    l_i0: tl.tensor[[Half]],
    l_i0_1: tl.tensor[[Half]] | Literal[0],
    l_i1: tl.tensor[[Half]],
    l_i1_1: tl.tensor[[Half]] | Literal[0],
    m_i0: tl.tensor[[Half]],
    m_i1: tl.tensor[[Half]],
    q0: tl.tensor[[Half, Dim]],
    q1: tl.tensor[[Half, Dim]],
    desc_k: tl.InputMatrixDescriptor[Flat, Dim, Dim, BN, Dim],
    desc_v: tl.InputMatrixDescriptor[Flat, Dim, Dim, BN, Dim],
    offset_y: int,
    dtype: object,
    start_m: tl.ProgramId,
    qk_scale: float,
    BLOCK_M: Int[Half * 2],
    HEAD_DIM: Int[Dim],
    BLOCK_N: Int[BN],
    STAGE: int,
    offs_m0: tl.Offsets[[Half]],
    offs_m1: tl.Offsets[[Half]],
    offs_n: tl.Offsets[[BN]],
    N_CTX: int,
    warp_specialize: bool,
    SUBTILING: bool,
    VECT_MUL: int,
    FADD2_REDUCE: bool,
) -> tuple[
    tl.tensor[[Half, Dim]],
    tl.tensor[[Half, Dim]],
    tl.tensor[[Half]],
    tl.tensor[[Half]] | Literal[0],
    tl.tensor[[Half]],
    tl.tensor[[Half]] | Literal[0],
    tl.tensor[[Half]],
    tl.tensor[[Half]],
]: ...


@triton.jit
def _attn_fwd_tma_dp[
    Batch: IntVar,
    Heads: IntVar,
    Tokens: IntVar,
    Half: IntVar,
    BN: IntVar,
    Dim: IntVar,
](
    sm_scale: float,
    M: tl.Attention3DStatsPointer[Batch, Heads, Tokens],  #
    Z: Int[Batch],
    H: tl.AttentionHeadCount[Heads],
    desc_q: tl.InputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, Half, Dim],
    desc_k: tl.InputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, BN, Dim],
    desc_v: tl.InputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, BN, Dim],
    desc_o: tl.OutputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, Half, Dim],
    pid: tl.ProgramId,
    off_hz: tl.ProgramId,
    N_CTX: Int[Tokens],  #
    HEAD_DIM: Int[Dim],  #
    BLOCK_M: Int[Half * 2],  #
    BLOCK_N: Int[BN],  #
    FP8_OUTPUT: bool,  #
    STAGE: int,  #
    warp_specialize: bool,  #
    dtype: object,
    SUBTILING: bool,
    VECT_MUL: int,
    FADD2_REDUCE: bool,
):
    start_m = pid  # tl.program_id(0)
    # off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    # initialize offsets
    offs_m0 = start_m * BLOCK_M + tl.arange(0, BLOCK_M // 2)
    offs_m1 = start_m * BLOCK_M + tl.arange(BLOCK_M // 2, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    m_i0 = tl.zeros([BLOCK_M // 2], dtype=tl.float32) - float("inf")
    l_i0_0 = tl.zeros([BLOCK_M // 2], dtype=tl.float32) + 1.0
    acc0 = tl.zeros([BLOCK_M // 2, HEAD_DIM], dtype=tl.float32)

    m_i1 = tl.zeros([BLOCK_M // 2], dtype=tl.float32) - float("inf")
    l_i1_0 = tl.zeros([BLOCK_M // 2], dtype=tl.float32) + 1.0
    acc1 = tl.zeros([BLOCK_M // 2, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    q0 = desc_q.load([qo_offset_y, 0])
    q1 = desc_q.load([qo_offset_y + BLOCK_M // 2, 0])

    if FADD2_REDUCE:
        l_i0_1 = tl.zeros([BLOCK_M // 2], dtype=tl.float32)
        l_i1_1 = tl.zeros([BLOCK_M // 2], dtype=tl.float32)
    else:
        l_i0_1 = 0
        l_i1_1 = 0

    if STAGE & 1:
        acc0, acc1, l_i0_0, l_i0_1, l_i1_0, l_i1_1, m_i0, m_i1 = _attn_fwd_inner_oss_dp(
            acc0,
            acc1,
            l_i0_0,
            l_i0_1,
            l_i1_0,
            l_i1_1,
            m_i0,
            m_i1,
            q0,
            q1,  #
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
            offs_m0,
            offs_m1,
            offs_n,
            N_CTX,  #
            warp_specialize,
            SUBTILING,
            VECT_MUL,
            FADD2_REDUCE,
        )
    if STAGE & 2:
        acc0, acc1, l_i0_0, l_i0_1, l_i1_0, l_i1_1, m_i0, m_i1 = _attn_fwd_inner_oss_dp(
            acc0,
            acc1,
            l_i0_0,
            l_i0_1,
            l_i1_0,
            l_i1_1,
            m_i0,
            m_i1,
            q0,
            q1,  #
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
            offs_m0,
            offs_m1,
            offs_n,
            N_CTX,  #
            warp_specialize,
            SUBTILING,
            VECT_MUL,
            FADD2_REDUCE,
        )

    if FADD2_REDUCE:
        l_i0 = l_i0_0 + l_i0_1
        l_i1 = l_i1_0 + l_i1_1
    else:
        l_i0 = l_i0_0
        l_i1 = l_i1_0

    m_i0 += tl.math.log2(l_i0)
    acc0 = acc0 / l_i0[:, None]
    m_ptrs0 = M + off_hz * N_CTX + offs_m0
    tl.store(m_ptrs0, m_i0)
    desc_o.store([qo_offset_y, 0], acc0.to(dtype))

    m_i1 += tl.math.log2(l_i1)
    acc1 = acc1 / l_i1[:, None]
    m_ptrs1 = M + off_hz * N_CTX + offs_m1
    tl.store(m_ptrs1, m_i1)
    desc_o.store([qo_offset_y + BLOCK_M // 2, 0], acc1.to(dtype))


def test_wrong_output_descriptor[
    Batch: IntVar,
    Heads: IntVar,
    Tokens: IntVar,
    Half: IntVar,
    BN: IntVar,
    Dim: IntVar,
    Other: IntVar,
](
    stats: tl.Attention3DStatsPointer[Batch, Heads, Tokens],
    batch: Int[Batch],
    heads: tl.AttentionHeadCount[Heads],
    q: tl.InputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, Half, Dim],
    k: tl.InputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, BN, Dim],
    v: tl.InputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, BN, Dim],
    wrong_o: tl.OutputMatrixDescriptor[Batch * Heads * Tokens, Other, Dim, Half, Dim],
    pid: tl.ProgramId,
    hz: tl.ProgramId,
    tokens: Int[Tokens],
    dim: Int[Dim],
    block_m: Int[Half * 2],
    block_n: Int[BN],
) -> None:
    _attn_fwd_tma_dp(
        1.0,
        stats,
        batch,
        heads,
        q,
        k,
        v,
        wrong_o,  # E: is not assignable
        pid,
        hz,
        tokens,
        dim,
        block_m,
        block_n,
        False,
        1,
        False,
        tl.float32,
        False,
        0,
        False,
    )


def test_consistent_contract[
    Batch: IntVar,
    Heads: IntVar,
    Tokens: IntVar,
    Half: IntVar,
    BN: IntVar,
    Dim: IntVar,
](
    stats: tl.Attention3DStatsPointer[Batch, Heads, Tokens],
    batch: Int[Batch],
    heads: tl.AttentionHeadCount[Heads],
    q: tl.InputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, Half, Dim],
    k: tl.InputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, BN, Dim],
    v: tl.InputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, BN, Dim],
    o: tl.OutputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, Half, Dim],
    pid: tl.ProgramId,
    hz: tl.ProgramId,
    tokens: Int[Tokens],
    dim: Int[Dim],
    block_m: Int[Half * 2],
    block_n: Int[BN],
) -> None:
    _attn_fwd_tma_dp(
        1.0,
        stats,
        batch,
        heads,
        q,
        k,
        v,
        o,
        pid,
        hz,
        tokens,
        dim,
        block_m,
        block_n,
        False,
        1,
        False,
        tl.float32,
        False,
        0,
        False,
    )


def test_wrong_statistics_extent[
    Batch: IntVar,
    Heads: IntVar,
    Tokens: IntVar,
    Other: IntVar,
    Half: IntVar,
    BN: IntVar,
    Dim: IntVar,
](
    wrong_stats: tl.Attention3DStatsPointer[Batch, Heads, Other],
    batch: Int[Batch],
    heads: tl.AttentionHeadCount[Heads],
    q: tl.InputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, Half, Dim],
    k: tl.InputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, BN, Dim],
    v: tl.InputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, BN, Dim],
    o: tl.OutputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, Half, Dim],
    pid: tl.ProgramId,
    hz: tl.ProgramId,
    tokens: Int[Tokens],
    dim: Int[Dim],
    block_m: Int[Half * 2],
    block_n: Int[BN],
) -> None:
    _attn_fwd_tma_dp(
        1.0,
        wrong_stats,
        batch,
        heads,
        q,  # E: is not assignable
        k,  # E: is not assignable
        v,  # E: is not assignable
        o,  # E: is not assignable
        pid,
        hz,
        tokens,  # E: is not assignable
        dim,
        block_m,  # E: is not assignable
        block_n,
        False,
        1,
        False,
        tl.float32,
        False,
        0,
        False,
    )

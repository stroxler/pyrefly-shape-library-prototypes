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

"""Descriptor-to-subtile shape propagation in warp-specialized attention."""

from typing import Literal

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


def _attn_fwd_subtile[Half: IntVar, BN: IntVar, Dim: IntVar](
    q: tl.tensor[[Half, Dim]],
    k: tl.tensor[[Dim, BN]],
    offs_m: tl.Offsets[[Half]],
    start_n: int,
    offs_n: tl.Offsets[[BN]],
    qk_scale: float,
    l_i0: tl.tensor[[Half]],
    l_i1: tl.tensor[[Half]] | Literal[0],
    m_i: tl.tensor[[Half]],
    acc: tl.tensor[[Half, Dim]],
    v: tl.tensor[[BN, Dim]],
    dtype: object,
    STAGE: int,
    SUBTILING: bool,
    VECT_MUL: int,
    FADD2_REDUCE: bool,
) -> tuple[
    tl.tensor[[Half]],
    tl.tensor[[Half]] | Literal[0],
    tl.tensor[[Half]],
    tl.tensor[[Half, Dim]],
]: ...


@triton.jit
def _attn_fwd_inner_oss_dp[
    Half: IntVar,
    BN: IntVar,
    Dim: IntVar,
    Tokens: IntVar,
    Stride: IntVar,
](
    acc0: tl.tensor[[Half, Dim]],
    acc1: tl.tensor[[Half, Dim]],
    l_i0: tl.tensor[[Half]],
    l_i0_1: tl.tensor[[Half]] | Literal[0],
    l_i1: tl.tensor[[Half]],
    l_i1_1: tl.tensor[[Half]] | Literal[0],
    m_i0: tl.tensor[[Half]],
    m_i1: tl.tensor[[Half]],
    q0: tl.tensor[[Half, Dim]],
    q1: tl.tensor[[Half, Dim]],  #
    desc_k: tl.InputMatrixDescriptor[Tokens, Dim, Stride, BN, Dim],
    desc_v: tl.InputMatrixDescriptor[Tokens, Dim, Stride, BN, Dim],  #
    offset_y: int,
    dtype: object,
    start_m: int | tl.ProgramId,
    qk_scale: float,  #
    BLOCK_M: Int[Half * 2],
    HEAD_DIM: Int[Dim],
    BLOCK_N: Int[BN],  #
    STAGE: int,
    offs_m0: tl.Offsets[[Half]],
    offs_m1: tl.Offsets[[Half]],  #
    offs_n: tl.Offsets[[BN]],  #
    N_CTX: Int[Tokens],
    warp_specialize: bool,
    SUBTILING: bool,
    VECT_MUL: int,
    FADD2_REDUCE: bool,
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
        lo, hi, BLOCK_N, warp_specialize=warp_specialize, disallow_acc_multi_buffer=True
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
        l_i1, l_i1_1, m_i1, acc1 = _attn_fwd_subtile(
            q1,
            k,
            offs_m1,
            start_n,
            offs_n,
            qk_scale,
            l_i1,
            l_i1_1,
            m_i1,
            acc1,
            v,
            dtype,
            STAGE,
            SUBTILING,
            VECT_MUL,
            FADD2_REDUCE,
        )

        offsetkv_y += BLOCK_N

    return acc0, acc1, l_i0, l_i0_1, l_i1, l_i1_1, m_i0, m_i1


def test_wrong_descriptor_block[
    Half: IntVar,
    BN: IntVar,
    Other: IntVar,
    Dim: IntVar,
    Tokens: IntVar,
](
    acc: tl.tensor[[Half, Dim]],
    stats: tl.tensor[[Half]],
    q: tl.tensor[[Half, Dim]],
    k: tl.InputMatrixDescriptor[Tokens, Dim, Dim, BN, Other],
    v: tl.InputMatrixDescriptor[Tokens, Dim, Dim, BN, Dim],
    rows: tl.Offsets[[Half]],
    cols: tl.Offsets[[BN]],
    block_m: Int[Half * 2],
    dim: Int[Dim],
    block_n: Int[BN],
    tokens: Int[Tokens],
) -> None:
    _attn_fwd_inner_oss_dp(
        acc,
        acc,
        stats,
        stats,
        stats,
        stats,
        stats,
        stats,
        q,
        q,
        k,  # E: is not assignable
        v,
        0,
        tl.float32,
        0,
        1.0,
        block_m,
        dim,
        block_n,
        1,
        rows,
        rows,
        cols,
        tokens,
        False,
        False,
        0,
        False,
    )

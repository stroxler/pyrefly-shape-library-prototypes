# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The helper body is copied from Triton's fused-attention-ws-device-tma-hopper-or-blackwell.py.
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

# Static-only parameter annotations are not accepted by Triton's runtime.
# @lint-ignore-every AUTODEPS2

"""Descriptor and attribute forwarding in device-TMA warp-specialized attention."""

from typing import Literal

import triton
import triton.language as tl
from shape_extensions import Int, IntVar
from tests.test_attention_ws_device_subtile_attrs import _attn_fwd_subtile, DotAttrs


@triton.jit
def _attn_fwd_inner_oss_dp[
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
    FWD_DOT_ATTRS: DotAttrs | None = None,
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
        merge_epilogue=True,
        merge_correction=True,
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
            l_i0_1,  # E: is not assignable
            m_i0,
            acc0,
            v,
            dtype,
            STAGE,
            SUBTILING,
            VECT_MUL,
            FADD2_REDUCE,
            FWD_DOT_ATTRS,
        )

        offsetkv_y += BLOCK_N

    return acc0, l_i0, l_i0_1, m_i0


def test_wrong_key_head_block[
    BM: IntVar,
    BN: IntVar,
    D: IntVar,
    Wrong: IntVar,
    Rows: IntVar,
    NCtx: IntVar,
](
    acc: tl.tensor[[BM, D]],
    row: tl.tensor[[BM]],
    q: tl.tensor[[BM, D]],
    wrong_k: tl.InputMatrixDescriptor[Rows, D, D, BN, Wrong],
    v: tl.InputMatrixDescriptor[Rows, D, D, BN, D],
    offs_m: tl.Offsets[[BM]],
    offs_n: tl.Offsets[[BN]],
    block_m: Int[BM],
    block_n: Int[BN],
    dim: Int[D],
    n_ctx: Int[NCtx],
    attrs: DotAttrs,
) -> None:
    _attn_fwd_inner_oss_dp(
        acc,
        row,
        row,
        row,
        q,
        wrong_k,  # E: is not assignable
        v,
        0,
        tl.float16,
        0,
        1.0,
        block_m,
        dim,
        block_n,
        3,
        offs_m,
        offs_n,
        n_ctx,
        False,
        False,
        0,
        False,
        2,
        attrs,
    )

# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The kernel body is copied from Triton's fused-attention-ws-device-tma-hopper-or-blackwell.py.
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

# Static-only; semantic annotations are not accepted by Triton's runtime.
# @lint-ignore-every AUTODEPS2

"""Full device-TMA attention output and statistics contract with dot attributes."""

from typing import Literal

import triton
import triton.language as tl
from shape_extensions import Int, IntVar
from tests.test_attention_ws_device_inner_attrs import _attn_fwd_inner_oss_dp
from tests.test_attention_ws_device_subtile_attrs import DotAttrs


@triton.jit
def _attn_fwd_tma_dp[
    Batch: IntVar,
    Heads: IntVar,
    Tokens: IntVar,
    BM: IntVar,
    BN: IntVar,
    Dim: IntVar,
](
    sm_scale: float,
    M: tl.Attention3DStatsPointer[Batch, Heads, Tokens],  #
    Z: Int[Batch],
    H: tl.AttentionHeadCount[Heads],
    desc_q: tl.InputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, BM, Dim],
    desc_k: tl.InputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, BN, Dim],
    desc_v: tl.InputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, BN, Dim],
    desc_o: tl.OutputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, BM, Dim],
    pid: tl.ProgramId,
    off_hz: tl.ProgramId,
    N_CTX: Int[Tokens],  #
    HEAD_DIM: Int[Dim],  #
    BLOCK_M: Int[BM],  #
    BLOCK_N: Int[BN],  #
    FP8_OUTPUT: bool,  #
    STAGE: int,  #
    warp_specialize: bool,  #
    dtype: object,
    SUBTILING: bool,
    VECT_MUL: int,
    FADD2_REDUCE: bool,
    DP_FACTOR: Literal[2],
    FWD_DOT_ATTRS: DotAttrs | None = None,
):
    start_m = pid  # tl.program_id(0)
    # off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    # initialize offsets
    offs_m0 = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    m_i0 = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i0_0 = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc0 = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    q0 = desc_q.load([qo_offset_y, 0])

    if FADD2_REDUCE:
        l_i0_1 = tl.zeros([BLOCK_M // 2], dtype=tl.float32)
    else:
        l_i0_1 = 0

    if STAGE & 1:
        acc0, l_i0_0, l_i0_1, m_i0 = _attn_fwd_inner_oss_dp(
            acc0,
            l_i0_0,
            l_i0_1,  # E: is not assignable
            m_i0,
            q0,
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
            offs_n,
            N_CTX,  #
            warp_specialize,
            SUBTILING,
            VECT_MUL,
            FADD2_REDUCE,
            DP_FACTOR,
            FWD_DOT_ATTRS,
        )
    if STAGE & 2:
        acc0, l_i0_0, l_i0_1, m_i0 = _attn_fwd_inner_oss_dp(
            acc0,
            l_i0_0,
            l_i0_1,  # E: is not assignable
            m_i0,
            q0,
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
            offs_n,
            N_CTX,  #
            warp_specialize,
            SUBTILING,
            VECT_MUL,
            FADD2_REDUCE,
            DP_FACTOR,
            FWD_DOT_ATTRS,
        )

    if FADD2_REDUCE:
        l_i0 = l_i0_0 + l_i0_1  # E: not supported
    else:
        l_i0 = l_i0_0

    m_i0 += tl.math.log2(l_i0)
    acc0 = acc0 / l_i0[:, None]
    m_ptrs0 = M + off_hz * N_CTX + offs_m0
    tl.store(m_ptrs0, m_i0)
    desc_o.store([qo_offset_y, 0], acc0.to(dtype))

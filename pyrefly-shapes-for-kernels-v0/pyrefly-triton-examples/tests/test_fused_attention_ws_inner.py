# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The kernel body is copied from Triton's python/tutorials/06-fused-attention-ws.py.
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
# @lint-ignore-every AUTODEPS2

"""Tile contracts for the first numbered warp-specialized attention helper."""

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def _attn_fwd_inner[BM: IntVar, BN: IntVar, D: IntVar, Y: IntVar, NC: IntVar](
    acc: tl.tensor[[BM, D]],
    l_i: tl.tensor[[BM]],
    m_i: tl.tensor[[BM]],
    q: tl.tensor[[BM, D]],  #
    desc_k: tl.tensor_descriptor[Y, D, D, BN, D],
    desc_v: tl.tensor_descriptor[Y, D, D, BN, D],  #
    offset_y: int,
    dtype: object,
    start_m: tl.ProgramId,
    qk_scale: float,  #
    BLOCK_M: Int[BM],
    HEAD_DIM: Int[D],
    BLOCK_N: Int[BN],  #
    STAGE: int,
    offs_m: tl.Offsets[[BM]],
    offs_n: tl.Offsets[[BN]],  #
    N_CTX: Int[NC],
    warp_specialize: bool,
    IS_HOPPER: bool,
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
    offsetk_y = offset_y + lo
    if dtype == tl.float8e5:
        offsetv_y = offset_y * HEAD_DIM + lo
    else:
        offsetv_y = offset_y + lo
    # loop over k, v and update accumulator
    for start_n in tl.range(
        lo, hi, BLOCK_N, warp_specialize=warp_specialize, disallow_acc_multi_buffer=True
    ):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        # -- update output accumulator --
        if not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = (
                acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()  # E: reshape
            )
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])  # E: join
        else:
            acc = acc * alpha[:, None]
        # prepare p and v for the dot
        if dtype == tl.float8e5:
            v = desc_v.load([0, offsetv_y]).T
        else:
            v = desc_v.load([offsetv_y, 0])
        p = p.to(dtype)
        # note that this non transposed v for FP8 is only supported on Blackwell
        acc = tl.dot(p, v, acc)  # E: not assignable
        # update m_i and l_i
        # place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i


def test_wrong_key_width[
    BM: IntVar,
    BN: IntVar,
    D: IntVar,
    Other: IntVar,
    Y: IntVar,
    NC: IntVar,
](
    acc: tl.tensor[[BM, D]],
    l_i: tl.tensor[[BM]],
    m_i: tl.tensor[[BM]],
    q: tl.tensor[[BM, D]],
    wrong_k: tl.tensor_descriptor[Y, Other, Other, BN, Other],
    v: tl.tensor_descriptor[Y, D, D, BN, D],
    offset_y: int,
    start_m: tl.ProgramId,
    block_m: Int[BM],
    dim: Int[D],
    block_n: Int[BN],
    offs_m: tl.Offsets[[BM]],
    offs_n: tl.Offsets[[BN]],
    n_ctx: Int[NC],
) -> None:
    _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        wrong_k,  # E: is not assignable
        v,
        offset_y,
        tl.float16,
        start_m,
        1.0,
        block_m,
        dim,
        block_n,
        1,
        offs_m,
        offs_n,
        n_ctx,
        True,
        False,
    )

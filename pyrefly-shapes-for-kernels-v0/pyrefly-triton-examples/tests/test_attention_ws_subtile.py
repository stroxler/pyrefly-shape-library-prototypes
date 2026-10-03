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

"""The numerical subtile in warp-specialized Triton forward attention."""

from typing import assert_type

import triton
import triton.language as tl
from shape_extensions import IntVar


def _fma_f32x2[Rows: IntVar, Cols: IntVar](
    a: tl.tensor[[Rows, Cols]], b: float, c: tl.tensor[[Rows, 1]]
) -> tl.tensor[[Rows, Cols]]: ...


def _mul_f32x2[Rows: IntVar, Cols: IntVar](
    a: tl.tensor[[Rows, Cols]], b: tl.tensor[[Rows, 1]]
) -> tl.tensor[[Rows, Cols]]: ...


def _reduce_fadd2(a: float, b: float, c: float, d: float) -> tuple[float, float]: ...


@triton.jit
def _attn_fwd_subtile[BM: IntVar, BN: IntVar, Dim: IntVar](
    q: tl.tensor[[BM, Dim]],
    k: tl.tensor[[Dim, BN]],
    offs_m: tl.Offsets[[BM]],
    start_n: int,
    offs_n: tl.Offsets[[BN]],
    qk_scale: float,
    l_i0: tl.tensor[[BM]],
    l_i1: tl.tensor[[BM]],  # used when FADD2_REDUCE is true
    m_i: tl.tensor[[BM]],
    acc: tl.tensor[[BM, Dim]],
    v: tl.tensor[[BN, Dim]],
    dtype: object,
    STAGE: int,
    SUBTILING: bool,
    VECT_MUL: int,
    FADD2_REDUCE: bool,
):
    qk = tl.dot(q, k)
    if STAGE == 2:
        mask = offs_m[:, None] >= (start_n + offs_n[None, :])
        qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
    else:
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        if VECT_MUL == 2 or VECT_MUL == 3:
            qk = _fma_f32x2(qk, qk_scale, -m_ij[:, None])
        else:
            qk = qk * qk_scale - m_ij[:, None]
    p = tl.math.exp2(qk)
    # -- compute correction factor
    alpha = tl.math.exp2(m_i - m_ij)
    if not FADD2_REDUCE:
        l_ij = tl.sum(p, 1)

    # -- update output accumulator --
    BM: tl.constexpr = acc.shape[0]
    BN: tl.constexpr = acc.shape[1]

    if SUBTILING:
        acc0, acc1 = (
            acc.reshape(  # E: No matching overload
                [BM, 2, BN // 2]
            )
            .permute(0, 2, 1)
            .split()
        )
        if VECT_MUL == 1 or VECT_MUL == 3:
            acc0 = _mul_f32x2(acc0, alpha[:, None])
            acc1 = _mul_f32x2(acc1, alpha[:, None])
        else:
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
        acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])  # E: No attribute
    else:
        acc = acc * alpha[:, None]

    # update m_i and l_i
    # place this at the end of the loop to reduce register pressure
    PM: tl.constexpr = p.shape[0]
    PN: tl.constexpr = p.shape[1]
    if FADD2_REDUCE:
        p0, p1 = (
            p.reshape([PM, 2, PN // 2])  # E: No matching overload
            .permute(0, 2, 1)
            .split()
        )
        l_ij0, l_ij1 = tl.reduce(  # E: No attribute
            (p0, p1), axis=1, combine_fn=_reduce_fadd2
        )
        l_i0 = l_i0 * alpha + l_ij0
        l_i1 = l_i1 * alpha + l_ij1

    # prepare p and v for the dot
    p = p.to(dtype)
    # note that this non transposed v for FP8 is only supported on Blackwell
    acc = tl.dot(p, v, acc)
    # update m_i and l_i
    # place this at the end of the loop to reduce register pressure
    if not FADD2_REDUCE:
        l_i0 = l_i0 * alpha + l_ij  # E: may be uninitialized
    m_i = m_ij

    return l_i0, l_i1, m_i, acc


def test_contracting_head_dimension[BM: IntVar, BN: IntVar, Dim: IntVar, Other: IntVar](
    q: tl.tensor[[BM, Dim]], wrong_k: tl.tensor[[Other, BN]]
) -> None:
    tl.dot(q, wrong_k)  # E: is not assignable


def test_value_head_dimension[BM: IntVar, BN: IntVar, Dim: IntVar, Other: IntVar](
    p: tl.tensor[[BM, BN]],
    wrong_v: tl.tensor[[BN, Other]],
    acc: tl.tensor[[BM, Dim]],
) -> None:
    tl.dot(p, wrong_v, acc)  # E: is not assignable


def test_wrong_causal_mask_tile[BM: IntVar, BN: IntVar, Other: IntVar](
    qk: tl.tensor[[BM, BN]],
    rows: tl.Offsets[[Other]],
    cols: tl.Offsets[[BN]],
) -> None:
    mask = rows[:, None] >= cols[None, :]
    qk + tl.where(mask, 0, -1.0e6)  # E: is not supported


def test_scalar_negation_preserves_tile[BM: IntVar, Dim: IntVar](
    row_max: tl.tensor[[BM]],
) -> None:
    assert_type(-row_max[:, None], tl.tensor[[BM, 1]])

# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The kernel body is copied from Triton's 15-multi-cta-layer-norm.py.
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

# This static-only kernel is checked by Pyrefly, not executed as a Buck test.
# @lint-ignore-every AUTODEPS2

"""The single-CTA baseline in Triton's multi-CTA layernorm tutorial."""

from typing import assert_type

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def _layer_norm_fwd_single_cta[
    Rows: IntVar,
    Cols: IntVar,
    Stride: IntVar,
    Block: IntVar,
](
    X: tl.InRowMajorPointer[Rows, Cols, Stride],
    Y: tl.OutRowMajorPointer[Rows, Cols, Stride],
    W: tl.InPointer[[Cols]],
    B: tl.InPointer[[Cols]],
    Mean: tl.OutPointer[[Rows]],
    Rstd: tl.OutPointer[[Rows]],
    stride: Int[Stride],
    N: Int[Cols],
    eps: float,
    BLOCK_SIZE: Int[Block],
):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride

    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N

    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)


def test_wrong_row_stride[
    Rows: IntVar,
    Cols: IntVar,
    Stride: IntVar,
    Other: IntVar,
](
    x: tl.InRowMajorPointer[Rows, Cols, Stride],
    wrong_stride: Int[Other],
) -> None:
    x.__iadd__(tl.program_id(0) * wrong_stride)  # E: is not assignable


def test_wrong_kernel_boundary[
    Rows: IntVar,
    Cols: IntVar,
    Stride: IntVar,
    Other: IntVar,
    Block: IntVar,
](
    x: tl.InRowMajorPointer[Rows, Cols, Stride],
    y: tl.OutRowMajorPointer[Rows, Cols, Stride],
    wrong_y: tl.OutRowMajorPointer[Rows, Other, Stride],
    w: tl.InPointer[[Cols]],
    b: tl.InPointer[[Cols]],
    mean: tl.OutPointer[[Rows]],
    wrong_mean: tl.OutPointer[[Other]],
    rstd: tl.OutPointer[[Rows]],
    stride: Int[Stride],
    n: Int[Cols],
    block: Int[Block],
) -> None:
    _layer_norm_fwd_single_cta(
        x,
        wrong_y,  # E: is not assignable to parameter
        w,
        b,
        mean,
        rstd,
        stride,
        n,
        1e-5,
        block,
    )
    _layer_norm_fwd_single_cta(
        x,
        y,
        w,
        b,
        wrong_mean,  # E: is not assignable to parameter
        rstd,
        stride,
        n,
        1e-5,
        block,
    )


def test_wrong_weight_mask[Cols: IntVar, Other: IntVar, Block: IntVar](
    w: tl.InPointer[[Cols]], offsets: tl.Offsets[[Block]], wrong_n: Int[Other]
) -> None:
    tl.load(w + offsets, mask=offsets < wrong_n)  # E: No matching overload


def test_unverified_offset_values[Cols: IntVar, Block: IntVar](
    w: tl.InPointer[[Cols]],
    offsets: tl.Offsets[[Block]],
    n: Int[Cols],
) -> None:
    # The matching mask cannot prove the values of the offsets it guards.
    assert_type(tl.load(w + offsets, mask=offsets < n), tl.tensor[[Block]])

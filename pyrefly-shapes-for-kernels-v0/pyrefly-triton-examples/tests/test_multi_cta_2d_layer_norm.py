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

"""The two-dimensional distributed layernorm in Triton's tutorial 15."""

from typing import assert_type

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def _layer_norm_fwd_multi_cta_2d[
    Rows: IntVar,
    Cols: IntVar,
    Stride: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    X: tl.In2DRowMajorPointer[Rows, Cols, Stride, BM],
    Y: tl.Out2DRowMajorPointer[Rows, Cols, Stride, BM],
    W: tl.InPointer[[Cols]],
    B: tl.InPointer[[Cols]],
    Mean: tl.OutPointer[[Rows]],
    Rstd: tl.OutPointer[[Rows]],
    stride: Int[Stride],
    M: Int[Rows],
    N: Int[Cols],
    eps: float,
    BLOCK_SIZE_M: Int[BM],
    BLOCK_SIZE_N: Int[BN],
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < M
    X += rows[:, None] * stride
    Y += rows[:, None] * stride

    _mean = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    for off in tl.range(0, N, BLOCK_SIZE_N, multi_cta=True):
        cols = off + tl.arange(0, BLOCK_SIZE_N)
        mask = row_mask[:, None] & (cols[None, :] < N)
        a = tl.load(X + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=1) / N

    _var = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    for off in tl.range(0, N, BLOCK_SIZE_N, multi_cta=True):
        cols = off + tl.arange(0, BLOCK_SIZE_N)
        mask = row_mask[:, None] & (cols[None, :] < N)
        x = tl.load(X + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        x = tl.where(mask, x - mean[:, None], 0.0)
        _var += x * x
    var = tl.sum(_var, axis=1) / N
    rstd = 1 / tl.sqrt(var + eps)

    tl.store(Mean + rows, mean, mask=row_mask)
    tl.store(Rstd + rows, rstd, mask=row_mask)

    for off in tl.range(0, N, BLOCK_SIZE_N, multi_cta=True):
        cols = off + tl.arange(0, BLOCK_SIZE_N)
        mask = row_mask[:, None] & (cols[None, :] < N)
        w = tl.load(W + cols[None, :], mask=cols[None, :] < N)
        b = tl.load(B + cols[None, :], mask=cols[None, :] < N)
        x = tl.load(X + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean[:, None]) * rstd[:, None]
        y = x_hat * w + b
        tl.store(Y + cols[None, :], y, mask=mask)


def test_wrong_row_stride_and_axis[
    Rows: IntVar,
    Cols: IntVar,
    Stride: IntVar,
    Other: IntVar,
    BM: IntVar,
](
    x: tl.In2DRowMajorPointer[Rows, Cols, Stride, BM],
    wrong_rows: tl.RowAxisOffsets[[Other]],
    wrong_axis: tl.ColumnAxisOffsets[[BM]],
    rows: tl.RowAxisOffsets[[BM]],
    stride: Int[Stride],
    wrong_stride: Int[Other],
) -> None:
    x.__iadd__(rows * wrong_stride)  # E: is not assignable
    x.__iadd__(wrong_rows * stride)  # E: is not assignable
    x.__iadd__(wrong_axis * stride)  # E: is not assignable


def test_wrong_input_mask[
    Rows: IntVar,
    Cols: IntVar,
    Other: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    ptrs: tl.In2DRowTilePointers[Rows, Cols, BM, BN],
    wrong_rows: tl.MatrixMask[Other, Cols, [BM], [BN]],
    wrong_columns: tl.MatrixMask[Rows, Other, [BM], [BN]],
    wrong_tile: tl.MatrixMask[Rows, Cols, [BM], [Other]],
) -> None:
    tl.load(ptrs, mask=wrong_rows, other=0.0)  # E: No matching overload
    tl.load(ptrs, mask=wrong_columns, other=0.0)  # E: No matching overload
    tl.load(ptrs, mask=wrong_tile, other=0.0)  # E: No matching overload


def test_wrong_weight_and_stat_masks[
    Rows: IntVar,
    Cols: IntVar,
    Other: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    weight: tl.InColumnTilePointers[Cols, BN],
    stat: tl.OutTilePointers[[Rows], [BM]],
    wrong_weight: tl.ColumnMask[Other, [BN]],
    wrong_stat: tl.Mask[[Other], [BM]],
    tile: tl.tensor[[BM]],
) -> None:
    tl.load(weight, mask=wrong_weight)  # E: No matching overload
    tl.store(stat, tile, mask=wrong_stat)  # E: No matching overload


def test_wrong_output_store[
    Rows: IntVar,
    Cols: IntVar,
    Other: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    ptrs: tl.Out2DRowTilePointers[Rows, Cols, BM, BN],
    value: tl.tensor[[BM, BN]],
    wrong_value: tl.tensor[[BM, Other]],
    wrong_mask: tl.MatrixMask[Rows, Other, [BM], [BN]],
    mask: tl.MatrixMask[Rows, Cols, [BM], [BN]],
) -> None:
    tl.store(ptrs, wrong_value, mask=mask)  # E: No matching overload
    tl.store(ptrs, value, mask=wrong_mask)  # E: No matching overload


def test_wrong_reduction_axis[Rows: IntVar, BM: IntVar, BN: IntVar](
    stats: tl.OutTilePointers[[Rows], [BM]],
    matrix: tl.tensor[[BM, BN]],
    mask: tl.Mask[[Rows], [BM]],
) -> None:
    tl.store(stats, tl.sum(matrix, axis=0), mask=mask)  # E: No matching overload


def test_wrong_kernel_boundary[
    Rows: IntVar,
    Cols: IntVar,
    Stride: IntVar,
    Other: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    x: tl.In2DRowMajorPointer[Rows, Cols, Stride, BM],
    y: tl.Out2DRowMajorPointer[Rows, Cols, Stride, BM],
    wrong_y: tl.Out2DRowMajorPointer[Rows, Other, Stride, BM],
    w: tl.InPointer[[Cols]],
    wrong_w: tl.InPointer[[Other]],
    b: tl.InPointer[[Cols]],
    mean: tl.OutPointer[[Rows]],
    wrong_mean: tl.OutPointer[[Other]],
    rstd: tl.OutPointer[[Rows]],
    stride: Int[Stride],
    rows: Int[Rows],
    cols: Int[Cols],
    bm: Int[BM],
    bn: Int[BN],
) -> None:
    _layer_norm_fwd_multi_cta_2d(
        x,
        wrong_y,  # E: is not assignable to parameter
        w,
        b,
        mean,
        rstd,
        stride,
        rows,
        cols,
        1e-5,
        bm,
        bn,
    )
    _layer_norm_fwd_multi_cta_2d(
        x,
        y,
        wrong_w,  # E: is not assignable to parameter
        b,
        mean,
        rstd,
        stride,
        rows,
        cols,
        1e-5,
        bm,
        bn,
    )
    _layer_norm_fwd_multi_cta_2d(
        x,
        y,
        w,
        b,
        wrong_mean,  # E: is not assignable to parameter
        rstd,
        stride,
        rows,
        cols,
        1e-5,
        bm,
        bn,
    )


def test_unverified_row_pointer_position[
    Rows: IntVar,
    Cols: IntVar,
    Stride: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    x: tl.In2DRowMajorPointer[Rows, Cols, Stride, BM],
    unrelated_cols: tl.ColumnAxisOffsets[[BN]],
    mask: tl.MatrixMask[Rows, Cols, [BM], [BN]],
) -> None:
    # The offset type alone cannot prove that the row pointer was advanced.
    assert_type(tl.load(x + unrelated_cols, mask=mask, other=0.0), tl.tensor[[BM, BN]])

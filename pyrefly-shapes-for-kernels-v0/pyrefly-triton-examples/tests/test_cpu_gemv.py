# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The kernel body is copied from Triton's python/tutorials/cpu/06-matrix-vector-multiplication.py.
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

"""Static matrix/vector allocation and reduction contract for CPU GEMV."""

from typing import assert_type

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def gemv_kernel[Rows: IntVar, Cols: IntVar, Stride: IntVar, BM: IntVar, BN: IntVar](
    Y: tl.CPUGemvOutputPointer[Rows, BM],
    A: tl.CPUGemvMatrixPointer[Rows, Cols, Stride, BM, BN],
    X: tl.CPUGemvVectorPointer[Cols, BN],
    M: Int[Rows],
    N: Int[Cols],
    stride_am: Int[Stride],
    BLOCK_SIZE_M: Int[BM],
    BLOCK_SIZE_N: Int[BN],
):
    start_m = tl.program_id(0)
    rm = start_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = tl.arange(0, BLOCK_SIZE_N)

    A = A + (rm[:, None] * stride_am + rn[None, :])
    X = X + rn

    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for n in range(N, 0, -BLOCK_SIZE_N):  # noqa: B007 - preserve upstream body
        a = tl.load(A)
        x = tl.load(X)
        acc += tl.sum(a * x[None, :], axis=1)
        A += BLOCK_SIZE_N
        X += BLOCK_SIZE_N

    Y = Y + rm
    tl.store(Y, acc)


def test_gemv_host_boundary[
    Rows: IntVar,
    OtherRows: IntVar,
    Cols: IntVar,
    OtherCols: IntVar,
    Stride: IntVar,
    OtherStride: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    output: tl.CPUGemvOutputPointer[Rows, BM],
    wrong_output: tl.CPUGemvOutputPointer[OtherRows, BM],
    matrix: tl.CPUGemvMatrixPointer[Rows, Cols, Stride, BM, BN],
    wrong_matrix: tl.CPUGemvMatrixPointer[Rows, OtherCols, Stride, BM, BN],
    vector: tl.CPUGemvVectorPointer[Cols, BN],
    wrong_vector: tl.CPUGemvVectorPointer[OtherCols, BN],
    rows: Int[Rows],
    cols: Int[Cols],
    stride: Int[Stride],
    wrong_stride: Int[OtherStride],
    block_rows: Int[BM],
    block_cols: Int[BN],
) -> None:
    gemv_kernel(output, matrix, vector, rows, cols, stride, block_rows, block_cols)
    gemv_kernel(
        wrong_output,
        matrix,  # E: is not assignable
        vector,
        rows,  # E: is not assignable
        cols,
        stride,
        block_rows,
        block_cols,
    )
    gemv_kernel(
        output,
        wrong_matrix,
        vector,  # E: is not assignable
        rows,
        cols,  # E: is not assignable
        stride,
        block_rows,
        block_cols,
    )
    gemv_kernel(
        output,
        matrix,
        wrong_vector,  # E: is not assignable
        rows,
        cols,
        stride,
        block_rows,
        block_cols,
    )
    gemv_kernel(
        output,
        matrix,
        vector,
        rows,
        cols,
        wrong_stride,  # E: is not assignable
        block_rows,
        block_cols,
    )


def test_gemv_tile_values[BM: IntVar, BN: IntVar, Other: IntVar, Rows: IntVar](
    matrix_tile: tl.tensor[[BM, BN]],
    vector_tile: tl.tensor[[BN]],
    output: tl.CPUGemvOutputPointer[Rows, BM],
    wrong: tl.tensor[[Other]],
) -> None:
    assert_type(tl.sum(matrix_tile * vector_tile[None, :], axis=1), tl.tensor[[BM]])
    tl.store(output, wrong)  # E: is not assignable


# Changing only the row-stride annotation rejects the original matrix address.
@triton.jit
def gemv_kernel_wrong_stride[
    Rows: IntVar,
    Cols: IntVar,
    Stride: IntVar,
    Other: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    Y: tl.CPUGemvOutputPointer[Rows, BM],
    A: tl.CPUGemvMatrixPointer[Rows, Cols, Stride, BM, BN],
    X: tl.CPUGemvVectorPointer[Cols, BN],
    M: Int[Rows],
    N: Int[Cols],
    stride_am: Int[Other],
    BLOCK_SIZE_M: Int[BM],
    BLOCK_SIZE_N: Int[BN],
):
    start_m = tl.program_id(0)
    rm = start_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = tl.arange(0, BLOCK_SIZE_N)

    A = A + (rm[:, None] * stride_am + rn[None, :])  # E: is not supported between
    X = X + rn

    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for n in range(N, 0, -BLOCK_SIZE_N):  # noqa: B007 - preserve upstream body
        a = tl.load(A)
        x = tl.load(X)
        acc += tl.sum(a * x[None, :], axis=1)
        A += BLOCK_SIZE_N
        X += BLOCK_SIZE_N

    Y = Y + rm
    tl.store(Y, acc)

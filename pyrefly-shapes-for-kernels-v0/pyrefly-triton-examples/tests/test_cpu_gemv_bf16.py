# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The kernel body is copied from Triton's python/tutorials/cpu/07-matrix-vector-multiplication-bf16.py.
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

"""Static shape contract for the BF16 CPU GEMV output conversion."""

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

    y = acc.to(tl.bfloat16)
    Y = Y + rm
    tl.store(Y, y)


def test_bf16_output_tile[Rows: IntVar, Block: IntVar, Other: IntVar](
    accumulator: tl.tensor[[Block]],
    output: tl.CPUGemvOutputPointer[Rows, Block],
    wrong_tile: tl.tensor[[Other]],
) -> None:
    assert_type(accumulator.to(tl.bfloat16), tl.tensor[[Block]])
    tl.store(output, wrong_tile)  # E: is not assignable


# Changing only the output tile annotation rejects the original row offset
# and output store, including the shape preserved by the BF16 conversion.
@triton.jit
def gemv_kernel_wrong_output_tile[
    Rows: IntVar,
    Cols: IntVar,
    Stride: IntVar,
    BM: IntVar,
    Other: IntVar,
    BN: IntVar,
](
    Y: tl.CPUGemvOutputPointer[Rows, Other],
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

    y = acc.to(tl.bfloat16)
    Y = Y + rm  # E: is not supported between
    tl.store(Y, y)  # E: is not assignable

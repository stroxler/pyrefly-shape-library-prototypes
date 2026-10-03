# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# All five kernel bodies are copied from Triton's
# python/tutorials/compilation-pipeline/11_coalesce_load_types.py.
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

"""Check one-dimensional and physical-layout copies without running the GPU."""

from typing import Literal

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def copy_1d[N: IntVar](
    src: tl.FullCopyInputPointer[N], dst: tl.FullCopyOutputPointer[N], N: Int[N]
):
    offs = tl.arange(0, N)
    tl.store(dst + offs, tl.load(src + offs))


@triton.jit
def copy_2d_rowmajor[M: IntVar, N: IntVar](
    src: tl.RowMajorCopyInputPointer[M, N],
    dst: tl.RowMajorCopyOutputPointer[M, N],
    M: Int[M],
    N: Int[N],
):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    idx = rm[:, None] * N + rn[None, :]  # inner axis (n) is contiguous
    tl.store(dst + idx, tl.load(src + idx))


@triton.jit
def copy_2d_colmajor[M: IntVar, N: IntVar](
    src: tl.ColumnMajorCopyInputPointer[M, N],
    dst: tl.ColumnMajorCopyOutputPointer[M, N],
    M: Int[M],
    N: Int[N],
):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    idx = rm[:, None] + rn[None, :] * M  # outer axis (m) is contiguous
    tl.store(dst + idx, tl.load(src + idx))


@triton.jit
def gather_strided[Source: IntVar, N: IntVar, S: IntVar](
    src: tl.StridedCopyInputPointer[Source, N, S],
    dst: tl.FullCopyOutputPointer[N],
    N: Int[N],
    S: Int[S],
):
    offs = tl.arange(0, N)
    tl.store(dst + offs, tl.load(src + offs * S))  # stride S => non-contiguous


@triton.jit
def copy_looped[N: IntVar, Block: IntVar, Steps: IntVar](
    src: tl.InPointer[[N]],
    dst: tl.OutPointer[[N]],
    n: Int[N],
    BLOCK: Int[Block],
    STEPS: Int[Steps],
    NS: Literal[1, 2],
):
    base = tl.program_id(0) * BLOCK * STEPS
    for i in tl.range(0, STEPS, num_stages=NS):
        offs = base + i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)


def test_full_copy_boundaries[
    M: IntVar,
    N: IntVar,
    Other: IntVar,
    S: IntVar,
    Source: IntVar,
    Block: IntVar,
    Steps: IntVar,
](
    src: tl.FullCopyInputPointer[N],
    dst: tl.FullCopyOutputPointer[N],
    wrong_src: tl.FullCopyInputPointer[Other],
    wrong_dst: tl.FullCopyOutputPointer[Other],
    rows: tl.RowMajorCopyInputPointer[M, N],
    row_out: tl.RowMajorCopyOutputPointer[M, N],
    wrong_row_out: tl.RowMajorCopyOutputPointer[M, Other],
    cols: tl.ColumnMajorCopyInputPointer[M, N],
    col_out: tl.ColumnMajorCopyOutputPointer[M, N],
    wrong_cols: tl.ColumnMajorCopyInputPointer[Other, N],
    gathered: tl.StridedCopyInputPointer[Source, N, S],
    wrong_stride: tl.StridedCopyInputPointer[Source, N, Other],
    loop_in: tl.InPointer[[N]],
    loop_out: tl.OutPointer[[N]],
    wrong_loop_out: tl.OutPointer[[Other]],
    m: Int[M],
    n: Int[N],
    stride: Int[S],
    block: Int[Block],
    steps: Int[Steps],
) -> None:
    copy_1d(src, dst, n)
    copy_1d(
        wrong_src,
        dst,  # E: is not assignable
        n,  # E: is not assignable
    )
    copy_1d(src, wrong_dst, n)  # E: is not assignable
    copy_2d_rowmajor(rows, row_out, m, n)
    copy_2d_rowmajor(rows, wrong_row_out, m, n)  # E: is not assignable
    copy_2d_colmajor(cols, col_out, m, n)
    copy_2d_colmajor(
        wrong_cols,
        col_out,  # E: is not assignable
        m,  # E: is not assignable
        n,
    )
    gather_strided(gathered, dst, n, stride)
    gather_strided(wrong_stride, dst, n, stride)  # E: is not assignable
    copy_looped(loop_in, loop_out, n, block, steps, 2)
    copy_looped(loop_in, wrong_loop_out, n, block, steps, 1)  # E: is not assignable
    copy_looped(loop_in, loop_out, n, block, steps, 3)  # E: is not assignable


# The staged copy is the only one that compares each access against `n`.
@triton.jit
def copy_looped_wrong_src[N: IntVar, Other: IntVar, Block: IntVar, Steps: IntVar](
    src: tl.InPointer[[Other]],
    dst: tl.OutPointer[[N]],
    n: Int[N],
    BLOCK: Int[Block],
    STEPS: Int[Steps],
    NS: Literal[1, 2],
):
    base = tl.program_id(0) * BLOCK * STEPS
    for i in tl.range(0, STEPS, num_stages=NS):
        offs = base + i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(
            dst + offs,
            tl.load(src + offs, mask=mask),  # E: is not assignable
            mask=mask,
        )

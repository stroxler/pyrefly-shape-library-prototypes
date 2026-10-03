# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The kernel body is copied from Triton's python/tutorials/cpu/03-matrix-multiplication.py.
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

"""Static tile and pointer-role checks for the CPU matmul padding helper."""

from typing import assert_type

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def pad_kernel[Rows: IntVar, Cols: IntVar, OutCols: IntVar, BM: IntVar, BN: IntVar](
    in_ptr: tl.CPUPadInputPointer[Rows, Cols, BN],
    out_ptr: tl.CPUPadOutputPointer[Rows, OutCols],
    N: Int[Cols],
    BLOCK_SIZE_M: Int[BM],
    BLOCK_SIZE_N: Int[BN],
    PADDING: Int[OutCols - Cols],
):
    in_offset = tl.program_id(axis=0) * N * BLOCK_SIZE_M
    out_offset = tl.program_id(axis=0) * (N + PADDING) * BLOCK_SIZE_M
    for row in tl.range(0, BLOCK_SIZE_M):  # noqa: B007 - preserve upstream body
        for block in tl.range(0, N // BLOCK_SIZE_N):
            val = tl.load(
                in_ptr + in_offset + block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            )
            tl.store(
                out_ptr
                + out_offset
                + block * BLOCK_SIZE_N
                + tl.arange(0, BLOCK_SIZE_N),
                val,
            )
        zero = tl.full((PADDING,), 0, dtype=in_ptr.type.element_ty)
        tl.store(out_ptr + out_offset + N + tl.arange(0, PADDING), zero)
        in_offset += N
        out_offset += N + PADDING


def test_host_shape_boundary[
    Rows: IntVar,
    OtherRows: IntVar,
    Cols: IntVar,
    OtherCols: IntVar,
    OutCols: IntVar,
    BM: IntVar,
    BN: IntVar,
    OtherP: IntVar,
](
    input: tl.CPUPadInputPointer[Rows, Cols, BN],
    wrong_input: tl.CPUPadInputPointer[Rows, OtherCols, BN],
    output: tl.CPUPadOutputPointer[Rows, OutCols],
    wrong_output_rows: tl.CPUPadOutputPointer[OtherRows, OutCols],
    wrong_output_cols: tl.CPUPadOutputPointer[Rows, OtherCols],
    n: Int[Cols],
    block_rows: Int[BM],
    block_cols: Int[BN],
    padding: Int[OutCols - Cols],
    wrong_padding: Int[OtherP],
) -> None:
    pad_kernel(input, output, n, block_rows, block_cols, padding)
    pad_kernel(
        wrong_input,
        output,
        n,  # E: is not assignable
        block_rows,
        block_cols,
        padding,  # E: is not assignable
    )
    pad_kernel(
        input,
        wrong_output_rows,  # E: is not assignable
        n,
        block_rows,
        block_cols,
        padding,  # E: cannot be inferred
    )
    pad_kernel(
        input,
        wrong_output_cols,
        n,
        block_rows,
        block_cols,
        padding,  # E: is not assignable
    )
    pad_kernel(
        input,
        output,
        n,
        block_rows,
        block_cols,
        wrong_padding,  # E: is not assignable
    )


def test_tile_widths[BN: IntVar, P: IntVar, Other: IntVar](
    input_tile: tl.CPUPadInputTile[BN],
    data_output: tl.CPUPadOutputTile[BN],
    padding_output: tl.CPUPadOutputTile[P],
    padding: Int[P],
    wrong_data: tl.tensor[[Other]],
) -> None:
    assert_type(tl.load(input_tile), tl.tensor[[BN]])
    assert_type(tl.full((padding,), 0, dtype=tl.float32), tl.tensor[[P]])
    tl.store(data_output, wrong_data)  # E: is not assignable
    tl.store(padding_output, wrong_data)  # E: is not assignable


# The input's declared tile width alone changes; its unchanged load must reject
# the source's BLOCK_SIZE_N offsets rather than returning an unknown value.
@triton.jit
def pad_kernel_wrong_input_tile[
    Rows: IntVar,
    Cols: IntVar,
    OutCols: IntVar,
    BM: IntVar,
    BN: IntVar,
    Other: IntVar,
](
    in_ptr: tl.CPUPadInputPointer[Rows, Cols, Other],
    out_ptr: tl.CPUPadOutputPointer[Rows, OutCols],
    N: Int[Cols],
    BLOCK_SIZE_M: Int[BM],
    BLOCK_SIZE_N: Int[BN],
    PADDING: Int[OutCols - Cols],
):
    in_offset = tl.program_id(axis=0) * N * BLOCK_SIZE_M
    out_offset = tl.program_id(axis=0) * (N + PADDING) * BLOCK_SIZE_M
    for row in tl.range(0, BLOCK_SIZE_M):  # noqa: B007 - preserve upstream body
        for block in tl.range(0, N // BLOCK_SIZE_N):
            val = tl.load(
                in_ptr  # E: is not supported between
                + in_offset
                + block * BLOCK_SIZE_N
                + tl.arange(0, BLOCK_SIZE_N)
            )
            tl.store(
                out_ptr
                + out_offset
                + block * BLOCK_SIZE_N
                + tl.arange(0, BLOCK_SIZE_N),
                val,
            )
        zero = tl.full((PADDING,), 0, dtype=in_ptr.type.element_ty)
        tl.store(out_ptr + out_offset + N + tl.arange(0, PADDING), zero)
        in_offset += N
        out_offset += N + PADDING


# An input pointer substituted for the output is rejected at the original stores.
@triton.jit
def pad_kernel_wrong_output_role[
    Rows: IntVar,
    Cols: IntVar,
    OutCols: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    in_ptr: tl.CPUPadInputPointer[Rows, Cols, BN],
    out_ptr: tl.CPUPadInputPointer[Rows, OutCols, BN],
    N: Int[Cols],
    BLOCK_SIZE_M: Int[BM],
    BLOCK_SIZE_N: Int[BN],
    PADDING: Int[OutCols - Cols],
):
    in_offset = tl.program_id(axis=0) * N * BLOCK_SIZE_M
    out_offset = tl.program_id(axis=0) * (N + PADDING) * BLOCK_SIZE_M
    for row in tl.range(0, BLOCK_SIZE_M):  # noqa: B007 - preserve upstream body
        for block in tl.range(0, N // BLOCK_SIZE_N):
            val = tl.load(
                in_ptr + in_offset + block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            )
            tl.store(  # E: No matching overload
                out_ptr
                + out_offset
                + block * BLOCK_SIZE_N
                + tl.arange(0, BLOCK_SIZE_N),
                val,
            )
        zero = tl.full((PADDING,), 0, dtype=in_ptr.type.element_ty)
        tl.store(
            out_ptr  # E: is not supported between
            + out_offset
            + N
            + tl.arange(0, PADDING),
            zero,
        )
        in_offset += N
        out_offset += N + PADDING


# The body also ties the declared physical input and output widths to their
# program-stride products. Neither failure proves the actual address is in bounds.
@triton.jit
def pad_kernel_wrong_allocations[
    Rows: IntVar,
    Cols: IntVar,
    OutCols: IntVar,
    OtherCols: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    in_ptr: tl.CPUPadInputPointer[Rows, OtherCols, BN],
    out_ptr: tl.CPUPadOutputPointer[Rows, OtherCols],
    N: Int[Cols],
    BLOCK_SIZE_M: Int[BM],
    BLOCK_SIZE_N: Int[BN],
    PADDING: Int[OutCols - Cols],
):
    in_offset = tl.program_id(axis=0) * N * BLOCK_SIZE_M
    out_offset = tl.program_id(axis=0) * (N + PADDING) * BLOCK_SIZE_M
    for row in tl.range(0, BLOCK_SIZE_M):  # noqa: B007 - preserve upstream body
        for block in tl.range(0, N // BLOCK_SIZE_N):
            val = tl.load(
                in_ptr  # E: is not supported between
                + in_offset
                + block * BLOCK_SIZE_N
                + tl.arange(0, BLOCK_SIZE_N)
            )
            tl.store(
                out_ptr  # E: is not supported between
                + out_offset
                + block * BLOCK_SIZE_N
                + tl.arange(0, BLOCK_SIZE_N),
                val,
            )
        zero = tl.full((PADDING,), 0, dtype=in_ptr.type.element_ty)
        tl.store(
            out_ptr  # E: is not supported between
            + out_offset
            + N
            + tl.arange(0, PADDING),
            zero,
        )
        in_offset += N
        out_offset += N + PADDING

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# These static-only kernels are checked by Pyrefly, not executed as Buck tests.
# @lint-ignore-every AUTODEPS2

"""Static-only copy of softmax_kernel in Triton's python/tutorials/02-fused-softmax.py."""

from typing import assert_type

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def softmax_kernel[
    Rows: IntVar,
    Cols: IntVar,
    InputStride: IntVar,
    OutputStride: IntVar,
    Block: IntVar,
](
    output_ptr: tl.OutRowMajorPointer[Rows, Cols, OutputStride],
    input_ptr: tl.InRowMajorPointer[Rows, Cols, InputStride],
    input_row_stride: Int[InputStride],
    output_row_stride: Int[OutputStride],
    n_rows: Int[Rows],
    n_cols: Int[Cols],
    BLOCK_SIZE: Int[Block],
    num_stages: int,
):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def test_wrong_input_rows[
    Rows: IntVar,
    Other: IntVar,
    Cols: IntVar,
    IS: IntVar,
    OS: IntVar,
    Block: IntVar,
](
    output: tl.OutRowMajorPointer[Rows, Cols, OS],
    input: tl.InRowMajorPointer[Other, Cols, IS],
    input_stride: Int[IS],
    output_stride: Int[OS],
    rows: Int[Rows],
    cols: Int[Cols],
    block: Int[Block],
) -> None:
    softmax_kernel(
        output,
        input,  # E: is not assignable to parameter
        input_stride,
        output_stride,
        rows,
        cols,
        block,
        2,
    )


def test_wrong_output_columns[
    Rows: IntVar,
    Cols: IntVar,
    Other: IntVar,
    IS: IntVar,
    OS: IntVar,
    Block: IntVar,
](
    output: tl.OutRowMajorPointer[Rows, Other, OS],
    input: tl.InRowMajorPointer[Rows, Cols, IS],
    input_stride: Int[IS],
    output_stride: Int[OS],
    rows: Int[Rows],
    cols: Int[Cols],
    block: Int[Block],
) -> None:
    softmax_kernel(
        output,
        input,  # E: is not assignable to parameter
        input_stride,
        output_stride,
        rows,
        cols,  # E: is not assignable to parameter
        block,
        2,
    )


def test_wrong_input_stride[
    Rows: IntVar,
    Cols: IntVar,
    IS: IntVar,
    Other: IntVar,
    OS: IntVar,
    Block: IntVar,
](
    output: tl.OutRowMajorPointer[Rows, Cols, OS],
    input: tl.InRowMajorPointer[Rows, Cols, IS],
    wrong_stride: Int[Other],
    output_stride: Int[OS],
    rows: Int[Rows],
    cols: Int[Cols],
    block: Int[Block],
) -> None:
    softmax_kernel(
        output,
        input,
        wrong_stride,  # E: is not assignable to parameter
        output_stride,
        rows,
        cols,
        block,
        2,
    )


def test_wrong_output_stride[
    Rows: IntVar,
    Cols: IntVar,
    IS: IntVar,
    OS: IntVar,
    Other: IntVar,
    Block: IntVar,
](
    output: tl.OutRowMajorPointer[Rows, Cols, OS],
    input: tl.InRowMajorPointer[Rows, Cols, IS],
    input_stride: Int[IS],
    wrong_stride: Int[Other],
    rows: Int[Rows],
    cols: Int[Cols],
    block: Int[Block],
) -> None:
    softmax_kernel(
        output,
        input,
        input_stride,
        wrong_stride,  # E: is not assignable to parameter
        rows,
        cols,
        block,
        2,
    )


def test_wrong_row_mask_bound[Rows: IntVar, Cols: IntVar, Other: IntVar, Block: IntVar](
    ptrs: tl.InRowTilePointers[Rows, Cols, [Block]],
    mask: tl.Mask[[Other], [Block]],
) -> None:
    tl.load(ptrs, mask=mask, other=0.0)  # E: is not assignable to parameter


def test_wrong_row_mask_tile[Rows: IntVar, Cols: IntVar, Block: IntVar, Other: IntVar](
    ptrs: tl.OutRowTilePointers[Rows, Cols, [Block]],
    mask: tl.Mask[[Cols], [Other]],
    value: tl.tensor[[Block]],
) -> None:
    tl.store(ptrs, value, mask=mask)  # E: No matching overload


def test_wrong_output_value_tile[
    Rows: IntVar,
    Cols: IntVar,
    Block: IntVar,
    Other: IntVar,
](
    ptrs: tl.OutRowTilePointers[Rows, Cols, [Block]],
    mask: tl.Mask[[Cols], [Block]],
    value: tl.tensor[[Other]],
) -> None:
    tl.store(ptrs, value, mask=mask)  # E: No matching overload


def test_wrong_reduction_axis[Block: IntVar](value: tl.tensor[[Block]]) -> None:
    tl.max(value, axis=1)  # E: is not assignable to parameter
    tl.sum(value, axis=1)  # E: is not assignable to parameter


def test_wrong_reduction_rank[Rows: IntVar, Cols: IntVar](
    value: tl.tensor[[Rows, Cols]],
) -> None:
    tl.max(value, axis=0)  # E: is not assignable to parameter
    assert_type(tl.sum(value, axis=0), tl.tensor[[Cols]])

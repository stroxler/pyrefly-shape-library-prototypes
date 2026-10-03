# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# These static-only kernels are checked by Pyrefly, not executed as Buck tests.
# @lint-ignore-every AUTODEPS2

"""Static-only copy of the kernel in Triton's python/tutorials/01-vector-add.py."""

from typing import assert_type

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def add_kernel[N: IntVar, Block: IntVar](
    x_ptr: tl.InPointer[[N]],  # *Pointer* to first input vector.
    y_ptr: tl.InPointer[[N]],  # *Pointer* to second input vector.
    output_ptr: tl.OutPointer[[N]],  # *Pointer* to output vector.
    n_elements: Int[N],  # Size of the vector.
    BLOCK_SIZE: Int[Block],  # Number of elements each program should process.
    # NOTE: `constexpr` so it can be used as a shape value.
):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


def test_mask_carries_allocation_and_tile[N: IntVar, Block: IntVar](
    n: Int[N], block: Int[Block]
) -> None:
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    assert_type(offsets < n, tl.Mask[[N], [Block]])


def test_mask_with_wrong_allocation[N: IntVar, Other: IntVar, Block: IntVar](
    ptrs: tl.InTilePointers[[N], [Block]], mask: tl.Mask[[Other], [Block]]
) -> None:
    tl.load(ptrs, mask=mask)  # E: is not assignable to parameter


def test_mask_with_wrong_tile[N: IntVar, Block: IntVar, Other: IntVar](
    ptrs: tl.InTilePointers[[N], [Block]], mask: tl.Mask[[N], [Other]]
) -> None:
    tl.load(ptrs, mask=mask)  # E: is not assignable to parameter


def test_output_mask_with_wrong_allocation[N: IntVar, Other: IntVar, Block: IntVar](
    ptrs: tl.OutTilePointers[[N], [Block]],
    value: tl.tensor[[Block]],
    mask: tl.Mask[[Other], [Block]],
) -> None:
    tl.store(ptrs, value, mask=mask)  # E: is not assignable to parameter


def test_wrong_input_shape[N: IntVar, Other: IntVar, Block: IntVar](
    x: tl.InPointer[[N]],
    y: tl.InPointer[[Other]],
    out: tl.OutPointer[[N]],
    n: Int[N],
    block: Int[Block],
) -> None:
    add_kernel(x, y, out, n, block)  # E: is not assignable to parameter


def test_wrong_output_shape[N: IntVar, Other: IntVar, Block: IntVar](
    x: tl.InPointer[[N]],
    y: tl.InPointer[[N]],
    out: tl.OutPointer[[Other]],
    n: Int[N],
    block: Int[Block],
) -> None:
    add_kernel(x, y, out, n, block)  # E: is not assignable to parameter

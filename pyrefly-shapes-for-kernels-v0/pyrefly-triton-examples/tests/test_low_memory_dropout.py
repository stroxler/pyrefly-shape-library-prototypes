# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# These static-only kernels are checked by Pyrefly, not executed as Buck tests.
# @lint-ignore-every AUTODEPS2

"""Static-only copies of both kernels in Triton's 04-low-memory-dropout.py."""

from typing import assert_type

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def _dropout[N: IntVar, Block: IntVar](
    x_ptr: tl.InPointer[[N]],  # pointer to the input
    x_keep_ptr: tl.InPointer[[N]],  # pointer to a mask of 0s and 1s
    output_ptr: tl.OutPointer[[N]],  # pointer to the output
    n_elements: Int[N],  # number of elements in the `x` tensor
    p: float,  # probability that an element of `x` is changed to zero
    BLOCK_SIZE: Int[Block],
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    # The line below is the crucial part, described in the paragraph above!
    output = tl.where(x_keep, x / (1 - p), 0.0)
    # Write-back output
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def _seeded_dropout[N: IntVar, Block: IntVar](
    x_ptr: tl.InPointer[[N]],
    output_ptr: tl.OutPointer[[N]],
    n_elements: Int[N],
    p: float,
    seed: int,
    BLOCK_SIZE: Int[Block],
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # randomly prune it
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def _dropout_wrong_keep_annotation[N: IntVar, Other: IntVar, Block: IntVar](
    x_ptr: tl.InPointer[[N]],  # pointer to the input
    x_keep_ptr: tl.InPointer[[Other]],  # pointer to a mask of 0s and 1s
    output_ptr: tl.OutPointer[[N]],  # pointer to the output
    n_elements: Int[N],  # number of elements in the `x` tensor
    p: float,  # probability that an element of `x` is changed to zero
    BLOCK_SIZE: Int[Block],
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(  # E: No matching overload
        x_keep_ptr + offsets,
        mask=mask,
    )
    # The line below is the crucial part, described in the paragraph above!
    output = tl.where(x_keep, x / (1 - p), 0.0)
    # Write-back output
    tl.store(output_ptr + offsets, output, mask=mask)


def test_dropout_mask_and_random_tile[N: IntVar, Block: IntVar](
    length: Int[N], block: Int[Block], seed: int, p: float
) -> None:
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    assert_type(offsets < length, tl.Mask[[N], [Block]])
    random = tl.rand(seed, offsets)
    assert_type(random, tl.tensor[[Block]])
    assert_type(random > p, tl.tensor[[Block]])


def test_baseline_wrong_keep_length[N: IntVar, Other: IntVar, Block: IntVar](
    x: tl.InPointer[[N]],
    keep: tl.InPointer[[Other]],
    output: tl.OutPointer[[N]],
    length: Int[N],
    block: Int[Block],
) -> None:
    _dropout(x, keep, output, length, 0.5, block)  # E: is not assignable to parameter


def test_baseline_wrong_output_length[N: IntVar, Other: IntVar, Block: IntVar](
    x: tl.InPointer[[N]],
    keep: tl.InPointer[[N]],
    output: tl.OutPointer[[Other]],
    length: Int[N],
    block: Int[Block],
) -> None:
    _dropout(x, keep, output, length, 0.5, block)  # E: is not assignable to parameter


def test_seeded_wrong_input_length[N: IntVar, Other: IntVar, Block: IntVar](
    x: tl.InPointer[[Other]],
    output: tl.OutPointer[[N]],
    length: Int[N],
    block: Int[Block],
) -> None:
    _seeded_dropout(
        x,
        output,  # E: is not assignable to parameter
        length,  # E: is not assignable to parameter
        0.5,
        123,
        block,
    )


def test_seeded_wrong_output_length[N: IntVar, Other: IntVar, Block: IntVar](
    x: tl.InPointer[[N]],
    output: tl.OutPointer[[Other]],
    length: Int[N],
    block: Int[Block],
) -> None:
    _seeded_dropout(
        x,
        output,  # E: is not assignable to parameter
        length,
        0.5,
        123,
        block,
    )


def test_seeded_wrong_seed[N: IntVar, Block: IntVar](
    x: tl.InPointer[[N]],
    output: tl.OutPointer[[N]],
    length: Int[N],
    block: Int[Block],
) -> None:
    _seeded_dropout(
        x,
        output,
        length,
        0.5,
        "123",  # E: is not assignable to parameter
        block,
    )


def test_wrong_mask_bound[N: IntVar, Other: IntVar, Block: IntVar](
    ptrs: tl.InTilePointers[[N], [Block]],
    offsets: tl.Offsets[[Block]],
    wrong_length: Int[Other],
) -> None:
    tl.load(ptrs, mask=offsets < wrong_length)  # E: is not assignable to parameter


def test_wrong_keep_mask_tile[N: IntVar, Block: IntVar, Other: IntVar](
    ptrs: tl.InTilePointers[[N], [Block]], mask: tl.Mask[[N], [Other]]
) -> None:
    tl.load(ptrs, mask=mask)  # E: is not assignable to parameter


def test_wrong_output_value_tile[N: IntVar, Block: IntVar, Other: IntVar](
    ptrs: tl.OutTilePointers[[N], [Block]],
    value: tl.tensor[[Other]],
    mask: tl.Mask[[N], [Block]],
) -> None:
    tl.store(ptrs, value, mask=mask)  # E: No matching overload


def test_wrong_random_offsets_tile[Block: IntVar, Other: IntVar](
    seed: int, offsets: tl.Offsets[[Other]], data: tl.tensor[[Block]], p: float
) -> None:
    tl.where(tl.rand(seed, offsets) > p, data, 0.0)  # E: No matching overload

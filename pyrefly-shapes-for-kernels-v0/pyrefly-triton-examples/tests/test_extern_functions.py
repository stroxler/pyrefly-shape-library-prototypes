# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# These static-only kernels are checked by Pyrefly, not executed as Buck tests.
# @lint-ignore-every AUTODEPS2

"""Static-only copy of Triton's python/tutorials/07-extern-functions.py kernel."""

from typing import assert_type

import triton
import triton.language as tl
from shape_extensions import Int, IntVar
from triton.language.extra import libdevice


@triton.jit
def asin_kernel[N: IntVar, Block: IntVar](
    x_ptr: tl.InPointer[[N]],
    y_ptr: tl.OutPointer[[N]],
    n_elements: Int[N],
    BLOCK_SIZE: Int[Block],
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x = libdevice.asin(x)
    tl.store(y_ptr + offsets, x, mask=mask)


def test_libdevice_preserves_tile[Block: IntVar, Other: IntVar](
    value: tl.tensor[[Block]], wrong: tl.tensor[[Other]]
) -> None:
    assert_type(libdevice.asin(value), tl.tensor[[Block]])
    incompatible: tl.tensor[[Block]] = libdevice.asin(wrong)  # E: is not assignable
    assert_type(incompatible, tl.tensor[[Block]])


def test_wrong_input_length[N: IntVar, Other: IntVar, Block: IntVar](
    x: tl.InPointer[[Other]],
    out: tl.OutPointer[[N]],
    length: Int[N],
    block: Int[Block],
) -> None:
    asin_kernel(
        x,
        out,  # E: is not assignable to parameter
        length,  # E: is not assignable to parameter
        block,
    )


def test_wrong_output_length[N: IntVar, Other: IntVar, Block: IntVar](
    x: tl.InPointer[[N]],
    out: tl.OutPointer[[Other]],
    length: Int[N],
    block: Int[Block],
) -> None:
    asin_kernel(x, out, length, block)  # E: is not assignable to parameter


def test_wrong_load_bound[N: IntVar, Other: IntVar, Block: IntVar](
    ptrs: tl.InTilePointers[[N], [Block]], mask: tl.Mask[[Other], [Block]]
) -> None:
    tl.load(ptrs, mask=mask)  # E: No matching overload


def test_wrong_store_tile[N: IntVar, Block: IntVar, Other: IntVar](
    ptrs: tl.OutTilePointers[[N], [Block]],
    mask: tl.Mask[[N], [Block]],
    value: tl.tensor[[Other]],
) -> None:
    tl.store(ptrs, libdevice.asin(value), mask=mask)  # E: No matching overload

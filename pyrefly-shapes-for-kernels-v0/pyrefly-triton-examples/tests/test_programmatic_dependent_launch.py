# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This static-only kernel is checked by Pyrefly, not executed as a Buck test.
# @lint-ignore-every AUTODEPS2

"""The add_kernel in Triton's 11-programmatic-dependent-launch.py."""

from typing import assert_type, Literal

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def add_kernel[N: IntVar, Block: IntVar](
    x_ptr: tl.InPointer[[N]],  #
    y_ptr: tl.InPointer[[N]],  #
    output_ptr: tl.OutPointer[[N]],  #
    n_elements: Int[N],  #
    BLOCK_SIZE: Int[Block],  #
    USE_GDC: bool,  #
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    if USE_GDC:
        # GDC wait waits for ALL programs in the the prior kernel to complete before continuing.
        # This ensures any memory operations happen before the wait in program order,
        # e.g. if the prior kernel writes to x or y the new values will be visible.
        tl.extra.cuda.gdc_wait()

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    if USE_GDC:
        # GDC launch dependents hints the runtime system to launch dependent kernels.
        # These dependent kernels must also be launched with PDL enabled.
        # Once GDC launch has been issued by ALL programs or
        # programs have finished, the dependent grid can begin if there are enough resources.
        # Note: this by itself provides no additional memory-ordering guarantees, unlike `gdc_wait`
        tl.extra.cuda.gdc_launch_dependents()
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def test_mask_carries_length_and_tile[N: IntVar, Block: IntVar](
    n: Int[N], block: Int[Block]
) -> None:
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    assert_type(offsets < n, tl.Mask[[N], [Block]])


def test_bad_input_mask[N: IntVar, Other: IntVar, Block: IntVar](
    ptrs: tl.InTilePointers[[N], [Block]], mask: tl.Mask[[Other], [Block]]
) -> None:
    tl.load(ptrs, mask=mask)  # E: is not assignable to parameter


def test_bad_output_mask[N: IntVar, Other: IntVar, Block: IntVar](
    ptrs: tl.OutTilePointers[[N], [Block]],
    value: tl.tensor[[Block]],
    mask: tl.Mask[[Other], [Block]],
) -> None:
    tl.store(ptrs, value, mask=mask)  # E: is not assignable to parameter


def test_bad_output_tile[N: IntVar, Block: IntVar, Other: IntVar](
    ptrs: tl.OutTilePointers[[N], [Block]],
    value: tl.tensor[[Other]],
    mask: tl.Mask[[N], [Block]],
) -> None:
    tl.store(ptrs, value, mask=mask)  # E: is not assignable to parameter


def test_wrong_input_length[N: IntVar, Other: IntVar, Block: IntVar](
    x: tl.InPointer[[Other]],
    y: tl.InPointer[[N]],
    out: tl.OutPointer[[N]],
    n: Int[N],
    block: Int[Block],
) -> None:
    add_kernel(
        x,
        y,  # E: is not assignable to parameter
        out,  # E: is not assignable to parameter
        n,  # E: is not assignable to parameter
        block,
        True,
    )


def test_wrong_output_length[N: IntVar, Other: IntVar, Block: IntVar](
    x: tl.InPointer[[N]],
    y: tl.InPointer[[N]],
    out: tl.OutPointer[[Other]],
    n: Int[N],
    block: Int[Block],
) -> None:
    add_kernel(x, y, out, n, block, True)  # E: is not assignable to parameter


def test_wrong_gdc_flag[N: IntVar, Block: IntVar](
    x: tl.InPointer[[N]],
    y: tl.InPointer[[N]],
    out: tl.OutPointer[[N]],
    n: Int[N],
    block: Int[Block],
) -> None:
    add_kernel(x, y, out, n, block, "enabled")  # E: is not assignable to parameter


def test_unverified_host_pdl_flag[N: IntVar, Block: IntVar](
    x: tl.InPointer[[N]],
    y: tl.InPointer[[N]],
    out: tl.OutPointer[[N]],
    n: Int[N],
    block: Int[Block],
    launch_pdl: Literal[False],
) -> None:
    # Known gap: the host launch flag is not coupled to the kernel's USE_GDC.
    assert_type(launch_pdl, Literal[False])
    assert_type(add_kernel(x, y, out, n, block, True), None)

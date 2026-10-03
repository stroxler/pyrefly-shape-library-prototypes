# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only original Python wrapper with semantic Torch parameters.
# @lint-ignore-every AUTODEPS2

"""Check the declared Torch boundary of the original TMA-gather wrapper."""

from typing import assert_type

import torch
from shape_extensions import Int, IntVar
from tests.test_gluon_tma_gather import async_gather_kernel
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


def async_gather[XMax: IntVar, YMax: IntVar, BX: IntVar, BY: IntVar](
    input: torch.Tensor[[XMax, YMax]],
    x_offsets: torch.Tensor[[BX]],
    y_offset: int,
    BLOCK_X: Int[BX],
    BLOCK_Y: Int[BY],
):
    gl_dtype = getattr(gl, str(input.dtype).split(".")[1])
    # When picking the shared memory layout, we use the dimensions of the shared
    # memory descriptor, which will be [BLOCK_X, BLOCK_Y]. But the block shape of the
    # tensor descriptor must still be [1, BLOCK_Y] to be used with async gather.
    layout = gl.NVMMASharedLayout.get_default_for([BLOCK_X, BLOCK_Y], gl_dtype)
    tensor_desc = TensorDescriptor.from_tensor(input, [1, BLOCK_Y], layout)
    out = torch.empty((BLOCK_X, BLOCK_Y), dtype=input.dtype, device="cuda")
    async_gather_kernel[(1,)](  # E: is not subscriptable
        out, *out.stride(), tensor_desc, x_offsets, y_offset, BLOCK_X
    )
    return out


def test_original_gather_wrapper_boundary[
    XMax: IntVar,
    YMax: IntVar,
    BX: IntVar,
    BY: IntVar,
    Other: IntVar,
](
    xmax: Int[XMax],
    ymax: Int[YMax],
    bx: Int[BX],
    by: Int[BY],
    other: Int[Other],
) -> None:
    source = torch.empty((xmax, ymax), dtype=torch.float32, device="cuda")
    offsets = torch.empty((bx,), dtype=torch.int32, device="cuda")
    wrong_offsets = torch.empty((other,), dtype=torch.int32, device="cuda")
    result = async_gather(source, offsets, 0, bx, by)
    assert_type(result, torch.Tensor[[BX, BY]])
    async_gather(source, wrong_offsets, 0, bx, by)  # E: is not assignable
    accepted_dtype = torch.empty((bx,), dtype=torch.float32, device="cuda")
    async_gather(source, accepted_dtype, 0, bx, by)

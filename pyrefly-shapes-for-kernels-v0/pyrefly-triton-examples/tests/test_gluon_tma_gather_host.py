# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only typed descriptor bridge; the original JIT bracket launch is separate.
# @lint-ignore-every AUTODEPS2

"""Check the Torch source descriptor and directly typed gather kernel boundary."""

from typing import assert_type

import torch
from shape_extensions import Int, IntListLiteral, IntVar
from tests.test_gluon_tma_gather import async_gather_kernel
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


def test_direct_torch_descriptor_boundary[
    XMax: IntVar,
    YMax: IntVar,
    BX: IntVar,
    BY: IntVar,
    Other: IntVar,
    StrideX: IntVar,
    StrideY: IntVar,
](
    xmax: Int[XMax],
    ymax: Int[YMax],
    bx: Int[BX],
    by: Int[BY],
    other: Int[Other],
    output: gl.OutMatrixPointer2D[BX, BY, StrideX, StrideY],
    wrong_output: gl.OutMatrixPointer2D[BX, Other, StrideX, StrideY],
    offsets: gl.GatherOffsetsPointer1D[BX],
    wrong_offsets: gl.GatherOffsetsPointer1D[Other],
    stride_x: Int[StrideX],
    stride_y: Int[StrideY],
) -> None:
    source = torch.empty((xmax, ymax), dtype=torch.float32, device="cuda")
    wrong_source = torch.empty((other, ymax), dtype=torch.float32, device="cuda")
    shape: IntListLiteral[[1, BY]] = [1, by]
    layout_shape: IntListLiteral[[BX, BY]] = [bx, by]
    layout = gl.NVMMASharedLayout.get_default_for(layout_shape, gl.float32)
    desc = TensorDescriptor.from_tensor(source, shape, layout)
    wrong_source_desc = TensorDescriptor.from_tensor(wrong_source, shape, layout)
    wrong_block_shape: IntListLiteral[[1, Other]] = [1, other]
    wrong_block = TensorDescriptor.from_tensor(wrong_source, wrong_block_shape, layout)
    wrong_first_shape: IntListLiteral[[Other, BY]] = [other, by]
    wrong_first_block = TensorDescriptor.from_tensor(source, wrong_first_shape, layout)
    assert_type(desc, gl.WgmmaDescriptorF32[XMax, YMax, 1, BY, int])

    async_gather_kernel(output, stride_x, stride_y, desc, offsets, 0, bx)
    async_gather_kernel(
        wrong_output,
        stride_x,
        stride_y,
        desc,  # E: is not assignable
        offsets,
        0,
        bx,
    )
    async_gather_kernel(
        output,
        stride_x,
        stride_y,
        wrong_block,  # E: is not assignable
        offsets,
        0,
        bx,
    )
    async_gather_kernel(
        output,
        stride_x,
        stride_y,
        wrong_first_block,  # E: is not assignable
        offsets,
        0,
        bx,
    )
    async_gather_kernel(
        output,
        stride_x,
        stride_y,
        desc,
        wrong_offsets,  # E: is not assignable
        0,
        bx,
    )

    # Gather only uses the input descriptor's block; full host XMax is unchecked.
    async_gather_kernel(output, stride_x, stride_y, wrong_source_desc, offsets, 0, bx)

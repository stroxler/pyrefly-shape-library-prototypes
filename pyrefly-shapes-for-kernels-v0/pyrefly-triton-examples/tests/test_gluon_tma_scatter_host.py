# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only typed descriptor bridge; the original JIT bracket launch is separate.
# @lint-ignore-every AUTODEPS2

"""Check Torch destination descriptor and directly typed scatter boundary."""

from typing import assert_type

import torch
from shape_extensions import Int, IntListLiteral, IntVar
from tests.test_gluon_tma_scatter import async_scatter_kernel
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


def test_direct_torch_scatter_descriptor[
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
    offsets: gl.GatherOffsetsPointer1D[BX],
    wrong_offsets: gl.GatherOffsetsPointer1D[Other],
    src: gl.InMatrixPointer2D[BX, BY, StrideX, StrideY],
    wrong_src: gl.InMatrixPointer2D[BX, Other, StrideX, StrideY],
    sx: Int[StrideX],
    sy: Int[StrideY],
) -> None:
    dest_tensor = torch.empty((xmax, ymax), dtype=torch.float32, device="cuda")
    wrong_host = torch.empty((other, ymax), dtype=torch.float32, device="cuda")
    shape: IntListLiteral[[1, BY]] = [1, by]
    layout_shape: IntListLiteral[[BX, BY]] = [bx, by]
    layout = gl.NVMMASharedLayout.get_default_for(layout_shape, gl.float32)
    dest = TensorDescriptor.from_tensor(dest_tensor, shape, layout)
    wrong_host_desc = TensorDescriptor.from_tensor(wrong_host, shape, layout)
    bad_block_shape: IntListLiteral[[Other, BY]] = [other, by]
    wrong_first_block = TensorDescriptor.from_tensor(
        dest_tensor, bad_block_shape, layout
    )
    wrong_width_shape: IntListLiteral[[1, Other]] = [1, other]
    wrong_width = TensorDescriptor.from_tensor(dest_tensor, wrong_width_shape, layout)
    assert_type(dest, gl.WgmmaDescriptorF32[XMax, YMax, 1, BY, int])

    async_scatter_kernel(dest, offsets, 0, src, sx, sy, bx)
    async_scatter_kernel(
        wrong_first_block,  # E: is not assignable
        offsets,
        0,
        src,
        sx,
        sy,
        bx,
    )
    async_scatter_kernel(
        wrong_width,
        offsets,
        0,
        src,  # E: is not assignable
        sx,
        sy,
        bx,
    )
    async_scatter_kernel(
        dest,
        wrong_offsets,
        0,
        src,  # E: is not assignable
        sx,
        sy,
        bx,  # E: is not assignable
    )
    async_scatter_kernel(
        dest,
        offsets,
        0,
        wrong_src,  # E: is not assignable
        sx,
        sy,
        bx,
    )

    # Full destination host extent is not checked by the unchanged scatter body.
    async_scatter_kernel(wrong_host_desc, offsets, 0, src, sx, sy, bx)

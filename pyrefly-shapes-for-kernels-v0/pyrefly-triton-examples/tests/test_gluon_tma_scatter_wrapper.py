# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only original Python wrapper with semantic Torch parameters.
# @lint-ignore-every AUTODEPS2

"""Check the declared Torch boundary of the original TMA-scatter wrapper."""

import torch
from shape_extensions import Int, IntVar
from tests.test_gluon_tma_scatter import async_scatter_kernel
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


def async_scatter[XMax: IntVar, YMax: IntVar, BX: IntVar, BY: IntVar](
    input: torch.Tensor[[XMax, YMax]],
    x_offsets: torch.Tensor[[BX]],
    y_offset: int,
    src: torch.Tensor[[BX, BY]],
    BLOCK_X: Int[BX],
    BLOCK_Y: Int[BY],
):
    gl_dtype = getattr(gl, str(input.dtype).split(".")[1])
    # When picking the shared memory layout, we use the dimensions of the shared
    # memory descriptor, which will be [BLOCK_X, BLOCK_Y]. But the block shape of the
    # tensor descriptor must still be [1, BLOCK_Y] to be used with async scatter.
    layout = gl.NVMMASharedLayout.get_default_for([BLOCK_X, BLOCK_Y], gl_dtype)
    tensor_desc = TensorDescriptor.from_tensor(input, [1, BLOCK_Y], layout)
    async_scatter_kernel[(1,)](  # E: is not subscriptable
        tensor_desc, x_offsets, y_offset, src, *src.stride(), BLOCK_X
    )


def test_original_scatter_wrapper_boundary[
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
    dest = torch.empty((xmax, ymax), dtype=torch.float32, device="cuda")
    offsets = torch.empty((bx,), dtype=torch.int32, device="cuda")
    wrong_offsets = torch.empty((other,), dtype=torch.int32, device="cuda")
    src = torch.empty((bx, by), dtype=torch.float32, device="cuda")
    wrong_src_rows = torch.empty((other, by), dtype=torch.float32, device="cuda")
    wrong_src_cols = torch.empty((bx, other), dtype=torch.float32, device="cuda")
    async_scatter(dest, offsets, 0, src, bx, by)
    # E: is not assignable
    async_scatter(dest, wrong_offsets, 0, src, bx, by)  # E: is not assignable
    async_scatter(dest, offsets, 0, wrong_src_rows, bx, by)  # E: is not assignable
    async_scatter(dest, offsets, 0, wrong_src_cols, bx, by)  # E: is not assignable
    # These unsafe runtime values and element dtype have no semantic type guard.
    float_offsets = torch.empty((bx,), dtype=torch.float32, device="cuda")
    async_scatter(dest, float_offsets, -1, src, bx, by)

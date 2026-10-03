# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; semantic parameter annotations are not executable Gluon annotations.
# @lint-ignore-every AUTODEPS2

"""Check the complete original Blackwell TMA-scatter kernel body."""

from shape_extensions import Int, IntVar
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    fence_async_shared,
    tma,
)
from triton.language import tensor


@gluon.jit
def async_scatter_kernel[
    XMax: IntVar,
    YMax: IntVar,
    BX: IntVar,
    BY: IntVar,
    StrideX: IntVar,
    StrideY: IntVar,
    Layout: IntVar,
](
    tensor_desc: gl.TmaOutputDescriptor2D[XMax, YMax, 1, BY, Layout],
    x_offsets_ptr: gl.GatherOffsetsPointer1D[BX],
    y_offset: int,
    src_ptr: gl.InMatrixPointer2D[BX, BY, StrideX, StrideY],
    src_stride_x: Int[StrideX],
    src_stride_y: Int[StrideY],
    BLOCK_X: Int[BX],
):
    BLOCK_Y: gl.constexpr = tensor_desc.block_type.shape[1]

    # Load the source using a coalesced layout for efficient load vectorization.
    coalesced_2d_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 32], [1, gl.num_warps()], [1, 0]
    )
    indices_x = (
        gl.arange(0, BLOCK_X, gl.SliceLayout(1, coalesced_2d_layout))[:, None]
        * src_stride_x
    )
    indices_y = (
        gl.arange(0, BLOCK_Y, gl.SliceLayout(0, coalesced_2d_layout))[None, :]
        * src_stride_y
    )
    src = gl.load(src_ptr + indices_x + indices_y)

    # Load the offsets using a coalesced layout for efficient load vectorization.
    coalesced_1d_layout: gl.constexpr = gl.BlockedLayout(
        [1], [32], [gl.num_warps()], [0]
    )
    x_offsets = gl.load(x_offsets_ptr + gl.arange(0, BLOCK_X, coalesced_1d_layout))

    # Convert the offsets layout to a slice layout that satisfies the constraints for `async_scatter`.
    offsets_layout: gl.constexpr = gl.SliceLayout(
        0, gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0])
    )
    x_offsets = gl.convert_layout(x_offsets, offsets_layout)

    # `async_scatter` stores the rows to a tensor descriptor from shared memory.
    smem_src = gl.allocate_shared_memory(
        tensor_desc.dtype, [BLOCK_X, BLOCK_Y], tensor_desc.layout
    )
    smem_src.store(src)
    # An async fence is required between the store to shared memory and the async scatter.
    # Recall from `04-tma` that a fence is needed when using different proxies to access shared
    # memory (generic proxy for the store, and async proxy for the `async_scatter`).
    fence_async_shared()
    tma.async_scatter(tensor_desc, x_offsets, y_offset, smem_src)
    # Wait for the completion of the async scatter using `store_wait`.
    tma.store_wait(0)


def test_scatter_interface[
    XMax: IntVar,
    YMax: IntVar,
    BX: IntVar,
    BY: IntVar,
    Other: IntVar,
    StrideX: IntVar,
    StrideY: IntVar,
    Layout: IntVar,
](
    dest: gl.TmaOutputDescriptor2D[XMax, YMax, 1, BY, Layout],
    wrong_dest_block: gl.TmaOutputDescriptor2D[XMax, YMax, 1, Other, Layout],
    offsets: gl.GatherOffsetsPointer1D[BX],
    wrong_offsets: gl.GatherOffsetsPointer1D[Other],
    src: gl.InMatrixPointer2D[BX, BY, StrideX, StrideY],
    wrong_src: gl.InMatrixPointer2D[Other, BY, StrideX, StrideY],
    sx: Int[StrideX],
    sy: Int[StrideY],
    bx: Int[BX],
) -> None:
    async_scatter_kernel(dest, offsets, 0, src, sx, sy, bx)
    async_scatter_kernel(
        wrong_dest_block,
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


def test_unchecked_blackwell_scatter4_rules[YMax: IntVar, BY: IntVar, Layout: IntVar](
    dest: gl.TmaOutputDescriptor2D[256, YMax, 1, BY, Layout],
    offsets: tensor[[256]],
    src: gl.TmaSharedTile2D[256, BY, Layout],
) -> None:
    # This tutorial's invalid layout and negative Y offset are accepted.
    bad_layout = gl.BlockedLayout([4], [32], [4], [0])
    bad_layout_offsets = gl.convert_layout(offsets, bad_layout)
    tma.async_scatter(dest, bad_layout_offsets, -1, src)

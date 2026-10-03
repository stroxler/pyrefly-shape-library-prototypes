# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; semantic parameter annotations are not executable Gluon annotations.
# @lint-ignore-every AUTODEPS2

"""Check the complete original Blackwell TMA-gather kernel body."""

from shape_extensions import Int, IntVar
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import mbarrier, tma
from triton.language import tensor


@gluon.jit
def async_gather_kernel[
    XMax: IntVar,
    YMax: IntVar,
    BX: IntVar,
    BY: IntVar,
    StrideX: IntVar,
    StrideY: IntVar,
    Layout: IntVar,
](
    out_ptr: gl.OutMatrixPointer2D[BX, BY, StrideX, StrideY],
    out_stride_x: Int[StrideX],
    out_stride_y: Int[StrideY],
    tensor_desc: gl.TmaInputDescriptor2D[XMax, YMax, 1, BY, Layout],
    x_offsets_ptr: gl.GatherOffsetsPointer1D[BX],
    y_offset: int,
    BLOCK_X: Int[BX],
):
    BLOCK_Y: gl.constexpr = tensor_desc.block_type.shape[1]

    # Load the offsets using a coalesced layout for efficient load vectorization.
    coalesced_1d_layout: gl.constexpr = gl.BlockedLayout(
        [1], [32], [gl.num_warps()], [0]
    )
    x_offsets = gl.load(x_offsets_ptr + gl.arange(0, BLOCK_X, coalesced_1d_layout))

    # Convert the offsets layout to a slice layout that satisfies the constraints for `async_gather`.
    offsets_layout: gl.constexpr = gl.SliceLayout(
        0, gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0])
    )
    x_offsets = gl.convert_layout(x_offsets, offsets_layout)

    # `async_gather` loads the rows from a tensor descriptor and writes them into shared memory.
    # The layout of the shared memory descriptor must match the shared memory layout of the tensor descriptor.
    smem_dest = gl.allocate_shared_memory(
        tensor_desc.dtype, [BLOCK_X, BLOCK_Y], tensor_desc.layout
    )

    # `async_gather` is an asynchronous operation that uses an mbarrier to track its completion.
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    # Invoke `mbarrier.expect` on the mbarrier with the number of bytes to be loaded.
    mbarrier.expect(bar, BLOCK_X * tensor_desc.block_type.nbytes)

    # Issue the async gather and wait.
    tma.async_gather(tensor_desc, x_offsets, y_offset, barrier=bar, result=smem_dest)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    # Write the result using a coalesced layout.
    coalesced_2d_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 32], [1, gl.num_warps()], [1, 0]
    )
    out = smem_dest.load(coalesced_2d_layout)

    indices_x = (
        gl.arange(0, BLOCK_X, gl.SliceLayout(1, coalesced_2d_layout))[:, None]
        * out_stride_x
    )
    indices_y = (
        gl.arange(0, BLOCK_Y, gl.SliceLayout(0, coalesced_2d_layout))[None, :]
        * out_stride_y
    )
    gl.store(out_ptr + indices_x + indices_y, out)


def test_gather_interface[
    XMax: IntVar,
    YMax: IntVar,
    BX: IntVar,
    BY: IntVar,
    Other: IntVar,
    StrideX: IntVar,
    StrideY: IntVar,
    Layout: IntVar,
](
    output: gl.OutMatrixPointer2D[BX, BY, StrideX, StrideY],
    wrong_output: gl.OutMatrixPointer2D[Other, BY, StrideX, StrideY],
    desc: gl.TmaInputDescriptor2D[XMax, YMax, 1, BY, Layout],
    wrong_block: gl.TmaInputDescriptor2D[XMax, YMax, 1, Other, Layout],
    offsets: gl.GatherOffsetsPointer1D[BX],
    wrong_offsets: gl.GatherOffsetsPointer1D[Other],
    stride_x: Int[StrideX],
    stride_y: Int[StrideY],
    bx: Int[BX],
) -> None:
    async_gather_kernel(output, stride_x, stride_y, desc, offsets, 0, bx)
    async_gather_kernel(
        wrong_output,
        stride_x,
        stride_y,
        desc,
        offsets,  # E: is not assignable
        0,
        bx,  # E: is not assignable
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
        desc,
        wrong_offsets,  # E: is not assignable
        0,
        bx,
    )


def test_unchecked_blackwell_gather4_rules[YMax: IntVar, BY: IntVar, Layout: IntVar](
    desc: gl.TmaInputDescriptor2D[256, YMax, 1, BY, Layout],
    offsets: tensor[[256]],
    dest: gl.TmaSharedTile2D[256, BY, Layout],
    bar: gl.BarrierBuffer1D,
) -> None:
    # At 256 offsets, the upstream tutorial identifies this layout as invalid.
    invalid_layout = gl.BlockedLayout([4], [32], [4], [0])
    bad_layout_offsets = gl.convert_layout(offsets, invalid_layout)
    tma.async_gather(desc, bad_layout_offsets, 2, bar, dest)

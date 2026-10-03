# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; semantic parameter annotations are not executable Gluon annotations.
# @lint-ignore-every AUTODEPS2

"""Check the unchanged float32 shared-to-tensor-memory copy kernel."""

from shape_extensions import Int, IntVar
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    allocate_tensor_memory,
    fence_async_shared,
    mbarrier,
    tcgen05_commit,
    tcgen05_copy,
    TensorMemoryLayout,
    TensorMemoryTileF32,
)


@gluon.jit
def tcgen05_copy_kernel[
    MRows: IntVar,
    NCols: IntVar,
    InRowStride: IntVar,
    InColStride: IntVar,
    OutRowStride: IntVar,
    OutColStride: IntVar,
](
    in_ptr: gl.InFloat32MatrixPointer2D[MRows, NCols, InRowStride, InColStride],
    in_stride0: Int[InRowStride],
    in_stride1: Int[InColStride],
    out_ptr: gl.OutMatrixPointer2D[MRows, NCols, OutRowStride, OutColStride],
    out_stride0: Int[OutRowStride],
    out_stride1: Int[OutColStride],
    M: Int[MRows],
    N: Int[NCols],
    smem_layout: gl.NVMMASharedLayout,
    tmem_layout: TensorMemoryLayout,
):
    coalesced_2d_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 32], [1, gl.num_warps()], [1, 0]
    )
    offs_m = gl.arange(0, M, gl.SliceLayout(1, coalesced_2d_layout))
    offs_n = gl.arange(0, N, gl.SliceLayout(0, coalesced_2d_layout))

    input = gl.load(
        in_ptr + offs_m[:, None] * in_stride0 + offs_n[None, :] * in_stride1
    )

    # Allocate shared memory and tensor memory with the tile shape [M, N].
    smem = gl.allocate_shared_memory(input.dtype, (M, N), smem_layout)
    tmem = allocate_tensor_memory(input.dtype, (M, N), tmem_layout)

    bar = gl.allocate_shared_memory(
        gl.int64,
        [1],
        gl.constexpr(mbarrier.MBarrierLayout()),  # E: Expected a callable
    )
    mbarrier.init(bar, count=1)

    # Copy data from shared memory to tensor memory.
    smem.store(input)
    # Fence generic and async proxies
    fence_async_shared()
    # Issue the async copy
    tcgen05_copy(smem, tmem)
    # Track completion of the async copy
    tcgen05_commit(bar)
    # Wait for the async copy to complete
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    # Read the data from tensor memory.
    output = tmem.load()

    # Write using a coalesced layout.
    output = gl.convert_layout(output, coalesced_2d_layout)
    gl.store(
        out_ptr + offs_m[:, None] * out_stride0 + offs_n[None, :] * out_stride1, output
    )


def test_tcgen05_copy_interface[
    M: IntVar,
    N: IntVar,
    Other: IntVar,
    InR: IntVar,
    InC: IntVar,
    OutR: IntVar,
    OutC: IntVar,
    Shared: IntVar,
](
    src: gl.InFloat32MatrixPointer2D[M, N, InR, InC],
    wrong_dtype_ptr: gl.InMatrixPointer2D[M, N, InR, InC],
    wrong_src_width: gl.InFloat32MatrixPointer2D[M, Other, InR, InC],
    dst: gl.OutMatrixPointer2D[M, N, OutR, OutC],
    wrong_dst_width: gl.OutMatrixPointer2D[M, Other, OutR, OutC],
    in_r: Int[InR],
    in_c: Int[InC],
    out_r: Int[OutR],
    wrong_out_r: Int[Other],
    out_c: Int[OutC],
    m: Int[M],
    n: Int[N],
    shared_layout: gl.NVMMASharedLayout,
    tensor_layout: TensorMemoryLayout,
    smem: gl.TmaSharedTile2D[M, N, Shared],
    tmem: TensorMemoryTileF32[M, N],
    wrong_tmem_rows: TensorMemoryTileF32[Other, N],
    wrong_tmem_width: TensorMemoryTileF32[M, Other],
) -> None:
    tcgen05_copy_kernel(
        src, in_r, in_c, dst, out_r, out_c, m, n, shared_layout, tensor_layout
    )
    tcgen05_copy_kernel(
        wrong_dtype_ptr,  # E: is not assignable
        in_r,
        in_c,
        dst,
        out_r,
        out_c,
        m,
        n,
        shared_layout,
        tensor_layout,
    )
    tcgen05_copy_kernel(
        wrong_src_width,
        in_r,
        in_c,
        dst,  # E: is not assignable
        out_r,
        out_c,
        m,
        n,  # E: is not assignable
        shared_layout,
        tensor_layout,
    )
    tcgen05_copy_kernel(
        src,
        in_r,
        in_c,
        wrong_dst_width,  # E: is not assignable
        out_r,
        out_c,
        m,
        n,
        shared_layout,
        tensor_layout,
    )
    tcgen05_copy_kernel(
        src,
        in_r,
        in_c,
        dst,
        wrong_out_r,  # E: is not assignable
        out_c,
        m,
        n,
        shared_layout,
        tensor_layout,
    )
    tcgen05_copy(smem, tmem)
    tcgen05_copy(smem, wrong_tmem_rows)  # E: is not assignable
    tcgen05_copy(smem, wrong_tmem_width)  # E: is not assignable
    # The shape contract does not validate Blackwell-specific tensor-memory blocks.
    unsupported_layout = TensorMemoryLayout(block=(16, 1), col_stride=0)
    tcgen05_copy_kernel(
        src, in_r, in_c, dst, out_r, out_c, m, n, shared_layout, unsupported_layout
    )

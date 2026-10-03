# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only, not an executable Gluon kernel.
# @lint-ignore-every AUTODEPS2

"""Check source-order double-buffered WGMMA with original loop body."""

from typing import assert_type

from shape_extensions import IntVar
from tests.test_gluon_wgmma_blocked import pick_wgmma_layout
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_init,
    warpgroup_mma_wait,
)


@gluon.jit
def blocked_matmul_pipelined_kernel[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    LA: IntVar,
    LB: IntVar,
    LC: IntVar,
](
    a_desc: gl.TmaInputDescriptorF16[M, K, BM, BK, LA],
    b_desc: gl.TmaInputDescriptorF16[K, N, BK, BN, LB],
    c_desc: gl.TmaOutputDescriptorF16[M, N, BM, BN, LC],
    num_warps: int,
):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    # Allocate 2 buffers for each A and B.
    a_smem = gl.allocate_shared_memory(
        dtype, [2] + a_desc.block_type.shape, a_desc.layout
    )
    b_smem = gl.allocate_shared_memory(
        dtype, [2] + b_desc.block_type.shape, b_desc.layout
    )
    index = 0

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    mma_layout: gl.constexpr = pick_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps)
    acc = warpgroup_mma_init(
        gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)
    )

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    phase = 0

    for k in range(0, K, BLOCK_K):
        a = a_smem.index(index)
        b = b_smem.index(index)

        mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_load(a_desc, [off_m, k], bar, a)
        tma.async_load(b_desc, [k, off_n], bar, b)
        mbarrier.wait(bar, phase=phase)
        phase ^= 1

        # Since `warpgroup_mma_wait` is a no-op when there are no WGMMAs in
        # flight, we can overlap the WGMMA by waiting first, then issuing the
        # async WGMMA.
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
        acc = warpgroup_mma(a, b, acc, is_async=True)

        # Move to the next buffer. The TMA load will start while the WGMMA is
        # still running.
        index ^= 1

    # Wait for the last WGMMA to complete.
    acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))

    mbarrier.invalidate(bar)

    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    c_smem.store(acc.to(dtype))
    fence_async_shared()
    tma.async_store(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


def test_prefix_concat_is_widened[
    Rows: IntVar,
    Cols: IntVar,
    Other: IntVar,
    Layout: IntVar,
](
    block: gl.TmaBlockShape2D[Rows, Cols],
    wrong_block: gl.TmaBlockShape2D[Other, Cols],
    layout: gl.WgmmaLayoutF16ForBlock[Rows, Cols, Layout],
) -> None:
    assert_type([2] + block, list[int])
    # The layout recovers tile dimensions, but cannot check rank or buffer count.
    assert_type(
        gl.allocate_shared_memory(gl.float16, [3] + block, layout),
        gl.WgmmaSharedRingF16[Rows, Cols, Layout],
    )
    assert_type(
        gl.allocate_shared_memory(gl.float16, [], layout),
        gl.WgmmaSharedRingF16[Rows, Cols, Layout],
    )
    assert_type(
        gl.allocate_shared_memory(gl.float16, [2] + wrong_block, layout),
        gl.WgmmaSharedRingF16[Rows, Cols, Layout],
    )


def test_ring_transfer_checks_descriptor_tile[
    Rows: IntVar,
    Cols: IntVar,
    BR: IntVar,
    BC: IntVar,
    Other: IntVar,
    Layout: IntVar,
    OtherLayout: IntVar,
](
    desc: gl.TmaInputDescriptorF16[Rows, Cols, BR, BC, Layout],
    barrier: gl.BarrierBuffer1D,
    tile: gl.WgmmaSharedF16[BR, BC, Layout],
    wrong_width: gl.WgmmaSharedF16[BR, Other, Layout],
    wrong_layout: gl.WgmmaSharedF16[BR, BC, OtherLayout],
) -> None:
    tma.async_load(desc, [0, 0], barrier, tile)
    tma.async_load(desc, [0, 0], barrier, wrong_width)  # E: No matching overload
    tma.async_load(desc, [0, 0], barrier, wrong_layout)  # E: No matching overload

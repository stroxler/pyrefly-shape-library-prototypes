# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; semantic parameter annotations are not executable Gluon annotations.
# @lint-ignore-every AUTODEPS2

"""Check original fused gather/scatter matmul's offset and tile load helper."""

from typing import assert_type

from shape_extensions import Int, IntVar
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import mbarrier, tma


@gluon.jit
def issue_loads[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    LX: IntVar,
    LW: IntVar,
    Depth: IntVar,
](
    producer: int,
    X_desc: gl.GatherInputDescriptorF16[M, K, BM, BK, LX],
    W_desc: gl.TmaInputDescriptorF16[K, N, BK, BN, LW],
    X_gather_indx_ptr: gl.GatherOffsetsPointer1D[M],
    off_m: gl.GluonTileStart[BM],
    off_n: gl.GluonTileStart[BN],
    k: int,
    bars: gl.TmaBarrierRing2D[Depth],
    x_bufs: gl.WgmmaSharedRingF16[BM, BK, LX],
    w_bufs: gl.WgmmaSharedRingF16[BK, BN, LW],
    BLOCK_M: Int[BM],
    num_buffers: Int[Depth],
    pred=True,
):
    # Load the M dimension offsets for the X tensor tile. We expect the load to be small
    # enough (no more than 128 elements) that we don't need to use a coalesced layout. Load directly into the layout
    # required by `async_gather` to avoid the layout conversion.
    gather_indx_layout: gl.constexpr = gl.SliceLayout(
        0, gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0])
    )
    offs_x_m = gl.load(
        X_gather_indx_ptr + off_m + gl.arange(0, BLOCK_M, gather_indx_layout)
    )

    index = producer % num_buffers
    producer += 1
    bar = bars.index(index)

    # The W tensor tile is loaded using a regular `async_load`.
    mbarrier.expect(bar, W_desc.block_type.nbytes + BLOCK_M * X_desc.block_type.nbytes)
    tma.async_gather(X_desc, offs_x_m, k, bar, x_bufs.index(index), pred)
    tma.async_load(W_desc, [k, off_n], bar, w_bufs.index(index), pred)
    return producer


def test_gather_scatter_issue_loads_interface[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Other: IntVar,
    LX: IntVar,
    LW: IntVar,
    Depth: IntVar,
](
    x: gl.GatherInputDescriptorF16[M, K, BM, BK, LX],
    wrong_x_bk: gl.GatherInputDescriptorF16[M, K, BM, Other, LX],
    w: gl.TmaInputDescriptorF16[K, N, BK, BN, LW],
    wrong_w_bn: gl.TmaInputDescriptorF16[K, N, BK, Other, LW],
    wrong_w_host_k: gl.TmaInputDescriptorF16[Other, N, BK, BN, LW],
    offs: gl.GatherOffsetsPointer1D[M],
    wrong_offs_host: gl.GatherOffsetsPointer1D[Other],
    wrong_off_m: gl.GluonTileStart[Other],
    off_m: gl.GluonTileStart[BM],
    off_n: gl.GluonTileStart[BN],
    bars: gl.TmaBarrierRing2D[Depth],
    xring: gl.WgmmaSharedRingF16[BM, BK, LX],
    wrong_xring: gl.WgmmaSharedRingF16[Other, BK, LX],
    wring: gl.WgmmaSharedRingF16[BK, BN, LW],
    wrong_wring: gl.WgmmaSharedRingF16[BK, Other, LW],
    block_m: Int[BM],
    depth: Int[Depth],
) -> None:
    assert_type(
        issue_loads(0, x, w, offs, off_m, off_n, 0, bars, xring, wring, block_m, depth),
        int,
    )
    issue_loads(
        0,
        wrong_x_bk,
        w,  # E: is not assignable
        offs,
        off_m,
        off_n,
        0,
        bars,
        xring,  # E: is not assignable
        wring,  # E: is not assignable
        block_m,
        depth,
    )
    issue_loads(
        0,
        x,
        wrong_w_bn,
        offs,
        off_m,
        off_n,  # E: is not assignable
        0,
        bars,
        xring,
        wring,  # E: is not assignable
        block_m,
        depth,
    )
    issue_loads(
        0,
        x,
        w,
        offs,
        wrong_off_m,  # E: is not assignable
        off_n,
        0,
        bars,
        xring,
        wring,
        block_m,
        depth,
    )
    issue_loads(
        0,
        x,
        w,
        offs,
        off_m,
        off_n,
        0,
        bars,
        wrong_xring,  # E: is not assignable
        wring,
        block_m,
        depth,
    )
    issue_loads(
        0,
        x,
        w,
        offs,
        off_m,
        off_n,
        0,
        bars,
        xring,
        wrong_wring,  # E: is not assignable
        block_m,
        depth,
    )
    issue_loads(
        0,
        x,
        wrong_w_host_k,  # E: is not assignable
        offs,
        off_m,
        off_n,
        0,
        bars,
        xring,
        wring,
        block_m,
        depth,
    )
    issue_loads(
        0,
        x,
        w,
        wrong_offs_host,  # E: is not assignable
        off_m,
        off_n,
        0,
        bars,
        xring,
        wring,
        block_m,
        depth,
    )

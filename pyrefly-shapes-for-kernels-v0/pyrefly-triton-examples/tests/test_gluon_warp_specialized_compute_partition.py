# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; semantic parameter annotations are not executable Gluon annotations.
# @lint-ignore-every AUTODEPS2

"""Check the original warp-specialized compute worker's A+B-to-C tile body."""

from shape_extensions import Int, IntVar
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
)


@gluon.jit
def compute_partition[
    Cols: IntVar,
    BR: IntVar,
    BC: IntVar,
    LA: IntVar,
    LB: IntVar,
    LC: IntVar,
    LoadDepth: IntVar,
    StoreDepth: IntVar,
](
    barriers: tuple[
        gl.TmaBarrierRing2D[LoadDepth],
        gl.TmaBarrierRing2D[LoadDepth],
        gl.TmaBarrierRing2D[StoreDepth],
        gl.TmaBarrierRing2D[StoreDepth],
    ],
    buffers: tuple[
        gl.TmaSharedRing2D[LoadDepth, BR, BC, LA],
        gl.TmaSharedRing2D[LoadDepth, BR, BC, LB],
        gl.TmaSharedRing2D[StoreDepth, BR, BC, LC],
    ],
    ynumel: Int[Cols],
    YBLOCK: Int[BC],
    layout: gl.Layout2D,
):
    load_empty_bars, load_ready_bars, c_empty_bars, c_ready_bars = barriers
    a_bufs, b_bufs, c_bufs = buffers

    num_load_buffers: gl.constexpr = a_bufs.type.shape[0]
    num_store_buffers: gl.constexpr = c_bufs.type.shape[0]

    for i in range(gl.cdiv(ynumel, YBLOCK)):
        load_index = i % num_load_buffers
        load_phase = i // num_load_buffers & 1
        a_buf = a_bufs.index(load_index)
        b_buf = b_bufs.index(load_index)
        load_ready_bar = load_ready_bars.index(load_index)
        load_empty_bar = load_empty_bars.index(load_index)

        # Wait for the operands then consume them.
        mbarrier.wait(load_ready_bar, load_phase)
        a_val = a_buf.load(layout)
        b_val = b_buf.load(layout)
        # Fence before signalling the load partitions so the TMA load is
        # ordered with the shared load.
        fence_async_shared()
        mbarrier.arrive(load_empty_bar, count=1)

        c_val = a_val + b_val

        store_idx = i % num_store_buffers
        store_phase = i // num_store_buffers & 1
        c_buf = c_bufs.index(store_idx)
        c_empty_bar = c_empty_bars.index(store_idx)
        c_ready_bar = c_ready_bars.index(store_idx)

        mbarrier.wait(c_empty_bar, store_phase ^ 1)
        c_buf.store(c_val)
        # Fence to order with TMA store.
        fence_async_shared()
        mbarrier.arrive(c_ready_bar, count=1)


def test_compute_partition_tile_contract[
    Cols: IntVar,
    BR: IntVar,
    BC: IntVar,
    Other: IntVar,
    LA: IntVar,
    LB: IntVar,
    LC: IntVar,
    LoadDepth: IntVar,
    StoreDepth: IntVar,
](
    load_empty_bars: gl.TmaBarrierRing2D[LoadDepth],
    load_ready_bars: gl.TmaBarrierRing2D[LoadDepth],
    c_empty_bars: gl.TmaBarrierRing2D[StoreDepth],
    c_ready_bars: gl.TmaBarrierRing2D[StoreDepth],
    a_bufs: gl.TmaSharedRing2D[LoadDepth, BR, BC, LA],
    b_bufs: gl.TmaSharedRing2D[LoadDepth, BR, BC, LB],
    wrong_b_width: gl.TmaSharedRing2D[LoadDepth, BR, Other, LB],
    wrong_b_depth: gl.TmaSharedRing2D[StoreDepth, BR, BC, LB],
    c_bufs: gl.TmaSharedRing2D[StoreDepth, BR, BC, LC],
    wrong_c_width: gl.TmaSharedRing2D[StoreDepth, BR, Other, LC],
    ynumel: Int[Cols],
    other_ynumel: Int[Other],
    yblock: Int[BC],
    wrong_yblock: Int[Other],
    layout: gl.Layout2D,
) -> None:
    barriers = (load_empty_bars, load_ready_bars, c_empty_bars, c_ready_bars)
    buffers = (a_bufs, b_bufs, c_bufs)
    compute_partition(barriers, buffers, ynumel, yblock, layout)
    compute_partition(
        barriers,
        (a_bufs, wrong_b_width, c_bufs),  # E: is not assignable
        ynumel,
        yblock,
        layout,
    )
    compute_partition(
        barriers,
        (a_bufs, wrong_b_depth, c_bufs),  # E: is not assignable
        ynumel,
        yblock,
        layout,
    )
    compute_partition(
        barriers,
        (a_bufs, b_bufs, wrong_c_width),  # E: is not assignable
        ynumel,
        yblock,
        layout,
    )
    compute_partition(
        barriers,
        buffers,
        ynumel,
        wrong_yblock,  # E: is not assignable
        layout,
    )
    # No host extent is retained by the shared-memory ring parameters.
    compute_partition(barriers, buffers, other_ynumel, yblock, layout)

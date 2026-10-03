# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; semantic parameter annotations are not executable Gluon annotations.
# @lint-ignore-every AUTODEPS2

"""Check the original first warp-specialized TMA load worker's full body."""

from typing import assert_type

from shape_extensions import Int, IntVar
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma


@gluon.jit
def load_partition[
    Rows: IntVar,
    Cols: IntVar,
    BR: IntVar,
    BC: IntVar,
    LA: IntVar,
    LB: IntVar,
    CRows: IntVar,
    CCols: IntVar,
    CBR: IntVar,
    CBC: IntVar,
    LC: IntVar,
    LoadDepth: IntVar,
    StoreDepth: IntVar,
](
    descs: tuple[
        gl.TmaInputDescriptor2D[Rows, Cols, BR, BC, LA],
        gl.TmaInputDescriptor2D[Rows, Cols, BR, BC, LB],
        gl.TmaOutputDescriptor2D[CRows, CCols, CBR, CBC, LC],
    ],
    barriers: tuple[
        gl.TmaBarrierRing2D[LoadDepth],
        gl.TmaBarrierRing2D[LoadDepth],
        gl.TmaBarrierRing2D[StoreDepth],
        gl.TmaBarrierRing2D[StoreDepth],
    ],
    buffers: tuple[
        gl.TmaSharedRing2D[LoadDepth, BR, BC, LA],
        gl.TmaSharedRing2D[LoadDepth, BR, BC, LB],
        gl.TmaSharedRing2D[StoreDepth, CBR, CBC, LC],
    ],
    xoff: gl.GluonTileStart[BR],
    numel: tuple[Int[Rows], Int[Cols]],
    YBLOCK: Int[BC],
):
    # Unpack the arguments.
    a_desc, b_desc, c_desc = descs
    load_empty_bars, load_ready_bars, c_empty_bars, c_ready_bars = barriers
    a_bufs, b_bufs, c_bufs = buffers
    xnumel, ynumel = numel

    num_buffers: gl.constexpr = a_bufs.type.shape[0]

    # All the partitions need to have the same number of inner loop iterations.
    for i in range(gl.cdiv(ynumel, YBLOCK)):
        index = i % num_buffers
        phase = i // num_buffers & 1
        a_buf = a_bufs.index(index)
        b_buf = b_bufs.index(index)
        load_empty_bar = load_empty_bars.index(index)
        load_ready_bar = load_ready_bars.index(index)

        # Wait for the current buffers to be empty. Recall that mbarriers are
        # initialized to phase 1 complete, so we wait starting with phase 1 to
        # allow the producer to begin filling the pipeline.
        mbarrier.wait(load_empty_bar, phase ^ 1)

        # Okay, a_buf and b_buf are empty. Issue the TMA loads, and have them
        # signal the operand buffers as ready when they complete.
        yoff = i * YBLOCK
        mbarrier.expect(
            load_ready_bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes
        )
        tma.async_load(a_desc, [xoff, yoff], load_ready_bar, a_buf)
        tma.async_load(b_desc, [xoff, yoff], load_ready_bar, b_buf)


def test_load_worker_block_and_ring_contract[
    Rows: IntVar,
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
    a_desc: gl.TmaInputDescriptor2D[Rows, Cols, BR, BC, LA],
    b_desc: gl.TmaInputDescriptor2D[Rows, Cols, BR, BC, LB],
    wrong_b_block: gl.TmaInputDescriptor2D[Rows, Cols, BR, Other, LB],
    c_desc: gl.TmaOutputDescriptor2D[Rows, Cols, BR, BC, LC],
    wrong_c_host: gl.TmaOutputDescriptor2D[Rows, Other, BR, BC, LC],
    load_empty_bars: gl.TmaBarrierRing2D[LoadDepth],
    load_ready_bars: gl.TmaBarrierRing2D[LoadDepth],
    wrong_ready_bars: gl.TmaBarrierRing2D[StoreDepth],
    c_empty_bars: gl.TmaBarrierRing2D[StoreDepth],
    c_ready_bars: gl.TmaBarrierRing2D[StoreDepth],
    a_bufs: gl.TmaSharedRing2D[LoadDepth, BR, BC, LA],
    b_bufs: gl.TmaSharedRing2D[LoadDepth, BR, BC, LB],
    wrong_b_ring: gl.TmaSharedRing2D[StoreDepth, BR, BC, LB],
    c_bufs: gl.TmaSharedRing2D[StoreDepth, BR, BC, LC],
    xoff: gl.GluonTileStart[BR],
    wrong_xoff: gl.GluonTileStart[Other],
    rows: Int[Rows],
    cols: Int[Cols],
    yblock: Int[BC],
    wrong_yblock: Int[Other],
) -> None:
    assert_type(a_bufs.type.shape[0], Int[LoadDepth])
    descs = (a_desc, b_desc, c_desc)
    barriers = (load_empty_bars, load_ready_bars, c_empty_bars, c_ready_bars)
    buffers = (a_bufs, b_bufs, c_bufs)
    load_partition(descs, barriers, buffers, xoff, (rows, cols), yblock)
    load_partition(
        (a_desc, wrong_b_block, c_desc),  # E: is not assignable
        barriers,
        buffers,
        xoff,
        (rows, cols),
        yblock,
    )
    load_partition(
        descs,
        barriers,
        (a_bufs, wrong_b_ring, c_bufs),  # E: is not assignable
        xoff,
        (rows, cols),
        yblock,
    )
    load_partition(
        descs,
        (  # E: is not assignable
            load_empty_bars,
            wrong_ready_bars,
            c_empty_bars,
            c_ready_bars,
        ),
        buffers,
        xoff,
        (rows, cols),
        yblock,
    )
    load_partition(
        descs,
        barriers,
        buffers,
        wrong_xoff,  # E: is not assignable
        (rows, cols),
        yblock,
    )
    load_partition(
        descs,
        barriers,
        buffers,
        xoff,
        (rows, cols),
        wrong_yblock,  # E: is not assignable
    )
    # The original worker never reads the C descriptor or its output ring.
    load_partition(
        (a_desc, b_desc, wrong_c_host),
        barriers,
        buffers,
        xoff,
        (rows, cols),
        yblock,
    )

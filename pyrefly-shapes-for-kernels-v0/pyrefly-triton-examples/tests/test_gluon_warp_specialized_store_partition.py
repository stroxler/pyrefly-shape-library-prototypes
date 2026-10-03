# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; semantic parameter annotations are not executable Gluon annotations.
# @lint-ignore-every AUTODEPS2

"""Check the original warp-specialized TMA output worker's full body."""

from shape_extensions import Int, IntVar
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma


@gluon.jit
def store_partition[
    InputRows: IntVar,
    InputCols: IntVar,
    IBR: IntVar,
    IBC: IntVar,
    LA: IntVar,
    LB: IntVar,
    Rows: IntVar,
    Cols: IntVar,
    BR: IntVar,
    BC: IntVar,
    LC: IntVar,
    LoadDepth: IntVar,
    StoreDepth: IntVar,
](
    descs: tuple[
        gl.TmaInputDescriptor2D[InputRows, InputCols, IBR, IBC, LA],
        gl.TmaInputDescriptor2D[InputRows, InputCols, IBR, IBC, LB],
        gl.TmaOutputDescriptor2D[Rows, Cols, BR, BC, LC],
    ],
    barriers: tuple[
        gl.TmaBarrierRing2D[LoadDepth],
        gl.TmaBarrierRing2D[LoadDepth],
        gl.TmaBarrierRing2D[StoreDepth],
        gl.TmaBarrierRing2D[StoreDepth],
    ],
    buffers: tuple[
        gl.TmaSharedRing2D[LoadDepth, IBR, IBC, LA],
        gl.TmaSharedRing2D[LoadDepth, IBR, IBC, LB],
        gl.TmaSharedRing2D[StoreDepth, BR, BC, LC],
    ],
    xoff: gl.GluonTileStart[BR],
    numel: tuple[Int[Rows], Int[Cols]],
    YBLOCK: Int[BC],
):
    a_desc, b_desc, c_desc = descs
    load_empty_bars, load_ready_bars, c_empty_bars, c_ready_bars = barriers
    a_bufs, b_bufs, c_bufs = buffers
    xnumel, ynumel = numel

    # This partition consumes the addition result, passed over smem, and stores
    # them to global memory.
    num_buffers: gl.constexpr = c_bufs.type.shape[0]
    # We will keep `num_buffers-1` stores in flight by software pipelining.
    outstanding_stores: gl.constexpr = num_buffers - 1

    for i in range(gl.cdiv(ynumel, YBLOCK)):
        index = i % num_buffers
        phase = i // num_buffers & 1
        c_buf = c_bufs.index(index)
        c_ready_bar = c_ready_bars.index(index)

        # Wait for the compute partition to produce c.
        mbarrier.wait(c_ready_bar, phase)
        yoff = i * YBLOCK
        tma.async_store(c_desc, [xoff, yoff], c_buf)

        tma.store_wait(outstanding_stores)
        c_empty_bar = c_empty_bars.index((i - outstanding_stores) % num_buffers)
        # Signal the compute partition that the buffer `outstanding_stores`
        # iterations ago is consumed, predicated on there having been at least
        # that many outstanding stores.
        mbarrier.arrive(c_empty_bar, count=1, pred=i >= outstanding_stores)

    # Since we waited for the last value of c, all the other partitions have
    # exited by now. We just need the final stores to complete.
    tma.store_wait(0)


def test_declared_output_worker_contract[
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
    wrong_a_host: gl.TmaInputDescriptor2D[Rows, Other, BR, BC, LA],
    wrong_b_host: gl.TmaInputDescriptor2D[Rows, Other, BR, BC, LB],
    c_desc: gl.TmaOutputDescriptor2D[Rows, Cols, BR, BC, LC],
    wrong_c_block: gl.TmaOutputDescriptor2D[Rows, Cols, BR, Other, LC],
    wrong_c_host: gl.TmaOutputDescriptor2D[Rows, Other, BR, BC, LC],
    load_empty_bars: gl.TmaBarrierRing2D[LoadDepth],
    load_ready_bars: gl.TmaBarrierRing2D[LoadDepth],
    c_empty_bars: gl.TmaBarrierRing2D[StoreDepth],
    c_ready_bars: gl.TmaBarrierRing2D[StoreDepth],
    a_bufs: gl.TmaSharedRing2D[LoadDepth, BR, BC, LA],
    b_bufs: gl.TmaSharedRing2D[LoadDepth, BR, BC, LB],
    c_bufs: gl.TmaSharedRing2D[StoreDepth, BR, BC, LC],
    wrong_c_buf: gl.TmaSharedRing2D[StoreDepth, BR, Other, LC],
    xoff: gl.GluonTileStart[BR],
    wrong_xoff: gl.GluonTileStart[Other],
    rows: Int[Rows],
    cols: Int[Cols],
    yblock: Int[BC],
) -> None:
    barriers = (load_empty_bars, load_ready_bars, c_empty_bars, c_ready_bars)
    buffers = (a_bufs, b_bufs, c_bufs)
    descs = (a_desc, b_desc, c_desc)
    store_partition(descs, barriers, buffers, xoff, (rows, cols), yblock)
    store_partition(
        (a_desc, b_desc, wrong_c_block),
        barriers,
        buffers,  # E: is not assignable
        xoff,
        (rows, cols),
        yblock,  # E: is not assignable
    )
    store_partition(
        descs,
        barriers,
        (a_bufs, b_bufs, wrong_c_buf),  # E: is not assignable
        xoff,
        (rows, cols),
        yblock,
    )
    store_partition(
        descs,
        barriers,
        buffers,
        wrong_xoff,  # E: is not assignable
        (rows, cols),
        yblock,
    )
    # Its declared C extent must match numel, not the independently typed inputs.
    store_partition(
        (a_desc, b_desc, wrong_c_host),
        barriers,
        buffers,
        xoff,
        (rows, cols),  # E: is not assignable
        yblock,
    )
    store_partition(
        (wrong_a_host, wrong_b_host, c_desc),
        barriers,
        buffers,
        xoff,
        (rows, cols),
        yblock,
    )

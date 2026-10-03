# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; semantic parameter annotations are not executable Gluon annotations.
# @lint-ignore-every AUTODEPS2

"""Check the full original host-launched warp-specialized vector-add body."""

from typing import assert_type

from shape_extensions import Int, IntListLiteral, IntVar
from tests.test_gluon_warp_specialized_compute_partition import compute_partition
from tests.test_gluon_warp_specialized_load_partition import load_partition
from tests.test_gluon_warp_specialized_store_partition import store_partition
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier


@gluon.jit
def elementwise_add_warp_specialized_kernel[
    Rows: IntVar,
    Cols: IntVar,
    BR: IntVar,
    BC: IntVar,
    LA: IntVar,
    LB: IntVar,
    LC: IntVar,
    LoadDepth: IntVar,
    StoreDepth: IntVar,
](  #
    a_desc: gl.TmaInputDescriptor2D[Rows, Cols, BR, BC, LA],
    b_desc: gl.TmaInputDescriptor2D[Rows, Cols, BR, BC, LB],
    c_desc: gl.TmaOutputDescriptor2D[Rows, Cols, BR, BC, LC],
    xnumel: Int[Rows],
    ynumel: Int[Cols],
    XBLOCK: Int[BR],
    YBLOCK: Int[BC],
    num_load_buffers: Int[LoadDepth],
    num_store_buffers: Int[StoreDepth],
    num_warps: int,
):
    # Pick a layout that makes it easy to avoid bank conflicts.
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0])

    # Allocate all the buffers and barriers.
    a_bufs = gl.allocate_shared_memory(
        a_desc.dtype, [num_load_buffers] + a_desc.block_type.shape, a_desc.layout
    )
    b_bufs = gl.allocate_shared_memory(
        b_desc.dtype, [num_load_buffers] + b_desc.block_type.shape, b_desc.layout
    )
    c_bufs = gl.allocate_shared_memory(
        c_desc.dtype, [num_store_buffers] + c_desc.block_type.shape, c_desc.layout
    )
    load_empty_bars = gl.allocate_shared_memory(
        gl.int64, [num_load_buffers, 1], mbarrier.MBarrierLayout()
    )
    load_ready_bars = gl.allocate_shared_memory(
        gl.int64, [num_load_buffers, 1], mbarrier.MBarrierLayout()
    )
    c_empty_bars = gl.allocate_shared_memory(
        gl.int64, [num_store_buffers, 1], mbarrier.MBarrierLayout()
    )
    c_ready_bars = gl.allocate_shared_memory(
        gl.int64, [num_store_buffers, 1], mbarrier.MBarrierLayout()
    )

    for i in gl.static_range(num_load_buffers):
        mbarrier.init(load_empty_bars.index(i), count=1)
        mbarrier.init(load_ready_bars.index(i), count=1)
    for i in gl.static_range(num_store_buffers):
        mbarrier.init(c_empty_bars.index(i), count=1)
        mbarrier.init(c_ready_bars.index(i), count=1)

    descs = (a_desc, b_desc, c_desc)
    barriers = (load_empty_bars, load_ready_bars, c_empty_bars, c_ready_bars)
    buffers = (a_bufs, b_bufs, c_bufs)
    numel = (xnumel, ynumel)

    pid = gl.program_id(0)
    xoff = pid * XBLOCK

    # `gl.warp_specialize` declares a warp-specialized section of the kernel.
    # It accepts arguments for the default partition function, which can include
    # tensors, and the default partition function. It takes arguments for all
    # the worker partitions, which cannot include tensors, and takes a list of
    # worker partition functions. The warps and register budget for each
    # partition are passed as lists.
    #
    # Note that warp and register allocation on NVIDIA GPUs is by warpgroup,
    # which are 4 consecutive warps. The number of warps used by a kernel is
    # rounded to the nearest multiple of 4. The compiler tries to organize the
    # warps to reduce the amount of registers allocated. The default partition
    # receives whatever registers are left over, based on `maxnreg` passed to
    # the kernel.
    gl.warp_specialize(
        [
            (compute_partition, (barriers, buffers, ynumel, YBLOCK, layout)),
            (load_partition, (descs, barriers, buffers, xoff, numel, YBLOCK)),
            (store_partition, (descs, barriers, buffers, xoff, numel, YBLOCK)),
        ],
        [1, 1],
        [24, 24],
    )


def test_declared_kernel_host_dimensions[
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
    c_desc: gl.TmaOutputDescriptor2D[Rows, Cols, BR, BC, LC],
    wrong_b_host: gl.TmaInputDescriptor2D[Rows, Other, BR, BC, LB],
    wrong_c_block: gl.TmaOutputDescriptor2D[Rows, Cols, BR, Other, LC],
    rows: Int[Rows],
    cols: Int[Cols],
    br: Int[BR],
    bc: Int[BC],
    load_depth: Int[LoadDepth],
    store_depth: Int[StoreDepth],
) -> None:
    elementwise_add_warp_specialized_kernel(
        a_desc, b_desc, c_desc, rows, cols, br, bc, load_depth, store_depth, 4
    )
    elementwise_add_warp_specialized_kernel(
        a_desc,
        wrong_b_host,  # E: is not assignable
        c_desc,
        rows,
        cols,
        br,
        bc,
        load_depth,
        store_depth,
        4,
    )
    elementwise_add_warp_specialized_kernel(
        a_desc,
        b_desc,
        wrong_c_block,  # E: is not assignable
        rows,
        cols,
        br,
        bc,
        load_depth,
        store_depth,
        4,
    )


def test_unchecked_ring_prefix_length[Depth: IntVar, BR: IntVar, BC: IntVar](
    depth: Int[Depth], block: gl.TmaBlockType2D[BR, BC]
) -> None:
    # The reflected-list contract cannot establish that the prefix has length one.
    assert_type([depth, depth] + block.shape, IntListLiteral[[Depth, BR, BC]])


def test_warp_partition_pairing[
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
    descs: tuple[
        gl.TmaInputDescriptor2D[Rows, Cols, BR, BC, LA],
        gl.TmaInputDescriptor2D[Rows, Cols, BR, BC, LB],
        gl.TmaOutputDescriptor2D[Rows, Cols, BR, BC, LC],
    ],
    bars: tuple[
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
    xoff: gl.GluonTileStart[BR],
    rows: Int[Rows],
    cols: Int[Cols],
    bc: Int[BC],
    other: Int[Other],
    layout: gl.Layout2D,
) -> None:
    gl.warp_specialize(
        [(load_partition, (descs, bars, buffers, xoff, (rows, cols), bc))],
        [1],
        [24],
    )
    gl.warp_specialize(
        [  # E: is not assignable
            (load_partition, (descs, bars, buffers, xoff, (rows, cols), other))
        ],
        [1],
        [24],
    )
    # A list of just one worker is allowed here: completeness is not checked.
    gl.warp_specialize(
        [(compute_partition, (bars, buffers, cols, bc, layout))], [1], [24]
    )

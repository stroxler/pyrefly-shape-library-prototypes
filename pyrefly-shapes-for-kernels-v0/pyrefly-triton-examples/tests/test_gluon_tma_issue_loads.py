# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The helper body is copied from Triton's python/tutorials/gluon/04-tma.py.
# Triton's LICENSE includes the following notice:
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
# THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Static-only, not an executable Gluon helper.
# @lint-ignore-every AUTODEPS2

"""Check 2D TMA async loads into separate A/B shared-memory rings."""

from typing import assert_type

from shape_extensions import Int, IntVar
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma


@gluon.jit
def issue_loads[
    Rows: IntVar,
    Cols: IntVar,
    BlockRows: IntVar,
    BlockCols: IntVar,
    Buffers: IntVar,
    Layout: IntVar,
](
    copy_index: int,
    a_desc: gl.TmaInputDescriptor2D[Rows, Cols, BlockRows, BlockCols, Layout],
    b_desc: gl.TmaInputDescriptor2D[Rows, Cols, BlockRows, BlockCols, Layout],
    a_smem: gl.TmaSharedRing2D[Buffers, BlockRows, BlockCols, Layout],
    b_smem: gl.TmaSharedRing2D[Buffers, BlockRows, BlockCols, Layout],
    bars: gl.TmaBarrierRing2D[Buffers],
    xoff: gl.GluonTileStart[BlockRows],
    YBLOCK: Int[BlockCols],
    num_buffers: Int[Buffers],
) -> int:
    # Track completion of both TMA reads with the same mbarrier.
    yoff = copy_index * YBLOCK
    bar = bars.index(copy_index % num_buffers)
    mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
    tma.async_load(a_desc, [xoff, yoff], bar, a_smem.index(copy_index % num_buffers))
    tma.async_load(b_desc, [xoff, yoff], bar, b_smem.index(copy_index % num_buffers))
    return copy_index + 1


def test_issue_loads_boundary[
    Rows: IntVar,
    Cols: IntVar,
    Other: IntVar,
    BR: IntVar,
    BC: IntVar,
    Buffers: IntVar,
    Layout: IntVar,
    OtherLayout: IntVar,
](
    a_desc: gl.TmaInputDescriptor2D[Rows, Cols, BR, BC, Layout],
    b_desc: gl.TmaInputDescriptor2D[Rows, Cols, BR, BC, Layout],
    b_bad_role: gl.TmaOutputDescriptor2D[Rows, Cols, BR, BC, Layout],
    b_bad_host_width: gl.TmaInputDescriptor2D[Rows, Other, BR, BC, Layout],
    b_bad_block: gl.TmaInputDescriptor2D[Rows, Cols, BR, Other, Layout],
    b_bad_layout: gl.TmaInputDescriptor2D[Rows, Cols, BR, BC, OtherLayout],
    a_ring: gl.TmaSharedRing2D[Buffers, BR, BC, Layout],
    b_ring: gl.TmaSharedRing2D[Buffers, BR, BC, Layout],
    b_bad_ring_layout: gl.TmaSharedRing2D[Buffers, BR, BC, OtherLayout],
    b_bad_ring_block: gl.TmaSharedRing2D[Buffers, BR, Other, Layout],
    b_bad_ring_count: gl.TmaSharedRing2D[Other, BR, BC, Layout],
    bars: gl.TmaBarrierRing2D[Buffers],
    wrong_bars: gl.TmaBarrierRing2D[Other],
    xoff: gl.GluonTileStart[BR],
    wrong_xoff: gl.GluonTileStart[Other],
    yblock: Int[BC],
    wrong_yblock: Int[Other],
    buffers: Int[Buffers],
    wrong_buffers: Int[Other],
) -> None:
    assert_type(
        issue_loads(0, a_desc, b_desc, a_ring, b_ring, bars, xoff, yblock, buffers), int
    )
    issue_loads(
        0,
        a_desc,
        # E: is not assignable
        b_bad_host_width,
        a_ring,
        b_ring,
        bars,
        xoff,
        yblock,
        buffers,
    )
    issue_loads(
        0,
        a_desc,
        # E: is not assignable
        b_bad_role,
        a_ring,
        b_ring,
        bars,
        xoff,
        yblock,
        buffers,
    )
    issue_loads(
        0,
        a_desc,
        # E: is not assignable
        b_bad_block,
        a_ring,
        b_ring,
        bars,
        xoff,
        yblock,
        buffers,
    )
    issue_loads(
        0,
        a_desc,
        # E: is not assignable
        b_bad_layout,
        a_ring,
        b_ring,
        bars,
        xoff,
        yblock,
        buffers,
    )
    issue_loads(
        0,
        a_desc,
        b_desc,
        a_ring,
        # E: is not assignable
        b_bad_ring_layout,
        bars,
        xoff,
        yblock,
        buffers,
    )
    issue_loads(
        0,
        a_desc,
        b_desc,
        a_ring,
        # E: is not assignable
        b_bad_ring_block,
        bars,
        xoff,
        yblock,
        buffers,
    )
    issue_loads(
        0,
        a_desc,
        b_desc,
        a_ring,
        # E: is not assignable
        b_bad_ring_count,
        bars,
        xoff,
        yblock,
        buffers,
    )
    issue_loads(
        0,
        a_desc,
        b_desc,
        a_ring,
        b_ring,
        # E: is not assignable
        wrong_bars,
        xoff,
        yblock,
        buffers,
    )
    issue_loads(
        0,
        a_desc,
        b_desc,
        a_ring,
        b_ring,
        bars,
        # E: is not assignable
        wrong_xoff,
        yblock,
        buffers,
    )
    issue_loads(
        0,
        a_desc,
        b_desc,
        a_ring,
        b_ring,
        bars,
        xoff,
        # E: is not assignable
        wrong_yblock,
        buffers,
    )
    issue_loads(
        0,
        a_desc,
        b_desc,
        a_ring,
        b_ring,
        bars,
        xoff,
        yblock,
        # E: is not assignable
        wrong_buffers,
    )


def test_allocation_and_transfer[
    Rows: IntVar,
    Cols: IntVar,
    BR: IntVar,
    BC: IntVar,
    Buffers: IntVar,
    Other: IntVar,
    Layout: IntVar,
    OtherLayout: IntVar,
](
    a_desc: gl.TmaInputDescriptor2D[Rows, Cols, BR, BC, Layout],
    wrong_role: gl.TmaOutputDescriptor2D[Rows, Cols, BR, BC, Layout],
    b_bad_layout: gl.TmaInputDescriptor2D[Rows, Cols, BR, BC, OtherLayout],
    buffers: Int[Buffers],
    bx: Int[BR],
    by: Int[BC],
    other: Int[Other],
    xoff: gl.GluonTileStart[BR],
) -> None:
    ring = gl.allocate_shared_memory(a_desc.dtype, [buffers, bx, by], a_desc.layout)
    bars = gl.allocate_shared_memory(gl.int64, [buffers, 1], mbarrier.MBarrierLayout())
    assert_type(ring, gl.TmaSharedRing2D[Buffers, BR, BC, Layout])
    assert_type(bars, gl.TmaBarrierRing2D[Buffers])
    bar = bars.index(0)
    tma.async_load(a_desc, [xoff, 0], bar, ring.index(0))
    tma.async_load(wrong_role, [xoff, 0], bar, ring.index(0))  # E: No matching overload
    wrong_width_ring = gl.allocate_shared_memory(
        a_desc.dtype, [buffers, bx, other], a_desc.layout
    )
    tma.async_load(  # E: No matching overload
        a_desc, [xoff, 0], bar, wrong_width_ring.index(0)
    )
    tma.async_load(  # E: No matching overload
        b_bad_layout, [xoff, 0], bar, ring.index(0)
    )
    tma.async_load(
        a_desc, [0, xoff], bar, ring.index(0)
    )  # Coordinate order is not checked.
    tma.async_load(
        a_desc, [xoff], bar, ring.index(0)
    )  # Coordinate rank is not checked.
    bars.index(-1)  # Index bounds are not checked.

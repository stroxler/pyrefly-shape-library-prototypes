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

"""Check 2D TMA input tiles, sum, and output tile for a pipeline stage."""

from typing import assert_type

from shape_extensions import Int, IntVar
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
)
from triton.language import tensor


@gluon.jit
def perform_add[
    Rows: IntVar,
    Cols: IntVar,
    BlockRows: IntVar,
    BlockCols: IntVar,
    Buffers: IntVar,
    Layout: IntVar,
](
    read_index: int,
    bars: gl.TmaBarrierRing2D[Buffers],
    a_smem: gl.TmaSharedRing2D[Buffers, BlockRows, BlockCols, Layout],
    b_smem: gl.TmaSharedRing2D[Buffers, BlockRows, BlockCols, Layout],
    c_smem: gl.TmaSharedTile2D[BlockRows, BlockCols, Layout],
    c_desc: gl.TmaOutputDescriptor2D[Rows, Cols, BlockRows, BlockCols, Layout],
    xoff: gl.GluonTileStart[BlockRows],
    layout: gl.Layout2D,
    YBLOCK: Int[BlockCols],
    num_buffers: Int[Buffers],
) -> int:
    # Wait for the copy from num_buffers-1 iterations ago to complete.
    read_phase = read_index // num_buffers & 1
    mbarrier.wait(bars.index(read_index % num_buffers), read_phase)
    a_val = a_smem.index(read_index % num_buffers).load(layout)
    b_val = b_smem.index(read_index % num_buffers).load(layout)
    c_val = a_val + b_val
    yoff = read_index * YBLOCK
    # Pipeline the store by rotating the store wait.
    tma.store_wait(pendings=0)
    c_smem.store(c_val)
    fence_async_shared()
    # Issue the store without waiting for it.
    tma.async_store(c_desc, [xoff, yoff], c_smem)
    return read_index + 1


def test_stage_boundary[
    Rows: IntVar,
    Cols: IntVar,
    BR: IntVar,
    BC: IntVar,
    Buffers: IntVar,
    Other: IntVar,
    Layout: IntVar,
    OtherLayout: IntVar,
](
    bars: gl.TmaBarrierRing2D[Buffers],
    wrong_bars: gl.TmaBarrierRing2D[Other],
    a_ring: gl.TmaSharedRing2D[Buffers, BR, BC, Layout],
    b_ring: gl.TmaSharedRing2D[Buffers, BR, BC, Layout],
    wrong_b_ring: gl.TmaSharedRing2D[Buffers, BR, Other, Layout],
    wrong_b_count: gl.TmaSharedRing2D[Other, BR, BC, Layout],
    c_tile: gl.TmaSharedTile2D[BR, BC, Layout],
    wrong_c_tile: gl.TmaSharedTile2D[BR, Other, Layout],
    wrong_c_layout: gl.TmaSharedTile2D[BR, BC, OtherLayout],
    c_desc: gl.TmaOutputDescriptor2D[Rows, Cols, BR, BC, Layout],
    wrong_c_block: gl.TmaOutputDescriptor2D[Rows, Cols, BR, Other, Layout],
    wrong_c_layout_desc: gl.TmaOutputDescriptor2D[Rows, Cols, BR, BC, OtherLayout],
    wrong_c_role: gl.TmaInputDescriptor2D[Rows, Cols, BR, BC, Layout],
    xoff: gl.GluonTileStart[BR],
    wrong_xoff: gl.GluonTileStart[Other],
    layout: gl.Layout2D,
    yblock: Int[BC],
    wrong_yblock: Int[Other],
    buffers: Int[Buffers],
    wrong_buffers: Int[Other],
) -> None:
    assert_type(
        perform_add(
            0, bars, a_ring, b_ring, c_tile, c_desc, xoff, layout, yblock, buffers
        ),
        int,
    )
    perform_add(
        0,
        wrong_bars,
        a_ring,  # E: is not assignable
        b_ring,  # E: is not assignable
        c_tile,
        c_desc,
        xoff,
        layout,
        yblock,
        buffers,  # E: is not assignable
    )
    perform_add(
        0,
        bars,
        a_ring,
        wrong_b_ring,  # E: is not assignable
        c_tile,
        c_desc,
        xoff,
        layout,
        yblock,
        buffers,
    )
    perform_add(
        0,
        bars,
        a_ring,
        wrong_b_count,  # E: is not assignable
        c_tile,
        c_desc,
        xoff,
        layout,
        yblock,
        buffers,
    )
    perform_add(
        0,
        bars,
        a_ring,
        b_ring,
        wrong_c_tile,  # E: is not assignable
        c_desc,
        xoff,
        layout,
        yblock,
        buffers,
    )
    perform_add(
        0,
        bars,
        a_ring,
        b_ring,
        wrong_c_layout,  # E: is not assignable
        c_desc,
        xoff,
        layout,
        yblock,
        buffers,
    )
    perform_add(
        0,
        bars,
        a_ring,
        b_ring,
        c_tile,
        wrong_c_block,  # E: is not assignable
        xoff,
        layout,
        yblock,
        buffers,
    )
    perform_add(
        0,
        bars,
        a_ring,
        b_ring,
        c_tile,
        wrong_c_layout_desc,  # E: is not assignable
        xoff,
        layout,
        yblock,
        buffers,
    )
    perform_add(
        0,
        bars,
        a_ring,
        b_ring,
        c_tile,
        wrong_c_role,  # E: is not assignable
        xoff,
        layout,
        yblock,
        buffers,
    )
    perform_add(
        0,
        bars,
        a_ring,
        b_ring,
        c_tile,
        c_desc,
        wrong_xoff,  # E: is not assignable
        layout,
        yblock,
        buffers,
    )
    perform_add(
        0,
        bars,
        a_ring,
        b_ring,
        c_tile,
        c_desc,
        xoff,
        layout,
        wrong_yblock,  # E: is not assignable
        buffers,
    )
    perform_add(
        0,
        bars,
        a_ring,
        b_ring,
        c_tile,
        c_desc,
        xoff,
        layout,
        yblock,
        wrong_buffers,  # E: is not assignable
    )


def test_output_tile_transfer[
    Rows: IntVar,
    Cols: IntVar,
    BR: IntVar,
    BC: IntVar,
    Other: IntVar,
    Layout: IntVar,
    OtherLayout: IntVar,
](
    c_desc: gl.TmaOutputDescriptor2D[Rows, Cols, BR, BC, Layout],
    wrong_c_host: gl.TmaOutputDescriptor2D[Other, Cols, BR, BC, Layout],
    wrong_role: gl.TmaInputDescriptor2D[Rows, Cols, BR, BC, Layout],
    wrong_layout: gl.TmaOutputDescriptor2D[Rows, Cols, BR, BC, OtherLayout],
    layout: gl.TmaLayout2D[Layout],
    bx: Int[BR],
    by: Int[BC],
    other: Int[Other],
    xoff: gl.GluonTileStart[BR],
    correct_value: tensor[[BR, BC]],
    wrong_value: tensor[[BR, Other]],
) -> None:
    tile = gl.allocate_shared_memory(gl.float32, [bx, by], layout)
    assert_type(tile, gl.TmaSharedTile2D[BR, BC, Layout])
    tile.store(correct_value)
    tile.store(wrong_value)  # E: is not assignable
    tma.async_store(c_desc, [xoff, 0], tile)
    tma.async_store(wrong_role, [xoff, 0], tile)  # E: No matching overload
    tma.async_store(wrong_layout, [xoff, 0], tile)  # E: No matching overload
    wrong_tile = gl.allocate_shared_memory(gl.float32, [bx, other], layout)
    tma.async_store(c_desc, [xoff, 0], wrong_tile)  # E: No matching overload
    tma.async_store(
        wrong_c_host, [xoff, 0], tile
    )  # Host extent is not checked by transfer.
    tma.async_store(c_desc, [xoff], tile)  # Coordinate count is not checked.


def test_shared_tile_addition[
    BR: IntVar,
    BC: IntVar,
    Other: IntVar,
    Buffers: IntVar,
    Layout: IntVar,
](
    a_ring: gl.TmaSharedRing2D[Buffers, BR, BC, Layout],
    b_ring: gl.TmaSharedRing2D[Buffers, BR, BC, Layout],
    c_tile: gl.TmaSharedTile2D[BR, BC, Layout],
    wrong_value: tensor[[BR, Other]],
) -> None:
    a_val = a_ring.index(0).load(gl.Layout2D())
    b_val = b_ring.index(0).load(gl.Layout2D())
    assert_type(a_val + b_val, tensor[[BR, BC]])
    c_tile.store(a_val + b_val)
    a_val + wrong_value  # E: No matching overload
    a_ring.index(0).load(gl.Layout1D())  # E: is not assignable

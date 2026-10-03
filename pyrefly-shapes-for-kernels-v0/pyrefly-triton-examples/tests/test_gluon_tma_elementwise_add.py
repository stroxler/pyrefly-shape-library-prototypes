# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The kernel body is copied from Triton's python/tutorials/gluon/04-tma.py.
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

# Static-only, not an executable Gluon kernel.
# @lint-ignore-every AUTODEPS2

"""Link pipelined 2D TMA read and write helper contracts inside a full kernel."""

from typing import assert_type

from shape_extensions import Int, IntVar
from tests.test_gluon_tma_issue_loads import issue_loads
from tests.test_gluon_tma_perform_add import perform_add
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma


@gluon.jit
def elementwise_add_tma_kernel[
    Rows: IntVar,
    Cols: IntVar,
    BlockRows: IntVar,
    BlockCols: IntVar,
    Layout: IntVar,
    Buffers: IntVar,
](
    a_desc: gl.TmaInputDescriptor2D[Rows, Cols, BlockRows, BlockCols, Layout],
    b_desc: gl.TmaInputDescriptor2D[Rows, Cols, BlockRows, BlockCols, Layout],
    c_desc: gl.TmaOutputDescriptor2D[Rows, Cols, BlockRows, BlockCols, Layout],
    xnumel: Int[Rows],
    ynumel: Int[Cols],
    XBLOCK: Int[BlockRows],
    YBLOCK: Int[BlockCols],
    num_buffers: Int[Buffers],
) -> None:
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    xoff = pid * XBLOCK

    dtype: gl.constexpr = a_desc.type.block_type.element_ty
    # Allocate multibuffered shared memory for the input buffers.
    a_smem = gl.allocate_shared_memory(
        dtype, [num_buffers, XBLOCK, YBLOCK], a_desc.layout
    )
    b_smem = gl.allocate_shared_memory(
        dtype, [num_buffers, XBLOCK, YBLOCK], b_desc.layout
    )

    # Allocate shared memory for the TMA store.
    c_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], c_desc.layout)

    # Allocate mbarriers to track completion of the TMA reads.
    bars = gl.allocate_shared_memory(
        gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout()
    )
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)

    copy_index = 0
    read_index = 0

    for _ in gl.static_range(num_buffers - 1):
        copy_index = issue_loads(
            copy_index, a_desc, b_desc, a_smem, b_smem, bars, xoff, YBLOCK, num_buffers
        )

    for _ in range(gl.cdiv(ynumel, YBLOCK) - (num_buffers - 1)):
        copy_index = issue_loads(
            copy_index, a_desc, b_desc, a_smem, b_smem, bars, xoff, YBLOCK, num_buffers
        )
        read_index = perform_add(
            read_index,
            bars,
            a_smem,
            b_smem,
            c_smem,
            c_desc,
            xoff,
            layout,
            YBLOCK,
            num_buffers,
        )

    for _ in gl.static_range(num_buffers - 1):
        read_index = perform_add(
            read_index,
            bars,
            a_smem,
            b_smem,
            c_smem,
            c_desc,
            xoff,
            layout,
            YBLOCK,
            num_buffers,
        )

    for i in gl.static_range(num_buffers):
        mbarrier.invalidate(bars.index(i))

    # Wait for the last store to complete.
    tma.store_wait(pendings=0)


def test_outer_interface[
    Rows: IntVar,
    Cols: IntVar,
    Other: IntVar,
    BR: IntVar,
    BC: IntVar,
    Layout: IntVar,
    OtherLayout: IntVar,
    Buffers: IntVar,
](
    a_desc: gl.TmaInputDescriptor2D[Rows, Cols, BR, BC, Layout],
    wrong_a_rows: gl.TmaInputDescriptor2D[Other, Cols, BR, BC, Layout],
    wrong_a_role: gl.TmaOutputDescriptor2D[Rows, Cols, BR, BC, Layout],
    b_desc: gl.TmaInputDescriptor2D[Rows, Cols, BR, BC, Layout],
    wrong_b_cols: gl.TmaInputDescriptor2D[Rows, Other, BR, BC, Layout],
    wrong_b_block: gl.TmaInputDescriptor2D[Rows, Cols, BR, Other, Layout],
    c_desc: gl.TmaOutputDescriptor2D[Rows, Cols, BR, BC, Layout],
    wrong_c_rows: gl.TmaOutputDescriptor2D[Other, Cols, BR, BC, Layout],
    wrong_c_block: gl.TmaOutputDescriptor2D[Rows, Cols, Other, BC, Layout],
    wrong_c_layout: gl.TmaOutputDescriptor2D[Rows, Cols, BR, BC, OtherLayout],
    wrong_c_role: gl.TmaInputDescriptor2D[Rows, Cols, BR, BC, Layout],
    xnumel: Int[Rows],
    ynumel: Int[Cols],
    wrong_numel: Int[Other],
    xblock: Int[BR],
    yblock: Int[BC],
    wrong_block: Int[Other],
    buffers: Int[Buffers],
) -> None:
    assert_type(
        elementwise_add_tma_kernel(
            a_desc, b_desc, c_desc, xnumel, ynumel, xblock, yblock, buffers
        ),
        None,
    )
    elementwise_add_tma_kernel(
        wrong_a_rows,
        b_desc,  # E: is not assignable
        c_desc,  # E: is not assignable
        xnumel,  # E: is not assignable
        ynumel,
        xblock,
        yblock,
        buffers,
    )
    elementwise_add_tma_kernel(
        wrong_a_role,  # E: is not assignable
        b_desc,
        c_desc,
        xnumel,
        ynumel,
        xblock,
        yblock,
        buffers,
    )
    elementwise_add_tma_kernel(
        a_desc,
        wrong_b_cols,  # E: is not assignable
        c_desc,
        xnumel,
        ynumel,
        xblock,
        yblock,
        buffers,
    )
    elementwise_add_tma_kernel(
        a_desc,
        wrong_b_block,  # E: is not assignable
        c_desc,
        xnumel,
        ynumel,
        xblock,
        yblock,
        buffers,
    )
    elementwise_add_tma_kernel(
        a_desc,
        b_desc,
        wrong_c_rows,  # E: is not assignable
        xnumel,
        ynumel,
        xblock,
        yblock,
        buffers,
    )
    elementwise_add_tma_kernel(
        a_desc,
        b_desc,
        wrong_c_block,  # E: is not assignable
        xnumel,
        ynumel,
        xblock,
        yblock,
        buffers,
    )
    elementwise_add_tma_kernel(
        a_desc,
        b_desc,
        wrong_c_layout,  # E: is not assignable
        xnumel,
        ynumel,
        xblock,
        yblock,
        buffers,
    )
    elementwise_add_tma_kernel(
        a_desc,
        b_desc,
        wrong_c_role,  # E: is not assignable
        xnumel,
        ynumel,
        xblock,
        yblock,
        buffers,
    )
    elementwise_add_tma_kernel(
        a_desc,
        b_desc,
        c_desc,
        wrong_numel,  # E: is not assignable
        ynumel,
        xblock,
        yblock,
        buffers,
    )
    elementwise_add_tma_kernel(
        a_desc,
        b_desc,
        c_desc,
        xnumel,
        wrong_numel,  # E: is not assignable
        xblock,
        yblock,
        buffers,
    )
    elementwise_add_tma_kernel(
        a_desc,
        b_desc,
        c_desc,
        xnumel,
        ynumel,
        wrong_block,  # E: is not assignable
        yblock,
        buffers,
    )
    elementwise_add_tma_kernel(
        a_desc,
        b_desc,
        c_desc,
        xnumel,
        ynumel,
        xblock,
        wrong_block,  # E: is not assignable
        buffers,
    )


def test_outer_ring_allocation[
    Rows: IntVar,
    Cols: IntVar,
    BR: IntVar,
    BC: IntVar,
    Buffers: IntVar,
    Other: IntVar,
    Layout: IntVar,
](
    a_desc: gl.TmaInputDescriptor2D[Rows, Cols, BR, BC, Layout],
    b_desc: gl.TmaInputDescriptor2D[Rows, Cols, BR, BC, Layout],
    c_desc: gl.TmaOutputDescriptor2D[Rows, Cols, BR, BC, Layout],
    xblock: Int[BR],
    yblock: Int[BC],
    buffers: Int[Buffers],
    other_buffers: Int[Other],
    xoff: gl.GluonTileStart[BR],
) -> None:
    a_ring = gl.allocate_shared_memory(
        a_desc.dtype, [buffers, xblock, yblock], a_desc.layout
    )
    b_ring = gl.allocate_shared_memory(
        b_desc.dtype, [buffers, xblock, yblock], b_desc.layout
    )
    c_tile = gl.allocate_shared_memory(c_desc.dtype, [xblock, yblock], c_desc.layout)
    bars = gl.allocate_shared_memory(gl.int64, [buffers, 1], mbarrier.MBarrierLayout())
    wrong_bars = gl.allocate_shared_memory(
        gl.int64, [other_buffers, 1], mbarrier.MBarrierLayout()
    )
    assert_type(a_ring, gl.TmaSharedRing2D[Buffers, BR, BC, Layout])
    assert_type(b_ring, gl.TmaSharedRing2D[Buffers, BR, BC, Layout])
    assert_type(c_tile, gl.TmaSharedTile2D[BR, BC, Layout])
    assert_type(bars, gl.TmaBarrierRing2D[Buffers])
    issue_loads(0, a_desc, b_desc, a_ring, b_ring, bars, xoff, yblock, buffers)
    perform_add(
        0, bars, a_ring, b_ring, c_tile, c_desc, xoff, gl.Layout2D(), yblock, buffers
    )
    issue_loads(
        0,
        a_desc,
        b_desc,
        a_ring,
        b_ring,
        wrong_bars,  # E: is not assignable
        xoff,
        yblock,
        buffers,
    )

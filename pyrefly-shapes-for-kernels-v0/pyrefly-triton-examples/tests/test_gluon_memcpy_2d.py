# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The kernel body is copied from Triton's python/tutorials/gluon/02-layouts.py.
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

"""Check 2D Gluon memcpy array, stride, tile, and slice-layout roles."""

from typing import assert_type

from shape_extensions import Int, IntVar
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def memcpy_2d_kernel[
    X: IntVar,
    Y: IntVar,
    SXIn: IntVar,
    SYIn: IntVar,
    SXOut: IntVar,
    SYOut: IntVar,
    XB: IntVar,
    YB: IntVar,
](
    in_ptr: gl.InMatrixPointer2D[X, Y, SXIn, SYIn],
    out_ptr: gl.OutMatrixPointer2D[X, Y, SXOut, SYOut],  #
    xnumel: Int[X],
    ynumel: Int[Y],
    xstride_in: Int[SXIn],
    ystride_in: Int[SYIn],
    xstride_out: Int[SXOut],
    ystride_out: Int[SYOut],  #
    layout: gl.Layout2D,
    XBLOCK: Int[XB],
    YBLOCK: Int[YB],
):
    pid_x = gl.program_id(0)
    pid_y = gl.program_id(1)

    start_x = pid_x * XBLOCK
    start_y = pid_y * YBLOCK
    # For the 1D indices, use a SliceLayout along the dimensions we will expand.
    indices_x = start_x + gl.arange(
        0, XBLOCK, layout=gl.SliceLayout(dim=1, parent=layout)
    )
    indices_y = start_y + gl.arange(
        0, YBLOCK, layout=gl.SliceLayout(dim=0, parent=layout)
    )

    # expand_dims along the slice dimension returns a tensor with the parent
    # layout, so this yields [XBLOCK, 1] and [1, YBLOCK] tensors with the same
    # layout which can be broadcasted together to [XBLOCK, YBLOCK].
    in_offsets = xstride_in * indices_x[:, None] + ystride_in * indices_y[None, :]
    out_offsets = xstride_out * indices_x[:, None] + ystride_out * indices_y[None, :]

    # Compute the mask the same way: select for indices along each dimension
    # that are in bounds and broadcast them together.
    mask = (indices_x[:, None] < xnumel) & (indices_y[None, :] < ynumel)

    value = gl.load(in_ptr + in_offsets, mask=mask)
    gl.store(out_ptr + out_offsets, value, mask=mask)


def test_boundary[
    X: IntVar,
    Y: IntVar,
    Other: IntVar,
    SXIn: IntVar,
    SYIn: IntVar,
    SXOut: IntVar,
    SYOut: IntVar,
    BX: IntVar,
    BY: IntVar,
](
    inp: gl.InMatrixPointer2D[X, Y, SXIn, SYIn],
    out: gl.OutMatrixPointer2D[X, Y, SXOut, SYOut],
    bad_in_x: gl.InMatrixPointer2D[Other, Y, SXIn, SYIn],
    bad_out_y: gl.OutMatrixPointer2D[X, Other, SXOut, SYOut],
    bad_in_stride: gl.InMatrixPointer2D[X, Y, SXIn, Other],
    x: Int[X],
    y: Int[Y],
    sx_in: Int[SXIn],
    sy_in: Int[SYIn],
    sx_out: Int[SXOut],
    sy_out: Int[SYOut],
    bx: Int[BX],
    by: Int[BY],
    layout: gl.Layout2D,
    wrong_rank: gl.Layout1D,
) -> None:
    memcpy_2d_kernel(inp, out, x, y, sx_in, sy_in, sx_out, sy_out, layout, bx, by)
    memcpy_2d_kernel(
        bad_in_x,
        out,  # E: is not assignable
        x,  # E: is not assignable
        y,
        sx_in,
        sy_in,
        sx_out,
        sy_out,
        layout,
        bx,
        by,
    )
    memcpy_2d_kernel(
        inp,
        bad_out_y,  # E: is not assignable
        x,
        y,
        sx_in,
        sy_in,
        sx_out,
        sy_out,
        layout,
        bx,
        by,
    )
    memcpy_2d_kernel(
        bad_in_stride,
        out,
        x,
        y,
        sx_in,
        sy_in,  # E: is not assignable
        sx_out,
        sy_out,
        layout,
        bx,
        by,
    )
    memcpy_2d_kernel(
        inp,
        out,
        x,
        y,
        sx_in,
        sy_in,
        sx_out,
        sy_out,
        wrong_rank,  # E: is not assignable
        bx,
        by,
    )


def test_layout_rank_and_axes[XB: IntVar, YB: IntVar](
    xb: Int[XB],
    yb: Int[YB],
    one: gl.Layout1D,
    two: gl.Layout2D,
) -> None:
    row = gl.arange(0, xb, layout=gl.SliceLayout(dim=1, parent=two))
    col = gl.arange(0, yb, layout=gl.SliceLayout(dim=0, parent=two))
    assert_type(row, gl.RowIndices[XB])
    assert_type(col, gl.ColumnIndices[YB])
    assert_type(row[:, None], gl.RowIndices2D[XB])
    assert_type(col[None, :], gl.ColumnIndices2D[YB])
    gl.SliceLayout(dim=1, parent=one)  # E: No matching overload
    gl.SliceLayout(dim=2, parent=two)  # E: No matching overload
    row[None, :]  # E: Cannot index
    col[:, None]  # E: Cannot index


def test_blocked_layout_rank() -> None:
    assert_type(gl.BlockedLayout([1], [32], [4], [0]), gl.Layout1D)
    assert_type(gl.BlockedLayout([1, 1], [4, 8], [4, 1], [1, 0]), gl.Layout2D)
    gl.BlockedLayout([1, 1], [32], [4, 1], [1, 0])  # E: No matching overload


def test_mask_and_layout_carry[
    X: IntVar,
    Y: IntVar,
    BX: IntVar,
    BY: IntVar,
    Other: IntVar,
](
    inp: gl.InMatrixTile2D[X, Y, BX, BY],
    out: gl.OutMatrixTile2D[X, Y, BX, BY],
    x: Int[X],
    y: Int[Y],
    other: Int[Other],
    row: gl.RowIndices2D[BX],
    col: gl.ColumnIndices2D[BY],
    bad_value: gl.tensor[[BX, Other]],
) -> None:
    mask = (row < x) & (col < y)
    assert_type(mask, gl.MatrixMask2D[X, Y, BX, BY])
    value = gl.load(inp, mask=mask)
    assert_type(value, gl.tensor[[BX, BY]])
    gl.store(out, value, mask=mask)
    gl.load(inp, mask=(row < other) & (col < y))  # E: is not assignable
    gl.store(out, value, mask=(row < x) & (col < other))  # E: is not assignable
    gl.store(out, bad_value, mask=mask)  # E: is not assignable

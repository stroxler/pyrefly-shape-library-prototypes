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

"""Check a Gluon layout's rank independently of Triton-style masked accesses."""

import triton.language as tl
from shape_extensions import Int, IntVar
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def memcpy_1d_kernel[N: IntVar, Block: IntVar](
    in_ptr: tl.InPointer[[N]],
    out_ptr: tl.OutPointer[[N]],
    xnumel: Int[N],
    XBLOCK: Int[Block],
    layout: gl.Layout1D,
):
    pid = gl.program_id(0)
    start = pid * XBLOCK

    # The main difference between writing this kernel in Triton and Gluon is
    # we need to specify the layout of the 1D tensor. Layouts are propagated
    # forwards through type inference, so we only need to specify the layout for
    # the indices tensor.
    indices = gl.arange(0, XBLOCK, layout=layout)

    offsets = start + indices
    in_ptrs = in_ptr + offsets
    mask = offsets < xnumel

    value = gl.load(in_ptrs, mask=mask)
    out_ptrs = out_ptr + offsets
    gl.store(out_ptrs, value, mask=mask)


def test_rank_and_allocation[N: IntVar, Block: IntVar, Other: IntVar](
    inp: tl.InPointer[[N]],
    out: tl.OutPointer[[N]],
    wrong_inp: tl.InPointer[[Other]],
    wrong_out: tl.OutPointer[[Other]],
    n: Int[N],
    block: Int[Block],
    one_dimensional: gl.Layout1D,
    wrong_rank: gl.Layout2D,
) -> None:
    memcpy_1d_kernel(inp, out, n, block, one_dimensional)
    # E: is not assignable
    # E: is not assignable
    memcpy_1d_kernel(wrong_inp, out, n, block, one_dimensional)
    memcpy_1d_kernel(inp, wrong_out, n, block, one_dimensional)  # E: is not assignable
    memcpy_1d_kernel(inp, out, n, block, wrong_rank)  # E: is not assignable


def test_wrong_mask_bound[N: IntVar, Block: IntVar, Other: IntVar](
    inp: tl.InPointer[[N]],
    n: Int[N],
    other: Int[Other],
    block: Int[Block],
    layout: gl.Layout1D,
) -> None:
    offsets = gl.arange(0, block, layout=layout)
    gl.load(inp + offsets, mask=offsets < n)
    gl.load(inp + offsets, mask=offsets < other)  # E: is not assignable

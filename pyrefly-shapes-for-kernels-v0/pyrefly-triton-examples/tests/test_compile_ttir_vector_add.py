# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The complete kernel bodies are copied from Triton's
# python/tutorials/compilation-pipeline/01_read_ttir.py and
# python/tutorials/compilation-pipeline/02_layout_assignment.py.
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
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# @lint-ignore-every AUTODEPS2

"""TTIR vector add checks the same masked boundary as the numbered tutorial."""

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def add_kernel[N: IntVar, Block: IntVar](
    x_ptr: tl.InPointer[[N]],
    y_ptr: tl.InPointer[[N]],
    o_ptr: tl.OutPointer[[N]],
    n: Int[N],
    BLOCK: Int[Block],
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(o_ptr + offs, x + y, mask=mask)


def test_wrong_output_allocation[N: IntVar, Other: IntVar, Block: IntVar](
    x: tl.InPointer[[N]],
    y: tl.InPointer[[N]],
    wrong_output: tl.OutPointer[[Other]],
    n: Int[N],
    block: Int[Block],
) -> None:
    add_kernel(x, y, wrong_output, n, block)  # E: is not assignable


# Only the second input's annotation changes; its original masked load fails.
@triton.jit
def add_kernel_wrong_input[N: IntVar, Other: IntVar, Block: IntVar](
    x_ptr: tl.InPointer[[N]],
    y_ptr: tl.InPointer[[Other]],
    o_ptr: tl.OutPointer[[N]],
    n: Int[N],
    BLOCK: Int[Block],
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)  # E: is not assignable
    tl.store(o_ptr + offs, x + y, mask=mask)


# The second compilation tutorial uses inline loads but the same bound mask.
@triton.jit
def add_kernel_layout[N: IntVar, Block: IntVar](
    x_ptr: tl.InPointer[[N]],
    y_ptr: tl.InPointer[[N]],
    o_ptr: tl.OutPointer[[N]],
    n: Int[N],
    BLOCK: Int[Block],
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(
        o_ptr + offs,
        tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask),
        mask=mask,
    )


@triton.jit
def add_kernel_layout_wrong_output[N: IntVar, Other: IntVar, Block: IntVar](
    x_ptr: tl.InPointer[[N]],
    y_ptr: tl.InPointer[[N]],
    o_ptr: tl.OutPointer[[Other]],
    n: Int[N],
    BLOCK: Int[Block],
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(  # E: No matching overload
        o_ptr + offs,
        tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask),
        mask=mask,
    )

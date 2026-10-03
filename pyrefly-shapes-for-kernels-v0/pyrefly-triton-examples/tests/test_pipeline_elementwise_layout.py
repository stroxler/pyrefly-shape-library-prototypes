# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The kernel body is copied from Triton's python/tutorials/compilation-pipeline/04_remove_layout_conversions.py.
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

"""Static allocation and tile contract for the layout-conversion comparison."""

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def elementwise_kernel[N: IntVar, Block: IntVar](
    x_ptr: tl.InPointer[[N]], o_ptr: tl.OutPointer[[N]], n: Int[N], BLOCK: Int[Block]
):
    offs = tl.arange(0, BLOCK)
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=offs < n) * 2.0, mask=offs < n)


def test_elementwise_boundary[N: IntVar, Other: IntVar, Block: IntVar](
    x: tl.InPointer[[N]],
    output: tl.OutPointer[[N]],
    wrong_output: tl.OutPointer[[Other]],
    n: Int[N],
    block: Int[Block],
) -> None:
    elementwise_kernel(x, output, n, block)
    elementwise_kernel(x, wrong_output, n, block)  # E: is not assignable


# Changing only the input extent makes the original nested load reject n's mask.
@triton.jit
def elementwise_kernel_wrong_input[N: IntVar, Other: IntVar, Block: IntVar](
    x_ptr: tl.InPointer[[Other]],
    o_ptr: tl.OutPointer[[N]],
    n: Int[N],
    BLOCK: Int[Block],
):
    offs = tl.arange(0, BLOCK)
    tl.store(
        o_ptr + offs,
        tl.load(x_ptr + offs, mask=offs < n) * 2.0,  # E: No matching overload
        mask=offs < n,
    )


# Changing only the output extent makes the original store reject n's mask.
@triton.jit
def elementwise_kernel_wrong_output[N: IntVar, Other: IntVar, Block: IntVar](
    x_ptr: tl.InPointer[[N]],
    o_ptr: tl.OutPointer[[Other]],
    n: Int[N],
    BLOCK: Int[Block],
):
    offs = tl.arange(0, BLOCK)
    tl.store(  # E: No matching overload
        o_ptr + offs, tl.load(x_ptr + offs, mask=offs < n) * 2.0, mask=offs < n
    )

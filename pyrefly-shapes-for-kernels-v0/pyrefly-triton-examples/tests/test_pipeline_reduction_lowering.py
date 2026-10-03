# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The complete kernel body is copied from Triton's
# python/tutorials/compilation-pipeline/07_reduction_lowering.py.
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

"""A masked source tile reduces to one scalar output in compiler tutorial 07."""

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def sum_kernel[N: IntVar, Block: IntVar](
    src: tl.InPointer[[N]],
    dst: tl.OutScalarPointer[[1]],
    N: Int[N],
    BLOCK: Int[Block],
):
    offs = tl.arange(0, BLOCK)
    x = tl.load(src + offs, mask=offs < N, other=0.0)
    tl.store(dst, tl.sum(x, axis=0))


def test_reduction_boundary[N: IntVar, Other: IntVar, Block: IntVar](
    src: tl.InPointer[[N]],
    wrong_src: tl.InPointer[[Other]],
    dst: tl.OutScalarPointer[[1]],
    wrong_dst: tl.OutScalarPointer[[Other]],
    n: Int[N],
    block: Int[Block],
) -> None:
    sum_kernel(src, dst, n, block)
    sum_kernel(wrong_src, dst, n, block)  # E: is not assignable
    sum_kernel(src, wrong_dst, n, block)  # E: is not assignable


# The source extent is grounded in its masked load, unlike the output extent.
@triton.jit
def sum_kernel_wrong_input[N: IntVar, Other: IntVar, Block: IntVar](
    src: tl.InPointer[[Other]],
    dst: tl.OutScalarPointer[[1]],
    N: Int[N],
    BLOCK: Int[Block],
):
    offs = tl.arange(0, BLOCK)
    x = tl.load(src + offs, mask=offs < N, other=0.0)  # E: is not assignable
    tl.store(dst, tl.sum(x, axis=0))

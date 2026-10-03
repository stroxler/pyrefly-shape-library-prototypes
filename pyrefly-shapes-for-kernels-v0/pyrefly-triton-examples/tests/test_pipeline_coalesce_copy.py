# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The kernel body is copied from Triton's python/tutorials/compilation-pipeline/03_coalesce_vectorization.py.
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

"""Allocation and tile contract for the coalescing tutorial's flat copy."""

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def copy_kernel[N: IntVar, Block: IntVar](
    src: tl.InPointer[[N]], dst: tl.OutPointer[[N]], n: Int[N], BLOCK: Int[Block]
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)


def test_copy_boundary[N: IntVar, Other: IntVar, Block: IntVar](
    src: tl.InPointer[[N]],
    dst: tl.OutPointer[[N]],
    wrong_src: tl.InPointer[[Other]],
    wrong_dst: tl.OutPointer[[Other]],
    n: Int[N],
    block: Int[Block],
) -> None:
    copy_kernel(src, dst, n, block)
    copy_kernel(
        wrong_src,
        dst,  # E: is not assignable
        n,  # E: is not assignable
        block,
    )
    copy_kernel(src, wrong_dst, n, block)  # E: is not assignable


# An incorrect source extent fails at the nested original load.
@triton.jit
def copy_kernel_wrong_src[N: IntVar, Other: IntVar, Block: IntVar](
    src: tl.InPointer[[Other]], dst: tl.OutPointer[[N]], n: Int[N], BLOCK: Int[Block]
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(
        dst + offs,
        tl.load(src + offs, mask=mask),  # E: No matching overload
        mask=mask,
    )


# An incorrect destination extent fails at the original store.
@triton.jit
def copy_kernel_wrong_dst[N: IntVar, Other: IntVar, Block: IntVar](
    src: tl.InPointer[[N]], dst: tl.OutPointer[[Other]], n: Int[N], BLOCK: Int[Block]
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(  # E: No matching overload
        dst + offs, tl.load(src + offs, mask=mask), mask=mask
    )

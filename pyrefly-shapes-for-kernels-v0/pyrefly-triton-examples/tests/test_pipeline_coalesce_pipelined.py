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

"""Static allocation contract for the looped coalescing tutorial copy."""

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def copy_kernel_pipelined[N: IntVar, Block: IntVar, Steps: IntVar](
    src: tl.InPointer[[N]],
    dst: tl.OutPointer[[N]],
    n: Int[N],
    BLOCK: Int[Block],
    STEPS: Int[Steps],
):
    # Same copy, but over a loop with num_stages=2. The software pipeliner then
    # multi-buffers the load and issues it ASYNCHRONOUSLY (cp.async) so the next
    # iteration's data is fetched while this one stores — the synchronous vector
    # `ld.global` becomes `cp.async`.
    base = tl.program_id(0) * BLOCK * STEPS
    for i in tl.range(0, STEPS, num_stages=2):
        offs = base + i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)


def test_pipelined_copy_boundary[
    N: IntVar,
    Other: IntVar,
    Block: IntVar,
    Steps: IntVar,
](
    src: tl.InPointer[[N]],
    dst: tl.OutPointer[[N]],
    wrong_dst: tl.OutPointer[[Other]],
    n: Int[N],
    block: Int[Block],
    steps: Int[Steps],
) -> None:
    copy_kernel_pipelined(src, dst, n, block, steps)
    copy_kernel_pipelined(src, wrong_dst, n, block, steps)  # E: is not assignable


# Only the input allocation extent differs. The original nested load must
# reject the mask derived from n even though the loop offset is an integer.
@triton.jit
def copy_kernel_pipelined_wrong_src[
    N: IntVar,
    Other: IntVar,
    Block: IntVar,
    Steps: IntVar,
](
    src: tl.InPointer[[Other]],
    dst: tl.OutPointer[[N]],
    n: Int[N],
    BLOCK: Int[Block],
    STEPS: Int[Steps],
):
    # Same copy, but over a loop with num_stages=2. The software pipeliner then
    # multi-buffers the load and issues it ASYNCHRONOUSLY (cp.async) so the next
    # iteration's data is fetched while this one stores — the synchronous vector
    # `ld.global` becomes `cp.async`.
    base = tl.program_id(0) * BLOCK * STEPS
    for i in tl.range(0, STEPS, num_stages=2):
        offs = base + i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(
            dst + offs,
            tl.load(src + offs, mask=mask),  # E: No matching overload
            mask=mask,
        )


# Only the output allocation extent differs. The original store must reject
# the bound carried by its own mask.
@triton.jit
def copy_kernel_pipelined_wrong_dst[
    N: IntVar,
    Other: IntVar,
    Block: IntVar,
    Steps: IntVar,
](
    src: tl.InPointer[[N]],
    dst: tl.OutPointer[[Other]],
    n: Int[N],
    BLOCK: Int[Block],
    STEPS: Int[Steps],
):
    # Same copy, but over a loop with num_stages=2. The software pipeliner then
    # multi-buffers the load and issues it ASYNCHRONOUSLY (cp.async) so the next
    # iteration's data is fetched while this one stores — the synchronous vector
    # `ld.global` becomes `cp.async`.
    base = tl.program_id(0) * BLOCK * STEPS
    for i in tl.range(0, STEPS, num_stages=2):
        offs = base + i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(  # E: No matching overload
            dst + offs, tl.load(src + offs, mask=mask), mask=mask
        )

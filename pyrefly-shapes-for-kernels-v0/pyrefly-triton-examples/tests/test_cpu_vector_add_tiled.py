# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The kernel body is copied from Triton's python/tutorials/cpu/01-vector-add.py.
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

# Static-only copy of add_kernel_tiled from Triton's
# python/tutorials/cpu/01-vector-add.py; Triton's JIT does not accept these annotations.
# @lint-ignore-every AUTODEPS2

"""Check a program that processes multiple smaller tiles of one allocation."""

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def add_kernel_tiled[N: IntVar, Block: IntVar, Tile: IntVar](
    x_ptr: tl.InPointer[[N]],  # *Pointer* to first input vector.
    y_ptr: tl.InPointer[[N]],  # *Pointer* to second input vector.
    output_ptr: tl.OutPointer[[N]],  # *Pointer* to output vector.
    n_elements: Int[N],  # Size of the vector.
    BLOCK_SIZE: Int[Block],  # Number of elements each program should process.
    TILE_SIZE: Int[Tile],  # Number of elements each iteration should process.
    # NOTE `constexpr` so it can be used as a shape value.
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    for i in range(0, tl.cdiv(BLOCK_SIZE, TILE_SIZE)):
        offsets = block_start + i * TILE_SIZE + tl.arange(0, TILE_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)


def test_incorrect_allocation_at_boundary[
    N: IntVar,
    Other: IntVar,
    Block: IntVar,
    Tile: IntVar,
](
    x: tl.InPointer[[N]],
    wrong_y: tl.InPointer[[Other]],
    output: tl.OutPointer[[N]],
    n: Int[N],
    block: Int[Block],
    tile: Int[Tile],
) -> None:
    add_kernel_tiled(x, wrong_y, output, n, block, tile)  # E: is not assignable


# Only the second pointer annotation differs from the source kernel above. The
# unchanged load must reject its mask's allocation bound inside the loop.
@triton.jit
def add_kernel_tiled_wrong_y[N: IntVar, Other: IntVar, Block: IntVar, Tile: IntVar](
    x_ptr: tl.InPointer[[N]],
    y_ptr: tl.InPointer[[Other]],
    output_ptr: tl.OutPointer[[N]],
    n_elements: Int[N],
    BLOCK_SIZE: Int[Block],
    TILE_SIZE: Int[Tile],
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    for i in range(0, tl.cdiv(BLOCK_SIZE, TILE_SIZE)):
        offsets = block_start + i * TILE_SIZE + tl.arange(0, TILE_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)  # E: is not assignable
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)


# Only the output pointer annotation differs from the source kernel. The store
# must reject its mask's allocation bound without changing its executable body.
@triton.jit
def add_kernel_tiled_wrong_output[
    N: IntVar,
    Other: IntVar,
    Block: IntVar,
    Tile: IntVar,
](
    x_ptr: tl.InPointer[[N]],
    y_ptr: tl.InPointer[[N]],
    output_ptr: tl.OutPointer[[Other]],
    n_elements: Int[N],
    BLOCK_SIZE: Int[Block],
    TILE_SIZE: Int[Tile],
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    for i in range(0, tl.cdiv(BLOCK_SIZE, TILE_SIZE)):
        offsets = block_start + i * TILE_SIZE + tl.arange(0, TILE_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)  # E: is not assignable

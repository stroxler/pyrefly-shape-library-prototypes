# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The kernel body is copied from Triton's python/tutorials/gluon/03-async-copy.py.
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

"""Check global-to-shared async-copy bounds, tile size, and memory roles."""

from typing import assert_type

from shape_extensions import Int, IntVar
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp
from triton.language import InPointer, OutPointer, tensor


@gluon.jit
def memcpy_1d_cpasync_kernel[N: IntVar, Block: IntVar](
    in_ptr: InPointer[[N]],
    out_ptr: OutPointer[[N]],
    xnumel: Int[N],
    XBLOCK: Int[Block],
):
    pid = gl.program_id(0)

    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    offsets = pid * XBLOCK + gl.arange(0, XBLOCK, layout=layout)
    mask = offsets < xnumel

    # For 1D tensor, pick a simple layout.
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[0]
    )
    smem = gl.allocate_shared_memory(gl.float32, [XBLOCK], layout=smem_layout)

    # Issue the async copy.
    cp.async_load(smem, in_ptr + offsets, mask=mask)
    # `commit_group` puts all previously issued async copies into a group.
    cp.commit_group()

    # Wait until the number of pending groups reaches 0. Then we can retrieve
    # the data from shared memory.
    cp.wait_group(0)

    value = smem.load(layout)
    gl.store(out_ptr + offsets, value, mask=mask)


def test_global_interface[N: IntVar, Other: IntVar, Block: IntVar](
    inp: InPointer[[N]],
    out: OutPointer[[N]],
    other_in: InPointer[[Other]],
    other_out: OutPointer[[Other]],
    length: Int[N],
    block: Int[Block],
) -> None:
    memcpy_1d_cpasync_kernel(inp, out, length, block)
    # E: is not assignable
    # E: is not assignable
    memcpy_1d_cpasync_kernel(other_in, out, length, block)
    memcpy_1d_cpasync_kernel(inp, other_out, length, block)  # E: is not assignable


def test_shared_memory_contract[N: IntVar, Block: IntVar, Other: IntVar](
    ptr: InPointer[[N]],
    output: OutPointer[[N]],
    n: Int[N],
    other: Int[Other],
    block: Int[Block],
    other_block: Int[Other],
    layout: gl.Layout1D,
    shared_layout: gl.SharedLayout1D,
) -> None:
    offsets = gl.arange(0, block, layout=layout)
    mask = offsets < n
    smem = gl.allocate_shared_memory(gl.float32, [block], layout=shared_layout)
    assert_type(smem, gl.SharedBuffer1D[Block])
    cp.async_load(smem, ptr + offsets, mask=mask)
    cp.commit_group()
    cp.wait_group(0)
    value = smem.load(layout)
    assert_type(value, tensor[[Block]])
    gl.store(output + offsets, value, mask=mask)

    other_smem = gl.allocate_shared_memory(
        gl.float32, [other_block], layout=shared_layout
    )
    # E: is not assignable
    # E: is not assignable
    cp.async_load(other_smem, ptr + offsets, mask=mask)
    cp.async_load(smem, ptr + offsets, mask=offsets < other)  # E: is not assignable
    cp.async_load(smem, output + offsets, mask=mask)  # E: is not assignable
    # E: No matching overload
    gl.allocate_shared_memory(
        gl.float32,
        [block],
        layout=layout,
    )
    smem.load(shared_layout)  # E: is not assignable
    # E: is not assignable
    gl.store(output + offsets, other_smem.load(layout), mask=mask)


def test_layout_rank() -> None:
    assert_type(
        gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0]),
        gl.SharedLayout1D,
    )
    gl.SwizzledSharedLayout(
        vec=1,
        per_phase=1,
        max_phase=1,
        order=[1, 0],  # E: is not assignable
    )


def test_constexpr_local_layouts[Block: IntVar](block: Int[Block]) -> None:
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[0]
    )
    assert_type(layout, gl.Layout1D)
    assert_type(smem_layout, gl.SharedLayout1D)
    gl.arange(0, block, layout=smem_layout)  # E: No matching overload
    # E: No matching overload
    gl.allocate_shared_memory(
        gl.float32,
        [block],
        layout=layout,
    )

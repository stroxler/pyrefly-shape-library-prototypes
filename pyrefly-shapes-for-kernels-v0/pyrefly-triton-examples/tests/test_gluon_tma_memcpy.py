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

"""Check Hopper TMA block and layout contracts across shared-memory transfers."""

from typing import assert_type

from shape_extensions import Int, IntVar
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma


@gluon.jit
def memcpy_1d_tma_kernel[N: IntVar, Block: IntVar, Layout: IntVar](
    in_desc: gl.TmaInputDescriptor1D[N, Block, Layout],
    out_desc: gl.TmaOutputDescriptor1D[N, Block, Layout],
    XBLOCK: Int[Block],
):
    # We don't need to pass the tensor strides because they are stored in the
    # tensor descriptors
    pid = gl.program_id(0)

    # Each tensor descriptor contains a shared memory layout. Data is
    # transferred between global and shared memory according to that layout.
    smem_layout: gl.constexpr = in_desc.layout
    smem = gl.allocate_shared_memory(in_desc.dtype, [XBLOCK], smem_layout)

    # Completion of async TMA reads are tracked by mbarrier objects. These
    # are 64-bit objects that live in shared memory.
    #
    # An mbarrier is initialized with a count. Each time a mbarrier is
    # "arrived" on, the count is decremented. When the count reaches 0, the
    # current phase of the mbarrier is marked as complete and it moves to the
    # next phase. The mbarrier only tracks the state of the current and
    # previous phase. This is important, because if an mbarrier's phase races
    # too far ahead, its waiter will become out of sync.
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())

    # Completion of an async TMA arrives on an mbarrier once. Thus, initialize
    # the mbarrier with a count of 1 so its phase will complete when the TMA is
    # complete.
    mbarrier.init(bar, count=1)

    # Tensor descriptors have an associated block shape. Each TMA request will
    # copy one block of the tensor descriptor. The coordinates of the TMA
    # request are specified as offsets to the beginning of the block. Masking
    # of out-of-bounds reads and writes is handled automatically by TMAs, using
    # the shape specified on the tensor descriptor.
    gl.static_assert(in_desc.block_type == out_desc.block_type)
    gl.static_assert(in_desc.layout == out_desc.layout)

    # Track completion of the TMA read based on the number of bytes copied.
    # mbarrier.expect sets the number of outstanding bytes tracked by the
    # mbarrier. If we pass the barrier to the TMA copy, it will atomically
    # decrement the number of outstanding bytes as transactions complete. When
    # it reaches 0, the mbarrier is arrived on once.
    mbarrier.expect(bar, in_desc.block_type.nbytes)
    tma.async_load(in_desc, [pid * XBLOCK], bar, smem)

    # Wait for completion of the read. We query the completion state of the
    # mbarrier using the parity of the phase, i.e. either 0 or 1. Mbarriers are
    # initialized to parity 1 complete, so we wait for parity 0.
    mbarrier.wait(bar, phase=0)

    # When we are done using the mbarrier, we need to invalidate it.
    mbarrier.invalidate(bar)

    # Since the TMA store reads from shared memory, we don't even need to load
    # the result into registers. We can just store the result directly.
    tma.async_store(out_desc, [pid * XBLOCK], smem)

    # Unlike TMA reads, the completion of TMA stores is tracked by commit
    # groups, just like async copies. Each async TMA store is implicitly
    # committed to an async store group. We can wait until there are at most
    # `pendings` outstanding TMA stores using `store_wait`. Note that the commit
    # groups for async copy and async TMA stores are separate.
    tma.store_wait(pendings=0)


def test_descriptor_contract[
    N: IntVar,
    Other: IntVar,
    Block: IntVar,
    OtherBlock: IntVar,
    Layout: IntVar,
    OtherLayout: IntVar,
](
    inp: gl.TmaInputDescriptor1D[N, Block, Layout],
    out: gl.TmaOutputDescriptor1D[N, Block, Layout],
    wrong_input_extent: gl.TmaInputDescriptor1D[Other, Block, Layout],
    wrong_output_extent: gl.TmaOutputDescriptor1D[Other, Block, Layout],
    wrong_output_block: gl.TmaOutputDescriptor1D[N, OtherBlock, Layout],
    wrong_output_layout: gl.TmaOutputDescriptor1D[N, Block, OtherLayout],
    block: Int[Block],
    wrong_block: Int[OtherBlock],
) -> None:
    memcpy_1d_tma_kernel(inp, out, block)
    memcpy_1d_tma_kernel(inp, wrong_output_extent, block)  # E: is not assignable
    # E: is not assignable
    memcpy_1d_tma_kernel(wrong_input_extent, out, block)
    memcpy_1d_tma_kernel(inp, wrong_output_block, block)  # E: is not assignable
    memcpy_1d_tma_kernel(inp, wrong_output_layout, block)  # E: is not assignable
    memcpy_1d_tma_kernel(inp, out, wrong_block)  # E: is not assignable


def test_tma_access[
    Length: IntVar,
    Block: IntVar,
    OtherBlock: IntVar,
    Layout: IntVar,
    OtherLayout: IntVar,
](
    inp: gl.TmaInputDescriptor1D[Length, Block, Layout],
    out: gl.TmaOutputDescriptor1D[Length, Block, Layout],
    other_out: gl.TmaOutputDescriptor1D[Length, Block, OtherLayout],
    block: Int[Block],
    other_block: Int[OtherBlock],
) -> None:
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    shared = gl.allocate_shared_memory(inp.dtype, [block], inp.layout)
    assert_type(shared, gl.TmaSharedBuffer1D[Block, Layout])
    start = gl.program_id(0) * block
    tma.async_load(inp, [start], bar, shared)
    tma.async_store(out, [start], shared)
    # E: No matching overload
    tma.async_load(
        inp,
        [gl.program_id(0) * other_block],
        bar,
        shared,
    )
    tma.async_store(other_out, [start], shared)  # E: No matching overload
    other_shared = gl.allocate_shared_memory(inp.dtype, [other_block], inp.layout)
    tma.async_load(inp, [start], bar, other_shared)  # E: No matching overload
    tma.async_store(out, [start], other_shared)  # E: No matching overload
    tma.async_store(inp, [start], shared)  # E: No matching overload
    tma.async_load(out, [start], bar, shared)  # E: No matching overload


def test_descriptor_assertions[Length: IntVar, Block: IntVar, Layout: IntVar](
    inp: gl.TmaInputDescriptor1D[Length, Block, Layout],
    out: gl.TmaOutputDescriptor1D[Length, Block, Layout],
) -> None:
    gl.static_assert(inp.block_type == out.block_type)
    gl.static_assert(inp.layout == out.layout)
    gl.static_assert(False)  # E: is not assignable


def test_mismatched_descriptor_metadata[
    Length: IntVar,
    Block: IntVar,
    OtherBlock: IntVar,
    Layout: IntVar,
    OtherLayout: IntVar,
](
    inp: gl.TmaInputDescriptor1D[Length, Block, Layout],
    different_block: gl.TmaOutputDescriptor1D[Length, OtherBlock, Layout],
    different_layout: gl.TmaOutputDescriptor1D[Length, Block, OtherLayout],
) -> None:
    gl.static_assert(
        inp.block_type == different_block.block_type  # E: is not assignable
    )
    gl.static_assert(inp.layout == different_layout.layout)  # E: is not assignable

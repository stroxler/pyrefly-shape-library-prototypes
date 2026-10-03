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

"""Check one-block TMA message descriptor, atomic flag, and output contracts."""

from typing import assert_type

from shape_extensions import Int, IntVar
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
)
from triton.language import OutPointer


@gluon.jit
def tma_message_passing_kernel[Size: IntVar, Layout: IntVar](
    message_desc: gl.MessageDescriptor1D[Size, Size, Layout],
    ready: gl.AtomicReadyFlagPointer1D,
    output: gl.MessageOutputPointer1D[Size],
    MESSAGE_SIZE: Int[Size],
):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    offsets = gl.arange(0, MESSAGE_SIZE, layout)
    smem = gl.allocate_shared_memory(
        message_desc.dtype, message_desc.block_shape, message_desc.layout
    )

    if pid == 0:
        # CTA 0 sends a message by staging it in shared memory and then using
        # TMA to write it to HBM.
        smem.store(offsets + 1000)
        fence_async_shared()
        tma.async_store(message_desc, [0], smem)

        # Before signaling CTA 1, wait for the TMA write to become visible in HBM.
        tma.store_wait(pendings=0, read_only=False)
        gl.atomic_xchg(ready, 1, sem="release", scope="gpu")
    else:
        # CTA 1 waits until the TMA message has been published, then reads it
        # back through TMA.
        ready_value = 0
        while ready_value != 1:
            ready_value = gl.atomic_add(ready, 0, sem="acquire", scope="gpu")

        bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
        mbarrier.init(bar, count=1)
        mbarrier.expect(bar, message_desc.block_type.nbytes)
        tma.async_load(message_desc, [0], bar, smem)
        mbarrier.wait(bar, phase=0, deps=[smem])
        mbarrier.invalidate(bar)
        gl.store(output + offsets, smem.load(layout))


def test_message_interface[Size: IntVar, Other: IntVar, Layout: IntVar](
    message: gl.MessageDescriptor1D[Size, Size, Layout],
    wrong_host_length: gl.MessageDescriptor1D[Other, Size, Layout],
    wrong_block: gl.MessageDescriptor1D[Size, Other, Layout],
    ready: gl.AtomicReadyFlagPointer1D,
    bad_flag: OutPointer[[1]],
    output: gl.MessageOutputPointer1D[Size],
    wrong_output: gl.MessageOutputPointer1D[Other],
    size: Int[Size],
    other: Int[Other],
) -> None:
    tma_message_passing_kernel(message, ready, output, size)
    tma_message_passing_kernel(
        wrong_host_length,  # E: is not assignable
        ready,
        output,  # E: is not assignable
        size,  # E: is not assignable
    )
    tma_message_passing_kernel(wrong_block, ready, output, size)  # E: is not assignable
    tma_message_passing_kernel(message, bad_flag, output, size)  # E: is not assignable
    tma_message_passing_kernel(
        message,
        ready,
        wrong_output,  # E: is not assignable
        size,
    )
    tma_message_passing_kernel(message, ready, output, other)  # E: is not assignable


def test_transfer_and_output[Size: IntVar, Other: IntVar, Layout: IntVar](
    message: gl.MessageDescriptor1D[Size, Size, Layout],
    wrong_host_length: gl.MessageDescriptor1D[Other, Size, Layout],
    output: gl.MessageOutputPointer1D[Size],
    wrong_output: gl.MessageOutputPointer1D[Other],
    size: Int[Size],
    other: Int[Other],
) -> None:
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    offsets = gl.arange(0, size, layout)
    wrong_offsets = gl.arange(0, other, layout)
    shared = gl.allocate_shared_memory(
        message.dtype, message.block_shape, message.layout
    )
    assert_type(shared, gl.MessageSharedBuffer1D[Size, Layout])
    shared.store(offsets + 1000)
    gl.store(output + offsets, shared.load(layout))
    shared.store(wrong_offsets + 1000)  # E: is not assignable
    gl.store(wrong_output + offsets, shared.load(layout))  # E: is not assignable
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    tma.async_store(message, [0], shared)
    tma.async_load(message, [0], bar, shared)
    tma.async_store(wrong_host_length, [0], shared)  # E: No matching overload
    tma.async_load(wrong_host_length, [0], bar, shared)  # E: No matching overload
    tma.store_wait(pendings=0, read_only=True)  # The stub does not prove visibility.


def test_atomic_roles(
    flag: gl.AtomicReadyFlagPointer1D,
    wrong_flag: OutPointer[[1]],
) -> None:
    gl.atomic_xchg(flag, 1, sem="release", scope="gpu")
    gl.atomic_add(flag, 0, sem="acquire", scope="gpu")
    gl.atomic_xchg(wrong_flag, 1, sem="release", scope="gpu")  # E: is not assignable
    gl.atomic_add(flag, 0, sem="release", scope="gpu")  # E: is not assignable
    gl.atomic_xchg(flag, 1, sem="acquire", scope="gpu")  # E: is not assignable


def test_warp_count_does_not_change_rank() -> None:
    assert_type(gl.BlockedLayout([1], [32], [1], [0]), gl.Layout1D)
    assert_type(gl.BlockedLayout([1], [32], [4], [0]), gl.Layout1D)

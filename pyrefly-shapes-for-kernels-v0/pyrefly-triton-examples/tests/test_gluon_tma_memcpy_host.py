# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The Python wrapper body is copied from Triton's python/tutorials/gluon/04-tma.py.
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

# Static-only semantic annotations are not executable Gluon annotations.
# @lint-ignore-every AUTODEPS2

"""Ground a TMA descriptor and actual grid call in Torch allocation shapes."""

from typing import assert_type

import torch
import triton
from shape_extensions import Int, IntVar
from tests.test_gluon_tma_memcpy import memcpy_1d_tma_kernel
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


def memcpy_1d_tma[Length: IntVar, Block: IntVar](
    input: torch.Tensor[[Length]],
    output: torch.Tensor[[Length]],
    XBLOCK: Int[Block] = 8192,
):
    assert input.shape == output.shape

    # The layout for a tensor descriptor is always an NVMMASharedLayout. We can
    # use this helper to grab the default NVMMASharedLayout, but sometimes you
    # might need a different layout.
    block_shape = [XBLOCK]
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, gl.float32)

    # Wrap the tensors in tensor descriptors.
    in_desc = TensorDescriptor.from_tensor(input, block_shape, layout)
    out_desc = TensorDescriptor.from_tensor(output, block_shape, layout)

    grid = (triton.cdiv(input.numel(), XBLOCK),)
    # Our kernel only uses scalars, so just a single warp is enough.
    # E: is not subscriptable
    memcpy_1d_tma_kernel[grid](in_desc, out_desc, XBLOCK, num_warps=1)


def test_torch_host_boundary[
    Length: IntVar,
    Other: IntVar,
    Block: IntVar,
    OtherBlock: IntVar,
](
    length: Int[Length],
    other: Int[Other],
    block: Int[Block],
    other_block: Int[OtherBlock],
) -> None:
    input = torch.empty((length,), device="cuda")
    output = torch.empty_like(input)
    wrong_input = torch.empty((other,), device="cuda")
    wrong_output = torch.empty((other,), device="cuda")
    assert_type(input, torch.Tensor[[Length]])
    assert_type(output, torch.Tensor[[Length]])
    memcpy_1d_tma(input, output, block)
    memcpy_1d_tma(input, wrong_output, block)  # E: is not assignable
    memcpy_1d_tma(wrong_input, output, block)  # E: is not assignable

    block_shape = [block]
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, gl.float32)
    input_desc = TensorDescriptor.from_tensor(input, block_shape, layout)
    output_desc = TensorDescriptor.from_tensor(output, block_shape, layout)
    wrong_extent = TensorDescriptor.from_tensor(wrong_output, block_shape, layout)
    wrong_block = TensorDescriptor.from_tensor(
        input,
        [other_block],
        gl.NVMMASharedLayout.get_default_for([other_block], gl.float32),
    )
    assert_type(input_desc, gl.TmaDescriptor1D[Length, Block, int])
    assert_type(output_desc, gl.TmaDescriptor1D[Length, Block, int])
    memcpy_1d_tma_kernel(input_desc, output_desc, block)
    memcpy_1d_tma_kernel(wrong_extent, output_desc, block)  # E: is not assignable
    # E: is not assignable
    memcpy_1d_tma_kernel(wrong_block, output_desc, block)  # E: is not assignable
    # A real TensorDescriptor may be loaded or stored: source/destination
    # direction is not established by the host-side constructor.
    memcpy_1d_tma_kernel(output_desc, input_desc, block)


def test_accepted_block_rank_gap[Length: IntVar, Block: IntVar](
    input: torch.Tensor[[Length]],
    output: torch.Tensor[[Length]],
    block: Int[Block],
) -> None:
    memcpy_1d_tma(input, output, block)
    # list[Int[Block]] forgets the number of entries; the real constructor
    # checks block-shape rank against the input tensor at runtime.
    wrong_rank = [block, block]
    layout = gl.NVMMASharedLayout.get_default_for(wrong_rank, gl.float32)
    TensorDescriptor.from_tensor(input, wrong_rank, layout)

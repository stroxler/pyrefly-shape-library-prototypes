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

"""Static shapes and contiguous row strides for the pipeline transpose."""

from typing import assert_type

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def transpose_kernel[M: IntVar, N: IntVar](
    x_ptr: tl.TransposeInputPointer[M, N],
    o_ptr: tl.TransposeOutputPointer[M, N],
    M: Int[M],
    N: Int[N],
):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    x = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
    tl.store(o_ptr + rn[:, None] * M + rm[None, :], tl.trans(x))


def test_transpose_boundary[M: IntVar, N: IntVar, Other: IntVar](
    x: tl.TransposeInputPointer[M, N],
    output: tl.TransposeOutputPointer[M, N],
    wrong_input: tl.TransposeInputPointer[M, Other],
    wrong_output: tl.TransposeOutputPointer[M, Other],
    m: Int[M],
    n: Int[N],
) -> None:
    transpose_kernel(x, output, m, n)
    transpose_kernel(
        wrong_input,
        output,  # E: is not assignable
        m,
        n,  # E: is not assignable
    )
    transpose_kernel(x, wrong_output, m, n)  # E: is not assignable


def test_transpose_value[M: IntVar, N: IntVar](
    output_tile: tl.TransposeOutputTile[M, N], input_tile: tl.tensor[[M, N]]
) -> None:
    assert_type(tl.trans(input_tile), tl.tensor[[N, M]])
    tl.store(output_tile, tl.trans(input_tile))
    tl.store(output_tile, input_tile)  # E: is not assignable


# Changing only the output allocation shape rejects the original store address.
@triton.jit
def transpose_kernel_wrong_output[M: IntVar, N: IntVar, Other: IntVar](
    x_ptr: tl.TransposeInputPointer[M, N],
    o_ptr: tl.TransposeOutputPointer[M, Other],
    M: Int[M],
    N: Int[N],
):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    x = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
    tl.store(  # E: No matching overload
        o_ptr  # E: is not supported between
        + rn[:, None] * M
        + rm[None, :],
        tl.trans(x),
    )


# Changing only the input allocation shape rejects the original load address.
@triton.jit
def transpose_kernel_wrong_input[M: IntVar, N: IntVar, Other: IntVar](
    x_ptr: tl.TransposeInputPointer[M, Other],
    o_ptr: tl.TransposeOutputPointer[M, N],
    M: Int[M],
    N: Int[N],
):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    x = tl.load(
        x_ptr  # E: is not supported between # E: is not supported between
        + rm[:, None] * N
        + rn[None, :]
    )
    tl.store(  # E: No matching overload
        o_ptr + rn[:, None] * M + rm[None, :], tl.trans(x)
    )

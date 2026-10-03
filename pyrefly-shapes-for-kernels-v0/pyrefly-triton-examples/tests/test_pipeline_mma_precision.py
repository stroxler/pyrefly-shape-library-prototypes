# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The bodies are copied from Triton's python/tutorials/compilation-pipeline/
# 10_mma_precision_numerics.py and 14_mma_tensorcore_vs_fma.py.
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

"""Check a complete single-tile matmul for both supported precision choices."""

from typing import Literal

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def dot_kernel[M: IntVar, N: IntVar, K: IntVar](
    a_ptr: tl.PipelineMatrixInputPointer[M, K, M, K],
    b_ptr: tl.PipelineMatrixInputPointer[K, N, K, N],
    c_ptr: tl.PipelineMatrixOutputPointer[M, N, M, N],
    M: Int[M],
    N: Int[N],
    K: Int[K],
    PREC: Literal["ieee", "tf32"],
):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    c = tl.dot(a, b, input_precision=PREC)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], c)


def test_host_shape_and_precision[M: IntVar, N: IntVar, K: IntVar, Other: IntVar](
    a: tl.PipelineMatrixInputPointer[M, K, M, K],
    b: tl.PipelineMatrixInputPointer[K, N, K, N],
    c: tl.PipelineMatrixOutputPointer[M, N, M, N],
    wrong_b: tl.PipelineMatrixInputPointer[Other, N, K, N],
    wrong_c: tl.PipelineMatrixOutputPointer[M, Other, M, N],
    m: Int[M],
    n: Int[N],
    k: Int[K],
) -> None:
    dot_kernel(a, b, c, m, n, k, "ieee")
    dot_kernel(a, b, c, m, n, k, "tf32")
    dot_kernel(a, b, c, m, n, k, "invalid")  # E: is not assignable
    dot_kernel(a, wrong_b, c, m, n, k, "ieee")  # E: is not assignable
    dot_kernel(a, b, wrong_c, m, n, k, "tf32")  # E: is not assignable


# Changing only C's allocation width rejects the unchanged output address.
@triton.jit
def dot_kernel_wrong_c[M: IntVar, N: IntVar, K: IntVar, Other: IntVar](
    a_ptr: tl.PipelineMatrixInputPointer[M, K, M, K],
    b_ptr: tl.PipelineMatrixInputPointer[K, N, K, N],
    c_ptr: tl.PipelineMatrixOutputPointer[M, Other, M, N],
    M: Int[M],
    N: Int[N],
    K: Int[K],
    PREC: Literal["ieee", "tf32"],
):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    c = tl.dot(a, b, input_precision=PREC)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], c)  # E: is not supported between


# Tutorial 14 uses the same operators as 10 but names the dot result `acc`.
@triton.jit
def dot_kernel_architectural[M: IntVar, N: IntVar, K: IntVar](
    a_ptr: tl.PipelineMatrixInputPointer[M, K, M, K],
    b_ptr: tl.PipelineMatrixInputPointer[K, N, K, N],
    c_ptr: tl.PipelineMatrixOutputPointer[M, N, M, N],
    M: Int[M],
    N: Int[N],
    K: Int[K],
    PREC: Literal["ieee", "tf32"],
):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    acc = tl.dot(a, b, input_precision=PREC)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)

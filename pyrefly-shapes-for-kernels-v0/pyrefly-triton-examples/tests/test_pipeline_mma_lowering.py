# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The kernel body is copied from Triton's python/tutorials/compilation-pipeline/09_dot_to_mma_lowering.py.
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

"""Static allocation and contraction checks for the MMA-lowering matmul."""

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def matmul_kernel[M: IntVar, N: IntVar, K: IntVar, BM: IntVar, BN: IntVar, BK: IntVar](
    a_ptr: tl.PipelineMatrixInputPointer[M, K, BM, BK],
    b_ptr: tl.PipelineMatrixInputPointer[K, N, BK, BN],
    c_ptr: tl.PipelineMatrixOutputPointer[M, N, BM, BN],
    M: Int[M],
    N: Int[N],
    K: Int[K],
    BM: Int[BM],
    BN: Int[BN],
    BK: Int[BK],
):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k in range(0, K, BK):
        rk = k + tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
        acc += tl.dot(a, b)  # fp16 inputs => natural tensor-core path
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.float16))


def test_host_shapes[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Other: IntVar,
](
    a: tl.PipelineMatrixInputPointer[M, K, BM, BK],
    b: tl.PipelineMatrixInputPointer[K, N, BK, BN],
    c: tl.PipelineMatrixOutputPointer[M, N, BM, BN],
    wrong_a: tl.PipelineMatrixInputPointer[M, Other, BM, BK],
    wrong_c: tl.PipelineMatrixOutputPointer[M, Other, BM, BN],
    m: Int[M],
    n: Int[N],
    k: Int[K],
    bm: Int[BM],
    bn: Int[BN],
    bk: Int[BK],
) -> None:
    matmul_kernel(a, b, c, m, n, k, bm, bn, bk)
    matmul_kernel(
        wrong_a,
        b,  # E: is not assignable
        c,
        m,
        n,
        k,  # E: is not assignable
        bm,
        bn,
        bk,
    )
    matmul_kernel(a, b, wrong_c, m, n, k, bm, bn, bk)  # E: is not assignable


# Changing the source allocation K extent rejects the original row stride.
@triton.jit
def matmul_kernel_wrong_a[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Other: IntVar,
](
    a_ptr: tl.PipelineMatrixInputPointer[M, Other, BM, BK],
    b_ptr: tl.PipelineMatrixInputPointer[K, N, BK, BN],
    c_ptr: tl.PipelineMatrixOutputPointer[M, N, BM, BN],
    M: Int[M],
    N: Int[N],
    K: Int[K],
    BM: Int[BM],
    BN: Int[BN],
    BK: Int[BK],
):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k in range(0, K, BK):
        rk = k + tl.arange(0, BK)
        a = tl.load(
            a_ptr + rm[:, None] * K + rk[None, :]  # E: is not supported between
        )
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
        acc += tl.dot(a, b)  # fp16 inputs => natural tensor-core path
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.float16))


# Changing the output allocation N extent rejects the original store row stride.
@triton.jit
def matmul_kernel_wrong_c[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Other: IntVar,
](
    a_ptr: tl.PipelineMatrixInputPointer[M, K, BM, BK],
    b_ptr: tl.PipelineMatrixInputPointer[K, N, BK, BN],
    c_ptr: tl.PipelineMatrixOutputPointer[M, Other, BM, BN],
    M: Int[M],
    N: Int[N],
    K: Int[K],
    BM: Int[BM],
    BN: Int[BN],
    BK: Int[BK],
):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k in range(0, K, BK):
        rk = k + tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
        acc += tl.dot(a, b)  # fp16 inputs => natural tensor-core path
    tl.store(
        c_ptr + rm[:, None] * N + rn[None, :],  # E: is not supported between
        acc.to(tl.float16),
    )

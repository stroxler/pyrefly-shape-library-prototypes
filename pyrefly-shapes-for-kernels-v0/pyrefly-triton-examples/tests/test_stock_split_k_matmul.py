# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The kernel body is copied from Triton's 12-split-k-matmul.py.
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
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# This static-only kernel is checked by Pyrefly, not executed as a Buck test.
# @lint-ignore-every AUTODEPS2

"""The unsplit baseline kernel in Triton's 12-split-k-matmul.py."""

from typing import assert_type

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def _stock_triton_kernel[
    MDim: IntVar,
    NDim: IntVar,
    KDim: IntVar,
    AM: IntVar,
    AK: IntVar,
    BK: IntVar,
    BN: IntVar,
    CM: IntVar,
    CN: IntVar,
    BM: IntVar,
    BNN: IntVar,
    BKK: IntVar,
    Group: IntVar,
](
    a_ptr: tl.InMatrixPointer[MDim, KDim, AM, AK],
    b_ptr: tl.InMatrixPointer[KDim, NDim, BK, BN],
    c_ptr: tl.OutMatrixPointer[MDim, NDim, CM, CN],
    M: Int[MDim],
    N: Int[NDim],
    K: Int[KDim],
    stride_am: Int[AM],
    stride_ak: Int[AK],
    stride_bk: Int[BK],
    stride_bn: Int[BN],
    stride_cm: Int[CM],
    stride_cn: Int[CN],
    BLOCK_M: Int[BM],
    BLOCK_N: Int[BNN],
    BLOCK_K: Int[BKK],
    GROUP_SIZE_M: Int[Group],
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def test_wrong_a_row_stride[
    M: IntVar,
    K: IntVar,
    RS: IntVar,
    Other: IntVar,
    CS: IntVar,
    BM: IntVar,
    BK: IntVar,
](
    a: tl.InMatrixPointer[M, K, RS, CS],
    row: tl.WrappedRowAxisOffsets[M, [BM]],
    col: tl.ColumnAxisOffsets[[BK]],
    wrong_stride: Int[Other],
    column_stride: Int[CS],
) -> None:
    a + row * wrong_stride + col * column_stride  # E: No matching overload


def test_wrong_b_column_extent[
    K: IntVar,
    N: IntVar,
    Other: IntVar,
    RS: IntVar,
    CS: IntVar,
    BK: IntVar,
    BN: IntVar,
](
    b: tl.InMatrixPointer[K, N, RS, CS],
    row: tl.RowAxisOffsets[[BK]],
    col: tl.WrappedColumnAxisOffsets[Other, [BN]],
    row_stride: Int[RS],
    column_stride: Int[CS],
) -> None:
    b + row * row_stride + col * column_stride  # E: is not assignable to parameter


def test_wrong_output_mask[
    M: IntVar,
    N: IntVar,
    Other: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    ptrs: tl.OutMatrixTilePointers[M, N, BM, BN],
    tile: tl.tensor[[BM, BN]],
    wrong: tl.MatrixMask[M, Other, [BM], [BN]],
) -> None:
    tl.store(ptrs, tile, mask=wrong)  # E: No matching overload


def test_wrong_boundary_output[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    Other: IntVar,
    AM: IntVar,
    AK: IntVar,
    BK: IntVar,
    BN: IntVar,
    CM: IntVar,
    CN: IntVar,
    BM: IntVar,
    BNN: IntVar,
    BKK: IntVar,
    Group: IntVar,
](
    a: tl.InMatrixPointer[M, K, AM, AK],
    b: tl.InMatrixPointer[K, N, BK, BN],
    c: tl.OutMatrixPointer[M, Other, CM, CN],
    m: Int[M],
    n: Int[N],
    k: Int[K],
    am: Int[AM],
    ak: Int[AK],
    bk: Int[BK],
    bn: Int[BN],
    cm: Int[CM],
    cn: Int[CN],
    bm: Int[BM],
    bnn: Int[BNN],
    bkk: Int[BKK],
    group: Int[Group],
) -> None:
    _stock_triton_kernel(
        a,
        b,
        c,  # E: is not assignable to parameter
        m,
        n,
        k,
        am,
        ak,
        bk,
        bn,
        cm,
        cn,
        bm,
        bnn,
        bkk,
        group,
    )


def test_unverified_k_mask_bound[
    M: IntVar,
    K: IntVar,
    BM: IntVar,
    BK: IntVar,
    RS: IntVar,
    CS: IntVar,
](
    pointers: tl.WrappedRowMatrixTilePointers[M, K, BM, BK, RS, CS],
    offsets: tl.ColumnAxisOffsets[[BK]],
    arbitrary_bound: int,
) -> None:
    # Iterating over K erases the runtime K-tail mask's specific bound.
    assert_type(
        tl.load(pointers, mask=offsets < arbitrary_bound, other=0.0),
        tl.tensor[[BM, BK]],
    )

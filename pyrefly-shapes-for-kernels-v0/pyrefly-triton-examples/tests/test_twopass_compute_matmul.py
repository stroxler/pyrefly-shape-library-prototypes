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

"""Split-K scratch production in Triton's two-pass matmul."""

from typing import assert_type

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def _twopass_compute_kernel[
    MDim: IntVar,
    NDim: IntVar,
    KDim: IntVar,
    SplitDim: IntVar,
    SplitLength: IntVar,
    AM: IntVar,
    AK: IntVar,
    BK: IntVar,
    BN: IntVar,
    SM: IntVar,
    SN: IntVar,
    SK: IntVar,
    BM: IntVar,
    BNN: IntVar,
    BKK: IntVar,
    Group: IntVar,
](
    a_ptr: tl.InMatrixPointer[MDim, KDim, AM, AK],
    b_ptr: tl.InMatrixPointer[KDim, NDim, BK, BN],
    scratch_ptr: tl.Scratch3DPointer[SplitDim, MDim, NDim, SK, SM, SN],
    M: Int[MDim],
    N: Int[NDim],
    K: Int[KDim],
    K_PER_SPLIT: Int[SplitLength],
    stride_am: Int[AM],
    stride_ak: Int[AK],
    stride_bk: Int[BK],
    stride_bn: Int[BN],
    stride_sm: Int[SM],  # scratch stride for M dim (within one split-k slice)
    stride_sn: Int[SN],  # scratch stride for N dim
    stride_sk: Int[SK],  # scratch stride between split-k slices (= M * N)
    SPLIT_K: Int[SplitDim],
    BLOCK_M: Int[BM],
    BLOCK_N: Int[BNN],
    BLOCK_K: Int[BKK],
    GROUP_SIZE_M: Int[Group],
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    k_start = pid_k * K_PER_SPLIT
    k_end = min(k_start + K_PER_SPLIT, K)
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = (
        a_ptr + offs_am[:, None] * stride_am + (k_start + offs_k[None, :]) * stride_ak
    )
    b_ptrs = (
        b_ptr + (k_start + offs_k[:, None]) * stride_bk + offs_bn[None, :] * stride_bn
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K_PER_SPLIT, BLOCK_K)):
        k_remaining = k_end - (k_start + k * BLOCK_K)
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    # Store fp32 partial result into scratch[pid_k, :, :]
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    scratch_ptrs = (
        scratch_ptr
        + pid_k * stride_sk
        + offs_cm[:, None] * stride_sm
        + offs_cn[None, :] * stride_sn
    )
    tl.store(scratch_ptrs, acc, mask=c_mask)


def test_wrong_scratch_strides[
    Split: IntVar,
    M: IntVar,
    N: IntVar,
    SK: IntVar,
    SM: IntVar,
    SN: IntVar,
    Other: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    scratch: tl.Scratch3DPointer[Split, M, N, SK, SM, SN],
    split_stride: Int[SK],
    row_stride: Int[SM],
    wrong_stride: Int[Other],
    row: tl.RowAxisOffsets[[BM]],
    col: tl.ColumnAxisOffsets[[BN]],
) -> None:
    scratch + tl.program_id(1) * wrong_stride  # E: is not assignable
    split_ptr = scratch + tl.program_id(1) * split_stride
    split_ptr + row * wrong_stride  # E: is not assignable
    row_ptr = split_ptr + row * row_stride
    row_ptr + col * wrong_stride  # E: is not assignable


def test_wrong_scratch_store[
    Split: IntVar,
    M: IntVar,
    N: IntVar,
    Other: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    pointers: tl.ScratchTilePointers[Split, M, N, BM, BN],
    tile: tl.tensor[[BM, BN]],
    wrong_tile: tl.tensor[[BM, Other]],
    wrong_row_mask: tl.MatrixMask[Other, N, [BM], [BN]],
    wrong_column_mask: tl.MatrixMask[M, Other, [BM], [BN]],
    mask: tl.MatrixMask[M, N, [BM], [BN]],
) -> None:
    tl.store(pointers, tile, mask=wrong_row_mask)  # E: No matching overload
    tl.store(pointers, tile, mask=wrong_column_mask)  # E: No matching overload
    tl.store(pointers, wrong_tile, mask=mask)  # E: No matching overload


def test_wrong_scratch_allocation[
    Split: IntVar,
    M: IntVar,
    N: IntVar,
    K: IntVar,
    Other: IntVar,
    AM: IntVar,
    AK: IntVar,
    BK: IntVar,
    BN: IntVar,
    SM: IntVar,
    SN: IntVar,
    SK: IntVar,
    BM: IntVar,
    BNN: IntVar,
    BKK: IntVar,
    Group: IntVar,
    SplitLength: IntVar,
](
    a: tl.InMatrixPointer[M, K, AM, AK],
    b: tl.InMatrixPointer[K, N, BK, BN],
    wrong_rows: tl.Scratch3DPointer[Split, Other, N, SK, SM, SN],
    wrong_columns: tl.Scratch3DPointer[Split, M, Other, SK, SM, SN],
    m: Int[M],
    n: Int[N],
    k: Int[K],
    k_per_split: Int[SplitLength],
    am: Int[AM],
    ak: Int[AK],
    bk: Int[BK],
    bn: Int[BN],
    sm: Int[SM],
    sn: Int[SN],
    sk: Int[SK],
    split: Int[Split],
    bm: Int[BM],
    bnn: Int[BNN],
    bkk: Int[BKK],
    group: Int[Group],
) -> None:
    _twopass_compute_kernel(
        a,
        b,
        wrong_rows,  # E: is not assignable to parameter
        m,
        n,
        k,
        k_per_split,
        am,
        ak,
        bk,
        bn,
        sm,
        sn,
        sk,
        split,
        bm,
        bnn,
        bkk,
        group,
    )
    _twopass_compute_kernel(
        a,
        b,
        wrong_columns,  # E: is not assignable to parameter
        m,
        n,
        k,
        k_per_split,
        am,
        ak,
        bk,
        bn,
        sm,
        sn,
        sk,
        split,
        bm,
        bnn,
        bkk,
        group,
    )


def test_unverified_split_program_axis[
    Split: IntVar,
    M: IntVar,
    N: IntVar,
    SK: IntVar,
    SM: IntVar,
    SN: IntVar,
](
    scratch: tl.Scratch3DPointer[Split, M, N, SK, SM, SN],
    split_stride: Int[SK],
) -> None:
    # program_id(0) is accepted for the split axis: the grid is untyped.
    assert_type(
        scratch + tl.program_id(0) * split_stride,
        tl.ScratchSlicePointer[Split, M, N, SM, SN],
    )


def test_unverified_k_bound[
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
    assert_type(
        tl.load(pointers, mask=offsets < arbitrary_bound, other=0.0),
        tl.tensor[[BM, BK]],
    )

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

"""Split-K scratch reduction in Triton's two-pass matmul."""

from typing import assert_type

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def _twopass_reduce_kernel[
    Split: IntVar,
    MDim: IntVar,
    NDim: IntVar,
    SM: IntVar,
    SN: IntVar,
    SK: IntVar,
    CM: IntVar,
    CN: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    scratch_ptr: tl.Scratch3DPointer[Split, MDim, NDim, SK, SM, SN],
    c_ptr: tl.OutMatrixPointer[MDim, NDim, CM, CN],
    M: Int[MDim],
    N: Int[NDim],
    stride_sm: Int[SM],
    stride_sn: Int[SN],
    stride_sk: tl.SplitStride[SK],
    stride_cm: Int[CM],
    stride_cn: Int[CN],
    SPLIT_K: Int[Split],
    BLOCK_M: Int[BM],
    BLOCK_N: Int[BN],
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Sum across split-K slices
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for sk in range(SPLIT_K):
        s_ptrs = (
            scratch_ptr
            + sk * stride_sk
            + offs_m[:, None] * stride_sm
            + offs_n[None, :] * stride_sn
        )
        partial = tl.load(s_ptrs, mask=mask, other=0.0)
        acc += partial
    # Store as fp16
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)


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
    slice_stride: tl.SplitStride[SK],
    wrong_slice_stride: tl.SplitStride[Other],
    row_stride: Int[SM],
    wrong_stride: Int[Other],
    row: tl.RowAxisOffsets[[BM]],
    col: tl.ColumnAxisOffsets[[BN]],
    index: int,
) -> None:
    scratch + index * wrong_slice_stride  # E: is not assignable
    slice_ptr = scratch + index * slice_stride
    slice_ptr + row * wrong_stride  # E: is not assignable
    row_ptr = slice_ptr + row * row_stride
    row_ptr + col * wrong_stride  # E: is not assignable


def test_wrong_scratch_load[
    Split: IntVar,
    M: IntVar,
    N: IntVar,
    Other: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    ptrs: tl.ScratchTilePointers[Split, M, N, BM, BN],
    wrong_rows: tl.MatrixMask[Other, N, [BM], [BN]],
    wrong_columns: tl.MatrixMask[M, Other, [BM], [BN]],
    wrong_tile: tl.MatrixMask[M, N, [BM], [Other]],
) -> None:
    tl.load(ptrs, mask=wrong_rows, other=0.0)  # E: is not assignable
    tl.load(ptrs, mask=wrong_columns, other=0.0)  # E: is not assignable
    tl.load(ptrs, mask=wrong_tile, other=0.0)  # E: is not assignable


def test_wrong_output_mask[
    M: IntVar,
    N: IntVar,
    Other: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    ptrs: tl.OutMatrixTilePointers[M, N, BM, BN],
    value: tl.tensor[[BM, BN]],
    wrong_columns: tl.MatrixMask[M, Other, [BM], [BN]],
) -> None:
    tl.store(ptrs, value, mask=wrong_columns)  # E: No matching overload


def test_wrong_boundary_allocations[
    Split: IntVar,
    M: IntVar,
    N: IntVar,
    Other: IntVar,
    SM: IntVar,
    SN: IntVar,
    SK: IntVar,
    CM: IntVar,
    CN: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    scratch_rows: tl.Scratch3DPointer[Split, Other, N, SK, SM, SN],
    scratch_columns: tl.Scratch3DPointer[Split, M, Other, SK, SM, SN],
    scratch: tl.Scratch3DPointer[Split, M, N, SK, SM, SN],
    c: tl.OutMatrixPointer[M, N, CM, CN],
    wrong_c: tl.OutMatrixPointer[M, Other, CM, CN],
    m: Int[M],
    n: Int[N],
    sm: Int[SM],
    sn: Int[SN],
    sk: tl.SplitStride[SK],
    cm: Int[CM],
    cn: Int[CN],
    split: Int[Split],
    bm: Int[BM],
    bn: Int[BN],
) -> None:
    _twopass_reduce_kernel(
        scratch_rows,
        c,  # E: is not assignable to parameter
        m,  # E: is not assignable to parameter
        n,
        sm,
        sn,
        sk,
        cm,
        cn,
        split,
        bm,
        bn,
    )
    _twopass_reduce_kernel(
        scratch_columns,
        c,  # E: is not assignable to parameter
        m,
        n,  # E: is not assignable to parameter
        sm,
        sn,
        sk,
        cm,
        cn,
        split,
        bm,
        bn,
    )
    _twopass_reduce_kernel(
        scratch,
        wrong_c,  # E: is not assignable to parameter
        m,
        n,
        sm,
        sn,
        sk,
        cm,
        cn,
        split,
        bm,
        bn,
    )


def test_unverified_split_index[
    Split: IntVar,
    M: IntVar,
    N: IntVar,
    SM: IntVar,
    SN: IntVar,
    SK: IntVar,
](
    scratch: tl.Scratch3DPointer[Split, M, N, SK, SM, SN],
    stride: tl.SplitStride[SK],
    unrelated_index: int,
) -> None:
    # An arbitrary index has the same stride role, without a split bound.
    assert_type(
        scratch + unrelated_index * stride,
        tl.ScratchSlicePointer[Split, M, N, SM, SN],
    )

# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The kernel bodies are copied from Triton's 09-persistent-matmul.py.
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

# These static-only kernels are checked by Pyrefly, not executed as Buck tests.
# @lint-ignore-every AUTODEPS2

"""The first persistent pointer matmul in Triton's 09-persistent-matmul.py."""

from typing import assert_type

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def _compute_pid[Group: IntVar, SMs: IntVar](
    tile_id: int,
    num_pid_in_group: int,
    num_pid_m: int,
    GROUP_SIZE_M: Int[Group],
    NUM_SMS: Int[SMs],
):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def matmul_kernel_persistent[
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
    SMs: IntVar,
](
    a_ptr: tl.InMatrixPointer[MDim, KDim, AM, AK],
    b_ptr: tl.InMatrixPointer[KDim, NDim, BK, BN],
    c_ptr: tl.OutMatrixPointer[MDim, NDim, CM, CN],  #
    M: Int[MDim],
    N: Int[NDim],
    K: Int[KDim],  #
    stride_am: Int[AM],
    stride_ak: Int[AK],  #
    stride_bk: Int[BK],
    stride_bn: Int[BN],  #
    stride_cm: Int[CM],
    stride_cn: Int[CN],  #
    BLOCK_SIZE_M: Int[BM],  #
    BLOCK_SIZE_N: Int[BNN],  #
    BLOCK_SIZE_K: Int[BKK],  #
    GROUP_SIZE_M: Int[Group],  #
    NUM_SMS: Int[SMs],  #
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    # NOTE: There is currently a bug in blackwell pipelining that means it can't handle a value being
    # used in both the prologue and epilogue, so we duplicate the counters as a work-around.
    tile_id_c = start_pid - NUM_SMS

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (
                offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
            )
            b_ptrs = b_ptr + (
                offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
            )

            a = tl.load(
                a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            b = tl.load(
                b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            accumulator = tl.dot(a, b, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if c_ptr.dtype.element_ty == tl.float8e4nv:
            c = accumulator.to(tl.float8e4nv)
        else:
            c = accumulator.to(tl.float16)
        tl.store(c_ptrs, c, mask=c_mask)


def test_clamp_retains_bound[Dim: IntVar, Other: IntVar, Block: IntVar](
    offsets: tl.Offsets[[Block]], bound: Int[Dim], wrong: Int[Other], block: Int[Block]
) -> None:
    clamped = tl.where(offsets < bound, offsets, 0)
    assert_type(
        tl.max_contiguous(tl.multiple_of(clamped, block), block),
        tl.ClampedOffsets[Dim, [Block]],
    )
    assert_type(
        tl.where(offsets < wrong, offsets, 0), tl.ClampedOffsets[Other, [Block]]
    )


def test_wrong_a_stride[
    M: IntVar,
    K: IntVar,
    RS: IntVar,
    Other: IntVar,
    CS: IntVar,
    BM: IntVar,
    BK: IntVar,
](
    a: tl.InMatrixPointer[M, K, RS, CS],
    row: tl.ClampedRowAxisOffsets[M, [BM]],
    col: tl.ColumnAxisOffsets[[BK]],
    wrong_stride: Int[Other],
    col_stride: Int[CS],
) -> None:
    a + (row * wrong_stride + col * col_stride)  # E: No matching overload


def test_wrong_b_clamp[
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
    col: tl.ClampedColumnAxisOffsets[Other, [BN]],
    row_stride: Int[RS],
    col_stride: Int[CS],
) -> None:
    b + (row * row_stride + col * col_stride)  # E: No matching overload


def test_wrong_b_column_stride[
    K: IntVar,
    N: IntVar,
    RS: IntVar,
    CS: IntVar,
    Other: IntVar,
    BK: IntVar,
    BN: IntVar,
](
    b: tl.InMatrixPointer[K, N, RS, CS],
    row: tl.RowAxisOffsets[[BK]],
    col: tl.ClampedColumnAxisOffsets[N, [BN]],
    row_stride: Int[RS],
    wrong_stride: Int[Other],
) -> None:
    b + (row * row_stride + col * wrong_stride)  # E: No matching overload


def test_wrong_c_row_stride[
    M: IntVar,
    N: IntVar,
    RS: IntVar,
    CS: IntVar,
    Other: IntVar,
    BM: IntVar,
](
    c: tl.OutMatrixPointer[M, N, RS, CS],
    rows: tl.RowAxisOffsets[[BM]],
    wrong_stride: Int[Other],
) -> None:
    c + rows * wrong_stride  # E: is not assignable to parameter


def test_wrong_k_mask_bound[
    M: IntVar,
    K: IntVar,
    Other: IntVar,
    BM: IntVar,
    BK: IntVar,
    RS: IntVar,
    CS: IntVar,
](
    ptrs: tl.ClampedRowMatrixTilePointers[M, K, BM, BK, RS, CS],
    mask: tl.ColumnMask[Other, [BK]],
) -> None:
    tl.load(ptrs, mask=mask, other=0.0)  # E: No matching overload


def test_unverified_runtime_k_bound[
    M: IntVar,
    K: IntVar,
    BM: IntVar,
    BK: IntVar,
    RS: IntVar,
    CS: IntVar,
](
    ptrs: tl.ClampedRowMatrixTilePointers[M, K, BM, BK, RS, CS],
    mask_offsets: tl.ColumnAxisOffsets[[BK]],
    arbitrary_bound: int,
) -> None:
    # Known gap: the actual `K - ki * BLOCK_SIZE_K` also widens to plain int.
    assert_type(
        tl.load(ptrs, mask=mask_offsets < arbitrary_bound, other=0.0),
        tl.tensor[[BM, BK]],
    )


def test_wrong_contraction[M: IntVar, N: IntVar, K: IntVar, Other: IntVar](
    a: tl.tensor[[M, K]], b: tl.tensor[[Other, N]], acc: tl.tensor[[M, N]]
) -> None:
    tl.dot(a, b, acc)  # E: is not assignable to parameter


def test_wrong_output_mask[
    M: IntVar,
    N: IntVar,
    Other: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    ptrs: tl.OutMatrixTilePointers[M, N, BM, BN],
    value: tl.tensor[[BM, BN]],
    mask: tl.MatrixMask[M, Other, [BM], [BN]],
) -> None:
    tl.store(ptrs, value, mask=mask)  # E: No matching overload


def test_wrong_output_value[
    M: IntVar,
    N: IntVar,
    BM: IntVar,
    BN: IntVar,
    Other: IntVar,
](
    ptrs: tl.OutMatrixTilePointers[M, N, BM, BN],
    value: tl.tensor[[BM, Other]],
    mask: tl.MatrixMask[M, N, [BM], [BN]],
) -> None:
    tl.store(ptrs, value, mask=mask)  # E: No matching overload


def test_wrong_input_allocation[
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
    SMs: IntVar,
](
    a: tl.InMatrixPointer[Other, K, AM, AK],
    b: tl.InMatrixPointer[K, N, BK, BN],
    c: tl.OutMatrixPointer[M, N, CM, CN],
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
    sms: Int[SMs],
) -> None:
    matmul_kernel_persistent(
        a,
        b,
        c,  # E: is not assignable to parameter
        m,  # E: is not assignable to parameter
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
        sms,
    )


def test_wrong_output_allocation[
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
    SMs: IntVar,
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
    sms: Int[SMs],
) -> None:
    matmul_kernel_persistent(
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
        sms,
    )


def test_unverified_tile_counter[Group: IntVar, SMs: IntVar](
    computed_tile: int,
    unrelated_output_tile: int,
    tiles_per_group: int,
    num_m_tiles: int,
    group: Int[Group],
    sms: Int[SMs],
) -> None:
    # Both tile indices remain `int`; the second is not required to match the first.
    assert_type(
        _compute_pid(unrelated_output_tile, tiles_per_group, num_m_tiles, group, sms),
        tuple[int, int],
    )
    assert_type(computed_tile, int)

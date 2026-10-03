# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Kernel body from Triton's python/tutorials/10-block-scaled-matmul.py.
# Copyright (c) 2023 - 2025 NVIDIA Corporation & Affiliates. All rights reserved.
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

# Static-only annotations cannot be passed to Triton's runtime.
# @lint-ignore-every AUTODEPS2
# flake8: noqa: B007

"""Semantic checks for CDNA4's packed matrices and preshuffled scales."""

from typing import assert_type, Literal

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def block_scaled_matmul_kernel_cdna4[
    MDim: IntVar,
    NDim: IntVar,
    PackedK: IntVar,
    AM: IntVar,
    AK: IntVar,
    BK: IntVar,
    BN: IntVar,
    CM: IntVar,
    CN: IntVar,
    ASM: IntVar,
    ASK: IntVar,
    BSN: IntVar,
    BSK: IntVar,
    BM: IntVar,
    BNN: IntVar,
    BKK: IntVar,
](
    a_ptr: tl.CDNA4PackedAPointer[MDim, PackedK, AM, AK, BM, BKK],
    b_ptr: tl.CDNA4PackedBPointer[PackedK, NDim, BK, BN, BNN, BKK],
    c_ptr: tl.OutMatrixPointer[MDim, NDim, CM, CN],
    a_scales_ptr: tl.CDNA4ScalePointer[MDim, PackedK, ASM, ASK, BM, BKK],
    b_scales_ptr: tl.CDNA4ScalePointer[NDim, PackedK, BSN, BSK, BNN, BKK],
    M: Int[MDim],
    N: Int[NDim],
    K: Int[PackedK],
    stride_am: Int[AM],
    stride_ak: Int[AK],
    stride_bk: Int[BK],
    stride_bn: Int[BN],
    stride_ck: int,
    stride_cm: Int[CM],
    stride_cn: Int[CN],
    stride_asm: Int[ASM],
    stride_ask: Int[ASK],
    stride_bsn: Int[BSN],
    stride_bsk: Int[BSK],
    BLOCK_M: Int[BM],
    BLOCK_N: Int[BNN],
    BLOCK_K: Int[BKK],
    mfma_nonkdim: Literal[16, 32],
):
    """Kernel for computing the matmul C = A x B.
    A and B inputs are in the microscale fp4 (mxfp4) format.
    A_scales and B_scales are in e8m0 format.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    pid = tl.program_id(axis=0)

    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # We assume 32 elements along K share the same scale.
    SCALE_GROUP_SIZE: tl.constexpr = 32
    num_k_iter = tl.cdiv(K, BLOCK_K // 2)
    # Create pointers for first block of A and B input matrices
    # The BLOCK sizes are of the elements and in fp4 we pack 2 per uint8 container.
    offs_k = tl.arange(0, BLOCK_K // 2)
    offs_k_split = offs_k
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Create pointers for the first block of A and B scales
    offs_asn = (pid_n * (BLOCK_N // 32) + tl.arange(0, (BLOCK_N // 32))) % N
    offs_ks = tl.arange(0, BLOCK_K // SCALE_GROUP_SIZE * 32)

    # B scales are N x K even though B operand is K x N.
    b_scale_ptrs = (
        b_scales_ptr + offs_asn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk
    )
    offs_asm = (pid_m * (BLOCK_M // 32) + tl.arange(0, (BLOCK_M // 32))) % M
    a_scale_ptrs = (
        a_scales_ptr + offs_asm[:, None] * stride_asm + offs_ks[None, :] * stride_ask
    )
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, num_k_iter):
        # Here we "undo" the shuffle done in global memory (shuffle_scales_cdna4 function).
        if mfma_nonkdim == 32:
            a_scales = (
                tl.load(a_scale_ptrs)
                .reshape(BLOCK_M // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 2, 32, 4, 1)
                .permute(0, 3, 1, 4, 2, 5)
                .reshape(BLOCK_M, BLOCK_K // SCALE_GROUP_SIZE)
            )
            b_scales = (
                tl.load(b_scale_ptrs)
                .reshape(BLOCK_N // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 2, 32, 4, 1)
                .permute(0, 3, 1, 4, 2, 5)
                .reshape(BLOCK_N, BLOCK_K // SCALE_GROUP_SIZE)
            )
        elif mfma_nonkdim == 16:
            a_scales = (
                tl.load(a_scale_ptrs)
                .reshape(
                    BLOCK_M // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2, 1
                )
                .permute(0, 5, 3, 1, 4, 2, 6)
                .reshape(BLOCK_M, BLOCK_K // SCALE_GROUP_SIZE)
            )
            b_scales = (
                tl.load(b_scale_ptrs)
                .reshape(
                    BLOCK_N // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2, 1
                )
                .permute(0, 5, 3, 1, 4, 2, 6)
                .reshape(BLOCK_N, BLOCK_K // SCALE_GROUP_SIZE)
            )

        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs, cache_modifier=None)

        accumulator += tl.dot_scaled(a, a_scales, "e2m1", b, b_scales, "e2m1")

        # Advance the ptrs to the next K block.
        a_ptrs += (BLOCK_K // 2) * stride_ak
        b_ptrs += (BLOCK_K // 2) * stride_bk

        a_scale_ptrs += BLOCK_K * stride_ask
        b_scale_ptrs += BLOCK_K * stride_bsk

    c = accumulator.to(c_ptr.type.element_ty)

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask, cache_modifier=".wt")


def test_host_boundary[
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
    ASM: IntVar,
    ASK: IntVar,
    BSN: IntVar,
    BSK: IntVar,
    BM: IntVar,
    BNN: IntVar,
    BKK: IntVar,
](
    a: tl.CDNA4PackedAPointer[M, K, AM, AK, BM, BKK],
    b: tl.CDNA4PackedBPointer[K, N, BK, BN, BNN, BKK],
    c: tl.OutMatrixPointer[M, N, CM, CN],
    scale_a: tl.CDNA4ScalePointer[M, K, ASM, ASK, BM, BKK],
    scale_b: tl.CDNA4ScalePointer[N, K, BSN, BSK, BNN, BKK],
    wrong_a: tl.CDNA4PackedAPointer[Other, K, AM, AK, BM, BKK],
    wrong_b: tl.CDNA4PackedBPointer[Other, N, BK, BN, BNN, BKK],
    wrong_c: tl.OutMatrixPointer[M, Other, CM, CN],
    wrong_scale_rows: tl.CDNA4ScalePointer[Other, K, ASM, ASK, BM, BKK],
    wrong_scale_k: tl.CDNA4ScalePointer[M, Other, ASM, ASK, BM, BKK],
    wrong_scale_stride: tl.CDNA4ScalePointer[N, K, BSN, Other, BNN, BKK],
    wrong_tile: tl.CDNA4ScalePointer[M, K, ASM, ASK, Other, BKK],
    m: Int[M],
    n: Int[N],
    k: Int[K],
    stride_am: Int[AM],
    stride_ak: Int[AK],
    stride_bk: Int[BK],
    stride_bn: Int[BN],
    stride_cm: Int[CM],
    stride_cn: Int[CN],
    stride_asm: Int[ASM],
    stride_ask: Int[ASK],
    stride_bsn: Int[BSN],
    stride_bsk: Int[BSK],
    bm: Int[BM],
    bnn: Int[BNN],
    bkk: Int[BKK],
) -> None:
    block_scaled_matmul_kernel_cdna4(
        a,
        b,
        c,
        scale_a,
        scale_b,
        m,
        n,
        k,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        0,
        stride_cm,
        stride_cn,
        stride_asm,
        stride_ask,
        stride_bsn,
        stride_bsk,
        bm,
        bnn,
        bkk,
        16,
    )
    block_scaled_matmul_kernel_cdna4(
        wrong_a,
        wrong_b,  # E: is not assignable
        wrong_c,  # E: is not assignable
        wrong_scale_rows,
        scale_b,
        m,  # E: is not assignable
        n,
        k,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        0,
        stride_cm,
        stride_cn,
        stride_asm,
        stride_ask,
        stride_bsn,
        stride_bsk,
        bm,
        bnn,
        bkk,
        32,
    )
    block_scaled_matmul_kernel_cdna4(
        a,
        b,
        c,
        wrong_scale_rows,  # E: is not assignable
        scale_b,
        m,
        n,
        k,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        0,
        stride_cm,
        stride_cn,
        stride_asm,
        stride_ask,
        stride_bsn,
        stride_bsk,
        bm,
        bnn,
        bkk,
        16,
    )
    block_scaled_matmul_kernel_cdna4(
        a,
        b,
        c,
        wrong_scale_k,  # E: is not assignable
        wrong_scale_stride,
        m,
        n,
        k,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        0,
        stride_cm,
        stride_cn,
        stride_asm,
        stride_ask,
        stride_bsn,
        stride_bsk,  # E: is not assignable
        bm,
        bnn,
        bkk,
        16,
    )
    block_scaled_matmul_kernel_cdna4(
        a,
        b,
        c,
        wrong_tile,  # E: is not assignable
        scale_b,
        m,
        n,
        k,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        0,
        stride_cm,
        stride_cn,
        stride_asm,
        stride_ask,
        stride_bsn,
        stride_bsk,
        bm,
        bnn,
        bkk,
        16,
    )


def test_scale_access[
    M: IntVar,
    K: IntVar,
    Other: IntVar,
    BM: IntVar,
    BK: IntVar,
    SR: IntVar,
    SK: IntVar,
](
    scale: tl.CDNA4ScalePointer[M, K, SR, SK, BM, BK],
    rows: tl.WrappedRowAddress[M, [BM // 32], SR],
    wrong_rows: tl.WrappedRowAddress[Other, [BM // 32], SR],
    columns: tl.ColumnAddress[[BK // 32 * 32], SK],
    wrong_columns: tl.ColumnAddress[[BK // 32 * 32], Other],
    scale_tile: tl.CDNA4ScaleTilePointers[M, K, SK, BM, BK],
    bad_tile: tl.CDNA4ScaleTile[BM, BK],
    bm: Int[BM],
    wrong_bm: Int[Other],
    groups: Int[BM // 32],
    bk: Int[BK],
    step: Int[BK * SK],
    wrong_step: Int[BK * Other],
) -> None:
    assert_type(scale + rows + columns, tl.CDNA4ScaleTilePointers[M, K, SK, BM, BK])
    scale + wrong_rows  # E: is not assignable
    scale + rows + wrong_columns  # E: is not assignable
    scale_tile += step
    scale_tile += wrong_step  # E: is not assignable
    bad_tile.reshape(groups, 1, 2, 32, 4, 1).permute(0, 3, 1, 4, 2, 5).reshape(
        wrong_bm,  # E: is not assignable
        bk // 32,
    )
    assert_type(
        bad_tile.reshape(groups, 1, 4, 16, 2, 2, 1)
        .permute(0, 5, 3, 1, 4, 2, 6)
        .reshape(bm, bk // 32),
        tl.tensor[[BM, BK // 32]],
    )


def test_output_mask[
    M: IntVar,
    N: IntVar,
    Other: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    ptr: tl.OutMatrixTilePointers[M, N, BM, BN],
    value: tl.tensor[[BM, BN]],
    bad_value: tl.tensor[[Other, BN]],
    mask: tl.MatrixMask[M, N, [BM], [BN]],
    bad_mask: tl.MatrixMask[Other, N, [BM], [BN]],
) -> None:
    tl.store(ptr, value, mask=mask, cache_modifier=".wt")
    tl.store(ptr, bad_value, mask=mask, cache_modifier=".wt")  # E: is not assignable
    tl.store(ptr, value, mask=bad_mask, cache_modifier=".wt")  # E: is not assignable


def test_packed_operand_addresses[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    Other: IntVar,
    AM: IntVar,
    AK: IntVar,
    BK: IntVar,
    BN: IntVar,
    BM: IntVar,
    BNN: IntVar,
    BKK: IntVar,
](
    a: tl.CDNA4PackedAPointer[M, K, AM, AK, BM, BKK],
    b: tl.CDNA4PackedBPointer[K, N, BK, BN, BNN, BKK],
    a_address: tl.WrappedRowMatrixAddress[M, [BM], [BKK // 2], AM, AK],
    b_address: tl.WrappedColumnMatrixAddress[N, [BKK // 2], [BNN], BK, BN],
    wrong_k: tl.WrappedRowMatrixAddress[M, [BM], [Other], AM, AK],
    wrong_stride: tl.WrappedColumnMatrixAddress[N, [BKK // 2], [BNN], Other, BN],
    a_tile: tl.CDNA4PackedATilePointers[M, K, BM, BKK, AK],
    b_tile: tl.CDNA4PackedBTilePointers[N, K, BNN, BKK, BK],
    step_a: Int[(BKK // 2) * AK],
    step_b: Int[(BKK // 2) * BK],
    wrong_step: Int[(Other // 2) * AK],
) -> None:
    assert_type(a + a_address, tl.CDNA4PackedATilePointers[M, K, BM, BKK, AK])
    assert_type(b + b_address, tl.CDNA4PackedBTilePointers[N, K, BNN, BKK, BK])
    a + wrong_k  # E: is not assignable
    b + wrong_stride  # E: is not assignable
    assert_type(tl.load(a_tile), tl.tensor[[BM, BKK // 2]])
    assert_type(tl.load(b_tile, cache_modifier=None), tl.tensor[[BKK // 2, BNN]])
    a_tile += step_a
    b_tile += step_b
    a_tile += wrong_step  # E: is not assignable


def test_unmasked_partial_scale_tile[
    M: IntVar,
    K: IntVar,
    SR: IntVar,
    SK: IntVar,
    BM: IntVar,
    BK: IntVar,
](
    scale_tile: tl.CDNA4ScaleTilePointers[M, K, SK, BM, BK],
) -> None:
    # Known gap: no proof that scale rows use a physical [M // 32] bound.
    assert_type(tl.load(scale_tile), tl.CDNA4ScaleTile[BM, BK])

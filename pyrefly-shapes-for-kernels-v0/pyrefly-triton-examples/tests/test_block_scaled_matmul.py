# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This kernel body is copied from Triton's 10-block-scaled-matmul.py:
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

# Static-only fixture; these parameter annotations are not valid Triton runtime code.
# @lint-ignore-every AUTODEPS2
# Keep the original loop variable in the unchanged upstream body.
# flake8: noqa: B007

"""Rank-five scale descriptors at the block-scaled matmul kernel boundary."""

from typing import assert_type, Literal

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


def _matmul_launch_metadata(grid: object, kernel: object, args: object) -> object:
    return None


@triton.jit(launch_metadata=_matmul_launch_metadata)
def block_scaled_matmul_kernel[
    MDim: IntVar,
    NDim: IntVar,
    KDim: IntVar,
    AElementsPerByte: IntVar,
    BElementsPerByte: IntVar,
    VecSize: IntVar,
    RepM: IntVar,
    RepN: IntVar,
    RepK: IntVar,
](  #
    a_desc: tl.BlockDataDescriptor[
        MDim, KDim, AElementsPerByte, RepM, RepK, VecSize
    ],  #
    a_scale_desc: tl.BlockScaleDescriptor[MDim, KDim, VecSize, RepM, RepK],  #
    b_desc: tl.BlockDataDescriptor[
        NDim, KDim, BElementsPerByte, RepN, RepK, VecSize
    ],  #
    b_scale_desc: tl.BlockScaleDescriptor[NDim, KDim, VecSize, RepN, RepK],  #
    c_desc: tl.BlockOutputDescriptor[MDim, NDim, RepM * 128, RepN * 128],  #
    M: Int[MDim],  #
    N: Int[NDim],  #
    K: Int[KDim],  #
    output_type: Literal[0, 1, 2],  #
    ELEM_PER_BYTE_A: Int[AElementsPerByte],  #
    ELEM_PER_BYTE_B: Int[BElementsPerByte],  #
    VEC_SIZE: Int[VecSize],  #
    BLOCK_M: Int[RepM * 128],  #
    BLOCK_N: Int[RepN * 128],  #
    BLOCK_K: Int[RepK * 4 * VecSize],  #
    rep_m: Int[RepM],  #
    rep_n: Int[RepN],  #
    rep_k: Int[RepK],  #
    NUM_STAGES: tl.constexpr,  #
    disallow_acc_multi_buffer: bool,  #
):  #
    if output_type == 0:
        output_dtype = tl.float32
    elif output_type == 1:
        output_dtype = tl.float16
    elif output_type == 2:
        output_dtype = tl.float8e4nv

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    offs_k_a = 0
    offs_k_b = 0
    offs_scale_m = pid_m * rep_m
    offs_scale_n = pid_n * rep_n
    offs_scale_k = 0

    MIXED_PREC: tl.constexpr = ELEM_PER_BYTE_A == 1 and ELEM_PER_BYTE_B == 2

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(
        0,
        tl.cdiv(K, BLOCK_K),
        num_stages=NUM_STAGES,
        disallow_acc_multi_buffer=disallow_acc_multi_buffer,
    ):
        a = a_desc.load([offs_am, offs_k_a])
        b = b_desc.load([offs_bn, offs_k_b])
        scale_a = a_scale_desc.load([0, offs_scale_m, offs_scale_k, 0, 0])
        scale_b = b_scale_desc.load([0, offs_scale_n, offs_scale_k, 0, 0])

        scale_a = (
            scale_a.reshape(rep_m, rep_k, 32, 4, 4)
            .trans(0, 3, 2, 1, 4)
            .reshape(BLOCK_M, BLOCK_K // VEC_SIZE)
        )
        scale_b = (
            scale_b.reshape(rep_n, rep_k, 32, 4, 4)
            .trans(0, 3, 2, 1, 4)
            .reshape(BLOCK_N, BLOCK_K // VEC_SIZE)
        )

        if MIXED_PREC:
            accumulator = tl.dot_scaled(
                a, scale_a, "e4m3", b.T, scale_b, "e2m1", accumulator
            )
        elif ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 2:
            accumulator = tl.dot_scaled(
                a, scale_a, "e2m1", b.T, scale_b, "e2m1", accumulator
            )
        else:
            accumulator = tl.dot_scaled(
                a, scale_a, "e4m3", b.T, scale_b, "e4m3", accumulator
            )

        offs_k_a += BLOCK_K // ELEM_PER_BYTE_A
        offs_k_b += BLOCK_K // ELEM_PER_BYTE_B
        offs_scale_k += rep_k

    c_desc.store([offs_am, offs_bn], accumulator.to(output_dtype))


def test_bad_output_tile[BM: IntVar, BN: IntVar, Other: IntVar](
    output: tl.BlockOutputDescriptor[int, int, BM, Other],
    bm: Int[BM],
    bn: Int[BN],
) -> None:
    output.store([0, 0], tl.zeros((bm, bn), dtype=tl.float32))  # E: is not assignable


def test_scale_first_reshape[RM: IntVar, RK: IntVar](
    scale: tl.BlockScaleDescriptor[int, int, int, RM, RK],
    rm: Int[RM],
    rk: Int[RK],
) -> None:
    assert_type(scale.load([0, 0, 0, 0, 0]), tl.tensor[[1, RM, RK, 2, 256]])
    assert_type(
        scale.load([0, 0, 0, 0, 0]).reshape(rm, rk, 32, 4, 4).trans(0, 3, 2, 1, 4),
        tl.tensor[[RM, 4, 32, RK, 4]],
    )


def test_boundary[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    EA: IntVar,
    EB: IntVar,
    V: IntVar,
    RM: IntVar,
    RN: IntVar,
    RK: IntVar,
    Other: IntVar,
](
    a: tl.BlockDataDescriptor[M, K, EA, RM, RK, V],
    b: tl.BlockDataDescriptor[N, K, EB, RN, RK, V],
    scale_a: tl.BlockScaleDescriptor[M, K, V, RM, RK],
    scale_b: tl.BlockScaleDescriptor[N, K, V, RN, RK],
    c: tl.BlockOutputDescriptor[M, N, RM * 128, RN * 128],
    bad_a: tl.BlockDataDescriptor[Other, K, EA, RM, RK, V],
    bad_scale: tl.BlockScaleDescriptor[Other, K, V, RM, RK],
    bad_scale_tile: tl.BlockScaleDescriptor[M, K, V, Other, RK],
    bad_packing: tl.BlockDataDescriptor[M, K, Other, RM, RK, V],
    bad_output: tl.BlockOutputDescriptor[M, Other, RM * 128, RN * 128],
    m: Int[M],
    n: Int[N],
    k: Int[K],
    ea: Int[EA],
    eb: Int[EB],
    vec: Int[V],
    rm: Int[RM],
    rn: Int[RN],
    rk: Int[RK],
    bm: Int[RM * 128],
    bn: Int[RN * 128],
    bk: Int[RK * 4 * V],
) -> None:
    block_scaled_matmul_kernel(
        a,
        scale_a,
        b,
        scale_b,
        c,
        m,
        n,
        k,
        1,
        ea,
        eb,
        vec,
        bm,
        bn,
        bk,
        rm,
        rn,
        rk,
        4,
        True,
    )
    block_scaled_matmul_kernel(
        bad_a,
        scale_a,  # E: is not assignable
        b,
        scale_b,
        c,  # E: is not assignable
        m,  # E: is not assignable
        n,
        k,
        1,
        ea,
        eb,
        vec,
        bm,
        bn,
        bk,
        rm,
        rn,
        rk,
        4,
        True,
    )
    block_scaled_matmul_kernel(
        a,
        bad_scale,  # E: is not assignable
        b,
        scale_b,
        c,
        m,
        n,
        k,
        1,
        ea,
        eb,
        vec,
        bm,
        bn,
        bk,
        rm,
        rn,
        rk,
        4,
        True,
    )
    block_scaled_matmul_kernel(
        a,
        bad_scale_tile,  # E: is not assignable
        b,
        scale_b,
        c,
        m,
        n,
        k,
        1,
        ea,
        eb,
        vec,
        bm,
        bn,
        bk,
        rm,
        rn,
        rk,
        4,
        True,
    )
    block_scaled_matmul_kernel(
        bad_packing,
        scale_a,
        b,
        scale_b,
        c,
        m,
        n,
        k,
        1,
        ea,  # E: is not assignable
        eb,
        vec,
        bm,
        bn,
        bk,
        rm,
        rn,
        rk,
        4,
        True,
    )
    block_scaled_matmul_kernel(
        a,
        scale_a,
        b,
        scale_b,
        bad_output,  # E: is not assignable
        m,
        n,
        k,
        1,
        ea,
        eb,
        vec,
        bm,
        bn,
        bk,
        rm,
        rn,
        rk,
        4,
        True,
    )


def test_dot_scaled_tile[BM: IntVar, BN: IntVar, K: IntVar, SK: IntVar, Other: IntVar](
    a: tl.tensor[[BM, K]],
    scale_a: tl.tensor[[BM, SK]],
    b: tl.tensor[[K, BN]],
    scale_b: tl.tensor[[BN, SK]],
    wrong_scale: tl.tensor[[Other, SK]],
    acc: tl.tensor[[BM, BN]],
) -> None:
    assert_type(
        tl.dot_scaled(a, scale_a, "e2m1", b, scale_b, "e2m1", acc),
        tl.tensor[[BM, BN]],
    )
    tl.dot_scaled(a, wrong_scale, "e2m1", b, scale_b, "e2m1", acc)  # E: not assignable


def test_unchecked_descriptor_coordinates[
    Rows: IntVar,
    KGroups: IntVar,
    RM: IntVar,
    RK: IntVar,
](scale: tl.BlockScaleDescriptor[Rows, KGroups, int, RM, RK]) -> None:
    # Known gap: offset list rank and bound are not verified by descriptor.load.
    assert_type(scale.load([0]), tl.tensor[[1, RM, RK, 2, 256]])


def test_unchecked_packed_contraction[
    BM: IntVar,
    BN: IntVar,
    KLeft: IntVar,
    KRight: IntVar,
    SK: IntVar,
](
    a: tl.tensor[[BM, KLeft]],
    b: tl.tensor[[KRight, BN]],
    scale_a: tl.tensor[[BM, SK]],
    scale_b: tl.tensor[[BN, SK]],
    acc: tl.tensor[[BM, BN]],
) -> None:
    # Known gap: the physical K extents can differ despite matching scale tiles.
    assert_type(
        tl.dot_scaled(a, scale_a, "e2m1", b, scale_b, "e2m1", acc),
        tl.tensor[[BM, BN]],
    )


def test_unchecked_fp_format[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    RM: IntVar,
    RN: IntVar,
    RK: IntVar,
    V: IntVar,
    SK: IntVar,
](
    a: tl.BlockDataDescriptor[M, K, 2, RM, RK, V],
    b: tl.BlockDataDescriptor[N, K, 2, RN, RK, V],
    scale_a: tl.tensor[[RM * 128, SK]],
    scale_b: tl.tensor[[RN * 128, SK]],
    acc: tl.tensor[[RM * 128, RN * 128]],
) -> None:
    # Known gap: packed FP4 data also accepts the FP8 format string.
    assert_type(
        tl.dot_scaled(
            a.load([0, 0]), scale_a, "e4m3", b.load([0, 0]).T, scale_b, "e4m3", acc
        ),
        tl.tensor[[RM * 128, RN * 128]],
    )

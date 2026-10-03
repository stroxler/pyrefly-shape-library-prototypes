# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This kernel body is copied from Triton's 08-grouped-gemm.py:
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

# This static-only kernel is checked by Pyrefly, not executed as a Buck test.
# @lint-ignore-every AUTODEPS2
# Preserve the upstream unused `kk` loop variable in the unchanged kernel body.
# flake8: noqa: B007

"""The grouped_matmul_tma_kernel in Triton's 08-grouped-gemm.py."""

from typing import assert_type

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def grouped_matmul_tma_kernel[Groups: IntVar, BM: IntVar, BN: IntVar, BK: IntVar](
    # device tensor of matrices pointers
    group_a_ptrs: tl.GroupAPointers[Groups],
    group_b_ptrs: tl.GroupBPointers[Groups],
    group_c_ptrs: tl.GroupCPointers[Groups],
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    group_gemm_sizes: tl.GroupSizes[Groups],
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds: tl.GroupLeadingDimensions[Groups],
    # number of gemms
    group_size: Int[Groups],
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: Int[BM],
    BLOCK_SIZE_N: Int[BN],
    BLOCK_SIZE_K: Int[BK],
    # is the output FP8 or FP16
    FP8: tl.constexpr,
):
    dtype = tl.float8e4nv if FP8 else tl.float16
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the gemm size of the current problem
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        if tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            # pick up a tile from the current gemm problem
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)

            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(dtype))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(dtype))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(dtype))

            a_desc = tl.make_tensor_descriptor(
                a_ptr,
                shape=[gm, gk],
                strides=[lda, 1],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
            )

            b_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[gn, gk],
                strides=[ldb, 1],
                block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
            )
            c_desc = tl.make_tensor_descriptor(
                c_ptr,
                shape=[gm, gn],
                strides=[ldc, 1],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
            )

            # iterate through the tiles in the current gemm problem
            while (
                tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles
            ):
                k = gk
                # figure out tile coordinates
                tile_idx_in_gemm = tile_idx - last_problem_end
                tile_m_idx = tile_idx_in_gemm // num_n_tiles
                tile_n_idx = tile_idx_in_gemm % num_n_tiles

                # do regular gemm here
                offs_am = tile_m_idx * BLOCK_SIZE_M
                offs_bn = tile_n_idx * BLOCK_SIZE_N

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                    a = a_desc.load([offs_am, kk * BLOCK_SIZE_K])
                    b = b_desc.load([offs_bn, kk * BLOCK_SIZE_K])
                    accumulator += tl.dot(a, b.T)

                offs_cm = tile_m_idx * BLOCK_SIZE_M
                offs_cn = tile_n_idx * BLOCK_SIZE_N

                c = accumulator.to(dtype)
                c_desc.store([offs_cm, offs_cn], c)

                # go to the next tile by advancing NUM_SM
                tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles


def test_transposed_b_descriptor[BM: IntVar, BN: IntVar, BK: IntVar](
    a: tl.tensor_descriptor[int, int, int, BM, BK],
    b: tl.tensor_descriptor[int, int, int, BN, BK],
    c: tl.tensor_descriptor[int, int, int, BM, BN],
) -> None:
    result = tl.dot(a.load([0, 0]), b.load([0, 0]).T)
    assert_type(result, tl.tensor[[BM, BN]])
    c.store([0, 0], result)


def test_wrong_b_descriptor_orientation[BM: IntVar, BN: IntVar, BK: IntVar](
    a: tl.tensor_descriptor[int, int, int, BM, BK],
    b: tl.tensor_descriptor[int, int, int, BN, BK],
) -> None:
    tl.dot(a.load([0, 0]), b.load([0, 0]))  # E: is not assignable to parameter


def test_wrong_c_descriptor_block[BM: IntVar, BN: IntVar, BK: IntVar, Other: IntVar](
    a: tl.tensor_descriptor[int, int, int, BM, BK],
    b: tl.tensor_descriptor[int, int, int, BN, BK],
    c: tl.tensor_descriptor[int, int, int, BM, Other],
) -> None:
    c.store([0, 0], tl.dot(a.load([0, 0]), b.load([0, 0]).T))  # E: is not assignable


def test_unverified_descriptor_metadata[Groups: IntVar, BM: IntVar, BK: IntVar](
    a: tl.GroupAMatrixPointer[Groups],
    wrong_rows: int,
    wrong_cols: int,
    wrong_stride: int,
    bm: Int[BM],
    bk: Int[BK],
) -> None:
    # Known gap: the descriptor's shape and stride need not describe its pointer.
    descriptor = tl.make_tensor_descriptor(
        a,
        shape=[wrong_rows, wrong_cols],
        strides=[wrong_stride, 1],
        block_shape=[bm, bk],
    )
    assert_type(descriptor.load([0, 0]), tl.tensor[[BM, BK]])


def test_unverified_descriptor_role[Groups: IntVar, BM: IntVar, BN: IntVar](
    a: tl.GroupAMatrixPointer[Groups],
    value: tl.tensor[[BM, BN]],
    bm: Int[BM],
    bn: Int[BN],
) -> None:
    # Known gap: the descriptor erases the input pointer's A role at the store.
    descriptor = tl.make_tensor_descriptor(
        a,
        shape=[1, 1],
        strides=[1, 1],
        block_shape=[bm, bn],
    )
    assert_type(descriptor.store([0, 0], value), None)


def test_wrong_group_count[
    Groups: IntVar,
    Other: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
](
    a: tl.GroupAPointers[Groups],
    b: tl.GroupBPointers[Groups],
    c: tl.GroupCPointers[Groups],
    sizes: tl.GroupSizes[Other],
    strides: tl.GroupLeadingDimensions[Groups],
    count: Int[Groups],
    sm: int,
    bm: Int[BM],
    bn: Int[BN],
    bk: Int[BK],
) -> None:
    grouped_matmul_tma_kernel(
        a,
        b,
        c,
        sizes,  # E: is not assignable to parameter
        strides,
        count,
        sm,
        bm,
        bn,
        bk,
        False,
    )

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

"""The first grouped_matmul_kernel in Triton's 08-grouped-gemm.py."""

from typing import assert_type

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def grouped_matmul_kernel[Groups: IntVar, BM: IntVar, BN: IntVar, BK: IntVar](
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
):
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
        # iterate through the tiles in the current gemm problem
        while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            # pick up a tile from the current gemm problem
            k = gk
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                # hint to Triton compiler to do proper loop pipelining
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                # assume full tile for now
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb
            c = accumulator.to(tl.float16)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]

            # assumes full tile for now
            tl.store(c_ptrs, c)

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles


def test_grouped_tile_shapes[BM: IntVar, BN: IntVar, BK: IntVar](
    a: tl.GroupATilePointers[BM, BK],
    b: tl.GroupBTilePointers[BK, BN],
    c: tl.GroupCTilePointers[BM, BN],
) -> None:
    result = tl.dot(tl.load(a), tl.load(b))
    assert_type(result, tl.tensor[[BM, BN]])
    tl.store(c, result)


def test_indirect_address_roles[Groups: IntVar](
    a: tl.GroupAPointers[Groups],
    b: tl.GroupBPointers[Groups],
    c: tl.GroupCPointers[Groups],
) -> None:
    assert_type(
        tl.load(a + 0).to(tl.pointer_type(tl.float16)),
        tl.GroupAMatrixPointer[Groups],
    )
    assert_type(
        tl.load(b + 0).to(tl.pointer_type(tl.float16)),
        tl.GroupBMatrixPointer[Groups],
    )
    assert_type(
        tl.load(c + 0).to(tl.pointer_type(tl.float16)),
        tl.GroupCMatrixPointer[Groups],
    )


def test_unverified_input_roles[BM: IntVar, BN: IntVar, BK: IntVar](
    a: tl.GroupBTilePointers[BM, BK], b: tl.GroupATilePointers[BK, BN]
) -> None:
    # Known gap: dot checks the tile dimensions, not the input allocation roles.
    assert_type(tl.dot(tl.load(a), tl.load(b)), tl.tensor[[BM, BN]])


def test_unverified_packed_metadata_index[Groups: IntVar](
    sizes: tl.GroupSizes[Groups], group: int
) -> None:
    # Known gap: an incorrect group stride still reads a typable scalar.
    assert_type(tl.load(sizes + group * 2 + 1), int)


def test_wrong_b_contraction[BM: IntVar, BN: IntVar, BK: IntVar, Other: IntVar](
    a: tl.GroupATilePointers[BM, BK], b: tl.GroupBTilePointers[Other, BN]
) -> None:
    tl.dot(tl.load(a), tl.load(b))  # E: is not assignable to parameter


def test_wrong_c_tile[BM: IntVar, BN: IntVar, BK: IntVar, Other: IntVar](
    a: tl.GroupATilePointers[BM, BK],
    b: tl.GroupBTilePointers[BK, BN],
    c: tl.GroupCTilePointers[BM, Other],
) -> None:
    tl.store(c, tl.dot(tl.load(a), tl.load(b)))  # E: No matching overload


def test_wrong_group_count[
    Groups: IntVar,
    Other: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
](
    a: tl.GroupAPointers[Groups],
    b: tl.GroupBPointers[Other],
    c: tl.GroupCPointers[Groups],
    sizes: tl.GroupSizes[Groups],
    strides: tl.GroupLeadingDimensions[Groups],
    count: Int[Groups],
    sm: int,
    bm: Int[BM],
    bn: Int[BN],
    bk: Int[BK],
) -> None:
    grouped_matmul_kernel(
        a,
        b,  # E: is not assignable to parameter
        c,
        sizes,
        strides,
        count,
        sm,
        bm,
        bn,
        bk,
    )


def test_wrong_pointer_role[Groups: IntVar, BM: IntVar, BN: IntVar, BK: IntVar](
    a: tl.GroupAPointers[Groups],
    b: tl.GroupBPointers[Groups],
    c: tl.GroupCPointers[Groups],
    sizes: tl.GroupSizes[Groups],
    strides: tl.GroupLeadingDimensions[Groups],
    count: Int[Groups],
    sm: int,
    bm: Int[BM],
    bn: Int[BN],
    bk: Int[BK],
) -> None:
    grouped_matmul_kernel(
        b,  # E: is not assignable to parameter
        a,  # E: is not assignable to parameter
        c,
        sizes,
        strides,
        count,
        sm,
        bm,
        bn,
        bk,
    )

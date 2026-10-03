# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The kernel body is copied from Triton's 09-persistent-matmul.py.
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

"""The nonpersistent TMA descriptor matmul in Triton's tutorial 09."""

from typing import assert_type

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def matmul_kernel_tma[
    MDim: IntVar,
    NDim: IntVar,
    KDim: IntVar,
    AS: IntVar,
    BS: IntVar,
    CS: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Group: IntVar,
](
    a_desc: tl.InputMatrixDescriptor[MDim, KDim, AS, BM, BK],
    b_desc: tl.InputMatrixDescriptor[NDim, KDim, BS, BN, BK],
    c_desc: tl.OutputMatrixDescriptor[MDim, NDim, CS, BM, BN],  #
    M: Int[MDim],
    N: Int[NDim],
    K: Int[KDim],  #
    BLOCK_SIZE_M: Int[BM],  #
    BLOCK_SIZE_N: Int[BN],  #
    BLOCK_SIZE_K: Int[BK],  #
    GROUP_SIZE_M: Int[Group],  #
    FP8_OUTPUT: bool,  #
    WARP_SPECIALIZE: bool,  #
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in tl.range(k_tiles, warp_specialize=WARP_SPECIALIZE):
        offs_k = k * BLOCK_SIZE_K
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

    c = accumulator.to(dtype)

    offs_cm = pid_m * BLOCK_SIZE_M
    offs_cn = pid_n * BLOCK_SIZE_N
    c_desc.store([offs_cm, offs_cn], c)


def test_descriptor_blocks[
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    AS: IntVar,
    BS: IntVar,
    CS: IntVar,
    M: IntVar,
    N: IntVar,
    K: IntVar,
](
    a: tl.InputMatrixDescriptor[M, K, AS, BM, BK],
    b: tl.InputMatrixDescriptor[N, K, BS, BN, BK],
    c: tl.OutputMatrixDescriptor[M, N, CS, BM, BN],
) -> None:
    value = tl.dot(a.load([0, 0]), b.load([0, 0]).T)
    assert_type(value, tl.tensor[[BM, BN]])
    c.store([0, 0], value)


def test_wrong_b_orientation[
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    AS: IntVar,
    BS: IntVar,
    M: IntVar,
    N: IntVar,
    K: IntVar,
](
    a: tl.InputMatrixDescriptor[M, K, AS, BM, BK],
    b: tl.InputMatrixDescriptor[N, K, BS, BN, BK],
) -> None:
    tl.dot(a.load([0, 0]), b.load([0, 0]))  # E: is not assignable to parameter


def test_wrong_c_block[
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Other: IntVar,
    AS: IntVar,
    BS: IntVar,
    CS: IntVar,
    M: IntVar,
    N: IntVar,
    K: IntVar,
](
    a: tl.InputMatrixDescriptor[M, K, AS, BM, BK],
    b: tl.InputMatrixDescriptor[N, K, BS, BN, BK],
    c: tl.OutputMatrixDescriptor[M, N, CS, BM, Other],
) -> None:
    c.store([0, 0], tl.dot(a.load([0, 0]), b.load([0, 0]).T))  # E: is not assignable


def test_wrong_b_block_k[
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Other: IntVar,
    AS: IntVar,
    BS: IntVar,
    M: IntVar,
    N: IntVar,
    K: IntVar,
](
    a: tl.InputMatrixDescriptor[M, K, AS, BM, BK],
    b: tl.InputMatrixDescriptor[N, K, BS, BN, Other],
) -> None:
    tl.dot(a.load([0, 0]), b.load([0, 0]).T)  # E: is not assignable to parameter


def test_wrong_boundary_shape[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    Other: IntVar,
    AS: IntVar,
    BS: IntVar,
    CS: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Group: IntVar,
](
    a: tl.InputMatrixDescriptor[M, K, AS, BM, BK],
    b: tl.InputMatrixDescriptor[N, Other, BS, BN, BK],
    c: tl.OutputMatrixDescriptor[M, N, CS, BM, BN],
    m: Int[M],
    n: Int[N],
    k: Int[K],
    bm: Int[BM],
    bn: Int[BN],
    bk: Int[BK],
    group: Int[Group],
) -> None:
    matmul_kernel_tma(
        a,
        b,  # E: is not assignable to parameter
        c,
        m,
        n,
        k,
        bm,
        bn,
        bk,
        group,
        False,
        False,
    )


def test_wrong_boundary_role[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    AS: IntVar,
    BS: IntVar,
    CS: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Group: IntVar,
](
    a: tl.InputMatrixDescriptor[M, K, AS, BM, BK],
    b: tl.InputMatrixDescriptor[N, K, BS, BN, BK],
    c: tl.InputMatrixDescriptor[M, N, CS, BM, BN],
    m: Int[M],
    n: Int[N],
    k: Int[K],
    bm: Int[BM],
    bn: Int[BN],
    bk: Int[BK],
    group: Int[Group],
) -> None:
    matmul_kernel_tma(
        a,
        b,
        c,  # E: is not assignable to parameter
        m,
        n,
        k,
        bm,
        bn,
        bk,
        group,
        False,
        False,
    )


def test_unverified_descriptor_offsets[
    M: IntVar,
    K: IntVar,
    S: IntVar,
    BM: IntVar,
    BK: IntVar,
](a: tl.InputMatrixDescriptor[M, K, S, BM, BK], wrong_row: int, wrong_k: int) -> None:
    # Offsets are arbitrary integers: neither bounds nor block alignment are proved.
    assert_type(a.load([wrong_row, wrong_k]), tl.tensor[[BM, BK]])

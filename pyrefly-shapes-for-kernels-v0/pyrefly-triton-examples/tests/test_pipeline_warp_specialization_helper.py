# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The kernel body is copied from Triton's python/tutorials/compilation-pipeline/06_warp_specialization.py.
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

"""Original persistent warp-specialized matmul helper with descriptor tiles."""

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def _matmul_persistent_ws[
    MDim: IntVar,
    NDim: IntVar,
    KDim: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Group: IntVar,
    SMs: IntVar,
](
    a_desc: tl.InputMatrixDescriptor[MDim, KDim, KDim, BM, BK],
    b_desc: tl.InputMatrixDescriptor[NDim, KDim, KDim, BN, BK],
    c_desc: tl.OutputMatrixDescriptor[MDim, NDim, NDim, BM, BN],
    M: Int[MDim],
    N: Int[NDim],
    K: Int[KDim],
    BLOCK_M: Int[BM],
    BLOCK_N: Int[BN],
    BLOCK_K: Int[BK],
    GROUP_M: Int[Group],
    NUM_SMS: Int[SMs],
    FLATTEN: bool,
):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_M * num_pid_n
    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=FLATTEN,
        warp_specialize=True,
        disallow_acc_multi_buffer=True,
        separate_epilogue_store=True,
    ):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (tile_id % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        offs_am = pid_m * BLOCK_M
        offs_bn = pid_n * BLOCK_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            acc = tl.dot(a, b.T, acc)
        c_desc.store([offs_am, offs_bn], acc.to(tl.float16))


def test_descriptor_host_widths[
    MDim: IntVar,
    NDim: IntVar,
    KDim: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Group: IntVar,
    SMs: IntVar,
    Other: IntVar,
](
    a: tl.InputMatrixDescriptor[MDim, KDim, KDim, BM, BK],
    b: tl.InputMatrixDescriptor[NDim, KDim, KDim, BN, BK],
    c: tl.OutputMatrixDescriptor[MDim, NDim, NDim, BM, BN],
    wrong_a: tl.InputMatrixDescriptor[MDim, Other, Other, BM, BK],
    wrong_b: tl.InputMatrixDescriptor[NDim, Other, Other, BN, BK],
    wrong_c: tl.OutputMatrixDescriptor[MDim, Other, Other, BM, BN],
    m: Int[MDim],
    n: Int[NDim],
    k: Int[KDim],
    block_m: Int[BM],
    block_n: Int[BN],
    block_k: Int[BK],
    group: Int[Group],
    sms: Int[SMs],
) -> None:
    _matmul_persistent_ws(
        a, b, c, m, n, k, block_m, block_n, block_k, group, sms, False
    )
    _matmul_persistent_ws(
        wrong_a,
        b,  # E: is not assignable
        c,
        m,
        n,
        k,  # E: is not assignable
        block_m,
        block_n,
        block_k,
        group,
        sms,
        False,
    )
    _matmul_persistent_ws(
        a,
        wrong_b,  # E: is not assignable
        c,
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        group,
        sms,
        False,
    )
    _matmul_persistent_ws(
        a,
        b,
        wrong_c,  # E: is not assignable
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        group,
        sms,
        False,
    )


# Changing only the A row block must reject its original contraction.
@triton.jit
def _matmul_persistent_ws_wrong_a[
    MDim: IntVar,
    NDim: IntVar,
    KDim: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Group: IntVar,
    SMs: IntVar,
    Other: IntVar,
](
    a_desc: tl.InputMatrixDescriptor[MDim, KDim, KDim, Other, BK],
    b_desc: tl.InputMatrixDescriptor[NDim, KDim, KDim, BN, BK],
    c_desc: tl.OutputMatrixDescriptor[MDim, NDim, NDim, BM, BN],
    M: Int[MDim],
    N: Int[NDim],
    K: Int[KDim],
    BLOCK_M: Int[BM],
    BLOCK_N: Int[BN],
    BLOCK_K: Int[BK],
    GROUP_M: Int[Group],
    NUM_SMS: Int[SMs],
    FLATTEN: bool,
):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_M * num_pid_n
    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=FLATTEN,
        warp_specialize=True,
        disallow_acc_multi_buffer=True,
        separate_epilogue_store=True,
    ):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (tile_id % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        offs_am = pid_m * BLOCK_M
        offs_bn = pid_n * BLOCK_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            acc = tl.dot(a, b.T, acc)  # E: is not assignable
        c_desc.store([offs_am, offs_bn], acc.to(tl.float16))  # E: is not assignable


# Changing only the B row block must reject its original contraction.
@triton.jit
def _matmul_persistent_ws_wrong_b[
    MDim: IntVar,
    NDim: IntVar,
    KDim: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Group: IntVar,
    SMs: IntVar,
    Other: IntVar,
](
    a_desc: tl.InputMatrixDescriptor[MDim, KDim, KDim, BM, BK],
    b_desc: tl.InputMatrixDescriptor[NDim, KDim, KDim, Other, BK],
    c_desc: tl.OutputMatrixDescriptor[MDim, NDim, NDim, BM, BN],
    M: Int[MDim],
    N: Int[NDim],
    K: Int[KDim],
    BLOCK_M: Int[BM],
    BLOCK_N: Int[BN],
    BLOCK_K: Int[BK],
    GROUP_M: Int[Group],
    NUM_SMS: Int[SMs],
    FLATTEN: bool,
):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_M * num_pid_n
    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=FLATTEN,
        warp_specialize=True,
        disallow_acc_multi_buffer=True,
        separate_epilogue_store=True,
    ):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (tile_id % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        offs_am = pid_m * BLOCK_M
        offs_bn = pid_n * BLOCK_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            acc = tl.dot(a, b.T, acc)  # E: is not assignable
        c_desc.store([offs_am, offs_bn], acc.to(tl.float16))  # E: is not assignable


# Changing only the C column block must reject its original store.
@triton.jit
def _matmul_persistent_ws_wrong_c[
    MDim: IntVar,
    NDim: IntVar,
    KDim: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Group: IntVar,
    SMs: IntVar,
    Other: IntVar,
](
    a_desc: tl.InputMatrixDescriptor[MDim, KDim, KDim, BM, BK],
    b_desc: tl.InputMatrixDescriptor[NDim, KDim, KDim, BN, BK],
    c_desc: tl.OutputMatrixDescriptor[MDim, NDim, NDim, BM, Other],
    M: Int[MDim],
    N: Int[NDim],
    K: Int[KDim],
    BLOCK_M: Int[BM],
    BLOCK_N: Int[BN],
    BLOCK_K: Int[BK],
    GROUP_M: Int[Group],
    NUM_SMS: Int[SMs],
    FLATTEN: bool,
):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_M * num_pid_n
    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=FLATTEN,
        warp_specialize=True,
        disallow_acc_multi_buffer=True,
        separate_epilogue_store=True,
    ):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (tile_id % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        offs_am = pid_m * BLOCK_M
        offs_bn = pid_n * BLOCK_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            acc = tl.dot(a, b.T, acc)
        c_desc.store([offs_am, offs_bn], acc.to(tl.float16))  # E: is not assignable

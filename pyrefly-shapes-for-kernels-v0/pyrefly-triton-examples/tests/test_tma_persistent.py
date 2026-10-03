# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# These kernel bodies are copied from Triton's 09-persistent-matmul.py.
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

"""The persistent TMA descriptor matmul in Triton's tutorial 09."""

from typing import assert_type, Literal, overload

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


@overload
def matmul_kernel_tma_persistent[
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
    SMs: IntVar,
](
    a_desc: tl.InputMatrixDescriptor[MDim, KDim, AS, BM, BK],
    b_desc: tl.InputMatrixDescriptor[NDim, KDim, BS, BN, BK],
    c_desc: tl.OutputMatrixDescriptor[MDim, NDim, CS, BM, BN],
    M: Int[MDim],
    N: Int[NDim],
    K: Int[KDim],
    BLOCK_SIZE_M: Int[BM],
    BLOCK_SIZE_N: Int[BN],
    BLOCK_SIZE_K: Int[BK],
    GROUP_SIZE_M: Int[Group],
    FP8_OUTPUT: bool,
    EPILOGUE_SUBTILE: Literal[False],
    NUM_SMS: Int[SMs],
    WARP_SPECIALIZE: bool,
) -> None: ...


@overload
def matmul_kernel_tma_persistent[
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
    SMs: IntVar,
](
    a_desc: tl.InputMatrixDescriptor[MDim, KDim, AS, BM, BK],
    b_desc: tl.InputMatrixDescriptor[NDim, KDim, BS, BN, BK],
    c_desc: tl.OutputMatrixDescriptor[MDim, NDim, CS, BM, BN // 2],
    M: Int[MDim],
    N: Int[NDim],
    K: Int[KDim],
    BLOCK_SIZE_M: Int[BM],
    BLOCK_SIZE_N: Int[BN],
    BLOCK_SIZE_K: Int[BK],
    GROUP_SIZE_M: Int[Group],
    FP8_OUTPUT: bool,
    EPILOGUE_SUBTILE: Literal[True],
    NUM_SMS: Int[SMs],
    WARP_SPECIALIZE: bool,
) -> None: ...


@triton.jit
def matmul_kernel_tma_persistent[
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
    SMs: IntVar,
](
    a_desc: tl.InputMatrixDescriptor[MDim, KDim, AS, BM, BK],
    b_desc: tl.InputMatrixDescriptor[NDim, KDim, BS, BN, BK],
    c_desc: tl.OutputMatrixDescriptor[MDim, NDim, CS, BM, BN]
    | tl.OutputMatrixDescriptor[MDim, NDim, CS, BM, BN // 2],  #
    M: Int[MDim],
    N: Int[NDim],
    K: Int[KDim],  #
    BLOCK_SIZE_M: Int[BM],  #
    BLOCK_SIZE_N: Int[BN],  #
    BLOCK_SIZE_K: Int[BK],  #
    GROUP_SIZE_M: Int[Group],  #
    FP8_OUTPUT: bool,  #
    EPILOGUE_SUBTILE: bool,  #
    NUM_SMS: Int[SMs],  #
    WARP_SPECIALIZE: bool,  #
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Enable warp specialization to leverage async warp scheduling on the GPU.
    # FIXME: This only works on Blackwell right now. On older GPUs, this will
    # use software pipelining.
    for tile_id in tl.range(
        start_pid, num_tiles, NUM_SMS, flatten=True, warp_specialize=WARP_SPECIALIZE
    ):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_am_c = pid_m * BLOCK_SIZE_M
        offs_bn_c = pid_n * BLOCK_SIZE_N

        # Epilogue subtiling is a technique to break our computation and stores into multiple pieces
        # By subtiling we can reduce shared memory consumption by the epilogue and instead use that
        # memory to increase our stage count.
        # In this case we partition the accumulator into 2 BLOCK_SIZE_M x BLOCK_SIZE_N // 2 tensors
        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], c0)  # E: is not assignable
            c1 = acc1.to(dtype)
            c_desc.store(
                [offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2],
                c1,  # E: is not assignable
            )
        else:
            accumulator = accumulator.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], accumulator)  # E: is not assignable


def test_subtiled_value_shape[Rows: IntVar, Cols: IntVar](
    acc: tl.tensor[[Rows, Cols]], rows: Int[Rows], cols: Int[Cols]
) -> None:
    subtiled = tl.reshape(acc, (rows, 2, cols // 2))
    first, second = tl.split(tl.permute(subtiled, (0, 2, 1)))
    assert_type(first, tl.tensor[[Rows, Cols // 2]])
    assert_type(second, tl.tensor[[Rows, Cols // 2]])


def test_wrong_subtile_shape[Rows: IntVar, Cols: IntVar, Other: IntVar](
    acc: tl.tensor[[Rows, Cols]], rows: Int[Rows], wrong_cols: Int[Other]
) -> None:
    tl.reshape(acc, (rows, 2, wrong_cols // 2))  # E: is not assignable


def test_wrong_permutation[Rows: IntVar, Half: IntVar](
    reshaped: tl.tensor[[Rows, 2, Half]],
) -> None:
    tl.permute(reshaped, (2, 0, 1))  # E: is not assignable


def test_wrong_output_subtile[Rows: IntVar, Cols: IntVar, S: IntVar](
    desc: tl.OutputMatrixDescriptor[Rows, Cols, S, Rows, Cols // 2],
    full_value: tl.tensor[[Rows, Cols]],
) -> None:
    desc.store([0, 0], full_value)  # E: is not assignable


def test_wrong_output_full_tile[Rows: IntVar, Cols: IntVar, S: IntVar](
    desc: tl.OutputMatrixDescriptor[Rows, Cols, S, Rows, Cols],
    half_value: tl.tensor[[Rows, Cols // 2]],
) -> None:
    desc.store([0, 0], half_value)  # E: is not assignable


def test_unverified_descriptor_offsets[
    M: IntVar,
    N: IntVar,
    S: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    desc: tl.OutputMatrixDescriptor[M, N, S, BM, BN],
    tile: tl.tensor[[BM, BN]],
    unrelated_m: int,
    unrelated_n: int,
) -> None:
    # Descriptor addressing does not require the current computed tile indices.
    assert_type(desc.store([unrelated_m, unrelated_n], tile), None)


def test_wrong_epilogue_flag[
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
    SMs: IntVar,
](
    a: tl.InputMatrixDescriptor[M, K, AS, BM, BK],
    b: tl.InputMatrixDescriptor[N, K, BS, BN, BK],
    half_output: tl.OutputMatrixDescriptor[M, N, CS, BM, BN // 2],
    m: Int[M],
    n: Int[N],
    k: Int[K],
    bm: Int[BM],
    bn: Int[BN],
    bk: Int[BK],
    group: Int[Group],
    sms: Int[SMs],
) -> None:
    matmul_kernel_tma_persistent(  # E: No matching overload
        a, b, half_output, m, n, k, bm, bn, bk, group, False, False, sms, False
    )


def test_matching_epilogue_modes[
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
    SMs: IntVar,
](
    a: tl.InputMatrixDescriptor[M, K, AS, BM, BK],
    b: tl.InputMatrixDescriptor[N, K, BS, BN, BK],
    full_output: tl.OutputMatrixDescriptor[M, N, CS, BM, BN],
    half_output: tl.OutputMatrixDescriptor[M, N, CS, BM, BN // 2],
    m: Int[M],
    n: Int[N],
    k: Int[K],
    bm: Int[BM],
    bn: Int[BN],
    bk: Int[BK],
    group: Int[Group],
    sms: Int[SMs],
) -> None:
    assert_type(
        matmul_kernel_tma_persistent(
            a, b, full_output, m, n, k, bm, bn, bk, group, False, False, sms, False
        ),
        None,
    )
    assert_type(
        matmul_kernel_tma_persistent(
            a, b, half_output, m, n, k, bm, bn, bk, group, False, True, sms, False
        ),
        None,
    )


def test_wrong_boundary_output[
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
    SMs: IntVar,
](
    a: tl.InputMatrixDescriptor[M, K, AS, BM, BK],
    b: tl.InputMatrixDescriptor[N, K, BS, BN, BK],
    c: tl.OutputMatrixDescriptor[M, Other, CS, BM, BN],
    m: Int[M],
    n: Int[N],
    k: Int[K],
    bm: Int[BM],
    bn: Int[BN],
    bk: Int[BK],
    group: Int[Group],
    sms: Int[SMs],
) -> None:
    matmul_kernel_tma_persistent(  # E: No matching overload
        a, b, c, m, n, k, bm, bn, bk, group, False, False, sms, False
    )


def test_wrong_boundary_b_k[
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
    SMs: IntVar,
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
    sms: Int[SMs],
) -> None:
    matmul_kernel_tma_persistent(  # E: No matching overload
        a, b, c, m, n, k, bm, bn, bk, group, False, False, sms, False
    )

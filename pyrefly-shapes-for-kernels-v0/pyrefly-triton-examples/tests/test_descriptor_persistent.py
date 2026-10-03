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

"""Device-side descriptor construction in Triton's tutorial 09."""

from typing import assert_type, Literal

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
def matmul_kernel_descriptor_persistent[
    MDim: IntVar,
    NDim: IntVar,
    KDim: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Group: IntVar,
    SMs: IntVar,
](
    a_ptr: tl.InMatrixPointer[MDim, KDim, KDim, 1],
    b_ptr: tl.InMatrixPointer[NDim, KDim, KDim, 1],
    c_ptr: tl.OutMatrixPointer[MDim, NDim, NDim, 1],  #
    M: Int[MDim],
    N: Int[NDim],
    K: Int[KDim],  #
    BLOCK_SIZE_M: Int[BM],  #
    BLOCK_SIZE_N: Int[BN],  #
    BLOCK_SIZE_K: Int[BK],  #
    GROUP_SIZE_M: Int[Group],  #
    EPILOGUE_SUBTILE: Literal[False],  #
    NUM_SMS: Int[SMs],  #
    WARP_SPECIALIZE: bool,  #
    FLATTEN: bool,
):
    # Matmul using TMA and device-side descriptor creation
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[
            BLOCK_SIZE_M,
            BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2,
        ],
    )

    # tile_id_c is used in the epilogue to break the dependency between
    # the prologue and the epilogue
    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(
        start_pid, num_tiles, NUM_SMS, flatten=FLATTEN, warp_specialize=WARP_SPECIALIZE
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
        offs_cm = pid_m * BLOCK_SIZE_M
        offs_cn = pid_n * BLOCK_SIZE_N

        if EPILOGUE_SUBTILE:
            acc = tl.reshape(
                accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2)
            )
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_cm, offs_cn], c0)  # E: is not assignable
            c1 = acc1.to(dtype)
            c_desc.store(
                [offs_cm, offs_cn + BLOCK_SIZE_N // 2],
                c1,  # E: is not assignable
            )
        else:
            c = accumulator.to(dtype)
            c_desc.store([offs_cm, offs_cn], c)


def test_constructed_descriptor_shapes[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
](
    a: tl.InMatrixPointer[M, K, K, 1],
    b: tl.InMatrixPointer[N, K, K, 1],
    c: tl.OutMatrixPointer[M, N, N, 1],
    m: Int[M],
    n: Int[N],
    k: Int[K],
    bm: Int[BM],
    bn: Int[BN],
    bk: Int[BK],
) -> None:
    a_desc = tl.make_tensor_descriptor(
        a, shape=[m, k], strides=[k, 1], block_shape=[bm, bk]
    )
    b_desc = tl.make_tensor_descriptor(
        b, shape=[n, k], strides=[k, 1], block_shape=[bn, bk]
    )
    c_desc = tl.make_tensor_descriptor(
        c, shape=[m, n], strides=[n, 1], block_shape=[bm, bn]
    )
    assert_type(a_desc, tl.InputMatrixDescriptor[M, K, K, BM, BK])
    assert_type(b_desc, tl.InputMatrixDescriptor[N, K, K, BN, BK])
    assert_type(c_desc, tl.OutputMatrixDescriptor[M, N, N, BM, BN])


def test_wrong_input_shape[M: IntVar, K: IntVar, Other: IntVar, BM: IntVar, BK: IntVar](
    wrong_a: tl.InMatrixPointer[Other, K, K, 1],
    m: Int[M],
    k: Int[K],
    bm: Int[BM],
    bk: Int[BK],
) -> None:
    tl.make_tensor_descriptor(  # E: No matching overload
        wrong_a, shape=[m, k], strides=[k, 1], block_shape=[bm, bk]
    )


def test_wrong_input_stride[
    M: IntVar,
    K: IntVar,
    Other: IntVar,
    BM: IntVar,
    BK: IntVar,
](
    wrong_a: tl.InMatrixPointer[M, K, Other, 1],
    m: Int[M],
    k: Int[K],
    bm: Int[BM],
    bk: Int[BK],
) -> None:
    tl.make_tensor_descriptor(  # E: No matching overload
        wrong_a, shape=[m, k], strides=[k, 1], block_shape=[bm, bk]
    )


def test_wrong_b_k[N: IntVar, K: IntVar, Other: IntVar, BN: IntVar, BK: IntVar](
    wrong_b: tl.InMatrixPointer[N, Other, Other, 1],
    n: Int[N],
    k: Int[K],
    bn: Int[BN],
    bk: Int[BK],
) -> None:
    tl.make_tensor_descriptor(  # E: No matching overload
        wrong_b, shape=[n, k], strides=[k, 1], block_shape=[bn, bk]
    )


def test_wrong_c_shape[M: IntVar, N: IntVar, Other: IntVar, BM: IntVar, BN: IntVar](
    wrong_c: tl.OutMatrixPointer[M, Other, Other, 1],
    m: Int[M],
    n: Int[N],
    bm: Int[BM],
    bn: Int[BN],
) -> None:
    tl.make_tensor_descriptor(  # E: No matching overload
        wrong_c, shape=[m, n], strides=[n, 1], block_shape=[bm, bn]
    )


def test_wrong_output_role[M: IntVar, N: IntVar, BM: IntVar, BN: IntVar](
    input_pointer: tl.InMatrixPointer[M, N, N, 1],
    m: Int[M],
    n: Int[N],
    bm: Int[BM],
    bn: Int[BN],
) -> None:
    descriptor = tl.make_tensor_descriptor(
        input_pointer, shape=[m, n], strides=[n, 1], block_shape=[bm, bn]
    )
    descriptor.store([0, 0], tl.zeros((bm, bn), dtype=tl.float32))  # E: store


def test_unsupported_epilogue_mode[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Group: IntVar,
    SMs: IntVar,
](
    a: tl.InMatrixPointer[M, K, K, 1],
    b: tl.InMatrixPointer[N, K, K, 1],
    c: tl.OutMatrixPointer[M, N, N, 1],
    m: Int[M],
    n: Int[N],
    k: Int[K],
    bm: Int[BM],
    bn: Int[BN],
    bk: Int[BK],
    group: Int[Group],
    sms: Int[SMs],
) -> None:
    matmul_kernel_descriptor_persistent(
        a,
        b,
        c,
        m,
        n,
        k,
        bm,
        bn,
        bk,
        group,
        True,  # E: is not assignable
        sms,
        False,
        True,
    )


def test_unverified_descriptor_index[M: IntVar, K: IntVar, BM: IntVar, BK: IntVar](
    a: tl.InMatrixPointer[M, K, K, 1],
    m: Int[M],
    k: Int[K],
    bm: Int[BM],
    bk: Int[BK],
    unrelated_row: int,
    unrelated_k: int,
) -> None:
    descriptor = tl.make_tensor_descriptor(
        a, shape=[m, k], strides=[k, 1], block_shape=[bm, bk]
    )
    # A bound-aware constructor does not prove subsequent descriptor indices.
    assert_type(descriptor.load([unrelated_row, unrelated_k]), tl.tensor[[BM, BK]])

# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The entrypoint body is copied from Triton's python/tutorials/compilation-pipeline/06_warp_specialization.py.
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

"""Static descriptor forwarding at the warp-specialized matmul entrypoint."""

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


# Trusted helper interface only. Its executable body is a separate kernel
# and is not checked by this entrypoint fixture.
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
) -> None: ...


@triton.jit
def matmul[
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
    FLATTEN: bool,
    BLOCK_M: Int[BM],
    BLOCK_N: Int[BN],
    BLOCK_K: Int[BK],
    GROUP_M: Int[Group],
    NUM_SMS: Int[SMs],
):
    _matmul_persistent_ws(
        a_desc,
        b_desc,
        c_desc,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        GROUP_M,
        NUM_SMS,
        FLATTEN,
    )


def test_descriptor_host_shape[
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
    wrong_b: tl.InputMatrixDescriptor[Other, KDim, KDim, BN, BK],
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
    matmul(a, b, c, m, n, k, False, block_m, block_n, block_k, group, sms)
    matmul(
        wrong_a,
        b,  # E: is not assignable
        c,
        m,
        n,
        k,  # E: is not assignable
        False,
        block_m,
        block_n,
        block_k,
        group,
        sms,
    )
    matmul(
        a,
        wrong_b,
        c,  # E: is not assignable
        m,
        n,  # E: is not assignable
        k,
        False,
        block_m,
        block_n,
        block_k,
        group,
        sms,
    )
    matmul(
        a,
        b,
        wrong_c,  # E: is not assignable
        m,
        n,
        k,
        False,
        block_m,
        block_n,
        block_k,
        group,
        sms,
    )


# Changing only the forwarded output descriptor rejects the original call.
@triton.jit
def matmul_wrong_output[
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
    c_desc: tl.OutputMatrixDescriptor[MDim, Other, Other, BM, BN],
    M: Int[MDim],
    N: Int[NDim],
    K: Int[KDim],
    FLATTEN: bool,
    BLOCK_M: Int[BM],
    BLOCK_N: Int[BN],
    BLOCK_K: Int[BK],
    GROUP_M: Int[Group],
    NUM_SMS: Int[SMs],
):
    _matmul_persistent_ws(
        a_desc,
        b_desc,
        c_desc,  # E: is not assignable
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        GROUP_M,
        NUM_SMS,
        FLATTEN,
    )

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only Torch descriptor bridge; does not run the Blackwell kernel.
# @lint-ignore-every AUTODEPS2

"""Check declared Torch-to-Blackwell MMA descriptor dimensions and roles."""

from typing import assert_type

import torch
from shape_extensions import Int, IntListLiteral, IntVar
from tests.test_gluon_tcgen05_small import small_mma_kernel
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


def test_small_mma_host[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    Other: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
](
    m: Int[M],
    n: Int[N],
    k: Int[K],
    other: Int[Other],
    bm: Int[BM],
    bn: Int[BN],
    bk: Int[BK],
) -> None:
    a = torch.empty((m, k), device="cuda", dtype=torch.float16)
    b = torch.empty((k, n), device="cuda", dtype=torch.float16)
    c = torch.empty((m, n), device="cuda", dtype=torch.float32)
    d = torch.empty((m, n), device="cuda", dtype=torch.float32)
    wrong_b = torch.empty((other, n), device="cuda", dtype=torch.float16)
    wrong_c = torch.empty((other, n), device="cuda", dtype=torch.float32)
    wrong_d = torch.empty((m, other), device="cuda", dtype=torch.float32)

    a_block: IntListLiteral[[BM, BK]] = [bm, bk]
    b_block: IntListLiteral[[BK, BN]] = [bk, bn]
    cd_block: IntListLiteral[[BM, BN]] = [bm, bn]
    other_cd_block: IntListLiteral[[BM, Other]] = [bm, other]
    a_layout = gl.NVMMASharedLayout.get_default_for(a_block, gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for(b_block, gl.float16)
    cd_layout = gl.NVMMASharedLayout.get_default_for(cd_block, gl.float32)
    other_cd_layout = gl.NVMMASharedLayout.get_default_for(other_cd_block, gl.float32)

    a_desc = TensorDescriptor.from_tensor(a, a_block, a_layout)
    b_desc = TensorDescriptor.from_tensor(b, b_block, b_layout)
    c_desc = TensorDescriptor.from_tensor(c, cd_block, cd_layout)
    d_desc = TensorDescriptor.from_tensor(d, cd_block, cd_layout)
    wrong_b_desc = TensorDescriptor.from_tensor(wrong_b, b_block, b_layout)
    wrong_c_desc = TensorDescriptor.from_tensor(wrong_c, cd_block, cd_layout)
    wrong_d_desc = TensorDescriptor.from_tensor(wrong_d, cd_block, cd_layout)

    assert_type(a_desc, gl.WgmmaDescriptorF16[M, K, BM, BK, int])
    assert_type(d_desc, gl.WgmmaDescriptorF32[M, N, BM, BN, int])
    # Unlike float16, float32 layouts do not remember their factory block.
    assert_type(
        TensorDescriptor.from_tensor(c, cd_block, other_cd_layout),
        gl.WgmmaDescriptorF32[M, N, BM, BN, int],
    )
    small_mma_kernel(a_desc, b_desc, c_desc, d_desc, (bm, bn), False, False, 4)
    small_mma_kernel(a_desc, b_desc, c_desc, d_desc, (bm, bn), True, True, 4)
    small_mma_kernel(
        a_desc,
        wrong_b_desc,  # E: is not assignable
        c_desc,
        d_desc,
        (bm, bn),
        False,
        False,
        4,
    )
    small_mma_kernel(
        a_desc,
        b_desc,
        wrong_c_desc,  # E: is not assignable
        d_desc,
        (bm, bn),
        False,
        False,
        4,
    )
    small_mma_kernel(
        a_desc,
        b_desc,
        c_desc,
        wrong_d_desc,  # E: is not assignable
        (bm, bn),
        False,
        False,
        4,
    )

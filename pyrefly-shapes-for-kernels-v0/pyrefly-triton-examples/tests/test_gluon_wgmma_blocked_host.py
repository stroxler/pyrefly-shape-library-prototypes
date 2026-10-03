# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only shape bridge; no Triton kernel or CUDA code executes.
# @lint-ignore-every AUTODEPS2

"""Follow blocked WGMMA's untransposed host shapes through 2D TMA descriptors."""

from typing import assert_type

import torch
from shape_extensions import Int, IntListLiteral, IntVar
from tests.test_gluon_wgmma_blocked import blocked_matmul_kernel
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


def test_torch_to_blocked_kernel[
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
    output = torch.empty((m, n), device="cuda", dtype=torch.float16)
    wrong_b_k = torch.empty((other, n), device="cuda", dtype=torch.float16)
    wrong_output_n = torch.empty((m, other), device="cuda", dtype=torch.float16)

    a_block: IntListLiteral[[BM, BK]] = [bm, bk]
    b_block: IntListLiteral[[BK, BN]] = [bk, bn]
    output_block: IntListLiteral[[BM, BN]] = [bm, bn]
    a_layout = gl.NVMMASharedLayout.get_default_for(a_block, gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for(b_block, gl.float16)
    output_layout = gl.NVMMASharedLayout.get_default_for(output_block, gl.float16)

    a_desc = TensorDescriptor.from_tensor(a, a_block, a_layout)
    b_desc = TensorDescriptor.from_tensor(b, b_block, b_layout)
    output_desc = TensorDescriptor.from_tensor(output, output_block, output_layout)
    wrong_b_desc = TensorDescriptor.from_tensor(wrong_b_k, b_block, b_layout)
    wrong_output_desc = TensorDescriptor.from_tensor(
        wrong_output_n, output_block, output_layout
    )

    assert_type(a_desc, gl.WgmmaDescriptorF16[M, K, BM, BK, int])
    assert_type(output_desc, gl.WgmmaDescriptorF16[M, N, BM, BN, int])
    blocked_matmul_kernel(a_desc, b_desc, output_desc, False, 4)
    blocked_matmul_kernel(
        a_desc,
        wrong_b_desc,  # E: is not assignable
        output_desc,
        False,
        4,
    )
    blocked_matmul_kernel(
        a_desc,
        b_desc,
        wrong_output_desc,  # E: is not assignable
        False,
        4,
    )

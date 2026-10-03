# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only host-to-kernel descriptor probe; no Hopper GPU is executed.
# @lint-ignore-every AUTODEPS2

"""Trace 2D Torch allocation shapes through Hopper TMA descriptors to MMA."""

from typing import assert_type

import torch
from shape_extensions import Int, IntListLiteral, IntVar
from tests.test_gluon_wgmma_small import small_mma_kernel
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


def test_host_descriptor_boundary[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    Other: IntVar,
](
    m: Int[M],
    n: Int[N],
    k: Int[K],
    other: Int[Other],
    bm: Int[M],
    bn: Int[N],
    bk: Int[K],
) -> None:
    a = torch.empty((m, k), device="cuda", dtype=torch.float16)
    b = torch.empty((k, n), device="cuda", dtype=torch.float16)
    c = torch.empty((m, n), device="cuda", dtype=torch.float32)
    d = torch.empty_like(c)
    wrong_b = torch.empty((other, n), device="cuda", dtype=torch.float16)
    wrong_d = torch.empty((m, other), device="cuda", dtype=torch.float32)
    wrong_a_dtype = torch.empty((m, k), device="cuda", dtype=torch.float32)

    a_block: IntListLiteral[[M, K]] = [bm, bk]
    b_block: IntListLiteral[[K, N]] = [bk, bn]
    cd_block: IntListLiteral[[M, N]] = [bm, bn]
    a_layout = gl.NVMMASharedLayout.get_default_for(a_block, gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for(b_block, gl.float16)
    cd_layout = gl.NVMMASharedLayout.get_default_for(cd_block, gl.float32)
    a_desc = TensorDescriptor.from_tensor(a, a_block, a_layout)
    b_desc = TensorDescriptor.from_tensor(b, b_block, b_layout)
    c_desc = TensorDescriptor.from_tensor(c, cd_block, cd_layout)
    d_desc = TensorDescriptor.from_tensor(d, cd_block, cd_layout)
    wrong_b_desc = TensorDescriptor.from_tensor(wrong_b, b_block, b_layout)
    wrong_d_desc = TensorDescriptor.from_tensor(wrong_d, cd_block, cd_layout)
    # The shape-aware Torch tensor does not retain the runtime dtype.
    unproved_dtype = TensorDescriptor.from_tensor(wrong_a_dtype, a_block, a_layout)

    assert_type(a_desc, gl.WgmmaDescriptorF16[M, K, M, K, int])
    assert_type(c_desc, gl.WgmmaDescriptorF32[M, N, M, N, int])
    small_mma_kernel(a_desc, b_desc, c_desc, d_desc, False, 16, 4)
    small_mma_kernel(unproved_dtype, b_desc, c_desc, d_desc, False, 16, 4)
    small_mma_kernel(
        a_desc,
        wrong_b_desc,  # E: is not assignable
        c_desc,
        d_desc,
        False,
        16,
        4,
    )
    small_mma_kernel(
        a_desc,
        b_desc,
        c_desc,
        wrong_d_desc,  # E: is not assignable
        False,
        16,
        4,
    )
    # from_tensor does not know whether the descriptor is read or written.
    small_mma_kernel(a_desc, b_desc, d_desc, c_desc, False, 16, 4)

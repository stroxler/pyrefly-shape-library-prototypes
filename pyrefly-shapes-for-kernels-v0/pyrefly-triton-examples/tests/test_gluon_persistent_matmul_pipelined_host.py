# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only direct kernel call, not the source wrapper's bracketed JIT launch.
# @lint-ignore-every AUTODEPS2

"""Check the declared Torch A/B/C interface to persistent Gluon matmul."""

import torch
from shape_extensions import Int, IntListLiteral, IntVar
from tests.test_gluon_persistent_issue_mma import MMAv5, WGMMA
from tests.test_gluon_persistent_matmul_pipelined import matmul_pipelined_kernel
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


def test_torch_persistent_matmul_boundary[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Depth: IntVar,
    Other: IntVar,
](
    m: Int[M],
    n: Int[N],
    k: Int[K],
    bm: Int[BM],
    bn: Int[BN],
    bk: Int[BK],
    depth: Int[Depth],
    other: Int[Other],
) -> None:
    a = torch.empty((m, k), device="cuda", dtype=torch.float16)
    b = torch.empty((k, n), device="cuda", dtype=torch.float16)
    c = torch.empty((m, n), device="cuda", dtype=torch.float16)
    wrong_b_k = torch.empty((other, n), device="cuda", dtype=torch.float16)
    wrong_c_n = torch.empty((m, other), device="cuda", dtype=torch.float16)
    wrong_a_dtype = torch.empty((m, k), device="cuda", dtype=torch.float32)

    a_block: IntListLiteral[[BM, BK]] = [bm, bk]
    b_block: IntListLiteral[[BK, BN]] = [bk, bn]
    c_block: IntListLiteral[[BM, BN]] = [bm, bn]
    a_layout = gl.NVMMASharedLayout.get_default_for(a_block, gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for(b_block, gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for(c_block, gl.float16)

    a_desc = TensorDescriptor.from_tensor(a, a_block, a_layout)
    b_desc = TensorDescriptor.from_tensor(b, b_block, b_layout)
    c_desc = TensorDescriptor.from_tensor(c, c_block, c_layout)
    wrong_b_desc = TensorDescriptor.from_tensor(wrong_b_k, b_block, b_layout)
    wrong_c_desc = TensorDescriptor.from_tensor(wrong_c_n, c_block, c_layout)
    wrong_dtype_desc = TensorDescriptor.from_tensor(wrong_a_dtype, a_block, a_layout)

    matmul_pipelined_kernel(a_desc, b_desc, c_desc, WGMMA, depth, 8)
    matmul_pipelined_kernel(a_desc, b_desc, c_desc, MMAv5, depth, 4)
    matmul_pipelined_kernel(
        a_desc,
        wrong_b_desc,  # E: is not assignable
        c_desc,
        WGMMA,
        depth,
        8,
    )
    matmul_pipelined_kernel(
        a_desc,
        b_desc,
        wrong_c_desc,  # E: is not assignable
        MMAv5,
        depth,
        4,
    )
    # Shape-only Torch tensor tracking does not prove float16 dtype.
    matmul_pipelined_kernel(wrong_dtype_desc, b_desc, c_desc, WGMMA, depth, 8)

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only original host wrapper; the JIT bracket launch remains untyped.
# @lint-ignore-every AUTODEPS2

"""Probe the original persistence tutorial wrapper without editing its body."""

import torch
import triton
from shape_extensions import Int, IntVar
from tests.test_gluon_persistent_issue_mma import MMAv5, WGMMA
from tests.test_gluon_persistent_matmul_pipelined import matmul_pipelined_kernel
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


def select_mma_impl():
    if torch.cuda.get_device_capability()[0] == 9:
        return WGMMA
    elif torch.cuda.get_device_capability()[0] == 10:
        return MMAv5
    else:
        return None


def matmul_pipelined[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Depth: IntVar,
](
    A: torch.Tensor[[M, K]],
    B: torch.Tensor[[K, N]],
    C: torch.Tensor[[M, N]],
    BLOCK_M: Int[BM],
    BLOCK_N: Int[BN],
    BLOCK_K: Int[BK],
    num_buffers: Int[Depth],
    num_warps: int,
):
    MMAImpl = select_mma_impl()
    M, N = C.shape

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_pipelined_kernel[grid](  # E: is not subscriptable
        a_desc, b_desc, c_desc, MMAImpl, num_buffers, num_warps=num_warps
    )


def test_typed_wrapper_boundary[
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
    wrong_b = torch.empty((other, n), device="cuda", dtype=torch.float16)
    wrong_c = torch.empty((m, other), device="cuda", dtype=torch.float16)
    matmul_pipelined(a, b, c, bm, bn, bk, depth, 8)
    matmul_pipelined(a, wrong_b, c, bm, bn, bk, depth, 8)  # E: is not assignable
    matmul_pipelined(a, b, wrong_c, bm, bn, bk, depth, 8)  # E: is not assignable

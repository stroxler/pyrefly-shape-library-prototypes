# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; the original bracketed JIT launch remains untyped.
# @lint-ignore-every AUTODEPS2

"""Preserve the original Blackwell accumulate wrapper's Torch interface."""

from typing import assert_type

import torch
import triton
from shape_extensions import Int, IntVar
from tests import test_gluon_persistent_issue_mma as t7
from tests.test_gluon_tcgen05_matmul_accumulate_kernel import (
    matmul_accumulate_kernel,
)
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


def matmul_accumulate[
    MRows: IntVar,
    NCols: IntVar,
    KReduction: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Group: IntVar,
    Depth: IntVar,
](
    A: torch.Tensor[[MRows, KReduction]],
    B: torch.Tensor[[KReduction, NCols]],
    C: torch.Tensor[[MRows, NCols]],
    BLOCK_M: Int[BM] = 128,
    BLOCK_N: Int[BN] = 128,
    BLOCK_K: Int[BK] = 64,
    GROUP_SIZE_M: Int[Group] = 8,
    num_buffers: Int[Depth] = 3,
):
    SchedulerImpl = t7.GroupedPersistentTileScheduler(GROUP_SIZE_M)  # E: No attribute
    M, N = C.shape

    dtype = getattr(gl, str(A.dtype).split(".")[1])
    acc_dtype = getattr(gl, str(C.dtype).split(".")[1])
    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], dtype)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], dtype)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], acc_dtype)

    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)
    D = torch.empty((M, N), dtype=C.dtype, device="cuda")

    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(num_sms, num_pid),)
    matmul_accumulate_kernel[grid](  # E: is not subscriptable
        a_desc, b_desc, c_desc, D, *D.stride(), SchedulerImpl, num_buffers
    )
    return D


def test_original_wrapper_declared_torch_boundary[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    Other: IntVar,
](m: Int[M], n: Int[N], k: Int[K], other: Int[Other]) -> None:
    a = torch.empty((m, k), dtype=torch.float16, device="cuda")
    b = torch.empty((k, n), dtype=torch.float16, device="cuda")
    c = torch.empty((m, n), dtype=torch.float32, device="cuda")
    wrong_b = torch.empty((other, n), dtype=torch.float16, device="cuda")
    wrong_c = torch.empty((m, other), dtype=torch.float32, device="cuda")
    wrong_a_dtype = torch.empty((m, k), dtype=torch.float32, device="cuda")
    wrong_c_dtype = torch.empty((m, n), dtype=torch.float16, device="cuda")
    d = matmul_accumulate(a, b, c)
    assert_type(d, torch.Tensor[[M, N]])
    matmul_accumulate(a, wrong_b, c)  # E: is not assignable
    matmul_accumulate(a, b, wrong_c)  # E: is not assignable
    matmul_accumulate(wrong_a_dtype, b, c)
    matmul_accumulate(a, b, wrong_c_dtype)

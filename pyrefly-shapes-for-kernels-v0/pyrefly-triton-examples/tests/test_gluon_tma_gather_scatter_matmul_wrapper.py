# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only original wrapper; the JIT bracket launch remains untyped.
# @lint-ignore-every AUTODEPS2

"""Check the original fused gather/scatter wrapper's Torch boundary."""

from typing import assert_type

import torch
import triton
from shape_extensions import Int, IntVar
from tests import test_gluon_persistent_issue_mma as t7
from tests.test_gluon_tma_gather_scatter_matmul import (
    matmul_fused_gather_scatter_kernel,
)
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


def matmul_fused_gather_scatter[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Group: IntVar,
    Depth: IntVar,
](
    X: torch.Tensor[[M, K]],
    X_gather_indx: torch.Tensor[[M]],
    W: torch.Tensor[[K, N]],
    out_scatter_indx: torch.Tensor[[M]],
    BLOCK_M: Int[BM] = 128,
    BLOCK_N: Int[BN] = 128,
    BLOCK_K: Int[BK] = 64,
    GROUP_SIZE_M: Int[Group] = 8,
    num_buffers: Int[Depth] = 3,
):
    M = X.shape[0]
    N = W.shape[1]
    out = torch.empty((M, N), dtype=X.dtype, device="cuda")

    # Convert torch dtype to gluon dtype.
    dtype = getattr(gl, str(X.dtype).split(".")[1])
    # Setup descriptors for inputs and outputs.
    X_desc_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], dtype)
    W_desc_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], dtype)
    out_desc_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], dtype)

    X_desc = TensorDescriptor.from_tensor(X, [1, BLOCK_K], X_desc_layout)
    W_desc = TensorDescriptor.from_tensor(W, [BLOCK_K, BLOCK_N], W_desc_layout)
    out_desc = TensorDescriptor.from_tensor(out, [1, BLOCK_N], out_desc_layout)

    # Persistent kernel grid.
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(num_sms, num_pid),)
    SchedulerImpl = t7.GroupedPersistentTileScheduler(GROUP_SIZE_M)  # E: No attribute
    matmul_fused_gather_scatter_kernel[grid](  # E: is not subscriptable
        X_desc,
        W_desc,
        out_desc,
        X_gather_indx,
        out_scatter_indx,
        BLOCK_M,
        SchedulerImpl,
        num_buffers,
    )
    return out


def test_wrapper_declared_torch_boundary[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    Other: IntVar,
](m: Int[M], n: Int[N], k: Int[K], other: Int[Other]) -> None:
    x = torch.empty((m, k), dtype=torch.float16, device="cuda")
    w = torch.empty((k, n), dtype=torch.float16, device="cuda")
    gather = torch.empty((m,), dtype=torch.int32, device="cuda")
    scatter = torch.empty((m,), dtype=torch.int32, device="cuda")
    wrong_w = torch.empty((other, n), dtype=torch.float16, device="cuda")
    wrong_gather = torch.empty((other,), dtype=torch.int32, device="cuda")
    wrong_scatter = torch.empty((other,), dtype=torch.int32, device="cuda")
    output = matmul_fused_gather_scatter(x, gather, w, scatter)
    assert_type(output, torch.Tensor[[M, N]])
    matmul_fused_gather_scatter(x, gather, wrong_w, scatter)  # E: is not assignable
    matmul_fused_gather_scatter(x, wrong_gather, w, scatter)  # E: is not assignable
    matmul_fused_gather_scatter(x, gather, w, wrong_scatter)  # E: is not assignable

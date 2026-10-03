# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only original host wrapper; the JIT bracket launch remains untyped.
# @lint-ignore-every AUTODEPS2

"""Check the source pipelined persistent wrapper's declared Torch boundary."""

import torch
import triton
from shape_extensions import Int, IntVar
from tests.test_gluon_persistent_matmul import PersistentTileScheduler
from tests.test_gluon_persistent_matmul_pipelined_stealb import (
    persistent_matmul_pipelined_kernel,
)
from tests.test_gluon_persistent_matmul_pipelined_wrapper import select_mma_impl
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


def persistent_matmul_pipelined[
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
    SchedulerImpl: type[PersistentTileScheduler],
):
    M, N = C.shape
    MMAImpl = select_mma_impl()

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)
    c_half_layout = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_M, BLOCK_N // 2], gl.float16
    )

    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)
    c_half_desc = TensorDescriptor.from_tensor(
        C, [BLOCK_M, BLOCK_N // 2], c_half_layout
    )

    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(num_sms, num_pid),)
    stealb = num_buffers == 4
    persistent_matmul_pipelined_kernel[grid](  # E: is not subscriptable
        a_desc,
        b_desc,
        c_desc,
        c_half_desc,
        MMAImpl,
        SchedulerImpl,
        num_buffers,
        STEALB=stealb,
        num_warps=num_warps,
    )


def test_wrapper_declares_host_output_shape[
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
    persistent_matmul_pipelined(a, b, c, bm, bn, bk, depth, 8, PersistentTileScheduler)
    persistent_matmul_pipelined(
        a,
        wrong_b,  # E: is not assignable
        c,
        bm,
        bn,
        bk,
        depth,
        8,
        PersistentTileScheduler,
    )
    persistent_matmul_pipelined(
        a,
        b,
        wrong_c,  # E: is not assignable
        bm,
        bn,
        bk,
        depth,
        8,
        PersistentTileScheduler,
    )

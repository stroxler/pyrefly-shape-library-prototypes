# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only typed direct bridge, distinct from the tutorial's JIT bracket launch.
# @lint-ignore-every AUTODEPS2

"""Separate checked W construction from trusted gather and scatter descriptors."""

import torch
from shape_extensions import Int, IntListLiteral, IntVar
from tests.test_gluon_persistent_matmul import PersistentTileScheduler
from tests.test_gluon_tma_gather_scatter_matmul import (
    matmul_fused_gather_scatter_kernel,
)
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


def test_torch_fused_boundary_with_trusted_gather_descriptors[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Other: IntVar,
    LX: IntVar,
    LO: IntVar,
    Depth: IntVar,
](
    m: Int[M],
    n: Int[N],
    k: Int[K],
    bm: Int[BM],
    bn: Int[BN],
    bk: Int[BK],
    other: Int[Other],
    depth: Int[Depth],
    x_desc: gl.GatherInputDescriptorF16[M, K, BM, BK, LX],
    out_desc: gl.ScatterOutputDescriptorF16[M, N, BM, BN, LO],
    wrong_out: gl.ScatterOutputDescriptorF16[M, Other, BM, BN, LO],
    gather_ptr: gl.GatherOffsetsPointer1D[M],
    scatter_ptr: gl.GatherOffsetsPointer1D[M],
    wrong_scatter_ptr: gl.GatherOffsetsPointer1D[Other],
) -> None:
    x = torch.empty((m, k), dtype=torch.float16, device="cuda")
    w = torch.empty((k, n), dtype=torch.float16, device="cuda")
    wrong_w = torch.empty((other, n), dtype=torch.float16, device="cuda")
    wrong_w_dtype = torch.empty((k, n), dtype=torch.float32, device="cuda")
    out = torch.empty((m, n), dtype=torch.float16, device="cuda")
    w_block: IntListLiteral[[BK, BN]] = [bk, bn]
    w_layout = gl.NVMMASharedLayout.get_default_for(w_block, gl.float16)
    w_desc = TensorDescriptor.from_tensor(w, w_block, w_layout)
    wrong_w_desc = TensorDescriptor.from_tensor(wrong_w, w_block, w_layout)
    wrong_dtype_desc = TensorDescriptor.from_tensor(wrong_w_dtype, w_block, w_layout)
    matmul_fused_gather_scatter_kernel(
        x_desc,
        w_desc,
        out_desc,
        gather_ptr,
        scatter_ptr,
        bm,
        PersistentTileScheduler,
        depth,
    )
    matmul_fused_gather_scatter_kernel(
        x_desc,
        wrong_w_desc,  # E: is not assignable
        out_desc,
        gather_ptr,
        scatter_ptr,
        bm,
        PersistentTileScheduler,
        depth,
    )
    matmul_fused_gather_scatter_kernel(
        x_desc,
        w_desc,
        wrong_out,  # E: is not assignable
        gather_ptr,
        scatter_ptr,
        bm,
        PersistentTileScheduler,
        depth,
    )
    matmul_fused_gather_scatter_kernel(
        x_desc,
        wrong_dtype_desc,
        out_desc,
        gather_ptr,
        scatter_ptr,
        bm,
        PersistentTileScheduler,
        depth,
    )
    matmul_fused_gather_scatter_kernel(
        x_desc,
        w_desc,
        out_desc,
        gather_ptr,
        wrong_scatter_ptr,  # E: is not assignable
        bm,
        PersistentTileScheduler,
        depth,
    )

    # The regular descriptor constructor requires its physical and layout tiles
    # to agree. It cannot construct the tutorial's gather/scatter descriptors.
    x_physical: IntListLiteral[[1, BK]] = [1, bk]
    x_layout = gl.NVMMASharedLayout.get_default_for([bm, bk], gl.float16)
    TensorDescriptor.from_tensor(x, x_physical, x_layout)  # E: No matching overload
    x_bad_layout = gl.NVMMASharedLayout.get_default_for([bm, bn], gl.float16)
    TensorDescriptor.from_tensor(x, x_physical, x_bad_layout)  # E: No matching overload
    out_physical: IntListLiteral[[1, BN]] = [1, bn]
    out_layout = gl.NVMMASharedLayout.get_default_for([bm, bn], gl.float16)
    TensorDescriptor.from_tensor(  # E: No matching overload
        out,
        out_physical,
        out_layout,
    )
    out_bad_layout = gl.NVMMASharedLayout.get_default_for([bm, bk], gl.float16)
    TensorDescriptor.from_tensor(  # E: No matching overload
        out,
        out_physical,
        out_bad_layout,
    )

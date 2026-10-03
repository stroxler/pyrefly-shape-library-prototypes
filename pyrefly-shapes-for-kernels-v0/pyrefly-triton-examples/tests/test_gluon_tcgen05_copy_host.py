# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only declared Torch-to-pointer adapter; no runtime pointer conversion.
# @lint-ignore-every AUTODEPS2

"""Distinguish Torch allocation dimensions from trusted float32 pointer roles."""

import torch
from shape_extensions import Int, IntVar
from tests.test_gluon_tcgen05_copy import tcgen05_copy_kernel
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import TensorMemoryLayout


def typed_copy_boundary[
    M: IntVar,
    N: IntVar,
    InR: IntVar,
    InC: IntVar,
    OutR: IntVar,
    OutC: IntVar,
](
    source: torch.Tensor[[M, N]],
    destination: torch.Tensor[[M, N]],
    source_ptr: gl.InFloat32MatrixPointer2D[M, N, InR, InC],
    source_row_stride: Int[InR],
    source_col_stride: Int[InC],
    destination_ptr: gl.OutMatrixPointer2D[M, N, OutR, OutC],
    destination_row_stride: Int[OutR],
    destination_col_stride: Int[OutC],
    m: Int[M],
    n: Int[N],
    shared_layout: gl.NVMMASharedLayout,
    tensor_layout: TensorMemoryLayout,
) -> None:
    tcgen05_copy_kernel(
        source_ptr,
        source_row_stride,
        source_col_stride,
        destination_ptr,
        destination_row_stride,
        destination_col_stride,
        m,
        n,
        shared_layout,
        tensor_layout,
    )


def test_torch_copy_bridge[
    M: IntVar,
    N: IntVar,
    Other: IntVar,
    InR: IntVar,
    InC: IntVar,
    OutR: IntVar,
    OutC: IntVar,
](
    m: Int[M],
    n: Int[N],
    other: Int[Other],
    src_ptr: gl.InFloat32MatrixPointer2D[M, N, InR, InC],
    in_r: Int[InR],
    in_c: Int[InC],
    dst_ptr: gl.OutMatrixPointer2D[M, N, OutR, OutC],
    out_r: Int[OutR],
    out_c: Int[OutC],
    shared_layout: gl.NVMMASharedLayout,
    tensor_layout: TensorMemoryLayout,
) -> None:
    source = torch.empty((m, n), dtype=torch.float32, device="cuda")
    dest = torch.empty((m, n), dtype=torch.float32, device="cuda")
    wrong_source = torch.empty((m, other), dtype=torch.float32, device="cuda")
    wrong_dest = torch.empty((other, n), dtype=torch.float32, device="cuda")
    accepted_wrong_dtype = torch.empty((m, n), dtype=torch.float16, device="cuda")
    typed_copy_boundary(
        source,
        dest,
        src_ptr,
        in_r,
        in_c,
        dst_ptr,
        out_r,
        out_c,
        m,
        n,
        shared_layout,
        tensor_layout,
    )
    typed_copy_boundary(
        wrong_source,
        dest,  # E: is not assignable
        src_ptr,  # E: is not assignable
        in_r,
        in_c,
        dst_ptr,  # E: is not assignable
        out_r,
        out_c,
        m,
        n,  # E: is not assignable
        shared_layout,
        tensor_layout,
    )
    typed_copy_boundary(
        source,
        wrong_dest,  # E: is not assignable
        src_ptr,
        in_r,
        in_c,
        dst_ptr,
        out_r,
        out_c,
        m,
        n,
        shared_layout,
        tensor_layout,
    )
    typed_copy_boundary(
        accepted_wrong_dtype,
        dest,
        src_ptr,
        in_r,
        in_c,
        dst_ptr,
        out_r,
        out_c,
        m,
        n,
        shared_layout,
        tensor_layout,
    )

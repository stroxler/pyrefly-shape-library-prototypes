# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only typed Torch bridge; the original dynamic wrapper remains separate.
# @lint-ignore-every AUTODEPS2

"""Separate regular A/B/C descriptor construction from the untyped launch."""

from typing import assert_type

import torch
from shape_extensions import Int, IntListLiteral, IntVar
from tests.test_gluon_persistent_matmul import PersistentTileScheduler
from tests.test_gluon_tcgen05_matmul_accumulate_kernel import (
    matmul_accumulate_kernel,
)
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


def output_pointer[
    Rows: IntVar,
    Cols: IntVar,
    RowStride: IntVar,
    ColStride: IntVar,
](
    output: torch.Tensor[[Rows, Cols]],
    row_stride: Int[RowStride],
    col_stride: Int[ColStride],
) -> gl.OutMatrixPointer2D[Rows, Cols, RowStride, ColStride]: ...


def test_accumulate_torch_descriptor_interface[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    Other: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    RowStride: IntVar,
    ColStride: IntVar,
    Depth: IntVar,
](
    m: Int[M],
    n: Int[N],
    k: Int[K],
    other: Int[Other],
    bm: Int[BM],
    bn: Int[BN],
    bk: Int[BK],
    depth: Int[Depth],
    row_stride: Int[RowStride],
    col_stride: Int[ColStride],
) -> None:
    a = torch.empty((m, k), dtype=torch.float16, device="cuda")
    b = torch.empty((k, n), dtype=torch.float16, device="cuda")
    c = torch.empty((m, n), dtype=torch.float32, device="cuda")
    d = torch.empty((m, n), dtype=torch.float32, device="cuda")
    wrong_a = torch.empty((m, other), dtype=torch.float16, device="cuda")
    wrong_b = torch.empty((other, n), dtype=torch.float16, device="cuda")
    wrong_c = torch.empty((m, other), dtype=torch.float32, device="cuda")
    wrong_d = torch.empty((m, other), dtype=torch.float32, device="cuda")
    wrong_dtype_a = torch.empty((m, k), dtype=torch.float32, device="cuda")

    a_block: IntListLiteral[[BM, BK]] = [bm, bk]
    b_block: IntListLiteral[[BK, BN]] = [bk, bn]
    c_block: IntListLiteral[[BM, BN]] = [bm, bn]
    wrong_a_block: IntListLiteral[[BM, Other]] = [bm, other]
    a_layout = gl.NVMMASharedLayout.get_default_for(a_block, gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for(b_block, gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for(c_block, gl.float32)
    wrong_a_layout = gl.NVMMASharedLayout.get_default_for(wrong_a_block, gl.float16)

    a_desc = TensorDescriptor.from_tensor(a, a_block, a_layout)
    b_desc = TensorDescriptor.from_tensor(b, b_block, b_layout)
    c_desc = TensorDescriptor.from_tensor(c, c_block, c_layout)
    wrong_a_desc = TensorDescriptor.from_tensor(wrong_a, a_block, a_layout)
    wrong_b_desc = TensorDescriptor.from_tensor(wrong_b, b_block, b_layout)
    wrong_c_desc = TensorDescriptor.from_tensor(wrong_c, c_block, c_layout)
    wrong_dtype_a_desc = TensorDescriptor.from_tensor(wrong_dtype_a, a_block, a_layout)
    output_ptr = output_pointer(d, row_stride, col_stride)
    wrong_output_ptr = output_pointer(wrong_d, row_stride, col_stride)
    TensorDescriptor.from_tensor(a, a_block, wrong_a_layout)  # E: No matching overload

    assert_type(a_desc, gl.WgmmaDescriptorF16[M, K, BM, BK, int])
    assert_type(c_desc, gl.WgmmaDescriptorF32[M, N, BM, BN, int])
    matmul_accumulate_kernel(
        a_desc,
        b_desc,
        c_desc,
        output_ptr,
        row_stride,
        col_stride,
        PersistentTileScheduler,
        depth,
    )
    matmul_accumulate_kernel(
        wrong_a_desc,
        b_desc,  # E: is not assignable
        c_desc,
        output_ptr,
        row_stride,
        col_stride,
        PersistentTileScheduler,
        depth,
    )
    matmul_accumulate_kernel(
        a_desc,
        wrong_b_desc,  # E: is not assignable
        c_desc,
        output_ptr,
        row_stride,
        col_stride,
        PersistentTileScheduler,
        depth,
    )
    matmul_accumulate_kernel(
        a_desc,
        b_desc,
        wrong_c_desc,  # E: is not assignable
        output_ptr,
        row_stride,
        col_stride,
        PersistentTileScheduler,
        depth,
    )
    matmul_accumulate_kernel(
        a_desc,
        b_desc,
        c_desc,
        wrong_output_ptr,  # E: is not assignable
        row_stride,
        col_stride,
        PersistentTileScheduler,
        depth,
    )
    # Element dtype is not carried by the shape-only torch.Tensor annotation.
    matmul_accumulate_kernel(
        wrong_dtype_a_desc,
        b_desc,
        c_desc,
        output_ptr,
        row_stride,
        col_stride,
        PersistentTileScheduler,
        depth,
    )

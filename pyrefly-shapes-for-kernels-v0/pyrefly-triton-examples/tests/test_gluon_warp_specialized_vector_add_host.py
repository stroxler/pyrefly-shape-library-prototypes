# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only direct typed kernel call, not the original bracketed JIT launch.
# @lint-ignore-every AUTODEPS2

"""Probe the declared Torch A/B/C descriptor boundary of warp-specialized add."""

import torch
from shape_extensions import Int, IntListLiteral, IntVar
from tests.test_gluon_warp_specialized_vector_add import (
    elementwise_add_warp_specialized_kernel,
)
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


def test_warp_specialized_direct_torch_boundary[
    Rows: IntVar,
    Cols: IntVar,
    BR: IntVar,
    BC: IntVar,
    Other: IntVar,
    LoadDepth: IntVar,
    StoreDepth: IntVar,
](
    rows: Int[Rows],
    cols: Int[Cols],
    br: Int[BR],
    bc: Int[BC],
    other: Int[Other],
    load_depth: Int[LoadDepth],
    store_depth: Int[StoreDepth],
) -> None:
    a = torch.empty((rows, cols), device="cuda", dtype=torch.float32)
    b = torch.empty((rows, cols), device="cuda", dtype=torch.float32)
    c = torch.empty((rows, cols), device="cuda", dtype=torch.float32)
    wrong_b = torch.empty((rows, other), device="cuda", dtype=torch.float32)
    wrong_c = torch.empty((rows, other), device="cuda", dtype=torch.float32)
    wrong_dtype = torch.empty((rows, cols), device="cuda", dtype=torch.float16)

    block_shape: IntListLiteral[[BR, BC]] = [br, bc]
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, gl.float32)
    a_desc = TensorDescriptor.from_tensor(a, block_shape, layout)
    b_desc = TensorDescriptor.from_tensor(b, block_shape, layout)
    c_desc = TensorDescriptor.from_tensor(c, block_shape, layout)
    wrong_b_desc = TensorDescriptor.from_tensor(wrong_b, block_shape, layout)
    wrong_c_desc = TensorDescriptor.from_tensor(wrong_c, block_shape, layout)
    wrong_dtype_desc = TensorDescriptor.from_tensor(wrong_dtype, block_shape, layout)

    elementwise_add_warp_specialized_kernel(
        a_desc, b_desc, c_desc, rows, cols, br, bc, load_depth, store_depth, 4
    )
    elementwise_add_warp_specialized_kernel(
        a_desc,
        wrong_b_desc,  # E: is not assignable
        c_desc,
        rows,
        cols,
        br,
        bc,
        load_depth,
        store_depth,
        4,
    )
    elementwise_add_warp_specialized_kernel(
        a_desc,
        b_desc,
        wrong_c_desc,  # E: is not assignable
        rows,
        cols,
        br,
        bc,
        load_depth,
        store_depth,
        4,
    )
    # Tensor-shape stubs do not prove the backing Torch element dtype.
    elementwise_add_warp_specialized_kernel(
        wrong_dtype_desc,
        b_desc,
        c_desc,
        rows,
        cols,
        br,
        bc,
        load_depth,
        store_depth,
        4,
    )

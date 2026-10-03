# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only original Python wrapper; the JIT bracket launch remains untyped.
# @lint-ignore-every AUTODEPS2

"""Check the source vector-add wrapper's declared Torch boundary."""

import torch
import triton
from shape_extensions import Int, IntVar
from tests.test_gluon_warp_specialized_vector_add import (
    elementwise_add_warp_specialized_kernel,
)
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


def elementwise_add_warp_specialized[
    Rows: IntVar,
    Cols: IntVar,
    BR: IntVar,
    BC: IntVar,
    LoadDepth: IntVar,
    StoreDepth: IntVar,
](
    a: torch.Tensor[[Rows, Cols]],
    b: torch.Tensor[[Rows, Cols]],
    c: torch.Tensor[[Rows, Cols]],
    XBLOCK: Int[BR] = 32,
    YBLOCK: Int[BC] = 64,
    num_load_buffers: Int[LoadDepth] = 2,
    num_store_buffers: Int[StoreDepth] = 2,
    num_warps: int = 4,
):
    xnumel, ynumel = a.shape
    grid = (triton.cdiv(xnumel, XBLOCK),)

    block_shape = [XBLOCK, YBLOCK]
    layout = gl.NVMMASharedLayout.get_default_for(  # E: No matching overload
        block_shape, gl.float32
    )
    a_desc = TensorDescriptor.from_tensor(  # E: No matching overload
        a, block_shape, layout
    )
    b_desc = TensorDescriptor.from_tensor(  # E: No matching overload
        b, block_shape, layout
    )
    c_desc = TensorDescriptor.from_tensor(  # E: No matching overload
        c, block_shape, layout
    )

    # By default, a warp-specialized kernel assumes maxnreg=256, the maximum
    # allowed per thread, in order to determine how to reallocate registers.
    # We need to intentionally set the register limit. Since the kernel will
    # have `num_warps+4` warps total, register usage will be
    #
    #     maxnreg * (num_warps+4) * 32
    #
    # Keep this in mind when deciding how much occupancy you want.
    elementwise_add_warp_specialized_kernel[grid](  # E: is not subscriptable
        a_desc,
        b_desc,
        c_desc,
        xnumel,
        ynumel,  #
        XBLOCK,
        YBLOCK,
        num_load_buffers,
        num_store_buffers,  #
        num_warps=num_warps,
        maxnreg=128,
    )


def test_typed_wrapper_caller[
    Rows: IntVar,
    Cols: IntVar,
    Other: IntVar,
    BR: IntVar,
    BC: IntVar,
    LoadDepth: IntVar,
    StoreDepth: IntVar,
](
    rows: Int[Rows],
    cols: Int[Cols],
    other: Int[Other],
    br: Int[BR],
    bc: Int[BC],
    load_depth: Int[LoadDepth],
    store_depth: Int[StoreDepth],
) -> None:
    a = torch.empty((rows, cols), device="cuda", dtype=torch.float32)
    b = torch.empty((rows, cols), device="cuda", dtype=torch.float32)
    c = torch.empty((rows, cols), device="cuda", dtype=torch.float32)
    wrong_b = torch.empty((rows, other), device="cuda", dtype=torch.float32)
    wrong_c = torch.empty((rows, other), device="cuda", dtype=torch.float32)
    elementwise_add_warp_specialized(a, b, c, br, bc, load_depth, store_depth, 4)
    elementwise_add_warp_specialized(
        a,
        wrong_b,  # E: is not assignable
        c,
        br,
        bc,
        load_depth,
        store_depth,
        4,
    )
    elementwise_add_warp_specialized(
        a,
        b,
        wrong_c,  # E: is not assignable
        br,
        bc,
        load_depth,
        store_depth,
        4,
    )

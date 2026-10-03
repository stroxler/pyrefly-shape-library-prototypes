# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only Torch bridge for a worker, not a host-launched warp-specialized kernel.
# @lint-ignore-every AUTODEPS2

"""Check the worker's declared input shapes against Torch descriptors."""

import torch
from shape_extensions import Int, IntListLiteral, IntVar
from tests.test_gluon_warp_specialized_load_partition import load_partition
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


def test_load_partition_torch_bridge[
    Rows: IntVar,
    Cols: IntVar,
    Other: IntVar,
    BR: IntVar,
    BC: IntVar,
    LoadDepth: IntVar,
    StoreDepth: IntVar,
    LA: IntVar,
    LB: IntVar,
    LC: IntVar,
](
    rows: Int[Rows],
    cols: Int[Cols],
    other: Int[Other],
    block_rows: Int[BR],
    block_cols: Int[BC],
    xoff: gl.GluonTileStart[BR],
    load_empty_bars: gl.TmaBarrierRing2D[LoadDepth],
    load_ready_bars: gl.TmaBarrierRing2D[LoadDepth],
    c_empty_bars: gl.TmaBarrierRing2D[StoreDepth],
    c_ready_bars: gl.TmaBarrierRing2D[StoreDepth],
    a_bufs: gl.TmaSharedRing2D[LoadDepth, BR, BC, LA],
    b_bufs: gl.TmaSharedRing2D[LoadDepth, BR, BC, LB],
    c_bufs: gl.TmaSharedRing2D[StoreDepth, BR, BC, LC],
) -> None:
    a = torch.empty((rows, cols), device="cuda", dtype=torch.float32)
    b = torch.empty((rows, cols), device="cuda", dtype=torch.float32)
    wrong_b = torch.empty((rows, other), device="cuda", dtype=torch.float32)
    c = torch.empty((rows, cols), device="cuda", dtype=torch.float32)
    wrong_c = torch.empty((rows, other), device="cuda", dtype=torch.float32)
    wrong_a_dtype = torch.empty((rows, cols), device="cuda", dtype=torch.float16)

    block_shape: IntListLiteral[[BR, BC]] = [block_rows, block_cols]
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, gl.float32)
    a_desc = TensorDescriptor.from_tensor(a, block_shape, layout)
    b_desc = TensorDescriptor.from_tensor(b, block_shape, layout)
    wrong_b_desc = TensorDescriptor.from_tensor(wrong_b, block_shape, layout)
    c_desc = TensorDescriptor.from_tensor(c, block_shape, layout)
    wrong_c_desc = TensorDescriptor.from_tensor(wrong_c, block_shape, layout)
    wrong_a_desc = TensorDescriptor.from_tensor(wrong_a_dtype, block_shape, layout)

    barriers = (load_empty_bars, load_ready_bars, c_empty_bars, c_ready_bars)
    buffers = (a_bufs, b_bufs, c_bufs)
    load_partition(
        (a_desc, b_desc, c_desc), barriers, buffers, xoff, (rows, cols), block_cols
    )
    load_partition(
        (a_desc, wrong_b_desc, c_desc),  # E: is not assignable
        barriers,
        buffers,
        xoff,
        (rows, cols),
        block_cols,
    )
    # This worker never reads C, so a wrong C extent still passes here.
    load_partition(
        (a_desc, b_desc, wrong_c_desc),
        barriers,
        buffers,
        xoff,
        (rows, cols),
        block_cols,
    )
    # Shape-only Torch tracking permits a wrong element dtype.
    load_partition(
        (wrong_a_desc, b_desc, c_desc),
        barriers,
        buffers,
        xoff,
        (rows, cols),
        block_cols,
    )

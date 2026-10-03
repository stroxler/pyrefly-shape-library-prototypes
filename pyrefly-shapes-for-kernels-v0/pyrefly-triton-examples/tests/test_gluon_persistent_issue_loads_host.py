# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only descriptor bridge to a JIT helper, not an executable Python launch.
# @lint-ignore-every AUTODEPS2

"""Check declared Torch dimensions on the first persistent-matmul load helper."""

import torch
from shape_extensions import Int, IntListLiteral, IntVar
from tests.test_gluon_persistent_issue_loads import issue_loads
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


def test_declared_torch_load_boundary[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    OtherK: IntVar,
    Depth: IntVar,
    LA: IntVar,
    LB: IntVar,
](
    m: Int[M],
    n: Int[N],
    k: Int[K],
    bm: Int[BM],
    bn: Int[BN],
    bk: Int[BK],
    other_k: Int[OtherK],
    num_buffers: Int[Depth],
    off_m: gl.GluonTileStart[BM],
    off_n: gl.GluonTileStart[BN],
    bars: gl.TmaBarrierRing2D[Depth],
    a_bufs: gl.WgmmaSharedRingF16[BM, BK, LA],
    b_bufs: gl.WgmmaSharedRingF16[BK, BN, LB],
) -> None:
    a = torch.empty((m, k), device="cuda", dtype=torch.float16)
    b = torch.empty((k, n), device="cuda", dtype=torch.float16)
    wrong_b = torch.empty((other_k, n), device="cuda", dtype=torch.float16)
    wrong_a_dtype = torch.empty((m, k), device="cuda", dtype=torch.float32)

    a_block: IntListLiteral[[BM, BK]] = [bm, bk]
    b_block: IntListLiteral[[BK, BN]] = [bk, bn]
    a_layout = gl.NVMMASharedLayout.get_default_for(a_block, gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for(b_block, gl.float16)
    a_desc = TensorDescriptor.from_tensor(a, a_block, a_layout)
    b_desc = TensorDescriptor.from_tensor(b, b_block, b_layout)
    wrong_b_desc = TensorDescriptor.from_tensor(wrong_b, b_block, b_layout)
    wrong_a_dtype_desc = TensorDescriptor.from_tensor(wrong_a_dtype, a_block, a_layout)

    issue_loads(0, a_desc, b_desc, off_m, off_n, 0, bars, a_bufs, b_bufs, num_buffers)
    issue_loads(
        0,
        a_desc,
        wrong_b_desc,  # E: is not assignable
        off_m,
        off_n,
        0,
        bars,
        a_bufs,
        b_bufs,
        num_buffers,
    )
    # Shape-only Torch types accept float32 backing for nominally float16 TMA.
    issue_loads(
        0,
        wrong_a_dtype_desc,
        b_desc,
        off_m,
        off_n,
        0,
        bars,
        a_bufs,
        b_bufs,
        num_buffers,
    )

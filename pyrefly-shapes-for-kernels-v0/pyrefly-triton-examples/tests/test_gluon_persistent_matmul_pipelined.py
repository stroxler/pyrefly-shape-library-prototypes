# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; semantic parameters are not executable Gluon kernel annotations.
# @lint-ignore-every AUTODEPS2

"""Check the source-order host-launched persistent matmul kernel's original body."""

from shape_extensions import Int, IntVar
from tests.test_gluon_persistent_issue_loads import issue_loads
from tests.test_gluon_persistent_issue_mma import issue_mma, MMAv5, WGMMA
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
)


@gluon.jit
def matmul_pipelined_kernel[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    LA: IntVar,
    LB: IntVar,
    LC: IntVar,
    Depth: IntVar,
](
    a_desc: gl.TmaInputDescriptorF16[M, K, BM, BK, LA],
    b_desc: gl.TmaInputDescriptorF16[K, N, BK, BN, LB],
    c_desc: gl.TmaOutputDescriptorF16[M, N, BM, BN, LC],
    MMAImpl: type[WGMMA] | type[MMAv5],
    num_buffers: Int[Depth],
    num_warps: int,
):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    gl.static_assert(
        num_buffers >= 2,  # E: is not assignable
        "expected at least 2 buffers",
    )
    a_bufs = gl.allocate_shared_memory(
        dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout
    )
    b_bufs = gl.allocate_shared_memory(
        dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout
    )
    bars = gl.allocate_shared_memory(
        gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout()
    )
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)
    # Separate producer and consumer indices, to support more than 2 buffers.
    producer = 0
    consumer = 0

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    # Use our MMA abstraction!
    mma = MMAImpl.initialize(dtype, BLOCK_M, BLOCK_N, num_warps)

    # Prefetch at most num_buffers-2 loads to allow the MMA to overlap.
    for k in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
        producer = issue_loads(
            producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, num_buffers
        )

    for k in range(BLOCK_K * (num_buffers - 2), K, BLOCK_K):
        producer = issue_loads(
            producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, num_buffers
        )
        consumer, mma = issue_mma(consumer, mma, bars, a_bufs, b_bufs, num_buffers)

    for _ in gl.static_range(num_buffers - 2):
        consumer, mma = issue_mma(consumer, mma, bars, a_bufs, b_bufs, num_buffers)

    mma = mma.wait_num_outstanding(0)
    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    c, mma = mma.take_result()
    c_smem.store(c.to(dtype))  # E: has no attribute `to`
    fence_async_shared()
    tma.async_store(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


def test_kernel_declared_tile_contract[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Other: IntVar,
    LA: IntVar,
    LB: IntVar,
    LC: IntVar,
    Depth: IntVar,
](
    a_desc: gl.TmaInputDescriptorF16[M, K, BM, BK, LA],
    b_desc: gl.TmaInputDescriptorF16[K, N, BK, BN, LB],
    wrong_b_k: gl.TmaInputDescriptorF16[K, N, Other, BN, LB],
    c_desc: gl.TmaOutputDescriptorF16[M, N, BM, BN, LC],
    wrong_c_block_n: gl.TmaOutputDescriptorF16[M, N, BM, Other, LC],
    depth: Int[Depth],
) -> None:
    matmul_pipelined_kernel(a_desc, b_desc, c_desc, WGMMA, depth, 8)
    matmul_pipelined_kernel(a_desc, b_desc, c_desc, MMAv5, depth, 4)
    matmul_pipelined_kernel(
        a_desc,
        wrong_b_k,  # E: is not assignable
        c_desc,
        WGMMA,
        depth,
        8,
    )
    matmul_pipelined_kernel(
        a_desc,
        b_desc,
        wrong_c_block_n,  # E: is not assignable
        MMAv5,
        depth,
        4,
    )

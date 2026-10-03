# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; semantic parameter annotations are not executable Gluon annotations.
# @lint-ignore-every AUTODEPS2

"""Check the original persistent pipelined matmul kernel's full body."""

from shape_extensions import Int, IntVar
from tests.test_gluon_persistent_issue_loads_stealb import issue_loads_stealb
from tests.test_gluon_persistent_issue_mma import MMAv5, WGMMA
from tests.test_gluon_persistent_issue_mma_stealb import issue_mma_stealb
from tests.test_gluon_persistent_matmul import PersistentTileScheduler
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
)


@gluon.jit
def persistent_matmul_pipelined_kernel[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    LA: IntVar,
    LB: IntVar,
    LC: IntVar,
    LH: IntVar,
    Depth: IntVar,
](
    a_desc: gl.TmaInputDescriptorF16[M, K, BM, BK, LA],
    b_desc: gl.TmaInputDescriptorF16[K, N, BK, BN, LB],
    c_desc: gl.TmaOutputDescriptorF16[M, N, BM, BN, LC],
    c_half_desc: gl.TmaOutputDescriptorF16[M, N, BM, BN // 2, LH],
    MMAImpl: type[WGMMA] | type[MMAv5],
    SchedulerImpl: type[PersistentTileScheduler],
    num_buffers: Int[Depth],
    STEALB: bool,
    num_warps: int,
):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    # All buffers share the same liverange.
    gl.static_assert(
        num_buffers >= 3,  # E: is not assignable
        "expected at least 3 buffers",
    )
    a_bufs = gl.allocate_shared_memory(
        dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout
    )
    # Add an extra B buffer when stealing.
    b_bufs = gl.allocate_shared_memory(
        dtype, [num_buffers + STEALB] + b_desc.block_type.shape, b_desc.layout
    )
    if not STEALB:
        c_smem = gl.allocate_shared_memory(
            dtype, c_desc.block_type.shape, c_desc.layout
        )
    else:
        gl.static_assert(
            BLOCK_M == BLOCK_K or BLOCK_M == 2 * BLOCK_K,  # E: is not assignable
            "expected one or two B tiles to cover the epilogue tile",
        )
    bars = gl.allocate_shared_memory(
        gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout()
    )
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)
    producer = 0
    consumer = 0

    mma = MMAImpl.initialize(dtype, BLOCK_M, BLOCK_N, num_warps)
    scheduler = SchedulerImpl.initialize(
        c_desc.shape[0], c_desc.shape[1], BLOCK_M, BLOCK_N
    )
    num_tiles = scheduler.get_num_tiles()

    # Peeled inner loop prologue.
    idx = 0
    pid_m, pid_n = scheduler.get_tile(idx)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
        producer = issue_loads_stealb(
            producer,
            a_desc,
            b_desc,
            off_m,
            off_n,
            ki,
            bars,
            a_bufs,
            b_bufs,
            STEALB,
            num_buffers,
        )
    k = BLOCK_K * (num_buffers - 2)
    producer = issue_loads_stealb(
        producer,
        a_desc,
        b_desc,
        off_m,
        off_n,
        k,
        bars,
        a_bufs,
        b_bufs,
        STEALB,
        num_buffers,
    )

    for _ in range(num_tiles):  # E: is not assignable
        consumer, mma = issue_mma_stealb(
            consumer, mma, bars, a_bufs, b_bufs, STEALB, num_buffers
        )
        if STEALB:
            # Wait for the epilogue before the first TMA load.
            tma.store_wait(pendings=0)
        for k in range(BLOCK_K * (num_buffers - 1), K, BLOCK_K):
            producer = issue_loads_stealb(
                producer,
                a_desc,
                b_desc,
                off_m,
                off_n,
                k,
                bars,
                a_bufs,
                b_bufs,
                STEALB,
                num_buffers,
            )
            consumer, mma = issue_mma_stealb(
                consumer, mma, bars, a_bufs, b_bufs, STEALB, num_buffers
            )

        epilogue_off_m = off_m
        epilogue_off_n = off_n

        # Peel the next prologue and fuse it with the pipeline drain loop.
        idx += 1
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N
        # Predicate the peeled prologue instead of using a conditional.
        pred = idx < num_tiles
        for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
            producer = issue_loads_stealb(
                producer,
                a_desc,
                b_desc,
                off_m,
                off_n,
                ki,
                bars,
                a_bufs,
                b_bufs,
                STEALB,
                num_buffers,
                pred,  # E: is not assignable
            )
            consumer, mma = issue_mma_stealb(
                consumer, mma, bars, a_bufs, b_bufs, STEALB, num_buffers
            )
        k = BLOCK_K * (num_buffers - 2)
        producer = issue_loads_stealb(
            producer,
            a_desc,
            b_desc,
            off_m,
            off_n,
            k,
            bars,
            a_bufs,
            b_bufs,
            STEALB,
            num_buffers,
        )

        mma = mma.wait_num_outstanding(0)
        use_split_n_load: gl.constexpr = STEALB and BLOCK_M != BLOCK_K
        c, mma = mma.take_result(splitn=use_split_n_load)
        c = c.to(dtype)  # E: has no attribute `to`
        if not STEALB:
            c_buf = c_smem  # E: may be uninitialized
            tma.store_wait(pendings=0)
            c_buf.store(c)
            fence_async_shared()
            tma.async_store(c_desc, [epilogue_off_m, epilogue_off_n], c_buf)
        elif BLOCK_M == BLOCK_K:
            c_buf = b_bufs.index(producer % (num_buffers + STEALB))
            c_buf.store(c)
            fence_async_shared()
            tma.async_store(  # E: No matching overload
                c_desc, [epilogue_off_m, epilogue_off_n], c_buf
            )
        else:
            # Steal the next 2 B buffers for the epilogue.
            c0, c1 = c.reshape((BLOCK_M, 2, BLOCK_N // 2)).permute(0, 2, 1).split()
            c0_buf = b_bufs.index(  # E: has no attribute `reinterpret`
                producer % (num_buffers + STEALB)
            ).reinterpret(
                shape=c_half_desc.block_type.shape,
                layout=c_half_desc.layout,
            )
            c1_buf = b_bufs.index(  # E: has no attribute `reinterpret`
                (producer + 1) % (num_buffers + STEALB)
            ).reinterpret(shape=c_half_desc.block_type.shape, layout=c_half_desc.layout)
            c0_buf.store(c0)
            c1_buf.store(c1)
            fence_async_shared()
            tma.async_store(c_half_desc, [epilogue_off_m, epilogue_off_n], c0_buf)
            tma.async_store(
                c_half_desc, [epilogue_off_m, epilogue_off_n + BLOCK_N // 2], c1_buf
            )
    tma.store_wait(pendings=0)


def test_declared_pipelined_output_descriptors[
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
    LH: IntVar,
    Depth: IntVar,
](
    a_desc: gl.TmaInputDescriptorF16[M, K, BM, BK, LA],
    b_desc: gl.TmaInputDescriptorF16[K, N, BK, BN, LB],
    wrong_b_k: gl.TmaInputDescriptorF16[K, N, Other, BN, LB],
    c_desc: gl.TmaOutputDescriptorF16[M, N, BM, BN, LC],
    c_half_desc: gl.TmaOutputDescriptorF16[M, N, BM, BN // 2, LH],
    wrong_half_host_n: gl.TmaOutputDescriptorF16[M, Other, BM, BN // 2, LH],
    wrong_half_block: gl.TmaOutputDescriptorF16[M, N, BM, Other, LH],
    depth: Int[Depth],
) -> None:
    persistent_matmul_pipelined_kernel(
        a_desc,
        b_desc,
        c_desc,
        c_half_desc,
        WGMMA,
        PersistentTileScheduler,
        depth,
        False,
        8,
    )
    persistent_matmul_pipelined_kernel(
        a_desc,
        b_desc,
        c_desc,
        c_half_desc,
        MMAv5,
        PersistentTileScheduler,
        depth,
        True,
        4,
    )
    persistent_matmul_pipelined_kernel(
        a_desc,
        wrong_b_k,  # E: is not assignable
        c_desc,
        c_half_desc,
        WGMMA,
        PersistentTileScheduler,
        depth,
        False,
        8,
    )
    persistent_matmul_pipelined_kernel(
        a_desc,
        b_desc,
        c_desc,
        wrong_half_host_n,  # E: is not assignable
        MMAv5,
        PersistentTileScheduler,
        depth,
        True,
        4,
    )
    persistent_matmul_pipelined_kernel(
        a_desc,
        b_desc,
        c_desc,
        wrong_half_block,  # E: is not assignable
        WGMMA,
        PersistentTileScheduler,
        depth,
        True,
        8,
    )

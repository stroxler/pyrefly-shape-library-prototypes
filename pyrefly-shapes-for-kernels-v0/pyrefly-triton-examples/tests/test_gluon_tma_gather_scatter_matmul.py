# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; semantic parameter annotations are not executable Gluon annotations.
# @lint-ignore-every AUTODEPS2

"""Check the original fused gather/scatter matmul entrypoint."""

from shape_extensions import Int, IntVar
from tests import test_gluon_persistent_issue_mma as t7
from tests.test_gluon_persistent_matmul import PersistentTileScheduler
from tests.test_gluon_tma_gather_scatter_issue_loads import issue_loads
from tests.test_gluon_tma_gather_scatter_issue_mma import issue_mma
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    fence_async_shared,
    mbarrier,
    tma,
)


@gluon.jit
def matmul_fused_gather_scatter_kernel[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    LX: IntVar,
    LW: IntVar,
    LO: IntVar,
    Depth: IntVar,
](
    X_desc: gl.GatherInputDescriptorF16[M, K, BM, BK, LX],
    W_desc: gl.TmaInputDescriptorF16[K, N, BK, BN, LW],
    out_desc: gl.ScatterOutputDescriptorF16[M, N, BM, BN, LO],
    X_gather_indx_ptr: gl.GatherOffsetsPointer1D[M],
    out_scatter_indx_ptr: gl.GatherOffsetsPointer1D[M],
    BLOCK_M: Int[BM],
    SchedulerImpl: type[PersistentTileScheduler],
    num_buffers: Int[Depth],
):
    BLOCK_N: gl.constexpr = W_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = W_desc.block_type.shape[0]
    dtype: gl.constexpr = X_desc.dtype
    M = X_desc.shape[0]
    N = W_desc.shape[1]
    K = X_desc.shape[1]

    # Allocate shared memory for the input tiles.
    x_bufs = gl.allocate_shared_memory(
        dtype, [num_buffers, BLOCK_M, BLOCK_K], X_desc.layout
    )
    w_bufs = gl.allocate_shared_memory(
        dtype, [num_buffers, BLOCK_K, BLOCK_N], W_desc.layout
    )

    # Allocate shared memory for the output tile.
    out_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, BLOCK_N], out_desc.layout)

    # Initialize barriers for multibuffering the loads.
    bars = gl.allocate_shared_memory(
        gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout()
    )
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)
    producer = 0
    consumer = 0

    mma = t7.MMAv5.initialize(dtype, BLOCK_M, BLOCK_N, gl.num_warps())
    scheduler = SchedulerImpl.initialize(M, N, BLOCK_M, BLOCK_N)
    num_tiles = scheduler.get_num_tiles()

    # Peeled inner loop prologue.
    idx = 0
    pid_m, pid_n = scheduler.get_tile(idx)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
        producer = issue_loads(
            producer,
            X_desc,
            W_desc,
            X_gather_indx_ptr,
            off_m,
            off_n,
            ki,
            bars,
            x_bufs,
            w_bufs,
            BLOCK_M,
            num_buffers,
        )
    k = BLOCK_K * (num_buffers - 2)
    producer = issue_loads(
        producer,
        X_desc,
        W_desc,
        X_gather_indx_ptr,
        off_m,
        off_n,
        k,
        bars,
        x_bufs,
        w_bufs,
        BLOCK_M,
        num_buffers,
    )

    for _ in range(num_tiles):  # E: is not assignable
        consumer, mma = issue_mma(consumer, mma, bars, x_bufs, w_bufs, num_buffers)
        for k in range(BLOCK_K * (num_buffers - 1), K, BLOCK_K):
            producer = issue_loads(
                producer,
                X_desc,
                W_desc,
                X_gather_indx_ptr,
                off_m,
                off_n,
                k,
                bars,
                x_bufs,
                w_bufs,
                BLOCK_M,
                num_buffers,
            )
            consumer, mma = issue_mma(consumer, mma, bars, x_bufs, w_bufs, num_buffers)

        epilogue_off_m = off_m
        epilogue_off_n = off_n

        # Load the M dimension offsets for the output tile. We expect the load to be small
        # enough (no more than 128 elements) that we don't need to use a coalesced layout.
        # Load directly into the layout required by `async_scatter` to avoid the layout conversion.
        scatter_indx_layout: gl.constexpr = gl.SliceLayout(
            0, gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0])
        )
        out_offs_m = gl.load(
            out_scatter_indx_ptr
            + epilogue_off_m
            + gl.arange(0, BLOCK_M, scatter_indx_layout)
        )

        # Peel the next prologue and fuse it with the pipeline drain loop.
        idx += 1
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N

        # Predicate the peeled prologue instead of using a conditional.
        pred = idx < num_tiles
        for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
            producer = issue_loads(
                producer,
                X_desc,
                W_desc,
                X_gather_indx_ptr,
                off_m,
                off_n,
                ki,
                bars,
                x_bufs,
                w_bufs,
                BLOCK_M,
                num_buffers,
                pred,
            )
            consumer, mma = issue_mma(consumer, mma, bars, x_bufs, w_bufs, num_buffers)
        k = BLOCK_K * (num_buffers - 2)
        producer = issue_loads(
            producer,
            X_desc,
            W_desc,
            X_gather_indx_ptr,
            off_m,
            off_n,
            k,
            bars,
            x_bufs,
            w_bufs,
            BLOCK_M,
            num_buffers,
        )

        mma = mma.wait_num_outstanding(0)
        out, mma = mma.take_result()
        out = out.to(dtype)  # E: has no attribute `to`
        # Pipeline the async scatter by waiting for the previous scatter to complete.
        tma.store_wait(pendings=0)
        out_smem.store(out)
        fence_async_shared()
        tma.async_scatter(out_desc, out_offs_m, epilogue_off_n, out_smem)
    # Wait for the last async scatter to complete.
    tma.store_wait(pendings=0)


def test_declared_kernel_interface[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Other: IntVar,
    LX: IntVar,
    LW: IntVar,
    LO: IntVar,
    Depth: IntVar,
](
    x: gl.GatherInputDescriptorF16[M, K, BM, BK, LX],
    w: gl.TmaInputDescriptorF16[K, N, BK, BN, LW],
    wrong_w_k: gl.TmaInputDescriptorF16[Other, N, BK, BN, LW],
    output: gl.ScatterOutputDescriptorF16[M, N, BM, BN, LO],
    wrong_output_n: gl.ScatterOutputDescriptorF16[M, Other, BM, BN, LO],
    wrong_output_tile: gl.ScatterOutputDescriptorF16[M, N, BM, Other, LO],
    gather_ptr: gl.GatherOffsetsPointer1D[M],
    wrong_gather_ptr: gl.GatherOffsetsPointer1D[Other],
    scatter_ptr: gl.GatherOffsetsPointer1D[M],
    wrong_scatter_ptr: gl.GatherOffsetsPointer1D[Other],
    block_m: Int[BM],
    depth: Int[Depth],
) -> None:
    matmul_fused_gather_scatter_kernel(
        x, w, output, gather_ptr, scatter_ptr, block_m, PersistentTileScheduler, depth
    )
    matmul_fused_gather_scatter_kernel(
        x,
        wrong_w_k,  # E: is not assignable
        output,
        gather_ptr,
        scatter_ptr,
        block_m,
        PersistentTileScheduler,
        depth,
    )
    matmul_fused_gather_scatter_kernel(
        x,
        w,
        wrong_output_n,  # E: is not assignable
        gather_ptr,
        scatter_ptr,
        block_m,
        PersistentTileScheduler,
        depth,
    )
    matmul_fused_gather_scatter_kernel(
        x,
        w,
        wrong_output_tile,  # E: is not assignable
        gather_ptr,
        scatter_ptr,
        block_m,
        PersistentTileScheduler,
        depth,
    )
    matmul_fused_gather_scatter_kernel(
        x,
        w,
        output,
        wrong_gather_ptr,  # E: is not assignable
        scatter_ptr,
        block_m,
        PersistentTileScheduler,
        depth,
    )
    matmul_fused_gather_scatter_kernel(
        x,
        w,
        output,
        gather_ptr,
        wrong_scatter_ptr,  # E: is not assignable
        block_m,
        PersistentTileScheduler,
        depth,
    )

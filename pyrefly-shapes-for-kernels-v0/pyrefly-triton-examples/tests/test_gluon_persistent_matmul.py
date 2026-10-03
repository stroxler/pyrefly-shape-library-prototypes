# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; Gluon aggregate and kernel semantic annotations are non-executable.
# @lint-ignore-every AUTODEPS2

"""Check the source-order persistent scheduler and matmul kernel bodies."""

from typing import assert_type

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


@gluon.aggregate
class PersistentTileScheduler:
    pid_start: gl.tensor
    pid_end: gl.tensor
    num_pid_m: gl.tensor

    @gluon.jit
    def initialize[M: IntVar, N: IntVar, BM: IntVar, BN: IntVar](
        M: Int[M],  # noqa: B902 - Gluon aggregate initializer has no self.
        N: Int[N],
        BLOCK_M: Int[BM],
        BLOCK_N: Int[BN],
    ):
        kernel_id = gl.program_id(axis=0)
        num_kernels = gl.num_programs(axis=0)
        num_pid_m = gl.cdiv(M, BLOCK_M)
        num_pid_n = gl.cdiv(N, BLOCK_N)
        num_pid = num_pid_m * num_pid_n
        pid_per_kernel = gl.cdiv(num_pid, num_kernels)
        pid_start = kernel_id * pid_per_kernel
        pid_end = min(pid_start + pid_per_kernel, num_pid)  # E: is not assignable
        return PersistentTileScheduler(
            pid_start,  # E: is not assignable
            pid_end,  # E: is not assignable
            num_pid_m,  # E: is not assignable
        )

    @gluon.jit
    def get_num_tiles(self):
        return self.pid_end - self.pid_start

    @gluon.jit
    def get_tile(self, idx: int):
        # Delinearize the tile ID along M.
        pid = self.pid_start + idx
        pid_m = pid % self.num_pid_m  # E: is not supported
        pid_n = pid // self.num_pid_m  # E: is not supported
        return pid_m, pid_n


@gluon.jit
def persistent_matmul_kernel[
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
    SchedulerImpl: type[PersistentTileScheduler],
    num_buffers: Int[Depth],
    num_warps: int,
):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    bars = gl.allocate_shared_memory(
        gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout()
    )
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)
    # Producer and consumer indices.
    producer = 0
    consumer = 0

    mma = MMAImpl.initialize(dtype, BLOCK_M, BLOCK_N, num_warps)
    scheduler = SchedulerImpl.initialize(
        c_desc.shape[0], c_desc.shape[1], BLOCK_M, BLOCK_N
    )
    for idx in range(scheduler.get_num_tiles()):  # E: is not assignable
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N

        a_bufs = gl.allocate_shared_memory(
            dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout
        )
        b_bufs = gl.allocate_shared_memory(
            dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout
        )
        for k in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
            producer = issue_loads(
                producer,
                a_desc,
                b_desc,
                off_m,
                off_n,
                k,
                bars,
                a_bufs,
                b_bufs,
                num_buffers,
            )

        for k in range(BLOCK_K * (num_buffers - 2), K, BLOCK_K):
            producer = issue_loads(
                producer,
                a_desc,
                b_desc,
                off_m,
                off_n,
                k,
                bars,
                a_bufs,
                b_bufs,
                num_buffers,
            )
            consumer, mma = issue_mma(consumer, mma, bars, a_bufs, b_bufs, num_buffers)

        for _ in gl.static_range(num_buffers - 2):
            consumer, mma = issue_mma(consumer, mma, bars, a_bufs, b_bufs, num_buffers)

        mma = mma.wait_num_outstanding(0)
        c_smem = gl.allocate_shared_memory(
            dtype, c_desc.block_type.shape, c_desc.layout
        )
        c, mma = mma.take_result()
        c_smem.store(c.to(dtype))  # E: has no attribute `to`
        fence_async_shared()
        tma.async_store(c_desc, [off_m, off_n], c_smem)
        tma.store_wait(pendings=0)


def test_scheduler_tile_coordinates_are_not_typed(
    scheduler: PersistentTileScheduler, idx: int
) -> None:
    assert_type(scheduler.get_tile(idx), tuple[int, int])  # E: failed


def test_runtime_block_does_not_prove_tile_size[BM: IntVar](
    runtime_block: int,
    expected_block: Int[BM],
) -> tuple[gl.GluonTileStart[BM], gl.GluonTileStart[BM]]:
    expected_tile: gl.GluonTileStart[BM] = gl.program_id(axis=0) * expected_block
    arbitrary_tile: gl.GluonTileStart[BM] = gl.program_id(axis=0) * runtime_block
    return expected_tile, arbitrary_tile


def test_persistent_kernel_declared_descriptor_contract[
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
    wrong_c_n: gl.TmaOutputDescriptorF16[M, N, BM, Other, LC],
    depth: Int[Depth],
) -> None:
    persistent_matmul_kernel(
        a_desc, b_desc, c_desc, WGMMA, PersistentTileScheduler, depth, 8
    )
    persistent_matmul_kernel(
        a_desc, b_desc, c_desc, MMAv5, PersistentTileScheduler, depth, 4
    )
    persistent_matmul_kernel(
        a_desc,
        wrong_b_k,  # E: is not assignable
        c_desc,
        WGMMA,
        PersistentTileScheduler,
        depth,
        8,
    )
    persistent_matmul_kernel(
        a_desc,
        b_desc,
        wrong_c_n,  # E: is not assignable
        MMAv5,
        PersistentTileScheduler,
        depth,
        4,
    )

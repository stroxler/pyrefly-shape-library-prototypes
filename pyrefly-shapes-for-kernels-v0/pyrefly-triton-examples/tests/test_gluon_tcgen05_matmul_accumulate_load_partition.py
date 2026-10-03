# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; semantic parameter annotations are not executable Gluon annotations.
# @lint-ignore-every AUTODEPS2

"""Check the original TCGen05 accumulate load worker against a declared interface."""

from typing import Protocol

from shape_extensions import IntVar
from tests.test_gluon_persistent_matmul import PersistentTileScheduler
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    mbarrier,
    tensor_memory_descriptor,
    tma,
)


@gluon.aggregate
class PartitionArgs:
    a_desc: tma.tensor_descriptor
    b_desc: tma.tensor_descriptor
    c_desc: tma.tensor_descriptor
    d_ptr: gl.tensor
    d_stride_m: gl.tensor
    d_stride_n: gl.tensor
    a_bufs: gl.shared_memory_descriptor
    b_bufs: gl.shared_memory_descriptor
    load_empty_bars: gl.shared_memory_descriptor
    load_ready_bars: gl.shared_memory_descriptor
    c_buf: gl.shared_memory_descriptor
    c_empty_bar: gl.shared_memory_descriptor
    c_ready_bar: gl.shared_memory_descriptor
    acc_bufs: tensor_memory_descriptor
    acc_empty_bars: gl.shared_memory_descriptor
    acc_ready_bars: gl.shared_memory_descriptor
    SchedulerImpl: gl.constexpr


@gluon.aggregate
class Counter:
    index: gl.tensor
    phase: gl.tensor
    num_barriers: gl.constexpr

    @gluon.jit
    def create(phase: int, num_barriers: int):  # noqa: B902  # E: self type
        return Counter(gl.to_tensor(0), gl.to_tensor(phase), num_barriers)

    @gluon.must_use_result
    @gluon.jit
    def next(self, pred=True):
        incr = self.index + gl.where(pred, 1, 0)  # E: No attribute `where`
        rollover = incr == self.num_barriers
        index = gl.where(rollover, 0, incr)  # E: No attribute `where`
        phase = gl.where(  # E: No attribute `where`
            rollover,
            self.phase ^ 1,  # E: is not supported
            self.phase,
        )
        return Counter(index, phase, self.num_barriers)


class t8:
    Counter = Counter


class LoadPartitionArgs[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BHostK: IntVar,
    AM: IntVar,
    AK: IntVar,
    BN: IntVar,
    BK: IntVar,
    CM: IntVar,
    CN: IntVar,
    LA: IntVar,
    LB: IntVar,
    LC: IntVar,
    Depth: IntVar,
    RingAM: IntVar,
    RingAK: IntVar,
    RingBK: IntVar,
    RingBN: IntVar,
    SharedCM: IntVar,
    SharedCN: IntVar,
](Protocol):
    a_desc: gl.TmaInputDescriptorF16[M, K, AM, AK, LA]
    b_desc: gl.TmaInputDescriptorF16[BHostK, N, BK, BN, LB]
    c_desc: gl.TmaInputDescriptor2D[M, N, CM, CN, LC]
    a_bufs: gl.WgmmaSharedRingF16[RingAM, RingAK, LA]
    b_bufs: gl.WgmmaSharedRingF16[RingBK, RingBN, LB]
    load_empty_bars: gl.TmaBarrierRing2D[Depth]
    load_ready_bars: gl.TmaBarrierRing2D[Depth]
    c_buf: gl.TmaSharedTile2D[SharedCM, SharedCN, LC]
    c_empty_bar: gl.BarrierBuffer1D
    c_ready_bar: gl.BarrierBuffer1D
    SchedulerImpl: type[PersistentTileScheduler]


@gluon.jit
def matmul_accumulate_load_partition[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    AK: IntVar,
    CM: IntVar,
    CN: IntVar,
    LA: IntVar,
    LB: IntVar,
    LC: IntVar,
    Depth: IntVar,
](
    p: LoadPartitionArgs[
        M,
        N,
        K,
        K,
        CM,
        AK,
        CN,
        AK,
        CM,
        CN,
        LA,
        LB,
        LC,
        Depth,
        CM,
        AK,
        AK,
        CN,
        CM,
        CN,
    ],
):
    BLOCK_M: gl.constexpr = p.c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = p.a_desc.block_type.shape[1]
    K = p.a_desc.shape[1]

    c_phase = 1
    state = t8.Counter.create(1, p.load_empty_bars.shape[0])
    scheduler = p.SchedulerImpl.initialize(
        p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N
    )
    for idx in range(scheduler.get_num_tiles()):  # E: is not assignable
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N
        # Issue the async TMA load for the C tile.
        mbarrier.wait(p.c_empty_bar, c_phase)
        mbarrier.expect(p.c_ready_bar, p.c_desc.block_type.nbytes)
        tma.async_load(p.c_desc, [off_m, off_n], p.c_ready_bar, p.c_buf)
        c_phase ^= 1
        # Inner loop loads.
        for k in range(0, K, BLOCK_K):
            bar = p.load_ready_bars.index(state.index)  # E: is not assignable
            mbarrier.wait(
                p.load_empty_bars.index(state.index),  # E: is not assignable
                state.phase,  # E: is not assignable
            )
            mbarrier.expect(
                bar, p.a_desc.block_type.nbytes + p.b_desc.block_type.nbytes
            )
            tma.async_load(
                p.a_desc,
                [off_m, k],
                bar,
                p.a_bufs.index(state.index),  # E: is not assignable
            )
            tma.async_load(
                p.b_desc,
                [k, off_n],
                bar,
                p.b_bufs.index(state.index),  # E: is not assignable
            )
            state = state.next()


type CanonicalLoadArgs[
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
] = LoadPartitionArgs[
    M,
    N,
    K,
    K,
    BM,
    BK,
    BN,
    BK,
    BM,
    BN,
    LA,
    LB,
    LC,
    Depth,
    BM,
    BK,
    BK,
    BN,
    BM,
    BN,
]


def test_load_partition_declared_interface[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    Other: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    LA: IntVar,
    LB: IntVar,
    LC: IntVar,
    Depth: IntVar,
](
    correct: CanonicalLoadArgs[M, N, K, BM, BN, BK, LA, LB, LC, Depth],
    wrong_b_host_k: LoadPartitionArgs[
        M,
        N,
        K,
        Other,
        BM,
        BK,
        BN,
        BK,
        BM,
        BN,
        LA,
        LB,
        LC,
        Depth,
        BM,
        BK,
        BK,
        BN,
        BM,
        BN,
    ],
    a_desc: gl.TmaInputDescriptorF16[M, K, BM, BK, LA],
    b_desc: gl.TmaInputDescriptorF16[K, N, BK, BN, LB],
    c_desc: gl.TmaInputDescriptor2D[M, N, BM, BN, LC],
    a_ring: gl.WgmmaSharedRingF16[BM, BK, LA],
    b_ring: gl.WgmmaSharedRingF16[BK, BN, LB],
    c_tile: gl.TmaSharedTile2D[BM, BN, LC],
    wrong_a_ring: gl.WgmmaSharedRingF16[BM, Other, LA],
    wrong_b_ring: gl.WgmmaSharedRingF16[Other, BN, LB],
    wrong_c_tile: gl.TmaSharedTile2D[BM, Other, LC],
    ready: gl.BarrierBuffer1D,
    source_aggregate: PartitionArgs,
) -> None:
    matmul_accumulate_load_partition(correct)
    matmul_accumulate_load_partition(wrong_b_host_k)  # E: is not assignable
    matmul_accumulate_load_partition(source_aggregate)  # E: is not assignable
    # These three calls check the consumer rule, not the original worker body.
    tma.async_load(a_desc, [0, 0], ready, a_ring.index(0))
    tma.async_load(a_desc, [0, 0], ready, wrong_a_ring.index(0))  # E: is not assignable
    tma.async_load(b_desc, [0, 0], ready, b_ring.index(0))
    tma.async_load(b_desc, [0, 0], ready, wrong_b_ring.index(0))  # E: is not assignable
    tma.async_load(c_desc, [0, 0], ready, c_tile)
    tma.async_load(c_desc, [0, 0], ready, wrong_c_tile)  # E: is not assignable

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; semantic parameter annotations are not executable Gluon annotations.
# @lint-ignore-every AUTODEPS2

"""Check the unchanged TCGen05 accumulate epilogue and its host contract."""

from typing import assert_type, Protocol

from shape_extensions import Int, IntVar
from tests.test_gluon_persistent_matmul import PersistentTileScheduler
from tests.test_gluon_tcgen05_matmul_accumulate_load_partition import (
    PartitionArgs,
    t8,
)
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    mbarrier,
    TensorMemoryRingF32,
)


class EpilogueScheduler:
    def get_num_tiles(self) -> gl.tensor: ...
    def get_tile(self, idx: int) -> tuple[gl.GluonTileId, gl.GluonTileId]: ...


class EpilogueSchedulerFactory:
    @staticmethod
    def initialize[M: IntVar, N: IntVar, BM: IntVar, BN: IntVar](
        rows: Int[M], cols: Int[N], block_rows: Int[BM], block_cols: Int[BN]
    ) -> EpilogueScheduler: ...


class EpiloguePartitionArgs[
    M: IntVar,
    N: IntVar,
    BM: IntVar,
    BN: IntVar,
    LC: IntVar,
    OutM: IntVar,
    OutN: IntVar,
    PtrRowStride: IntVar,
    PtrColStride: IntVar,
    ProvidedRowStride: IntVar,
    ProvidedColStride: IntVar,
    AccDepth: IntVar,
    AccM: IntVar,
    AccN: IntVar,
](Protocol):
    c_desc: gl.TmaInputDescriptor2D[M, N, BM, BN, LC]
    d_ptr: gl.OutMatrixPointer2D[OutM, OutN, PtrRowStride, PtrColStride]
    d_stride_m: Int[ProvidedRowStride]
    d_stride_n: Int[ProvidedColStride]
    acc_bufs: TensorMemoryRingF32[AccDepth, AccM, AccN]
    acc_empty_bars: gl.TmaBarrierRing2D[AccDepth]
    acc_ready_bars: gl.TmaBarrierRing2D[AccDepth]
    SchedulerImpl: type[EpilogueSchedulerFactory]


@gluon.jit
def matmul_accumulate_epilogue_partition[
    M: IntVar,
    N: IntVar,
    BM: IntVar,
    BN: IntVar,
    LC: IntVar,
    RowStride: IntVar,
    ColStride: IntVar,
    AccDepth: IntVar,
](
    p: EpiloguePartitionArgs[
        M,
        N,
        BM,
        BN,
        LC,
        M,
        N,
        RowStride,
        ColStride,
        RowStride,
        ColStride,
        AccDepth,
        BM,
        BN,
    ],
):
    BLOCK_M: gl.constexpr = p.c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.c_desc.block_type.shape[1]

    coalesced_2d_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 32], [1, gl.num_warps()], [1, 0]
    )
    range_m = gl.arange(0, BLOCK_M, gl.SliceLayout(1, coalesced_2d_layout))
    range_n = gl.arange(0, BLOCK_N, gl.SliceLayout(0, coalesced_2d_layout))

    acc_state = t8.Counter.create(0, p.acc_empty_bars.shape[0])
    scheduler = p.SchedulerImpl.initialize(
        p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N
    )
    for idx in range(scheduler.get_num_tiles()):  # E: is not assignable
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N
        # Wait for the accumulator.
        mbarrier.wait(
            p.acc_ready_bars.index(acc_state.index),  # E: is not assignable
            acc_state.phase,  # E: is not assignable
        )
        acc = p.acc_bufs.index(acc_state.index).load()  # E: is not assignable
        mbarrier.arrive(
            p.acc_empty_bars.index(acc_state.index),  # E: is not assignable
            count=1,
        )
        acc_state = acc_state.next()
        offs_m = off_m + range_m
        offs_n = off_n + range_n
        # This `convert_layout` is fairly expensive and it uses a lot of shared
        # memory, because the default TMEM register layout assigns contiguous
        # columns to the same thread, but the coalesced layout assigns
        # contiguous columns to different threads for efficient global writes.
        # We could subtile the store to reduce the shared memory usage.
        acc = gl.convert_layout(acc, coalesced_2d_layout)
        gl.store(
            p.d_ptr + offs_m[:, None] * p.d_stride_m + offs_n[None, :] * p.d_stride_n,
            acc,
        )


def test_epilogue_declared_interface[
    M: IntVar,
    N: IntVar,
    Other: IntVar,
    BM: IntVar,
    BN: IntVar,
    LC: IntVar,
    RowStride: IntVar,
    ColStride: IntVar,
    AccDepth: IntVar,
](
    correct: EpiloguePartitionArgs[
        M,
        N,
        BM,
        BN,
        LC,
        M,
        N,
        RowStride,
        ColStride,
        RowStride,
        ColStride,
        AccDepth,
        BM,
        BN,
    ],
    wrong_output_host_n: EpiloguePartitionArgs[
        M,
        N,
        BM,
        BN,
        LC,
        M,
        Other,
        RowStride,
        ColStride,
        RowStride,
        ColStride,
        AccDepth,
        BM,
        BN,
    ],
    wrong_row_stride: EpiloguePartitionArgs[
        M,
        N,
        BM,
        BN,
        LC,
        M,
        N,
        RowStride,
        ColStride,
        Other,
        ColStride,
        AccDepth,
        BM,
        BN,
    ],
    original: PartitionArgs,
    scheduler: type[PersistentTileScheduler],
    pid_m: gl.GluonTileId,
    pid_n: gl.GluonTileId,
    block_m: Int[BM],
    block_n: Int[BN],
) -> None:
    matmul_accumulate_epilogue_partition(correct)
    matmul_accumulate_epilogue_partition(wrong_output_host_n)  # E: is not assignable
    matmul_accumulate_epilogue_partition(wrong_row_stride)  # E: is not assignable
    matmul_accumulate_epilogue_partition(original)  # E: is not assignable
    _factory: type[EpilogueSchedulerFactory] = scheduler  # E: is not assignable
    assert_type(_factory, type[EpilogueSchedulerFactory])
    # The nominal ID alone cannot distinguish the scheduler's M/N axes.
    assert_type(pid_m * block_m, gl.GluonTileStart[BM])
    assert_type(pid_n * block_n, gl.GluonTileStart[BN])
    assert_type(pid_n * block_m, gl.GluonTileStart[BM])
    assert_type(pid_m * block_n, gl.GluonTileStart[BN])

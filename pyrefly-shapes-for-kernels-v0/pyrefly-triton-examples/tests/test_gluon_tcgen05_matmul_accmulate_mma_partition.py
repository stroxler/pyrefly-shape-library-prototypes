# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; semantic parameter annotations are not executable Gluon annotations.
# @lint-ignore-every AUTODEPS2

"""Check the original accumulate MMA worker, including erased ring indexes."""

from typing import Protocol

from shape_extensions import IntVar
from tests.test_gluon_tcgen05_matmul_accumulate_load_partition import (
    LoadPartitionArgs,
    PartitionArgs,
    t8,
)
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    mbarrier,
    tcgen05_commit,
    tcgen05_copy,
    tcgen05_mma,
    TensorMemoryRingF32,
    TensorMemoryTileF32,
)


class MmaPartitionArgs[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BHostK: IntVar,
    BBlockK: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    ARingK: IntVar,
    BRingK: IntVar,
    AccM: IntVar,
    AccN: IntVar,
    LA: IntVar,
    LB: IntVar,
    LC: IntVar,
    LoadDepth: IntVar,
    AccDepth: IntVar,
](
    LoadPartitionArgs[
        M,
        N,
        K,
        BHostK,
        BM,
        BK,
        BN,
        BBlockK,
        BM,
        BN,
        LA,
        LB,
        LC,
        LoadDepth,
        BM,
        ARingK,
        BRingK,
        BN,
        BM,
        BN,
    ],
    Protocol,
):
    acc_bufs: TensorMemoryRingF32[AccDepth, AccM, AccN]
    acc_empty_bars: gl.TmaBarrierRing2D[AccDepth]
    acc_ready_bars: gl.TmaBarrierRing2D[AccDepth]


@gluon.jit
def matmul_accmulate_mma_partition[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    LA: IntVar,
    LB: IntVar,
    LC: IntVar,
    LoadDepth: IntVar,
    AccDepth: IntVar,
](
    p: MmaPartitionArgs[
        M,
        N,
        K,
        K,
        BK,
        BM,
        BN,
        BK,
        BK,
        BK,
        BM,
        BN,
        LA,
        LB,
        LC,
        LoadDepth,
        AccDepth,
    ],
):
    BLOCK_M: gl.constexpr = p.c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = p.a_desc.block_type.shape[1]
    K = p.a_desc.shape[1]

    c_phase = 0
    load_state = t8.Counter.create(0, p.load_empty_bars.shape[0])
    acc_state = t8.Counter.create(1, p.acc_empty_bars.shape[0])
    scheduler = p.SchedulerImpl.initialize(
        p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N
    )
    for _ in range(scheduler.get_num_tiles()):  # E: is not assignable
        # We expect the load of C to take longer than the previous epilogue to
        # release the accumulator, so acquire c_buf first.
        mbarrier.wait(p.c_ready_bar, c_phase)
        mbarrier.wait(
            p.acc_empty_bars.index(acc_state.index),  # E: is not assignable
            acc_state.phase,  # E: is not assignable
        )
        acc_buf = p.acc_bufs.index(acc_state.index)  # E: is not assignable
        tcgen05_copy(p.c_buf, acc_buf)
        # Release c_buf when the copy is complete. We don't need to wait for the
        # copy to complete because it will be implicitly pipelined with the first MMA.
        tcgen05_commit(p.c_empty_bar)
        c_phase ^= 1
        for k in range(0, K, BLOCK_K):  # noqa: B007
            # Wait for the operands to be ready.
            mbarrier.wait(
                p.load_ready_bars.index(load_state.index),  # E: is not assignable
                load_state.phase,  # E: is not assignable
            )
            # Issue the MMA and release the load buffers then it completes.
            tcgen05_mma(
                p.a_bufs.index(load_state.index),  # E: is not assignable
                p.b_bufs.index(load_state.index),  # E: is not assignable
                acc_buf,
                use_acc=True,
            )
            tcgen05_commit(
                p.load_empty_bars.index(load_state.index)  # E: is not assignable
            )
            load_state = load_state.next()
        # Release the accumulator when the last MMA is complete.
        tcgen05_commit(p.acc_ready_bars.index(acc_state.index))  # E: is not assignable
        acc_state = acc_state.next()


def test_mma_worker_declared_interface[
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
    LoadDepth: IntVar,
    AccDepth: IntVar,
](
    correct: MmaPartitionArgs[
        M,
        N,
        K,
        K,
        BK,
        BM,
        BN,
        BK,
        BK,
        BK,
        BM,
        BN,
        LA,
        LB,
        LC,
        LoadDepth,
        AccDepth,
    ],
    wrong_b_host_k: MmaPartitionArgs[
        M,
        N,
        K,
        Other,
        BK,
        BM,
        BN,
        BK,
        BK,
        BK,
        BM,
        BN,
        LA,
        LB,
        LC,
        LoadDepth,
        AccDepth,
    ],
    original: PartitionArgs,
    a: gl.WgmmaSharedF16[BM, BK, LA],
    b: gl.WgmmaSharedF16[BK, BN, LB],
    wrong_b_k: gl.WgmmaSharedF16[Other, BN, LB],
    c: gl.TmaSharedTile2D[BM, BN, LC],
    wrong_c_width: gl.TmaSharedTile2D[BM, Other, LC],
    acc: TensorMemoryTileF32[BM, BN],
    wrong_acc_rows: TensorMemoryTileF32[Other, BN],
) -> None:
    matmul_accmulate_mma_partition(correct)
    matmul_accmulate_mma_partition(wrong_b_host_k)  # E: is not assignable
    matmul_accmulate_mma_partition(original)  # E: is not assignable
    tcgen05_copy(c, acc)
    tcgen05_copy(wrong_c_width, acc)  # E: is not assignable
    tcgen05_copy(c, wrong_acc_rows)  # E: is not assignable
    tcgen05_mma(a, b, acc)
    tcgen05_mma(a, wrong_b_k, acc)  # E: is not assignable
    tcgen05_mma(a, b, wrong_acc_rows)  # E: is not assignable

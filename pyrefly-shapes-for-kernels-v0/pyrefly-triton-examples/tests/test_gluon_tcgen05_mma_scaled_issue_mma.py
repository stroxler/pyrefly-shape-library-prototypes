# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; semantic parameter annotations are not executable Gluon annotations.
# @lint-ignore-every AUTODEPS2

"""Check the original packed-scale pipeline MMA helper's tile interface."""

from shape_extensions import IntVar
from tests.test_gluon_tcgen05_async_mma_scaled_impl import async_mma_scaled_impl
from tests.test_gluon_tcgen05_matmul_accumulate_load_partition import Counter
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    mbarrier,
    tcgen05_commit,
    TensorMemoryTileF32,
)


@gluon.jit
def issue_mma[
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    LA: IntVar,
    LB: IntVar,
    LSA: IntVar,
    LSB: IntVar,
    ConsumerDepth: IntVar,
    ProducerDepth: IntVar,
](
    consumer: Counter,
    c_bars: gl.TmaBarrierRing2D[ConsumerDepth],
    a_bufs: gl.WgmmaSharedRingF8[BM, BK, LA],
    b_bufs: gl.WgmmaSharedRingF8[BN, BK, LB],
    a_scale_bufs: gl.PackedScaleSharedRing[BM, BK, LSA],
    b_scale_bufs: gl.PackedScaleSharedRing[BN, BK, LSB],
    producer: Counter,
    p_bars: gl.TmaBarrierRing2D[ProducerDepth],
    acc_tmem: TensorMemoryTileF32[BM, BN],
    use_acc: bool,
    pred: bool,
):
    c_index = consumer.index
    mbarrier.wait(
        c_bars.index(c_index),  # E: is not assignable
        consumer.phase,  # E: is not assignable
        pred,
    )
    async_mma_scaled_impl(
        a_bufs.index(c_index),  # E: is not assignable
        b_bufs.index(c_index),  # E: is not assignable
        a_scale_bufs.index(c_index),  # E: is not assignable
        b_scale_bufs.index(c_index),  # E: is not assignable
        acc_tmem,
        use_acc,
        pred,
    )
    tcgen05_commit(p_bars.index(producer.index), pred)  # E: is not assignable
    return consumer.next(pred), producer.next(pred)


def test_issue_mma_declared_interface[
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Other: IntVar,
    LA: IntVar,
    LB: IntVar,
    LSA: IntVar,
    LSB: IntVar,
    Depth: IntVar,
](
    consumer: Counter,
    producer: Counter,
    c_bars: gl.TmaBarrierRing2D[Depth],
    another_depth: gl.TmaBarrierRing2D[Other],
    a_bufs: gl.WgmmaSharedRingF8[BM, BK, LA],
    b_bufs: gl.WgmmaSharedRingF8[BN, BK, LB],
    wrong_b_k: gl.WgmmaSharedRingF8[BN, Other, LB],
    a_scale_bufs: gl.PackedScaleSharedRing[BM, BK, LSA],
    wrong_a_scale_rows: gl.PackedScaleSharedRing[Other, BK, LSA],
    b_scale_bufs: gl.PackedScaleSharedRing[BN, BK, LSB],
    acc: TensorMemoryTileF32[BM, BN],
    wrong_acc_n: TensorMemoryTileF32[BM, Other],
) -> None:
    issue_mma(
        consumer,
        c_bars,
        a_bufs,
        b_bufs,
        a_scale_bufs,
        b_scale_bufs,
        producer,
        another_depth,
        acc,
        False,
        True,
    )
    # A ring's declared depth is not linked to either Counter's num_barriers.
    issue_mma(
        consumer,
        another_depth,
        a_bufs,
        b_bufs,
        a_scale_bufs,
        b_scale_bufs,
        producer,
        c_bars,
        acc,
        False,
        True,
    )
    issue_mma(
        consumer,
        c_bars,
        a_bufs,
        wrong_b_k,  # E: is not assignable
        a_scale_bufs,
        b_scale_bufs,
        producer,
        another_depth,
        acc,
        False,
        True,
    )
    issue_mma(
        consumer,
        c_bars,
        a_bufs,
        b_bufs,
        wrong_a_scale_rows,  # E: is not assignable
        b_scale_bufs,
        producer,
        another_depth,
        acc,
        False,
        True,
    )
    issue_mma(
        consumer,
        c_bars,
        a_bufs,
        b_bufs,
        a_scale_bufs,
        b_scale_bufs,
        producer,
        another_depth,
        wrong_acc_n,  # E: is not assignable
        False,
        True,
    )

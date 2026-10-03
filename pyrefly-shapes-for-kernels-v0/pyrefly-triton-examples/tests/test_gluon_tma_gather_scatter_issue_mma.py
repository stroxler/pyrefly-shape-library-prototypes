# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; semantic parameter annotations are not executable Gluon annotations.
# @lint-ignore-every AUTODEPS2

"""Check the original fused gather/scatter MMA helper's contraction."""

from typing import assert_type

from shape_extensions import Int, IntVar
from tests.test_gluon_persistent_issue_mma import MMAv5, WGMMA
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    mbarrier,
    TensorMemoryTileF32,
)
from triton.language import tensor


@gluon.jit
def issue_mma[
    BM: IntVar,
    BK: IntVar,
    BN: IntVar,
    LX: IntVar,
    LW: IntVar,
    Depth: IntVar,
](
    consumer: int,
    mma: WGMMA | MMAv5,
    bars: gl.TmaBarrierRing2D[Depth],
    x_bufs: gl.WgmmaSharedRingF16[BM, BK, LX],
    w_bufs: gl.WgmmaSharedRingF16[BK, BN, LW],
    num_buffers: Int[Depth],
):
    index = consumer % num_buffers
    b_index = consumer % num_buffers
    phase = consumer // num_buffers & 1
    consumer += 1
    mbarrier.wait(bars.index(index), phase)
    mma = mma.wait_num_outstanding(0)
    mma = mma.issue_async_mma(x_bufs.index(index), w_bufs.index(b_index))
    return consumer, mma


def test_issue_mma_contraction[
    BM: IntVar,
    BK: IntVar,
    BN: IntVar,
    Other: IntVar,
    LX: IntVar,
    LW: IntVar,
    Depth: IntVar,
](
    hopper: WGMMA,
    blackwell: MMAv5,
    bars: gl.TmaBarrierRing2D[Depth],
    wrong_bars: gl.TmaBarrierRing2D[Other],
    xring: gl.WgmmaSharedRingF16[BM, BK, LX],
    wring: gl.WgmmaSharedRingF16[BK, BN, LW],
    wrong_wk: gl.WgmmaSharedRingF16[Other, BN, LW],
    wrong_wn: gl.WgmmaSharedRingF16[BK, Other, LW],
    wrong_xm: gl.WgmmaSharedRingF16[Other, BK, LX],
    num_buffers: Int[Depth],
) -> None:
    assert_type(
        issue_mma(0, hopper, bars, xring, wring, num_buffers), tuple[int, WGMMA | MMAv5]
    )
    assert_type(
        issue_mma(0, blackwell, bars, xring, wring, num_buffers),
        tuple[int, WGMMA | MMAv5],
    )
    issue_mma(0, hopper, bars, xring, wrong_wk, num_buffers)  # E: is not assignable
    issue_mma(0, blackwell, bars, xring, wrong_wk, num_buffers)  # E: is not assignable
    # Unshaped aggregate accumulator fields do not anchor M or N to the output.
    issue_mma(0, hopper, bars, xring, wrong_wn, num_buffers)
    issue_mma(0, blackwell, bars, xring, wrong_wn, num_buffers)
    issue_mma(0, hopper, bars, wrong_xm, wring, num_buffers)
    issue_mma(0, blackwell, bars, wrong_xm, wring, num_buffers)
    issue_mma(0, hopper, wrong_bars, xring, wring, num_buffers)  # E: is not assignable
    issue_mma(
        0,
        blackwell,
        wrong_bars,
        xring,
        wring,
        num_buffers,  # E: is not assignable
    )


def test_erased_accumulator_output_gap[
    BM: IntVar,
    BK: IntVar,
    BN: IntVar,
    Other: IntVar,
    LX: IntVar,
    LW: IntVar,
    Depth: IntVar,
](
    wrong_hopper_acc: tensor[[BM, Other]],
    wrong_blackwell_acc: TensorMemoryTileF32[BM, Other],
    bar: gl.BarrierBuffer1D,
    bars: gl.TmaBarrierRing2D[Depth],
    xring: gl.WgmmaSharedRingF16[BM, BK, LX],
    wring: gl.WgmmaSharedRingF16[BK, BN, LW],
    num_buffers: Int[Depth],
) -> None:
    hopper = WGMMA(wrong_hopper_acc, gl.to_tensor(False))
    blackwell = MMAv5(gl.to_tensor(False), wrong_blackwell_acc, bar, gl.to_tensor(0))
    issue_mma(0, hopper, bars, xring, wring, num_buffers)
    issue_mma(0, blackwell, bars, xring, wring, num_buffers)

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; semantic parameter annotations are not executable Gluon annotations.
# @lint-ignore-every AUTODEPS2

"""Check the original MMA JIT helper with separately indexed B buffers."""

from shape_extensions import Int, IntVar
from tests.test_gluon_persistent_issue_mma import MMAv5, WGMMA
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier


@gluon.jit
def issue_mma_stealb[
    M: IntVar,
    K: IntVar,
    N: IntVar,
    LA: IntVar,
    LB: IntVar,
    Depth: IntVar,
](
    consumer: int,
    mma: WGMMA | MMAv5,
    bars: gl.TmaBarrierRing2D[Depth],
    a_bufs: gl.WgmmaSharedRingF16[M, K, LA],
    b_bufs: gl.WgmmaSharedRingF16[K, N, LB],
    stealb: int,
    num_buffers: Int[Depth],
):
    index = consumer % num_buffers
    b_index = consumer % (num_buffers + stealb)
    phase = consumer // num_buffers & 1
    consumer += 1
    mbarrier.wait(bars.index(index), phase)
    mma = mma.wait_num_outstanding(0)
    mma = mma.issue_async_mma(a_bufs.index(index), b_bufs.index(b_index))
    return consumer, mma


def test_stealb_mma_contract[
    M: IntVar,
    K: IntVar,
    N: IntVar,
    Other: IntVar,
    LA: IntVar,
    LB: IntVar,
    Depth: IntVar,
](
    hopper: WGMMA,
    blackwell: MMAv5,
    bars: gl.TmaBarrierRing2D[Depth],
    a_bufs: gl.WgmmaSharedRingF16[M, K, LA],
    b_bufs: gl.WgmmaSharedRingF16[K, N, LB],
    wrong_b: gl.WgmmaSharedRingF16[Other, N, LB],
    num_buffers: Int[Depth],
) -> None:
    issue_mma_stealb(0, hopper, bars, a_bufs, b_bufs, 1, num_buffers)
    issue_mma_stealb(0, blackwell, bars, a_bufs, b_bufs, 1, num_buffers)
    issue_mma_stealb(
        0,
        hopper,
        bars,
        a_bufs,
        wrong_b,  # E: is not assignable
        1,
        num_buffers,
    )
    issue_mma_stealb(
        0,
        blackwell,
        bars,
        a_bufs,
        wrong_b,  # E: is not assignable
        1,
        num_buffers,
    )
    # No capacity or nonnegative count is retained in the B-ring type.
    issue_mma_stealb(0, hopper, bars, a_bufs, b_bufs, -1, num_buffers)

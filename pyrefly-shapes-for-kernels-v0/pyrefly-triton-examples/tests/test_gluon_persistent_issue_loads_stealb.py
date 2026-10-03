# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; semantic parameter annotations are not executable Gluon annotations.
# @lint-ignore-every AUTODEPS2

"""Check the original source-order JIT helper with a spare B ring buffer."""

from typing import assert_type

from shape_extensions import Int, IntVar
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma


@gluon.jit
def issue_loads_stealb[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    LA: IntVar,
    LB: IntVar,
    Depth: IntVar,
](
    producer: int,
    a_desc: gl.TmaInputDescriptorF16[M, K, BM, BK, LA],
    b_desc: gl.TmaInputDescriptorF16[K, N, BK, BN, LB],
    off_m: gl.GluonTileStart[BM],
    off_n: gl.GluonTileStart[BN],
    k: int,
    bars: gl.TmaBarrierRing2D[Depth],
    a_bufs: gl.WgmmaSharedRingF16[BM, BK, LA],
    b_bufs: gl.WgmmaSharedRingF16[BK, BN, LB],
    stealb: int,
    num_buffers: Int[Depth],
    pred: bool = True,
):
    index = producer % num_buffers
    b_index = producer % (num_buffers + stealb)
    producer += 1
    bar = bars.index(index)
    mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes, pred=pred)
    tma.async_load(a_desc, [off_m, k], bar, a_bufs.index(index), pred)
    tma.async_load(b_desc, [k, off_n], bar, b_bufs.index(b_index), pred)
    return producer


def test_stealb_helper_contract[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Other: IntVar,
    OtherK: IntVar,
    LA: IntVar,
    LB: IntVar,
    Depth: IntVar,
](
    producer: int,
    a_desc: gl.TmaInputDescriptorF16[M, K, BM, BK, LA],
    b_desc: gl.TmaInputDescriptorF16[K, N, BK, BN, LB],
    wrong_b_tile_k: gl.TmaInputDescriptorF16[K, N, Other, BN, LB],
    wrong_b_full_k: gl.TmaInputDescriptorF16[OtherK, N, BK, BN, LB],
    bars: gl.TmaBarrierRing2D[Depth],
    a_bufs: gl.WgmmaSharedRingF16[BM, BK, LA],
    b_bufs: gl.WgmmaSharedRingF16[BK, BN, LB],
    wrong_b_tile: gl.WgmmaSharedRingF16[BK, Other, LB],
    num_buffers: Int[Depth],
    off_m: gl.GluonTileStart[BM],
    off_n: gl.GluonTileStart[BN],
    wrong_off_n: gl.GluonTileStart[Other],
) -> None:
    assert_type(
        issue_loads_stealb(
            producer,
            a_desc,
            b_desc,
            off_m,
            off_n,
            0,
            bars,
            a_bufs,
            b_bufs,
            1,
            num_buffers,
        ),
        int,
    )
    issue_loads_stealb(
        producer,
        a_desc,
        wrong_b_tile_k,  # E: is not assignable
        off_m,
        off_n,
        0,
        bars,
        a_bufs,
        b_bufs,
        1,
        num_buffers,
    )
    issue_loads_stealb(
        producer,
        a_desc,
        wrong_b_full_k,  # E: is not assignable
        off_m,
        off_n,
        0,
        bars,
        a_bufs,
        b_bufs,
        1,
        num_buffers,
    )
    issue_loads_stealb(
        producer,
        a_desc,
        b_desc,
        off_m,
        off_n,
        0,
        bars,
        a_bufs,
        wrong_b_tile,  # E: is not assignable
        1,
        num_buffers,
    )
    issue_loads_stealb(
        producer,
        a_desc,
        b_desc,
        off_m,
        wrong_off_n,  # E: is not assignable
        0,
        bars,
        a_bufs,
        b_bufs,
        1,
        num_buffers,
    )


def test_spare_buffer_capacity_is_not_grounded[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    LA: IntVar,
    LB: IntVar,
    Depth: IntVar,
](
    a_desc: gl.TmaInputDescriptorF16[M, K, BM, BK, LA],
    b_desc: gl.TmaInputDescriptorF16[K, N, BK, BN, LB],
    bars: gl.TmaBarrierRing2D[Depth],
    a_bufs: gl.WgmmaSharedRingF16[BM, BK, LA],
    b_bufs: gl.WgmmaSharedRingF16[BK, BN, LB],
    num_buffers: Int[Depth],
    off_m: gl.GluonTileStart[BM],
    off_n: gl.GluonTileStart[BN],
) -> None:
    # The ring type carries tile dimensions and layout, not its allocation depth.
    issue_loads_stealb(
        0, a_desc, b_desc, off_m, off_n, 0, bars, a_bufs, b_bufs, 999, num_buffers
    )
    issue_loads_stealb(
        0, a_desc, b_desc, off_m, off_n, 0, bars, a_bufs, b_bufs, -1, num_buffers
    )

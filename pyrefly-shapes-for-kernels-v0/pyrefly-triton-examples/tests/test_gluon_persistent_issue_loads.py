# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; semantic parameter annotations are not executable Gluon annotations.
# @lint-ignore-every AUTODEPS2

"""Check the original first source-order persistent-matmul JIT helper."""

from typing import assert_type

from shape_extensions import Int, IntVar
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma


@gluon.jit
def issue_loads[
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
    num_buffers: Int[Depth],
    pred: bool = True,
):
    index = producer % num_buffers
    producer += 1
    bar = bars.index(index)
    mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes, pred=pred)
    tma.async_load(a_desc, [off_m, k], bar, a_bufs.index(index), pred)
    tma.async_load(b_desc, [k, off_n], bar, b_bufs.index(index), pred)
    return producer


def test_load_interface[
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
    wrong_b_k: gl.TmaInputDescriptorF16[K, N, Other, BN, LB],
    wrong_b_host_k: gl.TmaInputDescriptorF16[OtherK, N, BK, BN, LB],
    wrong_a_layout: gl.WgmmaSharedRingF16[BM, BK, LB],
    bars: gl.TmaBarrierRing2D[Depth],
    a_bufs: gl.WgmmaSharedRingF16[BM, BK, LA],
    b_bufs: gl.WgmmaSharedRingF16[BK, BN, LB],
    num_buffers: Int[Depth],
    off_m: gl.GluonTileStart[BM],
    off_n: gl.GluonTileStart[BN],
    wrong_off_n: gl.GluonTileStart[Other],
) -> None:
    assert_type(
        issue_loads(
            producer, a_desc, b_desc, off_m, off_n, 0, bars, a_bufs, b_bufs, num_buffers
        ),
        int,
    )
    issue_loads(
        producer,
        a_desc,
        wrong_b_k,  # E: is not assignable
        off_m,
        off_n,
        0,
        bars,
        a_bufs,
        b_bufs,
        num_buffers,
    )
    issue_loads(
        producer,
        a_desc,
        wrong_b_host_k,  # E: is not assignable
        off_m,
        off_n,
        0,
        bars,
        a_bufs,
        b_bufs,
        num_buffers,
    )
    issue_loads(
        producer,
        a_desc,
        b_desc,
        off_m,
        off_n,
        0,
        bars,
        wrong_a_layout,  # E: is not assignable
        b_bufs,
        num_buffers,
    )
    issue_loads(
        producer,
        a_desc,
        b_desc,
        off_m,
        wrong_off_n,  # E: is not assignable
        0,
        bars,
        a_bufs,
        b_bufs,
        num_buffers,
    )


def test_independent_loads_do_not_contract_k[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    OtherK: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    LA: IntVar,
    LB: IntVar,
](
    a_desc: gl.TmaInputDescriptorF16[M, K, BM, BK, LA],
    b_desc: gl.TmaInputDescriptorF16[OtherK, N, BK, BN, LB],
    a_tile: gl.WgmmaSharedF16[BM, BK, LA],
    b_tile: gl.WgmmaSharedF16[BK, BN, LB],
    barrier: gl.BarrierBuffer1D,
) -> None:
    tma.async_load(a_desc, [0, 0], barrier, a_tile)
    tma.async_load(b_desc, [0, 0], barrier, b_tile)
    tma.async_load(a_desc, [], barrier, a_tile)


def test_ring_indices_are_not_bounded[
    Depth: IntVar,
    BM: IntVar,
    BK: IntVar,
    Layout: IntVar,
](
    bars: gl.TmaBarrierRing2D[Depth],
    a_bufs: gl.WgmmaSharedRingF16[BM, BK, Layout],
) -> None:
    assert_type(bars.index(1_000_000), gl.BarrierBuffer1D)
    assert_type(a_bufs.index(-1), gl.WgmmaSharedF16[BM, BK, Layout])

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; semantic parameter annotations are not executable Gluon annotations.
# @lint-ignore-every AUTODEPS2
# @lint-ignore-every SPELL

"""Inspect the original packed-scale pipeline TMA load helper."""

from shape_extensions import IntVar
from tests.test_gluon_tcgen05_matmul_accumulate_load_partition import Counter
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import mbarrier, tma


@gluon.jit
def issue_loads[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    HostAM: IntVar,
    HostAK: IntVar,
    HostBN: IntVar,
    HostBK: IntVar,
    LA: IntVar,
    LB: IntVar,
    LSA: IntVar,
    LSB: IntVar,
    Depth: IntVar,
](
    producer: Counter,
    pid_m: gl.GluonTileId,
    pid_n: gl.GluonTileId,
    k: int,
    a_desc: gl.TmaDescriptorF8[M, K, BM, BK, LA],
    b_desc: gl.TmaDescriptorF8[N, K, BN, BK, LB],
    a_scale_desc: gl.PackedScaleDescriptorU8[HostAM, HostAK, BM, BK, LSA],
    b_scale_desc: gl.PackedScaleDescriptorU8[HostBN, HostBK, BN, BK, LSB],
    a_bufs: gl.WgmmaSharedRingF8[BM, BK, LA],
    b_bufs: gl.WgmmaSharedRingF8[BN, BK, LB],
    a_scale_bufs: gl.PackedScaleSharedRing[BM, BK, LSA],
    b_scale_bufs: gl.PackedScaleSharedRing[BN, BK, LSB],
    bars: gl.TmaBarrierRing2D[Depth],
    pred: bool,
):
    A_ELEM_PER_BYTE: gl.constexpr = 2 if a_desc.dtype == gl.uint8 else 1
    B_ELEM_PER_BYTE: gl.constexpr = 2 if b_desc.dtype == gl.uint8 else 1
    BLOCK_M: gl.constexpr = a_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = b_desc.block_type.shape[0]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1] * A_ELEM_PER_BYTE
    REP_M: gl.constexpr = a_scale_desc.block_type.shape[1]
    REP_N: gl.constexpr = b_scale_desc.block_type.shape[1]
    A_REP_K: gl.constexpr = a_scale_desc.block_type.shape[2]
    B_REP_K: gl.constexpr = b_scale_desc.block_type.shape[2]

    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    off_m_a_scale = pid_m * REP_M
    off_n_b_scale = pid_n * REP_N
    off_k_a = k // A_ELEM_PER_BYTE
    off_k_b = k // B_ELEM_PER_BYTE
    off_k_a_scale = (k // BLOCK_K) * A_REP_K
    off_k_b_scale = (k // BLOCK_K) * B_REP_K

    index = producer.index
    bar = bars.index(index)  # E: is not assignable
    mbarrier.expect(
        bar,
        a_desc.block_type.nbytes
        + b_desc.block_type.nbytes
        + a_scale_desc.block_type.nbytes
        + b_scale_desc.block_type.nbytes,
        pred,
    )
    tma.async_load(
        a_desc,
        [off_m, off_k_a],
        bar,
        a_bufs.index(index),  # E: is not assignable
        pred,
    )
    tma.async_load(
        b_desc,
        [off_n, off_k_b],
        bar,
        b_bufs.index(index),  # E: is not assignable
        pred,
    )
    tma.async_load(
        a_scale_desc,
        [0, off_m_a_scale, off_k_a_scale, 0, 0],
        bar,
        a_scale_bufs.index(index),  # E: is not assignable
        pred,
    )
    tma.async_load(
        b_scale_desc,
        [0, off_n_b_scale, off_k_b_scale, 0, 0],
        bar,
        b_scale_bufs.index(index),  # E: is not assignable
        pred,
    )
    return producer.next(pred)


def test_issue_loads_declared_interface[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    Other: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    HostAM: IntVar,
    HostAK: IntVar,
    HostBN: IntVar,
    HostBK: IntVar,
    LA: IntVar,
    LB: IntVar,
    LSA: IntVar,
    LSB: IntVar,
    Depth: IntVar,
](
    producer: Counter,
    pid_m: gl.GluonTileId,
    pid_n: gl.GluonTileId,
    a_desc: gl.TmaDescriptorF8[M, K, BM, BK, LA],
    b_desc: gl.TmaDescriptorF8[N, K, BN, BK, LB],
    wrong_b_host_k: gl.TmaDescriptorF8[N, Other, BN, BK, LB],
    a_scale_desc: gl.PackedScaleDescriptorU8[HostAM, HostAK, BM, BK, LSA],
    b_scale_desc: gl.PackedScaleDescriptorU8[HostBN, HostBK, BN, BK, LSB],
    wrong_a_scale_host_k: gl.PackedScaleDescriptorU8[HostAM, Other, BM, BK, LSA],
    wrong_b_scale_block_k: gl.PackedScaleDescriptorU8[HostBN, HostBK, BN, Other, LSB],
    a_bufs: gl.WgmmaSharedRingF8[BM, BK, LA],
    wrong_a_ring_k: gl.WgmmaSharedRingF8[BM, Other, LA],
    b_bufs: gl.WgmmaSharedRingF8[BN, BK, LB],
    a_scale_bufs: gl.PackedScaleSharedRing[BM, BK, LSA],
    b_scale_bufs: gl.PackedScaleSharedRing[BN, BK, LSB],
    bars: gl.TmaBarrierRing2D[Depth],
) -> None:
    issue_loads(
        producer,
        pid_m,
        pid_n,
        0,
        a_desc,
        b_desc,
        a_scale_desc,
        b_scale_desc,
        a_bufs,
        b_bufs,
        a_scale_bufs,
        b_scale_bufs,
        bars,
        True,
    )
    issue_loads(
        producer,
        pid_m,
        pid_n,
        0,
        a_desc,
        wrong_b_host_k,  # E: is not assignable
        a_scale_desc,
        b_scale_desc,
        a_bufs,
        b_bufs,
        a_scale_bufs,
        b_scale_bufs,
        bars,
        True,
    )
    # The host repeat K of the scales is not linked to either operand host K.
    issue_loads(
        producer,
        pid_m,
        pid_n,
        0,
        a_desc,
        b_desc,
        wrong_a_scale_host_k,
        b_scale_desc,
        a_bufs,
        b_bufs,
        a_scale_bufs,
        b_scale_bufs,
        bars,
        True,
    )
    issue_loads(
        producer,
        pid_m,
        pid_n,
        0,
        a_desc,
        b_desc,
        a_scale_desc,
        wrong_b_scale_block_k,  # E: is not assignable
        a_bufs,
        b_bufs,
        a_scale_bufs,
        b_scale_bufs,
        bars,
        True,
    )
    issue_loads(
        producer,
        pid_m,
        pid_n,
        0,
        a_desc,
        b_desc,
        a_scale_desc,
        b_scale_desc,
        wrong_a_ring_k,  # E: is not assignable
        b_bufs,
        a_scale_bufs,
        b_scale_bufs,
        bars,
        True,
    )


def test_tma_coordinates_are_not_rank_or_bounds_checked[
    M: IntVar,
    K: IntVar,
    BM: IntVar,
    BK: IntVar,
    LA: IntVar,
](
    desc: gl.TmaDescriptorF8[M, K, BM, BK, LA],
    bar: gl.BarrierBuffer1D,
    tile: gl.WgmmaSharedF8[BM, BK, LA],
) -> None:
    tma.async_load(desc, [0], bar, tile)
    tma.async_load(desc, [0, 0, 0], bar, tile)
    tma.async_load(desc, [-1, 12345], bar, tile)

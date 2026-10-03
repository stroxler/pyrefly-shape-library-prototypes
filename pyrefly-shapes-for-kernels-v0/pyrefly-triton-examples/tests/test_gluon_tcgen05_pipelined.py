# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; semantic annotations are not executable Gluon annotations.
# @lint-ignore-every AUTODEPS2

"""Check the original two-tile Blackwell pipeline and its scheduling helper."""

from typing import assert_type

from shape_extensions import Int, IntVar
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    allocate_tensor_memory,
    fence_async_shared,
    mbarrier,
    tcgen05_commit,
    tcgen05_mma,
    TensorMemoryLayout,
    tma,
)


@gluon.jit
def get_and_increment(counter: int):
    return counter % 2, counter // 2 & 1, counter + 1


@gluon.jit
def blocked_matmul_pipelined_kernel[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    LA: IntVar,
    LB: IntVar,
    LC: IntVar,
](
    a_desc: gl.TmaInputDescriptorF16[M, K, BM, BK, LA],
    b_desc: gl.TmaInputDescriptorF16[K, N, BK, BN, LB],
    c_desc: gl.TmaOutputDescriptorF16[M, N, BM, BN, LC],
    num_warps: int,
):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * (2 * BLOCK_M)
    off_n = pid_n * BLOCK_N

    # u := upper tile, v := lower tile
    u_bufs = gl.allocate_shared_memory(
        dtype, [2] + a_desc.block_type.shape, a_desc.layout
    )
    v_bufs = gl.allocate_shared_memory(
        dtype, [2] + a_desc.block_type.shape, a_desc.layout
    )
    b_bufs = gl.allocate_shared_memory(
        dtype, [2] + b_desc.block_type.shape, b_desc.layout
    )

    # Use two accumulators!
    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    ub_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
    vb_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)

    mma_ub_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    mma_vb_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    load_ub_bars = gl.allocate_shared_memory(
        gl.int64, [2, 1], mbarrier.MBarrierLayout()
    )
    load_v_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(2):
        mbarrier.init(mma_ub_bars.index(i), count=1)
        mbarrier.init(mma_vb_bars.index(i), count=1)
        mbarrier.init(load_ub_bars.index(i), count=1)
        mbarrier.init(load_v_bars.index(i), count=1)

    load_counter = 0
    mma_counter = 0
    k = 0
    ub_acc = False
    vb_acc = False

    # U1, B1
    load_index, load_phase, load_counter = get_and_increment(load_counter)
    load_ub_bar = load_ub_bars.index(load_index)
    mbarrier.expect(load_ub_bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
    tma.async_load(a_desc, [off_m, k], load_ub_bar, u_bufs.index(load_index))
    tma.async_load(b_desc, [k, off_n], load_ub_bar, b_bufs.index(load_index))
    # V1
    load_v_bar = load_v_bars.index(load_index)
    mbarrier.expect(load_v_bar, a_desc.block_type.nbytes)
    tma.async_load(a_desc, [off_m + BLOCK_M, k], load_v_bar, v_bufs.index(load_index))
    k += BLOCK_K

    # U2, B2
    load_index, load_phase, load_counter = get_and_increment(load_counter)
    load_ub_bar = load_ub_bars.index(load_index)
    mbarrier.expect(load_ub_bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
    tma.async_load(a_desc, [off_m, k], load_ub_bar, u_bufs.index(load_index))
    tma.async_load(b_desc, [k, off_n], load_ub_bar, b_bufs.index(load_index))
    # V2
    load_v_bar = load_v_bars.index(load_index)
    mbarrier.expect(load_v_bar, a_desc.block_type.nbytes)
    tma.async_load(a_desc, [off_m + BLOCK_M, k], load_v_bar, v_bufs.index(load_index))
    k += BLOCK_K

    for _ in range(gl.cdiv(K, BLOCK_K) - 2):  # E: Fixpoint iteration did not converge
        # wait Ui and Bi, UBi
        mma_index, mma_phase, mma_counter = get_and_increment(mma_counter)
        mbarrier.wait(load_ub_bars.index(mma_index), mma_phase)
        tcgen05_mma(
            u_bufs.index(mma_index), b_bufs.index(mma_index), ub_tmem, use_acc=ub_acc
        )
        tcgen05_commit(mma_ub_bars.index(mma_index))
        ub_acc = True
        # wait Vi, VBi
        mbarrier.wait(load_v_bars.index(mma_index), mma_phase)
        tcgen05_mma(
            v_bufs.index(mma_index), b_bufs.index(mma_index), vb_tmem, use_acc=vb_acc
        )
        tcgen05_commit(mma_vb_bars.index(mma_index))
        vb_acc = True

        # wait UBi, U(i+2)
        load_index, load_phase, load_counter = get_and_increment(load_counter)
        mbarrier.wait(mma_ub_bars.index(mma_index), mma_phase)
        load_ub_bar = load_ub_bars.index(load_index)
        mbarrier.expect(
            load_ub_bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes
        )
        tma.async_load(a_desc, [off_m, k], load_ub_bar, u_bufs.index(load_index))

        # wait VBi, B(i+2), V(i+2)
        mbarrier.wait(mma_vb_bars.index(mma_index), mma_phase)
        tma.async_load(b_desc, [k, off_n], load_ub_bar, b_bufs.index(load_index))
        load_v_bar = load_v_bars.index(load_index)
        mbarrier.expect(load_v_bar, a_desc.block_type.nbytes)
        tma.async_load(
            a_desc, [off_m + BLOCK_M, k], load_v_bar, v_bufs.index(load_index)
        )
        k += BLOCK_K

    mma_index, mma_phase, mma_counter = get_and_increment(mma_counter)
    ub_bar = mma_ub_bars.index(mma_index)
    vb_bar = mma_vb_bars.index(mma_index)
    epilogue_phase = mma_phase

    # wait U(N-1) and B(N-1), UB(N-1)
    mbarrier.wait(load_ub_bars.index(mma_index), mma_phase)
    tcgen05_mma(u_bufs.index(mma_index), b_bufs.index(mma_index), ub_tmem, use_acc=True)
    # wait V(N-1), VB(N-1)
    mbarrier.wait(load_v_bars.index(mma_index), mma_phase)
    tcgen05_mma(v_bufs.index(mma_index), b_bufs.index(mma_index), vb_tmem, use_acc=True)

    # Wait UN and BN, UBN
    mma_index, mma_phase, mma_counter = get_and_increment(mma_counter)
    mbarrier.wait(load_ub_bars.index(mma_index), mma_phase)
    tcgen05_mma(u_bufs.index(mma_index), b_bufs.index(mma_index), ub_tmem, use_acc=True)
    tcgen05_commit(ub_bar)
    # Wait VN and VBN
    mbarrier.wait(load_v_bars.index(mma_index), mma_phase)
    tcgen05_mma(v_bufs.index(mma_index), b_bufs.index(mma_index), vb_tmem, use_acc=True)
    tcgen05_commit(vb_bar)

    # Wait UBN, UB epilogue
    mbarrier.wait(ub_bar, epilogue_phase)
    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    ub = ub_tmem.load()
    c_smem.store(ub.to(dtype))
    fence_async_shared()
    tma.async_store(c_desc, [off_m, off_n], c_smem)

    # Wait VBN, VB epilogue
    mbarrier.wait(vb_bar, epilogue_phase)
    vb = vb_tmem.load()
    tma.store_wait(pendings=0)
    c_smem.store(vb.to(dtype))
    fence_async_shared()
    tma.async_store(c_desc, [off_m + BLOCK_M, off_n], c_smem)
    tma.store_wait(pendings=0)


def test_double_tile_start[
    Rows: IntVar,
    Cols: IntVar,
    BM: IntVar,
    BK: IntVar,
    Other: IntVar,
    Layout: IntVar,
](
    pid: gl.GluonProgramId,
    bm: Int[BM],
    other: Int[Other],
    descriptor: gl.TmaInputDescriptorF16[Rows, Cols, BM, BK, Layout],
    wrong_row_descriptor: gl.TmaInputDescriptorF16[Rows, Cols, Other, BK, Layout],
    tile: gl.WgmmaSharedF16[BM, BK, Layout],
    wrong_row_tile: gl.WgmmaSharedF16[Other, BK, Layout],
    barrier: gl.BarrierBuffer1D,
) -> None:
    start = pid * (2 * bm)
    assert_type(start, gl.GluonTileStart[2 * BM])
    assert_type(start + bm, gl.GluonTileStart[2 * BM])
    start + other  # E: is not supported
    tma.async_load(descriptor, [start, 0], barrier, tile)
    tma.async_load(descriptor, [start + bm, 0], barrier, tile)
    tma.async_load(  # E: No matching overload
        wrong_row_descriptor, [start, 0], barrier, wrong_row_tile
    )

    # This shape relation does not prove alignment or address bounds.
    assert_type(pid * bm + bm // 2, gl.GluonTileStart[BM])


def test_shared_ring_depth_is_not_verified[
    Rows: IntVar,
    Cols: IntVar,
    BR: IntVar,
    BC: IntVar,
    Layout: IntVar,
](
    descriptor: gl.TmaInputDescriptorF16[Rows, Cols, BR, BC, Layout],
) -> None:
    assert_type(
        gl.allocate_shared_memory(
            descriptor.dtype, [3] + descriptor.block_type.shape, descriptor.layout
        ),
        gl.WgmmaSharedRingF16[BR, BC, Layout],
    )
    assert_type(
        gl.allocate_shared_memory(descriptor.dtype, [], descriptor.layout),
        gl.WgmmaSharedRingF16[BR, BC, Layout],
    )

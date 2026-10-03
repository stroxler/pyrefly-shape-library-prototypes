# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; semantic parameter annotations are not executable Gluon annotations.
# @lint-ignore-every AUTODEPS2
# @lint-ignore-every SPELL

"""Check the unchanged packed-scale TMA load and unswizzle helper."""

from typing import Literal

from shape_extensions import Int, IntVar
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    allocate_tensor_memory,
    fence_async_shared,
    mbarrier,
    tcgen05_commit,
    tcgen05_mma_scaled,
    TensorMemoryLayout,
    TensorMemoryScalesLayout,
    tma,
)


@gluon.jit
def unswizzle_scales_packed_block[Rows: IntVar, K: IntVar](
    scales: gl.PackedScaleTile[Rows, K],
    BLOCK_MN: Int[Rows],
    BLOCK_K: Int[K],
    VEC_SIZE: Literal[32],
):
    # Unswizzle the scales subtile from its packed block layout.
    scales = scales.reshape(  # E: is not assignable
        scales.shape[1], scales.shape[2], 32, 4, 4
    )
    scales = scales.permute(0, 3, 2, 1, 4)  # E: no attribute `permute`
    return scales.reshape(  # E: Missing argument `d32` # E: Missing argument `d4` # E: Missing argument `d4b`
        BLOCK_MN,  # E: is not assignable
        BLOCK_K // VEC_SIZE,  # E: is not assignable
    )


@gluon.jit
def mma_scaled_packed_block_kernel[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    LA: IntVar,
    LB: IntVar,
    LC: IntVar,
    LSA: IntVar,
    LSB: IntVar,
    AMHost: IntVar,
    AKHost: IntVar,
    BNHost: IntVar,
    BKHost: IntVar,
](
    a_desc: gl.TmaDescriptorF8[M, K, BM, BK, LA],
    b_desc: gl.TmaDescriptorF8[N, K, BN, BK, LB],
    c_desc: gl.WgmmaDescriptorF16[M, N, BM, BN, LC],
    a_scale_desc: gl.PackedScaleDescriptorU8[AMHost, AKHost, BM, BK, LSA],
    b_scale_desc: gl.PackedScaleDescriptorU8[BNHost, BKHost, BN, BK, LSB],
    VEC_SIZE: Literal[32],
):
    # ======= Begin unchanged code from `simple_mma_scaled_kernel` =======
    A_IS_FP4: gl.constexpr = a_desc.dtype == gl.uint8
    B_IS_FP4: gl.constexpr = b_desc.dtype == gl.uint8
    A_ELEM_PER_BYTE: gl.constexpr = 2 if A_IS_FP4 else 1
    B_ELEM_PER_BYTE: gl.constexpr = 2 if B_IS_FP4 else 1
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1] * A_ELEM_PER_BYTE
    K = a_desc.shape[1] * A_ELEM_PER_BYTE

    a_smem = gl.allocate_shared_memory(
        a_desc.dtype, a_desc.block_type.shape, a_desc.layout
    )
    b_smem = gl.allocate_shared_memory(
        b_desc.dtype, b_desc.block_type.shape, b_desc.layout
    )

    scale_layout: gl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(
        a_scale_desc.dtype, [BLOCK_M, BLOCK_K // VEC_SIZE], scale_layout
    )
    b_scale_tmem = allocate_tensor_memory(
        b_scale_desc.dtype, [BLOCK_N, BLOCK_K // VEC_SIZE], scale_layout
    )
    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
    use_acc = False

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.init(mma_bar, count=1)
    phase = 0
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    # ======= End unchanged code from `simple_mma_scaled_kernel` =======

    # Allocate shared memory to TMA load the scales.
    a_scale_smem = gl.allocate_shared_memory(
        a_scale_desc.dtype, a_scale_desc.block_type.shape, a_scale_desc.layout
    )
    b_scale_smem = gl.allocate_shared_memory(
        b_scale_desc.dtype, b_scale_desc.block_type.shape, b_scale_desc.layout
    )
    REP_M: gl.constexpr = a_scale_desc.block_type.shape[1]
    REP_N: gl.constexpr = b_scale_desc.block_type.shape[1]
    A_REP_K: gl.constexpr = a_scale_desc.block_type.shape[2]
    B_REP_K: gl.constexpr = b_scale_desc.block_type.shape[2]
    # Index the M and N subtiles along REP_M.
    off_m_a_scale = pid_m * REP_M
    off_n_b_scale = pid_n * REP_N

    for k in range(0, K, BLOCK_K):
        off_k_a = k // A_ELEM_PER_BYTE
        off_k_b = k // B_ELEM_PER_BYTE
        # Index the K subtile along REP_K for each scale.
        off_k_a_scale = (k // BLOCK_K) * A_REP_K
        off_k_b_scale = (k // BLOCK_K) * B_REP_K

        mbarrier.expect(
            bar,
            a_desc.block_type.nbytes
            + b_desc.block_type.nbytes
            + a_scale_desc.block_type.nbytes
            + b_scale_desc.block_type.nbytes,
        )
        tma.async_load(a_desc, [off_m, off_k_a], bar, a_smem)
        tma.async_load(b_desc, [off_n, off_k_b], bar, b_smem)
        tma.async_load(
            a_scale_desc, [0, off_m_a_scale, off_k_a_scale, 0, 0], bar, a_scale_smem
        )
        tma.async_load(
            b_scale_desc, [0, off_n_b_scale, off_k_b_scale, 0, 0], bar, b_scale_smem
        )
        mbarrier.wait(bar, phase)

        # We know the destination 2D layout of the scales required to store them
        # into tensor memory. You could work backwards to figure out the layout with
        # which to load the scales from shared memory such that after unswizzling,
        # they have the right 2D layout for the store to TMEM. Instead, we will use
        # AutoLayout to let the compiler backwards propagate the layout.
        a_scale_layout: gl.constexpr = a_scale_tmem.get_reg_layout()
        b_scale_layout: gl.constexpr = b_scale_tmem.get_reg_layout()

        # Load the scales with AutoLayout. Subsequent operations, including the unswizzling,
        # will be generic over the layout.
        a_scale = a_scale_smem.load(gl.AutoLayout())
        b_scale = b_scale_smem.load(gl.AutoLayout())
        a_scale = unswizzle_scales_packed_block(a_scale, BLOCK_M, BLOCK_K, VEC_SIZE)
        b_scale = unswizzle_scales_packed_block(b_scale, BLOCK_N, BLOCK_K, VEC_SIZE)

        # Use `set_auto_layout` with the concrete scale layouts to create an anchor.
        # The compiler will propagate the layout backwards to resolve the auto layouts.
        a_scale = gl.set_auto_layout(  # E: No attribute `set_auto_layout`
            a_scale, a_scale_layout
        )
        b_scale = gl.set_auto_layout(  # E: No attribute `set_auto_layout`
            b_scale, b_scale_layout
        )

        # ======= Begin unchanged code from `simple_mma_scaled_kernel` =======
        a_scale_tmem.store(a_scale)
        b_scale_tmem.store(b_scale)

        a_format: gl.constexpr = "e2m1" if A_IS_FP4 else "e4m3"
        b_format: gl.constexpr = "e2m1" if B_IS_FP4 else "e4m3"
        tcgen05_mma_scaled(
            a_smem,
            b_smem.permute((1, 0)),
            acc_tmem,
            a_scale_tmem,
            b_scale_tmem,
            a_format,
            b_format,
            use_acc=use_acc,
        )
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, phase)
        use_acc = True
        phase ^= 1

    mbarrier.invalidate(bar)
    mbarrier.invalidate(mma_bar)
    acc = acc_tmem.load()
    acc = acc.to(c_desc.dtype)
    acc_smem = gl.allocate_shared_memory(
        c_desc.dtype, c_desc.block_type.shape, c_desc.layout
    )
    acc_smem.store(acc)
    fence_async_shared()
    tma.async_store(c_desc, [off_m, off_n], acc_smem)
    tma.store_wait(0)
    # ======= End unchanged code from `simple_mma_scaled_kernel` =======


def test_packed_scale_tma_tile_contract[
    HostRows: IntVar,
    HostK: IntVar,
    Rows: IntVar,
    K: IntVar,
    Other: IntVar,
    Layout: IntVar,
](
    desc: gl.PackedScaleDescriptorU8[HostRows, HostK, Rows, K, Layout],
    other_host_k: gl.PackedScaleDescriptorU8[HostRows, Other, Rows, K, Layout],
    other_block_rows: gl.PackedScaleDescriptorU8[HostRows, HostK, Other, K, Layout],
    tile: gl.PackedScaleSharedTile[Rows, K, Layout],
    wrong_tile_rows: gl.PackedScaleSharedTile[Other, K, Layout],
    wrong_tile_k: gl.PackedScaleSharedTile[Rows, Other, Layout],
    bar: gl.BarrierBuffer1D,
    coordinates: list[int],
) -> None:
    tma.async_load(desc, coordinates, bar, tile)
    tma.async_load(other_host_k, coordinates, bar, tile)
    tma.async_load(desc, coordinates, bar, wrong_tile_rows)  # E: No matching overload
    tma.async_load(desc, coordinates, bar, wrong_tile_k)  # E: No matching overload
    tma.async_load(other_block_rows, coordinates, bar, tile)  # E: No matching overload


def test_packed_scale_declared_kernel_interface[
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
    LSA: IntVar,
    LSB: IntVar,
    HostAM: IntVar,
    HostAK: IntVar,
    HostBN: IntVar,
    HostBK: IntVar,
](
    a: gl.TmaDescriptorF8[M, K, BM, BK, LA],
    b: gl.TmaDescriptorF8[N, K, BN, BK, LB],
    wrong_b_host_k: gl.TmaDescriptorF8[N, Other, BN, BK, LB],
    c: gl.WgmmaDescriptorF16[M, N, BM, BN, LC],
    wrong_c_host_n: gl.WgmmaDescriptorF16[M, Other, BM, BN, LC],
    a_scales: gl.PackedScaleDescriptorU8[HostAM, HostAK, BM, BK, LSA],
    wrong_a_block_m: gl.PackedScaleDescriptorU8[HostAM, HostAK, Other, BK, LSA],
    b_scales: gl.PackedScaleDescriptorU8[HostBN, HostBK, BN, BK, LSB],
    wrong_b_block_k: gl.PackedScaleDescriptorU8[HostBN, HostBK, BN, Other, LSB],
    wrong_a_host_k: gl.PackedScaleDescriptorU8[HostAM, Other, BM, BK, LSA],
) -> None:
    mma_scaled_packed_block_kernel(a, b, c, a_scales, b_scales, 32)
    mma_scaled_packed_block_kernel(
        a,
        wrong_b_host_k,  # E: is not assignable
        c,
        a_scales,
        b_scales,
        32,
    )
    mma_scaled_packed_block_kernel(
        a,
        b,
        wrong_c_host_n,  # E: is not assignable
        a_scales,
        b_scales,
        32,
    )
    mma_scaled_packed_block_kernel(
        a,
        b,
        c,
        wrong_a_block_m,  # E: is not assignable
        b_scales,
        32,
    )
    mma_scaled_packed_block_kernel(
        a,
        b,
        c,
        a_scales,
        wrong_b_block_k,  # E: is not assignable
        32,
    )
    # Wrong full 5D scale allocation extent remains accepted by the kernel.
    mma_scaled_packed_block_kernel(a, b, c, wrong_a_host_k, b_scales, 32)

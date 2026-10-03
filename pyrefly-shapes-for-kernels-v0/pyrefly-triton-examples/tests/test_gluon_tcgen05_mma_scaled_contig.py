# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; semantic parameter annotations are not executable Gluon annotations.
# @lint-ignore-every AUTODEPS2

"""Check the unchanged contiguous-scale MMA kernel and its 1D-to-2D tile."""

from typing import Literal

from shape_extensions import IntVar
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
def mma_scaled_contig_kernel[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    LA: IntVar,
    LB: IntVar,
    LC: IntVar,
    AScaleElements: IntVar,
    BScaleElements: IntVar,
](
    a_desc: gl.TmaDescriptorF8[M, K, BM, BK, LA],
    b_desc: gl.TmaDescriptorF8[N, K, BN, BK, LB],
    c_desc: gl.WgmmaDescriptorF16[M, N, BM, BN, LC],
    a_scale_ptr: gl.ContiguousScalePointer1D[AScaleElements, BM, BK // 32],
    b_scale_ptr: gl.ContiguousScalePointer1D[BScaleElements, BN, BK // 32],
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
        a_scale_ptr.dtype.element_ty, [BLOCK_M, BLOCK_K // VEC_SIZE], scale_layout
    )
    b_scale_tmem = allocate_tensor_memory(
        b_scale_ptr.dtype.element_ty, [BLOCK_N, BLOCK_K // VEC_SIZE], scale_layout
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

    for k in range(0, K, BLOCK_K):
        off_k_a = k // A_ELEM_PER_BYTE
        off_k_b = k // B_ELEM_PER_BYTE

        mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_load(a_desc, [off_m, off_k_a], bar, a_smem)
        tma.async_load(b_desc, [off_n, off_k_b], bar, b_smem)
        mbarrier.wait(bar, phase)

        # ======= End unchanged code from `simple_mma_scaled_kernel` =======

        SCALE_K = K // VEC_SIZE
        SCALE_BLOCK_K: gl.constexpr = BLOCK_K // VEC_SIZE
        # We know the global memory tensor `a_scale` is contiguous with shape
        # [M // BLOCK_M, SCALE_K // SCALE_BLOCK_K, BLOCK_M, SCALE_BLOCK_K]. Each inner
        # loop tile will load `a_scale[pid_m, k // BLOCK_K, :, :]`.
        a_stride_k: gl.constexpr = BLOCK_M * SCALE_BLOCK_K
        a_stride_m = SCALE_K // SCALE_BLOCK_K * a_stride_k
        b_stride_k: gl.constexpr = BLOCK_N * SCALE_BLOCK_K
        b_stride_n = SCALE_K // SCALE_BLOCK_K * b_stride_k

        # Load `a_scale[pid_m, k // BLOCK_K, :, :]`. Since we know the inner two
        # dimensions are contiguous, we can use a 1D load for simplicity.
        coalesced_1d: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])

        a_scale_base = a_scale_ptr + pid_m * a_stride_m + k // BLOCK_K * a_stride_k
        b_scale_base = b_scale_ptr + pid_n * b_stride_n + k // BLOCK_K * b_stride_k
        a_scale = gl.load(
            a_scale_base + gl.arange(0, BLOCK_M * SCALE_BLOCK_K, coalesced_1d)
        )
        b_scale = gl.load(
            b_scale_base + gl.arange(0, BLOCK_N * SCALE_BLOCK_K, coalesced_1d)
        )
        a_scale = a_scale.reshape(BLOCK_M, SCALE_BLOCK_K)
        b_scale = b_scale.reshape(BLOCK_N, SCALE_BLOCK_K)

        # ======= Begin unchanged code from `simple_mma_scaled_kernel` =======
        a_scale_layout: gl.constexpr = a_scale_tmem.get_reg_layout()
        b_scale_layout: gl.constexpr = b_scale_tmem.get_reg_layout()
        a_scale = gl.convert_layout(a_scale, a_scale_layout)
        b_scale = gl.convert_layout(b_scale, b_scale_layout)
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


def test_declared_contiguous_scale_tile_interface[
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
    AElements: IntVar,
    BElements: IntVar,
](
    a: gl.TmaDescriptorF8[M, K, BM, BK, LA],
    b: gl.TmaDescriptorF8[N, K, BN, BK, LB],
    wrong_b_host_k: gl.TmaDescriptorF8[N, Other, BN, BK, LB],
    wrong_b_tile_k: gl.TmaDescriptorF8[N, K, BN, Other, LB],
    c: gl.WgmmaDescriptorF16[M, N, BM, BN, LC],
    wrong_c_host_n: gl.WgmmaDescriptorF16[M, Other, BM, BN, LC],
    a_scales: gl.ContiguousScalePointer1D[AElements, BM, BK // 32],
    b_scales: gl.ContiguousScalePointer1D[BElements, BN, BK // 32],
    wrong_a_scale_rows: gl.ContiguousScalePointer1D[AElements, Other, BK // 32],
    wrong_b_scale_tile_k: gl.ContiguousScalePointer1D[BElements, BN, Other],
    wrong_a_host_elements: gl.ContiguousScalePointer1D[Other, BM, BK // 32],
    vec: Literal[32],
    wrong_vec: Literal[16],
) -> None:
    mma_scaled_contig_kernel(a, b, c, a_scales, b_scales, vec)
    mma_scaled_contig_kernel(
        a,
        wrong_b_host_k,  # E: is not assignable
        c,
        a_scales,
        b_scales,
        vec,
    )
    mma_scaled_contig_kernel(
        a,
        wrong_b_tile_k,  # E: is not assignable
        c,
        a_scales,
        b_scales,
        vec,
    )
    mma_scaled_contig_kernel(
        a,
        b,
        wrong_c_host_n,  # E: is not assignable
        a_scales,
        b_scales,
        vec,
    )
    mma_scaled_contig_kernel(
        a,
        b,
        c,
        wrong_a_scale_rows,  # E: is not assignable
        b_scales,
        vec,
    )
    mma_scaled_contig_kernel(
        a,
        b,
        c,
        a_scales,
        wrong_b_scale_tile_k,  # E: is not assignable
        vec,
    )
    mma_scaled_contig_kernel(
        a,
        b,
        c,
        a_scales,
        b_scales,
        wrong_vec,  # E: is not assignable
    )
    # The full allocation length is independent of descriptors and vector size.
    mma_scaled_contig_kernel(a, b, c, wrong_a_host_elements, b_scales, vec)

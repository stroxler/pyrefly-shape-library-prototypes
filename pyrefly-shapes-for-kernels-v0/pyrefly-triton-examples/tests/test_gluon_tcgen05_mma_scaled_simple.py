# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; semantic parameter annotations are not executable Gluon annotations.
# @lint-ignore-every AUTODEPS2
# @lint-ignore-every SPELL

"""Check the source-identical simple scaled-MMA kernel's FP8 interface."""

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
    TensorMemoryScaleTile,
    TensorMemoryTileF32,
    tma,
)


@gluon.jit
def simple_mma_scaled_kernel[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    ScaleK: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    LA: IntVar,
    LB: IntVar,
    LC: IntVar,
    AScaleRowStride: IntVar,
    AScaleColStride: IntVar,
    BScaleRowStride: IntVar,
    BScaleColStride: IntVar,
    Vec: IntVar,
](
    a_desc: gl.TmaDescriptorF8[M, K, BM, BK, LA],
    b_desc: gl.TmaDescriptorF8[N, K, BN, BK, LB],
    c_desc: gl.WgmmaDescriptorF16[M, N, BM, BN, LC],
    a_scale_ptr: gl.ScalePointer2D[M, ScaleK, AScaleRowStride, AScaleColStride],
    a_scale_stride_m: Int[AScaleRowStride],
    a_scale_stride_k: Int[AScaleColStride],
    b_scale_ptr: gl.ScalePointer2D[N, ScaleK, BScaleRowStride, BScaleColStride],
    b_scale_stride_n: Int[BScaleRowStride],
    b_scale_stride_k: Int[BScaleColStride],
    VEC_SIZE: Int[Vec],
):
    # If the operand dtype is fp4, they will be packed into uint8.
    A_IS_FP4: gl.constexpr = a_desc.dtype == gl.uint8
    B_IS_FP4: gl.constexpr = b_desc.dtype == gl.uint8
    # fp4 is a sub-byte dtype, so we need to account for this when loading the
    # operands from a uint8 tensor descriptor.
    A_ELEM_PER_BYTE: gl.constexpr = 2 if A_IS_FP4 else 1
    B_ELEM_PER_BYTE: gl.constexpr = 2 if B_IS_FP4 else 1

    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    # BLOCK_K represents the number of actual elements along K.
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1] * A_ELEM_PER_BYTE
    K = a_desc.shape[1] * A_ELEM_PER_BYTE

    # Allocate shared memory for the operands.
    a_smem = gl.allocate_shared_memory(
        a_desc.dtype, a_desc.block_type.shape, a_desc.layout
    )
    b_smem = gl.allocate_shared_memory(
        b_desc.dtype, b_desc.block_type.shape, b_desc.layout
    )

    # Allocate tensor memory for the scales. The scales must have the layout
    # `TensorMemoryScalesLayout`. Note that the B scales are always passed to
    # `tcgen05_mma_scaled` as [BLOCK_N, BLOCK_K // VEC_SIZE].
    scale_layout: gl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(
        a_scale_ptr.dtype.element_ty, [BLOCK_M, BLOCK_K // VEC_SIZE], scale_layout
    )
    b_scale_tmem = allocate_tensor_memory(
        b_scale_ptr.dtype.element_ty, [BLOCK_N, BLOCK_K // VEC_SIZE], scale_layout
    )

    # Allocate tensor memory for the accumulator.
    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
    use_acc = False

    # Allocate a barrier to track the operand loads and MMA.
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
        # BLOCK_K is the number of logical elements along K to load in a tile.
        # For sub-byte dtypes like fp4, translate them into uint8 offset.
        off_k_a = k // A_ELEM_PER_BYTE
        off_k_b = k // B_ELEM_PER_BYTE

        # Load the A and B tiles.
        mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_load(a_desc, [off_m, off_k_a], bar, a_smem)
        tma.async_load(b_desc, [off_n, off_k_b], bar, b_smem)
        mbarrier.wait(bar, phase)

        # Load the scales. We must always feed `b_scales` into `tcgen05_mma_scaled`
        # as [BLOCK_N, BLOCK_K // VEC_SIZE].
        coalesced_2d_layout: gl.constexpr = gl.BlockedLayout(
            [1, 1], [1, 32], [1, gl.num_warps()], [1, 0]
        )

        # Compute the right offsets by dividing the offset along K by VEC_SIZE.
        a_scale_offs_m = off_m + gl.arange(
            0, BLOCK_M, layout=gl.SliceLayout(1, coalesced_2d_layout)
        )
        a_scale_offs_k = k // VEC_SIZE + gl.arange(
            0, BLOCK_K // VEC_SIZE, layout=gl.SliceLayout(0, coalesced_2d_layout)
        )
        a_scale = gl.load(
            a_scale_ptr
            + a_scale_offs_m[:, None] * a_scale_stride_m
            + a_scale_offs_k[None, :] * a_scale_stride_k
        )

        b_scale_offs_n = off_n + gl.arange(
            0, BLOCK_N, layout=gl.SliceLayout(1, coalesced_2d_layout)
        )
        b_scale_offs_k = k // VEC_SIZE + gl.arange(
            0, BLOCK_K // VEC_SIZE, layout=gl.SliceLayout(0, coalesced_2d_layout)
        )
        b_scale = gl.load(
            b_scale_ptr
            + b_scale_offs_n[:, None] * b_scale_stride_n
            + b_scale_offs_k[None, :] * b_scale_stride_k
        )

        # We have to write the scales to tensor memory. Convert them into a the right
        # layout so we can write into tensor memory with layout `TensorMemoryScalesLayout`.
        a_scale_layout: gl.constexpr = a_scale_tmem.get_reg_layout()
        b_scale_layout: gl.constexpr = b_scale_tmem.get_reg_layout()
        a_scale = gl.convert_layout(a_scale, a_scale_layout)
        b_scale = gl.convert_layout(b_scale, b_scale_layout)
        a_scale_tmem.store(a_scale)
        b_scale_tmem.store(b_scale)

        # Pass the operand and scale tensors to `tcgen05_mma_scaled` along with the right
        # operand format strings.
        a_format: gl.constexpr = "e2m1" if A_IS_FP4 else "e4m3"
        b_format: gl.constexpr = "e2m1" if B_IS_FP4 else "e4m3"
        # Pass the operand and scale tensors to `tcgen05_mma_scaled` along with the right
        # operand format strings. Accumulate in-place with `use_acc`, which is set to False
        # on the first iteration to zero-initialize the accumulator. The B operand must be
        # transposed in shared memory.
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
        # Commit the MMA and wait for it to complete.
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, phase)
        use_acc = True
        phase ^= 1

    # Make sure to invalidate the barriers after we are done with them to avoid
    # race conditions and memory corruption errors. This is especially important
    # because a few lines below we are allocating shared memory for the async TMA
    # store of the accumulator. Re-using mbarrier shared memory without calling
    # `invalidate` is undefined behaviour.
    mbarrier.invalidate(bar)
    mbarrier.invalidate(mma_bar)

    # Load the accumulator tile from tensor memory and convert it to the output dtype.
    acc = acc_tmem.load()
    acc = acc.to(c_desc.dtype)

    # Write the accumulator via TMA store.
    acc_smem = gl.allocate_shared_memory(
        c_desc.dtype, c_desc.block_type.shape, c_desc.layout
    )
    acc_smem.store(acc)
    fence_async_shared()
    tma.async_store(c_desc, [off_m, off_n], acc_smem)
    tma.store_wait(0)


def test_declared_fp8_interface[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    Other: IntVar,
    ScaleK: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    LA: IntVar,
    LB: IntVar,
    LC: IntVar,
    AR: IntVar,
    AC: IntVar,
    BR: IntVar,
    BC: IntVar,
    Vec: IntVar,
](
    a: gl.TmaDescriptorF8[M, K, BM, BK, LA],
    b: gl.TmaDescriptorF8[N, K, BN, BK, LB],
    wrong_b_host_k: gl.TmaDescriptorF8[N, Other, BN, BK, LB],
    wrong_b_tile_k: gl.TmaDescriptorF8[N, K, BN, Other, LB],
    c: gl.WgmmaDescriptorF16[M, N, BM, BN, LC],
    wrong_c_host_n: gl.WgmmaDescriptorF16[M, Other, BM, BN, LC],
    a_scale: gl.ScalePointer2D[M, ScaleK, AR, AC],
    b_scale: gl.ScalePointer2D[N, ScaleK, BR, BC],
    wrong_a_scale_rows: gl.ScalePointer2D[Other, ScaleK, AR, AC],
    wrong_a_scale_cols: gl.ScalePointer2D[M, Other, AR, AC],
    wrong_b_scale_cols: gl.ScalePointer2D[N, Other, BR, BC],
    a_row: Int[AR],
    a_col: Int[AC],
    b_row: Int[BR],
    b_col: Int[BC],
    wrong_a_row: Int[Other],
    vec: Int[Vec],
) -> None:
    simple_mma_scaled_kernel(a, b, c, a_scale, a_row, a_col, b_scale, b_row, b_col, vec)
    simple_mma_scaled_kernel(
        a,
        wrong_b_host_k,  # E: is not assignable
        c,
        a_scale,
        a_row,
        a_col,
        b_scale,
        b_row,
        b_col,
        vec,
    )
    simple_mma_scaled_kernel(
        a,
        wrong_b_tile_k,  # E: is not assignable
        c,
        a_scale,
        a_row,
        a_col,
        b_scale,
        b_row,
        b_col,
        vec,
    )
    simple_mma_scaled_kernel(
        a,
        b,
        wrong_c_host_n,  # E: is not assignable
        a_scale,
        a_row,
        a_col,
        b_scale,
        b_row,
        b_col,
        vec,
    )
    simple_mma_scaled_kernel(
        a,
        b,
        c,
        wrong_a_scale_rows,  # E: is not assignable
        a_row,
        a_col,
        b_scale,
        b_row,
        b_col,
        vec,
    )
    simple_mma_scaled_kernel(
        a,
        b,
        c,
        a_scale,
        wrong_a_row,  # E: is not assignable
        a_col,
        b_scale,
        b_row,
        b_col,
        vec,
    )
    simple_mma_scaled_kernel(
        a,
        b,
        c,
        a_scale,
        a_row,
        a_col,
        wrong_b_scale_cols,  # E: is not assignable
        b_row,
        b_col,
        vec,
    )
    # Both scale buffers can claim the same *wrong* full K axis: no K/Vec link.
    simple_mma_scaled_kernel(
        a,
        b,
        c,
        wrong_a_scale_cols,
        a_row,
        a_col,
        wrong_b_scale_cols,
        b_row,
        b_col,
        vec,
    )


def test_scaled_mma_intrinsic_tile_contract[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    Other: IntVar,
    ScaleK: IntVar,
    LA: IntVar,
    LB: IntVar,
](
    a: gl.WgmmaSharedF8[M, K, LA],
    b: gl.WgmmaSharedF8[N, K, LB],
    wrong_b_k: gl.WgmmaSharedF8[N, Other, LB],
    acc: TensorMemoryTileF32[M, N],
    wrong_acc_n: TensorMemoryTileF32[M, Other],
    a_scales: TensorMemoryScaleTile[M, ScaleK],
    b_scales: TensorMemoryScaleTile[N, ScaleK],
    wrong_scale_n: TensorMemoryScaleTile[Other, ScaleK],
    wrong_scale_k: TensorMemoryScaleTile[N, Other],
) -> None:
    tcgen05_mma_scaled(a, b.permute((1, 0)), acc, a_scales, b_scales, "e4m3", "e4m3")
    tcgen05_mma_scaled(
        a,
        wrong_b_k.permute((1, 0)),  # E: is not assignable
        acc,
        a_scales,
        b_scales,
        "e4m3",
        "e4m3",
    )
    tcgen05_mma_scaled(
        a,
        b.permute((1, 0)),
        wrong_acc_n,  # E: is not assignable
        a_scales,
        b_scales,
        "e4m3",
        "e4m3",
    )
    tcgen05_mma_scaled(
        a,
        b.permute((1, 0)),
        acc,
        a_scales,
        wrong_scale_n,  # E: is not assignable
        "e4m3",
        "e4m3",
    )
    tcgen05_mma_scaled(
        a,
        b.permute((1, 0)),
        acc,
        a_scales,
        wrong_scale_k,  # E: is not assignable
        "e4m3",
        "e4m3",
    )

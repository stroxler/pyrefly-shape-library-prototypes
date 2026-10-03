# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; semantic annotations are not executable Gluon annotations.
# @lint-ignore-every AUTODEPS2

"""Check original Blackwell blocked matmul without loosening MMA contraction."""

from typing import assert_type, Literal

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
    TensorMemoryTileF32,
    tma,
)


@gluon.jit
def blocked_matmul_kernel[
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
    TRANSPOSE_B: Literal[False],
    num_warps: int,
):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    # The block of C this program is processing is (pid_m, pid_n).
    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    a_smem = gl.allocate_shared_memory(dtype, a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(dtype, b_desc.block_type.shape, b_desc.layout)

    tma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(tma_bar, count=1)
    mma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(mma_bar, count=1)
    phase = 0

    # Determine the TMEM layout.
    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)

    # We can zero-initialize the accumulator by setting `use_acc=False` on the
    # first iteration.
    use_acc = False
    for k in range(0, K, BLOCK_K):
        mbarrier.expect(tma_bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_load(a_desc, [off_m, k], tma_bar, a_smem)
        tma.async_load(
            b_desc, [off_n, k] if TRANSPOSE_B else [k, off_n], tma_bar, b_smem
        )
        mbarrier.wait(tma_bar, phase=phase)

        # We can transpose B by creating a transposed view over tile of B in
        # shared memory. This forwards the transposition to tcgen05_mma, which
        # handles it for us.
        if TRANSPOSE_B:
            b = b_smem.permute((1, 0))
        else:
            b = b_smem

        # Issue and wait on the tcgen05_mma.
        tcgen05_mma(a_smem, b, acc_tmem, use_acc=use_acc)  # E: is not assignable
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, phase=phase)
        use_acc = True

        phase ^= 1  # toggle the parity phase between 0 and 1

    mbarrier.invalidate(tma_bar)
    mbarrier.invalidate(mma_bar)

    acc = acc_tmem.load()

    # Downcast accumulator and store tile of C.
    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    c_smem.store(acc.to(dtype))
    fence_async_shared()
    tma.async_store(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


def test_blocked_interface[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Other: IntVar,
    LA: IntVar,
    LB: IntVar,
    LC: IntVar,
](
    a: gl.TmaInputDescriptorF16[M, K, BM, BK, LA],
    b: gl.TmaInputDescriptorF16[K, N, BK, BN, LB],
    c: gl.TmaOutputDescriptorF16[M, N, BM, BN, LC],
    wrong_b_extent: gl.TmaInputDescriptorF16[Other, N, BK, BN, LB],
    wrong_c_extent: gl.TmaOutputDescriptorF16[M, Other, BM, BN, LC],
    wrong_b_tile: gl.TmaInputDescriptorF16[K, N, Other, BN, LB],
    transposed_b: gl.TmaInputDescriptorF16[N, K, BN, BK, LB],
    wrong_c_dtype: gl.TmaOutputDescriptor2D[M, N, BM, BN, LC],
) -> None:
    blocked_matmul_kernel(a, b, c, False, 4)
    blocked_matmul_kernel(
        a,
        wrong_b_extent,  # E: is not assignable
        c,
        False,
        4,
    )
    blocked_matmul_kernel(
        a,
        b,
        wrong_c_extent,  # E: is not assignable
        False,
        4,
    )
    blocked_matmul_kernel(a, wrong_b_tile, c, False, 4)  # E: is not assignable
    blocked_matmul_kernel(a, transposed_b, c, False, 4)  # E: is not assignable
    blocked_matmul_kernel(a, b, wrong_c_dtype, False, 4)  # E: is not assignable
    blocked_matmul_kernel(a, b, c, True, 4)  # E: is not assignable


def test_tma_conditional_coordinate_contract[
    Rows: IntVar,
    Cols: IntVar,
    BR: IntVar,
    BC: IntVar,
    Other: IntVar,
    Layout: IntVar,
](
    desc: gl.TmaInputDescriptorF16[Rows, Cols, BR, BC, Layout],
    destination: gl.WgmmaSharedF16[BR, BC, Layout],
    barrier: gl.BarrierBuffer1D,
    row: gl.GluonTileStart[BR],
    col: gl.GluonTileStart[BC],
    wrong_start: gl.GluonTileStart[Other],
) -> None:
    tma.async_load(desc, [row, col], barrier, destination)
    tma.async_load(  # E: No matching overload
        desc, [wrong_start, col], barrier, destination
    )
    # The covariant coordinate element type cannot validate rank or ordering.
    tma.async_load(desc, [], barrier, destination)
    tma.async_load(desc, [col, row], barrier, destination)


def test_transposed_b_mma_primitive[
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    LA: IntVar,
    LB: IntVar,
](
    a: gl.WgmmaSharedF16[BM, BK, LA],
    b_transposed: gl.WgmmaSharedF16[BN, BK, LB],
    accumulator: TensorMemoryTileF32[BM, BN],
) -> None:
    tcgen05_mma(a, b_transposed.permute((1, 0)), accumulator)
    tcgen05_mma(a, b_transposed, accumulator)  # E: is not assignable


def test_tmem_layout_block_is_not_linked[BM: IntVar, BN: IntVar, Other: IntVar](
    bm: Int[BM], bn: Int[BN], other: Int[Other]
) -> None:
    layout = TensorMemoryLayout([bm, other], col_stride=1)
    assert_type(
        allocate_tensor_memory(gl.float32, [bm, bn], layout),
        TensorMemoryTileF32[BM, BN],
    )

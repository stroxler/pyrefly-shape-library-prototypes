# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only, not executable Gluon kernels.
# @lint-ignore-every AUTODEPS2

"""Check the original blocked WGMMA body specialized for untransposed B."""

from typing import assert_type, Literal

import triton
from shape_extensions import IntVar
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_wait,
)
from triton.language import tensor


@gluon.constexpr_function
def get_warps_per_cta(BLOCK_M, BLOCK_N, num_warps):
    warps_per_cta = [4, 1]
    m = 16
    # Tile the atom until we have enough warps.
    while warps_per_cta[0] * warps_per_cta[1] != num_warps:
        # Tile along M only if it would not cause broadcasting.
        if BLOCK_M > m * warps_per_cta[0]:
            warps_per_cta[0] *= 2
        else:
            warps_per_cta[1] *= 2
    return warps_per_cta


@gluon.constexpr_function
def get_instr_shape_n(BLOCK_M, BLOCK_N, num_warps):
    m = 16
    mReps = triton.cdiv(BLOCK_M, m)
    nReps = triton.cdiv(num_warps, mReps)
    maxN = max(BLOCK_N // nReps, 8)
    n = 256
    while n > maxN or BLOCK_N % n != 0:
        n -= 8
    assert n >= 8, "expected to find a valid n"
    return n


@gluon.constexpr_function
def pick_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps):
    m = 16
    k = 256 // dtype.primitive_bitwidth
    n = get_instr_shape_n(BLOCK_M, BLOCK_N, num_warps)
    warps_per_cta = get_warps_per_cta(BLOCK_M, BLOCK_N, num_warps)
    return gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=warps_per_cta,
        instr_shape=[m, n, k],
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

    a_smem = gl.allocate_shared_memory(dtype, a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(dtype, b_desc.block_type.shape, b_desc.layout)

    # The block of C this program is processing is (pid_m, pid_n).
    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    # Determine the WGMMA layout.
    mma_layout: gl.constexpr = pick_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    phase = 0

    for k in range(0, K, BLOCK_K):
        # Load tiles of A and B.
        mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_load(a_desc, [off_m, k], bar, a_smem)
        if TRANSPOSE_B:
            tma.async_load(  # E: This code is unreachable
                b_desc, [off_n, k], bar, b_smem
            )
        else:
            tma.async_load(b_desc, [k, off_n], bar, b_smem)
        mbarrier.wait(bar, phase=phase)
        phase ^= 1  # toggle the parity phase between 0 and 1

        # We can transpose B by creating a transposed view over tile of B in
        # shared memory. This forwards the transposition to WGMMA, which handles
        # it for us.
        if TRANSPOSE_B:
            b = b_smem.permute((1, 0))
        else:
            b = b_smem

        acc = warpgroup_mma(a_smem, b, acc, is_async=True)  # E: is not assignable
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))

    mbarrier.invalidate(bar)

    # Downcast accumulator and store tile of C.
    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    c_smem.store(acc.to(dtype))
    fence_async_shared()
    tma.async_store(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


def test_orientation_and_tile_contract[
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
](
    a: gl.TmaInputDescriptorF16[M, K, BM, BK, LA],
    b: gl.TmaInputDescriptorF16[K, N, BK, BN, LB],
    transposed_b: gl.TmaInputDescriptorF16[N, K, BN, BK, LB],
    wrong_b: gl.TmaInputDescriptorF16[Other, N, Other, BN, LB],
    output: gl.TmaOutputDescriptorF16[M, N, BM, BN, LC],
    wrong_output: gl.TmaOutputDescriptorF16[M, Other, BM, Other, LC],
    wrong_output_role: gl.TmaInputDescriptorF16[M, N, BM, BN, LC],
    left_tile: gl.WgmmaSharedF16[BM, BK, LA],
    right_tile: gl.WgmmaSharedF16[BK, BN, LB],
    transposed_tile: gl.WgmmaSharedF16[BN, BK, LB],
    acc: tensor[[BM, BN]],
) -> None:
    blocked_matmul_kernel(a, b, output, False, 4)
    blocked_matmul_kernel(a, wrong_b, output, False, 4)  # E: is not assignable
    blocked_matmul_kernel(a, b, wrong_output, False, 4)  # E: is not assignable
    blocked_matmul_kernel(a, b, wrong_output_role, False, 4)  # E: is not assignable
    blocked_matmul_kernel(a, transposed_b, output, False, 4)  # E: is not assignable
    blocked_matmul_kernel(a, b, output, True, 4)  # E: is not assignable

    assert_type(
        warpgroup_mma(left_tile, right_tile, acc, is_async=True), tensor[[BM, BN]]
    )
    assert_type(
        warpgroup_mma(left_tile, transposed_tile.permute((1, 0)), acc, is_async=True),
        tensor[[BM, BN]],
    )
    warpgroup_mma(
        left_tile,
        transposed_tile,  # E: is not assignable
        acc,
        is_async=True,
    )

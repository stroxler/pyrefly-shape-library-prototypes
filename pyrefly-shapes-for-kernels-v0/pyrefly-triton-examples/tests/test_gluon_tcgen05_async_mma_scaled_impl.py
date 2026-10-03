# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; semantic parameter annotations are not executable Gluon annotations.
# @lint-ignore-every AUTODEPS2
# @lint-ignore-every SPELL

"""Inspect the source-identical packed-scale async MMA helper boundary."""

from typing import assert_type, Literal

from shape_extensions import Int, IntVar
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    allocate_tensor_memory,
    tcgen05_copy,
    tcgen05_mma_scaled,
    TensorMemoryScalesLayout,
    TensorMemoryTileF32,
)


@gluon.jit
def unswizzle_scales_shared_memory[Rows: IntVar, K: IntVar, Layout: IntVar](
    smem: gl.PackedScaleSharedTile[Rows, K, Layout],
    BLOCK_MN: Int[Rows],
    BLOCK_K: Int[K],
    VEC_SIZE: Literal[32],
):
    smem = smem.reshape(  # E: is not assignable
        (smem.shape[1], smem.shape[2], 32, 4, 4)
    )
    smem = smem.permute((0, 3, 2, 1, 4))  # E: no attribute `permute`
    return smem.reshape((BLOCK_MN, BLOCK_K // VEC_SIZE))  # E: is not assignable


@gluon.jit
def async_mma_scaled_impl[
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    LA: IntVar,
    LB: IntVar,
    LSA: IntVar,
    LSB: IntVar,
](
    a_smem: gl.WgmmaSharedF8[BM, BK, LA],
    b_smem: gl.WgmmaSharedF8[BN, BK, LB],
    a_scale_smem: gl.PackedScaleSharedTile[BM, BK, LSA],
    b_scale_smem: gl.PackedScaleSharedTile[BN, BK, LSB],
    acc_tmem: TensorMemoryTileF32[BM, BN],
    use_acc: bool,
    pred: bool,
):
    A_ELEM_PER_BYTE: gl.constexpr = 2 if a_smem.dtype == gl.uint8 else 1
    BLOCK_M: gl.constexpr = a_smem.shape[0]
    BLOCK_N: gl.constexpr = b_smem.shape[0]
    BLOCK_K: gl.constexpr = a_smem.shape[1] * A_ELEM_PER_BYTE
    # Recall we use `uint8` to represent fp4 elements.
    VEC_SIZE: gl.constexpr = 32 if a_scale_smem.dtype == gl.uint8 else 16

    a_scale = unswizzle_scales_shared_memory(a_scale_smem, BLOCK_M, BLOCK_K, VEC_SIZE)
    b_scale = unswizzle_scales_shared_memory(b_scale_smem, BLOCK_N, BLOCK_K, VEC_SIZE)

    # We don't need to hoist the scales tensor memory allocations outside of the loop,
    # so we can pull them into this helper function.
    scale_layout: gl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(
        a_scale.dtype,  # E: no attribute `dtype`
        a_scale.type.shape,  # E: no attribute `type`
        scale_layout,
    )
    b_scale_tmem = allocate_tensor_memory(
        b_scale.dtype,  # E: no attribute `dtype`
        b_scale.type.shape,  # E: no attribute `type`
        scale_layout,
    )
    tcgen05_copy(a_scale, a_scale_tmem)  # E: is not assignable # E: is not assignable
    tcgen05_copy(b_scale, b_scale_tmem)  # E: is not assignable # E: is not assignable

    a_format: gl.constexpr = "e2m1" if a_smem.dtype == gl.uint8 else "e4m3"
    b_format: gl.constexpr = "e2m1" if b_smem.dtype == gl.uint8 else "e4m3"
    tcgen05_mma_scaled(
        a_smem,
        b_smem.permute((1, 0)),
        acc_tmem,
        a_scale_tmem,
        b_scale_tmem,
        a_format,
        b_format,
        use_acc=use_acc,
        pred=pred,
    )


def test_mxfp8_dtype_branch() -> None:
    assert_type(gl.float8e4nv == gl.uint8, Literal[False])
    assert_type(gl.uint8 == gl.uint8, Literal[True])


def test_async_scaled_declared_interface[
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Other: IntVar,
    LA: IntVar,
    LB: IntVar,
    LSA: IntVar,
    LSB: IntVar,
](
    a: gl.WgmmaSharedF8[BM, BK, LA],
    b: gl.WgmmaSharedF8[BN, BK, LB],
    wrong_b_k: gl.WgmmaSharedF8[BN, Other, LB],
    a_scale: gl.PackedScaleSharedTile[BM, BK, LSA],
    b_scale: gl.PackedScaleSharedTile[BN, BK, LSB],
    wrong_a_scale_rows: gl.PackedScaleSharedTile[Other, BK, LSA],
    wrong_b_scale_k: gl.PackedScaleSharedTile[BN, Other, LSB],
    acc: TensorMemoryTileF32[BM, BN],
    wrong_acc_n: TensorMemoryTileF32[BM, Other],
) -> None:
    async_mma_scaled_impl(a, b, a_scale, b_scale, acc, False, True)
    async_mma_scaled_impl(
        a,
        wrong_b_k,  # E: is not assignable
        a_scale,
        b_scale,
        acc,
        False,
        True,
    )
    async_mma_scaled_impl(
        a,
        b,
        wrong_a_scale_rows,  # E: is not assignable
        b_scale,
        acc,
        False,
        True,
    )
    async_mma_scaled_impl(
        a,
        b,
        a_scale,
        wrong_b_scale_k,  # E: is not assignable
        acc,
        False,
        True,
    )
    async_mma_scaled_impl(
        a,
        b,
        a_scale,
        b_scale,
        wrong_acc_n,  # E: is not assignable
        False,
        True,
    )

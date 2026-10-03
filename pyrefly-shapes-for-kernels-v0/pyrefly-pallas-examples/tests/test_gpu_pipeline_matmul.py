# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The GPU kernel and nested pipeline callback come from JAX
# docs/pallas/gpu/pipelining.md (Apache-2.0); only their parameter types are new.
# @lint-ignore-every AUTODEPS2

"""Hopper GMEM/SMEM/ACC refs connect a float16 matmul host boundary."""

from __future__ import annotations

from collections.abc import Callable
from typing import assert_type, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar


def matmul[
    Rows: IntVar,
    Inner: IntVar,
    Cols: IntVar,
    TM: IntVar,
    TN: IntVar,
    Swizzle: IntVar,
](
    a: jax.Float16Array[[Rows, Inner]],
    b: jax.Float16Array[[Inner, Cols]],
    tile_m: Int[TM],
    tile_n: Int[TN],
    swizzle: Int[Swizzle],
) -> jax.Array[[Rows, Cols]]:
    m, k = a.shape
    _, n = b.shape
    dtype = jnp.float16
    swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
    tile_k = swizzle_elems
    grid_m = m // tile_m
    grid_k = k // tile_k
    grid_n = n // tile_n
    assert tile_m % swizzle_elems == 0

    # Note: Transforms will be inferred automatically
    # by Mosaic GPU in the future.
    transforms = (
        plgpu.TilingTransform((8, swizzle_elems)),
        plgpu.SwizzleTransform(swizzle),
    )

    def kernel(
        a_gmem: plgpu.GmemInRef[Rows, Inner],
        b_gmem: plgpu.GmemInRef[Inner, Cols],
        o_gmem: plgpu.GmemOutRef[Rows, Cols],
        o_smem: plgpu.SmemScratchRef[TM, TN],
        acc: plgpu.AccRef[TM, TN],
    ) -> None:
        def pipeline_step(
            _: int,
            a_smem: plgpu.SmemInRef[TM, Swizzle // 2],
            b_smem: plgpu.SmemInRef[Swizzle // 2, TN],
        ) -> None:
            plgpu.wgmma(acc, a_smem, b_smem)
            plgpu.wgmma_wait(1)

        # pl.program_id obtains the index into the grid.
        pid_m = pl.program_id(0)
        pid_n = pl.program_id(1)

        pipeline = plgpu.emit_pipeline(
            pipeline_step,
            in_specs=[
                plgpu.BlockSpec(
                    (tile_m, tile_k), lambda k: (pid_m, k), transforms=transforms
                ),
                plgpu.BlockSpec(
                    (tile_k, tile_n), lambda k: (k, pid_n), transforms=transforms
                ),
            ],
            grid=(grid_k,),
            max_concurrent_steps=2,
            delay_release=1,
        )

        pipeline(a_gmem, b_gmem)
        # Store WGMMA accumulator to SMEM and then to GMEM.
        o_smem[...] = acc[...].astype(dtype)
        plgpu.commit_smem()
        m_slice = pl.ds(pid_m * tile_m, tile_m)
        n_slice = pl.ds(pid_n * tile_n, tile_n)
        plgpu.copy_smem_to_gmem(o_smem, o_gmem.at[m_slice, n_slice])
        plgpu.wait_smem_to_gmem(0)

    return plgpu.kernel(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float16),
        scratch_shapes=dict(
            o_smem=plgpu.SMEM((tile_m, tile_n), jnp.float16),
            acc=plgpu.ACC((tile_m, tile_n), jnp.float32),
        ),
        # grid specifies the CUDA grid.
        # Instances of `kernel` will be executed in parallel over this grid.
        grid=(grid_m, grid_n),
        grid_names=("m", "n"),
    )(a, b)


def test_host_result[
    Rows: IntVar,
    Inner: IntVar,
    Cols: IntVar,
    TM: IntVar,
    TN: IntVar,
    Swizzle: IntVar,
](
    a: jax.Float16Array[[Rows, Inner]],
    b: jax.Float16Array[[Inner, Cols]],
    tm: Int[TM],
    tn: Int[TN],
    swizzle: Int[Swizzle],
) -> None:
    assert_type(matmul(a, b, tm, tn, swizzle), jax.Array[[Rows, Cols]])


def test_wrong_host_contracting_extent[
    Rows: IntVar,
    Inner: IntVar,
    Other: IntVar,
    Cols: IntVar,
    TM: IntVar,
    TN: IntVar,
    Swizzle: IntVar,
](
    a: jax.Float16Array[[Rows, Inner]],
    b: jax.Float16Array[[Other, Cols]],
    tm: Int[TM],
    tn: Int[TN],
    swizzle: Int[Swizzle],
) -> None:
    matmul(a, b, tm, tn, swizzle)  # E: is not assignable


def test_wrong_host_dtype[
    Rows: IntVar,
    Inner: IntVar,
    Cols: IntVar,
    TM: IntVar,
    TN: IntVar,
    Swizzle: IntVar,
](
    a: jax.Array[[Rows, Inner]],
    b: jax.Float16Array[[Inner, Cols]],
    tm: Int[TM],
    tn: Int[TN],
    swizzle: Int[Swizzle],
) -> None:
    matmul(a, b, tm, tn, swizzle)  # E: is not assignable


def test_explicit_host_float16_conversion[
    Rows: IntVar,
    Inner: IntVar,
    Cols: IntVar,
    TM: IntVar,
    TN: IntVar,
    Swizzle: IntVar,
](
    a: jax.Array[[Rows, Inner]],
    b: jax.Array[[Inner, Cols]],
    tm: Int[TM],
    tn: Int[TN],
    swizzle: Int[Swizzle],
) -> None:
    assert_type(
        matmul(a.astype(jnp.float16), b.astype(jnp.float16), tm, tn, swizzle),
        jax.Array[[Rows, Cols]],
    )


def test_wrong_wgmma_inner[TM: IntVar, TK: IntVar, TN: IntVar, Other: IntVar](
    accum: plgpu.AccRef[TM, TN],
    lhs: plgpu.SmemInRef[TM, TK],
    rhs: plgpu.SmemInRef[Other, TN],
) -> None:
    plgpu.wgmma(accum, lhs, rhs)  # E: is not assignable


def test_wrong_accumulator_width[TM: IntVar, TK: IntVar, TN: IntVar, Other: IntVar](
    accum: plgpu.AccRef[TM, Other],
    lhs: plgpu.SmemInRef[TM, TK],
    rhs: plgpu.SmemInRef[TK, TN],
) -> None:
    plgpu.wgmma(accum, lhs, rhs)  # E: is not assignable


def test_wrong_gmem_store_width[
    Rows: IntVar,
    Cols: IntVar,
    TM: IntVar,
    TN: IntVar,
    Other: IntVar,
](
    scratch: plgpu.SmemScratchRef[TM, TN],
    output: plgpu.GmemOutRef[Rows, Cols],
    row: pl.HalfRowSlice[TM],
    col: pl.HalfRowSlice[Other],
) -> None:
    plgpu.copy_smem_to_gmem(scratch, output.at[row, col])  # E: is not assignable


def test_wrong_pipeline_rhs_tile[TM: IntVar, TK: IntVar, TN: IntVar, Other: IntVar](
    step: Callable[[int, plgpu.SmemInRef[TM, TK], plgpu.SmemInRef[Other, TN]], None],
    lhs: plgpu.BlockSpec[TM, TK],
    rhs: plgpu.BlockSpec[TK, TN],
) -> None:
    plgpu.emit_pipeline(
        step,  # E: is not assignable
        in_specs=[lhs, rhs],
        grid=(1,),
        max_concurrent_steps=2,
        delay_release=1,
    )


def test_wrong_output_allocation[
    Rows: IntVar,
    Inner: IntVar,
    Cols: IntVar,
    TM: IntVar,
    TN: IntVar,
    Other: IntVar,
](
    body: Callable[
        [
            plgpu.GmemInRef[Rows, Inner],
            plgpu.GmemInRef[Inner, Cols],
            plgpu.GmemOutRef[Rows, Cols],
            plgpu.SmemScratchRef[TM, TN],
            plgpu.AccRef[TM, TN],
        ],
        None,
    ],
    scratch: plgpu.SMEM[TM, TN],
    accumulator: plgpu.ACC[TM, TN],
    rows: Int[Rows],
    wrong_cols: Int[Other],
) -> None:
    plgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct(  # E: is not assignable
            (rows, wrong_cols), jnp.float16
        ),
        scratch_shapes={"o_smem": scratch, "acc": accumulator},
        grid=(1, 1),
        grid_names=("m", "n"),
    )


def test_wrong_output_dtype_is_not_rejected[
    Rows: IntVar,
    Inner: IntVar,
    Cols: IntVar,
    TM: IntVar,
    TN: IntVar,
](
    body: Callable[
        [
            plgpu.GmemInRef[Rows, Inner],
            plgpu.GmemInRef[Inner, Cols],
            plgpu.GmemOutRef[Rows, Cols],
            plgpu.SmemScratchRef[TM, TN],
            plgpu.AccRef[TM, TN],
        ],
        None,
    ],
    scratch: plgpu.SMEM[TM, TN],
    accumulator: plgpu.ACC[TM, TN],
    rows: Int[Rows],
    cols: Int[Cols],
) -> None:
    plgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((rows, cols), jnp.float32),
        scratch_shapes={"o_smem": scratch, "acc": accumulator},
        grid=(999, 999),
        grid_names=("m", "n"),
    )


if not TYPE_CHECKING:
    import unittest

    class GpuPipelineMatmulHostTest(unittest.TestCase):
        def test_host_matrix_product_and_tile_counts(self) -> None:
            a = jnp.arange(8, dtype=jnp.float16).reshape(4, 2)
            b = jnp.arange(6, dtype=jnp.float16).reshape(2, 3)
            self.assertEqual((a @ b).shape, (4, 3))
            self.assertEqual((4 // 2, 3 // 1), (2, 3))

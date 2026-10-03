# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Kernel body from JAX docs/pallas/tpu/matmul.md (Apache-2.0).
# Semantic annotations exist only in the stub overlay.
# @lint-ignore-every AUTODEPS2

"""Pallas matrix multiplication accumulates into a scratch Ref over the K grid."""

from __future__ import annotations

import unittest
from typing import assert_type, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar


def matmul_kernel[RB: IntVar, KB: IntVar, CB: IntVar, Steps: IntVar](
    x_ref: pl.InRef[[RB, KB]],
    y_ref: pl.InRef[[KB, CB]],
    z_ref: pl.OutRef[[RB, CB]],
    acc_ref: pl.AccumRef[[RB, CB]],
    *,
    nsteps: Int[Steps],
) -> None:
    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jnp.dot(x_ref[...], y_ref[...], preferred_element_type=jnp.float32)

    @pl.when(pl.program_id(2) == nsteps - 1)
    def _():
        z_ref[...] = acc_ref[...].astype(z_ref.dtype)


def scratch_matmul[
    Rows: IntVar,
    Inner: IntVar,
    Cols: IntVar,
    RB: IntVar,
    KB: IntVar,
    CB: IntVar,
](
    x: jax.Array[[Rows, Inner]],
    y: jax.Array[[Inner, Cols]],
    rows: Int[Rows],
    inner: Int[Inner],
    cols: Int[Cols],
    row_block: Int[RB],
    inner_block: Int[KB],
    col_block: Int[CB],
    dtype: object,
) -> jax.Array[[Rows, Cols]]:
    x_spec: pl.BlockSpec[[RB, KB]] = pl.BlockSpec(
        (row_block, inner_block), lambda i, j, k: (i, k)
    )
    y_spec: pl.BlockSpec[[KB, CB]] = pl.BlockSpec(
        (inner_block, col_block), lambda i, j, k: (k, j)
    )
    out_spec: pl.BlockSpec[[RB, CB]] = pl.BlockSpec(
        (row_block, col_block), lambda i, j, k: (i, j)
    )
    scratch: pltpu.VMEM[[RB, CB]] = pltpu.VMEM((row_block, col_block), jnp.float32)
    grid_spec: pltpu.PrefetchScalarGridSpec[Rows, Cols, RB, CB, Inner, KB] = (
        pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(rows // row_block, cols // col_block, inner // inner_block),
            in_specs=(x_spec, y_spec),
            out_specs=out_spec,
            scratch_shapes=(scratch,),
        )
    )
    nsteps = inner // inner_block

    def kernel(
        x_ref: pl.InRef[[RB, KB]],
        y_ref: pl.InRef[[KB, CB]],
        z_ref: pl.OutRef[[RB, CB]],
        acc_ref: pl.AccumRef[[RB, CB]],
    ) -> None:
        matmul_kernel(x_ref, y_ref, z_ref, acc_ref, nsteps=nsteps)

    call = pl.pallas_call(
        kernel,
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((rows, cols), dtype),
        interpret=True,
    )
    return call(x, y)


def test_wrong_contraction_axis[
    Rows: IntVar,
    Inner: IntVar,
    Other: IntVar,
    Cols: IntVar,
    RB: IntVar,
    KB: IntVar,
    CB: IntVar,
](
    x: jax.Array[[Rows, Inner]],
    y: jax.Array[[Other, Cols]],
    rows: Int[Rows],
    inner: Int[Inner],
    cols: Int[Cols],
    rb: Int[RB],
    kb: Int[KB],
    cb: Int[CB],
) -> None:
    scratch_matmul(x, y, rows, inner, cols, rb, kb, cb, None)  # E: is not assignable


def test_wrong_scratch_tile[RB: IntVar, KB: IntVar, CB: IntVar, Other: IntVar](
    x: pl.InRef[[RB, KB]],
    y: pl.InRef[[KB, CB]],
    out: pl.OutRef[[RB, CB]],
    scratch: pl.AccumRef[[RB, Other]],
) -> None:
    scratch[...] += jnp.dot(  # E: is not supported
        x[...], y[...], preferred_element_type=jnp.float32
    )


def test_wrong_dot_inner[RB: IntVar, KB: IntVar, CB: IntVar, Other: IntVar](
    x: pl.InRef[[RB, KB]],
    y: pl.InRef[[Other, CB]],
) -> None:
    jnp.dot(
        x[...],
        y[...],  # E: is not assignable
        preferred_element_type=jnp.float32,
    )


def test_wrong_initialization_shape[RB: IntVar, CB: IntVar, Other: IntVar](
    scratch: pl.AccumRef[[RB, CB]],
    wrong: pl.AccumRef[[RB, Other]],
) -> None:
    scratch[...] = jnp.zeros_like(wrong)  # E: Cannot set item


def test_wrong_grid_inner_block[
    Rows: IntVar,
    Inner: IntVar,
    Cols: IntVar,
    RB: IntVar,
    KB: IntVar,
    CB: IntVar,
    Other: IntVar,
](
    rows: Int[Rows],
    inner: Int[Inner],
    cols: Int[Cols],
    rb: Int[RB],
    wrong_kb: Int[Other],
    cb: Int[CB],
    x_spec: pl.BlockSpec[[RB, KB]],
    y_spec: pl.BlockSpec[[KB, CB]],
    out_spec: pl.BlockSpec[[RB, CB]],
    scratch: pltpu.VMEM[[RB, CB]],
) -> None:
    spec: pltpu.PrefetchScalarGridSpec[Rows, Cols, RB, CB, Inner, KB] = (
        pltpu.PrefetchScalarGridSpec(  # E: No matching overload
            num_scalar_prefetch=0,
            grid=(rows // rb, cols // cb, inner // wrong_kb),
            in_specs=(x_spec, y_spec),
            out_specs=out_spec,
            scratch_shapes=(scratch,),
        )
    )
    _ = spec


def test_wrong_scratch_allocation[RB: IntVar, CB: IntVar, Other: IntVar](
    rb: Int[RB], wrong_cb: Int[Other]
) -> None:
    scratch: pltpu.VMEM[[RB, CB]] = pltpu.VMEM(  # E: is not assignable
        (rb, wrong_cb), jnp.float32
    )
    _ = scratch


def test_wrong_host_output_cols[
    Rows: IntVar,
    Inner: IntVar,
    Cols: IntVar,
    Other: IntVar,
    RB: IntVar,
    KB: IntVar,
    CB: IntVar,
](
    x: jax.Array[[Rows, Inner]],
    y: jax.Array[[Inner, Cols]],
    rows: Int[Rows],
    inner: Int[Inner],
    wrong_cols: Int[Other],
    rb: Int[RB],
    kb: Int[KB],
    cb: Int[CB],
) -> None:
    scratch_matmul(
        x,
        y,
        rows,
        inner,
        wrong_cols,  # E: is not assignable
        rb,
        kb,
        cb,
        None,
    )


def test_output_shape[
    Rows: IntVar,
    Inner: IntVar,
    Cols: IntVar,
    RB: IntVar,
    KB: IntVar,
    CB: IntVar,
](
    x: jax.Array[[Rows, Inner]],
    y: jax.Array[[Inner, Cols]],
    rows: Int[Rows],
    inner: Int[Inner],
    cols: Int[Cols],
    rb: Int[RB],
    kb: Int[KB],
    cb: Int[CB],
) -> None:
    assert_type(
        scratch_matmul(x, y, rows, inner, cols, rb, kb, cb, None),
        jax.Array[[Rows, Cols]],
    )


def test_initialization_order_is_not_proven[RB: IntVar, CB: IntVar](
    scratch: pl.AccumRef[[RB, CB]],
) -> pl.Tile[[RB, CB]]:
    return scratch[...]


def test_nsteps_termination_is_not_proven[
    Rows: IntVar,
    Inner: IntVar,
    Cols: IntVar,
    RB: IntVar,
    KB: IntVar,
    CB: IntVar,
    Other: IntVar,
](
    spec: pltpu.PrefetchScalarGridSpec[Rows, Cols, RB, CB, Inner, KB],
    rows: Int[Rows],
    cols: Int[Cols],
    wrong_nsteps: Int[Other],
    x: jax.Array[[Rows, Inner]],
    y: jax.Array[[Inner, Cols]],
) -> None:
    def kernel(
        x_ref: pl.InRef[[RB, KB]],
        y_ref: pl.InRef[[KB, CB]],
        out_ref: pl.OutRef[[RB, CB]],
        scratch: pl.AccumRef[[RB, CB]],
    ) -> None:
        matmul_kernel(x_ref, y_ref, out_ref, scratch, nsteps=wrong_nsteps)

    call = pl.pallas_call(
        kernel, grid_spec=spec, out_shape=jax.ShapeDtypeStruct((rows, cols), None)
    )
    assert_type(call(x, y), jax.Array[[Rows, Cols]])


if not TYPE_CHECKING:

    class ScratchMatmulTest(unittest.TestCase):
        def test_cpu_interpreter(self) -> None:
            x = jnp.arange(24, dtype=jnp.float32).reshape(4, 6)
            y = jnp.arange(24, dtype=jnp.float32).reshape(6, 4)
            result = scratch_matmul(x, y, 4, 6, 4, 2, 3, 2, x.dtype)
            self.assertEqual(result.tolist(), (x @ y).tolist())

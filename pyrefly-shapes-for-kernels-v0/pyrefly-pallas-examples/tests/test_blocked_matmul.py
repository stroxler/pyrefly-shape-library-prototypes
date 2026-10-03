# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Semantic annotation names exist only in the stub overlay.
# @lint-ignore-every AUTODEPS2

"""Typed version of the blocked matmul in JAX's Pallas quickstart."""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING

import jax
from jax.experimental import pallas as pl

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar
else:
    import jax.numpy as jnp


def matmul_kernel[RowBlock: IntVar, Inner: IntVar, ColBlock: IntVar](
    x_ref: pl.InRef[[RowBlock, Inner]],
    y_ref: pl.InRef[[Inner, ColBlock]],
    z_ref: pl.OutRef[[RowBlock, ColBlock]],
) -> None:
    z_ref[...] = x_ref[...] @ y_ref[...]


def blocked_matmul[
    Rows: IntVar,
    Inner: IntVar,
    Cols: IntVar,
    RowBlock: IntVar,
    ColBlock: IntVar,
](
    x: jax.Array[[Rows, Inner]],
    y: jax.Array[[Inner, Cols]],
    rows: Int[Rows],
    inner: Int[Inner],
    cols: Int[Cols],
    row_block: Int[RowBlock],
    col_block: Int[ColBlock],
    dtype: object,
) -> jax.Array[[Rows, Cols]]:
    x_spec: pl.BlockSpec[[RowBlock, Inner]] = pl.BlockSpec(
        (row_block, inner), lambda i, j: (i, 0)
    )
    y_spec: pl.BlockSpec[[Inner, ColBlock]] = pl.BlockSpec(
        (inner, col_block), lambda i, j: (0, j)
    )
    z_spec: pl.BlockSpec[[RowBlock, ColBlock]] = pl.BlockSpec(
        (row_block, col_block), lambda i, j: (i, j)
    )
    multiply = pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((rows, cols), dtype),
        grid=(pl.cdiv(rows, row_block), pl.cdiv(cols, col_block)),
        in_specs=(x_spec, y_spec),
        out_specs=z_spec,
        interpret=True,
    )
    return multiply(x, y)


def test_wrong_shared_inner[
    Rows: IntVar,
    Inner: IntVar,
    Other: IntVar,
    Cols: IntVar,
    RowBlock: IntVar,
    ColBlock: IntVar,
](
    x: jax.Array[[Rows, Inner]],
    y: jax.Array[[Other, Cols]],
    rows: Int[Rows],
    inner: Int[Inner],
    cols: Int[Cols],
    row_block: Int[RowBlock],
    col_block: Int[ColBlock],
) -> None:
    blocked_matmul(
        x,
        y,  # E: is not assignable to parameter
        rows,
        inner,
        cols,
        row_block,
        col_block,
        None,
    )


def test_wrong_output_cols[
    Rows: IntVar,
    Inner: IntVar,
    Cols: IntVar,
    Other: IntVar,
    RowBlock: IntVar,
    ColBlock: IntVar,
](
    x_spec: pl.BlockSpec[[RowBlock, Inner]],
    y_spec: pl.BlockSpec[[Inner, ColBlock]],
    z_spec: pl.BlockSpec[[RowBlock, ColBlock]],
    rows: Int[Rows],
    cols: Int[Cols],
    other_cols: Int[Other],
    row_block: Int[RowBlock],
    col_block: Int[ColBlock],
) -> None:
    pl.pallas_call(  # E: No matching overload
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((rows, other_cols), None),
        grid=(pl.cdiv(rows, row_block), pl.cdiv(cols, col_block)),
        in_specs=(x_spec, y_spec),
        out_specs=z_spec,
    )


def test_wrong_row_grid[
    Rows: IntVar,
    Inner: IntVar,
    Cols: IntVar,
    RowBlock: IntVar,
    ColBlock: IntVar,
    WrongBlock: IntVar,
](
    x_spec: pl.BlockSpec[[RowBlock, Inner]],
    y_spec: pl.BlockSpec[[Inner, ColBlock]],
    z_spec: pl.BlockSpec[[RowBlock, ColBlock]],
    rows: Int[Rows],
    cols: Int[Cols],
    wrong_block: Int[WrongBlock],
    col_block: Int[ColBlock],
) -> None:
    pl.pallas_call(  # E: No matching overload
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((rows, cols), None),
        grid=(pl.cdiv(rows, wrong_block), pl.cdiv(cols, col_block)),
        in_specs=(x_spec, y_spec),
        out_specs=z_spec,
    )


def test_wrong_y_block[
    Rows: IntVar,
    Inner: IntVar,
    Cols: IntVar,
    Wrong: IntVar,
    RowBlock: IntVar,
    ColBlock: IntVar,
](
    x_spec: pl.BlockSpec[[RowBlock, Inner]],
    y_spec: pl.BlockSpec[[Wrong, ColBlock]],
    z_spec: pl.BlockSpec[[RowBlock, ColBlock]],
    rows: Int[Rows],
    cols: Int[Cols],
    row_block: Int[RowBlock],
    col_block: Int[ColBlock],
) -> None:
    pl.pallas_call(  # E: No matching overload
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((rows, cols), None),
        grid=(pl.cdiv(rows, row_block), pl.cdiv(cols, col_block)),
        in_specs=(x_spec, y_spec),
        out_specs=z_spec,
    )


def test_wrong_output_block[
    Rows: IntVar,
    Inner: IntVar,
    Cols: IntVar,
    RowBlock: IntVar,
    ColBlock: IntVar,
    Wrong: IntVar,
](
    x_spec: pl.BlockSpec[[RowBlock, Inner]],
    y_spec: pl.BlockSpec[[Inner, ColBlock]],
    z_spec: pl.BlockSpec[[RowBlock, Wrong]],
    rows: Int[Rows],
    cols: Int[Cols],
    row_block: Int[RowBlock],
    col_block: Int[ColBlock],
) -> None:
    pl.pallas_call(  # E: No matching overload
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((rows, cols), None),
        grid=(pl.cdiv(rows, row_block), pl.cdiv(cols, col_block)),
        in_specs=(x_spec, y_spec),
        out_specs=z_spec,
    )


def test_wrong_index_map_rank[RowBlock: IntVar, Inner: IntVar](
    row_block: Int[RowBlock],
    inner: Int[Inner],
) -> None:
    pl.BlockSpec(  # E: No matching overload
        (row_block, inner),
        lambda i, j: (i,),
    )


def test_wrong_matmul_inner[
    RowBlock: IntVar,
    Inner: IntVar,
    Wrong: IntVar,
    ColBlock: IntVar,
](
    x_ref: pl.InRef[[RowBlock, Inner]],
    y_ref: pl.InRef[[Wrong, ColBlock]],
    z_ref: pl.OutRef[[RowBlock, ColBlock]],
) -> None:
    z_ref[...] = x_ref[...] @ y_ref[...]  # E: is not assignable to parameter


def test_index_map_attribution_is_not_proven[
    Rows: IntVar,
    Inner: IntVar,
    Cols: IntVar,
    RowBlock: IntVar,
    ColBlock: IntVar,
](
    x_spec: pl.BlockSpec[[RowBlock, Inner]],
    y_spec: pl.BlockSpec[[Inner, ColBlock]],
    rows: Int[Rows],
    cols: Int[Cols],
    row_block: Int[RowBlock],
    col_block: Int[ColBlock],
) -> None:
    # Correct rank and shape, but all programs write the same output block.
    z_spec: pl.BlockSpec[[RowBlock, ColBlock]] = pl.BlockSpec(
        (row_block, col_block), lambda i, j: (0, 0)
    )
    pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((rows, cols), None),
        grid=(pl.cdiv(rows, row_block), pl.cdiv(cols, col_block)),
        in_specs=(x_spec, y_spec),
        out_specs=z_spec,
    )


if not TYPE_CHECKING:

    class BlockedMatmulTest(unittest.TestCase):
        def test_two_dimensional_grid(self) -> None:
            x = jnp.arange(12, dtype=jnp.float32).reshape((4, 3))
            y = jnp.arange(18, dtype=jnp.float32).reshape((3, 6))
            actual = blocked_matmul(x, y, 4, 3, 6, 2, 3, x.dtype)
            self.assertEqual(actual.tolist(), (x @ y).tolist())

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Semantic annotation names exist only in the stub overlay.
# @lint-ignore-every AUTODEPS2

"""Two-dimensional blocked addition with a squeezed row axis."""

from __future__ import annotations

import unittest
from typing import Literal, TYPE_CHECKING

import jax
from jax.experimental import pallas as pl

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar
else:
    import jax.numpy as jnp


def add_kernel[Block: IntVar](
    x_ref: pl.InRef[[Block]], y_ref: pl.InRef[[Block]], o_ref: pl.OutRef[[Block]]
) -> None:
    x = x_ref[:]
    y = y_ref[:]
    o_ref[:] = x + y


def squeezed_row_add[Rows: IntVar, Cols: IntVar, Block: IntVar](
    x: jax.Array[[Rows, Cols]],
    y: jax.Array[[Rows, Cols]],
    rows: Int[Rows],
    cols: Int[Cols],
    block: Int[Block],
    dtype: object,
) -> jax.Array[[Rows, Cols]]:
    spec: pl.BlockSpec[[Block], Literal[True]] = pl.BlockSpec(
        (None, block), lambda i, j: (i, j)
    )
    add = pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct((rows, cols), dtype),
        grid=(rows, pl.cdiv(cols, block)),
        in_specs=(spec, spec),
        out_specs=spec,
        interpret=True,
    )
    return add(x, y)


def test_wrong_input_rows[Rows: IntVar, Other: IntVar, Cols: IntVar, Block: IntVar](
    x: jax.Array[[Rows, Cols]],
    y: jax.Array[[Other, Cols]],
    rows: Int[Rows],
    cols: Int[Cols],
    block: Int[Block],
) -> None:
    squeezed_row_add(x, y, rows, cols, block, None)  # E: is not assignable to parameter


def test_wrong_output_rows[Rows: IntVar, Other: IntVar, Cols: IntVar, Block: IntVar](
    spec: pl.BlockSpec[[Block], Literal[True]],
    rows: Int[Rows],
    other_rows: Int[Other],
    cols: Int[Cols],
    block: Int[Block],
) -> None:
    pl.pallas_call(  # E: No matching overload
        add_kernel,
        out_shape=jax.ShapeDtypeStruct((other_rows, cols), None),
        grid=(rows, pl.cdiv(cols, block)),
        in_specs=(spec, spec),
        out_specs=spec,
    )


def test_wrong_col_grid_block[Rows: IntVar, Cols: IntVar, Block: IntVar, Other: IntVar](
    spec: pl.BlockSpec[[Block], Literal[True]],
    rows: Int[Rows],
    cols: Int[Cols],
    other: Int[Other],
) -> None:
    pl.pallas_call(  # E: No matching overload
        add_kernel,
        out_shape=jax.ShapeDtypeStruct((rows, cols), None),
        grid=(rows, pl.cdiv(cols, other)),
        in_specs=(spec, spec),
        out_specs=spec,
    )


def test_wrong_index_map_rank[Block: IntVar](block: Int[Block]) -> None:
    pl.BlockSpec(  # E: No matching overload
        (None, block), lambda i, j: (i,)
    )


def test_wrong_kernel_ref_rank[Block: IntVar](
    x_ref: pl.InRef[[Block]],
    y_ref: pl.InRef[[Block]],
    o_ref: pl.OutRef[[Block, Block]],
) -> None:
    o_ref[:] = x_ref[:] + y_ref[:]  # E: is not assignable to parameter


def test_unsqueezed_spec_does_not_describe_2d_host[
    Rows: IntVar,
    Cols: IntVar,
    Block: IntVar,
](rows: Int[Rows], cols: Int[Cols], block: Int[Block]) -> None:
    one_dimensional_spec: pl.BlockSpec[[Block]] = pl.BlockSpec((block,), lambda i: (i,))
    pl.pallas_call(  # E: No matching overload
        add_kernel,
        out_shape=jax.ShapeDtypeStruct((rows, cols), None),
        grid=(rows, pl.cdiv(cols, block)),
        in_specs=(one_dimensional_spec, one_dimensional_spec),
        out_specs=one_dimensional_spec,
    )


def test_host_rank_must_remain_two[Rows: IntVar, Cols: IntVar, Block: IntVar](
    spec: pl.BlockSpec[[Block], Literal[True]],
    rows: Int[Rows],
    cols: Int[Cols],
    block: Int[Block],
) -> None:
    pl.pallas_call(  # E: No matching overload
        add_kernel,
        out_shape=jax.ShapeDtypeStruct((cols,), None),
        grid=(rows, pl.cdiv(cols, block)),
        in_specs=(spec, spec),
        out_specs=spec,
    )


def test_index_map_coverage_is_not_proven[Rows: IntVar, Cols: IntVar, Block: IntVar](
    rows: Int[Rows], cols: Int[Cols], block: Int[Block]
) -> None:
    # Rank two is accepted even when the map sends every row to row zero.
    spec: pl.BlockSpec[[Block], Literal[True]] = pl.BlockSpec(
        (None, block), lambda i, j: (0, j)
    )
    pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct((rows, cols), None),
        grid=(rows, pl.cdiv(cols, block)),
        in_specs=(spec, spec),
        out_specs=spec,
    )


if not TYPE_CHECKING:

    class SqueezedAxisTest(unittest.TestCase):
        def test_partial_last_column_block(self) -> None:
            x = jnp.arange(15, dtype=jnp.float32).reshape((3, 5))
            y = jnp.arange(15, 30, dtype=jnp.float32).reshape((3, 5))
            actual = squeezed_row_add(x, y, 3, 5, 2, x.dtype)
            self.assertEqual(actual.tolist(), (x + y).tolist())

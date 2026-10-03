# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Kernel body from JAX docs/pallas/tpu/sparse.md (Apache-2.0).
# Semantic annotations exist only in the stub overlay.
# @lint-ignore-every AUTODEPS2

"""Pallas scalar prefetch selects a block from a larger source array."""

from __future__ import annotations

import unittest
from typing import assert_type, Literal, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar


def dynamic_slice_kernel[RowBlock: IntVar, ColBlock: IntVar](
    indices: pl.PrefetchRef[[2]],
    x_ref: pl.InRef[[RowBlock, ColBlock]],
    o_ref: pl.OutRef[[RowBlock, ColBlock]],
) -> None:
    del indices
    o_ref[...] = x_ref[...]


def block_dynamic_slice[
    SourceRows: IntVar,
    SourceCols: IntVar,
    RowBlock: IntVar,
    ColBlock: IntVar,
](
    x: jax.Array[[SourceRows, SourceCols]],
    block_idx: jax.Array[[2]],
    row_block: Int[RowBlock],
    col_block: Int[ColBlock],
    dtype: object,
) -> jax.Array[[RowBlock, ColBlock]]:
    block_shape = (row_block, col_block)
    x_spec: pl.BlockSpec[[RowBlock, ColBlock], Literal["prefetch"]] = pl.BlockSpec(
        block_shape, lambda i, j, indices: (indices[0], indices[1])
    )
    o_spec: pl.BlockSpec[[RowBlock, ColBlock], Literal["prefetch"]] = pl.BlockSpec(
        block_shape, lambda *_: (0, 0)
    )
    grid_spec: pltpu.PrefetchScalarGridSpec[
        SourceRows, SourceCols, RowBlock, ColBlock
    ] = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(1, 1),
        in_specs=[x_spec],
        out_specs=o_spec,
    )
    slice_call = pl.pallas_call(
        dynamic_slice_kernel,
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct(block_shape, dtype),
        interpret=True,
    )
    return slice_call(block_idx, x)


def test_wrong_prefetch_shape[
    SourceRows: IntVar,
    SourceCols: IntVar,
    RowBlock: IntVar,
    ColBlock: IntVar,
](
    x: jax.Array[[SourceRows, SourceCols]],
    indices: jax.Array[[3]],
    row_block: Int[RowBlock],
    col_block: Int[ColBlock],
) -> None:
    block_dynamic_slice(x, indices, row_block, col_block, None)  # E: is not assignable


def test_wrong_input_rank[
    SourceRows: IntVar,
    RowBlock: IntVar,
    ColBlock: IntVar,
](
    x: jax.Array[[SourceRows]],
    indices: jax.Array[[2]],
    row_block: Int[RowBlock],
    col_block: Int[ColBlock],
) -> None:
    block_dynamic_slice(x, indices, row_block, col_block, None)  # E: is not assignable


def test_wrong_output_tile[Rows: IntVar, Cols: IntVar, Other: IntVar](
    indices: pl.PrefetchRef[[2]],
    x: pl.InRef[[Rows, Cols]],
    out: pl.OutRef[[Rows, Other]],
) -> None:
    del indices
    out[...] = x[...]  # E: Cannot set item


def test_wrong_index_map_rank[Rows: IntVar, Cols: IntVar](
    rows: Int[Rows], cols: Int[Cols]
) -> None:
    pl.BlockSpec(  # E: No matching overload
        (rows, cols), lambda i, j, indices: (indices[0],)
    )


def test_wrong_output_allocation[
    SourceRows: IntVar,
    SourceCols: IntVar,
    RowBlock: IntVar,
    ColBlock: IntVar,
    Other: IntVar,
](
    spec: pltpu.PrefetchScalarGridSpec[SourceRows, SourceCols, RowBlock, ColBlock],
    row_block: Int[RowBlock],
    other_col_block: Int[Other],
) -> None:
    pl.pallas_call(  # E: No matching overload
        dynamic_slice_kernel,
        grid_spec=spec,
        out_shape=jax.ShapeDtypeStruct((row_block, other_col_block), None),
    )


def test_wrong_prefetch_position[
    SourceRows: IntVar,
    SourceCols: IntVar,
    RowBlock: IntVar,
    ColBlock: IntVar,
](
    spec: pltpu.PrefetchScalarGridSpec[SourceRows, SourceCols, RowBlock, ColBlock],
    row_block: Int[RowBlock],
    col_block: Int[ColBlock],
) -> None:
    def wrong_order(
        x: pl.InRef[[RowBlock, ColBlock]],
        indices: pl.PrefetchRef[[2]],
        out: pl.OutRef[[RowBlock, ColBlock]],
    ) -> None:
        del x, indices, out

    pl.pallas_call(  # E: No matching overload
        wrong_order,
        grid_spec=spec,
        out_shape=jax.ShapeDtypeStruct((row_block, col_block), None),
    )


def test_wrong_prefetch_grid[Rows: IntVar, Cols: IntVar](
    x_spec: pl.BlockSpec[[Rows, Cols], Literal["prefetch"]],
    o_spec: pl.BlockSpec[[Rows, Cols], Literal["prefetch"]],
) -> None:
    pltpu.PrefetchScalarGridSpec(  # E: No matching overload
        num_scalar_prefetch=1,
        grid=(1, 2),
        in_specs=[x_spec],
        out_specs=o_spec,
    )


def test_index_map_bounds_are_not_proven[Rows: IntVar, Cols: IntVar](
    rows: Int[Rows], cols: Int[Cols]
) -> None:
    # This map swaps the row/column indices, potentially reading another block.
    x_spec: pl.BlockSpec[[Rows, Cols], Literal["prefetch"]] = pl.BlockSpec(
        (rows, cols), lambda i, j, indices: (indices[1], indices[0])
    )
    _ = x_spec


def test_output_shape[
    SourceRows: IntVar,
    SourceCols: IntVar,
    RowBlock: IntVar,
    ColBlock: IntVar,
](
    x: jax.Array[[SourceRows, SourceCols]],
    indices: jax.Array[[2]],
    row_block: Int[RowBlock],
    col_block: Int[ColBlock],
) -> None:
    assert_type(
        block_dynamic_slice(x, indices, row_block, col_block, None),
        jax.Array[[RowBlock, ColBlock]],
    )


if not TYPE_CHECKING:

    class ScalarPrefetchTest(unittest.TestCase):
        def test_cpu_interpreter(self) -> None:
            x = jnp.arange(30, dtype=jnp.int32).reshape(5, 6)
            indices = jnp.array([1, 1], dtype=jnp.int32)
            self.assertEqual(
                block_dynamic_slice(x, indices, 2, 3, jnp.int32).tolist(),
                [[15, 16, 17], [21, 22, 23]],
            )
            self.assertEqual(
                block_dynamic_slice(x, jnp.array([0, 1]), 2, 3, jnp.int32).tolist(),
                [[3, 4, 5], [9, 10, 11]],
            )

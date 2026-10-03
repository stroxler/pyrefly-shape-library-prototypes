# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Semantic annotation names exist only in the stub overlay.
# @lint-ignore-every AUTODEPS2

"""Typed version of the add_kernel in JAX's Pallas design document."""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING

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


def vector_add[Length: IntVar, Block: IntVar](
    x: jax.Array[[Length]],
    y: jax.Array[[Length]],
    length: Int[Length],
    block: Int[Block],
    dtype: object,
) -> jax.Array[[Length]]:
    spec: pl.BlockSpec[[Block]] = pl.BlockSpec((block,), lambda i: (i,))
    add = pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct((length,), dtype),
        grid=(pl.cdiv(length, block),),
        in_specs=(spec, spec),
        out_specs=spec,
        interpret=True,
    )
    return add(x, y)


def test_wrong_input_length[Length: IntVar, Other: IntVar, Block: IntVar](
    x: jax.Array[[Length]],
    y: jax.Array[[Other]],
    n: Int[Length],
    block: Int[Block],
) -> None:
    vector_add(x, y, n, block, None)  # E: is not assignable to parameter


def test_wrong_output_length[Length: IntVar, Other: IntVar, Block: IntVar](
    spec: pl.BlockSpec[[Block]],
    n: Int[Other],
    grid: pl.GridSize[Length, Block],
) -> None:
    pl.pallas_call(  # E: No matching overload
        add_kernel,
        out_shape=jax.ShapeDtypeStruct((n,), None),
        grid=(grid,),
        in_specs=(spec, spec),
        out_specs=spec,
    )


def test_wrong_grid_block[Length: IntVar, Block: IntVar, Other: IntVar](
    spec: pl.BlockSpec[[Block]],
    n: Int[Length],
    wrong_block: Int[Other],
) -> None:
    pl.pallas_call(  # E: No matching overload
        add_kernel,
        out_shape=jax.ShapeDtypeStruct((n,), None),
        grid=(pl.cdiv(n, wrong_block),),
        in_specs=(spec, spec),
        out_specs=spec,
    )


def test_wrong_index_map_rank[Length: IntVar, Block: IntVar](
    block: Int[Block],
) -> None:
    pl.BlockSpec(  # E: No matching overload
        (block,),
        lambda i: (i, i),
    )


def test_wrong_kernel_block[Block: IntVar, Other: IntVar](
    x_ref: pl.InRef[[Block]],
    y_ref: pl.InRef[[Block]],
    o_ref: pl.OutRef[[Other]],
) -> None:
    o_ref[:] = x_ref[:] + y_ref[:]  # E: is not assignable to parameter


def test_index_map_coverage_is_not_proven[Length: IntVar, Block: IntVar](
    x: jax.Array[[Length]],
    y: jax.Array[[Length]],
    n: Int[Length],
    block: Int[Block],
) -> jax.Array[[Length]]:
    # The map has the correct rank but sends every program to block zero.
    spec: pl.BlockSpec[[Block]] = pl.BlockSpec((block,), lambda i: (0,))
    add = pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct((n,), None),
        grid=(pl.cdiv(n, block),),
        in_specs=(spec, spec),
        out_specs=spec,
    )
    return add(x, y)


if not TYPE_CHECKING:

    class VectorAddTest(unittest.TestCase):
        def test_partial_final_block(self) -> None:
            x = jnp.arange(10, dtype=jnp.float32)
            y = jnp.arange(10, 20, dtype=jnp.float32)
            actual = vector_add(x, y, x.shape[0], 4, x.dtype)
            self.assertEqual(actual.tolist(), (x + y).tolist())

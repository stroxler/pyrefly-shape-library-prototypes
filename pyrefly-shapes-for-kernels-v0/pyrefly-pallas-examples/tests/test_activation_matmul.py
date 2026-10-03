# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Kernel body from JAX docs/pallas/quickstart.md (Apache-2.0).
# Semantic annotations are static-only.
# @lint-ignore-every AUTODEPS2

"""A Pallas tiled matmul with a host-provided activation callback."""

from __future__ import annotations

import unittest
from collections.abc import Callable
from functools import partial
from typing import assert_type, TYPE_CHECKING

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
    *,
    activation: Callable[
        [pl.Tile[[RowBlock, ColBlock]]], pl.Tile[[RowBlock, ColBlock]]
    ],
) -> None:
    z_ref[...] = activation(x_ref[...] @ y_ref[...])


def matmul_with_activation[
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
    activation: Callable[
        [pl.Tile[[RowBlock, ColBlock]]], pl.Tile[[RowBlock, ColBlock]]
    ],
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

    # The explicit callback preserves the activation's input and output tile.
    def bound_kernel(
        x_ref: pl.InRef[[RowBlock, Inner]],
        y_ref: pl.InRef[[Inner, ColBlock]],
        z_ref: pl.OutRef[[RowBlock, ColBlock]],
    ) -> None:
        matmul_kernel(x_ref, y_ref, z_ref, activation=activation)

    multiply = pl.pallas_call(
        bound_kernel,
        out_shape=jax.ShapeDtypeStruct((rows, cols), dtype),
        grid=(pl.cdiv(rows, row_block), pl.cdiv(cols, col_block)),
        in_specs=(x_spec, y_spec),
        out_specs=z_spec,
        interpret=True,
    )
    return multiply(x, y)


def test_bad_activation_width[RB: IntVar, Inner: IntVar, CB: IntVar, Other: IntVar](
    x: pl.InRef[[RB, Inner]],
    y: pl.InRef[[Inner, CB]],
    z: pl.OutRef[[RB, CB]],
    activation: Callable[[pl.Tile[[RB, CB]]], pl.Tile[[RB, Other]]],
) -> None:
    matmul_kernel(x, y, z, activation=activation)  # E: is not assignable


def test_partial_binding_of_wrong_activation_is_not_proven[
    RB: IntVar,
    CB: IntVar,
    Other: IntVar,
](
    activation: Callable[[pl.Tile[[RB, CB]]], pl.Tile[[RB, Other]]],
) -> None:
    # A partial's keyword arguments are not checked against the generic kernel.
    _ = partial(matmul_kernel, activation=activation)


def test_bad_contracting_dimension[
    RB: IntVar,
    Inner: IntVar,
    CB: IntVar,
    Other: IntVar,
](
    x: pl.InRef[[RB, Inner]],
    y: pl.InRef[[Other, CB]],
    z: pl.OutRef[[RB, CB]],
    activation: Callable[[pl.Tile[[RB, CB]]], pl.Tile[[RB, CB]]],
) -> None:
    matmul_kernel(x, y, z, activation=activation)  # E: is not assignable


def test_host_wrong_y_extent[
    Rows: IntVar,
    Inner: IntVar,
    Cols: IntVar,
    Other: IntVar,
    RB: IntVar,
    CB: IntVar,
](
    x: jax.Array[[Rows, Inner]],
    y: jax.Array[[Other, Cols]],
    rows: Int[Rows],
    inner: Int[Inner],
    cols: Int[Cols],
    rb: Int[RB],
    cb: Int[CB],
    activation: Callable[[pl.Tile[[RB, CB]]], pl.Tile[[RB, CB]]],
) -> None:
    matmul_with_activation(
        x,
        y,  # E: is not assignable
        rows,
        inner,
        cols,
        rb,
        cb,
        None,
        activation,
    )


def test_host_wrong_activation_width[
    Rows: IntVar,
    Inner: IntVar,
    Cols: IntVar,
    Other: IntVar,
    RB: IntVar,
    CB: IntVar,
](
    x: jax.Array[[Rows, Inner]],
    y: jax.Array[[Inner, Cols]],
    rows: Int[Rows],
    inner: Int[Inner],
    cols: Int[Cols],
    rb: Int[RB],
    cb: Int[CB],
    bad_activation: Callable[[pl.Tile[[RB, CB]]], pl.Tile[[RB, Other]]],
) -> None:
    matmul_with_activation(
        x,
        y,
        rows,
        inner,
        cols,
        rb,
        cb,
        None,
        bad_activation,  # E: is not assignable
    )


def test_output_shape[
    Rows: IntVar,
    Inner: IntVar,
    Cols: IntVar,
    RB: IntVar,
    CB: IntVar,
](
    x: jax.Array[[Rows, Inner]],
    y: jax.Array[[Inner, Cols]],
    rows: Int[Rows],
    inner: Int[Inner],
    cols: Int[Cols],
    rb: Int[RB],
    cb: Int[CB],
    activation: Callable[[pl.Tile[[RB, CB]]], pl.Tile[[RB, CB]]],
) -> None:
    assert_type(
        matmul_with_activation(x, y, rows, inner, cols, rb, cb, None, activation),
        jax.Array[[Rows, Cols]],
    )


if not TYPE_CHECKING:

    class ActivationMatmulTest(unittest.TestCase):
        def test_interpreter(self) -> None:
            x = jnp.arange(12, dtype=jnp.float32).reshape((4, 3)) - 5
            y = jnp.arange(18, dtype=jnp.float32).reshape((3, 6)) - 6
            actual = matmul_with_activation(x, y, 4, 3, 6, 2, 3, x.dtype, jax.nn.relu)
            self.assertEqual(actual.tolist(), jax.nn.relu(x @ y).tolist())

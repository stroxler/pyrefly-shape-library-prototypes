# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The kernel body is copied from JAX's docs/pallas/pipelining.md (Apache-2.0).
# Semantic annotations exist only in the stub overlay.
# @lint-ignore-every AUTODEPS2

"""An unblocked rank-two Pallas Ref spanning its host matrix allocation."""

from __future__ import annotations

import unittest
from typing import assert_type, TYPE_CHECKING

import jax
from jax.experimental import pallas as pl

if TYPE_CHECKING:
    from shape_extensions import IntVar
else:
    import jax.numpy as jnp


def add_matrices_kernel[Rows: IntVar, Cols: IntVar](
    x_sram_ref: pl.InRef[[Rows, Cols]],
    y_sram_ref: pl.InRef[[Rows, Cols]],
    z_sram_ref: pl.OutRef[[Rows, Cols]],
) -> None:
    # Load x and y from SRAM into registers
    x_regs = x_sram_ref[:, :]
    y_regs = y_sram_ref[:, :]
    # Execute a vectorized add
    z_regs = x_regs + y_regs
    # Store the output values in registers back into SRAM
    z_sram_ref[:, :] = z_regs


def add_matrices[Rows: IntVar, Cols: IntVar](
    x: jax.Array[[Rows, Cols]], y: jax.Array[[Rows, Cols]]
) -> jax.Array[[Rows, Cols]]:
    add = pl.pallas_call(
        add_matrices_kernel,
        out_shape=jax.ShapeDtypeStruct.like(x),
        interpret=True,
    )
    return add(x, y)


def test_wrong_input_columns[Rows: IntVar, Cols: IntVar, Other: IntVar](
    x: jax.Array[[Rows, Cols]], y: jax.Array[[Rows, Other]]
) -> None:
    add_matrices(x, y)  # E: is not assignable to parameter


def test_wrong_input_rows[Rows: IntVar, Cols: IntVar, Other: IntVar](
    x: jax.Array[[Rows, Cols]], y: jax.Array[[Other, Cols]]
) -> None:
    add_matrices(x, y)  # E: is not assignable to parameter


def test_wrong_output_allocation[Rows: IntVar, Cols: IntVar, Other: IntVar](
    x: jax.Array[[Rows, Cols]],
    y: jax.Array[[Rows, Cols]],
    wrong_output: jax.ShapeDtypeStruct[[Rows, Other]],
) -> None:
    add = pl.pallas_call(add_matrices_kernel, out_shape=wrong_output)
    add(
        x,  # E: is not assignable to parameter
        y,  # E: is not assignable to parameter
    )


def test_wrong_output_tile[Rows: IntVar, Cols: IntVar, Other: IntVar](
    x: pl.InRef[[Rows, Cols]],
    y: pl.InRef[[Rows, Cols]],
    wrong_output: pl.OutRef[[Rows, Other]],
) -> None:
    wrong_output[:, :] = x[:, :] + y[:, :]  # E: Cannot set item


def test_unblocked_output_shape[Rows: IntVar, Cols: IntVar](
    x: jax.Array[[Rows, Cols]], y: jax.Array[[Rows, Cols]]
) -> None:
    assert_type(add_matrices(x, y), jax.Array[[Rows, Cols]])


if not TYPE_CHECKING:

    class FullMatrixAddTest(unittest.TestCase):
        def test_interpreter(self) -> None:
            x = jnp.arange(12, dtype=jnp.float32).reshape((3, 4))
            y = jnp.ones((3, 4), dtype=jnp.float32)
            self.assertEqual(add_matrices(x, y).tolist(), (x + y).tolist())

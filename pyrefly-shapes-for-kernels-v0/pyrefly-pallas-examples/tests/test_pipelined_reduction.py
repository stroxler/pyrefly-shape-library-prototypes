# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Kernel body from JAX docs/pallas/pipelining.md (Apache-2.0).
# The annotations are static-only; the CPU interpreter tests the same body.
# @lint-ignore-every AUTODEPS2

"""Pallas grid-last reduction exposes an initialized output SRAM Ref."""

from __future__ import annotations

import unittest
from typing import assert_type, Literal, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar


def correct_sum_kernel[RB: IntVar, CB: IntVar](
    x_ref: pl.InRef[[RB, CB]], o_ref: pl.AccumRef[[RB, CB]]
) -> None:
    @pl.when(pl.program_id(2) == 0)
    def _():
        o_ref[...] = jnp.zeros_like(o_ref)

    o_ref[...] += x_ref[...]


def correct_sum[Reduce: IntVar, Rows: IntVar, Cols: IntVar, RB: IntVar, CB: IntVar](
    x: jax.Array[[Reduce, Rows, Cols]],
    reduce: Int[Reduce],
    rows: Int[Rows],
    cols: Int[Cols],
    row_block: Int[RB],
    col_block: Int[CB],
    dtype: object,
) -> jax.Array[[Rows, Cols]]:
    x_spec: pl.BlockSpec[[RB, CB], Literal["reduction"]] = pl.BlockSpec(
        (None, row_block, col_block), lambda i, j, k: (k, i, j)
    )
    o_spec: pl.BlockSpec[[RB, CB]] = pl.BlockSpec(
        (row_block, col_block), lambda i, j, k: (i, j)
    )
    reduce_call = pl.pallas_call(
        correct_sum_kernel,
        out_shape=jax.ShapeDtypeStruct((rows, cols), dtype),
        grid=(pl.cdiv(rows, row_block), pl.cdiv(cols, col_block), reduce),
        in_specs=[x_spec],
        out_specs=o_spec,
        interpret=True,
    )
    return reduce_call(x)


def test_wrong_input_reduction_axis[
    Reduce: IntVar,
    Other: IntVar,
    Rows: IntVar,
    Cols: IntVar,
    RB: IntVar,
    CB: IntVar,
](
    x: jax.Array[[Other, Rows, Cols]],
    reduce: Int[Reduce],
    rows: Int[Rows],
    cols: Int[Cols],
    rb: Int[RB],
    cb: Int[CB],
) -> None:
    correct_sum(x, reduce, rows, cols, rb, cb, None)  # E: is not assignable


def test_wrong_host_output_rows[
    Reduce: IntVar,
    Rows: IntVar,
    Other: IntVar,
    Cols: IntVar,
    RB: IntVar,
    CB: IntVar,
](
    x: jax.Array[[Reduce, Rows, Cols]],
    reduce: Int[Reduce],
    wrong_rows: Int[Other],
    cols: Int[Cols],
    rb: Int[RB],
    cb: Int[CB],
) -> None:
    correct_sum(x, reduce, wrong_rows, cols, rb, cb, None)  # E: is not assignable


def test_wrong_output_tile[RB: IntVar, CB: IntVar, Other: IntVar](
    x: pl.InRef[[RB, CB]], out: pl.AccumRef[[RB, Other]]
) -> None:
    out[...] += x[...]  # E: is not supported


def test_wrong_initialization_tile[RB: IntVar, CB: IntVar, Other: IntVar](
    out: pl.AccumRef[[RB, CB]], wrong: pl.AccumRef[[RB, Other]]
) -> None:
    out[...] = jnp.zeros_like(wrong)  # E: Cannot set item


def test_initialization_before_read_is_not_proven[RB: IntVar, CB: IntVar](
    out: pl.AccumRef[[RB, CB]],
) -> pl.Tile[[RB, CB]]:
    return out[...]


def test_wrong_reduction_index_map_is_not_proven[
    Reduce: IntVar,
    Rows: IntVar,
    Cols: IntVar,
    RB: IntVar,
    CB: IntVar,
](
    x: jax.Array[[Reduce, Rows, Cols]],
    reduce: Int[Reduce],
    rows: Int[Rows],
    cols: Int[Cols],
    rb: Int[RB],
    cb: Int[CB],
) -> None:
    # Mapping every iteration to the first reduction slice still type-checks.
    x_spec: pl.BlockSpec[[RB, CB], Literal["reduction"]] = pl.BlockSpec(
        (None, rb, cb), lambda i, j, k: (0, i, j)
    )
    o_spec: pl.BlockSpec[[RB, CB]] = pl.BlockSpec((rb, cb), lambda i, j, k: (i, j))
    call = pl.pallas_call(
        correct_sum_kernel,
        out_shape=jax.ShapeDtypeStruct((rows, cols), None),
        grid=(pl.cdiv(rows, rb), pl.cdiv(cols, cb), reduce),
        in_specs=[x_spec],
        out_specs=o_spec,
    )
    assert_type(call(x), jax.Array[[Rows, Cols]])


def test_output_shape[
    Reduce: IntVar,
    Rows: IntVar,
    Cols: IntVar,
    RB: IntVar,
    CB: IntVar,
](
    x: jax.Array[[Reduce, Rows, Cols]],
    reduce: Int[Reduce],
    rows: Int[Rows],
    cols: Int[Cols],
    rb: Int[RB],
    cb: Int[CB],
) -> None:
    assert_type(
        correct_sum(x, reduce, rows, cols, rb, cb, None), jax.Array[[Rows, Cols]]
    )


if not TYPE_CHECKING:

    class PipelinedReductionTest(unittest.TestCase):
        def test_cpu_interpreter(self) -> None:
            x = jnp.arange(32, dtype=jnp.float32).reshape((2, 4, 4))
            result = correct_sum(x, 2, 4, 4, 2, 2, jnp.float32)
            self.assertEqual(result.tolist(), jnp.sum(x, axis=0).tolist())

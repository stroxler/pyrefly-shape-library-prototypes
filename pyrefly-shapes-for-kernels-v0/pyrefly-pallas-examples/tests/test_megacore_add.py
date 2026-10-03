# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Kernel body from JAX docs/pallas/tpu/pipelining.md (Apache-2.0).
# Semantic annotations exist only in the stub overlay.
# @lint-ignore-every AUTODEPS2

"""A one-axis parallel grid tiles two-dimensional Pallas inputs by row."""

from __future__ import annotations

import unittest
from typing import assert_type, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar


def add_matrices_kernel[RowBlock: IntVar, Cols: IntVar](
    x_vmem_ref: pl.InRef[[RowBlock, Cols]],
    y_vmem_ref: pl.InRef[[RowBlock, Cols]],
    z_vmem_ref: pl.OutRef[[RowBlock, Cols]],
) -> None:
    # Load x and y from VMEM into VREGs
    x_vregs = x_vmem_ref[:, :]
    y_vregs = y_vmem_ref[:, :]
    # Execute a vectorized add
    z_vregs = x_vregs + y_vregs
    # Store the output values in VREGs back into VMEM
    z_vmem_ref[:, :] = z_vregs


def megacore_add[Rows: IntVar, Cols: IntVar, RowBlock: IntVar](
    x: jax.Array[[Rows, Cols]],
    y: jax.Array[[Rows, Cols]],
    rows: Int[Rows],
    cols: Int[Cols],
    row_block: Int[RowBlock],
) -> jax.Array[[Rows, Cols]]:
    block_spec: pl.BlockSpec[[RowBlock, Cols]] = pl.BlockSpec(
        (row_block, cols), lambda i: (i, 0)
    )
    call = pl.pallas_call(
        add_matrices_kernel,
        out_shape=jax.ShapeDtypeStruct.like(x),
        in_specs=(block_spec, block_spec),
        out_specs=block_spec,
        grid=(pl.cdiv(rows, row_block),),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=True,
    )
    return call(x, y)


def test_wrong_host_input_rows[Rows: IntVar, Other: IntVar, Cols: IntVar, RB: IntVar](
    x: jax.Array[[Rows, Cols]],
    y: jax.Array[[Other, Cols]],
    rows: Int[Rows],
    cols: Int[Cols],
    row_block: Int[RB],
) -> None:
    megacore_add(x, y, rows, cols, row_block)  # E: is not assignable


def test_wrong_host_input_columns[
    Rows: IntVar,
    Cols: IntVar,
    Other: IntVar,
    RB: IntVar,
](
    x: jax.Array[[Rows, Cols]],
    y: jax.Array[[Rows, Other]],
    rows: Int[Rows],
    cols: Int[Cols],
    row_block: Int[RB],
) -> None:
    megacore_add(x, y, rows, cols, row_block)  # E: is not assignable


def test_wrong_rhs_tile[RB: IntVar, Cols: IntVar, Other: IntVar](
    x: pl.InRef[[RB, Cols]],
    y: pl.InRef[[RB, Other]],
    z: pl.OutRef[[RB, Cols]],
) -> None:
    z[:, :] = x[:, :] + y[:, :]  # E: is not supported


def test_wrong_output_tile[RB: IntVar, Cols: IntVar, Other: IntVar](
    x: pl.InRef[[RB, Cols]],
    z: pl.OutRef[[RB, Other]],
) -> None:
    z[:, :] = x[:, :]  # E: Cannot set item


def test_wrong_index_map_rank[RB: IntVar, Cols: IntVar](
    rb: Int[RB],
    cols: Int[Cols],
) -> None:
    pl.BlockSpec[[RB, Cols]](  # E: No matching overload
        (rb, cols), lambda i: (i,)
    )


def test_wrong_grid_block[Rows: IntVar, Cols: IntVar, RB: IntVar, Other: IntVar](
    spec: pl.BlockSpec[[RB, Cols]],
    rows: Int[Rows],
    cols: Int[Cols],
    wrong_block: Int[Other],
) -> None:
    pl.pallas_call(  # E: No matching overload
        add_matrices_kernel,
        out_shape=jax.ShapeDtypeStruct((rows, cols), None),
        in_specs=(spec, spec),
        out_specs=spec,
        grid=(pl.cdiv(rows, wrong_block),),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )


def test_wrong_output_allocation[Rows: IntVar, Cols: IntVar, RB: IntVar, Other: IntVar](
    spec: pl.BlockSpec[[RB, Cols]],
    wrong_rows: Int[Other],
    cols: Int[Cols],
    rows: Int[Rows],
    rb: Int[RB],
) -> None:
    pl.pallas_call(  # E: No matching overload
        add_matrices_kernel,
        out_shape=jax.ShapeDtypeStruct((wrong_rows, cols), None),
        in_specs=(spec, spec),
        out_specs=spec,
        grid=(pl.cdiv(rows, rb),),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )


def test_wrong_compiler_grid_rank() -> None:
    pltpu.CompilerParams(  # E: No matching overload
        dimension_semantics=("parallel", "parallel")
    )


def test_index_map_coverage_is_not_proven[Rows: IntVar, Cols: IntVar, RB: IntVar](
    rows: Int[Rows],
    rb: Int[RB],
    cols: Int[Cols],
    x: jax.Array[[Rows, Cols]],
    y: jax.Array[[Rows, Cols]],
) -> None:
    bad_spec: pl.BlockSpec[[RB, Cols]] = pl.BlockSpec((rb, cols), lambda i: (0, 0))
    call = pl.pallas_call(
        add_matrices_kernel,
        out_shape=jax.ShapeDtypeStruct((rows, cols), None),
        in_specs=(bad_spec, bad_spec),
        out_specs=bad_spec,
        grid=(pl.cdiv(rows, rb),),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )
    assert_type(call(x, y), jax.Array[[Rows, Cols]])


def test_output_shape[Rows: IntVar, Cols: IntVar, RB: IntVar](
    x: jax.Array[[Rows, Cols]],
    y: jax.Array[[Rows, Cols]],
    rows: Int[Rows],
    cols: Int[Cols],
    rb: Int[RB],
) -> None:
    assert_type(megacore_add(x, y, rows, cols, rb), jax.Array[[Rows, Cols]])


if not TYPE_CHECKING:

    class MegacoreAddTest(unittest.TestCase):
        def test_cpu_interpreter(self) -> None:
            x = jnp.arange(24, dtype=jnp.float32).reshape(4, 6)
            y = jnp.ones_like(x)
            self.assertEqual(megacore_add(x, y, 4, 6, 2).tolist(), (x + y).tolist())

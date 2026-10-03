# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Kernel body from JAX docs/pallas/tpu/pipelining.md (Apache-2.0).
# Semantic annotations exist only in the stub overlay.
# @lint-ignore-every AUTODEPS2

"""Pallas copies an HBM row through scratch VMEM before writing output."""

from __future__ import annotations

import unittest
from typing import assert_type, Literal, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar


def hbm_vmem_kernel[Rows: IntVar, Cols: IntVar](
    x_hbm_ref: pl.UnconstrainedInRef[[Rows, Cols]],
    out_vmem_ref: pl.OutRef[[1, Cols]],
    scratch_vmem_ref: pltpu.VmemScratchRef[[1, Cols]],
) -> None:
    pltpu.sync_copy(x_hbm_ref.at[0:1], scratch_vmem_ref)
    out_vmem_ref[...] = scratch_vmem_ref[...] + 1


def first_row_plus_one[Rows: IntVar, Cols: IntVar](
    x: jax.Array[[Rows, Cols]], dtype: object
) -> jax.Array[[1, Cols]]:
    _, cols = x.shape
    x_spec: pl.BlockSpec[[Rows, Cols], Literal["unconstrained"]] = pl.BlockSpec(
        memory_space=pl.ANY
    )
    scratch: pltpu.VMEM[[1, Cols]] = pltpu.VMEM((1, cols), dtype)
    call = pl.pallas_call(
        hbm_vmem_kernel,
        in_specs=[x_spec],
        scratch_shapes=(scratch,),
        out_shape=jax.ShapeDtypeStruct((1, cols), dtype),
        interpret=True,
    )
    return call(x)


def test_wrong_host_input_rank[Rows: IntVar](
    x: jax.Array[[Rows]],
) -> None:
    first_row_plus_one(x, None)  # E: is not assignable


def test_wrong_host_input_columns[Rows: IntVar, Cols: IntVar, Other: IntVar](
    x_spec: pl.BlockSpec[[Rows, Cols], Literal["unconstrained"]],
    scratch: pltpu.VMEM[[1, Cols]],
    cols: Int[Cols],
    x: jax.Array[[Rows, Other]],
) -> None:
    call = pl.pallas_call(
        hbm_vmem_kernel,
        in_specs=[x_spec],
        scratch_shapes=(scratch,),
        out_shape=jax.ShapeDtypeStruct((1, cols), None),
    )
    call(x)  # E: is not assignable


def test_wrong_source_slice[Rows: IntVar, Cols: IntVar](
    x: pl.UnconstrainedInRef[[Rows, Cols]],
) -> None:
    x.at[1:2]  # E: Cannot index into


def test_wrong_scratch_copy[Rows: IntVar, Cols: IntVar, Other: IntVar](
    x: pl.UnconstrainedInRef[[Rows, Cols]],
    scratch: pltpu.VmemScratchRef[[1, Other]],
) -> None:
    pltpu.sync_copy(x.at[0:1], scratch)  # E: is not assignable


def test_wrong_output_width[Cols: IntVar, Other: IntVar](
    out: pl.OutRef[[1, Other]],
    scratch: pltpu.VmemScratchRef[[1, Cols]],
) -> None:
    out[...] = scratch[...] + 1  # E: Cannot set item


def test_wrong_output_allocation[Rows: IntVar, Cols: IntVar](
    x_spec: pl.BlockSpec[[Rows, Cols], Literal["unconstrained"]],
    scratch: pltpu.VMEM[[1, Cols]],
    cols: Int[Cols],
) -> None:
    pl.pallas_call(  # E: No matching overload
        hbm_vmem_kernel,
        in_specs=[x_spec],
        scratch_shapes=(scratch,),
        out_shape=jax.ShapeDtypeStruct((2, cols), None),
    )


def test_wrong_scratch_allocation[Rows: IntVar, Cols: IntVar, Other: IntVar](
    x_spec: pl.BlockSpec[[Rows, Cols], Literal["unconstrained"]],
    scratch: pltpu.VMEM[[1, Other]],
    cols: Int[Cols],
) -> None:
    pl.pallas_call(  # E: No matching overload
        hbm_vmem_kernel,
        in_specs=[x_spec],
        scratch_shapes=(scratch,),
        out_shape=jax.ShapeDtypeStruct((1, cols), None),
    )


def test_zero_source_rows_are_not_excluded[Cols: IntVar](
    x: jax.Array[[0, Cols]],
) -> None:
    assert_type(first_row_plus_one(x, None), jax.Array[[1, Cols]])


def test_output_shape[Rows: IntVar, Cols: IntVar](
    x: jax.Array[[Rows, Cols]],
) -> None:
    assert_type(first_row_plus_one(x, None), jax.Array[[1, Cols]])


if not TYPE_CHECKING:

    class HbmVmemCopyTest(unittest.TestCase):
        def test_cpu_interpreter(self) -> None:
            x = jnp.arange(24, dtype=jnp.float32).reshape(4, 6)
            self.assertEqual(
                first_row_plus_one(x, x.dtype).tolist(), [[1, 2, 3, 4, 5, 6]]
            )

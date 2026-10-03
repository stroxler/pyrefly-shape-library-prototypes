# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Semantic annotation names exist only in the stub overlay.
# @lint-ignore-every AUTODEPS2

"""Typed version of the explicit program-id example in JAX's Pallas quickstart."""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING

import jax
from jax.experimental import pallas as pl

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar
else:
    import jax.numpy as jnp


def iota_kernel[Length: IntVar](o_ref: pl.OutRef[[Length]]) -> None:
    i = pl.program_id(0)
    o_ref[i] = i


def iota[Length: IntVar](length: Int[Length], dtype: object) -> jax.Array[[Length]]:
    make = pl.pallas_call(
        iota_kernel,
        out_shape=jax.ShapeDtypeStruct((length,), dtype),
        grid=(length,),
        interpret=True,
    )
    return make()


def test_wrong_grid_length[Length: IntVar, Other: IntVar](
    length: Int[Length], other: Int[Other]
) -> None:
    pl.pallas_call(  # E: No matching overload
        iota_kernel,
        out_shape=jax.ShapeDtypeStruct((length,), None),
        grid=(other,),
    )


def test_wrong_output_rank[Rows: IntVar, Cols: IntVar](
    rows: Int[Rows], cols: Int[Cols]
) -> None:
    pl.pallas_call(  # E: No matching overload
        iota_kernel,
        out_shape=jax.ShapeDtypeStruct((rows, cols), None),
        grid=(rows,),
    )


def test_wrong_program_axis[Length: IntVar](o_ref: pl.OutRef[[Length]]) -> None:
    i = pl.program_id(1)
    o_ref[i] = i  # E: Cannot set item


def test_wrong_kernel_output_role[Length: IntVar](
    o_ref: pl.OutRef[[Length]], x_ref: pl.InRef[[Length]]
) -> None:
    o_ref[:] = x_ref[:]  # This verifies the permitted full-tile write.
    o_ref[:] = pl.program_id(0)  # E: Cannot set item


if not TYPE_CHECKING:

    class IotaTest(unittest.TestCase):
        def test_full_output_ref_program_index(self) -> None:
            self.assertEqual(iota(8, jnp.int32).tolist(), list(range(8)))

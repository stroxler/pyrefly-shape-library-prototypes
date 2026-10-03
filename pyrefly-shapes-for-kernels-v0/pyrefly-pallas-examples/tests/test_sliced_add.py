# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Semantic annotation names exist only in the stub overlay.
# @lint-ignore-every AUTODEPS2

"""Typed version of the compositional Ref slices in JAX's Pallas quickstart.

The kernel body comes from JAX (Apache-2.0, see the JAX repository LICENSE).
"""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING

import jax
from jax.experimental import pallas as pl

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar
else:
    import jax.numpy as jnp


def add_sliced_kernel[Half: IntVar](
    x_ref: pl.InRef[[2 * Half]],
    y_ref: pl.InRef[[2 * Half]],
    o_ref: pl.OutRef[[4 * Half]],
) -> None:
    small_mid = x_ref.shape[0] // 2

    x_left = x_ref.at[:small_mid]
    x_right = x_ref.at[small_mid:]
    y_left = y_ref.at[:small_mid]
    y_right = y_ref.at[small_mid:]

    # The output shape is (4*small_mid).
    large_mid = 2 * small_mid
    o_ref.at[:large_mid][:small_mid] = x_left[...] + y_left[...]
    o_ref.at[:large_mid][small_mid:] = x_left[...] + y_right[...]
    o_ref.at[large_mid:][:small_mid] = x_right[...] + y_left[...]
    o_ref.at[large_mid:][small_mid:] = x_right[...] + y_right[...]


def sliced_add[InputLength: IntVar, OutputLength: IntVar](
    x: jax.Array[[InputLength]],
    y: jax.Array[[InputLength]],
    input_length: Int[InputLength],
    output_length: Int[OutputLength],
    dtype: object,
) -> jax.Array[[OutputLength]]:
    x_spec: pl.BlockSpec[[InputLength]] = pl.BlockSpec((input_length,), lambda i: (0,))
    output_spec: pl.BlockSpec[[OutputLength]] = pl.BlockSpec(
        (output_length,), lambda i: (0,)
    )
    add = pl.pallas_call(  # E: No matching overload
        add_sliced_kernel,
        out_shape=jax.ShapeDtypeStruct((output_length,), dtype),
        grid=(1,),
        in_specs=(x_spec, x_spec),
        out_specs=output_spec,
        interpret=True,
    )
    return add(x, y)


def test_wrong_input_extent[InputLength: IntVar, Other: IntVar, OutputLength: IntVar](
    x: jax.Array[[InputLength]],
    y: jax.Array[[Other]],
    input_length: Int[InputLength],
    output_length: Int[OutputLength],
) -> None:
    sliced_add(
        x,
        y,  # E: is not assignable to parameter
        input_length,
        output_length,
        None,
    )


def test_wrong_input_length[InputLength: IntVar, Other: IntVar, OutputLength: IntVar](
    x: jax.Array[[InputLength]],
    y: jax.Array[[InputLength]],
    wrong_input_length: Int[Other],
    output_length: Int[OutputLength],
) -> None:
    sliced_add(
        x,
        y,
        wrong_input_length,  # E: is not assignable to parameter
        output_length,
        None,
    )


def test_wrong_input_split[Half: IntVar](
    x_ref: pl.InRef[[2 * Half]],
) -> None:
    x_ref.at[: x_ref.shape[0]]  # E: Cannot index into


def test_wrong_output_ref_width[Half: IntVar](
    x_ref: pl.InRef[[2 * Half]], o_ref: pl.OutRef[[2 * Half]]
) -> None:
    small_mid = x_ref.shape[0] // 2
    large_mid = 2 * small_mid
    o_ref.at[:large_mid]  # E: Cannot index into


def test_wrong_index_map_rank[Half: IntVar](
    length: Int[4 * Half],
) -> None:
    pl.BlockSpec(  # E: No matching overload
        (length,), lambda i: (i, i)
    )


def test_index_map_address_is_not_proven[Half: IntVar](
    length: Int[4 * Half],
) -> None:
    # The declared block size is correct, but the only program maps past the output.
    output_spec: pl.BlockSpec[[4 * Half]] = pl.BlockSpec((length,), lambda i: (i + 1,))
    _ = output_spec


if not TYPE_CHECKING:

    class SlicedAddTest(unittest.TestCase):
        def test_composed_slices(self) -> None:
            x = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
            y = jnp.array([10, 20, 30, 40], dtype=jnp.int32)
            self.assertEqual(
                sliced_add(x, y, 4, 8, x.dtype).tolist(),
                [11, 22, 31, 42, 13, 24, 33, 44],
            )

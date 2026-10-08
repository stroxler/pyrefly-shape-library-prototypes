"""Pallas's program-indexed output has no host inputs or block specs."""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any, assert_type, cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from shape_extensions import Int, IntVar


def iota_kernel[Length: IntVar](o_ref: pl.OutRef[[Length]]) -> None:
    # Body from the JAX Pallas quickstart's explicit program-id example.
    i = pl.program_id(0)
    o_ref[i] = i


def checked_iota[Length: IntVar](
    length: Int[Length], dtype: object
) -> jax.Array[[Length]]:
    """Tie the program grid and the single full-length output Ref together."""
    if type(length) is not int or length <= 0:
        raise ValueError("The program grid needs a positive length")
    launch = pl.pallas_call(
        iota_kernel,
        out_shape=jax.ShapeDtypeStruct((length,), dtype),
        grid=(length,),
        interpret=True,
    )
    return launch()


if TYPE_CHECKING:

    def wrong_grid[Length: IntVar, Other: IntVar](
        length: Int[Length], other: Int[Other]
    ) -> None:
        pl.pallas_call(  # pyrefly: ignore[no-matching-overload]
            iota_kernel,
            out_shape=jax.ShapeDtypeStruct((length,), None),
            grid=(other,),
        )

    def wrong_program_axis[Length: IntVar](o_ref: pl.OutRef[[Length]]) -> None:
        i = pl.program_id(1)
        o_ref[i] = i  # pyrefly: ignore[unsupported-operation]

    known_length: Int[8] = cast(Any, 8)
    assert_type(checked_iota(known_length, jnp.int32), jax.Array[[8]])


class IotaTest(unittest.TestCase):
    def test_program_index(self) -> None:
        self.assertEqual(checked_iota(8, jnp.int32).tolist(), list(range(8)))

    def test_empty_grid_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive length"):
            checked_iota(0, jnp.int32)

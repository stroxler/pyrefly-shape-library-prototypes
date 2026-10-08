"""A minimal TPU-style vector tile with an explicitly checked Pallas boundary.

The kernel body comes from docs/pallas/quickstart.md. The TPU-specific
quickstart uses a separate two-dimensional pipelined addition instead; this
single-dimensional example is the matching vector-add analogue.
CPU interpretation tests the computation, not TPU memory movement or scheduling.
"""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any, assert_type, cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from shape_extensions import Int, IntVar

from pallas_library.layout import checked_pallas_call, grid_axis, vector_layout

Block = IntVar("Block")


def tpu_add_kernel(
    x_ref: pl.InRef[[Block]], y_ref: pl.InRef[[Block]], o_ref: pl.OutRef[[Block]]
) -> None:
    x, y = x_ref[...], y_ref[...]
    o_ref[...] = x + y


def checked_tpu_add[Length: IntVar, Tile: IntVar](
    x: jax.Array[[Length]],
    y: jax.Array[[Length]],
    *,
    block_size: Int[Tile],
    output_dtype: object = jnp.float32,
) -> jax.Array[[Length]]:
    """Check host rank, matching dtype/device, and output allocation metadata."""
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("Inputs must be vectors of the same length")
    if x.dtype != output_dtype or y.dtype != output_dtype:
        raise ValueError("Inputs and declared output must have the same dtype")
    if x.device != y.device:
        raise ValueError("Input devices must match")
    length = x.shape[0]
    if length == 0:
        return cast(jax.Array[[Length]], jnp.empty((0,), dtype=output_dtype))
    layout = vector_layout(
        tpu_add_kernel,
        out_shape=jax.ShapeDtypeStruct((length,), output_dtype),
        grid=(grid_axis(length, block_size),),
        block=(block_size,),
        index_map=lambda i: (i,),
    )
    return checked_pallas_call(layout, interpret=True)(x, y)


class TpuVectorAddTest(unittest.TestCase):
    def test_cpu_tiles(self) -> None:
        for length in (0, 1, 8, 13):
            with self.subTest(length=length):
                x = cast(Any, jnp.arange)(length, dtype=jnp.float32)
                y = x * 3
                result = checked_tpu_add(x, y, block_size=8)
                self.assertEqual(result.shape, (length,))
                self.assertEqual(result.tolist(), (x + y).tolist())

    def test_invalid_host_contract(self) -> None:
        x = cast(Any, jnp.arange)(8, dtype=jnp.float32)
        with self.assertRaisesRegex(ValueError, "same length"):
            checked_tpu_add(x, x[:7], block_size=8)
        with self.assertRaisesRegex(ValueError, "declared output"):
            checked_tpu_add(x, x, block_size=8, output_dtype=jnp.int32)
        with self.assertRaisesRegex(ValueError, "block_size"):
            checked_tpu_add(x, x, block_size=cast(Any, 0))


if TYPE_CHECKING:
    typed_x: jax.Array[[13]] = cast(Any, jnp.arange)(13)
    assert_type(checked_tpu_add(typed_x, typed_x, block_size=8), jax.Array[[13]])
    wrong_y: jax.Array[[12]] = cast(Any, jnp.arange)(12)
    checked_tpu_add(typed_x, wrong_y, block_size=8)  # pyrefly: ignore[bad-argument-type]

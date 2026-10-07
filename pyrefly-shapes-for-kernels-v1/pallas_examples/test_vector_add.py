"""Run Pallas vector add with a checked JAX-array-to-Ref boundary."""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any, assert_type, cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from shape_extensions import Int, IntVar

from pallas_library.layout import checked_pallas_call, grid_axis, vector_layout

Block = IntVar("Block")


def add_kernel(
    x_ref: pl.InRef[[Block]], y_ref: pl.InRef[[Block]], o_ref: pl.OutRef[[Block]]
) -> None:
    x = x_ref[:]
    y = y_ref[:]
    o_ref[:] = x + y


def checked_add[Length: IntVar, Tile: IntVar](
    x: jax.Array[[Length]],
    y: jax.Array[[Length]],
    *,
    block_size: Int[Tile],
    output_dtype: object = jnp.float32,
) -> jax.Array[[Length]]:
    """Check allocation metadata before constructing a tiled Pallas call."""
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("Inputs must be one-dimensional with the same length")
    if x.dtype != output_dtype or y.dtype != output_dtype:
        raise ValueError("Inputs must have the output dtype")
    if x.device != y.device:
        raise ValueError("Input devices must match")
    length = x.shape[0]
    if length == 0:
        return cast(jax.Array[[Length]], cast(Any, jnp.empty)((0,), dtype=output_dtype))
    layout = vector_layout(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct((length,), output_dtype),
        grid=(grid_axis(length, block_size),),
        block=(block_size,),
        index_map=lambda i: (i,),
    )
    launch = checked_pallas_call(layout, interpret=True)
    return launch(x, y)


def design_doc_add(x: Any, y: Any) -> Any:
    """The design doc's inline 8-element call with explicit 1D index tuples."""
    add = pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct((8,), jnp.int32),
        in_specs=(
            pl.BlockSpec((2,), lambda i: (i,)),
            pl.BlockSpec((2,), lambda i: (i,)),
        ),
        out_specs=pl.BlockSpec((2,), lambda i: (i,)),
        grid=(pl.cdiv(8, 2),),
        interpret=True,
    )
    return add(x, y)


class VectorAddTest(unittest.TestCase):
    """Exercise complete and partial tiles, and reject invalid host inputs."""

    def test_checked_add(self) -> None:
        for length in (0, 1, 4, 10, 16):
            with self.subTest(length=length):
                x = cast(Any, jnp.arange)(length, dtype=jnp.float32)
                y = x * 3
                actual = checked_add(x, y, block_size=4)
                self.assertEqual(actual.shape, (length,))
                self.assertEqual(actual.tolist(), (x + y).tolist())
                self.assertEqual(
                    checked_add(y=y, x=x, block_size=4).tolist(),
                    actual.tolist(),
                )

    def test_checked_add_rejects_invalid_inputs(self) -> None:
        good = cast(Any, jnp.arange)(10, dtype=jnp.float32)
        for error, invalid in (
            ("same length", cast(Any, jnp.arange)(11, dtype=jnp.float32)),
            ("one-dimensional", good.reshape(2, 5)),
            ("dtype", cast(Any, jnp.arange)(10, dtype=jnp.int32)),
        ):
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    checked_add(
                        good,
                        invalid,
                        block_size=4,
                    )

        for invalid_block in (0, -1, 1.5):
            with self.subTest(block_size=invalid_block):
                with self.assertRaisesRegex(ValueError, "block_size"):
                    checked_add(
                        good,
                        good,
                        block_size=cast(Any, invalid_block),
                    )

    def test_layout_rejects_wrong_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "Grid size"):
            vector_layout(
                add_kernel,
                out_shape=jax.ShapeDtypeStruct((8,), jnp.float32),
                grid=(cast(Any, grid_axis(12, 4)),),
                block=(4,),
                index_map=lambda i: (i,),
            )

    def test_design_doc_call(self) -> None:
        x = cast(Any, jnp.arange)(8, dtype=jnp.int32)
        y = cast(Any, jnp.arange)(8, 16, dtype=jnp.int32)
        actual = design_doc_add(x, y)
        self.assertEqual(actual.tolist(), (x + y).tolist())
        checked = checked_add(
            x,
            y,
            block_size=2,
            output_dtype=jnp.int32,
        )
        self.assertEqual(actual.tolist(), checked.tolist())


if TYPE_CHECKING:
    typed_input: jax.Array[[8]] = cast(Any, jnp.arange)(8, dtype=jnp.float32)
    assert_type(
        checked_add(typed_input, typed_input, block_size=4),
        jax.Array[[8]],
    )

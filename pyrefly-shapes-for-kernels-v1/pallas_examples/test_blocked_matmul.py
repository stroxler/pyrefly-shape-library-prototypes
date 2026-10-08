"""Check the 2D BlockSpec mapping in JAX's Pallas quickstart matmul."""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any, assert_type, cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from shape_extensions import Int, IntVar

from pallas_library.layout import checked_pallas_call, grid_axis, matmul_layout

Rows = IntVar("Rows")
Inner = IntVar("Inner")
Cols = IntVar("Cols")
RowBlock = IntVar("RowBlock")
ColBlock = IntVar("ColBlock")


def matmul_kernel(
    x_ref: pl.InRef[[RowBlock, Inner]],
    y_ref: pl.InRef[[Inner, ColBlock]],
    z_ref: pl.OutRef[[RowBlock, ColBlock]],
) -> None:
    z_ref[...] = x_ref[...] @ y_ref[...]


def blocked_matmul[
    Rows: IntVar,
    Inner: IntVar,
    Cols: IntVar,
    RowBlock: IntVar,
    ColBlock: IntVar,
](
    x: jax.Array[[Rows, Inner]],
    y: jax.Array[[Inner, Cols]],
    *,
    row_block: Int[RowBlock],
    col_block: Int[ColBlock],
) -> jax.Array[[Rows, Cols]]:
    """Supply distinct left, right, and output block maps explicitly."""
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("Expected two-dimensional inputs")
    rows, inner = x.shape
    cols = y.shape[1]
    layout = matmul_layout(
        matmul_kernel,
        x_shape=x.shape,
        y_shape=y.shape,
        out_shape=jax.ShapeDtypeStruct((rows, cols), x.dtype),
        grid=(grid_axis(rows, row_block), grid_axis(cols, col_block)),
        x_block=(row_block, inner),
        y_block=(inner, col_block),
        out_block=(row_block, col_block),
        x_map=lambda i, j: (i, 0),
        y_map=lambda i, j: (0, j),
        out_map=lambda i, j: (i, j),
    )
    multiply = checked_pallas_call(layout, interpret=True)
    return multiply(x, y)


class BlockedMatmulTest(unittest.TestCase):
    """Check two-dimensional Refs and distinct aligned host block maps."""

    def test_reject_rank_one_input(self) -> None:
        x = cast(Any, jnp).ones((8,), dtype=jnp.float32)
        y = cast(Any, jnp).ones((8, 10), dtype=jnp.float32)
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            blocked_matmul(x, y, row_block=3, col_block=5)

    def test_mapped_matmul(self) -> None:
        x = cast(Any, jnp.arange)(48, dtype=jnp.float32).reshape(6, 8)
        y = cast(Any, jnp.arange)(80, dtype=jnp.float32).reshape(8, 10)
        result = blocked_matmul(x, y, row_block=3, col_block=5)
        self.assertEqual(result.shape, (6, 10))
        self.assertTrue(bool(cast(Any, jnp).allclose(result, x @ y)))

    def test_reject_inner_dimension_mismatch(self) -> None:
        x = cast(Any, jnp).ones((6, 8), dtype=jnp.float32)
        y = cast(Any, jnp).ones((9, 10), dtype=jnp.float32)
        with self.assertRaisesRegex(ValueError, "inner dimensions"):
            blocked_matmul(x, y, row_block=3, col_block=5)

    def test_reject_partial_output_blocks(self) -> None:
        x = cast(Any, jnp).ones((7, 8), dtype=jnp.float32)
        y = cast(Any, jnp).ones((8, 10), dtype=jnp.float32)
        with self.assertRaisesRegex(ValueError, "divisible"):
            blocked_matmul(x, y, row_block=3, col_block=5)

    def test_untyped_index_map_can_still_change_result(self) -> None:
        """Bypassing the typed constructor still allows an incorrect map."""
        x = cast(Any, jnp.arange)(48, dtype=jnp.float32).reshape(6, 8)
        y = cast(Any, jnp.arange)(80, dtype=jnp.float32).reshape(8, 10)
        # Deliberately bypass the static index-map contract to test its limit.
        layout = cast(Any, matmul_layout)(
            matmul_kernel,
            x_shape=(6, 8),
            y_shape=(8, 10),
            out_shape=jax.ShapeDtypeStruct((6, 10), jnp.float32),
            grid=(grid_axis(6, 3), grid_axis(10, 5)),
            x_block=(3, 8),
            y_block=(8, 5),
            out_block=(3, 5),
            x_map=lambda i, j: (0, 0),
            y_map=lambda i, j: (0, j),
            out_map=lambda i, j: (i, j),
        )
        multiply = checked_pallas_call(layout, interpret=True)
        result = multiply(x, y)
        self.assertFalse(bool(cast(Any, jnp).allclose(result, x @ y)))


if TYPE_CHECKING:
    typed_x: jax.Array[[6, 8]] = cast(Any, jnp).ones((6, 8))
    typed_y: jax.Array[[8, 10]] = cast(Any, jnp).ones((8, 10))
    row_block: Int[3] = cast(Int[3], 3)
    col_block: Int[5] = cast(Int[5], 5)
    assert_type(
        blocked_matmul(
            typed_x,
            typed_y,
            row_block=row_block,
            col_block=col_block,
        ),
        jax.Array[[6, 10]],
    )
    typed_layout = matmul_layout(
        matmul_kernel,
        x_shape=typed_x.shape,
        y_shape=typed_y.shape,
        out_shape=jax.ShapeDtypeStruct((6, 10), typed_x.dtype),
        grid=(
            grid_axis(typed_x.shape[0], row_block),
            grid_axis(typed_y.shape[1], col_block),
        ),
        x_block=(row_block, typed_x.shape[1]),
        y_block=(typed_y.shape[0], col_block),
        out_block=(row_block, col_block),
        x_map=lambda i, j: (i, 0),
        y_map=lambda i, j: (0, j),
        out_map=lambda i, j: (i, j),
    )
    assert_type(
        checked_pallas_call(typed_layout, interpret=True)(typed_x, typed_y),
        jax.Array[[6, 10]],
    )
    checked_pallas_call(typed_layout, interpret=True)(
        typed_y,  # pyrefly: ignore[bad-argument-type]
        typed_x,  # pyrefly: ignore[bad-argument-type]
    )
    # The checked layout rejects losing the row-grid index.
    matmul_layout(
        matmul_kernel,
        x_shape=typed_x.shape,
        y_shape=typed_y.shape,
        out_shape=jax.ShapeDtypeStruct((6, 10), typed_x.dtype),
        grid=(
            grid_axis(typed_x.shape[0], row_block),
            grid_axis(typed_y.shape[1], col_block),
        ),
        x_block=(row_block, typed_x.shape[1]),
        y_block=(typed_y.shape[0], col_block),
        out_block=(row_block, col_block),
        x_map=lambda i, j: (0, 0),  # pyrefly: ignore[bad-argument-type]
        y_map=lambda i, j: (0, j),
        out_map=lambda i, j: (i, j),
    )

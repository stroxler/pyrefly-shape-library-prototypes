"""Run the Pallas GPU softmax body on a matrix using vmap and CPU interpret mode."""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any, assert_type, cast

import jax
import jax.nn
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from shape_extensions import Int, IntVar

from pallas_library.layout import checked_pallas_call, row_layout

Rows = IntVar("Rows")
Cols = IntVar("Cols")
Block = IntVar("Block")


def _vmappable_softmax_kernel(
    input_ref: pl.InRef[[Cols]],
    probs_ref: pl.OutRef[[Cols]],
    *,
    block_row: Int[Block],
) -> None:
    row_len = input_ref.shape[-1]

    col_idx = jnp.arange(block_row)
    mask = col_idx < row_len
    row = plgpu.load(input_ref.at[col_idx], mask=mask, other=-float("inf"))

    row_max = jnp.max(row, axis=0)
    numerator = jnp.exp((row - row_max).astype(jnp.float32))
    denominator = jnp.sum(numerator, axis=0)

    plgpu.store(
        probs_ref.at[col_idx],
        (numerator / denominator).astype(probs_ref.dtype),
        mask=mask,
    )


def softmax(
    x: jax.Array[[Rows, Cols]],
    *,
    axis: int = -1,
    num_warps: int = 4,
    interpret: bool = True,
    debug: bool = False,
) -> jax.Array[[Rows, Cols]]:
    """Adapt upstream's vmapped host wrapper to CPU interpretation."""
    if x.ndim != 2:
        raise ValueError("Expected a two-dimensional array")
    axis = axis if axis >= 0 else len(x.shape) + axis
    if axis != len(x.shape) - 1:
        raise NotImplementedError("reductions along non-trailing dimension unsupported")

    row_len = x.shape[-1]
    block_row = pl.next_power_of_2(row_len)
    out_shape = jax.ShapeDtypeStruct(shape=(row_len,), dtype=x.dtype)

    def kernel(input_ref: pl.InRef[[Cols]], probs_ref: pl.OutRef[[Cols]]) -> None:
        _vmappable_softmax_kernel(input_ref, probs_ref, block_row=block_row)

    layout = row_layout(
        kernel,
        grid=(),
        out_shape=out_shape,
    )
    f = checked_pallas_call(
        layout,
        compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=1),
        debug=debug,
        interpret=interpret,
    )

    def apply_row(row: jax.Array[[Cols]]) -> jax.Array[[Cols]]:
        return f(row)

    return jax.vmap(apply_row)(x)


class MaskedSoftmaxTest(unittest.TestCase):
    """Verify mapped rows and a padded final column block on CPU."""

    def test_irregular_matrix(self) -> None:
        x = cast(Any, jnp.arange)(35, dtype=jnp.float32).reshape(5, 7)
        result = softmax(x)
        self.assertEqual(result.shape, (5, 7))
        self.assertTrue(
            bool(cast(Any, jnp).allclose(result, jax.nn.softmax(x, axis=-1), atol=1e-6))
        )

    def test_reject_nontrailing_axis(self) -> None:
        x = cast(Any, jnp.arange)(15, dtype=jnp.float32).reshape(3, 5)
        with self.assertRaisesRegex(NotImplementedError, "non-trailing"):
            softmax(x, axis=0)

    def test_reject_rank_one(self) -> None:
        x = cast(Any, jnp.arange)(5, dtype=jnp.float32)
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            softmax(x)

    def test_checked_row_layout_rejects_wrong_width(self) -> None:
        def kernel(input_ref: pl.InRef[[7]], probs_ref: pl.OutRef[[7]]) -> None:
            probs_ref[:] = input_ref[:]

        layout = row_layout(
            kernel,
            out_shape=jax.ShapeDtypeStruct((7,), jnp.float32),
            grid=(),
        )
        f = checked_pallas_call(
            layout,
            compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=1),
            interpret=True,
        )
        with self.assertRaisesRegex(ValueError, "declared layout"):
            f(cast(Any, jnp.arange)(8, dtype=jnp.float32))


if TYPE_CHECKING:

    def typed_kernel(x_ref: pl.InRef[[7]], y_ref: pl.OutRef[[7]]) -> None:
        y_ref[:] = x_ref[:]

    typed_row: jax.Array[[7]] = cast(Any, jnp.arange)(7, dtype=jnp.float32)
    typed_layout = row_layout(
        typed_kernel,
        out_shape=jax.ShapeDtypeStruct((7,), jnp.float32),
        grid=(),
    )
    typed_call = checked_pallas_call(
        typed_layout,
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=1),
        interpret=True,
    )
    assert_type(typed_call(typed_row), jax.Array[[7]])

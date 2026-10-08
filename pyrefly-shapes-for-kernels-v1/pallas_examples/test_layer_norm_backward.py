"""Check the unchanged JAX/Pallas layer-norm input-gradient kernel."""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any, assert_type, cast

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from shape_extensions import Int, IntVar

from pallas_library.layout import checked_pallas_call, row_input_gradient_layout

Features = IntVar("Features")
Block = IntVar("Block")


def layer_norm_backward_kernel_dx(
    # Inputs
    x_ref: pl.InRef[[Features]],
    weight_ref: pl.InRef[[Features]],
    bias_ref: pl.InRef[[Features]],
    do_ref: pl.InRef[[Features]],
    mean_ref: pl.InRef[[]],
    rstd_ref: pl.InRef[[]],
    # Outputs
    dx_ref: pl.OutRef[[Features]],
    *,
    eps: float,
    block_size: Int[Block],
):
    n_col = x_ref.shape[0]

    def mean_body(i: int, acc: tuple[pl.Tile[[Block]], pl.Tile[[Block]]]):
        col_idx = i * block_size + jnp.arange(block_size)
        mask = col_idx < n_col
        a = plgpu.load(
            x_ref.at[col_idx], mask=mask, other=0.0, eviction_policy="evict_last"
        ).astype(jnp.float32)
        dout = plgpu.load(
            do_ref.at[col_idx], mask=mask, other=0.0, eviction_policy="evict_last"
        ).astype(jnp.float32)
        weight = plgpu.load(
            weight_ref.at[col_idx],
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        ).astype(jnp.float32)
        a_hat = (a - mean_ref[...]) * rstd_ref[...]
        wdout = weight * dout
        mean1_acc, mean2_acc = acc
        return mean1_acc + a_hat * wdout, mean2_acc + wdout

    mean1, mean2 = lax.fori_loop(
        0,
        pl.cdiv(n_col, block_size),
        mean_body,
        init_val=(jnp.zeros(block_size), jnp.zeros(block_size)),
    )
    mean1 = mean1.sum() / n_col
    mean2 = mean2.sum() / n_col

    @pl.loop(0, pl.cdiv(n_col, block_size))
    def dx_body(i: int):
        col_idx = i * block_size + jnp.arange(block_size)
        mask = col_idx < n_col
        a = plgpu.load(
            x_ref.at[col_idx], mask=mask, other=0.0, eviction_policy="evict_last"
        ).astype(jnp.float32)
        dout = plgpu.load(
            do_ref.at[col_idx], mask=mask, other=0.0, eviction_policy="evict_last"
        ).astype(jnp.float32)
        weight = plgpu.load(
            weight_ref.at[col_idx],
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        ).astype(jnp.float32)
        a_hat = (a - mean_ref[...]) * rstd_ref[...]
        wdout = weight * dout
        da = (wdout - (a_hat * mean1 + mean2)) * rstd_ref[...]
        plgpu.store(dx_ref.at[col_idx], da.astype(dx_ref.dtype), mask=mask)


def layer_norm_input_grad[Features: IntVar, Block: IntVar](
    x: jax.Array[[Features]],
    weight: jax.Array[[Features]],
    bias: jax.Array[[Features]],
    dout: jax.Array[[Features]],
    mean: jax.Array[[]],
    rstd: jax.Array[[]],
    *,
    block_size: Int[Block],
) -> jax.Array[[Features]]:
    """Validate a saved-statistics row and run one Pallas input-gradient program."""
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be positive")

    def kernel(
        x_ref: pl.InRef[[Features]],
        weight_ref: pl.InRef[[Features]],
        bias_ref: pl.InRef[[Features]],
        do_ref: pl.InRef[[Features]],
        mean_ref: pl.InRef[[]],
        rstd_ref: pl.InRef[[]],
        dx_ref: pl.OutRef[[Features]],
    ) -> None:
        layer_norm_backward_kernel_dx(
            x_ref,
            weight_ref,
            bias_ref,
            do_ref,
            mean_ref,
            rstd_ref,
            dx_ref,
            eps=1e-5,
            block_size=block_size,
        )

    layout = row_input_gradient_layout(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    )
    return checked_pallas_call(layout, interpret=True)(
        x, weight, bias, dout, mean, rstd
    )


class LayerNormBackwardTest(unittest.TestCase):
    """Exercise partial blocks, shape mismatches, and saved scalar statistics."""

    def test_gradient(self) -> None:
        x = cast(Any, jnp).array([0.5, -2.0, 1.0, 4.0, 0.1], dtype=jnp.float32)
        w = cast(Any, jnp).array([1.0, 2.0, 0.5, 1.0, 3.0], dtype=jnp.float32)
        b = cast(Any, jnp).zeros(5, dtype=jnp.float32)
        dout = cast(Any, jnp).array([0.2, -0.8, 1.2, 0.1, 1.0], dtype=jnp.float32)
        mean = x.mean()
        rstd = 1 / jnp.sqrt(((x - mean) ** 2).mean() + 1e-5)
        result = layer_norm_input_grad(
            x, w, b, dout, mean, cast(jax.Array[[]], rstd), block_size=4
        )
        xhat = (x - mean) * rstd
        wdout = w * dout
        expected = (wdout - xhat * (xhat * wdout).mean() - wdout.mean()) * rstd
        self.assertEqual(result.shape, (5,))
        self.assertTrue(bool(cast(Any, jnp).allclose(result, expected, atol=1e-6)))

    def test_mismatched_saved_statistics(self) -> None:
        row = cast(Any, jnp).ones(5, dtype=jnp.float32)
        scalar = cast(Any, jnp).array(0.5, dtype=jnp.float32)
        with self.assertRaisesRegex(ValueError, "shapes must match"):
            layer_norm_input_grad(
                row, row, row, row, row, scalar, block_size=4
            )
        with self.assertRaisesRegex(ValueError, "dtypes must match"):
            layer_norm_input_grad(
                row,
                row,
                row,
                row,
                scalar,
                cast(Any, jnp).array(0.5, dtype=jnp.float16),
                block_size=4,
            )


if TYPE_CHECKING:

    def typed_gradient[Features: IntVar, Other: IntVar](
        row: jax.Array[[Features]],
        wrong: jax.Array[[Other]],
        stat: jax.Array[[]],
    ) -> None:
        assert_type(
            layer_norm_input_grad(
                row, row, row, row, stat, stat, block_size=4
            ),
            jax.Array[[Features]],
        )
        layer_norm_input_grad(
            row,
            wrong,  # pyrefly: ignore[bad-argument-type]
            row,
            row,
            stat,
            stat,
            block_size=4,
        )

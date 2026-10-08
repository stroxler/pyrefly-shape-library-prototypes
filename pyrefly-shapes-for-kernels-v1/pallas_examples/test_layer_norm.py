"""Run JAX's masked Pallas layer-norm forward kernel with checked row statistics."""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any, assert_type, cast

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from shape_extensions import Int, IntVar

from pallas_library.layout import checked_pallas_call, row_statistics_layout

Features = IntVar("Features")
Block = IntVar("Block")


def layer_norm_forward_kernel(
    x_ref: pl.InRef[[Features]],
    weight_ref: pl.InRef[[Features]],
    bias_ref: pl.InRef[[Features]],  # Input arrays
    o_ref: pl.OutRef[[Features]],
    mean_ref: pl.OutRef[[]] | None = None,
    rstd_ref: pl.OutRef[[]] | None = None,  # Output arrays
    *,
    eps: float,
    block_size: Int[Block],
):
    n_col = x_ref.shape[0]

    def mean_body(i: int, acc: pl.Tile[[Block]]):
        col_idx = i * block_size + jnp.arange(block_size)
        mask = col_idx < n_col
        a = plgpu.load(
            x_ref.at[col_idx], mask=mask, other=0.0, eviction_policy="evict_last"
        ).astype(jnp.float32)
        return acc + a

    mean = lax.fori_loop(
        0,
        pl.cdiv(n_col, block_size),
        mean_body,
        init_val=jnp.zeros(block_size),
    ).sum()
    mean /= n_col

    def var_body(i: int, acc: pl.Tile[[Block]]):
        col_idx = i * block_size + jnp.arange(block_size)
        mask = col_idx < n_col
        a = plgpu.load(
            x_ref.at[col_idx], mask=mask, other=0.0, eviction_policy="evict_last"
        ).astype(jnp.float32)
        a = jnp.where(mask, a - mean, 0.0)
        return acc + a * a

    var = lax.fori_loop(
        0,
        pl.cdiv(n_col, block_size),
        var_body,
        init_val=jnp.zeros(block_size),
    ).sum()
    var /= n_col
    rstd = 1 / jnp.sqrt(var + eps)
    if mean_ref is not None:
        mean_ref[...] = mean.astype(mean_ref.dtype)
    if rstd_ref is not None:
        rstd_ref[...] = rstd.astype(rstd_ref.dtype)

    @pl.loop(0, pl.cdiv(n_col, block_size))
    def body(i: int):
        col_idx = i * block_size + jnp.arange(block_size)
        mask = col_idx < n_col
        weight = plgpu.load(weight_ref.at[col_idx], mask=mask)
        bias = plgpu.load(bias_ref.at[col_idx], mask=mask)
        x = plgpu.load(
            x_ref.at[col_idx], mask=mask, other=0.0, eviction_policy="evict_first"
        ).astype(jnp.float32)
        out = (x - mean) * rstd * weight + bias
        plgpu.store(o_ref.at[col_idx], out.astype(o_ref.dtype), mask=mask)


def layer_norm_row[Features: IntVar, Block: IntVar](
    x: jax.Array[[Features]],
    weight: jax.Array[[Features]],
    bias: jax.Array[[Features]],
    *,
    block_size: Int[Block],
    eps: float = 1e-5,
) -> tuple[jax.Array[[Features]], jax.Array[[]], jax.Array[[]]]:
    """Bind three full rows and return the normalized row and scalar statistics."""
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be positive")

    def kernel(
        x_ref: pl.InRef[[Features]],
        weight_ref: pl.InRef[[Features]],
        bias_ref: pl.InRef[[Features]],
        out_ref: pl.OutRef[[Features]],
        mean_ref: pl.OutRef[[]],
        rstd_ref: pl.OutRef[[]],
    ) -> None:
        layer_norm_forward_kernel(
            x_ref,
            weight_ref,
            bias_ref,
            out_ref,
            mean_ref,
            rstd_ref,
            eps=eps,
            block_size=block_size,
        )

    row = jax.ShapeDtypeStruct((x.shape[0],), x.dtype)
    scalar = jax.ShapeDtypeStruct((), x.dtype)
    layout = row_statistics_layout(
        kernel,
        out_shape=(row, scalar, scalar),
        grid=(),
    )
    return checked_pallas_call(layout, interpret=True)(x, weight, bias)


class LayerNormTest(unittest.TestCase):
    """Exercise a partial final block and the checked three-output boundary."""

    def test_partial_block_and_statistics(self) -> None:
        x = cast(Any, jnp).asarray([1.0, -2.0, 3.0, 4.0, -1.0], dtype=jnp.float32)
        weight = cast(Any, jnp).asarray([1.0, 0.5, 1.5, 2.0, 0.25], dtype=jnp.float32)
        bias = cast(Any, jnp).asarray([0.1, 0.0, -0.1, 0.5, -0.5], dtype=jnp.float32)
        result, mean, rstd = layer_norm_row(x, weight, bias, block_size=4)
        expected_mean = cast(Any, jnp).mean(x)
        expected_rstd = 1 / jnp.sqrt(
            cast(Any, jnp).mean((x - expected_mean) ** 2) + 1e-5
        )
        self.assertEqual((result.shape, mean.shape, rstd.shape), ((5,), (), ()))
        self.assertTrue(bool(cast(Any, jnp).allclose(mean, expected_mean, atol=1e-6)))
        self.assertTrue(bool(cast(Any, jnp).allclose(rstd, expected_rstd, atol=1e-6)))
        self.assertTrue(
            bool(
                cast(Any, jnp).allclose(
                    result,
                    (x - expected_mean) * expected_rstd * weight + bias,
                    atol=1e-6,
                )
            )
        )

    def test_reject_wrong_input_shape(self) -> None:
        x = cast(Any, jnp.ones)(5, dtype=jnp.float32)
        weight = cast(Any, jnp.ones)(4, dtype=jnp.float32)
        with self.assertRaisesRegex(ValueError, "shapes must match"):
            layer_norm_row(x, weight, x, block_size=4)

    def test_reject_wrong_input_dtype(self) -> None:
        x = cast(Any, jnp.ones)(5, dtype=jnp.float32)
        weight = cast(Any, jnp.ones)(5, dtype=jnp.int32)
        with self.assertRaisesRegex(ValueError, "dtypes must match"):
            layer_norm_row(x, weight, x, block_size=4)

    def test_reject_non_scalar_statistics(self) -> None:
        def kernel(
            x_ref: pl.InRef[[5]],
            weight_ref: pl.InRef[[5]],
            bias_ref: pl.InRef[[5]],
            out_ref: pl.OutRef[[5]],
            mean_ref: pl.OutRef[[]],
            rstd_ref: pl.OutRef[[]],
        ) -> None:
            layer_norm_forward_kernel(
                x_ref,
                weight_ref,
                bias_ref,
                out_ref,
                mean_ref,
                rstd_ref,
                eps=1e-5,
                block_size=4,
            )

        row = jax.ShapeDtypeStruct((5,), jnp.float32)
        wrong_stat = cast(Any, jax.ShapeDtypeStruct((1,), jnp.float32))
        scalar = jax.ShapeDtypeStruct((), jnp.float32)
        with self.assertRaisesRegex(ValueError, "Statistics outputs must be scalars"):
            row_statistics_layout(
                kernel,
                out_shape=(row, wrong_stat, scalar),
                grid=(),
            )


if TYPE_CHECKING:

    def typed_row[Features: IntVar, Other: IntVar, Block: IntVar](
        x: jax.Array[[Features]],
        weight: jax.Array[[Features]],
        bias: jax.Array[[Features]],
        wrong: jax.Array[[Other]],
        block_size: Int[Block],
    ) -> None:
        assert_type(
            layer_norm_row(x, weight, bias, block_size=block_size),
            tuple[jax.Array[[Features]], jax.Array[[]], jax.Array[[]]],
        )
        layer_norm_row(x, wrong, bias, block_size=block_size)  # pyrefly: ignore[bad-argument-type]

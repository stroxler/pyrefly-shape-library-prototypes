"""Type and execute JAX's masked Pallas RMSNorm forward row kernel.

The kernel body is copied from JAX's deprecated
jax/experimental/pallas/ops/gpu/rms_norm.py (Apache-2.0); only its parameter
annotations are new.
"""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any, assert_type, cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from shape_extensions import Int, IntVar

from pallas_library.layout import checked_pallas_call, row_rms_layout

Features = IntVar("Features")
Block = IntVar("Block")


def rms_norm_forward_kernel(
    x_ref: pl.InRef[[Features]],
    weight_ref: pl.InRef[[Features]],
    bias_ref: pl.InRef[[Features]],
    o_ref: pl.OutRef[[Features]],
    rstd_ref: pl.OutRef[[]] | None = None,
    *,
    eps: float,
    block_size: Int[Block],
):
    n_col = x_ref.shape[0]

    def var_body(i: int, acc: pl.Tile[[Block]]):
        col_idx = i * block_size + jnp.arange(block_size)
        mask = col_idx < n_col
        a = plgpu.load(
            x_ref.at[col_idx], mask=mask, other=0.0, eviction_policy="evict_last"
        ).astype(jnp.float32)
        a = jnp.where(mask, a, 0.0)
        return acc + a * a

    var = lax.fori_loop(
        0, pl.cdiv(n_col, block_size), var_body, init_val=jnp.zeros(block_size)
    ).sum()
    var /= n_col
    rstd = 1 / jnp.sqrt(var + eps)
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
        out = x * rstd * weight + bias
        plgpu.store(o_ref.at[col_idx], out.astype(o_ref.dtype), mask=mask)


def rms_norm_row[Features: IntVar, Block: IntVar](
    x: jax.Array[[Features]],
    weight: jax.Array[[Features]],
    bias: jax.Array[[Features]],
    *,
    block_size: Int[Block],
    eps: float = 1e-5,
) -> tuple[jax.Array[[Features]], jax.Array[[]]]:
    """Bind three equal-length logical rows and one optional scalar result."""
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be positive")
    if x.ndim != 1 or weight.ndim != 1 or bias.ndim != 1:
        raise ValueError("RMSNorm inputs must be one-dimensional")
    if not isinstance(eps, (int, float)) or eps <= 0:
        raise ValueError("eps must be positive")

    def kernel(
        x_ref: pl.InRef[[Features]],
        weight_ref: pl.InRef[[Features]],
        bias_ref: pl.InRef[[Features]],
        out_ref: pl.OutRef[[Features]],
        rstd_ref: pl.OutRef[[]],
    ) -> None:
        rms_norm_forward_kernel(
            x_ref,
            weight_ref,
            bias_ref,
            out_ref,
            rstd_ref,
            eps=eps,
            block_size=block_size,
        )

    row = jax.ShapeDtypeStruct((x.shape[0],), x.dtype)
    scalar = jax.ShapeDtypeStruct((), x.dtype)
    layout = row_rms_layout(kernel, out_shape=(row, scalar))
    return checked_pallas_call(layout, interpret=True)(x, weight, bias)


class RmsNormTest(unittest.TestCase):
    """Check tail masking, output statistic, and rejected host signatures."""

    def test_partial_block_and_statistic(self) -> None:
        x = cast(Any, jnp).asarray([1.0, -2.0, 3.0, 4.0, -1.0], dtype=jnp.float32)
        weight = cast(Any, jnp).asarray([1.0, 0.5, 1.5, 2.0, 0.25], dtype=jnp.float32)
        bias = cast(Any, jnp).asarray([0.1, 0.0, -0.1, 0.5, -0.5], dtype=jnp.float32)
        output, rstd = rms_norm_row(x, weight, bias, block_size=4)
        expected_rstd = 1 / jnp.sqrt(cast(Any, jnp).mean(x * x) + 1e-5)
        self.assertEqual((output.shape, rstd.shape), ((5,), ()))
        self.assertTrue(bool(cast(Any, jnp).allclose(rstd, expected_rstd, atol=1e-6)))
        self.assertTrue(
            bool(
                cast(Any, jnp).allclose(
                    output, x * expected_rstd * weight + bias, atol=1e-6
                )
            )
        )

    def test_reject_wrong_input_shape_and_dtype(self) -> None:
        x = cast(Any, jnp).ones(5, dtype=jnp.float32)
        with self.assertRaisesRegex(ValueError, "shapes must match"):
            rms_norm_row(x, cast(Any, jnp).ones(4), x, block_size=4)
        with self.assertRaisesRegex(ValueError, "dtypes must match"):
            rms_norm_row(x, cast(Any, jnp).ones(5, dtype=jnp.int32), x, block_size=4)

    def test_reject_wrong_statistic_shape(self) -> None:
        def kernel(
            x_ref: pl.InRef[[5]],
            weight_ref: pl.InRef[[5]],
            bias_ref: pl.InRef[[5]],
            out_ref: pl.OutRef[[5]],
            rstd_ref: pl.OutRef[[]],
        ) -> None:
            pass

        row = jax.ShapeDtypeStruct((5,), jnp.float32)
        scalar = cast(Any, jax.ShapeDtypeStruct((1,), jnp.float32))
        with self.assertRaisesRegex(ValueError, "RMS statistic"):
            row_rms_layout(kernel, out_shape=(row, scalar))


if TYPE_CHECKING:

    def check_shape[Features: IntVar, Other: IntVar, Block: IntVar](
        x: jax.Array[[Features]],
        weight: jax.Array[[Features]],
        bias: jax.Array[[Features]],
        wrong: jax.Array[[Other]],
        block: Int[Block],
    ) -> None:
        assert_type(
            rms_norm_row(x, weight, bias, block_size=block),
            tuple[jax.Array[[Features]], jax.Array[[]]],
        )
        rms_norm_row(
            x,
            wrong,  # pyrefly: ignore[bad-argument-type]
            bias,
            block_size=block,
        )

"""Check JAX's unchanged layer-norm weight/bias gradient kernel."""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any, assert_type, cast

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from shape_extensions import Int, IntVar

from pallas_library.layout import (
    checked_pallas_call,
    layer_norm_weight_gradient_layout,
)

Rows = IntVar("Rows")
Cols = IntVar("Cols")
RowBlock = IntVar("RowBlock")
ColBlock = IntVar("ColBlock")


def layer_norm_backward_kernel_dw_db(
    # Inputs
    x_ref: pl.LayerNormMatrixRef[Rows, Cols],
    weight_ref: pl.LayerNormVectorRef[Cols],
    bias_ref: pl.LayerNormVectorRef[Cols],
    do_ref: pl.LayerNormMatrixRef[Rows, Cols],
    mean_ref: pl.LayerNormVectorRef[Rows],
    rstd_ref: pl.LayerNormVectorRef[Rows],
    # Outputs
    dw_ref: pl.LayerNormOutRef[Cols],
    db_ref: pl.LayerNormOutRef[Cols],
    *,
    eps: float,
    block_m: Int[RowBlock],
    block_n: Int[ColBlock],
):
    m, n_col = x_ref.shape
    j = pl.program_id(0)
    col_idx = j * block_n + jnp.arange(block_n)
    col_mask = col_idx < n_col

    def body(
        i: int,
        acc: tuple[pl.Tile[[ColBlock]], pl.Tile[[ColBlock]]],
    ):
        row_idx = i * block_m + jnp.arange(block_m)
        row_mask = row_idx < m
        mask = row_mask[:, None] & col_mask[None, :]
        a = plgpu.load(
            x_ref.at[row_idx[:, None], col_idx[None]], mask=mask, other=0.0
        ).astype(jnp.float32)
        dout = plgpu.load(
            do_ref.at[row_idx[:, None], col_idx[None]], mask=mask, other=0.0
        ).astype(jnp.float32)
        mean = plgpu.load(mean_ref.at[row_idx], mask=row_mask, other=0.0).astype(
            jnp.float32
        )
        rstd = plgpu.load(rstd_ref.at[row_idx], mask=row_mask, other=0.0).astype(
            jnp.float32
        )
        a_hat = (a - mean[:, None]) * rstd[:, None]
        dw_acc_ref, db_acc_ref = acc
        return dw_acc_ref + (dout * a_hat).sum(axis=0), db_acc_ref + dout.sum(axis=0)

    dw_acc, db_acc = lax.fori_loop(
        0,
        pl.cdiv(m, block_m),
        body,
        init_val=(jnp.zeros(block_n), jnp.zeros(block_n)),
    )
    plgpu.store(dw_ref.at[col_idx], dw_acc.astype(dw_ref.dtype), mask=col_mask)
    plgpu.store(db_ref.at[col_idx], db_acc.astype(db_ref.dtype), mask=col_mask)


def layer_norm_weight_grad[M: IntVar, N: IntVar, BM: IntVar, BN: IntVar](
    x: jax.Array[[M, N]],
    weight: jax.Array[[N]],
    bias: jax.Array[[N]],
    dout: jax.Array[[M, N]],
    mean: jax.Array[[M]],
    rstd: jax.Array[[M]],
    *,
    block_m: Int[BM],
    block_n: Int[BN],
) -> tuple[jax.Array[[N]], jax.Array[[N]]]:
    """Bind six host inputs to full Refs and both feature-gradient outputs."""

    def kernel(
        x_ref: pl.LayerNormMatrixRef[M, N],
        weight_ref: pl.LayerNormVectorRef[N],
        bias_ref: pl.LayerNormVectorRef[N],
        do_ref: pl.LayerNormMatrixRef[M, N],
        mean_ref: pl.LayerNormVectorRef[M],
        rstd_ref: pl.LayerNormVectorRef[M],
        dw_ref: pl.LayerNormOutRef[N],
        db_ref: pl.LayerNormOutRef[N],
    ) -> None:
        layer_norm_backward_kernel_dw_db(
            x_ref,
            weight_ref,
            bias_ref,
            do_ref,
            mean_ref,
            rstd_ref,
            dw_ref,
            db_ref,
            eps=1e-5,
            block_m=block_m,
            block_n=block_n,
        )

    layout = layer_norm_weight_gradient_layout(
        kernel,
        input_shape=(x.shape[0], x.shape[1]),
        out_shape=(
            jax.ShapeDtypeStruct(weight.shape, weight.dtype),
            jax.ShapeDtypeStruct(bias.shape, bias.dtype),
        ),
        block_m=block_m,
        block_n=block_n,
    )
    return checked_pallas_call(layout, interpret=True)(
        x, weight, bias, dout, mean, rstd
    )


class LayerNormWeightGradTest(unittest.TestCase):
    """Check masked column tiles and row reductions against independent JAX math."""

    def test_gradient(self) -> None:
        x = cast(Any, jnp).arange(21, dtype=jnp.float32).reshape(3, 7) / 7
        dout = cast(Any, jnp).linspace(-0.5, 1.5, 21).reshape(3, 7)
        weight = cast(Any, jnp).ones(7, dtype=jnp.float32)
        bias = cast(Any, jnp).zeros(7, dtype=jnp.float32)
        mean = x.mean(axis=1)
        rstd = cast(Any, jnp).reciprocal(
            cast(Any, jnp).sqrt(((x - mean[:, None]) ** 2).mean(axis=1) + 1e-5)
        )
        dw, db = layer_norm_weight_grad(
            x, weight, bias, dout, mean, rstd, block_m=2, block_n=4
        )
        expected_dw = (dout * (x - mean[:, None]) * rstd[:, None]).sum(axis=0)
        self.assertEqual(dw.shape, (7,))
        self.assertTrue(bool(cast(Any, jnp).allclose(dw, expected_dw, atol=1e-5)))
        self.assertTrue(bool(cast(Any, jnp).allclose(db, dout.sum(axis=0), atol=1e-5)))

    def test_reject_mismatched_input(self) -> None:
        x = cast(Any, jnp).ones((3, 7), dtype=jnp.float32)
        columns = cast(Any, jnp).ones(7, dtype=jnp.float32)
        rows = cast(Any, jnp).ones(3, dtype=jnp.float32)
        with self.assertRaisesRegex(ValueError, "shapes must match"):
            layer_norm_weight_grad(
                x, columns, columns, x, columns, rows, block_m=2, block_n=4
            )
        with self.assertRaisesRegex(ValueError, "column shape"):
            layer_norm_weight_grad(
                x, rows, columns, x, rows, rows, block_m=2, block_n=4
            )
        with self.assertRaisesRegex(ValueError, "dtypes must match"):
            layer_norm_weight_grad(
                x,
                columns,
                columns,
                cast(Any, jnp).ones((3, 7), dtype=jnp.float16),
                rows,
                rows,
                block_m=2,
                block_n=4,
            )


if TYPE_CHECKING:

    def typed_boundary[M: IntVar, N: IntVar, Other: IntVar](
        x: jax.Array[[M, N]],
        columns: jax.Array[[N]],
        rows: jax.Array[[M]],
        wrong: jax.Array[[Other]],
    ) -> None:
        assert_type(
            layer_norm_weight_grad(
                x, columns, columns, x, rows, rows, block_m=2, block_n=4
            ),
            tuple[jax.Array[[N]], jax.Array[[N]]],
        )
        layer_norm_weight_grad(
            x,
            columns,
            wrong,  # pyrefly: ignore[bad-argument-type]
            x,
            rows,
            rows,
            block_m=2,
            block_n=4,
        )

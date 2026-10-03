# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The entire kernel comes from JAX
# jax/experimental/pallas/ops/gpu/layer_norm.py;
# only the kernel and nested callback's semantic parameter types are added.
# @lint-ignore-every AUTODEPS2

"""Layer norm reduces batch rows into two masked feature gradients."""

from __future__ import annotations

from collections.abc import Callable
from typing import assert_type, TYPE_CHECKING

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar

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


def layer_norm_weight_grad[
    M: IntVar,
    N: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    x: jax.Array[[M, N]],
    weight: jax.Array[[N]],
    bias: jax.Array[[N]],
    dout: jax.Array[[M, N]],
    mean: jax.Array[[M]],
    rstd: jax.Array[[M]],
    block_m: Int[BM],
    block_n: Int[BN],
) -> tuple[jax.Array[[N]], jax.Array[[N]]]:
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

    return pl.pallas_call(
        kernel,
        grid=(pl.cdiv(x.shape[1], block_n),),
        out_shape=(
            jax.ShapeDtypeStruct(weight.shape, weight.dtype),
            jax.ShapeDtypeStruct(bias.shape, bias.dtype),
        ),
        interpret=True,
    )(x, weight, bias, dout, mean, rstd)


def test_correct_host_shapes[M: IntVar, N: IntVar, BM: IntVar, BN: IntVar](
    x: jax.Array[[M, N]],
    weight: jax.Array[[N]],
    bias: jax.Array[[N]],
    dout: jax.Array[[M, N]],
    mean: jax.Array[[M]],
    rstd: jax.Array[[M]],
    block_m: Int[BM],
    block_n: Int[BN],
) -> None:
    assert_type(
        layer_norm_weight_grad(x, weight, bias, dout, mean, rstd, block_m, block_n),
        tuple[jax.Array[[N]], jax.Array[[N]]],
    )


def test_wrong_host_shapes[M: IntVar, N: IntVar, Other: IntVar, BM: IntVar, BN: IntVar](
    x: jax.Array[[M, N]],
    weight: jax.Array[[N]],
    bias: jax.Array[[Other]],
    dout: jax.Array[[M, Other]],
    mean: jax.Array[[Other]],
    rstd: jax.Array[[M]],
    block_m: Int[BM],
    block_n: Int[BN],
) -> None:
    layer_norm_weight_grad(
        x,
        weight,
        bias,  # E: is not assignable
        dout,  # E: is not assignable
        mean,  # E: is not assignable
        rstd,
        block_m,
        block_n,
    )


def test_wrong_dw_store[N: IntVar, Block: IntVar, Other: IntVar](
    dw: pl.LayerNormOutRef[N],
    idx: pl.Indices[Block],
    bad_value: pl.Tile[[Other]],
    mask: pl.Mask[Block, N],
) -> None:
    plgpu.store(dw.at[idx], bad_value, mask=mask)  # E: No matching overload


def test_wrong_db_store[N: IntVar, Block: IntVar, Other: IntVar](
    db: pl.LayerNormOutRef[Other],
    idx: pl.Indices[Block],
    value: pl.Tile[[Block]],
    mask: pl.Mask[Block, N],
) -> None:
    plgpu.store(db.at[idx], value, mask=mask)  # E: No matching overload


def test_wrong_reduction_axis[N: IntVar, BM: IntVar, BN: IntVar](
    dw: pl.LayerNormOutRef[N],
    idx: pl.Indices[BN],
    matrix: pl.Tile[[BM, BN]],
    mask: pl.Mask[BN, N],
) -> None:
    plgpu.store(dw.at[idx], matrix.sum(axis=-1), mask=mask)  # E: No matching overload


def test_wrong_row_extent[
    M: IntVar,
    N: IntVar,
    Other: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    x: pl.LayerNormMatrixRef[M, N],
    rows: pl.Indices[BM],
    cols: pl.Indices[BN],
    row_mask: pl.Mask[BM, Other],
    col_mask: pl.Mask[BN, N],
) -> None:
    plgpu.load(  # E: No matching overload
        x.at[rows[:, None], cols[None]],
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    )


def test_wrong_column_extent[
    M: IntVar,
    N: IntVar,
    Other: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    x: pl.LayerNormMatrixRef[M, N],
    rows: pl.Indices[BM],
    cols: pl.Indices[BN],
    row_mask: pl.Mask[BM, M],
    col_mask: pl.Mask[BN, Other],
) -> None:
    plgpu.load(  # E: No matching overload
        x.at[rows[:, None], cols[None]],
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    )


def test_wrong_output_allocations[
    M: IntVar,
    N: IntVar,
    Other: IntVar,
    BN: IntVar,
](
    kernel: Callable[
        [
            pl.LayerNormMatrixRef[M, N],
            pl.LayerNormVectorRef[N],
            pl.LayerNormVectorRef[N],
            pl.LayerNormMatrixRef[M, N],
            pl.LayerNormVectorRef[M],
            pl.LayerNormVectorRef[M],
            pl.LayerNormOutRef[N],
            pl.LayerNormOutRef[N],
        ],
        None,
    ],
    x: jax.Array[[M, N]],
    weight: jax.Array[[N]],
    bias: jax.Array[[N]],
    dout: jax.Array[[M, N]],
    mean: jax.Array[[M]],
    rstd: jax.Array[[M]],
    wrong: jax.Array[[Other]],
    block_n: Int[BN],
) -> None:
    pl.pallas_call(  # E: No matching overload
        kernel,
        grid=(pl.cdiv(x.shape[1], block_n),),
        out_shape=(
            jax.ShapeDtypeStruct(wrong.shape, wrong.dtype),
            jax.ShapeDtypeStruct(bias.shape, bias.dtype),
        ),
        interpret=True,
    )(x, weight, bias, dout, mean, rstd)
    pl.pallas_call(  # E: No matching overload
        kernel,
        grid=(pl.cdiv(x.shape[1], block_n),),
        out_shape=(
            jax.ShapeDtypeStruct(weight.shape, weight.dtype),
            jax.ShapeDtypeStruct(wrong.shape, wrong.dtype),
        ),
        interpret=True,
    )(x, weight, bias, dout, mean, rstd)


if not TYPE_CHECKING:
    import unittest

    import numpy as np

    class LayerNormWeightGradBoundaryTest(unittest.TestCase):
        def test_partial_row_and_column_tiles(self) -> None:
            x = jnp.arange(15, dtype=jnp.float32).reshape((3, 5)) / 3
            dout = jnp.arange(15, dtype=jnp.float32).reshape((3, 5)) / 7
            weight = jnp.ones((5,), dtype=jnp.float32)
            bias = jnp.zeros((5,), dtype=jnp.float32)
            mean = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
            rstd = jnp.array([0.5, 1.0, 2.0], dtype=jnp.float32)
            dw, db = layer_norm_weight_grad(
                x, weight, bias, dout, mean, rstd, block_m=2, block_n=4
            )
            self.assertEqual(dw.shape, (5,))
            self.assertEqual(db.shape, (5,))
            normalized = (np.asarray(x) - np.asarray(mean)[:, None]) * np.asarray(rstd)[
                :, None
            ]
            np.testing.assert_allclose(
                np.asarray(dw), np.sum(np.asarray(dout) * normalized, axis=0), rtol=1e-6
            )
            np.testing.assert_allclose(
                np.asarray(db), np.sum(np.asarray(dout), axis=0), rtol=1e-6
            )

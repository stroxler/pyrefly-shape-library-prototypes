# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The full forward kernel is from JAX
# jax/experimental/pallas/ops/gpu/rms_norm.py (Apache-2.0); only parameter
# annotations are new. The upstream GPU module is deprecated.
# @lint-ignore-every AUTODEPS2

"""RMSNorm checks three feature axes, masked tiles and a scalar result."""

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar

    Length = IntVar("Length")
    Block = IntVar("Block")


def rms_norm_forward_kernel(
    x_ref: pl.InRef[[Length]],
    weight_ref: pl.InRef[[Length]],
    bias_ref: pl.InRef[[Length]],
    o_ref: pl.OutRef[[Length]],
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


def rms_norm_host[Batch: IntVar, Rows: IntVar, Features: IntVar, Tile: IntVar](
    x: jax.Array[[Batch, Rows, Features]],
    weight: jax.Array[[Features]],
    bias: jax.Array[[Features]],
    block_size: Int[Tile],
    eps: float = 1e-5,
) -> tuple[jax.Array[[Batch, Rows, Features]], jax.Array[[Batch, Rows]]]:
    def row_kernel(
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

    method = pl.pallas_call(
        row_kernel,
        grid=(),
        out_shape=(
            jax.ShapeDtypeStruct((x.shape[2],), x.dtype),
            jax.ShapeDtypeStruct((), x.dtype),
        ),
        interpret=True,
    )
    mapped_norm = jax.vmap(
        jax.vmap(method, in_axes=(0, None, None)), in_axes=(0, None, None)
    )
    return mapped_norm(x, weight, bias)


def test_host_shapes[Batch: IntVar, Rows: IntVar, Features: IntVar, Tile: IntVar](
    x: jax.Array[[Batch, Rows, Features]],
    weight: jax.Array[[Features]],
    bias: jax.Array[[Features]],
    block_size: Int[Tile],
) -> None:
    assert_type(
        rms_norm_host(x, weight, bias, block_size),
        tuple[jax.Array[[Batch, Rows, Features]], jax.Array[[Batch, Rows]]],
    )


def test_wrong_weight_width[
    Batch: IntVar,
    Rows: IntVar,
    Features: IntVar,
    Other: IntVar,
    Tile: IntVar,
](
    x: jax.Array[[Batch, Rows, Features]],
    weight: jax.Array[[Other]],
    bias: jax.Array[[Features]],
    block_size: Int[Tile],
) -> None:
    rms_norm_host(x, weight, bias, block_size)  # E: is not assignable


def test_wrong_bias_width[
    Batch: IntVar,
    Rows: IntVar,
    Features: IntVar,
    Other: IntVar,
    Tile: IntVar,
](
    x: jax.Array[[Batch, Rows, Features]],
    weight: jax.Array[[Features]],
    bias: jax.Array[[Other]],
    block_size: Int[Tile],
) -> None:
    rms_norm_host(x, weight, bias, block_size)  # E: is not assignable


def test_wrong_output_ref[Features: IntVar, Other: IntVar, Tile: IntVar](
    x: pl.InRef[[Features]],
    weight: pl.InRef[[Features]],
    bias: pl.InRef[[Features]],
    out: pl.OutRef[[Other]],
    rstd: pl.OutRef[[]],
    block_size: Int[Tile],
) -> None:
    rms_norm_forward_kernel(
        x,
        weight,
        bias,
        out,  # E: is not assignable
        rstd,
        eps=1e-5,
        block_size=block_size,
    )


def test_wrong_scalar_ref[Features: IntVar, Other: IntVar, Tile: IntVar](
    x: pl.InRef[[Features]],
    weight: pl.InRef[[Features]],
    bias: pl.InRef[[Features]],
    out: pl.OutRef[[Features]],
    rstd: pl.OutRef[[Other]],
    block_size: Int[Tile],
) -> None:
    rms_norm_forward_kernel(
        x,
        weight,
        bias,
        out,
        rstd,  # E: is not assignable
        eps=1e-5,
        block_size=block_size,
    )


def test_wrong_mask_source[Features: IntVar, Other: IntVar, Tile: IntVar](
    x: pl.InRef[[Features]],
    other: pl.InRef[[Other]],
    block: Int[Tile],
) -> None:
    col_idx = jnp.arange(block)
    mask = col_idx < other.shape[0]
    plgpu.load(x.at[col_idx], mask=mask)  # E: is not assignable


if not TYPE_CHECKING:
    import unittest

    class RmsNormBoundaryTest(unittest.TestCase):
        def test_cpu_reference_shares_feature_axis(self) -> None:
            x = jnp.arange(2 * 3 * 5, dtype=jnp.float32).reshape(2, 3, 5) / 5
            weight = jnp.arange(5, dtype=jnp.float32) + 1
            bias = jnp.arange(5, dtype=jnp.float32) / 10
            rstd = jax.lax.rsqrt(jnp.mean(x * x, axis=-1) + 1e-5)
            out = x * rstd[..., None] * weight + bias
            self.assertEqual(out.shape, (2, 3, 5))
            self.assertEqual(rstd.shape, (2, 3))

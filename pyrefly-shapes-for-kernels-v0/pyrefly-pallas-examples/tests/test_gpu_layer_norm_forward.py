# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The complete kernel comes from JAX
# jax/experimental/pallas/ops/gpu/layer_norm.py (Apache-2.0).
# Only parameter annotations are added; the GPU module is deprecated upstream.
# @lint-ignore-every AUTODEPS2

"""A full-row Pallas boundary for masked layer norm and scalar statistics."""

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
    block_size: Int[Block],
    eps: float = 1e-5,
) -> tuple[jax.Array[[Features]], jax.Array[[]], jax.Array[[]]]:
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

    return pl.pallas_call(
        kernel,
        grid=(),
        out_shape=(
            jax.ShapeDtypeStruct((x.shape[0],), x.dtype),
            jax.ShapeDtypeStruct((), x.dtype),
            jax.ShapeDtypeStruct((), x.dtype),
        ),
        interpret=True,
    )(x, weight, bias)


def test_matching_host_axes[Features: IntVar, Block: IntVar](
    x: jax.Array[[Features]],
    weight: jax.Array[[Features]],
    bias: jax.Array[[Features]],
    block_size: Int[Block],
) -> None:
    assert_type(
        layer_norm_row(x, weight, bias, block_size),
        tuple[jax.Array[[Features]], jax.Array[[]], jax.Array[[]]],
    )


def test_wrong_weight_and_bias[Features: IntVar, Other: IntVar, Block: IntVar](
    x: jax.Array[[Features]],
    weight: jax.Array[[Other]],
    bias: jax.Array[[Other]],
    block_size: Int[Block],
) -> None:
    layer_norm_row(
        x,
        weight,  # E: is not assignable
        bias,  # E: is not assignable
        block_size,
    )


def test_wrong_output_and_statistics_refs[
    Features: IntVar,
    Other: IntVar,
    Block: IntVar,
](
    x: pl.InRef[[Features]],
    weight: pl.InRef[[Features]],
    bias: pl.InRef[[Features]],
    out: pl.OutRef[[Other]],
    mean: pl.OutRef[[Other]],
    rstd: pl.OutRef[[Other]],
    block_size: Int[Block],
) -> None:
    layer_norm_forward_kernel(
        x,
        weight,
        bias,
        out,  # E: is not assignable
        mean,  # E: is not assignable
        rstd,  # E: is not assignable
        eps=1e-5,
        block_size=block_size,
    )


def test_wrong_input_mask[Features: IntVar, Other: IntVar, Block: IntVar](
    x: pl.InRef[[Features]], other: pl.InRef[[Other]], block: Int[Block]
) -> None:
    col_idx = jnp.arange(block)
    mask = col_idx < other.shape[0]
    plgpu.load(x.at[col_idx], mask=mask, other=0.0)  # E: No matching overload


def test_wrong_output_mask[Features: IntVar, Other: IntVar, Block: IntVar](
    out: pl.OutRef[[Features]], other: pl.InRef[[Other]], block: Int[Block]
) -> None:
    col_idx = jnp.arange(block)
    mask = col_idx < other.shape[0]
    plgpu.store(  # E: No matching overload
        out.at[col_idx], jnp.zeros(block), mask=mask
    )


def test_wrong_output_tile[Features: IntVar, Block: IntVar, Other: IntVar](
    out: pl.OutRef[[Features]],
    idx: pl.Indices[Block],
    value: pl.Tile[[Other]],
    mask: pl.Mask[Block, Features],
) -> None:
    plgpu.store(out.at[idx], value, mask=mask)  # E: No matching overload


def test_wrong_output_allocation[Features: IntVar, Other: IntVar](
    kernel: Callable[
        [
            pl.InRef[[Features]],
            pl.InRef[[Features]],
            pl.InRef[[Features]],
            pl.OutRef[[Features]],
            pl.OutRef[[]],
            pl.OutRef[[]],
        ],
        None,
    ],
    x: jax.Array[[Features]],
    weight: jax.Array[[Features]],
    bias: jax.Array[[Features]],
    wrong: jax.Array[[Other]],
) -> None:
    pl.pallas_call(  # E: No matching overload
        kernel,
        grid=(),
        out_shape=(
            jax.ShapeDtypeStruct(wrong.shape, wrong.dtype),
            jax.ShapeDtypeStruct((), x.dtype),
            jax.ShapeDtypeStruct((), x.dtype),
        ),
        interpret=True,
    )(x, weight, bias)

    pl.pallas_call(  # E: No matching overload
        kernel,
        grid=(),
        out_shape=(
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            jax.ShapeDtypeStruct(wrong.shape, wrong.dtype),
            jax.ShapeDtypeStruct((), x.dtype),
        ),
        interpret=True,
    )(x, weight, bias)

    pl.pallas_call(  # E: No matching overload
        kernel,
        grid=(),
        out_shape=(
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            jax.ShapeDtypeStruct((), x.dtype),
            jax.ShapeDtypeStruct(wrong.shape, wrong.dtype),
        ),
        interpret=True,
    )(x, weight, bias)


if not TYPE_CHECKING:
    import unittest

    import numpy as np

    class LayerNormForwardBoundaryTest(unittest.TestCase):
        def test_two_masked_blocks_and_scalar_outputs(self) -> None:
            x = jnp.array([1.0, -2.0, 3.0, 4.0, -1.0], dtype=jnp.float32)
            weight = jnp.array([0.5, 2.0, 1.0, -1.0, 3.0], dtype=jnp.float32)
            bias = jnp.array([1.0, -1.0, 0.0, 2.0, 0.5], dtype=jnp.float32)
            out, mean, rstd = layer_norm_row(x, weight, bias, block_size=4)
            expected_mean = np.mean(np.asarray(x))
            expected_rstd = 1 / np.sqrt(
                np.mean((np.asarray(x) - expected_mean) ** 2) + 1e-5
            )
            expected = (np.asarray(x) - expected_mean) * expected_rstd * np.asarray(
                weight
            ) + np.asarray(bias)
            self.assertEqual(out.shape, (5,))
            self.assertEqual(mean.shape, ())
            self.assertEqual(rstd.shape, ())
            np.testing.assert_allclose(np.asarray(mean), expected_mean, rtol=1e-6)
            np.testing.assert_allclose(np.asarray(rstd), expected_rstd, rtol=1e-6)
            np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-6)

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The complete kernel comes from JAX
# jax/experimental/pallas/ops/gpu/layer_norm.py (Apache-2.0).
# Only parameter annotations are added; the GPU module is deprecated upstream.
# @lint-ignore-every AUTODEPS2

"""A full-row input-gradient boundary for Pallas layer normalization."""

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
    block_size: Int[Block],
) -> jax.Array[[Features]]:
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

    return pl.pallas_call(
        kernel,
        grid=(),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=True,
    )(x, weight, bias, dout, mean, rstd)


def test_matching_host_axes[Features: IntVar, Block: IntVar](
    x: jax.Array[[Features]],
    weight: jax.Array[[Features]],
    bias: jax.Array[[Features]],
    dout: jax.Array[[Features]],
    mean: jax.Array[[]],
    rstd: jax.Array[[]],
    block_size: Int[Block],
) -> None:
    assert_type(
        layer_norm_input_grad(x, weight, bias, dout, mean, rstd, block_size),
        jax.Array[[Features]],
    )


def test_wrong_host_axes[Features: IntVar, Other: IntVar, Block: IntVar](
    x: jax.Array[[Features]],
    weight: jax.Array[[Other]],
    bias: jax.Array[[Other]],
    dout: jax.Array[[Other]],
    mean: jax.Array[[]],
    rstd: jax.Array[[Other]],
    block_size: Int[Block],
) -> None:
    layer_norm_input_grad(
        x,
        weight,  # E: is not assignable
        bias,  # E: is not assignable
        dout,  # E: is not assignable
        mean,
        rstd,  # E: is not assignable
        block_size,
    )


def test_wrong_scalar_refs[Features: IntVar, Other: IntVar, Block: IntVar](
    x: pl.InRef[[Features]],
    weight: pl.InRef[[Features]],
    bias: pl.InRef[[Features]],
    dout: pl.InRef[[Features]],
    mean: pl.InRef[[Other]],
    rstd: pl.InRef[[Other]],
    dx: pl.OutRef[[Features]],
    block_size: Int[Block],
) -> None:
    layer_norm_backward_kernel_dx(
        x,
        weight,
        bias,
        dout,
        mean,  # E: is not assignable
        rstd,  # E: is not assignable
        dx,
        eps=1e-5,
        block_size=block_size,
    )


def test_scalar_ref_read(mean: pl.InRef[[]]) -> None:
    assert_type(mean[...], pl.ScalarFloat)


def test_non_scalar_ref_read[Other: IntVar](mean: pl.InRef[[Other]]) -> None:
    value: pl.ScalarFloat = mean[...]  # E: is not assignable
    assert_type(value, pl.ScalarFloat)


def test_wrong_dout_mask[Features: IntVar, Other: IntVar, Block: IntVar](
    dout: pl.InRef[[Features]], other: pl.InRef[[Other]], block: Int[Block]
) -> None:
    idx = jnp.arange(block)
    mask = idx < other.shape[0]
    plgpu.load(dout.at[idx], mask=mask, other=0.0)  # E: No matching overload


def test_wrong_dx_store[Features: IntVar, Other: IntVar, Block: IntVar](
    dx: pl.OutRef[[Features]],
    idx: pl.Indices[Block],
    mask: pl.Mask[Block, Features],
    value: pl.Tile[[Other]],
) -> None:
    plgpu.store(dx.at[idx], value, mask=mask)  # E: No matching overload


def test_wrong_output_allocation[Features: IntVar, Other: IntVar](
    kernel: Callable[
        [
            pl.InRef[[Features]],
            pl.InRef[[Features]],
            pl.InRef[[Features]],
            pl.InRef[[Features]],
            pl.InRef[[]],
            pl.InRef[[]],
            pl.OutRef[[Features]],
        ],
        None,
    ],
    x: jax.Array[[Features]],
    weight: jax.Array[[Features]],
    bias: jax.Array[[Features]],
    dout: jax.Array[[Features]],
    mean: jax.Array[[]],
    rstd: jax.Array[[]],
    wrong: jax.Array[[Other]],
) -> None:
    pl.pallas_call(  # E: No matching overload
        kernel,
        grid=(),
        out_shape=jax.ShapeDtypeStruct(wrong.shape, wrong.dtype),
        interpret=True,
    )(x, weight, bias, dout, mean, rstd)


if not TYPE_CHECKING:
    import unittest

    import numpy as np

    class LayerNormInputGradBoundaryTest(unittest.TestCase):
        def test_partial_feature_block(self) -> None:
            x = jnp.array([1.0, -2.0, 3.0, 4.0, -1.0], dtype=jnp.float32)
            weight = jnp.array([0.5, 2.0, 1.0, -1.0, 3.0], dtype=jnp.float32)
            bias = jnp.array([1.0, -1.0, 0.0, 2.0, 0.5], dtype=jnp.float32)
            dout = jnp.array([1.0, -0.4, 2.0, -1.0, 0.3], dtype=jnp.float32)
            mean = jnp.mean(x)
            rstd = 1 / jnp.sqrt(jnp.mean((x - mean) ** 2) + 1e-5)
            actual = layer_norm_input_grad(
                x, weight, bias, dout, mean, rstd, block_size=4
            )
            normalized = (np.asarray(x) - np.asarray(mean)) * np.asarray(rstd)
            wdout = np.asarray(weight) * np.asarray(dout)
            expected = (
                wdout - normalized * np.mean(normalized * wdout) - np.mean(wdout)
            ) * np.asarray(rstd)
            self.assertEqual(actual.shape, (5,))
            np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6)

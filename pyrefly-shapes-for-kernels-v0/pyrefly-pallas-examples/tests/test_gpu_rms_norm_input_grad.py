# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The complete kernel comes from JAX
# jax/experimental/pallas/ops/gpu/rms_norm.py (Apache-2.0).
# Only semantic parameter annotations are added; the GPU module is deprecated.
# @lint-ignore-every AUTODEPS2

"""The masked RMSNorm input gradient uses a scalar row statistic."""

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


def rms_norm_backward_kernel_dx(
    # Inputs
    x_ref: pl.InRef[[Features]],
    weight_ref: pl.InRef[[Features]],
    bias_ref: pl.InRef[[Features]],
    do_ref: pl.InRef[[Features]],
    rstd_ref: pl.InRef[[]],
    # Outputs
    dx_ref: pl.OutRef[[Features]],
    *,
    eps: float,
    block_size: Int[Block],
):
    n_col = x_ref.shape[0]

    def mean_body(i: int, c1_acc: pl.Tile[[Block]]):
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
        a_hat = a * rstd_ref[...]
        wdout = weight * dout
        return c1_acc + a_hat * wdout

    c1 = lax.fori_loop(0, pl.cdiv(n_col, block_size), mean_body, jnp.zeros(block_size))
    c1 = c1.sum() / n_col

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
        a_hat = a * rstd_ref[...]
        wdout = weight * dout
        da = (wdout - (a_hat * c1)) * rstd_ref[...]
        plgpu.store(dx_ref.at[col_idx], da.astype(dx_ref.dtype), mask=mask)


def rms_norm_input_grad[Features: IntVar, Block: IntVar](
    x: jax.Array[[Features]],
    weight: jax.Array[[Features]],
    bias: jax.Array[[Features]],
    dout: jax.Array[[Features]],
    rstd: jax.Array[[]],
    block_size: Int[Block],
) -> jax.Array[[Features]]:
    def kernel(
        x_ref: pl.InRef[[Features]],
        weight_ref: pl.InRef[[Features]],
        bias_ref: pl.InRef[[Features]],
        do_ref: pl.InRef[[Features]],
        rstd_ref: pl.InRef[[]],
        dx_ref: pl.OutRef[[Features]],
    ) -> None:
        rms_norm_backward_kernel_dx(
            x_ref,
            weight_ref,
            bias_ref,
            do_ref,
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
    )(x, weight, bias, dout, rstd)


def test_matching_host_axes[Features: IntVar, Block: IntVar](
    x: jax.Array[[Features]],
    weight: jax.Array[[Features]],
    bias: jax.Array[[Features]],
    dout: jax.Array[[Features]],
    rstd: jax.Array[[]],
    block_size: Int[Block],
) -> None:
    assert_type(
        rms_norm_input_grad(x, weight, bias, dout, rstd, block_size),
        jax.Array[[Features]],
    )


def test_wrong_host_axes[Features: IntVar, Other: IntVar, Block: IntVar](
    x: jax.Array[[Features]],
    weight: jax.Array[[Other]],
    bias: jax.Array[[Other]],
    dout: jax.Array[[Other]],
    rstd: jax.Array[[Other]],
    block_size: Int[Block],
) -> None:
    rms_norm_input_grad(
        x,
        weight,  # E: is not assignable
        bias,  # E: is not assignable
        dout,  # E: is not assignable
        rstd,  # E: is not assignable
        block_size,
    )


def test_wrong_scalar_ref[Features: IntVar, Other: IntVar, Block: IntVar](
    x: pl.InRef[[Features]],
    weight: pl.InRef[[Features]],
    bias: pl.InRef[[Features]],
    dout: pl.InRef[[Features]],
    rstd: pl.InRef[[Other]],
    dx: pl.OutRef[[Features]],
    block_size: Int[Block],
) -> None:
    rms_norm_backward_kernel_dx(
        x,
        weight,
        bias,
        dout,
        rstd,  # E: is not assignable
        dx,
        eps=1e-5,
        block_size=block_size,
    )


def test_non_scalar_rstd_read[Other: IntVar](rstd: pl.InRef[[Other]]) -> None:
    value: pl.ScalarFloat = rstd[...]  # E: is not assignable
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
            pl.OutRef[[Features]],
        ],
        None,
    ],
    x: jax.Array[[Features]],
    weight: jax.Array[[Features]],
    bias: jax.Array[[Features]],
    dout: jax.Array[[Features]],
    rstd: jax.Array[[]],
    wrong: jax.Array[[Other]],
) -> None:
    pl.pallas_call(  # E: No matching overload
        kernel,
        grid=(),
        out_shape=jax.ShapeDtypeStruct(wrong.shape, wrong.dtype),
        interpret=True,
    )(x, weight, bias, dout, rstd)


if not TYPE_CHECKING:
    import unittest

    import numpy as np

    class RmsNormInputGradBoundaryTest(unittest.TestCase):
        def test_partial_feature_block_matches_reference_and_autodiff(self) -> None:
            x = jnp.array([1.0, -2.0, 3.0, 4.0, -1.0], dtype=jnp.float32)
            weight = jnp.array([0.5, 2.0, 1.0, -1.0, 3.0], dtype=jnp.float32)
            bias = jnp.array([1.0, -1.0, 0.0, 2.0, 0.5], dtype=jnp.float32)
            dout = jnp.array([1.0, -0.4, 2.0, -1.0, 0.3], dtype=jnp.float32)
            x_numpy = np.asarray(x)
            rstd = jnp.asarray(
                1 / np.sqrt(np.mean(x_numpy**2) + 1e-5), dtype=jnp.float32
            )

            actual = rms_norm_input_grad(x, weight, bias, dout, rstd, block_size=4)

            normalized = x_numpy * np.asarray(rstd)
            weighted_dout = np.asarray(weight) * np.asarray(dout)
            expected = (
                weighted_dout - normalized * np.mean(normalized * weighted_dout)
            ) * np.asarray(rstd)
            autodiff = jax.grad(
                lambda row: jnp.sum(
                    (row / jnp.sqrt(jnp.mean(row * row) + 1e-5) * weight + bias) * dout
                )
            )(x)

            self.assertEqual(actual.shape, (5,))
            np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6)
            np.testing.assert_allclose(
                np.asarray(actual), np.asarray(autodiff), rtol=1e-6
            )

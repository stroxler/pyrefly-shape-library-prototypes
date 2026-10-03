# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Kernel body from JAX jax/experimental/pallas/ops/gpu/softmax.py (Apache-2.0).
# Semantic annotations exist only in the stub overlay.
# @lint-ignore-every AUTODEPS2

"""Pallas softmax masks padded column loads and stores."""

from __future__ import annotations

import unittest
from collections.abc import Callable
from typing import assert_type, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar


def _vmappable_softmax_kernel[Length: IntVar, Block: IntVar](
    input_ref: pl.InRef[[Length]],
    probs_ref: pl.OutRef[[Length]],
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


def masked_softmax[Length: IntVar, Block: IntVar](
    x: jax.Array[[Length]],
    length: Int[Length],
    block_row: Int[Block],
    dtype: object,
) -> jax.Array[[Length]]:
    def kernel(x_ref: pl.InRef[[Length]], out_ref: pl.OutRef[[Length]]) -> None:
        _vmappable_softmax_kernel(x_ref, out_ref, block_row=block_row)

    softmax = pl.pallas_call(
        kernel,
        grid=(),
        out_shape=jax.ShapeDtypeStruct((length,), dtype),
        interpret=True,
    )
    return softmax(x)


def test_wrong_host_length[Length: IntVar, Other: IntVar, Block: IntVar](
    x: jax.Array[[Length]],
    wrong_length: Int[Other],
    block: Int[Block],
) -> None:
    masked_softmax(x, wrong_length, block, None)  # E: is not assignable


def test_wrong_output_allocation[Length: IntVar, Other: IntVar](
    kernel: Callable[[pl.InRef[[Length]], pl.OutRef[[Length]]], None],
    other_length: Int[Other],
) -> None:
    pl.pallas_call(  # E: No matching overload
        kernel,
        grid=(),
        out_shape=jax.ShapeDtypeStruct((other_length,), None),
    )


def test_wrong_input_output_ref[Length: IntVar, Other: IntVar, Block: IntVar](
    x: pl.InRef[[Length]],
    out: pl.OutRef[[Other]],
    block: Int[Block],
) -> None:
    _vmappable_softmax_kernel(x, out, block_row=block)  # E: is not assignable


def test_wrong_load_mask_length[Length: IntVar, Other: IntVar, Block: IntVar](
    x: pl.InRef[[Length]],
    other: pl.InRef[[Other]],
    block: Int[Block],
) -> None:
    col_idx = jnp.arange(block)
    mask = col_idx < other.shape[-1]
    plgpu.load(x.at[col_idx], mask=mask, other=0.0)  # E: is not assignable


def test_wrong_store_mask_length[Length: IntVar, Other: IntVar, Block: IntVar](
    out: pl.OutRef[[Length]],
    other: pl.InRef[[Other]],
    block: Int[Block],
    values: pl.Tile[[Block]],
) -> None:
    col_idx = jnp.arange(block)
    mask = col_idx < other.shape[-1]
    plgpu.store(out.at[col_idx], values, mask=mask)  # E: is not assignable


def test_wrong_load_mask_block[Length: IntVar, Block: IntVar, Other: IntVar](
    x: pl.InRef[[Length]],
    block: Int[Block],
    wrong_block: Int[Other],
) -> None:
    col_idx = jnp.arange(block)
    mask = jnp.arange(wrong_block) < x.shape[-1]
    plgpu.load(x.at[col_idx], mask=mask, other=0.0)  # E: is not assignable


def test_unmasked_load[Length: IntVar, Block: IntVar](
    x: pl.InRef[[Length]], block: Int[Block]
) -> None:
    col_idx = jnp.arange(block)
    plgpu.load(x.at[col_idx], other=0.0)  # E: Missing argument `mask`


def test_wrong_store_tile_width[Length: IntVar, Block: IntVar, Other: IntVar](
    out: pl.OutRef[[Length]],
    block: Int[Block],
    wrong_values: pl.Tile[[Other]],
) -> None:
    col_idx = jnp.arange(block)
    mask = col_idx < out.shape[-1]
    plgpu.store(out.at[col_idx], wrong_values, mask=mask)  # E: is not assignable


def test_undersized_block_is_not_proven[Length: IntVar, Block: IntVar](
    x: jax.Array[[Length]], length: Int[Length], block: Int[Block]
) -> None:
    # A block shorter than Length omits output elements despite a valid mask.
    assert_type(masked_softmax(x, length, block, None), jax.Array[[Length]])


if not TYPE_CHECKING:

    class MaskedSoftmaxTest(unittest.TestCase):
        def test_cpu_interpreter(self) -> None:
            x = jnp.arange(5, dtype=jnp.float32)
            actual = masked_softmax(x, 5, 8, x.dtype)
            expected = jax.nn.softmax(x)
            self.assertTrue(jnp.allclose(actual, expected, rtol=1e-6, atol=1e-6))

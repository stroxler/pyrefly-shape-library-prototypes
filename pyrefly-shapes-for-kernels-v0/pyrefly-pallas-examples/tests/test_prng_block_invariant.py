# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The full kernel factory and nested body are from JAX docs/pallas/tpu/prng.rst
# (Apache-2.0); only their parameter annotations are new.
# @lint-ignore-every AUTODEPS2

"""One full PRNG output admits two distinct, shape-linked Pallas grids."""

from __future__ import annotations

from collections.abc import Callable
from typing import assert_type, Literal, TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.random
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

if TYPE_CHECKING:
    from shape_extensions import IntVar

    Rows = IntVar("Rows")
    Cols = IntVar("Cols")


def make_kernel_body(
    index_map: Callable[[int, int], tuple[int, int]],
) -> Callable[[pltpu.PallasKeyRef, pl.ShapeReadableOutRef[[Rows, Cols]]], None]:
    def body(
        key_ref: pltpu.PallasKeyRef,
        o_ref: pl.ShapeReadableOutRef[[Rows, Cols]],
    ) -> None:
        key = key_ref[...]
        samples = pltpu.sample_block(
            jax.random.uniform,
            key,
            block_size=o_ref[...].shape,
            tile_size=(16, 128),
            total_size=(64, 512),
            block_index=index_map(pl.program_id(0), pl.program_id(1)),
            minval=0.0,
            maxval=1.0,
        )
        o_ref[...] = samples

    return body


def invariant_samples(
    host_key: jax.random.Key[Literal["threefry2x32"]],
) -> tuple[jax.Array[[64, 512]], jax.Array[[64, 512]]]:
    global_key = pltpu.to_pallas_key(host_key)
    o_shape = jnp.ones((64, 512), dtype=jnp.float32)
    key_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
    out_spec = pl.BlockSpec((16, 128), lambda i, j: (i, j))
    result_16x128 = pl.pallas_call(
        make_kernel_body(index_map=lambda i, j: (i, j)),
        out_shape=o_shape,
        in_specs=[key_spec],
        out_specs=out_spec,
        grid=(4, 4),
    )(global_key)

    out_spec = pl.BlockSpec((32, 256), lambda i, j: (j, i))
    result_32x256_transposed = pl.pallas_call(
        make_kernel_body(index_map=lambda i, j: (j, i)),
        in_specs=[key_spec],
        out_shape=o_shape,
        out_specs=out_spec,
        grid=(2, 2),
    )(global_key)
    return result_16x128, result_32x256_transposed


def test_host_results() -> None:
    assert_type(
        invariant_samples(jax.random.key(7)),
        tuple[jax.Array[[64, 512]], jax.Array[[64, 512]]],
    )


def test_wrong_host_key(key: jax.random.Key[Literal["rbg"]]) -> None:
    invariant_samples(key)  # E: is not assignable


def test_wrong_output_grid(key: pltpu.PallasKey) -> None:
    pl.pallas_call(  # E: No matching overload
        make_kernel_body(index_map=lambda i, j: (i, j)),
        out_shape=jnp.ones((64, 512), dtype=jnp.float32),
        in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)],
        out_specs=pl.BlockSpec((16, 128), lambda i, j: (i, j)),
        grid=(2, 4),
    )(key)


def test_wrong_full_output(key: pltpu.PallasKey) -> None:
    pl.pallas_call(  # E: No matching overload
        make_kernel_body(index_map=lambda i, j: (i, j)),
        out_shape=jnp.ones((64, 256), dtype=jnp.float32),
        in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)],
        out_specs=pl.BlockSpec((16, 128), lambda i, j: (i, j)),
        grid=(4, 4),
    )(key)


def test_wrong_total_size(
    key: pltpu.PallasKey,
) -> None:
    pltpu.sample_block(
        jax.random.uniform,
        key,
        block_size=(16, 128),
        tile_size=(16, 128),
        total_size=(32, 512),  # E: is not assignable
        block_index=(0, 0),
        minval=0.0,
        maxval=1.0,
    )


def test_wrong_tile_size(key: pltpu.PallasKey) -> None:
    pltpu.sample_block(
        jax.random.uniform,
        key,
        block_size=(16, 128),
        tile_size=(32, 128),  # E: is not assignable
        total_size=(64, 512),
        block_index=(0, 0),
        minval=0.0,
        maxval=1.0,
    )


def test_wrong_block_output(
    key: pltpu.PallasKey, out: pl.ShapeReadableOutRef[[16, 128]]
) -> None:
    out[...] = pltpu.sample_block(  # E: Cannot set item
        jax.random.uniform,
        key,
        block_size=(32, 256),
        tile_size=(16, 128),
        total_size=(64, 512),
        block_index=(0, 0),
        minval=0.0,
        maxval=1.0,
    )


def test_duplicate_block_indices_are_not_rejected(key: pltpu.PallasKey) -> None:
    pl.pallas_call(
        make_kernel_body(index_map=lambda i, j: (0, 0)),
        out_shape=jnp.ones((64, 512), dtype=jnp.float32),
        in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)],
        out_specs=pl.BlockSpec((16, 128), lambda i, j: (0, 0)),
        grid=(4, 4),
    )(key)


if not TYPE_CHECKING:
    import unittest

    class BlockInvariantBoundaryTest(unittest.TestCase):
        def test_two_host_tiling_geometries(self) -> None:
            samples = jax.random.uniform(jax.random.key(7), shape=(64, 512))
            small = samples[3 * 16 : 4 * 16, 2 * 128 : 3 * 128]
            large = samples[1 * 32 : 2 * 32, 1 * 256 : 2 * 256]
            self.assertEqual(samples.shape, (64, 512))
            self.assertEqual(small.shape, (16, 128))
            self.assertEqual(large.shape, (32, 256))

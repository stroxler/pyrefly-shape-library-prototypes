# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The entire body is from JAX docs/pallas/tpu/prng.rst (Apache-2.0);
# only its parameter annotations are new.
# @lint-ignore-every AUTODEPS2

"""A converted host PRNG key is read in SMEM to generate a full tile."""

from __future__ import annotations

from typing import assert_type, Literal, TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.random
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def body(
    key_ref: pltpu.PallasKeyRef,
    o_ref: pl.ShapeReadableOutRef[[8, 128]],
) -> None:
    o_ref[...] = jax.random.uniform(key_ref[...], shape=o_ref[...].shape)


def stateless_uniform(
    host_key: jax.random.Key[Literal["threefry2x32"]],
) -> jax.Array[[8, 128]]:
    key = pltpu.to_pallas_key(host_key)
    o_shape = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    return pl.pallas_call(
        body,
        in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)],
        out_shape=o_shape,
    )(key)


def test_threefry_host_key() -> None:
    host_key = jax.random.key(7, impl="threefry2x32")
    assert_type(stateless_uniform(host_key), jax.Array[[8, 128]])


def test_wrong_host_key_kind(host_key: jax.random.Key[Literal["rbg"]]) -> None:
    stateless_uniform(host_key)  # E: is not assignable


def test_raw_key_cannot_enter_kernel(
    raw_key: jax.random.Key[Literal["threefry2x32"]],
) -> None:
    pl.pallas_call(
        body,
        in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)],
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )(raw_key)  # E: is not assignable


def test_wrong_output_descriptor(key: pltpu.PallasKey) -> None:
    pl.pallas_call(  # E: No matching overload
        body,
        in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)],
        out_shape=jax.ShapeDtypeStruct((8, 64), jnp.float32),
    )(key)


def test_wrong_generated_tile_width(
    key_ref: pltpu.PallasKeyRef,
    out_ref: pl.OutRef[[8, 64]],
) -> None:
    out_ref[...] = jax.random.uniform(  # E: Cannot set item
        key_ref[...], shape=(8, 128)
    )


def test_raw_ref_cannot_sample(
    raw_key_ref: pl.InRef[[2]], out_ref: pl.ShapeReadableOutRef[[8, 128]]
) -> None:
    out_ref[...] = jax.random.uniform(
        raw_key_ref[...],  # E: is not assignable
        shape=out_ref[...].shape,
    )


if not TYPE_CHECKING:
    import unittest

    class StatelessPallasKeyBoundaryTest(unittest.TestCase):
        def test_host_threefry_reference_only(self) -> None:
            key = jax.random.key(7, impl="threefry2x32")
            one = jax.random.uniform(key, shape=(8, 128))
            two = jax.random.uniform(key, shape=(8, 128))
            self.assertEqual(one.shape, (8, 128))
            self.assertEqual(one.dtype, jnp.float32)
            self.assertTrue(jnp.array_equal(one, two))

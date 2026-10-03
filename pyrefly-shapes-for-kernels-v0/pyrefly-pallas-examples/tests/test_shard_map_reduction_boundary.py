# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# @lint-ignore-every AUTODEPS2

"""A trailing-axis shard map returns both a result and double-buffer scratch."""

from __future__ import annotations

from collections.abc import Callable
from typing import assert_type, TYPE_CHECKING

import jax

if TYPE_CHECKING:
    from shape_extensions import IntVar


def reduction_shard_map[Devices: IntVar, Rows: IntVar, Cols: IntVar](
    local: Callable[
        [jax.Array[[Rows, Cols]]],
        tuple[jax.Array[[Rows, Cols]], jax.Array[[2, Rows, Cols]]],
    ],
    mesh: jax.Mesh[Devices],
    x: jax.Array[[Rows, Devices * Cols]],
) -> tuple[jax.Array[[Rows, Devices * Cols]], jax.Array[[2, Devices * Rows, Cols]]]:
    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=jax.P(None, "x"),
        out_specs=jax.P(None, "x"),
        check_vma=False,
    )(x)


def test_wrong_input[Devices: IntVar, Rows: IntVar, Cols: IntVar, Other: IntVar](
    local: Callable[
        [jax.Array[[Other, Cols]]],
        tuple[jax.Array[[Rows, Cols]], jax.Array[[2, Rows, Cols]]],
    ],
    mesh: jax.Mesh[Devices],
    x: jax.Array[[Rows, Devices * Cols]],
) -> None:
    jax.shard_map(  # E: No matching overload
        local,
        mesh=mesh,
        in_specs=jax.P(None, "x"),
        out_specs=jax.P(None, "x"),
    )(x)


def test_wrong_scratch_output[
    Devices: IntVar,
    Rows: IntVar,
    Cols: IntVar,
    Other: IntVar,
](
    local: Callable[
        [jax.Array[[Rows, Cols]]],
        tuple[jax.Array[[Rows, Cols]], jax.Array[[2, Rows, Other]]],
    ],
    mesh: jax.Mesh[Devices],
    x: jax.Array[[Rows, Devices * Cols]],
) -> None:
    jax.shard_map(  # E: No matching overload
        local,
        mesh=mesh,
        in_specs=jax.P(None, "x"),
        out_specs=jax.P(None, "x"),
    )(x)


def test_wrong_global_input[Devices: IntVar, Rows: IntVar, Cols: IntVar](
    local: Callable[
        [jax.Array[[Rows, Cols]]],
        tuple[jax.Array[[Rows, Cols]], jax.Array[[2, Rows, Cols]]],
    ],
    mesh: jax.Mesh[Devices],
    x: jax.Array[[Rows, Cols]],
) -> None:
    reduction_shard_map(local, mesh, x)  # E: is not assignable


def test_wrong_output_partition[Devices: IntVar, Rows: IntVar, Cols: IntVar](
    local: Callable[
        [jax.Array[[Rows, Cols]]],
        tuple[jax.Array[[Rows, Cols]], jax.Array[[2, Rows, Cols]]],
    ],
    mesh: jax.Mesh[Devices],
) -> None:
    jax.shard_map(  # E: No matching overload
        local,
        mesh=mesh,
        in_specs=jax.P(None, "x"),
        out_specs=jax.P("x", None),
    )


def test_output_shape[Devices: IntVar, Rows: IntVar, Cols: IntVar](
    local: Callable[
        [jax.Array[[Rows, Cols]]],
        tuple[jax.Array[[Rows, Cols]], jax.Array[[2, Rows, Cols]]],
    ],
    mesh: jax.Mesh[Devices],
    x: jax.Array[[Rows, Devices * Cols]],
) -> None:
    result, scratch = reduction_shard_map(local, mesh, x)
    assert_type(result, jax.Array[[Rows, Devices * Cols]])
    assert_type(scratch, jax.Array[[2, Devices * Rows, Cols]])


if not TYPE_CHECKING:
    import unittest

    import jax.numpy as jnp

    class ShardMapReductionBoundaryTest(unittest.TestCase):
        def test_two_cpu_devices(self) -> None:
            if len(jax.devices("cpu")) < 2:
                self.skipTest("Set XLA_FLAGS=--xla_force_host_platform_device_count=2")
            mesh = jax.make_mesh((2,), ("x",), devices=jax.devices("cpu")[:2])
            partition = jax.P(None, "x")
            x = jax.device_put(
                jnp.arange(8 * 256).reshape(8, 256),
                jax.sharding.NamedSharding(mesh, partition),
            )
            result, scratch = jax.shard_map(
                lambda shard: (shard, jnp.stack((shard, shard))),
                mesh=mesh,
                in_specs=partition,
                out_specs=partition,
                check_vma=False,
            )(x)
            self.assertEqual(result.shape, (8, 256))
            self.assertEqual(scratch.shape, (2, 16, 128))

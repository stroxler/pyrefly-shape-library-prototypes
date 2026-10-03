# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# @lint-ignore-every AUTODEPS2

"""Check that shard_map binds the local callback to global dimensions."""

from __future__ import annotations

from collections.abc import Callable
from typing import assert_type, TYPE_CHECKING

import jax

if TYPE_CHECKING:
    from shape_extensions import IntVar


def shard_map_contract[Devices: IntVar, Rows: IntVar, Cols: IntVar](
    f: Callable[[jax.Array[[Rows, Cols]]], jax.Array[[Devices, Rows, Cols]]],
    mesh: jax.Mesh[Devices],
    x: jax.Array[[Devices * Rows, Cols]],
) -> jax.Array[[Devices * Devices, Rows, Cols]]:
    wrapped = jax.shard_map(
        f,
        mesh=mesh,
        in_specs=jax.P("x", None),
        out_specs=jax.P("x", None),
        check_vma=False,
    )
    return wrapped(x)


def test_wrong_callback_local_input[
    Devices: IntVar,
    Rows: IntVar,
    Cols: IntVar,
    Other: IntVar,
](
    f: Callable[[jax.Array[[Other, Cols]]], jax.Array[[Devices, Rows, Cols]]],
    mesh: jax.Mesh[Devices],
    x: jax.Array[[Devices * Rows, Cols]],
) -> None:
    jax.shard_map(  # E: No matching overload
        f,
        mesh=mesh,
        in_specs=jax.P("x", None),
        out_specs=jax.P("x", None),
    )(x)


def test_wrong_callback_local_output[
    Devices: IntVar,
    Rows: IntVar,
    Cols: IntVar,
    Other: IntVar,
](
    f: Callable[[jax.Array[[Rows, Cols]]], jax.Array[[Other, Rows, Cols]]],
    mesh: jax.Mesh[Devices],
    x: jax.Array[[Devices * Rows, Cols]],
) -> None:
    jax.shard_map(  # E: No matching overload
        f,
        mesh=mesh,
        in_specs=jax.P("x", None),
        out_specs=jax.P("x", None),
    )(x)


def test_wrong_global_input[Devices: IntVar, Rows: IntVar, Cols: IntVar, Other: IntVar](
    f: Callable[[jax.Array[[Rows, Cols]]], jax.Array[[Devices, Rows, Cols]]],
    mesh: jax.Mesh[Devices],
    x: jax.Array[[Other, Cols]],
) -> None:
    shard_map_contract(f, mesh, x)  # E: is not assignable


def test_wrong_partition[Devices: IntVar, Rows: IntVar, Cols: IntVar](
    f: Callable[[jax.Array[[Rows, Cols]]], jax.Array[[Devices, Rows, Cols]]],
    mesh: jax.Mesh[Devices],
) -> None:
    jax.shard_map(  # E: No matching overload
        f,
        mesh=mesh,
        in_specs=jax.P(None, "x"),
        out_specs=jax.P("x", None),
    )


def test_output_shape[Devices: IntVar, Rows: IntVar, Cols: IntVar](
    f: Callable[[jax.Array[[Rows, Cols]]], jax.Array[[Devices, Rows, Cols]]],
    mesh: jax.Mesh[Devices],
    x: jax.Array[[Devices * Rows, Cols]],
) -> None:
    assert_type(
        shard_map_contract(f, mesh, x),
        jax.Array[[Devices * Devices, Rows, Cols]],
    )


def test_concrete_mesh_rejects_wrong_local_output[Rows: IntVar, Cols: IntVar](
    f: Callable[[jax.Array[[Rows, Cols]]], jax.Array[[3, Rows, Cols]]],
    x: jax.Array[[2 * Rows, Cols]],
) -> None:
    mesh: jax.Mesh[2] = jax.make_mesh((2,), ("x",))
    jax.shard_map(  # E: No matching overload
        f,
        mesh=mesh,
        in_specs=jax.P("x", None),
        out_specs=jax.P("x", None),
    )(x)


if not TYPE_CHECKING:
    import unittest

    import jax.numpy as jnp

    class ShardMapBoundaryTest(unittest.TestCase):
        def test_two_cpu_devices(self) -> None:
            if len(jax.devices("cpu")) < 2:
                self.skipTest("Set XLA_FLAGS=--xla_force_host_platform_device_count=2")
            mesh = jax.make_mesh((2,), ("x",), devices=jax.devices("cpu")[:2])
            partition = jax.P("x", None)
            x = jax.device_put(
                jnp.arange(16 * 128).reshape(16, 128),
                jax.sharding.NamedSharding(mesh, partition),
            )
            gathered = jax.shard_map(
                lambda shard: jax.lax.all_gather(shard, "x"),
                mesh=mesh,
                in_specs=partition,
                out_specs=partition,
            )(x)
            self.assertEqual(gathered.shape, (4, 8, 128))
            self.assertEqual(gathered[:, 0, 0].tolist(), [0, 1024, 0, 1024])

"""Track global and per-device dimensions through JAX's shard_map boundary."""

from __future__ import annotations

import unittest
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, assert_type, cast

import jax
import jax.numpy as jnp
from shape_extensions import IntVar


def shard_map_contract[Devices: IntVar, Rows: IntVar, Cols: IntVar](
    f: Callable[[jax.Array[[Rows, Cols]]], jax.Array[[Devices, Rows, Cols]]],
    mesh: jax.Mesh[Devices],
    x: jax.Array[[Devices * Rows, Cols]],
) -> jax.Array[[Devices * Devices, Rows, Cols]]:
    """Keep local rows and the device count in the global host signature."""
    wrapped = jax.shard_map(
        f,
        mesh=mesh,
        in_specs=jax.P("x", None),
        out_specs=jax.P("x", None),
        check_vma=False,
    )
    return wrapped(x)


if TYPE_CHECKING:

    def wrong_local_input[Devices: IntVar, Rows: IntVar, Cols: IntVar, Other: IntVar](
        f: Callable[[jax.Array[[Other, Cols]]], jax.Array[[Devices, Rows, Cols]]],
        mesh: jax.Mesh[Devices],
        x: jax.Array[[Devices * Rows, Cols]],
    ) -> None:
        jax.shard_map(  # pyrefly: ignore[no-matching-overload]
            f, mesh=mesh, in_specs=jax.P("x", None), out_specs=jax.P("x", None)
        )(x)

    def wrong_local_output[Devices: IntVar, Rows: IntVar, Cols: IntVar, Other: IntVar](
        f: Callable[[jax.Array[[Rows, Cols]]], jax.Array[[Other, Rows, Cols]]],
        mesh: jax.Mesh[Devices],
        x: jax.Array[[Devices * Rows, Cols]],
    ) -> None:
        jax.shard_map(  # pyrefly: ignore[no-matching-overload]
            f, mesh=mesh, in_specs=jax.P("x", None), out_specs=jax.P("x", None)
        )(x)

    def right_shape[Devices: IntVar, Rows: IntVar, Cols: IntVar](
        f: Callable[[jax.Array[[Rows, Cols]]], jax.Array[[Devices, Rows, Cols]]],
        mesh: jax.Mesh[Devices],
        x: jax.Array[[Devices * Rows, Cols]],
    ) -> None:
        assert_type(
            shard_map_contract(f, mesh, x), jax.Array[[Devices * Devices, Rows, Cols]]
        )


class ShardMapBoundaryTest(unittest.TestCase):
    def test_two_cpu_devices(self) -> None:
        runtime_jax = cast(Any, jax)
        if len(runtime_jax.devices("cpu")) < 2:
            self.skipTest("Set XLA_FLAGS=--xla_force_host_platform_device_count=2")
        mesh = runtime_jax.make_mesh(
            (2,), ("x",), devices=runtime_jax.devices("cpu")[:2]
        )
        partition = jax.P("x", None)
        x = runtime_jax.device_put(
            cast(Any, jnp.arange)(16 * 128).reshape(16, 128),
            runtime_jax.sharding.NamedSharding(mesh, partition),
        )
        gathered = runtime_jax.shard_map(
            lambda shard: runtime_jax.lax.all_gather(shard, "x"),
            mesh=mesh,
            in_specs=partition,
            out_specs=partition,
        )(x)
        self.assertEqual(gathered.shape, (4, 8, 128))
        self.assertEqual(gathered[:, 0, 0].tolist(), [0, 1024, 0, 1024])

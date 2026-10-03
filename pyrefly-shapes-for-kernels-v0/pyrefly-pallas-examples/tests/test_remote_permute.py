# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Kernel body from JAX docs/pallas/tpu/distributed.md (Apache-2.0).
# Semantic annotations exist only in the stub overlay.
# @lint-ignore-every AUTODEPS2

"""Pallas remotely copies a shard to its right-hand mesh neighbor."""

from __future__ import annotations

from collections.abc import Callable
from typing import assert_type, Literal, TYPE_CHECKING

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar


def permute_local_shard[Rows: IntVar, Cols: IntVar](
    x: jax.Array[[Rows, Cols]],
    rows: Int[Rows],
    cols: Int[Cols],
    num_devices: int,
    dtype: object,
) -> jax.Array[[Rows, Cols]]:
    def right_permute_kernel(
        input_ref: pl.UnconstrainedInRef[[Rows, Cols]],
        output_ref: pl.OutRef[[Rows, Cols]],
        send_sem: pltpu.DmaSemaphore,
        recv_sem: pltpu.DmaSemaphore,
    ) -> None:
        my_id = lax.axis_index("x")
        right_neighbor = lax.rem(my_id + 1, num_devices)
        remote_copy_op = pltpu.make_async_remote_copy(
            src_ref=input_ref,
            dst_ref=output_ref,
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=(right_neighbor,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        remote_copy_op.start()
        remote_copy_op.wait()

    shard_spec: pl.BlockSpec[[Rows, Cols], Literal["unconstrained"]] = pl.BlockSpec(
        memory_space=pl.ANY
    )
    grid_spec: pltpu.PrefetchScalarGridSpec[Rows, Cols, Rows, Cols] = (
        pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=(shard_spec,),
            out_specs=shard_spec,
            scratch_shapes=(pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA),
        )
    )
    call = pl.pallas_call(
        right_permute_kernel,
        out_shape=jax.ShapeDtypeStruct((rows, cols), dtype),
        grid_spec=grid_spec,
    )
    return call(x)


def test_wrong_host_shard_width[Rows: IntVar, Cols: IntVar, Other: IntVar](
    x: jax.Array[[Rows, Other]],
    rows: Int[Rows],
    cols: Int[Cols],
) -> None:
    permute_local_shard(x, rows, cols, 2, None)  # E: is not assignable


def test_wrong_host_shard_rows[Rows: IntVar, Cols: IntVar, Other: IntVar](
    x: jax.Array[[Other, Cols]],
    rows: Int[Rows],
    cols: Int[Cols],
) -> None:
    permute_local_shard(x, rows, cols, 2, None)  # E: is not assignable


def test_wrong_remote_destination[Rows: IntVar, Cols: IntVar, Other: IntVar](
    source: pl.UnconstrainedInRef[[Rows, Cols]],
    output: pl.OutRef[[Rows, Other]],
    send: pltpu.DmaSemaphore,
    recv: pltpu.DmaSemaphore,
) -> None:
    pltpu.make_async_remote_copy(  # E: No matching overload
        src_ref=source,
        dst_ref=output,
        send_sem=send,
        recv_sem=recv,
        device_id=(0,),
        device_id_type=pl.DeviceIdType.MESH,
    )


def test_wrong_semaphore_count[Rows: IntVar, Cols: IntVar](
    spec: pl.BlockSpec[[Rows, Cols], Literal["unconstrained"]],
) -> None:
    pltpu.PrefetchScalarGridSpec[Rows, Cols, Rows, Cols](  # E: No matching overload
        num_scalar_prefetch=0,
        in_specs=(spec,),
        out_specs=spec,
        scratch_shapes=(pltpu.SemaphoreType.DMA,),
    )


def test_wrong_output_allocation[Rows: IntVar, Cols: IntVar, Other: IntVar](
    kernel: Callable[
        [
            pl.UnconstrainedInRef[[Rows, Cols]],
            pl.OutRef[[Rows, Cols]],
            pltpu.DmaSemaphore,
            pltpu.DmaSemaphore,
        ],
        None,
    ],
    spec: pltpu.PrefetchScalarGridSpec[Rows, Cols, Rows, Cols],
    rows: Int[Rows],
    wrong_cols: Int[Other],
) -> None:
    pl.pallas_call(  # E: No matching overload
        kernel,
        grid_spec=spec,
        out_shape=jax.ShapeDtypeStruct((rows, wrong_cols), None),
    )


def test_zero_devices_are_not_excluded[Rows: IntVar, Cols: IntVar](
    x: jax.Array[[Rows, Cols]], rows: Int[Rows], cols: Int[Cols]
) -> None:
    assert_type(permute_local_shard(x, rows, cols, 0, None), jax.Array[[Rows, Cols]])


def test_local_shard_shape[Rows: IntVar, Cols: IntVar](
    x: jax.Array[[Rows, Cols]], rows: Int[Rows], cols: Int[Cols]
) -> None:
    assert_type(permute_local_shard(x, rows, cols, 2, None), jax.Array[[Rows, Cols]])

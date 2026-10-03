# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Kernel body from JAX docs/pallas/tpu/distributed.md (Apache-2.0).
# Semantic annotations exist only in the stub overlay.
# @lint-ignore-every AUTODEPS2

"""Relate a distributed all-gather's global arrays to local Pallas Refs."""

from __future__ import annotations

from collections.abc import Callable
from typing import assert_type, Literal, TYPE_CHECKING

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar


def gather_global[Devices: IntVar, Rows: IntVar, Cols: IntVar](
    mesh: jax.Mesh[Devices],
    num_devices: Int[Devices],
    rows: Int[Rows],
    cols: Int[Cols],
    dtype: object,
    x: jax.Array[[Devices * Rows, Cols]],
) -> jax.Array[[Devices * Devices, Rows, Cols]]:
    def all_gather_kernel(
        input_ref: pl.UnconstrainedInRef[[Rows, Cols]],
        output_ref: pl.OutRef[[Devices, Rows, Cols]],
        local_copy_sem: pltpu.DmaSemaphore,
        send_sem: pltpu.DmaSemaphore,
        recv_sems: pltpu.DmaSemaphoreArray[Devices - 1],
    ) -> None:
        outer_step = pl.program_id(0)
        my_id = lax.axis_index("x")
        right_neighbor = lax.rem(my_id + 1, num_devices)
        copy_slot = my_id - outer_step
        copy_slot = lax.rem(copy_slot + num_devices, num_devices)

        @pl.when(outer_step == 0)
        def _():
            local_copy_op = pltpu.make_async_copy(
                src_ref=input_ref,
                dst_ref=output_ref.at[my_id],
                sem=local_copy_sem,
            )
            local_copy_op.start()
            local_copy_op.wait()

        # Copy to our right neighbor.
        # Note that we will also be receiving data from our left neighbor,
        # but at `copy_slot-1` rather than `copy_slot`! This makes use of the fact
        # that the indices do not need to be symmetric between remote DMAs.
        remote_copy_op = pltpu.make_async_remote_copy(
            src_ref=output_ref.at[copy_slot],
            dst_ref=output_ref.at[copy_slot],
            send_sem=send_sem,
            recv_sem=recv_sems.at[outer_step],
            device_id=(right_neighbor,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        remote_copy_op.start()
        remote_copy_op.wait()

    input_spec: pl.BlockSpec[[Rows, Cols], Literal["unconstrained"]] = pl.BlockSpec(
        memory_space=pl.ANY
    )
    output_spec: pl.BlockSpec[[Devices, Rows, Cols], Literal["unconstrained"]] = (
        pl.BlockSpec(memory_space=pl.ANY)
    )
    scratch: pltpu.DmaSemaphoreArrayAllocation[Devices - 1] = pltpu.SemaphoreType.DMA(
        (num_devices - 1,)
    )
    grid_spec: pltpu.PrefetchScalarGridSpec[Rows, Cols, Rows, Cols, 1, 1, Devices] = (
        pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=(input_spec,),
            out_specs=output_spec,
            scratch_shapes=(pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA, scratch),
            grid=(num_devices - 1,),
        )
    )
    local = pl.pallas_call(
        all_gather_kernel,
        out_shape=jax.ShapeDtypeStruct((num_devices, rows, cols), dtype),
        grid_spec=grid_spec,
    )
    global_gather = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=jax.P("x", None),
        out_specs=jax.P("x", None),
        check_vma=False,
    )
    return global_gather(x)


def test_wrong_global_input_extent[
    Devices: IntVar,
    Rows: IntVar,
    Cols: IntVar,
    Other: IntVar,
](
    x: jax.Array[[Other, Cols]],
    mesh: jax.Mesh[Devices],
    devices: Int[Devices],
    rows: Int[Rows],
    cols: Int[Cols],
) -> None:
    gather_global(mesh, devices, rows, cols, None, x)  # E: is not assignable


def test_wrong_local_copy_width[Rows: IntVar, Cols: IntVar, Other: IntVar](
    input_ref: pl.UnconstrainedInRef[[Rows, Cols]],
    output_ref: pl.OutRef[[Rows, Other]],
    sem: pltpu.DmaSemaphore,
) -> None:
    pltpu.make_async_copy(  # E: No matching overload
        src_ref=input_ref,
        dst_ref=output_ref,
        sem=sem,
    )


def test_wrong_output_allocation[
    Devices: IntVar,
    Rows: IntVar,
    Cols: IntVar,
    Other: IntVar,
](
    kernel: Callable[
        [
            pl.UnconstrainedInRef[[Rows, Cols]],
            pl.OutRef[[Devices, Rows, Cols]],
            pltpu.DmaSemaphore,
            pltpu.DmaSemaphore,
            pltpu.DmaSemaphoreArray[Devices - 1],
        ],
        None,
    ],
    spec: pltpu.PrefetchScalarGridSpec[Rows, Cols, Rows, Cols, 1, 1, Devices],
    wrong_devices: Int[Other],
    rows: Int[Rows],
    cols: Int[Cols],
) -> None:
    pl.pallas_call(  # E: No matching overload
        kernel,
        grid_spec=spec,
        out_shape=jax.ShapeDtypeStruct((wrong_devices, rows, cols), None),
    )


def test_wrong_recv_semaphore_count[
    Devices: IntVar,
    Rows: IntVar,
    Cols: IntVar,
    Other: IntVar,
](
    input_spec: pl.BlockSpec[[Rows, Cols], Literal["unconstrained"]],
    output_spec: pl.BlockSpec[[Devices, Rows, Cols], Literal["unconstrained"]],
    scratch: pltpu.DmaSemaphoreArrayAllocation[Other - 1],
    devices: Int[Devices],
) -> None:
    pltpu.PrefetchScalarGridSpec[
        Rows, Cols, Rows, Cols, 1, 1, Devices
    ](  # E: No matching overload
        num_scalar_prefetch=0,
        in_specs=(input_spec,),
        out_specs=output_spec,
        scratch_shapes=(pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA, scratch),
        grid=(devices - 1,),
    )


def test_output_slot_bounds_are_not_proven[Devices: IntVar, Rows: IntVar, Cols: IntVar](
    out: pl.OutRef[[Devices, Rows, Cols]],
) -> None:
    assert_type(out.at[999], pl.OutRef[[Rows, Cols]])


def test_output_shape[Devices: IntVar, Rows: IntVar, Cols: IntVar](
    x: jax.Array[[Devices * Rows, Cols]],
    mesh: jax.Mesh[Devices],
    devices: Int[Devices],
    rows: Int[Rows],
    cols: Int[Cols],
) -> None:
    assert_type(
        gather_global(mesh, devices, rows, cols, None, x),
        jax.Array[[Devices * Devices, Rows, Cols]],
    )

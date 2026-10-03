# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Kernel body from JAX docs/pallas/tpu/distributed.md (Apache-2.0).
# Semantic annotations exist only in the stub overlay.
# @lint-ignore-every AUTODEPS2

"""Distributed Pallas reduce-scatter changes the sharded host-array axis."""

from __future__ import annotations

import functools
from typing import assert_type, Literal, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar


def local_barrier(
    left_neighbor: int, right_neighbor: int, double_barrier: bool = True
) -> None:
    """Performs a barrier with neighbors on the global barrier semaphore."""
    barrier_sem = pltpu.get_barrier_semaphore()
    for neighbor in [left_neighbor, right_neighbor]:
        pl.semaphore_signal(
            barrier_sem,
            inc=1,
            device_id=(neighbor,),
            device_id_type=pl.DeviceIdType.MESH,
        )
    pl.semaphore_wait(barrier_sem, 2)
    if double_barrier:

        @functools.partial(pl.run_scoped, second_barrier=pltpu.SemaphoreType.REGULAR)
        def _(second_barrier: pltpu.RegularSemaphore) -> None:
            for neighbor in [left_neighbor, right_neighbor]:
                pl.semaphore_signal(
                    second_barrier,
                    inc=1,
                    device_id=(neighbor,),
                    device_id_type=pl.DeviceIdType.MESH,
                )
            pl.semaphore_wait(second_barrier, 2)


def scatter_global[Devices: IntVar, Rows: IntVar, Cols: IntVar](  # noqa: C901
    mesh: jax.Mesh[Devices],
    num_devices: Int[Devices],
    block_size: tuple[Int[Rows], Int[Cols]],
    x: jax.Array[[Devices * Rows, Devices * Cols]],
) -> jax.Array[[Devices * Rows, Cols]]:
    LEFT = 0
    RIGHT = 1

    def mod(x, n):
        return lax.rem(x + n, n)

    def signal(left_or_right, semaphore):
        my_id = lax.axis_index("x")
        if left_or_right == LEFT:
            neighbor = mod(my_id - 1, num_devices)
        else:
            neighbor = mod(my_id + 1, num_devices)
        pl.semaphore_signal(
            semaphore,
            inc=1,
            device_id=(neighbor,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def reduce_scatter_kernel(
        x_ref: pl.VmemInRef[[Devices, Rows, Cols]],
        o_ref: pl.ScatterOutRef[[Rows, Cols]],
        hbm_scratch: pl.OutRef[[2, Rows, Cols]],
        local_copy_sem: pltpu.DmaSemaphore,
        left_recv_sem: pltpu.DmaSemaphore,
        left_send_sem: pltpu.DmaSemaphore,
        right_recv_sem: pltpu.DmaSemaphore,
        right_send_sem: pltpu.DmaSemaphore,
        left_capacity_sem: pltpu.RegularSemaphore,
        right_capacity_sem: pltpu.RegularSemaphore,
        accum_scratch: pltpu.VmemScratchRef[[Rows // 2, Cols]],
    ) -> None:
        outer_step = pl.program_id(0)
        phase = pl.program_id(1)
        is_start = jnp.logical_and(outer_step == 0, phase == 0)
        last_iteration = outer_step == pl.num_programs(0) - 1

        working_slot = lax.rem(outer_step, 2)
        receiving_slot = 1 - working_slot
        my_id = lax.axis_index("x")
        right_neighbor = mod(my_id + 1, num_devices)
        left_neighbor = mod(my_id - 1, num_devices)

        left_copy_device = mod(my_id + outer_step + 1, num_devices)
        right_copy_device = mod(my_id - outer_step - 1, num_devices)
        # Slices can be specified using pl.ds(start, size)
        left_copy_slice = pl.ds(0, block_size[0] // 2)
        right_copy_slice = pl.ds(block_size[0] // 2, block_size[0] // 2)
        current_phase_slice = pl.ds(phase * (block_size[0] // 2), block_size[0] // 2)

        initial_left_copy = pltpu.make_async_remote_copy(
            src_ref=x_ref.at[my_id, left_copy_slice],
            dst_ref=hbm_scratch.at[working_slot, left_copy_slice],
            send_sem=left_send_sem,
            recv_sem=left_recv_sem,
            device_id=(left_neighbor,),
            device_id_type=pl.DeviceIdType.MESH,
        )

        initial_right_copy = pltpu.make_async_remote_copy(
            src_ref=x_ref.at[my_id, right_copy_slice],
            dst_ref=hbm_scratch.at[working_slot, right_copy_slice],
            send_sem=right_send_sem,
            recv_sem=right_recv_sem,
            device_id=(right_neighbor,),
            device_id_type=pl.DeviceIdType.MESH,
        )

        left_copy = pltpu.make_async_remote_copy(
            src_ref=hbm_scratch.at[working_slot, left_copy_slice],
            dst_ref=hbm_scratch.at[receiving_slot, left_copy_slice],
            send_sem=left_send_sem,
            recv_sem=left_recv_sem,
            device_id=(left_neighbor,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        right_copy = pltpu.make_async_remote_copy(
            # Note: Right copy is flipped with regards to slots since we are copying
            # to the next outer_step iteration.
            src_ref=hbm_scratch.at[receiving_slot, right_copy_slice],
            dst_ref=hbm_scratch.at[working_slot, right_copy_slice],
            send_sem=right_send_sem,
            recv_sem=right_recv_sem,
            device_id=(right_neighbor,),
            device_id_type=pl.DeviceIdType.MESH,
        )

        # --- Prologue ---
        @pl.when(is_start)
        def _():
            # Barrier with both neighbors at the start, since we will be
            # communicating with both.
            local_barrier(left_neighbor, right_neighbor)

            # Initialize o_ref, acc_scratch, and hbm_scratch with initial copies.
            o_ref[...] = jnp.zeros_like(o_ref[...])
            accum_scratch[...] = jnp.zeros_like(accum_scratch[...])

            initial_left_copy.start()
            initial_left_copy.wait()
            initial_right_copy.start()

            # We tell our left neighbor that it is allowed to send to the right.
            # (and vice versa for right neighbor)
            signal(LEFT, right_capacity_sem)
            signal(RIGHT, left_capacity_sem)

        # --- Body ---
        # At the beginning of our kernel body, we start a DMA which copies
        # the result we computed in the previous phase to our neighbor.
        # This allows us to overlap the communication of sending our previous phase
        # with the computation for the current phase.
        @pl.when(~is_start)  # E: deprecated
        def _():
            @pl.when(phase == LEFT)
            def _():
                # We block here until our right neighbor tells use we can send to
                # the right.
                pl.semaphore_wait(right_capacity_sem, 1)
                right_copy.start()

            @pl.when(phase == RIGHT)
            def _():
                # We block here until our left neighbor tells use we can send to
                # the left.
                pl.semaphore_wait(left_capacity_sem, 1)
                left_copy.start()

        local_copy = pltpu.make_async_copy(
            src_ref=hbm_scratch.at[working_slot, current_phase_slice],
            dst_ref=accum_scratch,
            sem=local_copy_sem,
        )
        local_copy.start()
        local_copy.wait()

        @pl.when(~last_iteration)  # E: deprecated
        def _():
            @pl.when(phase == LEFT)
            def _():
                accum_scratch[...] += x_ref[left_copy_device, left_copy_slice]

            @pl.when(phase == RIGHT)
            def _():
                accum_scratch[...] += x_ref[right_copy_device, right_copy_slice]

        local_copy = pltpu.make_async_copy(
            src_ref=accum_scratch,
            dst_ref=hbm_scratch.at[working_slot, current_phase_slice],
            sem=local_copy_sem,
        )
        local_copy.start()
        local_copy.wait()

        @pl.when(is_start)
        def _():
            initial_right_copy.wait()

        # At the end of our kernel body, we wait on the DMA of the previous phase
        # to make sure the results are ready for the next phase.
        @pl.when(~is_start)  # E: deprecated
        def _():
            @pl.when(phase == LEFT)
            def _():
                right_copy.wait()
                signal(LEFT, right_capacity_sem)

            @pl.when(phase == RIGHT)
            def _():
                left_copy.wait()
                signal(RIGHT, left_capacity_sem)

        # --- Epilogue ---
        # Store result on last iteration.
        @pl.when(last_iteration)
        def _():
            # Clean up semaphores so that they exit with a value of 0.
            @pl.when(phase == LEFT)
            def _():
                o_ref[left_copy_slice, ...] = accum_scratch[...]
                pl.semaphore_wait(right_capacity_sem, 1)

            @pl.when(phase == RIGHT)
            def _():
                o_ref[right_copy_slice, ...] = accum_scratch[...]
                pl.semaphore_wait(left_capacity_sem, 1)

    half_rows: Int[Rows // 2] = block_size[0] // 2
    input_spec: pl.BlockSpec[[Devices, Rows, Cols], Literal["vmem"]] = pl.BlockSpec(
        memory_space=pltpu.VMEM
    )
    result_spec: pl.BlockSpec[[Rows, Cols], Literal["vmem"]] = pl.BlockSpec(
        memory_space=pltpu.VMEM
    )
    scratch_spec: pl.BlockSpec[[2, Rows, Cols], Literal["unconstrained"]] = (
        pl.BlockSpec(memory_space=pl.ANY)
    )
    accum: pltpu.VMEM[[Rows // 2, Cols]] = pltpu.VMEM(
        (half_rows, block_size[1]), jnp.float32
    )
    spec: pltpu.PrefetchScalarGridSpec[
        Devices * Rows, Cols, Rows, Cols, 1, 1, Devices
    ] = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=(input_spec,),
        out_specs=(result_spec, scratch_spec),
        grid=(num_devices, 2),
        scratch_shapes=(
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.REGULAR,
            pltpu.SemaphoreType.REGULAR,
            accum,
        ),
    )
    local = pl.pallas_call(
        reduce_scatter_kernel,
        out_shape=(
            jax.ShapeDtypeStruct(block_size, jnp.float32),
            jax.ShapeDtypeStruct((2, *block_size), jnp.float32),
        ),
        grid_spec=spec,
        compiler_params=pltpu.CompilerParams(collective_id=0),
    )

    def local_scatter(
        shard: jax.Array[[Devices * Rows, Cols]],
    ) -> jax.Array[[Rows, Cols]]:
        return local(shard.reshape((num_devices, *block_size)))[0]

    return jax.shard_map(
        local_scatter,
        mesh=mesh,
        in_specs=jax.P(None, "x"),
        out_specs=jax.P("x", None),
        check_vma=False,
    )(x)


def test_wrong_global_column_extent[
    Devices: IntVar,
    Rows: IntVar,
    Cols: IntVar,
    Other: IntVar,
](
    mesh: jax.Mesh[Devices],
    devices: Int[Devices],
    rows: Int[Rows],
    cols: Int[Cols],
    x: jax.Array[[Devices * Rows, Devices * Other]],
) -> None:
    scatter_global(mesh, devices, (rows, cols), x)  # E: is not assignable


def test_wrong_global_row_extent[
    Devices: IntVar,
    Rows: IntVar,
    Cols: IntVar,
    Other: IntVar,
](
    mesh: jax.Mesh[Devices],
    devices: Int[Devices],
    rows: Int[Rows],
    cols: Int[Cols],
    x: jax.Array[[Devices * Other, Devices * Cols]],
) -> None:
    scatter_global(mesh, devices, (rows, cols), x)  # E: is not assignable


def test_half_row_width_must_match_accumulator[
    Devices: IntVar,
    Rows: IntVar,
    Cols: IntVar,
    Other: IntVar,
](
    x: pl.VmemInRef[[Devices, Rows, Cols]],
    accum: pltpu.VmemScratchRef[[Rows // 2, Other]],
    rows: Int[Rows],
) -> None:
    accum[...] += x[0, pl.ds(0, rows // 2)]  # E: is not supported


def test_full_row_slice_is_not_a_half[Devices: IntVar, Rows: IntVar, Cols: IntVar](
    x: pl.VmemInRef[[Devices, Rows, Cols]], rows: Int[Rows]
) -> None:
    x.at[0, pl.ds(0, rows)]  # E: Cannot index


def test_wrong_result_memory_space[Rows: IntVar, Cols: IntVar]() -> None:
    pl.BlockSpec[[Rows, Cols], Literal["vmem"]](  # E: No matching overload
        memory_space=pl.ANY
    )


def test_wrong_source_to_scratch_width[
    Devices: IntVar,
    Rows: IntVar,
    Cols: IntVar,
    Other: IntVar,
](
    x: pl.VmemInRef[[Devices, Rows, Cols]],
    scratch: pl.OutRef[[2, Rows, Other]],
    send: pltpu.DmaSemaphore,
    recv: pltpu.DmaSemaphore,
    rows: Int[Rows],
) -> None:
    pltpu.make_async_remote_copy(  # E: No matching overload
        src_ref=x.at[0, pl.ds(0, rows // 2)],
        dst_ref=scratch.at[0, pl.ds(0, rows // 2)],
        send_sem=send,
        recv_sem=recv,
        device_id=(1,),
        device_id_type=pl.DeviceIdType.MESH,
    )


def test_wrong_phase_grid[Devices: IntVar, Rows: IntVar, Cols: IntVar](
    source: pl.BlockSpec[[Devices, Rows, Cols], Literal["vmem"]],
    result: pl.BlockSpec[[Rows, Cols], Literal["vmem"]],
    buffer: pl.BlockSpec[[2, Rows, Cols], Literal["unconstrained"]],
    accum: pltpu.VMEM[[Rows // 2, Cols]],
    devices: Int[Devices],
) -> None:
    pltpu.PrefetchScalarGridSpec[
        Devices * Rows, Cols, Rows, Cols, 1, 1, Devices
    ](  # E: No matching overload
        num_scalar_prefetch=0,
        in_specs=(source,),
        out_specs=(result, buffer),
        grid=(devices, 3),
        scratch_shapes=(
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.REGULAR,
            pltpu.SemaphoreType.REGULAR,
            accum,
        ),
    )


def test_reshape_element_count_is_not_proven[
    Devices: IntVar,
    Rows: IntVar,
    Cols: IntVar,
    Other: IntVar,
](
    shard: jax.Array[[Other, Cols]],
    devices: Int[Devices],
    rows: Int[Rows],
    cols: Int[Cols],
) -> None:
    assert_type(shard.reshape((devices, rows, cols)), jax.Array[[Devices, Rows, Cols]])


def test_half_row_slice_start_is_not_proven[
    Devices: IntVar,
    Rows: IntVar,
    Cols: IntVar,
](
    x: pl.VmemInRef[[Devices, Rows, Cols]],
    rows: Int[Rows],
) -> None:
    assert_type(x[0, pl.ds(999, rows // 2)], pl.Tile[[Rows // 2, Cols]])


def test_output_shape[Devices: IntVar, Rows: IntVar, Cols: IntVar](
    mesh: jax.Mesh[Devices],
    devices: Int[Devices],
    rows: Int[Rows],
    cols: Int[Cols],
    x: jax.Array[[Devices * Rows, Devices * Cols]],
) -> None:
    assert_type(
        scatter_global(mesh, devices, (rows, cols), x),
        jax.Array[[Devices * Rows, Cols]],
    )


if not TYPE_CHECKING:
    import unittest

    import numpy as np

    class ReduceScatterBoundaryTest(unittest.TestCase):
        def test_two_cpu_devices(self) -> None:
            if len(jax.devices("cpu")) < 2:
                self.skipTest("Set XLA_FLAGS=--xla_force_host_platform_device_count=2")
            devices, rows, cols = 2, 16, 128
            mesh = jax.make_mesh((devices,), ("x",), devices=jax.devices("cpu")[:2])
            x = jax.device_put(
                jnp.arange(devices * rows * devices * cols).reshape(
                    devices * rows, devices * cols
                ),
                jax.sharding.NamedSharding(mesh, jax.P(None, "x")),
            )
            result = jax.shard_map(
                lambda shard: lax.psum_scatter(shard.reshape(devices, rows, cols), "x"),
                mesh=mesh,
                in_specs=jax.P(None, "x"),
                out_specs=jax.P("x", None),
                check_vma=False,
            )(x)
            self.assertEqual(result.shape, (devices * rows, cols))
            host = jax.device_get(x)
            self.assertTrue(
                np.array_equal(jax.device_get(result), host[:, :cols] + host[:, cols:])
            )

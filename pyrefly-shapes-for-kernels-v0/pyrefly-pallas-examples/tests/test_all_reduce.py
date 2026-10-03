# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Kernel and barrier helper bodies from JAX docs/pallas/tpu/distributed.md
# (Apache-2.0). Semantic annotations exist only in the stub overlay.
# @lint-ignore-every AUTODEPS2

"""Pallas all-reduce links a sharded host array to a VMEM accumulator."""

from __future__ import annotations

import functools
from collections.abc import Callable
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
    """Performs a barrier with neighbors on the global barrier semaphore.

    Optionally performs a second barrier, which prevents a potential race
    when reusing the same collective_id across kernel invocations.
    """
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
        # The double-barrier prevents a race condition where one neighbor can
        # re-enter the kernel again on a subsequent call and increment the
        # barrier semaphore a second time. This would unblock the current device
        # even if the other neighbor is not ready yet.
        # To implement a double-barrier, we stack-allocate a second REGULAR
        # semaphore using run_scoped.
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


def reduce_global[Devices: IntVar, Rows: IntVar, Cols: IntVar](
    mesh: jax.Mesh[Devices],
    num_devices: Int[Devices],
    rows: Int[Rows],
    cols: Int[Cols],
    dtype: object,
    x: jax.Array[[Rows, Devices * Cols]],
) -> jax.Array[[Rows, Devices * Cols]]:
    def all_reduce_kernel(
        x_ref: pl.VmemInRef[[Rows, Cols]],
        o_ref: pl.AccumRef[[Rows, Cols]],
        hbm_scratch: pl.OutRef[[2, Rows, Cols]],
        copy_sem: pltpu.DmaSemaphore,
        remote_recv_sem: pltpu.DmaSemaphore,
        remote_send_sem: pltpu.DmaSemaphore,
        capacity_sem: pltpu.RegularSemaphore,
        receive_scratch: pltpu.VmemScratchRef[[Rows, Cols]],
    ) -> None:
        outer_step = pl.program_id(0)
        working_slot = lax.rem(outer_step, 2)
        receiving_slot = 1 - working_slot

        my_id = lax.axis_index("x")
        right_neighbor = lax.rem(my_id + 1, num_devices)
        left_neighbor = lax.rem(my_id - 1 + num_devices, num_devices)

        @pl.when(outer_step == 0)
        def _():
            # Barrier with both neighbors at the start, since we will be
            # communicating with both.
            local_barrier(left_neighbor, right_neighbor)

            # Initialize o_ref, acc_scratch, and hbm_scratch.
            o_ref[...] = jnp.zeros_like(o_ref)
            receive_scratch[...] = jnp.zeros_like(receive_scratch)
            initial_copy = pltpu.make_async_remote_copy(
                src_ref=x_ref,
                dst_ref=hbm_scratch.at[working_slot],
                send_sem=remote_send_sem,
                recv_sem=remote_recv_sem,
                device_id=(right_neighbor,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            initial_copy.start()
            initial_copy.wait()

        # Signal to our left neighbor that we are ready to receive.
        # Without this signal, our left neighbor can be >=1 iteration ahead,
        # meaning it could write into our working slot.
        pl.semaphore_signal(
            capacity_sem,
            inc=1,
            device_id=(left_neighbor,),
            device_id_type=pl.DeviceIdType.MESH,
        )

        # Copy the partial result our left neighbor sent to us into VMEM for
        # computation.
        local_copy = pltpu.make_async_copy(
            src_ref=hbm_scratch.at[working_slot],
            dst_ref=receive_scratch,
            sem=copy_sem,
        )
        local_copy.start()

        # Block until our right neighbor is ready to receive.
        pl.semaphore_wait(capacity_sem, 1)
        # Pass the value to our right neighbor.
        remote_copy = pltpu.make_async_remote_copy(
            src_ref=hbm_scratch.at[working_slot],
            dst_ref=hbm_scratch.at[receiving_slot],
            send_sem=remote_send_sem,
            recv_sem=remote_recv_sem,
            device_id=(right_neighbor,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        remote_copy.start()
        # Finish local copy and accumulate while remote_copy is happening.
        local_copy.wait()
        o_ref[...] += receive_scratch[...]
        # Block until remote copy finishes.
        remote_copy.wait()

    input_spec: pl.BlockSpec[[Rows, Cols], Literal["vmem"]] = pl.BlockSpec(
        memory_space=pltpu.VMEM
    )
    result_spec: pl.BlockSpec[[Rows, Cols], Literal["vmem"]] = pl.BlockSpec(
        memory_space=pltpu.VMEM
    )
    scratch_spec: pl.BlockSpec[[2, Rows, Cols], Literal["unconstrained"]] = (
        pl.BlockSpec(memory_space=pl.ANY)
    )
    vmem: pltpu.VMEM[[Rows, Cols]] = pltpu.VMEM((rows, cols), dtype)
    grid_spec: pltpu.PrefetchScalarGridSpec[Rows, Cols, Rows, Cols, 1, 1, Devices] = (
        pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=(input_spec,),
            out_specs=(result_spec, scratch_spec),
            scratch_shapes=(
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.REGULAR,
                vmem,
            ),
            grid=(num_devices,),
        )
    )
    local = pl.pallas_call(
        all_reduce_kernel,
        out_shape=(
            jax.ShapeDtypeStruct((rows, cols), dtype),
            jax.ShapeDtypeStruct((2, rows, cols), dtype),
        ),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(collective_id=0),
    )
    result, scratch = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=jax.P(None, "x"),
        out_specs=jax.P(None, "x"),
        check_vma=False,
    )(x)
    assert_type(scratch, jax.Array[[2, Devices * Rows, Cols]])
    return result


def test_wrong_global_input[Devices: IntVar, Rows: IntVar, Cols: IntVar, Other: IntVar](
    mesh: jax.Mesh[Devices],
    devices: Int[Devices],
    rows: Int[Rows],
    cols: Int[Cols],
    x: jax.Array[[Rows, Other]],
) -> None:
    reduce_global(mesh, devices, rows, cols, None, x)  # E: is not assignable


def test_wrong_accumulator_width[Rows: IntVar, Cols: IntVar, Other: IntVar](
    result: pl.AccumRef[[Rows, Other]],
    scratch: pltpu.VmemScratchRef[[Rows, Cols]],
) -> None:
    result[...] += scratch[...]  # E: is not supported


def test_wrong_double_buffer_copy[Rows: IntVar, Cols: IntVar, Other: IntVar](
    input_ref: pl.VmemInRef[[Rows, Cols]],
    scratch: pl.OutRef[[2, Rows, Other]],
    send: pltpu.DmaSemaphore,
    recv: pltpu.DmaSemaphore,
) -> None:
    pltpu.make_async_remote_copy(  # E: No matching overload
        src_ref=input_ref,
        dst_ref=scratch.at[0],
        send_sem=send,
        recv_sem=recv,
        device_id=(1,),
        device_id_type=pl.DeviceIdType.MESH,
    )


def test_wrong_input_memory_space[Rows: IntVar, Cols: IntVar]() -> None:
    pl.BlockSpec[[Rows, Cols], Literal["vmem"]](  # E: No matching overload
        memory_space=pl.ANY
    )


def test_wrong_vmem_scratch_allocation[
    Devices: IntVar,
    Rows: IntVar,
    Cols: IntVar,
    Other: IntVar,
](
    input_spec: pl.BlockSpec[[Rows, Cols], Literal["vmem"]],
    result_spec: pl.BlockSpec[[Rows, Cols], Literal["vmem"]],
    scratch_spec: pl.BlockSpec[[2, Rows, Cols], Literal["unconstrained"]],
    wrong_vmem: pltpu.VMEM[[Rows, Other]],
    vmem: pltpu.VMEM[[Rows, Cols]],
    devices: Int[Devices],
) -> None:
    pltpu.PrefetchScalarGridSpec[
        Rows, Cols, Rows, Cols, 1, 1, Devices
    ](  # E: No matching overload
        num_scalar_prefetch=0,
        in_specs=(input_spec,),
        out_specs=(result_spec, scratch_spec),
        scratch_shapes=(
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.REGULAR,
            wrong_vmem,
        ),
        grid=(devices,),
    )
    pltpu.PrefetchScalarGridSpec[
        Rows, Cols, Rows, Cols, 1, 1, Devices
    ](  # E: No matching overload
        num_scalar_prefetch=0,
        in_specs=(input_spec,),
        out_specs=(result_spec, scratch_spec),
        scratch_shapes=(
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            vmem,
        ),
        grid=(devices,),
    )


def test_wrong_output_allocations[
    Devices: IntVar,
    Rows: IntVar,
    Cols: IntVar,
    Other: IntVar,
](
    kernel: Callable[
        [
            pl.VmemInRef[[Rows, Cols]],
            pl.AccumRef[[Rows, Cols]],
            pl.OutRef[[2, Rows, Cols]],
            pltpu.DmaSemaphore,
            pltpu.DmaSemaphore,
            pltpu.DmaSemaphore,
            pltpu.RegularSemaphore,
            pltpu.VmemScratchRef[[Rows, Cols]],
        ],
        None,
    ],
    spec: pltpu.PrefetchScalarGridSpec[Rows, Cols, Rows, Cols, 1, 1, Devices],
    rows: Int[Rows],
    cols: Int[Cols],
    wrong_cols: Int[Other],
) -> None:
    pl.pallas_call(  # E: No matching overload
        kernel,
        grid_spec=spec,
        out_shape=(
            jax.ShapeDtypeStruct((rows, wrong_cols), None),
            jax.ShapeDtypeStruct((2, rows, cols), None),
        ),
        compiler_params=pltpu.CompilerParams(collective_id=0),
    )
    pl.pallas_call(  # E: No matching overload
        kernel,
        grid_spec=spec,
        out_shape=(
            jax.ShapeDtypeStruct((rows, cols), None),
            jax.ShapeDtypeStruct((2, rows, wrong_cols), None),
        ),
        compiler_params=pltpu.CompilerParams(collective_id=0),
    )


def test_double_buffer_slot_bounds_not_proven[Rows: IntVar, Cols: IntVar](
    scratch: pl.OutRef[[2, Rows, Cols]],
) -> None:
    assert_type(scratch.at[99], pl.OutRef[[Rows, Cols]])


def test_output_shape[Devices: IntVar, Rows: IntVar, Cols: IntVar](
    mesh: jax.Mesh[Devices],
    devices: Int[Devices],
    rows: Int[Rows],
    cols: Int[Cols],
    x: jax.Array[[Rows, Devices * Cols]],
) -> None:
    assert_type(
        reduce_global(mesh, devices, rows, cols, None, x),
        jax.Array[[Rows, Devices * Cols]],
    )

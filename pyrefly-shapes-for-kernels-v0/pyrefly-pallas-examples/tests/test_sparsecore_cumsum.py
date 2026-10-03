# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The kernel body comes from JAX docs/pallas/tpu/sparsecore.md (Apache-2.0);
# only its parameter annotations are new.
# @lint-ignore-every AUTODEPS2

"""SparseCore scalar-subcore DMA connects each host row to a scratch row."""

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import jax
import jax.lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu, tpu_sc as plsc

if TYPE_CHECKING:
    from shape_extensions import IntVar

    scalar_mesh: plsc.ScalarSubcoreMesh[2]
else:
    # Constructing a SparseCore mesh requires TPU hardware, unavailable on CPU.
    scalar_mesh = None


@jax.jit
def cumsum[Lanes: IntVar](x: jax.Array[[2, Lanes]]) -> jax.Array[[2, Lanes]]:
    @pl.kernel(
        out_type=x,
        mesh=scalar_mesh,
        scratch_types=[
            pltpu.SMEM((x.shape[1],), x.dtype),
            pltpu.SemaphoreType.DMA,
        ],
    )
    def kernel(
        x_ref: pl.CoreInRef[2, Lanes],
        o_ref: pl.CoreOutRef[2, Lanes],
        tmp_ref: pltpu.SmemScratchRef[[Lanes]],
        sem: pltpu.DmaSemaphore,
    ) -> None:
        idx = jax.lax.axis_index("core")
        pltpu.async_copy(x_ref.at[idx], tmp_ref, sem).wait()

        @pl.loop(1, x.shape[1])
        def _(i):
            tmp_ref[i] += tmp_ref[i - 1]

        pltpu.async_copy(tmp_ref, o_ref.at[idx], sem).wait()

    return kernel(x)


def test_host_shape[Lanes: IntVar](x: jax.Array[[2, Lanes]]) -> None:
    assert_type(cumsum(x), jax.Array[[2, Lanes]])


def test_wrong_host_core_count[Lanes: IntVar](x: jax.Array[[3, Lanes]]) -> None:
    cumsum(x)  # E: is not assignable


def test_wrong_source_scratch_lanes[Lanes: IntVar, Other: IntVar](
    source: pl.InRef[[Lanes]],
    scratch: pltpu.SmemScratchRef[[Other]],
    semaphore: pltpu.DmaSemaphore,
) -> None:
    pltpu.async_copy(source, scratch, semaphore)  # E: No matching overload


def test_wrong_output_scratch_lanes[Lanes: IntVar, Other: IntVar](
    scratch: pltpu.SmemScratchRef[[Lanes]],
    output: pl.OutRef[[Other]],
    semaphore: pltpu.DmaSemaphore,
) -> None:
    pltpu.async_copy(scratch, output, semaphore)  # E: No matching overload


def test_wrong_scratch_descriptor_lanes[Lanes: IntVar, Other: IntVar](
    data: jax.Array[[2, Lanes]],
    scratch: pltpu.SMEM[[Other]],
    mesh: plsc.ScalarSubcoreMesh[2],
) -> None:
    pl.kernel(  # E: No matching overload
        out_type=data,
        mesh=mesh,
        scratch_types=[scratch, pltpu.SemaphoreType.DMA],
    )


def test_wrong_mesh_core_count[Lanes: IntVar](
    data: jax.Array[[2, Lanes]],
    mesh: plsc.ScalarSubcoreMesh[3],
) -> None:
    pl.kernel(  # E: No matching overload
        out_type=data,
        mesh=mesh,
        scratch_types=[
            pltpu.SMEM((data.shape[1],), data.dtype),
            pltpu.SemaphoreType.DMA,
        ],
    )


def test_wrong_scratch_order_is_not_rejected[Lanes: IntVar](
    data: jax.Array[[2, Lanes]],
    mesh: plsc.ScalarSubcoreMesh[2],
) -> None:
    pl.kernel(
        out_type=data,
        mesh=mesh,
        scratch_types=[
            pltpu.SemaphoreType.DMA,
            pltpu.SMEM((data.shape[1],), data.dtype),
        ],
    )


def test_wrong_scratch_dtype_is_not_rejected[Lanes: IntVar](
    data: jax.Array[[2, Lanes]],
    mesh: plsc.ScalarSubcoreMesh[2],
) -> None:
    pl.kernel(
        out_type=data,
        mesh=mesh,
        scratch_types=[
            pltpu.SMEM((data.shape[1],), jnp.bfloat16),
            pltpu.SemaphoreType.DMA,
        ],
    )


if not TYPE_CHECKING:
    import unittest

    class SparsecoreCumsumBoundaryTest(unittest.TestCase):
        def test_host_per_row_cumsum_shape(self) -> None:
            x = jnp.array([[2, 1, 5], [3, 4, 1]], dtype=jnp.int32)
            result = jnp.cumsum(x, axis=1)
            self.assertEqual(result.shape, x.shape)
            self.assertTrue(jnp.array_equal(result, jnp.array([[2, 3, 8], [3, 7, 8]])))

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Kernel and nested callback bodies from JAX docs/pallas/tpu/sparsecore.md
# (Apache-2.0); only parameter annotations are added.
# @lint-ignore-every AUTODEPS2

"""SparseCore scatter writes indirectly into a separately sized host array."""

from __future__ import annotations

from collections.abc import Callable
from typing import assert_type, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar


def scatter_global[Batch: IntVar, Num: IntVar, Cols: IntVar, Window: IntVar](
    x: jax.Array[[Num, Cols]],
    indices: jax.Array[[Num]],
    batch_size: Int[Batch],
    value_dim: Int[Cols],
    num_indices: Int[Num],
    gather_window_size: Int[Window],
    vector_mesh: object,
    dtype: object,
) -> jax.Array[[Batch, Cols]]:
    shaped_indices = indices.reshape((1, num_indices))

    @pl.kernel(
        out_type=jax.ShapeDtypeStruct((batch_size, value_dim), dtype),
        mesh=vector_mesh,
        scratch_types=[],
    )
    def kernel(
        x_hbm: pl.InRef[[Num, Cols]],
        i_hbm: pl.InRef[[1, Num]],
        o_hbm: pl.OutRef[[Batch, Cols]],
    ) -> None:
        def body(
            x_vmem: pl.InRef[[Window, Cols]],
            i_vmem: pl.GatherIndicesRef[Window],
        ) -> None:
            pltpu.sync_copy(x_vmem, o_hbm.at[i_vmem.at[0]])  # The scatter op

        pltpu.emit_pipeline(
            body,
            grid=(num_indices // gather_window_size,),
            in_specs=[
                pl.BlockSpec(
                    (gather_window_size, value_dim), index_map=lambda i: (i, 0)
                ),
                pl.BlockSpec((1, gather_window_size), index_map=lambda i: (0, i)),
            ],
            out_specs=[],
            core_axis_name="subcore",
            dimension_semantics=(pltpu.PARALLEL,),
        )(x_hbm, i_hbm)

    return kernel(x, shaped_indices)


def test_wrong_host_index_count[
    Batch: IntVar,
    Num: IntVar,
    Other: IntVar,
    Cols: IntVar,
    Window: IntVar,
](
    x: jax.Array[[Num, Cols]],
    indices: jax.Array[[Other]],
    batch: Int[Batch],
    cols: Int[Cols],
    num: Int[Num],
    window: Int[Window],
) -> None:
    scatter_global(
        x,
        indices,  # E: is not assignable
        batch,
        cols,
        num,
        window,
        None,
        None,
    )


def test_wrong_scatter_destination_width[
    Batch: IntVar,
    Cols: IntVar,
    Other: IntVar,
    Window: IntVar,
](
    values: pl.InRef[[Window, Cols]],
    destination: pl.OutRef[[Batch, Other]],
    indices: pl.Indices[Window],
) -> None:
    pltpu.sync_copy(values, destination.at[indices])  # E: No matching overload


def test_wrong_index_window[Batch: IntVar, Cols: IntVar, Window: IntVar, Other: IntVar](
    values: pl.InRef[[Window, Cols]],
    destination: pl.OutRef[[Batch, Cols]],
    indices: pl.Indices[Other],
) -> None:
    pltpu.sync_copy(values, destination.at[indices])  # E: No matching overload


def test_pipeline_rejects_wrong_index_count[
    Num: IntVar,
    Other: IntVar,
    Cols: IntVar,
    Window: IntVar,
](
    body: Callable[[pl.InRef[[Window, Cols]], pl.GatherIndicesRef[Window]], None],
    values: pl.InRef[[Num, Cols]],
    indices: pl.InRef[[1, Other]],
    window: Int[Window],
    cols: Int[Cols],
) -> None:
    pipeline = pltpu.emit_pipeline(
        body,
        grid=(1,),
        in_specs=[
            pl.BlockSpec((window, cols), index_map=lambda i: (i, 0)),
            pl.BlockSpec((1, window), index_map=lambda i: (0, i)),
        ],
        out_specs=[],
        core_axis_name="subcore",
        dimension_semantics=(pltpu.PARALLEL,),
    )
    pipeline(values, indices)  # E: is not assignable


def test_wrong_output_batch_allocation[
    Batch: IntVar,
    Num: IntVar,
    Other: IntVar,
    Cols: IntVar,
](
    body: Callable[
        [pl.InRef[[Num, Cols]], pl.InRef[[1, Num]], pl.OutRef[[Other, Cols]]],
        None,
    ],
    rows: Int[Batch],
    cols: Int[Cols],
) -> None:
    pl.kernel(
        out_type=jax.ShapeDtypeStruct((rows, cols), None),
        mesh=None,
        scratch_types=[],
    )(body)  # E: is not assignable


def test_duplicate_destination_indices_remain_accepted[
    Batch: IntVar,
    Cols: IntVar,
    Window: IntVar,
](
    values: pl.InRef[[Window, Cols]],
    destination: pl.OutRef[[Batch, Cols]],
    indices: pl.Indices[Window],
) -> None:
    pltpu.sync_copy(values, destination.at[indices])


def test_output_extent_is_batch_not_index_count[
    Batch: IntVar,
    Num: IntVar,
    Cols: IntVar,
    Window: IntVar,
](
    values: jax.Array[[Num, Cols]],
    indices: jax.Array[[Num]],
    batch: Int[Batch],
    cols: Int[Cols],
    num: Int[Num],
    window: Int[Window],
) -> None:
    assert_type(
        scatter_global(values, indices, batch, cols, num, window, None, None),
        jax.Array[[Batch, Cols]],
    )


if not TYPE_CHECKING:
    import unittest

    class SparsecoreScatterBoundaryTest(unittest.TestCase):
        def test_host_destination_rows_are_independent_of_index_count(self) -> None:
            values = jnp.array([[1, 2], [3, 4], [5, 6]])
            indices = jnp.array([4, 0, 2])
            result = jnp.zeros((5, 2), dtype=values.dtype).at[indices].set(values)
            self.assertEqual(result.shape, (5, 2))
            self.assertTrue(
                jnp.array_equal(
                    result, jnp.array([[3, 4], [0, 0], [5, 6], [0, 0], [1, 2]])
                )
            )

        def test_duplicate_indices_still_have_destination_shape(self) -> None:
            values = jnp.array([[1, 2], [3, 4]])
            result = (
                jnp.zeros((5, 2), dtype=values.dtype).at[jnp.array([1, 1])].set(values)
            )
            self.assertEqual(result.shape, (5, 2))

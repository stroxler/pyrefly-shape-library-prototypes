# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The nested kernel and callback bodies come from JAX docs/pallas/tpu/sparsecore.md
# (Apache-2.0); only their parameter annotations are new.
# @lint-ignore-every AUTODEPS2

"""SparseCore's indirect gather connects host index extent to output rows."""

from __future__ import annotations

from collections.abc import Callable
from typing import assert_type, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar


def gather_global[Batch: IntVar, Num: IntVar, Cols: IntVar, Window: IntVar](
    x: jax.Array[[Batch, Cols]],
    indices: jax.Array[[Num]],
    value_dim: Int[Cols],
    num_indices: Int[Num],
    gather_window_size: Int[Window],
    vector_mesh: object,
    dtype: object,
) -> jax.Array[[Num, Cols]]:
    shaped_indices = indices.reshape((1, num_indices))

    @pl.kernel(
        out_type=jax.ShapeDtypeStruct((num_indices, value_dim), dtype),
        mesh=vector_mesh,
    )
    def kernel(
        x_hbm: pl.InRef[[Batch, Cols]],
        i_hbm: pl.InRef[[1, Num]],
        o_hbm: pl.OutRef[[Num, Cols]],
    ) -> None:
        def body(
            i_vmem: pl.GatherIndicesRef[Window],
            o_vmem: pl.OutRef[[Window, Cols]],
        ) -> None:
            pltpu.sync_copy(x_hbm.at[i_vmem.at[0]], o_vmem)  # The gather op

        pltpu.emit_pipeline(
            body,
            grid=(num_indices // gather_window_size,),
            in_specs=[
                pl.BlockSpec((1, gather_window_size), index_map=lambda i: (0, i))
            ],
            out_specs=[
                pl.BlockSpec(
                    (gather_window_size, value_dim), index_map=lambda i: (i, 0)
                )
            ],
            core_axis_name="subcore",
            dimension_semantics=(pltpu.PARALLEL,),
        )(i_hbm, o_hbm)

    return kernel(x, shaped_indices)


def test_wrong_host_indices[
    Batch: IntVar,
    Num: IntVar,
    Other: IntVar,
    Cols: IntVar,
    Window: IntVar,
](
    x: jax.Array[[Batch, Cols]],
    indices: jax.Array[[Other]],
    cols: Int[Cols],
    num: Int[Num],
    window: Int[Window],
) -> None:
    gather_global(x, indices, cols, num, window, None, None)  # E: is not assignable


def test_wrong_host_data_width[
    Batch: IntVar,
    Num: IntVar,
    Other: IntVar,
    Cols: IntVar,
    Window: IntVar,
](
    x: jax.Array[[Batch, Other]],
    indices: jax.Array[[Num]],
    cols: Int[Cols],
    num: Int[Num],
    window: Int[Window],
) -> None:
    gather_global(x, indices, cols, num, window, None, None)  # E: is not assignable


def test_wrong_gather_tile_width[
    Batch: IntVar,
    Window: IntVar,
    Cols: IntVar,
    Other: IntVar,
](
    x: pl.InRef[[Batch, Cols]],
    indices: pl.Indices[Window],
    out: pl.OutRef[[Window, Other]],
) -> None:
    pltpu.sync_copy(x.at[indices], out)  # E: No matching overload


def test_wrong_pipeline_output_extent[
    Num: IntVar,
    Other: IntVar,
    Cols: IntVar,
    Window: IntVar,
](
    body: Callable[[pl.GatherIndicesRef[Window], pl.OutRef[[Window, Cols]]], None],
    indices: pl.InRef[[1, Num]],
    out: pl.OutRef[[Other, Cols]],
    window: Int[Window],
    cols: Int[Cols],
) -> None:
    pipeline = pltpu.emit_pipeline(
        body,
        grid=(1,),
        in_specs=[pl.BlockSpec((1, window), index_map=lambda i: (0, i))],
        out_specs=[pl.BlockSpec((window, cols), index_map=lambda i: (i, 0))],
        core_axis_name="subcore",
        dimension_semantics=(pltpu.PARALLEL,),
    )
    pipeline(indices, out)  # E: is not assignable


def test_wrong_kernel_output_allocation[
    Batch: IntVar,
    Num: IntVar,
    Other: IntVar,
    Cols: IntVar,
](
    body: Callable[
        [pl.InRef[[Batch, Cols]], pl.InRef[[1, Num]], pl.OutRef[[Other, Cols]]],
        None,
    ],
    num: Int[Num],
    cols: Int[Cols],
) -> None:
    pl.kernel(out_type=jax.ShapeDtypeStruct((num, cols), None), mesh=None)(
        body  # E: is not assignable
    )


def test_pipeline_grid_coverage_is_not_proven[
    Num: IntVar,
    Cols: IntVar,
    Window: IntVar,
](
    body: Callable[[pl.GatherIndicesRef[Window], pl.OutRef[[Window, Cols]]], None],
    indices: pl.InRef[[1, Num]],
    out: pl.OutRef[[Num, Cols]],
    window: Int[Window],
    cols: Int[Cols],
) -> None:
    pipeline = pltpu.emit_pipeline(
        body,
        grid=(999,),
        in_specs=[pl.BlockSpec((1, window), index_map=lambda i: (0, i))],
        out_specs=[pl.BlockSpec((window, cols), index_map=lambda i: (i, 0))],
        core_axis_name="subcore",
        dimension_semantics=(pltpu.PARALLEL,),
    )
    pipeline(indices, out)


def test_output_rows_come_from_index_array[
    Batch: IntVar,
    Num: IntVar,
    Cols: IntVar,
    Window: IntVar,
](
    x: jax.Array[[Batch, Cols]],
    indices: jax.Array[[Num]],
    cols: Int[Cols],
    num: Int[Num],
    window: Int[Window],
) -> None:
    assert_type(
        gather_global(x, indices, cols, num, window, None, None),
        jax.Array[[Num, Cols]],
    )


if not TYPE_CHECKING:
    import unittest

    class SparsecoreGatherBoundaryTest(unittest.TestCase):
        def test_host_index_extent_controls_result_rows(self) -> None:
            x = jnp.arange(5 * 3).reshape(5, 3)
            indices = jnp.array([0, 4, 1, 0])
            result = jnp.take(x, indices, axis=0)
            self.assertEqual(result.shape, (4, 3))
            self.assertTrue(
                jnp.array_equal(
                    result, jnp.array([[0, 1, 2], [12, 13, 14], [3, 4, 5], [0, 1, 2]])
                )
            )

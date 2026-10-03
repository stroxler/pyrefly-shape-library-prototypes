# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Both full kernel bodies come from JAX docs/pallas/tpu/core_map.md
# (Apache-2.0); only their parameter annotations are new.
# @lint-ignore-every AUTODEPS2

"""A one-element scalar prefetch chooses the input half-width window."""

from __future__ import annotations

from typing import assert_type, Literal, TYPE_CHECKING

import jax
import jax.lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar

    device_mesh: jax.Mesh[2]
else:
    # No TPU mesh exists on CPU; the sharded interface is checked statically.
    device_mesh = None

num_cores: Literal[2] = 2


def add_one_body(in_vmem: pl.InRef[[8, 128]], out_vmem: pl.OutRef[[8, 128]]) -> None:
    out_vmem[...] = in_vmem[...] + 1


def indexed_add_one[Rows: IntVar, Cols: IntVar](
    x: jax.Array[[Rows, Cols]],
    index: jax.Int32Array[[1]],
) -> jax.Array[[Rows, Cols // 2]]:
    output_shape = (x.shape[0], x.shape[1] // 2)
    tc_mesh = None

    def indexed_add_one_kernel(
        in_refs: tuple[pl.UnconstrainedInRef[[Rows, Cols]], pl.UnconstrainedInRef[[1]]],
        out_refs: pl.OutRef[[Rows, Cols // 2]],
        i_smem_ref: pltpu.SmemScratchRef[[1]],
    ) -> None:
        (x_hbm_ref, i_hbm_ref), o_hbm_ref = in_refs, out_refs
        in_shape = x_hbm_ref.shape
        pltpu.sync_copy(i_hbm_ref, i_smem_ref)

        core_idx = jax.lax.axis_index("core")
        core_slc_size = in_shape[0] // num_cores
        # split work among cores
        i_map = lambda i: core_idx * core_slc_size // 8 + i  # noqa: E731
        # use the prefetched offset
        j_map = lambda j: i_smem_ref[0] // 128 + j  # noqa: E731

        pltpu.emit_pipeline(
            add_one_body,
            grid=(core_slc_size // 8, output_shape[1] // 128),
            in_specs=[
                pl.BlockSpec(
                    block_shape=(8, 128),
                    index_map=lambda i, j: (i_map(i), j_map(j)),
                )
            ],
            out_specs=[
                pl.BlockSpec(
                    block_shape=(8, 128),
                    index_map=lambda i, j: (i_map(i), j),
                )
            ],
        )(x_hbm_ref, o_hbm_ref)

    out_type = jax.ShapeDtypeStruct((x.shape[0], x.shape[1] // 2), x.dtype)
    return pl.kernel(
        indexed_add_one_kernel,
        out_type=out_type,
        mesh=tc_mesh,
        scratch_types=[pltpu.SMEM((1,), jnp.int32)],
    )((x, index))


def sharded_indexed_add_one[LocalRows: IntVar, Cols: IntVar](
    local_rows: Int[LocalRows],
    x: jax.Array[[2 * LocalRows, Cols]],
    index: jax.Int32Array[[1]],
) -> jax.Array[[2 * LocalRows, Cols // 2]]:
    # A replicated scalar and row-sharded matrix use the guide's partitioning.
    in_spec = jax.P("device", None)

    def local_kernel(
        local_x: jax.Array[[LocalRows, Cols]], local_index: jax.Int32Array[[1]]
    ) -> jax.Array[[LocalRows, Cols // 2]]:
        return indexed_add_one(local_x, local_index)

    return jax.shard_map(
        local_kernel,
        mesh=device_mesh,
        in_specs=(in_spec, jax.P()),
        out_specs=in_spec,
        check_vma=False,
    )(x, index)


def test_global_sharded_output[LocalRows: IntVar, Cols: IntVar](
    local_rows: Int[LocalRows],
    x: jax.Array[[2 * LocalRows, Cols]],
    index: jax.Int32Array[[1]],
) -> None:
    assert_type(
        sharded_indexed_add_one(local_rows, x, index),
        jax.Array[[2 * LocalRows, Cols // 2]],
    )


def test_wrong_global_sharded_rows[LocalRows: IntVar, Cols: IntVar, Other: IntVar](
    local_rows: Int[LocalRows],
    x: jax.Array[[Other, Cols]],
    index: jax.Int32Array[[1]],
) -> None:
    sharded_indexed_add_one(local_rows, x, index)  # E: is not assignable


def test_host_output[Rows: IntVar, Cols: IntVar](
    x: jax.Array[[Rows, Cols]], index: jax.Int32Array[[1]]
) -> None:
    assert_type(indexed_add_one(x, index), jax.Array[[Rows, Cols // 2]])


def test_wrong_index_count[Rows: IntVar, Cols: IntVar, Other: IntVar](
    x: jax.Array[[Rows, Cols]], index: jax.Int32Array[[Other]]
) -> None:
    indexed_add_one(x, index)  # E: is not assignable


def test_wrong_index_dtype[Rows: IntVar, Cols: IntVar](
    x: jax.Array[[Rows, Cols]], index: jax.Array[[1]]
) -> None:
    indexed_add_one(x, index)  # E: is not assignable


def test_explicit_index_dtype[Rows: IntVar, Cols: IntVar](
    x: jax.Array[[Rows, Cols]], index: jax.Array[[1]]
) -> None:
    assert_type(
        indexed_add_one(x, index.astype(jnp.int32)),
        jax.Array[[Rows, Cols // 2]],
    )


def test_wrong_prefetch_scratch_extent[Other: IntVar](
    index: pl.UnconstrainedInRef[[1]],
    scratch: pltpu.SmemScratchRef[[Other]],
) -> None:
    pltpu.sync_copy(index, scratch)  # E: No matching overload


def test_wrong_pipeline_output_extent[Rows: IntVar, Cols: IntVar, Other: IntVar](
    pipeline: pltpu.IndexedAddPipeline,
    input_ref: pl.UnconstrainedInRef[[Rows, Cols]],
    output_ref: pl.OutRef[[Rows, Other]],
) -> None:
    pipeline(input_ref, output_ref)  # E: is not assignable


def test_wrong_pipeline_body_width[Other: IntVar](
    input_tile: pl.InRef[[8, 128]],
    output_tile: pl.OutRef[[8, Other]],
) -> None:
    output_tile[...] = input_tile[...] + 1  # E: Cannot set item


if not TYPE_CHECKING:
    import unittest

    class CoreMapIndexedAddBoundaryTest(unittest.TestCase):
        def test_half_width_indexed_reference(self) -> None:
            x = jnp.arange(4 * 1024, dtype=jnp.float32).reshape(4, 1024)
            idx = jnp.array([256], dtype=jnp.int32)
            result = x[:, int(idx[0]) : int(idx[0]) + x.shape[1] // 2] + 1
            self.assertEqual(result.shape, (4, 512))
            self.assertEqual(float(result[0, 0]), float(x[0, 256] + 1))

        def test_out_of_range_index_does_not_cover_half_width(self) -> None:
            x = jnp.zeros((4, 1024), dtype=jnp.float32)
            index = 700
            host_slice = x[:, index : index + x.shape[1] // 2]
            self.assertEqual(host_slice.shape, (4, 324))

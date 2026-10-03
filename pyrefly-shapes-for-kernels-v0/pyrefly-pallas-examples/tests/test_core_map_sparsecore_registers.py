# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Both kernel bodies are from JAX docs/pallas/tpu/core_map.md (Apache-2.0);
# only the kernel parameters carry new semantic annotations.
# @lint-ignore-every AUTODEPS2

"""Bounded SparseCore tiles are divided into 4-by-16 register operations."""

from __future__ import annotations

from typing import assert_type, Literal, TYPE_CHECKING

import jax
import jax.lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu, tpu_sc as plsc

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar

    sc_mesh: plsc.VectorSubcoreMesh[4, 16]
    device_mesh: jax.Mesh[2]
else:
    # SparseCore meshes require TPU hardware; CPU tests use a host reference.
    sc_mesh = None
    device_mesh = None

sc_num_cores: Literal[4] = 4
sc_num_subcores: Literal[16] = 16
SC_REG_OP_SHAPE: tuple[Literal[4], Literal[16]] = (4, 16)


def sc_add_one_body(
    in_vmem: pl.BoundedInRef[8, 128], out_vmem: pl.BoundedOutRef[8, 128]
) -> None:
    @pl.loop(0, in_vmem.shape[0], step=SC_REG_OP_SHAPE[0])
    def _reg_loop_0(c0):
        @pl.loop(0, in_vmem.shape[1], step=SC_REG_OP_SHAPE[1])
        def _reg_loop_1(c1):
            slc = (pl.ds(c0, SC_REG_OP_SHAPE[0]), pl.ds(c1, SC_REG_OP_SHAPE[1]))
            out_vmem[slc] = in_vmem[slc] + 1


def sc_add_one[Rows: IntVar](
    x: jax.Array[[Rows, 128]],
) -> jax.Array[[Rows, 128]]:
    def sc_add_one_kernel(
        x_hbm_ref: pl.UnconstrainedInRef[[Rows, 128]],
        o_hbm_ref: pl.OutRef[[Rows, 128]],
    ) -> None:
        in_shape = x_hbm_ref.shape
        core_idx = jax.lax.axis_index("core")
        subcore_idx = jax.lax.axis_index("subcore")
        cm_idx = core_idx * sc_num_subcores + subcore_idx  # index on the core_map
        slc_size = in_shape[0] // (sc_num_subcores * sc_num_cores)
        index_map = lambda i, j: (  # noqa: E731
            pl.ds(pl.multiple_of(cm_idx * slc_size + i * 8, 8), 8),
            j,
        )

        pltpu.emit_pipeline(
            sc_add_one_body,
            grid=(slc_size // 8, in_shape[1] // 128),
            in_specs=[
                pl.BlockSpec(
                    block_shape=(pl.BoundedSlice(8), 128),
                    index_map=index_map,
                )
            ],
            out_specs=[
                pl.BlockSpec(
                    block_shape=(pl.BoundedSlice(8), 128),
                    index_map=index_map,
                )
            ],
        )(x_hbm_ref, o_hbm_ref)

    return pl.kernel(sc_add_one_kernel, out_type=x, mesh=sc_mesh, scratch_types=[])(x)


def sharded_sc_add_one[LocalRows: IntVar](
    local_rows: Int[LocalRows], x: jax.Array[[2 * LocalRows, 128]]
) -> jax.Array[[2 * LocalRows, 128]]:
    in_spec = jax.P("device", None)

    def local_kernel(
        local_x: jax.Array[[LocalRows, 128]],
    ) -> jax.Array[[LocalRows, 128]]:
        return sc_add_one(local_x)

    return jax.shard_map(
        local_kernel,
        mesh=device_mesh,
        in_specs=in_spec,
        out_specs=in_spec,
        check_vma=False,
    )(x)


def test_host_extent[Rows: IntVar](x: jax.Array[[Rows, 128]]) -> None:
    assert_type(sc_add_one(x), jax.Array[[Rows, 128]])


def test_global_sharded_extent[Rows: IntVar](
    local_rows: Int[Rows], x: jax.Array[[2 * Rows, 128]]
) -> None:
    assert_type(sharded_sc_add_one(local_rows, x), jax.Array[[2 * Rows, 128]])


def test_wrong_global_rows[Rows: IntVar, Other: IntVar](
    local_rows: Int[Rows], x: jax.Array[[Other, 128]]
) -> None:
    sharded_sc_add_one(local_rows, x)  # E: is not assignable


def test_wrong_host_width[Rows: IntVar, Other: IntVar](
    x: jax.Array[[Rows, Other]],
) -> None:
    sc_add_one(x)  # E: is not assignable


def test_wrong_pipeline_output_rows[Rows: IntVar, Other: IntVar](
    pipeline: pltpu.SparseCoreRegisterPipeline,
    x: pl.UnconstrainedInRef[[Rows, 128]],
    output: pl.OutRef[[Other, 128]],
) -> None:
    pipeline(x, output)  # E: is not assignable


def test_wrong_register_write_width[Other: IntVar](
    output: pl.BoundedOutRef[8, 128],
    input_tile: pl.Tile[[4, Other]],
) -> None:
    output[pl.ds(0, 4), pl.ds(0, 16)] = input_tile  # E: Cannot set item


def test_unaligned_boundaries_do_not_form_a_sparsecore_block() -> None:
    pl.BlockSpec(  # E: No matching overload
        block_shape=(pl.BoundedSlice(8), 128),
        index_map=lambda i, j: (pl.ds(i * 8, 8), j),
    )


if not TYPE_CHECKING:
    import unittest

    class SparseCoreRegisterBoundaryTest(unittest.TestCase):
        def test_cpu_reference_shape_and_values(self) -> None:
            x = jnp.arange(1024 * 128, dtype=jnp.int32).reshape(1024, 128)
            y = x + 1
            self.assertEqual(y.shape, (1024, 128))
            self.assertEqual(int(y[0, 0]), 1)
            self.assertEqual(int(y[-1, -1]), 1024 * 128)

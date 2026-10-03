# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Kernel body from JAX docs/pallas/tpu/pipelining.md (Apache-2.0).
# Semantic annotations exist only in the stub overlay.
# @lint-ignore-every AUTODEPS2

"""Pallas copies dynamically sized row blocks using a scratch-memory index."""

from __future__ import annotations

from collections.abc import Callable
from typing import assert_type, Literal, TYPE_CHECKING

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar


def copy_dynamic_blocks[Rows: IntVar, SliceRows: IntVar](
    x: jax.Array[[Rows, 128]], slices: jax.Array[[SliceRows, 2]], dtype: object
) -> jax.Array[[Rows, 128]]:
    # The original example's free `slices` is captured by this wrapper.
    def dynamic_block_example_kernel(
        x_hbm: pl.UnconstrainedInRef[[Rows, 128]],
        slices_hbm: pl.UnconstrainedInRef[[SliceRows, 2]],
        o_hbm: pl.OutRef[[Rows, 128]],
        slices_smem: pltpu.SmemScratchRef[[SliceRows, 2]],
    ) -> None:
        pltpu.sync_copy(slices_hbm, slices_smem)  # Copy slices into SMEM.

        def pipeline_body(
            x_vmem: pl.BoundedInRef[8, 128],
            o_vmem: pl.BoundedOutRef[8, 128],
        ) -> None:
            o_vmem[...] = x_vmem[...]

        def index_map(i: int) -> tuple[pl.DynamicSlice, int]:
            start = slices_smem[i, 0]
            size = slices_smem[i, 1] - slices_smem[i, 0]
            return (pl.ds(start, size), 0)

        block_spec = pl.BlockSpec(
            block_shape=(pl.BoundedSlice(8), 128),
            index_map=index_map,
        )
        pltpu.emit_pipeline(
            pipeline_body,
            grid=(slices.shape[0],),
            in_specs=[block_spec],
            out_specs=block_spec,
        )(x_hbm, o_hbm)

    x_spec: pl.BlockSpec[[Rows, 128], Literal["unconstrained"]] = pl.BlockSpec(
        memory_space=pl.ANY
    )
    slices_spec: pl.BlockSpec[[SliceRows, 2], Literal["unconstrained"]] = pl.BlockSpec(
        memory_space=pl.ANY
    )
    count, width = slices.shape
    scratch: pltpu.SMEM[[SliceRows, 2]] = pltpu.SMEM((count, width), dtype)
    rows, cols = x.shape
    call = pl.pallas_call(
        dynamic_block_example_kernel,
        in_specs=(x_spec, slices_spec),
        out_specs=x_spec,
        out_shape=jax.ShapeDtypeStruct((rows, cols), dtype),
        scratch_shapes=(scratch,),
        interpret=True,
    )
    return call(x, slices)


def test_wrong_input_width[Rows: IntVar, SliceRows: IntVar](
    x: jax.Array[[Rows, 127]],
    slices: jax.Array[[SliceRows, 2]],
) -> None:
    copy_dynamic_blocks(x, slices, None)  # E: is not assignable


def test_wrong_slice_record_width[Rows: IntVar, SliceRows: IntVar](
    x: jax.Array[[Rows, 128]],
    slices: jax.Array[[SliceRows, 3]],
) -> None:
    copy_dynamic_blocks(x, slices, None)  # E: is not assignable


def test_swapped_host_inputs[Rows: IntVar, SliceRows: IntVar](
    x: jax.Array[[Rows, 128]],
    slices: jax.Array[[SliceRows, 2]],
) -> None:
    copy_dynamic_blocks(
        slices,  # E: is not assignable
        x,  # E: is not assignable
        None,
    )


def test_wrong_copy_scratch[SliceRows: IntVar, Other: IntVar](
    slices_hbm: pl.UnconstrainedInRef[[SliceRows, 2]],
    scratch: pltpu.SmemScratchRef[[Other, 2]],
) -> None:
    pltpu.sync_copy(slices_hbm, scratch)  # E: No matching overload


def test_wrong_dynamic_index_map() -> None:
    pl.BlockSpec[[8, 128], Literal["dynamic"]](  # E: No matching overload
        block_shape=(pl.BoundedSlice(8), 128),
        index_map=lambda i: (i, 0),
    )


def test_wrong_pipeline_tile(
    source: pl.BoundedInRef[8, 128],
    destination: pl.BoundedOutRef[8, 127],
) -> None:
    destination[...] = source[...]  # E: Cannot set item


def test_wrong_output_allocation[Rows: IntVar, SliceRows: IntVar, Other: IntVar](
    x_spec: pl.BlockSpec[[Rows, 128], Literal["unconstrained"]],
    slices_spec: pl.BlockSpec[[SliceRows, 2], Literal["unconstrained"]],
    scratch: pltpu.SMEM[[SliceRows, 2]],
    wrong_rows: Int[Other],
) -> None:
    def kernel(
        x_hbm: pl.UnconstrainedInRef[[Rows, 128]],
        slices_hbm: pl.UnconstrainedInRef[[SliceRows, 2]],
        out: pl.OutRef[[Rows, 128]],
        slices_smem: pltpu.SmemScratchRef[[SliceRows, 2]],
    ) -> None:
        pass

    pl.pallas_call(  # E: No matching overload
        kernel,
        in_specs=(x_spec, slices_spec),
        out_specs=x_spec,
        scratch_shapes=(scratch,),
        out_shape=jax.ShapeDtypeStruct((wrong_rows, 128), None),
    )


def test_wrong_scratch_allocation[Rows: IntVar, SliceRows: IntVar, Other: IntVar](
    kernel: Callable[
        [
            pl.UnconstrainedInRef[[Rows, 128]],
            pl.UnconstrainedInRef[[SliceRows, 2]],
            pl.OutRef[[Rows, 128]],
            pltpu.SmemScratchRef[[SliceRows, 2]],
        ],
        None,
    ],
    x_spec: pl.BlockSpec[[Rows, 128], Literal["unconstrained"]],
    slices_spec: pl.BlockSpec[[SliceRows, 2], Literal["unconstrained"]],
    scratch: pltpu.SMEM[[Other, 2]],
    rows: Int[Rows],
) -> None:
    pl.pallas_call(  # E: No matching overload
        kernel,
        in_specs=(x_spec, slices_spec),
        out_specs=x_spec,
        scratch_shapes=(scratch,),
        out_shape=jax.ShapeDtypeStruct((rows, 128), None),
    )


def test_wrong_output_spec[Rows: IntVar, SliceRows: IntVar, Other: IntVar](
    kernel: Callable[
        [
            pl.UnconstrainedInRef[[Rows, 128]],
            pl.UnconstrainedInRef[[SliceRows, 2]],
            pl.OutRef[[Rows, 128]],
            pltpu.SmemScratchRef[[SliceRows, 2]],
        ],
        None,
    ],
    x_spec: pl.BlockSpec[[Rows, 128], Literal["unconstrained"]],
    slices_spec: pl.BlockSpec[[SliceRows, 2], Literal["unconstrained"]],
    bad_out_spec: pl.BlockSpec[[Rows, Other], Literal["unconstrained"]],
    scratch: pltpu.SMEM[[SliceRows, 2]],
    rows: Int[Rows],
) -> None:
    pl.pallas_call(  # E: No matching overload
        kernel,
        in_specs=(x_spec, slices_spec),
        out_specs=bad_out_spec,
        scratch_shapes=(scratch,),
        out_shape=jax.ShapeDtypeStruct((rows, 128), None),
    )


def test_preserved_host_shape[Rows: IntVar, SliceRows: IntVar](
    x: jax.Array[[Rows, 128]],
    slices: jax.Array[[SliceRows, 2]],
) -> None:
    assert_type(copy_dynamic_blocks(x, slices, None), jax.Array[[Rows, 128]])

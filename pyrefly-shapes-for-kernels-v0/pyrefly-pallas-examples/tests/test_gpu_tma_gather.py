# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The complete kernel comes from JAX docs/pallas/gpu/reference.md.
# Only semantic annotations on its parameters are added.
# @lint-ignore-every AUTODEPS2

"""Blackwell TMA gather loads SMEM indices before an asynchronous GMEM copy."""

from __future__ import annotations

from typing import assert_type, Literal, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

if TYPE_CHECKING:
    from shape_extensions import IntVar

    SourceRows = IntVar("SourceRows")
    Rows = IntVar("Rows")
    Cols = IntVar("Cols")


def kernel(
    x_ref_gmem: plgpu.TmaGmemSourceRef[SourceRows, Cols],
    idx_ref: plgpu.TmaIndexRef[Rows],
    o_ref: plgpu.TmaSmemOutRef[Rows, Cols],
    barrier_ref: plgpu.TmaBarrier,
):
    idxs = plgpu.load(idx_ref, layout=plgpu.Layout.TMA_INDICES)
    plgpu.copy_gmem_to_smem(x_ref_gmem.at[idxs], o_ref, barrier_ref)
    plgpu.barrier_wait(barrier_ref)


def gpu_tma_gather[SourceRows: IntVar, Rows: IntVar, Cols: IntVar](
    source: jax.Array[[SourceRows, Cols]],
    indices: jax.Array[[Rows]],
) -> jax.Array[[Rows, Cols]]:
    # The guide's `self.pallas_call` is its test harness, not a Mosaic GPU API.
    source_spec: pl.BlockSpec[[SourceRows, Cols], Literal["tma_gmem"]] = pl.BlockSpec(
        memory_space=plgpu.GMEM
    )
    index_spec: pl.BlockSpec[[Rows], Literal["tma_smem_indices"]] = pl.BlockSpec(
        memory_space=plgpu.SMEM
    )
    out_spec: plgpu.BlockSpec[Rows, Cols] = plgpu.BlockSpec(
        memory_space=plgpu.SMEM, transforms=()
    )

    def body(
        x_ref_gmem: plgpu.TmaGmemSourceRef[SourceRows, Cols],
        idx_ref: plgpu.TmaIndexRef[Rows],
        o_ref: plgpu.TmaSmemOutRef[Rows, Cols],
        barrier_ref: plgpu.TmaBarrier,
    ) -> None:
        kernel(x_ref_gmem, idx_ref, o_ref, barrier_ref)

    return pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct(
            (indices.shape[0], source.shape[1]), source.dtype
        ),
        in_specs=(source_spec, index_spec),
        out_specs=out_spec,
        scratch_shapes=[plgpu.Barrier()],
    )(source, indices)


def test_correct_interface[SourceRows: IntVar, Rows: IntVar, Cols: IntVar](
    source: jax.Array[[SourceRows, Cols]],
    indices: jax.Array[[Rows]],
) -> None:
    assert_type(gpu_tma_gather(source, indices), jax.Array[[Rows, Cols]])


def test_wrong_indices_layout[Rows: IntVar](idx_ref: plgpu.TmaIndexRef[Rows]) -> None:
    plgpu.load(idx_ref, layout=plgpu.Layout.WG_STRIDED)  # E: is not assignable


def test_wrong_spec_spaces[SourceRows: IntVar, Rows: IntVar, Cols: IntVar]() -> None:
    source: pl.BlockSpec[[SourceRows, Cols], Literal["tma_gmem"]] = (
        pl.BlockSpec(  # E: is not assignable
            memory_space=plgpu.SMEM
        )
    )
    indices: pl.BlockSpec[[Rows], Literal["tma_smem_indices"]] = (
        pl.BlockSpec(  # E: is not assignable
            memory_space=plgpu.GMEM
        )
    )
    _ = source, indices


def test_wrong_output_spec_memory_space[Rows: IntVar, Cols: IntVar]() -> None:
    output: plgpu.BlockSpec[Rows, Cols] = plgpu.BlockSpec(
        memory_space=plgpu.GMEM,  # E: is not assignable
        transforms=(),
    )
    _ = output


def test_wrong_source_memory_at_copy[Rows: IntVar, Cols: IntVar](
    smem_source: plgpu.SmemInRef[Rows, Cols],
    output: plgpu.TmaSmemOutRef[Rows, Cols],
    barrier: plgpu.TmaBarrier,
) -> None:
    plgpu.copy_gmem_to_smem(smem_source, output, barrier)  # E: is not assignable


def test_wrong_destination_memory_at_copy[
    SourceRows: IntVar,
    Rows: IntVar,
    Cols: IntVar,
](
    source: plgpu.TmaGatherGmemSlice[SourceRows, Rows, Cols],
    gmem_output: plgpu.GmemOutRef[Rows, Cols],
    barrier: plgpu.TmaBarrier,
) -> None:
    plgpu.copy_gmem_to_smem(source, gmem_output, barrier)  # E: is not assignable


def test_wrong_output_rows[
    SourceRows: IntVar,
    Rows: IntVar,
    Other: IntVar,
    Cols: IntVar,
](
    source: plgpu.TmaGmemSourceRef[SourceRows, Cols],
    idx: plgpu.TmaIndexRef[Rows],
    wrong_output: plgpu.TmaSmemOutRef[Other, Cols],
    barrier: plgpu.TmaBarrier,
) -> None:
    loaded = plgpu.load(idx, layout=plgpu.Layout.TMA_INDICES)
    plgpu.copy_gmem_to_smem(
        source.at[loaded],
        wrong_output,  # E: is not assignable
        barrier,
    )


def test_wrong_output_columns[
    SourceRows: IntVar,
    Rows: IntVar,
    Cols: IntVar,
    Other: IntVar,
](
    source: plgpu.TmaGmemSourceRef[SourceRows, Cols],
    idx: plgpu.TmaIndexRef[Rows],
    wrong_output: plgpu.TmaSmemOutRef[Rows, Other],
    barrier: plgpu.TmaBarrier,
) -> None:
    loaded = plgpu.load(idx, layout=plgpu.Layout.TMA_INDICES)
    plgpu.copy_gmem_to_smem(
        source.at[loaded],
        wrong_output,  # E: is not assignable
        barrier,
    )


def test_wrong_copy_barrier[SourceRows: IntVar, Rows: IntVar, Cols: IntVar](
    source: plgpu.TmaGmemSourceRef[SourceRows, Cols],
    idx: plgpu.TmaIndexRef[Rows],
    output: plgpu.TmaSmemOutRef[Rows, Cols],
    not_a_barrier: int,
) -> None:
    loaded = plgpu.load(idx, layout=plgpu.Layout.TMA_INDICES)
    plgpu.copy_gmem_to_smem(
        source.at[loaded],
        output,
        not_a_barrier,  # E: is not assignable
    )


def test_wrong_host_output_columns[
    SourceRows: IntVar,
    Rows: IntVar,
    Cols: IntVar,
    Other: IntVar,
](
    source: jax.Array[[SourceRows, Cols]],
    indices: jax.Array[[Rows]],
    wrong: jax.Array[[Other]],
    source_spec: pl.BlockSpec[[SourceRows, Cols], Literal["tma_gmem"]],
    index_spec: pl.BlockSpec[[Rows], Literal["tma_smem_indices"]],
    out_spec: plgpu.BlockSpec[Rows, Cols],
) -> None:
    def body(
        x_ref_gmem: plgpu.TmaGmemSourceRef[SourceRows, Cols],
        idx_ref: plgpu.TmaIndexRef[Rows],
        o_ref: plgpu.TmaSmemOutRef[Rows, Cols],
        barrier_ref: plgpu.TmaBarrier,
    ) -> None:
        kernel(x_ref_gmem, idx_ref, o_ref, barrier_ref)

    pl.pallas_call(  # E: No matching overload
        body,
        out_shape=jax.ShapeDtypeStruct(
            (indices.shape[0], wrong.shape[0]), source.dtype
        ),
        in_specs=(source_spec, index_spec),
        out_specs=out_spec,
        scratch_shapes=[plgpu.Barrier()],
    )(source, indices)


def test_wrong_host_index_extent[
    SourceRows: IntVar,
    Rows: IntVar,
    Cols: IntVar,
    Other: IntVar,
](
    source: jax.Array[[SourceRows, Cols]],
    wrong_indices: jax.Array[[Other]],
    rows: jax.Array[[Rows]],
    source_spec: pl.BlockSpec[[SourceRows, Cols], Literal["tma_gmem"]],
    index_spec: pl.BlockSpec[[Rows], Literal["tma_smem_indices"]],
    out_spec: plgpu.BlockSpec[Rows, Cols],
) -> None:
    def body(
        x_ref_gmem: plgpu.TmaGmemSourceRef[SourceRows, Cols],
        idx_ref: plgpu.TmaIndexRef[Rows],
        o_ref: plgpu.TmaSmemOutRef[Rows, Cols],
        barrier_ref: plgpu.TmaBarrier,
    ) -> None:
        kernel(x_ref_gmem, idx_ref, o_ref, barrier_ref)

    call = pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((rows.shape[0], source.shape[1]), source.dtype),
        in_specs=(source_spec, index_spec),
        out_specs=out_spec,
        scratch_shapes=[plgpu.Barrier()],
    )
    call(source, wrong_indices)  # E: is not assignable


if not TYPE_CHECKING:
    import unittest

    import numpy as np

    class BlackwellTmaGatherHostReferenceTest(unittest.TestCase):
        def test_host_only_independent_reference(self) -> None:
            source = jnp.arange(15, dtype=jnp.float32).reshape((5, 3))
            indices = jnp.array([2, 0, 2], dtype=jnp.int32)
            result = np.asarray(source)[np.asarray(indices)]
            np.testing.assert_array_equal(
                result, np.array([[6, 7, 8], [0, 1, 2], [6, 7, 8]], dtype=np.float32)
            )

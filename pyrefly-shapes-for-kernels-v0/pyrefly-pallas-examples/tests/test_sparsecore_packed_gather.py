# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Both kernel bodies from JAX docs/pallas/tpu/sparsecore.md (Apache-2.0).
# Only parameter annotations are added to the kernels.
# @lint-ignore-every AUTODEPS2

"""SparseCore packed gather links host row packing to bf16 VMEM pairs."""

from __future__ import annotations

from typing import assert_type, Literal, TYPE_CHECKING

import jax
import jax.lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar


def gather_packed_global[Batch: IntVar, Num: IntVar, Cols: IntVar, Window: IntVar](
    x: jax.BFloatArray[[Batch, Cols]],
    indices: jax.Array[[Num]],
    batch_size: Int[Batch],
    value_dim: Int[Cols],
    num_indices: Int[Num],
    gather_window_size: Int[Window],
    vector_mesh: object,
) -> jax.Array[[Num, Cols]]:
    packing: Literal[2] = 2  # Two 16-bit values fit in one 32-bit word.
    # Pack adjacent 16-bit rows (2*i, 2*i+1) into 32-bit int32 words on HBM
    x_packed = x.reshape(batch_size // packing, packing * value_dim).view(jnp.int32)

    @pl.kernel(
        out_type=jax.ShapeDtypeStruct((num_indices, value_dim), x.dtype),
        mesh=vector_mesh,
        scratch_types=dict(
            gather_vmem=pltpu.VMEM((gather_window_size, value_dim), jnp.int32)
        ),
    )
    def kernel(
        x_packed_hbm: pl.InRef[[Batch // 2, Cols]],
        i_hbm: pl.InRef[[Num]],
        o_hbm: pl.OutRef[[Num, Cols]],
        *,
        gather_vmem: pltpu.VmemScratchRef[[Window, Cols]],
    ) -> None:
        def body(
            idx_vmem: pl.Indices[Window],
            o_vmem: pl.OutRef[[Window, Cols]],
        ) -> None:
            # Issue indirect 32-bit DMA gather using halved index
            pltpu.sync_copy(
                x_packed_hbm.at[jax.lax.div(idx_vmem, packing)], gather_vmem
            )
            # Vectorized pair selection across the gathered window
            pairs = gather_vmem.view(jnp.bfloat16).reshape(-1, packing, value_dim)
            is_odd = (idx_vmem % 2)[:, None]
            o_vmem[...] = jnp.where(is_odd == 1, pairs[:, 1], pairs[:, 0])

        pltpu.emit_pipeline(
            body,
            grid=(num_indices // gather_window_size,),
            in_specs=[pl.BlockSpec((gather_window_size,), index_map=lambda i: (i,))],
            out_specs=[
                pl.BlockSpec(
                    (gather_window_size, value_dim), index_map=lambda i: (i, 0)
                )
            ],
            core_axis_name="subcore",
            dimension_semantics=(pltpu.PARALLEL,),
        )(i_hbm, o_hbm)

    return kernel(x_packed, indices)


def test_wrong_host_batch_extent[
    Batch: IntVar,
    Num: IntVar,
    Cols: IntVar,
    Window: IntVar,
    Other: IntVar,
](
    x: jax.BFloatArray[[Other, Cols]],
    indices: jax.Array[[Num]],
    batch: Int[Batch],
    cols: Int[Cols],
    num: Int[Num],
    window: Int[Window],
) -> None:
    gather_packed_global(
        x,
        indices,
        batch,  # E: is not assignable
        cols,
        num,
        window,
        None,
    )


def test_wrong_host_index_count[
    Batch: IntVar,
    Num: IntVar,
    Cols: IntVar,
    Window: IntVar,
    Other: IntVar,
](
    x: jax.BFloatArray[[Batch, Cols]],
    indices: jax.Array[[Other]],
    batch: Int[Batch],
    cols: Int[Cols],
    num: Int[Num],
    window: Int[Window],
) -> None:
    gather_packed_global(
        x,
        indices,
        batch,
        cols,
        num,  # E: is not assignable
        window,
        None,
    )


def test_wrong_host_dtype[Batch: IntVar, Num: IntVar, Cols: IntVar, Window: IntVar](
    x: jax.Array[[Batch, Cols]],
    indices: jax.Array[[Num]],
    batch: Int[Batch],
    cols: Int[Cols],
    num: Int[Num],
    window: Int[Window],
) -> None:
    gather_packed_global(
        x,  # E: is not assignable
        indices,
        batch,
        cols,
        num,
        window,
        None,
    )


def test_wrong_gather_vmem_width[
    PackedRows: IntVar,
    Window: IntVar,
    Cols: IntVar,
    Other: IntVar,
](
    source: pl.InRef[[PackedRows, Cols]],
    indices: pl.Indices[Window],
    vmem: pltpu.VmemScratchRef[[Window, Other]],
) -> None:
    pltpu.sync_copy(source.at[indices], vmem)  # E: No matching overload


def test_wrong_bf16_pair_reshape[Window: IntVar, Cols: IntVar, Other: IntVar](
    vmem: pltpu.VmemScratchRef[[Window, Cols]],
    other: Int[Other],
) -> None:
    vmem.view(jnp.bfloat16).reshape(-1, 2, other)  # E: is not assignable


def test_wrong_selected_pair_width[Window: IntVar, Cols: IntVar, Other: IntVar](
    mask: pl.IndexMask[Window],
    left: pl.Tile[[Window, Cols]],
    right: pl.Tile[[Window, Other]],
) -> None:
    jnp.where(mask, left, right)  # E: is not assignable


def test_wrong_output_allocation[
    PackedRows: IntVar,
    Num: IntVar,
    Cols: IntVar,
    Window: IntVar,
    Other: IntVar,
](
    kernel: pl.PackedGatherKernel[PackedRows, Num, Cols, Window],
    scratch: pltpu.VMEM[[Window, Cols]],
    num: Int[Num],
    other: Int[Other],
) -> None:
    pl.kernel(  # E: No matching overload
        out_type=jax.ShapeDtypeStruct((num, other), None),
        mesh=None,
        scratch_types={"gather_vmem": scratch},
    )(kernel)


def test_wrong_packed_input_rows[
    Batch: IntVar,
    Other: IntVar,
    Num: IntVar,
    Cols: IntVar,
    Window: IntVar,
](
    body: pl.PackedGatherKernel[Other, Num, Cols, Window],
    scratch: pltpu.VMEM[[Window, Cols]],
    packed: jax.Array[[Batch // 2, Cols]],
    indices: jax.Array[[Num]],
    num: Int[Num],
    cols: Int[Cols],
) -> None:
    kernel = pl.kernel(
        out_type=jax.ShapeDtypeStruct((num, cols), None),
        mesh=None,
        scratch_types={"gather_vmem": scratch},
    )(body)
    kernel(packed, indices)  # E: is not assignable


def test_wrong_scratch_descriptor_width[
    PackedRows: IntVar,
    Num: IntVar,
    Cols: IntVar,
    Window: IntVar,
    Other: IntVar,
](
    body: pl.PackedGatherKernel[PackedRows, Num, Cols, Window],
    scratch: pltpu.VMEM[[Window, Other]],
    num: Int[Num],
    cols: Int[Cols],
) -> None:
    pl.kernel(  # E: No matching overload
        out_type=jax.ShapeDtypeStruct((num, cols), None),
        mesh=None,
        scratch_types={"gather_vmem": scratch},
    )(body)


def test_wrong_scratch_dtype_is_not_rejected[
    PackedRows: IntVar,
    Num: IntVar,
    Cols: IntVar,
    Window: IntVar,
](
    body: pl.PackedGatherKernel[PackedRows, Num, Cols, Window],
    window: Int[Window],
    cols: Int[Cols],
    num: Int[Num],
) -> None:
    wrong: pltpu.VMEM[[Window, Cols]] = pltpu.VMEM((window, cols), jnp.bfloat16)
    pl.kernel(
        out_type=jax.ShapeDtypeStruct((num, cols), None),
        mesh=None,
        scratch_types={"gather_vmem": wrong},
    )(body)


def test_wrong_scratch_keyword_is_not_rejected[
    PackedRows: IntVar,
    Num: IntVar,
    Cols: IntVar,
    Window: IntVar,
](
    body: pl.PackedGatherKernel[PackedRows, Num, Cols, Window],
    scratch: pltpu.VMEM[[Window, Cols]],
    num: Int[Num],
    cols: Int[Cols],
) -> None:
    pl.kernel(
        out_type=jax.ShapeDtypeStruct((num, cols), None),
        mesh=None,
        scratch_types={"incorrect_keyword": scratch},
    )(body)


def test_output_shape[Batch: IntVar, Num: IntVar, Cols: IntVar, Window: IntVar](
    x: jax.BFloatArray[[Batch, Cols]],
    indices: jax.Array[[Num]],
    batch: Int[Batch],
    cols: Int[Cols],
    num: Int[Num],
    window: Int[Window],
) -> None:
    assert_type(
        gather_packed_global(x, indices, batch, cols, num, window, None),
        jax.Array[[Num, Cols]],
    )


def test_explicit_bfloat_conversion[
    Batch: IntVar,
    Num: IntVar,
    Cols: IntVar,
    Window: IntVar,
](
    x: jax.Array[[Batch, Cols]],
    indices: jax.Array[[Num]],
    batch: Int[Batch],
    cols: Int[Cols],
    num: Int[Num],
    window: Int[Window],
) -> None:
    assert_type(
        gather_packed_global(
            x.astype(jnp.bfloat16), indices, batch, cols, num, window, None
        ),
        jax.Array[[Num, Cols]],
    )


def test_odd_batch_extent_is_not_rejected[Num: IntVar, Window: IntVar](
    x: jax.BFloatArray[[5, 3]],
    indices: jax.Array[[Num]],
    num: Int[Num],
    window: Int[Window],
) -> None:
    assert_type(
        gather_packed_global(x, indices, 5, 3, num, window, None),
        jax.Array[[Num, 3]],
    )


if not TYPE_CHECKING:
    import unittest

    class SparsecorePackedGatherBoundaryTest(unittest.TestCase):
        def test_host_pack_and_unpack_pair_shapes(self) -> None:
            x = jnp.arange(12, dtype=jnp.bfloat16).reshape(4, 3)
            packed = x.reshape(2, 6).view(jnp.int32)
            self.assertEqual(packed.shape, (2, 3))
            pairs = packed.view(jnp.bfloat16).reshape(-1, 2, 3)
            self.assertEqual(pairs.shape, (2, 2, 3))
            indices = jnp.array([3, 0, 2])
            selected = pairs[indices // 2, indices % 2]
            self.assertTrue(jnp.array_equal(selected, x[indices]))

        def test_odd_row_count_cannot_be_packed(self) -> None:
            x = jnp.arange(15, dtype=jnp.bfloat16).reshape(5, 3)
            with self.assertRaises(TypeError):
                x.reshape(5 // 2, 2 * 3)

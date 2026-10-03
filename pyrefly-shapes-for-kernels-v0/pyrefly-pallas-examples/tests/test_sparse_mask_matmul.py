# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The kernel and four index-map bodies come from JAX docs/pallas/tpu/sparse.md
# (Apache-2.0); only kernel parameter annotations are new.
# @lint-ignore-every AUTODEPS2

"""Sparse output masks tie four prefetch maps to the output block grid."""

from __future__ import annotations

from collections.abc import Callable
from typing import assert_type, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar


def x_map(i, j, k, block_mask, prefetch_mask, prefetch_i, prefetch_j):
    del prefetch_mask, prefetch_j
    # Zero-out the k index if the mask is zero, to avoid constantly fetching
    # new blocks in the inner loop for blocks we are skipping.
    k_fetch = (block_mask[i, j] != 0) * k
    return (prefetch_i[i, j], k_fetch)


def y_map(i, j, k, block_mask, prefetch_mask, prefetch_i, prefetch_j):
    del prefetch_mask, prefetch_i
    k_fetch = (block_mask[i, j] != 0) * k
    return (k_fetch, prefetch_j[i, j])


def mask_map(i, j, k, block_mask, prefetch_mask, *_):
    del k, block_mask
    return (prefetch_mask[i, j], 0, 0)


def o_map(i, j, k, *_):
    del k
    return (i, j)


def masked_matmul[
    Rows: IntVar,
    Cols: IntVar,
    Inner: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    MaskTypes: IntVar,
](
    M: Int[Rows],
    N: Int[Cols],
    K: Int[Inner],
    blk_M: Int[BM],
    blk_N: Int[BN],
    blk_K: Int[BK],
    block_mask: jax.Array[[Rows // BM, Cols // BN]],
    prefetch_mask: jax.Array[[Rows // BM, Cols // BN]],
    prefetch_i: jax.Array[[Rows // BM, Cols // BN]],
    prefetch_j: jax.Array[[Rows // BM, Cols // BN]],
    X: jax.Array[[Rows, Inner]],
    Y: jax.Array[[Inner, Cols]],
    sparse_mask_data: jax.Array[[MaskTypes, BM, BN]],
) -> jax.Array[[Rows, Cols]]:
    def sparse_mask_matmul(
        block_mask_ref: pl.PrefetchRef[[Rows // BM, Cols // BN]],
        prefetch_mask: pl.PrefetchRef[[Rows // BM, Cols // BN]],
        prefetch_i: pl.PrefetchRef[[Rows // BM, Cols // BN]],
        prefetch_j: pl.PrefetchRef[[Rows // BM, Cols // BN]],
        x_ref: pl.InRef[[BM, BK]],
        y_ref: pl.InRef[[BK, BN]],
        mask_ref: pl.InRef[[1, BM, BN]],
        o_ref: pl.OutRef[[BM, BN]],
        accum_scratch: pltpu.VmemScratchRef[[BM, BN]],
    ) -> None:
        del prefetch_mask, prefetch_i, prefetch_j
        i, j, k = pl.program_id(0), pl.program_id(1), pl.program_id(2)
        should_compute = block_mask_ref[i, j] != 0

        @pl.when(k == 0)
        def _():
            o_ref[...] = jnp.zeros_like(o_ref)
            accum_scratch[...] = jnp.zeros_like(accum_scratch[...])

        # We only compute the output for blocks with non-zero masks.
        # Otherwise we skip the computation entirely.
        @pl.when(should_compute)
        def _():
            result = jnp.dot(x_ref[...], y_ref[...], preferred_element_type=jnp.float32)
            accum_scratch[...] += result

            @pl.when(k == pl.num_programs(2) - 1)
            def _():
                o_ref[...] = (mask_ref[0, ...] * accum_scratch[...]).astype(o_ref.dtype)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=4,
        grid=(M // blk_M, N // blk_N, K // blk_K),
        in_specs=[
            pl.BlockSpec((blk_M, blk_K), x_map),
            pl.BlockSpec((blk_K, blk_N), y_map),
            pl.BlockSpec((1, blk_M, blk_N), mask_map),
        ],
        out_specs=pl.BlockSpec((blk_M, blk_N), o_map),
        scratch_shapes=[pltpu.VMEM((blk_M, blk_N), dtype=jnp.float32)],
    )
    kernel = pl.pallas_call(
        sparse_mask_matmul,
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.bfloat16),
    )
    args = (block_mask, prefetch_mask, prefetch_i, prefetch_j, X, Y, sparse_mask_data)
    return kernel(*args)


def test_host_result[
    Rows: IntVar,
    Cols: IntVar,
    Inner: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Types: IntVar,
](
    block: jax.Array[[Rows // BM, Cols // BN]],
    pm: jax.Array[[Rows // BM, Cols // BN]],
    pi: jax.Array[[Rows // BM, Cols // BN]],
    pj: jax.Array[[Rows // BM, Cols // BN]],
    x: jax.Array[[Rows, Inner]],
    y: jax.Array[[Inner, Cols]],
    data: jax.Array[[Types, BM, BN]],
    rows: Int[Rows],
    cols: Int[Cols],
    inner: Int[Inner],
    bm: Int[BM],
    bn: Int[BN],
    bk: Int[BK],
) -> None:
    assert_type(
        masked_matmul(rows, cols, inner, bm, bn, bk, block, pm, pi, pj, x, y, data),
        jax.Array[[Rows, Cols]],
    )


def test_wrong_prefetch_i_extent[
    Rows: IntVar,
    Cols: IntVar,
    Inner: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Types: IntVar,
    Other: IntVar,
](
    block: jax.Array[[Rows // BM, Cols // BN]],
    pm: jax.Array[[Rows // BM, Cols // BN]],
    pi: jax.Array[[Other, Cols // BN]],
    pj: jax.Array[[Rows // BM, Cols // BN]],
    x: jax.Array[[Rows, Inner]],
    y: jax.Array[[Inner, Cols]],
    data: jax.Array[[Types, BM, BN]],
    rows: Int[Rows],
    cols: Int[Cols],
    inner: Int[Inner],
    bm: Int[BM],
    bn: Int[BN],
    bk: Int[BK],
) -> None:
    masked_matmul(
        rows,
        cols,
        inner,
        bm,
        bn,
        bk,
        block,
        pm,
        pi,  # E: is not assignable
        pj,
        x,
        y,
        data,
    )


def test_wrong_sparse_mask_tile_width[
    Rows: IntVar,
    Cols: IntVar,
    Inner: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Types: IntVar,
    Other: IntVar,
](
    block: jax.Array[[Rows // BM, Cols // BN]],
    pm: jax.Array[[Rows // BM, Cols // BN]],
    pi: jax.Array[[Rows // BM, Cols // BN]],
    pj: jax.Array[[Rows // BM, Cols // BN]],
    x: jax.Array[[Rows, Inner]],
    y: jax.Array[[Inner, Cols]],
    data: jax.Array[[Types, BM, Other]],
    rows: Int[Rows],
    cols: Int[Cols],
    inner: Int[Inner],
    bm: Int[BM],
    bn: Int[BN],
    bk: Int[BK],
) -> None:
    masked_matmul(
        rows,
        cols,
        inner,
        bm,
        bn,
        bk,
        block,
        pm,
        pi,
        pj,
        x,
        y,
        data,  # E: is not assignable
    )


def test_wrong_kernel_mask_grid_extent[
    Rows: IntVar,
    Cols: IntVar,
    Inner: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Types: IntVar,
    Other: IntVar,
](
    body: Callable[
        [
            pl.PrefetchRef[[Other, Cols // BN]],
            pl.PrefetchRef[[Other, Cols // BN]],
            pl.PrefetchRef[[Other, Cols // BN]],
            pl.PrefetchRef[[Other, Cols // BN]],
            pl.InRef[[BM, BK]],
            pl.InRef[[BK, BN]],
            pl.InRef[[1, BM, BN]],
            pl.OutRef[[BM, BN]],
            pltpu.VmemScratchRef[[BM, BN]],
        ],
        None,
    ],
    spec: pltpu.PrefetchScalarGridSpec[int, int, BM, BN, int, BK, int],
    output: jax.ShapeDtypeStruct[[Rows, Cols]],
    block: jax.Array[[Rows // BM, Cols // BN]],
    pm: jax.Array[[Other, Cols // BN]],
    pi: jax.Array[[Other, Cols // BN]],
    pj: jax.Array[[Other, Cols // BN]],
    x: jax.Array[[Rows, Inner]],
    y: jax.Array[[Inner, Cols]],
    data: jax.Array[[Types, BM, BN]],
) -> None:
    kernel = pl.pallas_call(body, grid_spec=spec, out_shape=output)
    kernel(block, pm, pi, pj, x, y, data)  # E: is not assignable


def test_wrong_matmul_inner[BM: IntVar, BN: IntVar, BK: IntVar, Other: IntVar](
    x: pl.InRef[[BM, BK]],
    y: pl.InRef[[Other, BN]],
) -> None:
    jnp.dot(x[...], y[...], preferred_element_type=jnp.float32)  # E: is not assignable


def test_wrong_mask_tile_width[BM: IntVar, BN: IntVar, Other: IntVar](
    mask: pl.InRef[[1, BM, Other]],
    accum: pltpu.VmemScratchRef[[BM, BN]],
) -> None:
    mask[0, ...] * accum[...]  # E: is not supported


if not TYPE_CHECKING:
    import unittest

    class SparseMaskMatmulHostTest(unittest.TestCase):
        def test_masked_output_shape_and_values(self) -> None:
            x = jnp.arange(16, dtype=jnp.float32).reshape(4, 4)
            y = jnp.ones((4, 4), dtype=jnp.float32)
            mask = jnp.tril(jnp.ones((4, 4), dtype=jnp.int32))
            mask_blocks = mask.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3)
            block_mask = jnp.any(mask_blocks, axis=(2, 3))
            self.assertEqual(block_mask.shape, (2, 2))
            self.assertTrue(
                jnp.array_equal(block_mask, jnp.array([[True, False], [True, True]]))
            )
            sparse_mask_data = jnp.stack((mask_blocks[0, 0], mask_blocks[1, 0]))
            self.assertEqual(sparse_mask_data.shape, (2, 2, 2))
            result = mask * (x @ y)
            self.assertEqual(result.shape, mask.shape)
            self.assertTrue(jnp.array_equal(result, jnp.tril(x @ y)))

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The kernel and index-map bodies come from JAX docs/pallas/tpu/sparse.md
# (Apache-2.0); only kernel parameter annotations are new.
# @lint-ignore-every AUTODEPS2

"""Sparse block matmul connects index arrays and blocks to its host boundary."""

from __future__ import annotations

from collections.abc import Callable
from typing import assert_type, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar


def x_map(j, blk_idx, blk_idxs_i, blk_idxs_k):
    del j, blk_idxs_i, blk_idxs_k
    return (blk_idx, 0, 0)


def y_map(j, blk_idx, blk_idxs_i, blk_idxs_k):
    del blk_idxs_i
    return (blk_idxs_k[blk_idx], j)


def o_map(j, blk_idx, blk_idxs_i, blk_idxs_k):
    del blk_idxs_k
    return (blk_idxs_i[blk_idx], j)


def sparse_dense_matmul[
    Blocks: IntVar,
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
](
    indices_i: jax.Array[[Blocks]],
    indices_k: jax.Array[[Blocks]],
    X_blocks: jax.Array[[Blocks, BM, BK]],
    Y: jax.Array[[K, N]],
    zeros: jax.Array[[M, N]],
    M: Int[M],
    N: Int[N],
    blk_M: Int[BM],
    blk_N: Int[BN],
    blk_K: Int[BK],
) -> jax.Array[[M, N]]:
    num_blocks = X_blocks.shape[0]

    def dsd_kernel(
        idxs_i_ref: pl.PrefetchRef[[Blocks]],
        idxs_k_ref: pl.PrefetchRef[[Blocks]],
        x_ref: pl.InRef[[1, BM, BK]],
        y_ref: pl.InRef[[BK, BN]],
        _: pl.InRef[[BM, BN]],
        o_ref: pl.OutRef[[BM, BN]],
        accum_scratch: pltpu.VmemScratchRef[[BM, BN]],
    ) -> None:
        """A DSD (Dense = Sparse @ Dense) matmul kernel."""
        del idxs_k_ref
        blk_idx = pl.program_id(1)
        is_start = blk_idx == 0
        changed_blocks = idxs_i_ref[blk_idx] != idxs_i_ref[jnp.maximum(blk_idx - 1, 0)]

        @pl.when(is_start | changed_blocks)
        def _():
            accum_scratch[...] = jnp.zeros_like(accum_scratch)

        accum_scratch[...] += jnp.dot(
            x_ref[0, :, :], y_ref[...], preferred_element_type=jnp.float32
        )

        next_block_change = (
            idxs_i_ref[blk_idx] != idxs_i_ref[jnp.minimum(blk_idx + 1, num_blocks)]
        )
        is_end = blk_idx == (num_blocks - 1)

        @pl.when(is_end | next_block_change)
        def _():
            o_ref[...] = accum_scratch[...].astype(o_ref.dtype)

    out_shape = jax.ShapeDtypeStruct((M, N), dtype=jnp.bfloat16)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        grid=(N // blk_N, num_blocks),
        in_specs=[
            pl.BlockSpec((1, blk_M, blk_K), x_map),
            pl.BlockSpec((blk_K, blk_N), y_map),
            pl.BlockSpec((blk_M, blk_N), o_map),
        ],
        out_specs=pl.BlockSpec((blk_M, blk_N), o_map),
        scratch_shapes=[pltpu.VMEM((blk_M, blk_N), dtype=jnp.float32)],
    )
    kernel = pl.pallas_call(
        dsd_kernel,
        grid_spec=grid_spec,
        out_shape=out_shape,
        input_output_aliases={4: 0},
    )
    return kernel(indices_i, indices_k, X_blocks, Y, zeros)


def test_host_result[
    Blocks: IntVar,
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
](
    i: jax.Array[[Blocks]],
    k: jax.Array[[Blocks]],
    x: jax.Array[[Blocks, BM, BK]],
    y: jax.Array[[K, N]],
    zeros: jax.Array[[M, N]],
    m: Int[M],
    n: Int[N],
    bm: Int[BM],
    bn: Int[BN],
    bk: Int[BK],
) -> None:
    assert_type(
        sparse_dense_matmul(i, k, x, y, zeros, m, n, bm, bn, bk), jax.Array[[M, N]]
    )


def test_wrong_block_count[
    Blocks: IntVar,
    Other: IntVar,
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
](
    i: jax.Array[[Other]],
    k: jax.Array[[Blocks]],
    x: jax.Array[[Blocks, BM, BK]],
    y: jax.Array[[K, N]],
    zeros: jax.Array[[M, N]],
    m: Int[M],
    n: Int[N],
    bm: Int[BM],
    bn: Int[BN],
    bk: Int[BK],
) -> None:
    sparse_dense_matmul(
        i,
        k,  # E: is not assignable
        x,  # E: is not assignable
        y,
        zeros,
        m,
        n,
        bm,
        bn,
        bk,
    )


def test_wrong_output_alias_extent[
    Blocks: IntVar,
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Other: IntVar,
](
    i: jax.Array[[Blocks]],
    k: jax.Array[[Blocks]],
    x: jax.Array[[Blocks, BM, BK]],
    y: jax.Array[[K, N]],
    zeros: jax.Array[[Other, N]],
    m: Int[M],
    n: Int[N],
    bm: Int[BM],
    bn: Int[BN],
    bk: Int[BK],
) -> None:
    sparse_dense_matmul(
        i,
        k,
        x,
        y,
        zeros,
        m,  # E: is not assignable
        n,
        bm,
        bn,
        bk,
    )


def test_wrong_rhs_output_width[
    Blocks: IntVar,
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    Other: IntVar,
](
    i: jax.Array[[Blocks]],
    k: jax.Array[[Blocks]],
    x: jax.Array[[Blocks, BM, BK]],
    y: jax.Array[[K, Other]],
    zeros: jax.Array[[M, N]],
    m: Int[M],
    n: Int[N],
    bm: Int[BM],
    bn: Int[BN],
    bk: Int[BK],
) -> None:
    sparse_dense_matmul(
        i,
        k,
        x,
        y,
        zeros,  # E: is not assignable
        m,
        n,  # E: is not assignable
        bm,
        bn,
        bk,
    )


def test_wrong_matmul_inner[BM: IntVar, BK: IntVar, BN: IntVar, Other: IntVar](
    x: pl.InRef[[1, BM, BK]],
    y: pl.InRef[[Other, BN]],
    accum: pltpu.VmemScratchRef[[BM, BN]],
) -> None:
    accum[...] += jnp.dot(
        x[0, :, :],
        y[...],  # E: is not assignable
        preferred_element_type=jnp.float32,
    )


def test_wrong_accumulator_width[BM: IntVar, BK: IntVar, BN: IntVar, Other: IntVar](
    x: pl.InRef[[1, BM, BK]],
    y: pl.InRef[[BK, BN]],
    accum: pltpu.VmemScratchRef[[BM, Other]],
) -> None:
    accum[...] += jnp.dot(  # E: is not supported
        x[0, :, :], y[...], preferred_element_type=jnp.float32
    )


def test_wrong_alias_mapping[
    Blocks: IntVar,
    M: IntVar,
    N: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
](
    body: Callable[
        [
            pl.PrefetchRef[[Blocks]],
            pl.PrefetchRef[[Blocks]],
            pl.InRef[[1, BM, BK]],
            pl.InRef[[BK, BN]],
            pl.InRef[[BM, BN]],
            pl.OutRef[[BM, BN]],
            pltpu.VmemScratchRef[[BM, BN]],
        ],
        None,
    ],
    spec: pltpu.PrefetchScalarGridSpec[1, 1, BM, BN, BK, BK, Blocks],
    output: jax.ShapeDtypeStruct[[M, N]],
) -> None:
    pl.pallas_call(  # E: No matching overload
        body, grid_spec=spec, out_shape=output, input_output_aliases={3: 0}
    )


if not TYPE_CHECKING:
    import unittest

    class SparseDsdMatmulHostTest(unittest.TestCase):
        def test_sparse_block_count_and_dense_output_shape(self) -> None:
            blocks = jnp.ones((3, 2, 2), dtype=jnp.float32)
            row_ids = jnp.array([0, 0, 1], dtype=jnp.int32)
            k_ids = jnp.array([0, 1, 0], dtype=jnp.int32)
            rhs = jnp.ones((4, 2), dtype=jnp.float32)
            zeros = jnp.zeros((4, 2), dtype=jnp.float32)
            self.assertEqual(blocks.shape[0], row_ids.shape[0])
            self.assertEqual(blocks.shape[0], k_ids.shape[0])
            self.assertEqual(rhs.shape[1], zeros.shape[1])
            dense = jnp.zeros((zeros.shape[0], rhs.shape[0]), dtype=rhs.dtype)
            for block, row, inner in zip(blocks, row_ids, k_ids):
                row, inner = int(row), int(inner)
                dense = dense.at[2 * row : 2 * row + 2, 2 * inner : 2 * inner + 2].set(
                    block
                )
            result = dense @ rhs
            self.assertEqual(result.shape, zeros.shape)
            self.assertTrue(
                jnp.array_equal(result, jnp.array([[4, 4], [4, 4], [2, 2], [2, 2]]))
            )

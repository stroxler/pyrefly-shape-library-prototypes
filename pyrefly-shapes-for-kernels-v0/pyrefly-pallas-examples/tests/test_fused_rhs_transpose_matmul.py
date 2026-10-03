# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The complete kernel is from JAX docs/pallas/tpu/matmul.md.
# Only semantic parameter annotations are added to its executable body.
# @lint-ignore-every AUTODEPS2

"""The logical RHS transpose produces a differently oriented Pallas Ref."""

from __future__ import annotations

from typing import assert_type, Literal, TYPE_CHECKING

import jax
import jax.lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar


def matmul_kernel[
    RB: IntVar,
    KB: IntVar,
    CB: IntVar,
    Steps: IntVar,
](
    x_ref: pl.InRef[[RB, KB]],
    y_ref: pl.InRef[[CB, KB]],
    z_ref: pl.OutRef[[RB, CB]],
    acc_ref: pl.AccumRef[[RB, CB]],
    *,
    nsteps: Int[Steps],
    transpose_rhs: Literal[True],
):
    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    # dot_general expects a data structure (contraction_dims, batch_dims),
    # where contraction_dims are the set of dimensions for LHS and RHS that will
    # be contracted (reduced) in the matmul; batch_dims, on the other hand, are
    # looped over. The remaining dimensions will be the input and output dimension
    # of the matmul.
    if transpose_rhs:
        dims = ((1,), (1,)), ((), ())
    else:
        dims = ((1,), (0,)), ((), ())  # E: This code is unreachable

    acc_ref[...] += jax.lax.dot_general(  # E: No matching overload
        x_ref[...],
        y_ref[...],
        dims,
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(2) == nsteps - 1)
    def _():
        z_ref[...] = acc_ref[...].astype(z_ref.dtype)


def fused_transposed_matmul[
    M: IntVar,
    K: IntVar,
    N: IntVar,
    BM: IntVar,
    BK: IntVar,
    BN: IntVar,
](
    x: jax.Array[[M, K]],
    logical_rhs: jax.Array[[K, N]],
    m: Int[M],
    k: Int[K],
    n: Int[N],
    bm: Int[BM],
    bk: Int[BK],
    bn: Int[BN],
) -> jax.Array[[M, N]]:
    physical_rhs = logical_rhs.swapaxes(0, 1)
    x_spec: pl.BlockSpec[[BM, BK]] = pl.BlockSpec(
        (bm, bk), lambda i, j, step: (i, step)
    )
    rhs_spec: pl.BlockSpec[[BN, BK]] = pl.BlockSpec(
        (bn, bk), lambda i, j, step: (j, step)
    )
    out_spec: pl.BlockSpec[[BM, BN]] = pl.BlockSpec((bm, bn), lambda i, j, step: (i, j))
    scratch: pltpu.VMEM[[BM, BN]] = pltpu.VMEM((bm, bn), jnp.float32)
    grid_spec: pltpu.PrefetchScalarGridSpec[M, N, BM, BN, K, BK, 1, Literal[True]] = (
        pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(m // bm, n // bn, k // bk),
            in_specs=(x_spec, rhs_spec),
            out_specs=out_spec,
            scratch_shapes=(scratch,),
        )
    )

    def kernel(
        lhs: pl.InRef[[BM, BK]],
        rhs: pl.InRef[[BN, BK]],
        out: pl.OutRef[[BM, BN]],
        acc: pl.AccumRef[[BM, BN]],
    ) -> None:
        matmul_kernel(lhs, rhs, out, acc, nsteps=k // bk, transpose_rhs=True)

    return pl.pallas_call(
        kernel,
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        interpret=True,
    )(x, physical_rhs)


def test_correct_interface[
    M: IntVar,
    K: IntVar,
    N: IntVar,
    BM: IntVar,
    BK: IntVar,
    BN: IntVar,
](
    x: jax.Array[[M, K]],
    y: jax.Array[[K, N]],
    m: Int[M],
    k: Int[K],
    n: Int[N],
    bm: Int[BM],
    bk: Int[BK],
    bn: Int[BN],
) -> None:
    assert_type(
        fused_transposed_matmul(x, y, m, k, n, bm, bk, bn),
        jax.Array[[M, N]],
    )


def test_original_post_swap_column_extent[K: IntVar, N: IntVar](
    logical_rhs: jax.Array[[K, N]],
) -> None:
    # The upstream wrapper's `y = y.swapaxes(0, 1); _, n = y.shape` gets K.
    y = logical_rhs.swapaxes(0, 1)
    _, n = y.shape
    assert_type(n, Int[K])
    assert_type(n, Int[N])  # E: assert_type


def test_wrong_host_contraction[
    M: IntVar,
    K: IntVar,
    N: IntVar,
    Other: IntVar,
    BM: IntVar,
    BK: IntVar,
    BN: IntVar,
](
    x: jax.Array[[M, K]],
    y: jax.Array[[Other, N]],
    m: Int[M],
    k: Int[K],
    n: Int[N],
    bm: Int[BM],
    bk: Int[BK],
    bn: Int[BN],
) -> None:
    fused_transposed_matmul(x, y, m, k, n, bm, bk, bn)  # E: is not assignable


def test_wrong_host_output_columns[
    M: IntVar,
    K: IntVar,
    N: IntVar,
    Other: IntVar,
    BM: IntVar,
    BK: IntVar,
    BN: IntVar,
](
    x: jax.Array[[M, K]],
    y: jax.Array[[K, N]],
    m: Int[M],
    k: Int[K],
    wrong_n: Int[Other],
    bm: Int[BM],
    bk: Int[BK],
    bn: Int[BN],
) -> None:
    fused_transposed_matmul(x, y, m, k, wrong_n, bm, bk, bn)  # E: is not assignable


def test_wrong_rhs_block_orientation[
    M: IntVar,
    K: IntVar,
    N: IntVar,
    BM: IntVar,
    BK: IntVar,
    BN: IntVar,
](
    m: Int[M],
    k: Int[K],
    n: Int[N],
    bm: Int[BM],
    bk: Int[BK],
    bn: Int[BN],
    lhs: pl.BlockSpec[[BM, BK]],
    wrong_rhs: pl.BlockSpec[[BK, BN]],
    output: pl.BlockSpec[[BM, BN]],
    scratch: pltpu.VMEM[[BM, BN]],
) -> None:
    pltpu.PrefetchScalarGridSpec[
        M, N, BM, BN, K, BK, 1, Literal[True]
    ](  # E: No matching overload
        num_scalar_prefetch=0,
        grid=(m // bm, n // bn, k // bk),
        in_specs=(lhs, wrong_rhs),
        out_specs=output,
        scratch_shapes=(scratch,),
    )


def test_incorrect_transposed_contraction[BM: IntVar, BK: IntVar, BN: IntVar](
    lhs: pl.Tile[[BM, BK]],
    rhs: pl.Tile[[BN, BK]],
) -> None:
    jax.lax.dot_general(  # E: No matching overload
        lhs, rhs, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
    )


def test_unswapped_rhs_at_launch[
    M: IntVar,
    K: IntVar,
    N: IntVar,
    BM: IntVar,
    BK: IntVar,
    BN: IntVar,
](
    x: jax.Array[[M, K]],
    logical_rhs: jax.Array[[K, N]],
    m: Int[M],
    n: Int[N],
    grid_spec: pltpu.PrefetchScalarGridSpec[M, N, BM, BN, K, BK, 1, Literal[True]],
) -> None:
    def kernel(
        lhs: pl.InRef[[BM, BK]],
        rhs: pl.InRef[[BN, BK]],
        out: pl.OutRef[[BM, BN]],
        acc: pl.AccumRef[[BM, BN]],
    ) -> None:
        matmul_kernel(lhs, rhs, out, acc, nsteps=1, transpose_rhs=True)

    call = pl.pallas_call(
        kernel, grid_spec=grid_spec, out_shape=jax.ShapeDtypeStruct((m, n), x.dtype)
    )
    call(x, logical_rhs)  # E: is not assignable


if not TYPE_CHECKING:
    import unittest

    import numpy as np

    class FusedRhsTransposeBoundaryTest(unittest.TestCase):
        def test_nonsquare_rhs_with_partial_reduction_grid(self) -> None:
            x = jnp.arange(24, dtype=jnp.float32).reshape((4, 6))
            rhs = jnp.arange(48, dtype=jnp.float32).reshape((6, 8))
            result = fused_transposed_matmul(x, rhs, 4, 6, 8, 2, 3, 4)
            self.assertEqual(result.shape, (4, 8))
            np.testing.assert_allclose(np.asarray(result), np.asarray(x @ rhs))

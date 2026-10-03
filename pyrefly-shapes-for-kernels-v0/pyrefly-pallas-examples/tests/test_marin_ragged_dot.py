# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The complete kernel body comes from Marin
# lib/haliax/src/haliax/nn/ragged_dot.py;
# only its parameter annotations are added.
# @lint-ignore-every AUTODEPS2

"""A ragged contraction masks partial reduction and output blocks."""

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import jax
import jax.lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar

    Rows = IntVar("Rows")
    Inner = IntVar("Inner")
    Columns = IntVar("Columns")
    RowBlock = IntVar("RowBlock")
    InnerBlock = IntVar("InnerBlock")
    ColumnBlock = IntVar("ColumnBlock")
    Groups = IntVar("Groups")


def _triton_ragged_dot_kernel(
    a_ref: pl.RaggedLhsRef[Rows, Inner],
    b_ref: pl.RaggedRhsRef[Inner, ColumnBlock],
    lo_ref: pl.RaggedBoundRef[Rows],
    hi_ref: pl.RaggedBoundRef[Rows],
    out_ref: pl.RaggedOutRef[Rows, Columns, ColumnBlock],
    *,
    block_m: Int[RowBlock],
    block_k: Int[InnerBlock],
    n: Int[Columns],
):
    """Pallas-Triton ragged dot kernel (no quantization)."""
    lo = lo_ref[()]
    hi = hi_ref[()]
    start_m = lo + pl.program_id(0) * block_m

    @pl.when(start_m < hi)
    def _compute():
        span_m = pl.ds(start_m, block_m)
        start_n = pl.program_id(1) * out_ref.shape[1]
        acc = jnp.zeros((block_m, out_ref.shape[1]), dtype=jnp.float32)
        k = a_ref.shape[1]

        def body(i: int, acc: pl.Tile[[RowBlock, ColumnBlock]]):
            start_k = i * block_k
            span_k = pl.ds(start_k, block_k)
            if k % block_k:
                contraction_mask = start_k + jnp.arange(block_k) < k
                a = plgpu.load(
                    a_ref.at[span_m, span_k], mask=contraction_mask[None, :], other=0.0
                )
                b = plgpu.load(
                    b_ref.at[span_k, pl.ds(0, b_ref.shape[1])],
                    mask=contraction_mask[:, None],
                    other=0.0,
                )
            else:
                a = plgpu.load(a_ref.at[span_m, span_k])
                b = plgpu.load(b_ref.at[span_k, pl.ds(0, b_ref.shape[1])])
            dtype = jnp.result_type(a, b)
            return acc + pl.dot(a.astype(dtype), b.astype(dtype))

        num_k_blocks = pl.cdiv(k, block_k)
        acc = jax.lax.fori_loop(0, num_k_blocks, body, acc)
        # Tokamax's BlockRef masks logical output edges; raw Pallas refs do not.
        store_mask = (start_m + jnp.arange(block_m) < hi)[:, None]
        if n % out_ref.shape[1]:
            store_mask &= (start_n + jnp.arange(out_ref.shape[1]) < n)[None, :]
        plgpu.store(
            out_ref.at[span_m, pl.ds(0, out_ref.shape[1])],
            acc.astype(out_ref.dtype),
            mask=store_mask,
        )


def grouped_ragged_dot[
    M: IntVar,
    K: IntVar,
    N: IntVar,
    G: IntVar,
    BM: IntVar,
    BK: IntVar,
    BN: IntVar,
](
    lhs: jax.Array[[M, K]],
    rhs: jax.Array[[G, K, N]],
    group_sizes: jax.Array[[G]],
    block_m: Int[BM],
    block_k: Int[BK],
    block_n: Int[BN],
) -> jax.Array[[M, N]]:
    boundaries = jnp.cumulative_sum(group_sizes, include_initial=True)

    def kernel(
        a_ref: pl.RaggedLhsRef[M, K],
        b_ref: pl.RaggedRhsRef[K, BN],
        lo_ref: pl.RaggedBoundRef[M],
        hi_ref: pl.RaggedBoundRef[M],
        out_ref: pl.RaggedOutRef[M, N, BN],
    ) -> None:
        _triton_ragged_dot_kernel(
            a_ref,
            b_ref,
            lo_ref,
            hi_ref,
            out_ref,
            block_m=block_m,
            block_k=block_k,
            n=rhs.shape[2],
        )

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((lhs.shape[0], rhs.shape[2]), lhs.dtype),
        in_specs=(
            pl.no_block_spec,
            pl.BlockSpec((None, lhs.shape[1], block_n), lambda _, j, e: (e, 0, j)),
            pl.BlockSpec((None,), lambda _, __, e: (e,)),
            pl.BlockSpec((None,), lambda _, __, e: (e,)),
        ),
        out_specs=pl.BlockSpec((lhs.shape[0], block_n), lambda _, j, __: (0, j)),
        grid=(
            pl.cdiv(lhs.shape[0], block_m),
            pl.cdiv(rhs.shape[2], block_n),
            rhs.shape[0],
        ),
        interpret=True,
    )(lhs, rhs, boundaries[:-1], boundaries[1:])


def test_correct_grouping[
    M: IntVar,
    K: IntVar,
    N: IntVar,
    G: IntVar,
    BM: IntVar,
    BK: IntVar,
    BN: IntVar,
](
    lhs: jax.Array[[M, K]],
    rhs: jax.Array[[G, K, N]],
    groups: jax.Array[[G]],
    block_m: Int[BM],
    block_k: Int[BK],
    block_n: Int[BN],
) -> None:
    assert_type(
        grouped_ragged_dot(lhs, rhs, groups, block_m, block_k, block_n),
        jax.Array[[M, N]],
    )


def test_wrong_contraction[
    M: IntVar,
    K: IntVar,
    Other: IntVar,
    N: IntVar,
    G: IntVar,
    BM: IntVar,
    BK: IntVar,
    BN: IntVar,
](
    lhs: jax.Array[[M, K]],
    rhs: jax.Array[[G, Other, N]],
    groups: jax.Array[[G]],
    block_m: Int[BM],
    block_k: Int[BK],
    block_n: Int[BN],
) -> None:
    grouped_ragged_dot(
        lhs,
        rhs,  # E: is not assignable
        groups,
        block_m,
        block_k,
        block_n,
    )


def test_wrong_group_count[
    M: IntVar,
    K: IntVar,
    N: IntVar,
    G: IntVar,
    Other: IntVar,
    BM: IntVar,
    BK: IntVar,
    BN: IntVar,
](
    lhs: jax.Array[[M, K]],
    rhs: jax.Array[[G, K, N]],
    groups: jax.Array[[Other]],
    block_m: Int[BM],
    block_k: Int[BK],
    block_n: Int[BN],
) -> None:
    grouped_ragged_dot(
        lhs,
        rhs,
        groups,  # E: is not assignable
        block_m,
        block_k,
        block_n,
    )


def test_wrong_output_allocation[
    M: IntVar,
    K: IntVar,
    N: IntVar,
    G: IntVar,
    BM: IntVar,
    BK: IntVar,
    BN: IntVar,
](
    lhs: jax.Array[[M, K]],
    rhs: jax.Array[[G, K, N]],
    block_m: Int[BM],
    block_k: Int[BK],
    block_n: Int[BN],
) -> None:
    def kernel(
        a: pl.RaggedLhsRef[M, K],
        b: pl.RaggedRhsRef[K, BN],
        lo: pl.RaggedBoundRef[M],
        hi: pl.RaggedBoundRef[M],
        out: pl.RaggedOutRef[M, N, BN],
    ) -> None:
        _triton_ragged_dot_kernel(
            a,
            b,
            lo,
            hi,
            out,
            block_m=block_m,
            block_k=block_k,
            n=rhs.shape[2],
        )

    pl.pallas_call(  # E: No matching overload
        kernel,
        out_shape=jax.ShapeDtypeStruct(
            (lhs.shape[0], rhs.shape[2] + 1),
            lhs.dtype,
        ),
        in_specs=(
            pl.no_block_spec,
            pl.BlockSpec((None, lhs.shape[1], block_n), lambda _, j, e: (e, 0, j)),
            pl.BlockSpec((None,), lambda _, __, e: (e,)),
            pl.BlockSpec((None,), lambda _, __, e: (e,)),
        ),
        out_specs=pl.BlockSpec((lhs.shape[0], block_n), lambda _, j, __: (0, j)),
        grid=(
            pl.cdiv(lhs.shape[0], block_m),
            pl.cdiv(rhs.shape[2], block_n),
            rhs.shape[0],
        ),
        interpret=True,
    )


def test_wrong_rhs_contraction_mask[
    K: IntVar,
    Other: IntVar,
    BK: IntVar,
    BN: IntVar,
](
    rhs: pl.RaggedRhsRef[K, BN],
    block_k: Int[BK],
    other_k: Int[Other],
) -> None:
    indices = jnp.arange(block_k)
    wrong_mask = (indices < other_k)[:, None]
    plgpu.load(  # E: No matching overload
        rhs.at[pl.ds(0, block_k), pl.ds(0, rhs.shape[1])],
        mask=wrong_mask,
        other=0.0,
    )


def test_wrong_lhs_contraction_mask[
    M: IntVar,
    K: IntVar,
    Other: IntVar,
    BM: IntVar,
    BK: IntVar,
](
    lhs: pl.RaggedLhsRef[M, K],
    block_m: Int[BM],
    block_k: Int[BK],
    other_k: Int[Other],
) -> None:
    wrong_mask = (jnp.arange(block_k) < other_k)[None, :]
    plgpu.load(  # E: No matching overload
        lhs.at[pl.ds(0, block_m), pl.ds(0, block_k)],
        mask=wrong_mask,
        other=0.0,
    )


def test_wrong_output_row_mask[
    M: IntVar,
    Other: IntVar,
    N: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    output: pl.RaggedOutRef[M, N, BN],
    hi_ref: pl.RaggedBoundRef[Other],
    block_m: Int[BM],
    values: pl.Tile[[BM, BN]],
) -> None:
    mask = (jnp.arange(block_m) < hi_ref[()])[:, None]
    plgpu.store(  # E: No matching overload
        output.at[pl.ds(0, block_m), pl.ds(0, output.shape[1])],
        values,
        mask=mask,
    )


def test_wrong_output_column_mask[
    M: IntVar,
    N: IntVar,
    BM: IntVar,
    BN: IntVar,
    Other: IntVar,
](
    output: pl.RaggedOutRef[M, N, BN],
    hi_ref: pl.RaggedBoundRef[M],
    block_m: Int[BM],
    wrong_block_n: Int[Other],
    n: Int[N],
    values: pl.Tile[[BM, BN]],
) -> None:
    mask = (jnp.arange(block_m) < hi_ref[()])[:, None]
    mask &= (jnp.arange(wrong_block_n) < n)[None, :]
    plgpu.store(  # E: No matching overload
        output.at[pl.ds(0, block_m), pl.ds(0, output.shape[1])],
        values,
        mask=mask,
    )


def test_wrong_dot_inner[
    BM: IntVar,
    BK: IntVar,
    BN: IntVar,
    Other: IntVar,
](
    a: pl.Tile[[BM, BK]],
    b: pl.Tile[[Other, BN]],
) -> None:
    pl.dot(a, b)  # E: is not assignable


def test_prefix_may_be_out_of_bounds[G: IntVar](groups: jax.Array[[G]]) -> None:
    # No proof that nonnegative counts sum to the input row count.
    boundaries = jnp.cumulative_sum(groups, include_initial=True)
    assert_type(boundaries[:-1], jax.Array[[G]])
    assert_type(boundaries[1:], jax.Array[[G]])


if not TYPE_CHECKING:
    import unittest

    import numpy as np

    class MarinRaggedDotBoundaryTest(unittest.TestCase):
        def test_group_prefix_and_host_reference(self) -> None:
            lhs = np.arange(20, dtype=np.float32).reshape(5, 4)
            rhs = np.stack([np.eye(4, dtype=np.float32) * scale for scale in (1, 2, 3)])
            groups = jnp.array([2, 1, 2], dtype=jnp.int32)
            boundaries = jnp.cumulative_sum(groups, include_initial=True)
            np.testing.assert_array_equal(boundaries, [0, 2, 3, 5])
            result = np.concatenate(
                [
                    lhs[int(boundaries[g]) : int(boundaries[g + 1])] @ rhs[g]
                    for g in range(3)
                ]
            )
            np.testing.assert_array_equal(
                result,
                lhs * np.array([1, 1, 2, 3, 3], dtype=np.float32)[:, None],
            )

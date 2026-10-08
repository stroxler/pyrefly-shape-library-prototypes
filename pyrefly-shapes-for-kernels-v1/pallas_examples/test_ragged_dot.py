"""Validate the grouped host boundary of Marin's Pallas ragged-dot kernel.

The kernel body is copied from Marin's lib/haliax/src/haliax/nn/ragged_dot.py;
only its semantic parameter annotations are added.
"""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any, assert_type, cast
from unittest.mock import patch

import jax
import jax.lax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from shape_extensions import Int, IntVar

from pallas_library.layout import checked_pallas_call, ragged_dot_layout

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
    """Check the dynamic group partition before binding it to kernel Refs."""
    if lhs.ndim != 2 or rhs.ndim != 3 or group_sizes.ndim != 1:
        raise ValueError("Ragged dot expects rank-2, rank-3, and rank-1 inputs")
    if lhs.shape[1] != rhs.shape[1] or rhs.shape[0] != group_sizes.shape[0]:
        raise ValueError("Ragged dot contraction and group dimensions must match")
    if lhs.dtype != rhs.dtype or lhs.dtype not in (jnp.float16, jnp.float32):
        raise ValueError("Ragged dot matrices must have matching floating dtypes")
    counts = np.asarray(group_sizes)
    if counts.dtype.kind not in "iu":
        raise ValueError("Ragged dot group sizes must be integers")
    if any(
        type(block) is not int or block <= 0 for block in (block_m, block_k, block_n)
    ):
        raise ValueError("Ragged dot block dimensions must be positive integers")
    if np.any(counts < 0) or int(np.sum(counts, dtype=np.int64)) != lhs.shape[0]:
        raise ValueError("Ragged dot group sizes must be nonnegative and sum to rows")
    if rhs.shape[2] % block_n:
        raise ValueError("Pallas CPU interpreter requires whole output-column blocks")
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

    layout = ragged_dot_layout(
        kernel,
        lhs=lhs,
        rhs=rhs,
        boundaries=boundaries,
        block_m=block_m,
        block_n=block_n,
    )
    return checked_pallas_call(layout, interpret=True)(
        lhs, rhs, boundaries[:-1], boundaries[1:]
    )


class RaggedDotTest(unittest.TestCase):
    """Check the data-dependent group partition and masked output shape."""

    def test_reject_wrong_partition(self) -> None:
        x = cast(Any, jnp).ones((5, 4), dtype=jnp.float32)
        y = cast(Any, jnp).ones((3, 4, 6), dtype=jnp.float32)
        for counts in ([2, 1, 1], [2, -1, 4]):
            with (
                self.subTest(counts=counts),
                self.assertRaisesRegex(ValueError, "sum to rows"),
            ):
                grouped_ragged_dot(x, y, cast(Any, jnp).array(counts), 2, 4, 4)

    def test_reject_contraction_or_group_shape(self) -> None:
        x = cast(Any, jnp).ones((5, 4), dtype=jnp.float32)
        y = cast(Any, jnp).ones((3, 4, 6), dtype=jnp.float32)
        with self.assertRaisesRegex(ValueError, "contraction and group"):
            grouped_ragged_dot(x, y[:, :3, :], cast(Any, jnp).array([2, 1, 2]), 2, 4, 4)
        with self.assertRaisesRegex(ValueError, "contraction and group"):
            grouped_ragged_dot(x, y, cast(Any, jnp).array([3, 2]), 2, 4, 4)
        with self.assertRaisesRegex(ValueError, "whole output-column blocks"):
            grouped_ragged_dot(x, y, cast(Any, jnp).array([2, 1, 2]), 2, 4, 4)

    def test_grouped_result(self) -> None:
        x = cast(Any, jnp.arange)(20, dtype=jnp.float32).reshape(5, 4)
        y = cast(Any, jnp).stack([cast(Any, jnp).eye(4) * i for i in (1, 2, 3)])
        counts = cast(Any, jnp).array([2, 1, 2], dtype=jnp.int32)
        # JAX 0.12 removed pl.dot; preserve the upstream kernel body here.
        with patch.object(pl, "dot", jnp.dot, create=True):
            out = grouped_ragged_dot(x, y, counts, 2, 4, 4)
        self.assertEqual(out.shape, (5, 4))
        np.testing.assert_allclose(
            np.asarray(out), np.asarray(x) * np.array([1, 1, 2, 3, 3])[:, None]
        )

    def test_partial_contraction(self) -> None:
        x = cast(Any, jnp.arange)(25, dtype=jnp.float32).reshape(5, 5)
        y = cast(Any, jnp).stack([cast(Any, jnp).eye(5)[:, :4] * i for i in (1, 2, 3)])
        counts = cast(Any, jnp).array([2, 1, 2], dtype=jnp.int32)
        with patch.object(pl, "dot", jnp.dot, create=True):
            result = grouped_ragged_dot(x, y, counts, 2, 4, 4)
        np.testing.assert_allclose(
            np.asarray(result),
            np.asarray(x)[:, :4] * np.array([1, 1, 2, 3, 3])[:, None],
        )


if TYPE_CHECKING:

    def check_shape[M: IntVar, K: IntVar, N: IntVar, G: IntVar, Other: IntVar](
        lhs: jax.Array[[M, K]],
        rhs: jax.Array[[G, K, N]],
        groups: jax.Array[[G]],
        wrong: jax.Array[[Other]],
    ) -> None:
        assert_type(grouped_ragged_dot(lhs, rhs, groups, 2, 4, 4), jax.Array[[M, N]])
        grouped_ragged_dot(
            lhs,
            rhs,
            wrong,  # pyrefly: ignore[bad-argument-type]
            2,
            4,
            4,
        )

    def check_reduction[Rows: IntVar, Inner: IntVar, Cols: IntVar, Other: IntVar](
        lhs: pl.Tile[[Rows, Inner]],
        rhs: pl.Tile[[Inner, Cols]],
        bad_rhs: pl.Tile[[Other, Cols]],
    ) -> None:
        assert_type(pl.dot(lhs, rhs), pl.Tile[[Rows, Cols]])
        pl.dot(lhs, bad_rhs)  # pyrefly: ignore[bad-argument-type]

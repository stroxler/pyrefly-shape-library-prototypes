# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The complete kernel comes from JAX
# jax/experimental/pallas/ops/gpu/attention.py (Apache-2.0).
# Only semantic parameter annotations are added; the GPU module is deprecated.
# @lint-ignore-every AUTODEPS2

"""Backward attention reduces a padded head Ref into a permuted delta."""

from __future__ import annotations

from collections.abc import Callable
from typing import assert_type, Literal, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar

    Batch = IntVar("Batch")
    Queries = IntVar("Queries")
    Heads = IntVar("Heads")
    HeadDim = IntVar("HeadDim")
    PaddedDim = IntVar("PaddedDim")
    QueryBlock = IntVar("QueryBlock")


def _preprocess_backward_kernel(
    out_ref: pl.MhaPreprocessRef[QueryBlock, PaddedDim, HeadDim],
    dout_ref: pl.MhaPreprocessRef[QueryBlock, PaddedDim, HeadDim],
    delta_ref: pl.OutRef[[QueryBlock]],
    head_dim: Int[HeadDim],
):
    # load
    head_mask = (jnp.arange(out_ref.shape[-1]) < head_dim)[None, :]
    o = plgpu.load(out_ref, mask=head_mask, other=0.0)
    do = plgpu.load(dout_ref, mask=head_mask, other=0.0)
    # compute
    delta = jnp.sum(o * do, axis=1)
    # write-back
    delta_ref[...] = delta.astype(delta_ref.dtype)


def mha_backward_delta[
    B: IntVar,
    Q: IntVar,
    H: IntVar,
    D: IntVar,
    P: IntVar,
    BQ: IntVar,
](
    out: jax.Array[[B, Q, H, D]],
    dout: jax.Array[[B, Q, H, D]],
    batches: Int[B],
    queries: Int[Q],
    heads: Int[H],
    head_dim: Int[D],
    padded_dim: Int[P],
    block_q: Int[BQ],
) -> jax.Array[[B, H, Q]]:
    if block_q <= 0 or queries % block_q:
        raise ValueError("Query length must be divisible by a positive block")
    if head_dim <= 0 or padded_dim < head_dim or padded_dim & (padded_dim - 1):
        raise ValueError("Padded head dimension must be a power of two covering D")
    out_spec: pl.BlockSpec[[BQ, P], Literal["mha_query"]] = pl.BlockSpec(
        (None, block_q, None, padded_dim), lambda i, j, h: (j, i, h, 0)
    )
    delta_spec: pl.BlockSpec[[BQ], Literal["mha_lse"]] = pl.BlockSpec(
        (None, None, block_q), lambda i, j, h: (j, h, i)
    )

    def kernel(
        out_ref: pl.MhaPreprocessRef[BQ, P, D],
        dout_ref: pl.MhaPreprocessRef[BQ, P, D],
        delta_ref: pl.OutRef[[BQ]],
    ) -> None:
        _preprocess_backward_kernel(out_ref, dout_ref, delta_ref, head_dim)

    return pl.pallas_call(
        kernel,
        in_specs=(out_spec, out_spec),
        out_specs=delta_spec,
        grid=(pl.cdiv(queries, block_q), batches, heads),
        out_shape=jax.ShapeDtypeStruct((batches, heads, queries), out.dtype),
        interpret=True,
    )(out, dout)


def test_host_result[B: IntVar, Q: IntVar, H: IntVar, D: IntVar, P: IntVar, BQ: IntVar](
    out: jax.Array[[B, Q, H, D]],
    dout: jax.Array[[B, Q, H, D]],
    batches: Int[B],
    queries: Int[Q],
    heads: Int[H],
    head_dim: Int[D],
    padded_dim: Int[P],
    block_q: Int[BQ],
) -> None:
    assert_type(
        mha_backward_delta(
            out, dout, batches, queries, heads, head_dim, padded_dim, block_q
        ),
        jax.Array[[B, H, Q]],
    )


def test_wrong_host_axes[
    B: IntVar,
    Q: IntVar,
    H: IntVar,
    D: IntVar,
    P: IntVar,
    BQ: IntVar,
    Other: IntVar,
](
    out: jax.Array[[B, Q, H, D]],
    wrong_queries: jax.Array[[B, Other, H, D]],
    wrong_heads: jax.Array[[B, Q, Other, D]],
    wrong_dim: jax.Array[[B, Q, H, Other]],
    batches: Int[B],
    queries: Int[Q],
    heads: Int[H],
    dim: Int[D],
    padded: Int[P],
    block_q: Int[BQ],
) -> None:
    mha_backward_delta(
        out,
        wrong_queries,  # E: is not assignable
        batches,
        queries,
        heads,
        dim,
        padded,
        block_q,
    )
    mha_backward_delta(
        out,
        wrong_heads,  # E: is not assignable
        batches,
        queries,
        heads,
        dim,
        padded,
        block_q,
    )
    mha_backward_delta(
        out,
        wrong_dim,  # E: is not assignable
        batches,
        queries,
        heads,
        dim,
        padded,
        block_q,
    )


def test_wrong_padded_ref[Q: IntVar, P: IntVar, D: IntVar, Other: IntVar](
    out: pl.MhaPreprocessRef[Q, P, D],
    dout: pl.MhaPreprocessRef[Q, Other, D],
    delta: pl.OutRef[[Q]],
    dim: Int[D],
) -> None:
    _preprocess_backward_kernel(out, dout, delta, dim)  # E: is not assignable


def test_wrong_head_mask[Q: IntVar, P: IntVar, D: IntVar, Other: IntVar](
    dout: pl.MhaPreprocessRef[Q, P, D], wrong_dim: Int[Other]
) -> None:
    mask = (jnp.arange(dout.shape[-1]) < wrong_dim)[None, :]
    plgpu.load(dout, mask=mask, other=0.0)  # E: No matching overload


def test_wrong_delta_reduction[Q: IntVar, P: IntVar](
    delta: pl.OutRef[[Q]], matrix: pl.Tile[[Q, P]]
) -> None:
    delta[...] = jnp.sum(matrix, axis=0).astype(  # E: Cannot set item
        delta.dtype
    )


def test_wrong_delta_ref[Q: IntVar, P: IntVar, D: IntVar, Other: IntVar](
    out: pl.MhaPreprocessRef[Q, P, D],
    dout: pl.MhaPreprocessRef[Q, P, D],
    wrong_delta: pl.OutRef[[Other]],
    dim: Int[D],
) -> None:
    _preprocess_backward_kernel(out, dout, wrong_delta, dim)  # E: is not assignable


def test_wrong_padded_spec[
    B: IntVar,
    Q: IntVar,
    H: IntVar,
    D: IntVar,
    P: IntVar,
    BQ: IntVar,
    Other: IntVar,
](
    kernel: Callable[
        [
            pl.MhaPreprocessRef[BQ, P, D],
            pl.MhaPreprocessRef[BQ, P, D],
            pl.OutRef[[BQ]],
        ],
        None,
    ],
    good_spec: pl.BlockSpec[[BQ, P], Literal["mha_query"]],
    wrong_spec: pl.BlockSpec[[BQ, Other], Literal["mha_query"]],
    delta_spec: pl.BlockSpec[[BQ], Literal["mha_lse"]],
    out: jax.Array[[B, Q, H, D]],
    dout: jax.Array[[B, Q, H, D]],
    batches: Int[B],
    queries: Int[Q],
    heads: Int[H],
    block_q: Int[BQ],
) -> None:
    pl.pallas_call(  # E: No matching overload
        kernel,
        in_specs=(wrong_spec, good_spec),
        out_specs=delta_spec,
        grid=(pl.cdiv(queries, block_q), batches, heads),
        out_shape=jax.ShapeDtypeStruct((batches, heads, queries), out.dtype),
        interpret=True,
    )(out, dout)


def test_wrong_delta_allocation[
    B: IntVar,
    Q: IntVar,
    H: IntVar,
    D: IntVar,
    P: IntVar,
    BQ: IntVar,
    Other: IntVar,
](
    kernel: Callable[
        [
            pl.MhaPreprocessRef[BQ, P, D],
            pl.MhaPreprocessRef[BQ, P, D],
            pl.OutRef[[BQ]],
        ],
        None,
    ],
    spec: pl.BlockSpec[[BQ, P], Literal["mha_query"]],
    delta_spec: pl.BlockSpec[[BQ], Literal["mha_lse"]],
    out: jax.Array[[B, Q, H, D]],
    dout: jax.Array[[B, Q, H, D]],
    batches: Int[B],
    queries: Int[Q],
    heads: Int[H],
    wrong: Int[Other],
    block_q: Int[BQ],
) -> None:
    pl.pallas_call(  # E: No matching overload
        kernel,
        in_specs=(spec, spec),
        out_specs=delta_spec,
        grid=(pl.cdiv(queries, block_q), batches, heads),
        out_shape=jax.ShapeDtypeStruct((batches, heads, wrong), out.dtype),
        interpret=True,
    )(out, dout)
    pl.pallas_call(  # E: No matching overload
        kernel,
        in_specs=(spec, spec),
        out_specs=delta_spec,
        grid=(pl.cdiv(queries, block_q), batches, heads),
        out_shape=jax.ShapeDtypeStruct((batches, wrong, queries), out.dtype),
        interpret=True,
    )(out, dout)


def test_wrong_delta_spec[
    B: IntVar,
    Q: IntVar,
    H: IntVar,
    D: IntVar,
    P: IntVar,
    BQ: IntVar,
    Other: IntVar,
](
    kernel: Callable[
        [
            pl.MhaPreprocessRef[BQ, P, D],
            pl.MhaPreprocessRef[BQ, P, D],
            pl.OutRef[[BQ]],
        ],
        None,
    ],
    spec: pl.BlockSpec[[BQ, P], Literal["mha_query"]],
    wrong_delta_spec: pl.BlockSpec[[Other], Literal["mha_lse"]],
    out: jax.Array[[B, Q, H, D]],
    dout: jax.Array[[B, Q, H, D]],
    batches: Int[B],
    queries: Int[Q],
    heads: Int[H],
    block_q: Int[BQ],
) -> None:
    pl.pallas_call(  # E: No matching overload
        kernel,
        in_specs=(spec, spec),
        out_specs=wrong_delta_spec,
        grid=(pl.cdiv(queries, block_q), batches, heads),
        out_shape=jax.ShapeDtypeStruct((batches, heads, queries), out.dtype),
        interpret=True,
    )(out, dout)


if not TYPE_CHECKING:
    import unittest

    import numpy as np

    class MhaBackwardPreprocessBoundaryTest(unittest.TestCase):
        def test_padded_head_dim_and_permuted_delta(self) -> None:
            out = jnp.arange(2 * 4 * 2 * 3, dtype=jnp.float32).reshape((2, 4, 2, 3)) / 7
            dout = jnp.flip(out, axis=-1) + 0.25
            delta = mha_backward_delta(
                out,
                dout,
                batches=2,
                queries=4,
                heads=2,
                head_dim=3,
                padded_dim=4,
                block_q=2,
            )

            self.assertEqual(delta.shape, (2, 2, 4))
            np.testing.assert_allclose(
                np.asarray(delta),
                np.einsum("bqhd,bqhd->bhq", np.asarray(out), np.asarray(dout)),
                rtol=1e-6,
                atol=1e-6,
            )

        def test_invalid_padded_head_dim(self) -> None:
            out = jnp.zeros((1, 2, 1, 3), dtype=jnp.float32)
            with self.assertRaisesRegex(ValueError, "Padded head dimension"):
                mha_backward_delta(out, out, 1, 2, 1, 3, 2, 2)

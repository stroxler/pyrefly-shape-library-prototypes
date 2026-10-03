# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Complete mha_forward_kernel and segment_mask from JAX
# jax/experimental/pallas/ops/gpu/attention.py (Apache-2.0).
# Only parameter/return annotations are replaced by semantic shape types.
# The upstream module is deprecated.
# @lint-ignore-every AUTODEPS2

"""Squeezed batch/head BlockSpecs connect GPU attention to host arrays."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import assert_type, Literal, TYPE_CHECKING

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar

    Batch = IntVar("Batch")
    Queries = IntVar("Queries")
    Keys = IntVar("Keys")
    Heads = IntVar("Heads")
    Dim = IntVar("Dim")
    QueryBlock = IntVar("QueryBlock")
    KeyBlock = IntVar("KeyBlock")
    DEFAULT_MASK_VALUE: float

    def segment_mask[Q: IntVar, K: IntVar](
        q_segment_ids: jax.Array[[Q]], kv_segment_ids: jax.Array[[K]]
    ) -> pl.MhaMask[Q, K]: ...


else:
    import numpy as np

    DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


def mha_forward_kernel(
    q_ref: pl.MhaQueryRef[QueryBlock, Dim],
    k_ref: pl.MhaKvRef[Keys, Dim],
    v_ref: pl.MhaKvRef[Keys, Dim],  # Input arrays
    segment_ids_ref: pl.MhaSegmentRef[Keys] | None,  # segment_id arrays
    o_ref: pl.MhaOutputRef[QueryBlock, Dim],  # Output
    *residual_refs: pl.OutRef[[QueryBlock]],  # Residual outputs
    sm_scale: float,
    causal: bool,
    block_q: Int[QueryBlock],
    block_k: Int[KeyBlock],
    head_dim: Int[Dim],
):
    seq_len = k_ref.shape[0]
    start_q = pl.program_id(0)
    head_dim_padded = q_ref.shape[-1]

    # o is the buffer where we accumulate the output on sram.
    # m_i and l_i (see FlashAttention paper) are updated during the k,v loop.
    m_i = jnp.zeros(block_q, dtype=jnp.float32) - float("inf")
    l_i = jnp.zeros(block_q, dtype=jnp.float32)
    # acc is the buffer where we accumulate the output on sram.
    o = jnp.zeros((block_q, head_dim_padded), dtype=jnp.float32)

    # Load q: it will stay in L1 throughout. Indices form a matrix because we
    # read, compute, and write all in 2d chunks. 1 element ~= 1 CUDA thread index.
    # q tile has shape [block_q, head_dim_padded], head_dim_padded >= head_dim.
    curr_q_slice = pl.dslice(start_q * block_q, block_q)
    head_mask = (jnp.arange(head_dim_padded) < head_dim)[None, :]
    q = plgpu.load(q_ref, mask=head_mask, other=0.0)
    q_segment_ids = None if segment_ids_ref is None else segment_ids_ref[curr_q_slice]

    # In FlashAttention algorithm 1 there are 2 loops: slow over tiles of kv (size
    # (Bc == block_k here), and fast over blocks of q (size Br == block_q here).
    # Here we only loop over blocks of kv to process entire seq_len, the loop over
    # blocks of q is carried out by the grid.
    def body(
        start_k: int,
        carry: tuple[
            pl.Tile[[QueryBlock, Dim]], pl.Tile[[QueryBlock]], pl.Tile[[QueryBlock]]
        ],
    ):
        o_prev, m_prev, l_prev = carry
        curr_k_slice = pl.dslice(start_k * block_k, block_k)

        k = plgpu.load(k_ref.at[curr_k_slice, :], mask=head_mask, other=0.0)
        qk = plgpu.dot(q, k.T)  # [block_q, block_k]

        # Scale logits to convert from base-2 to the natural log domain.
        # This is based on the identity: e^x = 2^(x * log2(e)).
        qk_scale = math.log2(math.e)
        if sm_scale != 1.0:
            qk_scale *= sm_scale
        qk *= qk_scale

        # Avoids Triton crash.
        # if num_heads > 2:
        #   qk = qk.astype(q_ref.dtype)
        #   qk = qk.astype(jnp.float32)

        if causal or segment_ids_ref is not None:
            mask = None
            if segment_ids_ref is not None:
                assert q_segment_ids is not None
                kv_segment_ids = segment_ids_ref[curr_k_slice]
                mask = segment_mask(q_segment_ids, kv_segment_ids)
            if causal:
                span_q = start_q * block_q + jnp.arange(block_q)
                span_k = start_k * block_k + jnp.arange(block_k)
                causal_mask = span_q[:, None] >= span_k[None, :]
                mask = (
                    causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
                )
            # Apply mask to qk.
            assert mask is not None
            qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

        m_curr = jnp.max(qk, axis=-1)
        m_next = jnp.maximum(m_prev, m_curr)
        correction = jnp.exp2(m_prev - m_next)
        l_prev_corr = correction * l_prev
        s_curr = jnp.exp2(
            qk - m_next[:, None]
        )  # Use m_next instead of m_curr to avoid a correction on l_curr
        l_curr = s_curr.sum(axis=-1)
        l_next = l_prev_corr + l_curr
        o_prev_corr = correction[:, None] * o_prev
        v = plgpu.load(v_ref.at[curr_k_slice, :], mask=head_mask)
        o_curr = plgpu.dot(s_curr.astype(v.dtype), v)

        o_next = o_prev_corr + o_curr
        return o_next, m_next, l_next

    if causal:
        # Ceildiv (`pl.cdiv` and `//` do not work due to type of start_q)
        upper_bound = lax.div(block_q * (start_q + 1) + block_k - 1, block_k)
    else:
        upper_bound = pl.cdiv(seq_len, block_k)
    o, m_i, l_i = lax.fori_loop(0, upper_bound, body, (o, m_i, l_i))

    # We keep an unscaled version of o during the scan over seq_len. Scaling it
    # by the last l_i gives us the correct final output. See section 3.1.1 of
    # FlashAttention-2 paper: https://arxiv.org/pdf/2307.08691.
    o /= l_i[:, None]

    if residual_refs:
        lse_ref = residual_refs[0]
        lse_ref[...] = m_i + jnp.log2(l_i)
    # Write output to dram.
    plgpu.store(o_ref.at[:, : o.shape[-1]], o.astype(o_ref.dtype), mask=head_mask)


if not TYPE_CHECKING:

    def segment_mask(  # noqa: F811 - runtime body replaces the trusted static signature
        q_segment_ids: jax.Array,
        kv_segment_ids: jax.Array,
    ):
        # [B, T, 1] or [T, 1]
        q_segment_ids = jnp.expand_dims(q_segment_ids, axis=-1)
        # [B, 1, S] or [1, S]
        if kv_segment_ids.ndim == 1:
            kv_segment_ids = jnp.expand_dims(kv_segment_ids, axis=0)
        else:
            kv_segment_ids = jnp.expand_dims(kv_segment_ids, axis=1)
        return jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)


def mha_host[
    B: IntVar,
    Q: IntVar,
    K: IntVar,
    H: IntVar,
    D: IntVar,
    BQ: IntVar,
    BK: IntVar,
](
    q: jax.Array[[B, Q, H, D]],
    k: jax.Array[[B, K, H, D]],
    v: jax.Array[[B, K, H, D]],
    segments: jax.Array[[B, K]] | None,
    batches: Int[B],
    queries: Int[Q],
    keys: Int[K],
    heads: Int[H],
    dim: Int[D],
    block_q: Int[BQ],
    block_k: Int[BK],
    causal: bool,
) -> tuple[jax.Array[[B, Q, H, D]], jax.Array[[B, H, Q]]]:
    if queries % block_q or keys % block_k:
        raise ValueError("Q and KV lengths must each be divisible by their blocks")
    if segments is not None and queries > keys:
        raise ValueError("Segment IDs must cover every query position")
    if dim <= 0 or dim & (dim - 1):
        raise ValueError("This adapter requires an already power-of-two head dim")
    query_spec: pl.BlockSpec[[BQ, D], Literal["mha_query"]] = pl.BlockSpec(
        (None, block_q, None, dim), lambda i, j, h: (j, i, h, 0)
    )
    kv_spec: pl.BlockSpec[[K, D], Literal["mha_kv"]] = pl.BlockSpec(
        (None, keys, None, dim), lambda _, j, h: (j, 0, h, 0)
    )
    segment_spec: pl.BlockSpec[[K], Literal["mha_segment"]] | None = (
        None if segments is None else pl.BlockSpec((None, keys), lambda _, j, h: (j, 0))
    )
    lse_spec: pl.BlockSpec[[BQ], Literal["mha_lse"]] = pl.BlockSpec(
        (None, None, block_q), lambda i, j, h: (j, h, i)
    )

    def kernel(
        q_ref: pl.MhaQueryRef[BQ, D],
        k_ref: pl.MhaKvRef[K, D],
        v_ref: pl.MhaKvRef[K, D],
        segment_ref: pl.MhaSegmentRef[K] | None,
        out_ref: pl.MhaOutputRef[BQ, D],
        lse_ref: pl.OutRef[[BQ]],
    ) -> None:
        mha_forward_kernel(
            q_ref,
            k_ref,
            v_ref,
            segment_ref,
            out_ref,
            lse_ref,
            sm_scale=0.5,
            causal=causal,
            block_q=block_q,
            block_k=block_k,
            head_dim=dim,
        )

    return pl.pallas_call(
        kernel,
        in_specs=(query_spec, kv_spec, kv_spec, segment_spec),
        out_specs=(query_spec, lse_spec),
        grid=(pl.cdiv(queries, block_q), batches, heads),
        out_shape=(
            jax.ShapeDtypeStruct((batches, queries, heads, dim), q.dtype),
            jax.ShapeDtypeStruct((batches, heads, queries), jnp.float32),
        ),
        interpret=True,
    )(q, k, v, segments)


def test_host_result[
    B: IntVar,
    Q: IntVar,
    K: IntVar,
    H: IntVar,
    D: IntVar,
    BQ: IntVar,
    BK: IntVar,
](
    q: jax.Array[[B, Q, H, D]],
    k: jax.Array[[B, K, H, D]],
    v: jax.Array[[B, K, H, D]],
    segments: jax.Array[[B, K]] | None,
    batch: Int[B],
    queries: Int[Q],
    keys: Int[K],
    heads: Int[H],
    dim: Int[D],
    bq: Int[BQ],
    bk: Int[BK],
) -> None:
    assert_type(
        mha_host(q, k, v, segments, batch, queries, keys, heads, dim, bq, bk, True),
        tuple[jax.Array[[B, Q, H, D]], jax.Array[[B, H, Q]]],
    )


def test_wrong_host_kv_axes[
    B: IntVar,
    Q: IntVar,
    K: IntVar,
    H: IntVar,
    D: IntVar,
    Other: IntVar,
    BQ: IntVar,
    BK: IntVar,
](
    q: jax.Array[[B, Q, H, D]],
    k: jax.Array[[B, K, H, Other]],
    v: jax.Array[[B, K, Other, D]],
    segments: jax.Array[[B, Other]],
    batch: Int[B],
    queries: Int[Q],
    keys: Int[K],
    heads: Int[H],
    dim: Int[D],
    bq: Int[BQ],
    bk: Int[BK],
) -> None:
    mha_host(
        q,
        k,  # E: is not assignable
        v,  # E: is not assignable
        segments,  # E: is not assignable
        batch,
        queries,
        keys,
        heads,
        dim,
        bq,
        bk,
        True,
    )


def test_wrong_qk_contract[BQ: IntVar, BK: IntVar, D: IntVar, Other: IntVar](
    q: pl.Tile[[BQ, D]], k: pl.Tile[[BK, Other]]
) -> None:
    plgpu.dot(q, k.T)  # E: is not assignable


def test_wrong_output_store[BQ: IntVar, D: IntVar, Other: IntVar](
    out: pl.MhaOutputRef[BQ, D],
    value: pl.Tile[[BQ, Other]],
    mask: pl.ContractionHorizontalMask[D, D],
    dim: Int[D],
) -> None:
    plgpu.store(out.at[:, :dim], value, mask=mask)  # E: No matching overload


def test_wrong_head_mask[BQ: IntVar, D: IntVar, Other: IntVar](
    q: pl.MhaQueryRef[BQ, D], mask: pl.ContractionHorizontalMask[D, Other]
) -> None:
    plgpu.load(q, mask=mask, other=0.0)  # E: No matching overload


def test_wrong_attention_mask[BQ: IntVar, BK: IntVar, Other: IntVar](
    score: pl.Tile[[BQ, BK]], mask: pl.MhaMask[Other, BK]
) -> None:
    jnp.where(mask, score, DEFAULT_MASK_VALUE)  # E: No matching overload


def test_wrong_squeezed_spec[
    B: IntVar,
    Q: IntVar,
    K: IntVar,
    H: IntVar,
    D: IntVar,
    BQ: IntVar,
    BK: IntVar,
](
    kernel: Callable[
        [
            pl.MhaQueryRef[BQ, D],
            pl.MhaKvRef[K, D],
            pl.MhaKvRef[K, D],
            pl.MhaSegmentRef[K] | None,
            pl.MhaOutputRef[BQ, D],
            pl.OutRef[[BQ]],
        ],
        None,
    ],
    query: pl.BlockSpec[[BQ, D], Literal["mha_query"]],
    wrong_query: pl.BlockSpec[[BQ, D]],
    kv: pl.BlockSpec[[K, D], Literal["mha_kv"]],
    lse: pl.BlockSpec[[BQ], Literal["mha_lse"]],
    batch: Int[B],
    queries: Int[Q],
    heads: Int[H],
    bq: Int[BQ],
    dim: Int[D],
) -> None:
    pl.pallas_call(  # E: No matching overload
        kernel,
        in_specs=(wrong_query, kv, kv, None),
        out_specs=(query, lse),
        grid=(pl.cdiv(queries, bq), batch, heads),
        out_shape=(
            jax.ShapeDtypeStruct((batch, queries, heads, dim), jnp.float32),
            jax.ShapeDtypeStruct((batch, heads, queries), jnp.float32),
        ),
    )


def test_wrong_residual_allocation[
    B: IntVar,
    Q: IntVar,
    K: IntVar,
    H: IntVar,
    D: IntVar,
    BQ: IntVar,
    Other: IntVar,
](
    kernel: Callable[
        [
            pl.MhaQueryRef[BQ, D],
            pl.MhaKvRef[K, D],
            pl.MhaKvRef[K, D],
            pl.MhaSegmentRef[K] | None,
            pl.MhaOutputRef[BQ, D],
            pl.OutRef[[BQ]],
        ],
        None,
    ],
    query: pl.BlockSpec[[BQ, D], Literal["mha_query"]],
    kv: pl.BlockSpec[[K, D], Literal["mha_kv"]],
    lse: pl.BlockSpec[[BQ], Literal["mha_lse"]],
    batch: Int[B],
    queries: Int[Q],
    other: Int[Other],
    heads: Int[H],
    bq: Int[BQ],
    dim: Int[D],
) -> None:
    pl.pallas_call(  # E: No matching overload
        kernel,
        in_specs=(query, kv, kv, None),
        out_specs=(query, lse),
        grid=(pl.cdiv(queries, bq), batch, heads),
        out_shape=(
            jax.ShapeDtypeStruct((batch, queries, heads, dim), jnp.float32),
            jax.ShapeDtypeStruct((batch, heads, other), jnp.float32),
        ),
    )
    pl.pallas_call(  # E: No matching overload
        kernel,
        in_specs=(query, kv, kv, None),
        out_specs=(query, lse),
        grid=(pl.cdiv(queries, bq), batch, heads),
        out_shape=(
            jax.ShapeDtypeStruct((batch, queries, heads, other), jnp.float32),
            jax.ShapeDtypeStruct((batch, heads, queries), jnp.float32),
        ),
    )


def test_bad_index_map_is_accepted[
    B: IntVar,
    Q: IntVar,
    K: IntVar,
    H: IntVar,
    D: IntVar,
    BQ: IntVar,
](
    kernel: Callable[
        [
            pl.MhaQueryRef[BQ, D],
            pl.MhaKvRef[K, D],
            pl.MhaKvRef[K, D],
            pl.MhaSegmentRef[K] | None,
            pl.MhaOutputRef[BQ, D],
            pl.OutRef[[BQ]],
        ],
        None,
    ],
    kv: pl.BlockSpec[[K, D], Literal["mha_kv"]],
    lse: pl.BlockSpec[[BQ], Literal["mha_lse"]],
    batch: Int[B],
    queries: Int[Q],
    heads: Int[H],
    bq: Int[BQ],
    dim: Int[D],
) -> None:
    # Rank-valid yet every query program reads block zero.
    bad_query: pl.BlockSpec[[BQ, D], Literal["mha_query"]] = pl.BlockSpec(
        (None, bq, None, dim), lambda i, j, h: (j, 0, h, 0)
    )
    pl.pallas_call(
        kernel,
        in_specs=(bad_query, kv, kv, None),
        out_specs=(bad_query, lse),
        grid=(pl.cdiv(queries, bq), batch, heads),
        out_shape=(
            jax.ShapeDtypeStruct((batch, queries, heads, dim), jnp.float32),
            jax.ShapeDtypeStruct((batch, heads, queries), jnp.float32),
        ),
    )


if not TYPE_CHECKING:
    import unittest

    class MhaForwardBoundaryTest(unittest.TestCase):
        def test_segment_ids_cover_query_positions(self) -> None:
            q = jnp.ones((1, 4, 1, 16), dtype=jnp.float16)
            kv = jnp.ones((1, 2, 1, 16), dtype=jnp.float16)
            segments = jnp.ones((1, 2), dtype=jnp.int32)
            with self.assertRaisesRegex(ValueError, "cover every query"):
                mha_host(q, kv, kv, segments, 1, 4, 2, 1, 16, 2, 2, True)

        def test_partial_kv_block_rejected_at_host(self) -> None:
            q = jnp.ones((1, 4, 1, 16), dtype=jnp.float16)
            kv = jnp.ones((1, 3, 1, 16), dtype=jnp.float16)
            with self.assertRaisesRegex(ValueError, "divisible"):
                mha_host(q, kv, kv, None, 1, 4, 3, 1, 16, 2, 2, True)

        def test_unsegmented_noncausal_attention(self) -> None:
            q = jnp.arange(2 * 1 * 16, dtype=jnp.float16).reshape(1, 2, 1, 16) / 13
            k = jnp.arange(4 * 1 * 16, dtype=jnp.float16).reshape(1, 4, 1, 16) / 17
            v = jnp.arange(4 * 1 * 16, dtype=jnp.float16).reshape(1, 4, 1, 16) / 31
            out, lse = mha_host(q, k, v, None, 1, 2, 4, 1, 16, 2, 2, False)
            scores = (
                np.asarray(q[0, :, 0], dtype=np.float32)
                @ np.asarray(k[0, :, 0], dtype=np.float32).T
                * 0.5
            )
            weights = np.exp(scores - scores.max(axis=1, keepdims=True))
            weights /= weights.sum(axis=1, keepdims=True)
            expected = weights @ np.asarray(v[0, :, 0], dtype=np.float32)
            expected_lse = np.log(
                np.exp(scores - scores.max(axis=1, keepdims=True)).sum(axis=1)
            ) + scores.max(axis=1)
            np.testing.assert_allclose(
                np.asarray(out[0, :, 0]), expected, rtol=2e-2, atol=2e-2
            )
            np.testing.assert_allclose(
                np.asarray(lse[0, 0]), expected_lse / np.log(2), rtol=2e-2, atol=2e-2
            )

        def test_segmented_causal_attention_and_lse(self) -> None:
            q = jnp.arange(2 * 4 * 2 * 16, dtype=jnp.float16).reshape(2, 4, 2, 16) / 19
            k = jnp.arange(2 * 4 * 2 * 16, dtype=jnp.float16).reshape(2, 4, 2, 16) / 37
            v = jnp.arange(2 * 4 * 2 * 16, dtype=jnp.float16).reshape(2, 4, 2, 16) / 29
            segments = jnp.array([[0, 0, 1, 1], [0, 0, 0, 1]], dtype=jnp.int32)
            out, lse = mha_host(q, k, v, segments, 2, 4, 4, 2, 16, 2, 2, True)
            expected = np.empty((2, 4, 2, 16), dtype=np.float32)
            expected_lse = np.empty((2, 2, 4), dtype=np.float32)
            for batch in range(2):
                for head in range(2):
                    for qi in range(4):
                        valid = [
                            ki
                            for ki in range(4)
                            if ki <= qi and segments[batch, qi] == segments[batch, ki]
                        ]
                        scores = np.array(
                            [
                                np.dot(
                                    np.asarray(q[batch, qi, head], dtype=np.float32),
                                    np.asarray(k[batch, ki, head], dtype=np.float32),
                                )
                                * 0.5
                                for ki in valid
                            ]
                        )
                        maximum = scores.max()
                        weights = np.exp(scores - maximum)
                        weights /= weights.sum()
                        expected[batch, qi, head] = sum(
                            weights[i] * np.asarray(v[batch, ki, head])
                            for i, ki in enumerate(valid)
                        )
                        expected_lse[batch, head, qi] = (
                            maximum + np.log(np.sum(np.exp(scores - maximum)))
                        ) / np.log(2)
            self.assertEqual(out.shape, (2, 4, 2, 16))
            self.assertEqual(lse.shape, (2, 2, 4))
            np.testing.assert_allclose(np.asarray(out), expected, rtol=2e-2, atol=2e-2)
            np.testing.assert_allclose(
                np.asarray(lse), expected_lse, rtol=2e-2, atol=2e-2
            )

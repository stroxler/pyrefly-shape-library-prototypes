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
import unittest
from typing import TYPE_CHECKING, Any, Literal, assert_type, cast

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from shape_extensions import Int, IntVar

from pallas_library.layout import attention_layout, checked_pallas_call

if TYPE_CHECKING:
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
    ) -> pl.Mask[[Q, K], [int, int]]: ...


else:
    import numpy as np

    DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


def mha_forward_kernel(
    q_ref: pl.ValidInRef[[QueryBlock, Dim], [QueryBlock, Dim]],
    k_ref: pl.InRef[[Keys, Dim]],
    v_ref: pl.InRef[[Keys, Dim]],  # Input arrays
    segment_ids_ref: pl.MhaSegmentRef[Keys] | None,  # segment_id arrays
    o_ref: pl.OutRef[[QueryBlock, Dim]],  # Output
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


def attention_forward[
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
    *,
    block_q: Int[BQ],
    block_k: Int[BK],
    sm_scale: float = 0.5,
) -> tuple[jax.Array[[B, Q, H, D]], jax.Array[[B, H, Q]]]:
    """Check host attention axes against the original noncausal Pallas kernel."""
    batch, queries, heads, dim = q.shape
    key_batch, keys, key_heads, key_dim = k.shape
    if (key_batch, key_heads, key_dim) != (batch, heads, dim):
        raise ValueError("Query and key/value axes must match")

    def kernel(
        q_ref: pl.ValidInRef[[BQ, D], [BQ, D]],
        k_ref: pl.InRef[[K, D]],
        v_ref: pl.InRef[[K, D]],
        out_ref: pl.OutRef[[BQ, D]],
        lse_ref: pl.OutRef[[BQ]],
    ) -> None:
        mha_forward_kernel(
            q_ref,
            k_ref,
            v_ref,
            None,
            out_ref,
            lse_ref,
            sm_scale=sm_scale,
            causal=False,
            block_q=block_q,
            block_k=block_k,
            head_dim=dim,
        )

    output = jax.ShapeDtypeStruct((batch, queries, heads, dim), q.dtype)
    stats = jax.ShapeDtypeStruct((batch, heads, queries), jnp.float32)
    layout = attention_layout(
        kernel,
        query_shape=q.shape,
        key_shape=k.shape,
        query_block=block_q,
        key_block=block_k,
        out_shape=(output, stats),
        grid=(pl.cdiv(queries, block_q), batch, heads),
    )
    return checked_pallas_call(layout, interpret=True)(q, k, v)


class AttentionForwardTest(unittest.TestCase):
    """Exercise the two-output host boundary and a multi-block key scan."""

    def test_noncausal_forward_and_lse(self) -> None:
        import numpy as np

        q = cast(Any, jnp).arange(32, dtype=jnp.float16).reshape(1, 2, 1, 16) / 13
        k = cast(Any, jnp).arange(64, dtype=jnp.float16).reshape(1, 4, 1, 16) / 17
        v = cast(Any, jnp).arange(64, dtype=jnp.float16).reshape(1, 4, 1, 16) / 31
        out, lse = attention_forward(q, k, v, block_q=2, block_k=2)
        scores = (
            np.asarray(q[0, :, 0], dtype=np.float32)
            @ np.asarray(k[0, :, 0], dtype=np.float32).T
            * 0.5
        )
        weights = np.exp(scores - scores.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(
            np.asarray(cast(Any, out)[0, :, 0]),
            weights @ np.asarray(v[0, :, 0], dtype=np.float32),
            rtol=2e-2,
            atol=2e-2,
        )
        expected_lse = np.log(
            np.exp(scores - scores.max(axis=1, keepdims=True)).sum(axis=1)
        ) + scores.max(axis=1)
        np.testing.assert_allclose(
            np.asarray(cast(Any, lse)[0, 0]),
            expected_lse / np.log(2),
            rtol=2e-2,
            atol=2e-2,
        )

    def test_reject_mismatched_values(self) -> None:
        q = cast(Any, jnp).ones((1, 2, 1, 16), dtype=jnp.float16)
        k = cast(Any, jnp).ones((1, 4, 1, 16), dtype=jnp.float16)
        v = cast(Any, jnp).ones((1, 3, 1, 16), dtype=jnp.float16)
        with self.assertRaisesRegex(ValueError, "Input shapes"):
            attention_forward(q, k, v, block_q=2, block_k=2)

    def test_reject_mismatched_grid_and_statistics_axes(self) -> None:
        def kernel(
            q_ref: pl.ValidInRef[[2, 16], [2, 16]],
            k_ref: pl.InRef[[4, 16]],
            v_ref: pl.InRef[[4, 16]],
            out_ref: pl.OutRef[[2, 16]],
            stats_ref: pl.OutRef[[2]],
        ) -> None:
            pass

        output = jax.ShapeDtypeStruct((1, 2, 1, 16), jnp.float16)
        stats = jax.ShapeDtypeStruct((1, 1, 2), jnp.float32)
        shape = (1, 2, 1, 16)
        key_shape = (1, 4, 1, 16)
        with self.assertRaisesRegex(ValueError, "grid must cover"):
            attention_layout(
                kernel,
                query_shape=shape,
                key_shape=key_shape,
                query_block=2,
                key_block=2,
                out_shape=(output, stats),
                grid=(2, 1, 1),
            )
        wrong_stats = cast(Any, jax.ShapeDtypeStruct((1, 2, 1), jnp.float32))
        with self.assertRaisesRegex(ValueError, "statistics shapes"):
            attention_layout(
                kernel,
                query_shape=shape,
                key_shape=key_shape,
                query_block=2,
                key_block=2,
                out_shape=(output, wrong_stats),
                grid=(1, 1, 1),
            )


if TYPE_CHECKING:

    def check_ref_feature_masks[
        Q: IntVar,
        K: IntVar,
        D: IntVar,
        Block: IntVar,
        Other: IntVar,
    ](
        query: pl.ValidInRef[[Q, D], [Q, D]],
        keys: pl.InRef[[K, D]],
        output: pl.OutRef[[Q, D]],
        values: pl.Tile[[Q, D]],
        key_block: Int[Block],
        feature_mask: pl.Mask[[1, D], [1, D]],
        wrong_bound: pl.Mask[[1, D], [1, Other]],
        wrong_axis: pl.Mask[[D, 1], [D, 1]],
    ) -> None:
        key_tile = keys.at[pl.dslice(0, key_block), :]
        output_tile = output.at[:, : output.shape[-1]]
        assert_type(
            key_tile,
            pl.TransformedRef[[K, D], [Block, D], Literal["in"]],
        )
        assert_type(
            output_tile,
            pl.TransformedRef[[Q, D], [Q, D], Literal["out"]],
        )
        plgpu.load(query, mask=feature_mask, other=0.0)
        plgpu.load(key_tile, mask=feature_mask)
        plgpu.store(output_tile, values, mask=feature_mask)
        plgpu.load(query, mask=wrong_bound, other=0.0)  # pyrefly: ignore[no-matching-overload]
        plgpu.load(key_tile, mask=wrong_bound)  # pyrefly: ignore[no-matching-overload]
        plgpu.load(key_tile, mask=wrong_axis)  # pyrefly: ignore[no-matching-overload]
        plgpu.store(output_tile, values, mask=wrong_bound)  # pyrefly: ignore[no-matching-overload]

    def typed_attention[
        B: IntVar,
        Q: IntVar,
        K: IntVar,
        H: IntVar,
        D: IntVar,
        Other: IntVar,
    ](
        q: jax.Array[[B, Q, H, D]],
        k: jax.Array[[B, K, H, D]],
        v: jax.Array[[B, K, H, D]],
        wrong: jax.Array[[B, K, H, Other]],
    ) -> None:
        assert_type(
            attention_forward(q, k, v, block_q=2, block_k=2),
            tuple[jax.Array[[B, Q, H, D]], jax.Array[[B, H, Q]]],
        )
        attention_forward(q, k, wrong, block_q=2, block_k=2)  # pyrefly: ignore[bad-argument-type]

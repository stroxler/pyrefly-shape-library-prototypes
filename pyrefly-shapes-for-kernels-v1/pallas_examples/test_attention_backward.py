# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license found in the root LICENSE file.
# Kernel body copied from JAX jax/experimental/pallas/ops/gpu/attention.py
# (Apache-2.0); only the parameter annotations are semantic types.

"""Check the complete two-scan Pallas attention-backward kernel."""

from __future__ import annotations

import math
import unittest
from typing import TYPE_CHECKING, Any, assert_type, cast

import jax
import jax.nn as jnn
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from jax.experimental.pallas.ops.gpu.attention import DEFAULT_MASK_VALUE, segment_mask
from shape_extensions import Int, IntVar

from pallas_library.layout import attention_backward_layout, checked_pallas_call

Batch = IntVar("Batch")
Queries = IntVar("Queries")
Keys = IntVar("Keys")
Heads = IntVar("Heads")
Dim = IntVar("Dim")
Padded = IntVar("Padded")
QueryBlockDkv = IntVar("QueryBlockDkv")
KeyBlockDkv = IntVar("KeyBlockDkv")
QueryBlockDq = IntVar("QueryBlockDq")
KeyBlockDq = IntVar("KeyBlockDq")


def mha_backward_kernel(
    # Inputs
    q_ref: pl.MhaBackwardMatrixRef[Queries, Padded, Dim],
    k_ref: pl.MhaBackwardMatrixRef[Keys, Padded, Dim],
    v_ref: pl.MhaBackwardMatrixRef[Keys, Padded, Dim],
    segment_ids_ref: pl.MhaSegmentRef[Keys] | None,
    out_ref: pl.MhaBackwardMatrixRef[Queries, Padded, Dim],
    do_scaled_ref: pl.MhaBackwardMatrixRef[Queries, Padded, Dim],
    lse_ref: pl.MhaBackwardVectorRef[Queries],
    delta_ref: pl.MhaBackwardVectorRef[Queries],
    # Outputs
    dq_ref: pl.MhaBackwardOutputRef[QueryBlockDq, Padded],
    dk_ref: pl.MhaBackwardOutputRef[KeyBlockDkv, Padded],
    dv_ref: pl.MhaBackwardOutputRef[KeyBlockDkv, Padded],
    *,
    sm_scale: float,
    causal: bool,
    block_q_dkv: Int[QueryBlockDkv],
    block_kv_dkv: Int[KeyBlockDkv],
    block_q_dq: Int[QueryBlockDq],
    block_kv_dq: Int[KeyBlockDq],
    head_dim: Int[Dim],
):
    del out_ref  # Not needed
    q_seq_len = q_ref.shape[0]
    kv_seq_len = k_ref.shape[0]

    # Scan #1: dK and dV
    #   1. Load a block of K and V of size (block_kv_dkv, head_dim) in SMEM.
    #   2. Iterate through Q in chunks of (block_q_dkv, head_dim) to accumulate
    #      dK and dV.
    start_k = pl.program_id(2)
    curr_k_slice = pl.dslice(start_k * block_kv_dkv, block_kv_dkv)

    head_dim_padded = q_ref.shape[-1]
    dv = jnp.zeros([block_kv_dkv, head_dim_padded], dtype=jnp.float32)
    dk = jnp.zeros([block_kv_dkv, head_dim_padded], dtype=jnp.float32)

    head_mask = (jnp.arange(head_dim_padded) < head_dim)[None, :]
    v = plgpu.load(v_ref.at[curr_k_slice, :], mask=head_mask, other=0.0)
    k = plgpu.load(k_ref.at[curr_k_slice, :], mask=head_mask, other=0.0)
    span_k = start_k * block_kv_dkv + jnp.arange(block_kv_dkv)
    kv_segment_ids = None if segment_ids_ref is None else segment_ids_ref[curr_k_slice]

    def inner_loop_dkdv(start_q, carry):
        dv, dk = carry
        curr_q_slice = pl.dslice(start_q * block_q_dkv, block_q_dkv)

        q = plgpu.load(q_ref.at[curr_q_slice, :], mask=head_mask, other=0.0)
        qk = plgpu.dot(q, k.T)
        qk_scale = math.log2(math.e)
        if sm_scale != 1.0:
            qk_scale *= sm_scale
        qk *= qk_scale

        if causal or segment_ids_ref is not None:
            mask = None
            if segment_ids_ref is not None:
                assert kv_segment_ids is not None
                q_segment_ids = segment_ids_ref[curr_q_slice]
                mask = segment_mask(q_segment_ids, kv_segment_ids)

            if causal:
                span_q = start_q * block_q_dkv + jnp.arange(block_q_dkv)
                causal_mask = span_q[:, None] >= span_k[None, :]
                mask = (
                    causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
                )
            assert mask is not None
            qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

        lse = lse_ref[curr_q_slice]
        di = delta_ref[curr_q_slice]
        do = plgpu.load(do_scaled_ref.at[curr_q_slice, :], mask=head_mask, other=0.0)

        p = jnp.exp2(qk - lse[:, None])
        dv = dv + plgpu.dot(p.astype(do.dtype).T, do)
        dp = jnp.zeros((block_q_dkv, block_kv_dkv), dtype=jnp.float32) - di[:, None]
        dp = dp + plgpu.dot(do, v.T)
        ds = p * dp
        if sm_scale != 1.0:
            ds = ds * sm_scale
        dk = dk + plgpu.dot(ds.astype(q_ref.dtype).T, q)

        return dv, dk

    lower_bound = lax.div(start_k * block_kv_dkv, block_q_dkv) if causal else 0
    dv, dk = lax.fori_loop(
        lower_bound, pl.cdiv(q_seq_len, block_q_dkv), inner_loop_dkdv, (dv, dk)
    )
    plgpu.store(
        dv_ref.at[:, : dv.shape[-1]],
        dv.astype(dv_ref.dtype),
        mask=head_mask,
    )
    plgpu.store(
        dk_ref.at[:, : dk.shape[-1]],
        dk.astype(dk_ref.dtype),
        mask=head_mask,
    )

    # Scan #2: dQ
    #   1. Load a block of Q of size (block_q_dq, head_dim) in SMEM.
    #   2. Iterate through K and V in chunks of (block_kv_dq, head_dim) to
    #     accumulate dQ.
    start_q = pl.program_id(2)
    curr_q_slice = pl.ds(start_q * block_q_dq, block_q_dq)
    span_q = start_q * block_q_dq + jnp.arange(block_q_dq)
    dq = jnp.zeros([block_q_dq, head_dim_padded], dtype=jnp.float32)

    q = plgpu.load(q_ref.at[curr_q_slice, :], mask=head_mask, other=0.0)
    q_segment_ids = None if segment_ids_ref is None else segment_ids_ref[curr_q_slice]
    lse = lse_ref[curr_q_slice]
    do = plgpu.load(do_scaled_ref.at[curr_q_slice, :], mask=head_mask, other=0.0)
    di = delta_ref[curr_q_slice]

    def inner_loop_dq(start_k, dq):
        curr_k_slice = pl.dslice(start_k * block_kv_dq, block_kv_dq)
        k = plgpu.load(k_ref.at[curr_k_slice, :], mask=head_mask, other=0.0)
        v = plgpu.load(v_ref.at[curr_k_slice, :], mask=head_mask, other=0.0)

        qk = plgpu.dot(q, k.T)
        qk_scale = math.log2(math.e)
        if sm_scale != 1.0:
            qk_scale *= sm_scale
        qk *= qk_scale

        if causal or segment_ids_ref is not None:
            mask = None
            if segment_ids_ref is not None:
                assert q_segment_ids is not None
                kv_segment_ids = segment_ids_ref[curr_k_slice]
                mask = segment_mask(q_segment_ids, kv_segment_ids)

            if causal:
                span_k = start_k * block_kv_dq + jnp.arange(block_kv_dq)
                causal_mask = span_q[:, None] >= span_k[None, :]
                mask = (
                    causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
                )
            assert mask is not None
            qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

        p = jnp.exp2(qk - lse[:, None])
        dp = jnp.zeros((block_q_dq, block_kv_dq), dtype=jnp.float32) - di[:, None]
        dp = dp + plgpu.dot(do, v.T)
        ds = p * dp
        if sm_scale != 1.0:
            ds = ds * sm_scale

        dq = dq + plgpu.dot(ds.astype(k.dtype), k).astype(dq.dtype)

        return dq

    if causal:
        upper_bound = pl.cdiv((start_q + 1) * block_q_dq, block_kv_dq)
    else:
        upper_bound = pl.cdiv(kv_seq_len, block_kv_dq)

    dq = lax.fori_loop(0, upper_bound, inner_loop_dq, (dq))
    plgpu.store(dq_ref.at[:, : dq.shape[-1]], dq.astype(dq_ref.dtype), mask=head_mask)


def attention_backward[
    B: IntVar,
    Q: IntVar,
    K: IntVar,
    H: IntVar,
    D: IntVar,
    P: IntVar,
    BQDKV: IntVar,
    BKVDKV: IntVar,
    BQDQ: IntVar,
    BKVDQ: IntVar,
](
    q: jax.Array[[B, Q, H, D]],
    k: jax.Array[[B, K, H, D]],
    v: jax.Array[[B, K, H, D]],
    segments: jax.Array[[B, K]] | None,
    out: jax.Array[[B, Q, H, D]],
    dout: jax.Array[[B, Q, H, D]],
    lse: jax.Array[[B, H, Q]],
    delta: jax.Array[[B, H, Q]],
    *,
    padded_dim: Int[P],
    block_q_dkv: Int[BQDKV],
    block_kv_dkv: Int[BKVDKV],
    block_q_dq: Int[BQDQ],
    block_kv_dq: Int[BKVDQ],
    causal: bool = False,
) -> tuple[
    jax.Array[[B, Q, H, D]],
    jax.Array[[B, K, H, D]],
    jax.Array[[B, K, H, D]],
]:
    """Check both scan schedules, all host axes, and three gradient outputs."""
    batch, queries, heads, dim = q.shape
    keys = k.shape[1]
    if (causal or segments is not None) and queries > keys:
        raise ValueError("Key and segment coverage must include every query")

    def kernel(
        q_ref: pl.MhaBackwardMatrixRef[Q, P, D],
        k_ref: pl.MhaBackwardMatrixRef[K, P, D],
        v_ref: pl.MhaBackwardMatrixRef[K, P, D],
        segment_ref: pl.MhaSegmentRef[K] | None,
        out_ref: pl.MhaBackwardMatrixRef[Q, P, D],
        do_ref: pl.MhaBackwardMatrixRef[Q, P, D],
        lse_ref: pl.MhaBackwardVectorRef[Q],
        delta_ref: pl.MhaBackwardVectorRef[Q],
        dq_ref: pl.MhaBackwardOutputRef[BQDQ, P],
        dk_ref: pl.MhaBackwardOutputRef[BKVDKV, P],
        dv_ref: pl.MhaBackwardOutputRef[BKVDKV, P],
    ) -> None:
        mha_backward_kernel(
            q_ref,
            k_ref,
            v_ref,
            segment_ref,
            out_ref,
            do_ref,
            lse_ref,
            delta_ref,
            dq_ref,
            dk_ref,
            dv_ref,
            sm_scale=1.0,
            causal=causal,
            block_q_dkv=block_q_dkv,
            block_kv_dkv=block_kv_dkv,
            block_q_dq=block_q_dq,
            block_kv_dq=block_kv_dq,
            head_dim=dim,
        )

    layout = attention_backward_layout(
        kernel,
        query_shape=q.shape,
        key_shape=k.shape,
        padded_dim=padded_dim,
        block_q_dkv=block_q_dkv,
        block_kv_dkv=block_kv_dkv,
        block_q_dq=block_q_dq,
        block_kv_dq=block_kv_dq,
        has_segments=segments is not None,
        out_shape=(
            jax.ShapeDtypeStruct((batch, queries, heads, dim), q.dtype),
            jax.ShapeDtypeStruct((batch, keys, heads, dim), q.dtype),
            jax.ShapeDtypeStruct((batch, keys, heads, dim), q.dtype),
        ),
    )
    return checked_pallas_call(layout, interpret=True)(
        q, k, v, segments, out, dout, lse, delta
    )


class AttentionBackwardTest(unittest.TestCase):
    """Check both scan schedules and the optional segment input on CPU."""

    def test_two_scan_gradients(self) -> None:
        import numpy as np
        from jax.experimental.pallas.ops.gpu.attention import BlockSizes, mha

        q = (
            cast(Any, jnp)
            .linspace(-0.75, 0.6, 32 * 16, dtype=jnp.float16)
            .reshape(1, 32, 1, 16)
        )
        k = (
            cast(Any, jnp)
            .linspace(-0.5, 0.8, 64 * 16, dtype=jnp.float16)
            .reshape(1, 64, 1, 16)
        )
        v = cast(Any, jnp).flip(k, axis=-1) * 0.75
        dout = cast(Any, jnp).flip(q, axis=1) + 0.125
        blocks = BlockSizes(
            block_q=16,
            block_k=16,
            block_q_dkv=16,
            block_kv_dkv=32,
            block_q_dq=16,
            block_kv_dq=16,
        )
        out, lse = mha(
            q,
            k,
            v,
            segment_ids=None,
            block_sizes=blocks,
            interpret=True,
            return_residuals=True,
        )
        delta = (
            (out.astype(jnp.float32) * dout.astype(jnp.float32))
            .sum(axis=-1)
            .transpose((0, 2, 1))
        )
        actual = attention_backward(
            q,
            k,
            v,
            None,
            out,
            dout,
            lse,
            delta,
            padded_dim=16,
            block_q_dkv=16,
            block_kv_dkv=32,
            block_q_dq=16,
            block_kv_dq=16,
        )

        def reference(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
            logits = cast(Any, jnp).einsum(
                "bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32
            )
            weights = jnn.softmax(logits, axis=-1)
            return (
                cast(Any, jnp)
                .einsum(
                    "bhqk,bkhd->bqhd",
                    weights,
                    v,
                    preferred_element_type=jnp.float32,
                )
                .astype(q.dtype)
            )

        expected = cast(Any, jax).grad(
            lambda q, k, v: jnp.sum(reference(q, k, v) * dout),
            argnums=(0, 1, 2),
        )(q, k, v)
        for got, want in zip(actual, expected, strict=True):
            self.assertEqual(got.shape, want.shape)
            np.testing.assert_allclose(
                np.asarray(got), np.asarray(want), rtol=3e-2, atol=3e-2
            )

    def test_causal_segmented_gradients(self) -> None:
        import numpy as np
        from jax.experimental.pallas.ops.gpu.attention import BlockSizes, mha

        q = (
            cast(Any, jnp)
            .linspace(-0.3, 0.6, 32 * 16, dtype=jnp.float16)
            .reshape(1, 32, 1, 16)
        )
        k = cast(Any, jnp).flip(q, axis=-1) * 0.5
        v = cast(Any, jnp).cos(q.astype(jnp.float32)).astype(jnp.float16)
        dout = cast(Any, jnp).flip(q, axis=1) + 0.2
        segments = cast(Any, jnp).array([[0] * 16 + [1] * 16], dtype=jnp.int32)
        blocks = BlockSizes(
            block_q=16,
            block_k=16,
            block_q_dkv=16,
            block_kv_dkv=16,
            block_q_dq=16,
            block_kv_dq=16,
        )
        out, lse = mha(
            q,
            k,
            v,
            segment_ids=segments,
            block_sizes=blocks,
            causal=True,
            interpret=True,
            return_residuals=True,
        )
        delta = (
            (out.astype(jnp.float32) * dout.astype(jnp.float32))
            .sum(axis=-1)
            .transpose((0, 2, 1))
        )
        actual = attention_backward(
            q,
            k,
            v,
            segments,
            out,
            dout,
            lse,
            delta,
            padded_dim=16,
            block_q_dkv=16,
            block_kv_dkv=16,
            block_q_dq=16,
            block_kv_dq=16,
            causal=True,
        )

        def reference(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
            scores = cast(Any, jnp).einsum(
                "bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32
            )
            matching = segments[:, :, None] == segments[:, None, :]
            causal = jnp.arange(32)[:, None] >= jnp.arange(32)[None, :]
            scores = jnp.where(
                matching[:, None, :, :] & cast(Any, causal)[None, None, :, :],
                scores,
                -1e10,
            )
            return (
                cast(Any, jnp)
                .einsum(
                    "bhqk,bkhd->bqhd",
                    cast(Any, jnn).softmax(scores, axis=-1),
                    v,
                    preferred_element_type=jnp.float32,
                )
                .astype(q.dtype)
            )

        expected = cast(Any, jax).grad(
            lambda q, k, v: jnp.sum(reference(q, k, v) * dout),
            argnums=(0, 1, 2),
        )(q, k, v)
        for got, want in zip(actual, expected, strict=True):
            np.testing.assert_allclose(
                np.asarray(got), np.asarray(want), rtol=3e-2, atol=3e-2
            )

    def test_reject_mismatched_axes_and_grid(self) -> None:
        q = cast(Any, jnp).ones((1, 32, 1, 16), dtype=jnp.float16)
        k = cast(Any, jnp).ones((1, 64, 1, 16), dtype=jnp.float16)
        stats = cast(Any, jnp).ones((1, 1, 32), dtype=jnp.float32)
        with self.assertRaisesRegex(ValueError, "same grid"):
            attention_backward(
                q,
                k,
                k,
                None,
                q,
                q,
                stats,
                stats,
                padded_dim=16,
                block_q_dkv=16,
                block_kv_dkv=32,
                block_q_dq=32,
                block_kv_dq=16,
            )
        with self.assertRaisesRegex(ValueError, "Input shapes"):
            attention_backward(
                q,
                k,
                k[:, :32],
                None,
                q,
                q,
                stats,
                stats,
                padded_dim=16,
                block_q_dkv=16,
                block_kv_dkv=32,
                block_q_dq=16,
                block_kv_dq=16,
            )


if TYPE_CHECKING:

    def typed_matrix_initializer[Rows: IntVar, Cols: IntVar](
        rows: Int[Rows], cols: Int[Cols]
    ) -> None:
        assert_type(
            jnp.zeros([rows, cols], dtype=jnp.float32),
            pl.Tile[[Rows, Cols]],
        )

    def typed_host_boundary[
        B: IntVar, Q: IntVar, K: IntVar, H: IntVar, D: IntVar, Other: IntVar
    ](
        q: jax.Array[[B, Q, H, D]],
        k: jax.Array[[B, K, H, D]],
        stats: jax.Array[[B, H, Q]],
        wrong: jax.Array[[B, K, H, Other]],
    ) -> None:
        assert_type(
            attention_backward(
                q,
                k,
                k,
                None,
                q,
                q,
                stats,
                stats,
                padded_dim=16,
                block_q_dkv=16,
                block_kv_dkv=16,
                block_q_dq=16,
                block_kv_dq=16,
            ),
            tuple[
                jax.Array[[B, Q, H, D]],
                jax.Array[[B, K, H, D]],
                jax.Array[[B, K, H, D]],
            ],
        )
        attention_backward(
            q,
            k,
            wrong,  # pyrefly: ignore[bad-argument-type]
            None,
            q,
            q,
            stats,
            stats,
            padded_dim=16,
            block_q_dkv=16,
            block_kv_dkv=16,
            block_q_dq=16,
            block_kv_dq=16,
        )

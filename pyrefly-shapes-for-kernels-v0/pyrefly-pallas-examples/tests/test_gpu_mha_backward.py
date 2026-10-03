# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The full kernel is from JAX jax/experimental/pallas/ops/gpu/attention.py
# (Apache-2.0); only parameter annotations are added.
# @lint-ignore-every AUTODEPS2

"""Fused attention backward maps two scan schedules into three gradients."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import assert_type, Literal, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar

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
    DEFAULT_MASK_VALUE: float

    def segment_mask[Q: IntVar, K: IntVar](
        q_segment_ids: jax.Array[[Q]], kv_segment_ids: jax.Array[[K]]
    ) -> pl.MhaMask[Q, K]: ...

else:
    from jax.experimental.pallas.ops.gpu.attention import (
        DEFAULT_MASK_VALUE,
        segment_mask,
    )


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
    dv = jnp.zeros(  # E: No matching overload
        [block_kv_dkv, head_dim_padded], dtype=jnp.float32
    )
    dk = jnp.zeros(  # E: No matching overload
        [block_kv_dkv, head_dim_padded], dtype=jnp.float32
    )

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
    plgpu.store(  # E: No matching overload
        dv_ref.at[:, : dv.shape[-1]],  # E: is not assignable
        dv.astype(dv_ref.dtype),
        mask=head_mask,
    )
    plgpu.store(  # E: No matching overload
        dk_ref.at[:, : dk.shape[-1]],  # E: is not assignable
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
    dq = jnp.zeros(  # E: No matching overload
        [block_q_dq, head_dim_padded], dtype=jnp.float32
    )

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


def mha_backward_host[
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
    batches: Int[B],
    queries: Int[Q],
    keys: Int[K],
    heads: Int[H],
    dim: Int[D],
    padded: Int[P],
    block_q_dkv: Int[BQDKV],
    block_kv_dkv: Int[BKVDKV],
    block_q_dq: Int[BQDQ],
    block_kv_dq: Int[BKVDQ],
    causal: bool,
) -> tuple[jax.Array[[B, Q, H, D]], jax.Array[[B, K, H, D]], jax.Array[[B, K, H, D]]]:
    if min(block_q_dkv, block_kv_dkv, block_q_dq, block_kv_dq) <= 0:
        raise ValueError("Backward blocks must be positive")
    if queries % block_q_dkv or keys % block_kv_dkv:
        raise ValueError("The first scan must divide both sequence axes")
    if queries % block_q_dq or keys % block_kv_dq:
        raise ValueError("The second scan must divide both sequence axes")
    if queries // block_q_dq != keys // block_kv_dkv:
        raise ValueError("Both output gradients must use the same grid")
    if dim <= 0 or padded < dim or padded & (padded - 1):
        raise ValueError("Padded head dimension must cover the feature axis")
    if (causal or segments is not None) and queries > keys:
        raise ValueError("Key and segment coverage must include every query")

    q_spec: pl.BlockSpec[[Q, P], Literal["mha_backward_input"]] = pl.BlockSpec(
        (None, queries, None, padded), lambda i, j, _: (i, 0, j, 0)
    )
    k_spec: pl.BlockSpec[[K, P], Literal["mha_backward_input"]] = pl.BlockSpec(
        (None, keys, None, padded), lambda i, j, _: (i, 0, j, 0)
    )
    seg_spec: pl.BlockSpec[[K], Literal["mha_segment"]] | None = (
        None if segments is None else pl.BlockSpec((None, keys), lambda i, j, _: (i, 0))
    )
    residual_spec: pl.BlockSpec[[Q], Literal["mha_lse"]] = pl.BlockSpec(
        (None, None, queries), lambda i, j, _: (i, j, 0)
    )
    dq_spec: pl.BlockSpec[[BQDQ, P], Literal["mha_backward_output"]] = pl.BlockSpec(
        (None, block_q_dq, None, padded), lambda i, j, step: (i, step, j, 0)
    )
    dkv_spec: pl.BlockSpec[[BKVDKV, P], Literal["mha_backward_output"]] = pl.BlockSpec(
        (None, block_kv_dkv, None, padded),
        lambda i, j, step: (i, step, j, 0),
    )

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

    return pl.pallas_call(
        kernel,
        in_specs=(
            q_spec,
            k_spec,
            k_spec,
            seg_spec,
            q_spec,
            q_spec,
            residual_spec,
            residual_spec,
        ),
        out_specs=(dq_spec, dkv_spec, dkv_spec),
        grid=(batches, heads, pl.cdiv(keys, block_kv_dkv)),
        out_shape=(
            jax.ShapeDtypeStruct((batches, queries, heads, dim), q.dtype),
            jax.ShapeDtypeStruct((batches, keys, heads, dim), k.dtype),
            jax.ShapeDtypeStruct((batches, keys, heads, dim), v.dtype),
        ),
        interpret=True,
    )(q, k, v, segments, out, dout, lse, delta)


def test_host_result[
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
    b: Int[B],
    query_len: Int[Q],
    key_len: Int[K],
    h: Int[H],
    d: Int[D],
    p: Int[P],
    bqdkv: Int[BQDKV],
    bkvdkv: Int[BKVDKV],
    bqdq: Int[BQDQ],
    bkvdq: Int[BKVDQ],
) -> None:
    assert_type(
        mha_backward_host(
            q,
            k,
            v,
            segments,
            out,
            dout,
            lse,
            delta,
            b,
            query_len,
            key_len,
            h,
            d,
            p,
            bqdkv,
            bkvdkv,
            bqdq,
            bkvdq,
            False,
        ),
        tuple[
            jax.Array[[B, Q, H, D]], jax.Array[[B, K, H, D]], jax.Array[[B, K, H, D]]
        ],
    )


def test_wrong_host_axes[
    B: IntVar,
    Q: IntVar,
    K: IntVar,
    H: IntVar,
    D: IntVar,
    P: IntVar,
    Other: IntVar,
    BQDKV: IntVar,
    BKVDKV: IntVar,
    BQDQ: IntVar,
    BKVDQ: IntVar,
](
    q: jax.Array[[B, Q, H, D]],
    k: jax.Array[[B, K, H, D]],
    wrong_v: jax.Array[[B, K, Other, D]],
    segments: jax.Array[[B, Other]],
    out: jax.Array[[B, Q, H, D]],
    wrong_dout: jax.Array[[B, Q, H, Other]],
    wrong_lse: jax.Array[[B, H, Other]],
    delta: jax.Array[[B, H, Q]],
    b: Int[B],
    qlen: Int[Q],
    klen: Int[K],
    h: Int[H],
    d: Int[D],
    p: Int[P],
    bqdkv: Int[BQDKV],
    bkvdkv: Int[BKVDKV],
    bqdq: Int[BQDQ],
    bkvdq: Int[BKVDQ],
) -> None:
    mha_backward_host(
        q,
        k,
        wrong_v,  # E: is not assignable
        segments,  # E: is not assignable
        out,
        wrong_dout,  # E: is not assignable
        wrong_lse,  # E: is not assignable
        delta,
        b,
        qlen,
        klen,
        h,
        d,
        p,
        bqdkv,
        bkvdkv,
        bqdq,
        bkvdq,
        False,
    )


def test_wrong_dot_contraction[
    Q: IntVar,
    K: IntVar,
    P: IntVar,
    D: IntVar,
    Other: IntVar,
](
    q: pl.Tile[[Q, P]],
    wrong_k: pl.Tile[[K, Other]],
) -> None:
    plgpu.dot(q, wrong_k.T)  # E: is not assignable


def test_wrong_gradient_store[
    Block: IntVar,
    P: IntVar,
    D: IntVar,
    Other: IntVar,
](
    dq_ref: pl.MhaBackwardOutputRef[Block, P],
    bad_value: pl.Tile[[Block, Other]],
    padded: Int[P],
    dim: Int[D],
) -> None:
    mask = (jnp.arange(padded) < dim)[None, :]
    plgpu.store(  # E: No matching overload
        dq_ref.at[:, :padded], bad_value, mask=mask
    )


def test_wrong_call_specs_and_allocations[
    B: IntVar,
    Q: IntVar,
    K: IntVar,
    H: IntVar,
    D: IntVar,
    P: IntVar,
    BQ: IntVar,
    BK: IntVar,
    Other: IntVar,
](
    kernel: Callable[
        [
            pl.MhaBackwardMatrixRef[Q, P, D],
            pl.MhaBackwardMatrixRef[K, P, D],
            pl.MhaBackwardMatrixRef[K, P, D],
            pl.MhaSegmentRef[K] | None,
            pl.MhaBackwardMatrixRef[Q, P, D],
            pl.MhaBackwardMatrixRef[Q, P, D],
            pl.MhaBackwardVectorRef[Q],
            pl.MhaBackwardVectorRef[Q],
            pl.MhaBackwardOutputRef[BQ, P],
            pl.MhaBackwardOutputRef[BK, P],
            pl.MhaBackwardOutputRef[BK, P],
        ],
        None,
    ],
    query_spec: pl.BlockSpec[[Q, P], Literal["mha_backward_input"]],
    key_spec: pl.BlockSpec[[K, P], Literal["mha_backward_input"]],
    bad_query_spec: pl.BlockSpec[[Q, Other], Literal["mha_backward_input"]],
    segment_spec: pl.BlockSpec[[K], Literal["mha_segment"]] | None,
    residual_spec: pl.BlockSpec[[Q], Literal["mha_lse"]],
    dq_spec: pl.BlockSpec[[BQ, P], Literal["mha_backward_output"]],
    dkv_spec: pl.BlockSpec[[BK, P], Literal["mha_backward_output"]],
    bad_dkv_spec: pl.BlockSpec[[Other, P], Literal["mha_backward_output"]],
    q: jax.Array[[B, Q, H, D]],
    k: jax.Array[[B, K, H, D]],
    v: jax.Array[[B, K, H, D]],
    segments: jax.Array[[B, K]] | None,
    out: jax.Array[[B, Q, H, D]],
    dout: jax.Array[[B, Q, H, D]],
    lse: jax.Array[[B, H, Q]],
    delta: jax.Array[[B, H, Q]],
    batches: Int[B],
    heads: Int[H],
    queries: Int[Q],
    keys: Int[K],
    key_block: Int[BK],
    wrong: Int[Other],
) -> None:
    good_inputs = (
        query_spec,
        key_spec,
        key_spec,
        segment_spec,
        query_spec,
        query_spec,
        residual_spec,
        residual_spec,
    )
    good_outputs = (dq_spec, dkv_spec, dkv_spec)
    good_grid = (batches, heads, pl.cdiv(keys, key_block))
    q_shape = jax.ShapeDtypeStruct((batches, queries, heads, q.shape[3]), q.dtype)
    k_shape = jax.ShapeDtypeStruct((batches, keys, heads, q.shape[3]), k.dtype)
    v_shape = jax.ShapeDtypeStruct((batches, keys, heads, q.shape[3]), v.dtype)

    pl.pallas_call(  # E: No matching overload
        kernel,
        in_specs=(
            bad_query_spec,
            key_spec,
            key_spec,
            segment_spec,
            query_spec,
            query_spec,
            residual_spec,
            residual_spec,
        ),
        out_specs=good_outputs,
        grid=good_grid,
        out_shape=(q_shape, k_shape, v_shape),
    )(q, k, v, segments, out, dout, lse, delta)
    pl.pallas_call(  # E: No matching overload
        kernel,
        in_specs=good_inputs,
        out_specs=(dq_spec, bad_dkv_spec, dkv_spec),
        grid=good_grid,
        out_shape=(q_shape, k_shape, v_shape),
    )(q, k, v, segments, out, dout, lse, delta)
    pl.pallas_call(  # E: No matching overload
        kernel,
        in_specs=good_inputs,
        out_specs=good_outputs,
        grid=(batches, heads, pl.cdiv(queries, key_block)),
        out_shape=(q_shape, k_shape, v_shape),
    )(q, k, v, segments, out, dout, lse, delta)
    pl.pallas_call(  # E: No matching overload
        kernel,
        in_specs=good_inputs,
        out_specs=good_outputs,
        grid=good_grid,
        out_shape=(
            jax.ShapeDtypeStruct((batches, wrong, heads, q.shape[3]), q.dtype),
            k_shape,
            v_shape,
        ),
    )(q, k, v, segments, out, dout, lse, delta)
    pl.pallas_call(  # E: No matching overload
        kernel,
        in_specs=good_inputs,
        out_specs=good_outputs,
        grid=good_grid,
        out_shape=(
            q_shape,
            jax.ShapeDtypeStruct((batches, wrong, heads, k.shape[3]), k.dtype),
            v_shape,
        ),
    )(q, k, v, segments, out, dout, lse, delta)
    pl.pallas_call(  # E: No matching overload
        kernel,
        in_specs=good_inputs,
        out_specs=good_outputs,
        grid=good_grid,
        out_shape=(
            q_shape,
            k_shape,
            jax.ShapeDtypeStruct((batches, keys, heads, wrong), v.dtype),
        ),
    )(q, k, v, segments, out, dout, lse, delta)


if not TYPE_CHECKING:
    import unittest

    import numpy as np
    from jax.experimental.pallas.ops.gpu.attention import BlockSizes, mha

    class MhaBackwardBoundaryTest(unittest.TestCase):
        def test_cpu_interpreter_matches_independent_attention_gradients(self) -> None:
            q = jnp.linspace(-0.75, 0.6, 32 * 16, dtype=jnp.float16).reshape(
                1, 32, 1, 16
            )
            k = jnp.linspace(-0.5, 0.8, 64 * 16, dtype=jnp.float16).reshape(
                1, 64, 1, 16
            )
            v = jnp.flip(k, axis=-1) * 0.75
            dout = jnp.flip(q, axis=1) + 0.125
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
            delta = jnp.sum(out.astype(jnp.float32) * dout.astype(jnp.float32), axis=-1)
            delta = delta.transpose((0, 2, 1))

            actual = mha_backward_host(
                q,
                k,
                v,
                None,
                out,
                dout,
                lse,
                delta,
                1,
                32,
                64,
                1,
                16,
                16,
                16,
                32,
                16,
                16,
                False,
            )

            def attention_reference(
                q: jax.Array, k: jax.Array, v: jax.Array
            ) -> jax.Array:
                scores = jnp.einsum(
                    "bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32
                )
                weights = jax.nn.softmax(scores, axis=-1)
                return jnp.einsum(
                    "bhqk,bkhd->bqhd",
                    weights,
                    v,
                    preferred_element_type=jnp.float32,
                ).astype(q.dtype)

            expected = jax.grad(
                lambda q, k, v: jnp.sum(attention_reference(q, k, v) * dout),
                argnums=(0, 1, 2),
            )(q, k, v)
            for got, want in zip(actual, expected):
                self.assertEqual(got.shape, want.shape)
                np.testing.assert_allclose(
                    np.asarray(got),
                    np.asarray(want),
                    rtol=3e-2,
                    atol=3e-2,
                )

        def test_causal_segmented_scans_match_independent_gradients(self) -> None:
            q = jnp.linspace(-0.3, 0.6, 32 * 16, dtype=jnp.float16).reshape(
                1, 32, 1, 16
            )
            k = jnp.flip(q, axis=-1) * 0.5
            v = jnp.cos(q.astype(jnp.float32)).astype(jnp.float16)
            dout = jnp.flip(q, axis=1) + 0.2
            segments = jnp.array([[0] * 16 + [1] * 16], dtype=jnp.int32)
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
            delta = jnp.sum(out.astype(jnp.float32) * dout.astype(jnp.float32), axis=-1)
            delta = delta.transpose((0, 2, 1))
            actual = mha_backward_host(
                q,
                k,
                v,
                segments,
                out,
                dout,
                lse,
                delta,
                1,
                32,
                32,
                1,
                16,
                16,
                16,
                16,
                16,
                16,
                True,
            )

            def masked_reference(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
                scores = jnp.einsum(
                    "bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32
                )
                same_segment = segments[:, :, None] == segments[:, None, :]
                causal_mask = jnp.arange(32)[:, None] >= jnp.arange(32)[None, :]
                scores = jnp.where(
                    same_segment[:, None, :, :] & causal_mask[None, None, :, :],
                    scores,
                    -1e10,
                )
                probs = jax.nn.softmax(scores, axis=-1)
                return jnp.einsum(
                    "bhqk,bkhd->bqhd",
                    probs,
                    v,
                    preferred_element_type=jnp.float32,
                ).astype(q.dtype)

            expected = jax.grad(
                lambda q, k, v: jnp.sum(masked_reference(q, k, v) * dout),
                argnums=(0, 1, 2),
            )(q, k, v)
            for got, want in zip(actual, expected):
                self.assertEqual(got.shape, want.shape)
                np.testing.assert_allclose(
                    np.asarray(got),
                    np.asarray(want),
                    rtol=3e-2,
                    atol=3e-2,
                )

        def test_grid_mismatch_is_rejected(self) -> None:
            q = jnp.zeros((1, 32, 1, 16), dtype=jnp.float16)
            k = jnp.zeros((1, 64, 1, 16), dtype=jnp.float16)
            lse = jnp.zeros((1, 1, 32), dtype=jnp.float32)
            with self.assertRaisesRegex(ValueError, "same grid"):
                mha_backward_host(
                    q,
                    k,
                    k,
                    None,
                    q,
                    q,
                    lse,
                    lse,
                    1,
                    32,
                    64,
                    1,
                    16,
                    16,
                    16,
                    16,
                    16,
                    16,
                    False,
                )

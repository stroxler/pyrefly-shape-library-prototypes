# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The complete attn_forward_kernel and its nested callbacks come from JAX
# jax/experimental/pallas/ops/gpu/decode_attention.py (Apache-2.0).
# Only semantic parameter annotations differ; the upstream module is deprecated.
# @lint-ignore-every AUTODEPS2

"""Decode attention relates per-split KV Refs to one shared query-head tile."""

from __future__ import annotations

from collections.abc import Callable
from typing import assert_type, Literal, TYPE_CHECKING

import jax
import jax.lax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar

    Splits = IntVar("Splits")
    Heads = IntVar("Heads")
    HeadBlock = IntVar("HeadBlock")
    TotalKeys = IntVar("TotalKeys")
    SplitKeys = IntVar("SplitKeys")
    KeyBlock = IntVar("KeyBlock")
    Dim = IntVar("Dim")
    DotRows = IntVar("DotRows")
    DotInner = IntVar("DotInner")
    DotCols = IntVar("DotCols")


def attn_forward_kernel(
    # inputs
    q_ref: pl.DecodeQueryRef[HeadBlock, Dim],  # [num_heads, head_dim]
    k_ref: pl.DecodeKvRef[SplitKeys, Dim],  # [k_seq_len, head_dim]
    v_ref: pl.DecodeKvRef[SplitKeys, Dim],  # [k_seq_len, head_dim]
    start_idx_ref: pl.DecodeBoundRef | None,  # [] (i.e., scalar)
    kv_seq_len_ref: pl.DecodeBoundRef | None,  # [] (i.e., scalar)
    # outputs
    o_ref: pl.DecodeOutputRef[HeadBlock, Dim],  # [num_heads, head_dim]
    *residual_refs: pl.DecodeResidualRef[HeadBlock],  # [num_heads,], [num_heads,]
    sm_scale: float,
    block_k: Int[KeyBlock],
    block_h: Int[HeadBlock],
    num_heads: Int[Heads],
):
    _, head_dim = q_ref.shape
    split_k_seq_len, _ = k_ref.shape
    prog_i, prog_j = pl.program_id(0), pl.program_id(1)
    q_slice = pl.ds(0, block_h)
    q_mask = (jnp.arange(block_h) < num_heads - block_h * prog_i)[:, None]

    def _compute(
        start_idx: int,
        kv_seq_len: int,
        o: pl.Tile[[HeadBlock, Dim]],
        m_i: pl.Tile[[HeadBlock]],
        l_i: pl.Tile[[HeadBlock]],
    ):
        # Load q: it will stay in L1 throughout. Indices form a matrix because we
        # read, compute, and write all in 2d chunks. 1 element ~= 1 CUDA thread index.
        # q tile has shape [block_h, head_dim].
        q = plgpu.load(q_ref.at[q_slice, :], mask=q_mask)

        def _dot(a: pl.Tile[[DotRows, DotInner]], b: pl.Tile[[DotInner, DotCols]]):
            # if a.shape[0] == 1:
            #   # Use matrix vector product
            #   return (a.T * b).sum(axis=0, keepdims=True)
            return plgpu.dot(a, b)

        mask_indices = jnp.arange(block_k)

        # Loop over blocks of kv to process entire kv seq_len.
        # Grid loops over q blocks over num_heads.
        def body(
            start_k: int,
            carry: tuple[
                pl.Tile[[HeadBlock, Dim]], pl.Tile[[HeadBlock]], pl.Tile[[HeadBlock]]
            ],
        ):
            o_prev, m_prev, l_prev = carry
            curr_k_slice = pl.ds(start_k * block_k, block_k)

            k = k_ref[curr_k_slice, :]
            qk = _dot(q, k.T)  # [block_h, block_k]
            if sm_scale != 1.0:
                qk *= sm_scale  # [block_h, block_k]

            # apply mask if start or sequence length is specified
            if start_idx_ref is not None or kv_seq_len_ref is not None:
                indices = prog_j * split_k_seq_len + start_k * block_k + mask_indices
                mask = ((indices >= start_idx) & (indices < kv_seq_len))[None, :]
                qk += (~mask) * (0.7 * jnp.finfo(qk.dtype).min)

            m_curr = qk.max(axis=-1)
            m_next = jnp.maximum(m_prev, m_curr)
            correction = jnp.exp(m_prev - m_next)
            l_prev_corr = correction * l_prev
            s_curr = jnp.exp(
                qk - m_next[:, None]
            )  # Use m_next instead of m_curr to avoid a correction on l_curr
            l_curr = s_curr.sum(axis=-1)
            l_next = l_prev_corr + l_curr
            v = v_ref[curr_k_slice, :]
            o_curr = _dot(s_curr.astype(v.dtype), v)

            # flash2 unscaled_o
            o_next = correction[:, None] * o_prev + o_curr
            return o_next, m_next, l_next

        max_it = jnp.minimum(
            pl.cdiv((kv_seq_len - prog_j * split_k_seq_len), block_k),
            split_k_seq_len // block_k,
        )
        (o, m_i, l_i) = lax.fori_loop(0, max_it, body, (o, m_i, l_i))
        return o, m_i, l_i

    # o is the buffer where we accumulate the output on sram.
    # m_i and l_i (see FlashAttention2 paper) are updated during the k,v loop.
    m_i = jnp.zeros(block_h, dtype=jnp.float32) + jnp.finfo(jnp.float32).min
    l_i = jnp.zeros(block_h, dtype=jnp.float32)
    o = jnp.zeros((block_h, head_dim), dtype=jnp.float32)

    start_idx = split_k_seq_len * prog_j
    if start_idx_ref is not None:
        start_idx = jnp.maximum(start_idx, start_idx_ref[()])
    kv_seq_len = (prog_j + 1) * split_k_seq_len  # lower bound on actual k_seq_len
    if kv_seq_len_ref is not None:
        kv_seq_len = jnp.minimum(kv_seq_len, kv_seq_len_ref[()])

    if start_idx_ref is None and kv_seq_len is None:
        o, m_i, l_i = _compute(start_idx, kv_seq_len, o, m_i, l_i)
    else:
        o, m_i, l_i = jax.lax.cond(
            start_idx >= kv_seq_len,
            lambda: (o, m_i, l_i),
            lambda: _compute(start_idx, kv_seq_len, o, m_i, l_i),
        )

    # Write output to dram.
    if residual_refs:
        l_ref, m_ref = residual_refs
        vec_q_mask = q_mask.reshape(-1) if q_mask is not None else None
        plgpu.store(l_ref.at[q_slice], l_i, mask=vec_q_mask)
        plgpu.store(m_ref.at[q_slice], m_i, mask=vec_q_mask)
    o = o.astype(o_ref.dtype)
    plgpu.store(o_ref.at[q_slice, :], o, mask=q_mask)


def decode_partials[
    S: IntVar,
    H: IntVar,
    K: IntVar,
    SK: IntVar,
    D: IntVar,
    BH: IntVar,
    BK: IntVar,
](
    q: jax.Array[[H, D]],
    k: jax.Array[[K, D]],
    v: jax.Array[[K, D]],
    start_idx: jax.Array[[]] | None,
    length: jax.Array[[]] | None,
    splits: Int[S],
    heads: Int[H],
    total_keys: Int[K],
    split_keys: Int[SK],
    dim: Int[D],
    block_h: Int[BH],
    block_k: Int[BK],
) -> tuple[jax.Array[[S, H, D]], jax.Array[[S, H]], jax.Array[[S, H]]]:
    if block_k < 16:
        raise ValueError("The decode GPU kernel requires block_k >= 16")
    if total_keys != splits * split_keys or split_keys < 16 or split_keys % block_k:
        raise ValueError("Each KV split must contain whole blocks of at least 16 keys")
    q_spec: pl.BlockSpec[[BH, D]] = pl.BlockSpec((block_h, dim), lambda i, j: (i, 0))
    kv_spec: pl.BlockSpec[[SK, D], Literal["decode_kv"]] = pl.BlockSpec(
        (None, split_keys, dim), lambda i, j: (j, 0, 0)
    )
    bound_spec: pl.BlockSpec[[], Literal["decode_bound"]] = pl.BlockSpec(
        (), lambda i, j: ()
    )
    out_spec: pl.BlockSpec[[BH, D], Literal["decode_output"]] = pl.BlockSpec(
        (None, block_h, dim), lambda i, j: (j, i, 0)
    )
    residual_spec: pl.BlockSpec[[BH], Literal[True]] = pl.BlockSpec(
        (None, block_h), lambda i, j: (j, i)
    )

    def kernel(
        q_ref: pl.DecodeQueryRef[BH, D],
        k_ref: pl.DecodeKvRef[SK, D],
        v_ref: pl.DecodeKvRef[SK, D],
        start_ref: pl.DecodeBoundRef | None,
        length_ref: pl.DecodeBoundRef | None,
        out_ref: pl.DecodeOutputRef[BH, D],
        l_ref: pl.DecodeResidualRef[BH],
        m_ref: pl.DecodeResidualRef[BH],
    ) -> None:
        attn_forward_kernel(
            q_ref,
            k_ref,
            v_ref,
            start_ref,
            length_ref,
            out_ref,
            l_ref,
            m_ref,
            sm_scale=0.5,
            block_k=block_k,
            block_h=block_h,
            num_heads=heads,
        )

    return pl.pallas_call(
        kernel,
        in_specs=(
            q_spec,
            kv_spec,
            kv_spec,
            None if start_idx is None else bound_spec,
            None if length is None else bound_spec,
        ),
        out_specs=(out_spec, residual_spec, residual_spec),
        grid=(pl.cdiv(heads, block_h), splits),
        out_shape=(
            jax.ShapeDtypeStruct((splits, heads, dim), q.dtype),
            jax.ShapeDtypeStruct((splits, heads), jnp.float32),
            jax.ShapeDtypeStruct((splits, heads), jnp.float32),
        ),
        interpret=True,
    )(
        q,
        k.reshape((splits, split_keys, dim)),
        v.reshape((splits, split_keys, dim)),
        start_idx,
        length,
    )


def test_typed_host_result[
    S: IntVar,
    H: IntVar,
    K: IntVar,
    SK: IntVar,
    D: IntVar,
    BH: IntVar,
    BK: IntVar,
](
    q: jax.Array[[H, D]],
    k: jax.Array[[K, D]],
    v: jax.Array[[K, D]],
    splits: Int[S],
    heads: Int[H],
    keys: Int[K],
    split_keys: Int[SK],
    dim: Int[D],
    block_h: Int[BH],
    block_k: Int[BK],
) -> None:
    assert_type(
        decode_partials(
            q,
            k,
            v,
            None,
            None,
            splits,
            heads,
            keys,
            split_keys,
            dim,
            block_h,
            block_k,
        ),
        tuple[jax.Array[[S, H, D]], jax.Array[[S, H]], jax.Array[[S, H]]],
    )


def test_wrong_host_kv_axes[
    S: IntVar,
    H: IntVar,
    K: IntVar,
    SK: IntVar,
    D: IntVar,
    Other: IntVar,
    BH: IntVar,
    BK: IntVar,
](
    q: jax.Array[[H, D]],
    k: jax.Array[[K, Other]],
    v: jax.Array[[Other, D]],
    splits: Int[S],
    heads: Int[H],
    keys: Int[K],
    split_keys: Int[SK],
    dim: Int[D],
    block_h: Int[BH],
    block_k: Int[BK],
) -> None:
    decode_partials(
        q,
        k,  # E: is not assignable
        v,  # E: is not assignable
        None,
        None,
        splits,
        heads,
        keys,
        split_keys,
        dim,
        block_h,
        block_k,
    )


def test_wrong_kernel_ref_axes[
    H: IntVar,
    BH: IntVar,
    BK: IntVar,
    SK: IntVar,
    D: IntVar,
    Other: IntVar,
](
    q: pl.DecodeQueryRef[BH, D],
    k: pl.DecodeKvRef[SK, Other],
    v: pl.DecodeKvRef[SK, D],
    out: pl.DecodeOutputRef[BH, Other],
    l: pl.DecodeResidualRef[BH],
    m: pl.DecodeResidualRef[BH],
    heads: Int[H],
    block_h: Int[BH],
    block_k: Int[BK],
) -> None:
    attn_forward_kernel(
        q,
        k,  # E: is not assignable
        v,
        None,
        None,
        out,  # E: is not assignable
        l,
        m,
        sm_scale=0.5,
        block_k=block_k,
        block_h=block_h,
        num_heads=heads,
    )


def test_wrong_scalar_ref[
    H: IntVar,
    BH: IntVar,
    BK: IntVar,
    SK: IntVar,
    D: IntVar,
](
    q: pl.DecodeQueryRef[BH, D],
    kv: pl.DecodeKvRef[SK, D],
    not_scalar: pl.InRef[[1]],
    out: pl.DecodeOutputRef[BH, D],
    residual: pl.DecodeResidualRef[BH],
    heads: Int[H],
    block_h: Int[BH],
    block_k: Int[BK],
) -> None:
    attn_forward_kernel(
        q,
        kv,
        kv,
        not_scalar,  # E: is not assignable
        None,
        out,
        residual,
        residual,
        sm_scale=0.5,
        block_k=block_k,
        block_h=block_h,
        num_heads=heads,
    )


def test_wrong_query_mask[BH: IntVar, D: IntVar, Other: IntVar](
    q: pl.DecodeQuerySlice[BH, D],
    mask: pl.ContractionVerticalMask[Other, int],
) -> None:
    plgpu.load(q, mask=mask)  # E: No matching overload


def test_wrong_dot_inner[BH: IntVar, BK: IntVar, D: IntVar, Other: IntVar](
    q: pl.Tile[[BH, D]],
    k: pl.Tile[[BK, Other]],
) -> None:
    plgpu.dot(q, k.T)  # E: is not assignable


def test_wrong_residual_store[BH: IntVar, Other: IntVar](
    l: pl.DecodeResidualRef[BH],
    idx: pl.HalfRowSlice[BH],
    value: pl.Tile[[Other]],
    mask: pl.Mask[BH, int],
) -> None:
    plgpu.store(l.at[idx], value, mask=mask)  # E: No matching overload


def test_wrong_split_output_allocation[
    S: IntVar,
    H: IntVar,
    SK: IntVar,
    D: IntVar,
    BH: IntVar,
    Other: IntVar,
](
    kernel: Callable[
        [
            pl.DecodeQueryRef[BH, D],
            pl.DecodeKvRef[SK, D],
            pl.DecodeKvRef[SK, D],
            pl.DecodeBoundRef | None,
            pl.DecodeBoundRef | None,
            pl.DecodeOutputRef[BH, D],
            pl.DecodeResidualRef[BH],
            pl.DecodeResidualRef[BH],
        ],
        None,
    ],
    q: pl.BlockSpec[[BH, D]],
    kv: pl.BlockSpec[[SK, D], Literal["decode_kv"]],
    out: pl.BlockSpec[[BH, D], Literal["decode_output"]],
    residual: pl.BlockSpec[[BH], Literal[True]],
    splits: Int[S],
    heads: Int[H],
    other: Int[Other],
    dim: Int[D],
    bh: Int[BH],
) -> None:
    pl.pallas_call(  # E: No matching overload
        kernel,
        in_specs=(q, kv, kv, None, None),
        out_specs=(out, residual, residual),
        grid=(pl.cdiv(heads, bh), splits),
        out_shape=(
            jax.ShapeDtypeStruct((splits, heads, dim), jnp.float32),
            jax.ShapeDtypeStruct((splits, other), jnp.float32),
            jax.ShapeDtypeStruct((splits, heads), jnp.float32),
        ),
    )


def test_wrong_squeezed_kv_spec[
    S: IntVar,
    H: IntVar,
    SK: IntVar,
    D: IntVar,
    BH: IntVar,
](
    kernel: Callable[
        [
            pl.DecodeQueryRef[BH, D],
            pl.DecodeKvRef[SK, D],
            pl.DecodeKvRef[SK, D],
            pl.DecodeBoundRef | None,
            pl.DecodeBoundRef | None,
            pl.DecodeOutputRef[BH, D],
            pl.DecodeResidualRef[BH],
            pl.DecodeResidualRef[BH],
        ],
        None,
    ],
    q: pl.BlockSpec[[BH, D]],
    wrong_kv: pl.BlockSpec[[SK, D]],
    kv: pl.BlockSpec[[SK, D], Literal["decode_kv"]],
    out: pl.BlockSpec[[BH, D], Literal["decode_output"]],
    residual: pl.BlockSpec[[BH], Literal[True]],
    splits: Int[S],
    heads: Int[H],
    dim: Int[D],
    bh: Int[BH],
) -> None:
    pl.pallas_call(  # E: No matching overload
        kernel,
        in_specs=(q, wrong_kv, kv, None, None),
        out_specs=(out, residual, residual),
        grid=(pl.cdiv(heads, bh), splits),
        out_shape=(
            jax.ShapeDtypeStruct((splits, heads, dim), jnp.float32),
            jax.ShapeDtypeStruct((splits, heads), jnp.float32),
            jax.ShapeDtypeStruct((splits, heads), jnp.float32),
        ),
    )


def test_split_index_map_coverage_is_not_proved[SK: IntVar, D: IntVar](
    split_keys: Int[SK],
    dim: Int[D],
) -> None:
    # The map is rank-correct but every program would read the first KV split.
    bad: pl.BlockSpec[[SK, D], Literal["decode_kv"]] = pl.BlockSpec(
        (None, split_keys, dim), lambda i, j: (0, 0, 0)
    )
    assert_type(bad, pl.BlockSpec[[SK, D], Literal["decode_kv"]])


if not TYPE_CHECKING:
    import unittest

    import numpy as np

    class DecodeAttentionBoundaryTest(unittest.TestCase):
        def test_masked_split_attention_matches_independent_reference(self) -> None:
            q = jnp.arange(3 * 16, dtype=jnp.float16).reshape(3, 16) / 19
            k = jnp.arange(32 * 16, dtype=jnp.float16).reshape(32, 16) / 79
            v = jnp.arange(32 * 16, dtype=jnp.float16).reshape(32, 16) / 43
            o, l, m = decode_partials(
                q,
                k,
                v,
                jnp.array(4, dtype=jnp.int32),
                jnp.array(26, dtype=jnp.int32),
                2,
                3,
                32,
                16,
                16,
                2,
                16,
            )
            self.assertEqual(o.shape, (2, 3, 16))
            self.assertEqual(l.shape, (2, 3))
            self.assertEqual(m.shape, (2, 3))
            for split in range(2):
                indices = list(range(max(split * 16, 4), min((split + 1) * 16, 26)))
                scores = (
                    np.asarray(q, dtype=np.float32)
                    @ np.asarray(k, dtype=np.float32)[indices].T
                ) * 0.5
                peak = scores.max(axis=-1)
                weights = np.exp(scores - peak[:, None])
                np.testing.assert_allclose(
                    np.asarray(m[split]), peak, rtol=2e-2, atol=2e-2
                )
                np.testing.assert_allclose(
                    np.asarray(l[split]), weights.sum(axis=-1), rtol=2e-2, atol=2e-2
                )
                np.testing.assert_allclose(
                    np.asarray(o[split]),
                    weights @ np.asarray(v, dtype=np.float32)[indices],
                    rtol=2e-2,
                    atol=2e-2,
                )

            split_m = np.asarray(m)
            correction = np.exp(split_m - split_m.max(axis=0, keepdims=True))
            merged = (np.asarray(o) * correction[:, :, None]).sum(axis=0) / (
                (np.asarray(l) * correction).sum(axis=0)[:, None]
            )
            scores = (
                np.asarray(q, dtype=np.float32)
                @ np.asarray(k, dtype=np.float32)[4:26].T
            ) * 0.5
            weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
            weights /= weights.sum(axis=-1, keepdims=True)
            expected = weights @ np.asarray(v, dtype=np.float32)[4:26]
            np.testing.assert_allclose(merged, expected, rtol=2e-2, atol=2e-2)

        def test_split_contract_rejects_incompatible_length(self) -> None:
            q = jnp.ones((2, 16), dtype=jnp.float16)
            kv = jnp.ones((33, 16), dtype=jnp.float16)
            with self.assertRaisesRegex(ValueError, "whole blocks"):
                decode_partials(q, kv, kv, None, None, 2, 2, 33, 16, 16, 2, 16)

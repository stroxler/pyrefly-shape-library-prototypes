"""Unbatched split-KV decode attention from JAX's deprecated GPU Pallas op.

The executable kernel statements come from JAX
jax/experimental/pallas/ops/gpu/decode_attention.py (Apache-2.0);
only semantic annotations on its arguments and nested callbacks are added.

The checked entrypoint exercises unbounded K/V sequences and both residual
outputs. Optional scalar start/length bounds, batched MQA/GQA wrappers, and
paged attention are not part of this fixture. Static types connect the host
split/head/dim axes to kernel Refs and the output tuple, but cannot prove
arbitrary index-map lambda values or the log-sum-exp recurrence.
"""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any, Literal, assert_type, cast
from unittest.mock import patch

import jax
import jax.lax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
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
    splits: Int[S],
    heads: Int[H],
    total_keys: Int[K],
    split_keys: Int[SK],
    dim: Int[D],
    block_h: Int[BH],
    block_k: Int[BK],
    *,
    sm_scale: float = 0.5,
) -> tuple[jax.Array[[S, H, D]], jax.Array[[S, H]], jax.Array[[S, H]]]:
    """Validate the host axes and construct one query-head and KV-split grid."""
    if (
        q.ndim != 2
        or k.ndim != 2
        or v.ndim != 2
        or q.shape != (heads, dim)
        or k.shape != (total_keys, dim)
        or v.shape != (total_keys, dim)
        or min(heads, total_keys, dim) <= 0
    ):
        raise ValueError("Decode Q, K, V must share their head and key dimensions")
    if (
        q.dtype != k.dtype
        or q.dtype != v.dtype
        or q.dtype not in (jnp.float16, jnp.bfloat16)
    ):
        raise ValueError("Decode Q, K, V must share a supported 16-bit dtype")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Decode Q, K, V must be on the same device")
    if (
        splits <= 0
        or split_keys < 16
        or total_keys != splits * split_keys
        or block_h < 16
        or block_h & (block_h - 1)
        or block_k < 16
        or block_k & (block_k - 1)
        or split_keys % block_k
        or dim % 16
    ):
        raise ValueError("Decode splits and blocks need whole supported GPU tiles")

    q_spec: pl.BlockSpec[[BH, D]] = pl.BlockSpec((block_h, dim), lambda i, j: (i, 0))
    kv_spec: pl.BlockSpec[[SK, D], Literal["decode_kv"]] = pl.BlockSpec(
        (None, split_keys, dim), lambda i, j: (j, 0, 0)
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
            sm_scale=sm_scale,
            block_k=block_k,
            block_h=block_h,
            num_heads=heads,
        )

    return pl.pallas_call(
        kernel,
        in_specs=(q_spec, kv_spec, kv_spec, None, None),
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
        None,
        None,
    )


class DecodeAttentionTest(unittest.TestCase):
    """Check host shape mapping and the preserved kernel on CPU interpretation."""

    def test_split_layout_and_output_specs(self) -> None:
        q = jnp.ones((18, 16), dtype=jnp.float16)
        k = jnp.ones((32, 16), dtype=jnp.float16)
        with patch.object(
            pl,
            "pallas_call",
            side_effect=lambda body, **metadata: (
                lambda *args: (
                    jnp.zeros((2, 18, 16), dtype=jnp.float16),
                    jnp.zeros((2, 18), dtype=jnp.float32),
                    jnp.zeros((2, 18), dtype=jnp.float32),
                )
            ),
        ) as mocked:
            out, l, m = decode_partials(q, k, k, 2, 18, 32, 16, 16, 16, 16)
        self.assertEqual((out.shape, l.shape, m.shape), ((2, 18, 16), (2, 18), (2, 18)))
        metadata = mocked.call_args.kwargs
        self.assertEqual(metadata["grid"], (2, 2))
        q_spec, key_spec, value_spec, start_spec, length_spec = metadata["in_specs"]
        self.assertEqual(q_spec.block_shape, (16, 16))
        self.assertEqual(key_spec.block_shape, (None, 16, 16))
        self.assertIs(key_spec, value_spec)
        self.assertIsNone(start_spec)
        self.assertIsNone(length_spec)
        self.assertEqual(q_spec.index_map(1, 0), (1, 0))
        self.assertEqual(key_spec.index_map(1, 0), (0, 0, 0))
        out_spec, l_spec, m_spec = metadata["out_specs"]
        self.assertEqual(out_spec.index_map(1, 0), (0, 1, 0))
        self.assertEqual(l_spec.block_shape, (None, 16))
        self.assertIs(l_spec, m_spec)
        self.assertEqual(
            tuple(spec.shape for spec in metadata["out_shape"]),
            ((2, 18, 16), (2, 18), (2, 18)),
        )

    def test_host_contract_rejects_wrong_axes_and_dtypes(self) -> None:
        q = jnp.ones((16, 16), dtype=jnp.float16)
        k = jnp.ones((32, 16), dtype=jnp.float16)
        with self.assertRaisesRegex(ValueError, "head and key dimensions"):
            decode_partials(q, k, cast(Any, k[:, :8]), 2, 16, 32, 16, 16, 16, 16)
        with self.assertRaisesRegex(ValueError, "16-bit dtype"):
            decode_partials(
                q,
                jnp.ones((32, 16), dtype=jnp.float32),
                k,
                2,
                16,
                32,
                16,
                16,
                16,
                16,
            )
        with self.assertRaisesRegex(ValueError, "whole supported GPU tiles"):
            decode_partials(q, k, k, 3, 16, 32, 16, 16, 16, 16)

    def test_cpu_decode_partials_against_softmax(self) -> None:
        q = cast(
            "jax.Array[[18, 16]]",
            jnp.asarray(
                (np.arange(18 * 16).reshape(18, 16) % 11 / 10).astype(np.float16)
            ),
        )
        k = cast(
            "jax.Array[[32, 16]]",
            jnp.asarray(
                (np.arange(32 * 16).reshape(32, 16) % 13 / 10).astype(np.float16)
            ),
        )
        v = cast(
            "jax.Array[[32, 16]]",
            jnp.asarray(
                (np.arange(32 * 16).reshape(32, 16) % 17 / 10).astype(np.float16)
            ),
        )
        partial, denominator, max_logit = decode_partials(
            q, k, v, 2, 18, 32, 16, 16, 16, 16
        )
        weights = np.exp(
            np.asarray(max_logit) - np.asarray(max_logit).max(axis=0)[None, :]
        )
        actual = (np.asarray(partial, dtype=np.float32) * weights[:, :, None]).sum(
            axis=0
        ) / (np.asarray(denominator) * weights).sum(axis=0)[:, None]
        logits = (
            np.asarray(q, dtype=np.float32) @ np.asarray(k, dtype=np.float32).T * 0.5
        )
        logits -= logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=-1, keepdims=True)
        expected = probs @ np.asarray(v, dtype=np.float32)
        np.testing.assert_allclose(actual, expected, rtol=0.015, atol=0.015)


if TYPE_CHECKING:

    def check_decode_result[
        S: IntVar,
        H: IntVar,
        K: IntVar,
        SK: IntVar,
        D: IntVar,
        BH: IntVar,
        BK: IntVar,
        Other: IntVar,
    ](
        q: jax.Array[[H, D]],
        k: jax.Array[[K, D]],
        v: jax.Array[[K, D]],
        wrong: jax.Array[[Other, D]],
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
                q, k, v, splits, heads, keys, split_keys, dim, block_h, block_k
            ),
            tuple[jax.Array[[S, H, D]], jax.Array[[S, H]], jax.Array[[S, H]]],
        )
        decode_partials(
            q,
            k,
            wrong,  # pyrefly: ignore[bad-argument-type]
            splits,
            heads,
            keys,
            split_keys,
            dim,
            block_h,
            block_k,
        )

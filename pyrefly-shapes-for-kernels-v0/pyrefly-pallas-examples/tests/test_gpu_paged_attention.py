# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The full kernel body comes from JAX
# jax/experimental/pallas/ops/gpu/paged_attention.py.
# Only parameter annotations are added.
# @lint-ignore-every AUTODEPS2

"""Paged KV gathers connect a host page table to attention tile shapes."""

from __future__ import annotations

import math
from typing import assert_type, TYPE_CHECKING

import jax
import jax.lax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar

    Heads = IntVar("Heads")
    Dim = IntVar("Dim")
    Pages = IntVar("Pages")
    PageSize = IntVar("PageSize")
    TablePages = IntVar("TablePages")
    PageBlock = IntVar("PageBlock")


def paged_attention_kernel(
    # inputs
    q_ref: pl.PagedQueryRef[Heads, Dim],  # [block_h, head_dim]
    k_pages_ref: pl.PagedPoolRef[
        Pages, PageSize, Dim
    ],  # [total_num_pages, page_size, head_dim]
    k_scales_pages_ref: pl.PagedScaleRef[Pages, PageSize]
    | None,  # [total_num_pages, page_size]
    v_pages_ref: pl.PagedPoolRef[
        Pages, PageSize, Dim
    ],  # [total_num_pages, page_size, head_dim]
    v_scales_pages_ref: pl.PagedScaleRef[Pages, PageSize]
    | None,  # [total_num_pages, page_size]
    block_tables_ref: pl.PagedBlockTableRef[TablePages],  # [pages_per_partition]
    lengths_ref: pl.PagedLengthRef | None,  # [1]
    # outputs
    o_ref: pl.PagedOutputRef[Heads, Dim],  # [block_h, head_dim]
    *residual_refs: pl.PagedResidualRef[
        Heads
    ],  # Residual outputs: [block_h,], [block_h,]
    num_heads: int,
    pages_per_compute_block: Int[PageBlock],
    mask_value: float,
    attn_logits_soft_cap: float | None,
):
    partition_idx = pl.program_id(2)
    block_h, head_dim = q_ref.shape
    page_size = k_pages_ref.shape[-2]
    pages_per_partition = block_tables_ref.shape[0]
    block_k = pages_per_compute_block * page_size

    def _compute(
        start_page_idx: int,
        end_page_idx: int,
        o: pl.Tile[[Heads, Dim]],
        m_i: pl.Tile[[Heads]],
        l_i: pl.Tile[[Heads]],
    ):
        q_slice = pl.ds(0, block_h)
        q = q_ref[q_slice, :]

        # Loop over blocks of pages to process a entire page sequence partition.
        # Grid loops over q blocks over num_heads.
        def body(
            start_k: int,
            carry: tuple[pl.Tile[[Heads, Dim]], pl.Tile[[Heads]], pl.Tile[[Heads]]],
        ):
            o_prev, m_prev, l_prev = carry

            block_tables_slice = pl.ds(
                start_k * pages_per_compute_block, pages_per_compute_block
            )
            block_tables = block_tables_ref[block_tables_slice]
            k = k_pages_ref[block_tables].reshape(block_k, head_dim)
            v = v_pages_ref[block_tables].reshape(block_k, head_dim)
            if k_scales_pages_ref is not None:
                # dynamic lhs quantized dot is not currently implemented
                # so we cast rhs to the lhs dtype
                k = k.astype(q.dtype)
            uncapped_logits = plgpu.dot(q, k.T)  # [block_h, block_k]
            if k_scales_pages_ref is not None:
                # k_scales_pages_ref are one per head
                # they're laid out across the output dimension, so scale output
                k_scale = k_scales_pages_ref[block_tables].reshape((1, block_k))
                uncapped_logits *= k_scale.astype(uncapped_logits.dtype)
            if attn_logits_soft_cap is not None:
                logits = jnp.tanh(uncapped_logits / attn_logits_soft_cap)
                logits = logits * attn_logits_soft_cap
            else:
                logits = uncapped_logits

            if lengths_ref is not None:
                curr_start_page_idx = (
                    partition_idx * pages_per_partition
                    + start_k * pages_per_compute_block
                )
                curr_start_token_idx = curr_start_page_idx * page_size

                mask = jnp.arange(block_k) + curr_start_token_idx < lengths_ref[0]
                mask = lax.broadcast_in_dim(mask, (block_h, block_k), (1,))
                logits = jnp.where(mask, logits, mask_value)

            log2e = math.log2(math.e)
            m_curr = logits.max(axis=-1)
            m_next = jnp.maximum(m_prev, m_curr)
            correction = jnp.exp2((m_prev - m_next) * log2e)
            l_prev_corr = correction * l_prev
            s_curr = jnp.exp2((logits - m_next[:, None]) * log2e)
            l_curr = s_curr.sum(axis=-1)
            l_next = l_prev_corr + l_curr
            o_prev_corr = correction[:, None] * o_prev
            if v_scales_pages_ref is not None:
                # v_scales are 1 per head
                # they're laid out across the reduction dimension, so scale lhs
                v_scale = v_scales_pages_ref[block_tables].reshape((1, block_k))
                s_curr *= v_scale.astype(s_curr.dtype)
                # dynamic lhs quantized dot is not currently implemented
                # so we cast rhs to the lhs dtype
                v = v.astype(s_curr.dtype)
            o_curr = plgpu.dot(s_curr.astype(v.dtype), v)

            o_next = o_prev_corr + o_curr
            return o_next, m_next, l_next

        max_it = pl.cdiv(end_page_idx - start_page_idx, pages_per_compute_block)
        (o, m_i, l_i) = lax.fori_loop(0, max_it, body, (o, m_i, l_i))

        return o, m_i, l_i

    m_i = jnp.zeros(block_h, dtype=jnp.float32) + jnp.finfo(jnp.float32).min
    l_i = jnp.zeros(block_h, dtype=jnp.float32)
    o = jnp.zeros((block_h, head_dim), dtype=jnp.float32)

    start_page_idx = partition_idx * pages_per_partition
    end_page_idx = start_page_idx + pages_per_partition

    if lengths_ref is None:
        o, m_i, l_i = _compute(start_page_idx, end_page_idx, o, m_i, l_i)
    else:
        end_page_idx = jnp.minimum(pl.cdiv(lengths_ref[0], page_size), end_page_idx)

        o, m_i, l_i = jax.lax.cond(
            start_page_idx >= end_page_idx,
            lambda: (o, m_i, l_i),
            lambda: _compute(start_page_idx, end_page_idx, o, m_i, l_i),
        )

    o_ref[...] = o.astype(o_ref.dtype)

    if residual_refs is not None:
        l_ref, m_ref = residual_refs
        l_ref[...] = l_i
        m_ref[...] = m_i


def single_partition_paged_attention[
    HeadCount: IntVar,
    HeadDim: IntVar,
    TotalPages: IntVar,
    TokensPerPage: IntVar,
    TableLength: IntVar,
    ComputePages: IntVar,
](
    q: jax.Array[[HeadCount, HeadDim]],
    k_pages: jax.Array[[TotalPages, TokensPerPage, HeadDim]],
    v_pages: jax.Array[[TotalPages, TokensPerPage, HeadDim]],
    block_table: jax.Array[[TableLength]],
    pages_per_compute_block: Int[ComputePages],
) -> jax.Array[[HeadCount, HeadDim]]:
    def body(
        q_ref: pl.PagedQueryRef[HeadCount, HeadDim],
        k_ref: pl.PagedPoolRef[TotalPages, TokensPerPage, HeadDim],
        k_scales: None,
        v_ref: pl.PagedPoolRef[TotalPages, TokensPerPage, HeadDim],
        v_scales: None,
        table_ref: pl.PagedBlockTableRef[TableLength],
        lengths: None,
        out: pl.PagedOutputRef[HeadCount, HeadDim],
        residual_l: pl.PagedResidualRef[HeadCount],
        residual_m: pl.PagedResidualRef[HeadCount],
    ) -> None:
        paged_attention_kernel(
            q_ref,
            k_ref,
            k_scales,
            v_ref,
            v_scales,
            table_ref,
            lengths,
            out,
            residual_l,
            residual_m,
            num_heads=q.shape[0],
            pages_per_compute_block=pages_per_compute_block,
            mask_value=-1e4,
            attn_logits_soft_cap=None,
        )

    output, denominator, _ = pl.pallas_call(
        body,
        grid=(1, 1, 1),
        out_shape=(
            jax.ShapeDtypeStruct((q.shape[0], q.shape[1]), q.dtype),
            jax.ShapeDtypeStruct((q.shape[0],), jnp.float32),
            jax.ShapeDtypeStruct((q.shape[0],), jnp.float32),
        ),
        interpret=True,
    )(q, k_pages, None, v_pages, None, block_table, None)
    return output / denominator[:, None]


def test_correct_host_shapes[
    HeadCount: IntVar,
    HeadDim: IntVar,
    TotalPages: IntVar,
    TokensPerPage: IntVar,
    TableLength: IntVar,
    ComputePages: IntVar,
](
    q: jax.Array[[HeadCount, HeadDim]],
    keys: jax.Array[[TotalPages, TokensPerPage, HeadDim]],
    values: jax.Array[[TotalPages, TokensPerPage, HeadDim]],
    table: jax.Array[[TableLength]],
    compute_pages: Int[ComputePages],
) -> None:
    assert_type(
        single_partition_paged_attention(q, keys, values, table, compute_pages),
        jax.Array[[HeadCount, HeadDim]],
    )


def test_wrong_host_page_pool[
    HeadCount: IntVar,
    HeadDim: IntVar,
    TotalPages: IntVar,
    TokensPerPage: IntVar,
    TableLength: IntVar,
    ComputePages: IntVar,
    Other: IntVar,
](
    q: jax.Array[[HeadCount, HeadDim]],
    keys: jax.Array[[TotalPages, TokensPerPage, Other]],
    values: jax.Array[[TotalPages, Other, HeadDim]],
    table: jax.Array[[TableLength]],
    compute_pages: Int[ComputePages],
) -> None:
    single_partition_paged_attention(
        q,
        keys,  # E: is not assignable
        values,  # E: is not assignable
        table,
        compute_pages,
    )


def test_wrong_indirect_page_tile[
    TotalPages: IntVar,
    PageSize: IntVar,
    Dim: IntVar,
    PageBlock: IntVar,
    Other: IntVar,
](
    pages: pl.PagedPoolRef[TotalPages, PageSize, Dim],
    ids: pl.PagedPageIds[PageBlock],
    wrong_block_k: Int[Other * PageSize],
    dim: Int[Dim],
) -> None:
    pages[ids].reshape(wrong_block_k, dim)  # E: is not assignable


def test_wrong_gathered_key_dimension[
    Heads: IntVar,
    Dim: IntVar,
    Other: IntVar,
    PageBlock: IntVar,
    PageSize: IntVar,
](
    q: pl.Tile[[Heads, Dim]],
    keys: pl.Tile[[PageBlock * PageSize, Other]],
) -> None:
    plgpu.dot(q, keys.T)  # E: is not assignable


def test_wrong_gathered_value_block[
    Heads: IntVar,
    Dim: IntVar,
    Other: IntVar,
    PageBlock: IntVar,
    PageSize: IntVar,
](
    probabilities: pl.Tile[[Heads, PageBlock * PageSize]],
    values: pl.Tile[[Other, Dim]],
) -> None:
    plgpu.dot(probabilities, values)  # E: is not assignable


def test_wrong_length_mask_shape[Heads: IntVar, Tokens: IntVar, Other: IntVar](
    mask: pl.PagedAttentionMask[Heads, Other],
    logits: pl.Tile[[Heads, Tokens]],
) -> None:
    jnp.where(mask, logits, -1e4)  # E: No matching overload


def test_wrong_output_feature_extent[Heads: IntVar, Dim: IntVar, Other: IntVar](
    out: pl.PagedOutputRef[Heads, Other],
    value: pl.Tile[[Heads, Dim]],
) -> None:
    out[...] = value  # E: Cannot set item


if not TYPE_CHECKING:
    import unittest

    import numpy as np

    class PagedAttentionBoundaryTest(unittest.TestCase):
        def test_indexed_pages_and_softmax(self) -> None:
            q = jnp.arange(16 * 32, dtype=jnp.float32).reshape((16, 32)) / 512
            k_pages = (
                jnp.arange(3 * 16 * 32, dtype=jnp.float32).reshape((3, 16, 32)) / 100
            )
            v_pages = jnp.sin(
                jnp.arange(3 * 16 * 32, dtype=jnp.float32).reshape((3, 16, 32)) / 30
            )
            block_table = jnp.array([2, 0], dtype=jnp.int32)
            output = single_partition_paged_attention(
                q, k_pages, v_pages, block_table, 1
            )
            k = np.asarray(k_pages)[np.asarray(block_table)].reshape((32, 32))
            v = np.asarray(v_pages)[np.asarray(block_table)].reshape((32, 32))
            scores = np.asarray(q) @ k.T
            probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            np.testing.assert_allclose(
                np.asarray(output), probabilities @ v, rtol=2e-5, atol=2e-6
            )

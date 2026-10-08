# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# @lint-ignore-every AUTODEPS2

"""Check Triton's tutorial 06 forward kernel and its host attention contract."""

from __future__ import annotations

import os
import unittest
from typing import TYPE_CHECKING, Any, Literal, assert_type, cast

import torch
import triton
import triton.language as tl
from shape_extensions import Int, IntListLiteral, IntVar
from triton.backends.compiler import GPUTarget

from triton_examples.testing import compile_ttir
from triton_library.launch_layout import attention_output
from triton_library.semantic_jit import ConstExpr, semantic_jit
from triton_library.torch_views import checked_attention_input, checked_attention_stats

BM = IntVar("BM")
BN = IntVar("BN")
D = IntVar("D")
Y = IntVar("Y")
NC = IntVar("NC")
R = IntVar("R")
C = IntVar("C")
S = IntVar("S")
BR = IntVar("BR")
BC = IntVar("BC")
ZDim = IntVar("ZDim")
HDim = IntVar("HDim")
NDim = IntVar("NDim")


@semantic_jit
def _attn_fwd_inner(
    acc: tl.tensor[[BM, D]],
    l_i: tl.tensor[[BM]],
    m_i: tl.tensor[[BM]],
    q: tl.tensor[[BM, D]],  #
    desc_k: tl.tensor_descriptor[Y, D, D, BN, D],
    desc_v: tl.tensor_descriptor[Y, D, D, BN, D],  #
    offset_y: int,
    dtype: ConstExpr[object],
    start_m: tl.ProgramId[Literal[0]],
    qk_scale: float,  #
    BLOCK_M: ConstExpr[Int[BM]],
    HEAD_DIM: ConstExpr[Int[D]],
    BLOCK_N: ConstExpr[Int[BN]],  #
    STAGE: ConstExpr[int],
    offs_m: tl.Offsets[[BM], 1, str, Literal[0]],
    offs_n: tl.Offsets[[BN]],  #
    N_CTX: ConstExpr[Int[NC]],
    warp_specialize: ConstExpr[bool],
    IS_HOPPER: ConstExpr[bool],
):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    offsetk_y = offset_y + lo
    if dtype == tl.float8e5:
        offsetv_y = offset_y * HEAD_DIM + lo
    else:
        offsetv_y = offset_y + lo
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        # -- update output accumulator --
        if not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = (
                acc.reshape([BM, 2, BN // 2])  # pyrefly: ignore[no-matching-overload]
                .permute(0, 2, 1)
                .split()
            )
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = (
                tl.join(acc0, acc1)  # pyrefly: ignore[missing-attribute]
                .permute(0, 2, 1)
                .reshape([BM, BN])
            )
        else:
            acc = acc * alpha[:, None]
        # prepare p and v for the dot
        if dtype == tl.float8e5:
            v = desc_v.load([0, offsetv_y]).T
        else:
            v = desc_v.load([offsetv_y, 0])
        p = p.to(dtype)
        # note that this non transposed v for FP8 is only supported on Blackwell
        acc = tl.dot(p, v, acc)  # pyrefly: ignore[bad-argument-type]
        # update m_i and l_i
        # place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i


@semantic_jit
def _maybe_make_tensor_desc(
    desc_or_ptr: tl.tensor_descriptor[R, C, S, BR, BC] | tl.AttentionPointer[R, C, S],
    shape: IntListLiteral[[R, C]],
    strides: IntListLiteral[[S, 1]],
    block_shape: IntListLiteral[[BR, BC]],
) -> tl.tensor_descriptor[R, C, S, BR, BC]:
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@semantic_jit
def _attn_fwd(
    sm_scale: float,
    M: tl.AttentionStatsPointer[ZDim * HDim, NDim],  #
    Z: ConstExpr[Int[ZDim]],
    H: ConstExpr[Int[HDim]],
    desc_q: tl.AttentionPointer[ZDim * HDim * NDim, D, D]
    | tl.tensor_descriptor[ZDim * HDim * NDim, D, D, BM, D],
    desc_k: tl.AttentionPointer[ZDim * HDim * NDim, D, D]
    | tl.tensor_descriptor[ZDim * HDim * NDim, D, D, BN, D],
    desc_v: tl.AttentionPointer[ZDim * HDim * NDim, D, D]
    | tl.tensor_descriptor[ZDim * HDim * NDim, D, D, BN, D],
    desc_o: tl.AttentionPointer[ZDim * HDim * NDim, D, D]
    | tl.tensor_descriptor[ZDim * HDim * NDim, D, D, BM, D],
    N_CTX: ConstExpr[Int[NDim]],  #
    HEAD_DIM: ConstExpr[Int[D]],  #
    BLOCK_M: ConstExpr[Int[BM]],  #
    BLOCK_N: ConstExpr[Int[BN]],  #
    FP8_OUTPUT: ConstExpr[Literal[False]],  #
    STAGE: ConstExpr[int],  #
    warp_specialize: ConstExpr[bool],  #
    IS_HOPPER: ConstExpr[bool],  #
):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(
        desc_q,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    if FP8_OUTPUT:
        desc_v = _maybe_make_tensor_desc(
            desc_v,
            shape=[HEAD_DIM, y_dim],
            strides=[N_CTX, 1],
            block_shape=[HEAD_DIM, BLOCK_N],
        )
    else:
        desc_v = _maybe_make_tensor_desc(
            desc_v,
            shape=[y_dim, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=[BLOCK_N, HEAD_DIM],
        )
    desc_k = _maybe_make_tensor_desc(
        desc_k,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_o = _maybe_make_tensor_desc(
        desc_o,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = desc_q.load([qo_offset_y, 0])
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,  #
            desc_k,
            desc_v,  #
            offset_y,
            dtype,
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            4 - STAGE,
            offs_m,
            offs_n,
            N_CTX,  #
            warp_specialize,
            IS_HOPPER,
        )
    # stage 2: on-band
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,  #
            desc_k,
            desc_v,  #
            offset_y,
            dtype,
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            2,
            offs_m,
            offs_n,
            N_CTX,  #
            warp_specialize,
            IS_HOPPER,
        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))


def attention_forward[Batch: IntVar, Heads: IntVar, Tokens: IntVar, Dim: IntVar](
    q: torch.Tensor[[Batch, Heads, Tokens, Dim]],
    k: torch.Tensor[[Batch, Heads, Tokens, Dim]],
    v: torch.Tensor[[Batch, Heads, Tokens, Dim]],
    *,
    block_m: int = 16,
    block_n: int = 16,
    sm_scale: float = 0.5,
) -> tuple[
    torch.Tensor[[Batch, Heads, Tokens, Dim]],
    torch.Tensor[[Batch, Heads, Tokens]],
]:
    """Check contiguous equal-length half-precision self-attention before launch."""
    layout = attention_output(q, k, v, block_m=block_m, block_n=block_n)
    if q.device.type != "cuda":
        raise ValueError("Descriptor attention requires a CUDA GPU for execution")
    batch, heads, tokens, dim = q.shape
    output = torch.empty_like(q)
    stats = torch.empty((batch, heads, tokens), dtype=torch.float32, device=q.device)
    q_ptr = checked_attention_input(q, q.shape)
    k_ptr = checked_attention_input(k, q.shape)
    v_ptr = checked_attention_input(v, q.shape)
    out_ptr = checked_attention_input(output, q.shape)
    stats_ptr = checked_attention_stats(stats, (batch, heads, tokens))
    layout.launch(
        _attn_fwd,
        sm_scale,
        stats_ptr,
        batch,
        heads,
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        tokens,
        dim,
        block_m,
        block_n,
        False,
        1,
        False,
        False,
    )
    return output, stats


class AttentionForwardTest(unittest.TestCase):
    """Compile the unchanged descriptor-based attention path without a GPU."""

    def test_frontend(self) -> None:
        if os.environ.get("TRITON_INTERPRET") == "1":
            self.skipTest("Frontend compilation uses normal JIT mode")
        ir = compile_ttir(
            _attn_fwd,
            signature={
                "sm_scale": "fp32",
                "M": "*fp32",
                "desc_q": "*fp16",
                "desc_k": "*fp16",
                "desc_v": "*fp16",
                "desc_o": "*fp16",
            },
            constexprs={
                "Z": 1,
                "H": 1,
                "N_CTX": 16,
                "HEAD_DIM": 16,
                "BLOCK_M": 16,
                "BLOCK_N": 16,
                "FP8_OUTPUT": 0,
                "STAGE": 1,
                "warp_specialize": 0,
                "IS_HOPPER": 0,
            },
            target=GPUTarget("cuda", 90, 32),
        )
        self.assertIn("tt.func", ir)

    def test_checked_host_axes_and_grid(self) -> None:
        q = torch.ones((2, 1, 32, 16), dtype=torch.float16)
        layout = attention_output(q, q, q, block_m=16, block_n=16)
        self.assertEqual(layout.grid, (2, 2))
        self.assertIs(checked_attention_input(q, (2, 1, 32, 16)), q)
        stats = torch.empty((2, 1, 32), dtype=torch.float32)
        self.assertIs(checked_attention_stats(stats, (2, 1, 32)), stats)
        ptr = checked_attention_input(q, (2, 1, 32, 16))
        stats_ptr = checked_attention_stats(stats, (2, 1, 32))
        with self.assertRaisesRegex(ValueError, "HEAD_DIM"):
            layout.launch(
                _attn_fwd,
                0.5,
                stats_ptr,
                2,
                1,
                ptr,
                ptr,
                ptr,
                ptr,
                32,
                cast(Any, 32),
                16,
                16,
                False,
                1,
                False,
                False,
            )
        with self.assertRaisesRegex(ValueError, "key, and value shapes"):
            attention_output(q, cast(Any, q[:, :, :16]), q, block_m=16, block_n=16)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            checked_attention_input(q, cast(Any, (1, 2, 32, 16)))
        with self.assertRaisesRegex(ValueError, "statistics"):
            checked_attention_stats(stats, cast(Any, (1, 2, 32)))
        with self.assertRaisesRegex(ValueError, "divide tokens"):
            attention_output(q, q, q, block_m=64, block_n=16)
        with self.assertRaisesRegex(ValueError, "CUDA GPU"):
            attention_forward(q, q, q)


if TYPE_CHECKING:

    def typed_attention[
        Batch: IntVar,
        Heads: IntVar,
        Tokens: IntVar,
        Dim: IntVar,
        Other: IntVar,
    ](
        q: torch.Tensor[[Batch, Heads, Tokens, Dim]],
        k: torch.Tensor[[Batch, Heads, Tokens, Dim]],
        v: torch.Tensor[[Batch, Heads, Tokens, Dim]],
        wrong: torch.Tensor[[Batch, Heads, Other, Dim]],
    ) -> None:
        assert_type(
            attention_forward(q, k, v),
            tuple[
                torch.Tensor[[Batch, Heads, Tokens, Dim]],
                torch.Tensor[[Batch, Heads, Tokens]],
            ],
        )
        attention_forward(q, wrong, v)  # pyrefly: ignore[bad-argument-type]

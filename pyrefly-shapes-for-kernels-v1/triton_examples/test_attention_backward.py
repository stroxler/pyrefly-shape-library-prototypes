# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The kernel body is copied from Triton's 06-fused-attention.py.
# Triton's LICENSE includes the following notice:
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Check the two-helper, two-scan Triton attention backward kernel."""

from __future__ import annotations

import os
import unittest
from typing import TYPE_CHECKING, Any, Literal, assert_type, cast

import torch
import triton.language as tl
from shape_extensions import Int, IntVar

from triton_examples.testing import compile_ttir
from triton_library import tlt
from triton_library.launch_layout import attention_backward_output
from triton_library.semantic_jit import ConstExpr, semantic_jit
from triton_library.torch_views import (
    checked_attention_backward_input,
    checked_attention_backward_output,
    checked_attention_backward_stats,
)

Batch = IntVar("Batch")
Heads = IntVar("Heads")
Tokens = IntVar("Tokens")
Dim = IntVar("Dim")
BM1 = IntVar("BM1")
BN1 = IntVar("BN1")
BM2 = IntVar("BM2")
BN2 = IntVar("BN2")
BM = IntVar("BM")
BN = IntVar("BN")
TokenStride = IntVar("TokenStride")
FeatureStride = IntVar("FeatureStride")
SliceFactor = IntVar("SliceFactor")

type _SelectedHeadIn[T: IntVar, D: IntVar, TS: IntVar, FS: IntVar] = (
    tl.SelectedInPointer[
        [int, int, T, D],
        [int, int, TS, FS],
        [T, D],
        [TS, FS],
        Literal["grouped"],
    ]
)
type _SelectedHeadOut[T: IntVar, D: IntVar, TS: IntVar, FS: IntVar] = (
    tl.SelectedOutPointer[
        [int, int, T, D],
        [int, int, TS, FS],
        [T, D],
        [TS, FS],
        Literal["grouped"],
    ]
)
type _SelectedStatsIn[T: IntVar] = tl.SelectedInPointer[
    [int, int, T], [int, T, 1], [T], [1], Literal["row"]
]


@semantic_jit
def _attn_bwd_dkdv(
    dk: tl.tensor[[BN, Dim]],
    dv: tl.tensor[[BN, Dim]],  #
    Q: _SelectedHeadIn[Tokens, Dim, TokenStride, FeatureStride],
    k: tl.tensor[[BN, Dim]],
    v: tl.tensor[[BN, Dim]],
    sm_scale: float,  #
    DO: _SelectedHeadIn[Tokens, Dim, TokenStride, FeatureStride],  #
    M: _SelectedStatsIn[Tokens],
    D: _SelectedStatsIn[Tokens],  #
    stride_tok: Int[TokenStride],
    stride_d: Int[FeatureStride],  #
    H: tl.GroupSize[Heads],
    N_CTX: Int[Tokens],
    BLOCK_M1: ConstExpr[Int[BM]],  #
    BLOCK_N1: ConstExpr[Int[BN]],  #
    HEAD_DIM: ConstExpr[Int[Dim]],  #
    start_n: int,
    start_m: int,
    num_steps: int,  #
    MASK: ConstExpr[bool],
):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = offs_m[None, :] >= offs_n[:, None]
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(tl.float16)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)
        dk += tl.dot(dsT, tl.trans(qT))
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    return dk, dv


@semantic_jit
def _attn_bwd_dq(
    dq: tl.tensor[[BM, Dim]],
    q: tl.tensor[[BM, Dim]],
    K: _SelectedHeadIn[Tokens, Dim, TokenStride, FeatureStride],
    V: _SelectedHeadIn[Tokens, Dim, TokenStride, FeatureStride],  #
    do: tl.tensor[[BM, Dim]],
    m: tl.tensor[[BM, 1]],
    D: _SelectedStatsIn[Tokens],
    # shared by Q/K/V/DO.
    stride_tok: Int[TokenStride],
    stride_d: Int[FeatureStride],  #
    H: tl.GroupSize[Heads],
    N_CTX: Int[Tokens],  #
    BLOCK_M2: ConstExpr[Int[BM]],  #
    BLOCK_N2: ConstExpr[Int[BN]],  #
    HEAD_DIM: ConstExpr[Int[Dim]],
    # Filled in by the wrapper.
    start_m: int,
    start_n: int,
    num_steps: int,  #
    MASK: ConstExpr[bool],
):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = offs_m[:, None] >= offs_n[None, :]
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.float16)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq


@semantic_jit
def _attn_bwd(
    Q: tlt.InPointer[[Batch, Heads, Tokens, Dim], [int, int, Dim, 1]],
    K: tlt.InPointer[[Batch, Heads, Tokens, Dim], [int, int, Dim, 1]],
    V: tlt.InPointer[[Batch, Heads, Tokens, Dim], [int, int, Dim, 1]],
    sm_scale: float,
    DO: tlt.InPointer[[Batch, Heads, Tokens, Dim], [int, int, Dim, 1]],
    DQ: tlt.OutPointer[[Batch, Heads, Tokens, Dim], [int, int, Dim, 1]],
    DK: tlt.OutPointer[[Batch, Heads, Tokens, Dim], [int, int, Dim, 1]],
    DV: tlt.OutPointer[[Batch, Heads, Tokens, Dim], [int, int, Dim, 1]],
    M: tlt.InPointer[[Batch, Heads, Tokens], [int, Tokens, 1]],
    D: tlt.InPointer[[Batch, Heads, Tokens], [int, Tokens, 1]],
    stride_z: tl.AxisStride[Heads, int, Literal["quotient"]],
    stride_h: tl.AxisStride[Heads, int, Literal["remainder"]],
    stride_tok: Int[Dim],
    stride_d: Int[1],
    H: tl.GroupSize[Heads],
    N_CTX: Int[Tokens],
    BLOCK_M1: ConstExpr[Int[BM1]],
    BLOCK_N1: ConstExpr[Int[BN1]],
    BLOCK_M2: ConstExpr[Int[BM2]],
    BLOCK_N2: ConstExpr[Int[BN2]],
    BLK_SLICE_FACTOR: ConstExpr[Int[SliceFactor]],
    HEAD_DIM: ConstExpr[Int[Dim]],
    CAUSAL: ConstExpr[bool],
):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)  # pyrefly: ignore[bad-assignment]

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    # Upstream rebinds pointers here; selected heads have lower-rank types.
    q_head = Q + adj
    k_head = K + adj
    v_head = V + adj
    do_head = DO + adj
    dq_head = DQ + adj
    dk_head = DK + adj
    dv_head = DV + adj
    m_head = M + off_chz
    delta_head = D + off_chz

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1
    start_m = 0

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(k_head + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(v_head + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    if CAUSAL:
        start_m = start_n
        num_steps = BLOCK_N1 // MASK_BLOCK_M1
        dk, dv = _attn_bwd_dkdv(
            dk,
            dv,  #
            q_head,
            k,
            v,
            sm_scale,  #
            do_head,  #
            m_head,
            delta_head,  #
            stride_tok,
            stride_d,  #
            H,
            N_CTX,  #
            MASK_BLOCK_M1,
            BLOCK_N1,
            HEAD_DIM,  #
            start_n,
            start_m,
            num_steps,  #
            MASK=True,  #
        )

        start_m += num_steps * MASK_BLOCK_M1

    # Compute dK and dV for non-masked blocks.
    num_steps = (N_CTX - start_m) // BLOCK_M1
    dk, dv = _attn_bwd_dkdv(  #
        dk,
        dv,  #
        q_head,
        k,
        v,
        sm_scale,  #
        do_head,  #
        m_head,
        delta_head,  #
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        BLOCK_M1,
        BLOCK_N1,
        HEAD_DIM,  #
        start_n,
        start_m,
        num_steps,  #
        MASK=False,  #
    )

    dv_ptrs = dv_head + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = dk_head + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # THIS BLOCK DOES dq_head:
    start_m = pid * BLOCK_M2
    start_n = 0
    num_steps = N_CTX // BLOCK_N2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(q_head + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(do_head + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(m_head + offs_m)
    m = m[:, None]

    if CAUSAL:
        # Compute dQ for masked (diagonal) blocks.
        # NOTE: This code scans each row of QK^T backward (from right to left,
        # but inside each call to _attn_bwd_dq, from left to right), but that's
        # not due to anything important.  I just wanted to reuse the loop
        # structure for dK & dV above as much as possible.
        end_n = start_m + BLOCK_M2
        num_steps = BLOCK_M2 // MASK_BLOCK_N2
        dq = _attn_bwd_dq(
            dq,
            q,
            k_head,
            v_head,  #
            do,
            m,
            delta_head,  #
            stride_tok,
            stride_d,  #
            H,
            N_CTX,  #
            BLOCK_M2,
            MASK_BLOCK_N2,
            HEAD_DIM,  #
            start_m,
            end_n - num_steps * MASK_BLOCK_N2,
            num_steps,  #
            MASK=True,  #
        )
        end_n -= num_steps * MASK_BLOCK_N2
        # stage 2
        num_steps = end_n // BLOCK_N2
        start_n = end_n - num_steps * BLOCK_N2

    dq = _attn_bwd_dq(
        dq,
        q,
        k_head,
        v_head,  #
        do,
        m,
        delta_head,  #
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        BLOCK_M2,
        BLOCK_N2,
        HEAD_DIM,  #
        start_m,
        start_n,
        num_steps,  #
        MASK=False,  #
    )
    # Write back dQ.
    dq_ptrs = dq_head + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


def attention_backward[B: IntVar, H: IntVar, T: IntVar, D: IntVar](
    q: torch.Tensor[[B, H, T, D]],
    k: torch.Tensor[[B, H, T, D]],
    v: torch.Tensor[[B, H, T, D]],
    dout: torch.Tensor[[B, H, T, D]],
    m: torch.Tensor[[B, H, T]],
    delta: torch.Tensor[[B, H, T]],
    *,
    block_m1: int,
    block_n1: int,
    block_m2: int,
    block_n2: int,
    slice_factor: int,
) -> tuple[
    torch.Tensor[[B, H, T, D]],
    torch.Tensor[[B, H, T, D]],
    torch.Tensor[[B, H, T, D]],
]:
    """Bind three gradients to contiguous attention inputs and saved statistics."""
    if q.ndim != 4:
        raise ValueError("Attention backward expects [batch, heads, tokens, dim]")
    batch, heads, tokens, dim = q.shape
    dq, dk, dv = (torch.empty_like(q), torch.empty_like(q), torch.empty_like(q))
    layout = attention_backward_output(
        q,
        k,
        v,
        dout,
        m,
        delta,
        dq,
        dk,
        dv,
        block_m1=block_m1,
        block_n1=block_n1,
        block_m2=block_m2,
        block_n2=block_n2,
        slice_factor=slice_factor,
    )
    shape = (batch, heads, tokens, dim)
    stats_shape = (batch, heads, tokens)
    # The layout has verified these dense strides against the tensor allocation.
    stride_z = cast('tl.AxisStride[H, int, Literal["quotient"]]', q.stride(0))
    stride_h = cast('tl.AxisStride[H, int, Literal["remainder"]]', q.stride(1))
    head_count = cast("tl.GroupSize[H]", heads)
    layout.launch(
        _attn_bwd,
        checked_attention_backward_input(q, shape),
        checked_attention_backward_input(k, shape),
        checked_attention_backward_input(v, shape),
        1.0,
        checked_attention_backward_input(dout, shape),
        checked_attention_backward_output(dq, shape),
        checked_attention_backward_output(dk, shape),
        checked_attention_backward_output(dv, shape),
        checked_attention_backward_stats(m, stats_shape),
        checked_attention_backward_stats(delta, stats_shape),
        stride_z,
        stride_h,
        dim,
        1,
        head_count,
        tokens,
        block_m1,
        block_n1,
        block_m2,
        block_n2,
        slice_factor,
        dim,
        False,
    )
    return dq, dk, dv


class AttentionBackwardTest(unittest.TestCase):
    """Run the full two-helper kernel through Triton's frontend."""

    def test_frontend(self) -> None:
        if os.environ.get("TRITON_INTERPRET") == "1":
            self.skipTest("Frontend compilation uses normal JIT mode")
        ir = compile_ttir(
            _attn_bwd,
            signature={
                name: "*fp16" for name in ("Q", "K", "V", "DO", "DQ", "DK", "DV")
            }
            | {"M": "*fp32", "D": "*fp32", "sm_scale": "fp32"}
            | {
                name: "i32"
                for name in (
                    "stride_z",
                    "stride_h",
                    "stride_tok",
                    "stride_d",
                    "H",
                    "N_CTX",
                )
            },
            constexprs={
                "BLOCK_M1": 16,
                "BLOCK_N1": 32,
                "BLOCK_M2": 32,
                "BLOCK_N2": 16,
                "BLK_SLICE_FACTOR": 2,
                "HEAD_DIM": 16,
                "CAUSAL": 0,
            },
        )
        self.assertIn("tt.func", ir)

    def test_reject_wrong_axes_and_grid(self) -> None:
        q = torch.ones((1, 1, 32, 16), dtype=torch.float16)
        stats = torch.ones((1, 1, 32), dtype=torch.float32)
        args = dict(
            block_m1=16,
            block_n1=32,
            block_m2=32,
            block_n2=16,
            slice_factor=2,
        )
        with self.assertRaisesRegex(ValueError, "shapes, dtypes, devices"):
            attention_backward(q, q, q, cast(Any, q[:, :, :16]), stats, stats, **args)
        with self.assertRaisesRegex(ValueError, "share a grid"):
            attention_backward(q, q, q, q, stats, stats, **(args | {"block_m2": 16}))

    def test_cpu_gradients(self) -> None:
        if os.environ.get("TRITON_INTERPRET") != "1":
            self.skipTest("CPU launches use interpreter mode")
        torch.manual_seed(0)
        q = (torch.randn((1, 1, 32, 16)) * 0.2).to(torch.float16)
        k = (torch.randn((1, 1, 32, 16)) * 0.2).to(torch.float16)
        v = torch.randn((1, 1, 32, 16)).to(torch.float16)
        dout = (torch.randn((1, 1, 32, 16)) * 0.2).to(torch.float16)
        qr = q.float().detach().requires_grad_()
        kr = k.float().detach().requires_grad_()
        vr = v.float().detach().requires_grad_()
        scores = qr @ kr.transpose(-1, -2)
        probabilities = scores.softmax(dim=-1)
        output = probabilities @ vr
        (output * dout.float()).sum().backward()
        m = torch.logsumexp(scores.detach(), dim=-1) / 0.6931471805599453
        delta = (output.detach().half().float() * dout.float()).sum(dim=-1)
        actual = attention_backward(
            q,
            k,
            v,
            dout,
            m.contiguous(),
            delta.contiguous(),
            block_m1=16,
            block_n1=32,
            block_m2=32,
            block_n2=16,
            slice_factor=2,
        )
        for got, want in zip(actual, (qr.grad, kr.grad, vr.grad), strict=True):
            torch.testing.assert_close(got.float(), want, rtol=0.15, atol=0.05)


if TYPE_CHECKING:

    def typed_logical_mask_alignment[Rows: IntVar, Cols: IntVar, Other: IntVar](
        row_indices: tl.RowAxisOffsets[Rows],
        column_indices: tl.ColumnAxisOffsets[Cols],
        values: tl.tensor[[Rows, Cols]],
        loaded_condition: tl.tensor[[Rows, Cols]],
        wrong_values: tl.tensor[[Rows, Other]],
        pointers: tl.InTilePointers[
            [Rows, Cols], [Cols, 1], [Rows, Cols], Literal["indexed"]
        ],
    ) -> None:
        logical_mask = column_indices >= row_indices
        assert_type(logical_mask, tl.LogicalMask[[Rows, Cols]])
        assert_type(tl.where(logical_mask, values, 0.0), tl.tensor[[Rows, Cols]])
        assert_type(tl.where(loaded_condition, values, 0.0), tl.tensor[[Rows, Cols]])
        tl.where(logical_mask, wrong_values, 0.0)  # pyrefly: ignore[no-matching-overload]
        tl.where(loaded_condition, wrong_values, 0.0)  # pyrefly: ignore[no-matching-overload]
        tl.load(pointers, mask=logical_mask)  # pyrefly: ignore[no-matching-overload]

    def typed_selected_tiles[
        Tokens: IntVar,
        Dim: IntVar,
        TS: IntVar,
        FS: IntVar,
        Block: IntVar,
        Other: IntVar,
    ](
        source: _SelectedHeadIn[Tokens, Dim, TS, FS],
        destination: _SelectedHeadOut[Tokens, Dim, TS, FS],
        rows: tl.RowAddress[Block, TS],
        columns: tl.ColumnAddress[Block, TS],
        features: tl.ColumnAddress[Dim, FS],
        transpose_features: tl.RowAddress[Dim, FS],
        wrong_features: tl.ColumnAddress[Dim, Other],
        step: Int[Block * TS],
        wrong_step: Int[Block * FS],
        values: tl.tensor[[Block, Dim]],
        wrong_values: tl.tensor[[Block, Other]],
    ) -> None:
        forward = source + rows + features
        assert_type(
            forward,
            tl.InTilePointers[
                [Tokens, Dim], [TS, FS], [Block, Dim], Literal["selected_unchecked_0"]
            ],
        )
        assert_type(tl.load(forward), tl.tensor[[Block, Dim]])
        assert_type(
            tl.load(source + columns + transpose_features), tl.tensor[[Dim, Block]]
        )
        assert_type(
            forward.__iadd__(step),
            tl.InTilePointers[
                [Tokens, Dim], [TS, FS], [Block, Dim], Literal["selected_unchecked_0"]
            ],
        )
        forward.__iadd__(wrong_step)  # pyrefly: ignore[no-matching-overload]
        source + rows + wrong_features  # pyrefly: ignore[unsupported-operation]
        output = destination + rows + features
        assert_type(
            output,
            tl.OutTilePointers[
                [Tokens, Dim], [TS, FS], [Block, Dim], Literal["selected_unchecked_0"]
            ],
        )
        tl.store(output, values)
        tl.store(output, wrong_values)  # pyrefly: ignore[no-matching-overload]

    def typed_grouped_address[H: IntVar, Other: IntVar, SZ: IntVar, SH: IntVar](
        pid: tl.ProgramId[Literal[2]],
        count: tl.GroupSize[H],
        other_count: tl.GroupSize[Other],
        head_stride: tl.AxisStride[H, SH, Literal["remainder"]],
        batch_stride: tl.AxisStride[H, SZ, Literal["quotient"]],
    ) -> None:
        head = pid % count
        batch = pid // count
        assert_type(head, tl.GroupIndex[H])
        assert_type(batch, tl.GroupQuotient[H])
        assert_type(
            head_stride * head + batch_stride * batch,
            tl.CombinedAddress[H, [SZ, SH]],
        )
        head_stride * batch  # pyrefly: ignore[unsupported-operation]
        batch_stride * head  # pyrefly: ignore[unsupported-operation]
        head_stride * (pid % other_count)  # pyrefly: ignore[unsupported-operation]

    def typed_head_address[B: IntVar, H: IntVar, T: IntVar, D: IntVar, Other: IntVar](
        q: tlt.InPointer[[B, H, T, D], [int, int, D, 1]],
        correct_head: tl.CombinedAddress[H, [int, int]],
        wrong_head: tl.CombinedAddress[Other, [int, int]],
    ) -> None:
        assert_type(
            q + correct_head,
            tl.SelectedInPointer[
                [B, H, T, D],
                [int, int, D, 1],
                [T, D],
                [D, 1],
                Literal["grouped"],
            ],
        )
        q + wrong_head  # pyrefly: ignore[unsupported-operation]

    def typed_selected_stats[
        Batch: IntVar,
        Heads: IntVar,
        Tokens: IntVar,
        Block: IntVar,
        Other: IntVar,
    ](
        source: tlt.InPointer[[Batch, Heads, Tokens], [int, Tokens, 1]],
        destination: tlt.OutPointer[[Batch, Heads, Tokens], [int, Tokens, 1]],
        row: tl.TileStart[[Tokens], Literal[2]],
        good: tl.Offsets[[Block], [1], Literal["program"], Literal[0]],
        wrong_grid: tl.Offsets[[Block], [1], Literal["program"], Literal[1]],
        wrong_step: tl.Offsets[[Block], [Other], Literal["program"], Literal[0]],
        values: tl.tensor[[Block]],
        wrong_values: tl.tensor[[Other]],
    ) -> None:
        selected_input = source + row
        assert_type(
            selected_input,
            tl.SelectedInPointer[
                [Batch, Heads, Tokens],
                [int, Tokens, 1],
                [Tokens],
                [1],
                Literal["row"],
            ],
        )
        assert_type(tl.load(selected_input + good), tl.tensor[[Block]])
        selected_input + wrong_step  # pyrefly: ignore[unsupported-operation]
        selected_output = destination + row
        assert_type(tl.store(selected_output + good, values), None)
        selected_output + wrong_grid  # pyrefly: ignore[unsupported-operation]
        tl.store(selected_output + good, wrong_values)  # pyrefly: ignore[no-matching-overload]

    def typed_host_boundary[B: IntVar, H: IntVar, T: IntVar, D: IntVar, Other: IntVar](
        q: torch.Tensor[[B, H, T, D]],
        wrong: torch.Tensor[[B, H, Other, D]],
        stats: torch.Tensor[[B, H, T]],
    ) -> None:
        assert_type(
            attention_backward(
                q,
                q,
                q,
                q,
                stats,
                stats,
                block_m1=16,
                block_n1=32,
                block_m2=32,
                block_n2=16,
                slice_factor=2,
            ),
            tuple[
                torch.Tensor[[B, H, T, D]],
                torch.Tensor[[B, H, T, D]],
                torch.Tensor[[B, H, T, D]],
            ],
        )
        attention_backward(
            q,
            q,
            q,
            wrong,  # pyrefly: ignore[bad-argument-type]
            stats,
            stats,
            block_m1=16,
            block_n1=32,
            block_m2=32,
            block_n2=16,
            slice_factor=2,
        )

# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
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

"""Check Triton's unmasked attention output-gradient preprocessing pass."""

from __future__ import annotations

import os
import unittest
from typing import TYPE_CHECKING, Any, assert_type, cast

import torch
import triton.language as tl
from shape_extensions import Int, IntVar

from triton_examples.testing import compile_ttir
from triton_library import tlt
from triton_library.launch_layout import attention_preprocess_output
from triton_library.semantic_jit import ConstExpr, semantic_jit
from triton_library.torch_views import (
    checked_attention_backward_delta,
    checked_attention_backward_input,
)

Batch = IntVar("Batch")
Heads = IntVar("Heads")
Tokens = IntVar("Tokens")
Dim = IntVar("Dim")
BM = IntVar("BM")


@semantic_jit
def _attn_bwd_preprocess(
    O: tlt.InPointer[[Batch, Heads, Tokens, Dim], [int, int, Dim, 1]],
    DO: tlt.InPointer[[Batch, Heads, Tokens, Dim], [int, int, Dim, 1]],  #
    Delta: tlt.OutPointer[[Batch, Heads, Tokens], [int, Tokens, 1]],  #
    Z: Int[Batch],
    H: Int[Heads],
    N_CTX: Int[Tokens],  #
    BLOCK_M: ConstExpr[Int[BM]],
    HEAD_DIM: ConstExpr[Int[Dim]],  #
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(
        O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]
    )
    do = tl.load(
        DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


def attention_backward_delta[Batch: IntVar, Heads: IntVar, Tokens: IntVar, Dim: IntVar](
    output: torch.Tensor[[Batch, Heads, Tokens, Dim]],
    dout: torch.Tensor[[Batch, Heads, Tokens, Dim]],
    *,
    block_m: int,
) -> torch.Tensor[[Batch, Heads, Tokens]]:
    """Check dense attention axes and run a whole-block, unmasked reduction."""
    if output.ndim != 4:
        raise ValueError("Attention backward needs [batch, heads, tokens, dim]")
    batch, heads, tokens, dim = output.shape
    delta = torch.empty(
        (batch, heads, tokens), dtype=torch.float32, device=output.device
    )
    layout = attention_preprocess_output(output, dout, delta, block_m=block_m)
    shape = (batch, heads, tokens, dim)
    o_ptr = checked_attention_backward_input(output, shape)
    do_ptr = checked_attention_backward_input(dout, shape)
    delta_ptr = checked_attention_backward_delta(delta, (batch, heads, tokens))
    layout.launch(
        _attn_bwd_preprocess,
        o_ptr,
        do_ptr,
        delta_ptr,
        batch,
        heads,
        tokens,
        block_m,
        dim,
    )
    return delta


class AttentionBackwardPreprocessTest(unittest.TestCase):
    """Check the unmasked tiling boundary and CPU reduction semantics."""

    def test_frontend(self) -> None:
        if os.environ.get("TRITON_INTERPRET") == "1":
            self.skipTest("Frontend compilation uses normal JIT mode")
        ir = compile_ttir(
            _attn_bwd_preprocess,
            signature={"O": "*fp32", "DO": "*fp32", "Delta": "*fp32"}
            | {"Z": "i32", "H": "i32", "N_CTX": "i32"},
            constexprs={"BLOCK_M": 2, "HEAD_DIM": 4},
        )
        self.assertIn("tt.func", ir)

    def test_reject_mismatched_axes_and_unmasked_tail(self) -> None:
        output = torch.ones((2, 2, 4, 4), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "shapes, dtypes, devices"):
            attention_backward_delta(output, cast(Any, output[:, :1]), block_m=2)
        with self.assertRaisesRegex(ValueError, "shapes, dtypes, devices"):
            attention_backward_delta(output, output.to(torch.float16), block_m=2)
        with self.assertRaisesRegex(ValueError, "whole query blocks"):
            attention_backward_delta(output, output, block_m=3)
        with self.assertRaisesRegex(ValueError, "whole query blocks"):
            sliced = output[:, :, :, :3].contiguous()
            attention_backward_delta(sliced, sliced, block_m=2)

    def test_cpu_gradient(self) -> None:
        if os.environ.get("TRITON_INTERPRET") != "1":
            self.skipTest("CPU launches use interpreter mode")
        output = torch.arange(2 * 2 * 4 * 4, dtype=torch.float32).reshape(2, 2, 4, 4)
        dout = torch.flip(output, dims=(-1,)) + 0.25
        delta = attention_backward_delta(output, dout.contiguous(), block_m=2)
        self.assertEqual(delta.shape, (2, 2, 4))
        torch.testing.assert_close(delta, (output * dout).sum(dim=-1))


if TYPE_CHECKING:

    def typed_addresses[B: IntVar, H: IntVar, T: IntVar, D: IntVar, Other: IntVar](
        output: tlt.InPointer[[B, H, T, D], [int, int, D, 1]],
        wrong: tl.ScaledTileStart[Other, T],
    ) -> None:
        output + wrong  # pyrefly: ignore[unsupported-operation]

    def typed_boundary[B: IntVar, H: IntVar, T: IntVar, D: IntVar, Other: IntVar](
        output: torch.Tensor[[B, H, T, D]],
        wrong: torch.Tensor[[B, H, Other, D]],
    ) -> None:
        assert_type(
            attention_backward_delta(output, output, block_m=2),
            torch.Tensor[[B, H, T]],
        )
        attention_backward_delta(
            output,
            wrong,  # pyrefly: ignore[bad-argument-type]
            block_m=2,
        )

# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The kernel body is copied from Triton's python/tutorials/fused-attention-ws.py.
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
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

# This static-only kernel is checked by Pyrefly, not executed as a Buck test.
# @lint-ignore-every AUTODEPS2

"""Monolithic warp-specialized backward requires zeroed atomic dQ output."""

import triton
import triton.language as tl
from shape_extensions import Int, IntVar

# A fake autotune configuration preserves the source decorators without GPU setup.
configs_bwd: list[object] = []


@triton.autotune(configs=configs_bwd, key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_bwd[
    Batch: IntVar,
    Heads: IntVar,
    Tokens: IntVar,
    Dim: IntVar,
    StrideZ: IntVar,
    StrideH: IntVar,
    TokenStride: IntVar,
    FeatureStride: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    Q: tl.Attention4DStridedInputPointer[
        Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
    ],
    K: tl.Attention4DStridedInputPointer[
        Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
    ],
    V: tl.Attention4DStridedInputPointer[
        Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
    ],
    sm_scale: float,
    DO: tl.Attention4DStridedInputPointer[
        Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
    ],
    DQ: tl.ZeroedAttention4DStridedOutputPointer[
        Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
    ],
    DK: tl.Attention4DStridedOutputPointer[
        Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
    ],
    DV: tl.Attention4DStridedOutputPointer[
        Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
    ],
    M: tl.Attention3DStatsPointer[Batch, Heads, Tokens],
    D: tl.Attention3DStatsPointer[Batch, Heads, Tokens],
    stride_z: tl.AttentionBatchStride[Heads, StrideZ],
    stride_h: tl.AttentionHeadStride[Heads, StrideH],
    stride_tok: Int[TokenStride],
    stride_d: Int[FeatureStride],
    H: tl.AttentionHeadCount[Heads],
    N_CTX: Int[Tokens],
    BLOCK_M1: Int[BM],
    BLOCK_N1: Int[BN],
    HEAD_DIM: Int[Dim],
):
    """Monolithic backward kernel: one thread block per K/V block.
    Copied from the proven _bwd_simple pattern in test_bwd_debug.py."""
    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    offs_k = tl.arange(0, HEAD_DIM)
    start_n = pid * BLOCK_N1
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    # Load K and V for this block — they stay in SRAM for the entire inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # Iterate over all Q blocks (the entire inner loop is inlined here,
    # NOT delegated to a helper function — this is critical for correctness).
    RCP_LN2: tl.constexpr = 1.4426950408889634  # E: float
    curr_m = 0
    for _ in range(N_CTX // BLOCK_M1):  # E: Fixpoint iteration did not converge
        offs_m = curr_m + tl.arange(0, BLOCK_M1)

        q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
        do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
        m = tl.load(M + offs_m)
        Di = tl.load(D + offs_m)

        # Recompute P = softmax(QK^T * sm_scale) in log2 space
        qk = tl.dot(q, tl.trans(k))  # [M, N]
        qk = qk * (sm_scale * RCP_LN2)
        p = tl.math.exp2(qk - m[:, None])  # [M, N]

        # dV += P^T @ dO
        pp = p.to(tl.float16)
        dv += tl.dot(tl.trans(pp), do)

        # dP = dO @ V^T, dS = P * (dP - Delta)
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)  # [M, N]
        ds = p * (dp - Di[:, None])  # [M, N]
        ds = ds.to(tl.float16)

        # dK += dS^T @ Q
        dk += tl.dot(tl.trans(ds), q)

        # dQ += dS @ K * sm_scale (accumulated via atomic add)
        dq = tl.dot(ds, k)  # [M, D]
        dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
        tl.atomic_add(dq_ptrs, dq.to(tl.float32) * sm_scale)

        curr_m += BLOCK_M1

    # Store dK (scaled) and dV
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    dk = dk * sm_scale
    tl.store(dk_ptrs, dk)
    tl.store(dv_ptrs, dv)


def test_atomic_output_requires_zeroing[Tokens: IntVar, Dim: IntVar, BM: IntVar](
    ordinary_output: tl.AttentionOutputTilePointers[Tokens, Dim, BM],
    zeroed_output: tl.ZeroedAttentionOutputTilePointers[Tokens, Dim, BM],
    value: tl.tensor[[BM, Dim]],
) -> None:
    tl.atomic_add(ordinary_output, value)  # E: is not assignable
    tl.atomic_add(zeroed_output, value)

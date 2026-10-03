# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The kernel body is copied from Triton's fused-attention-ws-device-tma-hopper-or-blackwell.py.
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

# Static-only kernel; these semantic annotations are illegal in Triton's JIT.
# @lint-ignore-every AUTODEPS2

"""Check descriptor conversion before device-TMA attention's typed forward body."""

from typing import assert_type, Literal

import triton
import triton.language as tl
from shape_extensions import Int, IntVar
from tests.test_attention_ws_device_subtile_attrs import DotAttrs
from tests.test_attention_ws_device_tma_attrs import _attn_fwd_tma_dp
from tests.test_fused_attention import _maybe_make_tensor_desc

configs: list[object] = []


def keep(conf: object) -> bool:
    return True


def prune_invalid_configs(*args: object, **kwargs: object) -> list[object]:
    return []


@triton.autotune(
    configs=list(filter(keep, configs)),
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
    prune_configs_by={"early_config_prune": prune_invalid_configs},
)
@triton.jit
def _attn_fwd[
    Batch: IntVar,
    Heads: IntVar,
    Tokens: IntVar,
    Dim: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    sm_scale: float,
    M: tl.Attention3DStatsPointer[Batch, Heads, Tokens],  #
    Z: Int[Batch],
    H: tl.AttentionHeadCount[Heads],
    desc_q: tl.AttentionPointer[Batch * Heads * Tokens, Dim, Dim]
    | tl.tensor_descriptor[Batch * Heads * Tokens, Dim, Dim, BM, Dim],
    desc_k: tl.AttentionPointer[Batch * Heads * Tokens, Dim, Dim]
    | tl.tensor_descriptor[Batch * Heads * Tokens, Dim, Dim, BN, Dim],
    desc_v: tl.AttentionPointer[Batch * Heads * Tokens, Dim, Dim]
    | tl.tensor_descriptor[Batch * Heads * Tokens, Dim, Dim, BN, Dim],
    desc_o: tl.AttentionPointer[Batch * Heads * Tokens, Dim, Dim]
    | tl.tensor_descriptor[Batch * Heads * Tokens, Dim, Dim, BM, Dim],
    N_CTX: Int[Tokens],  #
    HEAD_DIM: Int[Dim],  #
    BLOCK_M: Int[BM],  #
    BLOCK_N: Int[BN],  #
    FP8_OUTPUT: bool,  #
    STAGE: int,  #
    warp_specialize: bool,  #
    dtype: object,
    SUBTILING: bool,
    VECT_MUL: int,
    FADD2_REDUCE: bool,
    DP_FACTOR: Literal[2],
    FWD_DOT_ATTRS: DotAttrs | None = None,
):
    pid = tl.program_id(0)
    off_hz = tl.program_id(1)
    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(
        desc_q,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
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

    _attn_fwd_tma_dp(
        sm_scale,
        M,
        Z,
        H,
        desc_q,  # E: is not assignable
        desc_k,  # E: is not assignable
        desc_v,  # E: is not assignable
        desc_o,  # E: is not assignable
        pid,
        off_hz,
        N_CTX,
        HEAD_DIM,
        BLOCK_M,
        BLOCK_N,
        FP8_OUTPUT,
        STAGE,
        warp_specialize,
        dtype,
        SUBTILING,
        VECT_MUL,
        FADD2_REDUCE,
        DP_FACTOR,
        FWD_DOT_ATTRS,
    )


def test_pointer_full_rows_accepted_gap[
    Batch: IntVar,
    Heads: IntVar,
    Tokens: IntVar,
    Dim: IntVar,
    BM: IntVar,
    Wrong: IntVar,
](
    pointer: tl.AttentionPointer[Wrong, Dim, Dim],
    batch: Int[Batch],
    heads: tl.AttentionHeadCount[Heads],
    tokens: Int[Tokens],
    dim: Int[Dim],
    block_m: Int[BM],
) -> None:
    # The result keeps the pointer's wrong host extent despite the supplied shape.
    assert_type(
        _maybe_make_tensor_desc(
            pointer,
            shape=[batch * heads * tokens, dim],
            strides=[dim, 1],
            block_shape=[block_m, dim],
        ),
        tl.tensor_descriptor[Wrong, Dim, Dim, BM, Dim],
    )

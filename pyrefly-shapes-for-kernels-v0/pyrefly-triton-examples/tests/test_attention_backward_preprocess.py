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

# This static-only kernel is checked by Pyrefly, not executed as a Buck test.
# @lint-ignore-every AUTODEPS2

"""Backward attention output-gradient preprocessing in Triton's tutorial 06."""

from typing import assert_type

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def _attn_bwd_preprocess[
    ZDim: IntVar,
    Heads: IntVar,
    Tokens: IntVar,
    HeadDim: IntVar,
    BM: IntVar,
](
    O: tl.Attention4DInputPointer[ZDim, Heads, Tokens, HeadDim],
    DO: tl.Attention4DInputPointer[ZDim, Heads, Tokens, HeadDim],  #
    Delta: tl.Attention3DStatsPointer[ZDim, Heads, Tokens],  #
    Z: Int[ZDim],
    H: Int[Heads],
    N_CTX: Int[Tokens],  #
    BLOCK_M: Int[BM],
    HEAD_DIM: Int[HeadDim],  #
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


def test_wrong_input_head_slice[
    Batch: IntVar,
    Heads: IntVar,
    Tokens: IntVar,
    Dim: IntVar,
    Other: IntVar,
](
    ptr: tl.Attention4DInputPointer[Batch, Heads, Tokens, Dim],
    wrong_dim: tl.ScaledTileStart[Other, Tokens],
    wrong_tokens: tl.ScaledTileStart[Dim, Other],
) -> None:
    ptr + wrong_dim  # E: is not assignable
    ptr + wrong_tokens  # E: is not assignable


def test_wrong_input_row_and_feature[
    Batch: IntVar,
    Heads: IntVar,
    Tokens: IntVar,
    Dim: IntVar,
    Other: IntVar,
    BM: IntVar,
](
    head: tl.Attention4DHeadPointer[Batch, Heads, Tokens, Dim],
    rows: tl.Attention4DRows[Batch, Heads, Tokens, Dim, BM],
    wrong_row_stride: tl.RowAddress[[BM], Other],
    wrong_feature_extent: tl.ColumnAxisOffsets[[Other]],
) -> None:
    head + wrong_row_stride  # E: is not assignable
    rows + wrong_feature_extent  # E: is not assignable


def test_wrong_delta_stride_and_tile[
    Batch: IntVar,
    Heads: IntVar,
    Tokens: IntVar,
    Other: IntVar,
    BM: IntVar,
](
    ptr: tl.Attention3DStatsPointer[Batch, Heads, Tokens],
    wrong_stride: tl.TileStart[[Other]],
    tile: tl.Attention3DTilePointers[Batch, Heads, Tokens, BM],
    wrong_value: tl.tensor[[Other]],
) -> None:
    ptr + wrong_stride  # E: is not assignable
    tl.store(tile, wrong_value)  # E: No matching overload


def test_wrong_reduction_axis[
    Batch: IntVar,
    Heads: IntVar,
    Tokens: IntVar,
    BM: IntVar,
    Dim: IntVar,
](
    tile: tl.Attention3DTilePointers[Batch, Heads, Tokens, BM],
    o: tl.tensor[[BM, Dim]],
    do: tl.tensor[[BM, Dim]],
) -> None:
    tl.store(tile, tl.sum(o * do, axis=0))  # E: No matching overload


def test_wrong_kernel_boundary[
    Batch: IntVar,
    Heads: IntVar,
    Tokens: IntVar,
    Dim: IntVar,
    Other: IntVar,
    BM: IntVar,
](
    o: tl.Attention4DInputPointer[Batch, Heads, Tokens, Dim],
    do: tl.Attention4DInputPointer[Batch, Heads, Tokens, Dim],
    wrong_o: tl.Attention4DInputPointer[Batch, Heads, Tokens, Other],
    wrong_do: tl.Attention4DInputPointer[Batch, Heads, Other, Dim],
    delta: tl.Attention3DStatsPointer[Batch, Heads, Tokens],
    wrong_delta: tl.Attention3DStatsPointer[Batch, Heads, Other],
    batch: Int[Batch],
    heads: Int[Heads],
    tokens: Int[Tokens],
    bm: Int[BM],
    dim: Int[Dim],
) -> None:
    _attn_bwd_preprocess(
        wrong_o,
        do,  # E: is not assignable to parameter
        delta,
        batch,
        heads,
        tokens,
        bm,
        dim,  # E: is not assignable to parameter
    )
    _attn_bwd_preprocess(
        o,
        wrong_do,  # E: is not assignable to parameter
        delta,
        batch,
        heads,
        tokens,
        bm,
        dim,
    )
    _attn_bwd_preprocess(
        o,
        do,
        wrong_delta,  # E: is not assignable to parameter
        batch,
        heads,
        tokens,
        bm,
        dim,
    )


def test_unverified_token_bounds_and_program_axis[
    Batch: IntVar,
    Heads: IntVar,
    Tokens: IntVar,
    BM: IntVar,
](
    stats: tl.Attention3DStatsPointer[Batch, Heads, Tokens],
    unrelated_rows: tl.Offsets[[BM]],
    head_stride: Int[Tokens],
) -> None:
    # No mask checks whether the unbounded row offsets fit inside Tokens.
    assert_type(
        stats + tl.program_id(0) * head_stride + unrelated_rows,
        tl.Attention3DTilePointers[Batch, Heads, Tokens, BM],
    )
    # The wrong program axis is accepted because grid dimensions are untyped.

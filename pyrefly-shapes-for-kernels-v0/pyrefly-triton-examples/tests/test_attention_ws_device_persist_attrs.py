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

"""Persistent device-TMA attention descriptor and grid-stride boundary."""

from typing import assert_type, Literal

import triton
import triton.language as tl
from shape_extensions import Int, IntVar
from tests.test_attention_ws_device_subtile_attrs import DotAttrs
from tests.test_attention_ws_device_tma_attrs import _attn_fwd_tma_dp

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
def _attn_fwd_persist[
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
    desc_q: tl.AttentionPointer[Batch * Heads * Tokens, Dim, Dim],
    desc_k: tl.AttentionPointer[Batch * Heads * Tokens, Dim, Dim],
    desc_v: tl.AttentionPointer[Batch * Heads * Tokens, Dim, Dim],
    desc_o: tl.AttentionPointer[Batch * Heads * Tokens, Dim, Dim],
    N_CTX: Int[Tokens],  #
    HEAD_DIM: Int[Dim],  #
    BLOCK_M: Int[BM],  #
    BLOCK_N: Int[BN],  #
    FP8_OUTPUT: bool,  #
    STAGE: int,  #
    warp_specialize: bool,  #
    OUTER_LOOP: bool,
    dtype: object,
    SUBTILING: bool,
    VECT_MUL: int,
    FADD2_REDUCE: bool,
    DP_FACTOR: Literal[2],
    FWD_DOT_ATTRS: DotAttrs | None = None,
):
    n_tile_num = tl.cdiv(N_CTX, BLOCK_M)
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)
    total_tiles = n_tile_num * Z * H

    tiles_per_sm = total_tiles // num_progs
    if prog_id < total_tiles % num_progs:
        tiles_per_sm += 1

    tile_idx = prog_id

    desc_q = tl.make_tensor_descriptor(  # E: is not assignable
        desc_q,
        shape=[Z * H * N_CTX, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    desc_k = tl.make_tensor_descriptor(  # E: is not assignable
        desc_k,
        shape=[Z * H * N_CTX, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_v = tl.make_tensor_descriptor(  # E: is not assignable
        desc_v,
        shape=[Z * H * N_CTX, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_o = tl.make_tensor_descriptor(  # E: is not assignable
        desc_o,
        shape=[Z * H * N_CTX, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )

    # inner loop warpspec vs. outer loop warpspec
    for _ in tl.range(
        0,
        tiles_per_sm,
        warp_specialize=warp_specialize and OUTER_LOOP,
        merge_epilogue=True,
        merge_correction=True,
        data_partition_factor=DP_FACTOR,
    ):
        pid = tile_idx % n_tile_num
        off_hz = tile_idx // n_tile_num
        _attn_fwd_tma_dp(
            sm_scale,
            M,
            Z,
            H,
            desc_q,  # E: is not assignable
            desc_k,  # E: is not assignable
            desc_v,  # E: is not assignable
            desc_o,  # E: is not assignable
            pid,  # E: is not assignable
            off_hz,  # E: is not assignable
            N_CTX,
            HEAD_DIM,
            BLOCK_M,
            BLOCK_N,
            FP8_OUTPUT,
            STAGE,
            warp_specialize and not OUTER_LOOP,
            dtype,
            SUBTILING,
            VECT_MUL,
            FADD2_REDUCE,
            DP_FACTOR,
            FWD_DOT_ATTRS,
        )
        tile_idx += num_progs


def test_wrong_pointer_host_rows_accepted_gap[
    Rows: IntVar,
    Wrong: IntVar,
    Dim: IntVar,
    BM: IntVar,
](
    pointer: tl.AttentionPointer[Wrong, Dim, Dim],
    rows: Int[Rows],
    dim: Int[Dim],
    block_m: Int[BM],
) -> None:
    # The generic pointer constructor does not check the supplied host extent.
    assert_type(
        tl.make_tensor_descriptor(
            pointer,
            shape=[rows, dim],
            strides=[dim, 1],
            block_shape=[block_m, dim],
        ),
        tl.tensor_descriptor[Wrong, Dim, Dim, int, int],
    )


def test_persistent_range_requires_two_partitions() -> None:
    tl.range(  # E: No matching overload found
        0,
        16,
        warp_specialize=False,
        merge_epilogue=True,
        merge_correction=True,
        data_partition_factor=3,
    )

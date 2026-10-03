# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The kernel body is copied from Triton's fused-attention-ws.py.
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

# Static-only, not an executable Triton kernel.
# @lint-ignore-every AUTODEPS2

"""The nonpersistent warp-attention entry point and its helper contract."""

from typing import assert_type

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


def _attn_fwd_tma_dp[
    Batch: IntVar,
    Heads: IntVar,
    Tokens: IntVar,
    Half: IntVar,
    BN: IntVar,
    Dim: IntVar,
](
    sm_scale: float,
    M: tl.Attention3DStatsPointer[Batch, Heads, Tokens],
    Z: Int[Batch],
    H: tl.AttentionHeadCount[Heads],
    desc_q: tl.InputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, Half, Dim],
    desc_k: tl.InputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, BN, Dim],
    desc_v: tl.InputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, BN, Dim],
    desc_o: tl.OutputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, Half, Dim],
    pid: tl.ProgramId,
    off_hz: tl.ProgramId,
    N_CTX: Int[Tokens],
    HEAD_DIM: Int[Dim],
    BLOCK_M: Int[Half * 2],
    BLOCK_N: Int[BN],
    FP8_OUTPUT: bool,
    STAGE: int,
    warp_specialize: bool,
    dtype: object,
    SUBTILING: bool,
    VECT_MUL: int,
    FADD2_REDUCE: bool,
) -> None: ...


configs: list[object] = []


def keep(config: object) -> bool: ...


def prune_invalid_configs(configs: list[object]) -> list[object]: ...


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
    Half: IntVar,
    BN: IntVar,
    Dim: IntVar,
](
    sm_scale: float,
    M: tl.Attention3DStatsPointer[Batch, Heads, Tokens],  #
    Z: Int[Batch],
    H: tl.AttentionHeadCount[Heads],
    desc_q: tl.InputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, Half, Dim],
    desc_k: tl.InputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, BN, Dim],
    desc_v: tl.InputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, BN, Dim],
    desc_o: tl.OutputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, Half, Dim],
    N_CTX: Int[Tokens],  #
    HEAD_DIM: Int[Dim],  #
    BLOCK_M: Int[Half * 2],  #
    BLOCK_N: Int[BN],  #
    FP8_OUTPUT: bool,  #
    STAGE: int,  #
    warp_specialize: bool,  #
    dtype: object,
    SUBTILING: bool,
    VECT_MUL: int,
    FADD2_REDUCE: bool,
):
    pid = tl.program_id(0)
    off_hz = tl.program_id(1)
    _attn_fwd_tma_dp(
        sm_scale,
        M,
        Z,
        H,
        desc_q,
        desc_k,
        desc_v,
        desc_o,
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
    )


def test_kernel_contract[
    Batch: IntVar,
    Heads: IntVar,
    Tokens: IntVar,
    Half: IntVar,
    BN: IntVar,
    Dim: IntVar,
    Other: IntVar,
](
    stats: tl.Attention3DStatsPointer[Batch, Heads, Tokens],
    batch: Int[Batch],
    heads: tl.AttentionHeadCount[Heads],
    q: tl.InputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, Half, Dim],
    k: tl.InputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, BN, Dim],
    v: tl.InputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, BN, Dim],
    o: tl.OutputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, Half, Dim],
    wrong_o: tl.OutputMatrixDescriptor[Batch * Heads * Tokens, Other, Dim, Half, Dim],
    tokens: Int[Tokens],
    dim: Int[Dim],
    block_m: Int[Half * 2],
    block_n: Int[BN],
) -> None:
    _attn_fwd(
        1.0,
        stats,
        batch,
        heads,
        q,
        k,
        v,
        o,
        tokens,
        dim,
        block_m,
        block_n,
        False,
        1,
        False,
        tl.float32,
        False,
        0,
        False,
    )
    _attn_fwd(
        1.0,
        stats,
        batch,
        heads,
        q,
        k,
        v,
        wrong_o,  # E: is not assignable
        tokens,
        dim,
        block_m,
        block_n,
        False,
        1,
        False,
        tl.float32,
        False,
        0,
        False,
    )


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
    Half: IntVar,
    BN: IntVar,
    Dim: IntVar,
](
    sm_scale: float,
    M: tl.Attention3DStatsPointer[Batch, Heads, Tokens],  #
    Z: Int[Batch],
    H: tl.AttentionHeadCount[Heads],
    desc_q: tl.InputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, Half, Dim],
    desc_k: tl.InputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, BN, Dim],
    desc_v: tl.InputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, BN, Dim],
    desc_o: tl.OutputMatrixDescriptor[Batch * Heads * Tokens, Dim, Dim, Half, Dim],
    N_CTX: Int[Tokens],  #
    HEAD_DIM: Int[Dim],  #
    BLOCK_M: Int[Half * 2],  #
    BLOCK_N: Int[BN],  #
    FP8_OUTPUT: bool,  #
    STAGE: int,  #
    warp_specialize: bool,  #
    OUTER_LOOP: bool,  #
    dtype: object,
    SUBTILING: bool,
    VECT_MUL: int,
    FADD2_REDUCE: bool,
):
    n_tile_num = tl.cdiv(N_CTX, BLOCK_M)
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)
    total_tiles = n_tile_num * Z * H

    tiles_per_sm = total_tiles // num_progs
    if prog_id < total_tiles % num_progs:
        tiles_per_sm += 1

    tile_idx = prog_id
    # inner loop warpspec vs. outer loop warpspec
    for _ in tl.range(0, tiles_per_sm, warp_specialize=warp_specialize and OUTER_LOOP):
        pid = tile_idx % n_tile_num
        off_hz = tile_idx // n_tile_num
        _attn_fwd_tma_dp(
            sm_scale,
            M,
            Z,
            H,
            desc_q,
            desc_k,
            desc_v,
            desc_o,
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
        )
        tile_idx += num_progs


def test_persistent_tile_is_not_a_program_id(
    initial: tl.ProgramId, count: tl.ProgramCount, tiles: int
) -> None:
    tile_idx = initial
    tile_idx += count
    assert_type(tile_idx, int)
    pid = tile_idx % tiles
    off_hz = tile_idx // tiles
    assert_type(pid, int)
    assert_type(off_hz, int)


def test_program_count_and_plain_modulus_roles(
    program: tl.ProgramId, count: tl.ProgramCount, tiles: int
) -> None:
    def needs_program_id(index: tl.ProgramId) -> None: ...

    assert_type(program % tiles, tl.GroupIndex[int])
    needs_program_id(count)  # E: is not assignable
    needs_program_id(program % tiles)  # E: is not assignable

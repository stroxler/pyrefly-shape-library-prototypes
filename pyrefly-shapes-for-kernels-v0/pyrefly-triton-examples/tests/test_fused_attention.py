# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# These static-only kernels are checked by Pyrefly, not executed as Buck tests.
# @lint-ignore-every AUTODEPS2

"""Static-only copy of the forward boundary and helpers in Triton's tutorial 06."""

from typing import assert_type, Literal

import triton
import triton.language as tl
from shape_extensions import Int, IntListLiteral, IntVar


@triton.jit
def _attn_fwd_inner[BM: IntVar, BN: IntVar, D: IntVar, Y: IntVar, NC: IntVar](
    acc: tl.tensor[[BM, D]],
    l_i: tl.tensor[[BM]],
    m_i: tl.tensor[[BM]],
    q: tl.tensor[[BM, D]],  #
    desc_k: tl.tensor_descriptor[Y, D, D, BN, D],
    desc_v: tl.tensor_descriptor[Y, D, D, BN, D],  #
    offset_y: int,
    dtype: object,
    start_m: tl.ProgramId,
    qk_scale: float,  #
    BLOCK_M: Int[BM],
    HEAD_DIM: Int[D],
    BLOCK_N: Int[BN],  #
    STAGE: int,
    offs_m: tl.Offsets[[BM]],
    offs_n: tl.Offsets[[BN]],  #
    N_CTX: Int[NC],
    warp_specialize: bool,
    IS_HOPPER: bool,
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
                acc.reshape([BM, 2, BN // 2])  # E: reshape
                .permute(0, 2, 1)
                .split()
            )
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = (
                tl.join(acc0, acc1)  # E: join
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
        acc = tl.dot(p, v, acc)  # E: not assignable
        # update m_i and l_i
        # place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i


@triton.jit
def _maybe_make_tensor_desc[R: IntVar, C: IntVar, S: IntVar, BR: IntVar, BC: IntVar](
    desc_or_ptr: tl.tensor_descriptor[R, C, S, BR, BC] | tl.AttentionPointer[R, C, S],
    shape: IntListLiteral[[R, C]],
    strides: IntListLiteral[[S, 1]],
    block_shape: IntListLiteral[[BR, BC]],
) -> tl.tensor_descriptor[R, C, S, BR, BC]:
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


# These stand-ins preserve the source decorator without importing its runtime tuning setup.
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
    ZDim: IntVar,
    HDim: IntVar,
    NDim: IntVar,
    D: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    sm_scale: float,
    M: tl.AttentionStatsPointer[ZDim * HDim, NDim],  #
    Z: Int[ZDim],
    H: Int[HDim],
    desc_q: tl.AttentionPointer[ZDim * HDim * NDim, D, D]
    | tl.tensor_descriptor[ZDim * HDim * NDim, D, D, BM, D],
    desc_k: tl.AttentionPointer[ZDim * HDim * NDim, D, D]
    | tl.tensor_descriptor[ZDim * HDim * NDim, D, D, BN, D],
    desc_v: tl.AttentionPointer[ZDim * HDim * NDim, D, D]
    | tl.tensor_descriptor[ZDim * HDim * NDim, D, D, BN, D],
    desc_o: tl.AttentionPointer[ZDim * HDim * NDim, D, D]
    | tl.tensor_descriptor[ZDim * HDim * NDim, D, D, BM, D],
    N_CTX: Int[NDim],  #
    HEAD_DIM: Int[D],  #
    BLOCK_M: Int[BM],  #
    BLOCK_N: Int[BN],  #
    FP8_OUTPUT: Literal[False],  #
    STAGE: int,  #
    warp_specialize: bool,  #
    IS_HOPPER: bool,  #
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
            shape=[HEAD_DIM, y_dim],  # E: not assignable
            strides=[N_CTX, 1],  # E: not assignable
            block_shape=[HEAD_DIM, BLOCK_N],  # E: not assignable
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


def test_descriptor_fields[R: IntVar, C: IntVar, S: IntVar, BR: IntVar, BC: IntVar](
    desc: tl.tensor_descriptor[R, C, S, BR, BC],
    rows: Int[R],
    cols: Int[C],
    stride: Int[S],
    block_rows: Int[BR],
    block_cols: Int[BC],
) -> None:
    _valid: tl.tensor_descriptor[R, C, S, BR, BC] = _maybe_make_tensor_desc(
        desc, [rows, cols], [stride, 1], [block_rows, block_cols]
    )
    assert_type(_valid, tl.tensor_descriptor[R, C, S, BR, BC])
    _maybe_make_tensor_desc(
        desc,
        [cols, rows],  # E: not assignable
        [stride, 1],
        [block_rows, block_cols],
    )
    _maybe_make_tensor_desc(
        desc,
        [rows, cols],
        [1, stride],  # E: not assignable
        [block_rows, block_cols],
    )
    _maybe_make_tensor_desc(
        desc,
        [rows, cols],
        [stride, 1],
        [block_cols, block_rows],  # E: not assignable
    )


def test_direct_constructor_does_not_check_shape_or_stride[
    R: IntVar,
    C: IntVar,
    S: IntVar,
    BR: IntVar,
    BC: IntVar,
    Other: IntVar,
](
    ptr: tl.AttentionPointer[R, C, S],
    wrong_rows: Int[Other],
    cols: Int[C],
    wrong_stride: Int[Other],
    block_rows: Int[BR],
    block_cols: Int[BC],
) -> None:
    # The helper checks these positional values, but the direct constructor does not.
    descriptor: tl.tensor_descriptor[R, C, S, BR, BC] = tl.make_tensor_descriptor(
        ptr,
        [wrong_rows, cols],
        [wrong_stride, 1],
        [block_rows, block_cols],
    )
    assert_type(descriptor, tl.tensor_descriptor[R, C, S, BR, BC])


def test_wrong_k_head_dimension[
    Rows: IntVar,
    KBlock: IntVar,
    Head: IntVar,
    WrongHead: IntVar,
    QBlock: IntVar,
](
    q: tl.tensor[[QBlock, Head]],
    k: tl.tensor_descriptor[Rows, WrongHead, WrongHead, KBlock, WrongHead],
) -> None:
    tl.dot(q, k.load([0, 0]).T)  # E: not assignable


def test_wrong_output_value_head_dimension[
    Rows: IntVar,
    BM: IntVar,
    Head: IntVar,
    WrongHead: IntVar,
](
    output: tl.tensor_descriptor[Rows, WrongHead, WrongHead, BM, WrongHead],
    value: tl.tensor[[BM, Head]],
) -> None:
    output.store([0, 0], value)  # E: not assignable


def test_wrong_query_allocation_length[
    Z: IntVar,
    H: IntVar,
    N: IntVar,
    D: IntVar,
    BM: IntVar,
    WrongRows: IntVar,
](
    q: tl.AttentionPointer[WrongRows, D, D],
    z: Int[Z],
    h: Int[H],
    n: Int[N],
    d: Int[D],
    bm: Int[BM],
) -> None:
    y_dim = z * h * n
    _maybe_make_tensor_desc(q, [y_dim, d], [d, 1], [bm, d])  # E: not assignable

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# These static-only kernels are checked by Pyrefly, not executed as Buck tests.
# @lint-ignore-every AUTODEPS2

"""Static-only copy of matmul_kernel in Triton's 03-matrix-multiplication.py."""

from typing import assert_type

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def matmul_kernel[
    MDim: IntVar,
    NDim: IntVar,
    KDim: IntVar,
    AM: IntVar,
    AK: IntVar,
    BK: IntVar,
    BN: IntVar,
    CM: IntVar,
    CN: IntVar,
    BM: IntVar,
    BNN: IntVar,
    BKK: IntVar,
    Group: IntVar,
](
    # Pointers to matrices
    a_ptr: tl.InMatrixPointer[MDim, KDim, AM, AK],
    b_ptr: tl.InMatrixPointer[KDim, NDim, BK, BN],
    c_ptr: tl.OutMatrixPointer[MDim, NDim, CM, CN],
    # Matrix dimensions
    M: Int[MDim],
    N: Int[NDim],
    K: Int[KDim],
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am: Int[AM],
    stride_ak: Int[AK],  #
    stride_bk: Int[BK],
    stride_bn: Int[BN],  #
    stride_cm: Int[CM],
    stride_cn: Int[CN],
    # Meta-parameters
    BLOCK_SIZE_M: Int[BM],
    BLOCK_SIZE_N: Int[BNN],
    BLOCK_SIZE_K: Int[BKK],  #
    GROUP_SIZE_M: Int[Group],  #
    ACTIVATION: str,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -----------------------------------------------------------
    # Add some integer bound assumptions.
    # This helps to guide integer analysis in the backend to optimize
    # load/store offset address calculation
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def leaky_relu[Rows: IntVar, Cols: IntVar](x: tl.tensor[[Rows, Cols]]):
    return tl.where(x >= 0, x, 0.01 * x)


def test_wrapped_axes[Rows: IntVar, Cols: IntVar, BlockM: IntVar, BlockN: IntVar](
    rows: Int[Rows], cols: Int[Cols], block_m: Int[BlockM], block_n: Int[BlockN]
) -> None:
    row_offsets = (tl.program_id(0) // 8 * block_m + tl.arange(0, block_m)) % rows
    col_offsets = (tl.program_id(0) // 8 * block_n + tl.arange(0, block_n)) % cols
    assert_type(row_offsets, tl.WrappedOffsets[Rows, [BlockM]])
    assert_type(col_offsets, tl.WrappedOffsets[Cols, [BlockN]])


def test_k_bound_widens_to_int[K: IntVar, BlockK: IntVar](
    k: Int[K], block_k: Int[BlockK], iteration: int
) -> None:
    offsets = tl.arange(0, block_k)
    assert_type(
        offsets[None, :] < k - iteration * block_k, tl.ColumnMask[int, [BlockK]]
    )
    assert_type(offsets[:, None] < k - iteration * block_k, tl.RowMask[int, [BlockK]])


def test_wrong_k_mask[
    Rows: IntVar,
    K: IntVar,
    Other: IntVar,
    BM: IntVar,
    BK: IntVar,
    RS: IntVar,
    CS: IntVar,
](
    ptrs: tl.WrappedRowMatrixTilePointers[Rows, K, BM, BK, RS, CS],
    mask: tl.ColumnMask[Other, [BK]],
) -> None:
    tl.load(ptrs, mask=mask, other=0.0)  # E: No matching overload


def test_wrong_k_mask_tile[
    Rows: IntVar,
    K: IntVar,
    BM: IntVar,
    BK: IntVar,
    Other: IntVar,
    RS: IntVar,
    CS: IntVar,
](
    ptrs: tl.WrappedRowMatrixTilePointers[Rows, K, BM, BK, RS, CS],
    mask: tl.ColumnMask[K, [Other]],
) -> None:
    tl.load(ptrs, mask=mask, other=0.0)  # E: No matching overload


def test_wrong_k_mask_axis[
    Rows: IntVar,
    K: IntVar,
    BM: IntVar,
    BK: IntVar,
    RS: IntVar,
    CS: IntVar,
](
    ptrs: tl.WrappedRowMatrixTilePointers[Rows, K, BM, BK, RS, CS],
    mask: tl.RowMask[Rows, [BM]],
) -> None:
    tl.load(ptrs, mask=mask, other=0.0)  # E: No matching overload


def test_unverified_k_mask_bound[
    Rows: IntVar,
    K: IntVar,
    BM: IntVar,
    BKK: IntVar,
    RS: IntVar,
    CS: IntVar,
](
    ptrs: tl.WrappedRowMatrixTilePointers[Rows, K, BM, BKK, RS, CS],
    offsets: tl.ColumnAxisOffsets[[BKK]],
    wrong_bound: int,
) -> None:
    # Known gap: the real K - k * BLOCK_SIZE_K bound widens to int too.
    assert_type(
        tl.load(ptrs, mask=offsets < wrong_bound, other=0.0), tl.tensor[[BM, BKK]]
    )


def test_wrong_output_mask[
    M: IntVar,
    N: IntVar,
    Other: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    ptrs: tl.OutMatrixTilePointers[M, N, BM, BN],
    value: tl.tensor[[BM, BN]],
    mask: tl.MatrixMask[M, Other, [BM], [BN]],
) -> None:
    tl.store(ptrs, value, mask=mask)  # E: No matching overload


def test_wrong_output_tile[
    M: IntVar,
    N: IntVar,
    BM: IntVar,
    BN: IntVar,
    Other: IntVar,
](
    ptrs: tl.OutMatrixTilePointers[M, N, BM, BN],
    value: tl.tensor[[BM, Other]],
    mask: tl.MatrixMask[M, N, [BM], [BN]],
) -> None:
    tl.store(ptrs, value, mask=mask)  # E: No matching overload


def test_wrong_dot_contraction[M: IntVar, N: IntVar, K: IntVar, Other: IntVar](
    a: tl.tensor[[M, K]], b: tl.tensor[[Other, N]], acc: tl.tensor[[M, N]]
) -> None:
    tl.dot(a, b, acc)  # E: is not assignable to parameter


def test_wrong_a_allocation[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    Other: IntVar,
    AM: IntVar,
    AK: IntVar,
    BK: IntVar,
    BN: IntVar,
    CM: IntVar,
    CN: IntVar,
    BM: IntVar,
    BNN: IntVar,
    BKK: IntVar,
    Group: IntVar,
](
    a: tl.InMatrixPointer[Other, K, AM, AK],
    b: tl.InMatrixPointer[K, N, BK, BN],
    c: tl.OutMatrixPointer[M, N, CM, CN],
    m: Int[M],
    n: Int[N],
    k: Int[K],
    am: Int[AM],
    ak: Int[AK],
    bk: Int[BK],
    bn: Int[BN],
    cm: Int[CM],
    cn: Int[CN],
    bm: Int[BM],
    bnn: Int[BNN],
    bkk: Int[BKK],
    group: Int[Group],
) -> None:
    matmul_kernel(
        a,
        b,
        c,  # E: is not assignable to parameter
        m,  # E: is not assignable to parameter
        n,
        k,
        am,
        ak,
        bk,
        bn,
        cm,
        cn,
        bm,
        bnn,
        bkk,
        group,
        "",
    )


def test_wrong_c_allocation[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    Other: IntVar,
    AM: IntVar,
    AK: IntVar,
    BK: IntVar,
    BN: IntVar,
    CM: IntVar,
    CN: IntVar,
    BM: IntVar,
    BNN: IntVar,
    BKK: IntVar,
    Group: IntVar,
](
    a: tl.InMatrixPointer[M, K, AM, AK],
    b: tl.InMatrixPointer[K, N, BK, BN],
    c: tl.OutMatrixPointer[M, Other, CM, CN],
    m: Int[M],
    n: Int[N],
    k: Int[K],
    am: Int[AM],
    ak: Int[AK],
    bk: Int[BK],
    bn: Int[BN],
    cm: Int[CM],
    cn: Int[CN],
    bm: Int[BM],
    bnn: Int[BNN],
    bkk: Int[BKK],
    group: Int[Group],
) -> None:
    matmul_kernel(
        a,
        b,
        c,  # E: is not assignable to parameter
        m,
        n,
        k,
        am,
        ak,
        bk,
        bn,
        cm,
        cn,
        bm,
        bnn,
        bkk,
        group,
        "",
    )


def test_wrong_b_k_allocation[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    Other: IntVar,
    AM: IntVar,
    AK: IntVar,
    BK: IntVar,
    BN: IntVar,
    CM: IntVar,
    CN: IntVar,
    BM: IntVar,
    BNN: IntVar,
    BKK: IntVar,
    Group: IntVar,
](
    a: tl.InMatrixPointer[M, K, AM, AK],
    b: tl.InMatrixPointer[Other, N, BK, BN],
    c: tl.OutMatrixPointer[M, N, CM, CN],
    m: Int[M],
    n: Int[N],
    k: Int[K],
    am: Int[AM],
    ak: Int[AK],
    bk: Int[BK],
    bn: Int[BN],
    cm: Int[CM],
    cn: Int[CN],
    bm: Int[BM],
    bnn: Int[BNN],
    bkk: Int[BKK],
    group: Int[Group],
) -> None:
    matmul_kernel(
        a,
        b,  # E: is not assignable to parameter
        c,
        m,
        n,
        k,
        am,
        ak,
        bk,
        bn,
        cm,
        cn,
        bm,
        bnn,
        bkk,
        group,
        "",
    )


def test_wrong_input_row_stride[
    M: IntVar,
    K: IntVar,
    AM: IntVar,
    AK: IntVar,
    Other: IntVar,
    BM: IntVar,
    BKK: IntVar,
](
    a: tl.InMatrixPointer[M, K, AM, AK],
    rows: tl.WrappedOffsets[M, [BM]],
    k_offsets: tl.Offsets[[BKK]],
    wrong_stride: Int[Other],
    k_stride: Int[AK],
) -> None:
    address = rows[:, None] * wrong_stride + k_offsets[None, :] * k_stride
    a + address  # E: No matching overload


def test_wrong_input_column_stride[
    M: IntVar,
    K: IntVar,
    AM: IntVar,
    AK: IntVar,
    Other: IntVar,
    BM: IntVar,
    BKK: IntVar,
](
    a: tl.InMatrixPointer[M, K, AM, AK],
    rows: tl.WrappedOffsets[M, [BM]],
    k_offsets: tl.Offsets[[BKK]],
    row_stride: Int[AM],
    wrong_stride: Int[Other],
) -> None:
    address = rows[:, None] * row_stride + k_offsets[None, :] * wrong_stride
    a + address  # E: No matching overload


def test_wrong_output_row_stride[
    M: IntVar,
    N: IntVar,
    CM: IntVar,
    CN: IntVar,
    Other: IntVar,
    BM: IntVar,
](
    c: tl.OutMatrixPointer[M, N, CM, CN],
    offsets: tl.Offsets[[BM]],
    wrong_stride: Int[Other],
) -> None:
    c + wrong_stride * offsets[:, None]  # E: not supported


def test_wrong_output_col_stride[
    M: IntVar,
    N: IntVar,
    CM: IntVar,
    CN: IntVar,
    Other: IntVar,
    BM: IntVar,
    BNN: IntVar,
](
    c: tl.OutMatrixPointer[M, N, CM, CN],
    row_offsets: tl.Offsets[[BM]],
    col_offsets: tl.Offsets[[BNN]],
    row_stride: Int[CM],
    wrong_stride: Int[Other],
) -> None:
    rows = c + row_stride * row_offsets[:, None]
    rows + wrong_stride * col_offsets[None, :]  # E: not supported


def test_wrong_wrapped_row_bound[
    M: IntVar,
    K: IntVar,
    Other: IntVar,
    AM: IntVar,
    AK: IntVar,
    BM: IntVar,
    BKK: IntVar,
](
    a: tl.InMatrixPointer[M, K, AM, AK],
    rows: tl.WrappedOffsets[Other, [BM]],
    k_offsets: tl.Offsets[[BKK]],
    row_stride: Int[AM],
    k_stride: Int[AK],
) -> None:
    address = rows[:, None] * row_stride + k_offsets[None, :] * k_stride
    a + address  # E: No matching overload


def test_wrong_output_row_mask[
    M: IntVar,
    N: IntVar,
    Other: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    ptrs: tl.OutMatrixTilePointers[M, N, BM, BN],
    value: tl.tensor[[BM, BN]],
    mask: tl.MatrixMask[Other, N, [BM], [BN]],
) -> None:
    tl.store(ptrs, value, mask=mask)  # E: No matching overload


def test_wrong_a_k_step[
    M: IntVar,
    K: IntVar,
    BM: IntVar,
    BlockK: IntVar,
    WrongBlock: IntVar,
    AM: IntVar,
    AK: IntVar,
](
    ptrs: tl.WrappedRowMatrixTilePointers[M, K, BM, BlockK, AM, AK],
    wrong_step: Int[WrongBlock * AK],
) -> None:
    ptrs += wrong_step  # E: not supported


def test_wrong_b_k_stride[
    K: IntVar,
    N: IntVar,
    BlockK: IntVar,
    BN: IntVar,
    WrongStride: IntVar,
    BK: IntVar,
    NS: IntVar,
](
    ptrs: tl.WrappedColumnMatrixTilePointers[K, N, BlockK, BN, BK, NS],
    wrong_step: Int[BlockK * WrongStride],
) -> None:
    ptrs += wrong_step  # E: not supported

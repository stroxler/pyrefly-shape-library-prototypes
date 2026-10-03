# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# These static-only kernels are checked by Pyrefly, not executed as Buck tests.
# @lint-ignore-every AUTODEPS2

"""Static-only copies of Triton's python/tutorials/05-layer-norm.py kernels."""

from typing import assert_type

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def _layer_norm_fwd_fused[Rows: IntVar, Cols: IntVar, Stride: IntVar, Block: IntVar](
    X: tl.InRowMajorPointer[Rows, Cols, Stride],  # pointer to the input
    Y: tl.OutRowMajorPointer[Rows, Cols, Stride],  # pointer to the output
    W: tl.InPointer[[Cols]],  # pointer to the weights
    B: tl.InPointer[[Cols]],  # pointer to the biases
    Mean: tl.OutPointer[[Rows]],  # pointer to the mean
    Rstd: tl.OutPointer[[Rows]],  # pointer to the 1/std
    stride: Int[Stride],  # how much to increase the pointer when moving by 1 row
    N: Int[Cols],  # number of columns in X
    eps: float,  # epsilon to avoid division by zero
    BLOCK_SIZE: Int[Block],
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)


@triton.jit
def _layer_norm_bwd_dx_fused[
    Rows: IntVar,
    Cols: IntVar,
    Stride: IntVar,
    Groups: IntVar,
    Block: IntVar,
](
    DX: tl.OutRowMajorPointer[Rows, Cols, Stride],  # pointer to the input gradient
    DY: tl.InRowMajorPointer[Rows, Cols, Stride],  # pointer to the output gradient
    DW: tl.GroupedScratchPointer[
        Groups, Cols, Block
    ],  # pointer to the partial sum of weights gradient
    DB: tl.GroupedScratchPointer[
        Groups, Cols, Block
    ],  # pointer to the partial sum of biases gradient
    X: tl.InRowMajorPointer[Rows, Cols, Stride],  # pointer to the input
    W: tl.InPointer[[Cols]],  # pointer to the weights
    Mean: tl.InPointer[[Rows]],  # pointer to the mean
    Rstd: tl.InPointer[[Rows]],  # pointer to the 1/std
    Lock: tl.LockArrayPointer[Groups, 2 * Groups],  # pointer to the lock
    stride: Int[Stride],  # how much to increase the pointer when moving by 1 row
    N: Int[Cols],  # number of columns in X
    GROUP_SIZE_M: Int[Groups],
    BLOCK_SIZE_N: Int[Block],
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    # Offset locks and weights/biases gradient pointer for parallel reduction
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols
    # Load data to SRAM
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    # Compute dx
    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd
    # Write dx
    tl.store(DX + cols, dx, mask=mask)
    # Accumulate partial sums for dw/db
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = (dy).to(w.dtype)
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    # First store doesn't accumulate
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)

    # need a barrier to ensure all threads finished before
    # releasing the lock
    tl.debug_barrier()

    # Release the lock
    tl.atomic_xchg(Lock, 0)


@triton.jit
def _layer_norm_bwd_dwdb[Groups: IntVar, Cols: IntVar, BlockM: IntVar, BlockN: IntVar](
    DW: tl.GroupedScratchPointer[
        Groups, Cols, BlockN
    ],  # pointer to the partial sum of weights gradient
    DB: tl.GroupedScratchPointer[
        Groups, Cols, BlockN
    ],  # pointer to the partial sum of biases gradient
    FINAL_DW: tl.OutPointer[[Cols]],  # pointer to the weights gradient
    FINAL_DB: tl.OutPointer[[Cols]],  # pointer to the biases gradient
    M: Int[Groups],  # GROUP_SIZE_M
    N: Int[Cols],  # number of columns
    BLOCK_SIZE_M: Int[BlockM],
    BLOCK_SIZE_N: Int[BlockN],
):
    # Map the program id to the elements of DW and DB it should compute.
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Iterate through the rows of DW and DB to sum the partial sums.
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)
        db += tl.load(DB + offs, mask=mask, other=0.0)
    # Write the final sum to the output.
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)


def test_forward_wrong_input_stride[
    Rows: IntVar,
    Cols: IntVar,
    Stride: IntVar,
    Other: IntVar,
    Block: IntVar,
](
    x: tl.InRowMajorPointer[Rows, Cols, Stride],
    y: tl.OutRowMajorPointer[Rows, Cols, Stride],
    w: tl.InPointer[[Cols]],
    b: tl.InPointer[[Cols]],
    mean: tl.OutPointer[[Rows]],
    rstd: tl.OutPointer[[Rows]],
    wrong_stride: Int[Other],
    n: Int[Cols],
    block: Int[Block],
) -> None:
    _layer_norm_fwd_fused(
        x,
        y,
        w,
        b,
        mean,
        rstd,
        wrong_stride,  # E: is not assignable to parameter
        n,
        1e-5,
        block,
    )


def test_forward_wrong_weight_columns[
    Rows: IntVar,
    Cols: IntVar,
    Stride: IntVar,
    Other: IntVar,
    Block: IntVar,
](
    x: tl.InRowMajorPointer[Rows, Cols, Stride],
    y: tl.OutRowMajorPointer[Rows, Cols, Stride],
    w: tl.InPointer[[Other]],
    b: tl.InPointer[[Cols]],
    mean: tl.OutPointer[[Rows]],
    rstd: tl.OutPointer[[Rows]],
    stride: Int[Stride],
    n: Int[Cols],
    block: Int[Block],
) -> None:
    _layer_norm_fwd_fused(
        x,
        y,
        w,  # E: is not assignable to parameter
        b,
        mean,
        rstd,
        stride,
        n,
        1e-5,
        block,
    )


def test_forward_wrong_output_columns[
    Rows: IntVar,
    Cols: IntVar,
    Stride: IntVar,
    Other: IntVar,
    Block: IntVar,
](
    x: tl.InRowMajorPointer[Rows, Cols, Stride],
    y: tl.OutRowMajorPointer[Rows, Other, Stride],
    w: tl.InPointer[[Cols]],
    b: tl.InPointer[[Cols]],
    mean: tl.OutPointer[[Rows]],
    rstd: tl.OutPointer[[Rows]],
    stride: Int[Stride],
    n: Int[Cols],
    block: Int[Block],
) -> None:
    _layer_norm_fwd_fused(
        x,
        y,  # E: is not assignable to parameter
        w,
        b,
        mean,
        rstd,
        stride,
        n,
        1e-5,
        block,
    )


def test_forward_wrong_stat_rows[
    Rows: IntVar,
    Cols: IntVar,
    Stride: IntVar,
    Other: IntVar,
    Block: IntVar,
](
    x: tl.InRowMajorPointer[Rows, Cols, Stride],
    y: tl.OutRowMajorPointer[Rows, Cols, Stride],
    w: tl.InPointer[[Cols]],
    b: tl.InPointer[[Cols]],
    mean: tl.OutPointer[[Other]],
    rstd: tl.OutPointer[[Rows]],
    stride: Int[Stride],
    n: Int[Cols],
    block: Int[Block],
) -> None:
    _layer_norm_fwd_fused(
        x,
        y,
        w,
        b,
        mean,  # E: is not assignable to parameter
        rstd,
        stride,
        n,
        1e-5,
        block,
    )


def test_backward_wrong_scratch_groups[
    Rows: IntVar,
    Cols: IntVar,
    Stride: IntVar,
    Groups: IntVar,
    Other: IntVar,
    Block: IntVar,
](
    dx: tl.OutRowMajorPointer[Rows, Cols, Stride],
    dy: tl.InRowMajorPointer[Rows, Cols, Stride],
    dw: tl.GroupedScratchPointer[Other, Cols, Block],
    db: tl.GroupedScratchPointer[Groups, Cols, Block],
    x: tl.InRowMajorPointer[Rows, Cols, Stride],
    w: tl.InPointer[[Cols]],
    mean: tl.InPointer[[Rows]],
    rstd: tl.InPointer[[Rows]],
    lock: tl.LockArrayPointer[Groups, 2 * Groups],
    stride: Int[Stride],
    n: Int[Cols],
    groups: Int[Groups],
    block: Int[Block],
) -> None:
    _layer_norm_bwd_dx_fused(
        dx,
        dy,
        dw,
        db,  # E: is not assignable to parameter
        x,
        w,
        mean,
        rstd,
        lock,  # E: is not assignable to parameter
        stride,
        n,
        groups,  # E: is not assignable to parameter
        block,
    )


def test_final_reduction_wrong_output_columns[
    Groups: IntVar,
    Cols: IntVar,
    Other: IntVar,
    BlockM: IntVar,
    BlockN: IntVar,
](
    dw: tl.GroupedScratchPointer[Groups, Cols, BlockN],
    db: tl.GroupedScratchPointer[Groups, Cols, BlockN],
    output_dw: tl.OutPointer[[Other]],
    output_db: tl.OutPointer[[Cols]],
    groups: Int[Groups],
    n: Int[Cols],
    bm: Int[BlockM],
    bn: Int[BlockN],
) -> None:
    _layer_norm_bwd_dwdb(
        dw,
        db,
        output_dw,  # E: is not assignable to parameter
        output_db,
        groups,
        n,
        bm,
        bn,
    )


def test_backward_wrong_lock_capacity[
    Rows: IntVar,
    Cols: IntVar,
    Stride: IntVar,
    Groups: IntVar,
    Other: IntVar,
    Block: IntVar,
](
    dx: tl.OutRowMajorPointer[Rows, Cols, Stride],
    dy: tl.InRowMajorPointer[Rows, Cols, Stride],
    dw: tl.GroupedScratchPointer[Groups, Cols, Block],
    db: tl.GroupedScratchPointer[Groups, Cols, Block],
    x: tl.InRowMajorPointer[Rows, Cols, Stride],
    w: tl.InPointer[[Cols]],
    mean: tl.InPointer[[Rows]],
    rstd: tl.InPointer[[Rows]],
    short_lock: tl.LockArrayPointer[Groups, Other],
    stride: Int[Stride],
    n: Int[Cols],
    groups: Int[Groups],
    block: Int[Block],
) -> None:
    _layer_norm_bwd_dx_fused(
        dx,
        dy,
        dw,
        db,
        x,
        w,
        mean,
        rstd,
        short_lock,  # E: is not assignable to parameter
        stride,
        n,
        groups,
        block,
    )


def test_final_reduction_wrong_scratch_columns[
    Groups: IntVar,
    Cols: IntVar,
    Other: IntVar,
    BlockM: IntVar,
    BlockN: IntVar,
](
    dw: tl.GroupedScratchPointer[Groups, Other, BlockN],
    db: tl.GroupedScratchPointer[Groups, Cols, BlockN],
    output_dw: tl.OutPointer[[Cols]],
    output_db: tl.OutPointer[[Cols]],
    groups: Int[Groups],
    n: Int[Cols],
    bm: Int[BlockM],
    bn: Int[BlockN],
) -> None:
    _layer_norm_bwd_dwdb(
        dw,
        db,  # E: is not assignable to parameter
        output_dw,  # E: is not assignable to parameter
        output_db,  # E: is not assignable to parameter
        groups,
        n,  # E: is not assignable to parameter
        bm,
        bn,
    )


def test_wrong_grouped_scratch_address[
    Groups: IntVar,
    Other: IntVar,
    Cols: IntVar,
    Block: IntVar,
](
    ptr: tl.GroupedScratchTile[Groups, Cols, Block],
    wrong_group_start: tl.GroupStart[Other, Cols],
) -> None:
    ptr + wrong_group_start  # E: No matching overload


def test_group_index_keeps_symbolic_group_size[Groups: IntVar, Cols: IntVar](
    groups: Int[Groups], cols: Int[Cols]
) -> None:
    group_index = tl.program_id(0) % groups
    assert_type(group_index, tl.GroupIndex[Groups])
    assert_type(group_index * cols, tl.GroupStart[Groups, Cols])


def test_wrong_grouped_scratch_mask[
    Groups: IntVar,
    Cols: IntVar,
    Other: IntVar,
    Block: IntVar,
](
    ptr: tl.GroupedScratchTile[Groups, Cols, Block],
    wrong_mask: tl.Mask[[Other], [Block]],
) -> None:
    tl.load(ptr, mask=wrong_mask)  # E: is not assignable to parameter


def test_wrong_row_pointer_advance[
    Rows: IntVar,
    Cols: IntVar,
    Stride: IntVar,
    Other: IntVar,
](
    ptr: tl.InRowMajorPointer[Rows, Cols, Stride],
    row: tl.ProgramId,
    wrong_stride: Int[Other],
) -> None:
    ptr.__iadd__(row * wrong_stride)  # E: is not assignable to parameter


def test_wrong_partial_gradient_tile[
    Groups: IntVar,
    Cols: IntVar,
    Block: IntVar,
    Other: IntVar,
](
    ptr: tl.GroupedScratchTile[Groups, Cols, Block],
    value: tl.tensor[[Other]],
    mask: tl.Mask[[Cols], [Block]],
) -> None:
    tl.store(ptr, value, mask=mask)  # E: No matching overload


def test_wrong_reduction_row_mask[
    Groups: IntVar,
    Other: IntVar,
    Cols: IntVar,
    BM: IntVar,
    BN: IntVar,
](
    ptr: tl.GroupedScratchMatrixPointers[Groups, Cols, BM, BN],
    wrong_mask: tl.MatrixMask[Other, Cols, [BM], [BN]],
) -> None:
    tl.load(ptr, mask=wrong_mask, other=0.0)  # E: is not assignable to parameter


def test_matrix_reduction_preserves_columns[BM: IntVar, BN: IntVar](
    matrix: tl.tensor[[BM, BN]],
) -> None:
    assert_type(tl.sum(matrix, axis=0), tl.tensor[[BN]])
    assert_type(tl.sum(matrix, axis=1), tl.tensor[[BM]])
    tl.sum(matrix, axis=2)  # E: is not assignable to parameter


def test_grouped_pointer_requires_complete_address[
    Groups: IntVar,
    Cols: IntVar,
    Block: IntVar,
](
    unoffset_ptr: tl.GroupedScratchPointer[Groups, Cols, Block],
    group_start: tl.GroupStart[Groups, Cols],
    offsets: tl.Offsets[[Block]],
    mask: tl.Mask[[Cols], [Block]],
) -> None:
    tl.load(unoffset_ptr, mask=mask)  # E: No matching overload
    row_ptr = unoffset_ptr + group_start
    tl.load(row_ptr, mask=mask)  # E: No matching overload
    assert_type(tl.load(row_ptr + offsets, mask=mask), tl.tensor[[Block]])


def test_unverified_double_column_offset[Groups: IntVar, Cols: IntVar, Block: IntVar](
    unoffset_ptr: tl.GroupedScratchPointer[Groups, Cols, Block],
    group_start: tl.GroupStart[Groups, Cols],
    offsets: tl.Offsets[[Block]],
    mask: tl.Mask[[Cols], [Block]],
) -> None:
    tile_ptr = unoffset_ptr + group_start + offsets
    # The stage proves at least one column step, not exactly one.
    assert_type(tl.load(tile_ptr + offsets, mask=mask), tl.tensor[[Block]])

"""Check Triton's layer-norm backward partials and their scratch allocations."""

from __future__ import annotations

import os
import unittest
from typing import TYPE_CHECKING, Any, Literal, assert_type, cast

import torch
import triton.language as tl
from shape_extensions import Int, IntVar

from triton_examples.testing import compile_ttir
from triton_library import host_tensor, tlt
from triton_library.launch_layout import row_output, tiled_output
from triton_library.semantic_jit import ConstExpr, semantic_jit
from triton_library.torch_views import (
    as_host_tensor,
    checked_group_locks,
    checked_grouped_scratch,
    checked_matrix,
    checked_vector,
)

Rows = IntVar("Rows")
Cols = IntVar("Cols")
Stride = IntVar("Stride")
Groups = IntVar("Groups")
Block = IntVar("Block")
BlockM = IntVar("BlockM")
BlockN = IntVar("BlockN")


@semantic_jit
def _layer_norm_bwd_dx_fused(
    DX: tlt.OutPointer[[Rows, Cols], [Stride, 1]],
    DY: tlt.InPointer[[Rows, Cols], [Stride, 1]],
    DW: tlt.InOutPointer[[Groups, Cols], [Cols, 1]],
    DB: tlt.InOutPointer[[Groups, Cols], [Cols, 1]],
    X: tlt.InPointer[[Rows, Cols], [Stride, 1]],
    W: tlt.InPointer[[Cols], [1]],
    Mean: tlt.InPointer[[Rows], [1]],
    Rstd: tlt.InPointer[[Rows], [1]],
    Lock: tl.LockArrayPointer[Groups, int],
    stride: Int[Stride],
    N: Int[Cols],
    GROUP_SIZE_M: ConstExpr[Int[Groups]],
    BLOCK_SIZE_N: ConstExpr[Int[Block]],
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    # Upstream rebinds X, DY, and DX with +=; fresh names expose 1D row types.
    x_row = X + row * stride
    dy_row = DY + row * stride
    dx_row = DX + row * stride
    # Offset locks and weights/biases gradient pointer for parallel reduction
    lock_id = row % GROUP_SIZE_M
    # Upstream rebinds Lock with +=; the selected lock is a scalar pointer.
    lock_ptr = Lock + lock_id
    Count = lock_ptr + GROUP_SIZE_M
    # Upstream rebinds DW and DB; these names distinguish scratch tiles.
    dw_ptrs = DW + lock_id * N + cols
    db_ptrs = DB + lock_id * N + cols
    # Load data to SRAM
    x = tl.load(x_row + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(dy_row + cols, mask=mask, other=0).to(tl.float32)
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
    tl.store(dx_row + cols, dx, mask=mask)
    # Accumulate partial sums for dw/db
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = (dy).to(w.dtype)
    while tl.atomic_cas(lock_ptr, 0, 1) == 1:
        pass
    count = tl.load(Count)
    # First store doesn't accumulate
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(dw_ptrs, mask=mask)
        partial_db += tl.load(db_ptrs, mask=mask)
    tl.store(dw_ptrs, partial_dw, mask=mask)
    tl.store(db_ptrs, partial_db, mask=mask)

    # need a barrier to ensure all threads finished before
    # releasing the lock
    tl.debug_barrier()

    # Release the lock
    tl.atomic_xchg(lock_ptr, 0)


@semantic_jit
def _layer_norm_bwd_dwdb(
    DW: tlt.InPointer[[Groups, Cols], [Cols, 1]],
    DB: tlt.InPointer[[Groups, Cols], [Cols, 1]],
    FINAL_DW: tlt.OutPointer[[Cols], [1]],
    FINAL_DB: tlt.OutPointer[[Cols], [1]],
    M: Int[Groups],
    N: Int[Cols],
    BLOCK_SIZE_M: ConstExpr[Int[BlockM]],
    BLOCK_SIZE_N: ConstExpr[Int[BlockN]],
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


def layer_norm_backward_partials[Rows: IntVar, Cols: IntVar, Stride: IntVar](
    x: host_tensor.Tensor[[Rows, Cols], [Stride, 1]],
    dy: host_tensor.Tensor[[Rows, Cols], [Stride, 1]],
    weight: host_tensor.Tensor[[Cols], [1]],
    mean: host_tensor.Tensor[[Rows], [1]],
    rstd: host_tensor.Tensor[[Rows], [1]],
    *,
    block_size: int,
    group_size: int,
) -> tuple[
    torch.Tensor[[Rows, Cols]], torch.Tensor[[int, Cols]], torch.Tensor[[int, Cols]]
]:
    """Validate saved statistics and allocate DX, grouped partials, and locks."""
    x_ptr, stride, rows, cols = checked_matrix(
        x, tlt.InPointer[[Rows, Cols], [Stride, 1]]
    )
    dy_ptr, dy_stride, dy_rows, dy_cols = checked_matrix(
        dy, tlt.InPointer[[Rows, Cols], [Stride, 1]]
    )
    w_ptr, w_cols = checked_vector(weight, tlt.InPointer[[Cols], [1]])
    mean_ptr, mean_rows = checked_vector(mean, tlt.InPointer[[Rows], [1]])
    rstd_ptr, rstd_rows = checked_vector(rstd, tlt.InPointer[[Rows], [1]])
    if (
        (dy_stride, dy_rows, dy_cols) != (stride, rows, cols)
        or w_cols != cols
        or mean_rows != rows
        or rstd_rows != rows
        or any(t.device != x.device for t in (dy, weight, mean, rstd))
    ):
        raise ValueError(
            "Backward inputs must have matching shapes, strides, and devices"
        )
    if (
        x.dtype != torch.float32
        or dy.dtype != x.dtype
        or weight.dtype != x.dtype
        or mean.dtype != torch.float32
        or rstd.dtype != torch.float32
    ):
        raise ValueError("Backward inputs currently require float32 dtype")
    if type(group_size) is not int or not 1 <= group_size <= rows:
        raise ValueError("group_size must be between one and the number of rows")
    if (
        type(block_size) is not int
        or block_size < cols
        or block_size & (block_size - 1)
    ):
        raise ValueError("block_size must be a power of two covering all columns")
    dx = torch.empty_strided((rows, cols), (stride, 1), dtype=x.dtype, device=x.device)
    dw_partial = torch.empty((group_size, cols), dtype=x.dtype, device=x.device)
    db_partial = torch.empty_like(dw_partial)
    locks = torch.zeros((2 * group_size,), dtype=torch.int32, device=x.device)
    dw_ptr = checked_grouped_scratch(dw_partial, group_size, cols, block_size)
    db_ptr = checked_grouped_scratch(db_partial, group_size, cols, block_size)
    lock_ptr = checked_group_locks(locks, group_size)
    dx_view = as_host_tensor(dx, host_tensor.Tensor[[Rows, Cols], [Stride, 1]])
    dx_ptr, _, _, _ = checked_matrix(dx_view, tlt.OutPointer[[Rows, Cols], [Stride, 1]])
    layout = row_output(
        dx_view,
        column_parameter="N",
        stride_parameter="stride",
        block_parameter="BLOCK_SIZE_N",
        block_width=block_size,
        metadata={"GROUP_SIZE_M": group_size},
    )
    layout.launch(
        _layer_norm_bwd_dx_fused,
        dx_ptr,
        dy_ptr,
        dw_ptr,
        db_ptr,
        x_ptr,
        w_ptr,
        mean_ptr,
        rstd_ptr,
        lock_ptr,
        stride,
        cols,
        group_size,
        block_size,
    )
    return dx, dw_partial, db_partial


def reduce_layer_norm_partials[Groups: IntVar, Cols: IntVar](
    dw_partial: torch.Tensor[[Groups, Cols]],
    db_partial: torch.Tensor[[Groups, Cols]],
    *,
    block_m: int,
    block_n: int,
) -> tuple[torch.Tensor[[Cols]], torch.Tensor[[Cols]]]:
    """Check grouped partials and launch the final columnwise reduction."""
    if (
        dw_partial.ndim != 2
        or dw_partial.shape[0] <= 0
        or dw_partial.shape[1] <= 0
        or db_partial.shape != dw_partial.shape
        or not dw_partial.is_contiguous()
        or not db_partial.is_contiguous()
        or dw_partial.dtype != torch.float32
        or db_partial.dtype != dw_partial.dtype
        or dw_partial.device != db_partial.device
    ):
        raise ValueError("Partial gradients need matching dense float32 matrices")
    if any(
        type(block) is not int or block <= 0 or block & (block - 1)
        for block in (block_m, block_n)
    ):
        raise ValueError("Reduction blocks must be positive powers of two")
    groups, cols = dw_partial.shape
    dw_view = as_host_tensor(dw_partial, host_tensor.Tensor[[Groups, Cols], [Cols, 1]])
    db_view = as_host_tensor(db_partial, host_tensor.Tensor[[Groups, Cols], [Cols, 1]])
    dw_ptr, _, _, _ = checked_matrix(dw_view, tlt.InPointer[[Groups, Cols], [Cols, 1]])
    db_ptr, _, _, _ = checked_matrix(db_view, tlt.InPointer[[Groups, Cols], [Cols, 1]])
    final_dw = torch.empty((cols,), dtype=dw_partial.dtype, device=dw_partial.device)
    final_db = torch.empty_like(final_dw)
    final_dw_view = as_host_tensor(final_dw)
    final_db_view = as_host_tensor(final_db)
    final_dw_ptr, _ = checked_vector(final_dw_view, tlt.OutPointer[[Cols], [1]])
    final_db_ptr, _ = checked_vector(final_db_view, tlt.OutPointer[[Cols], [1]])
    layout = tiled_output(
        final_dw_view,
        (block_n,),
        shape_parameters=("N",),
        tile_parameters=("BLOCK_SIZE_N",),
        metadata={"M": groups, "BLOCK_SIZE_M": block_m},
    )
    layout.launch(
        _layer_norm_bwd_dwdb,
        dw_ptr,
        db_ptr,
        final_dw_ptr,
        final_db_ptr,
        groups,
        cols,
        block_m,
        block_n,
    )
    return final_dw, final_db


class LayerNormBackwardTest(unittest.TestCase):
    """Check host boundary and compile the unchanged lock-based kernel."""

    def test_reject_wrong_scratch_and_locks(self) -> None:
        # An untyped caller still needs the runtime allocation check.
        with self.assertRaisesRegex(ValueError, "Grouped scratch"):
            checked_grouped_scratch(cast(Any, torch.empty((2, 6))), 2, 7, 8)
        with self.assertRaisesRegex(ValueError, "zero-initialized"):
            checked_group_locks(torch.ones((4,), dtype=torch.int32), 2)

    def test_reject_wrong_statistics(self) -> None:
        x = as_host_tensor(torch.ones((2, 7)))
        w = as_host_tensor(torch.ones(7))
        wrong = as_host_tensor(torch.ones(7))
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            layer_norm_backward_partials(
                x,
                x,
                w,
                wrong,  # pyrefly: ignore[bad-argument-type]
                wrong,  # pyrefly: ignore[bad-argument-type]
                block_size=8,
                group_size=2,
            )

    def test_frontend(self) -> None:
        if os.environ.get("TRITON_INTERPRET") == "1":
            self.skipTest("Frontend compilation runs in normal JIT mode")
        ir = compile_ttir(
            _layer_norm_bwd_dx_fused,
            signature={
                name: "*fp32"
                for name in ("DX", "DY", "DW", "DB", "X", "W", "Mean", "Rstd")
            }
            | {"Lock": "*i32", "stride": "i32", "N": "i32"},
            constexprs={"GROUP_SIZE_M": 2, "BLOCK_SIZE_N": 8},
        )
        self.assertIn("tt.func", ir)
        reduction_ir = compile_ttir(
            _layer_norm_bwd_dwdb,
            signature={name: "*fp32" for name in ("DW", "DB", "FINAL_DW", "FINAL_DB")}
            | {"M": "i32", "N": "i32"},
            constexprs={"BLOCK_SIZE_M": 2, "BLOCK_SIZE_N": 4},
        )
        self.assertIn("tt.func", reduction_ir)

    def test_reject_mismatched_partials(self) -> None:
        with self.assertRaisesRegex(ValueError, "matching dense float32"):
            reduce_layer_norm_partials(
                cast(Any, torch.ones((2, 7))),
                cast(Any, torch.ones((3, 7))),
                block_m=2,
                block_n=4,
            )
        dense = torch.ones((2, 7), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "matching dense float32"):
            reduce_layer_norm_partials(
                dense[:, ::2], dense[:, ::2], block_m=2, block_n=4
            )
        with self.assertRaisesRegex(ValueError, "matching dense float32"):
            reduce_layer_norm_partials(
                dense, dense.to(torch.float16), block_m=2, block_n=4
            )

    def test_final_reduction(self) -> None:
        if os.environ.get("TRITON_INTERPRET") != "1":
            self.skipTest("CPU launches use interpreter mode")
        dw = torch.arange(21, dtype=torch.float32).reshape(3, 7)
        db = dw * 2
        final_dw, final_db = reduce_layer_norm_partials(dw, db, block_m=2, block_n=4)
        torch.testing.assert_close(final_dw, dw.sum(dim=0))
        torch.testing.assert_close(final_db, db.sum(dim=0))

    def test_partial_gradients(self) -> None:
        if os.environ.get("TRITON_INTERPRET") != "1":
            self.skipTest("CPU launches use interpreter mode")
        x = torch.arange(22, dtype=torch.float32).reshape(2, 11)[:, :7]
        dy = torch.linspace(0.1, 1.4, 14).reshape(2, 7)
        dy = torch.nn.functional.pad(dy, (0, 4))[:, :7]
        weight = torch.linspace(0.5, 1.5, 7)
        mean = x.mean(dim=1)
        rstd = torch.rsqrt(((x - mean[:, None]) ** 2).mean(dim=1) + 1e-5)
        dx, dw, db = layer_norm_backward_partials(
            as_host_tensor(x),
            as_host_tensor(dy),
            as_host_tensor(weight),
            as_host_tensor(mean),
            as_host_tensor(rstd),
            block_size=8,
            group_size=2,
        )
        xhat = (x - mean[:, None]) * rstd[:, None]
        wdy = dy * weight
        expected = (
            wdy - xhat * (xhat * wdy).mean(1, keepdim=True) - wdy.mean(1, keepdim=True)
        ) * rstd[:, None]
        torch.testing.assert_close(dx, expected)
        torch.testing.assert_close(dw, dy * xhat)
        torch.testing.assert_close(db, dy)
        final_dw, final_db = reduce_layer_norm_partials(dw, db, block_m=2, block_n=4)
        torch.testing.assert_close(final_dw, (dy * xhat).sum(dim=0))
        torch.testing.assert_close(final_db, dy.sum(dim=0))

    def test_multiple_rows_share_a_scratch_group(self) -> None:
        if os.environ.get("TRITON_INTERPRET") != "1":
            self.skipTest("CPU launches use interpreter mode")
        x = torch.arange(21, dtype=torch.float32).reshape(3, 7)
        dy = torch.linspace(0.1, 2.1, 21).reshape(3, 7)
        weight = torch.linspace(0.5, 1.5, 7)
        mean = x.mean(dim=1)
        rstd = torch.rsqrt(((x - mean[:, None]) ** 2).mean(dim=1) + 1e-5)
        _, dw, db = layer_norm_backward_partials(
            as_host_tensor(x),
            as_host_tensor(dy),
            as_host_tensor(weight),
            as_host_tensor(mean),
            as_host_tensor(rstd),
            block_size=8,
            group_size=2,
        )
        contributions = dy * (x - mean[:, None]) * rstd[:, None]
        torch.testing.assert_close(
            dw, torch.stack((contributions[0] + contributions[2], contributions[1]))
        )
        torch.testing.assert_close(db, torch.stack((dy[0] + dy[2], dy[1])))


if TYPE_CHECKING:

    def typed_grouped_reduction[
        Groups: IntVar, Cols: IntVar, Other: IntVar, BM: IntVar, BN: IntVar
    ](
        scratch: tlt.InPointer[[Groups, Cols], [Cols, 1]],
        offsets: tl.GroupedMatrixOffsets[[BM], [BN], Cols],
        mask: tl.MatrixMask[Groups, Cols, [BM], [BN]],
        wrong: tl.MatrixMask[Groups, Other, [BM], [BN]],
    ) -> None:
        ptrs = scratch + offsets
        assert_type(
            ptrs,
            tl.InTilePointers[[Groups, Cols], [Cols, 1], [BM, BN], Literal["grouped"]],
        )
        assert_type(tl.load(ptrs, mask=mask, other=0.0), tl.tensor[[BM, BN]])
        tl.load(ptrs, mask=wrong, other=0.0)  # pyrefly: ignore[no-matching-overload]

    def typed_pointer_selection[
        Groups: IntVar, Cols: IntVar, Block: IntVar, Other: IntVar
    ](
        scratch: tlt.InOutPointer[[Groups, Cols], [Cols, 1]],
        locks: tl.LockArrayPointer[Groups, int],
        group: tl.GroupIndex[Groups],
        n: Int[Cols],
        group_size: Int[Groups],
        columns: tl.Offsets[[Block]],
        mask: tl.Mask[[Cols], [Block]],
        wrong_mask: tl.Mask[[Other], [Block]],
        values: tl.tensor[[Block]],
    ) -> None:
        row_ptr = scratch + group * n
        assert_type(row_ptr, tlt.InOutPointer[[Cols], [1]])
        tile_ptrs = row_ptr + columns
        assert_type(tile_ptrs, tl.InOutTilePointers[[Cols], [1], [Block]])
        assert_type(tl.load(tile_ptrs, mask=mask), tl.tensor[[Block]])
        tl.load(tile_ptrs, mask=wrong_mask)  # pyrefly: ignore[no-matching-overload]
        tl.store(tile_ptrs, values, mask=mask)
        tl.store(  # pyrefly: ignore[no-matching-overload]
            tile_ptrs, values, mask=wrong_mask
        )
        lock_ptr = locks + group
        assert_type(lock_ptr, tl.LockSlotPointer[Groups])
        assert_type(lock_ptr + group_size, tl.CountPointer[Groups])

    def typed_boundary[Rows: IntVar, Cols: IntVar, Stride: IntVar, Other: IntVar](
        x: host_tensor.Tensor[[Rows, Cols], [Stride, 1]],
        dy: host_tensor.Tensor[[Rows, Cols], [Stride, 1]],
        weight: host_tensor.Tensor[[Cols], [1]],
        stats: host_tensor.Tensor[[Rows], [1]],
        wrong: host_tensor.Tensor[[Other], [1]],
    ) -> None:
        assert_type(
            layer_norm_backward_partials(
                x, dy, weight, stats, stats, block_size=8, group_size=2
            ),
            tuple[
                torch.Tensor[[Rows, Cols]],
                torch.Tensor[[int, Cols]],
                torch.Tensor[[int, Cols]],
            ],
        )
        dw: torch.Tensor[[Rows, Cols]] = x
        db: torch.Tensor[[Rows, Cols]] = dy
        assert_type(
            reduce_layer_norm_partials(dw, db, block_m=2, block_n=4),
            tuple[torch.Tensor[[Cols]], torch.Tensor[[Cols]]],
        )
        layer_norm_backward_partials(
            x,
            dy,
            wrong,  # pyrefly: ignore[bad-argument-type]
            stats,
            stats,
            block_size=8,
            group_size=2,
        )

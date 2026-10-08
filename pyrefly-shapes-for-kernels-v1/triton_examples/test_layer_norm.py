"""Run Triton's layer-norm forward body against a checked padded-row boundary."""

import os
import unittest
from typing import TYPE_CHECKING, Any, assert_type, cast

import torch
import triton.language as tl
from shape_extensions import Int, IntVar

from triton_examples.testing import compile_ttir
from triton_library import host_tensor, tlt
from triton_library.launch_layout import row_output
from triton_library.semantic_jit import ConstExpr, semantic_jit
from triton_library.torch_views import as_host_tensor, checked_matrix, checked_vector

Rows = IntVar("Rows")
Cols = IntVar("Cols")
Stride = IntVar("Stride")
Block = IntVar("Block")


@semantic_jit
def _layer_norm_fwd_fused(
    X: tlt.InPointer[[Rows, Cols], [Stride, 1]],  # pointer to the input
    Y: tlt.OutPointer[[Rows, Cols], [Stride, 1]],  # pointer to the output
    W: tlt.InPointer[[Cols], [1]],  # pointer to the weights
    B: tlt.InPointer[[Cols], [1]],  # pointer to the biases
    Mean: tlt.OutPointer[[Rows], [1]],  # pointer to the mean
    Rstd: tlt.OutPointer[[Rows], [1]],  # pointer to the 1/std
    stride: Int[Stride],  # how much to increase the pointer when moving by 1 row
    N: Int[Cols],  # number of columns in X
    eps: float,  # epsilon to avoid division by zero
    BLOCK_SIZE: ConstExpr[Int[Block]],
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    # Upstream rebinds Y and X with +=; fresh names expose their 1D row types.
    y_row = Y + row * stride
    x_row = X + row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(x_row + cols, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_row + cols, mask=cols < N, other=0.0).to(tl.float32)
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
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(y_row + cols, y, mask=mask)


def layer_norm[Rows: IntVar, Cols: IntVar, Stride: IntVar](
    x: host_tensor.Tensor[[Rows, Cols], [Stride, 1]],
    weight: host_tensor.Tensor[[Cols], [1]],
    bias: host_tensor.Tensor[[Cols], [1]],
    *,
    block_size: int = 32,
    eps: float = 1e-5,
) -> tuple[torch.Tensor[[Rows, Cols]], torch.Tensor[[Rows]], torch.Tensor[[Rows]]]:
    """Preserve the input row stride in a checked output and launch one row per PID."""
    x_ptr, stride, rows, cols = checked_matrix(
        x, tlt.InPointer[[Rows, Cols], [Stride, 1]]
    )
    w_ptr, w_cols = checked_vector(weight, tlt.InPointer[[Cols], [1]])
    b_ptr, b_cols = checked_vector(bias, tlt.InPointer[[Cols], [1]])
    if (
        (w_cols, b_cols) != (cols, cols)
        or x.device != weight.device
        or x.device != bias.device
    ):
        raise ValueError("Input, weight, and bias shapes and devices must match")
    if (
        not x.dtype.is_floating_point
        or weight.dtype != x.dtype
        or bias.dtype != x.dtype
    ):
        raise ValueError("Input, weight, and bias dtypes must match and be floating")
    if rows == 0 or cols == 0:
        raise ValueError("Layer norm requires nonempty dimensions")
    if type(block_size) is not int or block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    y = torch.empty_strided((rows, cols), (stride, 1), dtype=x.dtype, device=x.device)
    mean = torch.empty((rows,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((rows,), dtype=torch.float32, device=x.device)
    y_view = as_host_tensor(y, host_tensor.Tensor[[Rows, Cols], [Stride, 1]])
    y_ptr, _, _, _ = checked_matrix(y_view, tlt.OutPointer[[Rows, Cols], [Stride, 1]])
    mean_ptr, _ = checked_vector(as_host_tensor(mean), tlt.OutPointer[[Rows], [1]])
    rstd_ptr, _ = checked_vector(as_host_tensor(rstd), tlt.OutPointer[[Rows], [1]])
    layout = row_output(
        y_view,
        column_parameter="N",
        stride_parameter="stride",
        block_parameter="BLOCK_SIZE",
        block_width=block_size,
    )
    layout.launch(
        _layer_norm_fwd_fused,
        x_ptr,
        y_ptr,
        w_ptr,
        b_ptr,
        mean_ptr,
        rstd_ptr,
        stride,
        cols,
        eps,
        block_size,
    )
    return y, mean, rstd


class LayerNormTest(unittest.TestCase):
    """Check frontend translation, padded rows, and multi-block reductions."""

    def test_row_layout_rejects_inconsistent_launch(self) -> None:
        x = as_host_tensor(torch.empty_strided((3, 7), (11, 1)))
        w = as_host_tensor(torch.ones(7))
        x_ptr, stride, _, cols = checked_matrix(x, tlt.InPointer[[3, 7], [11, 1]])
        w_ptr, _ = checked_vector(w, tlt.InPointer[[7], [1]])
        y_ptr, _, _, _ = checked_matrix(x, tlt.OutPointer[[3, 7], [11, 1]])
        mean_ptr, _ = checked_vector(
            as_host_tensor(torch.empty(3)), tlt.OutPointer[[3], [1]]
        )
        layout = row_output(
            x,
            column_parameter="N",
            stride_parameter="stride",
            block_parameter="BLOCK_SIZE",
            block_width=4,
        )
        self.assertEqual(layout.grid, (3,))
        for name, actual_stride, actual_cols, block in (
            ("stride", stride + 1, cols, 4),
            ("N", stride, cols - 1, 4),
            ("BLOCK_SIZE", stride, cols, 8),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                layout.launch(
                    _layer_norm_fwd_fused,
                    x_ptr,
                    y_ptr,
                    w_ptr,
                    w_ptr,
                    mean_ptr,
                    mean_ptr,
                    actual_stride,
                    actual_cols,
                    1e-5,
                    block,
                )

    def test_frontend(self) -> None:
        if os.environ.get("TRITON_INTERPRET") == "1":
            self.skipTest("Frontend compilation runs in normal JIT mode")
        ir = compile_ttir(
            _layer_norm_fwd_fused,
            signature={name: "*fp32" for name in ("X", "Y", "W", "B", "Mean", "Rstd")}
            | {"stride": "i32", "N": "i32", "eps": "fp32"},
            constexprs={"BLOCK_SIZE": 4},
        )
        self.assertIn("tt.func", ir)

    def test_padded_rows(self) -> None:
        if os.environ.get("TRITON_INTERPRET") != "1":
            self.skipTest("CPU launches use interpreter mode")
        x = torch.arange(55, dtype=torch.float32).reshape(5, 11)[:, :7]
        w = torch.linspace(0.5, 1.5, 7)
        b = torch.linspace(-1.0, 1.0, 7)
        y, mean, rstd = layer_norm(
            as_host_tensor(x), as_host_tensor(w), as_host_tensor(b), block_size=4
        )
        expected_mean = x.mean(dim=1)
        expected_rstd = torch.rsqrt(
            ((x - expected_mean[:, None]) ** 2).mean(dim=1) + 1e-5
        )
        torch.testing.assert_close(y, torch.nn.functional.layer_norm(x, (7,), w, b))
        torch.testing.assert_close(mean, expected_mean)
        torch.testing.assert_close(rstd, expected_rstd)
        self.assertEqual(y.stride(), (11, 1))

    def test_reject_wrong_weight_length(self) -> None:
        x = as_host_tensor(torch.ones((2, 7)))
        w = as_host_tensor(torch.ones(6))
        b = as_host_tensor(torch.ones(7))
        with self.assertRaisesRegex(ValueError, "shapes and devices"):
            layer_norm(x, w, b)  # pyrefly: ignore[bad-argument-type]


if TYPE_CHECKING:

    def row_selection_contract[
        Rows: IntVar, Cols: IntVar, Stride: IntVar, Tile: IntVar
    ](
        input_ptr: tlt.InPointer[[Rows, Cols], [Stride, 1]],
        output_ptr: tlt.OutPointer[[Rows, Cols], [Stride, 1]],
        stride: Int[Stride],
        tile: Int[Tile],
    ) -> None:
        columns = tl.arange(0, tile)
        input_ptr + columns  # pyrefly: ignore[unsupported-operation]
        output_ptr + columns  # pyrefly: ignore[unsupported-operation]
        row_offset = tl.program_id(0) * stride
        input_row = input_ptr + row_offset
        output_row = output_ptr + row_offset
        assert_type(input_row, tlt.InPointer[[Cols], [1]])
        assert_type(output_row, tlt.OutPointer[[Cols], [1]])
        assert_type(input_row + columns, tl.InTilePointers[[Cols], [1], [Tile]])
        assert_type(output_row + columns, tl.OutTilePointers[[Cols], [1], [Tile]])

    def reject_wrong_row_stride[
        Rows: IntVar, Cols: IntVar, Stride: IntVar, OtherStride: IntVar
    ](
        input_ptr: tlt.InPointer[[Rows, Cols], [Stride, 1]],
        other_stride: Int[OtherStride],
    ) -> None:
        input_ptr + tl.program_id(0) * other_stride  # pyrefly: ignore[unsupported-operation]

    def typed_boundary[Rows: IntVar, Cols: IntVar, Stride: IntVar, Other: IntVar](
        x: host_tensor.Tensor[[Rows, Cols], [Stride, 1]],
        w: host_tensor.Tensor[[Cols], [1]],
        b: host_tensor.Tensor[[Cols], [1]],
        wrong: host_tensor.Tensor[[Other], [1]],
    ) -> None:
        assert_type(
            layer_norm(x, w, b),
            tuple[
                torch.Tensor[[Rows, Cols]], torch.Tensor[[Rows]], torch.Tensor[[Rows]]
            ],
        )
        layer_norm(x, wrong, b)  # pyrefly: ignore[bad-argument-type]

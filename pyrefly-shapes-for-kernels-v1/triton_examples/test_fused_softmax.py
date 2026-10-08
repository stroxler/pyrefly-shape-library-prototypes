"""Run Triton's tutorial softmax body with semantic 2D allocation types."""

import os
import unittest
from typing import TYPE_CHECKING, Any, assert_type, cast

import torch
import triton
import triton.language as tl
from shape_extensions import Int, IntVar
from triton.runtime import driver

from triton_examples.testing import compile_ttir
from triton_library import host_tensor, tlt
from triton_library.launch_layout import GridStrideOutputLayout, grid_stride_output
from triton_library.semantic_jit import ConstExpr, semantic_jit
from triton_library.torch_views import as_host_tensor, checked_matrix

Rows = IntVar("Rows")
Cols = IntVar("Cols")
InputStride = IntVar("InputStride")
OutputStride = IntVar("OutputStride")
Block = IntVar("Block")


@semantic_jit
def softmax_kernel(
    output_ptr: tlt.OutPointer[[Rows, Cols], [OutputStride, 1]],
    input_ptr: tlt.InPointer[[Rows, Cols], [InputStride, 1]],
    input_row_stride: Int[InputStride],
    output_row_stride: Int[OutputStride],
    n_rows: Int[Rows],
    n_cols: Int[Cols],
    BLOCK_SIZE: ConstExpr[Int[Block]],
    num_stages: ConstExpr[int],
):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def softmax[Rows: IntVar, Cols: IntVar, InputStride: IntVar](
    x: host_tensor.Tensor[[Rows, Cols], [InputStride, 1]],
) -> torch.Tensor[[Rows, Cols]]:
    """Launch softmax from a checked host view of the input matrix."""
    input_ptr, input_row_stride, n_rows, n_cols = checked_matrix(
        x, tlt.InPointer[[Rows, Cols], [InputStride, 1]]
    )
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 2
    y = torch.empty_like(x)
    output_view = as_host_tensor(y, host_tensor.Tensor[[Rows, Cols], [OutputStride, 1]])
    output_ptr, output_row_stride, out_rows, out_cols = checked_matrix(
        output_view, tlt.OutPointer[[Rows, Cols], [OutputStride, 1]]
    )
    if x.device != y.device or (out_rows, out_cols) != (n_rows, n_cols):
        raise ValueError("Input and output must have the same shape and device")

    if os.environ.get("TRITON_INTERPRET") == "1":
        num_programs = min(n_rows, 4)
    else:
        active_driver = cast(Any, driver.active)
        properties = active_driver.utils.get_device_properties(x.device.index)
        NUM_SM = properties["multiprocessor_count"]
        NUM_REGS = properties["max_num_regs"]
        SIZE_SMEM = properties["max_shared_mem"]
        WARP_SIZE = properties["warpSize"]
        num_stages = 4 if SIZE_SMEM > 200000 else 2
        kernel = softmax_kernel.warmup(
            output_ptr,
            input_ptr,
            input_row_stride,
            output_row_stride,
            n_rows,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=num_stages,
            num_warps=num_warps,
            grid=(1,),
        )
        kernel._init_handles()
        target = active_driver.get_current_target()
        if target.backend == "hip":
            NUM_GPRS = NUM_REGS
            if target.arch in ("gfx940", "gfx941", "gfx942", "gfx90a", "gfx908"):
                NUM_GPRS = NUM_REGS * 2
            max_num_waves = properties["max_threads_per_sm"] // WARP_SIZE
            occupancy = min(NUM_GPRS // WARP_SIZE // kernel.n_regs, max_num_waves)
            occupancy //= num_warps
        else:
            occupancy = NUM_REGS // (kernel.n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // kernel.metadata.shared)
        num_programs = min(NUM_SM * occupancy, n_rows)

    layout = grid_stride_output(
        output_view,
        num_programs,
        column_block=BLOCK_SIZE,
        block_parameter="BLOCK_SIZE",
        shape_parameters=("n_rows", "n_cols"),
    )
    layout.launch(
        softmax_kernel,
        output_ptr,
        input_ptr,
        input_row_stride,
        output_row_stride,
        n_rows,
        n_cols,
        BLOCK_SIZE,
        num_stages,
    )
    return y


class FusedSoftmaxTest(unittest.TestCase):
    """Check the frontend and the handwritten host launch."""

    def test_grid_stride_layout_checks_shape_and_metadata(self) -> None:
        output = torch.empty((5, 7))
        view = as_host_tensor(output)
        layout = grid_stride_output(
            view,
            4,
            column_block=8,
            block_parameter="BLOCK_SIZE",
            shape_parameters=("n_rows", "n_cols"),
        )
        self.assertEqual(layout.grid, (4, 1, 1))
        input_ptr, input_stride, _, _ = checked_matrix(
            view, tlt.InPointer[[5, 7], [int, 1]]
        )
        output_ptr, output_stride, _, _ = checked_matrix(
            view, tlt.OutPointer[[5, 7], [int, 1]]
        )
        for name, rows, cols, block in (
            ("n_rows", 4, 7, 8),
            ("n_cols", 5, 8, 8),
            ("BLOCK_SIZE", 5, 7, 16),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                layout.launch(
                    softmax_kernel,
                    output_ptr,
                    input_ptr,
                    input_stride,
                    output_stride,
                    rows,
                    cols,
                    block,
                    2,
                )
        with self.assertRaisesRegex(ValueError, "Grid-stride programs"):
            grid_stride_output(
                view,
                6,
                column_block=8,
                block_parameter="BLOCK_SIZE",
                shape_parameters=("n_rows", "n_cols"),
            )
        with self.assertRaisesRegex(ValueError, "Column block"):
            grid_stride_output(
                view,
                4,
                column_block=4,
                block_parameter="BLOCK_SIZE",
                shape_parameters=("n_rows", "n_cols"),
            )

    def test_checked_host_views(self) -> None:
        x = torch.arange(55, dtype=torch.float32).reshape(5, 11)[:, :7]
        view = as_host_tensor(x)
        self.assertIs(view, x)
        pointer, stride, rows, cols = checked_matrix(
            view, tlt.InPointer[[5, 7], [InputStride, 1]]
        )
        self.assertIs(pointer, x)
        self.assertEqual((stride, rows, cols), (11, 5, 7))
        with self.assertRaisesRegex(ValueError, "stride"):
            as_host_tensor(x[:, ::2])
        with self.assertRaisesRegex(ValueError, "stride"):
            as_host_tensor(x[:, ::2], host_tensor.Tensor[[5, 4], [InputStride, 1]])
        overlapping = torch.as_strided(torch.empty(3), (2, 2), (1, 1))
        overlapping_view = as_host_tensor(
            overlapping, host_tensor.Tensor[[2, 2], [OutputStride, 1]]
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            checked_matrix(overlapping_view, tlt.OutPointer[[2, 2], [OutputStride, 1]])

    @unittest.skipIf(os.environ.get("TRITON_INTERPRET") == "1", "requires JIT")
    def test_frontend_ttir(self) -> None:
        self.assertEqual(
            ["", "", "", "", "", "", "constexpr", "constexpr"],
            [parameter.annotation for parameter in getattr(softmax_kernel, "params")],
        )
        module = compile_ttir(
            softmax_kernel,
            {
                "output_ptr": "*fp32",
                "input_ptr": "*fp32",
                "input_row_stride": "i32",
                "output_row_stride": "i32",
                "n_rows": "i32",
                "n_cols": "i32",
            },
            {"BLOCK_SIZE": 16, "num_stages": 2},
        )
        self.assertIn("tt.func public @softmax_kernel", module)

    @unittest.skipIf(os.environ.get("TRITON_INTERPRET") == "1", "requires JIT")
    def test_jit_launch_contract_without_gpu(self) -> None:
        x = torch.arange(55, dtype=torch.float32).reshape(5, 11)[:, :7]
        y = torch.empty_like(x)
        hook = getattr(softmax_kernel, "pre_run_hooks")[0]
        hook(y, x, x.stride(0), y.stride(0), 5, 7, 8, 2, num_warps=4)
        with self.assertRaisesRegex(ValueError, "input_row_stride"):
            hook(y, x, y.stride(0), x.stride(0), 5, 7, 8, 2, num_warps=4)

    @unittest.skipUnless(
        os.environ.get("TRITON_INTERPRET") == "1", "requires interpreter"
    )
    def test_irregular_matrix_and_padded_rows(self) -> None:
        x = torch.arange(55, dtype=torch.float32).reshape(5, 11)[:, :7]
        self.assertEqual(x.stride(), (11, 1))
        result = softmax(as_host_tensor(x))
        self.assertEqual(result.shape, x.shape)
        torch.testing.assert_close(
            result, torch.softmax(x, dim=1), atol=1e-6, rtol=1e-6
        )

    @unittest.skipUnless(
        os.environ.get("TRITON_INTERPRET") == "1", "requires interpreter"
    )
    def test_launch_rejects_wrong_stride_and_shape(self) -> None:
        x = torch.arange(55, dtype=torch.float32).reshape(5, 11)[:, :7]
        y = torch.empty_like(x)
        input_view = as_host_tensor(x, host_tensor.Tensor[[5, 7], [InputStride, 1]])
        output_view = as_host_tensor(y, host_tensor.Tensor[[5, 7], [OutputStride, 1]])
        input_ptr, input_stride, rows, cols = checked_matrix(
            input_view, tlt.InPointer[[5, 7], [InputStride, 1]]
        )
        output_ptr, output_stride, _, _ = checked_matrix(
            output_view, tlt.OutPointer[[5, 7], [OutputStride, 1]]
        )
        with self.assertRaisesRegex(ValueError, "input_row_stride"):
            softmax_kernel[(1,)](
                output_ptr, input_ptr, output_stride, input_stride, rows, cols, 8, 2
            )

        wrong_x = torch.empty((5, 8), dtype=x.dtype)
        wrong_view = as_host_tensor(
            wrong_x, host_tensor.Tensor[[5, 8], [InputStride, 1]]
        )
        wrong_input, wrong_stride, _, _ = checked_matrix(
            wrong_view, tlt.InPointer[[5, 8], [InputStride, 1]]
        )
        with self.assertRaisesRegex(ValueError, r"input_ptr\.shape\[1\]"):
            softmax_kernel[(1,)](
                output_ptr,
                wrong_input,
                wrong_stride,
                output_stride,
                rows,
                cols,
                8,
                2,
            )


if TYPE_CHECKING:
    typed_input: torch.Tensor[[5, 7]] = torch.empty((5, 7))
    checked_input = as_host_tensor(typed_input)
    assert_type(checked_input, host_tensor.Tensor[[5, 7], [int, 1]])
    assert_type(
        grid_stride_output(
            checked_input,
            4,
            column_block=8,
            block_parameter="BLOCK_SIZE",
            shape_parameters=("n_rows", "n_cols"),
        ),
        GridStrideOutputLayout[[5, 7]],
    )
    assert_type(softmax(as_host_tensor(typed_input)), torch.Tensor[[5, 7]])

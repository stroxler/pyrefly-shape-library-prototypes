"""Exercise strided pointer arrays without changing the tutorial kernels."""

import os
import unittest
from typing import Any, cast

import torch
import triton.language as tl
from shape_extensions import Int, IntVar

from triton_examples.testing import compile_ttir
from triton_library import tlt
from triton_library.semantic_jit import ConstExpr, semantic_jit

N = IntVar("N")
Stride = IntVar("Stride")
Block = IntVar("Block")
Rows = IntVar("Rows")
Cols = IntVar("Cols")
RowStride = IntVar("RowStride")
ColumnStride = IntVar("ColumnStride")
OutputStride = IntVar("OutputStride")


@semantic_jit
def copy_strided_kernel(
    input_ptr: tlt.InPointer[[N], [Stride]],
    output_ptr: tlt.OutPointer[[N], [1]],
    n_elements: Int[N],
    input_stride: Int[Stride],
    BLOCK_SIZE: ConstExpr[Int[Block]],
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets * input_stride, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, values, mask=mask)


@semantic_jit
def copy_strided_matrix_kernel(
    input_ptr: tlt.InPointer[[Rows, Cols], [RowStride, ColumnStride]],
    output_ptr: tlt.OutPointer[[Rows, Cols], [OutputStride, 1]],
    input_row_stride: Int[RowStride],
    input_column_stride: Int[ColumnStride],
    output_row_stride: Int[OutputStride],
    n_rows: Int[Rows],
    n_cols: Int[Cols],
    BLOCK_SIZE: ConstExpr[Int[Block]],
    num_stages: ConstExpr[int],
):
    for row in tl.range(
        tl.program_id(0), n_rows, tl.num_programs(0), num_stages=num_stages
    ):
        columns = tl.arange(0, BLOCK_SIZE)
        mask = columns < n_cols
        input_row = input_ptr + row * input_row_stride
        output_row = output_ptr + row * output_row_stride
        values = tl.load(
            input_row + columns * input_column_stride, mask=mask, other=0.0
        )
        tl.store(output_row + columns, values, mask=mask)


class StridedCopyTest(unittest.TestCase):
    """Check that the address step matches the input pointer's element stride."""

    @unittest.skipIf(os.environ.get("TRITON_INTERPRET") == "1", "requires JIT")
    def test_evaluated_pointer_contracts(self) -> None:
        x = torch.arange(60, dtype=torch.float32)[::2]
        output = torch.empty(30)
        vector_hook = cast(Any, copy_strided_kernel).pre_run_hooks[0]
        vector_hook(x, output, 30, 2, 16)
        with self.assertRaisesRegex(ValueError, "input_stride"):
            vector_hook(x, output, 30, 1, 16)

        matrix = torch.arange(77, dtype=torch.float32).reshape(7, 11)[:, ::2]
        out_matrix = torch.empty(matrix.shape)
        matrix_hook = cast(Any, copy_strided_matrix_kernel).pre_run_hooks[0]
        with self.assertRaisesRegex(ValueError, "input_column_stride"):
            matrix_hook(
                matrix,
                out_matrix,
                matrix.stride(0),
                1,
                out_matrix.stride(0),
                *matrix.shape,
                16,
                2,
            )

    @unittest.skipIf(os.environ.get("TRITON_INTERPRET") == "1", "requires JIT")
    def test_frontend_ttir(self) -> None:
        module = compile_ttir(
            copy_strided_kernel,
            {
                "input_ptr": "*fp32",
                "output_ptr": "*fp32",
                "n_elements": "i32",
                "input_stride": "i32",
            },
            {"BLOCK_SIZE": 16},
        )
        self.assertIn("tt.func public @copy_strided_kernel", module)

    @unittest.skipUnless(
        os.environ.get("TRITON_INTERPRET") == "1", "requires interpreter"
    )
    def test_sliced_input(self) -> None:
        for stride in (1, 2, 3):
            with self.subTest(stride=stride):
                x = torch.arange(30 * stride, dtype=torch.float32)[::stride]
                out = torch.empty((30,), dtype=torch.float32)
                cast(Any, copy_strided_kernel)[(2,)](x, out, 30, x.stride(0), 16)
                torch.testing.assert_close(out, x)

    @unittest.skipIf(os.environ.get("TRITON_INTERPRET") == "1", "requires JIT")
    def test_matrix_frontend_ttir(self) -> None:
        module = compile_ttir(
            copy_strided_matrix_kernel,
            {
                "input_ptr": "*fp32",
                "output_ptr": "*fp32",
                "input_row_stride": "i32",
                "input_column_stride": "i32",
                "output_row_stride": "i32",
                "n_rows": "i32",
                "n_cols": "i32",
            },
            {"BLOCK_SIZE": 16, "num_stages": 2},
        )
        self.assertIn("tt.func public @copy_strided_matrix_kernel", module)

    @unittest.skipUnless(
        os.environ.get("TRITON_INTERPRET") == "1", "requires interpreter"
    )
    def test_sliced_and_transposed_matrix(self) -> None:
        sliced = torch.arange(77, dtype=torch.float32).reshape(7, 11)[:, ::2]
        for x in (sliced, sliced.T):
            with self.subTest(strides=x.stride()):
                out = torch.empty(x.shape, dtype=x.dtype)
                cast(Any, copy_strided_matrix_kernel)[(2,)](
                    x, out, *x.stride(), out.stride(0), *x.shape, 16, 2
                )
                torch.testing.assert_close(out, x)

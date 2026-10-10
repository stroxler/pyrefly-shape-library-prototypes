"""Execute Triton's original vector-add body with static semantic annotations."""

import os
import unittest
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Literal, assert_type, cast

import torch
import triton
import triton.language as tl
from shape_extensions import Int, IntVar

from triton_examples.testing import compile_ttir
from triton_library import host_tensor, tlt
from triton_library.launch_layout import TiledOutputLayout, tiled_output
from triton_library.semantic_jit import ConstExpr, semantic_jit
from triton_library.torch_views import as_host_tensor, checked_vector

N = IntVar("N")
Block = IntVar("Block")


@semantic_jit
def add_kernel(
    x_ptr: tlt.InPointer[[N], [1]],
    y_ptr: tlt.InPointer[[N], [1]],
    output_ptr: tlt.OutPointer[[N], [1]],
    n_elements: Int[N],
    BLOCK_SIZE: ConstExpr[Int[Block]],
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def checked_add[Length: IntVar](
    x: host_tensor.Tensor[[Length], [1]],
    y: host_tensor.Tensor[[Length], [1]],
    *,
    block_size: int = 16,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor[[Length]]:
    """Validate host arrays and launch the original vector-add kernel."""
    if type(block_size) is not int or block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    if not isinstance(output_dtype, torch.dtype):
        raise ValueError("output_dtype must be a torch.dtype")
    x_ptr, n_elements = checked_vector(x, tlt.InPointer[[Length], [1]])
    y_ptr, y_length = checked_vector(y, tlt.InPointer[[Length], [1]])
    if y_length != n_elements:
        raise ValueError("Input lengths must match")
    if x.device != y.device:
        raise ValueError("Input devices must match")
    output = torch.empty((n_elements,), dtype=output_dtype, device=x.device)
    output_view = as_host_tensor(output, host_tensor.Tensor[[Length], [1]])
    output_ptr, _ = checked_vector(output_view, tlt.OutPointer[[Length], [1]])
    if n_elements:
        layout = tiled_output(
            output_view,
            (block_size,),
            shape_parameters=("n_elements",),
            tile_parameters=("BLOCK_SIZE",),
        )
        device = (
            torch.cuda.device(x.device) if x.device.type == "cuda" else nullcontext()
        )
        with device:
            layout.launch(add_kernel, x_ptr, y_ptr, output_ptr, n_elements, block_size)
    return output


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Upstream tutorial wrapper, with device detection local to the call."""
    output = torch.empty_like(x)
    assert x.device == y.device == output.device
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    cast(Any, add_kernel)[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


class VectorAddTest(unittest.TestCase):
    """Check the vector-add contract in both Triton execution modes."""

    def test_tiled_layout_checks_vector_launch(self) -> None:
        output = torch.empty(30)
        layout = tiled_output(
            as_host_tensor(output),
            (16,),
            shape_parameters=("n_elements",),
            tile_parameters=("BLOCK_SIZE",),
        )
        self.assertEqual(layout.grid, (2,))
        input_ptr, _ = checked_vector(as_host_tensor(output), tlt.InPointer[[30], [1]])
        output_ptr, _ = checked_vector(
            as_host_tensor(output), tlt.OutPointer[[30], [1]]
        )
        for name, n_elements, block_size in (
            ("n_elements", 29, 16),
            ("BLOCK_SIZE", 30, 32),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                layout.launch(
                    add_kernel,
                    input_ptr,
                    input_ptr,
                    output_ptr,
                    n_elements,
                    block_size,
                )

    @unittest.skipIf(os.environ.get("TRITON_INTERPRET") == "1", "requires normal JIT")
    def test_jit_annotations(self) -> None:
        kernel = cast(Any, add_kernel)
        self.assertEqual({"BLOCK_SIZE": tl.constexpr}, kernel.fn.__annotations__)
        self.assertEqual(
            ["", "", "", "", "constexpr"],
            [parameter.annotation for parameter in kernel.params],
        )

    @unittest.skipIf(os.environ.get("TRITON_INTERPRET") == "1", "requires normal JIT")
    def test_frontend_ttir(self) -> None:
        module = compile_ttir(
            add_kernel,
            {
                "x_ptr": "*fp32",
                "y_ptr": "*fp32",
                "output_ptr": "*fp32",
                "n_elements": "i32",
            },
            {"BLOCK_SIZE": 16},
        )
        self.assertIn("tt.func public @add_kernel", module)

    @unittest.skipIf(os.environ.get("TRITON_INTERPRET") == "1", "requires normal JIT")
    def test_direct_launch_contract_without_gpu(self) -> None:
        x = torch.arange(30, dtype=torch.float32)
        out = torch.empty_like(x)
        hook = cast(Any, add_kernel).pre_run_hooks[0]
        with self.assertRaisesRegex(ValueError, r"y_ptr\.shape\[0\]"):
            hook(x, torch.arange(31, dtype=torch.float32), out, 30, 16)

    @unittest.skipUnless(
        os.environ.get("TRITON_INTERPRET") == "1", "requires interpreter"
    )
    def test_direct_kernel_launch(self) -> None:
        x = torch.arange(30, dtype=torch.float32)
        y = torch.arange(30, dtype=torch.float32)
        out = torch.empty_like(x)
        cast(Any, add_kernel)[(2,)](x, y, out, 30, 16)
        torch.testing.assert_close(out, x + y)

    @unittest.skipUnless(
        os.environ.get("TRITON_INTERPRET") == "1", "requires interpreter"
    )
    def test_checked_add(self) -> None:
        for length in (0, 1, 15, 16, 17, 30, 33, 257):
            with self.subTest(length=length):
                x = torch.arange(length, dtype=torch.float32)
                y = x * 3
                result = checked_add(as_host_tensor(x), as_host_tensor(y))
                self.assertEqual(x.shape, result.shape)
                if length:
                    self.assertNotEqual(x.data_ptr(), result.data_ptr())
                torch.testing.assert_close(result, x + y)

    @unittest.skipUnless(
        os.environ.get("TRITON_INTERPRET") == "1", "requires interpreter"
    )
    def test_checked_add_rejects_invalid_inputs(self) -> None:
        good = torch.arange(30, dtype=torch.float32)
        bad_inputs = (
            ("length", torch.arange(31, dtype=torch.float32)),
            ("one-dimensional", good.reshape(2, 15)),
            ("stride", torch.arange(60, dtype=torch.float32)[::2]),
        )
        for error, invalid in bad_inputs:
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    checked_add(
                        as_host_tensor(good), as_host_tensor(cast(Any, invalid))
                    )

        for block_size in (0, 3):
            with self.subTest(block_size=block_size):
                with self.assertRaises(ValueError):
                    checked_add(
                        as_host_tensor(good),
                        as_host_tensor(good),
                        block_size=block_size,
                    )

        with self.assertRaisesRegex(ValueError, "output_dtype"):
            checked_add(
                as_host_tensor(good),
                as_host_tensor(good),
                output_dtype=cast(Any, None),
            )

    @unittest.skipUnless(
        os.environ.get("TRITON_INTERPRET") == "1", "requires interpreter"
    )
    def test_output_dtype_is_explicit_and_inputs_can_differ(self) -> None:
        x = torch.arange(30, dtype=torch.float32)
        y = torch.arange(30, dtype=torch.int32)
        result = checked_add(as_host_tensor(x), as_host_tensor(y))
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, x + y)

        double_result = checked_add(
            as_host_tensor(x), as_host_tensor(y), output_dtype=torch.float64
        )
        self.assertEqual(double_result.dtype, torch.float64)
        torch.testing.assert_close(double_result, (x + y).to(torch.float64))

    @unittest.skipUnless(
        os.environ.get("TRITON_INTERPRET") == "1", "requires interpreter"
    )
    def test_upstream_block_size(self) -> None:
        x = torch.arange(30, dtype=torch.float32)
        torch.testing.assert_close(add(x, x), x + x)
        torch.testing.assert_close(
            checked_add(x=as_host_tensor(x), y=as_host_tensor(x), block_size=1024),
            add(x, x),
        )


if TYPE_CHECKING:

    def check_scalar_program_shift[Length: IntVar](
        input: tlt.InPointer[[Length], [1]],
        output: tlt.OutPointer[[Length], [1]],
        strided: tlt.InPointer[[Length], [2]],
        strided_output: tlt.OutPointer[[Length], [2]],
        matrix: tlt.InPointer[[Length, Length], [Length, 1]],
        value: tl.tensor[[]],
    ) -> None:
        tl.load(input + tl.program_id(0))
        tl.store(output + tl.program_id(0), value)
        strided + tl.program_id(0)  # pyrefly: ignore[unsupported-operation]
        strided_output + tl.program_id(0)  # pyrefly: ignore[unsupported-operation]
        matrix + tl.program_id(0)  # pyrefly: ignore[unsupported-operation]

    def check_launch_axis_mask[Length: IntVar, Width: IntVar](
        pointer: tlt.InPointer[[Length], [1]],
        output: tlt.OutPointer[[Length], [1]],
        length: Int[Length],
        width: Int[Width],
    ) -> None:
        offsets_0 = tl.program_id(0) * width + tl.arange(0, width)
        offsets_1 = tl.program_id(1) * width + tl.arange(0, width)
        assert_type(offsets_0, tl.Offsets[[Width], [1], Literal["program"], Literal[0]])
        assert_type(offsets_1, tl.Offsets[[Width], [1], Literal["program"], Literal[1]])
        assert_type(
            offsets_0 < length,
            tl.Mask[[Length], [Width], Literal["program"], Literal[0]],
        )
        assert_type(
            offsets_1 < length,
            tl.Mask[[Length], [Width], Literal["program"], Literal[1]],
        )
        wrong_tag: tl.Mask[[Length], [Width], Literal["program"], Literal[0]] = (
            offsets_1 < length  # pyrefly: ignore[bad-assignment]
        )
        values = tl.load(pointer + offsets_0, mask=offsets_0 < length)
        tl.store(output + offsets_0, values, mask=offsets_0 < length)
        tl.load(  # pyrefly: ignore[no-matching-overload]
            pointer + offsets_0, mask=offsets_1 < length
        )
        tl.store(  # pyrefly: ignore[no-matching-overload]
            output + offsets_0, values, mask=offsets_1 < length
        )

    def check_joined_axis[Length: IntVar, Width: IntVar](
        pointer: tlt.InPointer[[Length], [1]],
        output: tlt.OutPointer[[Length], [1]],
        values: tl.tensor[[Width]],
        length: Int[Length],
        width: Int[Width],
        condition: bool,
    ) -> None:
        offsets_0 = tl.program_id(0) * width + tl.arange(0, width)
        offsets_1 = tl.program_id(1) * width + tl.arange(0, width)
        if condition:
            offsets = offsets_0
        else:
            offsets = offsets_1
        tl.load(  # pyrefly: ignore[no-matching-overload]
            pointer + offsets, mask=offsets_0 < length
        )
        tl.load(  # pyrefly: ignore[no-matching-overload]
            pointer + offsets_0, mask=offsets < length
        )
        tl.store(  # pyrefly: ignore[no-matching-overload]
            output + offsets, values, mask=offsets_0 < length
        )
        tl.store(  # pyrefly: ignore[no-matching-overload]
            output + offsets_0, values, mask=offsets < length
        )

    typed_input: torch.Tensor[[7]] = torch.arange(7)
    typed_view = as_host_tensor(typed_input)
    assert_type(
        tiled_output(
            typed_view,
            (16,),
            shape_parameters=("n_elements",),
            tile_parameters=("BLOCK_SIZE",),
        ),
        TiledOutputLayout[[7], [16]],
    )
    assert_type(
        checked_add(as_host_tensor(typed_input), as_host_tensor(typed_input)),
        torch.Tensor[[7]],
    )

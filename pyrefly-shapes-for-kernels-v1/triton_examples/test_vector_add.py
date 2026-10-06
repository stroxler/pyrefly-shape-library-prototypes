"""Execute Triton's original vector-add body with static semantic annotations."""

from __future__ import annotations

import inspect
import os
import unittest
from typing import Any, cast

import torch
import triton
import triton.language as tl
from shape_extensions import Int, IntVar

from triton_examples.testing import compile_ttir
from triton_library.semantic_jit import ConstExpr, semantic_jit
from triton_library.torch_wrapper import Launch1D, make_torch_wrapper

N = IntVar("N")
Block = IntVar("Block")


@semantic_jit
def add_kernel(
    x_ptr: tl.InPointer[[N]],
    y_ptr: tl.InPointer[[N]],
    output_ptr: tl.OutPointer[[N]],
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
    def test_generated_wrapper(self) -> None:
        add = make_torch_wrapper(add_kernel, Launch1D(16, torch.float32))
        self.assertEqual(["x", "y"], list(inspect.signature(add).parameters))
        for length in (0, 1, 15, 16, 17, 30, 33, 257):
            with self.subTest(length=length):
                x = torch.arange(length, dtype=torch.float32)
                y = x * 3
                result = add(x, y)
                self.assertEqual(x.shape, result.shape)
                if length:
                    self.assertNotEqual(x.data_ptr(), result.data_ptr())
                torch.testing.assert_close(result, x + y)

    @unittest.skipUnless(
        os.environ.get("TRITON_INTERPRET") == "1", "requires interpreter"
    )
    def test_wrapper_rejects_invalid_inputs(self) -> None:
        add = make_torch_wrapper(add_kernel, Launch1D(16, torch.float32))
        good = torch.arange(30, dtype=torch.float32)
        bad_inputs = (
            ("length", torch.arange(31, dtype=torch.float32)),
            ("one-dimensional", good.reshape(2, 15)),
            ("contiguous", torch.arange(60, dtype=torch.float32)[::2]),
        )
        for error, invalid in bad_inputs:
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    add(good, invalid)

        for block_size in (0, 3):
            with self.subTest(block_size=block_size):
                with self.assertRaises(ValueError):
                    make_torch_wrapper(add_kernel, Launch1D(block_size, torch.float32))

        with self.assertRaisesRegex(ValueError, "output_dtype"):
            make_torch_wrapper(add_kernel, Launch1D(16, cast(Any, None)))

    @unittest.skipUnless(
        os.environ.get("TRITON_INTERPRET") == "1", "requires interpreter"
    )
    def test_output_dtype_is_explicit_and_inputs_can_differ(self) -> None:
        add = make_torch_wrapper(add_kernel, Launch1D(16, torch.float32))
        x = torch.arange(30, dtype=torch.float32)
        y = torch.arange(30, dtype=torch.int32)
        result = add(x, y)
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, x + y)

        add_double = make_torch_wrapper(add_kernel, Launch1D(16, torch.float64))
        double_result = add_double(x, y)
        self.assertEqual(double_result.dtype, torch.float64)
        torch.testing.assert_close(double_result, (x + y).to(torch.float64))

    @unittest.skipUnless(
        os.environ.get("TRITON_INTERPRET") == "1", "requires interpreter"
    )
    def test_upstream_block_size(self) -> None:
        generated_add = make_torch_wrapper(add_kernel, Launch1D(1024, torch.float32))
        x = torch.arange(30, dtype=torch.float32)
        torch.testing.assert_close(add(x, x), x + x)
        torch.testing.assert_close(generated_add(x=x, y=x), add(x, x))

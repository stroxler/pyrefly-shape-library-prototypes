"""Check Triton's tutorial 07 libdevice asin kernel and its host boundary.

The kernel body is copied from Triton's python/tutorials/07-extern-functions.py;
only the semantic parameter annotations differ.
"""

import os
import unittest
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, assert_type, cast
from unittest.mock import patch

import torch
import triton
import triton.language as tl
from shape_extensions import Int, IntVar
from triton.language.extra import libdevice

from triton_examples.testing import compile_ttir
from triton_library import host_tensor, tlt
from triton_library.launch_layout import tiled_output
from triton_library.semantic_jit import ConstExpr, semantic_jit
from triton_library.torch_views import as_host_tensor, checked_vector

N = IntVar("N")
Block = IntVar("Block")


@semantic_jit
def asin_kernel(
    x_ptr: tlt.InPointer[[N], [1]],
    y_ptr: tlt.OutPointer[[N], [1]],
    n_elements: Int[N],
    BLOCK_SIZE: ConstExpr[Int[Block]],
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x = libdevice.asin(x)
    tl.store(y_ptr + offsets, x, mask=mask)


def checked_asin[Length: IntVar](
    x: host_tensor.Tensor[[Length], [1]], *, block_size: int = 16
) -> torch.Tensor[[Length]]:
    """Validate contiguous floating input before launching the original body."""
    if type(block_size) is not int or block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    x_ptr, length = checked_vector(x, tlt.InPointer[[Length], [1]])
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError("asin input must be float32 or float64")
    out = torch.empty((length,), dtype=x.dtype, device=x.device)
    out_view = as_host_tensor(out, host_tensor.Tensor[[Length], [1]])
    out_ptr, _ = checked_vector(out_view, tlt.OutPointer[[Length], [1]])
    if length:
        layout = tiled_output(
            out_view,
            (block_size,),
            shape_parameters=("n_elements",),
            tile_parameters=("BLOCK_SIZE",),
        )
        device = (
            torch.cuda.device(x.device) if x.device.type == "cuda" else nullcontext()
        )
        with device:
            layout.launch(asin_kernel, x_ptr, out_ptr, length, block_size)
    return out


def asin(x: torch.Tensor) -> torch.Tensor:
    """The tutorial wrapper, which leaves the shape/stride contract implicit."""
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    cast(Any, asin_kernel)[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output


class ExternFunctionsTest(unittest.TestCase):
    """Verify the semantic mask and the host-to-device launch metadata."""

    @unittest.skipIf(os.environ.get("TRITON_INTERPRET") == "1", "requires normal JIT")
    def test_frontend(self) -> None:
        ttir = compile_ttir(
            asin_kernel,
            {"x_ptr": "*fp32", "y_ptr": "*fp32", "n_elements": "i32"},
            {"BLOCK_SIZE": 16},
        )
        self.assertIn("tt.func public @asin_kernel", ttir)

    @unittest.skipUnless(
        os.environ.get("TRITON_INTERPRET") == "1", "requires interpreter"
    )
    def test_interpreter_mask_with_identity_extern(self) -> None:
        # CUDA libdevice functions have no CPU interpreter implementation.
        with patch.object(libdevice, "asin", lambda value: value):
            for length in (0, 1, 15, 16, 19, 33):
                with self.subTest(length=length):
                    x = cast(Any, torch.linspace)(-0.95, 0.95, length)
                    actual = checked_asin(as_host_tensor(x))
                    torch.testing.assert_close(actual, x)

    def test_boundary_checks(self) -> None:
        x = torch.linspace(-0.5, 0.5, 19)
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            checked_asin(as_host_tensor(cast(Any, x.reshape(1, 19))))
        with self.assertRaisesRegex(ValueError, "stride"):
            checked_asin(as_host_tensor(cast(Any, x[::2])))
        with self.assertRaisesRegex(ValueError, "float32 or float64"):
            checked_asin(as_host_tensor(cast(Any, torch.arange(19))))
        with self.assertRaisesRegex(ValueError, "power of two"):
            checked_asin(as_host_tensor(x), block_size=3)


if TYPE_CHECKING:

    def check_shape[Length: IntVar, Other: IntVar, Tile: IntVar](
        x: host_tensor.Tensor[[Length], [1]],
        value: tl.tensor[[Tile]],
    ) -> None:
        assert_type(checked_asin(x), torch.Tensor[[Length]])
        assert_type(libdevice.asin(value), tl.tensor[[Tile]])
        v: torch.Tensor[[Other]] = checked_asin(x)  # pyrefly: ignore[bad-assignment]

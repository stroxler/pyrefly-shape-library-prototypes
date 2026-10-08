# The kernel body is copied from Triton's python/tutorials/cpu/01-vector-add.py.
# Triton's LICENSE includes the following notice:
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Check Triton's CPU tutorial vector add with multiple tiles per program."""

import os
import unittest
from typing import Any, cast

import torch
import triton.language as tl
from shape_extensions import Int, IntVar

from triton_examples.testing import compile_ttir
from triton_library import host_tensor, tlt
from triton_library.launch_layout import tiled_output
from triton_library.semantic_jit import ConstExpr, semantic_jit
from triton_library.torch_views import as_host_tensor, checked_vector

Length = IntVar("Length")
Block = IntVar("Block")
Tile = IntVar("Tile")
Other = IntVar("Other")


@semantic_jit
def add_kernel_tiled(
    x_ptr: tlt.InPointer[[Length], [1]],
    y_ptr: tlt.InPointer[[Length], [1]],
    output_ptr: tlt.OutPointer[[Length], [1]],
    n_elements: Int[Length],
    BLOCK_SIZE: ConstExpr[Int[Block]],
    TILE_SIZE: ConstExpr[Int[Tile]],
):
    # Body from Triton's python/tutorials/cpu/01-vector-add.py:add_kernel_tiled.
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    for i in range(0, tl.cdiv(BLOCK_SIZE, TILE_SIZE)):
        offsets = block_start + i * TILE_SIZE + tl.arange(0, TILE_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)


def checked_tiled_add[Size: IntVar](
    x: host_tensor.Tensor[[Size], [1]],
    y: host_tensor.Tensor[[Size], [1]],
    *,
    block_size: int = 16,
    tile_size: int = 4,
) -> torch.Tensor[[Size]]:
    """Bind both tile sizes and allocate an output before launching."""
    if (
        type(block_size) is not int
        or type(tile_size) is not int
        or block_size <= 0
        or tile_size <= 0
        or block_size % tile_size
        or block_size & (block_size - 1)
        or tile_size & (tile_size - 1)
    ):
        raise ValueError("Block and tile sizes must be divisible powers of two")
    x_ptr, length = checked_vector(x, tlt.InPointer[[Size], [1]])
    y_ptr, y_length = checked_vector(y, tlt.InPointer[[Size], [1]])
    if y_length != length or x.device != y.device:
        raise ValueError("Input lengths and devices must match")
    if x.dtype != y.dtype:
        raise ValueError("Input dtypes must match")
    output = torch.empty((length,), device=x.device, dtype=x.dtype)
    output_view = as_host_tensor(output, host_tensor.Tensor[[Size], [1]])
    output_ptr, _ = checked_vector(output_view, tlt.OutPointer[[Size], [1]])
    if length:
        layout = tiled_output(
            output_view,
            (block_size,),
            shape_parameters=("n_elements",),
            tile_parameters=("BLOCK_SIZE",),
            metadata={"TILE_SIZE": tile_size},
        )
        layout.launch(
            add_kernel_tiled, x_ptr, y_ptr, output_ptr, length, block_size, tile_size
        )
    return output


@semantic_jit
def add_kernel_tiled_wrong_y(
    x_ptr: tlt.InPointer[[Length], [1]],
    y_ptr: tlt.InPointer[[Other], [1]],
    output_ptr: tlt.OutPointer[[Length], [1]],
    n_elements: Int[Length],
    BLOCK_SIZE: ConstExpr[Int[Block]],
    TILE_SIZE: ConstExpr[Int[Tile]],
):
    """Only the Y allocation extent differs from the upstream kernel."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    for i in range(0, tl.cdiv(BLOCK_SIZE, TILE_SIZE)):
        offsets = block_start + i * TILE_SIZE + tl.arange(0, TILE_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)  # pyrefly: ignore[no-matching-overload]
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)


class CpuTiledAddTest(unittest.TestCase):
    @unittest.skipIf(os.environ.get("TRITON_INTERPRET") == "1", "requires JIT")
    def test_frontend_and_metadata(self) -> None:
        ir = compile_ttir(
            add_kernel_tiled,
            {
                "x_ptr": "*fp32",
                "y_ptr": "*fp32",
                "output_ptr": "*fp32",
                "n_elements": "i32",
            },
            {"BLOCK_SIZE": 16, "TILE_SIZE": 4},
        )
        self.assertIn("tt.func public @add_kernel_tiled", ir)

    @unittest.skipUnless(
        os.environ.get("TRITON_INTERPRET") == "1", "requires interpreter"
    )
    def test_multiple_tiles_and_tail(self) -> None:
        x = torch.arange(30, dtype=torch.float32)
        y = x * 2
        result = checked_tiled_add(as_host_tensor(x), as_host_tensor(y))
        torch.testing.assert_close(result, x + y)

    def test_bad_host_contract(self) -> None:
        x = torch.arange(8, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "lengths"):
            checked_tiled_add(
                as_host_tensor(x), as_host_tensor(cast(Any, torch.ones(7)))
            )
        with self.assertRaisesRegex(ValueError, "divisible"):
            checked_tiled_add(as_host_tensor(x), as_host_tensor(x), tile_size=3)

# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# The add_kernel body is copied from Triton's beta numbered tutorial
# 11-programmatic-dependent-launch.py. Triton's LICENSE includes the following:
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Typed vector-add tiling with an optional CUDA dependent-launch hint.

The checked CUDA path sets USE_GDC and launch_pdl together. The latter is a
Triton launch option rather than a kernel argument; type checking cannot prove
its agreement with USE_GDC or the memory-ordering semantics of gdc_wait.
"""

from __future__ import annotations

import os
import unittest
from contextlib import nullcontext
from typing import Protocol, cast

import torch
import triton.language as tl
from shape_extensions import Int, IntVar

from triton_examples.testing import compile_ttir
from triton_library import host_tensor, tlt
from triton_library.launch_layout import tiled_output
from triton_library.semantic_jit import ConstExpr, semantic_jit
from triton_library.torch_views import as_host_tensor, checked_vector

N = IntVar("N")
Block = IntVar("Block")


@semantic_jit
def add_kernel(
    x_ptr: tlt.InPointer[[N], [1]],  #
    y_ptr: tlt.InPointer[[N], [1]],  #
    output_ptr: tlt.OutPointer[[N], [1]],  #
    n_elements: Int[N],  #
    BLOCK_SIZE: ConstExpr[Int[Block]],  #
    USE_GDC: ConstExpr[bool],  #
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    if USE_GDC:
        # GDC wait waits for ALL programs in the the prior kernel to complete before continuing.
        # This ensures any memory operations happen before the wait in program order,
        # e.g. if the prior kernel writes to x or y the new values will be visible.
        tl.extra.cuda.gdc_wait()

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    if USE_GDC:
        # GDC launch dependents hints the runtime system to launch dependent kernels.
        # These dependent kernels must also be launched with PDL enabled.
        # Once GDC launch has been issued by ALL programs or
        # programs have finished, the dependent grid can begin if there are enough resources.
        # Note: this by itself provides no additional memory-ordering guarentees, unlike `gdc_wait`
        tl.extra.cuda.gdc_launch_dependents()
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


class PdlLaunch(Protocol):
    """Triton's out-of-band launch flag, absent from the JIT function signature."""

    def __call__[Length: IntVar, Tile: IntVar](
        self,
        x: tlt.InPointer[[Length], [1]],
        y: tlt.InPointer[[Length], [1]],
        output: tlt.OutPointer[[Length], [1]],
        length: Int[Length],
        tile: Int[Tile],
        use_gdc: bool,
        *,
        launch_pdl: bool,
    ) -> None: ...


def checked_pdl_add[Length: IntVar](
    x: host_tensor.Tensor[[Length], [1]],
    y: host_tensor.Tensor[[Length], [1]],
    *,
    block_size: int = 1024,
    launch_pdl: bool = False,
) -> torch.Tensor[[Length]]:
    """Validate FP32 host axes and output before using Triton's PDL option."""
    if type(block_size) is not int or block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    if type(launch_pdl) is not bool:
        raise ValueError("launch_pdl must be a bool")
    x_ptr, length = checked_vector(x, tlt.InPointer[[Length], [1]])
    y_ptr, y_length = checked_vector(y, tlt.InPointer[[Length], [1]])
    if y_length != length or x.device != y.device:
        raise ValueError("Input lengths and devices must match")
    if x.dtype != torch.float32 or y.dtype != torch.float32:
        raise ValueError("Input and output dtypes must be float32")
    if launch_pdl and (
        x.device.type != "cuda" or torch.cuda.get_device_capability(x.device)[0] < 9
    ):
        raise ValueError("PDL requires a CUDA compute capability of at least 9")
    output = torch.empty((length,), dtype=torch.float32, device=x.device)
    output_view = as_host_tensor(output, host_tensor.Tensor[[Length], [1]])
    output_ptr, _ = checked_vector(output_view, tlt.OutPointer[[Length], [1]])
    if length:
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
            if launch_pdl:
                # The flag is not a kernel argument; this path uses the same checked grid.
                cast(PdlLaunch, add_kernel[layout.grid])(
                    x_ptr, y_ptr, output_ptr, length, block_size, True, launch_pdl=True
                )
            else:
                layout.launch(
                    add_kernel, x_ptr, y_ptr, output_ptr, length, block_size, False
                )
    return output


class ProgrammaticDependentLaunchTest(unittest.TestCase):
    """Check PDL's intrinsic branch and the ordinary interpreter addition."""

    def test_frontend(self) -> None:
        if os.environ.get("TRITON_INTERPRET") == "1":
            self.skipTest("Frontend compilation uses normal JIT mode")
        signature = {
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "output_ptr": "*fp32",
            "n_elements": "i32",
        }
        for use_gdc in (0, 1):
            with self.subTest(use_gdc=use_gdc):
                ir = compile_ttir(
                    add_kernel,
                    signature=signature,
                    constexprs={"BLOCK_SIZE": 128, "USE_GDC": use_gdc},
                )
                self.assertIn("tt.func public @add_kernel", ir)

    def test_invalid_host_contract(self) -> None:
        x = torch.arange(30, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "lengths"):
            checked_pdl_add(
                as_host_tensor(x),
                cast("host_tensor.Tensor[[30], [1]]", as_host_tensor(torch.ones(31))),
            )
        with self.assertRaisesRegex(ValueError, "float32"):
            checked_pdl_add(as_host_tensor(x), as_host_tensor(x.double()))
        with self.assertRaisesRegex(ValueError, "one-dimensional|stride"):
            checked_pdl_add(
                as_host_tensor(x),
                cast("host_tensor.Tensor[[30], [1]]", as_host_tensor(x[::2])),
            )
        with self.assertRaisesRegex(ValueError, "block_size"):
            checked_pdl_add(as_host_tensor(x), as_host_tensor(x), block_size=3)
        with self.assertRaisesRegex(ValueError, "compute capability"):
            checked_pdl_add(as_host_tensor(x), as_host_tensor(x), launch_pdl=True)

    def test_interpreter(self) -> None:
        if os.environ.get("TRITON_INTERPRET") != "1":
            self.skipTest("CPU launch requires interpreter mode")
        for length in (0, 1, 127, 128, 129, 257):
            with self.subTest(length=length):
                x = torch.arange(length, dtype=torch.float32)
                got = checked_pdl_add(
                    as_host_tensor(x), as_host_tensor(x * 2), block_size=128
                )
                torch.testing.assert_close(got, x * 3)

"""Check the explicit-mask and seeded Triton tutorial dropout kernels."""

import os
import unittest
from typing import TYPE_CHECKING, Any, assert_type, cast

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
def _dropout(
    x_ptr: tlt.InPointer[[N], [1]],
    x_keep_ptr: tlt.InPointer[[N], [1]],
    output_ptr: tlt.OutPointer[[N], [1]],
    n_elements: Int[N],
    p: float,
    BLOCK_SIZE: ConstExpr[Int[Block]],
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    # The line below is the crucial part, described in the paragraph above!
    output = tl.where(x_keep, x / (1 - p), 0.0)
    # Write-back output
    tl.store(output_ptr + offsets, output, mask=mask)


@semantic_jit
def _seeded_dropout(
    x_ptr: tlt.InPointer[[N], [1]],
    output_ptr: tlt.OutPointer[[N], [1]],
    n_elements: Int[N],
    p: float,
    seed: int,
    BLOCK_SIZE: ConstExpr[Int[Block]],
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # randomly prune it
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def _check_parameters(p: float, block_size: int) -> None:
    """Validate the probability and arange width assumed by both kernels."""
    if not isinstance(p, (int, float)) or not 0 <= p < 1:
        raise ValueError("p must be a probability in [0, 1)")
    if type(block_size) is not int or block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")


def checked_dropout[Length: IntVar](
    x: host_tensor.Tensor[[Length], [1]],
    keep: host_tensor.Tensor[[Length], [1]],
    *,
    p: float,
    block_size: int = 16,
) -> torch.Tensor[[Length]]:
    """Check aligned input and keep-mask allocations before launching dropout."""
    _check_parameters(p, block_size)
    x_ptr, length = checked_vector(x, tlt.InPointer[[Length], [1]])
    keep_ptr, keep_length = checked_vector(keep, tlt.InPointer[[Length], [1]])
    if length != keep_length or x.device != keep.device:
        raise ValueError("Input and keep-mask shapes and devices must match")
    if not x.dtype.is_floating_point or keep.dtype != torch.bool:
        raise ValueError("Input must be floating-point and keep-mask boolean")
    out = torch.empty_like(x)
    output_view = as_host_tensor(out, host_tensor.Tensor[[Length], [1]])
    output_ptr, _ = checked_vector(output_view, tlt.OutPointer[[Length], [1]])
    if length:
        layout = tiled_output(
            output_view,
            (block_size,),
            shape_parameters=("n_elements",),
            tile_parameters=("BLOCK_SIZE",),
        )
        layout.launch(_dropout, x_ptr, keep_ptr, output_ptr, length, p, block_size)
    return out


def checked_seeded_dropout[Length: IntVar](
    x: host_tensor.Tensor[[Length], [1]],
    *,
    p: float,
    seed: int,
    block_size: int = 16,
) -> torch.Tensor[[Length]]:
    """Check the input allocation while preserving the kernel's integer seed."""
    _check_parameters(p, block_size)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    x_ptr, length = checked_vector(x, tlt.InPointer[[Length], [1]])
    if not x.dtype.is_floating_point:
        raise ValueError("Input must be floating-point")
    out = torch.empty_like(x)
    output_view = as_host_tensor(out, host_tensor.Tensor[[Length], [1]])
    output_ptr, _ = checked_vector(output_view, tlt.OutPointer[[Length], [1]])
    if length:
        layout = tiled_output(
            output_view,
            (block_size,),
            shape_parameters=("n_elements",),
            tile_parameters=("BLOCK_SIZE",),
        )
        layout.launch(_seeded_dropout, x_ptr, output_ptr, length, p, seed, block_size)
    return out


class DropoutTest(unittest.TestCase):
    """Exercise both kernel signatures and their Torch boundaries."""

    @unittest.skipIf(os.environ.get("TRITON_INTERPRET") == "1", "requires JIT")
    def test_frontend_ttir(self) -> None:
        for kernel, args in (
            (
                _dropout,
                {
                    "x_ptr": "*fp32",
                    "x_keep_ptr": "*i1",
                    "output_ptr": "*fp32",
                    "n_elements": "i32",
                    "p": "fp32",
                },
            ),
            (
                _seeded_dropout,
                {
                    "x_ptr": "*fp32",
                    "output_ptr": "*fp32",
                    "n_elements": "i32",
                    "p": "fp32",
                    "seed": "i32",
                },
            ),
        ):
            with self.subTest(kernel=kernel):
                module = compile_ttir(kernel, args, {"BLOCK_SIZE": 16})
                self.assertIn("tt.func public", module)

    @unittest.skipIf(os.environ.get("TRITON_INTERPRET") == "1", "requires JIT")
    def test_direct_launch_rejects_mismatched_mask_length(self) -> None:
        x = torch.ones(17)
        out = torch.empty_like(x)
        hook = cast(Any, _dropout).pre_run_hooks[0]
        with self.assertRaisesRegex(ValueError, r"x_keep_ptr\.shape\[0\]"):
            hook(x, torch.ones(16, dtype=torch.bool), out, 17, 0.25, 16)

    @unittest.skipUnless(
        os.environ.get("TRITON_INTERPRET") == "1", "requires interpreter"
    )
    def test_explicit_mask_on_partial_tiles(self) -> None:
        for length in (0, 1, 16, 17, 31):
            with self.subTest(length=length):
                x = torch.arange(length, dtype=torch.float32)
                keep = torch.arange(length) % 3 != 0
                actual = checked_dropout(
                    as_host_tensor(x), as_host_tensor(keep), p=0.25
                )
                torch.testing.assert_close(actual, torch.where(keep, x / 0.75, 0.0))

    @unittest.skipUnless(
        os.environ.get("TRITON_INTERPRET") == "1", "requires interpreter"
    )
    def test_seeded_dropout_is_repeatable(self) -> None:
        x = torch.ones(33, dtype=torch.float32)
        first = checked_seeded_dropout(as_host_tensor(x), p=0.5, seed=42)
        second = checked_seeded_dropout(as_host_tensor(x), p=0.5, seed=42)
        torch.testing.assert_close(first, second)
        self.assertTrue(bool(torch.all((first == 0) | (first == 2))))

    def test_reject_incompatible_host_inputs(self) -> None:
        x = torch.ones(17)
        with self.assertRaisesRegex(ValueError, "shapes and devices"):
            checked_dropout(
                as_host_tensor(x),
                as_host_tensor(cast(Any, torch.ones(16, dtype=torch.bool))),
                p=0.2,
            )
        with self.assertRaisesRegex(ValueError, "keep-mask boolean"):
            checked_dropout(as_host_tensor(x), as_host_tensor(x), p=0.2)
        with self.assertRaisesRegex(ValueError, "probability"):
            checked_seeded_dropout(as_host_tensor(x), p=1.0, seed=42)
        with self.assertRaisesRegex(ValueError, "seed"):
            checked_seeded_dropout(as_host_tensor(x), p=0.2, seed=cast(Any, "bad"))


if TYPE_CHECKING:
    typed_x: torch.Tensor[[17]] = cast(Any, torch).ones(17)
    typed_keep: torch.Tensor[[17]] = cast(Any, torch).ones(17, dtype=torch.bool)
    assert_type(
        checked_dropout(as_host_tensor(typed_x), as_host_tensor(typed_keep), p=0.25),
        torch.Tensor[[17]],
    )
    assert_type(
        checked_seeded_dropout(as_host_tensor(typed_x), p=0.25, seed=42),
        torch.Tensor[[17]],
    )

"""Check direct-launch validation for semantic JIT annotations."""

from __future__ import annotations

import os
import unittest
from typing import Any, cast

import torch
from shape_extensions import Int, IntVar

from triton_library import tlt
from triton_library.semantic_jit import ConstExpr, semantic_jit
from triton_library.tlt import InPointer as InputPointerAlias
from triton_library.torch_views import checked_matrix, checked_vector

Length = IntVar("Length")
Block = IntVar("Block")


@semantic_jit
def postponed_kernel(
    x_ptr: tlt.InPointer[[Length], [1]],
    n_elements: Int[Length],
    BLOCK_SIZE: ConstExpr[Int[Block]],
) -> None:
    pass


@semantic_jit
def partially_annotated_kernel(
    x_ptr: tlt.InPointer[[Length], [1]],
    unused: int,
    BLOCK_SIZE: ConstExpr[Int[Block]],
    extra,
) -> None:
    pass


@semantic_jit
def dependent_constexpr_kernel(
    x_ptr: tlt.InPointer[[Length], [1]],
    n_elements: Int[Length],
    BLOCK_SIZE: ConstExpr[Int[Length]],
) -> None:
    pass


@semantic_jit
def output_view_kernel(
    output: tlt.OutPointer[[int, int], [int, int]],
) -> None:
    pass


@semantic_jit
def output_vector_kernel(output: tlt.OutPointer[[int], [int]]) -> None:
    pass


@semantic_jit
def aliased_pointer_kernel(
    x_ptr: InputPointerAlias[[Length], [1]], n_elements: Int[Length]
) -> None:
    pass


class SemanticJitTest(unittest.TestCase):
    """Validate pointer and scalar contracts before compiling or interpreting."""

    def test_postponed_annotations_register_a_pointer_hook(self) -> None:
        hooks = cast(Any, postponed_kernel).pre_run_hooks
        self.assertEqual(len(hooks), 1)
        with self.assertRaisesRegex(ValueError, "n_elements"):
            hooks[0](torch.ones(4), 5, 8)

    def test_aliased_postponed_pointer_still_registers_a_hook(self) -> None:
        (hook,) = cast(Any, aliased_pointer_kernel).pre_run_hooks
        with self.assertRaisesRegex(ValueError, "n_elements"):
            hook(torch.ones(4), 5)

    def test_unannotated_parameters_do_not_break_pointer_checks(self) -> None:
        (hook,) = cast(Any, partially_annotated_kernel).pre_run_hooks
        hook(torch.ones(4), 19, 8, object())
        with self.assertRaisesRegex(ValueError, r"x_ptr\.stride\(0\)"):
            hook(torch.ones(8)[::2], 19, 8, object())

    def test_constexpr_integer_checks_shared_dimension(self) -> None:
        (hook,) = cast(Any, dependent_constexpr_kernel).pre_run_hooks
        hook(torch.ones(4), 4, 4)
        with self.assertRaisesRegex(ValueError, "BLOCK_SIZE"):
            hook(torch.ones(4), 4, 8)

    def test_output_overlap_agrees_across_direct_and_checked_views(self) -> None:
        (matrix_hook,) = cast(Any, output_view_kernel).pre_run_hooks
        (vector_hook,) = cast(Any, output_vector_kernel).pre_run_hooks
        column_major = torch.zeros(8, 4).T
        padded = torch.zeros(4, 10)[:, :8]
        overlapping = torch.zeros(4, 1).expand(4, 8)
        aliased_strides = torch.as_strided(torch.zeros(3), (2, 2), (1, 1))
        for matrix in (column_major, padded):
            matrix_hook(matrix)
            checked_matrix(cast(Any, matrix), tlt.OutPointer[[int, int], [int, int]])
        for matrix in (overlapping, aliased_strides):
            with self.assertRaisesRegex(ValueError, "must not overlap"):
                matrix_hook(matrix)
            with self.assertRaisesRegex(ValueError, "must not overlap"):
                checked_matrix(
                    cast(Any, matrix), tlt.OutPointer[[int, int], [int, int]]
                )

        expanded = torch.zeros(1).expand(4)
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            vector_hook(expanded)
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            checked_vector(cast(Any, expanded), tlt.OutPointer[[int], [int]])

    @unittest.skipUnless(
        os.environ.get("TRITON_INTERPRET") == "1", "requires interpreter"
    )
    def test_direct_launch_runs_the_postponed_annotation_hook(self) -> None:
        with self.assertRaisesRegex(ValueError, "n_elements"):
            cast(Any, postponed_kernel)[(1,)](torch.ones(4), 5, 8)

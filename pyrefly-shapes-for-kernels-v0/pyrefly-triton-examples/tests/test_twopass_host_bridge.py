# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This static-only probe checks shape allocation, not a runnable Triton wrapper.
# @lint-ignore-every AUTODEPS2

"""Check host allocations and expose the untyped tutorial 12 launch seam.

Use a Python interpreter with installed Torch when running Pyrefly: the Torch
overlay's Tensor inherits ``torch._C.TensorBase``, which must be resolvable for
Tensor-to-Tensor and Tensor-to-pointer assignment checks to remain sound.
"""

from typing import assert_type

import torch
import triton.language as tl
from shape_extensions import Int, IntVar
from tests.test_twopass_compute_matmul import _twopass_compute_kernel
from tests.test_twopass_reduce_matmul import _twopass_reduce_kernel


def require_host_dimensions[Split: IntVar, M: IntVar, N: IntVar](
    split: Int[Split],
    rows: Int[M],
    cols: Int[N],
    scratch_shape: tuple[Int[Split], Int[M], Int[N]],
    output_shape: tuple[Int[M], Int[N]],
) -> None:
    """Check host shape *values*, independently of Triton's launch syntax."""


def require_semantic_pointer[Split: IntVar, M: IntVar, N: IntVar](
    scratch: tl.Scratch3DPointer[Split, M, N, M * N, N, 1],
) -> None:
    """Verify that a host tensor is not implicitly a Triton scratch pointer."""


def test_host_allocations[Split: IntVar, M: IntVar, N: IntVar, Other: IntVar](
    split: Int[Split], m: Int[M], n: Int[N], other_n: Int[Other]
) -> None:
    # These shapes come from torch.empty, as in skinny_twopass_matmul.
    scratch = torch.empty((split, m, n), device="cuda", dtype=torch.float32)
    output = torch.empty((m, n), device="cuda", dtype=torch.float16)
    assert_type(scratch, torch.Tensor[[Split, M, N]])
    assert_type(output, torch.Tensor[[M, N]])
    require_host_dimensions(split, m, n, scratch.shape, output.shape)

    wrong_axes = torch.empty((m, split, n), device="cuda", dtype=torch.float32)
    require_host_dimensions(
        split,
        m,
        n,
        wrong_axes.shape,  # E: is not assignable
        output.shape,
    )
    wrong_output = torch.empty((m, other_n), device="cuda", dtype=torch.float16)
    require_host_dimensions(
        split,
        m,
        n,
        scratch.shape,
        wrong_output.shape,  # E: is not assignable
    )

    # Tensor's shape generic rejects wrong axes when torch._C is resolvable.
    assert_type(wrong_axes, torch.Tensor[[Split, M, N]])  # E: failed

    # Dtype is not a Tensor type parameter: this incorrect scratch passes.
    wrong_dtype = torch.empty((split, m, n), device="cuda", dtype=torch.float16)
    assert_type(wrong_dtype, torch.Tensor[[Split, M, N]])
    # Initialization likewise does not distinguish a zeroed tensor from an empty one.
    zeroed = torch.zeros((split, m, n), device="cuda", dtype=torch.float32)
    assert_type(zeroed, torch.Tensor[[Split, M, N]])


def test_no_typed_launch_bridge[Split: IntVar, M: IntVar, N: IntVar](
    split: Int[Split], m: Int[M], n: Int[N]
) -> None:
    scratch = torch.empty((split, m, n), device="cuda", dtype=torch.float32)
    # Host Tensor is not a semantic device pointer without a typed conversion.
    require_semantic_pointer(scratch)  # E: is not assignable
    # The overlay retains a Python function, but `jit` does not model grid launch.
    _twopass_compute_kernel[(1, split)]  # E: is not subscriptable
    _twopass_reduce_kernel[(1,)]  # E: is not subscriptable


def test_literal_allocation_axes() -> None:
    scratch = torch.empty((2, 3, 4), device="cuda", dtype=torch.float32)
    assert_type(scratch, torch.Tensor[[3, 2, 4]])  # E: failed

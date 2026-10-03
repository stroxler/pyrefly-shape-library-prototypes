# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only shape adapter; no Blackwell execution or verified contiguous input.
# @lint-ignore-every AUTODEPS2

"""Check the declared Torch-to-Blackwell kernel shape contract."""

from typing import assert_type

import torch
from shape_extensions import Int, IntVar
from tests.test_gluon_tmem_example import tmem_example_kernel
from triton.experimental.gluon import language as gl


def input_pointer[M: IntVar, N: IntVar](
    host: torch.Tensor[[M, N]],
) -> gl.InContiguousMatrixPointer2D[M, N]: ...


def output_pointer[M: IntVar, N: IntVar](
    host: torch.Tensor[[M, N]],
) -> gl.OutContiguousMatrixPointer2D[M, N]: ...


def test_tmem_host_boundary[M: IntVar, N: IntVar, Other: IntVar](
    inp: torch.Tensor[[M, N]],
    m: Int[M],
    n: Int[N],
    other: Int[Other],
) -> None:
    out = torch.empty((m, n), device="cuda", dtype=torch.float32)
    wrong_out = torch.empty((m, other), device="cuda", dtype=torch.float32)
    assert_type(out, torch.Tensor[[M, N]])
    tmem_example_kernel(input_pointer(inp), output_pointer(out), m, n, 4)
    tmem_example_kernel(
        input_pointer(inp),
        output_pointer(wrong_out),  # E: is not assignable
        m,
        n,
        4,
    )

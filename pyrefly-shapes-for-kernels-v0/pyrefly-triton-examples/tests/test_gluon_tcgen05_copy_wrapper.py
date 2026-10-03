# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only original host wrapper; the bracketed launch remains untyped.
# @lint-ignore-every AUTODEPS2

"""Check the tutorial's original host allocation and copy launch boundary."""

from typing import assert_type

import torch
from shape_extensions import Int, IntVar
from tests.test_gluon_tcgen05_copy import tcgen05_copy_kernel
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import TensorMemoryLayout


def tcgen05_copy_example[MRows: IntVar, NCols: IntVar](
    M: Int[MRows],
    N: Int[NCols],
    smem_layout: gl.NVMMASharedLayout,
    tmem_layout: TensorMemoryLayout,
    dtype: torch.dtype,
):
    input = torch.randn(M, N, dtype=dtype, device="cuda")
    output = torch.empty_like(input)
    tcgen05_copy_kernel[(1,)](  # E: is not subscriptable
        input, *input.stride(), output, *output.stride(), M, N, smem_layout, tmem_layout
    )
    # Just check that the input and output are equal.
    torch.testing.assert_close(input, output, atol=0, rtol=0)


def test_wrapper_torch_extent[M: IntVar, N: IntVar](
    m: Int[M],
    n: Int[N],
    layout: gl.NVMMASharedLayout,
    tmem_layout: TensorMemoryLayout,
) -> None:
    inp = torch.randn(m, n, dtype=torch.float32, device="cuda")
    out = torch.empty_like(inp)
    assert_type(inp, torch.Tensor[[M, N]])
    assert_type(out, torch.Tensor[[M, N]])
    tcgen05_copy_example(m, n, layout, tmem_layout, torch.float32)
    tcgen05_copy_example(m, n, layout, tmem_layout, torch.float16)

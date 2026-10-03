# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This is a static-only host contract, not a working Triton launch adapter.
# @lint-ignore-every AUTODEPS2

"""Relate shaped Torch arrays to the already checked vector-add kernel."""

from typing import assert_type

import torch
import triton.language as tl
from shape_extensions import Int, IntVar
from tests.test_vector_add import add_kernel


def input_pointer[N: IntVar](host: torch.Tensor[[N]]) -> tl.InPointer[[N]]: ...


def output_pointer[N: IntVar](host: torch.Tensor[[N]]) -> tl.OutPointer[[N]]: ...


def vector_add[N: IntVar, Block: IntVar](
    x: torch.Tensor[[N]],
    y: torch.Tensor[[N]],
    n: Int[N],
    block: Int[Block],
) -> torch.Tensor[[N]]:
    output = torch.empty((n,), device="cuda", dtype=torch.float32)
    add_kernel(input_pointer(x), input_pointer(y), output_pointer(output), n, block)
    return output


def test_host_input_and_result[N: IntVar, Other: IntVar, Block: IntVar](
    x: torch.Tensor[[N]],
    y: torch.Tensor[[N]],
    other: torch.Tensor[[Other]],
    n: Int[N],
    block: Int[Block],
) -> None:
    assert_type(vector_add(x, y, n, block), torch.Tensor[[N]])
    vector_add(x, other, n, block)  # E: is not assignable


def test_host_allocation_and_pointer_role[N: IntVar, Other: IntVar, Block: IntVar](
    x: torch.Tensor[[N]],
    y: torch.Tensor[[N]],
    n: Int[N],
    wrong_n: Int[Other],
    block: Int[Block],
) -> None:
    output = torch.empty((wrong_n,), device="cuda", dtype=torch.float32)
    assert_type(output, torch.Tensor[[Other]])
    add_kernel(
        input_pointer(x),
        input_pointer(y),
        output_pointer(output),  # E: is not assignable
        n,
        block,
    )
    correctly_sized = torch.empty((n,), device="cuda", dtype=torch.float32)
    add_kernel(
        x,  # E: is not assignable
        input_pointer(y),
        output_pointer(correctly_sized),
        n,
        block,
    )

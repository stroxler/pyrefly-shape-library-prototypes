# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; the annotations are not executable Triton kernel annotations.
# @lint-ignore-every AUTODEPS2

"""Check Blackwell tensor-memory memcpy with the original kernel body."""

from typing import assert_type

from shape_extensions import Int, IntVar
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    allocate_tensor_memory,
    TensorMemoryLayout,
    TensorMemoryTileF32,
)


@gluon.jit
def tmem_example_kernel[M: IntVar, N: IntVar](
    in_ptr: gl.InContiguousMatrixPointer2D[M, N],
    out_ptr: gl.OutContiguousMatrixPointer2D[M, N],
    M: Int[M],
    N: Int[N],
    num_warps: int,
):
    global_memory_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 32], [1, num_warps], [1, 0]
    )

    offs_m = gl.arange(0, M, gl.SliceLayout(1, global_memory_layout))
    offs_n = gl.arange(0, N, gl.SliceLayout(0, global_memory_layout))
    offs = offs_m[:, None] * N + offs_n[None, :]

    input = gl.load(in_ptr + offs)

    # Allocate some tensor memory.
    tmem_layout: gl.constexpr = TensorMemoryLayout(
        block=(64, 64),
        col_stride=32 // in_ptr.dtype.element_ty.primitive_bitwidth,
    )

    tmem = allocate_tensor_memory(
        element_ty=in_ptr.dtype.element_ty,
        shape=[M, N],
        layout=tmem_layout,
    )

    # Get the register layout needed to access the tensor memory from the descriptor.
    tmem_reg_layout: gl.constexpr = tmem.get_reg_layout()

    input = gl.convert_layout(input, tmem_reg_layout)
    tmem.store(input)
    output = tmem.load()
    output = gl.convert_layout(output, global_memory_layout)

    gl.store(out_ptr + offs, output)


def test_tmem_boundary[M: IntVar, N: IntVar, Other: IntVar](
    inp: gl.InContiguousMatrixPointer2D[M, N],
    out: gl.OutContiguousMatrixPointer2D[M, N],
    wrong_input: gl.InContiguousMatrixPointer2D[Other, N],
    wrong_output: gl.OutContiguousMatrixPointer2D[M, Other],
    m: Int[M],
    n: Int[N],
) -> None:
    tmem_example_kernel(inp, out, m, n, 4)
    tmem_example_kernel(
        wrong_input,
        out,  # E: is not assignable
        m,  # E: is not assignable
        n,
        4,
    )
    tmem_example_kernel(inp, wrong_output, m, n, 4)  # E: is not assignable


def test_tmem_transfer_tile[M: IntVar, N: IntVar, Other: IntVar](
    m: Int[M],
    n: Int[N],
    wrong_tile: gl.tensor[[M, Other]],
) -> None:
    tile = allocate_tensor_memory(gl.float32, [m, n], TensorMemoryLayout((64, 64), 1))
    assert_type(tile, TensorMemoryTileF32[M, N])
    tile.store(wrong_tile)  # E: is not assignable


def test_flat_pointer_row_stride[M: IntVar, N: IntVar, Other: IntVar](
    inp: gl.InContiguousMatrixPointer2D[M, N],
    row: gl.RowIndices2D[M],
    col: gl.ColumnIndices2D[N],
    n: Int[N],
    wrong_stride: Int[Other],
) -> None:
    gl.load(inp + (row * n + col))
    inp + (row * wrong_stride + col)  # E: is not supported

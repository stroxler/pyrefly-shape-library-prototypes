"""Relate a tiled Triton launch to its checked Torch output allocation."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import overload

import torch
from shape_extensions import Int, IntTuple, IntVar

from triton_library.host_tensor import Tensor
from triton_library.semantic_jit import SemanticKernel


@dataclass(frozen=True)
class TiledOutputLayout[Shape: IntTuple, Tile: IntTuple]:
    """The intended output coverage, without a proof of kernel PID arithmetic."""

    output_shape: tuple[int, ...]
    tile_shape: tuple[int, ...]
    shape_parameters: tuple[str, ...]
    tile_parameters: tuple[str, ...]
    metadata: tuple[tuple[str, int], ...]

    @property
    def grid(self) -> tuple[int, ...]:
        """Flatten the output tile coordinates into one Triton grid axis."""
        programs = 1
        for size, tile in zip(self.output_shape, self.tile_shape, strict=True):
            programs *= (size + tile - 1) // tile
        return (programs,)

    def launch[**Args, Result](
        self,
        kernel: SemanticKernel[Callable[Args, Result]],
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Result:
        """Check named dimension/tiling arguments and invoke the typed kernel."""
        expected = (
            tuple(zip(self.shape_parameters, self.output_shape, strict=True))
            + tuple(zip(self.tile_parameters, self.tile_shape, strict=True))
            + self.metadata
        )
        return _launch_checked(kernel, self.grid, expected, *args, **kwargs)


@dataclass(frozen=True)
class GridStrideOutputLayout[Shape: IntTuple]:
    """A row-iterating grid, where every program processes rows at a fixed step."""

    output_shape: tuple[int, ...]
    programs: int
    column_block: int
    shape_parameters: tuple[str, ...]
    block_parameter: str
    metadata: tuple[tuple[str, int], ...]

    @property
    def grid(self) -> tuple[int, int, int]:
        """Keep the tutorial's three-axis grid, with only axis zero varying."""
        return (self.programs, 1, 1)

    def launch[**Args, Result](
        self,
        kernel: SemanticKernel[Callable[Args, Result]],
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Result:
        """Check the output shape and metadata before the grid-stride launch."""
        expected = (
            tuple(zip(self.shape_parameters, self.output_shape, strict=True))
            + ((self.block_parameter, self.column_block),)
            + self.metadata
        )
        return _launch_checked(kernel, self.grid, expected, *args, **kwargs)


@dataclass(frozen=True)
class RowOutputLayout[Shape: IntTuple]:
    """One program per output row, with an independent column-loop width."""

    rows: int
    cols: int
    stride: int
    block_width: int
    column_parameter: str
    stride_parameter: str
    block_parameter: str
    metadata: tuple[tuple[str, int], ...] = ()

    @property
    def grid(self) -> tuple[int]:
        """Launch one Triton program per row."""
        return (self.rows,)

    def launch[**Args, Result](
        self,
        kernel: SemanticKernel[Callable[Args, Result]],
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Result:
        """Bind the checked column count, row stride, and loop width."""
        expected = (
            (self.column_parameter, self.cols),
            (self.stride_parameter, self.stride),
            (self.block_parameter, self.block_width),
        ) + self.metadata
        return _launch_checked(kernel, self.grid, expected, *args, **kwargs)


@dataclass(frozen=True)
class AttentionOutputLayout[Shape: IntTuple]:
    """Tiled self-attention over contiguous batch, head, token, feature axes."""

    batch: int
    heads: int
    tokens: int
    dim: int
    block_m: int
    block_n: int

    @property
    def grid(self) -> tuple[int, int]:
        """Assign one query tile per batch/head program."""
        return (self.tokens // self.block_m, self.batch * self.heads)

    def launch[**Args, Result](
        self,
        kernel: SemanticKernel[Callable[Args, Result]],
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Result:
        """Check the launch dimensions before invoking the typed kernel."""
        expected = (
            ("Z", self.batch),
            ("H", self.heads),
            ("N_CTX", self.tokens),
            ("HEAD_DIM", self.dim),
            ("BLOCK_M", self.block_m),
            ("BLOCK_N", self.block_n),
        )
        return _launch_checked(kernel, self.grid, expected, *args, **kwargs)


def attention_output[Batch: IntVar, Heads: IntVar, Tokens: IntVar, Dim: IntVar](
    q: torch.Tensor[[Batch, Heads, Tokens, Dim]],
    k: torch.Tensor[[Batch, Heads, Tokens, Dim]],
    v: torch.Tensor[[Batch, Heads, Tokens, Dim]],
    *,
    block_m: int,
    block_n: int,
) -> AttentionOutputLayout[[Batch, Heads, Tokens, Dim]]:
    """Validate equal contiguous Q/K/V axes for the descriptor-based tutorial."""
    if q.ndim != 4 or any(type(n) is not int or n <= 0 for n in q.shape):
        raise ValueError("Attention expects nonempty [batch, heads, tokens, dim]")
    if any(
        tensor.shape != q.shape
        or tensor.device != q.device
        or tensor.dtype != q.dtype
        or not tensor.is_contiguous()
        for tensor in (q, k, v)
    ):
        raise ValueError(
            "Query, key, and value shapes, dtypes, devices, and layouts must match"
        )
    if not q.is_contiguous() or q.dtype != torch.float16:
        raise ValueError("Descriptor attention currently requires contiguous float16")
    batch, heads, tokens, dim = q.shape
    if dim < 16 or dim & (dim - 1):
        raise ValueError("Head dimension must be a power of two >= 16")
    if (
        any(
            type(block) is not int or block < 16 or block & (block - 1)
            for block in (block_m, block_n)
        )
        or block_n > dim
        or tokens % block_m
        or tokens % block_n
    ):
        raise ValueError(
            "Attention blocks must divide tokens and fit the head dimension"
        )
    return AttentionOutputLayout(batch, heads, tokens, dim, block_m, block_n)


def row_output[Rows: IntVar, Cols: IntVar, StrideRows: IntVar, StrideCols: IntVar](
    output: Tensor[[Rows, Cols], [StrideRows, StrideCols]],
    *,
    column_parameter: str,
    stride_parameter: str,
    block_parameter: str,
    block_width: int,
    metadata: Mapping[str, int] | None = None,
) -> RowOutputLayout[[Rows, Cols]]:
    """Check the host row layout and the legal Triton loop width."""
    if output.ndim != 2 or min(output.shape) <= 0:
        raise ValueError("Expected a nonempty two-dimensional output")
    rows, cols = output.shape
    stride = output.stride(0)
    if output.stride(1) != 1 or (rows > 1 and stride < cols):
        raise ValueError("Output rows must be unit-column-stride and nonoverlapping")
    if (
        type(block_width) is not int
        or block_width <= 0
        or block_width & (block_width - 1)
    ):
        raise ValueError("Block width must be a positive power of two")
    names = {column_parameter, stride_parameter, block_parameter, *(metadata or {})}
    if len(names) != 3 + len(metadata or {}):
        raise ValueError("Launch argument names must be distinct")
    return RowOutputLayout(
        rows,
        cols,
        stride,
        block_width,
        column_parameter,
        stride_parameter,
        block_parameter,
        tuple((metadata or {}).items()),
    )


def _launch_checked[**Args, Result](
    kernel: SemanticKernel[Callable[Args, Result]],
    grid: tuple[int, ...],
    expected: tuple[tuple[str, int], ...],
    *args: Args.args,
    **kwargs: Args.kwargs,
) -> Result:
    source = getattr(kernel, "fn", None)
    if source is None:
        raise TypeError("A checked layout requires a semantic Triton kernel")
    bound = inspect.signature(source).bind(*args, **kwargs)
    for name, value in expected:
        actual = bound.arguments.get(name)
        if type(actual) is not int or actual != value:
            raise ValueError(f"Launch argument {name} must match the output layout")
    return kernel[grid](*args, **kwargs)


@overload
def tiled_output[Length: IntVar, Stride: IntVar, Block: IntVar](
    output: Tensor[[Length], [Stride]],
    blocks: tuple[Int[Block]],
    *,
    shape_parameters: tuple[str],
    tile_parameters: tuple[str],
    metadata: Mapping[str, int] | None = None,
) -> TiledOutputLayout[[Length], [Block]]: ...


@overload
def tiled_output[
    Rows: IntVar,
    Cols: IntVar,
    StrideRows: IntVar,
    StrideCols: IntVar,
    BlockRows: IntVar,
    BlockCols: IntVar,
](
    output: Tensor[[Rows, Cols], [StrideRows, StrideCols]],
    blocks: tuple[Int[BlockRows], Int[BlockCols]],
    *,
    shape_parameters: tuple[str, str],
    tile_parameters: tuple[str, str],
    metadata: Mapping[str, int] | None = None,
) -> TiledOutputLayout[[Rows, Cols], [BlockRows, BlockCols]]: ...


def tiled_output(
    output: Tensor,
    blocks: tuple[int, ...],
    *,
    shape_parameters: tuple[str, ...],
    tile_parameters: tuple[str, ...],
    metadata: Mapping[str, int] | None = None,
) -> TiledOutputLayout:
    """Check output dimensions and declare their correspondence to the launch."""
    if output.ndim not in (1, 2) or len(blocks) != output.ndim:
        raise ValueError("Expected a one- or two-dimensional tiled output")
    shape = tuple(output.shape)
    if any(type(size) is not int or size <= 0 for size in (*shape, *blocks)):
        raise ValueError("Output dimensions and tiles must be positive integers")
    if len(shape_parameters) != output.ndim or len(tile_parameters) != output.ndim:
        raise ValueError("Output layout parameter count must match the output rank")
    extra = tuple((name, value) for name, value in (metadata or {}).items())
    names = (*shape_parameters, *tile_parameters, *(name for name, _ in extra))
    if len(set(names)) != len(names):
        raise ValueError("Launch parameter names must be distinct")
    if any(type(value) is not int or value <= 0 for _, value in extra):
        raise ValueError("Additional launch metadata must be positive integers")
    return TiledOutputLayout(shape, blocks, shape_parameters, tile_parameters, extra)


def grid_stride_output[
    Rows: IntVar,
    Cols: IntVar,
    StrideRows: IntVar,
    StrideCols: IntVar,
](
    output: Tensor[[Rows, Cols], [StrideRows, StrideCols]],
    programs: int,
    *,
    column_block: int,
    block_parameter: str,
    shape_parameters: tuple[str, str],
    metadata: Mapping[str, int] | None = None,
) -> GridStrideOutputLayout[[Rows, Cols]]:
    """Bind row-iterating grid metadata to an actual two-dimensional output."""
    if output.ndim != 2:
        raise ValueError("Expected a two-dimensional grid-stride output")
    shape = tuple(output.shape)
    if any(type(size) is not int or size <= 0 for size in shape):
        raise ValueError("Output dimensions must be positive integers")
    if type(programs) is not int or not 1 <= programs <= shape[0]:
        raise ValueError("Grid-stride programs must be between 1 and the row count")
    if (
        type(column_block) is not int
        or column_block < shape[1]
        or column_block & (column_block - 1)
    ):
        raise ValueError("Column block must be a power of two covering the output")
    extra = tuple((name, value) for name, value in (metadata or {}).items())
    names = (*shape_parameters, block_parameter, *(name for name, _ in extra))
    if len(set(names)) != len(names):
        raise ValueError("Launch parameter names must be distinct")
    if any(type(value) is not int or value <= 0 for _, value in extra):
        raise ValueError("Additional launch metadata must be positive integers")
    return GridStrideOutputLayout(
        shape, programs, column_block, shape_parameters, block_parameter, extra
    )

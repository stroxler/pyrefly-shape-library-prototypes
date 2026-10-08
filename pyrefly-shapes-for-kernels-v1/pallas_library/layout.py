"""Checked Pallas layouts that bind grid axes, block maps, and host shapes."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast, overload

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from shape_extensions import Int, IntVar


class BlockIndex[Extent: IntVar, Block: IntVar](int):
    """The ordinal grid index for a host axis tiled into blocks."""


def grid_axis[Extent: IntVar, Block: IntVar](
    extent: Int[Extent], block: Int[Block]
) -> pl.GridSize[Extent, Block]:
    """Declare that ceiling division represents a grid axis."""
    if type(extent) is not int or extent < 0 or type(block) is not int or block <= 0:
        raise ValueError("Grid extents must be nonnegative and blocks positive")
    return cast(Any, pl.cdiv(extent, block))


@dataclass(frozen=True)
class Layout[**HostArgs, Result]:
    """A host-to-Ref contract checked when constructed and when invoked."""

    kernel: Callable[..., None]
    grid: tuple[int, ...]
    in_specs: tuple[object, ...] | None
    out_spec: object | None
    out_shape: jax.ShapeDtypeStruct | tuple[jax.ShapeDtypeStruct, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[object, ...]


def row_layout[Cols: IntVar](
    kernel: Callable[[pl.InRef[[Cols]], pl.OutRef[[Cols]]], None],
    *,
    out_shape: jax.ShapeDtypeStruct[[Cols]],
    grid: tuple[()],
) -> Layout[[jax.Array[[Cols]]], jax.Array[[Cols]]]:
    """Bind one full input row to one output row without a tiled grid."""
    shape = tuple(out_shape.shape)
    if grid != () or len(shape) != 1 or type(shape[0]) is not int or shape[0] <= 0:
        raise ValueError("Expected a nonempty one-dimensional row and empty grid")
    return Layout(kernel, grid, None, None, out_shape, (shape,), (out_shape.dtype,))


def row_statistics_layout[Features: IntVar](
    kernel: Callable[
        [
            pl.InRef[[Features]],
            pl.InRef[[Features]],
            pl.InRef[[Features]],
            pl.OutRef[[Features]],
            pl.OutRef[[]],
            pl.OutRef[[]],
        ],
        None,
    ],
    *,
    out_shape: tuple[
        jax.ShapeDtypeStruct[[Features]],
        jax.ShapeDtypeStruct[[]],
        jax.ShapeDtypeStruct[[]],
    ],
    grid: tuple[()],
) -> Layout[
    [jax.Array[[Features]], jax.Array[[Features]], jax.Array[[Features]]],
    tuple[jax.Array[[Features]], jax.Array[[]], jax.Array[[]]],
]:
    """Bind three row inputs to a row output and two scalar statistics."""
    (output, mean, rstd) = out_shape
    shape = tuple(output.shape)
    if grid != () or len(shape) != 1 or type(shape[0]) is not int or shape[0] <= 0:
        raise ValueError("Expected a nonempty output row and empty grid")
    if (
        mean.shape != ()
        or rstd.shape != ()
        or any(spec.dtype != output.dtype for spec in (mean, rstd))
    ):
        raise ValueError("Statistics outputs must be scalars of the row dtype")
    return Layout(
        kernel, grid, None, None, out_shape, (shape,) * 3, (output.dtype,) * 3
    )


def attention_layout[
    Batch: IntVar,
    Queries: IntVar,
    Keys: IntVar,
    Heads: IntVar,
    Dim: IntVar,
    QueryBlock: IntVar,
    KeyBlock: IntVar,
](
    kernel: Callable[
        [
            pl.MhaQueryRef[QueryBlock, Dim],
            pl.MhaKvRef[Keys, Dim],
            pl.MhaKvRef[Keys, Dim],
            pl.MhaOutputRef[QueryBlock, Dim],
            pl.OutRef[[QueryBlock]],
        ],
        None,
    ],
    *,
    query_shape: tuple[Int[Batch], Int[Queries], Int[Heads], Int[Dim]],
    key_shape: tuple[Int[Batch], Int[Keys], Int[Heads], Int[Dim]],
    query_block: Int[QueryBlock],
    key_block: Int[KeyBlock],
    out_shape: tuple[
        jax.ShapeDtypeStruct[[Batch, Queries, Heads, Dim]],
        jax.ShapeDtypeStruct[[Batch, Heads, Queries]],
    ],
    grid: tuple[int, Int[Batch], Int[Heads]],
) -> Layout[
    [
        jax.Array[[Batch, Queries, Heads, Dim]],
        jax.Array[[Batch, Keys, Heads, Dim]],
        jax.Array[[Batch, Keys, Heads, Dim]],
    ],
    tuple[jax.Array[[Batch, Queries, Heads, Dim]], jax.Array[[Batch, Heads, Queries]]],
]:
    """Bind squeezed query/KV tiles and a reordered statistics output to host arrays."""
    batches, queries, heads, dim = query_shape
    key_batches, keys, key_heads, key_dim = key_shape
    if any(
        type(n) is not int or n <= 0
        for n in (*query_shape, keys, query_block, key_block)
    ):
        raise ValueError("Attention dimensions and blocks must be positive integers")
    if (key_batches, key_heads, key_dim) != (batches, heads, dim):
        raise ValueError(
            "Query and key/value batch, head, and feature dimensions must agree"
        )
    if queries % query_block or keys % key_block:
        raise ValueError("Query and key lengths must be divisible by their blocks")
    if dim & (dim - 1) or dim < 16:
        raise ValueError(
            "This GPU attention kernel needs a power-of-two head dimension >= 16"
        )
    if tuple(grid) != (queries // query_block, batches, heads):
        raise ValueError("Attention grid must cover query blocks, batches, and heads")
    if tuple(out_shape[0].shape) != query_shape or tuple(out_shape[1].shape) != (
        batches,
        heads,
        queries,
    ):
        raise ValueError("Output and statistics shapes must match the attention axes")
    if out_shape[1].dtype != jnp.float32:
        raise ValueError("Attention statistics require float32 output")
    query_spec = pl.BlockSpec(
        (None, query_block, None, dim), lambda i, j, h: (j, i, h, 0)
    )
    kv_spec = pl.BlockSpec((None, keys, None, dim), lambda _, j, h: (j, 0, h, 0))
    lse_spec = pl.BlockSpec((None, None, query_block), lambda i, j, h: (j, h, i))
    return Layout(
        kernel,
        grid,
        (query_spec, kv_spec, kv_spec),
        (query_spec, lse_spec),
        out_shape,
        (query_shape, key_shape, key_shape),
        (out_shape[0].dtype,) * 3,
    )


@overload
def vector_layout[Length: IntVar, Block: IntVar](
    kernel: Callable[[pl.InRef[[Block]], pl.OutRef[[Block]]], None],
    *,
    out_shape: jax.ShapeDtypeStruct[[Length]],
    grid: tuple[pl.GridSize[Length, Block]],
    block: tuple[Int[Block]],
    index_map: Callable[[BlockIndex[Length, Block]], tuple[BlockIndex[Length, Block]]],
    input_dtypes: tuple[object],
) -> Layout[[jax.Array[[Length]]], jax.Array[[Length]]]: ...


@overload
def vector_layout[Length: IntVar, Block: IntVar](
    kernel: Callable[[pl.InRef[[Block]], pl.InRef[[Block]], pl.OutRef[[Block]]], None],
    *,
    out_shape: jax.ShapeDtypeStruct[[Length]],
    grid: tuple[pl.GridSize[Length, Block]],
    block: tuple[Int[Block]],
    index_map: Callable[[BlockIndex[Length, Block]], tuple[BlockIndex[Length, Block]]],
    input_dtypes: tuple[object, object] | None = None,
) -> Layout[[jax.Array[[Length]], jax.Array[[Length]]], jax.Array[[Length]]]: ...


def vector_layout(
    kernel: Callable[..., None],
    *,
    out_shape: jax.ShapeDtypeStruct,
    grid: tuple[int],
    block: tuple[int],
    index_map: Callable[..., tuple[int, ...]],
    input_dtypes: tuple[object, ...] | None = None,
) -> Layout:
    """Bind one or two equally mapped inputs and their output to one grid axis."""
    (length,) = out_shape.shape
    (block_size,) = block
    if (
        type(length) is not int
        or length < 0
        or type(block_size) is not int
        or block_size <= 0
    ):
        raise ValueError("Vector length must be nonnegative and block_size positive")
    if tuple(grid) != ((length + block_size - 1) // block_size,):
        raise ValueError("Grid size must cover the vector with complete blocks")
    dtypes = (
        input_dtypes if input_dtypes is not None else (out_shape.dtype, out_shape.dtype)
    )
    if (
        len(dtypes) not in (1, 2)
        or len(inspect.signature(kernel).parameters) != len(dtypes) + 1
    ):
        raise ValueError("Input dtypes must match the kernel input arity")
    spec = pl.BlockSpec(block, cast(Any, index_map))
    return Layout(
        kernel=kernel,
        grid=grid,
        in_specs=(spec,) * len(dtypes),
        out_spec=spec,
        out_shape=out_shape,
        input_shapes=((length,),) * len(dtypes),
        input_dtypes=dtypes,
    )


def matmul_layout[
    Rows: IntVar,
    Inner: IntVar,
    Cols: IntVar,
    RowBlock: IntVar,
    ColBlock: IntVar,
](
    kernel: Callable[
        [
            pl.InRef[[RowBlock, Inner]],
            pl.InRef[[Inner, ColBlock]],
            pl.OutRef[[RowBlock, ColBlock]],
        ],
        None,
    ],
    *,
    x_shape: tuple[Int[Rows], Int[Inner]],
    y_shape: tuple[Int[Inner], Int[Cols]],
    out_shape: jax.ShapeDtypeStruct[[Rows, Cols]],
    grid: tuple[pl.GridSize[Rows, RowBlock], pl.GridSize[Cols, ColBlock]],
    x_block: tuple[Int[RowBlock], Int[Inner]],
    y_block: tuple[Int[Inner], Int[ColBlock]],
    out_block: tuple[Int[RowBlock], Int[ColBlock]],
    x_map: Callable[
        [BlockIndex[Rows, RowBlock], BlockIndex[Cols, ColBlock]],
        tuple[BlockIndex[Rows, RowBlock], Literal[0]],
    ],
    y_map: Callable[
        [BlockIndex[Rows, RowBlock], BlockIndex[Cols, ColBlock]],
        tuple[Literal[0], BlockIndex[Cols, ColBlock]],
    ],
    out_map: Callable[
        [BlockIndex[Rows, RowBlock], BlockIndex[Cols, ColBlock]],
        tuple[BlockIndex[Rows, RowBlock], BlockIndex[Cols, ColBlock]],
    ],
) -> Layout[
    [jax.Array[[Rows, Inner]], jax.Array[[Inner, Cols]]], jax.Array[[Rows, Cols]]
]:
    """Check the 2D matmul block correspondence and build its BlockSpecs."""
    rows, inner = x_shape
    y_inner, cols = y_shape
    row_block, col_block = out_block
    if any(
        type(value) is not int or value <= 0 for value in (*x_shape, cols, *out_block)
    ):
        raise ValueError("Matrix and block dimensions must be positive integers")
    if inner != y_inner or tuple(out_shape.shape) != (rows, cols):
        raise ValueError("Input inner dimensions and declared output shape must match")
    if x_block != (row_block, inner) or y_block != (inner, col_block):
        raise ValueError("Input block shapes must match the host and output shapes")
    if rows % row_block or cols % col_block:
        raise ValueError("Matrix dimensions must be divisible by blocks")
    if tuple(grid) != (rows // row_block, cols // col_block):
        raise ValueError("Grid size must match the tiled output shape")
    # BlockIndex is a static refinement of the ordinary indices Pallas supplies.
    return Layout(
        kernel=kernel,
        grid=grid,
        in_specs=(
            pl.BlockSpec(x_block, cast(Any, x_map)),
            pl.BlockSpec(y_block, cast(Any, y_map)),
        ),
        out_spec=pl.BlockSpec(out_block, cast(Any, out_map)),
        out_shape=out_shape,
        input_shapes=(x_shape, y_shape),
        input_dtypes=(out_shape.dtype, out_shape.dtype),
    )


def checked_pallas_call[**HostArgs, Result](
    layout: Layout[HostArgs, Result],
    *,
    interpret: bool,
    debug: bool = False,
    compiler_params: object | None = None,
) -> Callable[HostArgs, Result]:
    """Compile a typed layout and validate the host arrays at invocation."""
    options: dict[str, object] = {
        "out_shape": layout.out_shape,
        "grid": layout.grid,
        "interpret": interpret,
        "debug": debug,
    }
    if layout.in_specs is not None:
        options["in_specs"] = layout.in_specs
        options["out_specs"] = layout.out_spec
    if compiler_params is not None:
        options["compiler_params"] = compiler_params
    raw_call = cast(Any, pl.pallas_call)(layout.kernel, **options)

    def call(*args: HostArgs.args, **kwargs: HostArgs.kwargs) -> Result:
        if kwargs or len(args) != len(layout.input_shapes):
            raise ValueError("Input arguments must match the declared layout")
        device = None
        for array, shape, dtype in zip(
            args, layout.input_shapes, layout.input_dtypes, strict=True
        ):
            if not isinstance(array, jax.Array) or tuple(array.shape) != shape:
                raise ValueError("Input shapes must match the declared layout")
            if array.dtype != dtype:
                raise ValueError("Input dtypes must match the declared layout")
            array_device = getattr(array, "device", None)
            if (
                device is not None
                and array_device is not None
                and array_device != device
            ):
                raise ValueError("Input devices must match")
            if array_device is not None:
                device = array_device
        return cast(Result, raw_call(*args))

    return call

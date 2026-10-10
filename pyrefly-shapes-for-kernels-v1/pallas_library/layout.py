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
    input_shapes: tuple[tuple[int, ...] | None, ...]
    input_dtypes: tuple[object | None, ...]


@dataclass(frozen=True)
class InputBinding:
    """Map a host array's named axes to the Ref visible inside a kernel."""

    host: jax.ShapeDtypeStruct | None
    axes: tuple[str, ...]
    block: tuple[int | None, ...] | None = None
    index_map: Callable[..., tuple[int, ...]] | None = None


@dataclass(frozen=True)
class OutputBinding:
    """Map a kernel output Ref back to a host array's named axes."""

    host: jax.ShapeDtypeStruct
    axes: tuple[str, ...]
    block: tuple[int | None, ...] | None = None
    index_map: Callable[..., tuple[int, ...]] | None = None


@dataclass(frozen=True)
class GridBinding:
    """Tile an output's named host axis along one program-grid axis."""

    output: int
    axis: int
    block: int
    exact: bool = False


def ragged_dot_layout[
    Rows: IntVar,
    Inner: IntVar,
    Cols: IntVar,
    Groups: IntVar,
    RowBlock: IntVar,
    ColBlock: IntVar,
](
    kernel: Callable[
        [
            pl.RaggedLhsRef[Rows, Inner],
            pl.RaggedRhsRef[Inner, ColBlock],
            pl.AxisBoundRef[Rows],
            pl.AxisBoundRef[Rows],
            pl.ValidOutRef[[Rows, ColBlock], [Rows, Cols]],
        ],
        None,
    ],
    *,
    lhs: jax.Array[[Rows, Inner]],
    rhs: jax.Array[[Groups, Inner, Cols]],
    boundaries: jax.RaggedCumulative[Groups],
    block_m: Int[RowBlock],
    block_n: Int[ColBlock],
) -> Layout[
    [
        jax.Array[[Rows, Inner]],
        jax.Array[[Groups, Inner, Cols]],
        jax.Array[[Groups]],
        jax.Array[[Groups]],
    ],
    jax.Array[[Rows, Cols]],
]:
    """Bind a group-indexed RHS and scalar boundaries to a tiled output."""
    rows, inner = lhs.shape
    groups, rhs_inner, cols = rhs.shape
    if (
        rows <= 0
        or inner <= 0
        or groups <= 0
        or cols <= 0
        or rhs_inner != inner
        or boundaries.shape != (groups + 1,)
    ):
        raise ValueError("Ragged dot shapes must agree and be nonempty")
    if block_m <= 0 or block_n <= 0:
        raise ValueError("Ragged dot tiles must be positive")
    shape = jax.ShapeDtypeStruct((rows, cols), lhs.dtype)
    return Layout(
        kernel=kernel,
        grid=((rows + block_m - 1) // block_m, (cols + block_n - 1) // block_n, groups),
        in_specs=(
            pl.no_block_spec,
            cast(Any, pl.BlockSpec)((None, inner, block_n), lambda _, j, e: (e, 0, j)),
            cast(Any, pl.BlockSpec)((None,), lambda _, __, e: (e,)),
            cast(Any, pl.BlockSpec)((None,), lambda _, __, e: (e,)),
        ),
        out_spec=cast(Any, pl.BlockSpec)((rows, block_n), lambda _, j, __: (0, j)),
        out_shape=shape,
        input_shapes=((rows, inner), (groups, inner, cols), (groups,), (groups,)),
        input_dtypes=(lhs.dtype, rhs.dtype, boundaries.dtype, boundaries.dtype),
    )


def binding_layout(
    kernel: Callable[..., None],
    *,
    inputs: tuple[InputBinding, ...],
    outputs: tuple[OutputBinding, ...],
    grid: tuple[GridBinding, ...] = (),
    divisible: tuple[tuple[str, int], ...] = (),
) -> Layout:
    """Assemble and validate runtime metadata for a Pallas kernel.

    The typed layout factories validate the kernel's Ref signature; this
    general assembler cannot yet map an arbitrary parameter tuple to Refs.
    """
    if not outputs or len(inspect.signature(kernel).parameters) != len(inputs) + len(
        outputs
    ):
        raise ValueError("Bindings must match the kernel parameter count")
    dimensions: dict[str, int] = {}
    bindings = (*inputs, *outputs)
    for binding in bindings:
        if binding.host is None:
            if (
                not isinstance(binding, InputBinding)
                or binding.axes
                or binding.block is not None
                or binding.index_map is not None
            ):
                raise ValueError("Only absent inputs may omit their host shape")
            continue
        shape = tuple(binding.host.shape)
        if len(shape) != len(binding.axes) or len(set(binding.axes)) != len(
            binding.axes
        ):
            raise ValueError("Every host axis must have a distinct name")
        for axis, extent in zip(binding.axes, shape, strict=True):
            if type(extent) is not int or extent < 0:
                raise ValueError("Host dimensions must be nonnegative integers")
            if axis in dimensions and dimensions[axis] != extent:
                raise ValueError(f"Host dimension {axis!r} does not match")
            dimensions[axis] = extent
        if (binding.block is None) != (binding.index_map is None):
            raise ValueError("A block shape and index map must be supplied together")
        if binding.block is not None:
            if len(binding.block) != len(shape) or any(
                value is not None and (type(value) is not int or value <= 0)
                for value in binding.block
            ):
                raise ValueError("Block shape must match the host rank")
            if any(
                extent == 0 and block is None
                for extent, block in zip(shape, binding.block, strict=True)
            ):
                raise ValueError("Squeezed block dimensions must be nonempty")
    if any(
        binding.block is None for binding in bindings if binding.host is not None
    ) and any(
        binding.block is not None for binding in bindings if binding.host is not None
    ):
        raise ValueError("All bindings must either have block specs or omit them")
    for axis, block in divisible:
        if axis not in dimensions or type(block) is not int or block <= 0:
            raise ValueError(
                "Divisibility constraint needs a known axis and positive block"
            )
        if dimensions[axis] % block:
            raise ValueError(f"Host dimension {axis!r} must be divisible by {block}")
    grid_sizes = []
    for entry in grid:
        if type(entry.output) is not int or not 0 <= entry.output < len(outputs):
            raise ValueError("Grid output index is invalid")
        shape = outputs[entry.output].host.shape
        if type(entry.axis) is not int or not 0 <= entry.axis < len(shape):
            raise ValueError("Grid output axis is invalid")
        if type(entry.block) is not int or entry.block <= 0:
            raise ValueError("Grid block must be positive")
        extent = shape[entry.axis]
        if entry.exact and extent % entry.block:
            raise ValueError("Grid block must divide the output dimension")
        grid_sizes.append((extent + entry.block - 1) // entry.block)
    specs = tuple(
        (
            cast(Any, pl.BlockSpec)(binding.block, binding.index_map)
            if binding.block is not None
            else None
        )
        for binding in bindings
    )
    out_shapes = tuple(binding.host for binding in outputs)
    return Layout(
        kernel=kernel,
        grid=tuple(grid_sizes),
        in_specs=(
            specs[: len(inputs)] if any(spec is not None for spec in specs) else None
        ),
        out_spec=(
            (specs[-1] if len(outputs) == 1 else specs[-len(outputs) :])
            if any(spec is not None for spec in specs)
            else None
        ),
        out_shape=out_shapes[0] if len(outputs) == 1 else out_shapes,
        input_shapes=tuple(
            tuple(binding.host.shape) if binding.host is not None else None
            for binding in inputs
        ),
        input_dtypes=tuple(
            binding.host.dtype if binding.host is not None else None
            for binding in inputs
        ),
    )


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
    return cast(
        Any,
        binding_layout(
            kernel,
            inputs=(InputBinding(out_shape, ("cols",)),),
            outputs=(OutputBinding(out_shape, ("cols",)),),
        ),
    )


def row_rms_layout[Features: IntVar](
    kernel: Callable[
        [
            pl.InRef[[Features]],
            pl.InRef[[Features]],
            pl.InRef[[Features]],
            pl.OutRef[[Features]],
            pl.OutRef[[]],
        ],
        None,
    ],
    *,
    out_shape: tuple[jax.ShapeDtypeStruct[[Features]], jax.ShapeDtypeStruct[[]]],
) -> Layout[
    [jax.Array[[Features]], jax.Array[[Features]], jax.Array[[Features]]],
    tuple[jax.Array[[Features]], jax.Array[[]]],
]:
    """Bind three full-row inputs to a row output and one scalar statistic."""
    output, rstd = out_shape
    shape = tuple(output.shape)
    if len(shape) != 1 or type(shape[0]) is not int or shape[0] <= 0:
        raise ValueError("Expected a nonempty output row")
    if rstd.shape != () or rstd.dtype != output.dtype:
        raise ValueError("RMS statistic must be a scalar of the row dtype")
    return cast(
        Any,
        binding_layout(
            kernel,
            inputs=(InputBinding(output, ("features",)),) * 3,
            outputs=(
                OutputBinding(output, ("features",)),
                OutputBinding(rstd, ()),
            ),
        ),
    )


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
    return cast(
        Any,
        binding_layout(
            kernel,
            inputs=(InputBinding(output, ("features",)),) * 3,
            outputs=(
                OutputBinding(output, ("features",)),
                OutputBinding(mean, ()),
                OutputBinding(rstd, ()),
            ),
        ),
    )


def row_input_gradient_layout[Features: IntVar](
    kernel: Callable[
        [
            pl.InRef[[Features]],
            pl.InRef[[Features]],
            pl.InRef[[Features]],
            pl.InRef[[Features]],
            pl.InRef[[]],
            pl.InRef[[]],
            pl.OutRef[[Features]],
        ],
        None,
    ],
    *,
    out_shape: jax.ShapeDtypeStruct[[Features]],
) -> Layout[
    [
        jax.Array[[Features]],
        jax.Array[[Features]],
        jax.Array[[Features]],
        jax.Array[[Features]],
        jax.Array[[]],
        jax.Array[[]],
    ],
    jax.Array[[Features]],
]:
    """Bind four full rows and two saved scalar statistics to one gradient row."""
    (features,) = out_shape.shape
    if type(features) is not int or features <= 0:
        raise ValueError("Expected a nonempty output row")
    scalar = jax.ShapeDtypeStruct((), out_shape.dtype)
    return cast(
        Any,
        binding_layout(
            kernel,
            inputs=(InputBinding(out_shape, ("features",)),) * 4
            + (InputBinding(scalar, ()),) * 2,
            outputs=(OutputBinding(out_shape, ("features",)),),
        ),
    )


def layer_norm_weight_gradient_layout[Rows: IntVar, Cols: IntVar](
    kernel: Callable[
        [
            pl.InRef[[Rows, Cols]],
            pl.InRef[[Cols]],
            pl.InRef[[Cols]],
            pl.InRef[[Rows, Cols]],
            pl.InRef[[Rows]],
            pl.InRef[[Rows]],
            pl.OutRef[[Cols]],
            pl.OutRef[[Cols]],
        ],
        None,
    ],
    *,
    input_shape: tuple[Int[Rows], Int[Cols]],
    out_shape: tuple[jax.ShapeDtypeStruct[[Cols]], jax.ShapeDtypeStruct[[Cols]]],
    block_m: int,
    block_n: int,
) -> Layout[
    [
        jax.Array[[Rows, Cols]],
        jax.Array[[Cols]],
        jax.Array[[Cols]],
        jax.Array[[Rows, Cols]],
        jax.Array[[Rows]],
        jax.Array[[Rows]],
    ],
    tuple[jax.Array[[Cols]], jax.Array[[Cols]]],
]:
    """Bind full-matrix Pallas Refs to two column-gradient outputs."""
    rows, cols = input_shape
    if any(type(n) is not int or n <= 0 for n in (rows, cols, block_m, block_n)):
        raise ValueError("Layer-norm dimensions and blocks must be positive")
    if any(tuple(out.shape) != (cols,) for out in out_shape):
        raise ValueError("Both gradients must have the input's column shape")
    matrix = jax.ShapeDtypeStruct(input_shape, out_shape[0].dtype)
    columns = jax.ShapeDtypeStruct((cols,), out_shape[0].dtype)
    row_stats = jax.ShapeDtypeStruct((rows,), out_shape[0].dtype)
    return cast(
        Any,
        binding_layout(
            kernel,
            inputs=(
                InputBinding(matrix, ("rows", "cols")),
                InputBinding(columns, ("cols",)),
                InputBinding(columns, ("cols",)),
                InputBinding(matrix, ("rows", "cols")),
                InputBinding(row_stats, ("rows",)),
                InputBinding(row_stats, ("rows",)),
            ),
            outputs=(
                OutputBinding(out_shape[0], ("cols",)),
                OutputBinding(out_shape[1], ("cols",)),
            ),
            grid=(GridBinding(0, 0, block_n),),
        ),
    )


def attention_preprocess_layout[
    Batch: IntVar,
    Queries: IntVar,
    Heads: IntVar,
    Dim: IntVar,
    Padded: IntVar,
    QueryBlock: IntVar,
](
    kernel: Callable[
        [
            pl.ValidInRef[[QueryBlock, Padded], [QueryBlock, Dim]],
            pl.ValidInRef[[QueryBlock, Padded], [QueryBlock, Dim]],
            pl.OutRef[[QueryBlock]],
        ],
        None,
    ],
    *,
    input_shape: tuple[Int[Batch], Int[Queries], Int[Heads], Int[Dim]],
    padded_dim: Int[Padded],
    query_block: Int[QueryBlock],
    out_shape: jax.ShapeDtypeStruct[[Batch, Heads, Queries]],
) -> Layout[
    [jax.Array[[Batch, Queries, Heads, Dim]], jax.Array[[Batch, Queries, Heads, Dim]]],
    jax.Array[[Batch, Heads, Queries]],
]:
    """Map two logical attention outputs to permuted query-wise scalars."""
    batch, queries, heads, dim = input_shape
    if any(
        type(n) is not int or n <= 0 for n in (*input_shape, padded_dim, query_block)
    ):
        raise ValueError("Attention dimensions and blocks must be positive")
    if padded_dim < dim or padded_dim & (padded_dim - 1):
        raise ValueError("Padded head dimension must cover the input head")
    if queries % query_block:
        raise ValueError("Query block must divide the query length")
    if tuple(out_shape.shape) != (batch, heads, queries):
        raise ValueError("Delta must have permuted [batch, heads, queries] axes")
    input_host = jax.ShapeDtypeStruct(input_shape, out_shape.dtype)
    query_map = lambda i, j, h: (j, i, h, 0)
    delta_map = lambda i, j, h: (j, h, i)
    return cast(
        Any,
        binding_layout(
            kernel,
            inputs=(
                InputBinding(
                    input_host,
                    ("batch", "queries", "heads", "dim"),
                    (None, query_block, None, padded_dim),
                    query_map,
                ),
                InputBinding(
                    input_host,
                    ("batch", "queries", "heads", "dim"),
                    (None, query_block, None, padded_dim),
                    query_map,
                ),
            ),
            outputs=(
                OutputBinding(
                    out_shape,
                    ("batch", "heads", "queries"),
                    (None, None, query_block),
                    delta_map,
                ),
            ),
            grid=(
                GridBinding(0, 2, query_block, exact=True),
                GridBinding(0, 0, 1),
                GridBinding(0, 1, 1),
            ),
        ),
    )


def attention_backward_layout[
    Batch: IntVar,
    Queries: IntVar,
    Keys: IntVar,
    Heads: IntVar,
    Dim: IntVar,
    Padded: IntVar,
    QueryBlockDkv: IntVar,
    KeyBlockDkv: IntVar,
    QueryBlockDq: IntVar,
    KeyBlockDq: IntVar,
](
    kernel: Callable[
        [
            pl.ValidInRef[[Queries, Padded], [Queries, Dim]],
            pl.ValidInRef[[Keys, Padded], [Keys, Dim]],
            pl.ValidInRef[[Keys, Padded], [Keys, Dim]],
            pl.MhaSegmentRef[Keys] | None,
            pl.ValidInRef[[Queries, Padded], [Queries, Dim]],
            pl.ValidInRef[[Queries, Padded], [Queries, Dim]],
            pl.InRef[[Queries]],
            pl.InRef[[Queries]],
            pl.ValidOutRef[[QueryBlockDq, Padded], [QueryBlockDq, Dim]],
            pl.ValidOutRef[[KeyBlockDkv, Padded], [KeyBlockDkv, Dim]],
            pl.ValidOutRef[[KeyBlockDkv, Padded], [KeyBlockDkv, Dim]],
        ],
        None,
    ],
    *,
    query_shape: tuple[Int[Batch], Int[Queries], Int[Heads], Int[Dim]],
    key_shape: tuple[Int[Batch], Int[Keys], Int[Heads], Int[Dim]],
    padded_dim: Int[Padded],
    block_q_dkv: Int[QueryBlockDkv],
    block_kv_dkv: Int[KeyBlockDkv],
    block_q_dq: Int[QueryBlockDq],
    block_kv_dq: Int[KeyBlockDq],
    has_segments: bool,
    out_shape: tuple[
        jax.ShapeDtypeStruct[[Batch, Queries, Heads, Dim]],
        jax.ShapeDtypeStruct[[Batch, Keys, Heads, Dim]],
        jax.ShapeDtypeStruct[[Batch, Keys, Heads, Dim]],
    ],
) -> Layout[
    [
        jax.Array[[Batch, Queries, Heads, Dim]],
        jax.Array[[Batch, Keys, Heads, Dim]],
        jax.Array[[Batch, Keys, Heads, Dim]],
        jax.Array[[Batch, Keys]] | None,
        jax.Array[[Batch, Queries, Heads, Dim]],
        jax.Array[[Batch, Queries, Heads, Dim]],
        jax.Array[[Batch, Heads, Queries]],
        jax.Array[[Batch, Heads, Queries]],
    ],
    tuple[
        jax.Array[[Batch, Queries, Heads, Dim]],
        jax.Array[[Batch, Keys, Heads, Dim]],
        jax.Array[[Batch, Keys, Heads, Dim]],
    ],
]:
    """Bind two attention scans to a shared batch/head/output-block grid."""
    batches, queries, heads, dim = query_shape
    key_batches, keys, key_heads, key_dim = key_shape
    if any(
        type(n) is not int or n <= 0
        for n in (
            *query_shape,
            keys,
            padded_dim,
            block_q_dkv,
            block_kv_dkv,
            block_q_dq,
            block_kv_dq,
        )
    ):
        raise ValueError("Attention backward dimensions and blocks must be positive")
    if (key_batches, key_heads, key_dim) != (batches, heads, dim):
        raise ValueError("Query and key batch, head, and feature axes must match")
    if padded_dim < dim or padded_dim & (padded_dim - 1):
        raise ValueError("Padded head dimension must cover the feature axis")
    if (
        queries % block_q_dkv
        or keys % block_kv_dkv
        or queries % block_q_dq
        or keys % block_kv_dq
    ):
        raise ValueError("Attention scans need whole sequence blocks")
    if queries // block_q_dq != keys // block_kv_dkv:
        raise ValueError("Both backward scans must share the same grid")
    if tuple(out_shape[0].shape) != query_shape or any(
        tuple(out.shape) != key_shape for out in out_shape[1:]
    ):
        raise ValueError("Backward output axes must match the input arrays")
    if any(out.dtype != out_shape[0].dtype for out in out_shape):
        raise ValueError("Attention gradients currently require matching dtypes")
    q_host = jax.ShapeDtypeStruct(query_shape, out_shape[0].dtype)
    k_host = jax.ShapeDtypeStruct(key_shape, out_shape[0].dtype)
    stats = jax.ShapeDtypeStruct((batches, heads, queries), jnp.float32)
    full_q_map = lambda i, j, _step: (i, 0, j, 0)
    full_k_map = lambda i, j, _step: (i, 0, j, 0)
    stats_map = lambda i, j, _step: (i, j, 0)
    output_map = lambda i, j, step: (i, step, j, 0)
    q_axes = ("batch", "queries", "heads", "dim")
    k_axes = ("batch", "keys", "heads", "dim")
    q_input = InputBinding(
        q_host, q_axes, (None, queries, None, padded_dim), full_q_map
    )
    k_input = InputBinding(k_host, k_axes, (None, keys, None, padded_dim), full_k_map)
    stats_input = InputBinding(
        stats, ("batch", "heads", "queries"), (None, None, queries), stats_map
    )
    segment_input = (
        InputBinding(
            jax.ShapeDtypeStruct((batches, keys), jnp.int32),
            ("batch", "keys"),
            (None, keys),
            lambda i, _j, _step: (i, 0),
        )
        if has_segments
        else InputBinding(None, ())
    )
    return cast(
        Any,
        binding_layout(
            kernel,
            inputs=(
                q_input,
                k_input,
                k_input,
                segment_input,
                q_input,
                q_input,
                stats_input,
                stats_input,
            ),
            outputs=(
                OutputBinding(
                    out_shape[0],
                    q_axes,
                    (None, block_q_dq, None, padded_dim),
                    output_map,
                ),
                OutputBinding(
                    out_shape[1],
                    k_axes,
                    (None, block_kv_dkv, None, padded_dim),
                    output_map,
                ),
                OutputBinding(
                    out_shape[2],
                    k_axes,
                    (None, block_kv_dkv, None, padded_dim),
                    output_map,
                ),
            ),
            grid=(
                GridBinding(0, 0, 1),
                GridBinding(0, 2, 1),
                GridBinding(1, 1, block_kv_dkv, exact=True),
            ),
        ),
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
            pl.ValidInRef[[QueryBlock, Dim], [QueryBlock, Dim]],
            pl.InRef[[Keys, Dim]],
            pl.InRef[[Keys, Dim]],
            pl.OutRef[[QueryBlock, Dim]],
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
    query_map = lambda i, j, h: (j, i, h, 0)
    kv_map = lambda _i, j, h: (j, 0, h, 0)
    lse_map = lambda i, j, h: (j, h, i)
    q_host = jax.ShapeDtypeStruct(query_shape, out_shape[0].dtype)
    kv_host = jax.ShapeDtypeStruct(key_shape, out_shape[0].dtype)
    return cast(
        Any,
        binding_layout(
            kernel,
            inputs=(
                InputBinding(
                    q_host,
                    ("batch", "queries", "heads", "dim"),
                    (None, query_block, None, dim),
                    query_map,
                ),
                InputBinding(
                    kv_host,
                    ("batch", "keys", "heads", "dim"),
                    (None, keys, None, dim),
                    kv_map,
                ),
                InputBinding(
                    kv_host,
                    ("batch", "keys", "heads", "dim"),
                    (None, keys, None, dim),
                    kv_map,
                ),
            ),
            outputs=(
                OutputBinding(
                    out_shape[0],
                    ("batch", "queries", "heads", "dim"),
                    (None, query_block, None, dim),
                    query_map,
                ),
                OutputBinding(
                    out_shape[1],
                    ("batch", "heads", "queries"),
                    (None, None, query_block),
                    lse_map,
                ),
            ),
            grid=(
                GridBinding(0, 1, query_block, exact=True),
                GridBinding(0, 0, 1),
                GridBinding(0, 2, 1),
            ),
            divisible=(("keys", key_block),),
        ),
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
    return binding_layout(
        kernel,
        inputs=tuple(
            InputBinding(
                jax.ShapeDtypeStruct((length,), dtype), ("length",), block, index_map
            )
            for dtype in dtypes
        ),
        outputs=(OutputBinding(out_shape, ("length",), block, index_map),),
        grid=(GridBinding(0, 0, block_size),),
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
    return cast(
        Any,
        binding_layout(
            kernel,
            inputs=(
                InputBinding(
                    jax.ShapeDtypeStruct(x_shape, out_shape.dtype),
                    ("rows", "inner"),
                    x_block,
                    x_map,
                ),
                InputBinding(
                    jax.ShapeDtypeStruct(y_shape, out_shape.dtype),
                    ("inner", "cols"),
                    y_block,
                    y_map,
                ),
            ),
            outputs=(OutputBinding(out_shape, ("rows", "cols"), out_block, out_map),),
            grid=(
                GridBinding(0, 0, row_block, exact=True),
                GridBinding(0, 1, col_block, exact=True),
            ),
        ),
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
            if shape is None:
                if array is not None:
                    raise ValueError("Absent input must match the declared layout")
                continue
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

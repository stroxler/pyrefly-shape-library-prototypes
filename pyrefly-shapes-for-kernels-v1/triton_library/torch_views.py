"""Validate Torch allocations before presenting them as Triton pointer views."""

from __future__ import annotations

from typing import cast, get_args, get_origin, overload

import torch
import triton.language as tl
from shape_extensions import Int, IntVar

from triton_library import host_tensor, tlt


def checked_attention_input[Batch: IntVar, Heads: IntVar, Tokens: IntVar, Dim: IntVar](
    tensor: torch.Tensor[[Batch, Heads, Tokens, Dim]],
    shape: tuple[Int[Batch], Int[Heads], Int[Tokens], Int[Dim]],
) -> tl.AttentionPointer[Batch * Heads * Tokens, Dim, Dim]:
    """Present a contiguous four-axis host tensor as a flattened descriptor input."""
    if tensor.ndim != 4 or tuple(tensor.shape) != shape or not tensor.is_contiguous():
        raise ValueError("Attention input must match the contiguous host axes")
    return cast("tl.AttentionPointer[Batch * Heads * Tokens, Dim, Dim]", tensor)


def checked_attention_stats[Batch: IntVar, Heads: IntVar, Tokens: IntVar](
    tensor: torch.Tensor[[Batch, Heads, Tokens]],
    shape: tuple[Int[Batch], Int[Heads], Int[Tokens]],
) -> tl.AttentionStatsPointer[Batch * Heads, Tokens]:
    """Present contiguous per-query log-sum-exp as a flattened stats pointer."""
    if tensor.ndim != 3 or tuple(tensor.shape) != shape or not tensor.is_contiguous():
        raise ValueError("Attention statistics must match contiguous host axes")
    return cast("tl.AttentionStatsPointer[Batch * Heads, Tokens]", tensor)


@overload
def as_host_tensor[Length: IntVar](
    tensor: torch.Tensor[[Length]],
) -> host_tensor.Tensor[[Length], [1]]: ...


@overload
def as_host_tensor[Length: IntVar](
    tensor: torch.Tensor[[Length]],
    view_type: type[host_tensor.Tensor[[Length], [1]]],
) -> host_tensor.Tensor[[Length], [1]]: ...


@overload
def as_host_tensor[Rows: IntVar, Cols: IntVar](
    tensor: torch.Tensor[[Rows, Cols]],
) -> host_tensor.Tensor[[Rows, Cols], [int, 1]]: ...


@overload
def as_host_tensor[Rows: IntVar, Cols: IntVar, RowStride: IntVar, ColStride: IntVar](
    tensor: torch.Tensor[[Rows, Cols]],
    view_type: type[host_tensor.Tensor[[Rows, Cols], [RowStride, ColStride]]],
) -> host_tensor.Tensor[[Rows, Cols], [RowStride, ColStride]]: ...


def as_host_tensor(
    tensor: torch.Tensor, view_type: object | None = None
) -> host_tensor.Tensor:
    """Validate a Torch matrix before it enters a stride-aware host interface."""
    if view_type is None:
        if tensor.ndim == 1:
            view_type = host_tensor.Tensor[[int], [1]]
        else:
            view_type = host_tensor.Tensor[[int, int], [int, 1]]
    _validate_view(tensor, view_type, (host_tensor.Tensor,))
    return cast(host_tensor.Tensor, tensor)


@overload
def checked_vector[Length: IntVar](
    tensor: host_tensor.Tensor[[Length], [1]],
    pointer_type: type[tlt.InPointer[[Length], [1]]],
) -> tuple[tlt.InPointer[[Length], [1]], Int[Length]]: ...


@overload
def checked_vector[Length: IntVar](
    tensor: host_tensor.Tensor[[Length], [1]],
    pointer_type: type[tlt.OutPointer[[Length], [1]]],
) -> tuple[tlt.OutPointer[[Length], [1]], Int[Length]]: ...


def checked_vector(tensor: torch.Tensor, pointer_type: object) -> tuple[object, int]:
    """Present a checked host vector as the same Triton pointer allocation."""
    _validate_view(tensor, pointer_type, (tlt.InPointer, tlt.OutPointer))
    return tensor, tensor.shape[0]


@overload
def checked_matrix[Rows: IntVar, Cols: IntVar, RowStride: IntVar, ColStride: IntVar](
    tensor: host_tensor.Tensor[[Rows, Cols], [RowStride, ColStride]],
    pointer_type: type[tlt.InPointer[[Rows, Cols], [RowStride, ColStride]]],
) -> tuple[
    tlt.InPointer[[Rows, Cols], [RowStride, ColStride]],
    Int[RowStride],
    Int[Rows],
    Int[Cols],
]: ...


@overload
def checked_matrix[Rows: IntVar, Cols: IntVar, RowStride: IntVar, ColStride: IntVar](
    tensor: host_tensor.Tensor[[Rows, Cols], [RowStride, ColStride]],
    pointer_type: type[tlt.OutPointer[[Rows, Cols], [RowStride, ColStride]]],
) -> tuple[
    tlt.OutPointer[[Rows, Cols], [RowStride, ColStride]],
    Int[RowStride],
    Int[Rows],
    Int[Cols],
]: ...


def checked_matrix(
    tensor: torch.Tensor, pointer_type: object
) -> tuple[object, int, int, int]:
    """Present a checked host matrix as the same Triton pointer allocation."""
    _validate_view(
        tensor,
        pointer_type,
        (tlt.InPointer, tlt.OutPointer),
    )
    rows, cols = tensor.shape
    return tensor, tensor.stride(0), rows, cols


def _validate_view(
    tensor: torch.Tensor, view_type: object, kinds: tuple[type, ...]
) -> None:
    """Check the shape and element-stride restrictions of a view marker."""
    kind = get_origin(view_type)
    if kind not in kinds:
        raise ValueError("Unexpected tensor view type")
    shape, strides = get_args(view_type)
    if tensor.ndim != len(shape) or len(strides) != len(shape):
        rank = {1: "one", 2: "two"}.get(len(shape), str(len(shape)))
        raise ValueError(f"Expected a {rank}-dimensional tensor")
    for actual, expected in zip(tensor.shape, shape, strict=True):
        if isinstance(expected, int) and actual != expected:
            raise ValueError(f"Expected dimension {expected}, got {actual}")
    for actual, expected in zip(tensor.stride(), strides, strict=True):
        if isinstance(expected, int) and actual != expected:
            raise ValueError(f"Expected element stride {expected}, got {actual}")
    if kind is tlt.OutPointer and tensor.ndim == 2:
        rows, cols = tensor.shape
        if rows > 1 and tensor.stride(0) < cols:
            raise ValueError("Output rows must not overlap")

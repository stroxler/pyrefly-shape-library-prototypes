# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Only the shape-preserving initialization used by the Pallas example."""

from typing import Literal, overload

from jax import Array, RaggedCumulative
from jax.experimental.pallas import (
    AccumRef,
    ConvSegmentMask,
    IndexMask,
    Indices,
    Mask,
    MhaMask,
    OutRef,
    PagedAttentionMask,
    ScalarFloat,
    Tile,
    TiledLseRows,
    VocabMask,
)
from jax.experimental.pallas.tpu import VmemScratchRef
from shape_extensions import Int, IntTuple, IntVar

float32: object
bool_: object
floating: object
inf: float

def issubdtype(dtype: object, kind: object) -> bool: ...

class Int32Dtype: ...
class BFloat16Dtype: ...

class Float16Dtype:
    itemsize: Literal[2]

int32: Int32Dtype
bfloat16: BFloat16Dtype
float16: Float16Dtype

def empty[Length: IntVar](
    shape: tuple[Int[Length]], *, dtype: object
) -> Array[[Length]]: ...
def dtype(value: Float16Dtype) -> Float16Dtype: ...
def ones[Rows: IntVar, Cols: IntVar](
    shape: tuple[Int[Rows], Int[Cols]], *, dtype: object
) -> Array[[Rows, Cols]]: ...
@overload
def zeros[Length: IntVar](length: Int[Length]) -> Tile[[Length]]: ...
@overload
def zeros[Length: IntVar](length: Int[Length], *, dtype: object) -> Tile[[Length]]: ...
@overload
def zeros[Rows: IntVar, Cols: IntVar](
    shape: tuple[Int[Rows], Int[Cols]], *, dtype: object
) -> Tile[[Rows, Cols]]: ...
@overload
def zeros[Batch: IntVar, Rows: IntVar, Cols: IntVar](
    shape: tuple[Int[Batch], Int[Rows], Int[Cols]], dtype: object
) -> Array[[Batch, Rows, Cols]]: ...
def full[Batch: IntVar, Rows: IntVar](
    shape: tuple[Int[Batch], Int[Rows]], fill_value: int, dtype: object
) -> Array[[Batch, Rows]]: ...
@overload
def concatenate[Batch: IntVar, Pad: IntVar, Block: IntVar, Cols: IntVar](
    arrays: list[Array[[Batch, Pad, Cols]] | Array[[Batch, Block, Cols]]],
    *,
    axis: Literal[1],
) -> Array[[Batch, Pad + Block, Cols]]: ...
@overload
def concatenate[Batch: IntVar, Pad: IntVar, Block: IntVar](
    arrays: list[Array[[Batch, Pad]] | Array[[Batch, Block]]],
    *,
    axis: Literal[1],
) -> Array[[Batch, Pad + Block]]: ...
def sqrt(value: float) -> ScalarFloat: ...
def arange[Block: IntVar](
    stop: Int[Block], *, dtype: object = ...
) -> Indices[Block]: ...
def cumulative_sum[Groups: IntVar](
    values: Array[[Groups]], *, include_initial: Literal[True]
) -> RaggedCumulative[Groups]: ...
def result_type[ShapeA: IntTuple, ShapeB: IntTuple](
    a: Tile[ShapeA], b: Tile[ShapeB]
) -> object: ...
@overload
def max[Block: IntVar](x: Tile[[Block]], axis: int = 0) -> float: ...
@overload
def max[Rows: IntVar, Cols: IntVar](
    x: Tile[[Rows, Cols]], axis: Literal[-1]
) -> Tile[[Rows]]: ...
@overload
def exp[Block: IntVar](x: Tile[[Block]]) -> Tile[[Block]]: ...
@overload
def exp[Rows: IntVar, Cols: IntVar](x: Tile[[Rows, Cols]]) -> Tile[[Rows, Cols]]: ...
def log[Rows: IntVar, Cols: IntVar](x: Tile[[Rows, Cols]]) -> Tile[[Rows, Cols]]: ...
def tanh[Rows: IntVar, Cols: IntVar](x: Tile[[Rows, Cols]]) -> Tile[[Rows, Cols]]: ...
def exp2[Shape: IntTuple](x: Tile[Shape]) -> Tile[Shape]: ...

class FloatInfo:
    min: float

def finfo(dtype: object) -> FloatInfo: ...
def tile[Rows: IntVar](
    x: Tile[[Rows, 128]], repeats: tuple[Literal[1], int]
) -> TiledLseRows[Rows]: ...
def full_like[Rows: IntVar](
    ref: VmemScratchRef[[Rows, 128]], value: float
) -> Tile[[Rows, 128]]: ...
@overload
def sum[Block: IntVar](x: Tile[[Block]], axis: int = 0) -> float: ...
@overload
def sum[Rows: IntVar, Cols: IntVar](
    x: Tile[[Rows, Cols]], axis: Literal[0]
) -> Tile[[Cols]]: ...
@overload
def sum[Rows: IntVar, Cols: IntVar](
    x: Tile[[Rows, Cols]], axis: Literal[1]
) -> Tile[[Rows]]: ...
def dot[Rows: IntVar, Inner: IntVar, Cols: IntVar](
    x: Tile[[Rows, Inner]],
    y: Tile[[Inner, Cols]],
    *,
    preferred_element_type: object,
) -> Tile[[Rows, Cols]]: ...
@overload
def zeros_like[Shape: IntTuple](ref: AccumRef[Shape]) -> Tile[Shape]: ...
@overload
def zeros_like[Shape: IntTuple](ref: VmemScratchRef[Shape]) -> Tile[Shape]: ...
@overload
def zeros_like[Shape: IntTuple](ref: Tile[Shape]) -> Tile[Shape]: ...
@overload
def zeros_like[Shape: IntTuple](ref: OutRef[Shape]) -> Tile[Shape]: ...
@overload
def logical_and[Queries: IntVar, Keys: IntVar](
    x: MhaMask[Queries, Keys], y: MhaMask[Queries, Keys]
) -> MhaMask[Queries, Keys]: ...
@overload
def logical_and(x: bool, y: bool) -> bool: ...
def log2[Length: IntVar](x: Tile[[Length]]) -> Tile[[Length]]: ...
@overload
def maximum(x: int, y: int) -> int: ...
@overload
def maximum[Rows: IntVar](left: Tile[[Rows]], right: Tile[[Rows]]) -> Tile[[Rows]]: ...
@overload
def maximum[Rows: IntVar](
    left: Tile[[Rows, 128]], right: Tile[[Rows, 128]]
) -> Tile[[Rows, 128]]: ...
@overload
def maximum[Rows: IntVar](
    left: Tile[[Rows, 128]], right: Tile[[Rows, 1]]
) -> Tile[[Rows, 128]]: ...
def minimum(x: int, y: int) -> int: ...
@overload
def where[Shape: IntTuple](
    condition: Tile[Shape], true_value: Tile[Shape], false_value: float
) -> Tile[Shape]: ...
@overload
def where[Window: IntVar, Cols: IntVar](
    condition: IndexMask[Window],
    true_value: Tile[[Window, Cols]],
    false_value: Tile[[Window, Cols]],
) -> Tile[[Window, Cols]]: ...
@overload
def where[Queries: IntVar, Keys: IntVar](
    condition: MhaMask[Queries, Keys],
    true_value: Tile[[Queries, Keys]],
    false_value: float,
) -> Tile[[Queries, Keys]]: ...
@overload
def where[Length: IntVar, Block: IntVar](
    condition: Mask[Block, Length],
    true_value: Tile[[Block]],
    false_value: float,
) -> Tile[[Block]]: ...
@overload
def where[Rows: IntVar, Cols: IntVar](
    condition: ConvSegmentMask[Rows],
    true_value: Tile[[Rows, Cols]],
    false_value: Tile[[Rows, Cols]],
) -> Tile[[Rows, Cols]]: ...
@overload
def where[Rows: IntVar, Block: IntVar, Vocab: IntVar](
    condition: VocabMask[Block, Vocab],
    true_value: Tile[[Rows, Block]],
    false_value: float,
) -> Tile[[Rows, Block]]: ...
@overload
def where[Heads: IntVar, Block: IntVar](
    condition: PagedAttentionMask[Heads, Block],
    true_value: Tile[[Heads, Block]],
    false_value: float,
) -> Tile[[Heads, Block]]: ...

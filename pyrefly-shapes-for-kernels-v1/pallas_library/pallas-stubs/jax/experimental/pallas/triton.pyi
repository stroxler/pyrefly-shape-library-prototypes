# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal, overload

from jax.experimental.pallas import (
    ContractionHorizontalMask,
    ContractionVerticalMask,
    DecodeOutputSlice,
    DecodeQuerySlice,
    DecodeResidualSlice,
    IndexedInRef,
    IndexedOutRef,
    LayerNormMatrixMask,
    LayerNormMatrixSlice,
    LayerNormOutSlice,
    LayerNormVectorSlice,
    Mask,
    MhaBackwardMatrixSlice,
    MhaBackwardOutputSlice,
    MhaKvSlice,
    MhaOutputSlice,
    MhaPreprocessRef,
    MhaQueryRef,
    RaggedLhsSlice,
    RaggedOutSlice,
    RaggedRhsSlice,
    RaggedStoreMask,
    RaggedStoreRowMask,
    Tile,
)
from shape_extensions import IntVar

class CompilerParams:
    def __init__(self, *, num_warps: int, num_stages: int) -> None: ...

def dot[Rows: IntVar, Inner: IntVar, Cols: IntVar](
    lhs: Tile[[Rows, Inner]], rhs: Tile[[Inner, Cols]]
) -> Tile[[Rows, Cols]]: ...
@overload
def load[Heads: IntVar, Dim: IntVar](
    ref: DecodeQuerySlice[Heads, Dim], *, mask: ContractionVerticalMask[Heads, int]
) -> Tile[[Heads, Dim]]: ...
@overload
def store[Heads: IntVar, Dim: IntVar](
    ref: DecodeOutputSlice[Heads, Dim],
    value: Tile[[Heads, Dim]],
    *,
    mask: ContractionVerticalMask[Heads, int],
) -> None: ...
@overload
def store[Heads: IntVar](
    ref: DecodeResidualSlice[Heads],
    value: Tile[[Heads]],
    *,
    mask: Mask[Heads, int] | None,
) -> None: ...
@overload
def load[Queries: IntVar, Dim: IntVar](
    ref: MhaQueryRef[Queries, Dim],
    *,
    mask: ContractionHorizontalMask[Dim, Dim],
    other: float,
) -> Tile[[Queries, Dim]]: ...
@overload
def load[Queries: IntVar, PaddedDim: IntVar, HeadDim: IntVar](
    ref: MhaPreprocessRef[Queries, PaddedDim, HeadDim],
    *,
    mask: ContractionHorizontalMask[PaddedDim, HeadDim],
    other: float,
) -> Tile[[Queries, PaddedDim]]: ...
@overload
def load[Block: IntVar, PaddedDim: IntVar, HeadDim: IntVar](
    ref: MhaBackwardMatrixSlice[Block, PaddedDim, HeadDim],
    *,
    mask: ContractionHorizontalMask[PaddedDim, HeadDim],
    other: float,
) -> Tile[[Block, PaddedDim]]: ...
@overload
def store[Block: IntVar, PaddedDim: IntVar, HeadDim: IntVar](
    ref: MhaBackwardOutputSlice[Block, PaddedDim],
    value: Tile[[Block, PaddedDim]],
    *,
    mask: ContractionHorizontalMask[PaddedDim, HeadDim],
) -> None: ...
@overload
def load[Keys: IntVar, Dim: IntVar](
    ref: MhaKvSlice[Keys, Dim],
    *,
    mask: ContractionHorizontalMask[Dim, Dim],
    other: float = 0.0,
) -> Tile[[Keys, Dim]]: ...
@overload
def store[Queries: IntVar, Dim: IntVar](
    ref: MhaOutputSlice[Queries, Dim],
    value: Tile[[Queries, Dim]],
    *,
    mask: ContractionHorizontalMask[Dim, Dim],
) -> None: ...
@overload
def load[Length: IntVar, Block: IntVar](
    ref: IndexedInRef[Length, Block],
    *,
    mask: Mask[Block, Length],
    other: float = 0.0,
    eviction_policy: Literal["evict_last", "evict_first"] | None = None,
) -> Tile[[Block]]: ...
@overload
def store[Length: IntVar, Block: IntVar](
    ref: IndexedOutRef[Length, Block],
    value: Tile[[Block]],
    *,
    mask: Mask[Block, Length],
) -> None: ...
@overload
def load[RowBlock: IntVar, InnerBlock: IntVar, Inner: IntVar](
    ref: RaggedLhsSlice[RowBlock, InnerBlock, Inner],
    *,
    mask: ContractionHorizontalMask[InnerBlock, Inner],
    other: float,
) -> Tile[[RowBlock, InnerBlock]]: ...
@overload
def load[InnerBlock: IntVar, ColBlock: IntVar, Inner: IntVar](
    ref: RaggedRhsSlice[InnerBlock, ColBlock, Inner],
    *,
    mask: ContractionVerticalMask[InnerBlock, Inner],
    other: float,
) -> Tile[[InnerBlock, ColBlock]]: ...
@overload
def load[RowBlock: IntVar, InnerBlock: IntVar, Inner: IntVar](
    ref: RaggedLhsSlice[RowBlock, InnerBlock, Inner],
) -> Tile[[RowBlock, InnerBlock]]: ...
@overload
def load[InnerBlock: IntVar, ColBlock: IntVar, Inner: IntVar](
    ref: RaggedRhsSlice[InnerBlock, ColBlock, Inner],
) -> Tile[[InnerBlock, ColBlock]]: ...
@overload
def store[RowBlock: IntVar, Rows: IntVar, Cols: IntVar, ColBlock: IntVar](
    ref: RaggedOutSlice[RowBlock, Rows, Cols, ColBlock],
    value: Tile[[RowBlock, ColBlock]],
    *,
    mask: RaggedStoreRowMask[RowBlock, Rows]
    | RaggedStoreMask[RowBlock, Rows, ColBlock, Cols],
) -> None: ...
@overload
def load[Rows: IntVar, Cols: IntVar, RowBlock: IntVar, ColBlock: IntVar](
    ref: LayerNormMatrixSlice[Rows, Cols, RowBlock, ColBlock],
    *,
    mask: LayerNormMatrixMask[RowBlock, Rows, ColBlock, Cols],
    other: float,
) -> Tile[[RowBlock, ColBlock]]: ...
@overload
def load[Length: IntVar, Block: IntVar](
    ref: LayerNormVectorSlice[Length, Block],
    *,
    mask: Mask[Block, Length],
    other: float,
) -> Tile[[Block]]: ...
@overload
def store[Length: IntVar, Block: IntVar](
    ref: LayerNormOutSlice[Length, Block],
    value: Tile[[Block]],
    *,
    mask: Mask[Block, Length],
) -> None: ...

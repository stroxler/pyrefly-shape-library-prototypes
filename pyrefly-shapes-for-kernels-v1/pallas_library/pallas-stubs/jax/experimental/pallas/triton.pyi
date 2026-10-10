# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal, overload

from jax.experimental.pallas import (
    Mask,
    RaggedLhsSlice,
    RaggedRhsSlice,
    RectOutputBlock,
    Tile,
    TransformedRef,
    ValidInRef,
)
from shape_extensions import IntVar

class CompilerParams:
    def __init__(self, *, num_warps: int, num_stages: int) -> None: ...

def dot[Rows: IntVar, Inner: IntVar, Cols: IntVar](
    lhs: Tile[[Rows, Inner]], rhs: Tile[[Inner, Cols]]
) -> Tile[[Rows, Cols]]: ...
@overload
def load[Rows: IntVar, Block: IntVar, Dim: IntVar](
    ref: TransformedRef[[Rows, Dim], [Block, Dim], Literal["in"]],
    *,
    mask: Mask[[Block, 1], [int, 1]],
) -> Tile[[Block, Dim]]: ...
@overload
def store[Rows: IntVar, Block: IntVar, Dim: IntVar](
    ref: TransformedRef[[Rows, Dim], [Block, Dim], Literal["out"]],
    value: Tile[[Block, Dim]],
    *,
    mask: Mask[[Block, 1], [int, 1]],
) -> None: ...
@overload
def store[Length: IntVar, Block: IntVar](
    ref: TransformedRef[[Length], [Block], Literal["out"]],
    value: Tile[[Block]],
    *,
    mask: Mask[[Block], [int]] | None,
) -> None: ...
@overload
def load[Queries: IntVar, PaddedDim: IntVar, HeadDim: IntVar](
    ref: ValidInRef[[Queries, PaddedDim], [Queries, HeadDim]],
    *,
    mask: Mask[[1, PaddedDim], [1, HeadDim]],
    other: float,
) -> Tile[[Queries, PaddedDim]]: ...
@overload
def load[Rows: IntVar, Block: IntVar, PaddedDim: IntVar, HeadDim: IntVar](
    ref: TransformedRef[
        [Rows, PaddedDim], [Block, PaddedDim], Literal["in"], [Rows, HeadDim]
    ],
    *,
    mask: Mask[[1, PaddedDim], [1, HeadDim]],
    other: float,
) -> Tile[[Block, PaddedDim]]: ...
@overload
def store[Block: IntVar, PaddedDim: IntVar, HeadDim: IntVar](
    ref: TransformedRef[
        [Block, PaddedDim], [Block, PaddedDim], Literal["out"], [Block, HeadDim]
    ],
    value: Tile[[Block, PaddedDim]],
    *,
    mask: Mask[[1, PaddedDim], [1, HeadDim]],
) -> None: ...
@overload
def load[Rows: IntVar, Block: IntVar, Dim: IntVar](
    ref: TransformedRef[[Rows, Dim], [Block, Dim], Literal["in"]],
    *,
    mask: Mask[[1, Dim], [1, Dim]],
    other: float = 0.0,
) -> Tile[[Block, Dim]]: ...
@overload
def store[Rows: IntVar, Dim: IntVar](
    ref: TransformedRef[[Rows, Dim], [Rows, Dim], Literal["out"]],
    value: Tile[[Rows, Dim]],
    *,
    mask: Mask[[1, Dim], [1, Dim]],
) -> None: ...
@overload
def load[Length: IntVar, Block: IntVar](
    ref: TransformedRef[[Length], [Block], Literal["in"]],
    *,
    mask: Mask[[Block], [Length]],
    other: float = 0.0,
    eviction_policy: Literal["evict_last", "evict_first"] | None = None,
) -> Tile[[Block]]: ...
@overload
def store[Length: IntVar, Block: IntVar](
    ref: TransformedRef[[Length], [Block], Literal["out"]],
    value: Tile[[Block]],
    *,
    mask: Mask[[Block], [Length]],
) -> None: ...
@overload
def load[RowBlock: IntVar, InnerBlock: IntVar, Inner: IntVar](
    ref: RaggedLhsSlice[RowBlock, InnerBlock, Inner],
    *,
    mask: Mask[[1, InnerBlock], [1, Inner]],
    other: float,
) -> Tile[[RowBlock, InnerBlock]]: ...
@overload
def load[InnerBlock: IntVar, ColBlock: IntVar, Inner: IntVar](
    ref: RaggedRhsSlice[InnerBlock, ColBlock, Inner],
    *,
    mask: Mask[[InnerBlock, 1], [Inner, 1]],
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
    ref: RectOutputBlock[RowBlock, Rows, Cols, ColBlock],
    value: Tile[[RowBlock, ColBlock]],
    *,
    mask: Mask[[RowBlock, 1], [Rows, 1]] | Mask[[RowBlock, ColBlock], [Rows, Cols]],
) -> None: ...
@overload
def load[Rows: IntVar, Cols: IntVar, RowBlock: IntVar, ColBlock: IntVar](
    ref: TransformedRef[[Rows, Cols], [RowBlock, ColBlock], Literal["in"]],
    *,
    mask: Mask[[RowBlock, ColBlock], [Rows, Cols]],
    other: float,
) -> Tile[[RowBlock, ColBlock]]: ...

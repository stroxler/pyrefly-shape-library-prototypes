# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from types import EllipsisType
from typing import Literal, Protocol, overload

from jax import Array, Int32Array, ShapeDtypeStruct
from jax.experimental.pallas import mosaic_gpu as gpu
from jax.experimental.pallas import tpu, tpu_sc, triton
from shape_extensions import Int, IntTuple, IntVar

class Tile[Shape: IntTuple]:
    @property
    def T[Rows: IntVar, Cols: IntVar](
        self: Tile[[Rows, Cols]],
    ) -> Tile[[Cols, Rows]]: ...
    @overload
    def __getitem__[Cols: IntVar](
        self: Tile[[Cols]], key: tuple[None, slice]
    ) -> Tile[[1, Cols]]: ...
    @overload
    def __getitem__[Rows: IntVar](
        self: Tile[[Rows]], key: tuple[slice, None]
    ) -> Tile[[Rows, 1]]: ...
    @property
    def shape[Rows: IntVar, Cols: IntVar](
        self: Tile[[Rows, Cols]],
    ) -> tuple[Int[Rows], Int[Cols]]: ...
    @overload
    def __add__(self, other: Tile[Shape]) -> Tile[Shape]: ...
    @overload
    def __add__(self, other: int) -> Tile[Shape]: ...
    @overload
    def __add__(self, other: float) -> Tile[Shape]: ...
    @overload
    def __add__[Heads: IntVar, Keys: IntVar](
        self: Tile[[Heads, Keys]], other: DecodeLogitPenalty[Keys]
    ) -> Tile[[Heads, Keys]]: ...
    @overload
    def __sub__(self, other: float) -> Tile[Shape]: ...
    @overload
    def __sub__(self, other: Tile[Shape]) -> Tile[Shape]: ...
    @overload
    def __sub__[Rows: IntVar, Cols: IntVar](
        self: Tile[[Rows, Cols]], other: Tile[[Rows, 1]]
    ) -> Tile[[Rows, Cols]]: ...
    @overload
    def __sub__[Rows: IntVar, Cols: IntVar](
        self: Tile[[Rows, Cols]], other: TiledLseRows[Rows]
    ) -> Tile[[Rows, Cols]]: ...
    @overload
    def __mul__[Rows: IntVar, Cols: IntVar](
        self: Tile[[1, Cols]], other: Tile[[Rows, Cols]]
    ) -> Tile[[Rows, Cols]]: ...
    @overload
    def __mul__[Rows: IntVar, Cols: IntVar](
        self: Tile[[Rows, Cols]], other: Tile[[Rows, 1]]
    ) -> Tile[[Rows, Cols]]: ...
    @overload
    def __mul__[Rows: IntVar, Cols: IntVar](
        self: Tile[[Rows, Cols]], other: Tile[[1, Cols]]
    ) -> Tile[[Rows, Cols]]: ...
    @overload
    def __mul__[Rows: IntVar, Cols: IntVar](
        self: Tile[[Rows, 1]], other: Tile[[Rows, Cols]]
    ) -> Tile[[Rows, Cols]]: ...
    @overload
    def __mul__(self, other: Tile[Shape]) -> Tile[Shape]: ...
    @overload
    def __mul__(self, other: ScalarFloat) -> Tile[Shape]: ...
    @overload
    def __mul__(self, other: float) -> Tile[Shape]: ...
    @overload
    def __truediv__[Rows: IntVar, Cols: IntVar](
        self: Tile[[Rows, Cols]], other: Tile[[Rows, 1]]
    ) -> Tile[[Rows, Cols]]: ...
    @overload
    def __truediv__(self, other: float) -> Tile[Shape]: ...
    def astype(self, dtype: object) -> Tile[Shape]: ...
    @property
    def dtype(self) -> object: ...
    @overload
    def sum(self) -> ScalarFloat: ...
    @overload
    def sum[Rows: IntVar, Cols: IntVar](
        self: Tile[[Rows, Cols]], *, axis: Literal[-1]
    ) -> Tile[[Rows]]: ...
    @overload
    def sum[Rows: IntVar, Cols: IntVar](
        self: Tile[[Rows, Cols]], *, axis: Literal[0]
    ) -> Tile[[Cols]]: ...
    def max[Rows: IntVar, Cols: IntVar](
        self: Tile[[Rows, Cols]], *, axis: Literal[-1]
    ) -> Tile[[Rows]]: ...
    def __matmul__[Rows: IntVar, Inner: IntVar, Cols: IntVar](
        self: Tile[[Rows, Inner]], other: Tile[[Inner, Cols]]
    ) -> Tile[[Rows, Cols]]: ...

class PagedPageIds[PagesPerBlock: IntVar]: ...
class PagedAttentionMask[Heads: IntVar, Tokens: IntVar]: ...
class MhaMask[Queries: IntVar, Keys: IntVar]: ...

class DecodeKeyIndicesMask[Keys: IntVar]:
    def __and__(self, other: Mask[Keys, int]) -> DecodeKeyIndicesMask[Keys]: ...
    def __getitem__(self, key: tuple[None, slice]) -> DecodeKeyMask[Keys]: ...

class DecodeKeyMask[Keys: IntVar]:
    def __invert__(self) -> DecodeKeyMask[Keys]: ...
    def __mul__(self, scalar: float) -> DecodeLogitPenalty[Keys]: ...

class DecodeLogitPenalty[Keys: IntVar]: ...

class DecodeBoundRef:
    def __getitem__(self, key: tuple[()]) -> int: ...

class DecodeQuerySlice[Heads: IntVar, Dim: IntVar]: ...

class DecodeQueryAt[Heads: IntVar, Dim: IntVar]:
    def __getitem__(
        self, key: tuple[HalfRowSlice[Heads], slice]
    ) -> DecodeQuerySlice[Heads, Dim]: ...

class DecodeQueryRef[Heads: IntVar, Dim: IntVar]:
    @property
    def shape(self) -> tuple[Int[Heads], Int[Dim]]: ...
    @property
    def at(self) -> DecodeQueryAt[Heads, Dim]: ...

class DecodeKvRef[SplitKeys: IntVar, Dim: IntVar]:
    @property
    def shape(self) -> tuple[Int[SplitKeys], Int[Dim]]: ...
    def __getitem__[Block: IntVar](
        self, key: tuple[HalfRowSlice[Block], slice]
    ) -> Tile[[Block, Dim]]: ...

class DecodeOutputSlice[Heads: IntVar, Dim: IntVar]: ...

class DecodeOutputAt[Heads: IntVar, Dim: IntVar]:
    def __getitem__(
        self, key: tuple[HalfRowSlice[Heads], slice]
    ) -> DecodeOutputSlice[Heads, Dim]: ...

class DecodeOutputRef[Heads: IntVar, Dim: IntVar]:
    @property
    def dtype(self) -> object: ...
    @property
    def at(self) -> DecodeOutputAt[Heads, Dim]: ...

class DecodeResidualSlice[Heads: IntVar]: ...

class DecodeResidualAt[Heads: IntVar]:
    def __getitem__(self, key: HalfRowSlice[Heads]) -> DecodeResidualSlice[Heads]: ...

class DecodeResidualRef[Heads: IntVar]:
    @property
    def at(self) -> DecodeResidualAt[Heads]: ...

class MhaSegmentRef[Sequence: IntVar]:
    def __getitem__[Block: IntVar](
        self, index: HalfRowSlice[Block]
    ) -> Array[[Block]]: ...

class MhaPreprocessRef[Queries: IntVar, PaddedDim: IntVar, HeadDim: IntVar]:
    @property
    def shape(self) -> tuple[Int[Queries], Int[PaddedDim]]: ...

class MhaBackwardMatrixSlice[Block: IntVar, PaddedDim: IntVar, HeadDim: IntVar]: ...

class MhaBackwardMatrixAt[Sequence: IntVar, PaddedDim: IntVar, HeadDim: IntVar]:
    def __getitem__[Block: IntVar](
        self, key: tuple[HalfRowSlice[Block], slice]
    ) -> MhaBackwardMatrixSlice[Block, PaddedDim, HeadDim]: ...

class MhaBackwardMatrixRef[Sequence: IntVar, PaddedDim: IntVar, HeadDim: IntVar]:
    @property
    def shape(self) -> tuple[Int[Sequence], Int[PaddedDim]]: ...
    @property
    def dtype(self) -> object: ...
    @property
    def at(self) -> MhaBackwardMatrixAt[Sequence, PaddedDim, HeadDim]: ...

class MhaBackwardVectorRef[Sequence: IntVar]:
    def __getitem__[Block: IntVar](
        self, index: HalfRowSlice[Block]
    ) -> Tile[[Block]]: ...

class MhaBackwardOutputSlice[Block: IntVar, PaddedDim: IntVar]: ...

class MhaBackwardOutputAt[Block: IntVar, PaddedDim: IntVar]:
    def __getitem__(
        self, index: tuple[slice, slice[None, Int[PaddedDim], None]]
    ) -> MhaBackwardOutputSlice[Block, PaddedDim]: ...

class MhaBackwardOutputRef[Block: IntVar, PaddedDim: IntVar]:
    @property
    def dtype(self) -> object: ...
    @property
    def at(self) -> MhaBackwardOutputAt[Block, PaddedDim]: ...

class MhaKvSlice[KeyBlock: IntVar, Dim: IntVar]: ...

class MhaKvAt[Sequence: IntVar, Dim: IntVar]:
    def __getitem__[KeyBlock: IntVar](
        self, key: tuple[HalfRowSlice[KeyBlock], slice]
    ) -> MhaKvSlice[KeyBlock, Dim]: ...

class MhaKvRef[Sequence: IntVar, Dim: IntVar]:
    @property
    def shape(self) -> tuple[Int[Sequence], Int[Dim]]: ...
    @property
    def at(self) -> MhaKvAt[Sequence, Dim]: ...

class MhaQueryRef[Queries: IntVar, Dim: IntVar]:
    @property
    def shape(self) -> tuple[Int[Queries], Int[Dim]]: ...
    @property
    def dtype(self) -> object: ...

class MhaOutputSlice[Queries: IntVar, Dim: IntVar]: ...

class MhaOutputAt[Queries: IntVar, Dim: IntVar]:
    def __getitem__(
        self, key: tuple[slice, slice[None, Int[Dim], None]]
    ) -> MhaOutputSlice[Queries, Dim]: ...

class MhaOutputRef[Queries: IntVar, Dim: IntVar]:
    @property
    def dtype(self) -> object: ...
    @property
    def at(self) -> MhaOutputAt[Queries, Dim]: ...

class PagedBlockTableRef[TablePages: IntVar]:
    @property
    def shape(self) -> tuple[Int[TablePages]]: ...
    def __getitem__[PagesPerBlock: IntVar](
        self, section: HalfRowSlice[PagesPerBlock]
    ) -> PagedPageIds[PagesPerBlock]: ...

class PagedPageTiles[PagesPerBlock: IntVar, PageSize: IntVar, Dim: IntVar]:
    def reshape(
        self, rows: Int[PagesPerBlock * PageSize], cols: Int[Dim]
    ) -> Tile[[PagesPerBlock * PageSize, Dim]]: ...

class PagedPoolRef[TotalPages: IntVar, PageSize: IntVar, Dim: IntVar]:
    @property
    def shape(self) -> tuple[Int[TotalPages], Int[PageSize], Int[Dim]]: ...
    def __getitem__[PagesPerBlock: IntVar](
        self, ids: PagedPageIds[PagesPerBlock]
    ) -> PagedPageTiles[PagesPerBlock, PageSize, Dim]: ...

class PagedScaleTiles[PagesPerBlock: IntVar, PageSize: IntVar]:
    def reshape(
        self, shape: tuple[Literal[1], Int[PagesPerBlock * PageSize]]
    ) -> Tile[[1, PagesPerBlock * PageSize]]: ...

class PagedScaleRef[TotalPages: IntVar, PageSize: IntVar]:
    def __getitem__[PagesPerBlock: IntVar](
        self, ids: PagedPageIds[PagesPerBlock]
    ) -> PagedScaleTiles[PagesPerBlock, PageSize]: ...

class PagedQueryRef[Heads: IntVar, Dim: IntVar]:
    @property
    def shape(self) -> tuple[Int[Heads], Int[Dim]]: ...
    def __getitem__(
        self, key: tuple[HalfRowSlice[Heads], slice]
    ) -> Tile[[Heads, Dim]]: ...

class PagedLengthRef:
    def __getitem__(self, index: Literal[0]) -> int: ...

class PagedOutputRef[Heads: IntVar, Dim: IntVar]:
    @property
    def dtype(self) -> object: ...
    def __setitem__(self, key: EllipsisType, value: Tile[[Heads, Dim]]) -> None: ...

class PagedResidualRef[Heads: IntVar]:
    def __setitem__(self, key: EllipsisType, value: Tile[[Heads]]) -> None: ...

class TiledLseRows[Rows: IntVar]: ...

class VocabColumnIndices[Block: IntVar]:
    def __lt__[Vocab: IntVar](self, bound: Int[Vocab]) -> VocabMask[Block, Vocab]: ...

class VocabMask[Block: IntVar, Vocab: IntVar]: ...

class LayerNormRowIndices[Block: IntVar]:
    def __ge__[KeyBlock: IntVar](
        self, other: VocabColumnIndices[KeyBlock]
    ) -> MhaMask[Block, KeyBlock]: ...

class LayerNormColumnIndices[Block: IntVar]: ...
class LayerNormMatrixMask[
    RowBlock: IntVar,
    Rows: IntVar,
    ColBlock: IntVar,
    Cols: IntVar,
]: ...
class LayerNormMatrixSlice[
    Rows: IntVar,
    Cols: IntVar,
    RowBlock: IntVar,
    ColBlock: IntVar,
]: ...

class LayerNormMatrixAt[Rows: IntVar, Cols: IntVar]:
    def __getitem__[RowBlock: IntVar, ColBlock: IntVar](
        self,
        key: tuple[LayerNormRowIndices[RowBlock], LayerNormColumnIndices[ColBlock]],
    ) -> LayerNormMatrixSlice[Rows, Cols, RowBlock, ColBlock]: ...

class LayerNormMatrixRef[Rows: IntVar, Cols: IntVar]:
    @property
    def shape(self) -> tuple[Int[Rows], Int[Cols]]: ...
    @property
    def at(self) -> LayerNormMatrixAt[Rows, Cols]: ...

class LayerNormVectorSlice[Length: IntVar, Block: IntVar]: ...

class LayerNormVectorAt[Length: IntVar]:
    def __getitem__[Block: IntVar](
        self, key: Indices[Block]
    ) -> LayerNormVectorSlice[Length, Block]: ...

class LayerNormVectorRef[Length: IntVar]:
    @property
    def at(self) -> LayerNormVectorAt[Length]: ...

class LayerNormOutSlice[Length: IntVar, Block: IntVar]: ...

class LayerNormOutAt[Length: IntVar]:
    def __getitem__[Block: IntVar](
        self, key: Indices[Block]
    ) -> LayerNormOutSlice[Length, Block]: ...

class LayerNormOutRef[Length: IntVar]:
    @property
    def at(self) -> LayerNormOutAt[Length]: ...
    @property
    def dtype(self) -> object: ...

class LseInputRef[BatchBlock: IntVar, Hidden: IntVar]:
    def __getitem__(self, key: EllipsisType) -> Tile[[BatchBlock, Hidden]]: ...

class LseWeightChunk[Hidden: IntVar, ComputeBlock: IntVar]:
    def __getitem__(self, key: EllipsisType) -> Tile[[Hidden, ComputeBlock]]: ...

class LseWeightRef[Hidden: IntVar, VocabBlock: IntVar]:
    @property
    def shape(self) -> tuple[Int[Hidden], Int[VocabBlock]]: ...
    @property
    def dtype(self) -> object: ...
    def __getitem__[ComputeBlock: IntVar](
        self, key: tuple[slice, HalfRowSlice[ComputeBlock]]
    ) -> LseWeightChunk[Hidden, ComputeBlock]: ...
    def __setitem__[Remain: IntVar](
        self, key: tuple[slice, slice[int, None, None]], value: Tile[[Hidden, Remain]]
    ) -> None: ...

class LseOutputRef[BatchBlock: IntVar]:
    @property
    def dtype(self) -> object: ...
    def __setitem__(
        self, key: EllipsisType, value: Tile[[BatchBlock, 128]]
    ) -> None: ...

class RaggedBound[Rows: IntVar](int):
    def __add__(self, other: int) -> RaggedBound[Rows]: ...

class RaggedBoundRef[Rows: IntVar]:
    def __getitem__(self, key: tuple[()]) -> RaggedBound[Rows]: ...

class ContractionHorizontalMask[Block: IntVar, Inner: IntVar]: ...

class ContractionVerticalMask[Block: IntVar, Inner: IntVar]:
    def reshape(self, size: Literal[-1]) -> Mask[Block, Inner]: ...
    def __and__[ColBlock: IntVar, Cols: IntVar](
        self, other: ContractionHorizontalMask[ColBlock, Cols]
    ) -> LayerNormMatrixMask[Block, Inner, ColBlock, Cols]: ...

class RaggedStoreMask[
    RowBlock: IntVar,
    Rows: IntVar,
    ColBlock: IntVar,
    Cols: IntVar,
]: ...

class RaggedRowMask[Block: IntVar, Rows: IntVar]:
    def __getitem__(
        self, key: tuple[slice, None]
    ) -> RaggedStoreRowMask[Block, Rows]: ...

class RaggedStoreRowMask[Block: IntVar, Rows: IntVar]:
    def __iand__[ColsBlock: IntVar, Cols: IntVar](
        self, other: ContractionHorizontalMask[ColsBlock, Cols]
    ) -> RaggedStoreMask[Block, Rows, ColsBlock, Cols]: ...

class RaggedLhsAt[Rows: IntVar, Inner: IntVar]:
    def __getitem__[RowBlock: IntVar, InnerBlock: IntVar](
        self, key: tuple[HalfRowSlice[RowBlock], HalfRowSlice[InnerBlock]]
    ) -> RaggedLhsSlice[RowBlock, InnerBlock, Inner]: ...

class RaggedLhsSlice[RowBlock: IntVar, InnerBlock: IntVar, Inner: IntVar]: ...

class RaggedLhsRef[Rows: IntVar, Inner: IntVar]:
    @property
    def shape(self) -> tuple[Int[Rows], Int[Inner]]: ...
    @property
    def at(self) -> RaggedLhsAt[Rows, Inner]: ...

class RaggedRhsAt[Inner: IntVar, ColBlock: IntVar]:
    def __getitem__[InnerBlock: IntVar](
        self, key: tuple[HalfRowSlice[InnerBlock], HalfRowSlice[ColBlock]]
    ) -> RaggedRhsSlice[InnerBlock, ColBlock, Inner]: ...

class RaggedRhsSlice[InnerBlock: IntVar, ColBlock: IntVar, Inner: IntVar]: ...

class RaggedRhsRef[Inner: IntVar, ColBlock: IntVar]:
    @property
    def shape(self) -> tuple[Int[Inner], Int[ColBlock]]: ...
    @property
    def at(self) -> RaggedRhsAt[Inner, ColBlock]: ...

class RaggedOutAt[Rows: IntVar, Cols: IntVar, ColBlock: IntVar]:
    def __getitem__[RowBlock: IntVar](
        self, key: tuple[HalfRowSlice[RowBlock], HalfRowSlice[ColBlock]]
    ) -> RaggedOutSlice[RowBlock, Rows, Cols, ColBlock]: ...

class RaggedOutSlice[
    RowBlock: IntVar,
    Rows: IntVar,
    Cols: IntVar,
    ColBlock: IntVar,
]: ...

class RaggedOutRef[Rows: IntVar, Cols: IntVar, ColBlock: IntVar]:
    @property
    def shape(self) -> tuple[Int[Rows], Int[ColBlock]]: ...
    @property
    def at(self) -> RaggedOutAt[Rows, Cols, ColBlock]: ...
    @property
    def dtype(self) -> object: ...

class ScalarFloat(float):
    def astype(self, dtype: object) -> Tile[[]]: ...
    def __rtruediv__(self, other: int) -> ScalarFloat: ...
    def __truediv__(self, other: int) -> ScalarFloat: ...

class ConvValuesRef[Rows: IntVar, Cols: IntVar]:
    @property
    def shape(self) -> tuple[Literal[1], Int[Rows], Int[Cols]]: ...
    def __getitem__[Block: IntVar](
        self, key: tuple[Literal[0], HalfRowSlice[Block], slice]
    ) -> Tile[[Block, Cols]]: ...

class ConvSegments[Rows: IntVar]:
    def __eq__(self, other: ConvSegments[Rows]) -> ConvSegmentEquality[Rows]: ...

class ConvSegmentEquality[Rows: IntVar]:
    def __getitem__(self, key: tuple[slice, None]) -> ConvSegmentMask[Rows]: ...

class ConvSegmentMask[Rows: IntVar]: ...

class ConvSegmentRef[Rows: IntVar]:
    def __getitem__[Block: IntVar](
        self, key: tuple[Literal[0], HalfRowSlice[Block]]
    ) -> ConvSegments[Block]: ...

class ConvWeightRow[Cols: IntVar]:
    def __getitem__(self, key: tuple[None, slice]) -> Tile[[1, Cols]]: ...

class ConvWeightRef[Width: IntVar, Cols: IntVar]:
    def __getitem__(self, key: int) -> ConvWeightRow[Cols]: ...

class ConvOutRef[Block: IntVar, Cols: IntVar]:
    @property
    def shape(self) -> tuple[Literal[1], Int[Block], Int[Cols]]: ...
    def __setitem__(self, key: Literal[0], value: Tile[[Block, Cols]]) -> None: ...

class ConvPartialWeightRef[Width: IntVar, Cols: IntVar]:
    def __setitem__(
        self,
        key: tuple[Literal[0], HalfRowSlice[1], slice],
        value: Tile[[1, Cols]],
    ) -> None: ...

class InRef[Shape: IntTuple]:
    @overload
    def __getitem__(self: InRef[[]], key: EllipsisType) -> ScalarFloat: ...
    @overload
    def __getitem__(self, key: slice | EllipsisType) -> Tile[Shape]: ...
    @overload
    def __getitem__[Rows: IntVar, Cols: IntVar](
        self: InRef[[Rows, Cols]], key: tuple[slice, slice]
    ) -> Tile[[Rows, Cols]]: ...
    @overload
    def __getitem__[Rows: IntVar, Cols: IntVar](
        self: InRef[[1, Rows, Cols]], key: tuple[Literal[0], slice, slice]
    ) -> Tile[[Rows, Cols]]: ...
    @overload
    def __getitem__[Rows: IntVar, Cols: IntVar](
        self: InRef[[1, Rows, Cols]], key: tuple[Literal[0], EllipsisType]
    ) -> Tile[[Rows, Cols]]: ...
    @property
    def shape[Length: IntVar](self: InRef[[Length]]) -> tuple[Int[Length]]: ...
    @property
    def dtype(self) -> object: ...
    @overload
    @property
    def at[Length: IntVar](self: InRef[[Length]]) -> InRefAt[Length]: ...
    @overload
    @property
    def at[Rows: IntVar, Cols: IntVar](
        self: InRef[[Rows, Cols]],
    ) -> GatherSourceAt[Rows, Cols]: ...

class GatherSourceAt[Rows: IntVar, Cols: IntVar]:
    def __getitem__[Window: IntVar](
        self, indices: Indices[Window]
    ) -> IndirectGather[Rows, Cols, Window]: ...

class IndirectGather[Rows: IntVar, Cols: IntVar, Window: IntVar]: ...

class ScatterTargetAt[Rows: IntVar, Cols: IntVar]:
    def __getitem__[Window: IntVar](
        self, indices: Indices[Window]
    ) -> IndirectScatter[Rows, Cols, Window]: ...

class IndirectScatter[Rows: IntVar, Cols: IntVar, Window: IntVar]: ...

class GatherIndicesRef[Window: IntVar]:
    @property
    def at(self) -> GatherIndexRow[Window]: ...

class GatherIndexRow[Window: IntVar]:
    def __getitem__(self, index: Literal[0]) -> Indices[Window]: ...

class InRefAt[Length: IntVar]:
    @overload
    def __getitem__(
        self,
        key: slice[None, Int[Length // 2], None] | slice[Int[Length // 2], None, None],
    ) -> InRef[[Length // 2]]: ...
    @overload
    def __getitem__[Block: IntVar](
        self, key: Indices[Block]
    ) -> IndexedInRef[Length, Block]: ...

class CoreIndex(int):
    def __mul__(self, other: int) -> int: ...

class CoreInputAt[Core: IntVar, Lanes: IntVar]:
    def __getitem__(self, key: CoreIndex) -> InRef[[Lanes]]: ...

class CoreOutputAt[Core: IntVar, Lanes: IntVar]:
    def __getitem__(self, key: CoreIndex) -> OutRef[[Lanes]]: ...

class CoreInRef[Core: IntVar, Lanes: IntVar]:
    @property
    def at(self) -> CoreInputAt[Core, Lanes]: ...

class CoreOutRef[Core: IntVar, Lanes: IntVar]:
    @property
    def at(self) -> CoreOutputAt[Core, Lanes]: ...

class UnconstrainedInRef[Shape: IntTuple]:
    def __getitem__(self, key: EllipsisType) -> Tile[Shape]: ...
    @property
    def shape[Rows: IntVar, Cols: IntVar](
        self: UnconstrainedInRef[[Rows, Cols]],
    ) -> tuple[Int[Rows], Int[Cols]]: ...
    @property
    def at[Rows: IntVar, Cols: IntVar](
        self: UnconstrainedInRef[[Rows, Cols]],
    ) -> UnconstrainedInRefAt[Rows, Cols]: ...

class VmemInRef[Shape: IntTuple](UnconstrainedInRef[Shape]):
    @overload
    def __getitem__[Devices: IntVar, Rows: IntVar, Cols: IntVar](
        self: VmemInRef[[Devices, Rows, Cols]],
        key: tuple[int, HalfRowSlice[Rows // 2]],
    ) -> Tile[[Rows // 2, Cols]]: ...
    @property
    def at[Devices: IntVar, Rows: IntVar, Cols: IntVar](
        self: VmemInRef[[Devices, Rows, Cols]],
    ) -> VmemShardedInRefAt[Rows, Cols]: ...

class DynamicSlice: ...
class HalfRowSlice[Size: IntVar](DynamicSlice): ...
class AlignedInt[Alignment: IntVar](int): ...
class AlignedSlice[Size: IntVar](HalfRowSlice[Size]): ...

class VmemShardedInRefAt[Rows: IntVar, Cols: IntVar]:
    def __getitem__(
        self, key: tuple[int, HalfRowSlice[Rows // 2]]
    ) -> VmemInRef[[Rows // 2, Cols]]: ...

class UnconstrainedInRefAt[Rows: IntVar, Cols: IntVar]:
    def __getitem__(
        self, key: slice[Literal[0], Literal[1], None]
    ) -> UnconstrainedSlice[Cols]: ...

class UnconstrainedSlice[Cols: IntVar]: ...

class BoundedSlice[Max: IntVar]:
    def __init__(self, max_size: Int[Max]) -> None: ...

class BoundedTile[Max: IntVar, Cols: IntVar]: ...

class BoundedInRef[Max: IntVar, Cols: IntVar]:
    @overload
    def __getitem__(self, key: EllipsisType) -> BoundedTile[Max, Cols]: ...
    @property
    def shape(self) -> tuple[Int[Max], Int[Cols]]: ...
    @overload
    def __getitem__(
        self, key: tuple[HalfRowSlice[4], HalfRowSlice[16]]
    ) -> Tile[[4, 16]]: ...

class BoundedOutRef[Max: IntVar, Cols: IntVar]:
    @overload
    def __setitem__(self, key: EllipsisType, value: BoundedTile[Max, Cols]) -> None: ...
    @overload
    def __setitem__(
        self, key: tuple[HalfRowSlice[4], HalfRowSlice[16]], value: Tile[[4, 16]]
    ) -> None: ...

@overload
def ds(start: AlignedInt[8], size: Literal[8]) -> AlignedSlice[8]: ...
@overload
def ds[Size: IntVar](start: int, size: Int[Size]) -> HalfRowSlice[Size]: ...
@overload
def ds(start: int, size: int) -> DynamicSlice: ...
def dslice[Size: IntVar](start: int, size: Int[Size]) -> HalfRowSlice[Size]: ...
def multiple_of(value: int, alignment: Literal[8]) -> AlignedInt[8]: ...

class Indices[Block: IntVar]:
    def __radd__(self, other: int) -> Indices[Block]: ...
    def __add__(self, other: int) -> Indices[Block]: ...
    @overload
    def __getitem__(self, key: tuple[None, slice]) -> VocabColumnIndices[Block]: ...
    @overload
    def __getitem__(self, key: None) -> LayerNormColumnIndices[Block]: ...
    @overload
    def __getitem__(self, key: tuple[slice, None]) -> LayerNormRowIndices[Block]: ...
    @overload
    def __lt__[Rows: IntVar](
        self, limit: RaggedBound[Rows]
    ) -> RaggedRowMask[Block, Rows]: ...
    @overload
    def __lt__[Length: IntVar](self, limit: Int[Length]) -> Mask[Block, Length]: ...
    @overload
    def __lt__(self, limit: int) -> DecodeQueryMask[Block]: ...
    def __ge__(self, limit: int) -> DecodeKeyIndicesMask[Block]: ...
    def __mod__(self, divisor: Literal[2]) -> IndexParity[Block]: ...

class IndexParity[Block: IntVar]:
    def __getitem__(self, key: tuple[slice, None]) -> ParityColumn[Block]: ...

class ParityColumn[Block: IntVar]:
    def __eq__(self, other: Literal[1]) -> IndexMask[Block]: ...

class IndexMask[Block: IntVar]: ...

class Mask[Block: IntVar, Length: IntVar]:
    @overload
    def __getitem__(
        self, key: tuple[None, slice]
    ) -> ContractionHorizontalMask[Block, Length]: ...
    @overload
    def __getitem__(
        self, key: tuple[slice, None]
    ) -> ContractionVerticalMask[Block, Length]: ...

class IndexedInRef[Length: IntVar, Block: IntVar]: ...
class IndexedOutRef[Length: IntVar, Block: IntVar]: ...

class PrefetchRef[Shape: IntTuple]:
    @overload
    def __getitem__(self, index: int) -> int: ...
    @overload
    def __getitem__(self, index: tuple[int, int]) -> int: ...

class OutRef[Shape: IntTuple]:
    @property
    def shape[Length: IntVar](self: OutRef[[Length]]) -> tuple[Int[Length]]: ...
    @overload
    def __setitem__(self, key: slice | EllipsisType, value: Tile[Shape]) -> None: ...
    @overload
    def __setitem__[Rows: IntVar, Cols: IntVar](
        self: OutRef[[Rows, Cols]],
        key: tuple[slice, slice],
        value: Tile[[Rows, Cols]],
    ) -> None: ...
    @overload
    def __setitem__[Rows: IntVar, Cols: IntVar](
        self: OutRef[[Rows, Cols]],
        key: tuple[HalfRowSlice[Rows // 2], EllipsisType],
        value: Tile[[Rows // 2, Cols]],
    ) -> None: ...
    @overload
    def __setitem__[Length: IntVar](
        self: OutRef[[Length]], key: ProgramId[Literal[0]], value: int
    ) -> None: ...
    @overload
    def __setitem__[Length: IntVar](
        self: OutRef[[Length]],
        key: slice[None, Int[Length // 2], None] | slice[Int[Length // 2], None, None],
        value: Tile[[Length // 2]],
    ) -> None: ...
    @overload
    @property
    def at[Length: IntVar](self: OutRef[[Length]]) -> OutRefAt[Length]: ...
    @overload
    @property
    def at[Devices: IntVar, Rows: IntVar, Cols: IntVar](
        self: OutRef[[Devices, Rows, Cols]],
    ) -> OutRefShards[Devices, Rows, Cols]: ...
    @overload
    @property
    def at[Rows: IntVar, Cols: IntVar](
        self: OutRef[[Rows, Cols]],
    ) -> ScatterTargetAt[Rows, Cols]: ...
    @property
    def dtype(self) -> object: ...

class ShapeOnlyTile[Shape: IntTuple]:
    @property
    def shape[Rows: IntVar, Cols: IntVar](
        self: ShapeOnlyTile[[Rows, Cols]],
    ) -> tuple[Int[Rows], Int[Cols]]: ...

class ShapeReadableOutRef[Shape: IntTuple](OutRef[Shape]):
    def __getitem__(self, key: EllipsisType) -> ShapeOnlyTile[Shape]: ...

# This Ref role permits read-modify-write, but does not prove prior initialization.
class ScatterOutRef[Shape: IntTuple](OutRef[Shape]):
    def __getitem__(self, key: EllipsisType) -> Tile[Shape]: ...

# Pipelined accumulation reads a previously written output SRAM tile. The
# type cannot prove that an earlier grid iteration initialized this buffer.
class AccumRef[Shape: IntTuple]:
    def __getitem__(self, key: EllipsisType) -> Tile[Shape]: ...
    def __setitem__(self, key: EllipsisType, value: Tile[Shape]) -> None: ...

def semaphore_signal(
    sem: tpu.RegularSemaphore | tpu.BarrierSemaphore,
    *,
    inc: int,
    device_id: tuple[int],
    device_id_type: DeviceIdType,
) -> None: ...
def semaphore_wait(
    sem: tpu.RegularSemaphore | tpu.BarrierSemaphore, value: int
) -> None: ...
def run_scoped(
    body: Callable[[tpu.RegularSemaphore], None],
    *,
    second_barrier: tpu.RegularSemaphoreType,
) -> None: ...

class OutRefAt[Length: IntVar]:
    @overload
    def __getitem__(
        self,
        key: slice[None, Int[Length // 2], None] | slice[Int[Length // 2], None, None],
    ) -> OutRef[[Length // 2]]: ...
    @overload
    def __getitem__[Block: IntVar](
        self, key: Indices[Block]
    ) -> IndexedOutRef[Length, Block]: ...

class OutRefShards[Devices: IntVar, Rows: IntVar, Cols: IntVar]:
    @overload
    def __getitem__(self, index: int) -> OutRef[[Rows, Cols]]: ...
    @overload
    def __getitem__(
        self, key: tuple[int, HalfRowSlice[Rows // 2]]
    ) -> OutRef[[Rows // 2, Cols]]: ...

class ProgramId[Axis: int](int): ...

@overload
def program_id(axis: Literal[0]) -> ProgramId[Literal[0]]: ...
@overload
def program_id(axis: Literal[1]) -> ProgramId[Literal[1]]: ...
@overload
def program_id(axis: Literal[2]) -> ProgramId[Literal[2]]: ...
@overload
def program_id(axis: Literal[3]) -> ProgramId[Literal[3]]: ...
@overload
def program_id(axis: int) -> int: ...
def num_programs(axis: Literal[0, 1, 2]) -> int: ...

class GridSize[Length: IntVar, Block: IntVar](int): ...
class AnyMemorySpace: ...
class NoBlockSpec: ...

class DeviceIdType:
    MESH: DeviceIdType

ANY: AnyMemorySpace
no_block_spec: NoBlockSpec

def next_power_of_2(value: int) -> int: ...
@overload
def cdiv[Length: IntVar, Block: IntVar](
    length: Int[Length], block: Int[Block]
) -> GridSize[Length, Block]: ...
@overload
def cdiv(length: int, block: int) -> int: ...
def dot[Rows: IntVar, Inner: IntVar, Cols: IntVar](
    lhs: Tile[[Rows, Inner]], rhs: Tile[[Inner, Cols]]
) -> Tile[[Rows, Cols]]: ...

class BlockSpec[Shape: IntTuple, Layout: bool | str = Literal[False]]:
    @overload
    def __init__[Keys: IntVar, Dim: IntVar](
        self: BlockSpec[[Keys, Dim], Literal["decode_kv"]],
        block_shape: tuple[None, Int[Keys], Int[Dim]],
        index_map: Callable[[int, int], tuple[int, int, int]],
    ) -> None: ...
    @overload
    def __init__[Heads: IntVar, Dim: IntVar](
        self: BlockSpec[[Heads, Dim], Literal["decode_output"]],
        block_shape: tuple[None, Int[Heads], Int[Dim]],
        index_map: Callable[[int, int], tuple[int, int, int]],
    ) -> None: ...
    @overload
    def __init__(
        self: BlockSpec[[], Literal["decode_bound"]],
        block_shape: tuple[()],
        index_map: Callable[[int, int], tuple[()]],
    ) -> None: ...
    @overload
    def __init__[Queries: IntVar, Dim: IntVar](
        self: BlockSpec[[Queries, Dim], Literal["mha_query"]],
        block_shape: tuple[None, Int[Queries], None, Int[Dim]],
        index_map: Callable[[int, int, int], tuple[int, int, int, int]],
    ) -> None: ...
    @overload
    def __init__[Keys: IntVar, Dim: IntVar](
        self: BlockSpec[[Keys, Dim], Literal["mha_kv"]],
        block_shape: tuple[None, Int[Keys], None, Int[Dim]],
        index_map: Callable[[int, int, int], tuple[int, int, int, int]],
    ) -> None: ...
    @overload
    def __init__[Sequence: IntVar, Dim: IntVar](
        self: BlockSpec[[Sequence, Dim], Literal["mha_backward_input"]],
        block_shape: tuple[None, Int[Sequence], None, Int[Dim]],
        index_map: Callable[[int, int, int], tuple[int, int, int, int]],
    ) -> None: ...
    @overload
    def __init__[Block: IntVar, Dim: IntVar](
        self: BlockSpec[[Block, Dim], Literal["mha_backward_output"]],
        block_shape: tuple[None, Int[Block], None, Int[Dim]],
        index_map: Callable[[int, int, int], tuple[int, int, int, int]],
    ) -> None: ...
    @overload
    def __init__[Queries: IntVar](
        self: BlockSpec[[Queries], Literal["mha_lse"]],
        block_shape: tuple[None, None, Int[Queries]],
        index_map: Callable[[int, int, int], tuple[int, int, int]],
    ) -> None: ...
    @overload
    def __init__[Sequence: IntVar](
        self: BlockSpec[[Sequence], Literal["mha_segment"]],
        block_shape: tuple[None, Int[Sequence]],
        index_map: Callable[[int, int, int], tuple[int, int]],
    ) -> None: ...
    @overload
    def __init__[Rows: IntVar, Cols: IntVar](
        self: BlockSpec[[Rows, Cols], Literal["tma_gmem"]],
        *,
        memory_space: type[gpu.GMEM],
    ) -> None: ...
    @overload
    def __init__[Rows: IntVar](
        self: BlockSpec[[Rows], Literal["tma_smem_indices"]],
        *,
        memory_space: type[gpu.SMEM],
    ) -> None: ...
    @overload
    def __init__[Rows: IntVar, Cols: IntVar](
        self: BlockSpec[[Rows, Cols], Literal["lse_vmem"]],
        block_shape: tuple[Int[Rows], Int[Cols]],
        index_map: Callable[[int, int, int, int], tuple[int, int]],
        *,
        memory_space: type[tpu.VMEM[[Rows, Cols]]],
    ) -> None: ...
    @overload
    def __init__[Inner: IntVar, ColBlock: IntVar](
        self: BlockSpec[[Inner, ColBlock], Literal["ragged_group"]],
        block_shape: tuple[None, Int[Inner], Int[ColBlock]],
        index_map: Callable[[int, int, int], tuple[int, int, int]],
    ) -> None: ...
    @overload
    def __init__(
        self: BlockSpec[[], Literal["ragged_bound"]],
        block_shape: tuple[None],
        index_map: Callable[[int, int, int], tuple[int]],
    ) -> None: ...
    @overload
    def __init__(
        self: BlockSpec[[], Literal["key_smem"]],
        *,
        memory_space: type[tpu.SMEM],
    ) -> None: ...
    @overload
    def __init__[Rows: IntVar, Cols: IntVar](
        self: BlockSpec[[Rows, Cols], Literal["vmem"]],
        *,
        memory_space: type[tpu.VMEM[[Rows, Cols]]],
    ) -> None: ...
    @overload
    def __init__[Devices: IntVar, Rows: IntVar, Cols: IntVar](
        self: BlockSpec[[Devices, Rows, Cols], Literal["vmem"]],
        *,
        memory_space: type[tpu.VMEM[[Rows, Cols]]],
    ) -> None: ...
    @overload
    def __init__[Rows: IntVar, Cols: IntVar](
        self: BlockSpec[[Rows, Cols], Literal["unconstrained"]],
        *,
        memory_space: AnyMemorySpace,
    ) -> None: ...
    @overload
    def __init__[Devices: IntVar, Rows: IntVar, Cols: IntVar](
        self: BlockSpec[[Devices, Rows, Cols], Literal["unconstrained"]],
        *,
        memory_space: AnyMemorySpace,
    ) -> None: ...
    @overload
    def __init__[Max: IntVar, Cols: IntVar](
        self: BlockSpec[[Max, Cols], Literal["dynamic"]],
        block_shape: tuple[BoundedSlice[Max], Int[Cols]],
        index_map: Callable[[int], tuple[DynamicSlice, int]],
    ) -> None: ...
    @overload
    def __init__[Max: IntVar, Cols: IntVar](
        self: BlockSpec[[Max, Cols], Literal["sc_bounded"]],
        block_shape: tuple[BoundedSlice[Max], Int[Cols]],
        index_map: Callable[[int, int], tuple[AlignedSlice[8], int]],
    ) -> None: ...
    @overload
    def __init__[Block: IntVar](
        self: BlockSpec[[Block]],
        block_shape: tuple[Int[Block]],
        index_map: Callable[[int], tuple[int]],
    ) -> None: ...
    @overload
    def __init__[Rows: IntVar, Cols: IntVar](
        self: BlockSpec[[Rows, Cols]],
        block_shape: tuple[Int[Rows], Int[Cols]],
        index_map: Callable[[int], tuple[int, int]],
    ) -> None: ...
    @overload
    def __init__[Rows: IntVar, Cols: IntVar](
        self: BlockSpec[[Rows, Cols]],
        block_shape: tuple[Int[Rows], Int[Cols]],
        index_map: Callable[[int, int], tuple[int, int]],
    ) -> None: ...
    @overload
    def __init__[Rows: IntVar, Inner: IntVar](
        self: BlockSpec[[1, Rows, Inner]],
        block_shape: tuple[Literal[1], Int[Rows], Int[Inner]],
        index_map: Callable[
            [int, int, PrefetchRef[[int]], PrefetchRef[[int]]], tuple[int, int, int]
        ],
    ) -> None: ...
    @overload
    def __init__[Rows: IntVar, Cols: IntVar](
        self: BlockSpec[[Rows, Cols]],
        block_shape: tuple[Int[Rows], Int[Cols]],
        index_map: Callable[
            [int, int, PrefetchRef[[int]], PrefetchRef[[int]]], tuple[int, int]
        ],
    ) -> None: ...
    @overload
    def __init__[Rows: IntVar, Cols: IntVar](
        self: BlockSpec[[Rows, Cols]],
        block_shape: tuple[Int[Rows], Int[Cols]],
        index_map: Callable[
            [
                int,
                int,
                int,
                PrefetchRef[[int]],
                PrefetchRef[[int]],
                PrefetchRef[[int]],
                PrefetchRef[[int]],
            ],
            tuple[int, int],
        ],
    ) -> None: ...
    @overload
    def __init__[Rows: IntVar, Cols: IntVar](
        self: BlockSpec[[1, Rows, Cols]],
        block_shape: tuple[Literal[1], Int[Rows], Int[Cols]],
        index_map: Callable[
            [
                int,
                int,
                int,
                PrefetchRef[[int]],
                PrefetchRef[[int]],
                PrefetchRef[[int]],
                PrefetchRef[[int]],
            ],
            tuple[int, int, int],
        ],
    ) -> None: ...
    @overload
    def __init__[Rows: IntVar, Cols: IntVar](
        self: BlockSpec[[Rows, Cols]],
        block_shape: tuple[Int[Rows], Int[Cols]],
        index_map: Callable[[int, int, int], tuple[int, int]],
    ) -> None: ...
    @overload
    def __init__[Batch: IntVar, Rows: IntVar, Cols: IntVar](
        self: BlockSpec[[Batch, Rows, Cols]],
        block_shape: tuple[Int[Batch], Int[Rows], Int[Cols]],
        index_map: Callable[[int, int, int], tuple[int, int, int]],
    ) -> None: ...
    @overload
    def __init__[Block: IntVar](
        self: BlockSpec[[Block], Literal[True]],
        block_shape: tuple[None, Int[Block]],
        index_map: Callable[[int, int], tuple[int, int]],
    ) -> None: ...
    @overload
    def __init__[Rows: IntVar, Cols: IntVar](
        self: BlockSpec[[Rows, Cols], Literal["reduction"]],
        block_shape: tuple[None, Int[Rows], Int[Cols]],
        index_map: Callable[[int, int, int], tuple[int, int, int]],
    ) -> None: ...
    @overload
    def __init__[Rows: IntVar, Cols: IntVar](
        self: BlockSpec[[Rows, Cols], Literal["prefetch"]],
        block_shape: tuple[Int[Rows], Int[Cols]],
        index_map: Callable[[int, int, PrefetchRef[[2]]], tuple[int, int]],
    ) -> None: ...

def when(
    predicate: bool | int,
) -> Callable[[Callable[[], None]], Callable[[], None]]: ...
@overload
def kernel[Batch: IntVar, Num: IntVar, Cols: IntVar](
    *,
    out_type: ShapeDtypeStruct[[Num, Cols]],
    mesh: object,
) -> Callable[
    [Callable[[InRef[[Batch, Cols]], InRef[[1, Num]], OutRef[[Num, Cols]]], None]],
    Callable[[Array[[Batch, Cols]], Array[[1, Num]]], Array[[Num, Cols]]],
]: ...
@overload
def kernel[Batch: IntVar, Num: IntVar, Cols: IntVar](
    *,
    out_type: ShapeDtypeStruct[[Batch, Cols]],
    mesh: object,
    scratch_types: list[object],
) -> Callable[
    [Callable[[InRef[[Num, Cols]], InRef[[1, Num]], OutRef[[Batch, Cols]]], None]],
    Callable[[Array[[Num, Cols]], Array[[1, Num]]], Array[[Batch, Cols]]],
]: ...

class PackedGatherKernel[PackedRows: IntVar, Num: IntVar, Cols: IntVar, Window: IntVar](
    Protocol
):
    def __call__(
        self,
        x_packed_hbm: InRef[[PackedRows, Cols]],
        i_hbm: InRef[[Num]],
        o_hbm: OutRef[[Num, Cols]],
        *,
        gather_vmem: tpu.VmemScratchRef[[Window, Cols]],
    ) -> None: ...

@overload
def kernel[PackedRows: IntVar, Num: IntVar, Cols: IntVar, Window: IntVar](
    *,
    out_type: ShapeDtypeStruct[[Num, Cols]],
    mesh: object,
    scratch_types: dict[str, tpu.VMEM[[Window, Cols]]],
) -> Callable[
    [PackedGatherKernel[PackedRows, Num, Cols, Window]],
    Callable[[Array[[PackedRows, Cols]], Array[[Num]]], Array[[Num, Cols]]],
]: ...
@overload
def kernel[Core: IntVar, Lanes: IntVar](
    *,
    out_type: Array[[Core, Lanes]],
    mesh: tpu_sc.ScalarSubcoreMesh[Core],
    scratch_types: list[tpu.SMEM[[Lanes]] | tpu.DmaSemaphoreType],
) -> Callable[
    [
        Callable[
            [
                CoreInRef[Core, Lanes],
                CoreOutRef[Core, Lanes],
                tpu.SmemScratchRef[[Lanes]],
                tpu.DmaSemaphore,
            ],
            None,
        ]
    ],
    Callable[[Array[[Core, Lanes]]], Array[[Core, Lanes]]],
]: ...
@overload
def kernel[Rows: IntVar, Cols: IntVar](
    body: Callable[
        [
            tuple[UnconstrainedInRef[[Rows, Cols]], UnconstrainedInRef[[1]]],
            OutRef[[Rows, Cols // 2]],
            tpu.SmemScratchRef[[1]],
        ],
        None,
    ],
    *,
    out_type: ShapeDtypeStruct[[Rows, Cols // 2]],
    mesh: object,
    scratch_types: list[tpu.SMEM[[1]]],
) -> Callable[
    [tuple[Array[[Rows, Cols]], Int32Array[[1]]]],
    Array[[Rows, Cols // 2]],
]: ...
@overload
def kernel[Rows: IntVar](
    body: Callable[[UnconstrainedInRef[[Rows, 128]], OutRef[[Rows, 128]]], None],
    *,
    out_type: Array[[Rows, 128]],
    mesh: tpu_sc.VectorSubcoreMesh[4, 16],
    scratch_types: list[object],
) -> Callable[[Array[[Rows, 128]]], Array[[Rows, 128]]]: ...
def loop(
    start: int, stop: int, *, step: int = 1
) -> Callable[[Callable[[int], None]], Callable[[int], None]]: ...
@overload
def pallas_call[SourceRows: IntVar, Rows: IntVar, Cols: IntVar](
    body: Callable[
        [
            gpu.TmaGmemSourceRef[SourceRows, Cols],
            gpu.TmaIndexRef[Rows],
            gpu.TmaSmemOutRef[Rows, Cols],
            gpu.TmaBarrier,
        ],
        None,
    ],
    *,
    out_shape: ShapeDtypeStruct[[Rows, Cols]],
    out_specs: gpu.BlockSpec[Rows, Cols],
    in_specs: tuple[
        BlockSpec[[SourceRows, Cols], Literal["tma_gmem"]],
        BlockSpec[[Rows], Literal["tma_smem_indices"]],
    ],
    scratch_shapes: list[gpu.Barrier],
) -> Callable[[Array[[SourceRows, Cols]], Array[[Rows]]], Array[[Rows, Cols]]]: ...
@overload
def pallas_call[
    Splits: IntVar,
    Heads: IntVar,
    SplitKeys: IntVar,
    Dim: IntVar,
    HeadBlock: IntVar,
](
    body: Callable[
        [
            DecodeQueryRef[HeadBlock, Dim],
            DecodeKvRef[SplitKeys, Dim],
            DecodeKvRef[SplitKeys, Dim],
            DecodeBoundRef | None,
            DecodeBoundRef | None,
            DecodeOutputRef[HeadBlock, Dim],
            DecodeResidualRef[HeadBlock],
            DecodeResidualRef[HeadBlock],
        ],
        None,
    ],
    *,
    in_specs: tuple[
        BlockSpec[[HeadBlock, Dim]],
        BlockSpec[[SplitKeys, Dim], Literal["decode_kv"]],
        BlockSpec[[SplitKeys, Dim], Literal["decode_kv"]],
        BlockSpec[[], Literal["decode_bound"]] | None,
        BlockSpec[[], Literal["decode_bound"]] | None,
    ],
    out_specs: tuple[
        BlockSpec[[HeadBlock, Dim], Literal["decode_output"]],
        BlockSpec[[HeadBlock], Literal[True]],
        BlockSpec[[HeadBlock], Literal[True]],
    ],
    grid: tuple[GridSize[Heads, HeadBlock], Int[Splits]],
    out_shape: tuple[
        ShapeDtypeStruct[[Splits, Heads, Dim]],
        ShapeDtypeStruct[[Splits, Heads]],
        ShapeDtypeStruct[[Splits, Heads]],
    ],
    interpret: bool = False,
) -> Callable[
    [
        Array[[Heads, Dim]],
        Array[[Splits, SplitKeys, Dim]],
        Array[[Splits, SplitKeys, Dim]],
        Array[[]] | None,
        Array[[]] | None,
    ],
    tuple[
        Array[[Splits, Heads, Dim]],
        Array[[Splits, Heads]],
        Array[[Splits, Heads]],
    ],
]: ...
@overload
def pallas_call[
    Batch: IntVar,
    Queries: IntVar,
    Keys: IntVar,
    Heads: IntVar,
    Dim: IntVar,
    QueryBlock: IntVar,
](
    body: Callable[
        [
            MhaQueryRef[QueryBlock, Dim],
            MhaKvRef[Keys, Dim],
            MhaKvRef[Keys, Dim],
            MhaSegmentRef[Keys] | None,
            MhaOutputRef[QueryBlock, Dim],
            OutRef[[QueryBlock]],
        ],
        None,
    ],
    *,
    in_specs: tuple[
        BlockSpec[[QueryBlock, Dim], Literal["mha_query"]],
        BlockSpec[[Keys, Dim], Literal["mha_kv"]],
        BlockSpec[[Keys, Dim], Literal["mha_kv"]],
        BlockSpec[[Keys], Literal["mha_segment"]] | None,
    ],
    out_specs: tuple[
        BlockSpec[[QueryBlock, Dim], Literal["mha_query"]],
        BlockSpec[[QueryBlock], Literal["mha_lse"]],
    ],
    grid: tuple[GridSize[Queries, QueryBlock], Int[Batch], Int[Heads]],
    out_shape: tuple[
        ShapeDtypeStruct[[Batch, Queries, Heads, Dim]],
        ShapeDtypeStruct[[Batch, Heads, Queries]],
    ],
    interpret: bool = False,
) -> Callable[
    [
        Array[[Batch, Queries, Heads, Dim]],
        Array[[Batch, Keys, Heads, Dim]],
        Array[[Batch, Keys, Heads, Dim]],
        Array[[Batch, Keys]] | None,
    ],
    tuple[Array[[Batch, Queries, Heads, Dim]], Array[[Batch, Heads, Queries]]],
]: ...
@overload
def pallas_call[
    Batch: IntVar,
    Queries: IntVar,
    Heads: IntVar,
    HeadDim: IntVar,
    PaddedDim: IntVar,
    QueryBlock: IntVar,
](
    body: Callable[
        [
            MhaPreprocessRef[QueryBlock, PaddedDim, HeadDim],
            MhaPreprocessRef[QueryBlock, PaddedDim, HeadDim],
            OutRef[[QueryBlock]],
        ],
        None,
    ],
    *,
    in_specs: tuple[
        BlockSpec[[QueryBlock, PaddedDim], Literal["mha_query"]],
        BlockSpec[[QueryBlock, PaddedDim], Literal["mha_query"]],
    ],
    out_specs: BlockSpec[[QueryBlock], Literal["mha_lse"]],
    grid: tuple[GridSize[Queries, QueryBlock], Int[Batch], Int[Heads]],
    out_shape: ShapeDtypeStruct[[Batch, Heads, Queries]],
    interpret: bool = False,
) -> Callable[
    [Array[[Batch, Queries, Heads, HeadDim]], Array[[Batch, Queries, Heads, HeadDim]]],
    Array[[Batch, Heads, Queries]],
]: ...
@overload
def pallas_call[
    Batch: IntVar,
    Queries: IntVar,
    Keys: IntVar,
    Heads: IntVar,
    HeadDim: IntVar,
    PaddedDim: IntVar,
    QueryBlock: IntVar,
    KeyBlock: IntVar,
](
    body: Callable[
        [
            MhaBackwardMatrixRef[Queries, PaddedDim, HeadDim],
            MhaBackwardMatrixRef[Keys, PaddedDim, HeadDim],
            MhaBackwardMatrixRef[Keys, PaddedDim, HeadDim],
            MhaSegmentRef[Keys] | None,
            MhaBackwardMatrixRef[Queries, PaddedDim, HeadDim],
            MhaBackwardMatrixRef[Queries, PaddedDim, HeadDim],
            MhaBackwardVectorRef[Queries],
            MhaBackwardVectorRef[Queries],
            MhaBackwardOutputRef[QueryBlock, PaddedDim],
            MhaBackwardOutputRef[KeyBlock, PaddedDim],
            MhaBackwardOutputRef[KeyBlock, PaddedDim],
        ],
        None,
    ],
    *,
    in_specs: tuple[
        BlockSpec[[Queries, PaddedDim], Literal["mha_backward_input"]],
        BlockSpec[[Keys, PaddedDim], Literal["mha_backward_input"]],
        BlockSpec[[Keys, PaddedDim], Literal["mha_backward_input"]],
        BlockSpec[[Keys], Literal["mha_segment"]] | None,
        BlockSpec[[Queries, PaddedDim], Literal["mha_backward_input"]],
        BlockSpec[[Queries, PaddedDim], Literal["mha_backward_input"]],
        BlockSpec[[Queries], Literal["mha_lse"]],
        BlockSpec[[Queries], Literal["mha_lse"]],
    ],
    out_specs: tuple[
        BlockSpec[[QueryBlock, PaddedDim], Literal["mha_backward_output"]],
        BlockSpec[[KeyBlock, PaddedDim], Literal["mha_backward_output"]],
        BlockSpec[[KeyBlock, PaddedDim], Literal["mha_backward_output"]],
    ],
    grid: tuple[Int[Batch], Int[Heads], GridSize[Keys, KeyBlock]],
    out_shape: tuple[
        ShapeDtypeStruct[[Batch, Queries, Heads, HeadDim]],
        ShapeDtypeStruct[[Batch, Keys, Heads, HeadDim]],
        ShapeDtypeStruct[[Batch, Keys, Heads, HeadDim]],
    ],
    interpret: bool = False,
) -> Callable[
    [
        Array[[Batch, Queries, Heads, HeadDim]],
        Array[[Batch, Keys, Heads, HeadDim]],
        Array[[Batch, Keys, Heads, HeadDim]],
        Array[[Batch, Keys]] | None,
        Array[[Batch, Queries, Heads, HeadDim]],
        Array[[Batch, Queries, Heads, HeadDim]],
        Array[[Batch, Heads, Queries]],
        Array[[Batch, Heads, Queries]],
    ],
    tuple[
        Array[[Batch, Queries, Heads, HeadDim]],
        Array[[Batch, Keys, Heads, HeadDim]],
        Array[[Batch, Keys, Heads, HeadDim]],
    ],
]: ...
@overload
def pallas_call[
    Heads: IntVar,
    Dim: IntVar,
    Pages: IntVar,
    PageSize: IntVar,
    TablePages: IntVar,
](
    body: Callable[
        [
            PagedQueryRef[Heads, Dim],
            PagedPoolRef[Pages, PageSize, Dim],
            None,
            PagedPoolRef[Pages, PageSize, Dim],
            None,
            PagedBlockTableRef[TablePages],
            None,
            PagedOutputRef[Heads, Dim],
            PagedResidualRef[Heads],
            PagedResidualRef[Heads],
        ],
        None,
    ],
    *,
    grid: tuple[Literal[1], Literal[1], Literal[1]],
    out_shape: tuple[
        ShapeDtypeStruct[[Heads, Dim]],
        ShapeDtypeStruct[[Heads]],
        ShapeDtypeStruct[[Heads]],
    ],
    interpret: bool = False,
) -> Callable[
    [
        Array[[Heads, Dim]],
        Array[[Pages, PageSize, Dim]],
        None,
        Array[[Pages, PageSize, Dim]],
        None,
        Array[[TablePages]],
        None,
    ],
    tuple[Array[[Heads, Dim]], Array[[Heads]], Array[[Heads]]],
]: ...
@overload
def pallas_call[Rows: IntVar, Cols: IntVar, ColBlock: IntVar](
    kernel: Callable[
        [
            LayerNormMatrixRef[Rows, Cols],
            LayerNormVectorRef[Cols],
            LayerNormVectorRef[Cols],
            LayerNormMatrixRef[Rows, Cols],
            LayerNormVectorRef[Rows],
            LayerNormVectorRef[Rows],
            LayerNormOutRef[Cols],
            LayerNormOutRef[Cols],
        ],
        None,
    ],
    *,
    grid: tuple[GridSize[Cols, ColBlock]],
    out_shape: tuple[ShapeDtypeStruct[[Cols]], ShapeDtypeStruct[[Cols]]],
    interpret: bool = False,
) -> Callable[
    [
        Array[[Rows, Cols]],
        Array[[Cols]],
        Array[[Cols]],
        Array[[Rows, Cols]],
        Array[[Rows]],
        Array[[Rows]],
    ],
    tuple[Array[[Cols]], Array[[Cols]]],
]: ...
@overload
def pallas_call[Rows: IntVar, Cols: IntVar, ColBlock: IntVar](
    kernel: Callable[
        [
            LayerNormMatrixRef[Rows, Cols],
            LayerNormVectorRef[Cols],
            LayerNormVectorRef[Cols],
            LayerNormMatrixRef[Rows, Cols],
            LayerNormVectorRef[Rows],
            LayerNormOutRef[Cols],
            LayerNormOutRef[Cols],
        ],
        None,
    ],
    *,
    grid: tuple[GridSize[Cols, ColBlock]],
    out_shape: tuple[ShapeDtypeStruct[[Cols]], ShapeDtypeStruct[[Cols]]],
    interpret: bool = False,
) -> Callable[
    [
        Array[[Rows, Cols]],
        Array[[Cols]],
        Array[[Cols]],
        Array[[Rows, Cols]],
        Array[[Rows]],
    ],
    tuple[Array[[Cols]], Array[[Cols]]],
]: ...
@overload
def pallas_call[
    Batch: IntVar,
    Hidden: IntVar,
    Vocab: IntVar,
    BatchBlock: IntVar,
    VocabBlock: IntVar,
](
    body: Callable[
        [
            LseInputRef[BatchBlock, Hidden],
            LseWeightRef[Hidden, VocabBlock],
            LseOutputRef[BatchBlock],
            tpu.VmemScratchRef[[BatchBlock, 128]],
            tpu.VmemScratchRef[[BatchBlock, 128]],
        ],
        None,
    ],
    *,
    out_shape: ShapeDtypeStruct[[Batch, 128]],
    in_specs: tuple[
        BlockSpec[[BatchBlock, Hidden], Literal["lse_vmem"]],
        BlockSpec[[Hidden, VocabBlock], Literal["lse_vmem"]],
    ],
    out_specs: BlockSpec[[BatchBlock, 128], Literal["lse_vmem"]],
    grid: tuple[
        Literal[1], GridSize[Batch, BatchBlock], GridSize[Vocab, VocabBlock], Literal[1]
    ],
    scratch_shapes: tuple[tpu.VMEM[[BatchBlock, 128]], tpu.VMEM[[BatchBlock, 128]]],
    interpret: bool = False,
) -> Callable[
    [Array[[Batch, Hidden]], Array[[Hidden, Vocab]]],
    Array[[Batch, 128]],
]: ...
@overload
def pallas_call[
    Rows: IntVar,
    Inner: IntVar,
    Cols: IntVar,
    Groups: IntVar,
    RowBlock: IntVar,
    ColBlock: IntVar,
](
    body: Callable[
        [
            RaggedLhsRef[Rows, Inner],
            RaggedRhsRef[Inner, ColBlock],
            RaggedBoundRef[Rows],
            RaggedBoundRef[Rows],
            RaggedOutRef[Rows, Cols, ColBlock],
        ],
        None,
    ],
    *,
    out_shape: ShapeDtypeStruct[[Rows, Cols]],
    in_specs: tuple[
        NoBlockSpec,
        BlockSpec[[Inner, ColBlock], Literal["ragged_group"]],
        BlockSpec[[], Literal["ragged_bound"]],
        BlockSpec[[], Literal["ragged_bound"]],
    ],
    out_specs: BlockSpec[[Rows, ColBlock]],
    grid: tuple[GridSize[Rows, RowBlock], GridSize[Cols, ColBlock], Int[Groups]],
    interpret: bool = False,
) -> Callable[
    [
        Array[[Rows, Inner]],
        Array[[Groups, Inner, Cols]],
        Array[[Groups]],
        Array[[Groups]],
    ],
    Array[[Rows, Cols]],
]: ...
@overload
def pallas_call[Rows: IntVar, Cols: IntVar](
    body: Callable[[tpu.PallasKeyRef, ShapeReadableOutRef[[Rows, Cols]]], None],
    *,
    in_specs: list[BlockSpec[[], Literal["key_smem"]]],
    out_shape: ShapeDtypeStruct[[Rows, Cols]],
) -> Callable[[tpu.PallasKey], Array[[Rows, Cols]]]: ...
@overload
def pallas_call[BlockRows: IntVar, BlockCols: IntVar](
    body: Callable[
        [tpu.PallasKeyRef, ShapeReadableOutRef[[BlockRows, BlockCols]]], None
    ],
    *,
    out_shape: Array[[64, 512]],
    in_specs: list[BlockSpec[[], Literal["key_smem"]]],
    out_specs: BlockSpec[[BlockRows, BlockCols]],
    grid: tuple[Int[64 // BlockRows], Int[512 // BlockCols]],
) -> Callable[[tpu.PallasKey], Array[[64, 512]]]: ...
@overload
def pallas_call[Length: IntVar](
    kernel: Callable[
        [
            InRef[[Length]],
            InRef[[Length]],
            InRef[[Length]],
            InRef[[Length]],
            InRef[[]],
            InRef[[]],
            OutRef[[Length]],
        ],
        None,
    ],
    *,
    out_shape: ShapeDtypeStruct[[Length]],
    grid: tuple[()],
    interpret: bool = False,
) -> Callable[
    [
        Array[[Length]],
        Array[[Length]],
        Array[[Length]],
        Array[[Length]],
        Array[[]],
        Array[[]],
    ],
    Array[[Length]],
]: ...
@overload
def pallas_call[Length: IntVar](
    kernel: Callable[
        [
            InRef[[Length]],
            InRef[[Length]],
            InRef[[Length]],
            InRef[[Length]],
            InRef[[]],
            OutRef[[Length]],
        ],
        None,
    ],
    *,
    out_shape: ShapeDtypeStruct[[Length]],
    grid: tuple[()],
    interpret: bool = False,
) -> Callable[
    [
        Array[[Length]],
        Array[[Length]],
        Array[[Length]],
        Array[[Length]],
        Array[[]],
    ],
    Array[[Length]],
]: ...
@overload
def pallas_call[Length: IntVar](
    kernel: Callable[
        [
            InRef[[Length]],
            InRef[[Length]],
            InRef[[Length]],
            OutRef[[Length]],
            OutRef[[]],
            OutRef[[]],
        ],
        None,
    ],
    *,
    out_shape: tuple[
        ShapeDtypeStruct[[Length]],
        ShapeDtypeStruct[[]],
        ShapeDtypeStruct[[]],
    ],
    grid: tuple[()],
    interpret: bool = False,
) -> Callable[
    [Array[[Length]], Array[[Length]], Array[[Length]]],
    tuple[Array[[Length]], Array[[]], Array[[]]],
]: ...
@overload
def pallas_call[Length: IntVar](
    kernel: Callable[
        [
            InRef[[Length]],
            InRef[[Length]],
            InRef[[Length]],
            OutRef[[Length]],
            OutRef[[]],
        ],
        None,
    ],
    *,
    out_shape: tuple[ShapeDtypeStruct[[Length]], ShapeDtypeStruct[[]]],
    grid: tuple[()],
    interpret: bool = False,
) -> Callable[
    [Array[[Length]], Array[[Length]], Array[[Length]]],
    tuple[Array[[Length]], Array[[]]],
]: ...
@overload
def pallas_call[
    Batch: IntVar,
    Seq: IntVar,
    Cols: IntVar,
    Block: IntVar,
    Width: IntVar,
    Channels: IntVar,
    Halo: IntVar,
](
    kernel: Callable[
        [
            ConvValuesRef[Seq, Channels],
            ConvValuesRef[Halo, Channels],
            ConvSegmentRef[Seq],
            ConvSegmentRef[Halo],
            ConvWeightRef[Width, Channels],
            ConvOutRef[Block, Channels],
        ],
        None,
    ],
    *,
    out_shape: ShapeDtypeStruct[[Batch, Seq, Cols]],
    grid: tuple[Int[Batch], int, int],
    in_specs: tuple[
        BlockSpec[[1, Seq, Channels]],
        BlockSpec[[1, Halo, Channels]],
        BlockSpec[[1, Seq]],
        BlockSpec[[1, Halo]],
        BlockSpec[[Width, Channels]],
    ],
    out_specs: BlockSpec[[1, Block, Channels]],
    interpret: bool = False,
) -> Callable[
    [
        Array[[Batch, Seq, Cols]],
        Array[[Batch, Halo, Cols]],
        Array[[Batch, Seq]],
        Array[[Batch, Halo]],
        Array[[Width, Cols]],
    ],
    Array[[Batch, Seq, Cols]],
]: ...
@overload
def pallas_call[
    Batch: IntVar,
    Seq: IntVar,
    Cols: IntVar,
    Block: IntVar,
    Width: IntVar,
    Channels: IntVar,
    Halo: IntVar,
](
    kernel: Callable[
        [
            ConvValuesRef[Seq, Channels],
            ConvValuesRef[Halo, Channels],
            ConvSegmentRef[Seq],
            ConvSegmentRef[Halo],
            ConvValuesRef[Seq, Channels],
            ConvValuesRef[Halo, Channels],
            ConvSegmentRef[Halo],
            ConvWeightRef[Width, Channels],
            ConvOutRef[Block, Channels],
            ConvPartialWeightRef[Width, Channels],
        ],
        None,
    ],
    *,
    out_shape: tuple[
        ShapeDtypeStruct[[Batch, Seq, Cols]],
        ShapeDtypeStruct[[Batch * (Seq // Block), Width, Cols]],
    ],
    grid: tuple[Int[Batch], int, int],
    in_specs: tuple[
        BlockSpec[[1, Seq, Channels]],
        BlockSpec[[1, Halo, Channels]],
        BlockSpec[[1, Seq]],
        BlockSpec[[1, Halo]],
        BlockSpec[[1, Seq, Channels]],
        BlockSpec[[1, Halo, Channels]],
        BlockSpec[[1, Halo]],
        BlockSpec[[Width, Channels]],
    ],
    out_specs: tuple[BlockSpec[[1, Block, Channels]], BlockSpec[[1, Width, Channels]]],
    interpret: bool = False,
) -> Callable[
    [
        Array[[Batch, Seq, Cols]],
        Array[[Batch, Halo, Cols]],
        Array[[Batch, Seq]],
        Array[[Batch, Halo]],
        Array[[Batch, Seq, Cols]],
        Array[[Batch, Halo, Cols]],
        Array[[Batch, Halo]],
        Array[[Width, Cols]],
    ],
    tuple[
        Array[[Batch, Seq, Cols]],
        Array[[Batch * (Seq // Block), Width, Cols]],
    ],
]: ...
@overload
def pallas_call[
    Blocks: IntVar,
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
](
    kernel: Callable[
        [
            PrefetchRef[[Blocks]],
            PrefetchRef[[Blocks]],
            InRef[[1, BM, BK]],
            InRef[[BK, BN]],
            InRef[[BM, BN]],
            OutRef[[BM, BN]],
            tpu.VmemScratchRef[[BM, BN]],
        ],
        None,
    ],
    *,
    grid_spec: tpu.PrefetchScalarGridSpec[1, 1, BM, BN, BK, BK, Blocks],
    out_shape: ShapeDtypeStruct[[M, N]],
    input_output_aliases: dict[Literal[4], Literal[0]],
) -> Callable[
    [
        Array[[Blocks]],
        Array[[Blocks]],
        Array[[Blocks, BM, BK]],
        Array[[K, N]],
        Array[[M, N]],
    ],
    Array[[M, N]],
]: ...
@overload
def pallas_call[
    GridRows: IntVar,
    GridCols: IntVar,
    Rows: IntVar,
    Cols: IntVar,
    Inner: IntVar,
    RowBlock: IntVar,
    ColBlock: IntVar,
    InnerBlock: IntVar,
    MaskTypes: IntVar,
](
    kernel: Callable[
        [
            PrefetchRef[[GridRows, GridCols]],
            PrefetchRef[[GridRows, GridCols]],
            PrefetchRef[[GridRows, GridCols]],
            PrefetchRef[[GridRows, GridCols]],
            InRef[[RowBlock, InnerBlock]],
            InRef[[InnerBlock, ColBlock]],
            InRef[[1, RowBlock, ColBlock]],
            OutRef[[RowBlock, ColBlock]],
            tpu.VmemScratchRef[[RowBlock, ColBlock]],
        ],
        None,
    ],
    *,
    grid_spec: tpu.PrefetchScalarGridSpec[
        int, int, RowBlock, ColBlock, int, InnerBlock, int
    ],
    out_shape: ShapeDtypeStruct[[Rows, Cols]],
) -> Callable[
    [
        Array[[GridRows, GridCols]],
        Array[[GridRows, GridCols]],
        Array[[GridRows, GridCols]],
        Array[[GridRows, GridCols]],
        Array[[Rows, Inner]],
        Array[[Inner, Cols]],
        Array[[MaskTypes, RowBlock, ColBlock]],
    ],
    Array[[Rows, Cols]],
]: ...
@overload
def pallas_call[Devices: IntVar, Rows: IntVar, Cols: IntVar](
    kernel: Callable[
        [
            VmemInRef[[Rows, Cols]],
            AccumRef[[Rows, Cols]],
            OutRef[[2, Rows, Cols]],
            tpu.DmaSemaphore,
            tpu.DmaSemaphore,
            tpu.DmaSemaphore,
            tpu.RegularSemaphore,
            tpu.VmemScratchRef[[Rows, Cols]],
        ],
        None,
    ],
    out_shape: tuple[ShapeDtypeStruct[[Rows, Cols]], ShapeDtypeStruct[[2, Rows, Cols]]],
    *,
    grid_spec: tpu.PrefetchScalarGridSpec[
        Rows, Cols, Rows, Cols, Literal[1], Literal[1], Devices
    ],
    compiler_params: tpu.CompilerParams,
    interpret: bool = False,
) -> Callable[
    [Array[[Rows, Cols]]],
    tuple[Array[[Rows, Cols]], Array[[2, Rows, Cols]]],
]: ...
@overload
def pallas_call[Devices: IntVar, Rows: IntVar, Cols: IntVar](
    kernel: Callable[
        [
            VmemInRef[[Devices, Rows, Cols]],
            ScatterOutRef[[Rows, Cols]],
            OutRef[[2, Rows, Cols]],
            tpu.DmaSemaphore,
            tpu.DmaSemaphore,
            tpu.DmaSemaphore,
            tpu.DmaSemaphore,
            tpu.DmaSemaphore,
            tpu.RegularSemaphore,
            tpu.RegularSemaphore,
            tpu.VmemScratchRef[[Rows // 2, Cols]],
        ],
        None,
    ],
    out_shape: tuple[ShapeDtypeStruct[[Rows, Cols]], ShapeDtypeStruct[[2, Rows, Cols]]],
    *,
    grid_spec: tpu.PrefetchScalarGridSpec[
        Devices * Rows, Cols, Rows, Cols, Literal[1], Literal[1], Devices
    ],
    compiler_params: tpu.CompilerParams,
    interpret: bool = False,
) -> Callable[
    [Array[[Devices, Rows, Cols]]],
    tuple[Array[[Rows, Cols]], Array[[2, Rows, Cols]]],
]: ...
@overload
def pallas_call[Devices: IntVar, Rows: IntVar, Cols: IntVar](
    kernel: Callable[
        [
            UnconstrainedInRef[[Rows, Cols]],
            OutRef[[Devices, Rows, Cols]],
            tpu.DmaSemaphore,
            tpu.DmaSemaphore,
            tpu.DmaSemaphoreArray[Devices - 1],
        ],
        None,
    ],
    out_shape: ShapeDtypeStruct[[Devices, Rows, Cols]],
    *,
    grid_spec: tpu.PrefetchScalarGridSpec[
        Rows, Cols, Rows, Cols, Literal[1], Literal[1], Devices
    ],
    interpret: bool = False,
) -> Callable[[Array[[Rows, Cols]]], Array[[Devices, Rows, Cols]]]: ...
@overload
def pallas_call[Rows: IntVar, Cols: IntVar](
    kernel: Callable[
        [
            UnconstrainedInRef[[Rows, Cols]],
            OutRef[[Rows, Cols]],
            tpu.DmaSemaphore,
            tpu.DmaSemaphore,
        ],
        None,
    ],
    out_shape: ShapeDtypeStruct[[Rows, Cols]],
    *,
    grid_spec: tpu.PrefetchScalarGridSpec[Rows, Cols, Rows, Cols],
    interpret: bool = False,
) -> Callable[[Array[[Rows, Cols]]], Array[[Rows, Cols]]]: ...
@overload
def pallas_call[Rows: IntVar, Cols: IntVar, RowBlock: IntVar](
    kernel: Callable[
        [InRef[[RowBlock, Cols]], InRef[[RowBlock, Cols]], OutRef[[RowBlock, Cols]]],
        None,
    ],
    out_shape: ShapeDtypeStruct[[Rows, Cols]],
    *,
    grid: tuple[GridSize[Rows, RowBlock]],
    in_specs: tuple[BlockSpec[[RowBlock, Cols]], BlockSpec[[RowBlock, Cols]]],
    out_specs: BlockSpec[[RowBlock, Cols]],
    compiler_params: tpu.CompilerParams,
    interpret: bool = False,
) -> Callable[[Array[[Rows, Cols]], Array[[Rows, Cols]]], Array[[Rows, Cols]]]: ...
@overload
def pallas_call[Rows: IntVar, Cols: IntVar, SliceRows: IntVar](
    kernel: Callable[
        [
            UnconstrainedInRef[[Rows, Cols]],
            UnconstrainedInRef[[SliceRows, 2]],
            OutRef[[Rows, Cols]],
            tpu.SmemScratchRef[[SliceRows, 2]],
        ],
        None,
    ],
    out_shape: ShapeDtypeStruct[[Rows, Cols]],
    *,
    in_specs: tuple[
        BlockSpec[[Rows, Cols], Literal["unconstrained"]],
        BlockSpec[[SliceRows, 2], Literal["unconstrained"]],
    ],
    out_specs: BlockSpec[[Rows, Cols], Literal["unconstrained"]],
    scratch_shapes: tuple[tpu.SMEM[[SliceRows, 2]]],
    interpret: bool = False,
) -> Callable[[Array[[Rows, Cols]], Array[[SliceRows, 2]]], Array[[Rows, Cols]]]: ...
@overload
def pallas_call[Rows: IntVar, Cols: IntVar](
    kernel: Callable[
        [
            UnconstrainedInRef[[Rows, Cols]],
            OutRef[[1, Cols]],
            tpu.VmemScratchRef[[1, Cols]],
        ],
        None,
    ],
    out_shape: ShapeDtypeStruct[[1, Cols]],
    *,
    in_specs: list[BlockSpec[[Rows, Cols], Literal["unconstrained"]]],
    scratch_shapes: tuple[tpu.VMEM[[1, Cols]]],
    interpret: bool = False,
) -> Callable[[Array[[Rows, Cols]]], Array[[1, Cols]]]: ...
@overload
def pallas_call[Length: IntVar](
    kernel: Callable[[InRef[[Length]], OutRef[[Length]]], None],
    out_shape: ShapeDtypeStruct[[Length]],
    *,
    grid: tuple[()],
    compiler_params: triton.CompilerParams | None = None,
    debug: bool = False,
    interpret: bool = False,
) -> Callable[[Array[[Length]]], Array[[Length]]]: ...
@overload
def pallas_call[
    SourceRows: IntVar,
    SourceCols: IntVar,
    RowBlock: IntVar,
    ColBlock: IntVar,
](
    kernel: Callable[
        [PrefetchRef[[2]], InRef[[RowBlock, ColBlock]], OutRef[[RowBlock, ColBlock]]],
        None,
    ],
    out_shape: ShapeDtypeStruct[[RowBlock, ColBlock]],
    *,
    grid_spec: tpu.PrefetchScalarGridSpec[SourceRows, SourceCols, RowBlock, ColBlock],
    interpret: bool = False,
) -> Callable[
    [Array[[2]], Array[[SourceRows, SourceCols]]], Array[[RowBlock, ColBlock]]
]: ...
@overload
def pallas_call[
    Rows: IntVar,
    Inner: IntVar,
    Cols: IntVar,
    RowBlock: IntVar,
    InnerBlock: IntVar,
    ColBlock: IntVar,
](
    kernel: Callable[
        [
            InRef[[RowBlock, InnerBlock]],
            InRef[[ColBlock, InnerBlock]],
            OutRef[[RowBlock, ColBlock]],
            AccumRef[[RowBlock, ColBlock]],
        ],
        None,
    ],
    out_shape: ShapeDtypeStruct[[Rows, Cols]],
    *,
    grid_spec: tpu.PrefetchScalarGridSpec[
        Rows, Cols, RowBlock, ColBlock, Inner, InnerBlock, Literal[1], Literal[True]
    ],
    interpret: bool = False,
) -> Callable[[Array[[Rows, Inner]], Array[[Cols, Inner]]], Array[[Rows, Cols]]]: ...
@overload
def pallas_call[
    Rows: IntVar,
    Inner: IntVar,
    Cols: IntVar,
    RowBlock: IntVar,
    InnerBlock: IntVar,
    ColBlock: IntVar,
](
    kernel: Callable[
        [
            InRef[[RowBlock, InnerBlock]],
            InRef[[InnerBlock, ColBlock]],
            OutRef[[RowBlock, ColBlock]],
            AccumRef[[RowBlock, ColBlock]],
        ],
        None,
    ],
    out_shape: ShapeDtypeStruct[[Rows, Cols]],
    *,
    grid_spec: tpu.PrefetchScalarGridSpec[
        Rows, Cols, RowBlock, ColBlock, Inner, InnerBlock
    ],
    interpret: bool = False,
) -> Callable[[Array[[Rows, Inner]], Array[[Inner, Cols]]], Array[[Rows, Cols]]]: ...
@overload
def pallas_call[Length: IntVar, Block: IntVar](
    kernel: Callable[[InRef[[Block]], InRef[[Block]], OutRef[[Block]]], None],
    out_shape: ShapeDtypeStruct[[Length]],
    *,
    grid: tuple[GridSize[Length, Block]],
    in_specs: tuple[BlockSpec[[Block]], BlockSpec[[Block]]],
    out_specs: BlockSpec[[Block]],
    interpret: bool = False,
) -> Callable[[Array[[Length]], Array[[Length]]], Array[[Length]]]: ...
@overload
def pallas_call[
    Rows: IntVar,
    Inner: IntVar,
    Cols: IntVar,
    RowBlock: IntVar,
    ColBlock: IntVar,
](
    kernel: Callable[
        [
            InRef[[RowBlock, Inner]],
            InRef[[Inner, ColBlock]],
            OutRef[[RowBlock, ColBlock]],
        ],
        None,
    ],
    out_shape: ShapeDtypeStruct[[Rows, Cols]],
    *,
    grid: tuple[GridSize[Rows, RowBlock], GridSize[Cols, ColBlock]],
    in_specs: tuple[BlockSpec[[RowBlock, Inner]], BlockSpec[[Inner, ColBlock]]],
    out_specs: BlockSpec[[RowBlock, ColBlock]],
    interpret: bool = False,
) -> Callable[[Array[[Rows, Inner]], Array[[Inner, Cols]]], Array[[Rows, Cols]]]: ...
@overload
def pallas_call[Rows: IntVar, Cols: IntVar, Block: IntVar](
    kernel: Callable[[InRef[[Block]], InRef[[Block]], OutRef[[Block]]], None],
    out_shape: ShapeDtypeStruct[[Rows, Cols]],
    *,
    grid: tuple[Int[Rows], GridSize[Cols, Block]],
    in_specs: tuple[
        BlockSpec[[Block], Literal[True]], BlockSpec[[Block], Literal[True]]
    ],
    out_specs: BlockSpec[[Block], Literal[True]],
    interpret: bool = False,
) -> Callable[[Array[[Rows, Cols]], Array[[Rows, Cols]]], Array[[Rows, Cols]]]: ...
@overload
def pallas_call[Rows: IntVar, Cols: IntVar](
    kernel: Callable[
        [InRef[[Rows, Cols]], InRef[[Rows, Cols]], OutRef[[Rows, Cols]]], None
    ],
    out_shape: ShapeDtypeStruct[[Rows, Cols]],
    *,
    interpret: bool = False,
) -> Callable[[Array[[Rows, Cols]], Array[[Rows, Cols]]], Array[[Rows, Cols]]]: ...
@overload
def pallas_call[
    Reduce: IntVar,
    Rows: IntVar,
    Cols: IntVar,
    RowBlock: IntVar,
    ColBlock: IntVar,
](
    kernel: Callable[
        [InRef[[RowBlock, ColBlock]], AccumRef[[RowBlock, ColBlock]]], None
    ],
    out_shape: ShapeDtypeStruct[[Rows, Cols]],
    *,
    grid: tuple[GridSize[Rows, RowBlock], GridSize[Cols, ColBlock], Int[Reduce]],
    in_specs: list[BlockSpec[[RowBlock, ColBlock], Literal["reduction"]]],
    out_specs: BlockSpec[[RowBlock, ColBlock]],
    interpret: bool = False,
) -> Callable[[Array[[Reduce, Rows, Cols]]], Array[[Rows, Cols]]]: ...
@overload
def pallas_call[Length: IntVar](
    kernel: Callable[[OutRef[[Length]]], None],
    out_shape: ShapeDtypeStruct[[Length]],
    *,
    grid: tuple[Int[Length]],
    interpret: bool = False,
) -> Callable[[], Array[[Length]]]: ...

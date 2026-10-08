"""Type-level Triton pointer operations for host tensor view markers."""

from typing import Literal, overload

import triton.language as tl
from shape_extensions import IntTuple, IntVar

class InPointer[Target: IntTuple, Strides: IntTuple]:
    @overload
    def __add__[
        Batch: IntVar, Heads: IntVar, Tokens: IntVar, Dim: IntVar,
        StrideZ: IntVar, StrideH: IntVar, TokenStride: IntVar,
        FeatureStride: IntVar,
    ](
        self: InPointer[
            [Batch, Heads, Tokens, Dim],
            [StrideZ, StrideH, TokenStride, FeatureStride],
        ],
        offset: tl.AttentionBatchHeadOffset[Heads, StrideZ, StrideH],
    ) -> tl.AttentionHeadLocalInputPointer[
        Tokens, Dim, TokenStride, FeatureStride
    ]: ...
    @overload
    def __add__[Batch: IntVar, Heads: IntVar, Tokens: IntVar, GridAxis: int](
        self: InPointer[[Batch, Heads, Tokens], [int, Tokens, 1]],
        offset: tl.TileStart[[Tokens], GridAxis],
    ) -> tl.AttentionHeadLocalStatsPointer[Tokens]: ...
    @overload
    def __add__[Batch: IntVar, Heads: IntVar, Tokens: IntVar, Dim: IntVar](
        self: InPointer[[Batch, Heads, Tokens, Dim], [int, int, Dim, 1]],
        start: tl.ScaledTileStart[Dim, Tokens],
    ) -> tl.Attention4DHeadPointer[Batch, Heads, Tokens, Dim]: ...
    @overload
    def __add__[GridAxis: int](
        self, row: tl.ProgramId[GridAxis]
    ) -> tl.InScalarPointer[Target]: ...
    @overload
    def __add__[
        Length: IntVar, Stride: IntVar, Tile: IntTuple, Origin: str, GridAxis: int
    ](
        self: InPointer[[Length], [Stride]],
        offsets: tl.Offsets[Tile, Stride, Origin, GridAxis],
    ) -> tl.InTilePointers[[Length], [Stride], Tile, Origin, GridAxis]: ...
    @overload
    def __add__[Cols: IntVar, BC: IntVar](
        self: InPointer[[Cols], [1]], offsets: tl.ColumnAxisOffsets[[BC], 1]
    ) -> tl.InColumnTilePointers[Cols, BC]: ...
    @overload
    def __add__[
        Rows: IntVar,
        Cols: IntVar,
        Stride: IntVar,
        ColumnStride: IntVar,
        GridAxis: int,
    ](
        self: InPointer[[Rows, Cols], [Stride, ColumnStride]],
        offset: tl.TileStart[[Stride], GridAxis],
    ) -> InPointer[[Cols], [ColumnStride]]: ...
    @overload
    def __add__[Rows: IntVar, Cols: IntVar, Stride: IntVar, ColumnStride: IntVar](
        self: InPointer[[Rows, Cols], [Stride, ColumnStride]],
        offset: tl.RowOffset[Rows, Stride],
    ) -> InPointer[[Cols], [ColumnStride]]: ...
    @overload
    def __add__[Groups: IntVar, Cols: IntVar, TileRows: IntVar, TileCols: IntVar](
        self: InPointer[[Groups, Cols], [Cols, 1]],
        address: tl.GroupedMatrixOffsets[[TileRows], [TileCols], Cols],
    ) -> tl.InTilePointers[
        [Groups, Cols], [Cols, 1], [TileRows, TileCols], Literal["grouped"]
    ]: ...
    @overload
    def __add__[
        Rows: IntVar,
        Cols: IntVar,
        RowStride: IntVar,
        ColStride: IntVar,
        TileRows: IntVar,
        TileCols: IntVar,
    ](
        self: InPointer[[Rows, Cols], [RowStride, ColStride]],
        address: tl.WrappedRowMatrixAddress[
            Rows, [TileRows], [TileCols], RowStride, ColStride
        ],
    ) -> tl.InTilePointers[
        [Rows, Cols],
        [RowStride, ColStride],
        [TileRows, TileCols],
        Literal["wrapped_0"],
    ]: ...
    @overload
    def __add__[
        Rows: IntVar,
        Cols: IntVar,
        RowStride: IntVar,
        ColStride: IntVar,
        TileRows: IntVar,
        TileCols: IntVar,
    ](
        self: InPointer[[Rows, Cols], [RowStride, ColStride]],
        address: tl.WrappedColumnMatrixAddress[
            Cols, [TileRows], [TileCols], RowStride, ColStride
        ],
    ) -> tl.InTilePointers[
        [Rows, Cols],
        [RowStride, ColStride],
        [TileRows, TileCols],
        Literal["wrapped_1"],
    ]: ...
class OutPointer[Target: IntTuple, Strides: IntTuple]:
    @overload
    def __add__[
        Batch: IntVar, Heads: IntVar, Tokens: IntVar, Dim: IntVar,
        StrideZ: IntVar, StrideH: IntVar, TokenStride: IntVar,
        FeatureStride: IntVar,
    ](
        self: OutPointer[
            [Batch, Heads, Tokens, Dim],
            [StrideZ, StrideH, TokenStride, FeatureStride],
        ],
        offset: tl.AttentionBatchHeadOffset[Heads, StrideZ, StrideH],
    ) -> tl.AttentionHeadLocalOutputPointer[
        Tokens, Dim, TokenStride, FeatureStride
    ]: ...
    @overload
    def __add__[Batch: IntVar, Heads: IntVar, Tokens: IntVar, Axis: int](
        self: OutPointer[[Batch, Heads, Tokens], [int, Tokens, 1]],
        start: tl.TileStart[[Tokens], Axis],
    ) -> tl.Attention3DHeadPointer[Batch, Heads, Tokens]: ...
    @overload
    def __add__[GridAxis: int](
        self, row: tl.ProgramId[GridAxis]
    ) -> tl.OutScalarPointer[Target]: ...
    @overload
    def __add__[
        Length: IntVar, Stride: IntVar, Tile: IntTuple, Origin: str, GridAxis: int
    ](
        self: OutPointer[[Length], [Stride]],
        offsets: tl.Offsets[Tile, Stride, Origin, GridAxis],
    ) -> tl.OutTilePointers[[Length], [Stride], Tile, Origin, GridAxis]: ...
    @overload
    def __add__[
        Rows: IntVar,
        Cols: IntVar,
        Stride: IntVar,
        ColumnStride: IntVar,
        GridAxis: int,
    ](
        self: OutPointer[[Rows, Cols], [Stride, ColumnStride]],
        offset: tl.TileStart[[Stride], GridAxis],
    ) -> OutPointer[[Cols], [ColumnStride]]: ...
    @overload
    def __add__[Rows: IntVar, Cols: IntVar, Stride: IntVar, ColumnStride: IntVar](
        self: OutPointer[[Rows, Cols], [Stride, ColumnStride]],
        offset: tl.RowOffset[Rows, Stride],
    ) -> OutPointer[[Cols], [ColumnStride]]: ...
    @overload
    def __add__[
        Rows: IntVar,
        Cols: IntVar,
        RowStride: IntVar,
        ColStride: IntVar,
        TileRows: IntVar,
    ](
        self: OutPointer[[Rows, Cols], [RowStride, ColStride]],
        address: tl.RowAddress[[TileRows], RowStride],
    ) -> tl.OutTilePointers[
        [Rows, Cols], [RowStride, ColStride], [TileRows, 1], Literal["axis_0"]
    ]: ...

class InOutPointer[Target: IntTuple, Strides: IntTuple]:
    @overload
    def __add__[Groups: IntVar, Cols: IntVar](
        self: InOutPointer[[Groups, Cols], [Cols, 1]],
        start: tl.GroupStart[Groups, Cols],
    ) -> InOutPointer[[Cols], [1]]: ...
    @overload
    def __add__[Cols: IntVar, Tile: IntTuple, Origin: str, GridAxis: int](
        self: InOutPointer[[Cols], [1]],
        offsets: tl.Offsets[Tile, 1, Origin, GridAxis],
    ) -> tl.InOutTilePointers[[Cols], [1], Tile, Origin, GridAxis]: ...

"""Type-level Triton pointer operations for host tensor view markers."""

from typing import Literal, overload

import triton.language as tl
from shape_extensions import IntTuple, IntVar

class InPointer[Target: IntTuple, Strides: IntTuple]:
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

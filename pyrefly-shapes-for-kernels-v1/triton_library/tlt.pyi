"""Type-level Triton pointer operations for host tensor view markers."""

from typing import Literal, overload

import triton.language as tl
from shape_extensions import IntTuple, IntVar

class InPointer[Target: IntTuple, Strides: IntTuple]:
    @overload
    def __add__(self, row: tl.ProgramId) -> tl.InScalarPointer[Target]: ...
    @overload
    def __add__[Length: IntVar, Stride: IntVar, Tile: IntTuple, Origin: str](
        self: InPointer[[Length], [Stride]], offsets: tl.Offsets[Tile, Stride, Origin]
    ) -> tl.InTilePointers[[Length], [Stride], Tile, Origin]: ...
    @overload
    def __add__[
        Rows: IntVar,
        Cols: IntVar,
        Stride: IntVar,
        Tile: IntTuple,
        Origin: str,
    ](
        self: InPointer[[Rows, Cols], [Stride, 1]], offsets: tl.Offsets[Tile, 1, Origin]
    ) -> tl.InTilePointers[[Cols], [1], Tile, Origin]: ...
    @overload
    def __add__[Cols: IntVar, BC: IntVar](
        self: InPointer[[Cols], [1]], offsets: tl.ColumnAxisOffsets[[BC], 1]
    ) -> tl.InColumnTilePointers[Cols, BC]: ...
    @overload
    def __add__[Rows: IntVar, Cols: IntVar, Stride: IntVar, ColumnStride: IntVar](
        self: InPointer[[Rows, Cols], [Stride, ColumnStride]],
        offset: tl.TileStart[[Stride]],
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
    def __iadd__[Rows: IntVar, Cols: IntVar, Stride: IntVar, ColumnStride: IntVar](
        self: InPointer[[Rows, Cols], [Stride, ColumnStride]],
        offset: tl.TileStart[[Stride]],
    ) -> InPointer[[Rows, Cols], [Stride, ColumnStride]]: ...

class OutPointer[Target: IntTuple, Strides: IntTuple]:
    @overload
    def __add__(self, row: tl.ProgramId) -> tl.OutScalarPointer[Target]: ...
    @overload
    def __add__[Length: IntVar, Stride: IntVar, Tile: IntTuple, Origin: str](
        self: OutPointer[[Length], [Stride]], offsets: tl.Offsets[Tile, Stride, Origin]
    ) -> tl.OutTilePointers[[Length], [Stride], Tile, Origin]: ...
    @overload
    def __add__[
        Rows: IntVar,
        Cols: IntVar,
        Stride: IntVar,
        Tile: IntTuple,
        Origin: str,
    ](
        self: OutPointer[[Rows, Cols], [Stride, 1]],
        offsets: tl.Offsets[Tile, 1, Origin],
    ) -> tl.OutTilePointers[[Cols], [1], Tile, Origin]: ...
    @overload
    def __add__[Rows: IntVar, Cols: IntVar, Stride: IntVar, ColumnStride: IntVar](
        self: OutPointer[[Rows, Cols], [Stride, ColumnStride]],
        offset: tl.TileStart[[Stride]],
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
    def __iadd__[Rows: IntVar, Cols: IntVar, Stride: IntVar, ColumnStride: IntVar](
        self: OutPointer[[Rows, Cols], [Stride, ColumnStride]],
        offset: tl.TileStart[[Stride]],
    ) -> OutPointer[[Rows, Cols], [Stride, ColumnStride]]: ...

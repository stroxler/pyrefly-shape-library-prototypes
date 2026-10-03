# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal, overload, Sequence

from shape_extensions import IntVar
from triton.experimental.gluon.language import (
    BarrierBuffer1D,
    GluonTileStart,
    MessageDescriptor1D,
    MessageSharedBuffer1D,
    PackedScaleDescriptorU8,
    PackedScaleSharedTile,
    TmaDescriptorF8,
    TmaInputDescriptor1D,
    TmaInputDescriptor2D,
    TmaInputDescriptorF16,
    TmaOutputDescriptor1D,
    TmaOutputDescriptor2D,
    TmaOutputDescriptorF16,
    TmaSharedBuffer1D,
    TmaSharedTile2D,
    WgmmaSharedF8,
    WgmmaSharedF16,
)

@overload
def async_load[
    HostRepRows: IntVar,
    HostRepK: IntVar,
    BlockRows: IntVar,
    BlockK: IntVar,
    Layout: IntVar,
](
    descriptor: PackedScaleDescriptorU8[
        HostRepRows, HostRepK, BlockRows, BlockK, Layout
    ],
    coordinates: Sequence[GluonTileStart[BlockRows // 128] | int],
    barrier: BarrierBuffer1D,
    destination: PackedScaleSharedTile[BlockRows, BlockK, Layout],
    pred: bool = True,
) -> None: ...
@overload
def async_load[
    Rows: IntVar,
    Cols: IntVar,
    BlockRows: IntVar,
    BlockCols: IntVar,
    Layout: IntVar,
](
    descriptor: TmaDescriptorF8[Rows, Cols, BlockRows, BlockCols, Layout],
    coordinates: Sequence[GluonTileStart[BlockRows] | int],
    barrier: BarrierBuffer1D,
    destination: WgmmaSharedF8[BlockRows, BlockCols, Layout],
    pred: bool = True,
) -> None: ...
@overload
def async_load[
    Rows: IntVar,
    Cols: IntVar,
    BlockRows: IntVar,
    BlockCols: IntVar,
    Layout: IntVar,
](
    descriptor: TmaInputDescriptorF16[Rows, Cols, BlockRows, BlockCols, Layout],
    coordinates: Sequence[
        GluonTileStart[BlockRows]
        | GluonTileStart[2 * BlockRows]
        | GluonTileStart[BlockCols]
        | int
    ],
    barrier: BarrierBuffer1D,
    destination: WgmmaSharedF16[BlockRows, BlockCols, Layout],
    pred: bool = True,
) -> None: ...
@overload
def async_load[
    Rows: IntVar,
    Cols: IntVar,
    BlockRows: IntVar,
    BlockCols: IntVar,
    Layout: IntVar,
](
    descriptor: TmaInputDescriptor2D[Rows, Cols, BlockRows, BlockCols, Layout],
    coordinates: list[GluonTileStart[BlockRows] | int],
    barrier: BarrierBuffer1D,
    destination: TmaSharedTile2D[BlockRows, BlockCols, Layout],
) -> None: ...
@overload
def async_load[Length: IntVar, Block: IntVar, Layout: IntVar](
    descriptor: TmaInputDescriptor1D[Length, Block, Layout],
    coordinates: list[GluonTileStart[Block]],
    barrier: BarrierBuffer1D,
    destination: TmaSharedBuffer1D[Block, Layout],
) -> None: ...
@overload
def async_load[Block: IntVar, Layout: IntVar](
    descriptor: MessageDescriptor1D[Block, Block, Layout],
    coordinates: list[Literal[0]],
    barrier: BarrierBuffer1D,
    destination: MessageSharedBuffer1D[Block, Layout],
) -> None: ...
@overload
def async_store[
    Rows: IntVar,
    Cols: IntVar,
    BlockRows: IntVar,
    BlockCols: IntVar,
    Layout: IntVar,
](
    descriptor: TmaOutputDescriptorF16[Rows, Cols, BlockRows, BlockCols, Layout],
    coordinates: Sequence[
        GluonTileStart[BlockRows]
        | GluonTileStart[2 * BlockRows]
        | GluonTileStart[BlockCols]
        | int
    ],
    source: WgmmaSharedF16[BlockRows, BlockCols, Layout],
) -> None: ...
@overload
def async_store[
    Rows: IntVar,
    Cols: IntVar,
    BlockRows: IntVar,
    BlockCols: IntVar,
    Layout: IntVar,
](
    descriptor: TmaOutputDescriptor2D[Rows, Cols, BlockRows, BlockCols, Layout],
    coordinates: Sequence[GluonTileStart[BlockRows] | GluonTileStart[BlockCols] | int],
    source: TmaSharedTile2D[BlockRows, BlockCols, Layout],
) -> None: ...
@overload
def async_store[Length: IntVar, Block: IntVar, Layout: IntVar](
    descriptor: TmaOutputDescriptor1D[Length, Block, Layout],
    coordinates: list[GluonTileStart[Block]],
    source: TmaSharedBuffer1D[Block, Layout],
) -> None: ...
@overload
def async_store[Block: IntVar, Layout: IntVar](
    descriptor: MessageDescriptor1D[Block, Block, Layout],
    coordinates: list[Literal[0]],
    source: MessageSharedBuffer1D[Block, Layout],
) -> None: ...
def store_wait(pendings: int, read_only: bool = True) -> None: ...

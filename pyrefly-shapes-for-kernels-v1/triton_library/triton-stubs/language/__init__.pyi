# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Static-only semantic roles for a tile and the allocation it accesses."""

from collections.abc import Iterator
from typing import Literal, overload

from shape_extensions import Int, IntListLiteral, IntTuple, IntVar

from triton_library.tlt import InPointer, OutPointer

from . import extra, math

class tensor[Tile: IntTuple]:
    dtype: object
    shape: tuple[int, ...]
    def __neg__(self) -> tensor[Tile]: ...
    @property
    def T[Rows: IntVar, Cols: IntVar](
        self: tensor[[Rows, Cols]],
    ) -> tensor[[Cols, Rows]]: ...
    @overload
    def __getitem__[Rows: IntVar](
        self: tensor[[Rows]], index: tuple[slice, None]
    ) -> tensor[[Rows, 1]]: ...
    @overload
    def __getitem__[Cols: IntVar](
        self: tensor[[Cols]], index: tuple[None, slice]
    ) -> tensor[[1, Cols]]: ...
    @overload
    def __add__(self, other: tensor[Tile]) -> tensor[Tile]: ...
    @overload
    def __add__[Rows: IntVar, Cols: IntVar](
        self: tensor[[Rows, Cols]], other: tensor[[1, Cols]]
    ) -> tensor[[Rows, Cols]]: ...
    @overload
    def __add__(self, other: tensor[[]]) -> tensor[Tile]: ...
    @overload
    def __add__(self, other: float) -> tensor[Tile]: ...
    def __iadd__(self, other: tensor[Tile]) -> tensor[Tile]: ...
    def __rsub__(self, other: float) -> tensor[Tile]: ...
    @overload
    def __sub__(self, other: float) -> tensor[Tile]: ...
    @overload
    def __sub__(self, other: tensor[Tile]) -> tensor[Tile]: ...
    @overload
    def __sub__(self, other: tensor[[]]) -> tensor[Tile]: ...
    @overload
    def __sub__[Rows: IntVar, Cols: IntVar](
        self: tensor[[Rows, Cols]], other: tensor[[Rows, 1]]
    ) -> tensor[[Rows, Cols]]: ...
    @overload
    def __sub__[Rows: IntVar, Cols: IntVar](
        self: tensor[[Rows, Cols]], other: tensor[[1, Cols]]
    ) -> tensor[[Rows, Cols]]: ...
    @overload
    def __truediv__(self, other: tensor[[]]) -> tensor[Tile]: ...
    @overload
    def __truediv__(self, other: float) -> tensor[Tile]: ...
    def __rtruediv__(self, other: float) -> tensor[Tile]: ...
    def __eq__(self, other: object) -> bool: ...
    @overload
    def __mul__(self, other: tensor[Tile]) -> tensor[Tile]: ...
    @overload
    def __mul__(self, other: tensor[[]]) -> tensor[Tile]: ...
    @overload
    def __mul__(self, other: float) -> tensor[Tile]: ...
    @overload
    def __mul__[Rows: IntVar, Cols: IntVar](
        self: tensor[[Rows, Cols]], other: tensor[[Rows, 1]]
    ) -> tensor[[Rows, Cols]]: ...
    @overload
    def __mul__[Rows: IntVar, Cols: IntVar](
        self: tensor[[Rows, Cols]], other: tensor[[1, Cols]]
    ) -> tensor[[Rows, Cols]]: ...
    def __rmul__(self, other: float) -> tensor[Tile]: ...
    @overload
    def __truediv__[Rows: IntVar, Cols: IntVar](
        self: tensor[[Rows, Cols]], other: tensor[[Rows, 1]]
    ) -> tensor[[Rows, Cols]]: ...
    def __ge__(self, other: int) -> tensor[Tile]: ...
    def __gt__(self, other: float) -> tensor[Tile]: ...
    def to(self, dtype: object) -> tensor[Tile]: ...
    @overload
    def reshape[Rows: IntVar, K: IntVar](
        self: tensor[[1, Rows, K, 2, 256]],
        rows: Int[Rows],
        k: Int[K],
        d32: Literal[32],
        d4: Literal[4],
        d4b: Literal[4],
    ) -> tensor[[Rows, K, 32, 4, 4]]: ...
    @overload
    def reshape[Rows: IntVar, K: IntVar](
        self: tensor[[Rows, 4, 32, K, 4]],
        rows: Int[Rows * 128],
        k: Int[K * 4],
    ) -> tensor[[Rows * 128, K * 4]]: ...
    def trans[Rows: IntVar, K: IntVar](
        self: tensor[[Rows, K, 32, 4, 4]],
        a: Literal[0],
        b: Literal[3],
        c: Literal[2],
        d: Literal[1],
        e: Literal[4],
    ) -> tensor[[Rows, 4, 32, K, 4]]: ...

# Splitting a rank-two attention tile retains row and column sizes at each
# exact reshape/permute/split stage; this does not manufacture split tiles.
class AttentionSplitTile[Rows: IntVar, Cols: IntVar](tensor[[Rows, Cols]]):
    shape: tuple[Int[Rows], Int[Cols]]
    def reshape(
        self, shape: IntListLiteral[[Rows, 2, Cols // 2]]
    ) -> AttentionSplitReshaped[Rows, Cols]: ...

class AttentionSplitReshaped[Rows: IntVar, Cols: IntVar]:
    def permute(
        self, first: Literal[0], second: Literal[2], third: Literal[1]
    ) -> AttentionSplitPermuted[Rows, Cols]: ...

class AttentionSplitPermuted[Rows: IntVar, Cols: IntVar]:
    def split(
        self,
    ) -> tuple[
        AttentionSplitTile[Rows, Cols // 2], AttentionSplitTile[Rows, Cols // 2]
    ]: ...

class tensor_descriptor[
    Rows: IntVar,
    Cols: IntVar,
    Stride: IntVar,
    BlockRows: IntVar,
    BlockCols: IntVar,
]:
    def load(self, offsets: list[int]) -> tensor[[BlockRows, BlockCols]]: ...
    def store(
        self, offsets: list[int], value: tensor[[BlockRows, BlockCols]]
    ) -> None: ...

# The host allocation is [1, Rows // 128, K // VectorSize // 4, 2, 256].
class BlockScaleDescriptor[
    Rows: IntVar,
    K: IntVar,
    VectorSize: IntVar,
    TileRows: IntVar,
    TileK: IntVar,
]:
    def load(
        self, offsets: list[int | GroupStart[int, TileRows]]
    ) -> tensor[[1, TileRows, TileK, 2, 256]]: ...

# The host allocation is [Rows, K // ElementsPerByte].
class BlockDataDescriptor[
    Rows: IntVar,
    K: IntVar,
    ElementsPerByte: IntVar,
    TileRows: IntVar,
    TileK: IntVar,
    VectorSize: IntVar,
]:
    def load(
        self, offsets: list[int | GroupStart[int, TileRows * 128]]
    ) -> tensor[[TileRows * 128, TileK * 4 * VectorSize // ElementsPerByte]]: ...

class BlockOutputDescriptor[
    Rows: IntVar,
    Cols: IntVar,
    TileRows: IntVar,
    TileCols: IntVar,
]:
    def store(
        self,
        offsets: list[int | GroupStart[int, TileRows] | GroupStart[int, TileCols]],
        value: tensor[[TileRows, TileCols]],
    ) -> None: ...

# CDNA4 shuffled scales have host shape [Rows // 32, 2 * PackedK]. Their
# unmasked accesses use a logical row bound of Rows, not Rows // 32; the
# caller must establish physical bounds separately.
class CDNA4ScalePointer[
    Rows: IntVar,
    PackedK: IntVar,
    RowStride: IntVar,
    KStride: IntVar,
    BlockRows: IntVar,
    BlockK: IntVar,
]:
    def __add__(
        self, row: WrappedRowAddress[Rows, [BlockRows // 32], RowStride]
    ) -> CDNA4ScaleRows[Rows, PackedK, KStride, BlockRows, BlockK]: ...

class CDNA4ScaleRows[
    Rows: IntVar,
    PackedK: IntVar,
    KStride: IntVar,
    BlockRows: IntVar,
    BlockK: IntVar,
]:
    def __add__(
        self, k: ColumnAddress[[BlockK // 32 * 32], KStride]
    ) -> CDNA4ScaleTilePointers[Rows, PackedK, KStride, BlockRows, BlockK]: ...

class CDNA4ScaleTilePointers[
    Rows: IntVar,
    PackedK: IntVar,
    KStride: IntVar,
    BlockRows: IntVar,
    BlockK: IntVar,
]:
    def __iadd__(
        self, k: Int[BlockK * KStride]
    ) -> CDNA4ScaleTilePointers[Rows, PackedK, KStride, BlockRows, BlockK]: ...

class CDNA4ScaleTile[BlockRows: IntVar, BlockK: IntVar]:
    @overload
    def reshape(
        self,
        row_groups: Int[BlockRows // 32],
        k_groups: int,
        d2: Literal[2],
        d32: Literal[32],
        d4: Literal[4],
        d1: Literal[1],
    ) -> CDNA4ScaleMFMA32[BlockRows, BlockK]: ...
    @overload
    def reshape(
        self,
        row_groups: Int[BlockRows // 32],
        k_groups: int,
        d4: Literal[4],
        d16: Literal[16],
        d2a: Literal[2],
        d2b: Literal[2],
        d1: Literal[1],
    ) -> CDNA4ScaleMFMA16[BlockRows, BlockK]: ...

class CDNA4ScaleMFMA32[BlockRows: IntVar, BlockK: IntVar]:
    def permute(
        self,
        a: Literal[0],
        b: Literal[3],
        c: Literal[1],
        d: Literal[4],
        e: Literal[2],
        f: Literal[5],
    ) -> CDNA4ScaleReordered[BlockRows, BlockK]: ...

class CDNA4ScaleMFMA16[BlockRows: IntVar, BlockK: IntVar]:
    def permute(
        self,
        a: Literal[0],
        b: Literal[5],
        c: Literal[3],
        d: Literal[1],
        e: Literal[4],
        f: Literal[2],
        g: Literal[6],
    ) -> CDNA4ScaleReordered[BlockRows, BlockK]: ...

class CDNA4ScaleReordered[BlockRows: IntVar, BlockK: IntVar]:
    def reshape(
        self, rows: Int[BlockRows], scales: int
    ) -> tensor[[BlockRows, BlockK // 32]]: ...

# A host-constructed TMA descriptor promises its allocation and block geometry.
# These roles describe kernel arguments; their host origins remain unverified.
class InputMatrixDescriptor[
    Rows: IntVar,
    Cols: IntVar,
    Stride: IntVar,
    BlockRows: IntVar,
    BlockCols: IntVar,
]:
    def load(self, offsets: list[int]) -> tensor[[BlockRows, BlockCols]]: ...

class OutputMatrixDescriptor[
    Rows: IntVar,
    Cols: IntVar,
    Stride: IntVar,
    BlockRows: IntVar,
    BlockCols: IntVar,
]:
    def store(
        self, offsets: list[int], value: tensor[[BlockRows, BlockCols]]
    ) -> None: ...

class AttentionPointer[Rows: IntVar, Cols: IntVar, Stride: IntVar]: ...
class AttentionHeadCount[Heads: IntVar](Int[Heads]): ...
class AttentionBatchIndex[Heads: IntVar]: ...

class AttentionHeadStride[Heads: IntVar, StrideH: IntVar](Int[StrideH]):
    def __mul__(
        self, index: GroupIndex[Heads]
    ) -> AttentionHeadOffset[Heads, StrideH]: ...

class AttentionBatchStride[Heads: IntVar, StrideZ: IntVar](Int[StrideZ]):
    def __mul__(
        self, index: AttentionBatchIndex[Heads]
    ) -> AttentionBatchOffset[Heads, StrideZ]: ...

class AttentionHeadOffset[Heads: IntVar, StrideH: IntVar]:
    def __add__[StrideZ: IntVar](
        self, batch: AttentionBatchOffset[Heads, StrideZ]
    ) -> AttentionBatchHeadOffset[Heads, StrideZ, StrideH]: ...

class AttentionBatchOffset[Heads: IntVar, StrideZ: IntVar]: ...

class AttentionBatchHeadOffset[Heads: IntVar, StrideZ: IntVar, StrideH: IntVar]:
    def to(
        self, dtype: object
    ) -> AttentionBatchHeadOffset[Heads, StrideZ, StrideH]: ...

class AttentionHeadLocalOutputPointer[
    Tokens: IntVar,
    Dim: IntVar,
    TokenStride: IntVar,
    FeatureStride: IntVar,
]:
    def __add__[BM: IntVar](
        self, addresses: RowAddress[[BM], TokenStride]
    ) -> AttentionOutputRows[Tokens, Dim, FeatureStride, BM]: ...

class Attention4DStridedOutputPointer[
    Batch: IntVar,
    Heads: IntVar,
    Tokens: IntVar,
    Dim: IntVar,
    StrideZ: IntVar,
    StrideH: IntVar,
    TokenStride: IntVar,
    FeatureStride: IntVar,
](AttentionHeadLocalOutputPointer[Tokens, Dim, TokenStride, FeatureStride]):
    def __iadd__(
        self, offset: AttentionBatchHeadOffset[Heads, StrideZ, StrideH]
    ) -> Attention4DStridedOutputPointer[
        Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
    ]: ...

# Atomic attention output requires a distinct, initially zeroed host allocation.
class ZeroedAttention4DStridedOutputPointer[
    Batch: IntVar,
    Heads: IntVar,
    Tokens: IntVar,
    Dim: IntVar,
    StrideZ: IntVar,
    StrideH: IntVar,
    TokenStride: IntVar,
    FeatureStride: IntVar,
](
    Attention4DStridedOutputPointer[
        Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
    ]
):
    def __iadd__(
        self, offset: AttentionBatchHeadOffset[Heads, StrideZ, StrideH]
    ) -> ZeroedAttention4DStridedOutputPointer[
        Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
    ]: ...
    def __add__[BM: IntVar](
        self, addresses: RowAddress[[BM], TokenStride]
    ) -> ZeroedAttentionOutputRows[Tokens, Dim, FeatureStride, BM]: ...

class AttentionOutputRows[
    Tokens: IntVar,
    Dim: IntVar,
    FeatureStride: IntVar,
    BM: IntVar,
]:
    def __add__(
        self, addresses: ColumnAddress[[Dim], FeatureStride]
    ) -> AttentionOutputTilePointers[Tokens, Dim, BM]: ...

class AttentionOutputTilePointers[Tokens: IntVar, Dim: IntVar, BM: IntVar]: ...

class ZeroedAttentionOutputRows[
    Tokens: IntVar,
    Dim: IntVar,
    FeatureStride: IntVar,
    BM: IntVar,
](AttentionOutputRows[Tokens, Dim, FeatureStride, BM]):
    def __add__(
        self, addresses: ColumnAddress[[Dim], FeatureStride]
    ) -> ZeroedAttentionOutputTilePointers[Tokens, Dim, BM]: ...

class ZeroedAttentionOutputTilePointers[Tokens: IntVar, Dim: IntVar, BM: IntVar](
    AttentionOutputTilePointers[Tokens, Dim, BM]
): ...

class Attention4DInputPointer[
    Batch: IntVar,
    Heads: IntVar,
    Tokens: IntVar,
    Dim: IntVar,
]:
    def __add__(
        self, start: ScaledTileStart[Dim, Tokens]
    ) -> Attention4DHeadPointer[Batch, Heads, Tokens, Dim]: ...

class Attention4DHeadPointer[Batch: IntVar, Heads: IntVar, Tokens: IntVar, Dim: IntVar]:
    def __add__[BM: IntVar](
        self, rows: RowAddress[[BM], Dim]
    ) -> Attention4DRows[Batch, Heads, Tokens, Dim, BM]: ...

class Attention4DRows[
    Batch: IntVar,
    Heads: IntVar,
    Tokens: IntVar,
    Dim: IntVar,
    BM: IntVar,
]:
    def __add__(
        self, cols: ColumnAxisOffsets[[Dim]]
    ) -> Attention4DTilePointers[Batch, Heads, Tokens, Dim, BM]: ...

class Attention4DTilePointers[
    Batch: IntVar,
    Heads: IntVar,
    Tokens: IntVar,
    Dim: IntVar,
    BM: IntVar,
]: ...

class AttentionHeadLocalStatsPointer[Tokens: IntVar]:
    def __add__[BM: IntVar](
        self, offsets: Offsets[[BM]]
    ) -> AttentionHeadLocalStatsTilePointers[Tokens, BM]: ...

class Attention3DStatsPointer[Batch: IntVar, Heads: IntVar, Tokens: IntVar](
    AttentionHeadLocalStatsPointer[Tokens]
):
    def __iadd__[GridAxis: int](
        self, start: TileStart[[Tokens], GridAxis]
    ) -> Attention3DStatsPointer[Batch, Heads, Tokens]: ...
    @overload
    def __add__[GridAxis: int](
        self, start: TileStart[[Tokens], GridAxis]
    ) -> Attention3DHeadPointer[Batch, Heads, Tokens]: ...
    @overload
    def __add__[BM: IntVar](
        self, offsets: Offsets[[BM]]
    ) -> AttentionHeadLocalStatsTilePointers[Tokens, BM]: ...

class Attention3DHeadPointer[Batch: IntVar, Heads: IntVar, Tokens: IntVar]:
    def __add__[BM: IntVar](
        self, offsets: Offsets[[BM]]
    ) -> Attention3DTilePointers[Batch, Heads, Tokens, BM]: ...

class Attention3DTilePointers[
    Batch: IntVar,
    Heads: IntVar,
    Tokens: IntVar,
    BM: IntVar,
]: ...

class AttentionHeadLocalInputPointer[
    Tokens: IntVar,
    Dim: IntVar,
    TokenStride: IntVar,
    FeatureStride: IntVar,
]:
    @overload
    def __add__[BM: IntVar](
        self, addresses: ColumnAddress[[BM], TokenStride]
    ) -> AttentionTransposeColumns[Tokens, Dim, TokenStride, FeatureStride, BM]: ...
    @overload
    def __add__[BM: IntVar](
        self, addresses: RowAddress[[BM], TokenStride]
    ) -> AttentionForwardRows[Tokens, Dim, TokenStride, FeatureStride, BM]: ...

class Attention4DStridedInputPointer[
    Batch: IntVar,
    Heads: IntVar,
    Tokens: IntVar,
    Dim: IntVar,
    StrideZ: IntVar,
    StrideH: IntVar,
    TokenStride: IntVar,
    FeatureStride: IntVar,
](AttentionHeadLocalInputPointer[Tokens, Dim, TokenStride, FeatureStride]):
    def __iadd__(
        self, offset: AttentionBatchHeadOffset[Heads, StrideZ, StrideH]
    ) -> Attention4DStridedInputPointer[
        Batch, Heads, Tokens, Dim, StrideZ, StrideH, TokenStride, FeatureStride
    ]: ...

class AttentionTransposeColumns[
    Tokens: IntVar,
    Dim: IntVar,
    TokenStride: IntVar,
    FeatureStride: IntVar,
    BM: IntVar,
]:
    def __add__(
        self, addresses: RowAddress[[Dim], FeatureStride]
    ) -> AttentionQTransposeTilePointers[Tokens, Dim, TokenStride, BM]: ...

class AttentionForwardRows[
    Tokens: IntVar,
    Dim: IntVar,
    TokenStride: IntVar,
    FeatureStride: IntVar,
    BM: IntVar,
]:
    def __add__(
        self, addresses: ColumnAddress[[Dim], FeatureStride]
    ) -> AttentionDOTilePointers[Tokens, Dim, TokenStride, BM]: ...

class AttentionQTransposeTilePointers[
    Tokens: IntVar,
    Dim: IntVar,
    TokenStride: IntVar,
    BM: IntVar,
]:
    def __iadd__(
        self, step: Int[BM * TokenStride]
    ) -> AttentionQTransposeTilePointers[Tokens, Dim, TokenStride, BM]: ...

class AttentionDOTilePointers[
    Tokens: IntVar,
    Dim: IntVar,
    TokenStride: IntVar,
    BM: IntVar,
]:
    def __iadd__(
        self, step: Int[BM * TokenStride]
    ) -> AttentionDOTilePointers[Tokens, Dim, TokenStride, BM]: ...

class AttentionHeadLocalStatsTilePointers[Tokens: IntVar, BM: IntVar]: ...

class AttentionStatsPointer[Heads: IntVar, Tokens: IntVar]:
    @overload
    def __add__[GridAxis: int](
        self, offset: TileStart[[Tokens], GridAxis]
    ) -> AttentionStatsRow[Heads, Tokens]: ...
    @overload
    def __add__[Tile: IntTuple](
        self, offsets: Offsets[Tile]
    ) -> AttentionStatsTilePointers[Heads, Tokens, Tile]: ...

class AttentionStatsRow[Heads: IntVar, Tokens: IntVar]:
    def __add__[Tile: IntTuple, Origin: str, GridAxis: int](
        self, offsets: Offsets[Tile, 1, Origin, GridAxis]
    ) -> AttentionStatsTilePointers[Heads, Tokens, Tile]: ...

class AttentionStatsTilePointers[Heads: IntVar, Tokens: IntVar, Tile: IntTuple]: ...

@overload
def make_tensor_descriptor[
    Rows: IntVar,
    Cols: IntVar,
    Stride: IntVar,
    BR: IntVar,
    BC: IntVar,
](
    ptr: AttentionPointer[Rows, Cols, Stride],
    shape: list[int],
    strides: list[int],
    block_shape: list[int],
) -> tensor_descriptor[Rows, Cols, Stride, BR, BC]: ...
@overload
def make_tensor_descriptor[Groups: IntVar, BR: IntVar, BC: IntVar](
    ptr: GroupAMatrixPointer[Groups]
    | GroupBMatrixPointer[Groups]
    | GroupCMatrixPointer[Groups],
    shape: list[int],
    strides: list[int],
    block_shape: IntListLiteral[[BR, BC]],
) -> tensor_descriptor[int, int, int, BR, BC]: ...

class ProgramId[GridAxis: int = Literal[-1]]:
    def __mul__[Block: IntVar](
        self, block: Int[Block]
    ) -> TileStart[[Block], GridAxis]: ...
    def __add__(self, other: int) -> int: ...
    def __floordiv__(self, other: int) -> int: ...
    def __sub__(self, other: int) -> int: ...
    def __ge__(self, other: int) -> bool: ...
    def __lt__(self, other: int) -> bool: ...
    @overload
    def __mod__[Groups: IntVar](self, other: Int[Groups]) -> GroupIndex[Groups]: ...
    @overload
    def __mod__(self, other: int) -> int: ...

class AttentionBatchHeadProgramId(ProgramId[Literal[2]]):
    @overload
    def __floordiv__[Heads: IntVar](
        self, other: AttentionHeadCount[Heads]
    ) -> AttentionBatchIndex[Heads]: ...
    @overload
    def __floordiv__(self, other: int) -> int: ...

class GroupIndex[Groups: IntVar]:
    def __radd__(self, other: int) -> int: ...
    def __mul__[Cols: IntVar](self, n: Int[Cols]) -> GroupStart[Groups, Cols]: ...
    def __mod__(self, other: int) -> int: ...
    def __floordiv__(self, other: int) -> int: ...

class GroupStart[Groups: IntVar, Cols: IntVar]:
    def __radd__(self, other: int) -> int: ...
    def __add__[Tile: IntTuple](
        self, cols: Offsets[Tile]
    ) -> GroupOffsets[Groups, Cols, Tile]: ...

class GroupOffsets[Groups: IntVar, Cols: IntVar, Tile: IntTuple]:
    def __mod__[Dim: IntVar](self, bound: Int[Dim]) -> WrappedOffsets[Dim, Tile]: ...
    def __getitem__(
        self: GroupOffsets[int, Cols, [Cols]], index: tuple[None, slice]
    ) -> ColumnAxisOffsets[[Cols]]: ...

class GroupedScratchPointer[Groups: IntVar, Cols: IntVar, Tile: IntVar]:
    @overload
    def __add__(
        self, start: GroupStart[Groups, Cols]
    ) -> GroupedScratchRow[Groups, Cols, Tile]: ...
    @overload
    def __add__[TileRows: IntVar](
        self, address: GroupedMatrixOffsets[[TileRows], [Tile], Cols]
    ) -> GroupedScratchMatrixPointers[Groups, Cols, TileRows, Tile]: ...

class GroupedScratchRow[Groups: IntVar, Cols: IntVar, Tile: IntVar](
    GroupedScratchPointer[Groups, Cols, Tile]
):
    @overload
    def __add__(
        self, start: GroupStart[Groups, Cols]
    ) -> GroupedScratchRow[Groups, Cols, Tile]: ...
    @overload
    def __add__(
        self, cols: Offsets[[Tile]]
    ) -> GroupedScratchTile[Groups, Cols, Tile]: ...
    @overload
    def __add__[TileRows: IntVar](
        self, address: GroupedMatrixOffsets[[TileRows], [Tile], Cols]
    ) -> GroupedScratchMatrixPointers[Groups, Cols, TileRows, Tile]: ...

class GroupedScratchTile[Groups: IntVar, Cols: IntVar, Tile: IntVar](
    GroupedScratchRow[Groups, Cols, Tile]
): ...
class GroupedScratchMatrixPointers[
    Groups: IntVar,
    Cols: IntVar,
    TileRows: IntVar,
    TileCols: IntVar,
]: ...

class LockArrayPointer[Groups: IntVar, Capacity: IntVar]:
    def __iadd__(
        self, index: GroupIndex[Groups]
    ) -> LockArrayPointer[Groups, Capacity]: ...
    def __add__(self, span: Int[Groups]) -> CountPointer[Groups]: ...

class CountPointer[Groups: IntVar]: ...

class TileStart[Tile: IntTuple, GridAxis: int = Literal[-1]](int):
    def to(self, dtype: object) -> TileStart[Tile, GridAxis]: ...
    def __rsub__(self, other: int) -> int: ...
    def __radd__(self, other: int) -> int: ...
    def __mul__[HeadDim: IntVar, Tokens: IntVar](
        self: TileStart[[HeadDim], GridAxis], extent: Int[Tokens]
    ) -> ScaledTileStart[HeadDim, Tokens]: ...
    @overload
    def __add__(self, offset: int) -> int: ...
    @overload
    def __add__[Rows: IntTuple](
        self, offsets: RowAxisOffsets[Rows]
    ) -> RowAxisOffsets[Rows]: ...
    @overload
    def __add__[Cols: IntTuple](
        self, offsets: ColumnAxisOffsets[Cols]
    ) -> ColumnAxisOffsets[Cols]: ...
    @overload
    def __add__(
        self, offsets: Offsets[Tile, 1, Literal["local"], Literal[-1]]
    ) -> Offsets[Tile, 1, Literal["program"], GridAxis]: ...

class ScaledTileStart[HeadDim: IntVar, Tokens: IntVar]:
    def __iadd__(self, step: Int[HeadDim]) -> ScaledTileStart[HeadDim, Tokens]: ...
    # A loop-indexed offset loses its block/step relationship; the resulting
    # tile width and allocation mask are checked separately at load/store.
    def __add__(self, loop_offset: int) -> int: ...

class SplitAddress[Stride: IntVar]: ...

class SplitStride[Stride: IntVar](int):
    def __rmul__(self, index: int) -> SplitAddress[Stride]: ...

class Offsets[
    Tile: IntTuple,
    Stride: IntVar = 1,
    Origin: str = Literal["local"],
    GridAxis: int = Literal[-1],
]:
    def to(self, dtype: object) -> Offsets[Tile, Stride, Origin, GridAxis]: ...
    def __add__(self, value: int) -> tensor[Tile]: ...
    def __lt__[N: IntVar](
        self: Offsets[Tile, 1, Origin, GridAxis], bound: Int[N]
    ) -> Mask[[N], Tile, Origin, GridAxis]: ...
    def __radd__(self, start: int) -> Offsets[Tile, Stride, Literal["shifted"]]: ...
    def __mod__[N: IntVar](self, bound: Int[N]) -> WrappedOffsets[N, Tile]: ...
    def __mul__[Step: IntVar](
        self, stride: Int[Step]
    ) -> ScaledOffsets[Tile, Step, Origin, GridAxis]: ...
    @overload
    def __getitem__(
        self, index: tuple[slice, None]
    ) -> RowAxisOffsets[Tile, Stride]: ...
    @overload
    def __getitem__(
        self, index: tuple[None, slice]
    ) -> ColumnAxisOffsets[Tile, Stride]: ...

class WrappedOffsets[Dim: IntVar, Tile: IntTuple]:
    @overload
    def __getitem__(
        self, index: tuple[slice, None]
    ) -> WrappedRowAxisOffsets[Dim, Tile]: ...
    @overload
    def __getitem__(
        self, index: tuple[None, slice]
    ) -> WrappedColumnAxisOffsets[Dim, Tile]: ...

# Clamping with `where(offsets < bound, offsets, 0)` and wrapping with modulo
# each retain a bound, but are distinct transformations of the tile indices.
class ClampedOffsets[Dim: IntVar, Tile: IntTuple]:
    @overload
    def __getitem__(
        self, index: tuple[slice, None]
    ) -> ClampedRowAxisOffsets[Dim, Tile]: ...
    @overload
    def __getitem__(
        self, index: tuple[None, slice]
    ) -> ClampedColumnAxisOffsets[Dim, Tile]: ...

class RowAxisOffsets[Tile: IntTuple, Stride: IntVar = 1]:
    def __ge__[Rows: IntVar, Cols: IntVar](
        self: RowAxisOffsets[[Rows]], other: ColumnAxisOffsets[[Cols]]
    ) -> tensor[[Rows, Cols]]: ...
    def __lt__[Dim: IntVar](
        self: RowAxisOffsets[Tile, 1], bound: Int[Dim]
    ) -> RowMask[Dim, Tile]: ...
    def __mul__[Step: IntVar](
        self: RowAxisOffsets[Tile, 1], stride: Int[Step]
    ) -> RowAddress[Tile, Step]: ...
    def __rmul__[Step: IntVar](
        self: RowAxisOffsets[Tile, 1], stride: Int[Step]
    ) -> RowAddress[Tile, Step]: ...
    def __add__[Rows: IntVar, Cols: IntVar, Stride: IntVar](
        self: RowAxisOffsets[[Rows]], other: ColumnAddress[[Cols], Stride]
    ) -> ColumnMajorMatrixOffsets[Rows, Cols, Stride]: ...

class ColumnAxisOffsets[Tile: IntTuple, Stride: IntVar = 1]:
    def __ge__[Rows: IntVar, Cols: IntVar](
        self: ColumnAxisOffsets[[Cols]], other: RowAxisOffsets[[Rows]]
    ) -> tensor[[Rows, Cols]]: ...
    def __radd__(self, start: int) -> ColumnAxisOffsets[Tile, Stride]: ...
    def __lt__[Dim: IntVar](
        self: ColumnAxisOffsets[Tile, 1], bound: Int[Dim]
    ) -> ColumnMask[Dim, Tile]: ...
    def __mul__[Step: IntVar](
        self: ColumnAxisOffsets[Tile, 1], stride: Int[Step]
    ) -> ColumnAddress[Tile, Step]: ...
    def __rmul__[Step: IntVar](
        self: ColumnAxisOffsets[Tile, 1], stride: Int[Step]
    ) -> ColumnAddress[Tile, Step]: ...

class WrappedRowAxisOffsets[Dim: IntVar, Tile: IntTuple]:
    def __mul__[Stride: IntVar](
        self, stride: Int[Stride]
    ) -> WrappedRowAddress[Dim, Tile, Stride]: ...

class WrappedColumnAxisOffsets[Dim: IntVar, Tile: IntTuple]:
    def __mul__[Stride: IntVar](
        self, stride: Int[Stride]
    ) -> WrappedColumnAddress[Dim, Tile, Stride]: ...

class ClampedRowAxisOffsets[Dim: IntVar, Tile: IntTuple]:
    def __mul__[Stride: IntVar](
        self, stride: Int[Stride]
    ) -> ClampedRowAddress[Dim, Tile, Stride]: ...

class ClampedColumnAxisOffsets[Dim: IntVar, Tile: IntTuple]:
    def __mul__[Stride: IntVar](
        self, stride: Int[Stride]
    ) -> ClampedColumnAddress[Dim, Tile, Stride]: ...

class RowAddress[Tile: IntTuple, Stride: IntVar]:
    @overload
    def __add__[Cols: IntTuple](
        self, other: ColumnAxisOffsets[Cols]
    ) -> GroupedMatrixOffsets[Tile, Cols, Stride]: ...
    @overload
    def __add__[Dim: IntVar, Cols: IntTuple, ColumnStride: IntVar](
        self, other: WrappedColumnAddress[Dim, Cols, ColumnStride]
    ) -> WrappedColumnMatrixAddress[Dim, Tile, Cols, Stride, ColumnStride]: ...
    @overload
    def __add__[Dim: IntVar, Cols: IntTuple, ColumnStride: IntVar](
        self, other: ClampedColumnAddress[Dim, Cols, ColumnStride]
    ) -> ClampedColumnMatrixAddress[Dim, Tile, Cols, Stride, ColumnStride]: ...

class GroupedMatrixOffsets[TileRows: IntTuple, TileCols: IntTuple, Stride: IntVar]: ...
class ColumnAddress[Tile: IntTuple, Stride: IntVar]: ...
class ColumnMajorMatrixOffsets[Rows: IntVar, Cols: IntVar, Stride: IntVar]: ...
class ScaledOffsets[
    Tile: IntTuple,
    Stride: IntVar,
    Origin: str = Literal["local"],
    GridAxis: int = Literal[-1],
](
    Offsets[Tile, Stride, Origin, GridAxis]
): ...

# Grouped GEMM receives device arrays of raw addresses and packed metadata.
# Their common group count does not prove that entry g in each array agrees.
class GroupAPointers[Groups: IntVar]:
    def __add__(self, index: int) -> GroupASlot[Groups]: ...

class GroupBPointers[Groups: IntVar]:
    def __add__(self, index: int) -> GroupBSlot[Groups]: ...

class GroupCPointers[Groups: IntVar]:
    def __add__(self, index: int) -> GroupCSlot[Groups]: ...

class GroupSizes[Groups: IntVar]:
    def __add__(self, index: int) -> GroupMetadataSlot[Groups]: ...

class GroupLeadingDimensions[Groups: IntVar]:
    def __add__(self, index: int) -> GroupMetadataSlot[Groups]: ...

class GroupASlot[Groups: IntVar]: ...
class GroupBSlot[Groups: IntVar]: ...
class GroupCSlot[Groups: IntVar]: ...

class GroupMetadataSlot[Groups: IntVar]:
    def __add__(self, field: Literal[1, 2]) -> GroupMetadataSlot[Groups]: ...

class GroupAAddress[Groups: IntVar]:
    def to(self, dtype: object) -> GroupAMatrixPointer[Groups]: ...

class GroupBAddress[Groups: IntVar]:
    def to(self, dtype: object) -> GroupBMatrixPointer[Groups]: ...

class GroupCAddress[Groups: IntVar]:
    def to(self, dtype: object) -> GroupCMatrixPointer[Groups]: ...

class GroupAMatrixPointer[Groups: IntVar]:
    def __add__[M: IntVar, Stride: IntVar](
        self, offsets: RowAddress[[M], Stride]
    ) -> GroupARowPointers[M]: ...

class GroupBMatrixPointer[Groups: IntVar]:
    def __add__[K: IntVar, Stride: IntVar](
        self, offsets: RowAddress[[K], Stride]
    ) -> GroupBRowPointers[K]: ...

class GroupCMatrixPointer[Groups: IntVar]:
    def __add__[M: IntVar, Stride: IntVar](
        self, offsets: RowAddress[[M], Stride]
    ) -> GroupCRowPointers[M]: ...

class GroupARowPointers[M: IntVar]:
    def __add__[K: IntVar](
        self, columns: ColumnAxisOffsets[[K]]
    ) -> GroupATilePointers[M, K]: ...

class GroupBRowPointers[K: IntVar]:
    def __add__[N: IntVar](
        self, columns: ColumnAxisOffsets[[N]]
    ) -> GroupBTilePointers[K, N]: ...

class GroupCRowPointers[M: IntVar]:
    def __add__[N: IntVar](
        self, columns: ColumnAxisOffsets[[N]]
    ) -> GroupCTilePointers[M, N]: ...

class GroupATilePointers[M: IntVar, K: IntVar]:
    def __iadd__(self, step: Int[K]) -> GroupATilePointers[M, K]: ...

class GroupBTilePointers[K: IntVar, N: IntVar]:
    def __iadd__(self, step: int) -> GroupBTilePointers[K, N]: ...

class GroupCTilePointers[M: IntVar, N: IntVar]: ...

class WrappedRowAddress[Dim: IntVar, Tile: IntTuple, Stride: IntVar]:
    def __add__[Cols: IntTuple, ColumnStride: IntVar](
        self, other: ColumnAddress[Cols, ColumnStride]
    ) -> WrappedRowMatrixAddress[Dim, Tile, Cols, Stride, ColumnStride]: ...

class WrappedColumnAddress[Dim: IntVar, Tile: IntTuple, Stride: IntVar]: ...

class ClampedRowAddress[Dim: IntVar, Tile: IntTuple, Stride: IntVar]:
    def __add__[Cols: IntTuple, ColumnStride: IntVar](
        self, other: ColumnAddress[Cols, ColumnStride]
    ) -> ClampedRowMatrixAddress[Dim, Tile, Cols, Stride, ColumnStride]: ...

class ClampedColumnAddress[Dim: IntVar, Tile: IntTuple, Stride: IntVar]: ...
class ClampedRowMatrixAddress[
    Dim: IntVar,
    Rows: IntTuple,
    Cols: IntTuple,
    RowStride: IntVar,
    ColStride: IntVar,
]: ...
class ClampedColumnMatrixAddress[
    Dim: IntVar,
    Rows: IntTuple,
    Cols: IntTuple,
    RowStride: IntVar,
    ColStride: IntVar,
]: ...
class WrappedRowMatrixAddress[
    Dim: IntVar,
    Rows: IntTuple,
    Cols: IntTuple,
    RowStride: IntVar,
    ColStride: IntVar,
]: ...
class WrappedColumnMatrixAddress[
    Dim: IntVar,
    Rows: IntTuple,
    Cols: IntTuple,
    RowStride: IntVar,
    ColStride: IntVar,
]: ...

class RowMask[Dim: IntVar, Tile: IntTuple]:
    def __and__[Other: IntVar, OtherTile: IntTuple](
        self, other: ColumnMask[Other, OtherTile]
    ) -> MatrixMask[Dim, Other, Tile, OtherTile]: ...

class ColumnMask[Dim: IntVar, Tile: IntTuple]: ...
class MatrixMask[
    Rows: IntVar,
    Cols: IntVar,
    TileRows: IntTuple,
    TileCols: IntTuple,
]: ...

# The CDNA4 FP4 operand stores two logical K elements per packed byte.
class CDNA4PackedAPointer[
    Rows: IntVar,
    PackedK: IntVar,
    RowStride: IntVar,
    KStride: IntVar,
    BlockRows: IntVar,
    BlockK: IntVar,
]:
    def __add__(
        self,
        address: WrappedRowMatrixAddress[
            Rows, [BlockRows], [BlockK // 2], RowStride, KStride
        ],
    ) -> CDNA4PackedATilePointers[Rows, PackedK, BlockRows, BlockK, KStride]: ...

class CDNA4PackedBPointer[
    PackedK: IntVar,
    Cols: IntVar,
    KStride: IntVar,
    ColStride: IntVar,
    BlockCols: IntVar,
    BlockK: IntVar,
]:
    def __add__(
        self,
        address: WrappedColumnMatrixAddress[
            Cols, [BlockK // 2], [BlockCols], KStride, ColStride
        ],
    ) -> CDNA4PackedBTilePointers[Cols, PackedK, BlockCols, BlockK, KStride]: ...

class CDNA4PackedATilePointers[
    Rows: IntVar,
    PackedK: IntVar,
    BlockRows: IntVar,
    BlockK: IntVar,
    KStride: IntVar,
]:
    def __iadd__(
        self, step: Int[(BlockK // 2) * KStride]
    ) -> CDNA4PackedATilePointers[Rows, PackedK, BlockRows, BlockK, KStride]: ...

class CDNA4PackedBTilePointers[
    Cols: IntVar,
    PackedK: IntVar,
    BlockCols: IntVar,
    BlockK: IntVar,
    KStride: IntVar,
]:
    def __iadd__(
        self, step: Int[(BlockK // 2) * KStride]
    ) -> CDNA4PackedBTilePointers[Cols, PackedK, BlockCols, BlockK, KStride]: ...

class PointerDType:
    element_ty: object

class ClampedRowMatrixTilePointers[
    Rows: IntVar,
    Cols: IntVar,
    TileRows: IntVar,
    TileCols: IntVar,
    RowStride: IntVar,
    ColStride: IntVar,
]:
    def __iadd__(
        self, increment: Int[TileCols * ColStride]
    ) -> ClampedRowMatrixTilePointers[
        Rows, Cols, TileRows, TileCols, RowStride, ColStride
    ]: ...

class ClampedColumnMatrixTilePointers[
    Rows: IntVar,
    Cols: IntVar,
    TileRows: IntVar,
    TileCols: IntVar,
    RowStride: IntVar,
    ColStride: IntVar,
]:
    def __iadd__(
        self, increment: Int[TileRows * RowStride]
    ) -> ClampedColumnMatrixTilePointers[
        Rows, Cols, TileRows, TileCols, RowStride, ColStride
    ]: ...

class Scratch3DPointer[
    Split: IntVar,
    Rows: IntVar,
    Cols: IntVar,
    SplitStride: IntVar,
    RowStride: IntVar,
    ColStride: IntVar,
]:
    @overload
    def __add__(
        self, offset: TileStart[[SplitStride]]
    ) -> ScratchSlicePointer[Split, Rows, Cols, RowStride, ColStride]: ...
    @overload
    def __add__(
        self, offset: SplitAddress[SplitStride]
    ) -> ScratchSlicePointer[Split, Rows, Cols, RowStride, ColStride]: ...

class ScratchSlicePointer[
    Split: IntVar,
    Rows: IntVar,
    Cols: IntVar,
    RowStride: IntVar,
    ColStride: IntVar,
]:
    def __add__[TileRows: IntVar](
        self, address: RowAddress[[TileRows], RowStride]
    ) -> ScratchRows[Split, Rows, Cols, TileRows, ColStride]: ...

class ScratchRows[
    Split: IntVar,
    Rows: IntVar,
    Cols: IntVar,
    TileRows: IntVar,
    ColStride: IntVar,
]:
    def __add__[TileCols: IntVar](
        self, address: ColumnAddress[[TileCols], ColStride]
    ) -> ScratchTilePointers[Split, Rows, Cols, TileRows, TileCols]: ...

class ScratchTilePointers[
    Split: IntVar,
    Rows: IntVar,
    Cols: IntVar,
    TileRows: IntVar,
    TileCols: IntVar,
]: ...

class Mask[
    Target: IntTuple,
    Tile: IntTuple,
    Origin: str = Literal["local"],
    GridAxis: int = Literal[-1],
]:
    def __getitem__[Rows: IntVar, BR: IntVar](
        self: Mask[[Rows], [BR]], index: tuple[slice, None]
    ) -> RowMask[Rows, [BR]]: ...

class InScalarPointer[Target: IntTuple]: ...
class OutScalarPointer[Target: IntTuple]: ...

class InTilePointers[
    Target: IntTuple,
    Strides: IntTuple,
    Tile: IntTuple,
    Origin: str = Literal["local"],
    GridAxis: int = Literal[-1],
]:
    @overload
    def __iadd__[
        Rows: IntVar,
        Cols: IntVar,
        RS: IntVar,
        CS: IntVar,
        TileRows: IntVar,
        TileCols: IntVar,
    ](
        self: InTilePointers[
            [Rows, Cols], [RS, CS], [TileRows, TileCols], Literal["wrapped_0"], GridAxis
        ],
        step: Int[TileCols * CS],
    ) -> InTilePointers[
        [Rows, Cols], [RS, CS], [TileRows, TileCols], Literal["wrapped_0"], GridAxis
    ]: ...
    @overload
    def __iadd__[
        Rows: IntVar,
        Cols: IntVar,
        RS: IntVar,
        CS: IntVar,
        TileRows: IntVar,
        TileCols: IntVar,
    ](
        self: InTilePointers[
            [Rows, Cols], [RS, CS], [TileRows, TileCols], Literal["wrapped_1"], GridAxis
        ],
        step: Int[TileRows * RS],
    ) -> InTilePointers[
        [Rows, Cols], [RS, CS], [TileRows, TileCols], Literal["wrapped_1"], GridAxis
    ]: ...

class InColumnTilePointers[Cols: IntVar, BC: IntVar]: ...

class OutTilePointers[
    Target: IntTuple,
    Strides: IntTuple,
    Tile: IntTuple,
    Origin: str = Literal["local"],
    GridAxis: int = Literal[-1],
]:
    def __add__[
        Rows: IntVar,
        Cols: IntVar,
        RS: IntVar,
        CS: IntVar,
        TileRows: IntVar,
        TileCols: IntVar,
    ](
        self: OutTilePointers[[Rows, Cols], [RS, CS], [TileRows, 1], Literal["axis_0"]],
        address: ColumnAddress[[TileCols], CS],
    ) -> OutTilePointers[
        [Rows, Cols], [RS, CS], [TileRows, TileCols], Literal["indexed"]
    ]: ...

# The unmasked transpose uses contiguous input [M, N] and output [N, M].
# These roles retain shapes and strides but do not establish address bounds.
class TransposeInputPointer[M: IntVar, N: IntVar]:
    def __add__(self, address: RowAddress[[M], N]) -> TransposeInputRows[M, N]: ...

class TransposeOutputPointer[M: IntVar, N: IntVar]:
    def __add__(self, address: RowAddress[[N], M]) -> TransposeOutputRows[M, N]: ...

class TransposeInputRows[M: IntVar, N: IntVar]:
    def __add__(self, columns: ColumnAxisOffsets[[N]]) -> TransposeInputTile[M, N]: ...

class TransposeOutputRows[M: IntVar, N: IntVar]:
    def __add__(self, columns: ColumnAxisOffsets[[M]]) -> TransposeOutputTile[M, N]: ...

class TransposeInputTile[M: IntVar, N: IntVar]: ...
class TransposeOutputTile[M: IntVar, N: IntVar]: ...

# The software-pipelining matmul uses unmasked row-major matrix tiles. Shape
# and row-stride identities are checked, but numerical K bounds are not.
class PipelineMatrixInputPointer[Rows: IntVar, Cols: IntVar, BR: IntVar, BC: IntVar]:
    def __add__(
        self, address: RowAddress[[BR], Cols]
    ) -> PipelineMatrixInputRows[Rows, Cols, BR, BC]: ...

class PipelineMatrixOutputPointer[Rows: IntVar, Cols: IntVar, BR: IntVar, BC: IntVar]:
    def __add__(
        self, address: RowAddress[[BR], Cols]
    ) -> PipelineMatrixOutputRows[Rows, Cols, BR, BC]: ...

class PipelineMatrixInputRows[Rows: IntVar, Cols: IntVar, BR: IntVar, BC: IntVar]:
    def __add__(
        self, columns: ColumnAxisOffsets[[BC]]
    ) -> PipelineMatrixInputTile[Rows, Cols, BR, BC]: ...

class PipelineMatrixOutputRows[Rows: IntVar, Cols: IntVar, BR: IntVar, BC: IntVar]:
    def __add__(
        self, columns: ColumnAxisOffsets[[BC]]
    ) -> PipelineMatrixOutputTile[Rows, Cols, BR, BC]: ...

class PipelineMatrixInputTile[Rows: IntVar, Cols: IntVar, BR: IntVar, BC: IntVar]: ...
class PipelineMatrixOutputTile[Rows: IntVar, Cols: IntVar, BR: IntVar, BC: IntVar]: ...

# Compilation tutorial 11 copies complete unmasked allocations in one program.
# A tile's declared extent is checked here; physical capacity remains a host check.
class FullCopyInputPointer[Length: IntVar]:
    def __add__(self, offsets: Offsets[[Length]]) -> FullCopyInputTile[Length]: ...

class FullCopyOutputPointer[Length: IntVar]:
    def __add__(self, offsets: Offsets[[Length]]) -> FullCopyOutputTile[Length]: ...

class FullCopyInputTile[Length: IntVar]: ...
class FullCopyOutputTile[Length: IntVar]: ...

class RowMajorCopyInputPointer[Rows: IntVar, Cols: IntVar]:
    def __add__(
        self, address: GroupedMatrixOffsets[[Rows], [Cols], Cols]
    ) -> CopyMatrixInputTile[Rows, Cols]: ...

class RowMajorCopyOutputPointer[Rows: IntVar, Cols: IntVar]:
    def __add__(
        self, address: GroupedMatrixOffsets[[Rows], [Cols], Cols]
    ) -> CopyMatrixOutputTile[Rows, Cols]: ...

class ColumnMajorCopyInputPointer[Rows: IntVar, Cols: IntVar]:
    def __add__(
        self, address: ColumnMajorMatrixOffsets[Rows, Cols, Rows]
    ) -> CopyMatrixInputTile[Rows, Cols]: ...

class ColumnMajorCopyOutputPointer[Rows: IntVar, Cols: IntVar]:
    def __add__(
        self, address: ColumnMajorMatrixOffsets[Rows, Cols, Rows]
    ) -> CopyMatrixOutputTile[Rows, Cols]: ...

class CopyMatrixInputTile[Rows: IntVar, Cols: IntVar]: ...
class CopyMatrixOutputTile[Rows: IntVar, Cols: IntVar]: ...

class StridedCopyInputPointer[Source: IntVar, Length: IntVar, Stride: IntVar]:
    def __add__(
        self, offsets: ScaledOffsets[[Length], Stride]
    ) -> FullCopyInputTile[Length]: ...

# The CPU padding tutorial performs unmasked accesses. These roles check tile
# widths and read/write direction, but cannot prove address bounds or strides.
class CPUPadInputPointer[Rows: IntVar, Cols: IntVar, Block: IntVar]:
    type: PointerDType
    @overload
    def __add__[BlockRows: IntVar](
        self, offset: ScaledTileStart[Cols, BlockRows]
    ) -> CPUPadInputPointer[Rows, Cols, Block]: ...
    @overload
    def __add__(self, offset: int) -> CPUPadInputPointer[Rows, Cols, Block]: ...
    @overload
    def __add__(self, offsets: Offsets[[Block]]) -> CPUPadInputTile[Block]: ...

class CPUPadOutputPointer[Rows: IntVar, Cols: IntVar]:
    @overload
    def __add__[BlockRows: IntVar](
        self, offset: ScaledTileStart[Cols, BlockRows]
    ) -> CPUPadOutputPointer[Rows, Cols]: ...
    @overload
    def __add__(self, offset: int) -> CPUPadOutputPointer[Rows, Cols]: ...
    @overload
    def __add__[Block: IntVar](
        self, offsets: Offsets[[Block]]
    ) -> CPUPadOutputTile[Block]: ...

class CPUPadInputTile[Block: IntVar]: ...
class CPUPadOutputTile[Block: IntVar]: ...

# GEMV's unmasked tiles retain their allocation shapes and row stride, while
# divisibility, address bounds, and launch coverage remain caller obligations.
class CPUGemvMatrixPointer[
    Rows: IntVar,
    Cols: IntVar,
    Stride: IntVar,
    BlockRows: IntVar,
    BlockCols: IntVar,
]:
    def __add__(
        self: CPUGemvMatrixPointer[Rows, Cols, Stride, BlockRows, BlockCols],
        offsets: GroupedMatrixOffsets[[BlockRows], [BlockCols], Stride],
    ) -> CPUGemvMatrixPointer[Rows, Cols, Stride, BlockRows, BlockCols]: ...
    def __iadd__(
        self, step: Int[BlockCols]
    ) -> CPUGemvMatrixPointer[Rows, Cols, Stride, BlockRows, BlockCols]: ...

class CPUGemvVectorPointer[Cols: IntVar, BlockCols: IntVar]:
    def __add__(
        self, offsets: Offsets[[BlockCols]]
    ) -> CPUGemvVectorPointer[Cols, BlockCols]: ...
    def __iadd__(
        self, step: Int[BlockCols]
    ) -> CPUGemvVectorPointer[Cols, BlockCols]: ...

class CPUGemvOutputPointer[Rows: IntVar, BlockRows: IntVar]:
    def __add__(
        self, offsets: Offsets[[BlockRows]]
    ) -> CPUGemvOutputPointer[Rows, BlockRows]: ...

class In2DRowMajorPointer[Rows: IntVar, Cols: IntVar, Stride: IntVar, BM: IntVar]:
    def __iadd__(
        self, address: RowAddress[[BM], Stride]
    ) -> In2DRowMajorPointer[Rows, Cols, Stride, BM]: ...
    def __add__[BN: IntVar](
        self, offsets: ColumnAxisOffsets[[BN]]
    ) -> In2DRowTilePointers[Rows, Cols, BM, BN]: ...

class Out2DRowMajorPointer[Rows: IntVar, Cols: IntVar, Stride: IntVar, BM: IntVar]:
    def __iadd__(
        self, address: RowAddress[[BM], Stride]
    ) -> Out2DRowMajorPointer[Rows, Cols, Stride, BM]: ...
    def __add__[BN: IntVar](
        self, offsets: ColumnAxisOffsets[[BN]]
    ) -> Out2DRowTilePointers[Rows, Cols, BM, BN]: ...

class In2DRowTilePointers[Rows: IntVar, Cols: IntVar, BM: IntVar, BN: IntVar]: ...
class Out2DRowTilePointers[Rows: IntVar, Cols: IntVar, BM: IntVar, BN: IntVar]: ...
class RowOffset[Rows: IntVar, Stride: IntVar]: ...

class RowIndex[Rows: IntVar]:
    def __mul__[Stride: IntVar](
        self, stride: Int[Stride]
    ) -> RowOffset[Rows, Stride]: ...

class RowRange[Rows: IntVar]:
    def __iter__(self) -> Iterator[RowIndex[Rows]]: ...

@overload
def program_id(axis: Literal[2]) -> AttentionBatchHeadProgramId: ...
@overload
def program_id(axis: Literal[0]) -> ProgramId[Literal[0]]: ...
@overload
def program_id(axis: Literal[1]) -> ProgramId[Literal[1]]: ...
@overload
def program_id(axis: int) -> ProgramId: ...
def num_programs(axis: int) -> ProgramCount: ...

class ProgramCount(int): ...

@overload
def range[Rows: IntVar, GridAxis: int](
    start: ProgramId[GridAxis], stop: Int[Rows], step: ProgramCount, *, num_stages: int
) -> RowRange[Rows]: ...
@overload
def range(start: Literal[0], stop: int) -> Iterator[int]: ...
@overload
def range[Steps: IntVar](
    start: Literal[0], stop: Int[Steps], *, num_stages: Literal[1, 2]
) -> Iterator[int]: ...
@overload
def range(
    start: int | TileStart[[int]],
    stop: int | TileStart[[int]],
    step: int,
    *,
    warp_specialize: bool,
) -> Iterator[int]: ...
@overload
def range(stop: int, *, warp_specialize: bool) -> Iterator[int]: ...
@overload
def range(start: Literal[0], stop: int, *, warp_specialize: bool) -> Iterator[int]: ...
@overload
def range[Cols: IntVar, Block: IntVar](
    start: Literal[0], stop: Int[Cols], step: Int[Block], *, multi_cta: bool
) -> Iterator[int]: ...
@overload
def range[SMs: IntVar, GridAxis: int](
    start: ProgramId[GridAxis], stop: int, step: Int[SMs], *, flatten: Literal[True]
) -> Iterator[int]: ...
@overload
def range(
    start: int, stop: int, *, num_stages: int, disallow_acc_multi_buffer: bool
) -> Iterator[int]: ...
@overload
def range(
    start: int,
    stop: int,
    step: int,
    *,
    warp_specialize: bool,
    disallow_acc_multi_buffer: Literal[True],
    data_partition_factor: Literal[2],
) -> Iterator[int]: ...
@overload
def range(
    start: int,
    stop: int,
    step: int,
    *,
    warp_specialize: bool,
    merge_epilogue: Literal[True],
    merge_correction: Literal[True],
    data_partition_factor: Literal[2],
) -> Iterator[int]: ...
@overload
def range(
    start: Literal[0],
    stop: int,
    *,
    warp_specialize: bool,
    merge_epilogue: Literal[True],
    merge_correction: Literal[True],
    data_partition_factor: Literal[2],
) -> Iterator[int]: ...
@overload
def range(
    start: int,
    stop: int,
    step: int,
    *,
    warp_specialize: bool,
    disallow_acc_multi_buffer: bool,
) -> Iterator[int]: ...
@overload
def range[SMs: IntVar, GridAxis: int](
    start: ProgramId[GridAxis],
    stop: int,
    step: Int[SMs],
    *,
    flatten: Literal[True],
    warp_specialize: bool,
) -> Iterator[int]: ...
@overload
def range[SMs: IntVar, GridAxis: int](
    start: ProgramId[GridAxis],
    stop: int,
    step: Int[SMs],
    *,
    flatten: bool,
    warp_specialize: bool,
) -> Iterator[int]: ...
@overload
def range[SMs: IntVar, GridAxis: int](
    start: ProgramId[GridAxis],
    stop: int,
    step: Int[SMs],
    *,
    flatten: bool,
    warp_specialize: Literal[True],
    disallow_acc_multi_buffer: Literal[True],
    separate_epilogue_store: Literal[True],
) -> Iterator[int]: ...
@overload
def arange[Block: IntVar](start: Literal[0], end: Int[Block]) -> Offsets[[Block]]: ...
@overload
def arange[Start: IntVar, End: IntVar](
    start: Int[Start], end: Int[End]
) -> Offsets[[End - Start]]: ...
def cdiv(value: int, block: int) -> int: ...
def assume(predicate: bool) -> None: ...
def static_assert(predicate: bool) -> None: ...
@overload
def multiple_of(value: int, block: int) -> int: ...
@overload
def multiple_of[Tile: IntTuple](value: TileStart[Tile], block: int) -> int: ...
@overload
def multiple_of[Dim: IntVar, Tile: IntTuple](
    value: ClampedOffsets[Dim, Tile], block: int
) -> ClampedOffsets[Dim, Tile]: ...
def max_contiguous[Dim: IntVar, Tile: IntTuple](
    value: ClampedOffsets[Dim, Tile], block: int
) -> ClampedOffsets[Dim, Tile]: ...
@overload
def multiple_of[M: IntVar, K: IntVar](
    value: GroupATilePointers[M, K], block: list[int]
) -> GroupATilePointers[M, K]: ...
@overload
def multiple_of[K: IntVar, N: IntVar](
    value: GroupBTilePointers[K, N], block: list[int]
) -> GroupBTilePointers[K, N]: ...
def pointer_type(dtype: object) -> object: ...
@overload
def zeros[Rows: IntVar, Cols: IntVar](
    shape: tuple[Int[Rows], Int[Cols]], *, dtype: object
) -> tensor[[Rows, Cols]]: ...
@overload
def zeros[Block: IntVar](
    shape: tuple[Int[Block]], *, dtype: object
) -> tensor[[Block]]: ...
def full[Block: IntVar](
    shape: tuple[Int[Block]], value: int, *, dtype: object
) -> tensor[[Block]]: ...
@overload
def zeros[Block: IntVar](
    shape: list[Int[Block]], *, dtype: object
) -> tensor[[Block]]: ...
@overload
def zeros[Rows: IntVar, Cols: IntVar](
    shape: IntListLiteral[[Rows, Cols]], *, dtype: object
) -> tensor[[Rows, Cols]]: ...
def reshape[Rows: IntVar, Cols: IntVar](
    value: tensor[[Rows, Cols]],
    shape: tuple[Int[Rows], Literal[2], Int[Cols // 2]],
) -> tensor[[Rows, 2, Cols // 2]]: ...
def permute[Rows: IntVar, Half: IntVar](
    value: tensor[[Rows, 2, Half]], order: tuple[Literal[0], Literal[2], Literal[1]]
) -> tensor[[Rows, Half, 2]]: ...
def split[Rows: IntVar, Half: IntVar](
    value: tensor[[Rows, Half, 2]],
) -> tuple[tensor[[Rows, Half]], tensor[[Rows, Half]]]: ...
@overload
def dot[M: IntVar, K: IntVar, N: IntVar](
    a: tensor[[M, K]], b: tensor[[K, N]]
) -> tensor[[M, N]]: ...
@overload
def dot[M: IntVar, K: IntVar, N: IntVar](
    a: tensor[[M, K]], b: tensor[[K, N]], *, input_precision: Literal["ieee", "tf32"]
) -> tensor[[M, N]]: ...
@overload
def dot[M: IntVar, K: IntVar, N: IntVar](
    a: tensor[[M, K]], b: tensor[[K, N]], acc: tensor[[M, N]]
) -> tensor[[M, N]]: ...
@overload
def dot[M: IntVar, K: IntVar, N: IntVar](
    a: tensor[[M, K]], b: tensor[[K, N]], *, attrs: object | None
) -> tensor[[M, N]]: ...
@overload
def dot[M: IntVar, K: IntVar, N: IntVar](
    a: tensor[[M, K]], b: tensor[[K, N]], acc: tensor[[M, N]], *, attrs: object | None
) -> tensor[[M, N]]: ...
def trans[Rows: IntVar, Cols: IntVar](
    value: tensor[[Rows, Cols]],
) -> tensor[[Cols, Rows]]: ...
@overload
def dot_scaled[M: IntVar, N: IntVar, KA: IntVar, KB: IntVar, ScaleK: IntVar](
    a: tensor[[M, KA]],
    scale_a: tensor[[M, ScaleK]],
    format_a: str,
    b: tensor[[KB, N]],
    scale_b: tensor[[N, ScaleK]],
    format_b: str,
) -> tensor[[M, N]]: ...
@overload
def dot_scaled[
    M: IntVar,
    N: IntVar,
    KA: IntVar,
    KB: IntVar,
    ScaleK: IntVar,
](
    a: tensor[[M, KA]],
    scale_a: tensor[[M, ScaleK]],
    format_a: str,
    b: tensor[[KB, N]],
    scale_b: tensor[[N, ScaleK]],
    format_b: str,
    acc: tensor[[M, N]],
) -> tensor[[M, N]]: ...
@overload
def where[Tile: IntTuple](
    condition: tensor[Tile], x: tensor[Tile], y: tensor[Tile]
) -> tensor[Tile]: ...
@overload
def where[Tile: IntTuple](
    condition: tensor[Tile], x: tensor[Tile], y: float
) -> tensor[Tile]: ...
@overload
def where[Target: IntTuple, Tile: IntTuple, Origin: str](
    condition: Mask[Target, Tile, Origin], x: tensor[Tile], y: float
) -> tensor[Tile]: ...
@overload
def where[Rows: IntVar, Cols: IntVar, BM: IntVar, BN: IntVar](
    condition: MatrixMask[Rows, Cols, [BM], [BN]],
    x: tensor[[BM, BN]],
    y: float,
) -> tensor[[BM, BN]]: ...
@overload
def where[Dim: IntVar, Tile: IntTuple](
    condition: Mask[[Dim], Tile], x: Offsets[Tile], y: Literal[0]
) -> ClampedOffsets[Dim, Tile]: ...
@overload
def where[Tile: IntTuple](
    condition: tensor[Tile], x: float, y: float
) -> tensor[Tile]: ...
def rand[Tile: IntTuple, Origin: str, GridAxis: int](
    seed: int, offsets: Offsets[Tile, 1, Origin, GridAxis]
) -> tensor[Tile]: ...
def sqrt[Tile: IntTuple](value: tensor[Tile]) -> tensor[Tile]: ...
def atomic_cas[Groups: IntVar, Capacity: IntVar](
    ptr: LockArrayPointer[Groups, Capacity], old: int, new: int
) -> int: ...
@overload
def atomic_xchg[Groups: IntVar, Capacity: IntVar](
    ptr: LockArrayPointer[Groups, Capacity], new: int
) -> None: ...
@overload
def atomic_xchg[Groups: IntVar](ptr: CountPointer[Groups], new: int) -> None: ...
def debug_barrier() -> None: ...

float16: object
float32: object
bfloat16: object
float8e5: object
float8e4nv: object
int64: object
constexpr = int

def maximum[Tile: IntTuple](a: tensor[Tile], b: tensor[Tile]) -> tensor[Tile]: ...
@overload
def load[Target: IntTuple](ptr: InScalarPointer[Target]) -> tensor[[]]: ...
@overload
def load[M: IntVar, N: IntVar](ptr: TransposeInputTile[M, N]) -> tensor[[M, N]]: ...
@overload
def load[Length: IntVar](ptr: FullCopyInputTile[Length]) -> tensor[[Length]]: ...
@overload
def load[Rows: IntVar, Cols: IntVar](
    ptr: CopyMatrixInputTile[Rows, Cols],
) -> tensor[[Rows, Cols]]: ...
@overload
def load[Rows: IntVar, Cols: IntVar, BR: IntVar, BC: IntVar](
    ptr: PipelineMatrixInputTile[Rows, Cols, BR, BC],
) -> tensor[[BR, BC]]: ...
@overload
def load[Block: IntVar](ptr: CPUPadInputTile[Block]) -> tensor[[Block]]: ...
@overload
def load[Rows: IntVar, Cols: IntVar, Stride: IntVar, BM: IntVar, BN: IntVar](
    ptr: CPUGemvMatrixPointer[Rows, Cols, Stride, BM, BN],
) -> tensor[[BM, BN]]: ...
@overload
def load[Cols: IntVar, BN: IntVar](
    ptr: CPUGemvVectorPointer[Cols, BN],
) -> tensor[[BN]]: ...
@overload
def load[Groups: IntVar](ptr: GroupASlot[Groups]) -> GroupAAddress[Groups]: ...
@overload
def load[Groups: IntVar](ptr: GroupBSlot[Groups]) -> GroupBAddress[Groups]: ...
@overload
def load[Groups: IntVar](ptr: GroupCSlot[Groups]) -> GroupCAddress[Groups]: ...
@overload
def load[Groups: IntVar](ptr: GroupMetadataSlot[Groups]) -> int: ...
@overload
def load[M: IntVar, K: IntVar](ptr: GroupATilePointers[M, K]) -> tensor[[M, K]]: ...
@overload
def load[K: IntVar, N: IntVar](ptr: GroupBTilePointers[K, N]) -> tensor[[K, N]]: ...
@overload
def load[M: IntVar, K: IntVar, BM: IntVar, BK: IntVar, KS: IntVar](
    ptr: CDNA4PackedATilePointers[M, K, BM, BK, KS],
    *,
    cache_modifier: None = None,
) -> tensor[[BM, BK // 2]]: ...
@overload
def load[N: IntVar, K: IntVar, BN: IntVar, BK: IntVar, KS: IntVar](
    ptr: CDNA4PackedBTilePointers[N, K, BN, BK, KS],
    *,
    cache_modifier: None = None,
) -> tensor[[BK // 2, BN]]: ...
@overload
def load[R: IntVar, K: IntVar, S: IntVar, BM: IntVar, BK: IntVar](
    ptr: CDNA4ScaleTilePointers[R, K, S, BM, BK],
) -> CDNA4ScaleTile[BM, BK]: ...
@overload
def load[Batch: IntVar, Heads: IntVar, Tokens: IntVar, Dim: IntVar, BM: IntVar](
    ptrs: Attention4DTilePointers[Batch, Heads, Tokens, Dim, BM],
) -> tensor[[BM, Dim]]: ...
@overload
def load[Tokens: IntVar, Dim: IntVar, Stride: IntVar, BM: IntVar](
    ptrs: AttentionQTransposeTilePointers[Tokens, Dim, Stride, BM],
) -> tensor[[Dim, BM]]: ...
@overload
def load[Tokens: IntVar, Dim: IntVar, Stride: IntVar, BM: IntVar](
    ptrs: AttentionDOTilePointers[Tokens, Dim, Stride, BM],
) -> tensor[[BM, Dim]]: ...
@overload
def load[Tokens: IntVar, BM: IntVar](
    ptrs: AttentionHeadLocalStatsTilePointers[Tokens, BM],
) -> tensor[[BM]]: ...
@overload
def load[Groups: IntVar](ptr: CountPointer[Groups]) -> tensor[[]]: ...
@overload
def load[
    Target: IntTuple, Strides: IntTuple, Tile: IntTuple, Origin: str, GridAxis: int
](
    ptrs: InTilePointers[Target, Strides, Tile, Origin, GridAxis],
    mask: Mask[Target, Tile, Origin, GridAxis],
) -> tensor[Tile]: ...
@overload
def load[
    Target: IntTuple, Strides: IntTuple, Tile: IntTuple, Origin: str, GridAxis: int
](
    ptrs: InTilePointers[Target, Strides, Tile, Origin, GridAxis],
    mask: Mask[Target, Tile, Origin, GridAxis],
    *,
    other: float,
) -> tensor[Tile]: ...
@overload
def load[
    Rows: IntVar,
    Cols: IntVar,
    RS: IntVar,
    CS: IntVar,
    TileRows: IntVar,
    TileCols: IntVar,
](
    ptrs: InTilePointers[
        [Rows, Cols], [RS, CS], [TileRows, TileCols], Literal["wrapped_0"]
    ],
    mask: ColumnMask[Cols, [TileCols]],
    *,
    other: float,
) -> tensor[[TileRows, TileCols]]: ...
@overload
def load[
    Rows: IntVar,
    Cols: IntVar,
    RS: IntVar,
    CS: IntVar,
    TileRows: IntVar,
    TileCols: IntVar,
](
    ptrs: InTilePointers[
        [Rows, Cols], [RS, CS], [TileRows, TileCols], Literal["wrapped_1"]
    ],
    mask: RowMask[Rows, [TileRows]],
    *,
    other: float,
) -> tensor[[TileRows, TileCols]]: ...
@overload
def load[Cols: IntVar, BN: IntVar](
    ptrs: InColumnTilePointers[Cols, BN], mask: ColumnMask[Cols, [BN]]
) -> tensor[[1, BN]]: ...
@overload
def load[Rows: IntVar, Cols: IntVar, BM: IntVar, BN: IntVar](
    ptrs: In2DRowTilePointers[Rows, Cols, BM, BN],
    mask: MatrixMask[Rows, Cols, [BM], [BN]],
    *,
    other: float,
) -> tensor[[BM, BN]]: ...
@overload
def load[Groups: IntVar, Cols: IntVar, Tile: IntVar](
    ptrs: GroupedScratchTile[Groups, Cols, Tile],
    mask: Mask[[Cols], [Tile]],
) -> tensor[[Tile]]: ...
@overload
def load[Groups: IntVar, Cols: IntVar, TileRows: IntVar, TileCols: IntVar](
    ptrs: GroupedScratchMatrixPointers[Groups, Cols, TileRows, TileCols],
    mask: MatrixMask[Groups, Cols, [TileRows], [TileCols]],
    *,
    other: float,
) -> tensor[[TileRows, TileCols]]: ...
@overload
def load[
    Rows: IntVar,
    Cols: IntVar,
    TileRows: IntVar,
    TileCols: IntVar,
    RS: IntVar,
    CS: IntVar,
](
    ptrs: ClampedRowMatrixTilePointers[Rows, Cols, TileRows, TileCols, RS, CS],
    mask: ColumnMask[Cols, [TileCols]],
    *,
    other: float,
) -> tensor[[TileRows, TileCols]]: ...
@overload
def load[
    Rows: IntVar,
    Cols: IntVar,
    TileRows: IntVar,
    TileCols: IntVar,
    RS: IntVar,
    CS: IntVar,
](
    ptrs: ClampedColumnMatrixTilePointers[Rows, Cols, TileRows, TileCols, RS, CS],
    mask: RowMask[Rows, [TileRows]],
    *,
    other: float,
) -> tensor[[TileRows, TileCols]]: ...
@overload
def load[
    Split: IntVar,
    Rows: IntVar,
    Cols: IntVar,
    TileRows: IntVar,
    TileCols: IntVar,
](
    ptrs: ScratchTilePointers[Split, Rows, Cols, TileRows, TileCols],
    mask: MatrixMask[Rows, Cols, [TileRows], [TileCols]],
    *,
    other: float,
) -> tensor[[TileRows, TileCols]]: ...
@overload
def store[Target: IntTuple](
    ptr: OutScalarPointer[Target], value: tensor[[]]
) -> None: ...
@overload
def store[M: IntVar, N: IntVar](
    ptr: TransposeOutputTile[M, N], value: tensor[[N, M]]
) -> None: ...
@overload
def store[Length: IntVar](
    ptr: FullCopyOutputTile[Length], value: tensor[[Length]]
) -> None: ...
@overload
def store[Rows: IntVar, Cols: IntVar](
    ptr: CopyMatrixOutputTile[Rows, Cols], value: tensor[[Rows, Cols]]
) -> None: ...
@overload
def store[Rows: IntVar, Cols: IntVar, BR: IntVar, BC: IntVar](
    ptr: PipelineMatrixOutputTile[Rows, Cols, BR, BC], value: tensor[[BR, BC]]
) -> None: ...
@overload
def store[Block: IntVar](
    ptr: CPUPadOutputTile[Block], value: tensor[[Block]]
) -> None: ...
@overload
def store[Rows: IntVar, BM: IntVar](
    ptr: CPUGemvOutputPointer[Rows, BM], value: tensor[[BM]]
) -> None: ...
@overload
def store[Batch: IntVar, Heads: IntVar, Tokens: IntVar, BM: IntVar](
    ptrs: Attention3DTilePointers[Batch, Heads, Tokens, BM],
    value: tensor[[BM]],
) -> None: ...
@overload
def store[Tokens: IntVar, Dim: IntVar, BM: IntVar](
    ptrs: AttentionOutputTilePointers[Tokens, Dim, BM],
    value: tensor[[BM, Dim]],
) -> None: ...
@overload
def store[M: IntVar, N: IntVar](
    ptr: GroupCTilePointers[M, N], value: tensor[[M, N]]
) -> None: ...
@overload
def store[Heads: IntVar, Tokens: IntVar, Tile: IntTuple](
    ptr: AttentionStatsTilePointers[Heads, Tokens, Tile], value: tensor[Tile]
) -> None: ...
@overload
def store[Groups: IntVar, Cols: IntVar, Tile: IntVar](
    ptrs: GroupedScratchTile[Groups, Cols, Tile],
    value: tensor[[Tile]],
    mask: Mask[[Cols], [Tile]],
) -> None: ...
@overload
def store[
    Target: IntTuple, Strides: IntTuple, Tile: IntTuple, Origin: str, GridAxis: int
](
    ptrs: OutTilePointers[Target, Strides, Tile, Origin, GridAxis],
    value: tensor[Tile],
    mask: Mask[Target, Tile, Origin, GridAxis],
) -> None: ...
@overload
def store[
    Rows: IntVar,
    Cols: IntVar,
    RS: IntVar,
    CS: IntVar,
    TileRows: IntVar,
    TileCols: IntVar,
](
    ptrs: OutTilePointers[
        [Rows, Cols], [RS, CS], [TileRows, TileCols], Literal["indexed"]
    ],
    value: tensor[[TileRows, TileCols]],
    mask: MatrixMask[Rows, Cols, [TileRows], [TileCols]],
    cache_modifier: Literal[".wt"] | None = None,
) -> None: ...
@overload
def store[Rows: IntVar, Cols: IntVar, BM: IntVar, BN: IntVar](
    ptrs: Out2DRowTilePointers[Rows, Cols, BM, BN],
    value: tensor[[BM, BN]],
    mask: MatrixMask[Rows, Cols, [BM], [BN]],
) -> None: ...
@overload
def store[
    Split: IntVar,
    Rows: IntVar,
    Cols: IntVar,
    TileRows: IntVar,
    TileCols: IntVar,
](
    ptrs: ScratchTilePointers[Split, Rows, Cols, TileRows, TileCols],
    value: tensor[[TileRows, TileCols]],
    mask: MatrixMask[Rows, Cols, [TileRows], [TileCols]],
) -> None: ...
@overload
def atomic_add[Tokens: IntVar, Dim: IntVar, BM: IntVar](
    ptrs: ZeroedAttentionOutputTilePointers[Tokens, Dim, BM],
    value: tensor[[BM, Dim]],
) -> tensor[[BM, Dim]]: ...
@overload
def max[Block: IntVar](value: tensor[[Block]], axis: Literal[0]) -> tensor[[]]: ...
@overload
def max[Rows: IntVar, Cols: IntVar](
    value: tensor[[Rows, Cols]], axis: Literal[1]
) -> tensor[[Rows]]: ...

class ReductionOrdering:
    UNORDERED: ReductionOrdering
    INNER_TREE: ReductionOrdering

@overload
def sum[Block: IntVar](value: tensor[[Block]], axis: Literal[0]) -> tensor[[]]: ...
@overload
def sum[Block: IntVar](
    value: tensor[[Block]],
    axis: Literal[0],
    *,
    reduction_ordering: ReductionOrdering,
) -> tensor[[]]: ...
@overload
def sum[Rows: IntVar, Cols: IntVar](
    value: tensor[[Rows, Cols]], axis: Literal[0]
) -> tensor[[Cols]]: ...
@overload
def sum[Rows: IntVar, Cols: IntVar](
    value: tensor[[Rows, Cols]], axis: Literal[1]
) -> tensor[[Rows]]: ...
def exp[Tile: IntTuple](value: tensor[Tile]) -> tensor[Tile]: ...

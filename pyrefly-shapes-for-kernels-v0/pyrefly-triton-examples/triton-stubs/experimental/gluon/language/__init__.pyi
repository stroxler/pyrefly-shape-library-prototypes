# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Final, Literal, overload

from shape_extensions import Int, IntListLiteral, IntVar
from triton.language import (
    InPointer,
    InTilePointers,
    Mask,
    Offsets,
    OutTilePointers,
    tensor,
)

class Layout1D: ...
class Layout2D: ...
class RowSliceLayout(Layout1D): ...
class ColumnSliceLayout(Layout1D): ...
class SharedLayout1D: ...

class Float32DType:
    primitive_bitwidth: Literal[32]

class Float16DType:
    primitive_bitwidth: Literal[16]

class Float8E4DType:
    primitive_bitwidth: Literal[8]
    def __eq__(self, other: Uint8DType) -> Literal[False]: ...

class Uint8DType:
    primitive_bitwidth: Literal[8]
    def __eq__(self, other: Uint8DType) -> Literal[True]: ...

class Int32DType: ...
class Int64DType: ...
class BarrierSharedLayout1D: ...

float32: Float32DType
float16: Float16DType
float8e4nv: Float8E4DType
uint8: Uint8DType
int32: Int32DType
int64: Int64DType
constexpr = Final

class TmaBlockType1D[Block: IntVar]:
    nbytes: int
    def __eq__(self, other: TmaBlockType1D[Block]) -> Literal[True]: ...

class TmaLayout1D[Layout: IntVar](SharedLayout1D):
    def __eq__(self, other: TmaLayout1D[Layout]) -> Literal[True]: ...

class NVMMASharedLayout:
    @staticmethod
    @overload
    def get_default_for[Block: IntVar](
        block_shape: list[Int[Block]], dtype: Float32DType
    ) -> TmaLayout1D[Literal[0]]: ...
    @staticmethod
    @overload
    def get_default_for[Rows: IntVar, Cols: IntVar](
        block_shape: IntListLiteral[[Rows, Cols]], dtype: Float16DType
    ) -> WgmmaLayoutF16ForBlock[Rows, Cols, Literal[0]]: ...
    @staticmethod
    @overload
    def get_default_for[Rows: IntVar, Cols: IntVar](
        block_shape: IntListLiteral[[Rows, Cols]], dtype: Float32DType
    ) -> WgmmaLayoutF32[Literal[0]]: ...

class TmaLayout2D[Layout: IntVar](NVMMASharedLayout): ...
class WgmmaLayoutF16[Layout: IntVar](TmaLayout2D[Layout]): ...
class WgmmaLayoutF16ForBlock[
    Rows: IntVar,
    Cols: IntVar,
    Layout: IntVar,
](WgmmaLayoutF16[Layout]): ...
class WgmmaLayoutF32[Layout: IntVar](TmaLayout2D[Layout]): ...
class TmaRingBlockShape[Count: IntVar, Rows: IntVar, Cols: IntVar](
    IntListLiteral[[Count, Rows, Cols]]
): ...

class TmaBlockShape2D[Rows: IntVar, Cols: IntVar](
    IntListLiteral[[Rows, Cols]], list[int]
):
    @overload
    def __getitem__(self, index: Literal[0]) -> Int[Rows]: ...
    @overload
    def __getitem__(self, index: Literal[1]) -> Int[Cols]: ...

class PackedScaleRingBlockShape[Count: IntVar, Rows: IntVar, K: IntVar](
    IntListLiteral[[Count, 1, Rows // 128, K // 128, 2, 256]]
): ...

class PackedScaleBlockShape[Rows: IntVar, K: IntVar](
    IntListLiteral[[Literal[1], Rows // 128, K // 128, Literal[2], Literal[256]]],
    list[int],
):
    def __radd__[Count: IntVar](
        self, prefix: list[Int[Count]]
    ) -> PackedScaleRingBlockShape[Count, Rows, K]: ...
    @overload
    def __getitem__(self, index: Literal[1]) -> Int[Rows // 128]: ...
    @overload
    def __getitem__(self, index: Literal[2]) -> Int[K // 128]: ...

class PackedScaleBlockType[Rows: IntVar, K: IntVar]:
    shape: PackedScaleBlockShape[Rows, K]
    nbytes: int

class PackedScaleLayout[Rows: IntVar, K: IntVar, Layout: IntVar](NVMMASharedLayout): ...

class PackedScaleDescriptorU8[
    HostRepRows: IntVar,
    HostRepK: IntVar,
    BlockRows: IntVar,
    BlockK: IntVar,
    Layout: IntVar,
]:
    dtype: Uint8DType
    shape: tuple[Literal[1], Int[HostRepRows], Int[HostRepK], Literal[2], Literal[256]]
    block_type: PackedScaleBlockType[BlockRows, BlockK]
    layout: PackedScaleLayout[BlockRows, BlockK, Layout]

class PackedScaleTile[Rows: IntVar, K: IntVar](
    tensor[[Literal[1], Rows // 128, K // 128, Literal[2], Literal[256]]]
):
    shape: tuple[Literal[1], Int[Rows // 128], Int[K // 128], Literal[2], Literal[256]]
    def reshape(
        self,
        rows: Int[Rows // 128],
        k: Int[K // 128],
        d32: Literal[32],
        d4: Literal[4],
        d4b: Literal[4],
    ) -> PackedScaleReshaped[Rows, K]: ...

class PackedScaleReshaped[Rows: IntVar, K: IntVar](
    tensor[[Rows // 128, K // 128, Literal[32], Literal[4], Literal[4]]]
):
    def permute(
        self,
        d0: Literal[0],
        d1: Literal[3],
        d2: Literal[2],
        d3: Literal[1],
        d4: Literal[4],
    ) -> PackedScalePermuted[Rows, K]: ...

class PackedScalePermuted[Rows: IntVar, K: IntVar](
    tensor[[Rows // 128, Literal[4], Literal[32], K // 128, Literal[4]]]
):
    def reshape(self, rows: Int[Rows], k: Int[K // 32]) -> tensor[[Rows, K // 32]]: ...

class AutoLayout: ...

class PackedScaleSharedTile[Rows: IntVar, K: IntVar, Layout: IntVar]:
    dtype: Uint8DType
    shape: tuple[Literal[1], Int[Rows // 128], Int[K // 128], Literal[2], Literal[256]]
    def reshape(
        self,
        shape: tuple[
            Int[Rows // 128], Int[K // 128], Literal[32], Literal[4], Literal[4]
        ],
    ) -> PackedScaleSharedReshaped[Rows, K, Layout]: ...
    def load(self, layout: AutoLayout) -> PackedScaleTile[Rows, K]: ...

class PackedScaleSharedRing[Rows: IntVar, K: IntVar, Layout: IntVar]:
    def index(self, index: int) -> PackedScaleSharedTile[Rows, K, Layout]: ...

class PackedScaleSharedReshaped[Rows: IntVar, K: IntVar, Layout: IntVar]:
    def permute(
        self, order: tuple[Literal[0], Literal[3], Literal[2], Literal[1], Literal[4]]
    ) -> PackedScaleSharedPermuted[Rows, K, Layout]: ...

class PackedScaleSharedPermuted[Rows: IntVar, K: IntVar, Layout: IntVar]:
    def reshape(
        self, shape: tuple[Int[Rows], Int[K // 32]]
    ) -> PackedScaleShared2D[Rows, K // 32, Layout]: ...

class PackedScaleShared2D[Rows: IntVar, Cols: IntVar, Layout: IntVar]:
    dtype: Uint8DType

class TmaBlockShapeF32[Rows: IntVar, Cols: IntVar](TmaBlockShape2D[Rows, Cols]):
    def __radd__[Count: IntVar](
        self, prefix: list[Int[Count]]
    ) -> IntListLiteral[[Count, Rows, Cols]]: ...

class TmaBlockShapeF8[Rows: IntVar, Cols: IntVar](TmaBlockShape2D[Rows, Cols]):
    def __radd__[Count: IntVar](
        self, prefix: list[Int[Count]]
    ) -> TmaRingBlockShape[Count, Rows, Cols]: ...

class TmaBlockType2D[BlockRows: IntVar, BlockCols: IntVar]:
    nbytes: int
    element_ty: Float32DType
    shape: TmaBlockShapeF32[BlockRows, BlockCols]

class TmaBlockTypeF16[BlockRows: IntVar, BlockCols: IntVar]:
    nbytes: int
    shape: TmaBlockShape2D[BlockRows, BlockCols]

class TmaBlockTypeF8[BlockRows: IntVar, BlockCols: IntVar]:
    nbytes: int
    shape: TmaBlockShapeF8[BlockRows, BlockCols]

class WgmmaLayoutF8[Rows: IntVar, Cols: IntVar, Layout: IntVar](
    TmaLayout2D[Layout]
): ...

class TmaDescriptorF8[
    Rows: IntVar,
    Cols: IntVar,
    BlockRows: IntVar,
    BlockCols: IntVar,
    Layout: IntVar,
]:
    dtype: Float8E4DType
    shape: tuple[Int[Rows], Int[Cols]]
    block_type: TmaBlockTypeF8[BlockRows, BlockCols]
    layout: WgmmaLayoutF8[BlockRows, BlockCols, Layout]

class WgmmaSharedF8[Rows: IntVar, Cols: IntVar, Layout: IntVar]:
    dtype: Float8E4DType
    shape: tuple[Int[Rows], Int[Cols]]
    def permute(
        self, order: tuple[Literal[1], Literal[0]]
    ) -> WgmmaSharedF8[Cols, Rows, Layout]: ...

class WgmmaSharedRingF8[Rows: IntVar, Cols: IntVar, Layout: IntVar]:
    def index(self, index: int) -> WgmmaSharedF8[Rows, Cols, Layout]: ...

class ScalePointerDType:
    element_ty: Uint8DType

class TmaInputDescriptorF16[
    Rows: IntVar,
    Cols: IntVar,
    BlockRows: IntVar,
    BlockCols: IntVar,
    Layout: IntVar,
]:
    dtype: Float16DType
    shape: tuple[Int[Rows], Int[Cols]]
    block_type: TmaBlockTypeF16[BlockRows, BlockCols]
    layout: WgmmaLayoutF16ForBlock[BlockRows, BlockCols, Layout]

class GatherInputDescriptorF16[
    Rows: IntVar,
    Cols: IntVar,
    GatherRows: IntVar,
    BlockCols: IntVar,
    Layout: IntVar,
]:
    dtype: Float16DType
    shape: tuple[Int[Rows], Int[Cols]]
    block_type: TmaBlockTypeF16[1, BlockCols]
    layout: WgmmaLayoutF16ForBlock[GatherRows, BlockCols, Layout]

class ScatterOutputDescriptorF16[
    Rows: IntVar,
    Cols: IntVar,
    ScatterRows: IntVar,
    BlockCols: IntVar,
    Layout: IntVar,
]:
    dtype: Float16DType
    shape: tuple[Int[Rows], Int[Cols]]
    block_type: TmaBlockTypeF16[1, BlockCols]
    layout: WgmmaLayoutF16ForBlock[ScatterRows, BlockCols, Layout]

class TmaOutputDescriptorF16[
    Rows: IntVar,
    Cols: IntVar,
    BlockRows: IntVar,
    BlockCols: IntVar,
    Layout: IntVar,
]:
    dtype: Float16DType
    shape: tuple[Int[Rows], Int[Cols]]
    block_type: TmaBlockTypeF16[BlockRows, BlockCols]
    layout: WgmmaLayoutF16ForBlock[BlockRows, BlockCols, Layout]

class TmaDescriptorType2D[BlockRows: IntVar, BlockCols: IntVar]:
    block_type: TmaBlockType2D[BlockRows, BlockCols]

class TmaInputDescriptor2D[
    Rows: IntVar,
    Cols: IntVar,
    BlockRows: IntVar,
    BlockCols: IntVar,
    Layout: IntVar,
]:
    dtype: Float32DType
    shape: tuple[Int[Rows], Int[Cols]]
    type: TmaDescriptorType2D[BlockRows, BlockCols]
    layout: TmaLayout2D[Layout]
    block_type: TmaBlockType2D[BlockRows, BlockCols]

class TmaOutputDescriptor2D[
    Rows: IntVar,
    Cols: IntVar,
    BlockRows: IntVar,
    BlockCols: IntVar,
    Layout: IntVar,
]:
    dtype: Float32DType
    layout: TmaLayout2D[Layout]
    block_type: TmaBlockType2D[BlockRows, BlockCols]

class WgmmaDescriptorF16[
    Rows: IntVar,
    Cols: IntVar,
    BlockRows: IntVar,
    BlockCols: IntVar,
    Layout: IntVar,
](
    TmaInputDescriptorF16[Rows, Cols, BlockRows, BlockCols, Layout],
    TmaOutputDescriptorF16[Rows, Cols, BlockRows, BlockCols, Layout],
): ...
class WgmmaDescriptorF32[
    Rows: IntVar,
    Cols: IntVar,
    BlockRows: IntVar,
    BlockCols: IntVar,
    Layout: IntVar,
](
    TmaInputDescriptor2D[Rows, Cols, BlockRows, BlockCols, Layout],
    TmaOutputDescriptor2D[Rows, Cols, BlockRows, BlockCols, Layout],
): ...

class WgmmaSharedF16[Rows: IntVar, Cols: IntVar, Layout: IntVar]:
    type: WgmmaSharedF16Type[Layout]
    @overload
    def load(self, layout: DotOperandLayout) -> tensor[[Rows, Cols]]: ...
    @overload
    def load(self, layout: Layout2D) -> tensor[[Rows, Cols]]: ...
    def store(self, value: tensor[[Rows, Cols]]) -> None: ...
    def permute(
        self, order: tuple[Literal[1], Literal[0]]
    ) -> WgmmaSharedF16[Cols, Rows, Layout]: ...

class WgmmaSharedRingF16[Rows: IntVar, Cols: IntVar, Layout: IntVar]:
    def index(self, index: int) -> WgmmaSharedF16[Rows, Cols, Layout]: ...

class WgmmaSharedF16Type[Layout: IntVar]:
    layout: TmaLayout2D[Layout]

class NVMMADistributedLayout(Layout2D):
    def __init__(
        self,
        *,
        version: IntListLiteral[[Literal[3], Literal[0]]],
        warps_per_cta: list[int],
        instr_shape: list[int],
    ) -> None: ...

class DotOperandLayout(Layout2D):
    def __init__(
        self,
        *,
        operand_index: Literal[0],
        parent: NVMMADistributedLayout,
        k_width: int,
    ) -> None: ...

class TmaInputDescriptor1D[Length: IntVar, Block: IntVar, Layout: IntVar]:
    dtype: Float32DType
    layout: TmaLayout1D[Layout]
    block_type: TmaBlockType1D[Block]

class TmaOutputDescriptor1D[Length: IntVar, Block: IntVar, Layout: IntVar]:
    dtype: Float32DType
    layout: TmaLayout1D[Layout]
    block_type: TmaBlockType1D[Block]

class TmaDescriptor1D[Length: IntVar, Block: IntVar, Layout: IntVar](
    TmaInputDescriptor1D[Length, Block, Layout],
    TmaOutputDescriptor1D[Length, Block, Layout],
):
    strides: tuple[Int[1]]

class MessageDescriptor1D[Length: IntVar, Block: IntVar, Layout: IntVar]:
    dtype: Int32DType
    layout: TmaLayout1D[Layout]
    block_type: TmaBlockType1D[Block]
    block_shape: IntListLiteral[[Block]]

class TmaSharedBuffer1D[Block: IntVar, Layout: IntVar](SharedBuffer1D[Block]): ...

class TmaSharedTile2D[BlockRows: IntVar, BlockCols: IntVar, Layout: IntVar]:
    def load(self, layout: Layout2D) -> tensor[[BlockRows, BlockCols]]: ...
    def store(self, value: tensor[[BlockRows, BlockCols]]) -> None: ...

class TmaSharedRingType2D[
    Count: IntVar,
    BlockRows: IntVar,
    BlockCols: IntVar,
]:
    shape: tuple[Int[Count], Int[BlockRows], Int[BlockCols]]

class TmaSharedRing2D[
    Count: IntVar,
    BlockRows: IntVar,
    BlockCols: IntVar,
    Layout: IntVar,
]:
    type: TmaSharedRingType2D[Count, BlockRows, BlockCols]
    def index(self, index: int) -> TmaSharedTile2D[BlockRows, BlockCols, Layout]: ...

class TmaBarrierRing2D[Count: IntVar]:
    shape: tuple[Int[Count], Literal[1]]
    def index(self, index: int) -> BarrierBuffer1D: ...

type _WarpBars[LoadDepth: IntVar, StoreDepth: IntVar] = tuple[
    TmaBarrierRing2D[LoadDepth],
    TmaBarrierRing2D[LoadDepth],
    TmaBarrierRing2D[StoreDepth],
    TmaBarrierRing2D[StoreDepth],
]
type _WarpBuffers[
    LoadDepth: IntVar,
    StoreDepth: IntVar,
    BR: IntVar,
    BC: IntVar,
    LA: IntVar,
    LB: IntVar,
    LC: IntVar,
] = tuple[
    TmaSharedRing2D[LoadDepth, BR, BC, LA],
    TmaSharedRing2D[LoadDepth, BR, BC, LB],
    TmaSharedRing2D[StoreDepth, BR, BC, LC],
]
type _WarpDescs[
    Rows: IntVar,
    Cols: IntVar,
    BR: IntVar,
    BC: IntVar,
    LA: IntVar,
    LB: IntVar,
    LC: IntVar,
] = tuple[
    TmaInputDescriptor2D[Rows, Cols, BR, BC, LA],
    TmaInputDescriptor2D[Rows, Cols, BR, BC, LB],
    TmaOutputDescriptor2D[Rows, Cols, BR, BC, LC],
]

def warp_specialize[
    Rows: IntVar,
    Cols: IntVar,
    BR: IntVar,
    BC: IntVar,
    LA: IntVar,
    LB: IntVar,
    LC: IntVar,
    LoadDepth: IntVar,
    StoreDepth: IntVar,
](
    partitions: list[
        tuple[
            Callable[
                [
                    _WarpBars[LoadDepth, StoreDepth],
                    _WarpBuffers[LoadDepth, StoreDepth, BR, BC, LA, LB, LC],
                    Int[Cols],
                    Int[BC],
                    Layout2D,
                ],
                None,
            ],
            tuple[
                _WarpBars[LoadDepth, StoreDepth],
                _WarpBuffers[LoadDepth, StoreDepth, BR, BC, LA, LB, LC],
                Int[Cols],
                Int[BC],
                Layout2D,
            ],
        ]
        | tuple[
            Callable[
                [
                    _WarpDescs[Rows, Cols, BR, BC, LA, LB, LC],
                    _WarpBars[LoadDepth, StoreDepth],
                    _WarpBuffers[LoadDepth, StoreDepth, BR, BC, LA, LB, LC],
                    GluonTileStart[BR],
                    tuple[Int[Rows], Int[Cols]],
                    Int[BC],
                ],
                None,
            ],
            tuple[
                _WarpDescs[Rows, Cols, BR, BC, LA, LB, LC],
                _WarpBars[LoadDepth, StoreDepth],
                _WarpBuffers[LoadDepth, StoreDepth, BR, BC, LA, LB, LC],
                GluonTileStart[BR],
                tuple[Int[Rows], Int[Cols]],
                Int[BC],
            ],
        ]
    ],
    warps_per_partition: list[int],
    maxnregs: list[int],
) -> None: ...

class MessageSharedBuffer1D[Block: IntVar, Layout: IntVar]:
    def store(self, value: tensor[[Block]]) -> None: ...
    def load(self, layout: Layout1D) -> tensor[[Block]]: ...

class MessageOutputPointer1D[Length: IntVar]:
    def __add__[Block: IntVar](
        self, offsets: Offsets[[Block]]
    ) -> MessageOutputTile1D[Length, Block]: ...

class MessageOutputTile1D[Length: IntVar, Block: IntVar]: ...
class AtomicReadyFlagPointer1D: ...
class shared_memory_descriptor: ...
class BarrierBuffer1D(shared_memory_descriptor): ...

def static_assert(condition: Literal[True], message: str = "") -> None: ...
@overload
def static_range(stop: int) -> range: ...
@overload
def static_range(start: int, stop: int, step: int) -> range: ...
def cdiv(dividend: int, divisor: int) -> int: ...
def to_tensor(value: bool | int) -> tensor[[]]: ...
def zeros[Rows: IntVar, Cols: IntVar](
    shape: tuple[Int[Rows], Int[Cols]],
    *,
    dtype: Float32DType,
    layout: NVMMADistributedLayout,
) -> tensor[[Rows, Cols]]: ...
def SwizzledSharedLayout(
    *,
    vec: Literal[1],
    per_phase: Literal[1],
    max_phase: Literal[1],
    order: IntListLiteral[[int]],
) -> SharedLayout1D: ...

class SharedBuffer1D[Block: IntVar]:
    def load(self, layout: Layout1D) -> tensor[[Block]]: ...

@overload
def allocate_shared_memory[
    BlockRows: IntVar,
    BlockCols: IntVar,
    Layout: IntVar,
](
    dtype: Float32DType,
    shape: IntListLiteral[[BlockRows, BlockCols]],
    layout: TmaLayout2D[Layout],
) -> TmaSharedTile2D[BlockRows, BlockCols, Layout]: ...
@overload
def allocate_shared_memory[
    BlockRows: IntVar,
    BlockCols: IntVar,
    Layout: IntVar,
](
    dtype: Float32DType,
    shape: tuple[Int[BlockRows], Int[BlockCols]],
    layout: TmaLayout2D[Layout],
) -> TmaSharedTile2D[BlockRows, BlockCols, Layout]: ...
@overload
def allocate_shared_memory[BlockRows: IntVar, BlockCols: IntVar](
    dtype: Float32DType,
    shape: tuple[Int[BlockRows], Int[BlockCols]],
    layout: NVMMASharedLayout,
) -> TmaSharedTile2D[BlockRows, BlockCols, Literal[0]]: ...
@overload
def allocate_shared_memory[
    BlockRows: IntVar,
    BlockCols: IntVar,
    Layout: IntVar,
](
    dtype: Float16DType,
    shape: IntListLiteral[[BlockRows, BlockCols]],
    layout: TmaLayout2D[Layout],
) -> WgmmaSharedF16[BlockRows, BlockCols, Layout]: ...
@overload
def allocate_shared_memory[
    BlockRows: IntVar,
    BlockCols: IntVar,
    Layout: IntVar,
](
    dtype: Float8E4DType,
    shape: IntListLiteral[[BlockRows, BlockCols]],
    layout: WgmmaLayoutF8[BlockRows, BlockCols, Layout],
) -> WgmmaSharedF8[BlockRows, BlockCols, Layout]: ...
@overload
def allocate_shared_memory[
    Count: IntVar,
    BlockRows: IntVar,
    BlockCols: IntVar,
    Layout: IntVar,
](
    dtype: Float8E4DType,
    shape: TmaRingBlockShape[Count, BlockRows, BlockCols],
    layout: WgmmaLayoutF8[BlockRows, BlockCols, Layout],
) -> WgmmaSharedRingF8[BlockRows, BlockCols, Layout]: ...
@overload
def allocate_shared_memory[Rows: IntVar, K: IntVar, Layout: IntVar](
    dtype: Uint8DType,
    shape: PackedScaleBlockShape[Rows, K],
    layout: PackedScaleLayout[Rows, K, Layout],
) -> PackedScaleSharedTile[Rows, K, Layout]: ...
@overload
def allocate_shared_memory[Count: IntVar, Rows: IntVar, K: IntVar, Layout: IntVar](
    dtype: Uint8DType,
    shape: PackedScaleRingBlockShape[Count, Rows, K],
    layout: PackedScaleLayout[Rows, K, Layout],
) -> PackedScaleSharedRing[Rows, K, Layout]: ...
@overload
def allocate_shared_memory[
    BlockRows: IntVar,
    BlockCols: IntVar,
    Layout: IntVar,
](
    dtype: Float16DType,
    shape: list[int],
    layout: WgmmaLayoutF16ForBlock[BlockRows, BlockCols, Layout],
) -> WgmmaSharedRingF16[BlockRows, BlockCols, Layout]: ...
@overload
def allocate_shared_memory[
    Count: IntVar,
    BlockRows: IntVar,
    BlockCols: IntVar,
    Layout: IntVar,
](
    dtype: Float32DType,
    shape: IntListLiteral[[Count, BlockRows, BlockCols]],
    layout: TmaLayout2D[Layout],
) -> TmaSharedRing2D[Count, BlockRows, BlockCols, Layout]: ...
@overload
def allocate_shared_memory[Count: IntVar](
    dtype: Int64DType,
    shape: IntListLiteral[[Count, 1]],
    layout: BarrierSharedLayout1D,
) -> TmaBarrierRing2D[Count]: ...
@overload
def allocate_shared_memory[Block: IntVar](
    dtype: Float32DType,
    shape: IntListLiteral[[Block]],
    *,
    layout: SharedLayout1D,
) -> SharedBuffer1D[Block]: ...
@overload
def allocate_shared_memory[Block: IntVar, Layout: IntVar](
    dtype: Float32DType,
    shape: IntListLiteral[[Block]],
    layout: TmaLayout1D[Layout],
) -> TmaSharedBuffer1D[Block, Layout]: ...
@overload
def allocate_shared_memory[Block: IntVar, Layout: IntVar](
    dtype: Int32DType,
    shape: IntListLiteral[[Block]],
    layout: TmaLayout1D[Layout],
) -> MessageSharedBuffer1D[Block, Layout]: ...
@overload
def allocate_shared_memory(
    dtype: Int64DType,
    shape: IntListLiteral[[Literal[1]]],
    layout: BarrierSharedLayout1D,
) -> BarrierBuffer1D: ...
@overload
def BlockedLayout(
    size_per_thread: IntListLiteral[[int]],
    threads_per_warp: IntListLiteral[[int]],
    warps_per_cta: IntListLiteral[[int]],
    order: IntListLiteral[[int]],
) -> Layout1D: ...
@overload
def BlockedLayout(
    size_per_thread: IntListLiteral[[int, int]],
    threads_per_warp: IntListLiteral[[int, int]],
    warps_per_cta: IntListLiteral[[int, int]],
    order: IntListLiteral[[int, int]],
) -> Layout2D: ...
@overload
def SliceLayout(*, dim: Literal[1], parent: Layout2D) -> RowSliceLayout: ...
@overload
def SliceLayout(*, dim: Literal[0], parent: Layout2D) -> ColumnSliceLayout: ...
@overload
def SliceLayout(dim: Literal[1], parent: Layout2D) -> RowSliceLayout: ...
@overload
def SliceLayout(dim: Literal[0], parent: Layout2D) -> ColumnSliceLayout: ...

class GluonTileStart[Block: IntVar]:
    @overload
    def __add__(self, offset: Int[Block // 2]) -> GluonTileStart[Block]: ...
    @overload
    def __add__(self, offsets: Offsets[[Block]]) -> Offsets[[Block]]: ...
    @overload
    def __add__(self, offsets: RowIndices[Block]) -> RowIndices[Block]: ...
    @overload
    def __add__(self, offsets: ColumnIndices[Block]) -> ColumnIndices[Block]: ...

class GluonProgramId:
    def __mul__[Block: IntVar](self, block: Int[Block]) -> GluonTileStart[Block]: ...

class GluonTileId:
    def __mul__[Block: IntVar](self, block: Int[Block]) -> GluonTileStart[Block]: ...

def program_id(axis: Literal[0, 1]) -> GluonProgramId: ...
def num_programs(axis: Literal[0, 1]) -> int: ...

class RowIndices[Block: IntVar]:
    def __getitem__(self, index: tuple[slice, None]) -> RowIndices2D[Block]: ...

class ColumnIndices[Block: IntVar]:
    def __radd__(self, offset: int) -> ColumnIndices[Block]: ...
    def __getitem__(self, index: tuple[None, slice]) -> ColumnIndices2D[Block]: ...

class RowIndices2D[Block: IntVar]:
    def __mul__[Stride: IntVar](
        self, stride: Int[Stride]
    ) -> RowAddress2D[Block, Stride]: ...
    def __rmul__[Stride: IntVar](
        self, stride: Int[Stride]
    ) -> RowAddress2D[Block, Stride]: ...
    def __lt__[Rows: IntVar](self, rows: Int[Rows]) -> RowMask2D[Rows, Block]: ...

class ColumnIndices2D[Block: IntVar]:
    def __mul__[Stride: IntVar](
        self, stride: Int[Stride]
    ) -> ColumnAddress2D[Block, Stride]: ...
    def __rmul__[Stride: IntVar](
        self, stride: Int[Stride]
    ) -> ColumnAddress2D[Block, Stride]: ...
    def __lt__[Cols: IntVar](self, cols: Int[Cols]) -> ColumnMask2D[Cols, Block]: ...

class RowAddress2D[Block: IntVar, Stride: IntVar]:
    @overload
    def __add__[OtherBlock: IntVar, OtherStride: IntVar](
        self, other: ColumnAddress2D[OtherBlock, OtherStride]
    ) -> MatrixAddress2D[Block, OtherBlock, Stride, OtherStride]: ...
    @overload
    def __add__[OtherBlock: IntVar](
        self, other: ColumnIndices2D[OtherBlock]
    ) -> ContiguousMatrixAddress2D[Block, OtherBlock, Stride]: ...

class ColumnAddress2D[Block: IntVar, Stride: IntVar]: ...
class MatrixAddress2D[
    BlockRows: IntVar,
    BlockCols: IntVar,
    RowStride: IntVar,
    ColStride: IntVar,
]: ...
class ContiguousMatrixAddress2D[Rows: IntVar, Cols: IntVar, Stride: IntVar]: ...

class Float32PointerDType:
    element_ty: Float32DType

class InContiguousMatrixPointer2D[Rows: IntVar, Cols: IntVar]:
    dtype: Float32PointerDType
    def __add__(
        self, address: ContiguousMatrixAddress2D[Rows, Cols, Cols]
    ) -> InContiguousMatrixTile2D[Rows, Cols]: ...

class OutContiguousMatrixPointer2D[Rows: IntVar, Cols: IntVar]:
    def __add__(
        self, address: ContiguousMatrixAddress2D[Rows, Cols, Cols]
    ) -> OutContiguousMatrixTile2D[Rows, Cols]: ...

class InContiguousMatrixTile2D[Rows: IntVar, Cols: IntVar]: ...
class OutContiguousMatrixTile2D[Rows: IntVar, Cols: IntVar]: ...

class RowMask2D[Rows: IntVar, Block: IntVar]:
    def __and__[Cols: IntVar, OtherBlock: IntVar](
        self, other: ColumnMask2D[Cols, OtherBlock]
    ) -> MatrixMask2D[Rows, Cols, Block, OtherBlock]: ...

class ColumnMask2D[Cols: IntVar, Block: IntVar]: ...
class MatrixMask2D[
    Rows: IntVar,
    Cols: IntVar,
    BlockRows: IntVar,
    BlockCols: IntVar,
]: ...

class InMatrixPointer2D[
    Rows: IntVar,
    Cols: IntVar,
    RowStride: IntVar,
    ColStride: IntVar,
]:
    @overload
    def __add__[BR: IntVar, BC: IntVar](
        self, address: MatrixAddress2D[BR, BC, RowStride, ColStride]
    ) -> InMatrixTile2D[Rows, Cols, BR, BC]: ...
    @overload
    def __add__[BR: IntVar](
        self, address: RowAddress2D[BR, RowStride]
    ) -> InMatrixRowTile2D[Rows, Cols, BR, ColStride]: ...

class ScalePointer2D[
    Rows: IntVar,
    Cols: IntVar,
    RowStride: IntVar,
    ColStride: IntVar,
](InMatrixPointer2D[Rows, Cols, RowStride, ColStride]):
    dtype: ScalePointerDType
    @overload
    def __add__[BR: IntVar, BC: IntVar](
        self, address: MatrixAddress2D[BR, BC, RowStride, ColStride]
    ) -> ScaleMatrixTile2D[Rows, Cols, BR, BC]: ...
    @overload
    def __add__[BR: IntVar](
        self, address: RowAddress2D[BR, RowStride]
    ) -> ScaleMatrixRowTile2D[Rows, Cols, BR, ColStride]: ...

class ScaleMatrixRowTile2D[
    Rows: IntVar,
    Cols: IntVar,
    BR: IntVar,
    ColStride: IntVar,
](InMatrixRowTile2D[Rows, Cols, BR, ColStride]):
    def __add__[BC: IntVar](
        self, address: ColumnAddress2D[BC, ColStride]
    ) -> ScaleMatrixTile2D[Rows, Cols, BR, BC]: ...

class ScaleMatrixTile2D[Rows: IntVar, Cols: IntVar, BR: IntVar, BC: IntVar](
    InMatrixTile2D[Rows, Cols, BR, BC]
): ...

class ContiguousScalePointer1D[
    Elements: IntVar,
    Rows: IntVar,
    Cols: IntVar,
](InPointer[[Elements]]):
    dtype: ScalePointerDType
    @overload
    def __add__(
        self, base_offset: int
    ) -> ContiguousScalePointer1D[Elements, Rows, Cols]: ...
    @overload
    def __add__[Block: IntVar](
        self, base_offset: GluonTileStart[Block]
    ) -> ContiguousScalePointer1D[Elements, Rows, Cols]: ...
    @overload
    def __add__(
        self, offsets: Offsets[[Rows * Cols]]
    ) -> ContiguousScaleTilePointer1D[Elements, Rows, Cols]: ...

class ContiguousScaleTilePointer1D[
    Elements: IntVar,
    Rows: IntVar,
    Cols: IntVar,
]: ...

class ContiguousScaleTile1D[Rows: IntVar, Cols: IntVar](tensor[[Rows * Cols]]):
    def reshape(self, rows: Int[Rows], cols: Int[Cols]) -> tensor[[Rows, Cols]]: ...

class InMatrixRowTile2D[Rows: IntVar, Cols: IntVar, BR: IntVar, ColStride: IntVar]:
    def __add__[BC: IntVar](
        self, address: ColumnAddress2D[BC, ColStride]
    ) -> InMatrixTile2D[Rows, Cols, BR, BC]: ...

class OutMatrixPointer2D[
    Rows: IntVar,
    Cols: IntVar,
    RowStride: IntVar,
    ColStride: IntVar,
]:
    @overload
    def __add__[BR: IntVar, BC: IntVar](
        self, address: MatrixAddress2D[BR, BC, RowStride, ColStride]
    ) -> OutMatrixTile2D[Rows, Cols, BR, BC]: ...
    @overload
    def __add__[BR: IntVar](
        self, address: RowAddress2D[BR, RowStride]
    ) -> OutMatrixRowTile2D[Rows, Cols, BR, ColStride]: ...

class OutMatrixRowTile2D[Rows: IntVar, Cols: IntVar, BR: IntVar, ColStride: IntVar]:
    def __add__[BC: IntVar](
        self, address: ColumnAddress2D[BC, ColStride]
    ) -> OutMatrixTile2D[Rows, Cols, BR, BC]: ...

class InMatrixTile2D[Rows: IntVar, Cols: IntVar, BR: IntVar, BC: IntVar]: ...
class OutMatrixTile2D[Rows: IntVar, Cols: IntVar, BR: IntVar, BC: IntVar]: ...

class Float32TensorTile2D[Rows: IntVar, Cols: IntVar](tensor[[Rows, Cols]]):
    dtype: Float32DType

class InFloat32MatrixTile2D[Rows: IntVar, Cols: IntVar, BR: IntVar, BC: IntVar](
    InMatrixTile2D[Rows, Cols, BR, BC]
): ...

class InFloat32MatrixRowTile2D[
    Rows: IntVar,
    Cols: IntVar,
    BR: IntVar,
    ColStride: IntVar,
](InMatrixRowTile2D[Rows, Cols, BR, ColStride]):
    def __add__[BC: IntVar](
        self, address: ColumnAddress2D[BC, ColStride]
    ) -> InFloat32MatrixTile2D[Rows, Cols, BR, BC]: ...

class InFloat32MatrixPointer2D[
    Rows: IntVar,
    Cols: IntVar,
    RowStride: IntVar,
    ColStride: IntVar,
](InMatrixPointer2D[Rows, Cols, RowStride, ColStride]):
    @overload
    def __add__[BR: IntVar, BC: IntVar](
        self, address: MatrixAddress2D[BR, BC, RowStride, ColStride]
    ) -> InFloat32MatrixTile2D[Rows, Cols, BR, BC]: ...
    @overload
    def __add__[BR: IntVar](
        self, address: RowAddress2D[BR, RowStride]
    ) -> InFloat32MatrixRowTile2D[Rows, Cols, BR, ColStride]: ...

class GatherOffsetsPointer1D[Length: IntVar]:
    @overload
    def __add__(self, offsets: Offsets[[Length]]) -> GatherOffsetsTile1D[Length]: ...
    @overload
    def __add__[Block: IntVar](
        self, tile_start: GluonTileStart[Block]
    ) -> GatherOffsetsPointerWindow1D[Length, Block]: ...

class GatherOffsetsPointerWindow1D[Length: IntVar, Block: IntVar]:
    @overload
    def __add__(self, offsets: Offsets[[Block]]) -> GatherOffsetsTile1D[Block]: ...
    @overload
    def __add__(self, offsets: ColumnIndices[Block]) -> GatherOffsetsTile1D[Block]: ...

class GatherOffsetsTile1D[Length: IntVar]: ...

def num_warps() -> int: ...
def barrier() -> None: ...
@overload
def arange[Block: IntVar](
    start: Literal[0], end: Int[Block], *, layout: RowSliceLayout
) -> RowIndices[Block]: ...
@overload
def arange[Block: IntVar](
    start: Literal[0], end: Int[Block], *, layout: ColumnSliceLayout
) -> ColumnIndices[Block]: ...
@overload
def arange[Block: IntVar](
    start: Literal[0], end: Int[Block], layout: RowSliceLayout
) -> RowIndices[Block]: ...
@overload
def arange[Block: IntVar](
    start: Literal[0], end: Int[Block], layout: ColumnSliceLayout
) -> ColumnIndices[Block]: ...
@overload
def arange[Block: IntVar](
    start: Literal[0], end: Int[Block], layout: Layout1D
) -> Offsets[[Block]]: ...
@overload
def arange[Block: IntVar](
    start: Literal[0], end: Int[Block], *, layout: Layout1D
) -> Offsets[[Block]]: ...
@overload
def load[Target: IntVar, Block: IntVar](
    ptr: InTilePointers[[Target], [Block]], mask: Mask[[Target], [Block]]
) -> tensor[[Block]]: ...
@overload
def load[Rows: IntVar, Cols: IntVar, BR: IntVar, BC: IntVar](
    ptr: InMatrixTile2D[Rows, Cols, BR, BC],
    mask: MatrixMask2D[Rows, Cols, BR, BC],
) -> tensor[[BR, BC]]: ...
@overload
def load[Length: IntVar](ptr: GatherOffsetsTile1D[Length]) -> tensor[[Length]]: ...
@overload
def load[Rows: IntVar, Cols: IntVar, BR: IntVar, BC: IntVar](
    ptr: ScaleMatrixTile2D[Rows, Cols, BR, BC],
) -> tensor[[BR, BC]]: ...
@overload
def load[Elements: IntVar, Rows: IntVar, Cols: IntVar](
    ptr: ContiguousScaleTilePointer1D[Elements, Rows, Cols],
) -> ContiguousScaleTile1D[Rows, Cols]: ...
@overload
def store[Length: IntVar](
    ptr: MessageOutputTile1D[Length, Length], value: tensor[[Length]]
) -> None: ...
@overload
def store[Target: IntVar, Block: IntVar](
    ptr: OutTilePointers[[Target], [Block]],
    value: tensor[[Block]],
    mask: Mask[[Target], [Block]],
) -> None: ...
@overload
def store[Rows: IntVar, Cols: IntVar, BR: IntVar, BC: IntVar](
    ptr: OutMatrixTile2D[Rows, Cols, BR, BC],
    value: tensor[[BR, BC]],
    mask: MatrixMask2D[Rows, Cols, BR, BC],
) -> None: ...
@overload
def load[Rows: IntVar, Cols: IntVar](
    ptr: InContiguousMatrixTile2D[Rows, Cols],
) -> tensor[[Rows, Cols]]: ...
@overload
def load[Rows: IntVar, Cols: IntVar](
    ptr: InFloat32MatrixTile2D[Rows, Cols, Rows, Cols],
) -> Float32TensorTile2D[Rows, Cols]: ...
@overload
def load[Rows: IntVar, Cols: IntVar](
    ptr: InMatrixTile2D[Rows, Cols, Rows, Cols],
) -> tensor[[Rows, Cols]]: ...
@overload
def store[Rows: IntVar, Cols: IntVar](
    ptr: OutContiguousMatrixTile2D[Rows, Cols], value: tensor[[Rows, Cols]]
) -> None: ...
@overload
def store[Rows: IntVar, Cols: IntVar](
    ptr: OutMatrixTile2D[Rows, Cols, Rows, Cols], value: tensor[[Rows, Cols]]
) -> None: ...
@overload
def store[Rows: IntVar, Cols: IntVar, BR: IntVar, BC: IntVar](
    ptr: OutMatrixTile2D[Rows, Cols, BR, BC], value: tensor[[BR, BC]]
) -> None: ...
@overload
def convert_layout[Rows: IntVar, Cols: IntVar](
    input: tensor[[Rows, Cols]], target_layout: Layout2D
) -> tensor[[Rows, Cols]]: ...
@overload
def convert_layout[Length: IntVar](
    input: tensor[[Length]], target_layout: Layout1D
) -> tensor[[Length]]: ...
def atomic_xchg(
    ready: AtomicReadyFlagPointer1D,
    value: Literal[1],
    *,
    sem: Literal["release"],
    scope: Literal["gpu"],
) -> int: ...
def atomic_add(
    ready: AtomicReadyFlagPointer1D,
    value: Literal[0],
    *,
    sem: Literal["acquire"],
    scope: Literal["gpu"],
) -> int: ...

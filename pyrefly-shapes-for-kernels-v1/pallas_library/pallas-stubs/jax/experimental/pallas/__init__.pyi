# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from types import EllipsisType
from typing import Literal, overload

from jax import Array, ShapeDtypeStruct
from jax.experimental.pallas import mosaic_gpu as gpu
from jax.experimental.pallas import tpu, triton
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
        self: Tile[[Heads, Keys]], other: Tile[[1, Keys]]
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

class MhaSegmentRef[Sequence: IntVar]:
    def __getitem__[Block: IntVar](
        self, index: HalfRowSlice[Block]
    ) -> Array[[Block]]: ...

class RowIndices[Block: IntVar]:
    def __ge__[KeyBlock: IntVar](
        self, other: ColumnIndices[KeyBlock]
    ) -> Mask[[Block, KeyBlock], [int, int]]: ...

class ColumnIndices[Block: IntVar]: ...

class AxisBound[Rows: IntVar](int):
    def __add__(self, other: int) -> AxisBound[Rows]: ...

class AxisBoundRef[Rows: IntVar]:
    def __getitem__(self, key: tuple[()]) -> AxisBound[Rows]: ...

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

class RectOutputBlock[
    RowBlock: IntVar,
    Rows: IntVar,
    Cols: IntVar,
    ColBlock: IntVar,
]: ...

class ScalarFloat(float):
    def astype(self, dtype: object) -> Tile[[]]: ...
    def __rtruediv__(self, other: int) -> ScalarFloat: ...
    def __truediv__(self, other: int) -> ScalarFloat: ...

class InRef[Shape: IntTuple]:
    @overload
    def __getitem__(self: InRef[[]], key: EllipsisType) -> ScalarFloat: ...
    @overload
    def __getitem__(self: InRef[[]], key: tuple[()]) -> int: ...
    @overload
    def __getitem__(self, key: slice | EllipsisType) -> Tile[Shape]: ...
    @overload
    def __getitem__[Rows: IntVar, Cols: IntVar](
        self: InRef[[Rows, Cols]], key: tuple[slice, slice]
    ) -> Tile[[Rows, Cols]]: ...
    @overload
    def __getitem__[Rows: IntVar, Cols: IntVar, Block: IntVar](
        self: InRef[[Rows, Cols]], key: tuple[HalfRowSlice[Block], slice]
    ) -> Tile[[Block, Cols]]: ...
    @overload
    def __getitem__[Length: IntVar, Block: IntVar](
        self: InRef[[Length]], key: HalfRowSlice[Block]
    ) -> Tile[[Block]]: ...
    @overload
    def __getitem__[Rows: IntVar, Cols: IntVar](
        self: InRef[[1, Rows, Cols]], key: tuple[Literal[0], slice, slice]
    ) -> Tile[[Rows, Cols]]: ...
    @overload
    def __getitem__[Rows: IntVar, Cols: IntVar](
        self: InRef[[1, Rows, Cols]], key: tuple[Literal[0], EllipsisType]
    ) -> Tile[[Rows, Cols]]: ...
    @overload
    @property
    def shape[Length: IntVar](self: InRef[[Length]]) -> tuple[Int[Length]]: ...
    @overload
    @property
    def shape[Rows: IntVar, Cols: IntVar](
        self: InRef[[Rows, Cols]],
    ) -> tuple[Int[Rows], Int[Cols]]: ...
    @property
    def dtype(self) -> object: ...
    @overload
    @property
    def at[Length: IntVar](self: InRef[[Length]]) -> InRefAt[[Length]]: ...
    @overload
    @property
    def at[Rows: IntVar, Cols: IntVar](
        self: InRef[[Rows, Cols]],
    ) -> InRefAt[[Rows, Cols]]: ...

class ValidInRef[Shape: IntTuple, ValidShape: IntTuple]:
    @property
    def shape[Rows: IntVar, Padded: IntVar, Valid: IntVar](
        self: ValidInRef[[Rows, Padded], [Rows, Valid]],
    ) -> tuple[Int[Rows], Int[Padded]]: ...
    @property
    def dtype(self) -> object: ...
    @property
    def at[Rows: IntVar, Padded: IntVar, Valid: IntVar](
        self: ValidInRef[[Rows, Padded], [Rows, Valid]],
    ) -> InRefAt[[Rows, Padded], [Rows, Valid]]: ...

# An indexed Ref retains its base shape so the load/store mask can be checked
# against the original bounds, even though the selected tile has another shape.
class TransformedRef[
    BaseShape: IntTuple,
    TileShape: IntTuple,
    Role: str,
    ValidShape: IntTuple = BaseShape,
]: ...

class InRefAt[Shape: IntTuple, ValidShape: IntTuple = Shape]:
    @overload
    def __getitem__[Rows: IntVar, Padded: IntVar, Valid: IntVar, Block: IntVar](
        self: InRefAt[[Rows, Padded], [Rows, Valid]],
        key: tuple[HalfRowSlice[Block], slice],
    ) -> TransformedRef[
        [Rows, Padded], [Block, Padded], Literal["in"], [Rows, Valid]
    ]: ...
    @overload
    def __getitem__[Length: IntVar](
        self: InRefAt[[Length]],
        key: slice[None, Int[Length // 2], None] | slice[Int[Length // 2], None, None],
    ) -> InRef[[Length // 2]]: ...
    @overload
    def __getitem__[Length: IntVar, Block: IntVar](
        self: InRefAt[[Length]], key: Indices[Block]
    ) -> TransformedRef[[Length], [Block], Literal["in"]]: ...
    @overload
    def __getitem__[Rows: IntVar, Cols: IntVar, Block: IntVar](
        self: InRefAt[[Rows, Cols]], key: tuple[HalfRowSlice[Block], slice]
    ) -> TransformedRef[[Rows, Cols], [Block, Cols], Literal["in"]]: ...
    @overload
    def __getitem__[Rows: IntVar, Cols: IntVar, RowBlock: IntVar, ColBlock: IntVar](
        self: InRefAt[[Rows, Cols]],
        key: tuple[RowIndices[RowBlock], ColumnIndices[ColBlock]],
    ) -> TransformedRef[[Rows, Cols], [RowBlock, ColBlock], Literal["in"]]: ...

class DynamicSlice: ...
class HalfRowSlice[Size: IntVar](DynamicSlice): ...

@overload
def ds[Size: IntVar](start: int, size: Int[Size]) -> HalfRowSlice[Size]: ...
@overload
def ds(start: int, size: int) -> DynamicSlice: ...
def dslice[Size: IntVar](start: int, size: Int[Size]) -> HalfRowSlice[Size]: ...

class Indices[Block: IntVar]:
    def __radd__(self, other: int) -> Indices[Block]: ...
    def __add__(self, other: int) -> Indices[Block]: ...
    @overload
    def __getitem__(self, key: tuple[None, slice]) -> ColumnIndices[Block]: ...
    @overload
    def __getitem__(self, key: None) -> ColumnIndices[Block]: ...
    @overload
    def __getitem__(self, key: tuple[slice, None]) -> RowIndices[Block]: ...
    @overload
    def __lt__[Rows: IntVar](self, limit: AxisBound[Rows]) -> Mask[[Block], [Rows]]: ...
    @overload
    def __lt__[Length: IntVar](self, limit: Int[Length]) -> Mask[[Block], [Length]]: ...
    @overload
    def __lt__(self, limit: int) -> Mask[[Block], [int]]: ...
    def __ge__(self, limit: int) -> Mask[[Block], [int]]: ...

class Mask[Tile: IntTuple, Bounds: IntTuple]:
    def __invert__(self) -> Mask[Tile, Bounds]: ...
    def __mul__[Keys: IntVar](
        self: Mask[[1, Keys], [1, int]], scalar: float
    ) -> Tile[[1, Keys]]: ...
    @overload
    def __getitem__[Block: IntVar, Length: IntVar](
        self: Mask[[Block], [Length]], key: tuple[None, slice]
    ) -> Mask[[1, Block], [1, Length]]: ...
    @overload
    def __getitem__[Block: IntVar, Length: IntVar](
        self: Mask[[Block], [Length]], key: tuple[slice, None]
    ) -> Mask[[Block, 1], [Length, 1]]: ...
    def reshape[Block: IntVar, Length: IntVar](
        self: Mask[[Block, 1], [Length, 1]], size: Literal[-1]
    ) -> Mask[[Block], [Length]]: ...
    @overload
    def __and__(self, other: Mask[Tile, Bounds]) -> Mask[Tile, Bounds]: ...
    @overload
    def __and__[RowBlock: IntVar, Rows: IntVar, ColBlock: IntVar, Cols: IntVar](
        self: Mask[[RowBlock, 1], [Rows, 1]],
        other: Mask[[1, ColBlock], [1, Cols]],
    ) -> Mask[[RowBlock, ColBlock], [Rows, Cols]]: ...
    def __iand__[RowBlock: IntVar, Rows: IntVar, ColBlock: IntVar, Cols: IntVar](
        self: Mask[[RowBlock, 1], [Rows, 1]],
        other: Mask[[1, ColBlock], [1, Cols]],
    ) -> Mask[[RowBlock, ColBlock], [Rows, Cols]]: ...

class OutRef[Shape: IntTuple]:
    @overload
    @property
    def shape[Length: IntVar](self: OutRef[[Length]]) -> tuple[Int[Length]]: ...
    @overload
    @property
    def shape[Rows: IntVar, Cols: IntVar](
        self: OutRef[[Rows, Cols]],
    ) -> tuple[Int[Rows], Int[Cols]]: ...
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
    def at[Length: IntVar](self: OutRef[[Length]]) -> OutRefAt[[Length]]: ...
    @overload
    @property
    def at[Rows: IntVar, Cols: IntVar](
        self: OutRef[[Rows, Cols]],
    ) -> OutRefAt[[Rows, Cols]]: ...
    @property
    def dtype(self) -> object: ...

class ValidOutRef[Shape: IntTuple, ValidShape: IntTuple]:
    @property
    def shape[Rows: IntVar, VisibleCols: IntVar, Cols: IntVar](
        self: ValidOutRef[[Rows, VisibleCols], [Rows, Cols]],
    ) -> tuple[Int[Rows], Int[VisibleCols]]: ...
    @property
    def dtype(self) -> object: ...
    @property
    def at[Rows: IntVar, Padded: IntVar, Valid: IntVar](
        self: ValidOutRef[[Rows, Padded], [Rows, Valid]],
    ) -> OutRefAt[[Rows, Padded], [Rows, Valid]]: ...

# Pipelined accumulation reads a previously written output SRAM tile. The
# type cannot prove that an earlier grid iteration initialized this buffer.
class AccumRef[Shape: IntTuple]:
    def __getitem__(self, key: EllipsisType) -> Tile[Shape]: ...
    def __setitem__(self, key: EllipsisType, value: Tile[Shape]) -> None: ...

class OutRefAt[Shape: IntTuple, ValidShape: IntTuple = Shape]:
    @overload
    def __getitem__[Rows: IntVar, ColBlock: IntVar, Cols: IntVar, RowBlock: IntVar](
        self: OutRefAt[[Rows, ColBlock], [Rows, Cols]],
        key: tuple[HalfRowSlice[RowBlock], HalfRowSlice[ColBlock]],
    ) -> RectOutputBlock[RowBlock, Rows, Cols, ColBlock]: ...
    @overload
    def __getitem__[Rows: IntVar, Padded: IntVar, Valid: IntVar](
        self: OutRefAt[[Rows, Padded], [Rows, Valid]],
        key: tuple[slice, slice[None, Int[Padded], None]],
    ) -> TransformedRef[
        [Rows, Padded], [Rows, Padded], Literal["out"], [Rows, Valid]
    ]: ...
    @overload
    def __getitem__[Rows: IntVar, Cols: IntVar](
        self: OutRefAt[[Rows, Cols]],
        key: tuple[slice, slice[None, Int[Cols], None]],
    ) -> TransformedRef[[Rows, Cols], [Rows, Cols], Literal["out"]]: ...
    @overload
    def __getitem__[Length: IntVar](
        self: OutRefAt[[Length]],
        key: slice[None, Int[Length // 2], None] | slice[Int[Length // 2], None, None],
    ) -> OutRef[[Length // 2]]: ...
    @overload
    def __getitem__[Length: IntVar, Block: IntVar](
        self: OutRefAt[[Length]], key: Indices[Block]
    ) -> TransformedRef[[Length], [Block], Literal["out"]]: ...
    @overload
    def __getitem__[Length: IntVar, Block: IntVar](
        self: OutRefAt[[Length]], key: HalfRowSlice[Block]
    ) -> TransformedRef[[Length], [Block], Literal["out"]]: ...
    @overload
    def __getitem__[Rows: IntVar, Cols: IntVar, Block: IntVar](
        self: OutRefAt[[Rows, Cols]], key: tuple[HalfRowSlice[Block], slice]
    ) -> TransformedRef[[Rows, Cols], [Block, Cols], Literal["out"]]: ...

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
class NoBlockSpec: ...

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

def when(
    predicate: bool | int,
) -> Callable[[Callable[[], None]], Callable[[], None]]: ...
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
            InRef[[HeadBlock, Dim]],
            InRef[[SplitKeys, Dim]],
            InRef[[SplitKeys, Dim]],
            InRef[[]] | None,
            InRef[[]] | None,
            OutRef[[HeadBlock, Dim]],
            OutRef[[HeadBlock]],
            OutRef[[HeadBlock]],
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
            ValidInRef[[QueryBlock, Dim], [QueryBlock, Dim]],
            InRef[[Keys, Dim]],
            InRef[[Keys, Dim]],
            MhaSegmentRef[Keys] | None,
            OutRef[[QueryBlock, Dim]],
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
            ValidInRef[[QueryBlock, PaddedDim], [QueryBlock, HeadDim]],
            ValidInRef[[QueryBlock, PaddedDim], [QueryBlock, HeadDim]],
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
            ValidInRef[[Queries, PaddedDim], [Queries, HeadDim]],
            ValidInRef[[Keys, PaddedDim], [Keys, HeadDim]],
            ValidInRef[[Keys, PaddedDim], [Keys, HeadDim]],
            MhaSegmentRef[Keys] | None,
            ValidInRef[[Queries, PaddedDim], [Queries, HeadDim]],
            ValidInRef[[Queries, PaddedDim], [Queries, HeadDim]],
            InRef[[Queries]],
            InRef[[Queries]],
            ValidOutRef[[QueryBlock, PaddedDim], [QueryBlock, HeadDim]],
            ValidOutRef[[KeyBlock, PaddedDim], [KeyBlock, HeadDim]],
            ValidOutRef[[KeyBlock, PaddedDim], [KeyBlock, HeadDim]],
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
def pallas_call[Rows: IntVar, Cols: IntVar, ColBlock: IntVar](
    kernel: Callable[
        [
            InRef[[Rows, Cols]],
            InRef[[Cols]],
            InRef[[Cols]],
            InRef[[Rows, Cols]],
            InRef[[Rows]],
            InRef[[Rows]],
            OutRef[[Cols]],
            OutRef[[Cols]],
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
            InRef[[Rows, Cols]],
            InRef[[Cols]],
            InRef[[Cols]],
            InRef[[Rows, Cols]],
            InRef[[Rows]],
            OutRef[[Cols]],
            OutRef[[Cols]],
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
            AxisBoundRef[Rows],
            AxisBoundRef[Rows],
            ValidOutRef[[Rows, ColBlock], [Rows, Cols]],
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

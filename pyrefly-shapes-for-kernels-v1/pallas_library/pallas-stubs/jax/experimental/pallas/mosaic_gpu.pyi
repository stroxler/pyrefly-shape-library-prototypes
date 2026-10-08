# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from types import EllipsisType
from typing import Literal, overload

from jax import Array, Float16Array, ShapeDtypeStruct
from jax.experimental.pallas import HalfRowSlice, Tile
from shape_extensions import Int, IntVar

class GmemInRef[Rows: IntVar, Cols: IntVar]: ...
class GMEM: ...
class TmaIndicesLayout: ...
class OtherIndicesLayout: ...

class Layout:
    TMA_INDICES: TmaIndicesLayout
    WG_STRIDED: OtherIndicesLayout

class TmaIndices[Rows: IntVar]: ...
class TmaIndexRef[Rows: IntVar]: ...
class TmaGatherGmemSlice[SourceRows: IntVar, Rows: IntVar, Cols: IntVar]: ...

class TmaGmemSourceAt[SourceRows: IntVar, Cols: IntVar]:
    def __getitem__[Rows: IntVar](
        self, indices: TmaIndices[Rows]
    ) -> TmaGatherGmemSlice[SourceRows, Rows, Cols]: ...

class TmaGmemSourceRef[SourceRows: IntVar, Cols: IntVar]:
    @property
    def at(self) -> TmaGmemSourceAt[SourceRows, Cols]: ...

class TmaSmemOutRef[Rows: IntVar, Cols: IntVar]: ...
class TmaBarrier: ...
class Barrier: ...

def load[Rows: IntVar](
    ref: TmaIndexRef[Rows], *, layout: TmaIndicesLayout
) -> TmaIndices[Rows]: ...
def copy_gmem_to_smem[SourceRows: IntVar, Rows: IntVar, Cols: IntVar](
    source: TmaGatherGmemSlice[SourceRows, Rows, Cols],
    destination: TmaSmemOutRef[Rows, Cols],
    barrier: TmaBarrier,
) -> None: ...
def barrier_wait(barrier: TmaBarrier) -> None: ...

class GmemOutRef[Rows: IntVar, Cols: IntVar]:
    @property
    def at(self) -> GmemOutAt[Rows, Cols]: ...

class GmemOutAt[Rows: IntVar, Cols: IntVar]:
    def __getitem__[RowTile: IntVar, ColTile: IntVar](
        self, key: tuple[HalfRowSlice[RowTile], HalfRowSlice[ColTile]]
    ) -> GmemTile[RowTile, ColTile]: ...

class GmemTile[Rows: IntVar, Cols: IntVar]: ...
class SmemInRef[Rows: IntVar, Cols: IntVar]: ...

class SmemScratchRef[Rows: IntVar, Cols: IntVar]:
    def __setitem__(self, key: EllipsisType, value: Tile[[Rows, Cols]]) -> None: ...

class AccRef[Rows: IntVar, Cols: IntVar]:
    def __getitem__(self, key: EllipsisType) -> Tile[[Rows, Cols]]: ...

class SMEM[Rows: IntVar, Cols: IntVar]:
    def __init__(self, shape: tuple[Int[Rows], Int[Cols]], dtype: object) -> None: ...

class ACC[Rows: IntVar, Cols: IntVar]:
    def __init__(self, shape: tuple[Int[Rows], Int[Cols]], dtype: object) -> None: ...

class TilingTransform:
    def __init__(self, shape: tuple[int, int]) -> None: ...

class SwizzleTransform:
    def __init__(self, swizzle: int) -> None: ...

class BlockSpec[Rows: IntVar, Cols: IntVar]:
    @overload
    def __init__(
        self,
        *,
        memory_space: type[SMEM[Rows, Cols]],
        transforms: tuple[()],
    ) -> None: ...
    @overload
    def __init__(
        self,
        shape: tuple[Int[Rows], Int[Cols]],
        index_map: Callable[[int], tuple[int, int]],
        *,
        transforms: tuple[TilingTransform, SwizzleTransform],
        delay_release: int = 0,
    ) -> None: ...

def wgmma[Rows: IntVar, Inner: IntVar, Cols: IntVar](
    acc: AccRef[Rows, Cols],
    lhs: SmemInRef[Rows, Inner],
    rhs: SmemInRef[Inner, Cols],
) -> None: ...
def wgmma_wait(count: int) -> None: ...

class MatmulPipeline[Rows: IntVar, Inner: IntVar, Cols: IntVar]:
    def __call__[FullRows: IntVar, FullInner: IntVar, FullCols: IntVar](
        self,
        lhs: GmemInRef[FullRows, FullInner],
        rhs: GmemInRef[FullInner, FullCols],
    ) -> None: ...

def emit_pipeline[Rows: IntVar, Inner: IntVar, Cols: IntVar](
    body: Callable[[int, SmemInRef[Rows, Inner], SmemInRef[Inner, Cols]], None],
    *,
    in_specs: list[BlockSpec[Rows, Inner] | BlockSpec[Inner, Cols]],
    grid: tuple[int],
    max_concurrent_steps: Literal[2],
) -> MatmulPipeline[Rows, Inner, Cols]: ...
def commit_smem() -> None: ...
def copy_smem_to_gmem[Rows: IntVar, Cols: IntVar](
    source: SmemScratchRef[Rows, Cols], destination: GmemTile[Rows, Cols]
) -> None: ...
def wait_smem_to_gmem(count: int) -> None: ...
@overload
def kernel[
    Rows: IntVar,
    Inner: IntVar,
    Cols: IntVar,
    RowTile: IntVar,
    InnerTile: IntVar,
    ColTile: IntVar,
](
    body: Callable[
        [
            GmemInRef[Rows, Inner],
            GmemInRef[Inner, Cols],
            GmemOutRef[Rows, Cols],
            SmemScratchRef[RowTile, ColTile],
            AccRef[RowTile, ColTile],
        ],
        None,
    ],
    *,
    out_type: ShapeDtypeStruct[[Rows, Cols]],
    scratch_types: dict[str, SMEM[RowTile, ColTile] | ACC[RowTile, ColTile]],
    grid: tuple[int, int],
    grid_names: tuple[Literal["m"], Literal["n"]],
) -> Callable[
    [Float16Array[[Rows, Inner]], Float16Array[[Inner, Cols]]],
    Array[[Rows, Cols]],
]: ...

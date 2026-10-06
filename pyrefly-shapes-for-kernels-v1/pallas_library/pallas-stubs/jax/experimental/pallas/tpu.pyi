# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from types import EllipsisType
from typing import Literal, overload

from jax.experimental.pallas import (
    BlockSpec,
    BoundedInRef,
    BoundedOutRef,
    DeviceIdType,
    GatherIndicesRef,
    Indices,
    IndirectGather,
    IndirectScatter,
    InRef,
    OutRef,
    Tile,
    UnconstrainedInRef,
    UnconstrainedSlice,
    VmemInRef,
)
from jax.numpy import BFloat16Dtype
from jax.random import Key
from shape_extensions import Int, IntTuple, IntVar

class PallasKey: ...

class PallasKeyRef:
    def __getitem__(self, key: EllipsisType) -> PallasKey: ...

def to_pallas_key(key: Key[Literal["threefry2x32"]]) -> PallasKey: ...
def sample_block[Rows: IntVar, Cols: IntVar](
    sampler_function: Callable[..., object],
    global_key: PallasKey,
    *,
    block_size: tuple[Int[Rows], Int[Cols]],
    tile_size: tuple[Literal[16], Literal[128]],
    total_size: tuple[Literal[64], Literal[512]],
    block_index: tuple[int, int],
    minval: float,
    maxval: float,
) -> Tile[[Rows, Cols]]: ...

class VMEM[Shape: IntTuple]:
    def __init__[Rows: IntVar, Cols: IntVar](
        self: VMEM[[Rows, Cols]],
        shape: tuple[Int[Rows], Int[Cols]],
        dtype: object,
    ) -> None: ...

class VmemScratchRef[Shape: IntTuple]:
    @property
    def dtype(self) -> object: ...
    def __getitem__(self, key: EllipsisType) -> Tile[Shape]: ...
    def __setitem__(self, key: EllipsisType, value: Tile[Shape]) -> None: ...
    def view[Window: IntVar, Cols: IntVar](
        self: VmemScratchRef[[Window, Cols]], dtype: BFloat16Dtype
    ) -> BfloatPairView[Window, Cols]: ...

class BfloatPairView[Window: IntVar, Cols: IntVar]:
    def reshape(
        self, inferred_window: Literal[-1], packing: Literal[2], cols: Int[Cols]
    ) -> PairedBfloatRows[Window, Cols]: ...

class PairedBfloatRows[Window: IntVar, Cols: IntVar]:
    def __getitem__(self, key: tuple[slice, Literal[0, 1]]) -> Tile[[Window, Cols]]: ...

class SMEM[Shape: IntTuple]:
    @overload
    def __init__[Lanes: IntVar](
        self: SMEM[[Lanes]], shape: tuple[Int[Lanes]], dtype: object
    ) -> None: ...
    @overload
    def __init__[Rows: IntVar, Cols: IntVar](
        self: SMEM[[Rows, Cols]],
        shape: tuple[Int[Rows], Int[Cols]],
        dtype: object,
    ) -> None: ...

class SmemScratchRef[Shape: IntTuple]:
    @overload
    def __getitem__[Lanes: IntVar](self: SmemScratchRef[[Lanes]], key: int) -> int: ...
    @overload
    def __getitem__[Rows: IntVar](
        self: SmemScratchRef[[Rows, 2]], key: tuple[int, Literal[0, 1]]
    ) -> int: ...
    def __setitem__[Lanes: IntVar](
        self: SmemScratchRef[[Lanes]], key: int, value: int
    ) -> None: ...

class CompilerParams:
    @overload
    def __init__(self, *, collective_id: Literal[0]) -> None: ...
    @overload
    def __init__(
        self, *, dimension_semantics: tuple[Literal["parallel", "arbitrary"]]
    ) -> None: ...

class DmaSemaphore: ...
class RegularSemaphore: ...
class BarrierSemaphore: ...

class DmaSemaphoreArray[Count: IntVar]:
    @property
    def at(self) -> DmaSemaphoreArrayAt: ...

class DmaSemaphoreArrayAt:
    def __getitem__(self, index: int) -> DmaSemaphore: ...

class DmaSemaphoreArrayAllocation[Count: IntVar]: ...

class SemaphoreType:
    DMA: DmaSemaphoreType
    REGULAR: RegularSemaphoreType
    def __call__[Count: IntVar](
        self, shape: tuple[Int[Count]]
    ) -> DmaSemaphoreArrayAllocation[Count]: ...

class DmaSemaphoreType(SemaphoreType): ...
class RegularSemaphoreType(SemaphoreType): ...

class RemoteCopyOp:
    def start(self) -> None: ...
    def wait(self) -> None: ...

@overload
def async_copy[Lanes: IntVar](
    source: InRef[[Lanes]],
    destination: SmemScratchRef[[Lanes]],
    sem: DmaSemaphore,
) -> RemoteCopyOp: ...
@overload
def async_copy[Lanes: IntVar](
    source: SmemScratchRef[[Lanes]],
    destination: OutRef[[Lanes]],
    sem: DmaSemaphore,
) -> RemoteCopyOp: ...
def get_barrier_semaphore() -> BarrierSemaphore: ...
@overload
def make_async_remote_copy[Rows: IntVar, Cols: IntVar](
    *,
    src_ref: UnconstrainedInRef[[Rows, Cols]],
    dst_ref: OutRef[[Rows, Cols]],
    send_sem: DmaSemaphore,
    recv_sem: DmaSemaphore,
    device_id: tuple[int],
    device_id_type: DeviceIdType,
) -> RemoteCopyOp: ...
@overload
def make_async_remote_copy[Rows: IntVar, Cols: IntVar](
    *,
    src_ref: VmemInRef[[Rows, Cols]],
    dst_ref: OutRef[[Rows, Cols]],
    send_sem: DmaSemaphore,
    recv_sem: DmaSemaphore,
    device_id: tuple[int],
    device_id_type: DeviceIdType,
) -> RemoteCopyOp: ...
@overload
def make_async_remote_copy[Rows: IntVar, Cols: IntVar](
    *,
    src_ref: OutRef[[Rows, Cols]],
    dst_ref: OutRef[[Rows, Cols]],
    send_sem: DmaSemaphore,
    recv_sem: DmaSemaphore,
    device_id: tuple[int],
    device_id_type: DeviceIdType,
) -> RemoteCopyOp: ...
@overload
def make_async_copy[Rows: IntVar, Cols: IntVar](
    *,
    src_ref: UnconstrainedInRef[[Rows, Cols]],
    dst_ref: OutRef[[Rows, Cols]],
    sem: DmaSemaphore,
) -> RemoteCopyOp: ...
@overload
def make_async_copy[Rows: IntVar, Cols: IntVar](
    *,
    src_ref: VmemScratchRef[[Rows, Cols]],
    dst_ref: OutRef[[Rows, Cols]],
    sem: DmaSemaphore,
) -> RemoteCopyOp: ...
@overload
def make_async_copy[Rows: IntVar, Cols: IntVar](
    *,
    src_ref: OutRef[[Rows, Cols]],
    dst_ref: VmemScratchRef[[Rows, Cols]],
    sem: DmaSemaphore,
) -> RemoteCopyOp: ...
@overload
def sync_copy[Rows: IntVar](
    source: UnconstrainedInRef[[Rows, 2]], destination: SmemScratchRef[[Rows, 2]]
) -> None: ...
@overload
def sync_copy(
    source: UnconstrainedInRef[[1]], destination: SmemScratchRef[[1]]
) -> None: ...
@overload
def sync_copy[Cols: IntVar](
    source: UnconstrainedSlice[Cols], destination: VmemScratchRef[[1, Cols]]
) -> None: ...
@overload
def sync_copy[Batch: IntVar, Cols: IntVar, Window: IntVar](
    source: IndirectGather[Batch, Cols, Window], destination: OutRef[[Window, Cols]]
) -> None: ...
@overload
def sync_copy[Batch: IntVar, Cols: IntVar, Window: IntVar](
    source: InRef[[Window, Cols]], destination: IndirectScatter[Batch, Cols, Window]
) -> None: ...
@overload
def sync_copy[PackedRows: IntVar, Cols: IntVar, Window: IntVar](
    source: IndirectGather[PackedRows, Cols, Window],
    destination: VmemScratchRef[[Window, Cols]],
) -> None: ...

class IndexedAddPipeline:
    def __call__[Rows: IntVar, Cols: IntVar](
        self, x: UnconstrainedInRef[[Rows, Cols]], output: OutRef[[Rows, Cols // 2]]
    ) -> None: ...

class SparseCoreRegisterPipeline:
    def __call__[Rows: IntVar](
        self, x: UnconstrainedInRef[[Rows, 128]], output: OutRef[[Rows, 128]]
    ) -> None: ...

@overload
def emit_pipeline[Rows: IntVar, Cols: IntVar, Max: IntVar](
    body: Callable[[BoundedInRef[Max, Cols], BoundedOutRef[Max, Cols]], None],
    *,
    grid: tuple[int],
    in_specs: list[BlockSpec[[Max, Cols], Literal["dynamic"]]],
    out_specs: BlockSpec[[Max, Cols], Literal["dynamic"]],
) -> Callable[[UnconstrainedInRef[[Rows, Cols]], OutRef[[Rows, Cols]]], None]: ...
@overload
def emit_pipeline(
    body: Callable[[InRef[[8, 128]], OutRef[[8, 128]]], None],
    *,
    grid: tuple[int, int],
    in_specs: list[BlockSpec[[8, 128]]],
    out_specs: list[BlockSpec[[8, 128]]],
) -> IndexedAddPipeline: ...
@overload
def emit_pipeline(
    body: Callable[[BoundedInRef[8, 128], BoundedOutRef[8, 128]], None],
    *,
    grid: tuple[int, int],
    in_specs: list[BlockSpec[[8, 128], Literal["sc_bounded"]]],
    out_specs: list[BlockSpec[[8, 128], Literal["sc_bounded"]]],
) -> SparseCoreRegisterPipeline: ...
@overload
def emit_pipeline[Cols: IntVar, Window: IntVar](
    body: Callable[[GatherIndicesRef[Window], OutRef[[Window, Cols]]], None],
    *,
    grid: tuple[int],
    in_specs: list[BlockSpec[[1, Window]]],
    out_specs: list[BlockSpec[[Window, Cols]]],
    core_axis_name: Literal["subcore"],
    dimension_semantics: tuple[ParallelDimension],
) -> GatherPipeline[Cols, Window]: ...

class GatherPipeline[Cols: IntVar, Window: IntVar]:
    def __call__[Num: IntVar](
        self, indices: InRef[[1, Num]], out: OutRef[[Num, Cols]]
    ) -> None: ...

@overload
def emit_pipeline[Cols: IntVar, Window: IntVar](
    body: Callable[[InRef[[Window, Cols]], GatherIndicesRef[Window]], None],
    *,
    grid: tuple[int],
    in_specs: list[BlockSpec[[Window, Cols]] | BlockSpec[[1, Window]]],
    out_specs: list[object],
    core_axis_name: Literal["subcore"],
    dimension_semantics: tuple[ParallelDimension],
) -> ScatterPipeline[Cols, Window]: ...

class ScatterPipeline[Cols: IntVar, Window: IntVar]:
    def __call__[Num: IntVar](
        self, values: InRef[[Num, Cols]], indices: InRef[[1, Num]]
    ) -> None: ...

@overload
def emit_pipeline[Cols: IntVar, Window: IntVar](
    body: Callable[[Indices[Window], OutRef[[Window, Cols]]], None],
    *,
    grid: tuple[int],
    in_specs: list[BlockSpec[[Window]]],
    out_specs: list[BlockSpec[[Window, Cols]]],
    core_axis_name: Literal["subcore"],
    dimension_semantics: tuple[ParallelDimension],
) -> PackedGatherPipeline[Cols, Window]: ...

class PackedGatherPipeline[Cols: IntVar, Window: IntVar]:
    def __call__[Num: IntVar](
        self, indices: InRef[[Num]], out: OutRef[[Num, Cols]]
    ) -> None: ...

class ParallelDimension: ...

PARALLEL: ParallelDimension

class PrefetchScalarGridSpec[
    SourceRows: IntVar,
    SourceCols: IntVar,
    RowBlock: IntVar,
    ColBlock: IntVar,
    Inner: IntVar = Literal[1],
    InnerBlock: IntVar = Literal[1],
    Devices: IntVar = Literal[1],
    Transposed: bool = Literal[False],
]:
    @overload
    def __init__[
        RowBlock: IntVar,
        ColBlock: IntVar,
        InnerBlock: IntVar,
    ](
        self: PrefetchScalarGridSpec[
            int, int, RowBlock, ColBlock, int, InnerBlock, int
        ],
        *,
        num_scalar_prefetch: Literal[4],
        grid: tuple[int, int, int],
        in_specs: list[
            BlockSpec[[RowBlock, InnerBlock]]
            | BlockSpec[[InnerBlock, ColBlock]]
            | BlockSpec[[1, RowBlock, ColBlock]]
        ],
        out_specs: BlockSpec[[RowBlock, ColBlock]],
        scratch_shapes: list[VMEM[[RowBlock, ColBlock]]],
    ) -> None: ...
    @overload
    def __init__[
        Blocks: IntVar,
        RowBlock: IntVar,
        ColBlock: IntVar,
        InnerBlock: IntVar,
    ](
        self: PrefetchScalarGridSpec[
            Literal[1], Literal[1], RowBlock, ColBlock, InnerBlock, InnerBlock, Blocks
        ],
        *,
        num_scalar_prefetch: Literal[2],
        grid: tuple[int, Int[Blocks]],
        in_specs: list[
            BlockSpec[[1, RowBlock, InnerBlock]]
            | BlockSpec[[InnerBlock, ColBlock]]
            | BlockSpec[[RowBlock, ColBlock]]
        ],
        out_specs: BlockSpec[[RowBlock, ColBlock]],
        scratch_shapes: list[VMEM[[RowBlock, ColBlock]]],
    ) -> None: ...
    @overload
    def __init__(
        self: PrefetchScalarGridSpec[
            SourceRows,
            SourceCols,
            RowBlock,
            ColBlock,
            Inner,
            InnerBlock,
            Literal[1],
            Literal[True],
        ],
        num_scalar_prefetch: Literal[0],
        grid: tuple[
            Int[SourceRows // RowBlock],
            Int[SourceCols // ColBlock],
            Int[Inner // InnerBlock],
        ],
        in_specs: tuple[
            BlockSpec[[RowBlock, InnerBlock]], BlockSpec[[ColBlock, InnerBlock]]
        ],
        out_specs: BlockSpec[[RowBlock, ColBlock]],
        scratch_shapes: tuple[VMEM[[RowBlock, ColBlock]]],
    ) -> None: ...
    @overload
    def __init__(
        self: PrefetchScalarGridSpec[
            SourceRows, SourceCols, RowBlock, ColBlock, Literal[1], Literal[1]
        ],
        num_scalar_prefetch: Literal[1],
        grid: tuple[Literal[1], Literal[1]],
        in_specs: list[BlockSpec[[RowBlock, ColBlock], Literal["prefetch"]]],
        out_specs: BlockSpec[[RowBlock, ColBlock], Literal["prefetch"]],
    ) -> None: ...
    @overload
    def __init__(
        self: PrefetchScalarGridSpec[
            SourceRows,
            SourceCols,
            RowBlock,
            ColBlock,
            Inner,
            InnerBlock,
            Literal[1],
            Literal[False],
        ],
        num_scalar_prefetch: Literal[0],
        grid: tuple[
            Int[SourceRows // RowBlock],
            Int[SourceCols // ColBlock],
            Int[Inner // InnerBlock],
        ],
        in_specs: tuple[
            BlockSpec[[RowBlock, InnerBlock]], BlockSpec[[InnerBlock, ColBlock]]
        ],
        out_specs: BlockSpec[[RowBlock, ColBlock]],
        scratch_shapes: tuple[VMEM[[RowBlock, ColBlock]]],
    ) -> None: ...
    @overload
    def __init__(
        self: PrefetchScalarGridSpec[
            SourceRows, SourceCols, SourceRows, SourceCols, Literal[1], Literal[1]
        ],
        num_scalar_prefetch: Literal[0],
        in_specs: tuple[BlockSpec[[SourceRows, SourceCols], Literal["unconstrained"]]],
        out_specs: BlockSpec[[SourceRows, SourceCols], Literal["unconstrained"]],
        scratch_shapes: tuple[DmaSemaphoreType, DmaSemaphoreType],
    ) -> None: ...
    @overload
    def __init__(
        self: PrefetchScalarGridSpec[
            SourceRows,
            SourceCols,
            SourceRows,
            SourceCols,
            Literal[1],
            Literal[1],
            Devices,
        ],
        num_scalar_prefetch: Literal[0],
        in_specs: tuple[BlockSpec[[SourceRows, SourceCols], Literal["unconstrained"]]],
        out_specs: BlockSpec[
            [Devices, SourceRows, SourceCols], Literal["unconstrained"]
        ],
        scratch_shapes: tuple[
            DmaSemaphoreType,
            DmaSemaphoreType,
            DmaSemaphoreArrayAllocation[Devices - 1],
        ],
        grid: tuple[Int[Devices - 1]],
    ) -> None: ...
    @overload
    def __init__(
        self: PrefetchScalarGridSpec[
            SourceRows,
            SourceCols,
            SourceRows,
            SourceCols,
            Literal[1],
            Literal[1],
            Devices,
        ],
        num_scalar_prefetch: Literal[0],
        in_specs: tuple[BlockSpec[[SourceRows, SourceCols], Literal["vmem"]]],
        out_specs: tuple[
            BlockSpec[[SourceRows, SourceCols], Literal["vmem"]],
            BlockSpec[[2, SourceRows, SourceCols], Literal["unconstrained"]],
        ],
        scratch_shapes: tuple[
            DmaSemaphoreType,
            DmaSemaphoreType,
            DmaSemaphoreType,
            RegularSemaphoreType,
            VMEM[[SourceRows, SourceCols]],
        ],
        grid: tuple[Int[Devices]],
    ) -> None: ...
    @overload
    def __init__(
        self: PrefetchScalarGridSpec[
            Devices * SourceRows,
            SourceCols,
            SourceRows,
            SourceCols,
            Literal[1],
            Literal[1],
            Devices,
        ],
        num_scalar_prefetch: Literal[0],
        in_specs: tuple[BlockSpec[[Devices, SourceRows, SourceCols], Literal["vmem"]]],
        out_specs: tuple[
            BlockSpec[[SourceRows, SourceCols], Literal["vmem"]],
            BlockSpec[[2, SourceRows, SourceCols], Literal["unconstrained"]],
        ],
        scratch_shapes: tuple[
            DmaSemaphoreType,
            DmaSemaphoreType,
            DmaSemaphoreType,
            DmaSemaphoreType,
            DmaSemaphoreType,
            RegularSemaphoreType,
            RegularSemaphoreType,
            VMEM[[SourceRows // 2, SourceCols]],
        ],
        grid: tuple[Int[Devices], Literal[2]],
    ) -> None: ...

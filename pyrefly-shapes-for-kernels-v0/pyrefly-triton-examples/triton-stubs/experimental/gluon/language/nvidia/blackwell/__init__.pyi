# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# @lint-ignore-every AUTODEPS2

from typing import Literal, overload

from shape_extensions import Int, IntListLiteral, IntVar
from triton.experimental.gluon.language import (
    BarrierBuffer1D,
    Float8E4DType,
    Float16DType,
    Float32DType,
    Layout2D,
    TmaSharedTile2D,
    Uint8DType,
    WgmmaSharedF8,
    WgmmaSharedF16,
)
from triton.experimental.gluon.language.nvidia.blackwell import tma
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
)
from triton.language import tensor

class TensorMemoryLayout:
    def __init__(
        self, block: tuple[int, int] | IntListLiteral[[int, int]], col_stride: int
    ) -> None: ...

class TensorMemoryScalesLayout: ...

class tensor_memory_descriptor:
    def get_reg_layout(self, *, instr_variant: str = "") -> Layout2D: ...
    def load(self, layout: Layout2D | None = None) -> tensor[[int, int]]: ...

class TensorMemoryTile[Rows: IntVar, Cols: IntVar](tensor_memory_descriptor):
    def get_reg_layout(self, *, instr_variant: str = "") -> Layout2D: ...
    def store(self, value: tensor[[Rows, Cols]]) -> None: ...
    def load(self, layout: Layout2D | None = None) -> tensor[[Rows, Cols]]: ...

class TensorMemoryTileF16[Rows: IntVar, Cols: IntVar](TensorMemoryTile[Rows, Cols]): ...
class TensorMemoryTileF32[Rows: IntVar, Cols: IntVar](TensorMemoryTile[Rows, Cols]): ...
class TensorMemoryScaleTile[Rows: IntVar, Cols: IntVar](
    TensorMemoryTile[Rows, Cols]
): ...

class TensorMemoryRingF32[Count: IntVar, Rows: IntVar, Cols: IntVar](
    tensor_memory_descriptor
):
    def index(self, index: int) -> TensorMemoryTileF32[Rows, Cols]: ...

@overload
def allocate_tensor_memory[Count: IntVar, Rows: IntVar, Cols: IntVar](
    element_ty: Float32DType,
    shape: IntListLiteral[[Count, Rows, Cols]],
    layout: TensorMemoryLayout,
) -> TensorMemoryRingF32[Count, Rows, Cols]: ...
@overload
def allocate_tensor_memory[Rows: IntVar, Cols: IntVar](
    element_ty: Float32DType,
    shape: IntListLiteral[[Rows, Cols]],
    layout: TensorMemoryLayout,
) -> TensorMemoryTileF32[Rows, Cols]: ...
@overload
def allocate_tensor_memory[Rows: IntVar, Cols: IntVar](
    element_ty: Float32DType,
    shape: tuple[Int[Rows], Int[Cols]],
    layout: TensorMemoryLayout,
) -> TensorMemoryTileF32[Rows, Cols]: ...
@overload
def allocate_tensor_memory[Rows: IntVar, Cols: IntVar](
    element_ty: Float16DType,
    shape: IntListLiteral[[Rows, Cols]],
    layout: TensorMemoryLayout,
) -> TensorMemoryTileF16[Rows, Cols]: ...
@overload
def allocate_tensor_memory[Rows: IntVar, Cols: IntVar](
    element_ty: Uint8DType,
    shape: IntListLiteral[[Rows, Cols]],
    layout: TensorMemoryScalesLayout,
) -> TensorMemoryScaleTile[Rows, Cols]: ...
def tcgen05_mma[M: IntVar, K: IntVar, N: IntVar, LA: IntVar, LB: IntVar](
    a: WgmmaSharedF16[M, K, LA] | TensorMemoryTileF16[M, K],
    b: WgmmaSharedF16[K, N, LB],
    acc: TensorMemoryTileF32[M, N],
    *,
    use_acc: bool = True,
    mbarriers: list[BarrierBuffer1D] | None = None,
    mbarrier_preds: list[bool] | None = None,
) -> None: ...
def tcgen05_mma_scaled[
    M: IntVar,
    K: IntVar,
    N: IntVar,
    ScaleK: IntVar,
    LA: IntVar,
    LB: IntVar,
](
    a: WgmmaSharedF8[M, K, LA],
    b: WgmmaSharedF8[K, N, LB],
    acc: TensorMemoryTileF32[M, N],
    a_scale: TensorMemoryScaleTile[M, ScaleK],
    b_scale: TensorMemoryScaleTile[N, ScaleK],
    a_format: Literal["e4m3"],
    b_format: Literal["e4m3"],
    *,
    use_acc: bool = True,
    pred: bool = True,
) -> None: ...
def tcgen05_commit(barrier: BarrierBuffer1D, pred: bool = True) -> None: ...
def tcgen05_copy[Rows: IntVar, Cols: IntVar, Layout: IntVar](
    src: TmaSharedTile2D[Rows, Cols, Layout],
    dst: TensorMemoryTileF32[Rows, Cols],
) -> None: ...

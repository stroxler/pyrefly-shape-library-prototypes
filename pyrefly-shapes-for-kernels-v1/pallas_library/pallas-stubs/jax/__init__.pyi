# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from typing import Literal, overload

from jax.numpy import BFloat16Dtype, Float16Dtype, Int32Dtype
from shape_extensions import Int, IntTuple, IntVar

class Array[Shape: IntTuple]:
    @property
    def ndim(self) -> int: ...
    @property
    def device(self) -> object: ...
    def tolist(self) -> object: ...
    @overload
    def __getitem__[Rows: IntVar](
        self: Array[[Rows]], key: tuple[slice, None]
    ) -> Array[[Rows, 1]]: ...
    @overload
    def __getitem__[Batch: IntVar](
        self: Array[[Batch, 128]], key: tuple[slice, Literal[0]]
    ) -> Array[[Batch]]: ...
    # Models the tail slice `a[:, -block:, ...]` used by short convolution.
    # A positive start also matches: numerical signs and block <= Seq are not proved.
    @overload
    def __getitem__[Batch: IntVar, Seq: IntVar, Cols: IntVar, Start: IntVar](
        self: Array[[Batch, Seq, Cols]],
        key: tuple[slice, slice[Int[Start], None, None], slice],
    ) -> Array[[Batch, -Start, Cols]]: ...
    @overload
    def __getitem__[Batch: IntVar, Seq: IntVar, Start: IntVar](
        self: Array[[Batch, Seq]],
        key: tuple[slice, slice[Int[Start], None, None]],
    ) -> Array[[Batch, -Start]]: ...
    @overload
    def __getitem__[Batch: IntVar, Seq: IntVar, Cols: IntVar, Block: IntVar](
        self: Array[[Batch, Seq, Cols]],
        key: tuple[slice, slice[None, Int[Block], None], slice],
    ) -> Array[[Batch, Block, Cols]]: ...
    @overload
    def __getitem__[Batch: IntVar, Seq: IntVar, Block: IntVar](
        self: Array[[Batch, Seq]],
        key: tuple[slice, slice[None, Int[Block], None]],
    ) -> Array[[Batch, Block]]: ...
    @overload
    @property
    def shape[Length: IntVar](self: Array[[Length]]) -> tuple[Int[Length]]: ...
    @overload
    @property
    def shape[Rows: IntVar, Cols: IntVar](
        self: Array[[Rows, Cols]],
    ) -> tuple[Int[Rows], Int[Cols]]: ...
    @overload
    @property
    def shape[Blocks: IntVar, Rows: IntVar, Cols: IntVar](
        self: Array[[Blocks, Rows, Cols]],
    ) -> tuple[Int[Blocks], Int[Rows], Int[Cols]]: ...
    @overload
    @property
    def shape[Batch: IntVar, Rows: IntVar, Heads: IntVar, Dim: IntVar](
        self: Array[[Batch, Rows, Heads, Dim]],
    ) -> tuple[Int[Batch], Int[Rows], Int[Heads], Int[Dim]]: ...
    @overload
    def reshape[SourceRows: IntVar, Devices: IntVar, Rows: IntVar, Cols: IntVar](
        self: Array[[SourceRows, Cols]],
        shape: tuple[Int[Devices], Int[Rows], Int[Cols]],
    ) -> Array[[Devices, Rows, Cols]]: ...
    @overload
    def reshape[Length: IntVar](
        self: Array[[Length]], shape: tuple[Literal[1], Int[Length]]
    ) -> Array[[1, Length]]: ...
    @overload
    def reshape[Rows: IntVar, Cols: IntVar](
        self: Array[[Rows, Cols]],
        rows: Int[Rows // 2],
        cols: Int[2 * Cols],
    ) -> PairedArray[Rows, Cols]: ...
    def swapaxes[Rows: IntVar, Cols: IntVar](
        self: Array[[Rows, Cols]], axis1: Literal[0], axis2: Literal[1]
    ) -> Array[[Cols, Rows]]: ...
    def __truediv__[Rows: IntVar, Cols: IntVar](
        self: Array[[Rows, Cols]], rhs: Array[[Rows, 1]]
    ) -> Array[[Rows, Cols]]: ...
    @property
    def dtype(self) -> object: ...
    @overload
    def astype(self, dtype: BFloat16Dtype) -> BFloatArray[Shape]: ...
    @overload
    def astype(self, dtype: Float16Dtype) -> Float16Array[Shape]: ...
    @overload
    def astype(self, dtype: Int32Dtype) -> Int32Array[Shape]: ...

class RaggedCumulative[Groups: IntVar]:
    @overload
    def __getitem__(self, key: slice[None, Literal[-1], None]) -> Array[[Groups]]: ...
    @overload
    def __getitem__(self, key: slice[Literal[1], None, None]) -> Array[[Groups]]: ...

class BFloatArray[Shape: IntTuple](Array[Shape]): ...
class Float16Array[Shape: IntTuple](Array[Shape]): ...
class Int32Array[Shape: IntTuple](Array[Shape]): ...

class PairedArray[Rows: IntVar, Cols: IntVar](Array[[Rows // 2, 2 * Cols]]):
    def view(self, dtype: Int32Dtype) -> Array[[Rows // 2, Cols]]: ...

class Mesh[Devices: IntVar]: ...
class LeadingAxisPartition: ...
class TrailingAxisPartition: ...
class ReplicatedPartition: ...

class ShardedIndexedAdd[Devices: IntVar, Rows: IntVar, Cols: IntVar]:
    def __call__(
        self,
        x: Array[[Devices * Rows, Cols]],
        index: Int32Array[[1]],
    ) -> Array[[Devices * Rows, Cols // 2]]: ...

class ShardedSparseCoreAdd[Devices: IntVar, Rows: IntVar]:
    def __call__(
        self, x: Array[[Devices * Rows, 128]]
    ) -> Array[[Devices * Rows, 128]]: ...

class MappedRmsNorm[Length: IntVar]:
    def __call__[Rows: IntVar](
        self,
        x: Array[[Rows, Length]],
        weight: Array[[Length]],
        bias: Array[[Length]],
    ) -> tuple[Array[[Rows, Length]], Array[[Rows]]]: ...

class DoubleMappedRmsNorm[Length: IntVar]:
    def __call__[Batch: IntVar, Rows: IntVar](
        self,
        x: Array[[Batch, Rows, Length]],
        weight: Array[[Length]],
        bias: Array[[Length]],
    ) -> tuple[Array[[Batch, Rows, Length]], Array[[Batch, Rows]]]: ...

class MappedUnaryRows[Cols: IntVar]:
    def __call__[Rows: IntVar](self, x: Array[[Rows, Cols]]) -> Array[[Rows, Cols]]: ...

def jit[**P, R](f: Callable[P, R]) -> Callable[P, R]: ...
@overload
def vmap[Cols: IntVar](
    f: Callable[[Array[[Cols]]], Array[[Cols]]],
) -> MappedUnaryRows[Cols]: ...
@overload
def vmap[Length: IntVar](
    f: Callable[
        [Array[[Length]], Array[[Length]], Array[[Length]]],
        tuple[Array[[Length]], Array[[]]],
    ],
    *,
    in_axes: tuple[Literal[0], None, None],
) -> MappedRmsNorm[Length]: ...
@overload
def vmap[Length: IntVar](
    f: MappedRmsNorm[Length],
    *,
    in_axes: tuple[Literal[0], None, None],
) -> DoubleMappedRmsNorm[Length]: ...
def make_mesh[Devices: IntVar](
    shape: tuple[Int[Devices]], axis_names: tuple[Literal["x"]]
) -> Mesh[Devices]: ...
@overload
def P(axis: Literal["x"], other: None) -> LeadingAxisPartition: ...
@overload
def P(axis: Literal["device"], other: None) -> LeadingAxisPartition: ...
@overload
def P(axis: None, other: Literal["x"]) -> TrailingAxisPartition: ...
@overload
def P() -> ReplicatedPartition: ...
@overload
def shard_map[Devices: IntVar, Rows: IntVar, Cols: IntVar](
    f: Callable[[Array[[Rows, Cols]], Int32Array[[1]]], Array[[Rows, Cols // 2]]],
    *,
    mesh: Mesh[Devices],
    in_specs: tuple[LeadingAxisPartition, ReplicatedPartition],
    out_specs: LeadingAxisPartition,
    check_vma: bool = False,
) -> ShardedIndexedAdd[Devices, Rows, Cols]: ...
@overload
def shard_map[Devices: IntVar, Rows: IntVar](
    f: Callable[[Array[[Rows, 128]]], Array[[Rows, 128]]],
    *,
    mesh: Mesh[Devices],
    in_specs: LeadingAxisPartition,
    out_specs: LeadingAxisPartition,
    check_vma: bool = False,
) -> ShardedSparseCoreAdd[Devices, Rows]: ...
@overload
def shard_map[Devices: IntVar, Rows: IntVar, Cols: IntVar](
    f: Callable[[Array[[Rows, Cols]]], Array[[Devices, Rows, Cols]]],
    *,
    mesh: Mesh[Devices],
    in_specs: LeadingAxisPartition,
    out_specs: LeadingAxisPartition,
    check_vma: bool = False,
) -> Callable[
    [Array[[Devices * Rows, Cols]]], Array[[Devices * Devices, Rows, Cols]]
]: ...
@overload
def shard_map[Devices: IntVar, Rows: IntVar, Cols: IntVar](
    f: Callable[
        [Array[[Rows, Cols]]],
        tuple[Array[[Rows, Cols]], Array[[2, Rows, Cols]]],
    ],
    *,
    mesh: Mesh[Devices],
    in_specs: TrailingAxisPartition,
    out_specs: TrailingAxisPartition,
    check_vma: bool = False,
) -> Callable[
    [Array[[Rows, Devices * Cols]]],
    tuple[Array[[Rows, Devices * Cols]], Array[[2, Devices * Rows, Cols]]],
]: ...
@overload
def shard_map[Devices: IntVar, InputRows: IntVar, Rows: IntVar, Cols: IntVar](
    f: Callable[[Array[[InputRows, Cols]]], Array[[Rows, Cols]]],
    *,
    mesh: Mesh[Devices],
    in_specs: TrailingAxisPartition,
    out_specs: LeadingAxisPartition,
    check_vma: bool = False,
) -> Callable[[Array[[InputRows, Devices * Cols]]], Array[[Devices * Rows, Cols]]]: ...

class ShapeDtypeStruct[Shape: IntTuple]:
    @property
    def dtype(self) -> object: ...
    @overload
    @property
    def shape[Length: IntVar](
        self: ShapeDtypeStruct[[Length]],
    ) -> tuple[Int[Length]]: ...
    @overload
    @property
    def shape[Rows: IntVar, Cols: IntVar](
        self: ShapeDtypeStruct[[Rows, Cols]],
    ) -> tuple[Int[Rows], Int[Cols]]: ...
    @classmethod
    def like(cls, array: Array[Shape]) -> ShapeDtypeStruct[Shape]: ...
    @overload
    def __init__(
        self: ShapeDtypeStruct[[]], shape: tuple[()], dtype: object
    ) -> None: ...
    @overload
    def __init__[Length: IntVar](
        self: ShapeDtypeStruct[[Length]], shape: tuple[Int[Length]], dtype: object
    ) -> None: ...
    @overload
    def __init__[Rows: IntVar, Cols: IntVar](
        self: ShapeDtypeStruct[[Rows, Cols]],
        shape: tuple[Int[Rows], Int[Cols]],
        dtype: object,
    ) -> None: ...
    @overload
    def __init__[Devices: IntVar, Rows: IntVar, Cols: IntVar](
        self: ShapeDtypeStruct[[Devices, Rows, Cols]],
        shape: tuple[Int[Devices], Int[Rows], Int[Cols]],
        dtype: object,
    ) -> None: ...
    @overload
    def __init__[Batch: IntVar, Rows: IntVar, Heads: IntVar, Dim: IntVar](
        self: ShapeDtypeStruct[[Batch, Rows, Heads, Dim]],
        shape: tuple[Int[Batch], Int[Rows], Int[Heads], Int[Dim]],
        dtype: object,
    ) -> None: ...

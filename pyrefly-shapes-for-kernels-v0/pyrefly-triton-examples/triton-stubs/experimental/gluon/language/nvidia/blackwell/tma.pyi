# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import overload

from shape_extensions import IntVar
from triton.experimental.gluon.language import (
    BarrierBuffer1D,
    GatherInputDescriptorF16,
    ScatterOutputDescriptorF16,
    TmaInputDescriptor2D,
    TmaOutputDescriptor2D,
    TmaSharedTile2D,
    WgmmaSharedF16,
)
from triton.experimental.gluon.language.nvidia.hopper.tma import (
    async_load as async_load,
    async_store as async_store,
    store_wait as store_wait,
)
from triton.language import tensor

class tensor_descriptor: ...

@overload
def async_gather[
    XMax: IntVar,
    YMax: IntVar,
    BX: IntVar,
    BY: IntVar,
    Layout: IntVar,
](
    tensor_desc: TmaInputDescriptor2D[XMax, YMax, 1, BY, Layout],
    x_offsets: tensor[[BX]],
    y_offset: int,
    barrier: BarrierBuffer1D,
    result: TmaSharedTile2D[BX, BY, Layout],
    pred: bool = True,
    multicast: bool = False,
) -> None: ...
@overload
def async_gather[
    XMax: IntVar,
    K: IntVar,
    BM: IntVar,
    BK: IntVar,
    Layout: IntVar,
](
    tensor_desc: GatherInputDescriptorF16[XMax, K, BM, BK, Layout],
    x_offsets: tensor[[BM]],
    y_offset: int,
    barrier: BarrierBuffer1D,
    result: WgmmaSharedF16[BM, BK, Layout],
    pred: bool = True,
    multicast: bool = False,
) -> None: ...
@overload
def async_scatter[
    XMax: IntVar,
    YMax: IntVar,
    BX: IntVar,
    BY: IntVar,
    Layout: IntVar,
](
    tensor_desc: TmaOutputDescriptor2D[XMax, YMax, 1, BY, Layout],
    x_offsets: tensor[[BX]],
    y_offset: int,
    src: TmaSharedTile2D[BX, BY, Layout],
) -> None: ...
@overload
def async_scatter[
    XMax: IntVar,
    YMax: IntVar,
    BX: IntVar,
    BY: IntVar,
    Layout: IntVar,
](
    tensor_desc: ScatterOutputDescriptorF16[XMax, YMax, BX, BY, Layout],
    x_offsets: tensor[[BX]],
    y_offset: int,
    src: WgmmaSharedF16[BX, BY, Layout],
) -> None: ...

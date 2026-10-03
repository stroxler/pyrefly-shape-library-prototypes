# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal, overload

from shape_extensions import IntVar
from triton.experimental.gluon.language import (
    BarrierBuffer1D,
    BarrierSharedLayout1D,
    MessageSharedBuffer1D,
)

def MBarrierLayout() -> BarrierSharedLayout1D: ...
def init(bar: BarrierBuffer1D, *, count: Literal[1]) -> None: ...
def expect(bar: BarrierBuffer1D, bytes: int, pred: bool = True) -> None: ...
def arrive(
    bar: BarrierBuffer1D, *, count: Literal[1] = 1, pred: bool = True
) -> None: ...
@overload
def wait(bar: BarrierBuffer1D, phase: int, pred: bool = True) -> None: ...
@overload
def wait[Block: IntVar, Layout: IntVar](
    bar: BarrierBuffer1D,
    *,
    phase: Literal[0],
    deps: list[MessageSharedBuffer1D[Block, Layout]],
) -> None: ...
def invalidate(bar: BarrierBuffer1D) -> None: ...

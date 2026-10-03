# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal, overload

from shape_extensions import IntVar
from triton.experimental.gluon.language import WgmmaSharedF16
from triton.language import tensor

class warpgroup_mma_accumulator: ...

def fence_async_shared() -> None: ...
def warpgroup_mma_init[Rows: IntVar, Cols: IntVar](
    accumulator: tensor[[Rows, Cols]],
) -> tensor[[Rows, Cols]]: ...
def warpgroup_mma[M: IntVar, K: IntVar, N: IntVar, LA: IntVar, LB: IntVar](
    a: WgmmaSharedF16[M, K, LA] | tensor[[M, K]],
    b: WgmmaSharedF16[K, N, LB],
    c: tensor[[M, N]],
    *,
    is_async: Literal[True],
    use_acc: Literal[True] = True,
) -> tensor[[M, N]]: ...
@overload
def warpgroup_mma_wait[M: IntVar, N: IntVar](
    *, num_outstanding: Literal[0], deps: tuple[tensor[[M, N]]]
) -> tensor[[M, N]]: ...
@overload
def warpgroup_mma_wait[M: IntVar, N: IntVar](
    num_outstanding: int, deps: tuple[tensor[[M, N]]]
) -> tensor[[M, N]]: ...

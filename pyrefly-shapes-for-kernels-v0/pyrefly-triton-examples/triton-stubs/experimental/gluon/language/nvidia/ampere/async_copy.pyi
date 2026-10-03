# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from shape_extensions import IntVar
from triton.experimental.gluon.language import SharedBuffer1D
from triton.language import InTilePointers, Mask

def async_load[Target: IntVar, Block: IntVar](
    smem: SharedBuffer1D[Block],
    pointer: InTilePointers[[Target], [Block]],
    mask: Mask[[Target], [Block]],
) -> None: ...
def commit_group() -> None: ...
def wait_group(pending: int) -> None: ...

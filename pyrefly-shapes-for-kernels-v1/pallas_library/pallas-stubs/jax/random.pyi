# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal

from jax.experimental.pallas import Tile
from shape_extensions import Int, IntVar

class Key[Impl: str]: ...

def key(
    seed: int, *, impl: Literal["threefry2x32"] = "threefry2x32"
) -> Key[Literal["threefry2x32"]]: ...
def fold_in[Impl: str](key: Key[Impl], data: int) -> Key[Impl]: ...
def bernoulli[Block: IntVar](
    key: Key[Literal["threefry2x32"]], *, p: float, shape: tuple[Int[Block]]
) -> Tile[[Block]]: ...

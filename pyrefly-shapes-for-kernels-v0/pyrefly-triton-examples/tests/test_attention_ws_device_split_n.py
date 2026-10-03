# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The helper body is copied from Triton's fused-attention-ws-device-tma-hopper-or-blackwell.py.
# Triton's LICENSE includes the following notice:
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

# Static-only; semantic annotations are illegal in Triton's JIT.
# @lint-ignore-every AUTODEPS2

"""Check device-TMA attention's recursive head-dimension splitter."""

from typing import assert_type

import triton
import triton.language as tl
from shape_extensions import Int, IntVar


@triton.jit
def _split_n[  # E: Fixpoint iteration did not converge
    Rows: IntVar,
    Cols: IntVar,
    Factor: IntVar,
](x: tl.AttentionSplitTile[Rows, Cols], SPLIT_FACTOR: Int[Factor]):
    if SPLIT_FACTOR == 1:
        return (x,)
    else:
        x0, x1 = x.reshape([x.shape[0], 2, x.shape[1] // 2]).permute(0, 2, 1).split()
        return _split_n(x0, SPLIT_FACTOR // 2) + _split_n(x1, SPLIT_FACTOR // 2)


def test_exact_split_stages[Rows: IntVar, Cols: IntVar](
    tile: tl.AttentionSplitTile[Rows, Cols],
    ordinary: tl.tensor[[Rows, Cols]],
    factor: Int[2],
) -> None:
    halves = (
        tile.reshape([tile.shape[0], 2, tile.shape[1] // 2]).permute(0, 2, 1).split()
    )
    assert_type(
        halves,
        tuple[
            tl.AttentionSplitTile[Rows, Cols // 2],
            tl.AttentionSplitTile[Rows, Cols // 2],
        ],
    )
    tile.reshape([tile.shape[0], 2, tile.shape[1]])  # E: is not assignable
    tile.reshape([tile.shape[0], 3, tile.shape[1] // 2])  # E: is not assignable
    tile.reshape([tile.shape[0], 2, tile.shape[1] // 2]).permute(
        1,  # E: is not assignable
        2,
        0,  # E: is not assignable
    )
    _split_n(ordinary, factor)  # E: is not assignable

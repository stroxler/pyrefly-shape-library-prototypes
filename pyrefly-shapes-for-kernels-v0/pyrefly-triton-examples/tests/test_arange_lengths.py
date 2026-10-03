# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# @lint-ignore-every AUTODEPS2

"""Guard the tile extent of Triton's nonzero-start arange."""

from typing import assert_type

import triton.language as tl
from shape_extensions import Int, IntVar


def test_range_length[Start: IntVar, End: IntVar](
    start: Int[Start], end: Int[End]
) -> None:
    assert_type(tl.arange(0, end), tl.Offsets[[End]])
    assert_type(tl.arange(start, end), tl.Offsets[[End - Start]])
    assert_type(  # E: failed
        tl.arange(start, end),
        tl.Offsets[[End]],
    )


def test_second_half[Half: IntVar](halfway: Int[Half], full: Int[2 * Half]) -> None:
    assert_type(tl.arange(halfway, full), tl.Offsets[[Half]])

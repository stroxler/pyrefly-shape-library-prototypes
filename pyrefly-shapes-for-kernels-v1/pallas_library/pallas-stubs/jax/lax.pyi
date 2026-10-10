# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from typing import Literal, overload

from jax.experimental.pallas import Indices, Mask, Tile
from shape_extensions import Int, IntVar

def axis_index(name: Literal["x", "core", "subcore"]) -> int: ...
def rem(lhs: int, rhs: int) -> int: ...
@overload
def div[Window: IntVar](lhs: Indices[Window], rhs: int) -> Indices[Window]: ...
@overload
def div(lhs: int, rhs: int) -> int: ...
@overload
def fori_loop[Block: IntVar](
    lower: int,
    upper: int,
    body_fun: Callable[[int, Tile[[Block]]], Tile[[Block]]],
    init_val: Tile[[Block]],
) -> Tile[[Block]]: ...
@overload
def fori_loop[Rows: IntVar, Cols: IntVar](
    lower: int,
    upper: int,
    body_fun: Callable[[int, Tile[[Rows, Cols]]], Tile[[Rows, Cols]]],
    init_val: Tile[[Rows, Cols]],
) -> Tile[[Rows, Cols]]: ...
@overload
def fori_loop[Rows: IntVar](
    lower: int,
    upper: int,
    body_fun: Callable[
        [int, tuple[Tile[[Rows, 128]], Tile[[Rows, 128]]]],
        tuple[Tile[[Rows, 128]], Tile[[Rows, 128]]],
    ],
    init_val: tuple[Tile[[Rows, 128]], Tile[[Rows, 128]]],
    *,
    unroll: Literal[True],
) -> tuple[Tile[[Rows, 128]], Tile[[Rows, 128]]]: ...
@overload
def fori_loop[Block: IntVar](
    lower: int,
    upper: int,
    body_fun: Callable[
        [int, tuple[Tile[[Block]], Tile[[Block]]]],
        tuple[Tile[[Block]], Tile[[Block]]],
    ],
    init_val: tuple[Tile[[Block]], Tile[[Block]]],
) -> tuple[Tile[[Block]], Tile[[Block]]]: ...
@overload
def fori_loop[Rows: IntVar, Cols: IntVar](
    lower: int,
    upper: int,
    body_fun: Callable[
        [int, tuple[Tile[[Rows, Cols]], Tile[[Rows, Cols]]]],
        tuple[Tile[[Rows, Cols]], Tile[[Rows, Cols]]],
    ],
    init_val: tuple[Tile[[Rows, Cols]], Tile[[Rows, Cols]]],
) -> tuple[Tile[[Rows, Cols]], Tile[[Rows, Cols]]]: ...
@overload
def fori_loop[Heads: IntVar, Dim: IntVar](
    lower: int,
    upper: int,
    body_fun: Callable[
        [int, tuple[Tile[[Heads, Dim]], Tile[[Heads]], Tile[[Heads]]]],
        tuple[Tile[[Heads, Dim]], Tile[[Heads]], Tile[[Heads]]],
    ],
    init_val: tuple[Tile[[Heads, Dim]], Tile[[Heads]], Tile[[Heads]]],
) -> tuple[Tile[[Heads, Dim]], Tile[[Heads]], Tile[[Heads]]]: ...
@overload
def dot_general[Rows: IntVar, Inner: IntVar, Cols: IntVar](
    lhs: Tile[[Rows, Inner]],
    rhs: Tile[[Inner, Cols]],
    dimension_numbers: tuple[
        tuple[tuple[Literal[1]], tuple[Literal[0]]], tuple[tuple[()], tuple[()]]
    ],
    *,
    preferred_element_type: object | None,
    precision: object = None,
) -> Tile[[Rows, Cols]]: ...
@overload
def dot_general[Rows: IntVar, Inner: IntVar, Cols: IntVar](
    lhs: Tile[[Rows, Inner]],
    rhs: Tile[[Cols, Inner]],
    dimension_numbers: tuple[
        tuple[tuple[Literal[1]], tuple[Literal[1]]], tuple[tuple[()], tuple[()]]
    ],
    *,
    preferred_element_type: object | None,
    precision: object = None,
) -> Tile[[Rows, Cols]]: ...
@overload
def broadcast_in_dim[Rows: IntVar](
    value: Tile[[Rows]],
    shape: tuple[Int[Rows], Literal[128]],
    broadcast_dimensions: tuple[Literal[0]],
) -> Tile[[Rows, 128]]: ...
@overload
def broadcast_in_dim[Heads: IntVar, Block: IntVar, Length: IntVar](
    value: Mask[[Block], [Length]],
    shape: tuple[Int[Heads], Int[Block]],
    broadcast_dimensions: tuple[Literal[1]],
) -> Mask[[Heads, Block], [Heads, Length]]: ...
def cond[T](
    predicate: bool, true_branch: Callable[[], T], false_branch: Callable[[], T]
) -> T: ...

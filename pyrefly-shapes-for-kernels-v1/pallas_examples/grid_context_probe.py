"""Probe grid-driven contextual typing of Pallas index-map lambdas."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, assert_type, cast

from jax.experimental import pallas as pl
from shape_extensions import Int, IntVar


class BlockIndex[Extent: IntVar, Block: IntVar](int):
    """The ordinal index of a block along a host-array axis."""


if TYPE_CHECKING:

    def checked_layout[
        Rows: IntVar,
        Inner: IntVar,
        Cols: IntVar,
        RowBlock: IntVar,
        ColBlock: IntVar,
    ](
        grid: tuple[pl.GridSize[Rows, RowBlock], pl.GridSize[Cols, ColBlock]],
        x_block: tuple[Int[RowBlock], Int[Inner]],
        y_block: tuple[Int[Inner], Int[ColBlock]],
        out_block: tuple[Int[RowBlock], Int[ColBlock]],
        x_map: Callable[
            [BlockIndex[Rows, RowBlock], BlockIndex[Cols, ColBlock]],
            tuple[BlockIndex[Rows, RowBlock], Literal[0]],
        ],
        y_map: Callable[
            [BlockIndex[Rows, RowBlock], BlockIndex[Cols, ColBlock]],
            tuple[Literal[0], BlockIndex[Cols, ColBlock]],
        ],
        out_map: Callable[
            [BlockIndex[Rows, RowBlock], BlockIndex[Cols, ColBlock]],
            tuple[BlockIndex[Rows, RowBlock], BlockIndex[Cols, ColBlock]],
        ],
    ) -> None: ...

    rows: Int[6] = cast(Int[6], 6)
    row_block: Int[3] = cast(Int[3], 3)
    inner: Int[8] = cast(Int[8], 8)
    cols: Int[10] = cast(Int[10], 10)
    col_block: Int[5] = cast(Int[5], 5)
    checked_layout(
        grid=(pl.cdiv(rows, row_block), pl.cdiv(cols, col_block)),
        x_block=(row_block, inner),
        y_block=(inner, col_block),
        out_block=(row_block, col_block),
        x_map=lambda i, j: (
            (assert_type(i, BlockIndex[6, 3]), i)[1],
            0,
        ),
        y_map=lambda i, j: (0, j),
        out_map=lambda i, j: (i, j),
    )
    # This call must fail unless the first result preserves the row grid index.
    checked_layout(
        grid=(pl.cdiv(rows, row_block), pl.cdiv(cols, col_block)),
        x_block=(row_block, inner),
        y_block=(inner, col_block),
        out_block=(row_block, col_block),
        x_map=lambda i, j: (0, 0),  # pyrefly: ignore[bad-argument-type]
        y_map=lambda i, j: (0, j),
        out_map=lambda i, j: (i, j),
    )

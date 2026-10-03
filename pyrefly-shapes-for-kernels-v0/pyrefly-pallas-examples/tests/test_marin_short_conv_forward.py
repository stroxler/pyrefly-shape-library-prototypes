# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The six complete host helpers and kernel functions come from Marin
# lib/levanter/src/levanter/kernels/pallas/short_conv/pallas_gpu.py;
# only their parameter annotations are replaced or added.
# @lint-ignore-every AUTODEPS2

"""Marin's short convolution routes padded head and dynamic lag windows."""

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar

    Seq = IntVar("Seq")
    Batch = IntVar("Batch")
    Width = IntVar("Width")
    BlockSeq = IntVar("BlockSeq")
    Channels = IntVar("Channels")
    Halo = IntVar("Halo")

OOB_SEGMENT = -1


def _mul_round(
    weight_row: pl.ConvWeightRow[Channels],
    tile: pl.Tile[[BlockSeq, Channels]],
    dtype: object,
    exact: bool,
) -> pl.Tile[[BlockSeq, Channels]]:
    """``weight_row[None, :] * tile`` with the reference's rounding."""
    product = weight_row[None, :].astype(jnp.float32) * tile.astype(jnp.float32)
    return product.astype(dtype) if exact else product


def _add_round(
    acc: pl.Tile[[BlockSeq, Channels]] | None,
    term: pl.Tile[[BlockSeq, Channels]],
    dtype: object,
    exact: bool,
) -> pl.Tile[[BlockSeq, Channels]]:
    if acc is None:
        return term
    total = acc.astype(jnp.float32) + term.astype(jnp.float32)
    return total.astype(dtype) if exact else total


def _keep(
    seg_shifted: pl.ConvSegments[BlockSeq],
    seg_cur: pl.ConvSegments[BlockSeq],
    tile: pl.Tile[[BlockSeq, Channels]],
) -> pl.Tile[[BlockSeq, Channels]]:
    return jnp.where((seg_shifted == seg_cur)[:, None], tile, jnp.zeros_like(tile))


def _fwd_body(
    vals_ref: pl.ConvValuesRef[Seq, Channels],
    segs_ref: pl.ConvSegmentRef[Seq],
    w_ref: pl.ConvWeightRef[Width, Channels],
    out_ref: pl.ConvOutRef[BlockSeq, Channels],
    base: int,
    block_seq: Int[BlockSeq],
    kernel_size: Int[Width],
    exact: bool,
) -> None:
    """One block's forward, reading taps at ``base - lag`` out of ``vals_ref``.

    ``base`` is the row in ``vals_ref`` corresponding to this block's first output row, so
    the general path passes ``si * BS`` into the whole-sequence view and the first-block
    path passes ``W - 1`` into the pre-padded head view. Both then read identically.
    """
    seg_cur = segs_ref[0, pl.ds(base, block_seq)]
    tile = vals_ref[0, pl.ds(base, block_seq), :]
    dtype = tile.dtype
    # Ascending lags, left-nested, rounding after every op: the reference's exact order.
    acc = _mul_round(w_ref[0], tile, dtype, exact)
    for lag in range(1, kernel_size):
        shifted = vals_ref[0, pl.ds(base - lag, block_seq), :]
        seg_shifted = segs_ref[0, pl.ds(base - lag, block_seq)]
        shifted = _keep(seg_shifted, seg_cur, shifted)
        acc = _add_round(
            acc, _mul_round(w_ref[lag], shifted, dtype, exact), dtype, exact
        )
    out_ref[0] = acc.astype(dtype)


def _fwd_kernel(
    x_ref: pl.ConvValuesRef[Seq, Channels],
    xh_ref: pl.ConvValuesRef[Halo, Channels],
    seg_ref: pl.ConvSegmentRef[Seq],
    segh_ref: pl.ConvSegmentRef[Halo],
    w_ref: pl.ConvWeightRef[Width, Channels],
    out_ref: pl.ConvOutRef[BlockSeq, Channels],
    *,
    kernel_size: Int[Width],
    exact: bool,
):
    block_seq = out_ref.shape[1]
    si = pl.program_id(1)

    @pl.when(si == 0)
    def _first():
        _fwd_body(
            xh_ref,
            segh_ref,
            w_ref,
            out_ref,
            kernel_size - 1,
            block_seq,
            kernel_size,
            exact,
        )

    @pl.when(si != 0)
    def _general():
        _fwd_body(
            x_ref,
            seg_ref,
            w_ref,
            out_ref,
            si * block_seq,
            block_seq,
            kernel_size,
            exact,
        )


def _head_views(
    x: jax.Array[[Batch, Seq, Channels]],
    segment_ids: jax.Array[[Batch, Seq]],
    block_seq: Int[BlockSeq],
    width: Int[Width],
) -> tuple[
    jax.Array[[Batch, BlockSeq + Width - 1, Channels]],
    jax.Array[[Batch, BlockSeq + Width - 1]],
]:
    """``[B, BS+W-1, C]`` / ``[B, BS+W-1]``: ``W-1`` pad rows then the first ``BS`` rows.

    Exactly what ``jnp.pad(x, ((0,0),(W-1,0),(0,0)))`` would give for those rows, so the
    first-block path is the reference's semantics with no special casing inside the kernel.
    """
    pad_vals = jnp.zeros((x.shape[0], width - 1, x.shape[2]), x.dtype)
    pad_segs = jnp.full(
        (segment_ids.shape[0], width - 1), OOB_SEGMENT, segment_ids.dtype
    )
    return (
        jnp.concatenate([pad_vals, x[:, :block_seq, :]], axis=1),
        jnp.concatenate([pad_segs, segment_ids[:, :block_seq]], axis=1),
    )


def short_conv_forward[
    Batch: IntVar,
    Rows: IntVar,
    Features: IntVar,
    Width: IntVar,
    Block: IntVar,
    ColumnBlock: IntVar,
](
    block_seq: Int[Block],
    block_channels: Int[ColumnBlock],
    width: Int[Width],
    x: jax.Array[[Batch, Rows, Features]],
    segment_ids: jax.Array[[Batch, Rows]],
    weights: jax.Array[[Width, Features]],
    exact: bool,
) -> jax.Array[[Batch, Rows, Features]]:
    x_head, seg_head = _head_views(x, segment_ids, block_seq, width)

    def kernel(
        x_ref: pl.ConvValuesRef[Rows, ColumnBlock],
        xh_ref: pl.ConvValuesRef[Block + Width - 1, ColumnBlock],
        seg_ref: pl.ConvSegmentRef[Rows],
        segh_ref: pl.ConvSegmentRef[Block + Width - 1],
        w_ref: pl.ConvWeightRef[Width, ColumnBlock],
        out_ref: pl.ConvOutRef[Block, ColumnBlock],
    ) -> None:
        _fwd_kernel(
            x_ref,
            xh_ref,
            seg_ref,
            segh_ref,
            w_ref,
            out_ref,
            kernel_size=width,
            exact=exact,
        )

    whole = lambda b, si, ci: (b, 0, ci)  # noqa: E731
    whole_1d = lambda b, si, ci: (b, 0)  # noqa: E731
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(x.shape[0], x.shape[1] // block_seq, x.shape[2] // block_channels),
        in_specs=(
            pl.BlockSpec((1, x.shape[1], block_channels), whole),
            pl.BlockSpec((1, block_seq + width - 1, block_channels), whole),
            pl.BlockSpec((1, x.shape[1]), whole_1d),
            pl.BlockSpec((1, block_seq + width - 1), whole_1d),
            pl.BlockSpec((width, block_channels), lambda b, si, ci: (0, ci)),
        ),
        out_specs=pl.BlockSpec(
            (1, block_seq, block_channels), lambda b, si, ci: (b, si, ci)
        ),
        interpret=True,
    )(x, x_head, segment_ids, seg_head, weights)


def test_correct_shapes[
    Batch: IntVar,
    Rows: IntVar,
    Features: IntVar,
    Width: IntVar,
    Block: IntVar,
    ColumnBlock: IntVar,
](
    block_seq: Int[Block],
    block_channels: Int[ColumnBlock],
    width: Int[Width],
    x: jax.Array[[Batch, Rows, Features]],
    segments: jax.Array[[Batch, Rows]],
    weights: jax.Array[[Width, Features]],
) -> None:
    assert_type(
        short_conv_forward(
            block_seq, block_channels, width, x, segments, weights, True
        ),
        jax.Array[[Batch, Rows, Features]],
    )


def test_wrong_segment_rows[
    Batch: IntVar,
    Rows: IntVar,
    Features: IntVar,
    Width: IntVar,
    Block: IntVar,
    ColumnBlock: IntVar,
    Other: IntVar,
](
    block_seq: Int[Block],
    block_channels: Int[ColumnBlock],
    width: Int[Width],
    x: jax.Array[[Batch, Rows, Features]],
    segments: jax.Array[[Batch, Other]],
    weights: jax.Array[[Width, Features]],
) -> None:
    short_conv_forward(
        block_seq,
        block_channels,
        width,
        x,
        segments,  # E: is not assignable
        weights,
        True,
    )


def test_wrong_head_segments[
    Batch: IntVar,
    Rows: IntVar,
    Features: IntVar,
    Width: IntVar,
    Block: IntVar,
    Other: IntVar,
](
    x: jax.Array[[Batch, Rows, Features]],
    segments: jax.Array[[Batch, Other]],
    block_seq: Int[Block],
    width: Int[Width],
) -> None:
    _head_views(x, segments, block_seq, width)  # E: is not assignable


def test_wrong_head_padding[
    Batch: IntVar,
    Rows: IntVar,
    Features: IntVar,
    Width: IntVar,
    Block: IntVar,
](
    x: jax.Array[[Batch, Rows, Features]],
    block_seq: Int[Block],
    width: Int[Width],
) -> None:
    pad = jnp.zeros((x.shape[0], width, x.shape[2]), x.dtype)
    invalid = jnp.concatenate([pad, x[:, :block_seq, :]], axis=1)
    assert_type(  # E: assert_type
        invalid,
        jax.Array[[Batch, Block + Width - 1, Features]],
    )


def test_reversed_head_chunks_are_not_rejected[
    Batch: IntVar,
    Rows: IntVar,
    Features: IntVar,
    Width: IntVar,
    Block: IntVar,
](
    x: jax.Array[[Batch, Rows, Features]],
    block_seq: Int[Block],
    width: Int[Width],
) -> None:
    pad = jnp.zeros((x.shape[0], width - 1, x.shape[2]), x.dtype)
    reordered = jnp.concatenate([x[:, :block_seq, :], pad], axis=1)
    assert_type(reordered, jax.Array[[Batch, Block + Width - 1, Features]])


def test_wrong_weight_channels[
    Batch: IntVar,
    Rows: IntVar,
    Features: IntVar,
    Width: IntVar,
    Block: IntVar,
    ColumnBlock: IntVar,
    Other: IntVar,
](
    block_seq: Int[Block],
    block_channels: Int[ColumnBlock],
    width: Int[Width],
    x: jax.Array[[Batch, Rows, Features]],
    segments: jax.Array[[Batch, Rows]],
    weights: jax.Array[[Width, Other]],
) -> None:
    short_conv_forward(
        block_seq,
        block_channels,
        width,
        x,
        segments,
        weights,  # E: is not assignable
        True,
    )


def test_wrong_helper_segment_shape[Block: IntVar, Features: IntVar, Other: IntVar](
    shifted: pl.ConvSegments[Other],
    current: pl.ConvSegments[Block],
    values: pl.Tile[[Other, Features]],
) -> None:
    _keep(shifted, current, values)  # E: is not assignable


def test_wrong_window_segments[
    Rows: IntVar,
    Other: IntVar,
    Width: IntVar,
    Block: IntVar,
    Features: IntVar,
](
    values: pl.ConvValuesRef[Rows, Features],
    segments: pl.ConvSegmentRef[Other],
    weights: pl.ConvWeightRef[Width, Features],
    output: pl.ConvOutRef[Block, Features],
    block_size: Int[Block],
    width: Int[Width],
) -> None:
    _fwd_body(
        values,
        segments,  # E: is not assignable
        weights,
        output,
        0,
        block_size,
        width,
        True,
    )


def test_wrong_output_block_shape[Block: IntVar, Features: IntVar, Other: IntVar](
    output: pl.ConvOutRef[Block, Features],
    values: pl.Tile[[Other, Features]],
) -> None:
    output[0] = values  # E: Cannot set item


if not TYPE_CHECKING:
    import unittest

    class MarinShortConvBoundaryTest(unittest.TestCase):
        def test_padded_head_and_segment_mask_reference(self) -> None:
            x = jnp.arange(2 * 5 * 3, dtype=jnp.float32).reshape(2, 5, 3)
            seg = jnp.array([[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]])
            w = jnp.ones((3, 3), dtype=jnp.float32)
            head, head_seg = _head_views(x, seg, 4, 3)
            self.assertEqual(head.shape, (2, 6, 3))
            self.assertEqual(head_seg.shape, (2, 6))
            out = jnp.zeros_like(x)
            for lag in range(3):
                shifted = jnp.pad(x, ((0, 0), (lag, 0), (0, 0)))[:, :5]
                shifted_seg = jnp.pad(seg, ((0, 0), (lag, 0)), constant_values=-1)[
                    :, :5
                ]
                out += jnp.where((shifted_seg == seg)[..., None], shifted * w[lag], 0)
            self.assertEqual(out.shape, x.shape)
            self.assertTrue(bool(jnp.array_equal(out[0, 0], x[0, 0])))

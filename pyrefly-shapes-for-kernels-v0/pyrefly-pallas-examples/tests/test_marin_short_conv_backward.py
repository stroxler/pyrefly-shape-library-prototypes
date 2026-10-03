# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The four complete functions below come from Marin
# lib/levanter/src/levanter/kernels/pallas/short_conv/pallas_gpu.py;
# only their parameter annotations are replaced or added.
# @lint-ignore-every AUTODEPS2

"""Marin's backward short convolution reads a tail halo and writes partial weights."""

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from tests.test_marin_short_conv_forward import (
    _add_round,
    _head_views,
    _keep,
    _mul_round,
    OOB_SEGMENT,
)

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar

    Seq = IntVar("Seq")
    Batch = IntVar("Batch")
    Width = IntVar("Width")
    BlockSeq = IntVar("BlockSeq")
    Channels = IntVar("Channels")
    Halo = IntVar("Halo")


def _dx_body(
    dy_ref: pl.ConvValuesRef[Seq, Channels],
    segs_ref: pl.ConvSegmentRef[Seq],
    w_ref: pl.ConvWeightRef[Width, Channels],
    dx_ref: pl.ConvOutRef[BlockSeq, Channels],
    base: int,
    block_seq: Int[BlockSeq],
    kernel_size: Int[Width],
    exact: bool,
) -> None:
    """``dx[t] = sum_lag w[lag] * [seg[t] == seg[t+lag]] * dy[t+lag]``.

    Descending lags then tap 0, because that is the order JAX's transpose of the forward
    produces and matching it is what makes ``dx`` bit-identical rather than merely close.
    """
    seg_cur = segs_ref[0, pl.ds(base, block_seq)]
    dy = dy_ref[0, pl.ds(base, block_seq), :]
    dtype = dy.dtype
    acc = None
    for lag in range(kernel_size - 1, 0, -1):
        dy_ahead = dy_ref[0, pl.ds(base + lag, block_seq), :]
        seg_ahead = segs_ref[0, pl.ds(base + lag, block_seq)]
        term = _mul_round(w_ref[lag], dy_ahead, dtype, exact)
        acc = _add_round(acc, _keep(seg_ahead, seg_cur, term), dtype, exact)
    dx_ref[0] = _add_round(
        acc, _mul_round(w_ref[0], dy, dtype, exact), dtype, exact
    ).astype(dtype)


def _dw_body(
    x_ref: pl.ConvValuesRef[Seq, Channels],
    segs_ref: pl.ConvSegmentRef[Seq],
    dy: pl.Tile[[BlockSeq, Channels]],
    dw_ref: pl.ConvPartialWeightRef[Width, Channels],
    base: int,
    block_seq: Int[BlockSeq],
    kernel_size: Int[Width],
) -> None:
    """``dw[lag] = sum_t dy[t] * [seg[t-lag] == seg[t]] * x[t-lag]``, fp32, in registers.

    The shifted-``x`` construction is the forward's, so the mask semantics are shared by
    construction rather than by a comment asking you to keep them in sync.
    """
    seg_cur = segs_ref[0, pl.ds(base, block_seq)]
    dy_f32 = dy.astype(jnp.float32)
    for lag in range(kernel_size):
        shifted = x_ref[0, pl.ds(base - lag, block_seq), :]
        if lag:
            shifted = _keep(segs_ref[0, pl.ds(base - lag, block_seq)], seg_cur, shifted)
        partial = jnp.sum(dy_f32 * shifted.astype(jnp.float32), axis=0)
        # Store per tap rather than stacking: Triton's `stack` lowering takes exactly two
        # operands, and a W-way stack would build non-power-of-2 intermediates anyway.
        dw_ref[0, pl.ds(lag, 1), :] = partial[None, :]


def _bwd_kernel(
    x_ref: pl.ConvValuesRef[Seq, Channels],
    xh_ref: pl.ConvValuesRef[Halo, Channels],
    seg_ref: pl.ConvSegmentRef[Seq],
    segh_ref: pl.ConvSegmentRef[Halo],
    dy_ref: pl.ConvValuesRef[Seq, Channels],
    dyt_ref: pl.ConvValuesRef[Halo, Channels],
    segt_ref: pl.ConvSegmentRef[Halo],
    w_ref: pl.ConvWeightRef[Width, Channels],
    dx_ref: pl.ConvOutRef[BlockSeq, Channels],
    dw_partial_ref: pl.ConvPartialWeightRef[Width, Channels],
    *,
    kernel_size: Int[Width],
    exact: bool,
):
    block_seq = dx_ref.shape[1]
    si = pl.program_id(1)
    last = pl.num_programs(1) - 1

    # dx reads dy *ahead* of this block, so only the final block needs the tail view.
    @pl.when(si != last)
    def _dx_general():
        _dx_body(
            dy_ref,
            seg_ref,
            w_ref,
            dx_ref,
            si * block_seq,
            block_seq,
            kernel_size,
            exact,
        )

    @pl.when(si == last)
    def _dx_last():
        _dx_body(dyt_ref, segt_ref, w_ref, dx_ref, 0, block_seq, kernel_size, exact)

    # dw reads x *behind* this block, so only the first block needs the head view.
    @pl.when(si != 0)
    def _dw_general():
        _dw_body(
            x_ref,
            seg_ref,
            dy_ref[0, pl.ds(si * block_seq, block_seq), :],
            dw_partial_ref,
            si * block_seq,
            block_seq,
            kernel_size,
        )

    @pl.when(si == 0)
    def _dw_first():
        _dw_body(
            xh_ref,
            segh_ref,
            dy_ref[0, pl.ds(0, block_seq), :],
            dw_partial_ref,
            kernel_size - 1,
            block_seq,
            kernel_size,
        )


def _tail_views(
    dy: jax.Array[[Batch, Seq, Channels]],
    segment_ids: jax.Array[[Batch, Seq]],
    block_seq: Int[BlockSeq],
    width: Int[Width],
):
    """``[B, BS+W-1, C]`` / ``[B, BS+W-1]``: the last ``BS`` rows then ``W-1`` pad rows.

    The anti-causal mirror of ``_head_views``. Positions past the end contribute nothing to
    ``dx``, so the pad value is zero and its segment id can never match.
    """
    pad_vals = jnp.zeros((dy.shape[0], width - 1, dy.shape[2]), dy.dtype)
    pad_segs = jnp.full(
        (segment_ids.shape[0], width - 1), OOB_SEGMENT, segment_ids.dtype
    )
    return (
        jnp.concatenate([dy[:, -block_seq:, :], pad_vals], axis=1),
        jnp.concatenate([segment_ids[:, -block_seq:], pad_segs], axis=1),
    )


def short_conv_backward[
    B: IntVar,
    S: IntVar,
    C: IntVar,
    W: IntVar,
    BS: IntVar,
    BC: IntVar,
](
    block_seq: Int[BS],
    block_channels: Int[BC],
    width: Int[W],
    x: jax.Array[[B, S, C]],
    segment_ids: jax.Array[[B, S]],
    weight: jax.Array[[W, C]],
    dy: jax.Array[[B, S, C]],
    exact: bool,
) -> tuple[jax.Array[[B, S, C]], jax.Array[[B * (S // BS), W, C]]]:
    x_head, seg_head = _head_views(x, segment_ids, block_seq, width)
    dy_tail, seg_tail = _tail_views(dy, segment_ids, block_seq, width)

    def kernel(
        x_ref: pl.ConvValuesRef[S, BC],
        xh_ref: pl.ConvValuesRef[BS + W - 1, BC],
        seg_ref: pl.ConvSegmentRef[S],
        segh_ref: pl.ConvSegmentRef[BS + W - 1],
        dy_ref: pl.ConvValuesRef[S, BC],
        dyt_ref: pl.ConvValuesRef[BS + W - 1, BC],
        segt_ref: pl.ConvSegmentRef[BS + W - 1],
        w_ref: pl.ConvWeightRef[W, BC],
        dx_ref: pl.ConvOutRef[BS, BC],
        dw_ref: pl.ConvPartialWeightRef[W, BC],
    ) -> None:
        _bwd_kernel(
            x_ref,
            xh_ref,
            seg_ref,
            segh_ref,
            dy_ref,
            dyt_ref,
            segt_ref,
            w_ref,
            dx_ref,
            dw_ref,
            kernel_size=width,
            exact=exact,
        )

    whole = lambda b, si, ci: (b, 0, ci)  # noqa: E731
    whole_1d = lambda b, si, ci: (b, 0)  # noqa: E731
    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct(x.shape, dy.dtype),
            jax.ShapeDtypeStruct(
                (x.shape[0] * (x.shape[1] // block_seq), width, x.shape[2]),
                jnp.float32,
            ),
        ),
        grid=(x.shape[0], x.shape[1] // block_seq, x.shape[2] // block_channels),
        in_specs=(
            pl.BlockSpec((1, x.shape[1], block_channels), whole),
            pl.BlockSpec((1, block_seq + width - 1, block_channels), whole),
            pl.BlockSpec((1, x.shape[1]), whole_1d),
            pl.BlockSpec((1, block_seq + width - 1), whole_1d),
            pl.BlockSpec((1, x.shape[1], block_channels), whole),
            pl.BlockSpec((1, block_seq + width - 1, block_channels), whole),
            pl.BlockSpec((1, block_seq + width - 1), whole_1d),
            pl.BlockSpec((width, block_channels), lambda b, si, ci: (0, ci)),
        ),
        out_specs=(
            pl.BlockSpec((1, block_seq, block_channels), lambda b, si, ci: (b, si, ci)),
            pl.BlockSpec(
                (1, width, block_channels),
                lambda b, si, ci: (b * (x.shape[1] // block_seq) + si, 0, ci),
            ),
        ),
        interpret=True,
    )(x, x_head, segment_ids, seg_head, dy, dy_tail, seg_tail, weight)


def test_correct_backward[
    B: IntVar,
    S: IntVar,
    C: IntVar,
    W: IntVar,
    BS: IntVar,
    BC: IntVar,
](
    block_seq: Int[BS],
    block_channels: Int[BC],
    width: Int[W],
    x: jax.Array[[B, S, C]],
    seg: jax.Array[[B, S]],
    weight: jax.Array[[W, C]],
    dy: jax.Array[[B, S, C]],
) -> None:
    assert_type(
        short_conv_backward(block_seq, block_channels, width, x, seg, weight, dy, True),
        tuple[jax.Array[[B, S, C]], jax.Array[[B * (S // BS), W, C]]],
    )


def test_wrong_backward_dy[
    B: IntVar,
    S: IntVar,
    C: IntVar,
    W: IntVar,
    BS: IntVar,
    BC: IntVar,
    Other: IntVar,
](
    block_seq: Int[BS],
    block_channels: Int[BC],
    width: Int[W],
    x: jax.Array[[B, S, C]],
    seg: jax.Array[[B, S]],
    weight: jax.Array[[W, C]],
    dy: jax.Array[[B, Other, C]],
) -> None:
    short_conv_backward(
        block_seq,
        block_channels,
        width,
        x,
        seg,
        weight,
        dy,  # E: is not assignable
        True,
    )


def test_wrong_backward_segment[
    B: IntVar,
    S: IntVar,
    C: IntVar,
    W: IntVar,
    BS: IntVar,
    BC: IntVar,
    Other: IntVar,
](
    block_seq: Int[BS],
    block_channels: Int[BC],
    width: Int[W],
    x: jax.Array[[B, S, C]],
    seg: jax.Array[[B, Other]],
    weight: jax.Array[[W, C]],
    dy: jax.Array[[B, S, C]],
) -> None:
    short_conv_backward(
        block_seq,
        block_channels,
        width,
        x,
        seg,  # E: is not assignable
        weight,
        dy,
        True,
    )


def test_wrong_partial_store[W: IntVar, C: IntVar, Other: IntVar](
    dw_ref: pl.ConvPartialWeightRef[W, C],
    partial: pl.Tile[[1, Other]],
) -> None:
    dw_ref[0, pl.ds(0, 1), :] = partial  # E: Cannot set item


def test_wrong_dx_store[BS: IntVar, C: IntVar, Other: IntVar](
    dx_ref: pl.ConvOutRef[BS, C],
    value: pl.Tile[[Other, C]],
) -> None:
    dx_ref[0] = value  # E: Cannot set item


def test_wrong_tail_padding[B: IntVar, S: IntVar, C: IntVar, W: IntVar, BS: IntVar](
    dy: jax.Array[[B, S, C]],
    block_seq: Int[BS],
    width: Int[W],
) -> None:
    padding = jnp.zeros((dy.shape[0], width, dy.shape[2]), dy.dtype)
    invalid = jnp.concatenate([dy[:, -block_seq:, :], padding], axis=1)
    assert_type(  # E: assert_type
        invalid,
        jax.Array[[B, BS + W - 1, C]],
    )


def test_wrong_partial_extent[
    B: IntVar,
    S: IntVar,
    C: IntVar,
    W: IntVar,
    BS: IntVar,
    BC: IntVar,
](
    block_seq: Int[BS],
    block_channels: Int[BC],
    width: Int[W],
    x: jax.Array[[B, S, C]],
    seg: jax.Array[[B, S]],
    weight: jax.Array[[W, C]],
    dy: jax.Array[[B, S, C]],
) -> None:
    _, grad = short_conv_backward(
        block_seq, block_channels, width, x, seg, weight, dy, True
    )
    assert_type(  # E: assert_type
        grad,
        jax.Array[[B * (S // BS), W + 1, C]],
    )


def test_wrong_partial_block_count[
    B: IntVar,
    S: IntVar,
    C: IntVar,
    W: IntVar,
    BS: IntVar,
    BC: IntVar,
](
    block_seq: Int[BS],
    block_channels: Int[BC],
    width: Int[W],
    x: jax.Array[[B, S, C]],
    seg: jax.Array[[B, S]],
    weight: jax.Array[[W, C]],
    dy: jax.Array[[B, S, C]],
) -> None:
    _, grad = short_conv_backward(
        block_seq, block_channels, width, x, seg, weight, dy, True
    )
    assert_type(  # E: assert_type
        grad,
        jax.Array[[B * (S // BS) + 1, W, C]],
    )


def test_wrong_segment_window[
    S: IntVar,
    Other: IntVar,
    W: IntVar,
    BS: IntVar,
    C: IntVar,
](
    dy: pl.ConvValuesRef[S, C],
    seg: pl.ConvSegmentRef[Other],
    weight: pl.ConvWeightRef[W, C],
    dx: pl.ConvOutRef[BS, C],
    block_seq: Int[BS],
    width: Int[W],
) -> None:
    _dx_body(dy, seg, weight, dx, 0, block_seq, width, False)  # E: is not assignable


def test_reversed_tail_chunks_are_not_rejected[
    B: IntVar,
    S: IntVar,
    C: IntVar,
    W: IntVar,
    BS: IntVar,
](
    dy: jax.Array[[B, S, C]],
    block_seq: Int[BS],
    width: Int[W],
) -> None:
    padding = jnp.zeros((dy.shape[0], width - 1, dy.shape[2]), dy.dtype)
    reordered = jnp.concatenate([padding, dy[:, -block_seq:, :]], axis=1)
    assert_type(reordered, jax.Array[[B, BS + W - 1, C]])


if not TYPE_CHECKING:
    import unittest

    import numpy as np

    class MarinShortConvBackwardBoundaryTest(unittest.TestCase):
        def test_tail_halo_and_partial_shape(self) -> None:
            x = jnp.arange(2 * 4 * 3, dtype=jnp.float32).reshape(2, 4, 3)
            seg = jnp.array([[0, 0, 1, 1], [0, 1, 1, 1]])
            tail, tail_seg = _tail_views(x, seg, 2, 3)
            self.assertEqual(tail.shape, (2, 4, 3))
            self.assertEqual(tail_seg.shape, (2, 4))
            self.assertTrue(bool(jnp.array_equal(tail[:, :2], x[:, -2:])))
            self.assertTrue(bool(jnp.all(tail[:, 2:] == 0)))
            self.assertTrue(bool(jnp.all(tail_seg[:, 2:] == -1)))
            dx, dw_partials = short_conv_backward(
                2,
                3,
                3,
                x,
                seg,
                jnp.ones((3, 3), dtype=x.dtype),
                x,
                False,
            )
            self.assertEqual(dx.shape, x.shape)
            self.assertEqual(dw_partials.shape, (4, 3, 3))
            reference_dx = np.zeros_like(np.asarray(x))
            reference_dw = np.zeros_like(np.asarray(dw_partials))
            for batch in range(2):
                for row in range(4):
                    for lag in range(3):
                        if row + lag < 4 and seg[batch, row] == seg[batch, row + lag]:
                            reference_dx[batch, row] += np.asarray(x[batch, row + lag])
                        if row >= lag and seg[batch, row] == seg[batch, row - lag]:
                            index = batch * 2 + row // 2
                            reference_dw[index, lag] += np.asarray(
                                x[batch, row] * x[batch, row - lag]
                            )
            np.testing.assert_allclose(np.asarray(dx), reference_dx)
            np.testing.assert_allclose(np.asarray(dw_partials), reference_dw)

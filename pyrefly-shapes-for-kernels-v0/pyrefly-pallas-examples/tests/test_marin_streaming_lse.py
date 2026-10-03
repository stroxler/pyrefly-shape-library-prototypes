# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The complete `linear_softmax_lse_forward_fori_pallas_kernel` body comes from Marin
# lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_tpu.py;
# only its parameter annotations are added. The executable helper is also copied.
# @lint-ignore-every AUTODEPS2

"""Streaming vocabulary log-sum-exp carries two per-row scratch buffers."""

from __future__ import annotations

from typing import assert_type, Optional, TYPE_CHECKING

import jax
import jax.lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar

    BatchBlock = IntVar("BatchBlock")
    Hidden = IntVar("Hidden")
    Vocab = IntVar("Vocab")
    VocabBlock = IntVar("VocabBlock")
    ComputeBlock = IntVar("ComputeBlock")

    def _apply_logit_soft_cap(
        logits: pl.Tile[[BatchBlock, ComputeBlock]],
        logit_soft_cap: float | None,
    ) -> pl.Tile[[BatchBlock, ComputeBlock]]: ...

else:

    def _apply_logit_soft_cap(
        logits: jax.Array, logit_soft_cap: Optional[float]
    ) -> jax.Array:
        if logit_soft_cap is None:
            return logits
        return jnp.tanh(logits / logit_soft_cap) * logit_soft_cap


NUM_LANES = 128


def linear_softmax_lse_forward_fori_pallas_kernel(
    x_ref: pl.LseInputRef[BatchBlock, Hidden],
    w_ref: pl.LseWeightRef[Hidden, VocabBlock],
    lse_ref: pl.LseOutputRef[BatchBlock],
    m_scratch_ref: pltpu.VmemScratchRef[[BatchBlock, 128]],
    l_scratch_ref: pltpu.VmemScratchRef[[BatchBlock, 128]],
    *,
    v_dim: Int[Vocab],
    v_compute_block_size: Int[ComputeBlock],
    dtype: object | None,
    logit_soft_cap: float | None,
    precision: object,
    dot_preferred_element_type: object | None,
):
    """Forward kernel for streaming LSE with an inner fori loop over V subtile blocks."""
    core_index, b_index, v_index, h_index = (pl.program_id(i) for i in range(4))
    del core_index, b_index, h_index
    v_block_size = w_ref.shape[1]
    num_v_blocks = pl.num_programs(2)

    if v_block_size % v_compute_block_size != 0:
        raise NotImplementedError(
            f"{v_block_size=} must be divisible by {v_compute_block_size=}"
        )
    repeats, rem = divmod(v_compute_block_size, NUM_LANES)
    if rem != 0:
        raise NotImplementedError(
            f"{v_compute_block_size=} must be a multiple of {NUM_LANES}"
        )

    @pl.when(v_index == num_v_blocks - 1)
    def pad_non_aligned_v_block():
        if v_dim % v_block_size != 0:
            rem = v_dim % v_block_size
            w_ref[:, rem:] = jnp.zeros(
                (w_ref.shape[0], w_ref.shape[1] - rem), dtype=w_ref.dtype
            )

    @pl.when(v_index == 0)
    def init_accumulators():
        m_scratch_ref[...] = jnp.full_like(m_scratch_ref, -jnp.inf)
        l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)

    def body(
        i: int,
        state: tuple[pl.Tile[[BatchBlock, 128]], pl.Tile[[BatchBlock, 128]]],
    ):
        m_prev, l_prev = state
        slice_v = pl.ds(i * v_compute_block_size, v_compute_block_size)
        w_chunk = w_ref[:, slice_v]
        logits = jax.lax.dot_general(
            x_ref[...],
            w_chunk[...],
            (((1,), (0,)), ((), ())),
            preferred_element_type=dot_preferred_element_type,
            precision=precision,
        )
        if dtype is not None:
            logits = logits.astype(dtype)
        logits = _apply_logit_soft_cap(logits, logit_soft_cap)
        logits_f32 = logits.astype(jnp.float32)

        block_offset = v_index * v_block_size + i * v_compute_block_size
        cols = jnp.arange(v_compute_block_size, dtype=jnp.int32) + block_offset
        logits_f32 = jnp.where(cols[None, :] < v_dim, logits_f32, -jnp.inf)

        m_curr = jnp.max(logits_f32, axis=-1)[:, None]
        m_next = jnp.maximum(m_prev, m_curr)
        s_curr = jnp.exp(logits_f32 - jnp.tile(m_next, (1, repeats)))
        l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0,))
        alpha = jnp.exp(m_prev - m_next)
        l_next = l_curr + alpha * l_prev
        return m_next, l_next

    @pl.when(True)
    def accumulate_block():
        m_prev = m_scratch_ref[...].astype(jnp.float32)
        l_prev = l_scratch_ref[...].astype(jnp.float32)
        num_iters = v_block_size // v_compute_block_size
        # Keep the streaming LSE recurrence in one program over V subtiles so we
        # only carry per-row (m, l) state instead of materializing full logits tiles.
        # This materially reduces VMEM pressure on TPU v4.
        m_next, l_next = jax.lax.fori_loop(
            0, num_iters, body, (m_prev, l_prev), unroll=True
        )
        m_scratch_ref[...] = m_next.astype(m_scratch_ref.dtype)
        l_scratch_ref[...] = l_next.astype(l_scratch_ref.dtype)

    @pl.when(v_index == num_v_blocks - 1)
    def finalize():
        lse_ref[...] = (jnp.log(l_scratch_ref[...]) + m_scratch_ref[...]).astype(
            lse_ref.dtype
        )


def streaming_lse[
    B: IntVar,
    H: IntVar,
    V: IntVar,
    BB: IntVar,
    VB: IntVar,
    VC: IntVar,
](
    x: jax.Array[[B, H]],
    w: jax.Array[[H, V]],
    block_batch: Int[BB],
    block_vocab: Int[VB],
    compute_vocab: Int[VC],
) -> jax.Array[[B]]:
    def kernel(
        x_ref: pl.LseInputRef[BB, H],
        w_ref: pl.LseWeightRef[H, VB],
        out_ref: pl.LseOutputRef[BB],
        max_scratch: pltpu.VmemScratchRef[[BB, 128]],
        sum_scratch: pltpu.VmemScratchRef[[BB, 128]],
    ) -> None:
        linear_softmax_lse_forward_fori_pallas_kernel(
            x_ref,
            w_ref,
            out_ref,
            max_scratch,
            sum_scratch,
            v_dim=w.shape[1],
            v_compute_block_size=compute_vocab,
            dtype=None,
            logit_soft_cap=None,
            precision=None,
            dot_preferred_element_type=jnp.float32,
        )

    lanes = pl.pallas_call(
        kernel,
        in_specs=(
            pl.BlockSpec(
                (block_batch, x.shape[1]),
                lambda c, i, j, k: (i, 0),
                memory_space=pltpu.VMEM,
            ),
            pl.BlockSpec(
                (x.shape[1], block_vocab),
                lambda c, i, j, k: (0, j),
                memory_space=pltpu.VMEM,
            ),
        ),
        out_specs=pl.BlockSpec(
            (block_batch, 128),
            lambda c, i, j, k: (i, 0),
            memory_space=pltpu.VMEM,
        ),
        out_shape=jax.ShapeDtypeStruct((x.shape[0], 128), x.dtype),
        scratch_shapes=(
            pltpu.VMEM((block_batch, 128), x.dtype),
            pltpu.VMEM((block_batch, 128), x.dtype),
        ),
        grid=(1, pl.cdiv(x.shape[0], block_batch), pl.cdiv(w.shape[1], block_vocab), 1),
        interpret=True,
    )(x, w)
    return lanes[:, 0]


def test_correct_projection[
    B: IntVar,
    H: IntVar,
    V: IntVar,
    BB: IntVar,
    VB: IntVar,
    VC: IntVar,
](
    x: jax.Array[[B, H]],
    w: jax.Array[[H, V]],
    block_batch: Int[BB],
    block_vocab: Int[VB],
    compute_vocab: Int[VC],
) -> None:
    assert_type(
        streaming_lse(x, w, block_batch, block_vocab, compute_vocab),
        jax.Array[[B]],
    )


def test_wrong_host_hidden[
    B: IntVar,
    H: IntVar,
    Other: IntVar,
    V: IntVar,
    BB: IntVar,
    VB: IntVar,
    VC: IntVar,
](
    x: jax.Array[[B, H]],
    w: jax.Array[[Other, V]],
    block_batch: Int[BB],
    block_vocab: Int[VB],
    compute_vocab: Int[VC],
) -> None:
    streaming_lse(
        x,
        w,  # E: is not assignable
        block_batch,
        block_vocab,
        compute_vocab,
    )


def test_wrong_host_output_allocation[
    B: IntVar,
    H: IntVar,
    V: IntVar,
    BB: IntVar,
    VB: IntVar,
    VC: IntVar,
](
    x: jax.Array[[B, H]],
    w: jax.Array[[H, V]],
    block_batch: Int[BB],
    block_vocab: Int[VB],
    compute_vocab: Int[VC],
) -> None:
    def kernel(
        x_ref: pl.LseInputRef[BB, H],
        w_ref: pl.LseWeightRef[H, VB],
        out_ref: pl.LseOutputRef[BB],
        max_scratch: pltpu.VmemScratchRef[[BB, 128]],
        sum_scratch: pltpu.VmemScratchRef[[BB, 128]],
    ) -> None:
        linear_softmax_lse_forward_fori_pallas_kernel(
            x_ref,
            w_ref,
            out_ref,
            max_scratch,
            sum_scratch,
            v_dim=w.shape[1],
            v_compute_block_size=compute_vocab,
            dtype=None,
            logit_soft_cap=None,
            precision=None,
            dot_preferred_element_type=jnp.float32,
        )

    pl.pallas_call(  # E: No matching overload
        kernel,
        in_specs=(
            pl.BlockSpec(
                (block_batch, x.shape[1]),
                lambda c, i, j, k: (i, 0),
                memory_space=pltpu.VMEM,
            ),
            pl.BlockSpec(
                (x.shape[1], block_vocab),
                lambda c, i, j, k: (0, j),
                memory_space=pltpu.VMEM,
            ),
        ),
        out_specs=pl.BlockSpec(
            (block_batch, 128),
            lambda c, i, j, k: (i, 0),
            memory_space=pltpu.VMEM,
        ),
        out_shape=jax.ShapeDtypeStruct((x.shape[0], 129), x.dtype),
        scratch_shapes=(
            pltpu.VMEM((block_batch, 128), x.dtype),
            pltpu.VMEM((block_batch, 128), x.dtype),
        ),
        grid=(1, pl.cdiv(x.shape[0], block_batch), pl.cdiv(w.shape[1], block_vocab), 1),
        interpret=True,
    )


def test_wrong_weight_ref_hidden[
    BB: IntVar,
    H: IntVar,
    Other: IntVar,
    VB: IntVar,
    VC: IntVar,
    V: IntVar,
](
    x: pl.LseInputRef[BB, H],
    w: pl.LseWeightRef[Other, VB],
    out: pl.LseOutputRef[BB],
    max_scratch: pltpu.VmemScratchRef[[BB, 128]],
    sum_scratch: pltpu.VmemScratchRef[[BB, 128]],
    v: Int[V],
    compute: Int[VC],
) -> None:
    linear_softmax_lse_forward_fori_pallas_kernel(
        x,
        w,  # E: is not assignable
        out,
        max_scratch,
        sum_scratch,
        v_dim=v,
        v_compute_block_size=compute,
        dtype=None,
        logit_soft_cap=None,
        precision=None,
        dot_preferred_element_type=None,
    )


def test_wrong_scratch_lane_width[BB: IntVar, Other: IntVar](
    scratch: pltpu.VmemScratchRef[[BB, 128]],
    wrong: pl.Tile[[BB, Other]],
) -> None:
    scratch[...] = wrong  # E: Cannot set item


def test_wrong_kernel_scratch_rows[
    BB: IntVar,
    Other: IntVar,
    H: IntVar,
    VB: IntVar,
    VC: IntVar,
    V: IntVar,
](
    x: pl.LseInputRef[BB, H],
    w: pl.LseWeightRef[H, VB],
    out: pl.LseOutputRef[BB],
    max_scratch: pltpu.VmemScratchRef[[BB, 128]],
    wrong_scratch: pltpu.VmemScratchRef[[Other, 128]],
    vocab: Int[V],
    compute: Int[VC],
) -> None:
    linear_softmax_lse_forward_fori_pallas_kernel(
        x,
        w,
        out,
        max_scratch,
        wrong_scratch,  # E: is not assignable
        v_dim=vocab,
        v_compute_block_size=compute,
        dtype=None,
        logit_soft_cap=None,
        precision=None,
        dot_preferred_element_type=None,
    )


def test_weight_suffix_extent_is_not_checked[
    H: IntVar,
    VB: IntVar,
    Other: IntVar,
](
    weight: pl.LseWeightRef[H, VB],
    wrong: pl.Tile[[H, Other]],
) -> None:
    # We do not infer a slice's remaining extent from its runtime start value.
    weight[:, 1:] = wrong


def test_wrong_output_lane_width[BB: IntVar, Other: IntVar](
    output: pl.LseOutputRef[BB],
    wrong: pl.Tile[[BB, Other]],
) -> None:
    output[...] = wrong  # E: Cannot set item


def test_wrong_compute_block_mask[
    BB: IntVar,
    VC: IntVar,
    Other: IntVar,
    V: IntVar,
](
    logits: pl.Tile[[BB, VC]],
    wrong_block: Int[Other],
    vocab: Int[V],
) -> None:
    mask = jnp.arange(wrong_block, dtype=jnp.int32)[None, :] < vocab
    jnp.where(  # E: No matching overload
        mask,
        logits,
        -jnp.inf,
    )


def test_wrong_projection_rank[B: IntVar](lanes: jax.Array[[B, 128]]) -> None:
    assert_type(  # E: assert_type
        lanes[:, 0],
        jax.Array[[B, 128]],
    )


if not TYPE_CHECKING:
    import unittest

    import numpy as np

    class MarinStreamingLseBoundaryTest(unittest.TestCase):
        def test_host_logsumexp_reference_only(self) -> None:
            x = np.arange(12, dtype=np.float32).reshape(3, 4) / 8
            w = np.arange(20, dtype=np.float32).reshape(4, 5) / 10
            logits = x @ w
            expected = np.logaddexp.reduce(logits, axis=1)
            actual = jax.nn.logsumexp(jnp.asarray(x) @ jnp.asarray(w), axis=1)
            self.assertEqual(actual.shape, (3,))
            np.testing.assert_allclose(actual, expected, rtol=1e-6)

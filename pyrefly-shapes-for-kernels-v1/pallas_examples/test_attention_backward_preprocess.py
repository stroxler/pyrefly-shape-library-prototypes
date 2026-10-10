"""Check JAX's attention backward preprocessing and its permuted output."""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any, assert_type, cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from shape_extensions import Int, IntVar

from pallas_library.layout import attention_preprocess_layout, checked_pallas_call

Batch = IntVar("Batch")
Queries = IntVar("Queries")
Heads = IntVar("Heads")
HeadDim = IntVar("HeadDim")
PaddedDim = IntVar("PaddedDim")
QueryBlock = IntVar("QueryBlock")


# Body copied from JAX's jax/experimental/pallas/ops/gpu/attention.py.
def _preprocess_backward_kernel(
    out_ref: pl.ValidInRef[[QueryBlock, PaddedDim], [QueryBlock, HeadDim]],
    dout_ref: pl.ValidInRef[[QueryBlock, PaddedDim], [QueryBlock, HeadDim]],
    delta_ref: pl.OutRef[[QueryBlock]],
    head_dim: Int[HeadDim],
):
    # load
    head_mask = (jnp.arange(out_ref.shape[-1]) < head_dim)[None, :]
    o = plgpu.load(out_ref, mask=head_mask, other=0.0)
    do = plgpu.load(dout_ref, mask=head_mask, other=0.0)
    # compute
    delta = jnp.sum(o * do, axis=1)
    # write-back
    delta_ref[...] = delta.astype(delta_ref.dtype)


def attention_backward_delta[
    B: IntVar,
    Q: IntVar,
    H: IntVar,
    D: IntVar,
    P: IntVar,
    BQ: IntVar,
](
    out: jax.Array[[B, Q, H, D]],
    dout: jax.Array[[B, Q, H, D]],
    *,
    padded_dim: Int[P],
    block_q: Int[BQ],
) -> jax.Array[[B, H, Q]]:
    """Bind full host arrays to padded head tiles and the transposed delta."""
    batch, queries, heads, dim = out.shape

    def kernel(
        out_ref: pl.ValidInRef[[BQ, P], [BQ, D]],
        dout_ref: pl.ValidInRef[[BQ, P], [BQ, D]],
        delta_ref: pl.OutRef[[BQ]],
    ) -> None:
        _preprocess_backward_kernel(out_ref, dout_ref, delta_ref, dim)

    layout = attention_preprocess_layout(
        kernel,
        input_shape=(batch, queries, heads, dim),
        padded_dim=padded_dim,
        query_block=block_q,
        out_shape=jax.ShapeDtypeStruct((batch, heads, queries), out.dtype),
    )
    return checked_pallas_call(layout, interpret=True)(out, dout)


class AttentionBackwardPreprocessTest(unittest.TestCase):
    """Check padded features and the host-output permutation."""

    def test_padded_head_and_permuted_delta(self) -> None:
        out = (
            cast(Any, jnp)
            .arange(2 * 4 * 2 * 3, dtype=jnp.float32)
            .reshape((2, 4, 2, 3))
            / 7
        )
        dout = cast(Any, jnp).flip(out, axis=-1) + 0.25
        delta = attention_backward_delta(out, dout, padded_dim=4, block_q=2)
        expected = cast(Any, jnp).einsum("bqhd,bqhd->bhq", out, dout)
        self.assertEqual(delta.shape, (2, 2, 4))
        self.assertTrue(bool(cast(Any, jnp).allclose(delta, expected, atol=1e-6)))

    def test_reject_mismatched_inputs_and_padding(self) -> None:
        out = cast(Any, jnp).ones((1, 4, 2, 3), dtype=jnp.float32)
        with self.assertRaisesRegex(ValueError, "shapes must match"):
            attention_backward_delta(
                out, cast(Any, jnp).ones((1, 4, 1, 3)), padded_dim=4, block_q=2
            )
        with self.assertRaisesRegex(ValueError, "Padded head dimension"):
            attention_backward_delta(out, out, padded_dim=2, block_q=2)
        with self.assertRaisesRegex(ValueError, "Query block"):
            attention_backward_delta(out, out, padded_dim=4, block_q=3)


if TYPE_CHECKING:

    def typed_padded_load[Q: IntVar, P: IntVar, D: IntVar, Other: IntVar](
        ref: pl.ValidInRef[[Q, P], [Q, D]],
        correct: pl.Mask[[1, P], [1, D]],
        wrong: pl.Mask[[1, P], [1, Other]],
        wrong_axis: pl.Mask[[P, 1], [D, 1]],
    ) -> None:
        plgpu.load(ref, mask=correct, other=0.0)
        plgpu.load(ref, mask=wrong, other=0.0)  # pyrefly: ignore[no-matching-overload]
        plgpu.load(ref, mask=wrong_axis, other=0.0)  # pyrefly: ignore[no-matching-overload]

    def typed_boundary[B: IntVar, Q: IntVar, H: IntVar, D: IntVar, Other: IntVar](
        out: jax.Array[[B, Q, H, D]],
        wrong: jax.Array[[B, Q, H, Other]],
    ) -> None:
        assert_type(
            attention_backward_delta(out, out, padded_dim=4, block_q=2),
            jax.Array[[B, H, Q]],
        )
        attention_backward_delta(
            out,
            wrong,  # pyrefly: ignore[bad-argument-type]
            padded_dim=4,
            block_q=2,
        )

"""Parallel Pallas dropout probes: aligned mask input and per-program RNG."""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any, assert_type, cast

import jax
import jax.numpy as jnp
import jax.random
from jax.experimental import pallas as pl
from shape_extensions import Int, IntVar

from pallas_library.layout import checked_pallas_call, grid_axis, vector_layout

Block = IntVar("Block")


def dropout_kernel(
    x_ref: pl.InRef[[Block]],
    keep_ref: pl.InRef[[Block]],
    output_ref: pl.OutRef[[Block]],
    *,
    p: float,
) -> None:
    output_ref[:] = jnp.where(keep_ref[:], x_ref[:] / (1 - p), 0.0)


def seeded_dropout_kernel(
    x_ref: pl.InRef[[Block]],
    output_ref: pl.OutRef[[Block]],
    *,
    p: float,
    seed: int,
    block_size: Int[Block],
) -> None:
    key = jax.random.fold_in(jax.random.key(seed), pl.program_id(0))
    keep = jax.random.bernoulli(key, p=1 - p, shape=(block_size,))
    output_ref[:] = jnp.where(keep, x_ref[:] / (1 - p), 0.0)


def _check_parameters(p: float, block_size: int) -> None:
    """Require a finite keep probability and a legal tile width."""
    if not isinstance(p, (float, int)) or not 0 <= p < 1:
        raise ValueError("p must be a probability in [0, 1)")
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive integer")


def checked_dropout[Length: IntVar, Tile: IntVar](
    x: jax.Array[[Length]],
    keep: jax.Array[[Length]],
    *,
    p: float,
    block_size: Int[Tile],
) -> jax.Array[[Length]]:
    """Tie a boolean keep allocation to the values and output length."""
    _check_parameters(p, block_size)
    if x.ndim != 1 or keep.ndim != 1 or x.shape != keep.shape:
        raise ValueError("Input and keep-mask lengths must match")
    if x.device != keep.device:
        raise ValueError("Input and keep-mask devices must match")
    if not jnp.issubdtype(x.dtype, jnp.floating) or keep.dtype != jnp.bool_:
        raise ValueError("Input must be floating-point and keep-mask boolean")
    (length,) = x.shape
    if length == 0:
        return cast(jax.Array[[Length]], jnp.empty((0,), dtype=x.dtype))

    def kernel(
        x_ref: pl.InRef[[Tile]],
        keep_ref: pl.InRef[[Tile]],
        output_ref: pl.OutRef[[Tile]],
    ) -> None:
        dropout_kernel(x_ref, keep_ref, output_ref, p=p)

    layout = vector_layout(
        kernel,
        out_shape=jax.ShapeDtypeStruct((length,), x.dtype),
        grid=(grid_axis(length, block_size),),
        block=(block_size,),
        index_map=lambda i: (i,),
        input_dtypes=(x.dtype, keep.dtype),
    )
    return checked_pallas_call(layout, interpret=True)(x, keep)


def checked_seeded_dropout[Length: IntVar, Tile: IntVar](
    x: jax.Array[[Length]],
    *,
    p: float,
    seed: int,
    block_size: Int[Tile],
) -> jax.Array[[Length]]:
    """Bind one vector Ref and derive a distinct JAX key per program."""
    _check_parameters(p, block_size)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if x.ndim != 1 or not jnp.issubdtype(x.dtype, jnp.floating):
        raise ValueError("Input must be a floating-point vector")
    (length,) = x.shape
    if length == 0:
        return cast(jax.Array[[Length]], jnp.empty((0,), dtype=x.dtype))

    def kernel(x_ref: pl.InRef[[Tile]], output_ref: pl.OutRef[[Tile]]) -> None:
        seeded_dropout_kernel(x_ref, output_ref, p=p, seed=seed, block_size=block_size)

    layout = vector_layout(
        kernel,
        out_shape=jax.ShapeDtypeStruct((length,), x.dtype),
        grid=(grid_axis(length, block_size),),
        block=(block_size,),
        index_map=lambda i: (i,),
        input_dtypes=(x.dtype,),
    )
    return checked_pallas_call(layout, interpret=True)(x)


class DropoutTest(unittest.TestCase):
    """Compare both dropout forms on complete and partial Pallas blocks."""

    def test_explicit_mask(self) -> None:
        for length in (0, 1, 8, 9, 17):
            with self.subTest(length=length):
                x = cast(Any, jnp.arange)(length, dtype=jnp.float32)
                keep = cast(Any, jnp.arange)(length) % 3 != 0
                actual = checked_dropout(x, keep, p=0.25, block_size=4)
                self.assertEqual(actual.shape, (length,))
                self.assertTrue(
                    bool(
                        cast(Any, jnp).allclose(actual, jnp.where(keep, x / 0.75, 0.0))
                    )
                )

    def test_seeded_dropout(self) -> None:
        x = cast(Any, jnp).ones((17,), dtype=jnp.float32)
        first = checked_seeded_dropout(x, p=0.5, seed=13, block_size=4)
        second = checked_seeded_dropout(x, p=0.5, seed=13, block_size=4)
        self.assertEqual(first.tolist(), second.tolist())
        self.assertTrue(bool(cast(Any, jnp).all((first == 0) | (first == 2))))

    def test_reject_incompatible_host_contract(self) -> None:
        x = cast(Any, jnp).ones((9,), dtype=jnp.float32)
        with self.assertRaisesRegex(ValueError, "keep-mask lengths"):
            checked_dropout(
                x,
                cast(Any, jnp).ones((8,), dtype=jnp.bool_),
                p=0.25,
                block_size=4,
            )
        with self.assertRaisesRegex(ValueError, "keep-mask boolean"):
            checked_dropout(x, x, p=0.25, block_size=4)
        with self.assertRaisesRegex(ValueError, "probability"):
            checked_seeded_dropout(x, p=1.0, seed=13, block_size=4)
        with self.assertRaisesRegex(ValueError, "seed"):
            checked_seeded_dropout(x, p=0.5, seed=cast(Any, "bad"), block_size=4)

    def test_layout_validates_mask_dtype_at_launch(self) -> None:
        x = cast(Any, jnp).ones((8,), dtype=jnp.float32)
        mask = cast(Any, jnp).ones((8,), dtype=jnp.bool_)

        def kernel(
            x_ref: pl.InRef[[4]],
            keep_ref: pl.InRef[[4]],
            output_ref: pl.OutRef[[4]],
        ) -> None:
            dropout_kernel(x_ref, keep_ref, output_ref, p=0.25)

        layout = vector_layout(
            kernel,
            out_shape=jax.ShapeDtypeStruct((8,), jnp.float32),
            grid=(grid_axis(8, 4),),
            block=(4,),
            index_map=lambda i: (i,),
            input_dtypes=(x.dtype, mask.dtype),
        )
        call = checked_pallas_call(layout, interpret=True)
        with self.assertRaisesRegex(ValueError, "dtypes"):
            call(x, x)


if TYPE_CHECKING:
    x: jax.Array[[9]] = cast(Any, jnp).ones((9,))
    keep: jax.Array[[9]] = cast(Any, jnp).ones((9,), dtype=jnp.bool_)
    assert_type(
        checked_dropout(x, keep, p=0.25, block_size=4),
        jax.Array[[9]],
    )
    assert_type(
        checked_seeded_dropout(x, p=0.25, seed=13, block_size=4),
        jax.Array[[9]],
    )

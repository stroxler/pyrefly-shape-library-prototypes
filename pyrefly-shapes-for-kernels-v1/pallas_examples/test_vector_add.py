"""Run the original Pallas vector-add body through a generated JAX boundary."""

from __future__ import annotations

import inspect
import unittest
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from shape_extensions import IntVar

from pallas_library.jax_wrapper import Launch1D, make_jax_wrapper

Block = IntVar("Block")
Other = IntVar("Other")


def add_kernel(
    x_ref: pl.InRef[[Block]], y_ref: pl.InRef[[Block]], o_ref: pl.OutRef[[Block]]
) -> None:
    x = x_ref[:]
    y = y_ref[:]
    o_ref[:] = x + y


def design_doc_add(x: Any, y: Any) -> Any:
    """The design doc's inline 8-element call with explicit 1D index tuples."""
    add = pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct((8,), jnp.int32),
        in_specs=(
            pl.BlockSpec((2,), lambda i: (i,)),
            pl.BlockSpec((2,), lambda i: (i,)),
        ),
        out_specs=pl.BlockSpec((2,), lambda i: (i,)),
        grid=(pl.cdiv(8, 2),),
        interpret=True,
    )
    return add(x, y)


class VectorAddTest(unittest.TestCase):
    """Exercise complete and partial tiles, and reject invalid host inputs."""

    def test_generated_wrapper(self) -> None:
        add = make_jax_wrapper(add_kernel, Launch1D(4))
        self.assertEqual(["x", "y"], list(inspect.signature(add).parameters))
        self.assertEqual(
            "pl.InRef[[Block]]",
            inspect.signature(add_kernel).parameters["x_ref"].annotation,
        )
        for length in (0, 1, 4, 10, 16):
            with self.subTest(length=length):
                x = cast(Any, jnp.arange)(length, dtype=jnp.float32)
                y = x * 3
                actual = add(x, y)
                self.assertEqual(actual.shape, (length,))
                self.assertEqual(actual.tolist(), (x + y).tolist())
                self.assertEqual(add(y=y, x=x).tolist(), actual.tolist())

    def test_wrapper_rejects_invalid_inputs(self) -> None:
        add = make_jax_wrapper(add_kernel, Launch1D(4))
        good = cast(Any, jnp.arange)(10, dtype=jnp.float32)
        for error, invalid in (
            ("length", cast(Any, jnp.arange)(11, dtype=jnp.float32)),
            ("one-dimensional", good.reshape(2, 5)),
            ("dtype", cast(Any, jnp.arange)(10, dtype=jnp.int32)),
        ):
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    add(good, invalid)

        for invalid_block in (0, -1, 1.5):
            with self.subTest(block_size=invalid_block):
                with self.assertRaisesRegex(ValueError, "block_size"):
                    make_jax_wrapper(add_kernel, Launch1D(cast(Any, invalid_block)))

    def test_wrapper_rejects_unmatched_refs(self) -> None:
        def mismatched(x_ref: pl.InRef[[Block]], o_ref: pl.OutRef[[Other]]) -> None:
            """Expose conflicting tile dimensions to the wrapper factory."""

        with self.assertRaisesRegex(ValueError, "does not share block"):
            make_jax_wrapper(mismatched, Launch1D(4))

    def test_design_doc_call(self) -> None:
        x = cast(Any, jnp.arange)(8, dtype=jnp.int32)
        y = cast(Any, jnp.arange)(8, 16, dtype=jnp.int32)
        actual = design_doc_add(x, y)
        self.assertEqual(actual.tolist(), (x + y).tolist())
        generated = make_jax_wrapper(add_kernel, Launch1D(2))(
            x.astype(jnp.float32), y.astype(jnp.float32)
        )
        self.assertEqual(actual.tolist(), generated.tolist())

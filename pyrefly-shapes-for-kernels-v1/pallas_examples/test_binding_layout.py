"""Exercise the composable Pallas binding assembler and its host checks."""

from __future__ import annotations

import unittest
from typing import Any, cast

import jax
import jax.numpy as jnp

from pallas_examples.test_vector_add import add_kernel
from pallas_library.layout import (
    GridBinding,
    InputBinding,
    OutputBinding,
    binding_layout,
    checked_pallas_call,
)


class BindingLayoutTest(unittest.TestCase):
    """Check axis equality, grid coverage, and executable block metadata."""

    def test_vector_add_from_bindings(self) -> None:
        host = jax.ShapeDtypeStruct((10,), jnp.float32)
        index_map = lambda i: (i,)
        layout = binding_layout(
            add_kernel,
            inputs=(
                InputBinding(host, ("length",), (4,), index_map),
                InputBinding(host, ("length",), (4,), index_map),
            ),
            outputs=(OutputBinding(host, ("length",), (4,), index_map),),
            grid=(GridBinding(0, 0, 4),),
        )
        self.assertEqual(layout.grid, (3,))
        x = cast(Any, jnp.arange)(10, dtype=jnp.float32)
        self.assertEqual(
            checked_pallas_call(layout, interpret=True)(x, x).tolist(),
            (2 * x).tolist(),
        )

    def test_inconsistent_axes_and_grid_are_rejected(self) -> None:
        one = jax.ShapeDtypeStruct((8,), jnp.float32)
        other = jax.ShapeDtypeStruct((9,), jnp.float32)
        with self.assertRaisesRegex(ValueError, "dimension 'length' does not match"):
            binding_layout(
                add_kernel,
                inputs=(
                    InputBinding(one, ("length",)),
                    InputBinding(other, ("length",)),
                ),
                outputs=(OutputBinding(one, ("length",)),),
            )
        with self.assertRaisesRegex(ValueError, "must divide"):
            binding_layout(
                add_kernel,
                inputs=(InputBinding(other, ("length",)),) * 2,
                outputs=(OutputBinding(other, ("length",)),),
                grid=(GridBinding(0, 0, 4, exact=True),),
            )

    def test_squeezed_attention_axes_keep_host_relationships(self) -> None:
        q = jax.ShapeDtypeStruct((2, 16, 4, 32), jnp.float32)
        kv = jax.ShapeDtypeStruct((2, 64, 4, 32), jnp.float32)
        lse = jax.ShapeDtypeStruct((2, 4, 16), jnp.float32)

        def kernel(a: object, b: object, c: object) -> None:
            pass

        layout = binding_layout(
            kernel,
            inputs=(
                InputBinding(q, ("batch", "queries", "heads", "dim"),
                             (None, 8, None, 32), lambda i, j, h: (j, i, h, 0)),
            ),
            outputs=(
                OutputBinding(q, ("batch", "queries", "heads", "dim"),
                              (None, 8, None, 32), lambda i, j, h: (j, i, h, 0)),
                OutputBinding(lse, ("batch", "heads", "queries"),
                              (None, None, 8), lambda i, j, h: (j, h, i)),
            ),
            grid=(GridBinding(0, 1, 8), GridBinding(0, 0, 1), GridBinding(0, 2, 1)),
        )
        self.assertEqual(layout.grid, (2, 2, 4))
        with self.assertRaisesRegex(ValueError, "dimension 'batch' does not match"):
            binding_layout(
                kernel,
                inputs=(InputBinding(kv, ("batch", "keys", "heads", "dim")),),
                outputs=(OutputBinding(q, ("batch", "queries", "heads", "dim")),
                         OutputBinding(jax.ShapeDtypeStruct((3, 4, 16), jnp.float32),
                                       ("batch", "heads", "queries"))),
            )


if __name__ == "__main__":
    unittest.main()

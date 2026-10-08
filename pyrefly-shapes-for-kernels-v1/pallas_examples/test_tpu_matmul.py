"""The three-axis tiled TPU matmul in JAX's docs/pallas/tpu/matmul.md.

The kernel body is unchanged. CPU interpretation checks tile accumulation;
there is no claim that TPU pipelining or cross-program ordering is verified.
Types do not prove `pl.when` initialized the accumulator before its read or
that arbitrary index-map implementations select the intended grid blocks.
"""

from __future__ import annotations

import unittest
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, assert_type, cast
from unittest.mock import patch

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from shape_extensions import Int, IntVar

from pallas_library.layout import (
    BlockIndex,
    GridBinding,
    InputBinding,
    Layout,
    OutputBinding,
    binding_layout,
    checked_pallas_call,
)

BM = IntVar("BM")
BK = IntVar("BK")
BN = IntVar("BN")


def matmul_kernel(
    x_ref: pl.InRef[[BM, BK]],
    y_ref: pl.InRef[[BK, BN]],
    z_ref: pl.AccumRef[[BM, BN]],
) -> None:
    @pl.when(pl.program_id(2) == 0)
    def _():
        z_ref[...] = jnp.zeros_like(z_ref)

    z_ref[...] += x_ref[...] @ y_ref[...]


def tpu_matmul_layout[
    M: IntVar,
    K: IntVar,
    N: IntVar,
    TM: IntVar,
    TK: IntVar,
    TN: IntVar,
](
    kernel: Callable[
        [pl.InRef[[TM, TK]], pl.InRef[[TK, TN]], pl.AccumRef[[TM, TN]]], None
    ],
    *,
    x_shape: tuple[Int[M], Int[K]],
    y_shape: tuple[Int[K], Int[N]],
    out_shape: jax.ShapeDtypeStruct[[M, N]],
    block_m: Int[TM],
    block_k: Int[TK],
    block_n: Int[TN],
    x_map: Callable[
        [BlockIndex[M, TM], BlockIndex[N, TN], BlockIndex[K, TK]],
        tuple[BlockIndex[M, TM], BlockIndex[K, TK]],
    ],
    y_map: Callable[
        [BlockIndex[M, TM], BlockIndex[N, TN], BlockIndex[K, TK]],
        tuple[BlockIndex[K, TK], BlockIndex[N, TN]],
    ],
    out_map: Callable[
        [BlockIndex[M, TM], BlockIndex[N, TN], BlockIndex[K, TK]],
        tuple[BlockIndex[M, TM], BlockIndex[N, TN]],
    ],
) -> Layout[[jax.Array[[M, K]], jax.Array[[K, N]]], jax.Array[[M, N]]]:
    """Bind each reduction block and output tile to one three-axis grid."""
    m, k = x_shape
    y_k, n = y_shape
    if any(
        type(value) is not int or value <= 0
        for value in (m, k, n, block_m, block_k, block_n)
    ):
        raise ValueError("Matrix and block dimensions must be positive")
    if k != y_k or tuple(out_shape.shape) != (m, n):
        raise ValueError("Inner and declared output dimensions must match")
    if m % block_m or k % block_k or n % block_n:
        raise ValueError("Matmul blocks must divide all dimensions")
    x_spec = jax.ShapeDtypeStruct(x_shape, out_shape.dtype)
    y_spec = jax.ShapeDtypeStruct(y_shape, out_shape.dtype)
    output_layout = binding_layout(
        kernel,
        inputs=(
            InputBinding(x_spec, ("rows", "inner"), (block_m, block_k), x_map),
            InputBinding(y_spec, ("inner", "cols"), (block_k, block_n), y_map),
        ),
        outputs=(
            OutputBinding(out_shape, ("rows", "cols"), (block_m, block_n), out_map),
        ),
        grid=(
            GridBinding(0, 0, block_m, exact=True),
            GridBinding(0, 1, block_n, exact=True),
        ),
    )
    # The reduction grid is not a tiled output axis; the validated K is shared.
    return cast(
        Layout[[jax.Array[[M, K]], jax.Array[[K, N]]], jax.Array[[M, N]]],
        Layout(
            kernel=kernel,
            grid=(*output_layout.grid, k // block_k),
            in_specs=output_layout.in_specs,
            out_spec=output_layout.out_spec,
            out_shape=output_layout.out_shape,
            input_shapes=output_layout.input_shapes,
            input_dtypes=output_layout.input_dtypes,
        ),
    )


def checked_tpu_matmul[
    M: IntVar,
    K: IntVar,
    N: IntVar,
    TM: IntVar,
    TK: IntVar,
    TN: IntVar,
](
    x: jax.Array[[M, K]],
    y: jax.Array[[K, N]],
    *,
    bm: Int[TM],
    bk: Int[TK],
    bn: Int[TN],
    interpret: bool = True,
) -> jax.Array[[M, N]]:
    """Check host axes, dtype, device, output, and whole reduction tiles."""
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[0]:
        raise ValueError("Matrix inner dimensions must match")
    if x.dtype != y.dtype or x.dtype != jnp.float32:
        raise ValueError("Inputs and declared float32 output must share a dtype")
    if x.device != y.device:
        raise ValueError("Input devices must match")
    if not interpret and getattr(x.device, "platform", None) != "tpu":
        raise ValueError("TPU execution requires input arrays on a TPU")
    m, _ = x.shape
    _, n = y.shape
    layout = tpu_matmul_layout(
        matmul_kernel,
        x_shape=x.shape,
        y_shape=y.shape,
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        block_m=bm,
        block_k=bk,
        block_n=bn,
        x_map=lambda i, _j, q: (i, q),
        y_map=lambda _i, j, q: (q, j),
        out_map=lambda i, j, _q: (i, j),
    )
    params = pltpu.CompilerParams(
        dimension_semantics=("parallel", "parallel", "arbitrary")
    )
    return checked_pallas_call(layout, interpret=interpret, compiler_params=params)(
        x, y
    )


class TpuMatmulTest(unittest.TestCase):
    def test_reduction_tiles_cpu(self) -> None:
        x = cast(Any, jnp.arange)(64, dtype=jnp.float32).reshape(8, 8) / 16
        y = cast(Any, jnp.arange)(48, dtype=jnp.float32).reshape(8, 6) / 8
        actual = checked_tpu_matmul(x, y, bm=4, bk=2, bn=3)
        self.assertEqual(actual.shape, (8, 6))
        self.assertTrue(bool(cast(Any, jnp).allclose(actual, x @ y, atol=1e-5)))

    def test_host_rejections(self) -> None:
        x = cast(Any, jnp.ones)((8, 8), dtype=jnp.float32)
        y = cast(Any, jnp.ones)((8, 6), dtype=jnp.float32)
        with self.assertRaisesRegex(ValueError, "inner dimensions"):
            checked_tpu_matmul(x, y[1:], bm=4, bk=2, bn=3)
        with self.assertRaisesRegex(ValueError, "blocks must divide"):
            checked_tpu_matmul(x, y, bm=3, bk=2, bn=3)
        with self.assertRaisesRegex(ValueError, "dtype"):
            checked_tpu_matmul(x, y.astype(jnp.float16), bm=4, bk=2, bn=3)
        with self.assertRaisesRegex(ValueError, "on a TPU"):
            checked_tpu_matmul(x, y, bm=4, bk=2, bn=3, interpret=False)

    def test_tpu_launch_metadata(self) -> None:
        x = cast(Any, jnp.ones)((8, 8), dtype=jnp.float32)
        y = cast(Any, jnp.ones)((8, 6), dtype=jnp.float32)
        with patch(
            "pallas_library.layout.pl.pallas_call", return_value=lambda a, b: a @ b
        ) as launch:
            checked_tpu_matmul(x, y, bm=4, bk=2, bn=3)
        self.assertEqual(launch.call_args.kwargs["grid"], (2, 2, 4))
        self.assertEqual(launch.call_args.kwargs["out_shape"].shape, (8, 6))
        self.assertEqual(
            tuple(spec.block_shape for spec in launch.call_args.kwargs["in_specs"]),
            ((4, 2), (2, 3)),
        )
        self.assertEqual(launch.call_args.kwargs["out_specs"].block_shape, (4, 3))
        self.assertEqual(
            launch.call_args.kwargs["compiler_params"].dimension_semantics,
            ("parallel", "parallel", "arbitrary"),
        )


if TYPE_CHECKING:
    typed_x: jax.Array[[8, 8]] = cast(Any, jnp.ones)((8, 8))
    typed_y: jax.Array[[8, 6]] = cast(Any, jnp.ones)((8, 6))
    assert_type(
        checked_tpu_matmul(typed_x, typed_y, bm=4, bk=2, bn=3),
        jax.Array[[8, 6]],
    )
    wrong_inner: jax.Array[[7, 6]] = cast(Any, jnp.ones)((7, 6))
    checked_tpu_matmul(
        typed_x,
        wrong_inner,  # pyrefly: ignore[bad-argument-type]
        bm=4,
        bk=2,
        bn=3,
    )
    tpu_matmul_layout(
        matmul_kernel,
        x_shape=typed_x.shape,
        y_shape=typed_y.shape,
        out_shape=jax.ShapeDtypeStruct((8, 6), jnp.float32),
        block_m=4,
        block_k=2,
        block_n=3,
        x_map=lambda i, _j, q: (i, q),
        y_map=lambda _i, j, q: (q, j),
        out_map=lambda i, j, _q: (i, _q),  # pyrefly: ignore[bad-argument-type]
    )

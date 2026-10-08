"""Model Mosaic GPU's pipelined matmul and its JAX-to-GMEM boundary.

The nested kernel and pipeline callback come from JAX's
docs/pallas/gpu/pipelining.md (Apache-2.0); only semantic annotations are new.
"""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any, assert_type, cast
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from shape_extensions import Int, IntVar


def pipelined_matmul[
    Rows: IntVar,
    Inner: IntVar,
    Cols: IntVar,
    TM: IntVar,
    TN: IntVar,
    Swizzle: IntVar,
](
    a: jax.Float16Array[[Rows, Inner]],
    b: jax.Float16Array[[Inner, Cols]],
    tile_m: Int[TM],
    tile_n: Int[TN],
    swizzle: Int[Swizzle],
) -> jax.Array[[Rows, Cols]]:
    """Check hardware tiling and shape compatibility before GPU lowering."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Pipelined matmul contraction dimensions must match")
    if a.dtype != jnp.float16 or b.dtype != jnp.float16 or a.device != b.device:
        raise ValueError("Pipelined matmul inputs must be float16 on one device")
    if type(tile_m) is not int or type(tile_n) is not int or tile_m <= 0 or tile_n <= 0:
        raise ValueError("Pipelined matmul tile dimensions must be positive integers")
    if type(swizzle) is not int or swizzle not in (32, 64, 128):
        raise ValueError("Pipelined matmul swizzle must be 32, 64, or 128 bytes")
    m, k = a.shape
    _, n = b.shape
    dtype = jnp.float16
    swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
    tile_k = swizzle_elems
    if min(m, k, n) <= 0 or m % tile_m or k % tile_k or n % tile_n:
        raise ValueError("Pipelined matmul requires whole positive matrix tiles")
    if tile_m < 64 or tile_m % 64 or tile_n < 8 or tile_n % 8 or tile_m % swizzle_elems:
        raise ValueError("Pipelined matmul tiles must satisfy WGMMA/swizzle alignment")
    grid_m = m // tile_m
    grid_k = k // tile_k
    grid_n = n // tile_n
    assert tile_m % swizzle_elems == 0

    # Note: Transforms will be inferred automatically
    # by Mosaic GPU in the future.
    transforms = (
        plgpu.TilingTransform((8, swizzle_elems)),
        plgpu.SwizzleTransform(swizzle),
    )

    def kernel(
        a_gmem: plgpu.GmemInRef[Rows, Inner],
        b_gmem: plgpu.GmemInRef[Inner, Cols],
        o_gmem: plgpu.GmemOutRef[Rows, Cols],
        o_smem: plgpu.SmemScratchRef[TM, TN],
        acc: plgpu.AccRef[TM, TN],
    ) -> None:
        def pipeline_step(
            _: int,
            a_smem: plgpu.SmemInRef[TM, Swizzle // 2],
            b_smem: plgpu.SmemInRef[Swizzle // 2, TN],
        ) -> None:
            plgpu.wgmma(acc, a_smem, b_smem)
            plgpu.wgmma_wait(1)

        # pl.program_id obtains the index into the grid.
        pid_m = pl.program_id(0)
        pid_n = pl.program_id(1)

        pipeline = plgpu.emit_pipeline(
            pipeline_step,
            in_specs=[
                plgpu.BlockSpec(
                    (tile_m, tile_k),
                    lambda k: (pid_m, k),
                    transforms=transforms,
                    delay_release=1,  # JAX 0.11 moved this from emit_pipeline.
                ),
                plgpu.BlockSpec(
                    (tile_k, tile_n),
                    lambda k: (k, pid_n),
                    transforms=transforms,
                    delay_release=1,  # JAX 0.11 moved this from emit_pipeline.
                ),
            ],
            grid=(grid_k,),
            max_concurrent_steps=2,
        )

        pipeline(a_gmem, b_gmem)
        # Store WGMMA accumulator to SMEM and then to GMEM.
        o_smem[...] = acc[...].astype(dtype)
        plgpu.commit_smem()
        m_slice = pl.ds(pid_m * tile_m, tile_m)
        n_slice = pl.ds(pid_n * tile_n, tile_n)
        plgpu.copy_smem_to_gmem(o_smem, o_gmem.at[m_slice, n_slice])
        plgpu.wait_smem_to_gmem(0)

    return plgpu.kernel(
        kernel,
        out_type=jax.ShapeDtypeStruct((m, n), jnp.float16),
        scratch_types=dict(
            o_smem=plgpu.SMEM((tile_m, tile_n), jnp.float16),
            acc=plgpu.ACC((tile_m, tile_n), jnp.float32),
        ),
        # grid specifies the CUDA grid.
        # Instances of `kernel` will be executed in parallel over this grid.
        grid=(grid_m, grid_n),
        grid_names=("m", "n"),
    )(a, b)


class PipelineMatmulTest(unittest.TestCase):
    """Check the GPU boundary metadata without claiming CPU kernel execution."""

    def test_pipeline_delay_release_specs(self) -> None:
        a = cast(Any, jnp).ones((128, 128), dtype=jnp.float16)
        b = cast(Any, jnp).ones((128, 64), dtype=jnp.float16)
        with patch.object(
            plgpu,
            "kernel",
            return_value=lambda lhs, rhs: jnp.zeros((128, 64), dtype=jnp.float16),
        ) as mock_kernel:
            pipelined_matmul(a, b, 64, 64, 128)
        kernel = mock_kernel.call_args.args[0]
        with (
            patch.object(pl, "program_id", side_effect=(0, 0)),
            patch.object(plgpu, "emit_pipeline", return_value=MagicMock()) as pipeline,
            patch.object(plgpu, "commit_smem"),
            patch.object(plgpu, "copy_smem_to_gmem"),
            patch.object(plgpu, "wait_smem_to_gmem"),
        ):
            kernel(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
        self.assertEqual(
            [spec.delay_release for spec in pipeline.call_args.kwargs["in_specs"]],
            [1, 1],
        )
        self.assertNotIn("delay_release", pipeline.call_args.kwargs)

    def test_grid_output_and_scratch_metadata(self) -> None:
        a = cast(Any, jnp).ones((128, 128), dtype=jnp.float16)
        b = cast(Any, jnp).ones((128, 64), dtype=jnp.float16)
        with patch.object(
            plgpu,
            "kernel",
            return_value=lambda lhs, rhs: jnp.zeros((128, 64), dtype=jnp.float16),
        ) as mock:
            output = pipelined_matmul(a, b, 64, 64, 128)
        self.assertEqual(output.shape, (128, 64))
        self.assertEqual(mock.call_args.kwargs["grid"], (2, 1))
        self.assertEqual(mock.call_args.kwargs["out_type"].shape, (128, 64))
        self.assertEqual(set(mock.call_args.kwargs["scratch_types"]), {"o_smem", "acc"})

    def test_reject_bad_contract_and_partial_tiles(self) -> None:
        a = cast(Any, jnp).ones((128, 128), dtype=jnp.float16)
        b = cast(Any, jnp).ones((128, 64), dtype=jnp.float16)
        with self.assertRaisesRegex(ValueError, "contraction dimensions"):
            pipelined_matmul(a, cast(Any, b[:120, :]), 64, 64, 128)
        with self.assertRaisesRegex(ValueError, "whole positive matrix tiles"):
            pipelined_matmul(cast(Any, a[:127, :]), b, 64, 64, 128)
        with self.assertRaisesRegex(ValueError, "WGMMA/swizzle alignment"):
            pipelined_matmul(a, b, 16, 64, 128)
        with self.assertRaisesRegex(ValueError, "float16"):
            pipelined_matmul(cast(Any, a.astype(jnp.float32)), b, 64, 64, 128)


if TYPE_CHECKING:

    def check_shapes[M: IntVar, K: IntVar, N: IntVar, Other: IntVar](
        a: jax.Float16Array[[M, K]],
        b: jax.Float16Array[[K, N]],
        wrong: jax.Float16Array[[Other, N]],
    ) -> None:
        assert_type(pipelined_matmul(a, b, 64, 64, 128), jax.Array[[M, N]])
        pipelined_matmul(a, wrong, 64, 64, 128)  # pyrefly: ignore[bad-argument-type]

    def check_wgmma[M: IntVar, K: IntVar, N: IntVar, Other: IntVar](
        acc: plgpu.AccRef[M, N],
        lhs: plgpu.SmemInRef[M, K],
        rhs: plgpu.SmemInRef[K, N],
        wrong: plgpu.SmemInRef[Other, N],
    ) -> None:
        plgpu.wgmma(acc, lhs, rhs)
        plgpu.wgmma(acc, lhs, wrong)  # pyrefly: ignore[bad-argument-type]

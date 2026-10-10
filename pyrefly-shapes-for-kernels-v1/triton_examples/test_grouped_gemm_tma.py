"""Check the descriptor-based kernel from Triton's tutorial 08 grouped GEMM.

The kernel body is copied from Triton's 08-grouped-gemm.py, NVIDIA copyright
2023–2025, under Triton's MIT license; only parameter annotations are added.
"""

import os
import unittest
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
import triton
import triton.language as tl
from shape_extensions import Int, IntVar
from triton.backends.compiler import GPUTarget

from triton_examples.testing import compile_ttir
from triton_library.semantic_jit import ConstExpr, semantic_jit

Groups = IntVar("Groups")
BM = IntVar("BM")
BN = IntVar("BN")
BK = IntVar("BK")


@semantic_jit
def grouped_matmul_tma_kernel(
    group_a_ptrs: "tl.PointerTable[Groups, Literal['read']]",
    group_b_ptrs: "tl.PointerTable[Groups, Literal['read']]",
    group_c_ptrs: "tl.PointerTable[Groups, Literal['write']]",
    group_gemm_sizes: "tl.PackedIntTable[Groups, 3]",
    g_lds: "tl.PackedIntTable[Groups, 3]",
    group_size: Int[Groups],
    NUM_SM: ConstExpr[int],
    BLOCK_SIZE_M: ConstExpr[Int[BM]],
    BLOCK_SIZE_N: ConstExpr[Int[BN]],
    BLOCK_SIZE_K: ConstExpr[Int[BK]],
    FP8: ConstExpr[bool],
):
    dtype = tl.float8e4nv if FP8 else tl.float16
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the gemm size of the current problem
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        if tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            # pick up a tile from the current gemm problem
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)

            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(dtype))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(dtype))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(dtype))

            a_desc = tl.make_tensor_descriptor(
                a_ptr,
                shape=[gm, gk],
                strides=[lda, 1],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
            )

            b_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[gn, gk],
                strides=[ldb, 1],
                block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
            )
            c_desc = tl.make_tensor_descriptor(
                c_ptr,
                shape=[gm, gn],
                strides=[ldc, 1],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
            )

            # iterate through the tiles in the current gemm problem
            while (
                tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles
            ):
                k = gk
                # figure out tile coordinates
                tile_idx_in_gemm = tile_idx - last_problem_end
                tile_m_idx = tile_idx_in_gemm // num_n_tiles
                tile_n_idx = tile_idx_in_gemm % num_n_tiles

                # do regular gemm here
                offs_am = tile_m_idx * BLOCK_SIZE_M
                offs_bn = tile_n_idx * BLOCK_SIZE_N

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                    a = a_desc.load([offs_am, kk * BLOCK_SIZE_K])
                    b = b_desc.load([offs_bn, kk * BLOCK_SIZE_K])
                    accumulator += tl.dot(a, b.T)

                offs_cm = tile_m_idx * BLOCK_SIZE_M
                offs_cn = tile_n_idx * BLOCK_SIZE_N

                c = accumulator.to(dtype)
                c_desc.store([offs_cm, offs_cn], c)

                # go to the next tile by advancing NUM_SM
                tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles


@dataclass(frozen=True)
class TmaGemmProblem[M: IntVar, K: IntVar, N: IntVar]:
    """One problem with B stored as [N, K] for descriptor loads."""

    a: torch.Tensor[[M, K]]
    b: torch.Tensor[[N, K]]


def checked_tma_problem[M: IntVar, K: IntVar, N: IntVar](
    a: torch.Tensor[[M, K]], b: torch.Tensor[[N, K]]
) -> TmaGemmProblem[M, K, N]:
    """Check the shared contracting dimension before packing a group."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("TMA grouped GEMM contraction dimensions must match")
    return TmaGemmProblem(a, b)


def checked_grouped_gemm_tma(
    problems: Sequence[TmaGemmProblem],
    *,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
    num_sm: int = 2,
) -> list[torch.Tensor]:
    """Check unmasked descriptor inputs and pack the five device arrays."""
    if not problems:
        raise ValueError("TMA grouped GEMM needs at least one problem")
    if any(
        type(block) is not int or block < 16 or block & (block - 1)
        for block in (block_m, block_n, block_k)
    ):
        raise ValueError("TMA tile sizes must be powers of two of at least 16")
    if type(num_sm) is not int or num_sm <= 0:
        raise ValueError("num_sm must be positive")
    device = problems[0].a.device
    shapes = []
    strides = []
    outputs = []
    for problem in problems:
        a, b = problem.a, problem.b
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
            raise ValueError("TMA grouped GEMM contraction dimensions must match")
        m, k = a.shape
        n = b.shape[0]
        if min(m, n, k) <= 0 or m % block_m or n % block_n or k % block_k:
            raise ValueError("Unmasked TMA grouped GEMM requires whole positive tiles")
        if (
            a.dtype != torch.float16
            or b.dtype != torch.float16
            or a.device != device
            or b.device != device
        ):
            raise ValueError("TMA grouped GEMM inputs must be float16 on one device")
        if a.stride(1) != 1 or b.stride(1) != 1 or a.stride(0) < k or b.stride(0) < k:
            raise ValueError(
                "TMA grouped GEMM requires nonoverlapping row-major strides"
            )
        # TMA requires at least 16-byte row alignment for float16 descriptors.
        if a.stride(0) % 8 or b.stride(0) % 8:
            raise ValueError("TMA grouped GEMM requires 16-byte aligned row strides")
        c = torch.empty((m, n), dtype=torch.float16, device=device)
        outputs.append(c)
        shapes.extend((m, n, k))
        strides.extend((a.stride(0), b.stride(0), c.stride(0)))
    if device.type != "cuda":
        raise ValueError("TMA grouped GEMM requires CUDA")
    if torch.cuda.get_device_capability(device)[0] < 9:
        raise ValueError("TMA grouped GEMM requires compute capability 9 or newer")
    a_ptrs = torch.tensor(
        [p.a.data_ptr() for p in problems], dtype=torch.int64, device=device
    )
    b_ptrs = torch.tensor(
        [p.b.data_ptr() for p in problems], dtype=torch.int64, device=device
    )
    c_ptrs = torch.tensor(
        [c.data_ptr() for c in outputs], dtype=torch.int64, device=device
    )
    sizes = torch.tensor(shapes, dtype=torch.int32, device=device)
    leading = torch.tensor(strides, dtype=torch.int32, device=device)

    # Triton's descriptor lowering allocates device-side scratch through this hook.
    def alloc_fn(size: int, alignment: int, stream: int | None) -> torch.Tensor:
        return torch.empty(size, dtype=torch.int8, device=device)

    cast(Any, triton).set_allocator(alloc_fn)
    grouped_matmul_tma_kernel[(num_sm,)](
        cast("tl.PointerTable[int, Literal['read']]", a_ptrs),
        cast("tl.PointerTable[int, Literal['read']]", b_ptrs),
        cast("tl.PointerTable[int, Literal['write']]", c_ptrs),
        cast("tl.PackedIntTable[int, 3]", sizes),
        cast("tl.PackedIntTable[int, 3]", leading),
        len(problems),
        num_sm,
        block_m,
        block_n,
        block_k,
        False,
    )
    return outputs


class GroupedGemmTmaTest(unittest.TestCase):
    """Exercise the grouped descriptor frontend and checked host boundary."""

    @unittest.skipIf(os.environ.get("TRITON_INTERPRET") == "1", "requires normal JIT")
    def test_frontend(self) -> None:
        for fp8 in (False, True):
            with self.subTest(fp8=fp8):
                ttir = compile_ttir(
                    grouped_matmul_tma_kernel,
                    {
                        "group_a_ptrs": "*i64",
                        "group_b_ptrs": "*i64",
                        "group_c_ptrs": "*i64",
                        "group_gemm_sizes": "*i32",
                        "g_lds": "*i32",
                        "group_size": "i32",
                    },
                    {
                        "NUM_SM": 2,
                        "BLOCK_SIZE_M": 128,
                        "BLOCK_SIZE_N": 128,
                        "BLOCK_SIZE_K": 64,
                        "FP8": fp8,
                    },
                    target=GPUTarget("cuda", 90, 32),
                )
                self.assertIn("tt.func public @grouped_matmul_tma_kernel", ttir)

    def test_reject_invalid_shapes_and_dtype(self) -> None:
        a = torch.ones((128, 64), dtype=torch.float16)
        b = torch.ones((128, 64), dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "contraction dimensions"):
            checked_tma_problem(a, cast(Any, b[:, :32]))
        with self.assertRaisesRegex(ValueError, "whole positive tiles"):
            checked_grouped_gemm_tma([checked_tma_problem(a[:127], b)])
        with self.assertRaisesRegex(ValueError, "float16"):
            checked_grouped_gemm_tma([checked_tma_problem(a.float(), b)])
        with self.assertRaisesRegex(ValueError, "row-major"):
            checked_grouped_gemm_tma(
                [
                    checked_tma_problem(
                        a, cast(Any, torch.ones((64, 128), dtype=torch.float16).T)
                    )
                ]
            )
        with self.assertRaisesRegex(ValueError, "aligned row strides"):
            checked_grouped_gemm_tma(
                [
                    checked_tma_problem(
                        a,
                        torch.empty_strided((128, 64), (65, 1), dtype=torch.float16),
                    )
                ]
            )
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            checked_grouped_gemm_tma([checked_tma_problem(a, b)])


if TYPE_CHECKING:

    def check_tma_roles[Groups: IntVar](
        a: tl.PointerTable[Groups, Literal["read"]],
        b: tl.PointerTable[Groups, Literal["read"]],
        c: tl.PointerTable[Groups, Literal["write"]],
        sizes: tl.PackedIntTable[Groups, 3],
        leading: tl.PackedIntTable[Groups, 3],
        count: Int[Groups],
        wrong_b: tl.PointerTable[Groups, Literal["write"]],
    ) -> None:
        grouped_matmul_tma_kernel(
            a, b, c, sizes, leading, count, 2, 128, 128, 64, False
        )
        grouped_matmul_tma_kernel(
            a,
            wrong_b,  # pyrefly: ignore[bad-argument-type]
            c,
            sizes,
            leading,
            count,
            2,
            128,
            128,
            64,
            False,
        )

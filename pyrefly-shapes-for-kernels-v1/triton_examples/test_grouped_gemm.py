"""Check Triton's tutorial 08 grouped GEMM against per-matrix host metadata.

The kernel body is copied from Triton's 08-grouped-gemm.py, NVIDIA copyright
2023–2025, under Triton's MIT license; only parameter annotations are added.
"""

import os
import unittest
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
import triton.language as tl
from shape_extensions import Int, IntVar

from triton_examples.testing import compile_ttir
from triton_library.semantic_jit import ConstExpr, semantic_jit

Groups = IntVar("Groups")
BM = IntVar("BM")
BN = IntVar("BN")
BK = IntVar("BK")


@semantic_jit
def grouped_matmul_kernel(
    group_a_ptrs: "tl.GroupAPointers[Groups]",
    group_b_ptrs: "tl.GroupBPointers[Groups]",
    group_c_ptrs: "tl.GroupCPointers[Groups]",
    group_gemm_sizes: "tl.GroupSizes[Groups]",
    g_lds: "tl.GroupLeadingDimensions[Groups]",
    group_size: Int[Groups],
    NUM_SM: ConstExpr[int],
    BLOCK_SIZE_M: ConstExpr[Int[BM]],
    BLOCK_SIZE_N: ConstExpr[Int[BN]],
    BLOCK_SIZE_K: ConstExpr[Int[BK]],
):
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
        # iterate through the tiles in the current gemm problem
        while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            # pick up a tile from the current gemm problem
            k = gk
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                # hint to Triton compiler to do proper loop pipelining
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                # assume full tile for now
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb
            c = accumulator.to(tl.float16)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]

            # assumes full tile for now
            tl.store(c_ptrs, c)

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles


@dataclass(frozen=True)
class GemmProblem[M: IntVar, K: IntVar, N: IntVar]:
    """One heterogeneous problem with matching contracting dimensions."""

    a: torch.Tensor[[M, K]]
    b: torch.Tensor[[K, N]]


def checked_problem[M: IntVar, K: IntVar, N: IntVar](
    a: torch.Tensor[[M, K]], b: torch.Tensor[[K, N]]
) -> GemmProblem[M, K, N]:
    """Type-check each problem's shape relationship before grouping it."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Grouped GEMM contraction dimensions must match")
    return GemmProblem(a, b)


def checked_grouped_gemm(
    problems: Sequence[GemmProblem],
    *,
    block_m: int = 16,
    block_n: int = 16,
    block_k: int = 16,
    num_sm: int = 2,
) -> list[torch.Tensor]:
    """Validate each unmasked matrix and construct the five device arrays."""
    if not problems:
        raise ValueError("Grouped GEMM needs at least one problem")
    if any(
        type(block) is not int or block < 16 or block & (block - 1)
        for block in (block_m, block_n, block_k)
    ):
        raise ValueError("Grouped GEMM tile sizes must be powers of two of at least 16")
    if type(num_sm) is not int or num_sm <= 0:
        raise ValueError("num_sm must be positive")
    device = problems[0].a.device
    shapes = []
    strides = []
    outputs = []
    for problem in problems:
        a, b = problem.a, problem.b
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise ValueError("Grouped GEMM contraction dimensions must match")
        m, k = a.shape
        n = b.shape[1]
        if min(m, k, n) <= 0 or m % block_m or n % block_n or k % block_k:
            raise ValueError("Unmasked grouped GEMM requires whole positive tiles")
        if (
            a.dtype != torch.float16
            or b.dtype != torch.float16
            or a.device != device
            or b.device != device
        ):
            raise ValueError("Grouped GEMM inputs must be float16 on one device")
        if a.stride(1) != 1 or b.stride(1) != 1 or a.stride(0) < k or b.stride(0) < n:
            raise ValueError(
                "Grouped GEMM requires nonoverlapping row-major input strides"
            )
        c = torch.empty((m, n), dtype=torch.float16, device=device)
        outputs.append(c)
        shapes.extend((m, n, k))
        strides.extend((a.stride(0), b.stride(0), c.stride(0)))
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
    grouped_matmul_kernel[(num_sm,)](
        cast("tl.GroupAPointers[int]", a_ptrs),
        cast("tl.GroupBPointers[int]", b_ptrs),
        cast("tl.GroupCPointers[int]", c_ptrs),
        cast("tl.GroupSizes[int]", sizes),
        cast("tl.GroupLeadingDimensions[int]", leading),
        len(problems),
        num_sm,
        block_m,
        block_n,
        block_k,
    )
    return outputs


class GroupedGemmTest(unittest.TestCase):
    """Test full-tile preconditions and grouped pointer frontend."""

    @unittest.skipIf(os.environ.get("TRITON_INTERPRET") == "1", "requires normal JIT")
    def test_frontend(self) -> None:
        ttir = compile_ttir(
            grouped_matmul_kernel,
            {
                "group_a_ptrs": "*i64",
                "group_b_ptrs": "*i64",
                "group_c_ptrs": "*i64",
                "group_gemm_sizes": "*i32",
                "g_lds": "*i32",
                "group_size": "i32",
            },
            {"NUM_SM": 2, "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 16},
        )
        self.assertIn("tt.func public @grouped_matmul_kernel", ttir)

    def test_reject_partial_tile_and_bad_contraction(self) -> None:
        a = torch.ones((16, 16), dtype=torch.float16)
        b = torch.ones((16, 16), dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "whole positive tiles"):
            checked_grouped_gemm([checked_problem(a[:15], b)])
        with self.assertRaisesRegex(ValueError, "contraction dimensions"):
            checked_problem(a, cast(Any, b[:15]))

    def test_reject_dtype_or_stride(self) -> None:
        a = torch.ones((16, 16), dtype=torch.float16)
        b = torch.ones((16, 16), dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "float16"):
            checked_grouped_gemm([checked_problem(a.float(), b)])
        with self.assertRaisesRegex(ValueError, "row-major"):
            checked_grouped_gemm([checked_problem(a.T, b)])

    @unittest.skipUnless(
        os.environ.get("TRITON_INTERPRET") == "1", "requires interpreter"
    )
    def test_grouped_result(self) -> None:
        problems = [
            checked_problem(
                torch.ones((16, 16), dtype=torch.float16),
                torch.eye(16, dtype=torch.float16),
            ),
            checked_problem(
                torch.ones((32, 16), dtype=torch.float16),
                2 * torch.eye(16, dtype=torch.float16),
            ),
        ]
        out = checked_grouped_gemm(problems)
        for result, problem in zip(out, problems, strict=True):
            torch.testing.assert_close(result, problem.a @ problem.b)


if TYPE_CHECKING:

    def check_problem[M: IntVar, K: IntVar, N: IntVar, Other: IntVar](
        a: torch.Tensor[[M, K]],
        b: torch.Tensor[[K, N]],
        wrong: torch.Tensor[[Other, N]],
    ) -> None:
        checked_problem(a, b)
        checked_problem(a, wrong)  # pyrefly: ignore[bad-argument-type]

    def check_tile[Rows: IntVar, Inner: IntVar, Cols: IntVar, Other: IntVar](
        a: tl.GroupATilePointers[Rows, Inner],
        b: tl.GroupBTilePointers[Inner, Cols],
        bad_b: tl.GroupBTilePointers[Other, Cols],
        c: tl.GroupCTilePointers[Rows, Cols],
    ) -> None:
        tl.store(c, tl.dot(tl.load(a), tl.load(b)))
        tl.dot(tl.load(a), tl.load(bad_b))  # pyrefly: ignore[bad-argument-type]

    def check_roles[Groups: IntVar](
        a: tl.GroupAPointers[Groups],
        b: tl.GroupBPointers[Groups],
        c: tl.GroupCPointers[Groups],
        sizes: tl.GroupSizes[Groups],
        strides: tl.GroupLeadingDimensions[Groups],
        count: Int[Groups],
    ) -> None:
        grouped_matmul_kernel(a, b, c, sizes, strides, count, 2, 16, 16, 16)
        grouped_matmul_kernel(
            a,
            a,  # pyrefly: ignore[bad-argument-type]
            c,
            sizes,
            strides,
            count,
            2,
            16,
            16,
            16,
        )

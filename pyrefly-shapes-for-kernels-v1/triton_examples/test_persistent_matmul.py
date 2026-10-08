"""Preserve Triton's tutorial 09 persistent matmul and check its host launch.

The two kernel bodies are copied from Triton's 09-persistent-matmul.py
(MIT; copyright 2018–2020 Philippe Tillet and 2020–2022 OpenAI).
Only semantic annotations are new.
"""

import os
import unittest
from typing import TYPE_CHECKING, Any, assert_type, cast

import torch
import triton
import triton.language as tl
from shape_extensions import Int, IntVar

from triton_examples.testing import compile_ttir
from triton_library import host_tensor, tlt
from triton_library.launch_layout import tiled_output
from triton_library.semantic_jit import ConstExpr, semantic_jit
from triton_library.torch_views import as_host_tensor, checked_matrix

M = IntVar("M")
N = IntVar("N")
K = IntVar("K")
AM = IntVar("AM")
AK = IntVar("AK")
BK = IntVar("BK")
BN = IntVar("BN")
CM = IntVar("CM")
CN = IntVar("CN")
BM = IntVar("BM")
BlockN = IntVar("BlockN")
BlockK = IntVar("BlockK")
Group = IntVar("Group")
SMs = IntVar("SMs")


@triton.jit
def _compute_pid(
    tile_id: int,
    num_pid_in_group: int,
    num_pid_m: int,
    GROUP_SIZE_M: "Int[Group]",
    NUM_SMS: "Int[SMs]",
):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@semantic_jit
def matmul_kernel_persistent(
    a_ptr: tlt.InPointer[[M, K], [AM, AK]],
    b_ptr: tlt.InPointer[[K, N], [BK, BN]],
    c_ptr: tlt.OutPointer[[M, N], [CM, CN]],
    M: Int[M],
    N: Int[N],
    K: Int[K],
    stride_am: Int[AM],
    stride_ak: Int[AK],
    stride_bk: Int[BK],
    stride_bn: Int[BN],
    stride_cm: Int[CM],
    stride_cn: Int[CN],
    BLOCK_SIZE_M: ConstExpr[Int[BM]],
    BLOCK_SIZE_N: ConstExpr[Int[BlockN]],
    BLOCK_SIZE_K: ConstExpr[Int[BlockK]],
    GROUP_SIZE_M: ConstExpr[Int[Group]],
    NUM_SMS: ConstExpr[Int[SMs]],
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    # NOTE: There is currently a bug in blackwell pipelining that means it can't handle a value being
    # used in both the prologue and epilogue, so we duplicate the counters as a work-around.
    tile_id_c = start_pid - NUM_SMS

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (
                offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
            )
            b_ptrs = b_ptr + (
                offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
            )

            a = tl.load(
                a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            b = tl.load(
                b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            accumulator = tl.dot(a, b, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if c_ptr.dtype.element_ty == tl.float8e4nv:
            c = accumulator.to(tl.float8e4nv)
        else:
            c = accumulator.to(tl.float16)
        tl.store(c_ptrs, c, mask=c_mask)


def persistent_matmul[Rows: IntVar, Inner: IntVar, Cols: IntVar](
    a: host_tensor.Tensor[[Rows, Inner], [int, int]],
    b: host_tensor.Tensor[[Inner, Cols], [int, int]],
    *,
    block_m: int = 16,
    block_n: int = 16,
    block_k: int = 16,
    group_m: int = 8,
    num_sms: int = 2,
) -> torch.Tensor[[Rows, Cols]]:
    """Bind two matrices to a persistent output-tile schedule."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Persistent matmul contraction dimensions must match")
    if min(*a.shape, b.shape[1]) <= 0:
        raise ValueError("Persistent matmul dimensions must be positive")
    if any(
        type(block) is not int or block < 16 or block & (block - 1)
        for block in (block_m, block_n, block_k)
    ):
        raise ValueError(
            "Persistent matmul blocks must be powers of two of at least 16"
        )
    if (
        type(group_m) is not int
        or group_m <= 0
        or type(num_sms) is not int
        or num_sms <= 0
    ):
        raise ValueError("Persistent matmul group and program counts must be positive")
    if a.device != b.device or a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("Persistent matmul inputs must be float16 on one device")
    if any(s <= 0 for s in (*a.stride(), *b.stride())):
        raise ValueError("Persistent matmul requires positive input strides")
    rows, inner = a.shape
    cols = b.shape[1]
    a_ptr, am, _, _ = checked_matrix(a, tlt.InPointer[[Rows, Inner], [int, int]])
    b_ptr, bk, _, _ = checked_matrix(b, tlt.InPointer[[Inner, Cols], [int, int]])
    out = torch.empty((rows, cols), dtype=torch.float16, device=a.device)
    c_view = as_host_tensor(out, host_tensor.Tensor[[Rows, Cols], [int, 1]])
    c_ptr, cm, _, _ = checked_matrix(c_view, tlt.OutPointer[[Rows, Cols], [int, 1]])
    layout = tiled_output(
        c_view,
        (block_m, block_n),
        shape_parameters=("M", "N"),
        tile_parameters=("BLOCK_SIZE_M", "BLOCK_SIZE_N"),
        metadata={"GROUP_SIZE_M": group_m, "NUM_SMS": num_sms},
        persistent_programs=num_sms,
    )
    layout.launch(
        matmul_kernel_persistent,
        a_ptr,
        b_ptr,
        c_ptr,
        rows,
        cols,
        inner,
        am,
        a.stride(1),
        bk,
        b.stride(1),
        cm,
        out.stride(1),
        block_m,
        block_n,
        block_k,
        group_m,
        num_sms,
    )
    return out


class PersistentMatmulTest(unittest.TestCase):
    """Check persistent launch metadata, frontend and CPU numerical behavior."""

    def test_reject_wrong_contract_and_dtype(self) -> None:
        a = torch.ones((20, 24), dtype=torch.float16)
        b = torch.ones((24, 19), dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "contraction dimensions"):
            persistent_matmul(as_host_tensor(a), as_host_tensor(cast(Any, b[:23])))
        with self.assertRaisesRegex(ValueError, "float16"):
            persistent_matmul(as_host_tensor(a.float()), as_host_tensor(b))

    def test_persistent_grid(self) -> None:
        out = torch.empty((36, 19), dtype=torch.float16)
        view = as_host_tensor(out, host_tensor.Tensor[[36, 19], [int, 1]])
        layout = tiled_output(
            view,
            (16, 16),
            shape_parameters=("M", "N"),
            tile_parameters=("BLOCK_SIZE_M", "BLOCK_SIZE_N"),
            metadata={"GROUP_SIZE_M": 8, "NUM_SMS": 2},
            persistent_programs=2,
        )
        self.assertEqual(layout.grid, (2,))  # Six output tiles, two programs.
        a = torch.empty((36, 16), dtype=torch.float16)
        b = torch.empty((16, 19), dtype=torch.float16)
        a_ptr, _, _, _ = checked_matrix(
            as_host_tensor(a), tlt.InPointer[[36, 16], [int, 1]]
        )
        b_ptr, _, _, _ = checked_matrix(
            as_host_tensor(b), tlt.InPointer[[16, 19], [int, 1]]
        )
        c_ptr, _, _, _ = checked_matrix(view, tlt.OutPointer[[36, 19], [int, 1]])
        with self.assertRaisesRegex(ValueError, "NUM_SMS"):
            layout.launch(
                matmul_kernel_persistent,
                a_ptr,
                b_ptr,
                c_ptr,
                36,
                19,
                16,
                16,
                1,
                19,
                1,
                19,
                1,
                16,
                16,
                16,
                8,
                3,
            )

    @unittest.skipIf(os.environ.get("TRITON_INTERPRET") == "1", "requires normal JIT")
    def test_frontend(self) -> None:
        ttir = compile_ttir(
            matmul_kernel_persistent,
            {
                "a_ptr": "*fp16",
                "b_ptr": "*fp16",
                "c_ptr": "*fp16",
                "M": "i32",
                "N": "i32",
                "K": "i32",
                "stride_am": "i32",
                "stride_ak": "i32",
                "stride_bk": "i32",
                "stride_bn": "i32",
                "stride_cm": "i32",
                "stride_cn": "i32",
            },
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 16,
                "BLOCK_SIZE_K": 16,
                "GROUP_SIZE_M": 8,
                "NUM_SMS": 2,
            },
        )
        self.assertIn("tt.func public @matmul_kernel_persistent", ttir)

    @unittest.skipUnless(
        os.environ.get("TRITON_INTERPRET") == "1", "requires interpreter"
    )
    def test_irregular_shapes_with_persistent_programs(self) -> None:
        a = torch.arange(37 * 23, dtype=torch.float16).reshape(37, 23) / 100
        b = torch.arange(23 * 19, dtype=torch.float16).reshape(23, 19) / 100
        out = persistent_matmul(as_host_tensor(a), as_host_tensor(b), num_sms=2)
        torch.testing.assert_close(out, a @ b, atol=0.05, rtol=0.02)

    @unittest.skipUnless(
        os.environ.get("TRITON_INTERPRET") == "1", "requires interpreter"
    )
    def test_padded_lhs_and_transposed_rhs(self) -> None:
        a = (torch.arange(37 * 32, dtype=torch.float16) / 100).reshape(37, 32)[:, :23]
        b = (torch.arange(19 * 23, dtype=torch.float16) / 100).reshape(19, 23).T
        out = persistent_matmul(
            as_host_tensor(a, host_tensor.Tensor[[37, 23], [int, int]]),
            as_host_tensor(b, host_tensor.Tensor[[23, 19], [int, int]]),
            num_sms=2,
        )
        torch.testing.assert_close(out, a @ b, atol=0.05, rtol=0.02)


if TYPE_CHECKING:

    def check_interface[Rows: IntVar, Inner: IntVar, Cols: IntVar, Other: IntVar](
        a: host_tensor.Tensor[[Rows, Inner], [int, int]],
        b: host_tensor.Tensor[[Inner, Cols], [int, int]],
        wrong: host_tensor.Tensor[[Other, Cols], [int, int]],
    ) -> None:
        assert_type(persistent_matmul(a, b), torch.Tensor[[Rows, Cols]])
        persistent_matmul(a, wrong)  # pyrefly: ignore[bad-argument-type]

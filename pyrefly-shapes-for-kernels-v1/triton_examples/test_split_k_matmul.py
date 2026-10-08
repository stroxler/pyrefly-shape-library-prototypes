"""All four executable JIT kernels from Triton's tutorial 12.

Adapted from 12-split-k-matmul.py (MIT; copyright 2018–2020 Philippe Tillet
and 2020–2022 OpenAI). Semantic annotations leave executable bodies unchanged.
The host adapter enforces shared input dimensions, output dtype, scratch
shape/strides, and zero initialization for split-K atomics. It does not prove
grouped program-ID coverage, the arithmetic partitioning K into splits, or
that separate launch stages execute in order beyond Python control flow.
"""

from __future__ import annotations

import os
import unittest
from typing import TYPE_CHECKING, Any, Literal, assert_type, cast

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
Split = IntVar("Split")
PerSplit = IntVar("PerSplit")
SK = IntVar("SK")
SM = IntVar("SM")
SN = IntVar("SN")


@semantic_jit
def _stock_triton_kernel(
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
    BLOCK_M: ConstExpr[Int[BM]],
    BLOCK_N: ConstExpr[Int[BlockN]],
    BLOCK_K: ConstExpr[Int[BlockK]],
    GROUP_SIZE_M: ConstExpr[Int[Group]],
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@semantic_jit
def _skinny_atomic_kernel(
    a_ptr: tlt.InPointer[[M, K], [AM, AK]],
    b_ptr: tlt.InPointer[[K, N], [BK, BN]],
    c_ptr: tlt.OutPointer[[M, N], [CM, CN]],
    M: Int[M],
    N: Int[N],
    K: Int[K],
    K_PER_SPLIT: Int[PerSplit],
    stride_am: Int[AM],
    stride_ak: Int[AK],
    stride_bk: Int[BK],
    stride_bn: Int[BN],
    stride_cm: Int[CM],
    stride_cn: Int[CN],
    SPLIT_K: ConstExpr[Int[Split]],
    BLOCK_M: ConstExpr[Int[BM]],
    BLOCK_N: ConstExpr[Int[BlockN]],
    BLOCK_K: ConstExpr[Int[BlockK]],
    GROUP_SIZE_M: ConstExpr[Int[Group]],
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    k_start = pid_k * K_PER_SPLIT
    k_end = min(k_start + K_PER_SPLIT, K)
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = (
        a_ptr + offs_am[:, None] * stride_am + (k_start + offs_k[None, :]) * stride_ak
    )
    b_ptrs = (
        b_ptr + (k_start + offs_k[:, None]) * stride_bk + offs_bn[None, :] * stride_bn
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K_PER_SPLIT, BLOCK_K)):
        k_remaining = k_end - (k_start + k * BLOCK_K)
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)


@semantic_jit
def _twopass_compute_kernel(
    a_ptr: tlt.InPointer[[M, K], [AM, AK]],
    b_ptr: tlt.InPointer[[K, N], [BK, BN]],
    scratch_ptr: tl.Scratch3DPointer[Split, M, N, SK, SM, SN],
    M: Int[M],
    N: Int[N],
    K: Int[K],
    K_PER_SPLIT: Int[PerSplit],
    stride_am: Int[AM],
    stride_ak: Int[AK],
    stride_bk: Int[BK],
    stride_bn: Int[BN],
    stride_sm: Int[SM],
    stride_sn: Int[SN],
    stride_sk: tl.SplitStride[SK],
    SPLIT_K: ConstExpr[Int[Split]],
    BLOCK_M: ConstExpr[Int[BM]],
    BLOCK_N: ConstExpr[Int[BlockN]],
    BLOCK_K: ConstExpr[Int[BlockK]],
    GROUP_SIZE_M: ConstExpr[Int[Group]],
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    k_start = pid_k * K_PER_SPLIT
    k_end = min(k_start + K_PER_SPLIT, K)
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = (
        a_ptr + offs_am[:, None] * stride_am + (k_start + offs_k[None, :]) * stride_ak
    )
    b_ptrs = (
        b_ptr + (k_start + offs_k[:, None]) * stride_bk + offs_bn[None, :] * stride_bn
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K_PER_SPLIT, BLOCK_K)):
        k_remaining = k_end - (k_start + k * BLOCK_K)
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    # Store fp32 partial result into scratch[pid_k, :, :]
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    scratch_ptrs = (
        scratch_ptr
        + pid_k * stride_sk
        + offs_cm[:, None] * stride_sm
        + offs_cn[None, :] * stride_sn
    )
    tl.store(scratch_ptrs, acc, mask=c_mask)


@semantic_jit
def _twopass_reduce_kernel(
    scratch_ptr: tl.Scratch3DPointer[Split, M, N, SK, SM, SN],
    c_ptr: tlt.OutPointer[[M, N], [CM, CN]],
    M: Int[M],
    N: Int[N],
    stride_sm: Int[SM],
    stride_sn: Int[SN],
    stride_sk: tl.SplitStride[SK],
    stride_cm: Int[CM],
    stride_cn: Int[CN],
    SPLIT_K: ConstExpr[Int[Split]],
    BLOCK_M: ConstExpr[Int[BM]],
    BLOCK_N: ConstExpr[Int[BlockN]],
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Sum across split-K slices
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for sk in range(SPLIT_K):
        s_ptrs = (
            scratch_ptr
            + sk * stride_sk
            + offs_m[:, None] * stride_sm
            + offs_n[None, :] * stride_sn
        )
        partial = tl.load(s_ptrs, mask=mask, other=0.0)
        acc += partial
    # Store as fp16
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)


def checked_scratch[Parts: IntVar, Rows: IntVar, Cols: IntVar](
    scratch: torch.Tensor[[Parts, Rows, Cols]],
    split: int,
    rows: int,
    cols: int,
) -> tl.Scratch3DPointer[Parts, Rows, Cols, Rows * Cols, Cols, 1]:
    """Check the temporary reduction allocation before exposing its pointer role."""
    if (
        tuple(scratch.shape) != (split, rows, cols)
        or tuple(scratch.stride()) != (rows * cols, cols, 1)
        or scratch.dtype != torch.float32
        or not scratch.is_contiguous()
    ):
        raise ValueError("Split-K scratch must be dense float32 [split, rows, cols]")
    return cast("tl.Scratch3DPointer[Parts, Rows, Cols, Rows * Cols, Cols, 1]", scratch)


def split_k_matmul[Rows: IntVar, Inner: IntVar, Cols: IntVar](
    a: host_tensor.Tensor[[Rows, Inner], [int, int]],
    b: host_tensor.Tensor[[Inner, Cols], [int, int]],
    *,
    mode: Literal["stock", "atomic", "twopass"] = "twopass",
    split_k: int = 2,
    block_m: int = 16,
    block_n: int = 16,
    block_k: int = 16,
    group_m: int = 8,
) -> torch.Tensor[[Rows, Cols]]:
    """Check matrix shapes, split scratch, and atomic initialization before launch."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Split-K contraction dimensions must match")
    if min(*a.shape, b.shape[1]) <= 0:
        raise ValueError("Split-K dimensions must be positive")
    if a.device != b.device or a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("Split-K inputs must be float16 on one device")
    if any(stride <= 0 for stride in (*a.stride(), *b.stride())):
        raise ValueError("Split-K requires positive input strides")
    if any(
        type(block) is not int or block < 16 or block & (block - 1)
        for block in (block_m, block_n, block_k)
    ):
        raise ValueError("Split-K blocks must be powers of two of at least 16")
    if (
        type(group_m) is not int
        or group_m < 1
        or type(split_k) is not int
        or not 1 <= split_k <= a.shape[1]
    ):
        raise ValueError("Split-K needs positive group and 1 <= split <= K")
    if mode not in ("stock", "atomic", "twopass"):
        raise ValueError("Unknown split-K reduction mode")
    rows, inner = a.shape
    cols = b.shape[1]
    per_split = (inner + split_k - 1) // split_k
    a_ptr, am, _, _ = checked_matrix(a, tlt.InPointer[[Rows, Inner], [int, int]])
    b_ptr, bk, _, _ = checked_matrix(b, tlt.InPointer[[Inner, Cols], [int, int]])
    # Atomic FP16 updates need a zeroed allocation when split_k > 1.
    alloc = torch.zeros if mode == "atomic" and split_k > 1 else torch.empty
    out = alloc((rows, cols), dtype=torch.float16, device=a.device)
    c_view = as_host_tensor(out, host_tensor.Tensor[[Rows, Cols], [int, 1]])
    c_ptr, cm, _, _ = checked_matrix(c_view, tlt.OutPointer[[Rows, Cols], [int, 1]])
    layout = tiled_output(
        c_view,
        (block_m, block_n),
        shape_parameters=("M", "N"),
        tile_parameters=("BLOCK_M", "BLOCK_N"),
        metadata={"GROUP_SIZE_M": group_m},
    )
    if mode == "stock":
        layout.launch(
            _stock_triton_kernel,
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
        )
    elif mode == "atomic":
        _skinny_atomic_kernel[(layout.grid[0], split_k)](
            a_ptr,
            b_ptr,
            c_ptr,
            rows,
            cols,
            inner,
            per_split,
            am,
            a.stride(1),
            bk,
            b.stride(1),
            cm,
            out.stride(1),
            split_k,
            block_m,
            block_n,
            block_k,
            group_m,
        )
    else:
        scratch = torch.empty(
            (split_k, rows, cols), dtype=torch.float32, device=a.device
        )
        scratch_ptr = checked_scratch(scratch, split_k, rows, cols)
        slice_stride = cast("tl.SplitStride[Rows * Cols]", scratch.stride(0))
        _twopass_compute_kernel[(layout.grid[0], split_k)](
            a_ptr,
            b_ptr,
            scratch_ptr,
            rows,
            cols,
            inner,
            per_split,
            am,
            a.stride(1),
            bk,
            b.stride(1),
            scratch.stride(1),
            scratch.stride(2),
            slice_stride,
            split_k,
            block_m,
            block_n,
            block_k,
            group_m,
        )
        _twopass_reduce_kernel[layout.grid](
            scratch_ptr,
            c_ptr,
            rows,
            cols,
            scratch.stride(1),
            scratch.stride(2),
            slice_stride,
            cm,
            out.stride(1),
            split_k,
            block_m,
            block_n,
        )
    return out


class SplitKMatmulTest(unittest.TestCase):
    """Exercise four unchanged kernels and their checked split-K launch seam."""

    @unittest.skipIf(os.environ.get("TRITON_INTERPRET") == "1", "requires normal JIT")
    def test_all_frontends(self) -> None:
        for name, kernel, extra, constexpr in (
            ("_stock_triton_kernel", _stock_triton_kernel, {}, {}),
            (
                "_skinny_atomic_kernel",
                _skinny_atomic_kernel,
                {"K_PER_SPLIT": "i32"},
                {"SPLIT_K": 2},
            ),
            (
                "_twopass_compute_kernel",
                _twopass_compute_kernel,
                {
                    "scratch_ptr": "*fp32",
                    "K_PER_SPLIT": "i32",
                    "stride_sm": "i32",
                    "stride_sn": "i32",
                    "stride_sk": "i32",
                },
                {"SPLIT_K": 2},
            ),
            (
                "_twopass_reduce_kernel",
                _twopass_reduce_kernel,
                {
                    "scratch_ptr": "*fp32",
                    "stride_sm": "i32",
                    "stride_sn": "i32",
                    "stride_sk": "i32",
                },
                {"SPLIT_K": 2},
            ),
        ):
            with self.subTest(kernel=name):
                signature = {
                    "c_ptr": "*fp16",
                    "M": "i32",
                    "N": "i32",
                    "stride_cm": "i32",
                    "stride_cn": "i32",
                }
                if name != "_twopass_reduce_kernel":
                    signature.update(
                        {
                            "a_ptr": "*fp16",
                            "b_ptr": "*fp16",
                            "K": "i32",
                            "stride_am": "i32",
                            "stride_ak": "i32",
                            "stride_bk": "i32",
                            "stride_bn": "i32",
                        }
                    )
                if name == "_twopass_compute_kernel":
                    del signature["c_ptr"]
                    del signature["stride_cm"]
                    del signature["stride_cn"]
                signature.update(extra)
                metadata: dict[str, int | str] = {"BLOCK_M": 16, "BLOCK_N": 16}
                if name != "_twopass_reduce_kernel":
                    metadata.update({"BLOCK_K": 16, "GROUP_SIZE_M": 8})
                metadata.update(constexpr)
                ttir = compile_ttir(kernel, signature, metadata)
                self.assertIn(f"tt.func public @{name}", ttir)

    def test_reject_invalid_host_shapes(self) -> None:
        a = torch.ones((18, 34), dtype=torch.float16)
        b = torch.ones((34, 21), dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "contraction dimensions"):
            split_k_matmul(as_host_tensor(a), as_host_tensor(cast(Any, b[:33])))
        with self.assertRaisesRegex(ValueError, "float16"):
            split_k_matmul(as_host_tensor(a.float()), as_host_tensor(b))
        with self.assertRaisesRegex(ValueError, "1 <= split"):
            split_k_matmul(as_host_tensor(a), as_host_tensor(b), split_k=35)
        scratch = torch.empty((2, 18, 21), dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "dense float32"):
            checked_scratch(scratch, 2, 18, 21)
        padded = torch.empty((2, 18, 24), dtype=torch.float32)[:, :, :21]
        with self.assertRaisesRegex(ValueError, "dense float32"):
            checked_scratch(padded, 2, 18, 21)

    @unittest.skipUnless(
        os.environ.get("TRITON_INTERPRET") == "1", "requires interpreter"
    )
    def test_cpu_irregular_k_and_masked_tiles(self) -> None:
        a = (torch.arange(18 * 34, dtype=torch.float16) / 100).reshape(18, 34)
        b = (torch.arange(34 * 21, dtype=torch.float16) / 120).reshape(34, 21)
        for mode, split in (
            ("stock", 1),
            ("atomic", 1),
            ("atomic", 2),
            ("twopass", 1),
            ("twopass", 2),
        ):
            with self.subTest(mode=mode, split=split):
                out = split_k_matmul(
                    as_host_tensor(a),
                    as_host_tensor(b),
                    mode=mode,
                    split_k=split,
                )
                torch.testing.assert_close(out, a @ b, atol=0.1, rtol=0.04)

    @unittest.skipUnless(
        os.environ.get("TRITON_INTERPRET") == "1", "requires interpreter"
    )
    def test_padded_inputs_and_transposed_rhs(self) -> None:
        a = (torch.arange(18 * 40, dtype=torch.float16) / 100).reshape(18, 40)[:, :34]
        b = (torch.arange(21 * 34, dtype=torch.float16) / 120).reshape(21, 34).T
        for mode in ("stock", "atomic", "twopass"):
            with self.subTest(mode=mode):
                out = split_k_matmul(
                    as_host_tensor(a, host_tensor.Tensor[[18, 34], [int, int]]),
                    as_host_tensor(b, host_tensor.Tensor[[34, 21], [int, int]]),
                    mode=mode,
                    split_k=2,
                )
                torch.testing.assert_close(out, a @ b, atol=0.1, rtol=0.04)


if TYPE_CHECKING:

    def check_split_grid_axis[
        Parts: IntVar,
        Rows: IntVar,
        Cols: IntVar,
        Slice: IntVar,
        RowStride: IntVar,
        ColumnStride: IntVar,
    ](
        scratch: tl.Scratch3DPointer[Parts, Rows, Cols, Slice, RowStride, ColumnStride],
        slice_stride: tl.SplitStride[Slice],
    ) -> None:
        scratch + tl.program_id(1) * slice_stride
        scratch + tl.program_id(0) * slice_stride  # pyrefly: ignore[unsupported-operation]

    def check_address_and_atomic_masks[
        Rows: IntVar,
        Cols: IntVar,
        Inner: IntVar,
        RS: IntVar,
        CS: IntVar,
        Other: IntVar,
        BR: IntVar,
        BC: IntVar,
    ](
        a: tlt.InPointer[[Rows, Inner], [RS, CS]],
        row: tl.WrappedRowAddress[Rows, [BR], RS],
        wrong_row: tl.WrappedRowAddress[Rows, [BR], Other],
        column: tl.ColumnAddress[[BC], CS],
        output: tl.OutTilePointers[
            [Rows, Cols], [RS, CS], [BR, BC], Literal["indexed"]
        ],
        tile: tl.tensor[[BR, BC]],
        mask: tl.MatrixMask[Rows, Cols, [BR], [BC]],
        wrong_mask: tl.MatrixMask[Rows, Other, [BR], [BC]],
    ) -> None:
        a + row + column
        a + wrong_row + column  # pyrefly: ignore[unsupported-operation]
        tl.atomic_add(output, tile, mask=mask)
        tl.atomic_add(output, tile, mask=wrong_mask)  # pyrefly: ignore[bad-argument-type]

    def check_split_contract[Rows: IntVar, Inner: IntVar, Cols: IntVar, Other: IntVar](
        a: host_tensor.Tensor[[Rows, Inner], [int, int]],
        b: host_tensor.Tensor[[Inner, Cols], [int, int]],
        wrong: host_tensor.Tensor[[Other, Cols], [int, int]],
    ) -> None:
        assert_type(split_k_matmul(a, b), torch.Tensor[[Rows, Cols]])
        split_k_matmul(a, wrong)  # pyrefly: ignore[bad-argument-type]

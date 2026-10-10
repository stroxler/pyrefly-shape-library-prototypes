"""Preserve Triton's tutorial 03 grouped matmul with checked matrix views."""

import os
import unittest
from typing import TYPE_CHECKING, Any, Literal, assert_type, cast

import torch
import triton
import triton.language as tl
from shape_extensions import Int, IntVar

from triton_examples.testing import compile_ttir
from triton_library import host_tensor, tlt
from triton_library.launch_layout import TiledOutputLayout, tiled_output
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


@semantic_jit
def matmul_kernel(
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
    ACTIVATION: ConstExpr[str],
):
    """Kernel for computing the matmul C = A x B."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def leaky_relu(x: "tl.tensor[[BM, BlockN]]") -> "tl.tensor[[BM, BlockN]]":
    return tl.where(x >= 0, x, 0.01 * x)


def checked_matmul[Rows: IntVar, Inner: IntVar, Cols: IntVar](
    a: host_tensor.Tensor[[Rows, Inner], [int, int]],
    b: host_tensor.Tensor[[Inner, Cols], [int, int]],
    *,
    block_m: int = 16,
    block_n: int = 16,
    block_k: int = 16,
    group_m: int = 8,
) -> torch.Tensor[[Rows, Cols]]:
    """Validate matrix shapes and strides before launching grouped matmul."""
    rows, inner = a.shape
    if b.shape[0] != inner:
        raise ValueError("Matrix contraction dimensions must match")
    cols = b.shape[1]
    if min(rows, inner, cols, block_m, block_n, block_k, group_m) <= 0:
        raise ValueError("Matrix dimensions and tiling metadata must be positive")
    if any(block < 16 or block & (block - 1) for block in (block_m, block_n, block_k)):
        raise ValueError("Matmul blocks must be powers of two of at least 16")
    if any(stride <= 0 for stride in (*a.stride(), *b.stride())):
        raise ValueError("Matmul inputs must have positive element strides")
    if a.device != b.device or a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("Matmul inputs must be float16 and on one device")
    a_ptr, stride_am, _, _ = checked_matrix(a, tlt.InPointer[[Rows, Inner], [int, int]])
    b_ptr, stride_bk, _, _ = checked_matrix(b, tlt.InPointer[[Inner, Cols], [int, int]])
    out = torch.empty((rows, cols), device=a.device, dtype=torch.float16)
    c_view = as_host_tensor(out, host_tensor.Tensor[[Rows, Cols], [int, 1]])
    c_ptr, stride_cm, _, _ = checked_matrix(
        c_view, tlt.OutPointer[[Rows, Cols], [int, 1]]
    )
    layout = tiled_output(
        c_view,
        (block_m, block_n),
        shape_parameters=("M", "N"),
        tile_parameters=("BLOCK_SIZE_M", "BLOCK_SIZE_N"),
        metadata={"GROUP_SIZE_M": group_m},
    )
    layout.launch(
        matmul_kernel,
        a_ptr,
        b_ptr,
        c_ptr,
        rows,
        cols,
        inner,
        stride_am,
        a.stride(1),
        stride_bk,
        b.stride(1),
        stride_cm,
        out.stride(1),
        block_m,
        block_n,
        block_k,
        group_m,
        "",
    )
    return out


class GroupedMatmulTest(unittest.TestCase):
    """Check the matmul frontend and the shape/stride launch boundary."""

    def test_reject_zero_input_stride(self) -> None:
        a = torch.empty((1, 24), dtype=torch.float16).expand(16, 24)
        b = torch.empty((24, 16), dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "positive element strides"):
            checked_matmul(
                as_host_tensor(a, host_tensor.Tensor[[16, 24], [int, int]]),
                as_host_tensor(b, host_tensor.Tensor[[24, 16], [int, int]]),
            )

    def test_reject_wrong_declared_column_stride(self) -> None:
        b = cast(torch.Tensor[[24, 19]], torch.empty((19, 24), dtype=torch.float16).T)
        with self.assertRaisesRegex(ValueError, "element stride"):
            as_host_tensor(b, host_tensor.Tensor[[24, 19], [int, 1]])

    def test_tiled_output_layout_checks_launch_metadata(self) -> None:
        a = torch.empty((20, 24), dtype=torch.float16)
        b = torch.empty((24, 19), dtype=torch.float16)
        out = torch.empty((20, 19), dtype=torch.float16)
        view = as_host_tensor(out, host_tensor.Tensor[[20, 19], [int, 1]])
        layout = tiled_output(
            view,
            (16, 16),
            shape_parameters=("M", "N"),
            tile_parameters=("BLOCK_SIZE_M", "BLOCK_SIZE_N"),
            metadata={"GROUP_SIZE_M": 8},
        )
        self.assertEqual(layout.grid, (4,))
        args = (
            a,
            b,
            out,
            20,
            19,
            24,
            *a.stride(),
            *b.stride(),
            *out.stride(),
            16,
            16,
            16,
            8,
            "",
        )
        for index, value, name in (
            (3, 21, "M"),
            (13, 32, "BLOCK_SIZE_N"),
            (15, 4, "GROUP_SIZE_M"),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                bad_args = list(args)
                bad_args[index] = value
                # These calls deliberately violate the checked launch contract.
                cast(Any, layout.launch)(matmul_kernel, *bad_args)

    @unittest.skipIf(os.environ.get("TRITON_INTERPRET") == "1", "requires JIT")
    def test_frontend_ttir(self) -> None:
        signature = {
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
        }
        constexprs = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 16,
            "BLOCK_SIZE_K": 16,
            "GROUP_SIZE_M": 8,
            "ACTIVATION": "",
        }
        module = compile_ttir(matmul_kernel, signature, constexprs)
        self.assertIn("tt.func public @matmul_kernel", module)
        activated = compile_ttir(
            matmul_kernel,
            signature,
            {**constexprs, "ACTIVATION": "leaky_relu"},
        )
        self.assertIn("tt.func public @matmul_kernel", activated)

    @unittest.skipIf(os.environ.get("TRITON_INTERPRET") == "1", "requires JIT")
    def test_launch_contract_without_gpu(self) -> None:
        a = torch.ones((16, 24), dtype=torch.float16)
        b = torch.ones((24, 16), dtype=torch.float16)
        out = torch.empty((16, 16), dtype=torch.float16)
        hook = cast(Any, matmul_kernel).pre_run_hooks[0]
        with self.assertRaisesRegex(ValueError, "stride_bn"):
            hook(
                a,
                b,
                out,
                16,
                16,
                24,
                *a.stride(),
                b.stride(0),
                0,
                *out.stride(),
                16,
                16,
                16,
                8,
                "",
            )

    @unittest.skipUnless(
        os.environ.get("TRITON_INTERPRET") == "1", "requires interpreter"
    )
    def test_matmul_with_padded_rows_and_transposed_rhs(self) -> None:
        a = (torch.arange(20 * 32, dtype=torch.float16) / 100).reshape(20, 32)[:, :24]
        b = (torch.arange(24 * 19, dtype=torch.float16) / 100).reshape(19, 24).T
        result = checked_matmul(
            as_host_tensor(a, host_tensor.Tensor[[20, 24], [int, int]]),
            as_host_tensor(b, host_tensor.Tensor[[24, 19], [int, int]]),
        )
        torch.testing.assert_close(result, a @ b, atol=0.1, rtol=1e-2)

    @unittest.skipUnless(
        os.environ.get("TRITON_INTERPRET") == "1", "requires interpreter"
    )
    def test_annotated_activation_helper(self) -> None:
        a = torch.eye(16, dtype=torch.float16)
        b = torch.arange(256, dtype=torch.float16).reshape(16, 16) / 32 - 4
        out = torch.empty((16, 16), dtype=torch.float16)
        a_ptr, stride_am, _, _ = checked_matrix(
            as_host_tensor(a, host_tensor.Tensor[[16, 16], [16, 1]]),
            tlt.InPointer[[16, 16], [16, 1]],
        )
        b_ptr, stride_bk, _, _ = checked_matrix(
            as_host_tensor(b, host_tensor.Tensor[[16, 16], [16, 1]]),
            tlt.InPointer[[16, 16], [16, 1]],
        )
        c_ptr, stride_cm, _, _ = checked_matrix(
            as_host_tensor(out, host_tensor.Tensor[[16, 16], [16, 1]]),
            tlt.OutPointer[[16, 16], [16, 1]],
        )
        matmul_kernel[(1,)](
            a_ptr,
            b_ptr,
            c_ptr,
            16,
            16,
            16,
            stride_am,
            1,
            stride_bk,
            1,
            stride_cm,
            1,
            16,
            16,
            16,
            8,
            "leaky_relu",
        )
        torch.testing.assert_close(out, torch.where(b >= 0, b, 0.01 * b))


if TYPE_CHECKING:

    def check_clamped_address[
        Rows: IntVar,
        Cols: IntVar,
        BR: IntVar,
        BC: IntVar,
        RS: IntVar,
        CS: IntVar,
    ](
        ptr: tlt.InPointer[[Rows, Cols], [RS, CS]],
        row: tl.BoundedAxisAddress[Rows, [BR], RS, Literal["clamped"], Literal[0]],
        wrong_bound: tl.BoundedAxisAddress[
            Cols, [BR], RS, Literal["clamped"], Literal[0]
        ],
        wrong_step: tl.BoundedAxisAddress[
            Rows, [BR], CS, Literal["clamped"], Literal[0]
        ],
        col: tl.ColumnAddress[BC, CS],
    ) -> None:
        assert_type(
            ptr + (row + col),
            tl.InTilePointers[[Rows, Cols], [RS, CS], [BR, BC], Literal["clamped_0"]],
        )
        ptr + (wrong_bound + col)  # pyrefly: ignore[unsupported-operation]
        ptr + (wrong_step + col)  # pyrefly: ignore[unsupported-operation]

    helper_tile: tl.tensor[[16, 19]] = cast(Any, None)
    assert_type(leaky_relu(helper_tile), tl.tensor[[16, 19]])
    helper_vector: tl.tensor[[16]] = cast(Any, None)
    leaky_relu(helper_vector)  # pyrefly: ignore[bad-argument-type]

    tile_ptrs: tl.InTilePointers[
        [20, 24], [int, int], [16, 16], Literal["wrapped_0"]
    ] = cast(Any, None)
    k_mask: tl.Mask[[1, 24], [1, 16]] = cast(Any, None)
    assert_type(tl.load(tile_ptrs, mask=k_mask, other=0.0), tl.tensor[[16, 16]])
    row_mask: tl.Mask[[20, 1], [16, 1]] = cast(Any, None)
    tl.load(tile_ptrs, mask=row_mask, other=0.0)  # pyrefly: ignore[no-matching-overload]
    wrong_k_tile: tl.Mask[[1, 24], [1, 8]] = cast(Any, None)
    tl.load(tile_ptrs, mask=wrong_k_tile, other=0.0)  # pyrefly: ignore[no-matching-overload]
    clamped_tile_ptrs: tl.InTilePointers[
        [20, 24], [int, int], [16, 16], Literal["clamped_0"]
    ] = cast(Any, None)
    assert_type(tl.load(clamped_tile_ptrs, mask=k_mask, other=0.0), tl.tensor[[16, 16]])
    tl.load(clamped_tile_ptrs, mask=row_mask, other=0.0)  # pyrefly: ignore[no-matching-overload]
    tl.load(clamped_tile_ptrs, mask=wrong_k_tile, other=0.0)  # pyrefly: ignore[no-matching-overload]
    assert_type(row_mask & k_mask, tl.Mask[[20, 24], [16, 16]])
    typed_a: torch.Tensor[[20, 24]] = torch.empty((20, 24))
    typed_b: torch.Tensor[[24, 19]] = torch.empty((24, 19))
    typed_out: torch.Tensor[[20, 19]] = torch.empty((20, 19))
    typed_layout = tiled_output(
        as_host_tensor(typed_out, host_tensor.Tensor[[20, 19], [int, 1]]),
        (16, 16),
        shape_parameters=("M", "N"),
        tile_parameters=("BLOCK_SIZE_M", "BLOCK_SIZE_N"),
        metadata={"GROUP_SIZE_M": 8},
    )
    assert_type(typed_layout, TiledOutputLayout[[20, 19], [16, 16]])
    # The layout launcher keeps the kernel's pointer-direction signature.
    bad_a: tlt.OutPointer[[20, 24], [int, int]] = cast(Any, None)
    good_b: tlt.InPointer[[24, 19], [int, int]] = cast(Any, None)
    good_c: tlt.OutPointer[[20, 19], [int, 1]] = cast(Any, None)
    typed_layout.launch(
        matmul_kernel,
        bad_a,  # pyrefly: ignore[bad-argument-type]
        good_b,
        good_c,
        20,
        19,
        24,
        24,
        1,
        19,
        1,
        19,
        1,
        16,
        16,
        16,
        8,
        "",
    )
    host_a = as_host_tensor(typed_a, host_tensor.Tensor[[20, 24], [int, int]])
    assert_type(host_a, host_tensor.Tensor[[20, 24], [int, int]])
    pointer_a, _, _, _ = checked_matrix(host_a, tlt.InPointer[[20, 24], [int, int]])
    assert_type(pointer_a, tlt.InPointer[[20, 24], [int, int]])
    assert_type(
        checked_matmul(
            as_host_tensor(typed_a, host_tensor.Tensor[[20, 24], [int, int]]),
            as_host_tensor(typed_b, host_tensor.Tensor[[24, 19], [int, int]]),
        ),
        torch.Tensor[[20, 19]],
    )

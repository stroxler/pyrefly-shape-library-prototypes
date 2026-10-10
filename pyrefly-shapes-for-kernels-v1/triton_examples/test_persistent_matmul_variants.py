"""Tutorial 09's four additional matmul kernels with unchanged executable bodies.

Adapted from Triton's 09-persistent-matmul.py (MIT license; copyright
2018–2020 Philippe Tillet and 2020–2022 OpenAI). The fifth kernel and shared
grouped-program helper live in test_persistent_matmul.py.

Only the nonpersistent pointer kernel has a checked host wrapper here. The
host-constructed TMA descriptors have shape-aware kernel signatures but no
validated Python-to-descriptor bridge. The device-constructed descriptor
checks dense pointer shapes and strides; its output block width remains
gradual because EPILOGUE_SUBTILE selects between full- and half-width stores.
None of the kernel types proves that grouped program IDs cover output tiles.
"""

from __future__ import annotations

import os
import unittest
from typing import TYPE_CHECKING, Any, Literal, assert_type, cast

import torch
import triton.language as tl
from shape_extensions import Int, IntVar
from triton.backends.compiler import GPUTarget

from triton_examples.test_persistent_matmul import _compute_pid
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
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
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

    if c_ptr.dtype.element_ty == tl.float8e4nv:
        c = accumulator.to(tl.float8e4nv)
    else:
        c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@semantic_jit
def matmul_kernel_tma(
    a_desc: tl.tensor_descriptor[[M, K], [K, 1], [BM, BlockK], Literal["read"]],
    b_desc: tl.tensor_descriptor[[N, K], [K, 1], [BlockN, BlockK], Literal["read"]],
    c_desc: tl.tensor_descriptor[[M, N], [N, 1], [BM, BlockN], Literal["write"]],
    M: Int[M],
    N: Int[N],
    K: Int[K],
    BLOCK_SIZE_M: ConstExpr[Int[BM]],
    BLOCK_SIZE_N: ConstExpr[Int[BlockN]],
    BLOCK_SIZE_K: ConstExpr[Int[BlockK]],
    GROUP_SIZE_M: ConstExpr[Int[Group]],
    FP8_OUTPUT: ConstExpr[bool],
    WARP_SPECIALIZE: ConstExpr[bool],
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in tl.range(k_tiles, warp_specialize=WARP_SPECIALIZE):
        offs_k = k * BLOCK_SIZE_K
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

    c = accumulator.to(dtype)

    offs_cm = pid_m * BLOCK_SIZE_M
    offs_cn = pid_n * BLOCK_SIZE_N
    c_desc.store([offs_cm, offs_cn], c)


@semantic_jit
def matmul_kernel_tma_persistent(
    a_desc: tl.tensor_descriptor[[M, K], [K, 1], [BM, BlockK], Literal["read"]],
    b_desc: tl.tensor_descriptor[[N, K], [K, 1], [BlockN, BlockK], Literal["read"]],
    c_desc: tl.tensor_descriptor[[M, N], [N, 1], [BM, int], Literal["write"]],
    M: Int[M],
    N: Int[N],
    K: Int[K],
    BLOCK_SIZE_M: ConstExpr[Int[BM]],
    BLOCK_SIZE_N: ConstExpr[Int[BlockN]],
    BLOCK_SIZE_K: ConstExpr[Int[BlockK]],
    GROUP_SIZE_M: ConstExpr[Int[Group]],
    FP8_OUTPUT: ConstExpr[bool],
    EPILOGUE_SUBTILE: ConstExpr[bool],
    NUM_SMS: ConstExpr[Int[SMs]],
    WARP_SPECIALIZE: ConstExpr[bool],
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Enable warp specialization to leverage async warp scheduling in the GPU.
    # FIXME: This only works on Blackwell right now. On older GPUs, this will
    # use software pipelining.
    for tile_id in tl.range(
        start_pid, num_tiles, NUM_SMS, flatten=True, warp_specialize=WARP_SPECIALIZE
    ):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_am_c = pid_m * BLOCK_SIZE_M
        offs_bn_c = pid_n * BLOCK_SIZE_N

        # Epilogue subtiling is a technique to break our computation and stores into multiple pieces
        # By subtiling we can reduce shared memory consumption by the epilogue and instead use that
        # memory to increase our stage count.
        # In this case we partition the accumulator into 2 BLOCK_SIZE_M x BLOCK_SIZE_N // 2 tensors
        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)
        else:
            accumulator = accumulator.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], accumulator)


@semantic_jit
def matmul_kernel_descriptor_persistent(
    a_ptr: tlt.InPointer[[M, K], [K, 1]],
    b_ptr: tlt.InPointer[[N, K], [K, 1]],
    c_ptr: tlt.OutPointer[[M, N], [N, 1]],
    M: Int[M],
    N: Int[N],
    K: Int[K],
    BLOCK_SIZE_M: ConstExpr[Int[BM]],
    BLOCK_SIZE_N: ConstExpr[Int[BlockN]],
    BLOCK_SIZE_K: ConstExpr[Int[BlockK]],
    GROUP_SIZE_M: ConstExpr[Int[Group]],
    EPILOGUE_SUBTILE: ConstExpr[bool],
    NUM_SMS: ConstExpr[Int[SMs]],
    WARP_SPECIALIZE: ConstExpr[bool],
    FLATTEN: ConstExpr[bool],
):
    # Matmul using TMA and device-side descriptor creation
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[
            BLOCK_SIZE_M,
            BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2,
        ],
    )

    # tile_id_c is used in the epilogue to break the dependency between
    # the prologue and the epilogue
    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(
        start_pid, num_tiles, NUM_SMS, flatten=FLATTEN, warp_specialize=WARP_SPECIALIZE
    ):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_cm = pid_m * BLOCK_SIZE_M
        offs_cn = pid_n * BLOCK_SIZE_N

        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_cm, offs_cn], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_cm, offs_cn + BLOCK_SIZE_N // 2], c1)
        else:
            c = accumulator.to(dtype)
            c_desc.store([offs_cm, offs_cn], c)


def tiled_matmul[Rows: IntVar, Inner: IntVar, Cols: IntVar](
    a: host_tensor.Tensor[[Rows, Inner], [int, int]],
    b: host_tensor.Tensor[[Inner, Cols], [int, int]],
    *,
    block_m: int = 16,
    block_n: int = 16,
    block_k: int = 16,
    group_m: int = 8,
) -> torch.Tensor[[Rows, Cols]]:
    """Validate host matrix views and launch one program per output tile."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Tiled matmul contraction dimensions must match")
    if min(*a.shape, b.shape[1]) <= 0:
        raise ValueError("Tiled matmul dimensions must be positive")
    if any(
        type(block) is not int or block < 16 or block & (block - 1)
        for block in (block_m, block_n, block_k)
    ):
        raise ValueError("Tiled matmul blocks must be powers of two of at least 16")
    if type(group_m) is not int or group_m <= 0:
        raise ValueError("Tiled matmul group size must be positive")
    if a.device != b.device or a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("Tiled matmul inputs must be float16 on one device")
    if any(stride <= 0 for stride in (*a.stride(), *b.stride())):
        raise ValueError("Tiled matmul requires positive input strides")

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
    return out


class Tutorial09VariantsTest(unittest.TestCase):
    """Check all four frontend bodies and the nonpersistent host boundary."""

    @unittest.skipIf(os.environ.get("TRITON_INTERPRET") == "1", "requires normal JIT")
    def test_naive_frontend(self) -> None:
        ttir = compile_ttir(
            matmul_kernel,
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
            },
        )
        self.assertIn("tt.func public @matmul_kernel", ttir)

    @unittest.skipIf(os.environ.get("TRITON_INTERPRET") == "1", "requires normal JIT")
    def test_device_descriptor_frontend(self) -> None:
        for subtile in (False, True):
            with self.subTest(subtile=subtile):
                ttir = compile_ttir(
                    matmul_kernel_descriptor_persistent,
                    {
                        "a_ptr": "*fp16",
                        "b_ptr": "*fp16",
                        "c_ptr": "*fp16",
                        "M": "i32",
                        "N": "i32",
                        "K": "i32",
                    },
                    {
                        "BLOCK_SIZE_M": 32,
                        "BLOCK_SIZE_N": 32,
                        "BLOCK_SIZE_K": 32,
                        "GROUP_SIZE_M": 8,
                        "EPILOGUE_SUBTILE": subtile,
                        "NUM_SMS": 2,
                        "WARP_SPECIALIZE": False,
                        "FLATTEN": True,
                    },
                    target=GPUTarget("cuda", 90, 32),
                )
                self.assertIn(
                    "tt.func public @matmul_kernel_descriptor_persistent", ttir
                )

    @unittest.skipIf(os.environ.get("TRITON_INTERPRET") == "1", "requires normal JIT")
    def test_host_descriptor_frontends(self) -> None:
        for name, kernel in (
            ("matmul_kernel_tma", matmul_kernel_tma),
            ("matmul_kernel_tma_persistent", matmul_kernel_tma_persistent),
        ):
            for subtile in (
                (False, True) if kernel is matmul_kernel_tma_persistent else (False,)
            ):
                with self.subTest(kernel=name, subtile=subtile):
                    output_width = 16 if subtile else 32
                    constexprs: dict[str, int | str] = {
                        "BLOCK_SIZE_M": 32,
                        "BLOCK_SIZE_N": 32,
                        "BLOCK_SIZE_K": 32,
                        "GROUP_SIZE_M": 8,
                        "FP8_OUTPUT": False,
                        "WARP_SPECIALIZE": False,
                    }
                    if kernel is matmul_kernel_tma_persistent:
                        constexprs.update({"EPILOGUE_SUBTILE": subtile, "NUM_SMS": 2})
                    ttir = compile_ttir(
                        kernel,
                        {
                            "a_desc": "tensordesc<fp16[32,32]>",
                            "b_desc": "tensordesc<fp16[32,32]>",
                            "c_desc": f"tensordesc<fp16[32,{output_width}]>",
                            "M": "i32",
                            "N": "i32",
                            "K": "i32",
                        },
                        constexprs,
                        target=GPUTarget("cuda", 90, 32),
                    )
                    self.assertIn(f"tt.func public @{name}", ttir)

    def test_reject_invalid_host_contract(self) -> None:
        a = torch.ones((20, 24), dtype=torch.float16)
        b = torch.ones((24, 19), dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "contraction dimensions"):
            tiled_matmul(as_host_tensor(a), as_host_tensor(cast(Any, b[:23])))
        with self.assertRaisesRegex(ValueError, "float16"):
            tiled_matmul(as_host_tensor(a.float()), as_host_tensor(b))

    @unittest.skipUnless(
        os.environ.get("TRITON_INTERPRET") == "1", "requires interpreter"
    )
    def test_irregular_shapes_with_strides(self) -> None:
        a = (torch.arange(37 * 32, dtype=torch.float16) / 100).reshape(37, 32)[:, :23]
        b = (torch.arange(19 * 23, dtype=torch.float16) / 100).reshape(19, 23).T
        out = tiled_matmul(
            as_host_tensor(a, host_tensor.Tensor[[37, 23], [int, int]]),
            as_host_tensor(b, host_tensor.Tensor[[23, 19], [int, int]]),
        )
        torch.testing.assert_close(out, a @ b, atol=0.05, rtol=0.02)


if TYPE_CHECKING:

    def check_descriptor_contract[
        Rows: IntVar,
        Inner: IntVar,
        BlockRows: IntVar,
        BlockInner: IntVar,
        Other: IntVar,
    ](
        a: tlt.InPointer[[Rows, Inner], [Inner, 1]],
        rows: Int[Rows],
        inner: Int[Inner],
        other: Int[Other],
        block_rows: Int[BlockRows],
        block_inner: Int[BlockInner],
    ) -> None:
        desc = tl.make_tensor_descriptor(
            a,
            shape=[rows, inner],
            strides=[inner, 1],
            block_shape=[block_rows, block_inner],
        )
        assert_type(
            desc,
            tl.tensor_descriptor[
                [Rows, Inner], [Inner, 1], [BlockRows, BlockInner], Literal["read"]
            ],
        )
        tl.make_tensor_descriptor(  # pyrefly: ignore[no-matching-overload]
            a,
            shape=[other, inner],
            strides=[inner, 1],
            block_shape=[block_rows, block_inner],
        )

    def check_naive_shape_interface[
        Rows: IntVar,
        Inner: IntVar,
        Cols: IntVar,
        Other: IntVar,
    ](
        a: host_tensor.Tensor[[Rows, Inner], [int, int]],
        b: host_tensor.Tensor[[Inner, Cols], [int, int]],
        wrong: host_tensor.Tensor[[Other, Cols], [int, int]],
    ) -> None:
        assert_type(tiled_matmul(a, b), torch.Tensor[[Rows, Cols]])
        tiled_matmul(a, wrong)  # pyrefly: ignore[bad-argument-type]

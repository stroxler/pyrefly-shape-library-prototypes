"""Executable block-scaled matmul kernel bodies from Triton's tutorial 10.

Kernel bodies retain the NVIDIA tutorial's MIT-licensed statements; parameter
annotations are specialized to v1's semantic scale and packed-data overlays.
The host contract checks allocations and metadata, but does not launch either
hardware-specific kernel or establish the provenance of shuffled scale values.
"""

from __future__ import annotations

import os
import unittest
from typing import TYPE_CHECKING, Literal

import torch
import triton.language as tl
from shape_extensions import Int, IntVar
from triton.backends.compiler import GPUTarget

from triton_examples.testing import compile_ttir
from triton_library.semantic_jit import ConstExpr, semantic_jit
from triton_library.tlt import OutPointer

MDim = IntVar("MDim")
NDim = IntVar("NDim")
KDim = IntVar("KDim")
AElementsPerByte = IntVar("AElementsPerByte")
BElementsPerByte = IntVar("BElementsPerByte")
VecSize = IntVar("VecSize")
RepM = IntVar("RepM")
RepN = IntVar("RepN")
RepK = IntVar("RepK")
AM = IntVar("AM")
AK = IntVar("AK")
BStrideK = IntVar("BStrideK")
BN = IntVar("BN")
CM = IntVar("CM")
CN = IntVar("CN")
ASM = IntVar("ASM")
ASK = IntVar("ASK")
BSN = IntVar("BSN")
BSK = IntVar("BSK")
BM = IntVar("BM")
BNN = IntVar("BNN")
BKK = IntVar("BKK")


@semantic_jit
def block_scaled_matmul_kernel(
    a_desc: tl.BlockDataDescriptor[MDim, KDim, AElementsPerByte, RepM, RepK, VecSize],
    a_scale_desc: tl.BlockScaleDescriptor[MDim, KDim, VecSize, RepM, RepK],
    b_desc: tl.BlockDataDescriptor[NDim, KDim, BElementsPerByte, RepN, RepK, VecSize],
    b_scale_desc: tl.BlockScaleDescriptor[NDim, KDim, VecSize, RepN, RepK],
    c_desc: tl.BlockOutputDescriptor[MDim, NDim, RepM * 128, RepN * 128],
    M: ConstExpr[Int[MDim]],
    N: ConstExpr[Int[NDim]],
    K: ConstExpr[Int[KDim]],
    output_type: ConstExpr[Literal[0, 1, 2]],
    ELEM_PER_BYTE_A: ConstExpr[Int[AElementsPerByte]],
    ELEM_PER_BYTE_B: ConstExpr[Int[BElementsPerByte]],
    VEC_SIZE: ConstExpr[Int[VecSize]],
    BLOCK_M: ConstExpr[Int[RepM * 128]],
    BLOCK_N: ConstExpr[Int[RepN * 128]],
    BLOCK_K: ConstExpr[Int[RepK * 4 * VecSize]],
    rep_m: ConstExpr[Int[RepM]],
    rep_n: ConstExpr[Int[RepN]],
    rep_k: ConstExpr[Int[RepK]],
    NUM_STAGES: ConstExpr[int],
    disallow_acc_multi_buffer: ConstExpr[bool],
):
    if output_type == 0:
        output_dtype = tl.float32
    elif output_type == 1:
        output_dtype = tl.float16
    elif output_type == 2:
        output_dtype = tl.float8e4nv

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    offs_k_a = 0
    offs_k_b = 0
    offs_scale_m = pid_m * rep_m
    offs_scale_n = pid_n * rep_n
    offs_scale_k = 0

    MIXED_PREC: tl.constexpr = ELEM_PER_BYTE_A == 1 and ELEM_PER_BYTE_B == 2

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(
        0,
        tl.cdiv(K, BLOCK_K),
        num_stages=NUM_STAGES,
        disallow_acc_multi_buffer=disallow_acc_multi_buffer,
    ):
        a = a_desc.load([offs_am, offs_k_a])
        b = b_desc.load([offs_bn, offs_k_b])
        scale_a = a_scale_desc.load([0, offs_scale_m, offs_scale_k, 0, 0])
        scale_b = b_scale_desc.load([0, offs_scale_n, offs_scale_k, 0, 0])

        scale_a = (
            scale_a.reshape(rep_m, rep_k, 32, 4, 4)
            .trans(0, 3, 2, 1, 4)
            .reshape(BLOCK_M, BLOCK_K // VEC_SIZE)
        )
        scale_b = (
            scale_b.reshape(rep_n, rep_k, 32, 4, 4)
            .trans(0, 3, 2, 1, 4)
            .reshape(BLOCK_N, BLOCK_K // VEC_SIZE)
        )

        if MIXED_PREC:
            accumulator = tl.dot_scaled(
                a, scale_a, "e4m3", b.T, scale_b, "e2m1", accumulator
            )
        elif ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 2:
            accumulator = tl.dot_scaled(
                a, scale_a, "e2m1", b.T, scale_b, "e2m1", accumulator
            )
        else:
            accumulator = tl.dot_scaled(
                a, scale_a, "e4m3", b.T, scale_b, "e4m3", accumulator
            )

        offs_k_a += BLOCK_K // ELEM_PER_BYTE_A
        offs_k_b += BLOCK_K // ELEM_PER_BYTE_B
        offs_scale_k += rep_k

    c_desc.store([offs_am, offs_bn], accumulator.to(output_dtype))


@semantic_jit
def block_scaled_matmul_kernel_cdna4(
    a_ptr: tl.CDNA4PackedAPointer[MDim, KDim, AM, AK, BM, BKK],
    b_ptr: tl.CDNA4PackedBPointer[KDim, NDim, BStrideK, BN, BNN, BKK],
    c_ptr: OutPointer[[MDim, NDim], [CM, CN]],
    a_scales_ptr: tl.CDNA4ScalePointer[MDim, KDim, ASM, ASK, BM, BKK],
    b_scales_ptr: tl.CDNA4ScalePointer[NDim, KDim, BSN, BSK, BNN, BKK],
    M: Int[MDim],
    N: Int[NDim],
    K: Int[KDim],
    stride_am: Int[AM],
    stride_ak: Int[AK],
    stride_bk: Int[BStrideK],
    stride_bn: Int[BN],
    stride_ck: int,
    stride_cm: Int[CM],
    stride_cn: Int[CN],
    stride_asm: Int[ASM],
    stride_ask: Int[ASK],
    stride_bsn: Int[BSN],
    stride_bsk: Int[BSK],
    BLOCK_M: ConstExpr[Int[BM]],
    BLOCK_N: ConstExpr[Int[BNN]],
    BLOCK_K: ConstExpr[Int[BKK]],
    mfma_nonkdim: ConstExpr[Literal[16, 32]],
):
    """Kernel for computing the matmul C = A x B.
    A and B inputs are in the microscale fp4 (mxfp4) format.
    A_scales and B_scales are in e8m0 format.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    pid = tl.program_id(axis=0)

    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # We assume 32 elements along K share the same scale.
    SCALE_GROUP_SIZE: tl.constexpr = 32
    num_k_iter = tl.cdiv(K, BLOCK_K // 2)
    # Create pointers for first block of A and B input matrices
    # The BLOCK sizes are of the elements and in fp4 we pack 2 per uint8 container.
    offs_k = tl.arange(0, BLOCK_K // 2)
    offs_k_split = offs_k
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Create pointers for the first block of A and B scales
    offs_asn = (pid_n * (BLOCK_N // 32) + tl.arange(0, (BLOCK_N // 32))) % N
    offs_ks = tl.arange(0, BLOCK_K // SCALE_GROUP_SIZE * 32)

    # B scales are N x K even though B operand is K x N.
    b_scale_ptrs = (
        b_scales_ptr + offs_asn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk
    )
    offs_asm = (pid_m * (BLOCK_M // 32) + tl.arange(0, (BLOCK_M // 32))) % M
    a_scale_ptrs = (
        a_scales_ptr + offs_asm[:, None] * stride_asm + offs_ks[None, :] * stride_ask
    )
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, num_k_iter):
        # Here we "undo" the shuffle done in global memory (shuffle_scales_cdna4 function).
        if mfma_nonkdim == 32:
            a_scales = (
                tl.load(a_scale_ptrs)
                .reshape(BLOCK_M // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 2, 32, 4, 1)
                .permute(0, 3, 1, 4, 2, 5)
                .reshape(BLOCK_M, BLOCK_K // SCALE_GROUP_SIZE)
            )
            b_scales = (
                tl.load(b_scale_ptrs)
                .reshape(BLOCK_N // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 2, 32, 4, 1)
                .permute(0, 3, 1, 4, 2, 5)
                .reshape(BLOCK_N, BLOCK_K // SCALE_GROUP_SIZE)
            )
        elif mfma_nonkdim == 16:
            a_scales = (
                tl.load(a_scale_ptrs)
                .reshape(
                    BLOCK_M // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2, 1
                )
                .permute(0, 5, 3, 1, 4, 2, 6)
                .reshape(BLOCK_M, BLOCK_K // SCALE_GROUP_SIZE)
            )
            b_scales = (
                tl.load(b_scale_ptrs)
                .reshape(
                    BLOCK_N // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2, 1
                )
                .permute(0, 5, 3, 1, 4, 2, 6)
                .reshape(BLOCK_N, BLOCK_K // SCALE_GROUP_SIZE)
            )

        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs, cache_modifier=None)

        accumulator += tl.dot_scaled(a, a_scales, "e2m1", b, b_scales, "e2m1")

        # Advance the ptrs to the next K block.
        a_ptrs += (BLOCK_K // 2) * stride_ak
        b_ptrs += (BLOCK_K // 2) * stride_bk

        a_scale_ptrs += BLOCK_K * stride_ask
        b_scale_ptrs += BLOCK_K * stride_bsk

    c = accumulator.to(c_ptr.type.element_ty)

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask, cache_modifier=".wt")


def validate_blackwell_block_scaled(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    output: torch.Tensor,
    *,
    k: int,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 128,
    vector: int = 32,
    a_elements_per_byte: int = 1,
    b_elements_per_byte: int = 1,
    output_type: Literal[0, 1, 2] = 1,
) -> None:
    """Check the host allocations and metadata before descriptor construction.

    The packed scales' values and their preshuffling provenance are assumed;
    this only checks the physical layout consumed by the unchanged kernel.
    """
    if (a_elements_per_byte, b_elements_per_byte, vector) not in (
        (1, 1, 32),
        (2, 2, 16),
        (2, 2, 32),
        (1, 2, 32),
    ):
        raise ValueError("Unsupported block scale format")
    if (
        any(
            type(size) is not int or size <= 0
            for size in (k, block_m, block_n, block_k)
        )
        or block_m % 128
        or block_n % 128
        or block_k % (4 * vector)
        or k % block_k
        or block_m & (block_m - 1)
        or block_n & (block_n - 1)
        or block_k & (block_k - 1)
    ):
        raise ValueError("Block scale dimensions must be positive whole tiles")
    if a.ndim != 2 or b.ndim != 2 or output.ndim != 2:
        raise ValueError("Block scale matrices must have rank two")
    m, a_k = a.shape
    n, b_k = b.shape
    if (
        m <= 0
        or n <= 0
        or m % block_m
        or n % block_n
        or a_k != k // a_elements_per_byte
        or b_k != k // b_elements_per_byte
        or output.shape != (m, n)
    ):
        raise ValueError("Packed matrices and output must match M, N, K whole tiles")
    if a_scale.shape != (1, m // 128, k // vector // 4, 2, 256) or b_scale.shape != (
        1,
        n // 128,
        k // vector // 4,
        2,
        256,
    ):
        raise ValueError("Scale descriptor shapes must match the packed matrices")
    if output_type not in (0, 1, 2):
        raise ValueError("Unsupported output type")
    # The local Torch typing overlay does not export these runtime dtypes.
    fp8_dtype = getattr(torch, "float8_e4m3fn")
    byte_dtype = getattr(torch, "uint8")
    expected_output_dtype = (torch.float32, torch.float16, fp8_dtype)[output_type]
    if (
        a.dtype != (fp8_dtype if a_elements_per_byte == 1 else byte_dtype)
        or b.dtype != (fp8_dtype if b_elements_per_byte == 1 else byte_dtype)
        or a_scale.dtype != (fp8_dtype if vector == 16 else byte_dtype)
        or b_scale.dtype != (fp8_dtype if vector == 16 else byte_dtype)
        or output.dtype != expected_output_dtype
    ):
        raise ValueError("Packed data, scale, or declared output dtype mismatch")
    arrays = (a, a_scale, b, b_scale, output)
    if any(not tensor.is_contiguous() for tensor in arrays):
        raise ValueError("TMA descriptors require contiguous row-major allocations")
    if any(tensor.device != a.device for tensor in arrays):
        raise ValueError("Block scale arrays must share a device")
    if a.device.type != "cuda":
        raise ValueError("Blackwell block scale kernel requires CUDA")
    if torch.cuda.get_device_capability(a.device)[0] < 10:
        raise ValueError("Blackwell block scale kernel requires compute capability 10")


def validate_cdna4_block_scaled(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    output: torch.Tensor,
    *,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 256,
    mfma_nonkdim: Literal[16, 32] = 32,
) -> None:
    """Check packed-byte dimensions and preshuffled CDNA4 scale allocation."""
    if mfma_nonkdim not in (16, 32):
        raise ValueError("CDNA4 mfma_nonkdim must be 16 or 32")
    if (
        any(
            type(block) is not int or block <= 0 or block & (block - 1)
            for block in (block_m, block_n, block_k)
        )
        or block_m % 32
        or block_n % 32
        or block_k % 256
    ):
        raise ValueError("CDNA4 blocks must be positive whole MFMA tiles")
    if any(tensor.ndim != 2 for tensor in (a, b, a_scale, b_scale, output)):
        raise ValueError("CDNA4 inputs and output must have rank two")
    m, packed_k = a.shape
    b_packed_k, n = b.shape
    if (
        m <= 0
        or n <= 0
        or m % block_m
        or n % block_n
        or packed_k <= 0
        or packed_k % (block_k // 2)
        or b_packed_k != packed_k
        or output.shape != (m, n)
    ):
        raise ValueError("CDNA4 packed matrices and output must match M, N, K")
    if a_scale.shape != (m // 32, packed_k * 2) or b_scale.shape != (
        n // 32,
        packed_k * 2,
    ):
        raise ValueError("CDNA4 shuffled scale shapes must match packed K")
    byte_dtype = getattr(torch, "uint8")
    if (
        a.dtype != byte_dtype
        or b.dtype != byte_dtype
        or a_scale.dtype != byte_dtype
        or b_scale.dtype != byte_dtype
        or output.dtype != torch.float32
    ):
        raise ValueError("CDNA4 packed data, scale, or declared output dtype mismatch")
    if (
        not a.is_contiguous()
        or b.stride(0) != 1
        or b.stride(1) != packed_k
        or not a_scale.is_contiguous()
        or not b_scale.is_contiguous()
        or not output.is_contiguous()
    ):
        raise ValueError("CDNA4 requires contiguous packed and shuffled layouts")
    if any(tensor.device != a.device for tensor in (b, a_scale, b_scale, output)):
        raise ValueError("CDNA4 arrays must share a device")
    if a.device.type != "cuda":
        raise ValueError("CDNA4 kernel requires a HIP device")
    if torch.cuda.get_device_properties(a.device).gcnArchName.split(":")[0] != "gfx950":
        raise ValueError("CDNA4 kernel requires gfx950")


class BlockScaledMatmulTest(unittest.TestCase):
    """Compile both original bodies without requiring physical GPUs."""

    def test_blackwell_host_contract(self) -> None:
        fp8 = getattr(torch, "float8_e4m3fn")
        byte = getattr(torch, "uint8")
        a = torch.empty((128, 128), dtype=fp8)
        b = torch.empty((128, 128), dtype=fp8)
        a_scale = torch.empty((1, 1, 1, 2, 256), dtype=byte)
        b_scale = torch.empty((1, 1, 1, 2, 256), dtype=byte)
        output = torch.empty((128, 128), dtype=torch.float16)
        args = (a, a_scale, b, b_scale, output)
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            validate_blackwell_block_scaled(*args, k=128)
        with self.assertRaisesRegex(ValueError, "Scale descriptor shapes"):
            validate_blackwell_block_scaled(
                a, a_scale[:, :, :, :, :255], b, b_scale, output, k=128
            )
        with self.assertRaisesRegex(ValueError, "declared output dtype"):
            validate_blackwell_block_scaled(*args, k=128, output_type=0)
        with self.assertRaisesRegex(ValueError, "Packed matrices"):
            validate_blackwell_block_scaled(*args, k=256)

    def test_cdna4_host_contract(self) -> None:
        byte = getattr(torch, "uint8")
        a = torch.empty((128, 128), dtype=byte)
        b = torch.empty((128, 128), dtype=byte).T
        a_scale = torch.empty((4, 256), dtype=byte)
        b_scale = torch.empty((4, 256), dtype=byte)
        output = torch.empty((128, 128), dtype=torch.float32)
        args = (a, b, a_scale, b_scale, output)
        with self.assertRaisesRegex(ValueError, "requires a HIP device"):
            validate_cdna4_block_scaled(*args)
        with self.assertRaisesRegex(ValueError, "shuffled scale shapes"):
            validate_cdna4_block_scaled(a, b, a_scale[:3], b_scale, output)
        with self.assertRaisesRegex(ValueError, "declared output dtype"):
            validate_cdna4_block_scaled(a, b, a_scale, b_scale, output.half())
        with self.assertRaisesRegex(ValueError, "packed matrices"):
            validate_cdna4_block_scaled(a, b[:, :127], a_scale, b_scale, output)

    @unittest.skipIf(os.environ.get("TRITON_INTERPRET") == "1", "requires normal JIT")
    def test_blackwell_frontend(self) -> None:
        for a_bytes, b_bytes, vector in ((1, 1, 32), (2, 2, 16), (1, 2, 32)):
            with self.subTest(a_bytes=a_bytes, b_bytes=b_bytes):
                block_k = 4 * vector
                ttir = compile_ttir(
                    block_scaled_matmul_kernel,
                    {
                        "a_desc": (
                            "tensordesc<fp8e4nv[128,128]>"
                            if a_bytes == 1
                            else "tensordesc<u8[128,64]>"
                        ),
                        "a_scale_desc": "tensordesc<u8[1,1,1,2,256]>",
                        "b_desc": (
                            "tensordesc<fp8e4nv[128,128]>"
                            if b_bytes == 1
                            else "tensordesc<u8[128,64]>"
                        ),
                        "b_scale_desc": "tensordesc<u8[1,1,1,2,256]>",
                        "c_desc": "tensordesc<fp16[128,128]>",
                    },
                    {
                        "M": 128,
                        "N": 128,
                        "K": block_k,
                        "output_type": 1,
                        "ELEM_PER_BYTE_A": a_bytes,
                        "ELEM_PER_BYTE_B": b_bytes,
                        "VEC_SIZE": vector,
                        "BLOCK_M": 128,
                        "BLOCK_N": 128,
                        "BLOCK_K": block_k,
                        "rep_m": 1,
                        "rep_n": 1,
                        "rep_k": 1,
                        "NUM_STAGES": 1,
                        "disallow_acc_multi_buffer": False,
                    },
                    target=GPUTarget("cuda", 100, 32),
                )
                self.assertIn("tt.func public @block_scaled_matmul_kernel", ttir)

    @unittest.skipIf(os.environ.get("TRITON_INTERPRET") == "1", "requires normal JIT")
    def test_cdna4_frontend(self) -> None:
        for mfma_nonkdim in (16, 32):
            with self.subTest(mfma_nonkdim=mfma_nonkdim):
                ttir = compile_ttir(
                    block_scaled_matmul_kernel_cdna4,
                    {
                        "a_ptr": "*u8",
                        "b_ptr": "*u8",
                        "c_ptr": "*fp32",
                        "a_scales_ptr": "*u8",
                        "b_scales_ptr": "*u8",
                        "M": "i32",
                        "N": "i32",
                        "K": "i32",
                        "stride_am": "i32",
                        "stride_ak": "i32",
                        "stride_bk": "i32",
                        "stride_bn": "i32",
                        "stride_ck": "i32",
                        "stride_cm": "i32",
                        "stride_cn": "i32",
                        "stride_asm": "i32",
                        "stride_ask": "i32",
                        "stride_bsn": "i32",
                        "stride_bsk": "i32",
                    },
                    {
                        "BLOCK_M": 128,
                        "BLOCK_N": 128,
                        "BLOCK_K": 256,
                        "mfma_nonkdim": mfma_nonkdim,
                    },
                    target=GPUTarget("hip", "gfx950", 64),
                )
                self.assertIn("tt.func public @block_scaled_matmul_kernel_cdna4", ttir)


if TYPE_CHECKING:

    def check_output_tile[
        Rows: IntVar,
        Cols: IntVar,
        BM: IntVar,
        BN: IntVar,
        Other: IntVar,
    ](
        output: tl.BlockOutputDescriptor[Rows, Cols, BM, BN],
        good: tl.tensor[[BM, BN]],
        wrong: tl.tensor[[BM, Other]],
    ) -> None:
        output.store([0, 0], good)
        output.store([0, 0], wrong)  # pyrefly: ignore[bad-argument-type]

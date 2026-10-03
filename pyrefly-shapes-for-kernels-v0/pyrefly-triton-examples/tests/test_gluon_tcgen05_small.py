# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only, not an executable Gluon kernel.
# @lint-ignore-every AUTODEPS2

"""Check the full Blackwell MMA kernel's unchanged shared/TMEM data flow."""

from shape_extensions import IntVar
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    allocate_tensor_memory,
    fence_async_shared,
    mbarrier,
    tcgen05_commit,
    tcgen05_mma,
    TensorMemoryLayout,
    TensorMemoryTileF16,
    TensorMemoryTileF32,
    tma,
)


@gluon.jit
def small_mma_kernel[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    LA: IntVar,
    LB: IntVar,
    LC: IntVar,
    LD: IntVar,
](
    a_desc: gl.TmaInputDescriptorF16[M, K, BM, BK, LA],
    b_desc: gl.TmaInputDescriptorF16[K, N, BK, BN, LB],
    c_desc: gl.TmaInputDescriptor2D[M, N, BM, BN, LC],
    d_desc: gl.TmaOutputDescriptor2D[M, N, BM, BN, LD],
    tmem_block: tuple[int, int],  #
    LHS_IN_TMEM: bool,
    USE_COMMIT: bool,
    num_warps: int,
):
    # Load A, B, and C tiles.
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    # A has shape [M, K].
    a_smem = gl.allocate_shared_memory(
        a_desc.dtype, a_desc.block_type.shape, a_desc.layout
    )
    # B has shape [K, N].
    b_smem = gl.allocate_shared_memory(
        b_desc.dtype, b_desc.block_type.shape, b_desc.layout
    )
    # C has shape [M, N].
    c_smem = gl.allocate_shared_memory(
        c_desc.dtype, c_desc.block_type.shape, c_desc.layout
    )

    mbarrier.expect(
        bar,
        a_desc.block_type.nbytes + b_desc.block_type.nbytes + c_desc.block_type.nbytes,
    )
    tma.async_load(a_desc, [0, 0], bar, a_smem)
    tma.async_load(b_desc, [0, 0], bar, b_smem)
    tma.async_load(c_desc, [0, 0], bar, c_smem)
    mbarrier.wait(bar, phase=0)

    # Reusing an mbarrier for TMAs and tcgen05_mma can lead to undefined
    # behaviour. Make sure to use a separate mbarrier or re-initialize it.
    mbarrier.invalidate(bar)
    mbarrier.init(bar, count=1)

    # The accumulator operand must be provided in TMEM. The LHS operand can be
    # provided in either SMEM or TMEM. The RHS operand must be provided in SMEM.
    # SMEM operands must have an NVMMASharedLayout.
    M: gl.constexpr = d_desc.block_type.shape[0]
    N: gl.constexpr = d_desc.block_type.shape[1]
    K: gl.constexpr = a_desc.block_type.shape[1]

    # Copy operands into TMEM.
    # TODO: Use `tcgen05.cp` when it is exposed in Gluon.
    acc_tmem_layout: gl.constexpr = TensorMemoryLayout(
        tmem_block,
        col_stride=32 // d_desc.dtype.primitive_bitwidth,
    )
    acc_tmem = allocate_tensor_memory(d_desc.dtype, [M, N], acc_tmem_layout)
    acc_reg_layout: gl.constexpr = acc_tmem.get_reg_layout()
    acc = c_smem.load(acc_reg_layout)
    acc_tmem.store(acc)

    if LHS_IN_TMEM:
        # When the LHS operand is fp16 or fp8, it is packed in TMEM.
        lhs_tmem_layout: gl.constexpr = TensorMemoryLayout(
            tmem_block,
            col_stride=1,
        )
        lhs_tmem = allocate_tensor_memory(a_desc.dtype, [M, K], lhs_tmem_layout)

        lhs_reg_layout: gl.constexpr = lhs_tmem.get_reg_layout()
        lhs = a_smem.load(lhs_reg_layout)
        lhs_tmem.store(lhs)
        a = lhs_tmem
    else:
        a = a_smem

    # tcgen05_mma is an asynchronous operation. Until the operation is complete,
    # we cannot read or write to the accumulator memory and we cannot write to
    # the operand memory. tcgen05_mma accesses shared memory through the async
    # proxy:
    #
    # ```python
    # b_smem.store(b)
    # fence_async_shared()
    # tcgen05_mma(a, b_smem, acc_tmem)
    # ```
    #
    # A fence is required between the shared store and tcgen05_mma to order
    # their shared memory accesses. Completion of the tcgen05_mma operation
    # implies its reads from shared memory are complete, thus it would be safe
    # to write to the shared memory inputs after waiting without a fence.
    #
    # Completion of tcgen05_mma operations is tracked with mbarriers. Invoking
    # tcgen05_commit on an mbarrier causes the mbarrier to be arrived on when
    # all previously issued tcgen05_mma operations have been completed. See
    # 04-tma.py for more details on how mbarriers work.
    #
    # To commit on an mbarrier, we can either explicitly invoke tcgen05_commit
    # or pass the mbarrier directly to tcgen05_mma. We can also conditionally
    # commit an mbarrier if necessary.
    #
    # tcgen05_mma is comprised of multiple async MMA instructions. The shape of
    # each instruction is determined by the TMEM layout. Selecting larger
    # instruction shapes generally results in better performance. Note that
    # tcgen05_mma only supports blockM=64 when there is 1 block.
    if USE_COMMIT:
        tcgen05_mma(a, b_smem, acc_tmem)
        tcgen05_commit(bar)
    else:
        tcgen05_mma(a, b_smem, acc_tmem, mbarriers=[bar], mbarrier_preds=[True])

    # Wait for the completion of the MMA.
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    # Another important flag to consider is `use_acc`. When `use_acc=False`, the
    # current value of the accumulator in TMEM is ignored. This is an efficient
    # way to zero the accumulator.

    d_smem = gl.allocate_shared_memory(
        d_desc.dtype, d_desc.block_type.shape, d_desc.layout
    )
    acc = acc_tmem.load()
    d_smem.store(acc)
    fence_async_shared()
    tma.async_store(d_desc, [0, 0], d_smem)
    tma.store_wait(pendings=0)


def test_mma_tile_contract[
    M: IntVar,
    K: IntVar,
    N: IntVar,
    Other: IntVar,
    LA: IntVar,
    LB: IntVar,
](
    a: gl.WgmmaSharedF16[M, K, LA],
    a_tmem: TensorMemoryTileF16[M, K],
    b: gl.WgmmaSharedF16[K, N, LB],
    wrong_b_k: gl.WgmmaSharedF16[Other, N, LB],
    acc: TensorMemoryTileF32[M, N],
    wrong_acc_n: TensorMemoryTileF32[M, Other],
    wrong_acc_dtype: TensorMemoryTileF16[M, N],
) -> None:
    tcgen05_mma(a, b, acc)
    tcgen05_mma(a_tmem, b, acc)
    tcgen05_mma(a, wrong_b_k, acc)  # E: is not assignable
    tcgen05_mma(a, b, wrong_acc_n)  # E: is not assignable
    tcgen05_mma(a, b, wrong_acc_dtype)  # E: is not assignable

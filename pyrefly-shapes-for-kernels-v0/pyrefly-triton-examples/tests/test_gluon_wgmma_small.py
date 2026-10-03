# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only, not an executable Gluon kernel.
# @lint-ignore-every AUTODEPS2

"""Check the full small WGMMA matmul's descriptor and MMA tile interface."""

from shape_extensions import IntVar
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_wait,
)


@gluon.jit
def small_mma_kernel[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    LA: IntVar,
    LB: IntVar,
    LC: IntVar,
](
    a_desc: gl.TmaInputDescriptorF16[M, K, M, K, LA],
    b_desc: gl.TmaInputDescriptorF16[K, N, K, N, LB],
    c_desc: gl.TmaInputDescriptor2D[M, N, M, N, LC],
    d_desc: gl.TmaOutputDescriptor2D[M, N, M, N, LC],
    LHS_IN_REG: bool,
    INSTR_SHAPE_N: int,
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
    mbarrier.invalidate(bar)

    # Let's parameterize the kernel over LHS_IN_REG and INSTR_SHAPE_N to see how
    # it can affect performance.
    m: gl.constexpr = 16
    k: gl.constexpr = 256 // a_desc.dtype.primitive_bitwidth
    n: gl.constexpr = INSTR_SHAPE_N
    warps_per_cta: gl.constexpr = [num_warps, 1]

    # The MMA shape is passed through the layout of `c`, which must always have
    # an NVMMADistributedLayout.
    c_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=warps_per_cta,
        instr_shape=[m, n, k],
    )

    # When A is passed through registers, it must have the following layout:
    a_reg_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=c_layout,
        k_width=32 // a_desc.dtype.primitive_bitwidth,
    )

    # When an operand is passed through shared memory, it must have an
    # NVMMASharedLayout. TMA requires using an NVMMASharedLayout.
    gl.static_assert(
        isinstance(a_smem.type.layout, gl.NVMMASharedLayout)  # E: is not assignable
    )
    gl.static_assert(
        isinstance(b_smem.type.layout, gl.NVMMASharedLayout)  # E: is not assignable
    )

    if LHS_IN_REG:
        a = a_smem.load(a_reg_layout)
    else:
        a = a_smem

    c = c_smem.load(c_layout)
    # Issue the async WGMMA. Note that `is_async=False` is the default value,
    # and all this does is immediately wait for 0 outstanding operations. In
    # this tutorial, we will always use `is_async=True`.
    #
    # Another important flag to consider is `use_acc`. When `use_acc=False`, the
    # `c` input is ignored and the accumulator is zero-initialized. This can
    # be an efficient way to zero the accumulator.
    d = warpgroup_mma(a, b_smem, c, is_async=True, use_acc=True)

    # To ensure correct ordering between `warpgroup_mma`, the wait, and uses of
    # the result, you must thread the `warpgroup_mma` result through the wait
    # via the `deps` argument and use the return value of the
    # `warpgroup_mma_wait`.
    #
    # Wait for 0 outstanding operations, so we know the WGMMA is complete.
    d = warpgroup_mma_wait(num_outstanding=0, deps=(d,))

    d_smem = gl.allocate_shared_memory(
        d_desc.dtype, d_desc.block_type.shape, d_desc.layout
    )
    d_smem.store(d)
    fence_async_shared()
    tma.async_store(d_desc, [0, 0], d_smem)
    tma.store_wait(pendings=0)


def test_kernel_interface[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    Other: IntVar,
    LA: IntVar,
    LB: IntVar,
    LC: IntVar,
    OtherLayout: IntVar,
](
    a: gl.TmaInputDescriptorF16[M, K, M, K, LA],
    b: gl.TmaInputDescriptorF16[K, N, K, N, LB],
    c: gl.TmaInputDescriptor2D[M, N, M, N, LC],
    d: gl.TmaOutputDescriptor2D[M, N, M, N, LC],
    wrong_b_k: gl.TmaInputDescriptorF16[Other, N, Other, N, LB],
    wrong_b_n: gl.TmaInputDescriptorF16[K, Other, K, Other, LB],
    wrong_c_m: gl.TmaInputDescriptor2D[Other, N, Other, N, LC],
    wrong_d_n: gl.TmaOutputDescriptor2D[M, Other, M, Other, LC],
    wrong_d_layout: gl.TmaOutputDescriptor2D[M, N, M, N, OtherLayout],
    wrong_a_dtype: gl.TmaInputDescriptor2D[M, K, M, K, LA],
    wrong_d_role: gl.TmaInputDescriptor2D[M, N, M, N, LC],
) -> None:
    small_mma_kernel(a, b, c, d, False, 16, 4)
    small_mma_kernel(a, b, c, d, True, 16, 4)
    small_mma_kernel(a, wrong_b_k, c, d, False, 16, 4)  # E: is not assignable
    small_mma_kernel(
        a,
        wrong_b_n,
        c,  # E: is not assignable
        d,  # E: is not assignable
        False,
        16,
        4,
    )
    small_mma_kernel(a, b, wrong_c_m, d, False, 16, 4)  # E: is not assignable
    small_mma_kernel(a, b, c, wrong_d_n, False, 16, 4)  # E: is not assignable
    small_mma_kernel(a, b, c, wrong_d_layout, False, 16, 4)  # E: is not assignable
    small_mma_kernel(wrong_a_dtype, b, c, d, False, 16, 4)  # E: is not assignable
    small_mma_kernel(a, b, c, wrong_d_role, False, 16, 4)  # E: is not assignable

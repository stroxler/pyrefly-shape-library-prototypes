# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; semantic parameter annotations are not executable Gluon annotations.
# @lint-ignore-every AUTODEPS2

"""Check the original Blackwell accumulate entrypoint's descriptor boundary."""

from shape_extensions import Int, IntVar
from tests.test_gluon_persistent_matmul import PersistentTileScheduler
from tests.test_gluon_tcgen05_matmul_accmulate_mma_partition import (
    matmul_accmulate_mma_partition,
)
from tests.test_gluon_tcgen05_matmul_accumulate_epilogue_partition import (
    matmul_accumulate_epilogue_partition,
)
from tests.test_gluon_tcgen05_matmul_accumulate_load_partition import (
    matmul_accumulate_load_partition,
    PartitionArgs,
)
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    allocate_tensor_memory,
    mbarrier,
    TensorMemoryLayout,
)


@gluon.jit(do_not_specialize=["d_stride_m", "d_stride_n"])
def matmul_accumulate_kernel[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    LA: IntVar,
    LB: IntVar,
    LC: IntVar,
    RowStride: IntVar,
    ColStride: IntVar,
    Depth: IntVar,
](
    a_desc: gl.WgmmaDescriptorF16[M, K, BM, BK, LA],
    b_desc: gl.WgmmaDescriptorF16[K, N, BK, BN, LB],
    c_desc: gl.WgmmaDescriptorF32[M, N, BM, BN, LC],
    d_ptr: gl.OutMatrixPointer2D[M, N, RowStride, ColStride],
    d_stride_m: Int[RowStride],
    d_stride_n: Int[ColStride],
    SchedulerImpl: type[PersistentTileScheduler],
    num_buffers: Int[Depth],
):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype

    a_bufs = gl.allocate_shared_memory(
        dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout
    )
    b_bufs = gl.allocate_shared_memory(
        dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout
    )
    load_empty_bars = gl.allocate_shared_memory(
        gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout()
    )
    load_ready_bars = gl.allocate_shared_memory(
        gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout()
    )
    for i in gl.static_range(num_buffers):
        mbarrier.init(load_empty_bars.index(i), count=1)
        mbarrier.init(load_ready_bars.index(i), count=1)

    c_buf = gl.allocate_shared_memory(
        c_desc.dtype, c_desc.block_type.shape, c_desc.layout
    )
    c_empty_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    c_ready_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(c_empty_bar, count=1)
    mbarrier.init(c_ready_bar, count=1)

    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_bufs = allocate_tensor_memory(gl.float32, [2, BLOCK_M, BLOCK_N], tmem_layout)
    acc_empty_bars = gl.allocate_shared_memory(
        gl.int64, [2, 1], mbarrier.MBarrierLayout()
    )
    acc_ready_bars = gl.allocate_shared_memory(
        gl.int64, [2, 1], mbarrier.MBarrierLayout()
    )
    for i in gl.static_range(2):
        mbarrier.init(acc_empty_bars.index(i), count=1)
        mbarrier.init(acc_ready_bars.index(i), count=1)

    p = PartitionArgs(
        a_desc,  # E: is not assignable
        b_desc,  # E: is not assignable
        c_desc,  # E: is not assignable
        d_ptr,  # E: is not assignable
        d_stride_m,  # E: is not assignable
        d_stride_n,  # E: is not assignable
        a_bufs,  # E: is not assignable
        b_bufs,  # E: is not assignable
        load_empty_bars,  # E: is not assignable
        load_ready_bars,  # E: is not assignable
        c_buf,  # E: is not assignable
        c_empty_bar,
        c_ready_bar,
        acc_bufs,
        acc_empty_bars,  # E: is not assignable
        acc_ready_bars,  # E: is not assignable
        SchedulerImpl,
    )
    gl.warp_specialize(
        [  # E: is not assignable
            (matmul_accumulate_epilogue_partition, (p,)),
            (matmul_accmulate_mma_partition, (p,)),
            (matmul_accumulate_load_partition, (p,)),
        ],
        [1, 1],
        [24, 24],
    )


def test_launcher_declared_matrix_interface[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    Other: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    LA: IntVar,
    LB: IntVar,
    LC: IntVar,
    RowStride: IntVar,
    ColStride: IntVar,
    Depth: IntVar,
](
    a: gl.WgmmaDescriptorF16[M, K, BM, BK, LA],
    b: gl.WgmmaDescriptorF16[K, N, BK, BN, LB],
    wrong_b_host_k: gl.WgmmaDescriptorF16[Other, N, BK, BN, LB],
    c: gl.WgmmaDescriptorF32[M, N, BM, BN, LC],
    wrong_c_host_n: gl.WgmmaDescriptorF32[M, Other, BM, BN, LC],
    d: gl.OutMatrixPointer2D[M, N, RowStride, ColStride],
    wrong_d_host_n: gl.OutMatrixPointer2D[M, Other, RowStride, ColStride],
    row_stride: Int[RowStride],
    col_stride: Int[ColStride],
    wrong_row_stride: Int[Other],
    depth: Int[Depth],
) -> None:
    matmul_accumulate_kernel(
        a, b, c, d, row_stride, col_stride, PersistentTileScheduler, depth
    )
    matmul_accumulate_kernel(
        a,
        wrong_b_host_k,  # E: is not assignable
        c,
        d,
        row_stride,
        col_stride,
        PersistentTileScheduler,
        depth,
    )
    matmul_accumulate_kernel(
        a,
        b,
        wrong_c_host_n,  # E: is not assignable
        d,
        row_stride,
        col_stride,
        PersistentTileScheduler,
        depth,
    )
    matmul_accumulate_kernel(
        a,
        b,
        c,
        wrong_d_host_n,  # E: is not assignable
        row_stride,
        col_stride,
        PersistentTileScheduler,
        depth,
    )
    matmul_accumulate_kernel(
        a,
        b,
        c,
        d,
        wrong_row_stride,  # E: is not assignable
        col_stride,
        PersistentTileScheduler,
        depth,
    )

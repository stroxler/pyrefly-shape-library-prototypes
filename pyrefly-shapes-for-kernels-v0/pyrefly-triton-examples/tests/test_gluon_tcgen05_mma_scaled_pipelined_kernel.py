# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; semantic parameter annotations are not executable Gluon annotations.
# @lint-ignore-every AUTODEPS2
# @lint-ignore-every SPELL

"""Check the source-identical pipelined scaled-MMA kernel boundary."""

from shape_extensions import Int, IntVar
from tests.test_gluon_tcgen05_matmul_accumulate_epilogue_partition import (
    EpilogueSchedulerFactory,
)
from tests.test_gluon_tcgen05_matmul_accumulate_load_partition import Counter
from tests.test_gluon_tcgen05_mma_scaled_issue_loads import issue_loads
from tests.test_gluon_tcgen05_mma_scaled_issue_mma import issue_mma
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    allocate_tensor_memory,
    fence_async_shared,
    mbarrier,
    TensorMemoryLayout,
    tma,
)
from triton.language import tensor


class t8:
    Counter = Counter


@gluon.jit
def mma_scaled_pipelined_kernel[
    M: IntVar,
    N: IntVar,
    K: IntVar,
    BM: IntVar,
    BN: IntVar,
    BK: IntVar,
    LA: IntVar,
    LB: IntVar,
    LC: IntVar,
    LSA: IntVar,
    LSB: IntVar,
    HostAM: IntVar,
    HostAK: IntVar,
    HostBN: IntVar,
    HostBK: IntVar,
    Depth: IntVar,
](
    a_desc: gl.TmaDescriptorF8[M, K, BM, BK, LA],
    b_desc: gl.TmaDescriptorF8[N, K, BN, BK, LB],
    c_desc: gl.TmaOutputDescriptorF16[M, N, BM, BN, LC],
    a_scale_desc: gl.PackedScaleDescriptorU8[HostAM, HostAK, BM, BK, LSA],
    b_scale_desc: gl.PackedScaleDescriptorU8[HostBN, HostBK, BN, BK, LSB],
    num_buffers: Int[Depth],
    SchedulerImpl: type[EpilogueSchedulerFactory],
):
    A_ELEM_PER_BYTE: gl.constexpr = 2 if a_desc.dtype == gl.uint8 else 1
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1] * A_ELEM_PER_BYTE
    K = a_desc.shape[1] * A_ELEM_PER_BYTE

    a_bufs = gl.allocate_shared_memory(
        a_desc.dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout
    )
    b_bufs = gl.allocate_shared_memory(
        b_desc.dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout
    )
    # The scale loads are much smaller than the operand loads (by a factor of VEC_SIZE).
    # We could use fewer buffers for the scales than the operands to save shared memory
    # as the scale load latency is lower, but this is left as an exercise for the reader.
    a_scale_bufs = gl.allocate_shared_memory(
        a_scale_desc.dtype,
        [num_buffers] + a_scale_desc.block_type.shape,
        a_scale_desc.layout,
    )
    b_scale_bufs = gl.allocate_shared_memory(
        b_scale_desc.dtype,
        [num_buffers] + b_scale_desc.block_type.shape,
        b_scale_desc.layout,
    )
    acc_smem = gl.allocate_shared_memory(
        c_desc.dtype, c_desc.block_type.shape, c_desc.layout
    )

    load_bars = gl.allocate_shared_memory(
        gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout()
    )
    for i in gl.static_range(num_buffers):
        mbarrier.init(load_bars.index(i), count=1)
    load_producer = t8.Counter.create(0, num_buffers)
    load_consumer = t8.Counter.create(0, num_buffers)

    # If BLOCK_N=256, double-buffering the accumulator will use all 512 columns
    # of tensor memory, which leaves no room for the scales' tensor memory.
    num_acc_buffers: gl.constexpr = 2 if BLOCK_N < 256 else 1
    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_bufs = allocate_tensor_memory(
        gl.float32, [num_acc_buffers, BLOCK_M, BLOCK_N], tmem_layout
    )
    acc_idx = 0

    # We double buffer the mma barriers so we can have 2 in flight simultaneously
    num_mma_bars: gl.constexpr = 2
    mma_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_mma_bars):
        mbarrier.init(mma_bars.index(i), count=1)
    mma_producer = t8.Counter.create(0, num_mma_bars)
    mma_consumer = t8.Counter.create(0, num_mma_bars)

    scheduler = SchedulerImpl.initialize(
        c_desc.shape[0], c_desc.shape[1], BLOCK_M, BLOCK_N
    )
    num_tiles = scheduler.get_num_tiles()

    # Peeled inner loop prologue. Use predicates to mask peeled iterations that
    # would be out-of-bounds if K is too small, but assume K > 0, i.e. we execute
    # at least one inner loop iteration.
    idx = 0
    pid_m, pid_n = scheduler.get_tile(idx)
    for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
        load_producer = issue_loads(
            load_producer,
            pid_m,
            pid_n,
            ki,
            a_desc,
            b_desc,
            a_scale_desc,
            b_scale_desc,
            a_bufs,
            b_bufs,
            a_scale_bufs,
            b_scale_bufs,
            load_bars,
            pred=ki < K,
        )
    k = BLOCK_K * (num_buffers - 2)
    load_producer = issue_loads(
        load_producer,
        pid_m,
        pid_n,
        k,
        a_desc,
        b_desc,
        a_scale_desc,
        b_scale_desc,
        a_bufs,
        b_bufs,
        a_scale_bufs,
        b_scale_bufs,
        load_bars,
        pred=k < K,
    )

    load_consumer, mma_producer = issue_mma(
        load_consumer,
        load_bars,
        a_bufs,
        b_bufs,
        a_scale_bufs,
        b_scale_bufs,
        mma_producer,
        mma_bars,
        acc_bufs.index(acc_idx),
        use_acc=False,
        pred=True,
    )
    for _ in range(num_tiles):  # E: is not assignable
        for k in range(BLOCK_K * (num_buffers - 1), K, BLOCK_K):
            load_producer = issue_loads(
                load_producer,
                pid_m,
                pid_n,
                k,
                a_desc,
                b_desc,
                a_scale_desc,
                b_scale_desc,
                a_bufs,
                b_bufs,
                a_scale_bufs,
                b_scale_bufs,
                load_bars,
                pred=True,
            )
            load_consumer, mma_producer = issue_mma(
                load_consumer,
                load_bars,
                a_bufs,
                b_bufs,
                a_scale_bufs,
                b_scale_bufs,
                mma_producer,
                mma_bars,
                acc_bufs.index(acc_idx),
                use_acc=True,
                pred=True,
            )
            # Wait for the N-1th MMA to complete so we can keep issuing loads.
            mbarrier.wait(
                mma_bars.index(mma_consumer.index),  # E: is not assignable
                mma_consumer.phase,  # E: is not assignable
            )
            mma_consumer = mma_consumer.next()

        # Peel the next prologue and fuse it with the pipeline drain loop.
        epilogue_pid_m, epilogue_pid_n = pid_m, pid_n
        idx += 1
        pid_m, pid_n = scheduler.get_tile(idx)
        has_next_tile = idx < num_tiles
        for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
            load_producer = issue_loads(
                load_producer,
                pid_m,
                pid_n,
                ki,
                a_desc,
                b_desc,
                a_scale_desc,
                b_scale_desc,
                a_bufs,
                b_bufs,
                a_scale_bufs,
                b_scale_bufs,
                load_bars,
                has_next_tile and ki < K,  # E: is not assignable
            )

            pred = K > ki + BLOCK_K
            load_consumer, mma_producer = issue_mma(
                load_consumer,
                load_bars,
                a_bufs,
                b_bufs,
                a_scale_bufs,
                b_scale_bufs,
                mma_producer,
                mma_bars,
                acc_bufs.index(acc_idx),
                use_acc=True,
                pred=pred,
            )
            mbarrier.wait(
                mma_bars.index(mma_consumer.index),  # E: is not assignable
                mma_consumer.phase,  # E: is not assignable
                pred,
            )
            mma_consumer = mma_consumer.next(pred)

        k = BLOCK_K * (num_buffers - 2)
        load_producer = issue_loads(
            load_producer,
            pid_m,
            pid_n,
            k,
            a_desc,
            b_desc,
            a_scale_desc,
            b_scale_desc,
            a_bufs,
            b_bufs,
            a_scale_bufs,
            b_scale_bufs,
            load_bars,
            pred=has_next_tile and k < K,  # E: is not assignable
        )
        cur_acc_buf = acc_bufs.index(acc_idx)

        # Compared to Hopper, we can overlap Blackwell MMAs a little bit more because
        # the accumulator is stored in tensor memory. When the accumulator is not
        # double-buffered, we will start the MMA of the next tile after loading the
        # final accumulator of the current tile, but before initiating the TMA store.
        # When the accumulator is double-buffered, we can the start first MMA of the next tile
        # before the last MMA of the current tile completes.
        if num_acc_buffers == 2:
            acc_idx ^= 1
            load_consumer, mma_producer = issue_mma(
                load_consumer,
                load_bars,
                a_bufs,
                b_bufs,
                a_scale_bufs,
                b_scale_bufs,
                mma_producer,
                mma_bars,
                acc_bufs.index(acc_idx),
                use_acc=False,
                pred=has_next_tile,  # E: is not assignable
            )
        mbarrier.wait(
            mma_bars.index(mma_consumer.index),  # E: is not assignable
            mma_consumer.phase,  # E: is not assignable
        )
        mma_consumer = mma_consumer.next()
        acc = cur_acc_buf.load()
        if num_acc_buffers == 1:
            # Wait for all threads to finish loading from accumulator
            gl.barrier()
            load_consumer, mma_producer = issue_mma(
                load_consumer,
                load_bars,
                a_bufs,
                b_bufs,
                a_scale_bufs,
                b_scale_bufs,
                mma_producer,
                mma_bars,
                acc_bufs.index(acc_idx),
                use_acc=False,
                pred=has_next_tile,  # E: is not assignable
            )

        acc = acc.to(c_desc.dtype)
        # Pipeline the store by waiting for the previous store to complete.
        tma.store_wait(0)
        acc_smem.store(acc)
        fence_async_shared()
        tma.async_store(
            c_desc, [epilogue_pid_m * BLOCK_M, epilogue_pid_n * BLOCK_N], acc_smem
        )

    # Wait for the last store.
    tma.store_wait(0)
    for i in gl.static_range(num_buffers):
        mbarrier.invalidate(load_bars.index(i))
    for i in gl.static_range(num_mma_bars):
        mbarrier.invalidate(mma_bars.index(i))


def test_reflected_ring_block_shape_controls[
    Depth: IntVar,
    BM: IntVar,
    BK: IntVar,
    Other: IntVar,
    LA: IntVar,
    LSA: IntVar,
](
    depth: Int[Depth],
    operand_shape: gl.TmaBlockShapeF8[BM, BK],
    packed_shape: gl.PackedScaleBlockShape[BM, BK],
    operand_layout: gl.WgmmaLayoutF8[BM, BK, LA],
    wrong_operand_layout: gl.WgmmaLayoutF8[BM, Other, LA],
    packed_layout: gl.PackedScaleLayout[BM, BK, LSA],
    wrong_packed_layout: gl.PackedScaleLayout[BM, Other, LSA],
) -> None:
    gl.allocate_shared_memory(gl.float8e4nv, [depth] + operand_shape, operand_layout)
    gl.allocate_shared_memory(gl.uint8, [depth] + packed_shape, packed_layout)
    gl.allocate_shared_memory(  # E: No matching overload
        gl.float8e4nv,
        [depth] + operand_shape,
        wrong_operand_layout,
    )
    gl.allocate_shared_memory(  # E: No matching overload
        gl.uint8, [depth] + packed_shape, wrong_packed_layout
    )
    gl.allocate_shared_memory(  # E: No matching overload
        gl.float8e4nv, [depth, BM, Other], operand_layout
    )
    # Reflected list addition does not prove that the prefix contains exactly one element.
    gl.allocate_shared_memory(
        gl.float8e4nv, [depth, depth] + operand_shape, operand_layout
    )
    gl.allocate_shared_memory(gl.uint8, [depth, depth] + packed_shape, packed_layout)


def test_pipelined_declared_host_interface[
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
    LSA: IntVar,
    LSB: IntVar,
    HostAM: IntVar,
    HostAK: IntVar,
    HostBN: IntVar,
    HostBK: IntVar,
    Depth: IntVar,
](
    a: gl.TmaDescriptorF8[M, K, BM, BK, LA],
    b: gl.TmaDescriptorF8[N, K, BN, BK, LB],
    wrong_b_host_k: gl.TmaDescriptorF8[N, Other, BN, BK, LB],
    c: gl.TmaOutputDescriptorF16[M, N, BM, BN, LC],
    wrong_c_host_n: gl.TmaOutputDescriptorF16[M, Other, BM, BN, LC],
    a_scale: gl.PackedScaleDescriptorU8[HostAM, HostAK, BM, BK, LSA],
    wrong_a_scale_host_k: gl.PackedScaleDescriptorU8[HostAM, Other, BM, BK, LSA],
    b_scale: gl.PackedScaleDescriptorU8[HostBN, HostBK, BN, BK, LSB],
    wrong_b_scale_tile_k: gl.PackedScaleDescriptorU8[HostBN, HostBK, BN, Other, LSB],
    depth: Int[Depth],
    scheduler: type[EpilogueSchedulerFactory],
) -> None:
    mma_scaled_pipelined_kernel(a, b, c, a_scale, b_scale, depth, scheduler)
    mma_scaled_pipelined_kernel(
        a,
        wrong_b_host_k,  # E: is not assignable
        c,
        a_scale,
        b_scale,
        depth,
        scheduler,
    )
    mma_scaled_pipelined_kernel(
        a,
        b,
        wrong_c_host_n,  # E: is not assignable
        a_scale,
        b_scale,
        depth,
        scheduler,
    )
    mma_scaled_pipelined_kernel(
        a, b, c, wrong_a_scale_host_k, b_scale, depth, scheduler
    )
    mma_scaled_pipelined_kernel(
        a,
        b,
        c,
        a_scale,
        wrong_b_scale_tile_k,  # E: is not assignable
        depth,
        scheduler,
    )


def test_output_tile_intrinsic_controls[
    M: IntVar,
    N: IntVar,
    BM: IntVar,
    BN: IntVar,
    Other: IntVar,
    LC: IntVar,
](
    output: gl.TmaOutputDescriptorF16[M, N, BM, BN, LC],
    shared: gl.WgmmaSharedF16[BM, BN, LC],
    wrong_shared: gl.WgmmaSharedF16[BM, Other, LC],
    acc: tensor[[BM, BN]],
    wrong_acc: tensor[[BM, Other]],
) -> None:
    shared.store(acc)
    shared.store(wrong_acc)  # E: is not assignable
    tma.async_store(output, [0, 0], shared)
    tma.async_store(output, [0, 0], wrong_shared)  # E: No matching overload

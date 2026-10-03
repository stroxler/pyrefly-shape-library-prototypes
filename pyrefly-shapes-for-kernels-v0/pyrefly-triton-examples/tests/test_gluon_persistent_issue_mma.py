# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only; Gluon aggregate annotations are non-executable semantic types.
# @lint-ignore-every AUTODEPS2

"""Check the source-order persistent MMA helper and both aggregate producers."""

from typing import Union

from shape_extensions import Int, IntVar
from tests import test_gluon_wgmma_blocked as t5
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    allocate_tensor_memory,
    mbarrier,
    tcgen05_commit,
    tcgen05_mma,
    tensor_memory_descriptor,
    TensorMemoryLayout,
    TensorMemoryTileF32,
)
from triton.experimental.gluon.language.nvidia.hopper import (
    warpgroup_mma,
    warpgroup_mma_accumulator,
    warpgroup_mma_wait,
)
from triton.language import tensor


@gluon.aggregate
class WGMMA:
    acc: Union[warpgroup_mma_accumulator, gl.tensor]
    use_acc: gl.tensor

    @gluon.jit
    def initialize[BM: IntVar, BN: IntVar](  # E: self type
        dtype: gl.Float16DType,  # noqa: B902 - Gluon aggregate JIT factory.
        BLOCK_M: Int[BM],
        BLOCK_N: Int[BN],
        num_warps: int,
    ):
        mma_layout: gl.constexpr = t5.pick_wgmma_layout(
            dtype, BLOCK_M, BLOCK_N, num_warps
        )
        acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)
        return WGMMA(acc, gl.to_tensor(False))

    @gluon.jit
    def issue_async_mma[M: IntVar, K: IntVar, N: IntVar, LA: IntVar, LB: IntVar](
        self,
        a: gl.WgmmaSharedF16[M, K, LA],
        b: gl.WgmmaSharedF16[K, N, LB],
    ):
        acc = warpgroup_mma(
            a,
            b,
            self.acc,  # E: is not assignable
            is_async=True,
            use_acc=self.use_acc,  # E: is not assignable
        )
        # Note that aggregates don't support in-place mutation, so we need to
        # return a new instance and re-assign it at the callsite.
        return WGMMA(acc, gl.to_tensor(True))

    @gluon.jit
    def wait_num_outstanding(self, num_outstanding: int):
        acc = warpgroup_mma_wait(
            num_outstanding,
            (self.acc,),  # E: is not assignable
        )
        return WGMMA(acc, self.use_acc)

    # Take the result and reset the accumulator.
    @gluon.jit
    def take_result(self, splitn: bool = False):
        return self.acc, WGMMA(self.acc, gl.to_tensor(False))


@gluon.aggregate
class MMAv5:
    use_acc: gl.tensor
    acc_tmem: tensor_memory_descriptor
    bar: gl.shared_memory_descriptor
    counter: gl.tensor

    @gluon.jit
    def initialize[BM: IntVar, BN: IntVar](  # E: self type
        dtype: gl.Float16DType,  # noqa: B902 - Gluon aggregate JIT factory.
        BLOCK_M: Int[BM],
        BLOCK_N: Int[BN],
        num_warps: int,
    ):
        layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
        acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], layout)
        bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
        mbarrier.init(bar, count=1)
        return MMAv5(gl.to_tensor(False), acc_tmem, bar, gl.to_tensor(0))

    @gluon.jit
    def issue_async_mma[M: IntVar, K: IntVar, N: IntVar, LA: IntVar, LB: IntVar](
        self,
        a: gl.WgmmaSharedF16[M, K, LA],
        b: gl.WgmmaSharedF16[K, N, LB],
    ):
        tcgen05_mma(
            a,
            b,
            self.acc_tmem,  # E: is not assignable
            use_acc=self.use_acc,  # E: is not assignable
        )
        tcgen05_commit(self.bar)  # E: is not assignable
        return MMAv5(gl.to_tensor(True), self.acc_tmem, self.bar, self.counter + 1)

    @gluon.jit
    def wait_num_outstanding(self, num_outstanding: int):
        mbarrier.wait(
            self.bar,  # E: is not assignable
            (self.counter - 1 - num_outstanding) & 1,  # E: is not supported
        )
        return self

    @gluon.jit
    def take_result(self, splitn: bool = False):
        next = MMAv5(gl.to_tensor(False), self.acc_tmem, self.bar, self.counter)
        if splitn:
            layout: gl.constexpr = self.acc_tmem.get_reg_layout(
                instr_variant="32x32b_splitn"
            )
            return self.acc_tmem.load(layout), next
        else:
            return self.acc_tmem.load(), next


@gluon.jit
def issue_mma[
    M: IntVar,
    K: IntVar,
    N: IntVar,
    LA: IntVar,
    LB: IntVar,
    Depth: IntVar,
](
    consumer: int,
    mma: WGMMA | MMAv5,
    bars: gl.TmaBarrierRing2D[Depth],
    a_bufs: gl.WgmmaSharedRingF16[M, K, LA],
    b_bufs: gl.WgmmaSharedRingF16[K, N, LB],
    num_buffers: Int[Depth],
):
    index = consumer % num_buffers
    phase = consumer // num_buffers & 1
    consumer += 1
    mbarrier.wait(bars.index(index), phase)
    mma = mma.wait_num_outstanding(0)
    mma = mma.issue_async_mma(a_bufs.index(index), b_bufs.index(index))
    return consumer, mma


def test_contraction_at_method_boundary[
    M: IntVar,
    K: IntVar,
    N: IntVar,
    Other: IntVar,
    LA: IntVar,
    LB: IntVar,
    Depth: IntVar,
](
    hopper: WGMMA,
    blackwell: MMAv5,
    bars: gl.TmaBarrierRing2D[Depth],
    a_bufs: gl.WgmmaSharedRingF16[M, K, LA],
    b_bufs: gl.WgmmaSharedRingF16[K, N, LB],
    wrong_b: gl.WgmmaSharedRingF16[Other, N, LB],
    num_buffers: Int[Depth],
) -> None:
    issue_mma(0, hopper, bars, a_bufs, b_bufs, num_buffers)
    issue_mma(0, blackwell, bars, a_bufs, b_bufs, num_buffers)
    issue_mma(0, hopper, bars, a_bufs, wrong_b, num_buffers)  # E: is not assignable
    issue_mma(0, blackwell, bars, a_bufs, wrong_b, num_buffers)  # E: is not assignable


def test_unshaped_aggregate_fields_accept_wrong_accumulators[
    M: IntVar,
    K: IntVar,
    N: IntVar,
    Other: IntVar,
    LA: IntVar,
    LB: IntVar,
    Depth: IntVar,
](
    wrong_hopper_acc: tensor[[M, Other]],
    wrong_blackwell_acc: TensorMemoryTileF32[M, Other],
    bar: gl.BarrierBuffer1D,
    bars: gl.TmaBarrierRing2D[Depth],
    a_bufs: gl.WgmmaSharedRingF16[M, K, LA],
    b_bufs: gl.WgmmaSharedRingF16[K, N, LB],
    num_buffers: Int[Depth],
) -> None:
    hopper = WGMMA(wrong_hopper_acc, gl.to_tensor(False))
    blackwell = MMAv5(gl.to_tensor(False), wrong_blackwell_acc, bar, gl.to_tensor(0))
    issue_mma(0, hopper, bars, a_bufs, b_bufs, num_buffers)
    issue_mma(0, blackwell, bars, a_bufs, b_bufs, num_buffers)

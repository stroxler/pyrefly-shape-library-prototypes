# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Pure typing probe: the real Triton JIT and grid launch are not modeled here.
# @lint-ignore-every AUTODEPS2

"""Test whether Python typing preserves a generic kernel behind grid launch."""

from collections.abc import Callable
from typing import Protocol

import triton.language as tl
from shape_extensions import Int, IntVar
from triton.experimental.gluon import language as gl


class GridLaunched[**P, R]:
    def __getitem__(self, grid: tuple[int]) -> Callable[P, R]: ...


def with_grid[**P, R](fn: Callable[P, R]) -> GridLaunched[P, R]: ...


class SignaturePreservingGrid[F: Callable[..., object]]:
    def __getitem__(self, grid: tuple[int]) -> F: ...


def with_preserved_signature[F: Callable[..., object]](
    fn: F,
) -> SignaturePreservingGrid[F]: ...


def identity[F: Callable[..., object]](fn: F) -> F: ...


@identity
def identity_kernel[N: IntVar](pointer: tl.InPointer[[N]], length: Int[N]) -> None: ...


@with_grid
def generic_kernel[N: IntVar](pointer: tl.InPointer[[N]], length: Int[N]) -> None: ...


def direct_generic[N: IntVar](pointer: tl.InPointer[[N]], length: Int[N]) -> None: ...


@with_preserved_signature
def retained_kernel[N: IntVar](pointer: tl.InPointer[[N]], length: Int[N]) -> None: ...


@with_grid
def fixed_kernel(pointer: tl.InPointer[[4]], length: Int[4]) -> None: ...


def test_generic_grid_call[N: IntVar, Other: IntVar](
    pointer: tl.InPointer[[N]],
    wrong_pointer: tl.InPointer[[Other]],
    length: Int[N],
) -> None:
    direct_generic(pointer, length)
    direct_generic(wrong_pointer, length)  # E: is not assignable
    identity_kernel(pointer, length)
    # An identity decorator keeps the bound dimension until grid indexing.
    identity_kernel(wrong_pointer, length)  # E: is not assignable
    generic_kernel[(1,)](pointer, length)
    # Accepted gap: the callback loses its generic relationship when wrapped.
    generic_kernel[(1,)](wrong_pointer, length)
    retained_kernel[(1,)](pointer, length)
    # Keeping the function type F also does not retain its generic constraint.
    retained_kernel[(1,)](wrong_pointer, length)


def test_fixed_grid_call(
    pointer: tl.InPointer[[4]], wrong_pointer: tl.InPointer[[5]], length: Int[4]
) -> None:
    fixed_kernel[(1,)](pointer, length)
    fixed_kernel[(1,)](wrong_pointer, length)  # E: is not assignable


class TmaKernelBody(Protocol):
    def __call__[N: IntVar, Block: IntVar, Layout: IntVar](
        self,
        inp: gl.TmaInputDescriptor1D[N, Block, Layout],
        out: gl.TmaOutputDescriptor1D[N, Block, Layout],
        block: Int[Block],
    ) -> None: ...


class TmaLaunchedKernel(TmaKernelBody):
    def __getitem__(self, grid: tuple[int]) -> TmaLaunchedKernel: ...


def typed_tma_jit(fn: TmaKernelBody) -> TmaLaunchedKernel: ...


# This protocol rejects even a correctly shaped generic kernel definition.
@typed_tma_jit  # E: is not assignable
def protocol_kernel[N: IntVar, Block: IntVar, Layout: IntVar](
    inp: gl.TmaInputDescriptor1D[N, Block, Layout],
    out: gl.TmaOutputDescriptor1D[N, Block, Layout],
    block: Int[Block],
) -> None: ...


# An incorrect kernel is indistinguishable at the failed decorator check.
@typed_tma_jit  # E: is not assignable
def wrong_protocol_kernel[N: IntVar, Block: IntVar, Layout: IntVar](
    inp: gl.TmaInputDescriptor1D[N, Block, Layout],
    out: gl.TmaOutputDescriptor1D[N, Block, Layout],
    block: Int[4],
) -> None: ...

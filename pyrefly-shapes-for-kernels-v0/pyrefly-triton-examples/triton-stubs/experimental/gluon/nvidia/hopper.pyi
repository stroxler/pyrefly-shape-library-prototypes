# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Typed host-side tensor descriptor construction for TMA tiles."""

from typing import overload

import torch
from shape_extensions import Int, IntListLiteral, IntVar
from triton.experimental.gluon import language as gl

class TensorDescriptor:
    @staticmethod
    @overload
    def from_tensor[
        Rows: IntVar,
        Cols: IntVar,
        BlockRows: IntVar,
        BlockCols: IntVar,
        Layout: IntVar,
    ](
        tensor: torch.Tensor[[Rows, Cols]],
        block_shape: IntListLiteral[[BlockRows, BlockCols]],
        layout: gl.WgmmaLayoutF16ForBlock[BlockRows, BlockCols, Layout],
    ) -> gl.WgmmaDescriptorF16[Rows, Cols, BlockRows, BlockCols, Layout]: ...
    @staticmethod
    @overload
    def from_tensor[
        Rows: IntVar,
        Cols: IntVar,
        BlockRows: IntVar,
        BlockCols: IntVar,
        Layout: IntVar,
    ](
        tensor: torch.Tensor[[Rows, Cols]],
        block_shape: IntListLiteral[[BlockRows, BlockCols]],
        layout: gl.WgmmaLayoutF32[Layout],
    ) -> gl.WgmmaDescriptorF32[Rows, Cols, BlockRows, BlockCols, Layout]: ...
    @staticmethod
    @overload
    def from_tensor[Length: IntVar, Block: IntVar, Layout: IntVar](
        tensor: torch.Tensor[[Length]],
        block_shape: list[Int[Block]],
        layout: gl.TmaLayout1D[Layout],
    ) -> gl.TmaDescriptor1D[Length, Block, Layout]: ...

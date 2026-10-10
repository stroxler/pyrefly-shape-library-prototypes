# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal, overload

class CompilerParams:
    @overload
    def __init__(self, *, collective_id: Literal[0]) -> None: ...
    @overload
    def __init__(
        self, *, dimension_semantics: tuple[Literal["parallel", "arbitrary"]]
    ) -> None: ...
    @overload
    def __init__(
        self,
        *,
        dimension_semantics: tuple[
            Literal["parallel"], Literal["parallel"], Literal["arbitrary"]
        ],
    ) -> None: ...

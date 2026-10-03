# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal

from shape_extensions import Int, IntVar

class ScalarSubcoreMesh[Core: IntVar]:
    def __init__(self, *, axis_name: Literal["core"], num_cores: Int[Core]) -> None: ...

class VectorSubcoreMesh[Cores: IntVar, Subcores: IntVar]:
    def __init__(
        self,
        *,
        core_axis_name: Literal["core"],
        subcore_axis_name: Literal["subcore"],
        num_cores: Int[Cores],
    ) -> None: ...

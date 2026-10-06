"""Generate a checked 1D Torch boundary from a semantic Triton signature."""

import inspect
import re
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, cast

import torch
import triton


@dataclass(frozen=True)
class Launch1D:
    """Launch settings and output dtype not determined by semantic annotations."""

    block_size: int
    output_dtype: torch.dtype


def make_torch_wrapper(kernel: object, launch: Launch1D) -> Callable[..., torch.Tensor]:
    """Bind one allocation dimension to inputs, scalars, output, and launch grid."""
    if type(launch.block_size) is not int or launch.block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if launch.block_size & (launch.block_size - 1):
        raise ValueError("block_size must be a power of two for tl.arange")
    if not isinstance(launch.output_dtype, torch.dtype):
        raise ValueError("output_dtype must be a torch.dtype")

    runtime_kernel = cast(Any, kernel)
    annotations = getattr(runtime_kernel.fn, "__semantic_annotations__", None)
    if not isinstance(annotations, dict):
        raise ValueError("Kernel must be decorated with @semantic_jit")

    inputs: list[str] = []
    output: str | None = None
    dimension_arg: str | None = None
    block_arg: str | None = None
    dimension: str | None = None
    scalar_symbol: str | None = None
    for name in inspect.signature(runtime_kernel.fn).parameters:
        annotation = annotations.get(name)
        if not isinstance(annotation, str):
            raise ValueError(f"Missing semantic annotation for {name}")
        role = re.fullmatch(
            r"tl\.(InPointer|OutPointer)\[\[([A-Za-z_]\w*)\]\]", annotation
        )
        scalar = re.fullmatch(r"Int\[([A-Za-z_]\w*)\]", annotation)
        constexpr = re.fullmatch(r"ConstExpr\[Int\[([A-Za-z_]\w*)\]\]", annotation)
        if role:
            symbol = role.group(2)
            if dimension is not None and symbol != dimension:
                raise ValueError(f"Pointer {name} does not share dimension {dimension}")
            dimension = symbol
            if role.group(1) == "InPointer":
                inputs.append(name)
            elif output is None:
                output = name
            else:
                raise ValueError("This prototype supports exactly one output")
        elif scalar and dimension_arg is None:
            dimension_arg = name
            scalar_symbol = scalar.group(1)
        elif constexpr and block_arg is None:
            block_arg = name
        else:
            raise ValueError(
                f"Unsupported semantic annotation for {name}: {annotation}"
            )

    if not inputs or output is None or dimension_arg is None or block_arg is None:
        raise ValueError(
            "Expected input and output pointers, length, and block constexpr"
        )
    if scalar_symbol != dimension:
        raise ValueError("Length parameter must name the allocation dimension")
    host_names = [name.removesuffix("_ptr") for name in inputs]
    if len(set(host_names)) != len(host_names):
        raise ValueError("Input pointer names must map to distinct host argument names")
    host_signature = inspect.Signature(
        [
            inspect.Parameter(
                name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=torch.Tensor
            )
            for name in host_names
        ],
        return_annotation=torch.Tensor,
    )

    def wrapper(*args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        """Validate host inputs before deriving kernel arguments and launching."""
        bound = host_signature.bind(*args, **kwargs)
        arrays = dict(zip(inputs, bound.arguments.values(), strict=True))
        length: int | None = None
        device: torch.device | None = None
        for name, array in arrays.items():
            if not isinstance(array, torch.Tensor) or array.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional Torch tensor")
            if not array.is_contiguous():
                raise ValueError(f"{name} must be contiguous (element stride 1)")
            if length is not None and len(array) != length:
                raise ValueError(f"{name} must have length {length}")
            if device is not None and array.device != device:
                raise ValueError(f"{name} must be on device {device}")
            length = len(array)
            device = array.device

        assert length is not None and device is not None
        result = torch.empty((length,), dtype=launch.output_dtype, device=device)
        if length:
            kernel_args = {
                **arrays,
                output: result,
                dimension_arg: length,
                block_arg: launch.block_size,
            }
            grid = (triton.cdiv(length, launch.block_size),)
            context = (
                torch.cuda.device(device) if device.type == "cuda" else nullcontext()
            )
            with context:
                runtime_kernel[grid](**kernel_args)
        return result

    setattr(wrapper, "__signature__", host_signature)
    return wrapper

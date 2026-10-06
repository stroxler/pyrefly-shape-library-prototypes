"""Generate a checked 1D JAX boundary from a semantic Pallas signature."""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


@dataclass(frozen=True)
class Launch1D:
    """Tile size and execution mode supplied outside the kernel signature."""

    block_size: int
    interpret: bool = True


def make_jax_wrapper(
    kernel: Callable[..., Any], launch: Launch1D
) -> Callable[..., Any]:
    """Derive aligned 1D input specs, an output spec, and a host callable."""
    if type(launch.block_size) is not int or launch.block_size <= 0:
        raise ValueError("block_size must be a positive integer")

    inputs: list[str] = []
    output: str | None = None
    block_symbol: str | None = None
    for name, parameter in inspect.signature(kernel).parameters.items():
        annotation = parameter.annotation
        if not isinstance(annotation, str):
            raise ValueError(f"Missing postponed semantic annotation for {name}")
        role = re.fullmatch(r"pl\.(InRef|OutRef)\[\[([A-Za-z_]\w*)\]\]", annotation)
        if role is None:
            raise ValueError(
                f"Unsupported semantic annotation for {name}: {annotation}"
            )
        if block_symbol is not None and role.group(2) != block_symbol:
            raise ValueError(f"Reference {name} does not share block {block_symbol}")
        block_symbol = role.group(2)
        if role.group(1) == "InRef":
            if output is not None:
                raise ValueError("Input references must precede the output reference")
            inputs.append(name)
        elif output is None:
            output = name
        else:
            raise ValueError("This prototype supports exactly one output reference")

    if not inputs or output is None:
        raise ValueError("Expected input references and one output reference")

    host_names = [name.removesuffix("_ref") for name in inputs]
    if len(set(host_names)) != len(host_names):
        raise ValueError("Input references must map to distinct host names")
    host_signature = inspect.Signature(
        [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for name in host_names
        ]
    )
    spec = pl.BlockSpec((launch.block_size,), lambda i: (i,))

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Check the aligned allocation contract, then launch the kernel."""
        bound = host_signature.bind(*args, **kwargs)
        arrays = tuple(bound.arguments.values())
        length: int | None = None
        device: Any = None
        for name, array in zip(inputs, arrays, strict=True):
            if not isinstance(array, jax.Array) or cast(Any, array).ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional JAX array")
            # The stub describes symbolic shapes; runtime JAX attributes are checked here.
            checked = cast(Any, array)
            if checked.dtype != jnp.float32:
                raise ValueError(f"{name} must have dtype float32")
            if length is not None and len(checked) != length:
                raise ValueError(f"{name} must have length {length}")
            if device is not None and checked.device != device:
                raise ValueError(f"{name} must be on device {device}")
            length, device = len(checked), checked.device

        assert length is not None
        if length == 0:
            return cast(Any, jnp).empty((0,), dtype=jnp.float32)
        # Pallas's stub overloads have fixed arities; this runtime factory
        # verifies the arguments above before invoking their shared implementation.
        add = cast(Any, pl.pallas_call)(
            kernel,
            out_shape=jax.ShapeDtypeStruct((length,), jnp.float32),
            grid=(pl.cdiv(length, launch.block_size),),
            in_specs=(spec,) * len(inputs),
            out_specs=spec,
            interpret=launch.interpret,
        )
        return add(*arrays)

    setattr(wrapper, "__signature__", host_signature)
    return wrapper

"""Preserve semantic annotations for Pyrefly while presenting Triton valid source."""

import ast
import inspect
from collections.abc import Callable
from typing import Annotated, Any, cast, get_args, get_origin

import torch
import triton
import triton.language as tl
from shape_extensions import Int, IntVar
from triton.runtime.interpreter import InterpretedFunction
from triton.runtime.jit import JITFunction

from triton_library import tlt

type ConstExpr[T] = Annotated[T, "triton.constexpr"]


class SemanticKernel[F: Callable[..., object]]:
    """Static launch signature of a JIT function; Triton supplies the runtime object."""

    def __getitem__(self, grid: tuple[int, ...]) -> F:
        raise NotImplementedError("Triton's JIT function implements this method")

    def warmup(self, *args: object, **kwargs: object) -> Any:
        raise NotImplementedError("Triton's JIT function implements this method")


def strip_semantic_annotations[F: Callable[..., object]](fn: F) -> F:
    """Leave the source alone and replace only Python's runtime annotations."""
    fn.__annotations__ = {
        name: tl.constexpr
        for name, annotation in fn.__annotations__.items()
        if (
            (isinstance(annotation, str) and annotation.startswith("ConstExpr["))
            or get_origin(annotation) is ConstExpr
            or (
                get_origin(annotation) is Annotated
                and "triton.constexpr" in get_args(annotation)[1:]
            )
        )
    }
    return fn


def _triton_source(source: str) -> str:
    """Erase parameter annotations without changing executable kernel statements."""
    tree = ast.parse(source)
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    lines = source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def position(line: int, column: int) -> int:
        # AST column offsets count UTF-8 bytes, while string slices count codepoints.
        return starts[line - 1] + len(lines[line - 1].encode()[:column].decode())

    edits = []
    arguments = (
        function.args.posonlyargs + function.args.args + function.args.kwonlyargs
    )
    for arg in arguments:
        annotation = arg.annotation
        if annotation is None:
            continue
        is_constexpr = (
            isinstance(annotation, ast.Subscript)
            and isinstance(annotation.value, ast.Name)
            and annotation.value.id == "ConstExpr"
        )
        assert annotation.end_lineno is not None
        assert annotation.end_col_offset is not None
        start = position(arg.lineno, arg.col_offset) + len(arg.arg)
        end = position(annotation.end_lineno, annotation.end_col_offset)
        replacement = ": tl.constexpr" if is_constexpr else ""
        replacement += "\n" * source[start:end].count("\n")
        edits.append((start, end, replacement))
    for start, end, replacement in reversed(edits):
        source = source[:start] + replacement + source[end:]
    return source


def _validate_pointer_arguments(
    fn: Callable[..., object], *args: object, **kwargs: object
) -> None:
    """Check declared dimensions and strides against actual launch arguments."""
    signature = inspect.signature(fn)
    bound = signature.bind(
        *args,
        **{
            name: value
            for name, value in kwargs.items()
            if name in signature.parameters
        },
    )
    symbols: dict[IntVar, int] = {}

    def check(symbol: object, value: int, name: str) -> None:
        if isinstance(symbol, int):
            if value != symbol:
                raise ValueError(f"{name} must be {symbol}, got {value}")
        elif type(symbol) is IntVar:
            previous = symbols.setdefault(symbol, value)
            if value != previous:
                raise ValueError(f"{name} must match {symbol}: {previous}, got {value}")
        elif symbol is int:
            return
        else:
            raise ValueError(f"Unsupported shape constraint for {name}: {symbol!r}")

    annotations = getattr(fn, "__semantic_annotations__")
    for name, value in bound.arguments.items():
        annotation = annotations[name]
        kind = get_origin(annotation)
        if kind not in (tlt.InPointer, tlt.OutPointer):
            continue
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} must be a Torch tensor")
        shape, strides = get_args(annotation)
        if value.ndim != len(shape) or len(strides) != len(shape):
            raise ValueError(f"{name} has the wrong rank")
        for index, (dimension, actual) in enumerate(
            zip(shape, value.shape, strict=True)
        ):
            check(dimension, actual, f"{name}.shape[{index}]")
        for index, (stride, actual) in enumerate(
            zip(strides, value.stride(), strict=True)
        ):
            check(stride, actual, f"{name}.stride({index})")
        if (
            kind is tlt.OutPointer
            and value.ndim == 2
            and value.stride(1) == 1
            and value.shape[0] > 1
            and value.stride(0) < value.shape[1]
        ):
            raise ValueError(f"{name} must not have overlapping elements")

    for name, value in bound.arguments.items():
        annotation = annotations[name]
        if get_origin(annotation) is Int:
            if type(value) is not int:
                raise ValueError(f"{name} must be an integer")
            check(get_args(annotation)[0], value, name)


def semantic_jit[F: Callable[..., object]](fn: F) -> SemanticKernel[F]:
    """Translate semantic annotations for both the compiler and interpreter."""
    setattr(fn, "__semantic_annotations__", dict(fn.__annotations__))
    strip_semantic_annotations(fn)
    kernel = triton.jit(fn)
    if isinstance(kernel, JITFunction):
        kernel._unsafe_update_src(_triton_source(kernel.src))
    elif isinstance(kernel, InterpretedFunction):
        prepare = kernel.rewriter._prepare_source
        kernel.rewriter._prepare_source = lambda lines: _triton_source(prepare(lines))
    else:
        raise TypeError(f"Unsupported Triton JIT implementation: {type(kernel)}")
    if any(
        get_origin(annotation) in (tlt.InPointer, tlt.OutPointer)
        for annotation in getattr(fn, "__semantic_annotations__").values()
    ):
        kernel.add_pre_run_hook(
            lambda *args, **kwargs: _validate_pointer_arguments(fn, *args, **kwargs)
        )
    return cast(SemanticKernel[F], kernel)

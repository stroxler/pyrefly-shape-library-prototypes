"""Preserve semantic annotations for Pyrefly while presenting Triton valid source."""

import ast
from collections.abc import Callable
from typing import Annotated

import triton
import triton.language as tl
from triton.runtime.interpreter import InterpretedFunction
from triton.runtime.jit import JITFunction

type ConstExpr[T] = Annotated[T, "triton.constexpr"]


def strip_semantic_annotations[F: Callable[..., object]](fn: F) -> F:
    """Leave the source alone and replace only Python's runtime annotations."""
    fn.__annotations__ = {
        name: tl.constexpr
        for name, annotation in fn.__annotations__.items()
        if isinstance(annotation, str) and annotation.startswith("ConstExpr[")
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


def semantic_jit[F: Callable[..., object]](fn: F) -> F:
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
    return kernel

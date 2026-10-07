"""Shared frontend-only test setup for annotated Triton examples."""

from typing import Any, cast

from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, make_backend


def compile_ttir(
    kernel: object,
    signature: dict[str, str],
    constexprs: dict[str, int | str],
    *,
    target: GPUTarget = GPUTarget("cuda", 80, 32),
) -> str:
    """Run Triton's real frontend without requiring a GPU or backend lowering."""
    backend = cast(Any, make_backend(target))
    options = backend.parse_options({})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(cast(Any, kernel), signature, constexprs=constexprs)
    result = source.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    return str(result)

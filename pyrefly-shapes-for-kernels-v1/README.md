# Kernel shape experiments (v1)

The [Triton library](triton_library/README.md) and
[Triton examples](triton_examples/README.md) are separate from the
[Pallas library](pallas_library/README.md) and
[Pallas examples](pallas_examples/README.md). The v1 overlays were derived
from v0 but are independently owned; Gluon stubs remain only in v0. The current
executable experiment uses a single `@semantic_jit` decorator around Triton's
JIT, starting with the vector-add body from
`python/tutorials/01-vector-add.py`. The kernel body is unchanged; the local
Triton stubs model allocation and mask semantics. Module-level `IntVar` declarations bind
symbolic `N` and `Block` with legacy-style generics; the kernel uses the same
`tl.InPointer[[N]]` and `Int[N]` annotations as v0. Triton's source extractor
currently only recognizes `def name(`, so PEP 695 function type parameters
are not suitable for this first runtime probe. This experiment makes no changes
to Pyrefly core or v0. The independent Pallas fixture also runs its original
vector-add kernel through a generated JAX boundary, using semantic annotations
directly without source rewriting.

Each vector-add fixture also shows the handwritten host call beside the
generated boundary. Triton's `add(x, y)` follows its tutorial wrapper with a
call-local device check for CPU testing. JAX's Pallas design document has no
named wrapper: `design_doc_add(x, y)` packages its inline `pallas_call`, with
the doc's fixed 8-element output, two-element blocks, and four-program grid.
The [Triton example](triton_examples/README.md) and
[Pallas example](pallas_examples/README.md) record the small adaptations needed
to run these calls against our annotated kernels and current libraries.

`ConstExpr[T]` is an `Annotated[T, "triton.constexpr"]` type alias: Pyrefly sees
the underlying `Int[Block]`, while the runtime decorator sees the postponed
annotation string and changes the function's runtime annotations to
`{"BLOCK_SIZE": tl.constexpr}`. All other runtime annotations are removed.
The decorator rewrites only the parameter-annotation spans in Triton's copy of
the source; the file's annotations and the executable kernel body are unchanged.
The rewritten JIT source retains `BLOCK_SIZE: tl.constexpr`, with the original
number of lines for useful source locations. This relies on Triton's private
`_unsafe_update_src` API, so version upgrades will require revalidation.
For interpreter mode, the same rewrite is installed on that function's own
source preparer; no global monkey-patch is applied.

Run these probes from this directory:

```sh
../.venv/bin/pyrefly check -c pyrefly.toml
/home/stroxler/.kernel-shapes-venv/bin/python -m unittest -v triton_examples.test_vector_add
TRITON_INTERPRET=1 /home/stroxler/.kernel-shapes-venv/bin/python -m unittest -v triton_examples.test_vector_add
/home/stroxler/.kernel-shapes-venv/bin/python -m unittest -v pallas_examples.test_vector_add
```

Pyrefly reports zero errors for the symbolic body. A normal JIT captures the
intended parameter annotations and computes its dependency hash over the
translated source. Frontend compilation with a virtual CUDA SM80 target produces
TTIR without an attached GPU; `triton_examples.testing.compile_ttir` returns
it as text for inspection. The
interpreter runs with an ordinary `16` and verifies masked vector addition for
30-element inputs, whose second tile extends past the allocation. A real GPU
launch has not been tested here, so runtime and device compatibility remain
unverified.

The bare JIT kernel is not itself a safe Python/Triton boundary. The next
section experiments with generating validation and a launch for this kernel's
one-dimensional allocation contract.

## A generated vector-add boundary

`make_torch_wrapper(add_kernel, Launch1D(block_size=1024, output_dtype=torch.float32))`
returns a dynamic `add(x, y)` callable. Its signature is derived from the
kernel parameter names and the original semantic annotations saved by
`@semantic_jit`, not written as a vector-add-specific wrapper. For this first
probe the generator deliberately accepts only one-dimensional `InPointer`,
`OutPointer`, `Int`, and `ConstExpr[Int[...]]` annotations; unknown roles fail at
construction time. `Launch1D` is explicit metadata asserting a one-dimensional
`ceildiv(length, block_size)` grid. It does not prove the program-ID arithmetic
in an arbitrary kernel body implements that mapping.

| Source of contract | Generated behavior | Upstream `add` counterpart |
| --- | --- | --- |
| Two `InPointer[[N]]` parameters | Accept `x` and `y`; check both are 1D, length `N`, contiguous, same dtype and device. | Pass `x, y` as pointers. Upstream checks device but assumes shape/layout compatibility. |
| `OutPointer[[N]]`, explicit `output_dtype` | Allocate a separate output of the shared checked length and device with the selected dtype. | `output = torch.empty_like(x)`; for upstream float32 `x`, this is float32. |
| `n_elements: Int[N]` | Supply the checked input length; no host `n_elements` argument. | `n_elements = output.numel()`. |
| `BLOCK_SIZE: ConstExpr[Int[Block]]` and `Launch1D(block_size=1024)` | Validate power-of-two tile size and inject `BLOCK_SIZE`. | `BLOCK_SIZE=1024`. |
| `Launch1D` grid rule | Launch `(triton.cdiv(N, BLOCK_SIZE),)`; return the allocated output. | `grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)` and `add_kernel[grid](...)`. |

The generated wrapper rejects incompatible shapes, element strides, and
devices before launching, and launches on the validated CUDA device when using
CUDA. Output dtype is chosen explicitly at wrapper
construction, not inferred from an input or passed to the kernel as a separate
constexpr. The wrapper does not constrain input dtypes: Triton specializes on
their actual pointer types, and its compiler decides whether operations and
casts in the kernel body are supported for each combination. CPU interpreter
tests cover a float32 input combined with an int32 input and both float32 and
float64 output allocations. Since every input has the same checked 1D shape
and device, this example needs no separate choice of an input to copy for
output allocation. The upstream example uses float32 inputs.
It skips the launch for empty inputs and returns an empty output. The CPU tests
covers empty arrays, full blocks, partial final blocks, invalid host inputs,
and both block sizes 16 (for testing) and 1024 (matching upstream).

This is a *runtime-validated boundary*, not a statically checked mapping from
Torch shape types to kernel shape types. The generated callable advertises
ordinary `torch.Tensor` inputs; static correspondence between its host
signature and the kernel's semantic annotations would be separate work. In
particular, the 1D tiling rule is declarative, and its agreement with this
kernel's `pid * BLOCK_SIZE + arange` body is currently manually audited.

## A generated Pallas boundary

`make_jax_wrapper(add_kernel, Launch1D(block_size=4))` returns `add(x, y)` for
the Pallas fixture. The kernel annotations stay in place: Pallas accepts them
without a Triton-style JIT or AST adaptation. From the three aligned
`pl.InRef[[Block]]`/`pl.OutRef[[Block]]` parameters, the factory derives host
input names, matching `BlockSpec((4,), lambda i: (i,))` specifications,
`out_shape=(length,)`, and `grid=(pl.cdiv(length, 4),)`. It checks that host
arrays are one-dimensional, equally long, float32, and on the same device;
`interpret=True` executes the original body on CPU. The host caller supplies
neither length nor output storage. A zero-length input returns an empty output
without launching a zero-sized grid.

As with Triton, the grid/index-map rule is launch metadata whose agreement
with the kernel must be audited. The wrapper does not type-check JAX's host
signature statically: its runtime factory intentionally crosses the
fixed-arity `pallas_call` stub overloads dynamically. Its scope is one output
and a single shared one-dimensional block size, not arbitrary Pallas calls.

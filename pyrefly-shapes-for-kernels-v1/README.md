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
`tl.InPointer[[N], [1]]` and `Int[N]` annotations. Triton's source extractor
currently only recognizes `def name(`, so PEP 695 function type parameters
are not suitable for this first runtime probe. This experiment makes no changes
to Pyrefly core or v0. The independent Pallas fixture runs its original
vector-add kernel through an explicit checked JAX boundary, using semantic
annotations directly without source rewriting.

Each vector-add fixture also shows the upstream host call beside an explicit
checked boundary. Triton's `add(x, y)` follows its tutorial wrapper with a
call-local device check for CPU testing. JAX's Pallas design document has no
named wrapper: `design_doc_add(x, y)` packages its inline `pallas_call`, with
the doc's fixed 8-element output, two-element blocks, and four-program grid.
The [Triton example](triton_examples/README.md) and
[Pallas example](pallas_examples/README.md) record the small adaptations needed
to run these calls against our annotated kernels and current libraries.

The next fixtures add [Triton fused softmax](triton_examples/test_fused_softmax.py)
and [Pallas masked softmax](pallas_examples/test_masked_softmax.py). They keep
their handwritten host logic. Triton explicitly models 2D allocation shape and element
strides with independent input/output row strides, a grid-stride row loop, and
a column tile.
Pallas types a one-row kernel and uses an explicit `vmap` to lift it to a 2D
host array. The Pallas stub checks the one-row output shape against its ref
and maps that shape across rows, but neither prototype statically proves
general grid coverage or layout and bounds conditions.
The [strided Triton copy probe](triton_examples/test_strided_copy.py) checks
independent row and column strides for sliced and transposed inputs, while the
tutorial bodies remain unchanged. These static-only types check that the
kernel's explicit address arithmetic agrees with its declared array layout;
Triton itself does not enforce that relationship.
Fused softmax additionally experiments with an explicit Torch-to-host-layout
conversion: callers write `softmax(as_host_tensor(x))`, and the handwritten
wrapper converts the checked host view to a Triton pointer view. The host
function has a symbolic matrix shape; the evaluated pointer markers are real
Python objects, but the launch still does not statically equate every symbolic
stride across kernel arguments.
A pre-run hook validates the missing equalities against actual tensors and
scalars before executing the kernel.

The next [Triton grouped matmul](triton_examples/test_matrix_multiplication.py)
and [Pallas blocked matmul](pallas_examples/test_blocked_matmul.py) fixtures
stress 2D tiling. Triton now uses rank-independent allocation pointers and
indexed-tile pointer types, but the operations that build 2D addresses and
masks retain specialized v0 stub types. Pallas accepts independently mapped
2D Refs and contextually types their index-map lambdas, but it cannot prove
arbitrary index-map arithmetic. A fixture deliberately bypasses the type
contract to demonstrate the wrong result such a map can produce. Both
fixtures keep the upstream kernel statements and exercise the supported CPU
interpretation mode.

`ConstExpr[T]` is an `Annotated[T, "triton.constexpr"]` type alias: Pyrefly sees
the underlying `Int[Block]`, while the runtime decorator recognizes either a
postponed string or an evaluated `Annotated` value and changes annotations to
`tl.constexpr` for each constexpr parameter. All other runtime annotations are
removed.
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
TRITON_INTERPRET=1 /home/stroxler/.kernel-shapes-venv/bin/python -m unittest -v triton_examples.test_fused_softmax
/home/stroxler/.kernel-shapes-venv/bin/python -m unittest -v triton_examples.test_fused_softmax
/home/stroxler/.kernel-shapes-venv/bin/python -m unittest -v triton_examples.test_strided_copy
TRITON_INTERPRET=1 /home/stroxler/.kernel-shapes-venv/bin/python -m unittest -v triton_examples.test_strided_copy
/home/stroxler/.kernel-shapes-venv/bin/python -m unittest -v pallas_examples.test_masked_softmax
/home/stroxler/.kernel-shapes-venv/bin/python -m unittest -v pallas_examples.test_blocked_matmul
/home/stroxler/.kernel-shapes-venv/bin/python -m unittest -v triton_examples.test_matrix_multiplication
TRITON_INTERPRET=1 /home/stroxler/.kernel-shapes-venv/bin/python -m unittest -v triton_examples.test_matrix_multiplication
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

## Explicit vector-add boundaries

Triton `checked_add(as_host_tensor(x), as_host_tensor(y))` validates the host
rank, unit element strides, shared length and device, block size, and output
dtype. `checked_vector` presents each unchanged Torch array as a typed
`InPointer` or `OutPointer` and derives `n_elements`; the handwritten launch
passes it to the kernel. The output is allocated with the selected dtype.
The checks also run when the kernel is launched directly, through the
evaluated semantic annotations and pre-run hook. CPU tests include empty,
partial, and full tiles; mixed input dtypes; and the upstream block size.

Pallas `checked_add(as_pallas_input(x), as_pallas_input(y), block_size=4)`
constructs a `vector_layout` with a typed grid axis, block, and index-map
lambda. `checked_pallas_call` builds a callable that checks runtime input
shape, dtype, and device against that layout before launching. The 2D matmul
fixture builds a different typed layout and uses the same checked call. Pallas itself
allocates the output and supplies its Ref to the kernel. Empty inputs return
an empty output without launching a zero-sized grid. The upstream inline
`pallas_call` remains alongside this example for comparison.

Contextual typing rejects simple index-map mistakes in the Pallas layouts,
but neither checked boundary proves that arbitrary index-map arithmetic or
grid traversal aligns with the operations inside the kernel. Mapping whole host shapes to Ref tile
shapes without explicit conversion would require a more general parameter-list
mapping operator or a Pyrefly hook; neither is part of v1.

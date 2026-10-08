# Kernel shape experiments (v1)

The [v1 assessment](V1_ASSESSMENT.md) summarizes the corpus, the checks
that bind a Python-facing shape contract to a kernel, and the remaining
trust boundaries. This is an experimental type system and checked-boundary
library, not a proof of GPU execution safety or a change to Pyrefly core.
The [boundary and corpus guide](BOUNDARY_AND_CORPUS.md) suggests an order for
reading the examples and records remaining example families. The
[Gluon/CuTe notes](OTHER_KERNEL_DSLS.md) capture insights from those
explorations without expanding the v1 implementation to either DSL.

The [Triton library](triton_library/README.md) and
[Triton examples](triton_examples/README.md) are separate from the
[Pallas library](pallas_library/README.md) and
[Pallas examples](pallas_examples/README.md). The v1 overlays were derived
from an earlier exploratory overlay but are independently owned; Gluon has no
v1 stubs. The
executable Triton experiments use `@semantic_jit` around kernel entrypoints,
starting with the vector-add body from
`python/tutorials/01-vector-add.py`. The kernel body is unchanged; the local
Triton stubs model allocation and mask semantics. Module-level `IntVar` declarations bind
symbolic `N` and `Block` with legacy-style generics; the kernel uses the same
`tlt.InPointer[[N], [1]]` and `Int[N]` annotations. Triton's source extractor
currently only recognizes `def name(`, so PEP 695 function type parameters
are not suitable for this first runtime probe. This experiment makes no changes
to Pyrefly core. The independent Pallas fixture runs its original
vector-add kernel through an explicit checked JAX boundary, using semantic
annotations directly without source rewriting.

## Kernel-body changes in v1

Keep the upstream algorithm, indexing, and control flow. A mechanical change
to introduce a fresh local is allowed when an existing parameter or local is
reassigned a pointer of a different semantic shape, for example changing
`X += row * stride` and later `X + cols` to `x_row = X + row * stride` and
`x_row + cols`. Comment next to each departure so readers can compare it
with upstream. This gives Pyrefly a true lower-rank pointer and makes the
transition visible in IDE inlay hints. It does not introduce typed semantic
helpers or assert that GPU performance is unchanged; the frontend and CPU
interpreter tests check the preserved behavior, not GPU performance.

Each vector-add fixture also shows the upstream host call beside an explicit
checked boundary. Triton's `add(x, y)` follows its tutorial wrapper with a
call-local device check for CPU testing. JAX's Pallas design document has no
named wrapper: `design_doc_add(x, y)` packages its inline `pallas_call`, with
the doc's fixed 8-element output, two-element blocks, and four-program grid.
The [Triton example](triton_examples/README.md) and
[Pallas example](pallas_examples/README.md) record the small adaptations needed
to run these calls against our annotated kernels and current libraries.

The [Triton fused softmax](triton_examples/test_fused_softmax.py)
and [Pallas masked softmax](pallas_examples/test_masked_softmax.py) fixtures
retain their handwritten host logic. Triton explicitly models 2D allocation shape and element
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

The [Triton grouped matmul](triton_examples/test_matrix_multiplication.py)
and [Pallas blocked matmul](pallas_examples/test_blocked_matmul.py) fixtures
stress 2D tiling. Triton uses rank-independent allocation pointers and
indexed-tile pointer types, but the operations that build 2D addresses and
masks retain specialized v0 stub types. Pallas accepts independently mapped
2D Refs and contextually types their index-map lambdas, but it cannot prove
arbitrary index-map arithmetic. A fixture deliberately bypasses the type
contract to demonstrate the wrong result such a map can produce. Triton's
matmul host boundary also builds a typed output-tile layout, deriving its
program count and checking the dimension and block-size arguments at launch;
the grouped PID mapping inside the kernel remains unchecked. Both
fixtures keep the upstream algorithm (with documented pointer-local renamings
where the semantic pointer type changes) and exercise the supported CPU
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
../.venv/bin/python -m unittest discover -s triton_examples -t . -p 'test_*.py'
TRITON_INTERPRET=1 ../.venv/bin/python -m unittest discover -s triton_examples -t . -p 'test_*.py'
../.venv/bin/python -m unittest discover -s pallas_examples -t . -p 'test_*.py'
```

Zero Pyrefly errors do not imply that every kernel operation has a semantic
shape rule: individual fixtures document narrow diagnostic suppressions
and properties that remain unchecked. A normal JIT captures the
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
Direct launches recheck only annotated pointer ranks, shapes, and strides,
plus shared symbolic integers, through the pre-run hook. Device agreement,
output dtype, block-size restrictions, and grid size rely on the checked
adapter and layout. CPU tests include empty, partial, and full tiles; mixed
input dtypes; and the upstream block size.

Pallas `checked_add(x, y, block_size=4)`
constructs a `vector_layout` with a typed grid axis, block, and index-map
lambda. `checked_pallas_call` builds a callable that checks runtime input
shape and dtype against that layout, and requires concrete input devices
to agree when available. The 2D matmul fixture builds a different typed
layout and uses the same checked call. Pallas itself
allocates the output and supplies its Ref to the kernel. Empty inputs return
an empty output without launching a zero-sized grid. The upstream inline
`pallas_call` remains alongside this example for comparison.

Contextual typing rejects simple index-map mistakes in the Pallas layouts,
but neither checked boundary proves that arbitrary index-map arithmetic or
grid traversal aligns with the operations inside the kernel. Mapping whole
host shapes to Ref tile shapes for arbitrary kernels would require a more general
parameter-list mapping operator or a Pyrefly hook; neither is part of v1.

## Review findings and v2 questions

The checked Python interfaces are useful shape contracts, but they are not
proofs that a kernel accesses or writes every intended element. Triton's
output layouts check grid sizes and named launch arguments; they do not prove
program-ID arithmetic. Pallas's typed index maps reject simple axis mistakes,
but its pattern-specific layout factories construct `BlockSpec`s behind a
generic `Layout` that stores them as `object`, and `checked_pallas_call` trusts
the constructed layout when it casts the underlying call's result. Existing
negative tests cover some wrong indexing and masks; more varied cases are
needed to characterize the boundaries of those rules.
For example, Triton's two-dimensional pointers require a stride-matched
row-selection operation before adding column offsets; adapted kernels use
fresh lower-rank pointer locals where upstream reassigned a parameter with
`+=`. This does not establish that the row index is in bounds or that each
row is visited. The 1D mask checks match allocation length, tile width,
offset-origin category, and known grid axis. They do not prove that a
predicate guards the exact address in each lane; the 2D rules do not retain
the same grid-axis information.

Triton's direct-launch validator resolves runtime-checkable semantic
annotations even when they are postponed strings or direct marker imports,
ignores unannotated parameters, and checks symbolic integers nested inside
`ConstExpr[...]`.
`triton_examples/test_semantic_jit.py` exercises these cases in both JIT and
interpreter modes. This is not a general annotation evaluator: stub-only
annotations such as attention's `tl.AttentionPointer` cannot register this
pointer hook. The checked attention host adapter validates its inputs; direct
launches of that kernel bypass those adapter checks. Checked host adapters
remain the intended safety boundary in v1.

For v2, try a composable description of host axes, tile/Ref axes, and grid
mappings before adding more kernel-specific layout factories or address/mask
overloads. [Three candidate experiments](NEXT_SEMANTIC_TYPE_EXPERIMENTS.md)
sketch row selection, mask provenance, and composable Pallas layouts, with
specific negative probes. The [Triton addressing design](TRITON_ADDRESSING_DESIGN.md)
explores a possible future combination of one reviewed PID intent annotation,
semantic operator rules, and runtime-identity wrappers, without changing the
v1 kernel bodies. A runtime bounds sanitizer could complement static
checks, but neither that nor static per-lane validity or full grid coverage is
implemented here.

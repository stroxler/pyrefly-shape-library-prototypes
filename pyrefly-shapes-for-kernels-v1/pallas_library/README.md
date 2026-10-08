# Pallas library

`pallas-stubs/` is an independent copy of the v0 JAX/Pallas stub overlay,
included on v1's Pyrefly search path. Unlike Triton, Pallas accepts semantic
`pl.InRef` and `pl.OutRef` annotations directly; no decorator or source
rewrite is needed. JAX arrays do not expose the Torch-style element-stride
contract checked in the Triton experiment.

`layout.py` accepts shape-typed `jax.Array` inputs directly, bringing grid,
block dimensions, and contextually typed index-map lambdas into one call.
`grid_axis` marks ceiling division as a grid dimension, while `BlockIndex`
tracks the host dimension and block width of each lambda parameter. The
`vector_layout` and `matmul_layout` check distinct block patterns against a
Ref-typed kernel and build actual Pallas `BlockSpec` objects. `row_layout`
models an empty-grid, full-row kernel using Pallas's default input/output
specs. `row_statistics_layout` supports multiple outputs, and
`attention_layout` binds query, key/value, and statistics axes, while
`row_input_gradient_layout` binds saved scalar statistics and input gradients.
Each factory
produces a `Layout` with a typed host input/output signature.

The factories use `InputBinding` and `OutputBinding` to assemble their runtime
metadata. Each binding names the axes of a host array and, where needed,
states its Pallas block shape and index map. `GridBinding` identifies an
output axis and its per-program block width. Shared axis names must have
equal host extents even when their arrays have different ranks; `None` in a
block shape removes that host axis from the kernel Ref. See
[concrete layouts](LAYOUT_BINDINGS.md) for every current example.

`checked_pallas_call(layout, interpret=...)` uses one parameter-list-generic
implementation for these patterns: it invokes `pallas_call` and validates
the input arrays' concrete shape, individually declared input dtypes, and shared device when
available before launch. JAX tracers inside `vmap` have shape and dtype but
do not expose a concrete device. The matmul layout requires positive dimensions
and exact divisibility; the vector layout permits a partial last block and
supports one or two inputs with distinct dtypes (e.g. boolean keep-mask and
floating-point values for dropout). Lambda return types reject
replacing a required grid index with zero or with an unrelated axis, but do
not prove arbitrary arithmetic, mask correctness, or complete output coverage.
Pallas supplies ordinary index values at runtime; `BlockIndex` is only a
static refinement, and the constructor explicitly bridges that difference.
`binding_layout` is an executable, runtime-checked metadata assembler, but
does **not** statically relate an arbitrary kernel's parameter tuple to its
input and output Refs. Keep the pattern-specific constructors for that static
check. Neither path proves index-map arithmetic, complete output coverage, or
that a kernel respects a given Ref's declared shape internally.

The copied JAX/Pallas stub overlay still includes semantic roles and overloads
for other v0 tutorials. These examples use its shared `InRef`, `OutRef`, tile,
masked-access, and `BlockSpec` types; removing unrelated overlay entries
requires pruning their cross-module references without weakening those rules.

For softmax, `row_layout` checks that the kernel's input and output Ref
shapes agree with `out_shape`; the shared launcher checks the actual row
length before invoking Pallas, which allocates and passes the output Ref to
the kernel. This layout is specific to a full-row input, one output, and
`grid=()`; it does not infer an arbitrary mapping from `BlockSpec` or prove
that every output element is written. Statically deriving a general host
callable from the kernel and specs remains future work.

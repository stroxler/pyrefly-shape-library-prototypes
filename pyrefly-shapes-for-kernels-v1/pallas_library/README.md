# Pallas library

`pallas-stubs/` is a shape-aware JAX/Pallas stub overlay for the included
examples, included on Pyrefly's search path. Unlike Triton, Pallas accepts semantic
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
[representative layouts](LAYOUT_BINDINGS.md) for several current examples.

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

The JAX/Pallas stub overlay models the included v1 examples using shared
`InRef`, `OutRef`, tile, masked-access, and `BlockSpec` types. The decode
attention kernel uses ordinary shape-parameterized Refs for query, key/value,
scalar bounds, outputs, and residuals. A dynamically selected decode tile is
a `TransformedRef`, rather than a separate attention-specific Ref role. Indexed
input/output Refs use a shared
`TransformedRef[BaseShape, TileShape, Role, ValidShape]` to check a tile load
or store against the original allocation bounds and tile-sized mask. The
valid shape defaults to the base shape for unpadded selections. Padded inputs
use `ValidInRef[Shape, ValidShape]`, and padded outputs use
`ValidOutRef[Shape, ValidShape]`; their selected views preserve the real
feature bound. Neither type tracks the symbolic selection location.
Boolean predicates use `Mask[TileShape, ValidBounds]`: a one-dimensional mask
can acquire a singleton broadcast axis, and intersecting row and column masks
produces a two-dimensional mask with both bounds. Loads and stores match the
mask's tile and valid bounds to the selected Ref. Bounds of `int` remain
gradual; this model does not prove that a predicate actually guards the
corresponding memory address. Mask composition with `&=` on an already-typed
local cannot change that local's shape parameters in Pyrefly; use of the
operator in the ragged-dot kernel is not proof of its final mask bounds.
The Pallas overlay does not include the disconnected paged-attention, LSE,
and convolution Ref families and their `pallas_call` overloads; these kernels
are not in the v1 corpus. Forward attention uses full and selected general
Refs; its query's valid extent is explicit even though it equals the visible
block extent. Backward attention's padded feature extent is retained through
both selected input loads and output stores; its optional segment input is
still a special Ref. Ragged dot's output is a `ValidOutRef` whose visible
column block and full valid-column bound remain distinct. Selecting its
rectangle produces a `RectOutputBlock`, whose store accepts either a row-only
mask or a combined row/column mask. Scalar boundary Refs produce an
`AxisBound[Rows]` for row-index comparisons. Ragged LHS/RHS selected inputs
retain dedicated types because the unchanged kernel loads them both masked
and unmasked in different runtime branches; their unmasked safety requires
branch-sensitive reasoning that the current type system does not provide.
TPU compiler parameters are modeled for the matmul example; TPU scratch-memory,
remote-copy, subcore, and prefetch typing is outside this example corpus.

For softmax, `row_layout` checks that the kernel's input and output Ref
shapes agree with `out_shape`; the shared launcher checks the actual row
length before invoking Pallas, which allocates and passes the output Ref to
the kernel. This layout is specific to a full-row input, one output, and
`grid=()`; it does not infer an arbitrary mapping from `BlockSpec` or prove
that every output element is written. Statically deriving a general host
callable from the kernel and specs remains future work.

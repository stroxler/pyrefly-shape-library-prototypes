# Pallas library

`pallas-stubs/` is an independent copy of the v0 JAX/Pallas stub overlay,
included on v1's Pyrefly search path. Unlike Triton, Pallas accepts semantic
`pl.InRef` and `pl.OutRef` annotations directly; no decorator or source
rewrite is needed. JAX arrays do not expose the Torch-style element-stride
contract checked in the Triton experiment.

`host_input.py` gives a checked host JAX array its own static role;
`as_pallas_input` checks rank without copying it. `layout.py` brings the grid,
block dimensions, and contextually typed index-map lambdas into one call.
`grid_axis` marks ceiling division as a grid dimension, while `BlockIndex`
tracks the host dimension and block width of each lambda parameter. The
`vector_layout` and `matmul_layout` check distinct block patterns against a
Ref-typed kernel and build actual Pallas `BlockSpec` objects. `row_layout`
models an empty-grid, full-row kernel using Pallas's default input/output
specs. All three produce a `Layout` with a typed host input/output signature.

`checked_pallas_call(layout, interpret=...)` uses one parameter-list-generic
implementation for all three patterns: it invokes `pallas_call` and validates
the input arrays' concrete shape, output dtype, and shared device when
available before launch. JAX tracers inside `vmap` have shape and dtype but
do not expose a concrete device. The matmul layout requires positive dimensions
and exact divisibility;
the vector layout permits a partial last block. Lambda return types reject
replacing a required grid index with zero or with an unrelated axis, but do
not prove arbitrary arithmetic, mask correctness, or complete output coverage.
Pallas supplies ordinary index values at runtime; `BlockIndex` is only a
static refinement, and the constructor explicitly bridges that difference.
The pattern-specific layout constructors still require typed signatures and
runtime checks, so this is not yet an arbitrary-kernel layout API.

The copied JAX/Pallas stub overlay still includes semantic roles and overloads
for other v0 tutorials. These examples use its shared `InRef`, `OutRef`, tile,
masked-access, and `BlockSpec` types; removing unrelated overlay entries
requires pruning their cross-module references without weakening those rules.

`checked_call.py` supplies `as_pallas_input(row)` to check rank and view the
*unchanged* JAX array as a checked host input. For softmax, `row_layout` checks
that the kernel's input and output Ref shapes agree with `out_shape`; the
shared launcher checks the actual input length before invoking Pallas, which
allocates and passes the output Ref to the kernel. This static view is specific
to a full-row input, one output, and `grid=()`; it does not infer a general mapping
from `BlockSpec` or prove that an arbitrary kernel writes its whole output.
The general parameter-list mapping needed to remove explicit conversion
remains future work.

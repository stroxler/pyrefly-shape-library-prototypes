# Triton library

`semantic_jit.py` translates static-only semantic annotations into Triton's
runtime signature and source. Its static `SemanticKernel` adapter keeps the
decorated function signature for grid-indexed launches; a JIT function is
still the actual runtime object. With runtime-checkable pointer annotations,
including postponed `tlt.InPointer`, `tlt.OutPointer`, and `tlt.InOutPointer`
annotations, a pre-run
hook checks actual Torch shapes, strides, and matching scalar arguments for
both JIT and interpreter launches. It also checks `Int[...]` inside
`ConstExpr[...]` and permits unannotated parameters. Postponed annotations
currently recognize those exact names, rather than arbitrary imported aliases;
stub-only pointer forms that cannot be evaluated at runtime do not register
this hook. `tlt.py` provides
evaluable pointer markers, with their arithmetic specified in `tlt.pyi`.
`host_tensor.py` is a separate Torch-side layout view: `as_host_tensor(x)`
checks a vector's unit element stride or, by default, a matrix's unit inner
stride, preserves the original Torch object, and marks a matrix's row stride
as unknown `int`. An explicit host-view type permits independently checked
matrix strides, including transposed inputs. `torch_views.py` converts checked
host views to Triton pointer views and returns observed dimensions and strides.
The pre-run hook validates symbolic relationships not retained statically at
the launch (see the Triton examples README).

The vector-add example supplies its own explicit launch metadata, including
the output dtype. Its checked host function verifies matching device, length,
and layout but leaves input dtypes unconstrained; Triton specializes them
when launching the kernel.
The `triton-stubs/` overlay is local to v1 and can diverge from v0. Its
ordinary Triton stubs were copied from v0; Gluon stubs are out of scope here.
The first executable usage is in `../triton_examples/`.
The 1D pointer-array overloads check that logical offsets were scaled by the
pointer's element stride, and their load/store masks must use the same logical
allocation length, tile width, offset-origin category, and known launch grid
axis. Distinct grid axes are checked separately from matrix address
orientation. The untagged axis (`-1`) from `program_id(axis: int)` or
adding arbitrary scalar indices to offsets can match another untagged axis;
that match supplies **no** grid-axis guarantee. Branches joining axis-zero
and axis-one offsets retain a union that is rejected against an axis-zero-only
mask or pointer, in both `tl.load` and `tl.store`.
For rank-two matrix pointers, column offsets require a prior row selection:
`x_row = X + row * stride` produces a rank-one pointer. Rebinding annotated
parameters with `+=` would retain their declared rank, so these examples use
fresh local names. This records a stride-matched shift but does not prove
that the selected row remains in bounds. `InOutPointer` is a generic
read/write allocation pointer used for grouped scratch and lock storage;
selecting one scratch row lowers its rank, while vector offsets produce
`InOutTilePointers`, an array of addresses. The lock/count allocation retains
a separate `LockArrayPointer` role so arbitrary read/write vectors cannot be
passed to atomic lock operations. Its length is checked as `2 * Groups` at
runtime; selecting one lock yields `LockSlotPointer`, then its count address
has a distinct role. The stubs do not prove predicate
implication or track masked-lane validity through tensors.
Seeded dropout also preserves the tile width when `tl.rand` receives shifted
unit-stride offsets, so `tl.where` and `tl.store` check its value-tile shape.

Grouped matmul uses evaluable generic `tlt.InPointer` and `tlt.OutPointer`
markers with two shape and two stride parameters. The launch hook checks
these against runtime tensors and scalar arguments. Generic pointer overloads
accept the 2D addresses and return generic `InTilePointers` and
`OutTilePointers`, retaining allocation shape, strides, tile shape, and a
wrapped-axis tag. Address expressions and masks still originate in
row/column-specific v0 overlay types, which remain a migration target. The
overlay also retains types for tutorials beyond these v1 fixtures; removing
them independently of the generic pointer migration would risk discarding
currently useful address and mask checks.

`launch_layout.py` checks one-dimensional grid size against a checked 1D or 2D
output view and its tile shape. Its `TiledOutputLayout` carries those symbolic
shapes; `launch` validates the named dimension, block-size, and optional
metadata arguments actually passed to the kernel. Vector add and grouped
matmul use it to bind lengths/dimensions, tile sizes, and grouping metadata to
their host output and launch configuration. Softmax instead uses
`GridStrideOutputLayout`, which binds output dimensions and block width while
allowing an occupancy-dependent number of programs up to the row count.
The mapping from `tl.program_id(0)`
to each output tile is still a kernel-side contract, not a verified property
of the layout.

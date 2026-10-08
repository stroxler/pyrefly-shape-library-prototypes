# Semantic-type experiment status and remaining directions

These checks concern how a kernel uses its declared inputs. They are
separate from the checked Python shape/layout boundary, which is useful even
when tiling, address selection, or per-lane validity remains unproved.
The [matmul addressing design](TRITON_ADDRESSING_DESIGN.md) works through
program-ID decoding and reduction-pointer updates as a possible type system
for this prototype or a later iteration; it is not an implemented experiment.

## 1. Require row selection before column access in Triton (v1, partial)

Softmax uses `row_ptr = input_ptr + row * row_stride`, then
`row_ptr + column_offsets`. Layer norm instead uses `X += row * stride`.
Pyrefly retains a subtype returned by `__iadd__` when it is assignable to
the annotated pointer. `tlt.pyi` therefore returns `SelectedInRow` or
`SelectedOutRow`, each a subtype of the original rank-two pointer, from a
row-stride shift. Column offsets apply to a lower-rank pointer or to a
selected-row subtype, but no longer to an unselected rank-two pointer.
The unchanged softmax, layer norm, and strided-copy bodies type-check. A
type-only layer-norm fixture checks both selected-pointer types and expects
`unsupported-operation` when a matrix pointer accesses columns before row
selection.

The operator rule checks that **at least one** shift uses the declared row
stride. Repeating the shift preserves the selected-row subtype and may move
the pointer out of bounds. It does not prove that the program ID chooses the
correct row, that all rows are visited, or that the row index is in bounds.
The selected-row subtype is a static state of the original rank-two pointer,
not a runtime object.
An explicitly described flattening pattern is still needed for kernels
that intentionally access a matrix without selecting a logical row.

## 2. Track the grid axis in Triton masks and addresses (1D v1)

`Offsets` and `Mask` carry the allocation shape, tile width, and a coarse
origin such as `"local"` or `"program"`. They now also carry a separate grid-axis
parameter through 1D program-id-scaled offsets, masks, and pointer arrays.
`tl.load` and `tl.store` require that parameter to match. A type-only
vector-add fixture shows that the preexisting origin category alone accepted
offsets from `program_id(0)` with a mask from `program_id(1)`; after adding
the axis parameter, those mismatches produce `no-matching-overload` for both
load and store. The checker also rejects a union of axes 0 and 1 on either
the pointer or the mask side. The unchanged executable vector-add body
type-checks. A discovered unittest runs Pyrefly with `--error unused-ignore`
so a negative probe failing to produce its expected diagnostic fails the
test suite.

An unrecognized scalar index or arithmetic path drops provenance rather
than manufacturing an axis guarantee. Unknown-axis offsets and masks can
match each other without proving either uses a particular grid axis; the
`program_id(axis: int)` fallback and adding an arbitrary scalar to offsets
are such unknown paths. The existing `"axis_0"` *origin*
marker describes a matrix address orientation, not grid axis zero. The
1D axis-tag result is not a complete multi-axis kernel analysis: row/column
offsets and `MatrixMask` still discard grid-axis provenance. Extending it
to real multi-axis kernels needs a separate typed combination rule for the
origins of different tensor axes.

Matching axes is only a necessary consistency check. Two different offset
expressions using the same axis can still be incompatible, and a matching
mask type does not prove that the predicate guards every accessed address.
Expression-level provenance, bounds implications, and validity of data after
a masked load would require a substantially richer analysis. Stop after the
axis-tag experiment if the stub changes proliferate without catching useful
mutants in existing examples.

## 3. Compose Pallas layouts from axis and block relationships

The common `checked_pallas_call` preserves the host callable signature, but
`vector_layout`, `matmul_layout`, `row_layout`, `row_statistics_layout`, and
`attention_layout` each encode their own Ref shapes, grid, `BlockSpec`s, and
host shapes. The generic `Layout` stores the constructed specs as `object`;
the typed correspondence lives in each factory rather than in those fields.

Try an explicit descriptor for each host-array axis: its symbolic extent,
block extent, grid axis (if any), and a contextually typed index map. Compose
these descriptors into input and output `BlockSpec`s beside the checked output
shape, then validate their concrete shape and grid obligations at runtime.
An axis reused by two arrays should retain the same symbolic extent; a
squeezed or reordered Ref axis must be described explicitly. Keep the kernel's
annotated Ref signature as a constraint on the resulting descriptor, not a
second unchecked statement of its interface.

Start with one proposed builder expressing both vector add and blocked
matmul; next challenge it with attention's distinct query/key lengths and
tuple of differently shaped outputs. The experiments should include a wrong
axis map and a wrong output shape, as well as a successful CPU call. Typed
index-map lambdas may work for fixed arities; mapping arbitrary parameter and
output tuples may need a Pyrefly tuple-mapping operation or a checker hook.
The aim is fewer kernel-specific factories with *at least* the current shape
guarantees, not fewer lines of wrapper code. Runtime checks alone cannot
establish that arbitrary Python index-map callbacks return the intended
coordinates or that the kernel writes every output tile.

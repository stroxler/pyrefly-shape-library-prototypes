# Three possible semantic-type experiments

These sketches concern how a kernel uses its declared inputs. They are
separate from the checked Python shape/layout boundary, which is useful even
when tiling, address selection, or per-lane validity remains unproved. No
proposal below has been implemented or shown to work across the corpus.
The [matmul addressing design](TRITON_ADDRESSING_DESIGN.md) works through
program-ID decoding and reduction-pointer updates as a possible type system
for this prototype or a later iteration; it is not an implemented experiment.

## 1. Require row selection before column access in Triton

The current `InPointer[[Rows, Cols], [RowStride, 1]] + column_offsets`
overload returns a tile of `Cols` even if no row was selected. Softmax already
uses the more informative sequence `row_ptr = input_ptr + row * row_stride`,
then `row_ptr + column_offsets`. Layer norm instead uses `X += row * stride`;
Pyrefly keeps the annotated matrix-pointer type of `X` after the augmented
assignment. Deleting the direct matrix-plus-columns overload would therefore
reject an unchanged, valid kernel as well as the missing-row mistake.

The target rule is: selecting a row with its declared stride yields a pointer
to the remaining column axis, and only that pointer accepts column offsets.
First, make a small type-only probe of an in-place row shift whose `__iadd__`
result has the lower-rank pointer type. If Pyrefly cannot retain that type
after `+=`, investigate flow-sensitive augmented-assignment typing in Pyrefly
or an explicit, out-of-band row-selection witness. Test against *both*
unchanged softmax and layer norm, plus a mutant that omits the row shift.

This rule would establish that the stride and selected axis line up; it would
not prove that a program ID chooses the correct row, that all rows are visited,
or that the row index is in bounds. Some kernels intentionally flatten a 2D
allocation, so any strict rule needs an explicit way to describe that access
pattern rather than treating all matrix-plus-offset code as an error.

## 2. Track the grid axis in Triton masks and addresses

`Offsets` and `Mask` currently carry the allocation shape, tile width, and a
coarse origin such as `"local"` or `"program"`. Two offsets produced from
different `tl.program_id` axes can consequently appear equivalent.

A narrow prototype could make `program_id(axis=0)` return a
`ProgramId[Literal[0]]` and carry that axis through `TileStart`, `Offsets`,
pointer arrays, and masks. Then `tl.load`/`tl.store` would require a mask and
pointer array with the same grid-axis tag. Test ordinary vector add and
multi-axis kernels, plus a mutant that constructs its pointer array from one
axis and its mask from another. Arithmetic that loses a recognizable origin
should widen conservatively instead of manufacturing an axis guarantee.

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

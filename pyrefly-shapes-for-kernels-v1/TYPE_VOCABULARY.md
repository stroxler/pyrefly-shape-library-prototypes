# Semantic type vocabulary

This guide is for reading the kernels as well as reviewing their types. A
fundamental type should describe a feature of a kernel value, an address, an
allocation, or its checked host boundary. A type named for an example is not
accepted as a fundamental concept merely because it makes that example check.
The tables distinguish the intended vocabulary from example-specific modeling
that still needs work. They describe static promises, not a proof of memory
safety or grid coverage.

Start with [Triton vector add](triton_examples/test_vector_add.py) and
[Pallas vector add](pallas_examples/test_vector_add.py). Both accept two
host arrays of length `N`, process a tile of size `Block`, and write a result
of length `N`; they differ in who computes the addresses:

| Stage | Triton vector add | Pallas vector add |
| --- | --- | --- |
| Host inputs | Torch tensors, checked as shape `[N]`, element stride `[1]`, and a shared device. | JAX arrays, checked as shape `[N]`, equal dtype, and a shared device. |
| Kernel inputs | `InPointer[[N], [1]]` denotes a pointer to the *whole allocation*. | `InRef[[Block]]` denotes the *current block*, selected from the array by a `BlockSpec` index map. |
| Which tile? | The kernel reads `program_id(0)`, multiplies it by `BLOCK_SIZE`, and adds `arange(0, BLOCK_SIZE)`. The resulting offsets select addresses when added to each pointer. | `pallas_call` applies the grid and `BlockSpec`s, then hands the kernel its selected Refs. The kernel reads `x_ref[:]` directly. |
| Partial final tile | `offsets < n_elements` is a `Mask` associated with the selected pointer array for masked `tl.load`/`tl.store`. | Pallas handles boundary writes for this fixture; more complex Pallas reads use explicit indices and masks. |
| Result inside the kernel | `tl.load` yields a device `tensor[[Block]]`, not a Torch tensor; `tl.store` writes through an array of output pointers. | `x_ref[:]` yields a device `Tile[[Block]]`; assignment to `o_ref[:]` writes the output block. |

The important distinction is between *the whole host allocation*, *the block
handled by one program*, and *the current values in that block*. Triton exposes
the address construction; Pallas's `BlockSpec` usually does it before entering
the kernel. Neither language's grid size alone proves that the right addresses
are visited, and a correctly shaped mask does not prove it was computed from
the same offsets as the pointer array.

Names such as `InPointer`, `OutPointer`, `InRef`, `ValidOutRef`, and
`TransformedRef` are *our static annotations*, not new GPU classes. The runtime
still sees Triton pointers, pointer arrays, tensors, and descriptors, or Pallas
Refs, selected TransformedRefs, and device values. An access or validity tag
adds a type-checking promise; it does not make the kernel compiler check that
promise. The checked host adapter and kernel-body rules provide different,
partial evidence for that promise.

In attention backward, `mask = offs_m[None, :] >= offs_n[:, None]` is a causal
predicate used by `tl.where(mask, pT, 0.0)` to zero a value tile. It has
type `LogicalMask[[BlockN, BlockM]]`: Pyrefly checks that `pT` has the same
tile shape, but does not check that `>=` is the right predicate. It is *not*
the bounds `Mask` supplied to `tl.load` or `tl.store`. Calls such as
`tl.load(qT_ptrs)` have no mask argument. Their selected-pointer annotations
check the host shape, view strides, and tile composition, but do not prove
that a particular program's unmasked address is in bounds or that its causal
predicate is correct.

## Triton: intended fundamental concepts

| Types | What the type tracks | Why it is distinct |
| --- | --- | --- |
| `Int`, `IntVar`, `IntTuple` (shape extensions) | Symbolic extents and strides shared between a host call and a kernel. | They are the common language for all shape relationships, not new Triton runtime values. |
| `ConstExpr[T]` | A compile-time kernel parameter whose symbolic value can also appear in shape types. | Triton recompiles/specializes for constexpr values; the semantic JIT adapter turns this annotation into `tl.constexpr` in Triton's copy of the source. |
| `host_tensor.Tensor[Shape, Strides]` | A checked Torch host-array view with shape and element strides. | Ordinary Torch tensor types do not presently expose strides statically; the host adapter checks them before a tensor is presented as a kernel pointer. |
| `InPointer[Shape, Strides]`, `OutPointer[Shape, Strides]`, `InOutPointer[Shape, Strides]` | A base device pointer, the full host allocation's logical shape and element strides, and read/write capability. | Shape and strides constrain address formation; access direction constrains loads and stores. These are static views of actual Triton pointer arguments. |
| `SelectedInPointer[Parent, ParentStrides, View, ViewStrides, Selection]`, `SelectedOutPointer[...]` | A single pointer after selecting a lower-rank view from a host allocation, retaining the parent shape and strides, the selected view, and access direction. | The resulting pointer is still a scalar pointer, not a pointer array. Its selection tag limits which offset composition can yield an unmasked load/store pointer array; the tag does **not** prove the selected origin is in bounds. |
| `ProgramId[Axis]`, `TileStart[Tile, Axis]`, `GroupSize[Groups]`, `GroupIndex[Groups]`, `GroupQuotient[Groups]` | A program's grid coordinate, an origin derived from a tile extent, and the remainder/quotient of splitting a program coordinate into groups. | The grid coordinate is not itself a memory address. Division and modulo with the same group size yield distinct index kinds; arbitrary integer arithmetic still loses this relationship. |
| `AxisStride`, `AxisAddress`, `CombinedAddress` | An address contribution from a group index times an axis stride, and the composition of quotient/remainder contributions. | The stride and index kind are checked before a combined offset can select a lower-rank pointer. This is a useful intermediate address, not a new attention-specific pointer kind; it still does not prove grid coverage. |
| `AxisRange[Extent, Axis]`, `AxisIndex[Extent, Axis]`, `AxisOffset[Extent, Stride, Axis]` | A loop over logical axis positions, one scalar position, and its stride-scaled address contribution. | A scalar position is not an offset tile; multiplying by the intended stride is checked before selecting a lower-rank pointer. `Axis` describes an allocation axis, not a Triton grid axis. |
| `Offsets[Tile, Steps, Origin, GridAxis]` | The extent and element-address step of each axis of an offset tile, plus its origin when known. | An offset tile is a device value that has not yet been added to a base pointer. `Tile` and `Steps` have the same rank. Rank insertion and step insertion are modeled together in `_shapes.pyi`. |
| `AxisOffsets[Tile, Steps, Axis, Role]` | An offset tile with an axis and whether it is a raw index or an address contribution. | Its common 2D row/column forms are type aliases, not distinct pointer or offset classes. An axis step of zero means broadcasting; retaining the index/address role prevents a raw index from silently standing in for a scaled address. |
| `BoundedOffsets[Dim, Tile, Transform]`, `BoundedAxisOffsets`, `BoundedAxisAddress`, `BoundedAddress[Dim, Tile, Steps, Transform, Axis]` | Offsets constrained by a wrap or clamp, retaining the bounded dimension, expanded axis, tile extent, and address strides as arithmetic proceeds. | These are intermediate operations on offset values, not separate wrap-, clamp-, or matrix-specific runtime values. `Tile` and `Steps` retain their axes together; `Transform` and the bounded axis distinguish the resulting pointer-array checks. |
| `InTilePointers`, `OutTilePointers`, `InOutTilePointers` | An array of device pointers with allocation shape/strides, selected tile, access capability, and (where known) bounds origin. | A vector of pointers is a different Triton value from one base pointer. It is the value passed to `tl.load` or `tl.store`; a mask must agree with its tile and allocation bounds. |
| `PackedPointer[Logical, Storage, Strides, Block, Layout]`, `PackedTilePointers` | The distinction between a logical packed array and its physical byte allocation, including the storage steps and block selected by an address. | Two FP4 elements per byte and shuffled scales are physical layouts. One base pointer and one pointer-array type cover both packed-axis orientations and scales; layout tags select the applicable address step and load rule. |
| `PackedScaleTile[Logical, Physical, Stage]` | The physical tile shape across a packed-scale reshape/permute sequence. | The value lives in device registers, not in a pointer array. The modeled MFMA-16 and MFMA-32 paths retain different physical ranks and permitted permutations without separate hardware-named tile classes. |
| `LogicalMask[Tile]` | A boolean device tile used for elementwise selection, including causal attention predicates. | It inherits the device tensor's tile shape so `tl.where` checks alignment with its value tiles. It carries no allocation bounds and cannot substitute for a `tl.load`/`tl.store` bounds mask. |
| `Mask[Target, Tile, Origin, GridAxis]` | Active lanes of an access tile, the target allocation shape used for the comparison, and provenance when expressible. | A mask is a boolean device value; it must be checked against the pointer array being accessed, not inferred from tile shape alone. Not every mask identity or compound boolean expression is proven in v1. |
| `tensor[Tile]` | A loaded or computed device tensor's tile shape. | A `tl.tensor` is a device value, not the full host `torch.Tensor` and not a pointer array. |
| `tensor_descriptor[Shape, Strides, Block, Access]` | A TMA descriptor's allocation shape, element strides, load/store block, and read/write capability. | A descriptor is a Triton runtime object with block loads/stores rather than an array of pointers. Host construction still needs validation; `read`, `write`, and `read_write` operations are distinguished statically. Specialized method overloads check grouped-offset steps for the modeled 2D/5D transfers, but do not prove which position of a Python list belongs to which axis. |
| `PointerTable`, `PointerTableSlot`, `DeviceAddress`, `IndirectInPointer`, `IndirectOutPointer` | Indirect device addresses loaded from a table and their read/write capability. | A pointer fetched from a table has no statically known allocation extent; its runtime address must not be conflated with a pointer into a known host tensor. |

`RowAxisOffsets`, `ColumnAxisOffsets`, `RowAddress`, and `ColumnAddress` are
short aliases for 2D instances of `AxisOffsets`, not separate classes.
Masked scalar pointer variants remain intermediate
scaffolding until their transformations can be encoded compositionally.
Tags such as `Origin` record what a check has established; a tag alone does
not establish that every program visits the correct tile.

The stored dropout keep-mask is loaded from a Torch boolean tensor whose
dtype is checked by the host adapter. The Triton pointer types do not yet
carry dtype, so that `tl.load` produces a gradual `tensor[Tile]`; one
`tl.where` overload accepts that condition while still checking its tile
shape. Computed comparisons yield `LogicalMask[Tile]` directly. This does
not claim static dtype validation for arbitrary loaded conditions.

`Offsets[Tile, Steps, Origin, GridAxis]` scales its address steps using one
rank-generic shape-DSL expression:
`dsl.IntTuple(step * scale for step in steps)`. Pyrefly evaluates the generator
element by element when `Steps` has a known rank, retaining symbolic products,
zero steps, and tuple length. Scaling rejects offset tiles whose extent and
step tuples have different ranks. In a type expression, the symbolic scalar must
be passed as `Int[Scale]`, not a raw `Scale` type variable. Tests cover ranks
one through three and reject an incorrect step. If the rank of `Steps` itself
is unknown, this evaluator cannot retain a deferred elementwise mapping and
the result becomes gradual; that larger case is not needed for these kernels.

Mask-axis insertion remains rank-specific: a generic `insert_extent` return
does not currently preserve the existing wrong-grid mask rejections at
`tl.load`. Those negative checks must stay in place before generalizing
`Mask.__getitem__`.

## Pallas: intended fundamental concepts

| Types | What the type tracks | Why it is distinct |
| --- | --- | --- |
| `jax.Array[Shape]` | The Python-facing JAX array shape shared with a checked Pallas layout. | It is a whole host array, not the per-program Pallas Ref; JAX handles its physical strides. |
| `InRef[Shape]`, `OutRef[Shape]` | The logical block shape of a Pallas `Ref` supplied by a `BlockSpec`, and its read/write capability. | Pallas indexes logically; physical element strides are not exposed as Triton pointer arithmetic. A whole-allocation block is simply a degenerate block Ref. |
| `ValidInRef[Shape, ValidShape]`, `ValidOutRef[Shape, ValidShape]` | A padded block's visible shape and the extent containing actual input or output values. | Visible shape alone cannot tell whether reading or writing a padded lane is valid. Padded Ref access must not inherit an unmasked operation that applies only to fully valid blocks. |
| `TransformedRef[BaseShape, TileShape, Role, ValidShape]`, `InRefAt`, `OutRefAt` | A selection made inside a kernel, retaining original shape, selected shape, read/write role, and valid extent. | A `.at` selection is not another `BlockSpec` block; its parent extent matters for masked indexed accesses. `TransformedRef` currently has no methods of its own: only explicitly stubbed load/store operations work. The indexed-view classes are candidates for consolidation. |
| `Tile[Shape]`, `Indices[Block]`, `Mask[Tile, Bounds]` | Device values, logical index arrays, and active lanes relative to a bound. | Value, index, and mask have different supported operations; none needs a named attention, decode, or matrix subtype. |
| `ScalarFloat`, `DynamicSlice`, `HalfRowSlice[Size]` | A scalar computed in a kernel and a dynamic Ref selection with a known slice extent where expressible. | A scalar and a slice/indexing instruction are neither a Ref nor a tile; the specialized half-row slice may be reducible to a general extent-bearing slice. |
| `AxisBound[Extent]`, `AxisBoundRef[Extent]` | A scalar bound and the Ref supplying it for one logical axis. | The bound is a scalar device value, but its symbolic extent must remain associated with the input's row domain for comparisons. |
| `RectOutputBlock` | A selected output block allowing its documented row-only or full-bound store masks. | Its store capability cannot be applied to arbitrary padded output views: that would let backward attention write outside its valid head dimension. This is an access distinction, not a ragged-kernel identity. |
| `BlockSpec[Shape, Layout]`, `GridSize[Length, Block]`, `ProgramId[Axis]` | Per-program block shape, index-map intent, grid extent, and current program coordinate. | These describe how Pallas maps whole inputs to block Refs. The v1 layout builder checks selected relationships, not arbitrary index-map arithmetic. |
| `AccumRef[Shape]` | An accumulation Ref's writable values. | Read-modify-write capability differs from input-only or output-only access; the type does not prove initialization. GPU/TPU memory-space variants are not part of the current core-Pallas overlay. |

## Families that do **not** yet pass this review

These are active or internally reachable stub families, not a list of
fundamental abstractions. Deleting a class simply because a test file never
names it would remove the inferred type of a live intermediate expression.

| Family | Current reason for its existence | General form to investigate |
| --- | --- | --- |
| Triton unmasked selected-view accesses | Use generic selected pointers and pointer arrays after head/row selection. These examples make full-tile `tl.load`/`tl.store` calls without passing a mask. | Runtime host checks constrain shape and contiguity. The types check view and tile composition but do not prove that every selected position is in bounds or that programs cover the allocation without overlap. |
| Triton packed-scale reshape/permute overloads | Model the two exact reshaping paths of the CDNA4 example using one `PackedScaleTile` class, with partially gradual physical dimensions where K-group arithmetic is not known. | A general shape-preserving `tensor.reshape`/`permute` model should retain element count and axis order. The physical layout and permitted permutations are still expressed by example-specific overloads, not by general reshaping algebra. |
| Triton `Lock*`, `CountPointer`, `SplitStride`, `SplitAddress` | Atomic lock/count roles and split-axis stride arithmetic. Split-K scratch itself is a generic 3D `InOutPointer`, reduced to a 2D pointer and then a pointer array. | Explore a general typed index-times-stride operation while preserving the split-grid-axis rejection and the lock/count atomic capabilities. |
| Pallas `MhaSegmentRef` | Carries segment-length behavior specific to a live attention fixture. | Determine whether a generic scalar bound or block Ref plus explicit indexing expresses the same operations without losing checks. |
| Pallas `RaggedLhsRef/At/Slice`, `RaggedRhsRef/At/Slice` | Keep track of selected row/inner/column extents and, crucially, permit unmasked loads on a full block in one branch of the existing kernel. | Generic `TransformedRef` works for masked selection, but allowing all selected views to load without a mask would be unsound. A branch-sensitive proof of a full block, or a targeted semantic annotation, is needed before removing this distinction. |
| Pallas TPU scratch, DMA, prefetch, distributed blocks, and indirect accesses | These facilities have real runtime semantics but are outside the v1 overlay's kernel corpus. | Add a representative kernel and its runtime/stub checks before deciding which distinct memory-space or access-capability types are needed. |

The hard gap is *where* a program points within an allocation. Shape and
strides check many host/kernel contracts, but the general relationship between
grid coordinates, intermediate integer arithmetic, and addresses is not yet
proven. The [addressing sketch](TRITON_ADDRESSING_DESIGN.md) discusses a
possible later experiment. This inventory should shrink as live examples are
expressed using the fundamental concepts; an example-specific class does not
become fundamental simply by appearing in this table.

## Reading the next examples

In [Triton attention backward](triton_examples/test_attention_backward.py),
`bhid = tl.program_id(2)` is a coordinate in a grid flattened across batches
and heads. `bhid % H` identifies the head within a batch; `bhid // H`
identifies the batch. Multiplying each typed index by its matching head or
batch stride and adding the results produces a `CombinedAddress`. Adding that
address to a rank-4 base pointer selects a lower-rank pointer for one head.
This catches mixing a head count or quotient/remainder stride from another
allocation; it does not prove that the grid size or all subsequent tile
addresses are correct.

In [Pallas attention backward](pallas_examples/test_attention_backward.py), a
Ref can expose a padded feature width while only a smaller `HeadDim` is valid.
`ValidInRef`/`ValidOutRef` retain both widths; selecting a tile produces a
`TransformedRef` that still knows the valid width. A matching mask can guard
the load or store; a mask using the padded width is rejected. Compare this
with [Pallas ragged dot](pallas_examples/test_ragged_dot.py): its LHS/RHS
selections still have dedicated types because their unmasked loads rely on a
runtime full-block branch, not merely on a shape or a mask type.

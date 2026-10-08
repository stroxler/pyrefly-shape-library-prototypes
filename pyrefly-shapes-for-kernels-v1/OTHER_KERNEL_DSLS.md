# What Gluon and CuTe add to the kernel-shape model

This tracked design record preserves evidence from Gluon and CuTe experiments
without adding either language to the v1 implementation. The CuTe fixture
sources were kept outside version control pending a review of their copied
sources' provenance and redistribution terms; this is a precaution, not a
finding that CuTe itself cannot be documented or redistributed. The results
below describe those local experiments, not reproducible v1 tests. In
particular, a Triton allocation pointer's
`InPointer[[M, N], [StrideM, StrideN]]` captures *host geometry*, not every
layout concept in another DSL. The decisive question is
whether a typed operation actually consumes a claimed dimension, rather than
whether a signature happens to mention it.

## Gluon: storage and asynchronous staging

Gluon extends Triton with explicit blocked/thread layouts, tensor-memory and
shared-memory staging, and descriptors for asynchronous copies. A useful
future contract has at least four distinct layers:

1. The Python allocation's logical shape, actual strides, dtype and device.
2. A descriptor's full allocation extent, strides, physical block extent and
   address-space role. A descriptor block `[1, BY]` is not the host's full
   `[XMax, YMax]` shape.
3. The shared-memory or tensor-memory tile's shape and thread layout.
4. The worker/producer-consumer relationship and its ordering protocol.

The first two layers are close to v1 Triton's pointer and checked-boundary
model, but descriptors must retain both *whole-allocation* and *tile* axes.
For example, a 2D TMA elementwise add can reject a mismatched B extent at
its load while still failing to establish C's full host extent from the body.
An annotation for C's host width alone is a promise until a typed operation
consumes it. A checked launch must validate descriptor construction and its
attachment to the actual Torch allocation.

Gluon TMA gather/scatter loads a source `[XMax,YMax]` through an index vector
`[BX]` into a tile `[BX,BY]`. Index-vector length, descriptor block width,
and the tile presented to a store can be checked; numerical row indices,
duplicate scatter targets, offset alignment, and host bounds are not consequences
of shape equality. In a fused gather/scatter matmul, the input contraction
dimension K reaches the MMA consumer, but the full output N and accumulator
result shape can be lost if the accumulator type erases them. Do not equate the
physical transfer block `[1,BY]` with the logical shared tile `[BM,BY]`.

Warp-specialized vector add and persistent MMA carry host `[Rows,Cols]` through
descriptors and shared-memory rings; typing each worker's inputs can catch
wrong B/C widths at the worker call. Static shape preservation does not prove
that every worker runs, a ring buffer is filled before reading, barriers are
paired, or a persistent schedule visits every output tile. A tensor-memory
copy can additionally check that SMEM and TMEM tile shapes match, but shared
and tensor memory remain separate roles, even when their numeric shapes agree.

The v1 Triton `InPointer`/`OutPointer` shape-and-strides vocabulary is a sound
starting point for a Gluon host interface. A future experiment should add
*composable descriptor and memory-space roles*, not force asynchronous
descriptors into a generic pointer or claim temporal correctness from a shape.

## CuTe: shape/strides are necessary, not sufficient

CuTe's `Tensor` carries logical shape and a layout mapping logical indices to
storage; an allocation pointer, a CTA tile, a per-thread fragment and a
register tensor need different contracts. Unlike a Triton pointer, a CuTe
view may have nested layout shape/stride trees and a per-thread value layout.
Its `from_dlpack` bridge is a promising place to check the host's actual
shape, strides, dtype and device before assigning semantic roles. The flat
GEMM example illustrates that `[M,K]`, `[K,N]`, `[M,N]` must remain distinct
even if the kernel receives flat lengths `M*K`, `K*N`, `M*N`: Pyrefly cannot
generally recover M and K by inferring backwards through their product.

### A concrete layout example

Consider a host allocation `A` of logical shape `[M, K]` and element strides
`[K, 1]`, divided into CTA tiles `[BM, BK]`. Its tile/rest coordinates can be
described at a fixed depth with existing shape-extension tuple types:

```text
host       shape [M, K]                            strides [K, 1]
CTA tile   shape [BM, BK]                          strides [K, 1]
CTA rest   shape [ceil(M/BM), ceil(K/BK)]          strides [BM*K, BK]
thread     a partition of that tile by the chosen thread/value layout
register   the fragment copied from that thread partition
```

`zipped_divide` exposes the first two local modes together with the rest
modes. `local_tile` chooses a CTA view, and thread partition and `copy`
subsequently refine its layout. This is *not* just the v1 Triton pointer
with a smaller shape: the same host allocation participates in several
layouts, and tile/rest coordinates, per-thread ownership and memory space
are separate properties. For noncontiguous host layouts, replace `[K,1]`
with the actual host strides and carry them through the same transformation.
The expressions illustrate the mapping, not a new v1 type API or a claim
that arbitrary CuTe layout algebra is statically modeled.

The static experiment represented one tile/rest level as a pair of
`IntTuple`s (`IntTuples`) and several *fixed* two-level layout patterns with
specialized overloads. It preserved shape and strides under selected
`composition`, `flat_divide`, `local_tile` and `zipped_divide` uses without
changing Pyrefly core. Existing shape operators could compare a recorded
tile product with a block size, but could not generally infer several unknown
thread/vector/unroll factors backwards from that product. The lesson is to
carry known factors forward or give them explicit semantic roles; it is not
that arbitrary nested CuTe layouts are already solved by `IntTuples`.

### Where the local fixtures actually reached

The Ampere pointwise-add exploration followed the example under
`ai_acceleration/cute_dsl/examples/ampere/elementwise_add.py`. Separate
local fixtures followed its `make_copy_atom`, `make_tiled_copy_tv`,
`partition_S`, `make_fragment_like` and `copy` operations and then the
unchanged *whole device body*. The latter checked consistent per-thread
shapes through both input copies, register arithmetic, output copy and
coordinate-predicated tail, with negatives for incorrect fragment width
and predicate tile. The host wrapper was not thereby proved: after tiling,
a rest dimension such as `ceil(M/BM)` cannot reconstruct M, and a fragment
type does not prove it points into the declared input allocation.

Another *whole kernel plus JIT wrapper* fixture, the TMA elementwise add
under `ai_acceleration/gs_opt/learn_01_cuteDSL_tma_element_add.py`, followed
host-shaped A/B/C views through separate load/store descriptors, CTA
partitions and shared-memory tiles. It rejected mismatched descriptor roles
and tile/host dimensions at actual typed operations, while trusting the
annotations on host input pointers. It did not check hardware alignment,
barrier ordering, actual Torch storage, or partial-tile safety. This is a
useful v1 analogy: a checked constructor needs to validate both the
*source allocation* and the descriptor's tile, not only their named shapes.

The CUTLASS automatic-predication copy example tests non-contiguous
`[Rows, Cols, Layers]` storage with element strides
`[1, Rows, Rows*Cols]`: it preserves those dimensions through an existing
two-level nesting and an actual `from_dlpack` conversion. Runtime copy
semantics handle partial CTA tiles; a static type preserving the original
extent does *not* prove that the generated predicate is right or that a
program's block coordinates are in bounds. In the RMSNorm weight broadcast,
a vector `[N]` becomes a logical matrix `[TileRows,N]` with strides `[0,1]`:
stride zero is useful metadata, not evidence of an independent 2D allocation.

For Flash Attention, indexing a 4D host tensor
`[Batch,Sequence,Heads,Dim]` at a batch and head gives a 2D logical
`[Sequence,Dim]` view with row stride `Heads*Dim`, *not* `Dim`. Q and K/V
can have different sequence lengths but share head and feature axes. Source
slices preserved those dimensions through Q/K/V `local_tile` and output
`partition_D`; they did not cover the intervening attention, allocation
provenance or launch. This is a concrete reason v1's separate shape/strides
parameters scale better than named `RowMajorPointer` classes, while still
leaving an additional per-thread layout to model.

A Blackwell batched GEMM experiment used first-axis-major inputs
`A[M,K,Batch]`, `B[N,K,Batch]`, `D[M,N,Batch]` with strides
`[1, Rows, Rows*Cols]`. Its grid had separate
`[ceil(M/BM), ceil(N/BN), Batch]` axes while its device tiles grouped rest
dimensions into a nested mode. Separate source slices checked the host
converter, grid, K subtiles, distinct A/B CTA maps and `tma_load`, and an
epilogue `tma_store`. Wrong A/B K, B batch, output N, stage width, or CTA
map role reached original operations in some slices. The independently
checked host/device slices did *not* establish their intervening tiler,
SMEM allocation, launch or source-to-physical-buffer identity. In particular,
Pyrefly widened some epilogue map factors to `int`, allowing a map made with
the wrong tile width to reach a store despite matching outer shapes.

In the Flash Attention output source slice, query/output sequence equality
was only declared: the query was not read there. The JAX-based CuTe launcher
demonstrated the reverse problem: a concrete shape-specific launcher can be
incorrect even when the host shape declaration itself is checked. Check the
*actual host-to-kernel call*, not just its two annotated ends.

The CUTLASS JAX fused-bias-ReLU example is an especially useful warning:
typing a generic `cutlass_call` can widen the dimensions of a polymorphic
launcher to ordinary `int`. A concrete launcher specialized to `[512,32]`
was accepted alongside declarations for input/output `[1024]` and bias
`[64]`. Other checks correctly rejected a wrong declared JAX bias or CuTe
output in isolation, but did not tie *that launcher* to *those inputs*.
Conversely, contiguous Torch `[M,K]` flattened to `[M*K]` and converted via
`from_dlpack` kept the original M/K roles in a narrowly tagged experiment;
the original JAX bridge did not gain that guarantee automatically.

### Provenance, masks and future type-system work

A predicated `cute.copy` needs the mask's host bound to refer to the very
allocation being accessed. A shape-correct predicate derived from an
independent coordinate tensor can still protect the *wrong* array. The
SGEMM coordinate experiment tagged identity tensors with their A/B host
roles and checked `elem_less` against corresponding host bounds; an opaque
role tag was necessary because treating those extents as `Int[M]` subclasses
accepted a swapped B extent for an A comparison. But an upstream predicate
buffer built only from per-thread fragment shape contains no M/N host
extent in its arguments, so a stub cannot soundly invent the missing bound.
It requires a checked producer, a cross-operation relation or a visibly
trusted assertion. Full layout-tree support would not repair that gap.

The Helion pointwise example illustrates conditional-specialization limits:
the full `addcmul`-mode body kept two expected errors in the inactive
optional-bias branch because Pyrefly did not refine
`cutlass.const_expr(self.bias_n > 0)` to a constant Boolean. The checked
thread and value tiles did not verify the scalar operation's numerical
formula, unmasked last tile, or Python wrapper launch. The RMSNorm
pointer-boundary and weight-broadcast slices likewise checked `[M,N]` and
`[N]` view construction separately from an optional weight-present kernel
branch; an annotation declaring weight presence did not make a `const_expr`
guard narrow an optional value across the wrapper call.

General CuTe layout composition/coalescing and arbitrary nesting likely need
an `IntTree`-like representation with equivalence rules, symbolic
divisibility, and potentially swizzle tags. A restricted first prototype
could still use flat shape/stride tuples, fixed-depth tuples and nominal
roles for CTA maps, predicates, fragments and memory spaces, following the
v1 Triton host contract. A general system would also need to compare
normalized/coalesced layouts rather than requiring identical syntax, and
represent shape/coordinate refinement separately from stride equality and
non-affine swizzles. That extension would not by itself repair missing
physical-allocation or coordinate-predicate provenance.
A predicate buffer allocated only from its per-thread tile geometry cannot
soundly acquire an M- or N-host-bound tag: distinct host extents can produce
identical allocation arguments. Some role tags must originate in a checked
producer or be explicitly trusted. Exact divisibility belongs to *unmasked*
operations that require it; `zipped_divide` and predicated copies can cover
partial tails.

## Consequences for another prototype

Keep the v1 distinction between a host allocation's shape/strides, a launch
layout, a local tile and a checked access. For Gluon add descriptors and
distinct memory-space/thread layouts; for CuTe add hierarchical layout
transforms and fragment provenance. Separate static evidence established by
the unchanged kernel body from runtime validation of host allocations. Neither
DSL makes scheduling, numerical indices, actual hardware alignment, or
barrier ordering follow from a shape annotation alone. These conclusions
refer to the upstream Triton/Gluon and CUTLASS/CuTe examples, not to an
implemented Gluon or CuTe v1 overlay.

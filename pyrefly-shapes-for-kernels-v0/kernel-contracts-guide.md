# Reading the kernel shape prototypes

These prototypes ask a specific question: can semantic annotations on a kernel's
parameters, together with typed library operations, check that the kernel's
*unchanged body* is consistent with the shapes claimed at its Python boundary?
The annotations are deliberately not valid runtime annotations for these DSLs.
Use the separate [Triton](pyrefly-triton-examples/README.md),
[Pallas](pyrefly-pallas-examples/README.md), and
[CuTe](pyrefly-cute-examples/README.md) READMEs for the evidence and limitations
of each fixture.

## A useful route through the examples

Start with [Triton vector add](pyrefly-triton-examples/tests/test_vector_add.py):
`[N]` names the extent of each host allocation, while `[Block]` names the
elements handled by one program. A program ID chooses a tile; the offsets
and comparison produce a mask carrying both the allocation bound and tile
shape. Typed loads and stores check that bound against the respective
pointer's declared extent. Triton's `tl.tensor` is a value inside a kernel,
not the entire Torch allocation; the pointer annotation names the allocation
that the program's tile accesses. This does not establish that the launch grid
covers every element or that the comparison protects the *same numerical*
offsets used by the access.

Next compare [Pallas vector add](pyrefly-pallas-examples/tests/test_vector_add.py):
the kernel sees `Ref[[Block]]`, and the `BlockSpec` and `pallas_call` boundary
relate that block to host arrays `[Length]`. The index map and launch grid
take over work that Triton's kernel spells out with pointer arithmetic and a
mask. The type-level relationship does not prove that an arbitrary Python
index-map lambda selects the intended blocks or handles the final partial
block correctly.

Then read the [CuTe Ampere add](pyrefly-cute-examples/tests/test_full_ampere_kernel.py):
the kernel explicitly partitions a host-shaped tensor into CTA and
thread/value views. Shapes and strides belong to *distinct* dimensions of
its `cute.Tensor` contract. Some restricted nested layout shapes are
represented with existing `IntTuple`-based types, without implying that
general layout trees or coordinate provenance are solved.

## Which boundary is actually checked?

An annotation declares a contract. A passing kernel body is evidence for it
only where typed accesses consume the annotated information; otherwise the
annotation might simply be an unchecked promise. For example, changing the
second Triton vector-add input annotation to an unrelated allocation extent
causes its unchanged load to fail because its mask still carries `[N]`.
The [Gluon 2D TMA add](pyrefly-triton-examples/tests/test_gluon_tma_elementwise_add.py)
provides the complementary counterexample: the unchanged load helper checks
the A/B extents, but its body does not check C's *full host extent* against
them. That part of the output contract still depends on a wrapper or a
different checked operation.

There are several boundaries in a complete launch: the host allocation and
its actual shape/strides; an annotated Python wrapper or descriptor; the
kernel parameter; the local view or block; and the typed loads and stores.
Following one symbol through only two of these boundaries does not establish
the rest. [Pallas all-gather](pyrefly-pallas-examples/tests/test_shard_map_boundary.py)
and [all-reduce](pyrefly-pallas-examples/tests/test_shard_map_reduction_boundary.py)
show how a typed `shard_map` can relate global and local dimensions, whereas
the [CuTe JAX launcher](pyrefly-cute-examples/tests/test_full_jax_fused_bias_relu.py)
records a case where a generic higher-order call loses the dimension
relationship to its concrete launcher. These are deliberately different
confidence levels, not interchangeable success counts.

When exploring a fixture, check its source-body identity statement, a
negative test that changes a claimed dimension, and its accepted-gap tests.
Passing a negative test is useful only if the diagnostic comes from the
operation that should enforce the claim. None of the three prototype overlays
proves arbitrary numeric bounds, device synchronization, full grid coverage,
or the dtype/layout rules of the actual accelerator runtime.

## Three ways an array becomes a tile

| Example | Host-to-kernel relationship | What the unchanged body checks | What still needs a contract |
| --- | --- | --- | --- |
| [Triton vector add](pyrefly-triton-examples/tests/test_vector_add.py) | Pointer allocation `[N]` and program tile `[Block]` meet at offsets and mask. | Each masked load/store uses the declared allocation bound and tile. | Actual host storage, correspondence between compared and accessed offsets, and launch coverage. |
| [Pallas vector add](pyrefly-pallas-examples/tests/test_vector_add.py) | `BlockSpec` and `pallas_call` present host `[Length]` as local `Ref[[Block]]`. | Addition and output assignment agree on the local tile. | Numeric index-map values and whether all host blocks are visited. |
| [CuTe RMSNorm weight](pyrefly-cute-examples/tests/test_rmsnorm_pointer_views.py) | A declared pointer `[N]` gets a unit-stride tensor view; a [separate device slice](pyrefly-cute-examples/tests/test_rmsnorm_weight_broadcast.py) turns that view into a zero-row-stride broadcast tile. | The wrapper's view construction agrees with the pointer tag; the device slice checks broadcast width and stride. | Real allocation size, the intervening launch connecting the two slices, and actual kernel bounds. |

These rows intentionally differ in proof strength. A static host adapter can
reject incompatible *declared* arrays without proving that an independently
annotated allocation has the promised physical size, dtype, or strides.
Conversely, a type-correct tile computation alone does not verify the Python
interface if the host-to-tile transformation drops the allocation extent.

## Reading the harder boundary cases

After the vector-add examples, the following fixtures are useful for learning
where each DSL puts its indexing decisions. In each case, read the named
unchanged operation alongside a negative test, then ask whether the rejected
dimension came from that operation or merely from the declared signature.

| Fixture | Where the indexing decision lives | Body-grounded relationship | Declared or unchecked relationship |
| --- | --- | --- | --- |
| [Triton persistent MMA helper](pyrefly-triton-examples/tests/test_gluon_persistent_issue_mma.py) | The consumer indexes two shared-memory rings and passes their tiles to the MMA aggregate. | Both aggregate implementations reject an A/B mismatch in the contraction width `K` at the original `issue_async_mma` call in `issue_mma`. | The upstream aggregate's unparameterized accumulator fields discard output `M,N`; wrong accumulator width is accepted even when A/B agree. The Python launch has not yet been checked in this fixture. |
| [Gluon warp-specialized vector add](pyrefly-triton-examples/tests/test_gluon_warp_specialized_vector_add.py) | Torch A/B/C `[Rows,Cols]` become separate TMA descriptors and shared-memory rings passed to load, compute and store workers. | Changing only the launched kernel's B or C host-width annotation, or B tile-width annotation, produces an error at the original `warp_specialize` call. | Worker presence/order, the concatenated ring prefix's length, dtype, actual bracketed JIT launch, offset bounds and barrier timing remain unproved. The original wrapper's direct typed signature checks shapes but does not type its launch. |
| [Gluon TMA gather](pyrefly-triton-examples/tests/test_gluon_tma_gather.py) | A source descriptor `[XMax,YMax]` and row-offset vector `[BX]` feed an irregular gather into output `[BX,BY]`. | The unchanged kernel checks offset-vector length and descriptor block `[1,BY]` at the original gather; its output store checks `[BX,BY]` and both named strides. | Row-index values, source host bounds, `gather4` layout legality, Y-offset alignment, Torch output-pointer construction and the original wrapper's bracketed launch remain unverified. |
| [Gluon TMA scatter](pyrefly-triton-examples/tests/test_gluon_tma_scatter.py) | A source tile `[BX,BY]` and row offsets `[BX]` address a destination descriptor `[XMax,YMax]` in irregular row order. | The unchanged source load checks both tile axes and strides; the original scatter checks destination tile `[1,BY]`, shared tile `[BX,BY]` and offset count. | Destination host bounds, negative or duplicated row values, `scatter4` layout, Y-offset alignment, dtype and the original bracketed launch are not checked. |
| [Gluon fused gather/scatter matmul](pyrefly-triton-examples/tests/test_gluon_tma_gather_scatter_matmul.py) | X `[M,K]`, W `[K,N]`, two row-index arrays `[M]` and output `[M,N]` flow through gather-load, MMA and scatter workers. | The full unchanged kernel rejects an X row-extent or W K-extent mutation at original helper calls, and an output tile-width mutation at `async_scatter`; the MMA helper checks X/W K contraction. | Output full host N and accumulator result shape remain unproved. The original wrapper's dynamic dtype and JIT launch are untyped; a trial constructor equating the widths of physical block `[1,B]` and shared layout `[BM,B]` failed a mismatched-width probe and was removed. |
| [Gluon tensor-memory copy](pyrefly-triton-examples/tests/test_gluon_tcgen05_copy.py) | A nominal float32 host `[M,N]` tile flows through SMEM `[M,N]` to TMEM `[M,N]` and back to a host output of the same dimensions. | The original load and store reject wrong full extents or stride roles; a separate negative intrinsic call rejects incompatible SMEM/TMEM shapes. | The original copy allocates both tiles from the same `(M,N)`, so no parameter-only copy-site mismatch is possible. Torch pointer dtype, GPU layout legality, barrier ordering and bracketed launch remain unproved. |
| [Pallas indexed add](pyrefly-pallas-examples/tests/test_core_map_indexed_add.py) | A scalar index is prefetched to shared memory and used inside a `BlockSpec` index-map lambda for a half-width input window. | The original pipeline call matches input/output Refs and `[8,128]` callback tiles; a separate host contract tracks `[Rows,Cols]` to `[Rows,Cols // 2]`. | The value of the index can select a window outside the input, and the type checker does not calculate the grid's coverage. The CPU tests illustrate these facts without executing TPU DMA. |
| [Pallas PRNG key](pyrefly-pallas-examples/tests/test_prng_stateless_key.py) | A host Threefry key is explicitly converted before a key Ref reaches the generator. | The unchanged body produces a tile with the output Ref's declared `[8,128]` shape from a semantic Pallas key. | The hardware key format, physical shared-memory placement, output dtype, and actual generated values are not verified. |
| [Marin Pallas short convolution](pyrefly-pallas-examples/tests/test_marin_short_conv_forward.py) | The original `_head_views` concatenates `Width-1` pad rows with a `BlockSeq`-row input head, then a wrapper passes that halo to a block kernel. | The unchanged host construction derives halo `[Batch,BlockSeq+Width-1,Channels]` and matching segment length; the kernel consumes those declared Ref dimensions through lag reads and output tiles. | The checker does not prove that concat order is correct, lag addresses are in bounds, or grid blocks cover the host sequence. |
| [Marin Pallas ragged dot](pyrefly-pallas-examples/tests/test_marin_ragged_dot.py) | A host `[M,K]` matrix and `[Groups,K,N]` weights use group sizes to select rows and grouped output columns. | The unchanged kernel checks tile contraction, both masked loads, and the row/column-masked output store. | The numeric group boundaries may be inconsistent, and the checker does not prove row coverage; the CPU test runs only a host reference because the installed JAX lacks `pl.dot`. |
| [Marin Pallas streaming LSE](pyrefly-pallas-examples/tests/test_marin_streaming_lse.py) | Host `[B,H]` and `[H,V]` tiles feed two `[BB,128]` scratch Refs and a padded `[B,128]` output; selecting column zero exposes `[B]`. | The unchanged kernel checks the contracting hidden dimension, vocabulary-tail mask and recurrence tile widths, and the host projection checks the result rank. | A separate declaration trusts the source helper's shape-preserving result; the checker does not prove numerical recurrence, program ordering, or complete vocabulary coverage. The CPU test is a host reference, not TPU execution. |
| [Pallas layer-norm weight gradients](pyrefly-pallas-examples/tests/test_upstream_layer_norm_weight_grad.py) | Input and upstream gradient `[M,N]` and per-row statistics `[M]` reduce into two independent host gradients `[N]`. | The unchanged kernel combines row/column masks at both loads, reduces over rows and validates both masked output stores; a CPU Pallas interpreter exercises partial rows and columns against NumPy. | The original kernel never reads its weight or bias Ref, so those input widths are signature declarations; statistic provenance, numeric bounds, grid coverage and dtype remain unchecked. |
| [Pallas transposed RHS matmul](pyrefly-pallas-examples/tests/test_fused_rhs_transpose_matmul.py) | A custom adapter swaps logical RHS `[K,N]` to physical `[N,K]`, connects `[BN,BK]` Ref tiles and produces `[M,N]`; its nonsquare CPU interpreter matches NumPy. | The original full kernel body deliberately has an expected error at `dot_general`: Pyrefly merges both contraction axes even with `transpose_rhs: Literal[True]`, so the contraction is **not** validated. | The upstream tutorial wrapper takes `n` from the post-swap RHS, yielding `K` instead of `N` for rectangular inputs; a negative probe exposes this. The custom adapter does not claim to validate that original wrapper. |
| [Pallas Blackwell TMA gather](pyrefly-pallas-examples/tests/test_gpu_tma_gather.py) | Source GMEM `[S,C]` and SMEM indices `[R]` feed a copied SMEM output `[R,C]`. | The unchanged kernel loads indices with the TMA layout and checks index count, memory-space roles, output width and barrier at the original asynchronous copy. | The `BlockSpec` shape tags without explicit block dimensions are interface declarations; index bounds, Blackwell execution and copy completion are not proved. Its CPU test is host-reference-only. |
| [CuTe Flash Attention output partition](pyrefly-cute-examples/tests/test_flash_attention_output_partition.py) | A 4D host output is projected to a 2D head, locally tiled, then passed to a copy thread's destination partition. | The original `local_tile` and `partition_D` operations retain output sequence length, head stride, and destination tile size. | Query/output sequence equality is only a signature promise: the query is not read by this source slice, and no output copy or bounds predicate is checked. |
| [CuTe Blackwell input load](pyrefly-cute-examples/tests/test_blackwell_batched_input_load.py) | A/B host dimensions and batch strides flow through K subtiles, staging slices, and distinct A/B CTA maps. | The unchanged TMA-load calls reject a swapped map or incompatible host K or staging tile. | The disconnected source slices do not prove physical allocation identity, K-index bounds, CTA-map tile factors, layout-atom conformance, or synchronization. |

The same dimension can have different evidence at different boundaries. A
wrong `K` reaches Triton's original MMA call, while a wrong CuTe query length
is rejected before the source slice starts because its parameter annotation
requires agreement with the output. Pallas exposes the opposite sort of
boundary: the half-width output extent is represented statically, but the
runtime index choosing the source window remains a value-level obligation.

Across these DSLs, the reusable pieces are a host-to-tile shape and access
contract, axis-aware index or mask provenance, and a separate placement map.
Pallas constructs local Refs through `pallas_call` and `BlockSpec`; Triton
constructs descriptors, program offsets, shared rings and worker arguments
explicitly. Neither a compatible tile shape nor a correct host signature by
itself proves that runtime coordinates cover the intended host elements.

Marin's [backward short convolution](pyrefly-pallas-examples/tests/test_marin_short_conv_backward.py)
uses the complementary *tail* halo and emits per-block weight-gradient
partials `[Batch * (Seq // BlockSeq), Width, Channels]`. Its unchanged
helper and kernel bodies check tile and output dimensions, and the separate
CPU interpreter test compares values with a scalar reference. A matching
static shape does not prove that the tail chunks were concatenated in the
right order or that every backward lag stays in bounds.

The next CuTe Flash Attention output-copy slice makes this boundary especially
clear. Its boolean predicate buffer is allocated from the destination's
*thread-tile geometry*, not the output allocation's `Query` or `Dim`. Two
allocations with different host extents can therefore have the same predicate
buffer type. The subsequent coordinate comparisons, writes into the buffer,
row guard, and predicated copy cannot be reduced to a proof of safe host writes
by assigning a bounds-aware type to that allocation. Such a proof would need
to track which coordinate produced each mask element, whether every element
was initialized, and which guarded destination is copied. The earlier
`partition_D` fixture checks the 4D-to-tile relationship without claiming
those later value-level properties. A CuTe copy that performs predication
internally can instead expose a typed library contract, but then the mask's
correctness is a trusted runtime-library guarantee.

The [Triton persistent scheduler fixture](pyrefly-triton-examples/tests/test_gluon_persistent_matmul.py)
pinpoints another loss of evidence: the original scheduler stores its tile
coordinates as unparameterized `gl.tensor` fields. Its modulo/division body
cannot preserve a relation between the program ID and `BM`/`BN`, so the
kernel's coordinate arguments reach the load helper as `Unknown`. Even the
apparently more precise `program_id * block` stub has an accepted false
positive: multiplying by an unrelated plain `int` can still pass as a tile
start with a particular block width. Pyrefly currently treats `Int[int]`
as compatible with a symbolic `Int[BM]`, so a stub overload cannot reliably
distinguish a runtime integer from an annotated block extent. This is not a
proven scheduler mapping;
the fixture checks B's tile contraction elsewhere in the unchanged kernel
and records the coordinate-to-output relationship as open.

## Where a general layout system might help

Avik's [CuTe layout type-system sketch](https://docs.google.com/document/d/1vOodinAqhBgWybbVY5oNlX4uMLDqp8hoxHfqfG2Fops/edit?tab=t.0#heading=h.yw6rtyoeqv77)
proposes hierarchical shapes **and** strides, with two distinct judgments:
shape-compatible subtyping for coordinate domains, and exact layout
conformance after `coalesce` normalization for hardware copy/MMA fragments.
It also proposes symbolic divisibility witnesses such as a dimension known
to be `128 * SomeInt`, and a separate composition type for non-affine shared
memory swizzles. These are design proposals, not rules implemented by the
fixed-depth `IntTuple` stub overlays here.

That distinction helps locate future work. A layout normal form could check
whether a CuTe register fragment fits an MMA atom even when host dimensions
are dynamic; a divisibility witness could justify hierarchical tile sizes.
Neither alone proves that a mutable Flash Attention mask was initialized
from the right coordinates, that a Triton scheduler's erased `gl.tensor`
field retains program-ID provenance, or that an original Python JIT launch
consumes the kernel contract. Those are separate data-flow and boundary
questions exposed by the current fixtures.

# Pallas shape-contract spike

The [shared host-boundary note](../kernel-boundary-prototype.md) records the
possible host declaration and runtime checks for both Pallas and Triton. This
prototype tests kernel-body semantics; its existing host adapters are examples,
not generated wrappers or a general `pallas_call` type implementation.

## Modeling approach

Start with [vector add](tests/test_vector_add.py) and the
[Pallas stub overlay](pallas-stubs/jax/experimental/pallas/__init__.pyi). Pallas
kernels are ordinary Python functions called via `pallas_call`, which supplies
a launch grid, output allocation shape, and `BlockSpec`s. Each `BlockSpec`
provides a block shape and an index-map function that selects which part of a
host array a program sees. The kernel receives `Ref`s to these blocks;
`InRef[[Block]]` and `OutRef[[Block]]` describe per-program views, while
`jax.Array[[Length]]` and `ShapeDtypeStruct` describe the host allocations.
The example ties `Length`, `Block`, grid, specs, and Ref operations together
without writing Triton-style pointer arithmetic or an edge mask. Its grid
uses ceiling division: Pallas pads partial reads and drops out-of-bounds
writes, so elementwise vector add does not require a divisible length. These
types do not prove the index-map values select the correct blocks.

The stub overlay models the host-to-Ref relationship for selected
`pallas_call` patterns, block indexing, and shape-carrying JAX operations.
Later examples test squeezed axes, padded blocks, masked loads, and attention
matrix dimensions; see [masked softmax](tests/test_masked_softmax.py) and
[attention backward](tests/test_gpu_mha_backward.py). Tests combine expected
type errors for wrong relationships with JAX CPU-interpreter checks where
available. Index-map *rank* can be checked, but an arbitrary function that
sends every program to the same block can still type-check: coverage and
numerical index validity need separate reasoning or runtime enforcement. The
types also cannot prove all value-shape flows inside the advanced kernels;
the detailed reports below distinguish verified operations from trusted
signatures and unknowns. Consult the [boundary note](../kernel-boundary-prototype.md)
for the proposed host-side declaration, which is not implemented.

## Kernel observations and remaining gaps

The full deprecated JAX GPU `mha_backward_kernel` follows the preprocessing
example. Its executable AST, including both scan callbacks, matches
`jax/experimental/pallas/ops/gpu/attention.py` after removing only semantic
parameter annotations. The host adapter accepts query, forward output and
upstream derivative `[B,Q,H,D]`; key/value `[B,K,H,D]`; optional segment IDs
`[B,K]`; and the forward log-sum-exp and preprocessing delta `[B,H,Q]`.
Eight input specs expose entire Q/K sequences and their padded head width `P`
while squeezing batch/head axes. Three output specs expose `[BQ_dq,P]` for
query gradients and `[BKV_dkv,P]` for each key/value gradient, allocated as
`[B,Q,H,D]` and twice `[B,K,H,D]`. The `(B,H,cdiv(K,BKV_dkv))` grid serves
both scans: a runtime check requires `Q/BQ_dq == K/BKV_dkv`, and all four
independent scan widths must divide their corresponding sequence length.
Runtime also checks positive block sizes, power-of-two `P >= D`, and `Q <= K`
when causal or segmented. The narrow callable signature relates all eight
input and three output shapes/specs; static diagnostics reject wrong host
axes, input/output specs, grid axis, and each output allocation separately.
Original dot contraction and masked gradient-store sites have independent
wrong-shape probes. The CPU interpreter runs both noncausal `Q=32,K=64` and
causal segmented `Q=K=32`, all three gradients agreeing with autodiff of an
independent attention computation to float16 tolerance.

Seven *expected type errors in the unchanged kernel* are an important limit:
three `jnp.zeros([block,padded], ...)` calls use a Python list, whose
homogeneous inferred type loses the order of the two symbolic dimensions.
The two key/value accumulator stores consequently raise four secondary
shape diagnostics. The query accumulator/store also lacks end-to-end shape
proof despite no extra diagnostic, because its list-shaped initializer
propagates an unknown type. An overload claiming that an arbitrary list
retains positional row/column brands would be unsound; the independent
dot/store mutation probes demonstrate narrower contracts, not a proof of
these original accumulator flows. `segment_mask` uses a trusted rank-aware
signature, as in the forward example, rather than a check of its variable
rebindings. The host does not statically prove index-map values or coverage,
causal/segment correctness, reciprocal normalizers from the matching forward
pass, floating-point precision, or GPU execution. The CPU test supplies the
forward output and log-sum-exp from the upstream Pallas forward kernel and
computes delta separately; it does not check their static provenance.

The GPU attention example's complete backward-preprocessing kernel
`_preprocess_backward_kernel` keeps its original executable body, adding only
semantic parameter annotations. It computes the rowwise inner product of the
attention output and its upstream derivative before the full backward pass.
Unlike the existing forward-attention adapter, its host input head dimension
`D` need not be a power of two: `[Batch,Queries,Heads,D]` arrays map through
`BlockSpec((None,BQ,None,PaddedD),...)` to `[BQ,PaddedD]` Refs. A `[1,PaddedD]`
mask prevents loads of the padded head positions. The kernel reduces the head
axis to `[BQ]`; a squeezed output spec maps it to the permuted host allocation
`[Batch,Heads,Queries]`. The narrow `pallas_call` overload relates the
distinct host dimension `D` and Ref dimension `PaddedD` to both input specs,
the output spec and the three-dimensional grid. A runtime check requires
`PaddedD` to be a power of two at least as large as `D`, and `Queries` to be
divisible by positive `BQ`. Eleven expected errors reject wrong host query,
head and feature axes, padded Ref and load-mask dimensions, the wrong
reduction axis, output Ref width, mismatched padded input or squeezed output
specs, and independently wrong output head and query allocations. The Pallas
CPU interpreter runs `D=3`, `PaddedD=4`, two batches and heads and four queries
in two-query blocks; its values and permutation match an independent NumPy
inner product.

The static contract does not prove that `PaddedD` is the *smallest* covering
power of two, that masks correspond to valid head indices, or that index-map
callbacks choose the right batch, head and query block. Positive query block
size, query divisibility and padding are enforced only at runtime. This
adapter models
the upstream preprocessing call, not the subsequent full attention-backward
kernel, and does not prove GPU execution, dtype or numerical precision.

The same upstream GPU RMSNorm module also supplies the complete input-gradient
kernel `rms_norm_backward_kernel_dx`, unchanged except for semantic parameter
annotations. Together with the forward and parameter-gradient examples, this
covers all three RMSNorm kernel bodies. A separate single-row adapter accepts
`x`, `weight`, `bias` and upstream derivative `dout` as `[Features]`, plus a
scalar reciprocal standard deviation. `grid=()` and no `BlockSpec` expose the
whole row and scalar to the kernel, whose masked loops internally process
feature blocks; its output allocation is `[Features]`. A narrow `pallas_call`
overload binds the five host inputs to these full-row Refs and the output.
Nine expected errors reject mismatched host and Ref ranks, a non-scalar Ref
read, a foreign-width derivative mask, a wrong output tile and a wrong output
allocation. The CPU interpreter evaluates five features in four-wide blocks;
its input gradient agrees with both an independent NumPy derivative and JAX
autodiff of the corresponding RMSNorm operation.

`bias_ref` is declared but never read by this kernel; `eps` is also unused by
the kernel because the host supplies the reciprocal standard deviation.
Types do not prove that statistic came from the same input, that each masked
feature is visited exactly once, that `block_size` is positive, or that GPU
execution and floating-point behavior match this CPU case. The adapter models
the upstream inner row call, not the wrapper's batch vmap or launch heuristics.

The deprecated upstream GPU RMSNorm parameter-gradient kernel
`rms_norm_backward_kernel_dw_db` is complete and executable, with only
semantic parameter annotations added. It pairs with the existing RMSNorm
forward example. The host adapter takes input and upstream derivative matrices
`[Rows,Features]`, shared weight and bias vectors `[Features]`, and reciprocal
standard deviations `[Rows]`. A one-dimensional feature grid
`(cdiv(Features,ColBlock),)` gives each program the entire matrix and row
statistics without `BlockSpec`; the kernel internally masks row and feature
blocks, reduces over rows, and writes two `[Features]` output allocations.
The narrow `pallas_call` overload reuses the existing matrix/vector Ref roles
from the layer-norm parameter-gradient example because both have this same
layout; the kernel's arithmetic still computes RMS normalization, with no
mean subtraction. Ten expected diagnostics reject mismatched weight,
derivative and statistic axes, two incompatible matrix masks, a reduction
over features instead of rows, two mismatched stores, and each output
allocation independently. The actual CPU Pallas interpreter uses three rows,
five features, and 2-by-4 inner blocks so both masks handle partial tiles;
both parameter gradients match independent NumPy sums.

`weight_ref` and `bias_ref` are passed through the unchanged kernel but are
not read; their axes are declarations only. `eps` is also unused by the body.
The host wrapper models the upstream inner call after flattening batch axes,
not the original wrapper's batch reshaping, GPU launch heuristics or input
gradient kernel. Neither the source of the supplied reciprocal standard
deviations nor the numerical grid coverage and mask-index values are proved
statically. Positive block sizes, dtype precision and GPU execution remain
unchecked.

The deprecated JAX GPU decode-attention example contributes the complete
`attn_forward_kernel` and its nested compute, reduction and dot callbacks
from `jax/experimental/pallas/ops/gpu/decode_attention.py`. Its executable
AST matches upstream after removing only semantic parameter annotations.
The separate host adapter accepts query `[Heads,D]` and flat K/V `[K,D]`,
checks `K == Splits*SplitKeys` and a whole-block `SplitKeys` at runtime, then
reshapes both pools to `[Splits,SplitKeys,D]`. An optional pair of scalar
arrays gives the start/end bounds. A `(cdiv(Heads,BH),Splits)` grid presents
`[BH,D]` query and output Refs, a `[SplitKeys,D]` KV Ref with its split axis
squeezed, and two `[BH]` residual Refs. `BlockSpec` declarations connect
these roles to three output allocations: partial unnormalized attention
`[Splits,Heads,D]` and two `[Splits,Heads]` normalization vectors. Scalar
bounds use rank-zero specs when present; `None` represents omitted bounds.

Ten expected errors reject wrong flat K/V dimensions, incompatible query/KV
or output Refs, a non-scalar bound, wrong query mask width, mismatched dot
contraction, wrong residual write width, wrong split-residual allocation and
an unsqueezed KV spec. A rank-correct counterexample sends every program to
KV split zero; static types cannot prove index-map values or coverage. The
actual CPU Pallas interpreter exercises a partial last query-head block,
two KV splits, and start/end masking that excludes tokens at both ends. All
three outputs for each split match independent NumPy attention reductions;
combining the split outputs also matches a separately computed full softmax.

This adapter stops at the kernel's split outputs: the NumPy combination test
does not statically check the upstream wrapper's final split reduction.
The `K == Splits*SplitKeys`, minimum block size and split-key divisibility
preconditions are runtime checks, not type proofs. The query mask retains
its `[BH]` extent, but the subtraction from `num_heads` loses the symbolic
`Heads` brand, so types do not verify that the mask values correspond to
valid heads. Start/end values, KV index ordering, whether every head and
split is visited, floating-point precision, dtype requirements and actual
GPU execution remain unproved. The residual store stub admits an optional
mask because Pyrefly merges a provably non-`None` local mask with the
unreachable `else None` arm; it does not prove the mask is present at runtime.

The deprecated JAX GPU multi-head-attention forward example supplies the
complete `mha_forward_kernel` and `segment_mask` executable ASTs from
`jax/experimental/pallas/ops/gpu/attention.py`. Only semantic parameter and
return annotations differ. A separate host adapter accepts `q` of shape
`[Batch,Q,Heads,D]`, `k` and `v` of `[Batch,KV,Heads,D]`, and optional segment
IDs `[Batch,KV]`. It allocates output `[Batch,Q,Heads,D]` plus log-sum-exp
`[Batch,Heads,Q]` and launches `(cdiv(Q,BQ),Batch,Heads)`. `BlockSpec` squeezes
batch and head singleton axes: each query/output Ref sees `[BQ,D]`, each
key/value Ref sees the full `[KV,D]`, and the residual Ref sees `[BQ]`.
The corresponding source maps select `(batch,query_block,head,0)` for q/output,
`(batch,0,head,0)` for k/v, `(batch,0)` for segment IDs, and
`(batch,head,query_block)` for the residual. The specialized `pallas_call`
signature links the specs, grid and all host dimensions; these are declared
roles, not proofs that the map callbacks return those coordinates.

The unchanged body checks `[BQ,D] @ [D,BK] -> [BQ,BK]` and its reduction,
mask, value, residual and output-store dimensions. Ten expected negative
diagnostics reject host head/feature/segment mismatches, wrong contraction
axis, mismatched head and attention masks, wrong output tile, unsqueezed input
spec and separately wrong output and residual allocations. A passing
counterexample maps every query program to query block zero. The actual CPU
Pallas interpreter runs both causal segmented attention over two batches and
heads and noncausal attention without segments; outputs and residuals match
independent NumPy softmax and log-sum-exp calculations.

The executable `segment_mask` helper runs unmodified, but its rank-one to
rank-two variable rebindings are not checked by Pyrefly: the kernel uses a
separate, trusted declaration that ties its query and key lengths to the
returned mask. The adapter always allocates the residual and does not check
the no-residual call branch. Unlike upstream's wrapper, this adapter accepts
only an already power-of-two head dimension, validated at runtime, rather
than padding an arbitrary `D`. It also rejects nondivisible query or KV
lengths at runtime, as the upstream wrapper does, and rejects segment IDs
that cannot cover every query position; a partial KV block can
produce non-finite CPU output without this precondition. Types do not prove
numerical index-map selection, query/key coverage, causal ordering, actual
segment-ID equality, dtype and precision, or GPU behavior. The static
overlay tracks shapes but not the example's float16 input requirement.

The same deprecated upstream layer-norm module supplies the complete
`layer_norm_backward_kernel_dx` input-gradient body, with only semantic
parameter annotations added. A separate single-row host adapter accepts
`x`, `weight`, `bias`, and upstream derivative `dout` as `[Features]`, plus
zero-dimensional mean and reciprocal standard deviation. `grid=()` gives
each Ref the full row without an explicit `BlockSpec`; the sole output
allocation is `[Features]`. The scalar input Ref's `[...]` read is modeled
as `ScalarFloat`, matching the zero-dimensional runtime value used by both
unmodified masked gradient passes. An expected error confirms that a vector
Ref cannot replace the scalar read. Ten expected errors reject that read,
wrong host and Ref axes, a foreign-width load mask, a wrong gradient tile,
and wrong output allocation. The CPU Pallas interpreter executes a partial
second feature block and checks the actual gradient against the independent
NumPy layer-norm derivative.

The source receives `bias_ref` but never reads it, and does not use `eps`:
their relationship to the computation is a declared interface, not a body
check. Types do not establish that supplied mean and reciprocal standard
deviation came from the same `x`, that masked loops cover each feature
exactly once, that `block_size` is positive, or that this derivative matches
the true numerical derivative outside the tested case. The original host
wrapper reshapes and vmaps batch rows; this adapter covers its inner
single-row call only. GPU execution, dtypes, and rounding are not proved.

The deprecated upstream GPU layer-norm forward kernel in
`jax/experimental/pallas/ops/gpu/layer_norm.py` runs unchanged apart from
semantic parameter annotations, including both masked feature-reduction
callbacks and its masked output loop. The separate host adapter receives
`x`, `weight`, and `bias` of shape `[Features]`, launches `grid=()` with
no `BlockSpec`, and allocates `[Features]` output plus two scalar outputs
`[]` for mean and reciprocal standard deviation. The empty grid means
each Ref sees the full row; `block_size` controls the kernel's internal
masked iterations, not host-to-program tiling. A dedicated `pallas_call`
signature ties all three inputs, the full-row Ref, and all three distinct
allocations together. Eleven expected errors reject input and output axis
mismatches, incorrect scalar Ref ranks, foreign-length load/store masks,
wrong store-tile width, and each wrong output allocation independently.
The actual Pallas CPU interpreter computes a five-element row with a partial
second four-element block; all three outputs match independent NumPy
statistics and normalization.

Types do not prove that the masks identify exactly the valid feature indices,
that `block_size` is positive, that the loop visits each element exactly once,
or that `eps` and the reductions compute the intended values. The runtime
case checks one numeric configuration only. The host adapter deliberately
does not model upstream's doubly-vmapped batch wrapper or its GPU launch
heuristics; GPU execution, dtype and floating-point precision are not
statically verified. The kernel permits either scalar output Ref to be
`None`; the typed adapter supplies both, without claiming every possible
direct call writes both outputs.

The GPU paged-attention fixture copies the complete `paged_attention_kernel`
and nested callbacks from
`jax/jax/experimental/pallas/ops/gpu/paged_attention.py`.
The executable AST matches upstream after removing only parameter annotations.
A separate unquantized, no-length, single-partition host adapter accepts query
`[Heads,D]`, key/value page pools `[TotalPages,PageSize,D]`, and a page table
`[TablePages]`; its three output allocations have shapes `[Heads,D]`,
`[Heads]`, and `[Heads]`. The original body reads page identifiers from the
table, gathers both page pools with those identifiers, reshapes each result
to `[PageBlock*PageSize,D]`, contracts the shared `D` axis and token axis,
and checks the `[Heads,PageBlock*PageSize]` mask and `[Heads,D]` output store.
The adapter divides the kernel's unnormalized output by its returned
normalization term. An actual CPU Pallas interpreter test compares this
adapter to an independent NumPy softmax over pages visited in nontrivial
order `[2,0]`. Seven negative checks reject wrong host K/V dimensions,
wrong gathered page-block size, both dot-contraction axes, wrong mask width,
and wrong output width.

This adapter models neither the original wrapper's grouped-query heads and
head padding nor its `k_splits` reduction; it launches just one query block
and partition, using the public `pl.pallas_call` CPU interpreter. The
optional quantization, soft cap, and sequence-length branches remain in the
original checked body but are not exercised in the CPU example. Shape types
do not establish that page IDs lie in `[0,TotalPages)`, page-table length is
divisible by the compute-page block, the loop covers exactly all pages, or
that masks correspond to numeric length values. Repeated page IDs are
accepted; index dtype, floating-point precision and GPU execution are not
verified. The tuple of two residual stores is tied to the separate host
output allocations, while the general `*residual_refs` signature does not
itself require exactly two outputs at every possible call.

The Blackwell Mosaic GPU TMA gather fixture copies the complete three-statement
kernel from `jax/docs/pallas/gpu/reference.md`. Its
executable AST is unchanged after adding only semantic parameter types. A
separate host adapter uses the public `pl.pallas_call`; the guide's
`self.pallas_call` belongs to its test harness, while `plgpu.kernel` has a
different argument convention. The adapter connects the source `[S,C]`,
SMEM indices `[R]`, and SMEM output `[R,C]` to its `ShapeDtypeStruct`,
GMEM/SMEM `BlockSpec` roles, and barrier scratch allocation. The original
body loads indices in `Layout.TMA_INDICES`, produces a GMEM gather view
`[R,C]`, and sends it to the SMEM output through `copy_gmem_to_smem` before
`barrier_wait`. Negative checks reject wrong index layout, incompatible
memory-space specs, SMEM-as-GMEM source, GMEM-as-SMEM destination, wrong
output row/column extent, wrong barrier role, output allocation, and index
extent at host launch. Unlike the SparseCore gather, this operation checks
the typed transition between GPU memory spaces and an asynchronous channel.

Only an independent host gather reference runs on CPU. Executing this kernel
requires Blackwell TMA hardware; the stub does not prove index values lie in
`[0,S)`, TMA-index encoding, SMEM transform/alignment constraints (the
example adapter passes empty transforms), proper DMA completion, or hardware
support. Host `BlockSpec` shape tags on specs without a `block_shape`
argument are declarations, not derived from a runtime spec constructor;
the call ties those declared roles to the actual host arrays and unchanged
kernel body. Repeated indices are valid for gathering and do not establish
any numerical bounds guarantee.

The TPU matmul guide's fused-RHS-transpose example contributes the complete
`matmul_kernel` from `jax/docs/pallas/tpu/matmul.md`.
Only parameter annotations are added; the executable AST is unchanged. The
separate adapter accepts logical `x: [M,K]`, `y: [K,N]`, swaps the latter to
physical `[N,K]`, passes `[BM,BK]` and `[BN,BK]` Refs into the kernel, and
allocates `[M,N]`. A transpose-specific `PrefetchScalarGridSpec` marker checks
that the RHS `BlockSpec` is `[BN,BK]`, not `[BK,BN]`, and connects the
three-axis grid with host dimensions. Negative controls reject mismatched
host K/N, an unswapped RHS at launch, and the wrong RHS tile orientation.
On CPU, the actual Pallas interpreter computes `[4,6] @ [6,8]` with partial
K tiles and agrees with the host matmul reference.

This is a **partial** static fixture. Precise `dot_general` stubs reject
using the ordinary RHS contraction axis with the transposed Ref, but the
unchanged kernel body still has an expected `dot_general` error: Pyrefly
marks `else` unreachable for `transpose_rhs: Literal[True]` yet merges its
axis tuple with the true branch before solving the call. Consequently the
body's contraction and downstream accumulator/store are not proved. Making
the stub accept both axis tuples would conceal that gap. The guide's own
wrapper also contains a distinct host bug for nonsquare logical `[K,N]`:
it calls `y.swapaxes(0,1)` and then reads `_, n = y.shape`, obtaining `K`
rather than `N`; a negative shape assertion pins that exact expression.
Our custom adapter explicitly takes `n: Int[N]` and is **not** claimed to
validate the guide's original wrapper. Grid coverage, scratch initialization
ordering, numerical divisibility, actual memory orientation, and dtype
precision are not proved. Type-driven pruning of the unreachable branch's
Phi arm would let the precise contraction overload type-check the body.

JAX's deprecated GPU layer-norm example
`jax/experimental/pallas/ops/gpu/layer_norm.py` supplies the complete
`layer_norm_backward_kernel_dw_db` body and its nested reduction callback.
Both executable ASTs match the upstream source after removing only semantic
parameter annotations. This is a prototype of the example, not a claim of
support for its deprecated public API. A separate host adapter relates
`x` and `dout` `[M,N]`, row statistics `[M]`, weight and bias `[N]`,
the feature-grid `cdiv(N,BN)`, and two separately allocated `[N]` gradients.
Inside the original body, row and column masks must agree with both matrix
loads, row statistic loads have extent `M`, and the row reductions produce
`[BN]` values accepted by each `[N]` masked output store. Negative controls
reject incompatible host inputs, either output allocation independently,
both matrix-mask bounds, and either wrong output store. An actual Pallas CPU
interpreter run exercises partial `2x4` tiles for a `3x5` input and compares
both resulting gradients with an independent NumPy row-reduction reference.
Reducing over columns instead of rows also fails the masked output store.

The source does not read its weight or bias Refs: their `[N]` input shape is
an interface declaration tied to output allocation, not validated by an
in-kernel read. Types do not establish correct numerical row/column indices,
grid coverage, row-loop bounds, non-overlapping stores, mask truth values,
or dtype and precision semantics. In particular, `eps` is unused by the
source, and the checker cannot establish that the input row statistics are
the actual mean and reciprocal standard deviation of `x`.

Marin's TPU streaming log-sum-exp fixture copies the complete
`linear_softmax_lse_forward_fori_pallas_kernel` body from
`marin/lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_tpu.py`.
The kernel and executable `_apply_logit_soft_cap` helper retain their upstream
ASTs after stripping parameter annotations. The helper's shape-preserving
return is declared separately for static checking because the source's
`jax.Array` return annotation does not express a register tile; its body
is not proved by that declaration. A one-core host adapter relates input
`x: [B,H]`, weights `[H,V]`, input tiles `[BB,H]` and `[H,VB]`, and a
four-axis grid `(1,cdiv(B,BB),cdiv(V,VB),1)`. The kernel maintains two
`VMEM[[BB,128]]` scratch tiles and writes a `[BB,128]` lane tile into
the host `[B,128]` allocation, whose lane-zero projection exposes `[B]`.
The original wrapper also supports multiple tensor cores; this adapter
models only its single-core branch.

Within the unchanged body, the typed `dot_general` checks the shared
`H` axis and yields `[BB,VC]`, the vocabulary-tail mask checks the `VC`
tile, and the nested loop carries two `[BB,128]` accumulator tiles through
the final scratch and output stores. Expected errors reject wrong host and
kernel hidden axes, wrong allocation lane width, wrong scratch batch rows
or store width, a wrong vocabulary-mask block, an output store width, and
confusing the lane-buffer rank with the projected API rank. The CPU test
validates **only an independent host log-sum-exp reference**; no TPU exists
locally, so it does not execute this TPU kernel.

Numerical alignment (128 lanes), batch/vocabulary divisibility, the number
of vocabulary subblocks, scratch initialization before reads, final-V-block
ordering, and the actual VMEM placement are not proved. In particular,
the `jnp.tile` stub records broadcasting by row but cannot prove that
`128 * repeats == VC`, and the accepted-gap control shows that a weight
slice's dynamically selected remaining width is not checked. The labels,
external loss weights, and host-level fused cross-entropy calculation are
outside this specific LSE kernel boundary; dtype and precision are also
described but not statically verified.

The Marin ragged-dot fixture copies the complete
`_triton_ragged_dot_kernel` from
`marin/lib/haliax/src/haliax/nn/ragged_dot.py`.
Its executable AST matches upstream after stripping only parameter
annotations, including the nested reduction callback. The typed host
adapter presents an unblocked lhs `[M,K]`, per-group rhs `[G,K,N]`, group
sizes `[G]`, output `[M,N]`, and scalar lower/upper group bounds obtained
from `jnp.cumulative_sum(group_sizes, include_initial=True)`. `BlockSpec`
connects rhs `[K,BN]`, both scalar bounds, and output `[M,BN]` to their
Ref roles and a grid of `(cdiv(M,BM), cdiv(N,BN), G)`.

Inside the unchanged body, both contracting-axis loads require a mask of
`[BK,K]` with the corresponding row/column broadcasting direction; `pl.dot`
checks `[BM,BK] @ [BK,BN]`, and the row/column output masks tie back to
`M`, `N`, and `BN`. Eight negative checks reject wrong host `K` and group
count, wrong allocated output `N`, wrong load masks, wrong output masks, and
a mismatched dot contraction. The CPU test evaluates group prefixes and an
independent grouped matmul reference, **not this kernel**: installed JAX
0.11.2 lacks the source's `pl.dot`, so this Pallas-Triton kernel is not
interpretable in this environment.

The checker cannot prove group sizes are nonnegative or sum to `M`, lower
and upper bounds are ordered, per-group scheduling covers the right rows,
block sizes or `K`/`N` partial tails are numerically safe, or index maps
select the intended group. The stub permits either row-only or combined
row-and-column output masks because the original branches on `n % BN`;
types do not prove the condition matches the actual masked-store choice.
The host adapter carries shape, not dtype or backend availability.

The Marin short-convolution backward fixture copies `_dx_body`, `_dw_body`,
`_bwd_kernel`, and `_tail_views` from
`marin/lib/levanter/src/levanter/kernels/pallas/short_conv/pallas_gpu.py`.
The four full executable ASTs match the source after stripping parameter
annotations. It shares the forward fixture's unchanged `_head_views` and
arithmetic helpers. Host `x`, `dy`, and segments share `[B,S,C]` / `[B,S]`;
weights have `[W,C]`; head and tail inputs have `[B,BS+W-1,C]` and
`[B,BS+W-1]`. Two output allocations differ: `dx` has `[B,S,C]`, while
per-block fp32 `dw_partials` has `[B*(S//BS),W,C]`. The second output's
block map indexes `(b * (S//BS) + si, 0, ci)`. The Ref annotations check
the sliced `dy` tile, segment width, `dx` write, and width-one per-tap
partial write. Expected-error controls reject mismatched host `dy`/segment
sizes, the partial/output write widths, a segment Ref mismatch, wrong tail
padding extent, and wrong partial width or block count. The 2-CPU interpreter compares both
outputs to independent scalar loop references over a segment-boundary case.

Numerical base/lag bounds, segment mask meaning, block divisibility,
`S >= BS`, the sign of a tail slice's start, tail chunk ordering, and whether
the partial block map is injective and covers every block remain unchecked.
The general negative-index slice stub models the expected `-BS` path but
would also type a positive start incorrectly; the accepted-gap test shows
reversing tail chunks preserves dimensions. The host callback is explicitly
typed to validate its argument roles and shapes; the original wrapper's
`functools.partial` binding is not thereby verified. The host declares fp32
for the partial allocation, but `Array`/`Tile` currently track shape rather
than dtype, so this is not a type-level proof of fp32 stores.

The vector-add fixture copies `add_kernel` from
`jax/docs/pallas/design/design.md`; the matmul fixture
copies `matmul_kernel` from `jax/docs/pallas/quickstart.md`.
The squeezed-axis fixture reuses the vector-add kernel body with the
`BlockSpec((None, block), lambda i, j: (i, j))` layout documented in
`jax/docs/pallas/grid_blockspec.md`.
The iota fixture copies `iota_kernel` from
`jax/docs/pallas/quickstart.md`.
The sliced-add fixture copies `add_sliced_kernel` from the same quickstart.
The full-matrix addition fixture copies `add_matrices_kernel` from
`jax/docs/pallas/pipelining.md`.
The activation matmul fixture copies the second `matmul_kernel` from
`jax/docs/pallas/quickstart.md`.
The masked softmax fixture copies `_vmappable_softmax_kernel` from
`jax/jax/experimental/pallas/ops/gpu/softmax.py`.
The scratch-backed matmul fixture copies the second `matmul_kernel` from
`jax/docs/pallas/tpu/matmul.md`.
The manual-copy fixture copies `hbm_vmem_kernel` from
`jax/docs/pallas/tpu/pipelining.md`.
The dynamic-block copy and megacore-add fixtures copy their respective
kernels from the same TPU pipelining guide.
The remote-permute fixture copies `right_permute_kernel` from
`jax/docs/pallas/tpu/distributed.md`.
The all-gather fixture copies `all_gather_kernel` from the same guide.
The all-reduce fixture copies `all_reduce_kernel` and its `local_barrier`
helper from that guide.
The reduce-scatter fixture copies `reduce_scatter_kernel` from the same guide;
it uses the same barrier helper and the guide's `mod` and `signal` helpers.
The SparseCore gather fixture copies `kernel` and its nested `body` callback
from `jax/docs/pallas/tpu/sparsecore.md`.
The SparseCore scatter fixture copies the following `kernel` and nested `body`
callback from the same guide.
The packed-bfloat16 gather fixture copies `gather_bf16_packed`'s `kernel`
and its nested `body` from the same guide.
The Hopper GPU matmul fixture copies the nested `kernel` and `pipeline_step`
from `jax/docs/pallas/gpu/pipelining.md`.
The scalar-prefetch indexed-add fixture copies `indexed_add_one_kernel` and
`add_one_body` from `jax/docs/pallas/tpu/core_map.md`.
Their semantic `pl.InRef[...]` / `pl.OutRef[...]` annotations are static-only;
`from __future__ import annotations` keeps them unevaluated. Most kernels run
on CPU through `interpret=True` with JAX 0.11.2; the dynamic TPU pipeline
and distributed remote-DMA kernels cannot be interpreted on CPU.

The first fixture explores one-dimensional blocked *identity* maps. The kernel sees
Refs of `[Block]` elements, while the host supplies JAX arrays of `[Length]`.
`BlockSpec[[Block]]` describes the slice size, and `GridSize[Length, Block]`
connects the launch grid to `cdiv(length, block)` and the output shape.
Wrong host-array lengths,
output shape, grid block identity, index-map rank, and kernel write-tile shape
are expected to produce Pyrefly errors. These are static marker types, not
real JAX APIs; `GridSize` and parameterized `BlockSpec` do not exist at runtime.

The blocked matmul uses a two-dimensional grid. Its input Refs have shapes
`[RowBlock, Inner]` and `[Inner, ColBlock]`, and its output Ref has shape
`[RowBlock, ColBlock]`. The three expected index maps are `(i, 0)`, `(0, j)`,
and `(i, j)`. Pyrefly validates the shared `Inner` dimension, the tile
shapes, index-map ranks, and both grid dimensions against the output shape.
A CPU interpreter test checks `[4, 3] @ [3, 6]` on a `(2, 2)` grid.

With a squeezed row axis, host inputs and output are `[Rows, Cols]` but the
kernel receives rank-one `[Block]` Refs. Pallas interprets `None` as a
single-element row block and removes that dimension before executing the
kernel. The two-dimensional index map still returns `(row, col_block)`;
the grid is `(Rows, cdiv(Cols, Block))`. A second `BlockSpec` marker,
`BlockSpec[[Block], Literal[True]]`, identifies this specific host-block
layout separately from the `[Block]` Ref shape. Ordinary one-dimensional
`BlockSpec[[Block]]` values cannot stand in for this two-dimensional layout.
The runtime test checks three rows and a partial final column block.

Iota exercises a different boundary: without an explicit `BlockSpec`, each
program receives a Ref for the **entire** `[Length]` output. The kernel reads
`program_id(0)` and writes one scalar element at that index. The specialized
`pallas_call` overload relates its output allocation `[Length]` to a
one-dimensional grid `(Length,)`, while the `OutRef` scalar-write overload
requires an axis-zero `ProgramId`. Negative tests reject a mismatched grid
length, a rank-two output, an axis-one program id, and using a scalar as a
whole-tile value. A CPU interpreter test verifies all eight output elements.
The `ProgramId` marker records the index's origin, but the type checker does
not prove numerical bounds, complete coverage, or whether different programs
write to distinct elements. This overload handles only a one-dimensional unblocked
output; an axis-one program id can legitimately index an output Ref in other
Pallas grids.

Sliced addition illustrates compositional `Ref.at` views. Each input Ref
has `[2 * Half]` elements; a half-size slice produces a `[Half]` input Ref.
The `[4 * Half]` output Ref is sliced twice before each assignment, so all
four writes receive `[Half]` tiles. The body checks without modifications:
the stub rejects splitting at the full input length or using an undersized
output Ref. The host runs one program with full-input/full-output
`BlockSpec`s (`lambda i: (0,)`). The wrapper's input array shape, input
block size, and input size argument share a type variable, and a wrong
input or block size is rejected. The output shape and block size share
another variable, and the `BlockSpec` checks the index-map's return rank.
An output map `(i + 1,)` still type-checks even though the only program
would address an unallocated block.

The central host boundary is **not** yet verified for this kernel: Pyrefly
cannot instantiate the generic callback through `2 * Half` and `4 * Half`
at `pallas_call`. The fixture deliberately expects that diagnostic. It
independently checks the body and the host's declared dimensions, but the
host `OutputLength` remains unrelated to the input `InputLength` and the
checker cannot prove the allocated output has twice as many elements as an
input. The runtime interpreter test checks this claim for a length-four
input and a length-eight output only. Checking or constraining
higher-order callbacks with symbolic dimension arithmetic would close this
specific gap; we should not treat a permissive wrapper overload as proof.
Putting `[2 * Half]` inputs and a `[4 * Half]` output directly on a generic
host wrapper does not currently work around it: Pyrefly cannot infer `Half`
from a caller's symbolic `[2 * Half]` input, so even a valid symbolic call
fails. This is a first-order inference limitation in addition to the
`pallas_call` callback limitation.
The method overloads assume half-slices of even-length Refs, not arbitrary
Python slices or odd-length inputs. We also do not prove the four writes
cover distinct regions in their required order.

The pipelining guide's full-matrix addition uses unblocked rank-two Refs.
`ShapeDtypeStruct.like(x)` and the no-grid `pallas_call` overload tie
the output allocation, both `[Rows, Cols]` host inputs, and the three
whole-matrix Refs together. Two-axis Ref reads yield matching tiles; the
output store checks the resulting tile's rows and columns. Negative controls
reject either wrong input axis, an incompatible output allocation, and a
wrong-width output Ref. A CPU interpreter test checks a `[3, 4]` addition.
Pallas's TPU pipelining guide describes these Refs as SRAM and the loaded
values as registers, but our `InRef`/`OutRef` and `Tile` markers do not
track physical memory space, data transfers, capacity, or aliasing. This
unblocked host/Ref equality does not address grid-to-block mapping or
partial-tile padding.

The fused-activation matmul retains the quickstart kernel's single-statement
body. The activation must take and return a `[RowBlock, ColBlock]` tile,
which the `[RowBlock, Inner] @ [Inner, ColBlock]` expression supplies and
the output Ref stores. The host wrapper shares symbolic input/output axes
and `BlockSpec` tiles with its ordinary matmul counterpart; negative
controls reject a wrong contraction axis and an activation with a different
output tile, both in direct kernel calls and at the host wrapper boundary.
The CPU interpreter checks `relu(x @ y)` over four output tiles.
JAX's example binds the activation with `functools.partial`, but a
passing accepted-gap control shows Pyrefly does not check that generic
partial's keyword binding: a wrong-width activation is silently accepted.
The prototype wrapper therefore uses an explicitly annotated local
callback that calls the unchanged kernel body, so its activation binding
is checked before `pallas_call`. This models a safe wrapper, not a proof
that the upstream `partial` wrapper is checked. The `BlockSpec` index-map
lambda, actual array strides/dtypes, and complete tile coverage remain
unverified as in the ordinary matmul fixture.

The pipelining guide's grid-last reduction adds a distinct case: a host
array `[Reduce, Rows, Cols]` is presented as a rank-two `[RowBlock,
ColBlock]` input Ref because its block shape is `(None, RowBlock,
ColBlock)`. The output allocation `[Rows, Cols]` and the rank-two
accumulating Ref share that tile, with a grid of row blocks, column blocks,
then the reduction extent. A separate `BlockSpec` layout marker distinguishes
the squeezed reduction input from the ordinary output block without
inventing a different runtime constructor. The unchanged `correct_sum_kernel`
body uses `@pl.when(pl.program_id(2) == 0)` to initialize the output Ref,
then reads and writes it with `+=`. Narrow `zeros_like` and Ref signatures
check the initialized tile, the addition, and the store; negative controls
reject a wrong input reduction extent, wrong accumulator tile, and
wrong-width initialization; the host wrapper also rejects a wrong output
row extent. The JAX CPU interpreter verifies the same body
on a `[2, 4, 4]` array reduced to `[4, 4]` using `[2, 2]` blocks.

This is only shape safety. The type of `AccumRef` permits a read even when
no prior iteration wrote it; `when` checks a boolean callback, not the
claim that the first visit to every output block initializes SRAM. An
accepted-gap test maps every reduction iteration to input slice zero while
still satisfying the same `BlockSpec` and host types. Consequently neither
the index-map semantics, proper grid iteration order, nor accumulator
liveness are validated. The JAX CPU interpreter confirms only the one
correctly mapped test case; it does not prove arbitrary grid shape,
nonoverlapping writes, or dtype behavior on TPU.

The TPU sparse guide's block-dynamic-slice kernel adds scalar prefetch.
Its unchanged body copies an input `[RowBlock, ColBlock]` Ref into a matching
output Ref; a `[2]` prefetch Ref appears first in the kernel's argument list.
The host supplies a `[SourceRows, SourceCols]` array and a `[2]` array of
block indices. A specialized `PrefetchScalarGridSpec` marker connects a
one-by-one grid, one scalar prefetch argument, two-dimensional input/output
`BlockSpec`s, and the output allocation `[RowBlock, ColBlock]` to the kernel
signature. Negative controls reject a wrong prefetch-array length or kernel
argument order, source rank, output allocation, grid shape, output Ref tile,
and index-map rank.
The CPU interpreter checks two index positions against a non-square source.
The source host dimensions are independent of the block dimensions because
the selected source block can come from a larger array; the grid-spec
annotation carries the declared source shape but does not establish that a
block fits inside it. The index-map lambda's block-index values are not
verified: swapping its row and column indices passes despite potentially
reading the wrong tile. Prefetch Ref element bounds, numerical block bounds,
scalar-memory placement, and the correspondence of declared prefetch count
to an arbitrary kernel are outside this narrow overload.

The GPU softmax kernel exercises an explicit masked read and write. Its
unchanged body computes `Indices[Block]` via `jnp.arange(block_row)` and
compares them to the input Ref's `Length`, obtaining a `Mask[Block, Length]`.
The specialized `plgpu.load` and `plgpu.store` stubs require that mask to
match both the indexed Ref's host length and its padded vector width. The
output Ref and host allocation share `[Length]`, and the return is also
`Array[[Length]]`. Negative controls reject a different host length, an
output Ref with a different length, masks based on another array or block,
an incorrect store-tile width, and an unmasked indexed load. A CPU interpreter
test confirms the original body returns JAX softmax values for a length-five
row padded to eight elements. The host uses a typed local callback to bind
`block_row`, since generic `functools.partial` keyword binding is not
validated by this prototype (as the activation matmul illustrates).
This is a narrow model of *masked* indexed access, not a general declaration
that Pallas forbids unmasked indexing. We do not prove `Block >= Length`,
that the mask actually covers all host elements, that `arange` or the
reduction computes the right numbers, or the input/output dtype. A passing
accepted-gap control permits an undersized block, even though it can leave
host output elements unwritten.

The TPU scratch-backed matmul adds a third, reducing grid axis and a device
scratch allocation. The host arrays `[Rows, Inner]` and `[Inner, Cols]`
yield input Refs `[RowBlock, InnerBlock]` and `[InnerBlock, ColBlock]`;
the output allocation `[Rows, Cols]`, output Ref, and VMEM accumulator
share `[RowBlock, ColBlock]`. `PrefetchScalarGridSpec` uses zero scalar
prefetch arguments and carries all six dimensions across the grid, the
two input `BlockSpec`s, output `BlockSpec`, and scratch `VMEM` allocation.
Inside the unchanged kernel, a dot product accumulates into the scratch Ref
on each `InnerBlock` iteration, with conditional initialization on the first
iteration and a conversion/store to the output on the last iteration.
Negative controls reject wrong host contraction/output axes, wrong dot
contraction or scratch tile, wrong scratch initialization, and an incompatible
third grid extent. The CPU interpreter matches `[4,6] @ [6,4]` on a
`(2,2,2)` grid with an `[2,2]` accumulator.

These signatures prove the *declared* host/Ref/scratch dimensions, not that
the three index-map lambdas select the right source/output blocks. The
conditional `@pl.when` and scratch Ref types do not prove initialization
before reading or that the captured `nsteps` equals the third grid extent:
a passing accepted-gap fixture binds a different number of steps. The host
uses an explicitly typed callback to bind `nsteps`, since `functools.partial`
alone does not check that binding. This kernel assumes exact block coverage
along the contraction axis; no type-level divisibility witness currently
prevents a partial, potentially unsafe reduction block.

The TPU pipelining guide's manual copy moves the first row of an input
array `[Rows, Cols]` through a scratch `VMEM[[1, Cols]]` allocation and
writes an output `[1, Cols]`. The unchanged kernel body calls
`pltpu.sync_copy(x_hbm_ref.at[0:1], scratch_vmem_ref)` before adding one
to the scratch tile and storing it. A specialized `pl.ANY` input
`BlockSpec` makes the kernel's input Ref refer to the full host array;
the zero-to-one slice has one row and preserves `Cols`. Negative controls
reject wrong host input rank or columns, source row-slice coordinates,
scratch-copy shape, output Ref width, scratch allocation, and output
allocation. The CPU interpreter checks an input `[4, 6]` becoming `[1, 6]`.
The stub calls this `UnconstrainedInRef`: `pl.ANY` usually results in HBM
placement but may choose VMEM, so its name does not certify HBM placement.
The `VMEM` scratch descriptor identifies a requested allocation, but the
types do not prove execution ordering, memory placement, or that `Rows >= 1`;
a source with zero rows passes the host type despite the first-row access.

The TPU pipelining guide's dynamic-block copy takes a host `[Rows, 128]`
array and `[SliceRows, 2]` start/end records, and returns `[Rows, 128]`.
The original kernel body is unchanged: it copies the records to an
`SMEM[[SliceRows, 2]]` scratch Ref and constructs a `BoundedSlice(8)`
`BlockSpec` whose `index_map` reads dynamic row ranges from that Ref.
`emit_pipeline` types the per-iteration input/output Refs with an eight-row
**upper bound**, rather than claiming every dynamic chunk has exactly eight
rows. Negative controls reject swapped or wrong-width host inputs, a scratch
copy/allocation with the wrong row count, a non-dynamic index-map result,
an incompatible pipeline tile, and wrong output allocation or `BlockSpec`.
This ties the host's declared shape, scratch table, and kernel Ref signature
together, but does not prove that the record ranges are ordered, in bounds,
nonoverlapping, cover every output row, or individually contain at most eight
rows. Nor does it prove that the callback's input and output dynamic ranges
are numerically equal; only their declared maximum extents match. JAX's
TPU-only `emit_pipeline` rejects CPU interpretation (`Unsupported TPU device
kind: cpu`), so this fixture has static checks but no CPU runtime test.

The TPU pipelining guide's megacore kernel adds rank-two arrays using a
**one-dimensional** parallel grid. The body sees two `[RowBlock, Cols]`
input Refs and a `[RowBlock, Cols]` output Ref, while both host inputs and
the output allocation share `[Rows, Cols]`. A rank-one grid computed from
`cdiv(Rows, RowBlock)` feeds rank-two `BlockSpec`s whose index maps each
return two block coordinates; `compiler_params` declares one `"parallel"`
or `"arbitrary"` axis. Negative controls reject mismatched host rows or
columns, incompatible input/output tiles, a wrong index-map arity, grid
block size, output allocation, and compiler-axis count. The CPU interpreter
checks `[4, 6]` inputs in two row blocks.
The index-map **values** are not validated: an accepted-gap control sends
every program to block `(0, 0)` despite correct shapes, so complete and
disjoint output coverage is not proved. The `"parallel"` compiler annotation
does not prove independent writes, `cdiv` has no numerical proof, and the
Ref types do not certify VMEM residency or TPU scheduling.

The distributed TPU guide's right-permute kernel copies a local input
shard `[Rows, Cols]` into a remote output shard of the same shape. A
zero-scalar-prefetch grid spec binds the two full-shard `BlockSpec`s and
exactly two DMA semaphore allocations to the unchanged kernel's four
Refs/handles. `make_async_remote_copy` requires source and destination
shapes to agree; negative controls reject mismatched host dimensions,
remote output widths, semaphore counts, and output allocations. This
is a **per-shard** Python-to-Pallas contract: the enclosing `jax.shard_map`
in the original guide turns a global `[Rows, Cols * Devices]` array into
per-device `[Rows, Cols]` arrays, but that global sharding boundary is
not typed by the current overlay. The types do not prove that
`num_devices` equals the mesh size (zero is accepted), that the chosen
neighbor is on the mesh, which semaphore belongs to sending or receiving,
or whether `.start()` and `.wait()` synchronize successfully. `pl.ANY`
does not guarantee HBM placement. The JAX CPU interpreter rejects this
remote DMA even under a two-device CPU `shard_map`, so the fixture has
static checks but no CPU runtime assertion.

The distributed-guide `all_gather_kernel` exercises the missing global
boundary. With `Mesh[Devices]` and `P("x", None)` partitioning the first
axis, the narrow `jax.shard_map` stub relates a global input
`[Devices * Rows, Cols]` to the callback's local input `[Rows, Cols]`.
The unchanged Pallas kernel writes a local `[Devices, Rows, Cols]` output
Ref via individual `[Rows, Cols]` output slots. A typed `pallas_call`
binds that output allocation and two single DMA semaphores plus a
`[Devices - 1]` receive-semaphore allocation to the kernel. The same
`shard_map` rule then describes global output
`[Devices * Devices, Rows, Cols]`. The actual `pallas_call` callable is
passed to `shard_map`, so mismatched local callback inputs or outputs are
rejected, including with an abstract `Mesh[Devices]` and a concrete
`Mesh[2]`; this is not a manually restated kernel signature. Other negative
controls reject a wrong global input shape, partition axis, local copy
width, output allocation, and receive-semaphore count. A two-device CPU
test of JAX's ordinary `lax.all_gather` (not Pallas DMA) checks that
`[16, 128]` becomes local `[8, 128]` and global `[4, 8, 128]`.

This is a specialized trusted `shard_map` contract for a leading mesh axis,
not a proof of arbitrary sharding layouts: the mesh device count and
partition mapping are accepted from their typed JAX boundary. Pyrefly
currently needs the mesh and scalar dimensions passed before the global
array so it can bind variables before checking `[Devices * Rows, Cols]`;
an array-first generic wrapper leaves those variables unresolved. Bounds
of each dynamic output slot (`.at[999]` is accepted), remote DMA completion,
per-device routing, synchronization and HBM residency are not proved. The
original TPU remote-DMA kernel has no CPU interpreter assertion.

The distributed-guide `all_reduce_kernel` adds a different partition axis,
two output arrays, VMEM scratch, and a double-buffered remote copy. A
`P(None, "x")` mesh partition presents global `[Rows, Devices * Cols]`
input as a local `[Rows, Cols]` Ref. The unchanged kernel accumulates into
`[Rows, Cols]` VMEM and copies equal-size tiles into an output
`[2, Rows, Cols]` double buffer. The `pallas_call` overload checks both
declared output allocations, the typed input/output `BlockSpec`s, the
`VMEM[[Rows, Cols]]` scratch allocation, and the DMA versus regular
semaphore descriptors. The actual returned callable passes through a
specialized `shard_map` rule, which checks its local callback signature and
exposes global result `[Rows, Devices * Cols]` and global scratch
`[2, Devices * Rows, Cols]`. Negative controls reject wrong global input,
accumulator and copy widths, input memory space, scratch descriptor and
allocation, either output allocation, callback input and scratch output,
and partition axis. A two-device CPU `shard_map` test checks these local
and global dimensions using an ordinary JAX callback, not TPU DMA.

The shape contract does not establish numerical device bounds, whether
both slots are initialized, semaphore balance or deadlock freedom, remote
copy ordering, collective-id uniqueness, actual HBM placement under
`pl.ANY`, or dtype agreement. An out-of-bounds `.at[99]` slot passes the
current Ref shape rules. The barrier helper's body has explicit semaphore
types, but `functools.partial(pl.run_scoped, ...)` does not statically
verify their callback binding. TPU remote DMA cannot run in the CPU
interpreter, so the original all-reduce has no CPU execution claim.

The distributed-guide `reduce_scatter_kernel` changes the host sharding
axis: global input `[Devices * Rows, Devices * Cols]` partitions columns
via `P(None, "x")`, so each device reshapes `[Devices * Rows, Cols]`
into `[Devices, Rows, Cols]`. The original kernel reads one half-row tile
`[Rows // 2, Cols]` from a selected source device, moves it through
double-buffered `[2, Rows, Cols]` HBM scratch and VMEM accumulator
`[Rows // 2, Cols]`, and writes local output `[Rows, Cols]`. The
`P("x", None)` output spec yields global `[Devices * Rows, Cols]`.
The complete kernel body matches the guide's Python AST after removing
only parameter annotations. The typed `pallas_call` relates its eleven
parameters to both output allocations, the two-axis grid `(Devices, 2)`,
memory-space specs, five DMA and two regular semaphores, and VMEM scratch.
Negative controls reject wrong host rows or columns, whole-row slices
used as half-row tiles, incompatible VMEM accumulation/copy widths, and
incorrect output memory space. A two-CPU-device JAX `shard_map` test uses
`lax.psum_scatter` to check both the resulting shape and numeric reduction;
it does not execute the TPU Pallas DMA kernel.

This is a *partial* host proof. Pyrefly does not currently infer a symbolic
product from a generic higher-order callback parameter. The specialized
`jax.shard_map` stub consequently checks the local callback's input/output
shapes and the global sharding on each axis, but does not assert internally
that the input's local row extent equals `Devices * Rows`. The explicitly
annotated wrapper declares that equality at its boundary. The `reshape`
stub also cannot prove element-count conservation: an accepted-gap test
reshapes `[Other, Cols]` to `[Devices, Rows, Cols]`. Nor does a typed
half-row slice check its start coordinate (a slice starting at 999 passes).
The checker does not prove that `Rows` is even, phase/device indices are
in bounds, scratch is initialized before use, semaphore roles and DMA
ordering are correct, or that `~` on JAX predicates differs from Python
boolean inversion. The latter produces three expected Pyrefly deprecation
warnings in this unchanged AST; they are not evidence of a bad kernel.

The SparseCore guide's gather introduces an **indirect index array**. A host
data array `[Batch, Cols]` and indices `[Num]` produce output `[Num, Cols]`;
the typed reshape presents indices to the kernel as `[1, Num]`. Within the
original outer kernel, `emit_pipeline` presents indices `[1, Window]` and
output `[Window, Cols]` to the unchanged nested callback. Its gather
`x_hbm.at[i_vmem.at[0]]` returns an indirect `[Window, Cols]` source, and
`pltpu.sync_copy` requires its width and window size to match the output
block. A narrow `pl.kernel` decorator checks the declared `[Num, Cols]`
output allocation against the kernel Ref and its input Ref types; the
pipeline call checks that the full index Ref's `Num` rows agree with the
full output Ref's `Num` rows. Negative controls reject mismatched host
index count, source/output widths, pipeline output extent, and allocation.
The guide's outer and nested kernel ASTs are unchanged apart from the new
parameter annotations. A CPU JAX `take` test checks the host index/output
relationship, **not** execution of this SparseCore kernel on CPU.

These types do not know the *values* stored in the index array, so a
well-shaped but out-of-range indirect gather remains accepted. Grid coverage
is also unproved: `emit_pipeline` currently accepts any integer grid extent,
including 999 for an unrelated `Num`, because inferring `Num` from the
generic symbolic quotient `Num // Window` fails. The `BlockSpec` index-map
values, divisibility of `Num` by `Window`, actual memory placement, index
dtype, and the SparseCore mesh are not validated. This separates the
trustworthy host array shape contract from the unchecked numeric mapping.

The guide's SparseCore scatter exercises the opposite direction: values
`[Num, Cols]` and indices `[Num]` write into a separately allocated output
`[Batch, Cols]`. The original outer kernel and nested callback ASTs are
unchanged after removing parameter annotations. A typed `pl.kernel`
decorator connects the full input, reshaped `[1, Num]` index array and
declared output allocation; `emit_pipeline` connects `[Window, Cols]`
values to `[1, Window]` indices, then checks their `Num` extents at its
call site. `o_hbm.at[i_vmem.at[0]]` produces an indirect destination;
`pltpu.sync_copy` rejects mismatched source/destination column or window
extents. Negative tests catch wrong index count, indirect store width,
pipeline value/index count, and output allocation. Crucially, **output
`Batch` is not derived from input `Num`**: it is supplied by the host
allocation and the wrong-Batch negative is interface-only, not a proof
from body reads/writes. CPU JAX `.at[indices].set(values)` tests validate
the resulting host shape and one concrete permutation, not this TPU kernel.

The index values remain opaque: neither numeric output bounds nor
uniqueness, collision ordering, complete coverage, or initialization of
untouched output rows is proved. A CPU shape test with duplicate destination
indices demonstrates that duplicates are permitted by the host operation,
while `Indices[Window]` carries no values with which to reject them in the
kernel. Grid length/divisibility, index-map values, dtype, VMEM/HBM residency,
and the SparseCore mesh likewise remain unverified.

The guide's packed-bfloat16 gather makes the host's dtype and packing
material to the kernel interface. A `BFloatArray[[Batch, Cols]]` is reshaped
to `[Batch // 2, 2 * Cols]`, then viewed as int32 `[Batch // 2, Cols]`
before the kernel. The dtype marker comes from `jax.Array.astype(jnp.bfloat16)`;
passing a generic `jax.Array[[Batch, Cols]]` is rejected. The original
outer and nested kernel bodies are unchanged after stripping only their
parameter annotations. The body gathers `Indices[Window] // 2` packed rows
into int32 VMEM `[Window, Cols]`, views each word as two bfloat16 values,
reshapes to `[Window, 2, Cols]`, then selects the even or odd half into
an output `[Window, Cols]`. Narrow `sync_copy`, `VMEM.view`, and `jnp.where`
signatures check this tile-width correspondence. `pl.kernel` connects
the packed host input Ref, `[Num]` indices, `[Num, Cols]` output allocation,
and `[Window, Cols]` scratch descriptor. Negative controls reject a wrong
host dtype, host batch/index extent, packed input-Ref rows, scratch or
gather width, bf16 pair width, pair selection, or output allocation.
A CPU JAX test verifies the pack/view/unpack shapes and selected values;
it does not run SparseCore DMA.

The host wrapper uses the literal `packing: Literal[2] = 2`: Pyrefly infers
only `int` from the guide's equivalent `32 // 16` assignment and otherwise
loses the `[Batch // 2, 2 * Cols]` relation. The type system still does
not prove that `Batch` is even. An accepted-gap test permits `Batch = 5`,
while a CPU test shows the corresponding reshape fails. Nor does the
current `VMEM` marker track the int32 dtype of its allocation: the body
can request a bfloat16 view of a scratch Ref allocated with a different
dtype. `scratch_types` accepts arbitrary dictionary keys rather than
checking the spelling `gather_vmem`. Numeric index bounds, parity value
correctness, dtype of the output and indices, grid coverage, and the
`BlockSpec` mappings also remain outside this shape contract.

The SparseCore guide's scalar-subcore `cumsum` demonstrates a host-row to
core mapping and a one-dimensional SMEM scratch buffer. The unchanged
`@jax.jit` wrapper passes its `[2, Lanes]` host input to a `pl.kernel`
decorator whose `ScalarSubcoreMesh[2]` selects one row per core. The
unchanged kernel uses `jax.lax.axis_index("core")` to copy that row into
`SmemScratchRef[[Lanes]]`, scans the scratch values with `pl.loop`, then
copies back to the corresponding output row. The `async_copy` overloads
reject scratch/source or scratch/destination lane mismatches; the kernel
decorator rejects incompatible host core count, scratch extent, or mesh
extent. Negative tests demonstrate each mismatch. Constructing an actual
SparseCore mesh requires TPU hardware, so the CPU JAX test checks the host
row-wise `cumsum` shape and values but does not execute this kernel.

The `CoreIndex` type records the name of the chosen axis but does not prove
its numerical range or that either DMA's row index matches the mesh core.
The `scratch_types` stub checks element shapes but not list order or dtype:
accepted-gap tests deliberately reverse the semaphore and scratch elements
and allocate a bfloat16 scratch descriptor for generic input dtype. The
per-core scan's arithmetic and loop bounds are not statically checked.

The TPU sparse guide's dense-output sparse-block matmul uses two `[Blocks]`
prefetch index arrays to map sparse LHS blocks `[Blocks, BM, BK]` and dense
RHS `[K, N]` into output `[M, N]`. The original `dsd_kernel` and all three
index-map bodies remain unchanged after stripping parameter annotations.
`PrefetchScalarGridSpec` ties `Blocks`, `BM`, `BK`, and `BN` to the kernel's
input and output Refs and its `[BM, BN]` VMEM accumulator. The
`pallas_call` overload connects the full index and LHS block counts, dense
RHS and output widths, and the declared `[M, N]` zero buffer to
the returned `[M, N]` array. It requires `input_output_aliases={4: 0}`:
the zero buffer is input position 4 and the output is position 0. Negative
probes reject host count/width mismatches, an incorrect alias slot, a bad
matmul contracting dimension, and a mismatched accumulator width. A CPU
test reconstructs the dense LHS from sparse blocks and checks the reference
matmul values; the TPU kernel itself cannot run on CPU.

The index maps' returned values are not inspected by the type system.
In particular, block row indices must be grouped to make the accumulator
initialize and flush correctly, block indices must remain in bounds, and
the grid must cover the output without duplicating or skipping blocks.
The stub's `in_specs` list union also does not enforce order or length,
and the full `K` dimension is not constrained to be divisible by `BK`.
Output/operand dtypes and the contents of the zero buffer remain unproved;
the alias marker checks the mapping slot and shape but cannot prove the
caller initialized every output block to zero.

The sparse guide's masked-output matmul adds a second sparse-matrix
interface: four metadata arrays (`block_mask`, `prefetch_mask`,
`prefetch_i`, `prefetch_j`) share an output-grid shape
`[Rows // BM, Cols // BN]`, dense inputs have shapes `[Rows, Inner]` and
`[Inner, Cols]`, and sparse mask data has shape `[MaskTypes, BM, BN]`.
The full unchanged `sparse_mask_matmul` body reads the block mask to
conditionally accumulate `BM × BK @ BK × BN` into a `[BM, BN]` tile and
multiplies by a `[BM, BN]` sparse mask tile before writing output. The
unchanged four index maps select blocks based on metadata values. Negative
probes reject metadata grid extent, sparse mask tile width, kernel-prefetch
Ref extent, matrix contraction width, and tile multiplication mismatches.
The prefetched `block_mask` is consumed in the kernel body; the other three
maps are consumed by index-map callbacks, so their extent controls check
the boundary rather than operations inside `sparse_mask_matmul`. CPU tests
check dense reference masking and grid shapes, not the TPU kernel.

For this wrapper the scalar dimensions precede the array parameters:
Pyrefly can check `[Rows // BM, Cols // BN]` after binding `Rows`, `BM`,
`Cols`, and `BN`, but does not infer those variables from quotient-shaped
arrays in first argument position. Moreover it cannot infer arbitrary
symbolic quotient shapes from `PrefetchScalarGridSpec.grid`; the stub
checks grid arity and block shapes but does not prove grid cardinality.
Its type links all four metadata arrays through the kernel's parameter
annotations and the `pallas_call` argument types, rather than deriving
the mapping from the index-map callback values. Correct prefetch values,
in-bounds block indices, output grid coverage, dtype, and mask-preprocessing
behavior remain unproved.

The GPU pipelining guide's Hopper matrix multiply tests the Mosaic GPU
memory boundary. Two host float16 matrices `[Rows, Inner]` and
`[Inner, Cols]` feed full GMEM input Refs; the unchanged outer kernel's
pipeline callback receives SMEM tiles `[TM, Swizzle // 2]` and
`[Swizzle // 2, TN]` and issues `wgmma` into an accumulator `[TM, TN]`.
The original kernel stores that accumulator into SMEM `[TM, TN]`, then
copies exactly one `[TM, TN]` tile to a sliced GMEM output `[Rows, Cols]`.
The typed `plgpu.kernel` call checks host operand shapes against GMEM,
scratch/accumulator tile sizes against the output allocation, and the
returned `[Rows, Cols]` shape. Wrong float16 input type, contracting
dimension, pipeline tile, accumulator width, output tile copy, or host
output allocation produces an error in the negative controls. The wrapper
derives the guide's free `m`, `k`, `n` globals from the typed operand
shapes; all statements of the nested pipeline callback and outer kernel
remain unchanged after erasing their new parameter annotations. A CPU
test checks reference matrix shapes, not Mosaic GPU execution.

The checker does not derive CUDA grid coverage from `grid_m` or `grid_n`,
validate swizzle transforms or the divisibility assertion, or check
the runtime ordering of asynchronous operations. Input float16 dtype is
tracked by `Float16Array`, but `ShapeDtypeStruct` currently tracks only
shape: even a float32 output declaration is accepted, so this prototype
returns the general `jax.Array[[Rows, Cols]]` without claiming a verified
output dtype. Scratch allocation dtypes and dictionary key spellings are
likewise not checked.

The TPU `core_map` indexed-add guide uses a scalar prefetch to select a
half-width window of a full input array. An int32 index array `[1]` is
copied into SMEM `[1]`; its scalar value shifts the input's column-block
index map. The original outer kernel and the `add_one_body` callback are
unchanged after erasing only semantic parameter annotations. Typed Refs
connect local input `[Rows, Cols]` to output `[Rows, Cols // 2]`, while the
`emit_pipeline` callback checks each input and output tile is `[8, 128]`.
Host probes reject a wrong index count or dtype; body probes reject a
wrong SMEM index scratch, output Ref width, or tile write width. A separate
`shard_map` fixture connects a global row-sharded array
`[2 * LocalRows, Cols]` to per-device `[LocalRows, Cols]`, broadcasts the
index `[1]`, and reconstructs global output
`[2 * LocalRows, Cols // 2]`. An explicit `local_rows` scalar grounds
Pyrefly's inference: it cannot infer `LocalRows` directly from an array
whose leading dimension is `2 * LocalRows` in first argument position.

The typed `CoreIndex` does not prove which rows each core actually covers.
Neither the offset in SMEM nor its bounds (`0 <= index <= Cols // 2`) are
tracked; an accepted high index would sample outside the input window.
CPU reference tests demonstrate a valid half-width slice and the numeric
range issue without running the TPU DMA pipeline. The original guide's
`shard_map` decorators are represented by a separate explicit host wrapper
so the local kernel interface and global sharding relation can each be
checked; numeric grid coverage, dtype of the SMEM allocation and index-map
callback values remain unproved.

The next kernel in `docs/pallas/tpu/core_map.md` runs on SparseCore vector
subcores. Its bounded `[8, 128]` block is divided into `[4, 16]` register
operations by two nested `pl.loop` decorators; `BoundedInRef` reads and
`BoundedOutRef` writes now preserve that register shape, so a wrong register
write width fails. An aligned-slice marker also rejects a `BlockSpec` index
map that omits the original `pl.multiple_of(..., 8)` alignment hint. The
array/Ref interface preserves local `[Rows, 128]`, rejecting an incorrect
host width or an output Ref with the wrong row count. A separate explicit
`shard_map` wrapper ties global `[2 * LocalRows, 128]` to local
`[LocalRows, 128]` and back; its `local_rows` scalar grounds generic
inference and a mismatched global row extent is rejected. Both full official
kernel bodies are unchanged apart from parameter annotations. The host CPU
check only tests the shape and values of `x + 1`; it cannot run SparseCore.
The overlay does not verify that `Rows` is divisible by the 64 subcores
times the eight-row block, that the 4-by-16 register loops cover each
bounded tile, that the index map visits every block exactly once, or that
the hardware has exactly four cores and sixteen subcores. In particular,
the 4-by-16 register Ref records an operation's maximum size, not its
runtime-valid extent. `multiple_of` is an asserted alignment hint, not a
proof that the arithmetic expression really is divisible by eight. Pyrefly
cannot infer a generic `Groups` solely from an array dimension `512 * Groups`;
the host-facing annotation therefore leaves the first dimension generic
instead of falsely claiming to enforce row divisibility.

The TPU PRNG guide's stateless example adds a qualitatively different host
boundary: a host `jax.random.Key["threefry2x32"]` is explicitly converted
with `pltpu.to_pallas_key`, passed through an `SMEM`-selected input
`BlockSpec`, and used by the unchanged kernel body to generate an output
tile `[8, 128]`. The semantic key Ref and `pallas_call` overload reject
an unconverted key, an incompatible key family, and an incorrectly sized
output declaration. Inside the body, `jax.random.uniform` receives only a
Pallas key and returns a tile matching the Ref's declared shape; a wrong
output write width or a raw-array key is rejected. A shape-readable output
Ref exposes only `.shape` through `o_ref[...]` without blessing arbitrary
reads of uninitialized output data. CPU tests exercise a JAX Threefry host
reference, not a TPU random generator. The checker does not establish the
actual hardware key representation, SMEM placement, random-number values,
or the output's float32 dtype: `ShapeDtypeStruct` currently tracks shape
but not dtype.

The following PRNG guide example draws a block-invariant `[64, 512]`
array through two tilings: `[16, 128]` with a `(4, 4)` grid, and
`[32, 256]` with a `(2, 2)` transposed grid. The complete original
`make_kernel_body` and nested `body` ASTs are retained with semantic
parameter annotations. The typed host call relates the output array,
`BlockSpec` block shape, grid division and converted input key. The body
derives the generated tile shape from its output Ref and `sample_block`,
checking wrong tile writes and the guide's declared full and tile sizes.
Negative controls reject wrong host key, output extent, grid, sampling
tile size and block write width. A duplicate index-map callback still
passes: the checker cannot prove it covers the output or that permuting
two grid axes preserves sampled random values. It also does not prove
the tile divides every block, or that two layouts produce identical
random numbers. CPU tests check only host tiling geometry.

The official GPU `ops/gpu/rms_norm.py` forward kernel is a larger
Marin-relevant normalization example (the upstream module is deprecated
in favor of tokamax). Its complete kernel, including the nested masked
reduction callback and the subsequent masked store loop, is unchanged
apart from semantic parameter annotations. A row kernel takes three
`[Features]` input Refs (activation, weight, bias), writes a
`[Features]` output Ref and optionally a scalar `[]` reciprocal-standard-
deviation Ref. Two typed `jax.vmap` applications relate the host inputs
`x: [Batch, Rows, Features]`, `weight/bias: [Features]` to outputs
`[Batch, Rows, Features]` and `[Batch, Rows]`. Wrong host weight/bias widths,
wrong output/scalar Ref shapes, and a mask derived from a different input
length are rejected. The tiled variance accumulation checks the mask and
feature shape through a `[Block]` accumulator, while scalar output retains
rank zero. CPU tests evaluate a host JAX reference, not the GPU kernel.
The checker does not prove `Block > 0` or that the masked loop visits every
feature exactly once; it also does not validate floating-point accuracy,
dtype conversions, epsilon positivity or JAX's `vmap` runtime semantics.

Marin's `lib/levanter/src/levanter/kernels/pallas/short_conv/pallas_gpu.py`
adds a real-world halo pattern to this corpus. The five complete forward
kernel/helpers and the host `_head_views` helper remain AST-identical after
removing only semantic parameter annotations. The host adapter accepts
activations `[Batch, Seq, Channels]`, segment IDs `[Batch, Seq]`, and
weights `[Width, Channels]`, and calls the unchanged `_head_views` to
derive `[Batch, BlockSeq + Width - 1, Channels]` and
`[Batch, BlockSeq + Width - 1]` padded views. The kernel output is
`[Batch, Seq, Channels]`.
`BlockSpec` shapes also distinguish full-sequence windows, head windows,
and a `[1, BlockSeq, ChannelBlock]` output tile. The first program calls
the body with head Refs and `Width - 1` as its base; later programs use
full-sequence Refs and their program-id-derived base. Inside the body,
`pl.ds` retains the dynamic block extent, segment equality conditions
retain row dimensions, and the output tile must match its block Ref.
Negative probes reject an incorrectly padded head length, wrong segment
rows or host weight channels, mismatched window/segment extents and output
tile rows. A CPU test runs the original `_head_views` and checks the
segment-masked reference, not GPU execution.

Pyrefly cannot infer `BlockSeq` and `Width` from the nested dimension
`BlockSeq + Width - 1`; the adapter takes those scalar witnesses before
the arrays. The `_fwd_kernel` parameter annotation keeps an independent
symbolic `Halo` to avoid a second nested-inference failure; the host
adapter is where its required length is checked. This prototype does not
verify the *order* of concatenated head chunks (the reversed order has the
same type), that segment sentinel values cannot collide with live IDs,
that the sequence/channel lengths divide their block sizes, that the
program-id dispatch covers all tiles, or the numeric safety of `base - lag`.

This makes the Python-to-kernel contrast concrete: Pallas typically declares
tile location in the host-side `BlockSpec` index map and gives the body a
pre-sliced Ref, while this unblocked example addresses the full host array
explicitly by program id. Triton instead computes a pointer offset from its
program id inside the kernel and usually carries an explicit bounds mask.
Our current Pallas annotations check the host allocation shape and declared
grid/slice shapes more directly, but do not validate the *meaning* of a
`BlockSpec` index-map lambda. A reusable grid-to-view mapping abstraction
could help both DSLs: Pallas supplies the mapping out of band through
`BlockSpec`, whereas Triton's corresponding mapping is expressed by pointer
arithmetic in the body. Connecting those two representations without
assuming a particular index-map lambda remains future work.
Compositional `Ref.at` slices describe a Pallas tile's subregions locally;
their Triton counterpart is explicit two-dimensional pointer arithmetic
using host matrix strides and bounds masks. The Pallas body avoids those
explicit pointer/mask operations, but the boundary must still relate the
host array's dimensions to the Ref extents and the mapping that supplies it.

This does **not** prove that `length` is the actual length of either input
array at runtime, that `cdiv` computes the correct numerical value, that
`lambda i: (i,)` actually maps each program to its own block, that different
programs' writes cannot overlap, or that a host JAX array has any particular
storage strides. The only modeled `pallas_call` signatures handle this vector
add and matmul; they are not a general Pallas stub.
`test_index_map_coverage_is_not_proven` intentionally passes a broken map
that sends all programs to block zero, demonstrating the missing proof.
Matmul's `test_index_map_attribution_is_not_proven` similarly type-checks an
output map sending every program to block `(0, 0)`, despite a correctly typed
grid and block shape. The input maps' relationships to the output map are
also unchecked.
The two-dimensional `BlockSpec` constructor in this overlay assumes a
two-dimensional grid, although real Pallas allows grid rank and array rank to
differ. These specialized overloads are examples, not general indexing rules.
The squeezed-row marker checks the selected layout, but not the semantics of
the index-map lambda: `(0, j)` passes static checking even though every
program targets row zero. It also does not derive an arbitrary host shape
from an arbitrary `BlockSpec`; each `pallas_call` overload states the
host-array and Ref shapes explicitly. Other squeezed-axis placements and
grids are not represented by this stub.
Pallas pads partial reads and discards out-of-bounds writes in its blocked
indexing mode, so the final partial block does not require the Triton-style
explicit bounds mask. The padding values are unspecified and cannot safely be
used in an in-block reduction.

From this directory, check static expectations with:

```sh
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations 'tests/test_*.py'
```

If you later install JAX in `.venv`, run the optional CPU-interpreter tests with:

```sh
../../.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Exercise the two-device CPU `shard_map` shape tests with:

```sh
XLA_FLAGS=--xla_force_host_platform_device_count=2 ../../.venv/bin/python -m unittest discover -s tests -p 'test_shard_map*boundary.py'
```

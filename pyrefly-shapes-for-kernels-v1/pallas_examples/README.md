# Pallas examples

`test_iota.py` models the quickstart's full-output Ref indexed by a program
ID, with one program per element and no input or block spec. It rejects a
different grid length statically and invalid grid length at runtime;
`test_shard_map_boundary.py` checks the global/local `[Devices*Rows,Cols]`
relationship through `jax.shard_map`, including a two-device CPU test when
available. These are distinct from tiled `BlockSpec` kernels: a general
Pallas boundary must not assume every Ref is one block of a grid tile.

`test_tpu_vector_add.py` models a TPU launch of the Pallas quickstart's
vector-add body. `test_tpu_matmul.py` models the TPU guide's three-axis
tiled matmul. `test_decode_attention.py` models an unbatched split-KV GPU
decode kernel. Their checked host boundaries relate
input/output shapes, grid tiles, and Ref signatures; CPU interpretation
checks representative results. These tests do not claim TPU or GPU hardware
execution, compiler lowering, or proof of arbitrary index-map callbacks.
The TPU matmul also cannot prove that its conditional accumulator
initialization runs before every update. Decode attention checks an unbatched
FP16/BF16 case with whole KV splits; optional sequence bounds, batched
variants, and other branches remain outside the fixture. The narrow
`jax.numpy.asarray` overlay exists to construct NumPy-backed CPU test inputs,
not to infer their symbolic shapes automatically.

`test_pipeline_matmul.py` checks the Hopper Mosaic GPU pipeline from JAX's
GPU pipelining guide. The host signature ties FP16 `[M,K]` and `[K,N]` to
`[M,N]`; runtime checks require matching devices, whole positive output/K
tiles and WGMMA-compatible tile/swizzle alignment. Kernel Refs distinguish
GMEM inputs/output, SMEM pipeline tiles and an accumulator, with static
checks of their matmul contraction axis and final store tile. Tests inspect
the grid, output specification and scratch metadata with the GPU launcher
mocked; no GPU execution or kernel lowering is claimed. The launch and local
stub use JAX's `out_type` and `scratch_types` APIs. The fixture places
`delay_release=1` on both input `BlockSpec`s, as required by the installed
JAX API, and omits it from `emit_pipeline`. This placement preserves the
pipeline's intended buffer lifetime. Focused tests check the captured
pipeline arguments, but frontend/GPU lowering remains untested. The eager
host adapter checks device identity and does not run under JAX
tracing; a trace-compatible adapter would need to avoid inspecting tracer
devices while retaining the same shape and dtype checks.
As in the other layouts, static types do not prove the index-map lambdas
agree with the intended grid tiling.

`test_ragged_dot.py` ports Marin's group-indexed Pallas contraction. The
checked layout binds LHS `[Rows,Inner]`, RHS `[Groups,Inner,Cols]`, two
`[Groups]` bound arrays and output `[Rows,Cols]` to the three-axis grid.
Before the call, the host adapter validates that nonnegative integer group
sizes sum to `Rows`, making the prefix boundaries valid; this check transfers
the small group-size array to the host and is not JIT-traceable. The kernel
body checks reduction and output tile shapes/masks; CPU interpretation covers
partial reduction blocks. The CPU adapter rejects partial output-column
blocks because JAX's CPU interpreter cannot lower the `program_id` reached
through the upstream conditional store-mask branch; this is an interpreter
restriction, not a proof that the GPU kernel cannot handle those columns.
Its output block carries the full valid-column bound even though the visible
Ref has only one column tile. Static fixtures reject output masks with wrong
row or column bounds. Its specialized LHS/RHS input slices remain necessary
because the same slices are loaded with and without masks in separate branches;
the types cannot prove from the runtime branch that the unmasked loads stay
within the logical reduction length.
With `JAX_DISABLE_JIT=1`, eager Ref indexing also rejects the masked final
contraction slice before the load; the partial-K numerical test is skipped
only in that mode. Normal CPU interpretation executes the test. Neither mode
establishes safety on an actual GPU.
The type system does
not verify the values returned by the index-map lambdas, nor associate a
specific RHS matrix with each runtime boundary. The CPU test substitutes
`jnp.dot` for `pl.dot` in test scope: the installed JAX omits `pl.dot`
from the upstream body. Production execution needs that upstream API
migration; the substitution does not change the preserved kernel source.
The local JAX stub overlay defines `shape` and `dtype` on its synthetic
`RaggedCumulative` result so the checked boundary can verify the prefix.

`test_rms_norm.py` checks JAX's original GPU RMSNorm row kernel with
three `[Features]` inputs and a `[Features]` output plus one optional scalar
reciprocal standard deviation. `row_rms_layout` checks the output row and
statistic shapes, and the checked call validates all input shapes and dtypes.
CPU interpretation compares an irregular row against an independent JAX
reference, exercising the partial final block. The same grid-free Ref pattern
works for layer norm; the two-output builder still duplicates some of the
three-output `row_statistics_layout` signature, a useful case for a general
checked output-tuple mapping later. Neither type rule proves the numerical
reduction nor the provenance of the returned statistic.

`test_attention_backward.py` contains JAX's full two-scan attention-backward
kernel. Its checked layout relates Q/O/dO/dQ `[B,Q,H,D]`, K/V/dK/dV
`[B,K,H,D]`, optional integer segment IDs `[B,K]`, and float32 LSE/Delta
`[B,H,Q]` to a shared `(B,H,K/block_kv_dkv)` grid. Pallas can support
different query and key lengths here; both scans must have the same number
of output blocks. The shared layout builder permits a genuinely absent
input (`None`) without shifting subsequent `BlockSpec`s, and the checked
call enforces presence, shapes, dtypes, and devices. CPU tests compare both
noncausal and causal/segmented gradients against independent JAX derivatives.
The kernel body is unchanged. `IntListLiteral` recovers the two ordered
dimensions in the upstream `jnp.zeros([block_rows, padded_dim])` calls;
the generic two-matrix `fori_loop` carry then checks both scan accumulators
and their stores without local suppressions. Index-map lambda values and
the alignment of externally saved LSE/Delta with the inputs remain outside
the proof.
Selected Q/K/V inputs and dQ/dK/dV outputs retain both padded head width and
the original valid `D` extent in `TransformedRef`. Feature-mask bounds and
orientation are checked for both loads and stores; a mask that marks the full
padded width as valid is rejected for the output.

`test_attention_backward_preprocess.py` retains JAX's preprocessing body for
attention backward. A checked layout binds two `[B,Q,H,D]` inputs to
`[B,H,Q]` delta and groups the grid as `(Q/block_q,B,H)`. Input BlockSpecs
select a `[block_q,padded_dim]` logical Ref; the kernel masks feature lanes
beyond the physical `D`, allowing a non-power-of-two host head dimension.
The feature mask has type `Mask[[1, padded_dim], [1, D]]`; its singleton row
axis broadcasts across the query tile, while its valid feature bound matches
the host head dimension. A mask with the wrong bound or broadcast axis fails
the static `plgpu.load` check.
The output BlockSpec reorders the batch/head/query axes, and the checked call
validates both input arrays. The builder checks the spec shapes and output
permutation but, as for forward attention, cannot prove arbitrary index-map
lambda bodies. CPU interpretation covers padded features and multiple heads.

`test_layer_norm_backward.py` keeps JAX's GPU layer-norm input-gradient
kernel body unchanged. A typed `row_input_gradient_layout` relates four
same-length input rows, two saved scalar statistics, and a same-length output
gradient, using the empty grid and Pallas's full-row Refs. The CPU interpreter
checks a partial final block against the independent layer-norm derivative.
The wrapper validates input shapes and dtypes at launch; it does not prove
that saved mean and reciprocal standard deviation were computed from this
particular input. `test_layer_norm_weight_grad.py` contains the upstream
weight/bias-gradient kernel. Its checked layout ties two full `[Rows, Cols]`
matrix Refs, two `[Cols]` vectors, two `[Rows]` saved statistics, and both
`[Cols]` outputs to a grid of column tiles. The CPU test covers partial row
and column tiles and compares against an independent batch reduction. The
kernel inputs and outputs use generic shape-parameterized `InRef`/`OutRef`.
Indexing an input Ref produces a `TransformedRef` carrying both the original
matrix bounds and the selected tile shape, so a masked load can compare its
mask against both. The Ref type does not track where within the matrix the
selection starts.

`test_attention_forward.py` retains JAX's deprecated GPU `mha_forward_kernel`
and the optional segment-mask helper unchanged, but exercises the noncausal,
unsegmented branch. `attention_layout` relates Q `[B,Q,H,D]`, K/V `[B,K,H,D]`,
output `[B,Q,H,D]`, and base-2 log-sum-exp `[B,H,Q]` to a
`(Q/block_q, B, H)` grid. Its typed callback receives squeezed query
`[block_q,D]` and full-key/value `[K,D]` Refs plus a `[block_q]` stats Ref.
The full query read carries an explicit valid extent; key/value and output
slices use general `TransformedRef` types whose feature masks must match `D`.
The builder constructs the three input BlockSpecs and both output BlockSpecs;
the checked call validates the three concrete input shapes and dtypes. The
CPU-interpreter test compares noncausal attention and log-sum-exp with an
independent NumPy calculation, with Q and K having different lengths.

For a v2: the layout proves axis agreement and grid size but not the *values*
of arbitrary index-map callbacks or per-lane mask coverage. Its noncausal
interface does not expose optional segment IDs, causal attention, or the
optional absence of residual outputs, although the original kernel body
contains those branches. The general `Layout` container stores heterogeneous
`BlockSpec`s as objects after the typed factory constructs them; an
arity-independent mapping abstraction might retain the relationship without
a new factory per attention signature. Contrast Triton's tutorial: Pallas
can type distinct Q and K lengths, whereas the current Triton descriptor
path assumes same-length self-attention.

`test_layer_norm.py` preserves the forward kernel body from JAX's deprecated
`jax/experimental/pallas/ops/gpu/layer_norm.py` and runs it in CPU interpret
mode. Three `[Features]` input Refs produce one `[Features]` output Ref and
two scalar `[]` statistics Refs. `row_statistics_layout` binds all six Ref
shapes to the typed `out_shape` tuple; `checked_pallas_call` validates host
shapes and dtypes before JAX traces the kernel. An irregular five-element
row with four-element blocks exercises all three masked passes through the
unchanged kernel. There is no host stride restriction because Pallas Refs
represent logical indices rather than Triton-style pointer addresses.

The layout rejects a non-scalar statistic output shape in a focused runtime
test; the scalar JAX stubs describe `Array[[]]` and
`ShapeDtypeStruct[[]]`. The multi-output layouts still have kernel-specific
signatures. Generalizing checked layouts across heterogeneous output tuples
and optional kernel outputs requires more than the existing arity overloads.
Neither the loop's grid coverage nor its per-lane mask implication is proved
by these types. The Triton forward equivalent uses a row-per-program grid,
while this Pallas row kernel has an empty grid and could be vmapped across
rows in a separate host adapter.

`test_dropout.py` creates two Pallas analogues to Triton's dropout tutorial;
these bodies are not copied from an upstream Pallas example. An explicit
boolean keep-mask and the floating-point values have the same host length
and block mapping but different dtypes. `vector_layout` checks each input
dtype independently while preserving their shared shape. The seeded version
uses a single input Ref; its closure captures checked probability and integer
seed metadata, and folds the program ID into a JAX key before sampling a
tile-shaped Bernoulli keep-mask. The shared builder supports one or two inputs
and a partial final tile. CPU tests cover both forms and repeatability; the
Pallas and Triton PRNGs need not return identical masks. Types do not prove
that every program samples independently or that every accelerator accepts
the seeded kernel.
The seeded Pallas kernel samples with a program-specific key and a block
shape, rather than using each element's global index as its random counter.
Changing block width can therefore change which random value an element gets
even when the seed is unchanged. This distinction matters if a later design
promises reproducibility across launch configurations; it does not weaken
the checked host shapes. A JAX array already carries a shape in the local
stubs, so these layouts accept `jax.Array[Shape]` directly and validate the
concrete shape at launch without an intermediate host-array marker.

`test_vector_add.py` preserves the executable vector-add body from the v0
Pallas fixture and tests an explicit checked boundary on CPU. Its
`design_doc_add(x, y)` packages the inline `pallas_call` in JAX's
`docs/pallas/design/design.md` into a named callable for comparison: the
design doc does not define a named host wrapper. It keeps the fixed 8-element
shape, int32 output, two-element blocks, and four-program grid. The index maps
return `(i,)` instead of the doc's `i`, the input specs are a tuple instead of
a list, and `grid=(pl.cdiv(8, 2),)` spells out the doc's `(4,)`; these forms
fit the current v1 stubs while preserving the mapping. `interpret=True` allows
CPU testing. `checked_add(x, y, block_size=2)`
also runs the same 8-element inputs, declaring their int32 output dtype.

The
`pl.InRef[[Block]]` and `pl.OutRef[[Block]]` annotations remain on the kernel
at runtime; no annotation-stripping decorator is needed. The checked host
function explicitly supplies the output shape, grid, and shared index-map
`BlockSpec` through `vector_layout`; the shared `checked_pallas_call` relates
the kernel and layout to validated host inputs. The fixture checks complete tiles, a 10-element input with
a partial final tile, empty arrays, and invalid host inputs. Its kernel
annotations are checked using the
independent `../pallas_library/pallas-stubs/` overlay.

Run from the v1 directory:

```sh
../.venv/bin/pyrefly check -c pyrefly.toml
../.venv/bin/python -m unittest discover -s pallas_examples -t . -p 'test_*.py'
JAX_DISABLE_JIT=1 ../.venv/bin/python -m unittest discover -s pallas_examples -t . -p 'test_*.py'
```

`test_masked_softmax.py` preserves the body of JAX's
`jax/experimental/pallas/ops/gpu/softmax.py` kernel and uses its explicit
`out_shape`, `grid=()`, `compiler_params`, and `vmap` host pattern. The Pallas
kernel receives a **one-dimensional row**; `jax.vmap` maps that row operation
over the leading matrix dimension. The v1 example supports a 2D input and
uses `interpret=True` to run on CPU. A typed nested kernel closure replaces
upstream's `functools.partial` so Pyrefly can check the bound input and output
ref dimensions. `row_layout` keeps the row kernel's Ref signature and
matches both Ref shapes to `out_shape`. The `vmap` callback passes its
shape-typed JAX row directly to the checked callable and returns the one-row
result; the narrowly typed `vmap` stub carries the trailing dimension
through to the 2D return type. Shared `checked_pallas_call` validates shape
and dtype even inside `vmap`; tracers do not expose a concrete device for a
device check. The original
`@jax.jit` wrapper is omitted from this CPU-focused test. The stub does not
prove that `next_power_of_2(row_len)` covers every column, and this example
does not prove that the comparison creating the mask protects every address
used by the load and store. Its typed `vmap` rule assumes the callback maps
independent rows; zero-column inputs, arbitrary mappings, and GPU execution
are not covered by these tests.

`test_blocked_matmul.py` preserves the one-line matmul kernel from JAX's
Pallas quickstart. It types two different 2D input Refs and a 2D output Ref;
the handwritten boundary passes distinct left, right, and output maps and a
two-axis grid into `matmul_layout`, which creates Pallas `BlockSpec`s.
`checked_pallas_call` then checks host dimensions, dtype, and device at launch;
the layout constructor checks the declared output shape, block shapes, and grid size.
CPU tests cover a correctly mapped matmul and reject incompatible inner
dimensions or partial output tiles. Contextual typing rejects a left map
that selects row block zero instead of `i`. A negative experiment deliberately
bypasses that type check and shows the resulting matrix is wrong; arbitrary
index-map behavior and external casts remain outside the static guarantee.

`grid_context_probe.py` is a type-only experiment for a future checked layout
API. A single generic call receives the grid dimensions, three block shapes,
and three index-map lambdas. Pyrefly contextually types the lambdas from the
grid's `GridSize` parameters: the valid matmul maps type-check, and replacing
the left map's row index with zero produces a `bad-argument-type` diagnostic
(suppressed in the fixture so project checking stays clean). This establishes
that ordinary lambdas can work for the simple index-preservation rule; the
runtime layout constructor builds Pallas `BlockSpec`s from those typed maps.

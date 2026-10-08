# Pallas examples

`test_layer_norm_backward.py` keeps JAX's GPU layer-norm input-gradient
kernel body unchanged. A typed `row_input_gradient_layout` relates four
same-length input rows, two saved scalar statistics, and a same-length output
gradient, using the empty grid and Pallas's full-row Refs. The CPU interpreter
checks a partial final block against the independent layer-norm derivative.
The wrapper validates input shapes and dtypes at launch; it does not prove
that saved mean and reciprocal standard deviation were computed from this
particular input. Pallas's weight-gradient kernel remains a separate example.

`test_attention_forward.py` retains JAX's deprecated GPU `mha_forward_kernel`
and the optional segment-mask helper unchanged, but exercises the noncausal,
unsegmented branch. `attention_layout` relates Q `[B,Q,H,D]`, K/V `[B,K,H,D]`,
output `[B,Q,H,D]`, and base-2 log-sum-exp `[B,H,Q]` to a
`(Q/block_q, B, H)` grid. Its typed callback receives squeezed query
`[block_q,D]` and full-key/value `[K,D]` Refs plus a `[block_q]` stats Ref.
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

For v2, test the statistic-output shape mismatch as well as input mismatch:
the scalar JAX stubs now describe `Array[[]]` and `ShapeDtypeStruct[[]]`,
but the current multi-output layouts have kernel-specific signatures. The
row-statistics layout is deliberately specific to this kernel; generalizing
checked layouts across heterogeneous output tuples and optional kernel outputs
requires more than the existing arity overloads.
Neither the loop's grid coverage nor its per-lane mask implication is proved
by these types. The Triton forward equivalent uses a row-per-program grid,
while this Pallas row kernel has an empty grid and could be vmapped across
rows in a separate host adapter.

`test_dropout.py` creates two Pallas analogues to Triton's dropout tutorial;
these bodies are not copied from an upstream Pallas example. An explicit
boolean keep-mask and the floating-point values have the same host length
and block mapping but different dtypes. `vector_layout` now checks each input
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
Pallas fixture and adds a CPU test for an explicit checked boundary. Its
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
/home/stroxler/.kernel-shapes-venv/bin/python -m unittest -v pallas_examples.test_vector_add
/home/stroxler/.kernel-shapes-venv/bin/python -m unittest -v pallas_examples.test_masked_softmax
/home/stroxler/.kernel-shapes-venv/bin/python -m unittest -v pallas_examples.test_blocked_matmul
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

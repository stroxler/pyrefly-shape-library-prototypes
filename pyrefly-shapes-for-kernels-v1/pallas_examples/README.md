# Pallas examples

`test_vector_add.py` preserves the executable vector-add body from the v0
Pallas fixture and adds a CPU test for an explicit checked boundary. Its
`design_doc_add(x, y)` packages the inline `pallas_call` in JAX's
`docs/pallas/design/design.md` into a named callable for comparison: the
design doc does not define a named host wrapper. It keeps the fixed 8-element
shape, int32 output, two-element blocks, and four-program grid. The index maps
return `(i,)` instead of the doc's `i`, the input specs are a tuple instead of
a list, and `grid=(pl.cdiv(8, 2),)` spells out the doc's `(4,)`; these forms
fit the current v1 stubs while preserving the mapping. `interpret=True` allows
CPU testing. `checked_add(as_pallas_input(x), as_pallas_input(y), block_size=2)`
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
matches both Ref shapes to `out_shape`. The `vmap` callback explicitly calls
`as_pallas_input(row)` before invoking the checked callable, then returns its
one-row result; the narrowly typed `vmap` stub carries the trailing dimension
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

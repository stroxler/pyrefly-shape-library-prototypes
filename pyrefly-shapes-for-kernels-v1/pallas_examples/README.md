# Pallas examples

`test_vector_add.py` preserves the executable vector-add body from the v0
Pallas fixture and adds a CPU test for the generated boundary. Its
`design_doc_add(x, y)` packages the inline `pallas_call` in JAX's
`docs/pallas/design/design.md` into a named callable for comparison: the
design doc does not define a named host wrapper. It keeps the fixed 8-element
shape, int32 output, two-element blocks, and four-program grid. The index maps
return `(i,)` instead of the doc's `i`, the input specs are a tuple instead of
a list, and `grid=(pl.cdiv(8, 2),)` spells out the doc's `(4,)`; these forms
fit the current v1 stubs while preserving the mapping. `interpret=True` allows
CPU testing. The generated wrapper also runs the same 8-element inputs after
converting them to its current float32-only boundary.

The
`pl.InRef[[Block]]` and `pl.OutRef[[Block]]` annotations remain on the kernel
at runtime; no annotation-stripping decorator is needed. The fixture checks
complete tiles, a 10-element input with a partial final tile, empty arrays,
and invalid host inputs. Its kernel annotations are checked using the
independent `../pallas_library/pallas-stubs/` overlay.

Run from the v1 directory:

```sh
../.venv/bin/pyrefly check -c pyrefly.toml
/home/stroxler/.kernel-shapes-venv/bin/python -m unittest -v pallas_examples.test_vector_add
```

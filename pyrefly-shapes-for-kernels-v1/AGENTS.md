# Kernel-shape v1 working notes

- Keep runtime library code separate from example kernel bodies. Triton and
  Pallas each have their own `*_library/` and `*_examples/` directories.
- `triton_library/triton-stubs/` and `pallas_library/pallas-stubs/` are v1-owned
  overlays derived from v0. Gluon stays in v0; change v1 stubs without
  mutating v0.
- Preserve original executable kernel statements when prototyping semantic
  parameter annotations; inspect the upstream source named by each example.
- A checked boundary must validate host shape/layout/device and declared output
  dtype before launching, and distinguish metadata it assumes from properties
  checked in the kernel body.
- Run `../.venv/bin/pyrefly check -c pyrefly.toml` from v1 and the relevant
  example's two-mode unittest suite in `README.md`. Put fixture-specific tests
  beside each example kernel; keep reusable compiler helpers in `testing.py`.
  The project config
  excludes stub overlays from project-wide diagnostics but uses them for imports.
- Triton and Pallas vector-add both demonstrate explicit checked boundaries.
  Pallas accepts semantic annotations without Triton's source rewriting.

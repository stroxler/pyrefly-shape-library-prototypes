# Working on the kernel shape prototype

This experiment asks how far Pyrefly's existing type system can validate
semantic host-array and tile contracts for unchanged Triton and JAX Pallas
kernel bodies. Triton is the
first priority, Pallas the second; CuTe is paused and kept locally only.

- Keep executable kernel bodies identical to their cited upstream examples.
  Change annotations, static stubs, negative controls, and documentation to
  explore the contract. Triton annotations are intentionally illegal at JIT
  runtime; check these fixtures statically rather than executing them.
- `# E:` comments assert expected diagnostics. A passing expectation suite
  need not have zero errors. Describe whether a boundary property is checked
  in the body, declared by a stub, enforced by host runtime validation, or
  still unverified. Do not widen an overload just to suppress an error.
- The language `pyrefly.toml` files select the repository's `.venv` and local
  stub overlays. Pyrefly, shape extensions, and Torch stubs use `1.4.0.dev3`.
  From an example directory, run `../../.venv/bin/pyrefly check -c
  pyrefly.toml --expectations tests/test_*.py`; see the v0 README. The local
  Python is 3.12. Running Pallas kernels in the CPU interpreter is optional
  and requires installing JAX. Torch host-boundary probes require the actual
  Torch package alongside the partial Torch shape stubs to recover all their
  expected diagnostics.
- This directory's `pyproject.toml` records optional runtime dependencies as
  well as the Pyrefly tools; fixture-specific `*-stubs/` folders are static
  overlays. No Pyrefly core fork is part of this repository. The fbsource
  original draft stack was left untouched.
- See `kernel-boundary-prototype.md` for wrapper-generation requirements.
  Ordinary Triton's 15 compilation-pipeline scripts have body fixtures for
  all 22 JIT definitions, but CPU matmul layouts 03/04/08 remain gaps.
  Pallas MHA backward has a typed host contract and CPU tests but seven
  expected body errors due to list-shaped zero initialization.
- Do not publish copied source fixtures before reviewing their upstream
  license notices and provenance. The locally copied CuTe directory is
  ignored by Git pending that review.

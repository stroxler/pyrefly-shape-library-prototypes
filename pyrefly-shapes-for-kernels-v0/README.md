# Pyrefly shapes for kernels: v0

Can Pyrefly describe the shape contract between Python arrays and an accelerator
kernel, then find evidence for that contract in the kernel body? This experiment
uses Pyrefly's existing type system, semantic parameter annotations, and local
stub overlays; it does not modify Pyrefly core. The executable statements of
the copied kernels remain unchanged. Some annotations are intentionally
**illegal at kernel runtime**: these are static-analysis fixtures, not an API
for writing deployable Triton kernels.

The long-term goal is a declaration that describes host shapes and layout
requirements, explains how programs reach tiles of those arrays, and can
eventually generate a safe Python launch wrapper. Runtime checks or explicit
input conversions can enforce properties that Python types do not express,
including strides, device, dtype, and alignment. That wrapper generator is
*not* implemented here. An accepted annotation alone is not evidence of a
safe interface: the experiments distinguish checks grounded in loads and
stores from host-side promises, runtime checks, and properties still unproved
(such as general grid coverage).

This is a standalone copy of the Pyrefly kernel-typing experiments. The original
fbsource draft stack is unchanged. The example directories have static-only
semantic stubs, annotated copies of kernel bodies, negative checks, and detailed
notes. Run Pyrefly on the Triton fixtures; do not run their decorated kernels.

The language guides describe distinct approaches:

- [Triton](pyrefly-triton-examples/README.md), the primary target: explicit
  program IDs, pointer offsets, masks, and tiles.
- [JAX Pallas](pyrefly-pallas-examples/README.md), the second target:
  `BlockSpec` and `pallas_call` supply much of the host-to-block mapping.
  Several fixtures also run in JAX's CPU interpreter.
- CuTe DSL (`pyrefly-cute-examples/README.md`), exploratory and currently
  local-only: layouts, strides, nested tile/rest views, and copy fragments.
  This directory is excluded from Git pending source-provenance review.

The [host-boundary note](kernel-boundary-prototype.md) separates properties
validated inside the kernel from host-side validation or sanitization that a
future generated wrapper would need. For a closer reading, start with the
[Triton vector add](pyrefly-triton-examples/tests/test_vector_add.py) and the
[Pallas vector add](pyrefly-pallas-examples/tests/test_vector_add.py); each
directory has a README describing coverage and known gaps.

The [kernel-contracts guide](kernel-contracts-guide.md) compares the experiments
and their remaining proof obligations. [AGENTS.md](AGENTS.md) provides a
shorter working guide for agents continuing this kernel experiment.

The local [pyproject.toml](pyproject.toml) pins Pyrefly, its shape extensions,
and its Torch stubs to `1.4.0.dev3`. These configs resolve `shape_extensions`
and host libraries from the selected Python interpreter, not a neighboring
fbsource checkout. With `~/.kernel-shapes-venv` already installed:

```sh
cd pyrefly-shapes-for-kernels-v0/pyrefly-triton-examples
~/.kernel-shapes-venv/bin/pyrefly check -c pyrefly.toml \
  --python-interpreter-path ~/.kernel-shapes-venv/bin/python \
  --expectations tests/test_vector_add.py

cd ../pyrefly-pallas-examples
~/.kernel-shapes-venv/bin/pyrefly check -c pyrefly.toml \
  --python-interpreter-path ~/.kernel-shapes-venv/bin/python \
  --expectations tests/test_vector_add.py
~/.kernel-shapes-venv/bin/python -m unittest discover -s tests \
  -p 'test_vector_add.py' -q
```

`--expectations` checks the negative-control diagnostics embedded in fixtures;
success does not mean the output has zero errors. The long example READMEs
retain their detailed exploration notes, with commands adjusted for this
standalone venv; the commands above are the supported starting point here.
Four Triton expectation comments were removed because the released Pyrefly
build does not emit those unreachable-code diagnostics. No kernel body's
executable statements were changed for this copy.
The CuTe folder is available locally but ignored for public Git publishing
pending a source-provenance review. In particular, full host launch safety,
Triton GPU execution, and all kernel-body relationships are not established.

This is a research prototype, not a safe launch API or drop-in typing package.
Before publishing copied Triton and JAX kernels, review upstream license
notices and attributions. The local CuTe copy must also pass a source-provenance
and redistribution review before it is added to Git. No files have been
published from this working tree by this setup.

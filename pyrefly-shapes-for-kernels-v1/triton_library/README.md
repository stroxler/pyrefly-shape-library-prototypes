# Triton library

`semantic_jit.py` translates static-only semantic annotations into Triton's
runtime signature and source. `torch_wrapper.py` builds a validated 1D Torch
boundary using those preserved annotations plus explicit launch metadata,
including the output dtype. The wrapper checks matching device, length, and
layout for inputs but leaves their dtypes unconstrained; Triton specializes
them when launching the kernel.
The `triton-stubs/` overlay is local to v1 and can diverge from v0. Its
ordinary Triton stubs were copied from v0; Gluon stubs are out of scope here.
The first executable usage is in `../triton_examples/`.

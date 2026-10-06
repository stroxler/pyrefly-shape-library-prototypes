# Pyrefly shape library prototypes

An open-ended workspace for experiments with semantic shape types and Pyrefly.
Each prototype lives in its own directory with its own goals, modeling notes,
and instructions. Kernel authoring is one exploration, not a constraint on
future experiments; a semantically typed linear algebra library, for example,
would be a separate prototype here.

Current experiment: [shapes for kernels, v0](pyrefly-shapes-for-kernels-v0/README.md).
It explores contracts between host arrays and Triton, JAX Pallas, and CuTe DSL
kernels. Its README explains the research question, evidence, limitations, and
how to run the fixtures. Its dependencies and agent instructions live alongside
the experiment, leaving the repository root available for other prototypes.

[Shapes for kernels, v1](pyrefly-shapes-for-kernels-v1/README.md) tests whether a
decorator can hide semantic kernel annotations from Triton's runtime signature
while preserving them for Pyrefly, without modifying kernel bodies. Triton and
Pallas each have their own v1 library, example directory, and independent copy
of the relevant v0 stubs for future experiments; Gluon remains in v0 only.

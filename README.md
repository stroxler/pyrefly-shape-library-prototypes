# Pyrefly shape library prototypes

An open-ended workspace for experiments with semantic shape types and Pyrefly.
Each prototype lives in its own directory with its own goals, modeling notes,
and instructions. Kernel authoring is one exploration, not a constraint on
future experiments; a semantically typed linear algebra library, for example,
would be a separate prototype here.

Current experiment: [shapes for kernels, v1](pyrefly-shapes-for-kernels-v1/README.md).
It explores checked contracts between host arrays and Triton and JAX Pallas
kernels. The [boundary/corpus guide](pyrefly-shapes-for-kernels-v1/BOUNDARY_AND_CORPUS.md)
explains which shape relationships the examples exercise; the
[Gluon/CuTe notes](pyrefly-shapes-for-kernels-v1/OTHER_KERNEL_DSLS.md) retain
lessons for a future prototype without mixing those DSLs into v1. Its
dependencies and agent instructions live alongside the experiment, leaving
the repository root available for other shape-library experiments.

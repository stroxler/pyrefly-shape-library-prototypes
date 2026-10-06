# Pallas library

`pallas-stubs/` is an independent copy of the v0 JAX/Pallas stub overlay,
included on v1's Pyrefly search path. `jax_wrapper.py` generates a checked
one-dimensional host callable from a kernel whose parameters are semantic
`pl.InRef[[Block]]` and `pl.OutRef[[Block]]` annotations. Unlike Triton, Pallas
accepts these annotations directly, so the library does not rewrite source or
wrap the kernel in a JIT decorator.

`Launch1D` supplies the tile size and execution mode. The wrapper verifies
that all references share a symbolic block dimension, accepts the input
references as host arguments, checks 1D shape, length, float32 dtype, and
device at runtime, and derives the output allocation, aligned `BlockSpec`s,
and ceil-divided grid. This is intentionally restricted to one output and
float32 inputs; unsupported annotations fail at wrapper construction. JAX
arrays do not expose the Torch-style element-stride contract checked in the
Triton experiment. The inferred alignment of input and output tiles depends
on the `Launch1D` index-map assumption; the types do not prove index-map
coverage or the kernel's indexing behavior. The dynamic wrapper is not yet
given a precise host-side static function type.

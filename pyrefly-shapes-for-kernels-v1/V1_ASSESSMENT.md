# What the v1 kernel-shape prototype checks

This is a prototype for describing a Python-facing shape contract and checking
that a kernel uses at least some of the same dimensions. It adds library code,
local stubs, semantic parameter annotations, and checked host adapters; it
does not modify Pyrefly core. Kernel algorithms are preserved, apart from
documented mechanical pointer-local renamings needed to retain lower-rank
types. It is not a formal proof that a launch computes every output correctly.

## Corpus and evidence

- [Triton examples](triton_examples/README.md) include every JIT kernel body
  in the top-level numbered tutorials 01–12 in the version surveyed, including
  the separately numbered warp-specialized attention tutorial. Tutorial 15
  needs frontend features absent from the installed PyPI Triton and is not
  part of this corpus. Additional unnumbered tutorial trees and Gluon remain
  outside v1. Coverage of a kernel body does not imply coverage of every
  upstream host wrapper, branch, or accelerator feature.
- [Pallas examples](pallas_examples/README.md) include vector operations,
  norms, matrix multiplication, attention forward and backward, a grouped
  ragged dot from Marin, TPU kernels, and a Mosaic GPU pipeline. Pallas
  dropout bodies are parallel exercises rather than upstream ports. This
  is a representative sample, not an exhaustive JAX or Pallas corpus.
- Pyrefly checks the examples with local Triton and JAX/Pallas stub overlays.
  Type-only negative examples exercise rejected shapes, strides, reduction
  dimensions, and some mask/grid-axis mismatches. The runnable fixtures use
  Triton's frontend or interpreter and Pallas's CPU interpreter where
  available. Some hardware-only paths check host contracts and frontend
  acceptance or inspect launcher metadata, without executing on a GPU/TPU.
  Diagnostics suppressed for deliberate negative probes must not be read as
  unchecked implementation errors; neither is zero remaining diagnostics a
  measure of semantic completeness.

## The boundary: what is checked, and by whom

| Relationship | Static checks | Runtime checks or trusted inputs |
| --- | --- | --- |
| Host input/output dimensions | The handwritten, shape-typed host signatures relate axes such as matmul `[M,K] × [K,N] → [M,N]`. Typed Pallas layout factories constrain their kernel Ref signatures and result types. | Adapters validate concrete input shapes; Triton adapters also validate element strides. Output shape and dtype are explicitly allocated or supplied to Pallas. An untyped caller needs these checks. |
| Launch metadata and Ref/tile shape | Triton output-layout types relate grid and tile parameters to symbolic host dimensions; Pallas layout factories relate declared Ref blocks and host axes. The checker sees a constrained callable, not an arbitrary `Any` launch. | Grid sizes, block sizes, dtypes, devices, and required divisibility are validated where the particular adapter requires them. Pallas's general `Layout` stores heterogeneous `BlockSpec`s as objects, and `checked_pallas_call` trusts the typed factory when it casts the underlying call's result. |
| Kernel access to declared axes | Triton pointer/offset/load/store overloads retain some allocation shape, strides, tile width, and mask categories. Pallas Ref/tile operations retain block dimensions and check contraction-axis compatibility. | Kernel parameter annotations assert intent. Stubs do not prove that an index-map callback, program ID, or pointer expression addresses the intended elements of a concrete allocation. |

Triton has two complementary paths. Handwritten checked wrappers validate
host tensors before presenting them as kernel pointer views; for evaluable
pointer annotations, `semantic_jit` also checks concrete pointer shapes,
strides, and related scalar launch arguments immediately before a direct
launch. The hook alone does not validate the launch grid, dtype, shared
device, pointer direction, or all metadata constraints; those belong to
the handwritten checked adapter and typed launch layout. The hook does not
cover every stub-only annotation (notably some attention descriptors), so
the checked adapter is the reliable boundary for those examples. The
runtime pointer views are still the original Torch objects; they do not
turn PyTorch tensors into actual Triton pointers.
Triton's `ConstExpr` is runtime annotation metadata, not a separate
compile-time symbolic-integer proof in Pyrefly.
Both paths reject overlapping writable tensor views. This check handles
one- and two-dimensional strides, including column-major and padded rows;
the higher-rank case uses a conservative disjoint-span condition, which
may reject some nonoverlapping sparse layouts. Neither path checks that
distinct output arguments do not alias *each other*.

Pallas receives logical Ref tiles selected by `BlockSpec`s rather than raw
host addresses. The type information is split across an annotated kernel,
typed layout factories, and the checked call. Each factory checks a known
pattern's host-to-Ref correspondence; the generic binding assembler alone
does not infer arbitrary kernel signatures. A JAX `vmap`/tracer may expose
shape and dtype without a concrete device, and adapters that materialize
data on the host (such as ragged-dot group sizes) are not JIT-traceable.
`checked_pallas_call` validates inputs; it declares the output through
`out_shape` and does not independently inspect the returned output.

## Inside the kernel: useful guarantees and limits

Vector add rejects several wrong-size or mismatched 1D load/store masks;
the 1D Triton rules track the known `program_id` axis separately from
offset origin. Strided copy checks that logical offsets are multiplied by
the pointer's element stride. Selected row pointers lose a dimension,
making an unselected matrix pointer plus column offsets a type error.
Matmul tiles retain their contraction dimensions; Pallas's ragged dot also
checks the orientations and dimensions of its masked K loads. These are
meaningful body checks, not merely descriptions of a foreign-function
interface. Some advanced paths still rely on specialized semantic roles,
targeted overloads, or local suppressions rather than a small composable
set of primitives.
Adding a raw program ID directly to an allocation pointer is typed only
for a unit-stride vector: a strided vector requires an explicit scaled
offset. A negative probe rejects a raw program ID on strided and rank-two
pointers.

The following are **not** proved:

- Exact mask implication: equal shape, allocation length, or axis tags do
  not establish that every enabled lane points in bounds. Value validity
  after a masked load is not tracked. The 2D rules lose some axis provenance.
- Complete or correct output coverage: the checked grid does not prove
  `program_id` arithmetic, a grouped scheduler, a Pallas `index_map`
  callback, or a reduction-pointer update visits the intended tiles exactly
  once. In particular, Pallas index-map lambdas are not symbolically
  evaluated, even when simple axis mistakes are statically rejected.
- Data-dependent alignment: types alone do not establish that ragged-group
  boundaries match the RHS selected by each program, that saved attention
  statistics came from the current inputs, or that a loop's state was
  initialized and advanced at the right time. Adapters can validate some
  concrete conditions, such as nonnegative ragged group sizes summing to
  the row count, without making them static properties.
- Hardware behavior: virtual-target Triton frontend compilation and CPU
  interpretation do not validate GPU lowering, tensor descriptors, TMA,
  numerical behavior in uncovered branches, or TPU execution. Library API
  drift may require test-local substitutions or an upstream port without
  changing the shape-typing conclusion.

## Next experiments

1. **Triton addressing, with explicit trust points.** The
   [addressing design](TRITON_ADDRESSING_DESIGN.md) proposes a shared
   host/kernel grid intent, a reviewed PID claim, semantic operators for
   tile coordinates and stride-specific steps, and a few identity helpers
   where operators cannot propagate those roles. Start with grouped matmul;
   require negative tests for swapped strides, wrong grid axes, incorrect
   K advances, and mismatched pointer/mask origins. These would be v2
   changes to the current rule of preserving kernel bodies.
2. **Make the safety boundary explicit across all Triton launch forms.**
   Test the typed host-to-kernel arguments even when semantic annotations
   are stub-only or a direct launch bypasses the handwritten adapter. A
   composable layout and an explicit checked-launch entry point may be
   simpler than depending on Triton's private source/launch hooks. Keep
   runtime layout validation for constraints the type system cannot express.
   Test compiler launch options, aliased annotation forms beyond direct
   marker imports, defaulted kernel parameters, and distinct output tensors
   that alias the same storage. The present checked-launch binder handles
   kernel parameters, not arbitrary Triton launch-only options.
3. **Challenge composability with Pallas's varied Ref layouts.** The
   [semantic-type experiments](NEXT_SEMANTIC_TYPE_EXPERIMENTS.md) propose
   reusing host-axis, block-axis, grid-axis, and index-map bindings across
   multiple input and output arities. Vector add, matmul, heterogeneous
   attention outputs, optional arguments, and ragged grouping should share
   a builder only if it preserves the present static shape guarantees.
   Investigate a typed parameter-list mapping or narrow checker hook if
   pattern-specific factories remain unavoidable. This is useful design
   work, but does not need to precede a Triton-focused v2.

These experiments prioritize a trustworthy, inspectable Python-to-kernel
shape contract and helpful internal diagnostics. Full bounds, scheduling,
and numerical correctness are distinct, substantially harder goals.

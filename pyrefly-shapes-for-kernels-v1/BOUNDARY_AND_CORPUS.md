# Reading the kernel contracts and choosing future examples

The purpose of v1 is to check the relationship among (1) the host allocation
and Python-facing signature, (2) launch metadata, (3) the Ref/pointer visible
to one program, and (4) the operations actually performed by that program.
These are different contracts. A correctly typed `tl.load` does not establish
that the Python caller supplied the declared allocation; an accurately shaped
output annotation does not prove that all programs wrote every output tile.
The [assessment](V1_ASSESSMENT.md) lists the precise limits of the current
Triton and Pallas implementations.

## A route through the examples

Start with [Triton vector add](triton_examples/test_vector_add.py): the
allocation is `[N]`, one program forms `[Block]` offsets and compares those
offsets with `n_elements: Int[N]`. The resulting mask has both the allocation
bound and tile shape, so changing only the Y allocation's declared length
breaks its original masked load. `tl.tensor` is the local value tile, not the
Torch allocation. The checked Torch adapter separately tests physical shape,
strides, dtype and device and derives the length and grid. The
[CPU tiled add](triton_examples/test_cpu_vector_add_tiled.py) preserves this
negative test even when a program visits *several* tiles in a loop.

Compare [Pallas vector add](pallas_examples/test_vector_add.py): `BlockSpec`
and `pallas_call` present blocks of whole host arrays as `Ref[[Block]]`, so
tile address calculation moves into the index-map callback and Pallas runtime.
The checked layout connects equal input/output lengths, block size and grid;
the body checks value-tile addition. Partial Pallas writes need no explicit
Triton-style pointer mask, but this does not prove an arbitrary index map
correct. [Pallas iota](pallas_examples/test_iota.py) provides a different
case: `grid=(Length,)` directly indexes a full `OutRef[[Length]]` with a
program ID, with no input, tiled Ref or block spec. Neither architecture
should assume every Pallas grid axis is a `ceil(Length/Block)` tile grid.

The [Pallas shard-map boundary](pallas_examples/test_shard_map_boundary.py)
adds another axis: a global `[Devices*Rows,Cols]` array becomes a local
`[Rows,Cols]` callback argument, and `[Devices,Rows,Cols]` callback output
becomes `[Devices*Devices,Rows,Cols]` globally. This is a host/local shape
relation distinct from a tile's element strides. A typed `shard_map` can
reject mismatched local shapes; a runtime mesh and partition still need
validation. See [the Gluon and CuTe notes](OTHER_KERNEL_DSLS.md) for related
but nonidentical host/view boundaries.

For harder bodies, see [Triton matmul](triton_examples/test_matrix_multiplication.py)
and [Pallas blocked matmul](pallas_examples/test_blocked_matmul.py). Check
whether the wrong K is rejected at an original contraction operation or just
in a separately declared adapter. Grouped PID arithmetic and arbitrary
`BlockSpec` index-map values are not certified by their symbolic shape tags.

## Checked launch layers

The v1 runtime adapter does not aim to hide the grid or specs. It can ask the
author for those components, check shape/stride relationships against actual
host arrays, and then present a simpler *typed* host signature. In Triton,
the interpreter and virtual-target frontend let us exercise a source-stripped
`@semantic_jit` without a GPU; real device execution remains untested. In
Pallas, annotations can remain in the ordinary Python function and
`checked_pallas_call` validates launch metadata and concrete host inputs.
The generic Pallas `binding_layout` handles multiple named input and output
axes at runtime but cannot yet statically map arbitrary parameter tuples to
`Ref`s. Specialized typed factories fill that gap selectively.

The split between static and runtime validation is deliberate: host strides,
device, dtype, alignment and supported block sizes can be checked or sanitized
at launch even if ordinary `torch.Tensor`/`jax.Array` stubs do not represent
them. A dynamic wrapper generated from a declarative contract might take
input/output shapes, allocation policy, metadata such as grid/block/specs,
and invariants such as divisibility or dtype, but it must preserve the
kernel-to-host symbolic dimension mapping. If Pyrefly cannot infer that
mapping through the ordinary library call, explicitly specifying and checking
*both ends* is more useful than an unchecked FFI.

A generated boundary would validate requirements *before* launching or
allocating dependent descriptors, bind actual shapes to symbolic dimensions,
choose kernel-only constexpr/block values, inject size and stride scalars,
construct Pallas specs or Triton descriptors, and allocate outputs using a
declared dtype/device, alias and initialization policy. An input may be
rejected for noncontiguity or explicitly copied, but a copy changes aliasing
and performance. Warp-specialized attention's atomic accumulation into dQ
needs a **zero-initialized** output, not merely a correctly shaped one.
The CPU matmul tutorial's padding helper shows why a divisibility assertion
after an unmasked kernel launch cannot make that launch safe. Input/output
roles must also survive descriptor construction: equal-sized Q and O
descriptors cannot recover their role from their shapes alone.

For every claim, distinguish body-checked (a negative shape mutation fails
at an original operation), bridge-checked (a host value is tied to that
kernel parameter), runtime-enforced (the wrapper checks or transforms it),
and trusted (the checker cannot establish it). A future mapping primitive
may need to relate the whole host parameter tuple or output tree to a kernel
parameter tuple and its specs; multiplying fixed-arity `pallas_call`
overloads is not a general solution, and `MapIntTuples` alone cannot express
a relationship among different inputs and outputs. User-facing declarations
could start on the host side and derive kernel semantic types internally;
v1's kernel annotations test the semantics without choosing that syntax.

## Scope of the consolidated examples

The top-level numbered Triton tutorials 01–12 have all their JIT kernel
bodies represented in v1, including helpers and multiple variants. The
numbered 15 multi-CTA example requires `tl.range(multi_cta=True)`, which the
installed PyPI Triton 3.8 frontend does not accept; it is not in the standard
website tutorial sequence. The separate CPU tutorial family is not fully
ported: tiled vector add is a representative extra case, while CPU GEMV,
matrix padding and CPU fused softmax remain candidates if they expose a new
v1 semantic failure. Compilation-pipeline tests (transposes, reductions,
coalescing, precision and warp-specialization examples) are valuable
compiler-layout probes, but are not additional numbered tutorial coverage.
Several separate unnumbered warp-specialized attention fixtures likewise
exercise architecture-specific lowering; v1 covers the numbered attention
and warp-specialized bodies without promising that all those variants run.
The Gluon case studies are recorded in [OTHER_KERNEL_DSLS.md](OTHER_KERNEL_DSLS.md),
not supported by the v1 Triton overlay.

Pallas is a representative collection rather than an exhaustive fixture
port. Existing v1 attention, matmul, dropout, layer norm, RMSNorm, ragged dot,
TPU/GPU and pipelined examples illustrate Refs, squeezed/block axes,
multiple outputs, masking and memory/pipeline roles. The program-indexed
output and global/local sharded-array fixtures retain two additional
independent contracts. Useful further candidates from the JAX/Marin
ecosystem, when driven by a specific user, include:

- Short convolution's `[Batch,BlockSeq+Width-1,Channels]` halo assembled
  from padding plus a host slice; a symbolic halo is different from an ordinary
  tile, and static shape does not prove concatenation order or lag bounds.
- A transposed-RHS matmul whose source wrapper mistakenly takes N from the
  *post-transpose* RHS (K on rectangular arrays). Its `dot_general` contraction
  was not statically validated when the checker merged the contraction axes
  even under `transpose_rhs: Literal[True]`. Do not count a kernel signature
  as body-checked if the original contraction produces a diagnostic.
- GPU paged attention and TMA gather, where dynamic page/index contents and
  asynchronous copies introduce non-shape validity conditions; RMSNorm
  input/weight gradients for distinct input vs reduction contracts.
- TPU DMA collectives and SparseCore, where physical memory placement,
  global/local shard shape and temporal synchronization must be distinguished.

These are documented future fixtures, not claimed v1 coverage. A Pallas
prototype shared with a JAX team should explicitly disclose that distinction
and invite a representative user kernel. The [other DSL notes](OTHER_KERNEL_DSLS.md)
similarly preserve CuTe and Gluon design questions without importing another
stub overlay into v1.

# Host boundaries for the Triton and Pallas prototypes

This note records a possible direction, not an implemented wrapper generator or
a proposed public annotation syntax. The kernel fixtures remain experiments in
checking **unchanged kernel bodies** with semantic parameter types. A future
user-facing declaration might instead describe host arrays and the launch,
with Pyrefly deriving those kernel parameter types internally. The declaration
should be the single source for both projections so they cannot disagree.

The immediate question is whether a declaration could determine a *safe* host
boundary. It need not express every requirement in the host-language type
system: generated code could validate a property at runtime or deliberately
transform an input to satisfy it. The exploration should distinguish three
claims:

1. The host shape and layout restrictions can be stated accurately.
2. The boundary checks or establishes those restrictions before launching.
3. The unchanged kernel body and the program-to-tile mapping justify the
   restrictions and the claimed output shape.

None of these claims follows automatically from a type-checking kernel
signature. Unknown properties must remain explicit obligations, not be treated
as established merely because the surrounding stub accepts a call.

## Information a declaration would need

| Part | Representative information | Possible enforcement |
| --- | --- | --- |
| Host arguments | Related dimensions, input/output roles, dtype and device relationships | Host types where useful; runtime checks otherwise |
| Storage | Required strides or contiguity, alignment, aliasing, and any permitted copies | Runtime validation or an explicit sanitization policy |
| Outputs | Shape, dtype, device, allocation and alias policy | Generate allocations from checked input metadata |
| Kernel arguments | Which host dimensions/strides become scalar arguments, constexpr choices | Derive scalars and specialize the launch |
| Scheduling | Grid dimensions, program-ID-to-tile mapping, tile sizes and edge policy | Generate the launch; check mappings against accesses where possible |

This table is an inventory, not a claim that our present Torch/JAX stubs
express all its entries. Dtype, device, ownership, numerical index bounds,
and general grid coverage are not currently established by the shape types.
Sometimes a precondition is a choice: rejecting a noncontiguous input keeps
the original allocation, whereas copying it changes cost and aliasing. Either
can be safe if the declaration makes the policy explicit. A Triton-heavy host
codebase might eventually benefit from layout-aware Torch types, but runtime
validation and sanitization are sufficient for this prototype's goal.

## Triton: vector add as a boundary sketch

`pyrefly-triton-examples/tests/test_vector_add.py` annotates the two input
pointers and output pointer with the same full allocation length `N`, ties
`n_elements` to `N`, and types a program's tile with `Block`. Its unchanged
body constructs `pid * BLOCK_SIZE + arange(0, BLOCK_SIZE)`; the mask comparison
retains the `N` and `Block` relationships at both loads and the store. Wrong
input or output lengths and incompatible mask bounds are rejected by targeted
negative controls.

A host-facing declaration for this kernel could say:

- Accept two equal-length one-dimensional inputs. Require a supported dtype,
  compatible devices, and unit element strides; choose either rejection or a
  specified contiguous copy when an input does not meet those requirements.
- Allocate a fresh result of the same length and compatible dtype/device.
  Supply the actual length as `n_elements` and choose `BLOCK_SIZE = 1024`.
- Launch a one-dimensional grid of `ceildiv(N, BLOCK_SIZE)` programs, passing
  inputs and output in their declared pointer roles. For each program, the
  corresponding logical positions are `pid * BLOCK_SIZE + arange(BLOCK_SIZE)`.

The block-size choice and copy policy are launch policy, not consequences of
the input's shape. The current type overlay checks local pointer, mask and
value shapes but does **not** establish that comparison and access use exactly
the same offsets, prove that the grid writes every output element once, or
check host contiguity, dtype and device. The existing static-only host adapter
calls a typed kernel directly: it is not a working bracketed Triton launch.
The original tutorial wrapper is an example against which to compare a future
generated boundary, not code that the prototype must preserve or type as-is.

Matmul adds independently named `[M, K]`, `[K, N]` and `[M, N]` allocations,
six element-stride arguments, three tile sizes and a grouped program-ID
ordering. `pyrefly-triton-examples/tests/test_matrix_multiplication.py` checks
many of the dimension/stride relationships at original loads, `dot` and store.
It does not prove that the grouped ordering and grid visit each output tile
exactly once. A declaration must keep host shape/strides, per-program tile
mapping, and internal tile layout distinguishable rather than folding them
into one notion of “tensor shape.”

The CPU matmul tutorial's optional `pad_kernel` illustrates why enforcement
order matters. Its source wrapper can launch unmasked padding before its
divisibility assertion. A safe generated boundary would check compatible
input/output storage, positive valid block sizes, divisible row and column
extents, and a matching grid **before** that launch; writing those conditions
down after the call is not enough. The current overlay does not yet model the
padding kernel's unmasked address bounds or its block-pointer matmul.

Warp-specialized attention adds an allocation obligation that shape alone
cannot describe: its backward kernel uses `atomic_add` to accumulate dQ into
an output allocated with zeros. A declaration capable of generating a safe
boundary must select zero initialization for that output, not merely allocate
the right `[Batch, Heads, Tokens, Dim]` shape. The kernel-side zeroed-output
type is a promise made at this boundary; the prototype does not currently
prove the Torch allocation, dtype or device satisfies that promise.

The device-TMA attention entrypoint exposes a different boundary issue:
the helper converting a host pointer or an existing descriptor into a
`tensor_descriptor` returns no input/output role. Equal-shaped Q and O
descriptors cannot recover that role from their dimensions, so its calls
into a role-specific forward helper currently produce explicit diagnostics.
A possible stub-only direction is an invariant role carried by the host
pointer, descriptor, constructor and conversion helper, with an unknown role
for unannotated descriptors. That design still needs an actual Pyrefly test:
it must reject an input/output swap even when dimensions match, and reject
a wrong full host extent separately. The current generic constructor does
not verify that host extent, so this entrypoint does not yet establish a
safe descriptor boundary.

## Pallas: the launch is closer to a declaration already

Pallas usually gives the kernel a `Ref` instead of making it form global
pointers with `program_id` and a bounds mask. At `pallas_call`, the caller
declares output shape and launch geometry, and can supply input/output
`BlockSpec` values. Each `BlockSpec` provides a block shape and an index map
from grid coordinates to the part of an array exposed to the kernel. Other
Pallas kernels, such as the full-row layer-norm forward fixture, use an
implicit whole-array Ref with `grid=()` and no `BlockSpec`. The Pallas
vector-add fixture (`pyrefly-pallas-examples/tests/test_vector_add.py`)
relates host length, output length, block size, grid and all three Refs.
The blocked-matmul fixture
relates `[M, K] @ [K, N] -> [M, N]`, different input/output block mappings,
and a two-dimensional grid. Both have negative checks for incompatible shapes
or tile parameters and CPU interpreter tests. Their `pallas_call` overloads
are specialized to these examples, not a general checker for arbitrary Pallas
call sites or index maps.

This call site is a useful model for declarative metadata, but it is not a
proof that arbitrary index-map callbacks cover the arrays. In the vector-add
fixture, an index map that sends **every** program to block zero passes the
current static checks. The matmul fixture has an analogous accepted mapping.
Pallas's blocked-indexing edge behavior is also different from Triton's
explicit masks: padded reads and discarded out-of-bounds writes do not imply
that padded values are safe to use in a reduction. A safe host declaration
would still need shape/dtype/device and allocation relationships, any required
runtime constraints, and a justified grid-to-view map.

More complex Pallas examples separate host array shape from a Ref's tile shape,
memory space or scalar-prefetch role. GPU-specific Mosaic behavior in this
prototype is checked statically rather than on GPU hardware; CPU interpreter
tests and negative checks demonstrate particular paths, not every branch of
every specialized kernel. The layer-norm forward and input-gradient fixtures
also show a whole-row call with no `BlockSpec`: the input-gradient kernel reads
scalar mean and reciprocal-standard-deviation Refs supplied by the host, so
their rank and the feature-row extent remain separate boundary obligations.

The GPU attention-forward fixture exercises another distinction a generated
boundary would need to preserve: host arrays have batch and head axes, but
their `BlockSpec` values squeeze those axes so a kernel sees a two-dimensional
query tile and a full key/value sequence. The host declaration relates
`[Batch, Queries, Heads, Dim]` and `[Batch, Keys, Heads, Dim]` to a
three-dimensional grid, output plus log-sum-exp allocations, and these local
Refs. Its adapter checks block divisibility, a power-of-two head dimension,
and whether supplied segment IDs cover all query positions at runtime. The
body checks attention tile and mask shapes, while the precise
index-map coordinates and the segment-mask helper remain trusted contracts,
not properties established by the current checker.

The attention-backward fixture extends that declaration to eight host inputs
and three gradients: query and dQ are `[Batch, Queries, Heads, Dim]`, key,
value, dK and dV use `[Batch, Keys, Heads, Dim]`, and the log-sum-exp and
delta residuals use `[Batch, Heads, Queries]`. The GPU kernel sees padded
two-dimensional Refs and runs two different query/key scan granularities
over a *shared* `(Batch, Heads, KeyBlock)` grid. A safe host boundary checks
that both gradient outputs need the same number of programs, that scan block
sizes divide their corresponding sequence axes, and that the padded feature
dimension covers the actual feature dimension. The current adapter checks
these numerical conditions at runtime and connects the host arrays to the
Ref, output and grid shapes statically. Independent CPU interpreter tests
compare both causal and noncausal gradients with automatic differentiation.
The original body's list-shaped `jnp.zeros` initializers lose the positional
dimension relationships required to type its accumulators and stores, so the
fixture retains seven expected body errors; it does not yet justify those
output tiles from their stores. This distinction matters for wrapper
generation: the metadata can describe a boundary while proof of the whole
kernel implementation remains incomplete.

## How to use this note during the example sweeps

For each kernel, record the host arrays and output allocation, the shape and
storage preconditions, scalar/constexpr sources, program-to-tile mapping,
and the actions a host call would take when a precondition fails. Classify
each claim as *checked in the body*, *enforced or sanitized at the host
boundary*, or *unverified*. A shape relation stated only in a trusted stub
is not body evidence; a local masked access is not a proof of full grid
coverage. For dynamic or hardware-specific cases, note the exact unverified
obligation instead of broadening an overload until the fixture passes.

This is a test of whether enough information exists to generate a safe
boundary later, not a request to implement generation now. Triton remains the
priority for the kernel-body sweep; Pallas supplies useful comparisons and a
second practical target. CuTe is paused while these two prototypes advance.

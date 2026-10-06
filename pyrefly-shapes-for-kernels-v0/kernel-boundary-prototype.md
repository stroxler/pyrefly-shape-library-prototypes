# Host boundaries for the Triton and Pallas prototypes

This note records a possible direction, not an implemented wrapper generator or
a proposed public annotation syntax. The kernel fixtures remain experiments in
checking **unchanged kernel bodies** with semantic parameter types. A future
user-facing declaration might instead describe host arrays and the launch,
with Pyrefly deriving those kernel parameter types internally. It is also
acceptable to declare both host and kernel types explicitly, provided a bridge
checker can verify their correspondence. Neither set of annotations should be
an unchecked promise about the other.

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

## A shared boundary model

Triton and Pallas have the same three layers even though they distribute the
work differently:

| Layer | What it describes | Vector-add example |
| --- | --- | --- |
| Python | Callable inputs and results: host allocations, related dimensions, permitted layouts and dtypes. | Two arrays of length `N` return one array of length `N`. |
| Launch metadata | Program-to-view mapping, grid, tile sizes, scalar and compile-time arguments, output allocation and edge policy. | A block of size `B` chosen for each of `ceildiv(N, B)` programs. |
| Kernel | What one program can read and write, with semantic parameter types checked against its unchanged body. | Two length-`B` input views produce one length-`B` output view. |

The desired fourth component is a *checked correspondence* among these layers,
not necessarily inference of any layer from the others. In particular, knowing
only a kernel's `[B]` view does not determine the host length `N`, which output
must be allocated, or how a program chooses its view. A user's declared host
callable could be checked against an explicit metadata declaration and the
kernel signature. Alternatively, a later design might derive the host
callable from that declaration; automatic inference is not a prerequisite for
avoiding an unchecked FFI.

For the two vector adds, the same correspondence has different realizations:

| Relationship to check | Triton | Pallas |
| --- | --- | --- |
| Host allocations to kernel parameters | Two host inputs `[N]` become input pointers with full allocation bound `[N]`; a newly allocated `[N]` result becomes an output pointer. The wrapper supplies `n_elements: Int[N]` and a chosen `BLOCK_SIZE: Int[B]`. | Two host inputs `[N]` become `[B]` input `Ref`s through `in_specs`; an allocated `[N]` result becomes a `[B]` output `Ref` through `out_shape` and `out_specs`. |
| Program-to-view mapping | `pid * B + arange(0, B)` produces offsets; comparing them to `N` produces the explicit tail mask used by loads and the store. | `BlockSpec((B,), lambda i: (i,))` maps program `i` to block `i`; blocked indexing handles a partial final block for this elementwise kernel. |
| Callable and launch | The proposed host signature is `(torch.Tensor[[N]], torch.Tensor[[N]]) -> torch.Tensor[[N]]`; metadata derives a grid of `ceildiv(N, B)` and a Triton bracketed launch. The fixture's direct typed kernel call is *not* that launch. | The proposed host signature is `(jax.Array[[N]], jax.Array[[N]]) -> jax.Array[[N]]`; metadata supplies `grid=(cdiv(N, B),)` and the three specs to `pallas_call`, which constructs the host callable. |

These rows describe a desired check, not one already established by the
prototype. The Pallas stub has a specialized `pallas_call` overload for this
vector add; multiplying overloads for every arity, output tree and block
pattern is not itself a scalable bridge design. A generic checker might need
to pair corresponding host arguments, specs and kernel parameters across a
tuple or pytree, transform their types, and check a *relation* among the
entries. `MapIntTuples` illustrates mapping over a tuple of shape values but
does not express that multi-input, multi-output relation by itself. A narrow
library-specific hook remains possible if a reusable relation cannot be
expressed in stubs; the experiment should identify the missing primitive
before hardcoding either DSL into Pyrefly.

The kernel types might not be the user-facing entry point. A host-first
declaration could derive the semantic kernel parameter types internally;
the present illegal Triton annotations are a way to test those types without
settling the eventual syntax. Conversely, declaring both sides is useful if
the checker cross-references them. In either case the runtime wrapper would
be controlled and generated by us, rather than hand-typed by users. Its
*exposed signature* would still be statically checked; its generated
implementation would use dynamic checks and launch code, not depend on users
manually type-checking that implementation.

## What a bridge checker must establish

The checker needs to relate each named host input and output to the appropriate
kernel parameter, its storage role, and the metadata that supplies it. It must
also account for parameters absent from the host callable: injected dimension
and stride values, `constexpr` tile choices, Pallas scratch allocations,
descriptors, and allocated or preinitialized outputs. Matching only the
number of arguments or their tile shapes would miss swapped equal-shaped
input/output roles and unsatisfied dtype or initialization requirements.

For each correspondence, record the strength of evidence:

- **Body-checked:** a semantic type reaches a real kernel operation, with
  negative controls demonstrating that a wrong shape or role fails there.
- **Bridge-checked:** a host shape/role is connected to the intended kernel
  parameter and to its `BlockSpec`, pointer view or descriptor. An overload
  that merely assumes this relation is not body evidence.
- **Runtime-enforced:** generated code checks or transforms a property before
  constructing descriptors, allocating outputs, or launching. A type promise
  is not a substitute for a required runtime shape or layout check.
- **Unverified or trusted:** arbitrary index-map values, general grid coverage,
  numerical bounds, hardware layout legality, and operations whose shape
  stubs are too broad must remain explicit obligations.

The mapping may not be a simple slicing formula. Pallas specs can squeeze or
pad axes, prefetch scalars, and map a grid coordinate to different input and
output positions. Triton may group program IDs, use strides and descriptors,
or accumulate into shared outputs. A structured, checkable mapping can give
stronger evidence for common patterns; an arbitrary Python lambda or pointer
arithmetic expression cannot be certified simply because its declared
input/output types match. If a safety-critical property is unverified, the
prototype must not describe that boundary as safe.

## What generating a dynamic wrapper would require

A generator would turn the checked declaration into a host callable. At each
call, or when specializing a cached callable, it would bind actual input
shapes and runtime metadata to symbolic dimensions; check dimensions,
dtype/device compatibility, strides, alignment, aliasing and other
preconditions that the host type system does not guarantee;
then reject or explicitly sanitize inputs according to a declared policy.
Only after those checks would it choose compile-time tile values, derive
scalar/stride arguments and descriptors, allocate outputs with the specified
shape and initialization, calculate the grid, and invoke the backend launch.
For Pallas this includes constructing `out_shape`, specs and `pallas_call`;
for Triton it includes the bracketed JIT launch. A wrapper returns the output
or a declared output tree, preserving the agreed ownership and alias policy.

All such checks must precede any access that relies on them; copying an input
has a performance and aliasing cost that must be explicit. Generation could
cache backend-specialized callables without changing the declared host
signature. We are **not** implementing that generator here. Its feasibility
depends on knowing, for each kernel, which values come from the call, which
are fixed at construction, and which requirements remain too complex to
establish statically or enforce economically at runtime.

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

For each kernel, record the proposed host signature; each named host-to-kernel
parameter correspondence; the shape and storage preconditions; scalar,
scratch and compile-time sources; output allocation and initialization; the
program-to-view map; and what happens when a host precondition fails. Label
each relationship as body-checked, bridge-checked, runtime-enforced or
unverified. Note what a generated wrapper would compute, validate, sanitize,
allocate, launch and return. A shape relation stated only in a trusted stub
is not body evidence; a local masked access is not a proof of full grid
coverage. For dynamic or hardware-specific cases, note the exact unverified
obligation instead of broadening an overload until the fixture passes.

This is a test of whether enough information exists to generate a safe
boundary later, not a request to implement generation now. Triton remains the
priority for the kernel-body sweep; Pallas supplies useful comparisons and a
second practical target. CuTe is paused while these two prototypes advance.

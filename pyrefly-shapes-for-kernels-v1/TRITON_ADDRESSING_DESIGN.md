# A possible semantic type system for Triton addressing

This is a design sketch, not an implemented safety claim. It uses the
unchanged grouped matmul in
[`triton_examples/test_matrix_multiplication.py`](triton_examples/test_matrix_multiplication.py)
as a worked example. The aim is to type the relationship between a checked
Python launch, its program IDs, in-program pointer arrays, reduction steps,
and masks. In particular, type-checking an allocation's shape and strides is
not the same as proving that a kernel addresses that allocation safely. The
v1 examples continue to preserve the upstream executable kernel bodies;
exploring new annotations and identity calls inside kernels is a candidate
for a subsequent iteration. This document records the design without
changing that v1 constraint.

## What v1 does and does not establish

The checked host call validates the shapes and strides of A, B, and C, their
devices and dtypes, and the launch arguments. Its `tiled_output` descriptor
derives a one-dimensional grid with
`ceildiv(M, BLOCK_SIZE_M) * ceildiv(N, BLOCK_SIZE_N)` programs. The kernel
parameter types tie A to `[M, K]`, B to `[K, N]`, C to `[M, N]`, and their
element strides to the corresponding scalar arguments. Tile pointer types
retain allocation strides and tile widths. In particular, `InTilePointers`
has `__iadd__` overloads intended to require a K-block displacement using
A's column stride or B's row stride, respectively. A swapped-stride mutation
still needs a dedicated negative regression test; do not infer that every
combination of symbolic and gradual strides is rejected. Inside the kernel,
the stride parameters have distinct symbolic types; the checked host adapter
currently handles some strides as gradual `int` and validates concrete
arguments at launch. These are different guarantees.

These facts do **not** prove addressing safety. An incorrect `pid_m` or
`pid_n`, a wrong pointer origin, an omitted or repeated reduction-pointer
advance, or a mask computed from different addresses can still access the
wrong elements. Matched tile dimensions and mask *categories* are not a
proof that each enabled lane is in bounds. Conversely, a valid masked load
does not prove that every output element is computed and stored once. A
checked launch grid says how many programs run, not what any program does.

## Separate the concepts, then compose them

The following are illustrative type expressions, not proposed Python syntax
or stub declarations. An axis is identified by its role in an allocation,
not only by the numerical value of its extent or stride:

| Concept | Illustrative form | Meaning |
| --- | --- | --- |
| Allocation | `Array[A, axes=(M, K), strides=(AM, AK)]` | Host allocation and its logical-to-physical mapping. |
| Grid | `Grid[(ceildiv(M, BM) * ceildiv(N, BN),)]` | Number of launched programs, checked on the host. |
| Scheduling | `GroupedTiles[axes=(M/BM, N/BN), group_m=G]` | Intended mapping from `pid(0)` to output tile coordinates. |
| Tile coordinate | `TileIndex[layout, M]`, `TileIndex[layout, N]` | Which output block a program owns; distinct from a lane. |
| Reduction coordinate | `BlockIndex[K, BK]` | Which K block the loop is processing. |
| Lane | `Lane[K, BK]` | An element within a K block, supplied by `tl.arange`. |
| Addressed region | `Pointers[A, tile=(BM, BK), axes=(M, K), origin=(m, AnyK)]` | An array of addresses, still tied to A's axes and strides. |
| Displacement | `Step[A, axis=K, block=BK, stride=AK]` | Advance an A pointer array by one K block. |
| Guard | `Valid[A, axes=(M, K), origin=(m, k), lanes]` | Claim that enabled address lanes belong to A. |

`A`, `B`, and `C` are allocation identities or roles; `M`, `N`, and `K`
are logical axes, not merely equal-sized integers. The implementation might
use generic tuples and axis descriptors rather than these named types. For
example, the tile-coordinate, lane, and displacement forms should compose
for other ranks and access orders instead of creating a separate pointer
class for each matmul operand. Distinguishing A's K stride from B's K stride
by *role* matters even when their concrete integer strides happen to match.

## The proposed trust boundary

The Python launch descriptor and the kernel should refer to the *same*
layout definition: for this example, output axes `(M, N)`, blocks `(BM,
BN)`, grouped-M scheduling with group width `G`, and one grid axis of length
`ceildiv(M, BM) * ceildiv(N, BN)`. The existing `tiled_output` type records
the output shape and blocks, but its grouping metadata is not currently a
shared static type. A future descriptor could include the schedule, or the
prototype could explicitly name the same schedule on both sides and ask a
reviewer to check that equality. The runtime boundary must continue to
check actual shapes, strides, scalar arguments, and grid size independently
of any kernel-side annotation.

The preferred *kernel-side* trust point is `pid`: a quoted annotation states
that `tl.program_id(axis=0)` is the program ID for that named layout. This
link is manually reviewed; Pyrefly cannot derive a kernel's layout from a
Python launch in general. A quoted narrowing annotation is still checked
as an assignment by Pyrefly, so the first experiment must decide how to
introduce the acknowledged trust. Possible forms, shown only as sketches,
are:

```python
# The explicitly suppressed assignment is the single reviewed assertion.
pid: "GridPid[MatmulLayout[M, N, BM, BN, G], 0]" = tl.program_id(axis=0)  # pyrefly: ignore[bad-assignment]

# Alternatively, one Triton-JIT-compatible identity helper supplies it.
pid: "GridPid[MatmulLayout[M, N, BM, BN, G], 0]" = claim_pid(
    tl.program_id(axis=0), M, N, BM, BN, G
)
```

Neither example is validated Pyrefly code. The suppression route needs a
probe showing that Pyrefly retains the declared local type afterward. The
helper route needs a way for its static signature to bind all the layout
parameters without treating them as arbitrary unbound type variables;
the extra arguments are compile-time/type evidence, not runtime checks.
Use quoted syntax here because Triton's frontend visits local annotations;
a CPU-only frontend probe accepted a quoted local semantic annotation.

No later wrapper may silently create a new PID/layout relationship. A
wrapper on the result of `min`, or on loop `k`, is a distinct trusted claim
about that result and must be counted as such. Prefer ordinary operators
when they can carry the existing claim through an expression. Keep each
unavoidable claim visible in source and audit its consequence separately.

If an identity helper is used, its JIT implementation must literally return
its first argument without changing its value, shape, dtype, or pointer
address. A paired stub can advertise a stronger semantic return type while
the JIT function remains legal Triton source. This is a *cast with a named
review obligation*, not a proof. Passing layout parameters that the helper
ignores can allow static role checking, but does not make an arbitrary value
match those parameters. One generic implementation with multiple typed
overloads is preferable to one runtime helper per kernel or operator.

## Thread the types through this kernel

1. **Launch and PID.** The checked launch binds `M`, `N`, `BM`, `BN`, and
   `G` to both the grid descriptor and the kernel parameters. `tl.program_id(0)`
   could be given a grid-axis provenance such as `ProgramId[layout, 0]`,
   not just `int`. A layout descriptor specifies the *intended* grouped schedule. For
   `Pm = ceildiv(M, BM)` and `Pn = ceildiv(N, BN)`, each group begins at
   `first_m = group_id * G` and has
   `group_m = min(Pm - first_m, G)` rows. Within the group, the program's
   remainder must map to `(first_m + remainder % group_m,
   remainder // group_m)`. These are the kernel's existing `pid_m` and
   `pid_n` expressions. A successful schedule proof must establish that
   every launched PID maps to one in-range `(m, n)` pair and that these pairs
   cover the output grid exactly once, including the last partial group.

2. **Output region.** `pid_m * BM + tl.arange(0, BM)` produces M-axis
   coordinates; the analogous expression produces N-axis coordinates. Adding
   their stride-scaled addresses to C creates
   `Pointers[C, (BM, BN), origin=(m, n)]`. The store guard
   `(offs_cm < M) & (offs_cn < N)` should establish validity for those
   *same* coordinates. Matching only the guard's dimensions is insufficient.

3. **Input regions.** A's rows start at `(pid_m * BM + row_lanes) % M`;
   B's columns start at `(pid_n * BN + col_lanes) % N`. These modulo
   operations deliberately make loads safe in padded output tiles, assuming
   positive `M` and `N`. They do not replace C's output store mask. The
   initial `offs_k` is a 1D array of integer K lanes, not pointers.
   `offs_k[None, :]` broadcasts across A's selected rows;
   `offs_k[:, None]` broadcasts across B's selected columns. The resulting
   `a_ptrs` is a `(BM, BK)` array of A addresses and `b_ptrs` is a
   `(BK, BN)` array of B addresses.

4. **Reduction loop.** In `for k in range(ceildiv(K, BK))`, `k` is a K
   *block index*. An optional semantic identity JIT helper could mark
   `k` as `BlockIndex[K, BK]` without replacing Triton's ordinary `range`.
   The load guards `offs_k < K - k * BK` have shapes `(1, BK)` for A and
   `(BK, 1)` for B after broadcasting. They guard the final partial K block.
   `tl.load` yields value tiles `(BM, BK)` and `(BK, BN)`, which `tl.dot`
   contracts along BK into the `(BM, BN)` accumulator.

5. **Pointer advances.** At the *start* of iteration `k`, the intended
   pointer origins are `(m, k)` for A and `(k, n)` for B. The updates
   `a_ptrs += BK * stride_ak` and `b_ptrs += BK * stride_bk` move along their
   respective K axes. Local operator rules could require
   `Step[A, K, BK, AK]` and `Step[B, K, BK, BK_stride]` and preserve an
   `AnyK` pointer type. The spelling `BK_stride` here means B's **row**
   stride (`stride_bk`), distinct from the K block width `BK`. A swapped
   `stride_ak`/`stride_bk` should then be rejected. This local check need
   not track the current block number, but consequently cannot show that
   either update runs exactly once per iteration or that the pointers and
   load guards refer to the same `k`.

The repeated `BK` names in ordinary matmul code (K block width versus B's
K-axis stride) are a useful reason to encode *axis and allocation roles*,
not to identify a stride by a scalar's spelling alone.

## Possible checking levels

**Stub-only, local checks.** Keep the upstream statements; retain axes,
allocation identity, tile shape, and stride role through arithmetic. Let
pointer updates accept only displacements on their reduction axes. Type the
two broadcasts, dot contraction, and guard/pointer shape compatibility.
This aims to catch wrong axes and strides without requiring proof of loop
progress, and does not modify the executable matmul body. Verify the
swapped-stride case with a negative static test before claiming it works.

**Shared, inspectable grid intent.** Use one composable launch-layout
descriptor on the Python side and as the named intended layout for PID
coordinates in the kernel. A quoted *local* annotation on `pid` could state
the intended correspondence with the Python grid for a person or agent to
review. Triton's frontend accepts a quoted local annotation in a CPU-only
frontend probe. But `pid: "GridPid[Layout]" = tl.program_id(0)` is **not** an
unchecked cast for Pyrefly: the right-hand side must still be assignable to
that type. A narrow, explicitly acknowledged type-check suppression or a
single runtime-identity `as_grid_pid(...)` at this boundary could introduce
the asserted type. Do not make `tl.program_id` globally return `Any` simply
to hide this obligation. Merely writing the same layout name in Python and
the kernel does not establish that they agree.

**Operator propagation after the PID claim.** If `pid` carries a semantic
type, its `%` and `//` overloads can return a *typed* remainder and group
index. The other inputs must carry meaning too: `tl.cdiv(M, BM)` and
`tl.cdiv(N, BN)` should retain tile counts, multiplying the N count by `G`
should produce a group span, and multiplying the group index by `G` should
produce a first M-tile index. The partial last group then involves
`min(Pm - first_m, G)`, followed by `% group_m`, `// group_m`, and addition.
The current `ProgramId` operators mostly return `int`; annotating `pid`
alone therefore does not type `pid_m` and `pid_n`. This is not one special
`GroupedMatmulPointer` type: ideally it is an algebra of tile counts, group
spans, coordinates, remainders, and axis-specific displacements that can
also describe other kernels. Overloads could check that those roles line
up, but role propagation alone cannot prove that every PID is mapped
bijectively to the output, especially in the truncated final group.

An illustrative type flow for the existing header is:

```text
tl.cdiv(M, BM)                 -> TileCount[M, BM]                (Pm)
tl.cdiv(N, BN)                 -> TileCount[N, BN]                (Pn)
G * Pn                        -> GroupSpan[Layout]
pid // GroupSpan              -> GroupIndex[Layout]
GroupIndex * G                -> FirstMTile[Layout]
Pm - FirstMTile               -> RemainingMTiles[Layout]
min(RemainingMTiles, G)       -> int           # Type lost by builtin min.
mark_group_rows(min_result, RemainingMTiles, G)
                              -> GroupRows[Layout]  # Trusted identity.
pid % GroupSpan               -> GroupLocal[Layout]
GroupLocal % GroupRows        -> RowWithinGroup[Layout]
FirstMTile + RowWithinGroup  -> TileIndex[Layout, M]
GroupLocal // GroupRows       -> TileIndex[Layout, N]
```

Here `mark_group_rows` is *one possible generic identity operation*, not
an implemented API or a function that computes `min`. Giving it the two
operands could let its signature check their axis/layout roles; it still
cannot verify that its first argument is their minimum. If that loss is
unacceptable, a future Pyrefly rule for the existing `min` call or a
separate static expression analysis would be needed. The transition table
specifies what an overload experiment should attempt, not a claim that
Pyrefly currently supports every symbolic relation shown.

### Minimal rules worth prototyping

These are *signature obligations*, not stub code that is known to work:

```text
cdiv(Int[M], Int[BM])                     -> TileCount[M, BM]
Int[G] * TileCount[N, BN]                 -> GroupSpan[N, BN, G]
GridPid[M, N, BM, BN, G] // GroupSpan[N, BN, G]
                                          -> GroupIndex[M, N, BM, BN, G]
GridPid[M, N, BM, BN, G] % GroupSpan[N, BN, G]
                                          -> GroupLocal[M, N, BM, BN, G]
GroupIndex[M, N, BM, BN, G] * Int[G]     -> FirstMTile[Layout]
TileCount[M, BM] - FirstMTile[Layout]    -> RemainingMTiles[Layout]
identity(min(RemainingMTiles, Int[G]), RemainingMTiles, Int[G])
                                          -> GroupRows[Layout]
GroupLocal[Layout] % GroupRows[Layout]   -> RowWithinGroup[Layout]
GroupLocal[Layout] // GroupRows[Layout]  -> TileIndex[Layout, N]
FirstMTile[Layout] + RowWithinGroup[Layout]
                                          -> TileIndex[Layout, M]
TileIndex[Layout, M] * Int[BM]           -> MTileStart[Layout]
TileIndex[Layout, N] * Int[BN]           -> NTileStart[Layout]
```

The common `Layout` in this display abbreviates the *same* `M`, `N`, `BM`,
`BN`, `G`, and grid axis. Both a matching type and correct arithmetic matter:
`pid % GroupSpan` has a useful role only because `%` really computes the
remainder. A branch that uses `pid + GroupSpan` should fail at the first
rule whose input or output role it violates. Specializations must not leave
a broad `int` overload that quietly accepts the mutated expression and
widens away the error. But checking these signatures alone still cannot
establish that two equal-looking `GroupIndex[Layout]` values came from the
*same invocation's PID*; that would need value-level provenance.

The proposed rules must match Python's operator-dispatch order. In
particular, `G * num_pid_n` invokes the left operand's multiplication first;
an overload only on `TileCount.__rmul__` might not be selected if the left
`Int` operation accepts the argument and returns a plain integer. Inspect
Pyrefly's actual inferred types of every intermediate before adding more
classes. Similarly, ordinary `min` is not a method of `RemainingMTiles`,
and Python's builtin `range` is not automatically a semantic iterator.
Where an existing shape-integer rule already retains useful information,
reuse it rather than defining another integer algebra.

The first goal is *role-checking the unchanged grouped formula*: derive
`pid_m: TileIndex[Layout, M]` and `pid_n: TileIndex[Layout, N]` without
independent casts on `pid_m` and `pid_n`. If this requires many special-case
types tied only to grouped matmul, record that failure rather than adding
nominal types for each source-code line. Full bounds and bijection proofs
are a distinct, more ambitious goal.

If overloads work as hoped, the complete set of *new executable statements*
needed in the matmul body may be as small as this illustrative sketch:

```python
pid: "GridPid[MatmulLayout[M, N, BM, BlockN, Group], 0]" = tl.program_id(axis=0)
# Existing grouped arithmetic is unchanged until its `min` result:
group_size_m = mark_group_rows(
    min(num_pid_m - first_pid_m, GROUP_SIZE_M),
    num_pid_m, first_pid_m, GROUP_SIZE_M,
)
# The remaining pid_m / pid_n arithmetic uses typed operators, not casts.
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    k_block = mark_reduction_block(k, K, BLOCK_SIZE_K)  # Optional.
    # Existing masks use k_block in place of k; pointer += lines are unchanged.
```

This is *not* a drop-in replacement for the real kernel: the PID type still
needs the explicit type-checking trust mechanism described above, the
proposed algebra must be implemented, and the wrapper signatures must be
tested with Pyrefly. `mark_group_rows` returns its first argument, as does
`mark_reduction_block`. Extra evidence parameters are ignored at runtime
but can require consistent symbolic axes and blocks in the static signature.
Neither wrapper can inspect and prove that its first argument came from the
displayed arithmetic. If K can instead be typed by a stub overload for the
existing `range`, omit the optional loop helper altogether.

**Identity wrappers at algebraic gaps.** If a primitive loses a needed
role (for example `min`), prefer an identity JIT helper on its *existing*
result over rewriting the algorithm. A runtime body `return value` preserves
the computed value; a typed signature may refine it. A helper that computes
`min(left, right)` is not a no-op even if it replaces an existing `min`, and
does not meet this design's strict identity-only goal. Likewise, a helper
claiming `pid_m` is an M-tile index without checking how it was computed is
an additional trusted assertion, not a consequence of the annotated PID.
Try to minimize such claims; if preserving meaning from `pid` requires too
many special cases or additional assertions, defer the deeper algebra rather
than presenting it as a proof. A scalar identity JIT call compiled in a
frontend probe and disappeared from optimized TTIR except for source-location
metadata. Its interpreter and device behavior, and other wrapper patterns,
still need separate tests.

**Reduction and in-place addresses.** A semantic K block index could flow
from `range(ceildiv(K, BK))` through a typed iterator if one is expressible
without changing loop semantics. Otherwise one identity call inside the
ordinary loop can mark `k` for downstream K-bound and mask typing. The
ordinary `__iadd__` rules should continue to check A/B reduction-axis
steps; a wrapper around each pointer update is unnecessary. With an `AnyK`
pointer type, these checks do *not* establish that the two pointer arrays
refer to the exact `k` used by the masks. Proving that stronger statement
requires an inductive loop invariant, even if no general linear type
system is needed.

The desired invariant can be stated without changing the loop. Write `t`
for the value of `k` **at the start** of an iteration, `i` for an M tile
lane, `j` for an N tile lane, and `q` for a K tile lane. With element
strides `S_AM`, `S_AK`, `S_BK`, and `S_BN`, the addresses should satisfy:

```text
a_ptrs[i, q] = A_base + ((pid_m * BM + i) % M) * S_AM
                      + (t * BK + q) * S_AK
b_ptrs[q, j] = B_base + (t * BK + q) * S_BK
                      + ((pid_n * BN + j) % N) * S_BN
K guard[q]   = (t * BK + q < K)
```

At `t = 0`, the initial pointer arrays establish these equations. The two
updates after the loads preserve them for `t + 1` if each executes exactly
once. The row and column modulo expressions protect the other input axes;
the K guard protects the reduction axis. In contrast, the output store uses
**unwrapped** `(pid_m * BM + i, pid_n * BN + j)` and needs its own M/N
guard. The v1 tile and `__iadd__` types can check selected shapes and the
physical *step* `BK * S_AK` versus `BK * S_BK`. Even a future
`BlockIndex[K, BK]` on `t` cannot establish the full equations unless the
pointer type tracks the current reduction-block provenance and the checker
proves the loop update. The equations are a concrete target for later
analysis, not a guarantee from the proposed stubs.

**Optional deeper analysis.** An integer-expression checker could normalize
ceil-divisions, modulo, and the final-group `min` to verify the grouped
PID-to-tile bijection. A loop analysis could prove by induction that the
pointer origins at iteration `k` equal the K block named by the guards.
This does not inherently require general linear types, but it does require
reasoning about augmented assignment and loop control flow beyond these
stubs. A per-lane validity analysis must additionally establish that every
enabled load/store address lies within its host allocation. Until those
checks exist, label the layout and recurrence as checked *intent*, not
addressing-safety guarantees.

## Relation to the other three experiments

- **Row selection.** Axes and strides from this design could check that a
  row shift uses the allocation's row stride. But an in-place `X += row *
  stride` does not currently refine an annotated matrix pointer to a
  lower-rank row pointer in Pyrefly. A separate `row = X + ...` assignment
  or an explicit identity *row-selection assertion* might help if modifying
  the body is allowed. Neither follows automatically from a typed PID.
- **Grid axes and masks.** A PID carrying the host layout and grid-axis tag
  could propagate that provenance into offsets, pointers, and masks. It
  would reject some mixed-axis expressions without extra masking wrappers.
  Even identical axis tags do not imply that a particular mask protects a
  particular address; exact per-lane validity needs more reasoning.
- **Pallas layouts.** Shared axis, tile, and grid descriptors may be useful
  for a composable `checked_pallas_call`. Pallas already expresses index maps
  and block specs outside the kernel, so Triton JIT identity helpers do not
  solve Pallas's tuple-mapping, contextual-typing, or arbitrary-index-map
  validation problems. The common gain is a vocabulary for the boundary,
  not one shared no-op wrapper implementation.

## Small probes before extending the corpus

Keep v1's executable bodies unchanged. When ready to test a body-editing
prototype, work on one *separate* matmul fixture or copy so that each extra
semantic statement is visible against upstream. Proceed in this order:

1. **Baseline and oracle.** Preserve current v1 diagnostics and CPU results.
   Add an independent, pure-Python oracle that enumerates every PID in the
   grouped launch and checks that the `(pid_m, pid_n)` pairs exactly equal
   the Cartesian product of valid output tiles. Include a full group, a
   truncated final group, `Pm < G`, and non-multiple M/N/K sizes. The oracle
   tests *this formula on sampled sizes*, not a theorem for all dimensions.
2. **One PID trust site.** Prototype the quoted `pid` annotation and one
   narrow way to introduce its type. Assert Pyrefly's inferred type of the
   next expression; run frontend compilation with the annotation in the
   real kernel. Verify that the host descriptor checks the layout parameters
   named in the annotation. If a PID identity helper is needed, check both
   normal JIT and interpreter mode before proceeding.
3. **Group algebra.** Add the smallest `cdiv`, multiply, `//`, `%`, subtract,
   and add rules that derive `pid_m` and `pid_n` from the original header.
   At `min`, test whether an overload on the existing call can preserve
   information; if not, try one typed identity wrapper around its result.
   Inspect the inferred type of *every* intermediate. Mutate the N tile
   count, `G`, group-tail expression, remainder/division order, and a PID
   axis. Record which mistakes produce a static error and which are caught
   only by the oracle. An identity wrapper that accepts `max(...)` in place
   of `min(...)` remains a reviewed assumption, not a passing safety test.
4. **Addresses and reduction.** Check that A's update rejects `stride_bk`
   and B's rejects `stride_ak`, including configurations where the concrete
   strides coincide; distinguish in-kernel symbolic tests from host-side
   gradual strides. If a K-index helper is useful, add only one identity
   use per iteration, leaving ordinary `range` and `+=` intact. Mutate a
   mask to use the wrong axis or K block. Also delete, duplicate, or move a
   pointer update: mark these as **expected misses** until loop provenance
   is analyzed. A same-shaped guard is not necessarily a valid guard.
5. **Runtime and cost.** Require Pyrefly positive/negative fixtures, actual
   Triton frontend compilation, CPU interpreter tests where supported,
   and a comparison of optimized IR *operations* with the original (ignore
   source-location metadata). In one scalar identity probe the call
   disappeared after optimization; that does not establish parity for
   every helper, compiler stage, or device. No GPU execution is required
   for the initial type-system experiment.

Track a small budget of *call sites*, not only library helpers: ideally one
PID trust point (zero extra calls if an explicit suppression works), at most
one identity at `min`, and optionally one per K-loop iteration. There is no
need for a wrapper on each pointer addition or stride update. If the
positive matmul requires a long chain of identity casts or the negative
mutations pass through broad `int` fallbacks, stop and reconsider the
operator model. A shared wrapper implementation does not make many
independent unchecked assertions safe.

This staged design prioritizes the checked Python shape/stride contract while
making the remaining kernel-addressing assumptions visible. It does not
require solving grouped PID arithmetic or loop invariants before improving
the local type rules.

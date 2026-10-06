# Triton example shape contracts (exploration)

The [shared host-boundary note](../kernel-boundary-prototype.md) sketches how
types and runtime metadata could eventually describe safe Python launches for
both Triton and Pallas. This directory remains focused on checking unchanged
kernel bodies, not generating or copying host wrappers.

This experiment starts from the Triton source examples on master. Its purpose is
to find out whether semantic types on kernel parameters can be justified by
checking the **unchanged kernel body** using Pyrefly and a stub overlay. The
fixtures deliberately use annotations that Triton's JIT does not accept; they
are static-only and must not be run as kernels.

## Modeling approach

Read [vector add](tests/test_vector_add.py) alongside the
[Triton stub overlay](triton-stubs/language/__init__.pyi). `InPointer[[N]]` and
`OutPointer[[N]]` describe roles and the **full host allocation** length;
`Int[N]` ties the kernel's `n_elements` argument to that length. A different
symbol `Block` describes the size of one program's tile. In the unchanged
kernel, `program_id * BLOCK_SIZE` identifies a tile boundary, `arange` adds
per-tile offsets, and comparison with `n_elements` creates a mask carrying
both the allocation bound and tile size. The `load` and `store` overloads
compare these brands against their pointers and per-program values.
`tl.tensor` is Triton's own name for a value in a program, **not** a full
Torch tensor; do not conflate its tile shape with the host allocation shape.

Other examples add independent row and column dimensions, element strides,
multi-axis program IDs, descriptors, and tile-level value shapes. Their stub
overloads attempt to preserve the relevant relationships through existing
operators, rather than requiring edits to executable kernel statements. A
matching signature is only a hypothesis: check which original loads, stores,
and operations consume each shape, then use deliberately incorrect shapes or
accesses as negative controls. A mask with the right bound and tile brand
does not, by itself, prove numerical bounds or that it guards the same offsets
as the access. Nor does a correctly typed tile prove the launch grid covers
the output. Read [the shared boundary note](../kernel-boundary-prototype.md)
for the separate host validation and wrapper-generation obligations.

## Corpus and measurement

The source corpus is `third-party/triton/beta/triton/python/tutorials/`,
`python/examples/`, and `examples/`. The initial inventory contains 68 Python
files, including support scripts, CPU tutorials, and Gluon examples. These
categories must be recorded separately when measuring Triton-kernel coverage.
The 68 files comprise 58 under `python/tutorials/`, nine under
`python/examples/`, and one plugin test under `examples/`. They are not 68
GPU kernels: the tutorials include nine CPU-backend files, helper modules,
and 16 compilation-pipeline files, of which 15 scripts contain 22 JIT
definitions. The number of fixtures also includes helper slices, host
adapters, wrappers and synthetic probes, so it is not a kernel coverage
percentage. A meaningful count must classify individual JIT definitions
and verify full-body identity and launch checks separately.
Each fixture names the original file and identifies the copied body or
source slice. Semantic annotations change parameters, while imports and
fixture scaffolding may differ; unchanged executable bodies are checked
against their originals rather than inferred from a passing type check.

For each kernel, record separately:

- whether the declared allocation shapes, dimension arguments, dtypes, and
  pointer roles are supported by the accesses in the kernel body;
- which loads, stores, masks, and value shapes are checked, and which operations
  remain unknown (an unknown access cannot validate its part of the contract);
- whether launch-grid size and program-ID mapping justify coverage of the
  allocation, which requires more than checking a tile locally.

Passing without diagnostics is not sufficient: a negative control must show
that incorrect annotations or incompatible accesses are rejected. A signature
whose output pointer never reaches a checked store is not validated merely
because its inputs were accepted. Later fixtures include original Python
wrappers and static-only Torch-to-kernel adapters, but a direct typed call
does not validate the original bracketed JIT launch. Where such a launch
remains untyped, the fixture records an expected diagnostic rather than
claiming that the Python boundary has been proved.

The initial fixture is the vector-add kernel in `python/tutorials/01-vector-add.py`.
Its kernel body has the same Python AST as the source. The fixture checks its
input and output pointer shapes against the same symbolic length and checks
that each load/store mask refers to the pointer's declared allocation.
The tests also check that the constructed mask carries both `[N]` and
`[Block]`, and reject a wrong allocation bound or tile size at `load`, plus a
wrong allocation bound at `store`. Two negative calls reject incorrect input
and output shapes. As an additional
body-grounding probe, temporarily changing only the second input annotation
to `InPointer[[Other]]` makes its unchanged `tl.load` fail because the mask
refers to `[N]` instead of `[Other]`.

This is one checked kernel, not coverage of the corpus. In particular, the
stub does not yet track element dtypes, prove that a mask protects each active
index, prove that every output element is written, or check the launch grid.
The mask type records its compared bound and tile shape, but does not establish
that the offsets compared are the same offsets later used for an access.
Later fixtures investigate grouped-ID matmul (tutorial 03), a grid-stride
loop (tutorial 09), and a runtime-selected group (tutorial 08). Each fixture
reports separately which mapping relationships reach a checked operation.

The first helper in numbered `06-fused-attention-ws.py`,
`_attn_fwd_inner`, has a separate AST-identical fixture in
`tests/test_fused_attention_ws_inner.py`. Its `tl.range` passes
`disallow_acc_multi_buffer=True`, which is accepted by the narrow existing
range overload. The parameter types tie the query tile's head dimension to
the key descriptor's physical block width; changing only the key descriptor
annotation to an unrelated width produces an error at the original
`tl.dot(q, k)`. A wrong key descriptor is also rejected at a direct call.
As in the regular tutorial 06 helper, the optional accumulator reshape and
join lack typed contracts, and the float16-versus-FP8 value layout merges at
the final `tl.dot`: these retain expected body diagnostics. The descriptor's
full host rows and stride are declared input assumptions, not proved from a
Torch allocation here; the helper cannot establish launch coverage, FP8
layout/hardware support, scheduling legality, or bounds of descriptor loads.
The next numbered warp-specialized JIT helper, `_maybe_make_tensor_desc`,
and launched `_attn_fwd` have exactly the same executable AST as their
regular tutorial 06 counterparts in `tests/test_fused_attention.py`;
`_attn_bwd_preprocess` likewise matches the already checked
`tests/test_attention_backward_preprocess.py`. They reuse those body and
negative-control results without duplicated fixtures. The first *distinct*
backward body is `_attn_bwd_dkdv`, checked in
`tests/test_attention_ws_numbered_backward_dkdv.py`: its original loop uses
`tl.range(..., warp_specialize=...)`. An incompatible key-head parameter
annotation triggers an error at the unchanged `tl.dot(k, qT)`, and an
incompatible query row-stride annotation triggers one at the original pointer
arithmetic; the fixture also rejects the wrong key shape at a direct call.
It does not establish that loop iterations cover all tokens, that token
addresses remain in allocation bounds, or that warp specialization is legal
for the target hardware.
The next distinct numbered-06 helper, `_attn_bwd_dq`, is checked with its
original `tl.range(..., warp_specialize=...)` in
`tests/test_attention_ws_numbered_backward_dq.py`. A wrong softmax-row
annotation produces a source-body diagnostic at `qk - m`; a wrong K-pointer
row-stride annotation produces one when forming `kT_ptrs`. Its direct-call
negative rejects a mismatched softmax-row tile. The declared head-local
pointer extents and row strides are not converted from or checked against
an actual Torch host tensor here; the helper also does not prove loop
coverage, memory bounds, or device-level scheduling legality.
The numbered 06-WS backward entrypoint `_attn_bwd` is checked with the
complete original body in `tests/test_attention_ws_numbered_backward_outer.py`.
It forwards `warp_specialize` into both typed backward helpers; input and
output pointers share host `[Batch, Heads, Tokens, Dim]` dimensions and
batch/head/token/feature strides. An annotation-only wrong DV feature extent
fails at its original `dv_ptrs` arithmetic and `tl.store`, and a wrong DQ
head stride fails at the original `DQ += adj`. These are kernel-body checks,
not proofs that a Torch host tensor has the declared dtype, device, strides,
or that the launch grid covers the tensor. The existing bare `tl.constexpr`
annotation for `LN2` produces an expected float-versus-int diagnostic;
we do not suppress that independent limitation.

The second fixture is `softmax_kernel` from `python/tutorials/02-fused-softmax.py`.
Its body AST also matches the original. The semantic matrix pointers declare
`[Rows, Cols]` and a row-stride identity, while promising a unit-stride inner
axis. `tl.range` ties each loop row to `n_rows`; row-index multiplication and
row-pointer addition require the stride supplied for that specific allocation.
The column comparison produces `Mask[[Cols], [Block]]`; the single input load
and output store both require that column bound and block tile. The reduction,
exponentiation, and division retain the tile or scalar shape through the
output store. `tl.max` only reduces one-dimensional tiles along axis zero;
`tl.sum` also reduces two-dimensional tiles along axis zero, preserving their
column dimension for tutorial 05. Negative checks reject another axis and
two-dimensional operands for `tl.max`.
Negative checks also reject mismatched input rows, output columns,
either row stride, a wrong column mask bound or tile, and a wrong output value
tile. The output-column negative call produces *two* diagnostics (the shared
input column and the supplied `n_cols`), both checked in the fixture.
Temporarily changing only the output pointer's column parameter from `Cols`
to a fresh `Other` produces an error on the unchanged `tl.store`, directly
grounding the output shape in the body.

The promise of unit inner stride is **not** proved from a host tensor: the
kernel adds `col_offsets` directly, and the wrapper passes only `stride(0)`.
This signature must not be advertised as supporting arbitrary column-strided
matrices. Nor do these stubs prove that `BLOCK_SIZE >= n_cols`, that the
strided rows are nonoverlapping, that `num_programs(0) > 0`, or that the
occupancy-derived launch covers every row. They do check the dimensions and
stride identities used at both accesses, rather than assuming an accepted
kernel body establishes the whole host contract.

The CPU backend's `01-vector-add.py:add_kernel_tiled` uses the same standard
`triton.language` API but gives each program a loop over smaller tiles. Its
other two JIT definitions reuse already checked executable bodies:
`add_kernel` is AST-identical to the GPU tutorial-01 kernel, and
`add_kernel_tiled_autotuned` is AST-identical to `add_kernel_tiled` despite
its different autotuning decorator. Thus all three CPU-01 bodies have
coverage, though the autotuner configuration and launch remain unverified. Its
fixture checks the allocation bound and tile shape at the original masked
loads and store; annotation-only input and output length changes fail inside
the unchanged loop body. Because the loop index and `tl.cdiv` are ordinary
integers, these checks do not establish positive tile sizes, divisibility,
unique iteration offsets, grid coverage or CPU execution. The CPU
`02-fused-softmax.py` kernel differs from the GPU tutorial: one program
handles one row. Its fixture checks both row-stride identities at pointer
addition and the column mask/tile at the load and store. Changing only the
input stride annotation fails at its original pointer addition. This kernel
has no row-count argument, so the type system cannot prove that every program
ID selects a valid row, that the grid covers all rows, or that the block is
large enough. The CPU `05-layer-norm.py` kernels have executable ASTs identical
to their already checked top-level fixtures. The CPU `03-matrix-multiplication.py`
padding helper now has an AST-identical fixture declaring input `[Rows, Cols]`,
output `[Rows, OutCols]`, and `PADDING = OutCols - Cols`. Its original load,
two stores, and pointer advances reject wrong column extents, tile widths,
and input/output roles; direct calls reject incompatible host dimensions.
These checks do not establish numerical address bounds for its unmasked
accesses, physical allocation size, row coverage, or CPU execution. A safe
host launch must check positive tile sizes, divisibility, contiguous storage,
dtype, capacity, and grid size **before** launching padding: the upstream
wrapper currently asserts some divisibility conditions only after that launch.
CPU `03`'s matmul body and CPU `04` still require block-pointer or
packed-descriptor contracts not represented by these stubs.
In CPU `03`'s matmul, an initial `USE_BLOCK_POINTERS` branch constructs
either block pointers or arithmetic tile pointers; a later test of the same
flag selects `tl.advance` or `+=`. After the first merge Pyrefly cannot
recover which pointer kind accompanies the flag, even when its parameter is
annotated `Literal[True]`. A union-accepting operation would hide an invalid
pointer operation, so this full body remains uncovered. Separate branch
fixtures could validate individual pointer modes without claiming they type
the unchanged combined kernel.
CPU `04-blocked-matmul.py` and `08-sfc-matmul.py` also remain outside full-body
coverage. Their four-dimensional packed descriptors, VNNI-dependent physical
strides, and selectable block layouts require separately checked allocation
contracts. CPU `08`'s partial-accumulation kernel constructs `c_tmp_desc`
only when the K-blocking factor is greater than one, but reads it under a
separate first-block flag; a safe host contract must relate those choices
before launch. A fixture that merely accepts an untyped descriptor load
would not validate either the output shape or that conditional initialization.

The compilation-pipeline tutorials are also standard Triton kernels, even
though their host scripts primarily inspect compiler IR. The first two
`add_kernel` bodies from `01_read_ttir.py` and `02_layout_assignment.py` are
copied in `tests/test_compile_ttir_vector_add.py`. Each executable body is
AST-identical after parameter annotations; both reuse the strict vector-add
allocation mask. A wrong input extent fails at the first body's original
masked load, and a wrong output extent fails at the second body's original
masked store. Their IR-inspection helpers, kernel launches, runtime dtypes,
and compiler behavior are outside the kernel-body fixture.

The `03_coalesce_vectorization.py:copy_kernel` fixture also preserves the
entire executable body. The input and output pointers share a symbolic
allocation length, and the original nested masked load/store reject a wrong
input or output extent. The distinct `copy_kernel_pipelined` has its own
AST-identical fixture: the two-stage loop still checks the original nested
masked load/store against the same allocation length. Its loop index and
scaled program base widen to ordinary integers before adding `tl.arange`,
so types do not prove the offsets' relationship to `BLOCK * STEPS`, loop
coverage, or the grid. Neither fixture proves contiguous host storage,
address-to-mask identity, `num_warps`-dependent coalescing or PTX
vectorization, asynchronous `cp.async` lowering, or bitwise runtime results.

The `04_remove_layout_conversions.py:elementwise_kernel` fixture preserves
the complete single-program body. Its masked input load, multiplication and
output store retain the `[BLOCK]` tile and reject a wrong input or output
allocation bound at the original expression. This shape check does
not prove that `BLOCK >= n`, the single-program launch, contiguous host
storage, the number of compiler layout conversions, or value equality.

The distinct `04_remove_layout_conversions.py:transpose_kernel` now has its own
unchanged-body fixture. Input `[M, N]` and output `[N, M]` nominal pointers
require matching contiguous row strides at both original unmasked address
expressions. The input load yields `[M, N]`; `tl.trans` produces `[N, M]`
for the output store. Direct calls reject wrong input/output allocation
shapes, and annotation-only body mutations reject incorrect input or output
dimensions at their original accesses. These checks neither prove the
indices are in bounds nor validate Torch host layout, dtype, device,
single-program launch, compiler conversion counts, or bitwise equality.

The `05_software_pipelining.py:matmul_kernel` body is distinct from the
top-level tutorial-03 matmul and has an AST-identical fixture. Its unmasked
row-major input tiles `[BM, BK]` and `[BK, BN]` retain the declared A `[M, K]`
and B `[K, N]` allocation shapes, while `tl.dot(..., input_precision="ieee")`
produces `[BM, BN]` for the C `[M, N]` store. Wrong A or C allocation extents
fail at the original load/store addresses; a direct call rejects a wrong C
shape. The row strides are checked as K and N, but no host Torch layout is
validated. The M extent of A/C and K extent of B are declared host dimensions,
not grounded by the unmasked body: it never compares offsets against them.
These types do not establish K-loop divisibility, bounds of the unmasked
accesses, grid coverage, device/dtype, the existence of async copies, or
bitwise equivalence across `num_stages`.

The `06_warp_specialization.py:matmul` entrypoint has an AST-identical fixture
for its single forwarding call. A narrow, trusted declaration of the separate
`_matmul_persistent_ws` helper accepts A `[M, K]`, B `[N, K]`, C `[M, N]`
descriptors with corresponding `[BM, BK]`, `[BN, BK]`, `[BM, BN]` blocks and
contiguous row strides. Wrong A, B, or C shapes are rejected at direct
entrypoint calls; a wrong C shape also fails at an annotation-only mutation
of the unchanged forwarding call.
The entrypoint's helper declaration is trusted locally, but a separate
AST-identical `_matmul_persistent_ws` fixture now checks its complete body.
The original descriptor loads feed `[BM, BK]` and `[BN, BK]` tiles into
`tl.dot(a, b.T, acc)`, and the original store requires `[BM, BN]`. Changing
only the A or B descriptor tile fails at the dot and subsequent store;
changing only C's tile width fails at the store. Direct calls also reject
wrong descriptor widths. The narrow `tl.range` signature accepts exactly
this warp-specialization keyword combination, without proving scheduler or
hardware legality. Descriptor offsets are plain integers, so neither
allocation bounds nor row-stride identities are derived from the helper's
load/store sites. Host descriptor construction, launch-grid mapping,
dtype, and Blackwell warp-specialization legality remain unproved.

The `07_reduction_lowering.py:sum_kernel` fixture retains the complete
original reduction body. Its masked input load ties the declared one-dimensional
source extent to the comparison bound, and the reduction produces a scalar
for the original output store. Changing only the source extent fails at that
load, and direct calls reject incorrect input and output extents. The scalar
store does **not** inspect the output allocation's length: its declared
one-element output extent is an interface assumption, not an independent
body-grounded proof. The body of
`08_reduction_order_numerics.py:sum_kernel` has the same executable AST, so
this fixture checks both kernels; their host-side compiler
IR inspection and numeric checks are outside this prototype. Neither body
proves that its block covers the source or that the output allocation exists.

The complete `09_dot_to_mma_lowering.py:matmul_kernel` body has a separate
fixture because its `tl.dot` omits `input_precision` and its output is cast.
The same `[M,K]`/`[K,N]` input and `[M,N]` output tile contract checks the
original source row address and output store address; annotation-only wrong
input K and output N each fail at their original operations. Its fixed
full-tile host launch and numeric or instruction-lowering behavior remain
outside the fixture. The distinct `10_mma_precision_numerics.py:dot_kernel`
uses a single whole-matrix tile. Its original `tl.dot` accepts precisely
`"ieee"` or `"tf32"`; direct calls reject other precision strings and wrong
input/output extents, and changing only the output extent fails at the
original store address. Element dtype, numeric accuracy, and which hardware
instruction is emitted are not checked by these shape contracts.
Tutorial `12_inner_tree_reduction.py:sum_kernel` likewise retains its full
original body: its masked source read grounds `[N]`, while the one-element
output is declared-only. Its `ReductionOrdering` argument is a typed choice;
the checker does not establish floating-point reduction order, numeric
equality, or independence from `num_warps`. Tutorial 14 has a separately
checked body with the same precision and tile operations as tutorial 10,
but with the accumulator named `acc`. Tutorials 13 and 15 have identical
complete matmul bodies and share one AST-identical fixture with a wrong-output
mutation rejected at the original store. Their different compilation
targets and tensor-core instruction counts are outside static shape checking.

All five unchanged bodies in `11_coalesce_load_types.py` are checked together:
full-length one-dimensional copy, row-major and column-major two-dimensional
copies, strided gather, and a staged masked copy. Distinct nominal pointer
roles keep row-major stride N separate from column-major stride M, while the
gather's output has `[N]` and its input has a separate declared allocation
length and element stride S. Direct calls reject wrong lengths, row/column
extents, and gather stride; a changed source length fails at the staged copy's
original masked read. For the four *unmasked* kernels, an accepted tile
establishes a local shape/stride relationship but does not prove address
bounds, actual source capacity, base-pointer origin, or that `(N - 1) * S`
fits the gather allocation. Their full-allocation host layouts, dtype and
the compiler's memory-coalescing choices need separate validation. The staged
copy's `NS` accepts the example's 1 or 2, not a general proof of async
lowering or grid coverage.

These fixtures cover all 22 JIT definitions in the 15 ordinary
compilation-pipeline scripts, counting identical executable bodies only once
where documented. Coverage here means each original body has a typed fixture;
it does **not** mean that all its shape/interface properties, launches,
compiler IR or numerical behavior have been verified.

The CPU `06-matrix-vector-multiplication.py:gemv_kernel` has an unchanged-body
fixture with input matrix `[Rows, Cols]`, input vector `[Cols]`, output
`[Rows]`, and an explicit matrix row stride. The original matrix pointer
addition rejects a different row stride; tile-specific pointer additions,
loads, reduction and store retain `[BM, BN]`, `[BN]`, and `[BM]` shapes.
Direct calls reject mismatched host dimensions or stride. The unmasked
pointer roles are cursors: they do not prove that the initial address was
computed before a load, that the K-loop visits only valid columns, that
`M` and `N` are divisible by the tile dimensions, that host storage is
contiguous or dtype-compatible, or that the launch covers each output row.
The CPU `07` BF16 GEMV fixture also preserves the full executable body. Its
output cast retains the `[BM]` tile through the original store; an annotation
change to the output tile fails at the original row offset and store. Dtypes
are modeled as `object`, so this does not prove that the conversion produces
BF16 values or that the host output has a BF16 dtype. Neither CPU kernel has
been run through the CPU backend here.

To check the fixture with Pyrefly, run from this directory:

```sh
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_vector_add.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_cpu_vector_add_tiled.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_cpu_gemv.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_cpu_gemv_bf16.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_pipeline_coalesce_copy.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_pipeline_coalesce_pipelined.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_pipeline_elementwise_layout.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_pipeline_transpose_layout.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_pipeline_software_matmul.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_pipeline_warp_specialization_entry.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_pipeline_warp_specialization_helper.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_pipeline_reduction_lowering.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_pipeline_mma_lowering.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_pipeline_mma_precision.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_pipeline_inner_tree_reduction.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_pipeline_mma_architecture.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_pipeline_coalesce_load_types.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_pipeline_warp_specialization_helper.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_cpu_fused_softmax.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_cpu_matrix_padding.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_compile_ttir_vector_add.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_fused_softmax.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_matrix_multiplication.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_low_memory_dropout.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_layer_norm.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_fused_attention.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_extern_functions.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_grouped_gemm.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_grouped_gemm_tma.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_naive_matmul.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_tma_matmul.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_persistent_matmul.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_tma_persistent.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_descriptor_persistent.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_block_scaled_matmul.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_block_scaled_matmul_cdna4.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_arange_lengths.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_programmatic_dependent_launch.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_stock_split_k_matmul.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_skinny_atomic_matmul.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_twopass_compute_matmul.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_twopass_reduce_matmul.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_single_cta_layer_norm.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_multi_cta_layer_norm.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_multi_cta_2d_layer_norm.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_attention_backward_preprocess.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_attention_backward_dkdv.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_attention_backward_dq.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_attention_backward_outer.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_attention_ws_subtile.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_attention_ws_inner.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_attention_ws_tma_dp.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_attention_ws_entry.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_fused_attention_ws_inner.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_attention_ws_numbered_backward_dkdv.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_attention_ws_numbered_backward_dq.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_attention_ws_numbered_backward_outer.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_attention_ws_device_tma.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_attention_ws_device_inner_oss_dp.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_gluon_memcpy_1d.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_gluon_memcpy_2d.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_gluon_async_copy.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_gluon_tma_memcpy.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_gluon_tma_message_passing.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_gluon_tma_issue_loads.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_gluon_tma_perform_add.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_gluon_tma_elementwise_add.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_gluon_wgmma_small.py
../../.venv/bin/pyrefly check -c pyrefly-host-bridge.toml --expectations tests/test_gluon_wgmma_host.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_gluon_wgmma_blocked.py
../../.venv/bin/pyrefly check -c pyrefly-host-bridge.toml --expectations tests/test_gluon_wgmma_blocked_host.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_gluon_wgmma_pipelined.py
../../.venv/bin/pyrefly check -c pyrefly-host-bridge.toml --expectations tests/test_gluon_wgmma_pipelined_host.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_gluon_tmem_example.py
../../.venv/bin/pyrefly check -c pyrefly-host-bridge.toml --expectations tests/test_gluon_tmem_example_host.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_gluon_tcgen05_small.py
../../.venv/bin/pyrefly check -c pyrefly-host-bridge.toml --expectations tests/test_gluon_tcgen05_small_host.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_gluon_tcgen05_blocked.py
../../.venv/bin/pyrefly check -c pyrefly-host-bridge.toml --expectations tests/test_gluon_tcgen05_blocked_host.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_gluon_tcgen05_pipelined.py
../../.venv/bin/pyrefly check -c pyrefly-host-bridge.toml --expectations tests/test_gluon_tcgen05_pipelined_host.py
../../.venv/bin/pyrefly check -c pyrefly.toml --expectations tests/test_launch_protocol_probe.py
```

The stub overlay is deliberately small and must not use permissive catch-all
overloads to make new examples appear to pass.
The `arange` length guard checks nonzero-start ranges as `[End - Start]`,
not `[End]`. It rejects the incorrect full-range extent and confirms that
`arange(Half, 2 * Half)` has a `[Half]` tile. This matters for the two
attention subtile halves in the remaining warp-specialized examples; it does
not prove that either runtime bound is positive or power-of-two.

The third fixture is `matmul_kernel` (and its `leaky_relu` helper) from
`python/tutorials/03-matrix-multiplication.py`. Both copied bodies have the
same Python AST as the source; only signatures and surrounding test code
differ. Semantic pointer types carry A `[M, K]`, B `[K, N]`, C `[M, N]`, and
all six distinct stride identities. Input tile construction checks the wrapped
M rows of A, wrapped N columns of B, operand axis orientation, and the
associated row/column strides. A and B use different tile pointer types, so
the K-column mask cannot be substituted for B's K-row mask or vice versa.
`tl.dot` checks `[BM, BK] @ [BK, BN] -> [BM, BN]`; the C store requires a
`[BM, BN]` value and a conjunction of an M-row and N-column mask, with each
bound and tile extent intact. Negative controls reject mismatched matrix
dimensions, input and output strides, wrapped bounds, K mask tile/axis (and a
wrong bound when supplied as a precise symbolic dimension),
dot contraction, output mask, output tile shape, and K-axis pointer steps
using the tile width and the respective K stride. In an additional
signature-only probe, changing C's annotated columns from `N` to `K` caused
the unchanged body to fail at `tl.store` (pointer columns K versus mask N).

These checks do **not** prove the grouped program-ID permutation, the launch
grid, positivity of the group/block/dimensions, or unique/complete coverage of
C. `tl.cdiv` and the group-ID arithmetic currently return ordinary integers;
the semantic link first appears when offsets wrap modulo M/N or are compared
with output bounds. K pointer advances require the precise symbolic product
`BLOCK_SIZE_K * stride_ak` for A (respectively `* stride_bk` for B), and
negative controls reject an unrelated K block or stride. The actual K-mask
expression `K - k * BLOCK_SIZE_K` widens to plain `int` because the built-in
`range` yields an `int` for `k`. Its mask therefore **does not retain the K
bound**: a mask comparing the offsets against any plain `int` is accepted at
`tl.load`. A passing known-gap test records this false positive; the precise
wrong-bound negative check alone does not establish validation of the body.
The stubs also do not prove that each K mask corresponds to the current
pointer-advance iteration or that wrapping M/N cannot divide by zero.
There is no host-launch or dtype check. These are *unproved interface
obligations*, not accepted proofs inferred from the lack of diagnostics.

The fourth fixture copies both `_dropout` and `_seeded_dropout` from
`python/tutorials/04-low-memory-dropout.py`, with unchanged body ASTs. Both
signatures require their input and output allocations to have the same symbolic
length as `n_elements`; the baseline also requires that length for `x_keep`.
Each body's masked load and store check that the bound and block tile agree
with the allocation, and the `where` result must retain the input tile shape
at the store. Seeded dropout's `tl.rand` produces a tile of the offsets' shape;
comparison with `p` and `where` preserve that shape, while `seed: int` rejects
a string seed. Negative controls reject wrong allocation lengths at the
boundary, the wrong mask bound or tile at a load, a wrong value tile at a store,
and a wrong RNG offset tile as `where`'s condition.
Changing only the baseline keep-pointer annotation to a fresh length `Other`
in a duplicate unchanged body makes its `tl.load` reject the mask carrying
`n_elements: Int[N]`, independently of the call-site checks.

These shape checks do **not** prove that the random offsets are the exact same
offsets used for the input and output addresses: a different offset vector
with the same tile shape would also type-check. Nor do they validate the keep
mask's integer/boolean dtype, input/output element dtype, PRNG determinism,
the int32 seed representation, `p` in `[0, 1)`, contiguous host allocations,
or the host launch grid's complete coverage. The wrapper in the tutorial
checks only `x.is_contiguous()`; its `x_keep` layout and size require an
additional host-side contract despite this kernel signature.

The fifth fixture copies all three kernels from `python/tutorials/05-layer-norm.py`:
`_layer_norm_fwd_fused`, `_layer_norm_bwd_dx_fused`, and
`_layer_norm_bwd_dwdb`. Their bodies match the source AST. In the forward
kernel, X and Y have `[Rows, Cols]` shapes with the same explicitly supplied
row-stride identity, W and B have `[Cols]`, and Mean and Rstd have `[Rows]`.
The kernel's row-pointer advances require that stride, and its column masks
check the `[Cols]` bound and `[Block]` tile at the X/W/B loads and Y store.
Mean/Rstd stores require scalar values, while the mean/variance reductions and
normalization preserve the tile dimensions. A wrong forward stride, output
width, weight length, or mean length is rejected at the interface.

Backward stage 1 checks that X/DY/DX use the same row dimensions and stride,
W uses `[Cols]`, Mean/Rstd use `[Rows]`, and DW/DB partial-gradient buffers
use `[Groups, Cols]` and the `[Block]` tile. The lock and count pointers carry
the same group identity as `GROUP_SIZE_M`, and the lock backing allocation
declares its full `2 * Groups` capacity. The group-index times N address
step checks that identity and the column dimension. A staged scratch address
type requires **both** the group start and column offsets before DW/DB can be
loaded or stored; negative checks reject the raw pointer and group-only row.
The masked partial-gradient loads and stores check the column bound and tile;
the DX store checks its own column mask and tile. A wrong scratch-group size
or mask bound or lock capacity is rejected.

Backward stage 2 reads `[Groups, Cols]` scratch through a two-dimensional
`[BlockM, BlockN]` tile, using the row/column mask bounds at each load. Its
axis-zero reduction preserves the column tile, which the `[Cols]` output
vector stores require. Negative checks reject the wrong scratch row-mask
bound, reduction axis, partial-gradient width, and final-gradient length. The
scratch pointer's second-stage type retains the same `[Groups, Cols]`
allocation contract, but the two-dimensional access has its own row-stride
and mask types.

There are important unproved obligations. `tl.program_id(0)` is not known to
be less than `Rows`; unmasked Mean/Rstd accesses therefore **do not establish
that the `[Rows]` allocations are large enough for the actual launch**. The
built-in `range` supplies an ordinary `int` to the forward and reduction
loops: the mask retains its `[Cols]` (or `[Groups, Cols]`) bounds, but does not
prove that successive offsets visit each element exactly once. In backward
stage 1, the staged type proves which kinds of address operation occurred,
not that they happened exactly once or that DW, DB, and Lock use the *same*
group-index value; a passing negative-proof probe accepts a double column
offset, and callers must pass actual base pointers rather than a pre-offset
pointer masquerading as one. The lock allocation declares **two** groups of size
`GROUP_SIZE_M` (lock plus count), but the types do not verify the atomic
protocol or prove that launch rows produce valid indices. Dtype conversions,
numerical formulas, aliasing,
row-major/contiguity promises, legal launch sizes, and coverage of the
allocation all still require independent validation at the host boundary.

The sixth fixture copies the first Python-facing tile kernel `_attn_fwd` from
`python/tutorials/06-fused-attention.py` together with its direct JIT helpers
`_attn_fwd_inner` and `_maybe_make_tensor_desc`. All three bodies match the
source AST. This first boundary signature is **restricted to non-FP8 output**:
`FP8_OUTPUT: Literal[False]`. Q, K, V, and O declare the flattened allocation
`[Z * H * N_CTX, HEAD_DIM]`; M declares `[Z * H, N_CTX]`. Their descriptors
track the row stride, unit column stride, and `[BLOCK_M, HEAD_DIM]` or
`[BLOCK_N, HEAD_DIM]` tile. The descriptor-conversion helper uses positional
integer-list literal types for the two dimensions, two strides, and two block
extents. Swapping any pair is rejected for a descriptor whose contract is
already fixed. The ordinary Q/K/V/O construction calls pass this check;
changing the query allocation length is rejected. The output descriptor's
store checks the `[BLOCK_M, HEAD_DIM]` value, while the M store retains the
`BLOCK_M` vector extent. A transposed K tile with a mismatched head dimension
fails `tl.dot`, and an incorrect output head tile fails descriptor `store`.

This is **partial body validation, not complete attention verification**. The
FP8 value descriptor is physically transposed (`[HEAD_DIM, Z * H * N_CTX]`,
stride `[N_CTX, 1]`, block `[HEAD_DIM, BLOCK_N]`) and is excluded. Expected
diagnostics in that unreachable branch and at the inner value `tl.dot`
record that the uncorrelated dtype/descriptors cannot establish the correct
branch-specific tile. The special accumulator reshape/permute/split/join
path is likewise unmodeled; its missing operations remain explicit expected
diagnostics, rather than being granted an unshaped fallback. The helper's
`tl.make_tensor_descriptor` stub accepts the already-checked list parameters
as ordinary `list[int]`; constructor shape checking is justified only at
calls to the annotated helper, not at arbitrary direct calls to that stub.
An accepted-gap control passes incompatible shape and stride lists directly
to the constructor and still receives the declared descriptor type.
`IntListLiteral` retains positional information for list-literal arguments
but lowers to `list[int]` when the helper's parameters become values; requiring
precise lists at the inner constructor would reject that unchanged helper
body. A first-class symbolic list type or checker support for retaining its
element identities is needed to validate both paths.
The host-side pre-hook that changes descriptor block shapes is not checked.
Tensor element dtypes, descriptor/pointer role, index bounds for unmasked
loads and stores, legal blocks and stage values, host stride authenticity,
program-ID flattening, and launch coverage remain unproved. In particular,
the M store has no mask, so its allocation size must be guaranteed by the
launch and wrapper, not inferred from this passing tile type.

The seventh fixture copies `asin_kernel` from
`python/tutorials/07-extern-functions.py` with its body unchanged. Its input,
output, and `n_elements` share the allocation length, while `BLOCK_SIZE`
determines the tile. The bound-bearing mask is required by the input load and
output store. `libdevice.asin` preserves the tile shape through the store;
negative controls reject the wrong boundary allocation lengths, wrong load
mask bound, and wrong result tile. This validates the shape of the value
crossing the external-function call, but **not** its element dtype or the
device implementation selected by Triton. In particular, the stub does not
prove that `asin` accepts the host tensor's dtype, that the GPU supports the
operation, or that the default or caller-supplied `extern_libs` path resolves.
As with vector add, a matching mask type does not prove actual index safety or
launch-grid coverage.

The first kernel in `python/tutorials/08-grouped-gemm.py` is the first
**indirect-device-array** fixture. Its body is unchanged; only parameter
annotations and the surrounding static tests differ. All three address
arrays, the packed GEMM-size array, the packed leading-dimension array, and
`group_size` carry a shared group-count identity. Dedicated A/B/C address
roles survive `tl.load(... + g)` and the `pointer_type` cast, so the unmasked
two-dimensional tile loads feed a checked `[BM, BK] @ [BK, BN] -> [BM, BN]`
dot, whose result must have exactly the output store's `[BM, BN]` tile shape.
Tests reject a pointer-array group-count mismatch and swapped A/B roles at
the kernel boundary, a wrong dot contraction dimension, and a wrong output
tile size. The output role is exercised by the body at `tl.store`; the A/B
roles are not preserved by numeric `tl.load` results, so a known-gap test
confirms that shape-compatible swapped input tile roles pass `tl.dot`. The
grouping metadata loads are deliberately modeled as ordinary integers: their
runtime-varying scalar values lose dimension and stride identities.

The accepted body does **not** verify that entry `g` of the pointer arrays
points to buffers shaped `[M_g, K_g]`, `[K_g, N_g]`, and `[M_g, N_g]` with the
sizes and leading dimensions loaded from entries `3*g`, `3*g+1`, `3*g+2`.
These are five independently supplied device allocations; sharing a group
count cannot prove cross-array, per-index correspondence, or even that those
index expressions remain in bounds; a known-gap test accepts `g*2+1` as a
metadata index where the kernel needs `g*3+1`. Nor is the runtime allocation
length of any device table inferred from its annotated group count. Their
role-branded address values also do not validate the host-provided raw
addresses, the FP16 cast or element dtype, positive nonoverlapping strides
and unit inner strides. There are no load/store masks: the source explicitly
assumes every group's `M_g`, `N_g`, and `K_g` are multiples of the corresponding
tile sizes. Finally, the type check does not establish positive `NUM_SM`, a
matching launch grid, or the group-offset/`tile_idx += NUM_SM` scheduling
proof. These are **unproved interface obligations**, not properties silently
inferred from a passing body. Proving arbitrary heterogeneous groups would
need a way to associate each runtime `g` with matching pointer, shape and
stride entries; making all groups share symbolic M/N/K would misrepresent
this kernel.

The second kernel in tutorial 08, `grouped_matmul_tma_kernel`, creates TMA
descriptors from the same indirect pointer and packed-metadata arrays. Its
body remains unchanged. Descriptor construction preserves the requested
block shapes: A loads `[BM, BK]`, B loads `[BN, BK]` and is transposed before
`tl.dot`, and C stores `[BM, BN]`. Negative controls reject omitting B's
transpose, a wrong C descriptor block width, and an incompatible declared
group count for the sizes array. Both FP8 and FP16 branches select a type at
runtime; this fixture checks their tile shapes, not their element dtypes.

The **descriptor boundary does not validate the heterogeneous interface**.
The `shape=[gm,gk]`, `shape=[gn,gk]`, `shape=[gm,gn]` and row-stride lists are
ordinary integer lists: Pyrefly does not relate their values to an individual
raw address, identify `g` across all five device arrays, verify the packed
metadata's field order, or establish device-table lengths from annotations.
A known-gap test constructs an A descriptor with unrelated shape and stride
values and still reads a correctly shaped tile. Descriptor construction also
erases A/B/C access roles, so a second known-gap test can store through a
descriptor made from an A pointer. Descriptor load/store offsets are plain
integer lists without a bounds or tile-alignment proof. The source uses
unmasked descriptors: valid addresses and full, aligned tiles for every group
must be provided by the host. TMA alignment, hardware support, allocation
lifetimes, pointer casts and FP8/FP16 dtype consistency, launch coverage, and
the group/CTA scheduler remain unchecked.

The first kernel in `python/tutorials/09-persistent-matmul.py` is the
nonpersistent pointer `matmul_kernel`, copied without changing its body.
Like the later persistent variant, its A/B/C parameter contract is
`[M, K]`/`[K, N]`/`[M, N]` with six independently typed strides. It uses
the same bound-aware clamped offsets for M/N, shape-checked A/B loads and
matrix multiplication, and masked output store. Unlike the persistent
variant, it constructs tile pointers once and advances each through K:
the `+=` overloads require exactly `BLOCK_SIZE_K * stride_ak` for A and
`BLOCK_SIZE_K * stride_bk` for B. Negative tests reject a different K step,
the wrong A mask axis, output mask dimension, and C boundary shape. As in
tutorial 03, the built-in loop index erases the K-bound identity in the
load mask; a known-gap test demonstrates acceptance of an arbitrary integer
bound. Element dtype, launch-grid coverage, legal group/block sizes and
actual host tensor strides remain unproved. The grouped program-ID
arithmetic yields ordinary integer tile coordinates: an added `GroupIndex`
addition overload allows `first_pid_m + (pid % group_size_m)` but does not
prove that the grouping is a permutation, or that accesses cover every C
element. This kernel requires no new core typing primitive.

The next kernel in tutorial 09, nonpersistent `matmul_kernel_tma`, uses
host-constructed TMA descriptors rather than per-element pointer arithmetic.
Its unchanged body loads A blocks `[BM, BK]` from a declared `[M, K]`
descriptor and B blocks `[BN, BK]` from a declared `[N, K]` descriptor,
transposes B, then stores `[BM, BN]` into a declared `[M, N]` output
descriptor. Input/output descriptor roles, allocation dimensions, row-stride
identities, and block shapes are part of the annotated kernel interface;
negative controls reject wrong B allocation K, a descriptor in the wrong
input/output role, the wrong B K block size, omission of B's transpose, and
a wrong C output block. `tl.range(k_tiles, warp_specialize=...)` yields
integer iteration values without introducing new tile-shape assumptions.

The body **grounds block geometry**, because descriptor load shapes feed
`dot` and its value must match the output store. It does **not ground the
descriptor allocation shapes or strides**: its descriptor offsets are plain
integer lists, so an accepted-gap test can load arbitrary row/K indices and
receive the declared tile shape. The annotations require A, B, C descriptors
to claim matching M/N/K, but the body does not prove those claims describe
real backing arrays or the offsets are in range or block-aligned. The host
`TensorDescriptor.from_tensor` calls and configuration pre-hook that sets the
actual `block_shape` are not checked. FP8 versus FP16 element dtype, valid
hardware/alignment, grouped program-ID permutation, and launch coverage
remain separate unproved obligations. This is a partial boundary contract,
not a validated host-to-device descriptor construction.

The first persistent kernel in `python/tutorials/09-persistent-matmul.py`
copies `_compute_pid` and `matmul_kernel_persistent` without changing either
body. A, B, and C declare `[M, K]`, `[K, N]`, and `[M, N]` allocations and six
distinct strides. `tl.where(offsets < bound, offsets, 0)` produces a clamped
offset type distinct from tutorial 03's modulo-wrapped offsets; the type
retains its bound through `multiple_of` and `max_contiguous`. The input tile
addresses check the appropriate clamped M or N dimension, both strides, K
axis orientation, and tile extents. Each load checks its K-mask axis and tile,
`dot` checks the contraction and accumulator shapes, and the C store checks
the M/N mask bounds and output value tile. Negative controls reject a wrong
A/C boundary allocation, an A stride or B clamp, K-mask bound, dot
contraction, C mask, and C value shape. `tl.range(..., flatten=True)` retains
the explicitly supplied `NUM_SMS` input while providing scalar tile indices;
this does not establish how many tiles are processed.

The **priority is the Python-to-kernel interface**, followed by as much
kernel-access validation as the types can substantiate. The inner built-in
`range` yields a plain integer `ki`, so the expression `K - ki * BLOCK_SIZE_K`
also has type `int`: a known-gap test demonstrates that an unrelated integer
bound would be accepted by the A load. The separately maintained `tile_id_c`
also has type `int`, independent of the `tile_id` returned by `tl.range`;
the types accept passing an unrelated output tile to `_compute_pid`. Thus this
fixture cannot prove that each computed tile is stored to its corresponding
output tile, that the launch grid `min(NUM_SMS, num_tiles)` is correct, or
that the persistent grid-stride loop covers every output exactly once.
Clamping does not establish positive M/N, provenance of the compared offsets,
or safety of every unmasked input element. Element dtypes (including the
output's FP8 branch), valid block sizes, positive NUM_SMS and GROUP_SIZE_M,
the grouped-ID permutation, and actual host strides remain unproved boundary
obligations. A passing body does not certify those obligations.

The persistent TMA variant `matmul_kernel_tma_persistent` also copies its
direct `_compute_pid` helper, with both original bodies unchanged. A/B are
declared input descriptors of `[M, K]` and `[N, K]`; C is an output descriptor
of `[M, N]`. Two overloads on the kernel boundary mirror the host pre-hook:
`EPILOGUE_SUBTILE=False` requires C block `[BM, BN]`, while `True` requires
`[BM, BN // 2]`. A negative call rejects a half-block output descriptor in
the full-block mode; others reject an incompatible B K dimension and C
allocation width. This is a **declared host-to-kernel constraint**, not
a proof that `TensorDescriptor.from_tensor` or the pre-hook constructed such
a descriptor: neither host function is checked in this fixture.

Inside the kernel, A/B tile loads and `dot` check `[BM, BK]` and the
transposed `[BN, BK]` orientation. In the split epilogue, narrow `reshape`,
`permute`, and `split` stubs check that accumulator `[BM, BN]` yields two
`[BM, BN // 2]` values; negative tests reject the wrong split dimensions,
permutation, and both full/half output value mismatches. Each output store
still produces an **explicit expected diagnostic**: a single unchanged
function implementation sees both descriptor block widths in its union,
but Pyrefly cannot narrow C's width using the independent boolean
`EPILOGUE_SUBTILE`. Giving one descriptor a `store` overload for both full
and half tiles would incorrectly certify the inactive branch. The accepted
boundary overloads therefore do not mean the kernel's output store has been
verified for each branch. An accepted-gap test also stores to arbitrary
descriptor offsets, illustrating that M/N allocation bounds, stride
identities, tile alignment, and the relationship between `tile_id_c` and
the computed tile remain ungrounded. `NUM_SMS` positivity and launch
coverage, grouping correctness, hardware support, and FP8/FP16 element
dtypes are likewise unproved. The shape-preserving `reshape` stub also
assumes an even, positive `BLOCK_SIZE_N` without establishing that
precondition from the annotated integer.

The final tutorial 09 kernel, `matmul_kernel_descriptor_persistent`, is a
stronger **interface-grounding** case because it constructs descriptors
inside the unchanged kernel body from A/B/C pointers. The static signature
declares A `[M, K]` and B `[N, K]`, each with element strides `[K, 1]`,
and C `[M, N]` with `[N, 1]`. Typed `make_tensor_descriptor` overloads require
that the constructor's `shape`, `strides`, and `block_shape` lists agree with
those pointer allocation and stride identities. The unchanged A/B/C calls
pass, while identical constructor calls with an incompatible A row count or
row stride, B K length, or C allocation width are rejected. Constructing a
descriptor from an input pointer cannot furnish a writable output descriptor.
This grounds more of the declared Python-to-kernel contract in the kernel
body than externally constructed descriptors do. It still **assumes** that
the pointer objects handed over by the Python wrapper really have their
annotated allocation shapes and contiguous strides; the wrapper itself is
not checked.

For this fixture, `EPILOGUE_SUBTILE` is deliberately `Literal[False]`. The
full-width C descriptor is constructed and its output store accepts the
`[BM, BN]` accumulator; an attempted `True` call fails at the boundary.
Pyrefly also checks the inactive split path and reports two explicit
expected full/half store mismatches (and one unreachable-code warning),
instead of claiming that one descriptor accepts both block widths. The
source supports the half-width branch, but that branch is **not validated
by this signature**. A future experiment could encode a branch-dependent
descriptor constructor without losing the relationship between the block
width and the flag. Descriptor load/store indices are still ordinary
integers: a known-gap test accepts unrelated indices. As with the other
persistent kernels, `tile_id_c` correspondence, group scheduling, launch
coverage, FP8/FP16 element dtypes, hardware alignment, and legal positive
block sizes remain unproved.

The first kernel of `python/tutorials/10-block-scaled-matmul.py` checks a
different descriptor boundary: A and B are physically `[M, K // EA]` and
`[N, K // EB]`, their scales are five-dimensional
`[1, M // 128, K // VEC_SIZE // 4, 2, 256]` and
`[1, N // 128, K // VEC_SIZE // 4, 2, 256]`, and C is `[M, N]`.
Here `EA` and `EB` are packed elements per byte. The new descriptor stubs
retain those logical dimensions and each block shape; the kernel body matches
upstream exactly. Two strictly shape-preserving reshapes and one permutation
turn a loaded five-dimensional scale tile into a two-dimensional tile;
`dot_scaled` checks the scale rows, common scale width, and accumulator shape;
the store checks the output tile. The checked equations within the body are the scale
tile element counts in both reshapes (`1 * RM * RK * 2 * 256 =
RM * RK * 32 * 4 * 4 = (RM * 128) * (RK * 4)`), their permuted axes,
and the `[RM * 128, RN * 128]` result accepted by the C store. The fixture
rejects wrong descriptor rows, scale
rows or block size, input packing argument, output allocation width, wrong
scale row in `dot_scaled`, and wrong output store tile.

This is **not** a proof that the host tensors have those shapes. The wrapper
passes descriptors alongside independent M/N/K/configuration arguments;
`TensorDescriptor.from_tensor` and the host allocations are not checked in
this fixture. Physical backing dimensions `[M, K // EA]`,
`[N, K // EB]`, and the five-dimensional scale shapes are *axioms* of the
annotated descriptor types, not inferred from their `load` calls or checked
against host tensors. The descriptor types describe physical dimensions but their
unmasked `load`/`store` do not validate offsets, bounds, row strides, or
launch coverage. An accepted-gap test passes a one-element offset list to a
five-dimensional descriptor. The `dot_scaled` stub accepts incompatible
physical K extents (also tested) because the exact relation depends on
format-specific packing. Element dtypes, format string versus packed dtype,
output dtype, positive dimensions, divisible block/scale sizes, GPU support,
and distinct A/B/C roles remain unproved; do not infer them merely from a
passing kernel body. In particular, the signature's logical K and packing
parameters need a host-side check against the actual backing tensors. An
accepted-gap control passes FP8 format strings to a packed two-elements-per-
byte descriptor. The wrapper's `TensorDescriptor.from_tensor` is not in this
overlay; testing a mismatched *actual* backing allocation requires modeling
that constructor rather than assuming a descriptor's annotation is authentic.

Tutorial 11's `add_kernel` is a vector-add kernel with optional programmatic
dependent launch (PDL). Its unchanged body uses the same `InPointer[[N]]`,
`OutPointer[[N]]`, `Int[N]`, and `Int[Block]` contract as tutorial 01; its
masked input loads and output store validate the shared allocation length,
mask bound, value tile, and output pointer role in the kernel body. Negative
controls reject incompatible input/output allocation lengths, a load/store
mask bound or output value tile, and a non-boolean `USE_GDC`. Strict stubs
for the two zero-argument `tl.extra.cuda` operations model calls only.

Those calls do **not** establish synchronization or launch ordering. The
kernel's `USE_GDC` flag is not linked to the host's `launch_pdl` launch
option; a known-gap test accepts `USE_GDC=True` alongside a host flag typed
`Literal[False]`. Nothing here proves that the GPU supports PDL, that all
programs issue the launch-dependents hint, or that a preceding grid has
completed before a particular load except under Triton's runtime guarantees.
As in tutorial 01, the mask types cannot prove provenance of the individual
offset values, actual host allocation sizes, or complete launch-grid coverage.

Tutorial 12 begins with `_stock_triton_kernel`, an unsplit matmul baseline.
Its unchanged body checks the same A `[M, K]`, B `[K, N]`, C `[M, N]`,
six stride identities, wrapped input axes, K-tail load masks, dot tile, and
M/N output store as tutorial 03. This variant adds input row and column
addresses to the base pointers in two steps; narrowly typed intermediate
pointers retain the allocation bounds, tile sizes, and strides through each
addition. Negative controls reject a wrong A row stride, wrapped B-column
bound, output mask, and output allocation width. The baseline introduces no
new semantic interface relation beyond tutorial 03; split-K and atomic output
accumulation occur only in the later kernels of this tutorial.

The host wrapper is not type-checked and the annotations still assume that
the actual A/B/C allocations have the declared shapes and strides. The
`K - k * BLOCK_K` mask bound becomes `int` in the loop, so a known-gap test
also accepts an unrelated integer K-mask bound. Local mask compatibility
cannot establish the indices' values, grouped program mapping, grid
coverage, positive dimensions, or element dtype.

Tutorial 12's `_skinny_atomic_kernel` adds a split-K program axis and an
atomic C update. Its unchanged body accepts the same matrix allocations and
six distinct strides as the stock baseline, with an independent
`K_PER_SPLIT` and a representative `SPLIT_K: Literal[2]` signature. The
input pointer additions and K-tile advances check tile orientations, wrapped
M/N bounds, strides, and block widths; the output `atomic_add` checks a
`[BM, BN]` value and both allocation bounds in its mask. A specialized
`ZeroedOutMatrixPointer[M, N, CM, CN]` preserves a promised zero-initialized
role through construction of the C tile pointer. Negative controls reject an
ordinary output pointer at the kernel boundary and at `atomic_add`, an
incorrect output allocation width, wrong atomic value/mask tiles, and wrong
A stride or wrapped B bound. The source's `SPLIT_K == 1` store is still
type-checked with this role, but the annotated callable contract intentionally
does not cover the unsplit case or arbitrary runtime split counts.

Zero initialization is an **unverified host-side precondition**, not a fact
proved by the kernel: neither the `torch.zeros`/`torch.empty` wrapper branches
nor Triton's autotune pre-hook are checked. There is no model of atomic
ordering, launch grid, program-ID-to-K partition, or a proof that the split
tiles cover K exactly once. `TileStart + offsets` retains the one-dimensional
K tile width and axis but does not prove `k_start` came from the matching
program ID or lies within `[0, K)`. An accepted-gap call passes an unrelated
`K_PER_SPLIT` because integer typing does not establish
`K_PER_SPLIT == ceil(K / SPLIT_K)`; another accepts an unrelated integer
K-tail mask bound after loop arithmetic widens the bound to `int`.

Tutorial 12's `_twopass_compute_kernel` writes partial matmul tiles to a
scratch allocation declared `[Split, M, N]` with three independent strides
`[SK, SM, SN]`. The unchanged body carries its split-stride step, M-row
address, and N-column address through narrowly typed intermediate pointers.
The final store validates both M/N allocation bounds against the mask and
the `[BM, BN]` accumulator. Direct negative probes reject each of the three
wrong scratch strides, either incorrect scratch dimension or output mask
bound, and an incompatible scratch tile; the A/B matrix contracts reuse the
atomic variant's wrapped input offsets, strides, masked K loads, and dot.

This establishes local shape/stride compatibility of the *declared* scratch
allocation with the write, not that the host's `torch.empty((split_k, M, N),
dtype=torch.float32)` has those actual dimensions or FP32 dtype. The stubs
do not track element dtype, nor prove `SK == M * N`, `SM == N`, or `SN == 1`:
arbitrary noncontiguous three-dimensional scratch strides are permitted.
The `SPLIT_K` argument is linked to the declared scratch split dimension in
the signature, but the body cannot prove that its `program_id(1)` index is
within that dimension or that the launch matches `SPLIT_K`; a known-gap
probe also accepts `program_id(0)` as a scratch split index. K-split
partitioning, `K_PER_SPLIT == ceil(K / SPLIT_K)`, and the runtime K-tail
mask bound remain unproved. The scratch consumer, reduction order, and
final FP16 C allocation are outside this compute-kernel fixture.

Tutorial 12's `_twopass_reduce_kernel` consumes the **same** scratch
allocation type `[Split, M, N]` with strides `[SK, SM, SN]` and writes a
separate output `[M, N]` with strides `[CM, CN]`. Its unchanged body checks
all three scratch address strides; `tl.load` checks the M/N bounds and
`[BM, BN]` tile against its mask, while the final output `tl.store` checks
the same bounds and tile shape against the distinct C allocation. Negative
controls reject each wrong scratch stride, both scratch dimensions (which
produce output/dimension argument errors at the kernel call), three wrong
scratch-load masks, a wrong output mask, and an incorrect output allocation
width. The split stride uses a dedicated `SplitStride[SK]` parameter type:
the built-in `range(SPLIT_K)` yields ordinary `int` indices, and this role
preserves the stride identity through `sk * stride_sk` rather than accepting
an arbitrary integer scratch offset.

This signature states an interface between compute and reduce fixtures, but
neither checks the Python wrapper, the actual FP32 scratch buffer, its
initialized contents, nor whether producer stores finish before reduction
loads. `SplitStride` does **not** bound an index: a known-gap test constructs
a valid scratch slice using an arbitrary integer index, even if it exceeds
the declared `Split` extent. Positive dimensions, contiguous stride
relations, launch coverage, and FP32 scratch and FP16 output dtypes remain
unverified; `tl.float16` conversion only preserves the tile shape in these
stubs. In this reduction's program mapping, `% tl.cdiv(N, BLOCK_N)` can
produce a generic group-index type in the overlay, so only its one-axis
offset shape is retained; this is not a proof that each C tile is written.

Tutorial 15 starts with `_layer_norm_fwd_single_cta`, a single-CTA forward
baseline with exactly the same semantic boundary as tutorial 05: X and Y
are row-major `[Rows, Cols]` with a shared row stride, weight and bias are
`[Cols]`, and mean and reciprocal standard deviation outputs are `[Rows]`.
The unchanged body checks the row stride, column masks and tile sizes on
three input loads, weight/bias loads, and normalized Y writes, and checks the
scalar mean and reciprocal standard deviation writes for each row. No new
stubs are required. Negative controls reject a wrong row stride, Y columns,
statistics length, and weight-mask bound; an accepted-gap test illustrates
that mask compatibility does not establish the numeric column-offset values.

The host wrapper reshapes X to `[M, N]`, allocates Y using `empty_like(x)`,
and passes the reshaped X row stride, but this fixture does not check those
host tensors. In particular, the shared X/Y row-stride promise should be
validated for noncontiguous inputs instead of inferred from `empty_like`.
The grid `(M,)`, row coverage, loop coverage when `BLOCK_SIZE < N`, positive
dimensions, unit inner stride, and element dtypes are not proved by these
types. The subsequent one-dimensional multi-CTA variant mainly replaces
three `range` loops with `tl.range(..., multi_cta=True)`; the two-dimensional
variant changes the tile access pattern substantially.

The next tutorial 15 kernel, `_layer_norm_fwd_multi_cta`, retains exactly
that one-dimensional X/Y/W/B/Mean/Rstd interface, replacing three standard
`range` loops with `tl.range(0, N, BLOCK_SIZE, multi_cta=True)`. A narrow
`tl.range` overload accepts those arguments and returns integer loop offsets;
the unchanged body's row stride, mask, input loads, reductions, and three
output stores use the already checked baseline contracts. The same negative
controls reject wrong row stride, output columns, statistics rows, and weight
mask bound; there is no new host-array shape or stride invariant to claim.

The iterator overload does **not** prove CTA assignment, loop coverage,
positive `BLOCK_SIZE`, or the wrapper's `N // NUM_CTAS` divisibility checks:
an accepted-gap test uses an unrelated symbolic stop value and obtains the
same iterator type. The host wrapper's grid `(M, NUM_CTAS)` and
`ctas_per_cga=(1, NUM_CTAS, 1)` are outside this kernel signature, as are
the distributed shared-memory reductions and their synchronization. The
actual host allocations, shared X/Y row stride, unit inner stride, and
element dtypes have the same unverified status as the baseline fixture.

Tutorial 15's `_layer_norm_fwd_multi_cta_2d` checks a new two-axis interface:
X and Y are `[M, N]` with a shared row stride and unit inner stride, W/B
are `[N]`, and Mean/Rstd are `[M]`. The unchanged body checks that row
addresses use a `[BM]` row axis and that row stride, while matrix input loads
and output stores require a combined M-row/N-column mask and `[BM, BN]` tile.
Weight/bias loads use a `[1, BN]` column tile and an N-bound column mask;
broadcast arithmetic with the `[BM, BN]` input retains the output tile
shape. `tl.sum(axis=1)` yields `[BM]` per-row mean and reciprocal-standard-
deviation values accepted by the row-masked `[M]` statistics stores. Strict
negative probes reject a wrong row stride, swapped axis or tile width, wrong
M/N input mask, wrong weight/statistic mask, output mask/value tile, wrong
reduction axis, and incompatible Y/W/Mean allocations at the kernel boundary.

These checks do **not** verify pointer position after `X +=` and `Y +=`:
the current type system keeps annotated parameter types fixed through
augmented assignment, so the column-address operation is available even
before applying the row offset. An accepted-gap probe loads a tile without
first advancing X. Masks likewise retain dimension and tile identities but
not the exact values of addresses. The host reshape, X/Y shared-stride
assumption, actual allocations and dtypes, positive block dimensions,
column-loop coverage, grid `(ceil(M / BM), NUM_CTAS)`, required cluster
layout `ctas_per_cga=(1, NUM_CTAS, 1)`, distributed reduction and its
synchronization remain unverified.

The first backward kernel of tutorial 06, `_attn_bwd_preprocess`, computes
the per-token reduction `Delta = sum(O * DO, head-dimension)` without any
mask. The declared interface is O/DO `[Z, H, N_CTX, HEAD_DIM]` and Delta
`[Z, H, N_CTX]`. Its unchanged body checks that the flattened-head address
uses `HEAD_DIM * N_CTX`, each token row uses `HEAD_DIM`, each input tile has
shape `[BLOCK_M, HEAD_DIM]`, and `tl.sum(axis=1)` produces the `[BLOCK_M]`
Delta store. Negative controls reject incorrect head/token/row strides or
feature width, incorrect Delta head stride or output tile, the wrong
reduction axis, and incompatible O/DO/Delta allocation dimensions at the
call boundary. This fixture does not include Q/K/V, which this kernel does
not access.

The `Z` and `H` arguments do **not** occur inside the kernel body; their
link to the O/DO/Delta shapes is a typed boundary assertion, not a validated
access bound. In particular, `program_id(1)` is not proved to be in
`[0, Z*H)` or even to be the correct program axis. An accepted-gap probe
uses `program_id(0)` for the head slice and arbitrary row offsets: both
still yield well-shaped, unmasked pointers, even when rows exceed N_CTX.
The host wrapper asserts `N_CTX % 128 == 0` and launches
`(N_CTX // 128, Z * H)`, but neither the assertion nor the grid is checked
here. The actual host O/DO/delta allocations, their contiguity/strides,
FP32 Delta element dtype, possible FP8 behavior, and completion before the
subsequent backward kernels are also unverified.

Tutorial 06's `_attn_bwd_dkdv` is a **head-local tile helper**, not a
kernel receiving host dK/dV allocation pointers. Q and DO enter as
head-local `[N_CTX, HEAD_DIM]` pointers with separate token and feature
strides; M and D (the preprocessed Delta) enter as `[N_CTX]` head-local
statistics pointers. Inputs k/v and accumulators dk/dv are already-loaded
Triton tiles `[BLOCK_N1, HEAD_DIM]`; they do not establish the original
host K/V/DK/DV allocation shapes or strides. The unchanged body checks
Q's transposed `[HEAD_DIM, BLOCK_M1]` address axes against its two strides,
DO's `[BLOCK_M1, HEAD_DIM]` address axes, `[BLOCK_M1]` M/Delta tile widths,
and token-pointer advances by `BLOCK_M1 * stride_tok`. Exact transpose and
dot stubs validate all matrix contractions and preserve `[BLOCK_N1,
HEAD_DIM]` dK/dV accumulator shapes. The causal comparison retains the
`[BLOCK_N1, BLOCK_M1]` condition shape; a swapped mask is rejected. Negative
controls also reject wrong token/feature axes or stride, advance width,
K/dO contraction, accumulator tile, and incompatible DO/M/Delta boundary
allocations. No catch-all dot, transpose, or pointer overload is used.

The helper has no masked Q/DO/M/Delta loads and never uses its `H`, `N_CTX`,
or `sm_scale` arguments to establish address bounds or scaling. The
`BLOCK_N1 % BLOCK_M1 == 0` runtime/compiler assertion is retained verbatim
but not proved by these types. An accepted-gap probe loads arbitrary
`[BLOCK_M1]` statistics offsets even if they exceed N_CTX. Likewise, the
causal comparison constrains tensor tile orientation but not token-range
provenance; `start_m`, `start_n`, and `num_steps` remain plain integers.
All batch/head pointer adjustments, host Q/K/V/DO and dK/dV allocations,
FP8/FP16 and FP32 element dtypes, loop bounds, mask correctness, and
eventual gradient stores depend on the later outer `_attn_bwd` kernel and
wrapper and are **not** validated by this helper fixture.

Tutorial 06's `_attn_bwd_dq` is the complementary **head-local helper**.
It receives K/V pointers with `[N_CTX, HEAD_DIM]` roles and separate token
and feature strides, a `[N_CTX]` Delta pointer, and already-loaded q/do/dq
tiles `[BLOCK_M2, HEAD_DIM]` and an m tile `[BLOCK_M2, 1]`. Its unchanged
body checks both transposed K/V tile axes `[HEAD_DIM, BLOCK_N2]` against
the matching strides and checks each pointer advance by
`BLOCK_N2 * stride_tok`. Delta loads yield `[BLOCK_M2]`; dot, transpose,
broadcast, and causal comparison constrain `[BLOCK_M2, BLOCK_N2]` scores
and the `[BLOCK_M2, HEAD_DIM]` dQ accumulator. Negative controls reject
wrong tile contraction, mask orientation, pointer axes, steps, and parameter
roles. Existing strict stubs suffice without new catch-all overloads.

The K/V and Delta `N_CTX` extents are **asserted** at the helper boundary,
not grounded by a bounds check: K/V and Delta loads have no mask, and an
accepted-gap probe loads arbitrary tile offsets. Likewise, q/do/dq/m
are tiles, not host Q/DO/DQ/M allocations; the helper cannot prove their
original shapes, strides, or the output DQ store. `H`, `N_CTX`, start/step
provenance, the `BLOCK_M2 % BLOCK_N2` assertion, host buffer initialization
and dtype, and the eventual batch/head grid are not verified. The outer
`_attn_bwd` kernel and host wrapper need separate fixtures.

Tutorial 06's outer `_attn_bwd` fixture checks the seven Q/K/V/DO and
dQ/dK/dV allocation roles `[Batch, Heads, N_CTX, HEAD_DIM]` with shared
batch, head, token, and feature strides. The flattened program ID `bhid`
selects a head via `bhid % H` and a batch via `bhid // H`; multiplying these
by their separately typed strides yields one batch/head adjustment accepted
only by matching 4D allocation pointers. Only axis-2 program IDs receive
the batch/head division rule; ordinary program-ID division remains integer
arithmetic. Pyrefly currently infers `AttentionBatchIndex[int]`, not
`AttentionBatchIndex[Heads]`, for `bhid // H`: the exact head-count relation
on that batch index is **not proved**. Statistics M and Delta declare
`[Batch, Heads, N_CTX]` and shift by `bhid * N_CTX` into the same head.
Within that head, the unchanged body loads K/V and Q/DO tiles, feeds them
into both previously checked dK/dV and dQ helpers, loads the M tile, and
stores all three gradient tiles through their declared output role. A store
must receive `[BLOCK_N1, HEAD_DIM]` for dK/dV and `[BLOCK_M2, HEAD_DIM]`
for dQ; pointer axes must use the declared token and feature strides.
Negative controls reject wrong batch/head adjustment strides, different
head counts, wrong Q/K/V/DO/DQ/DK/DV or M/Delta parameter dimensions,
misoriented gradient pointers, and incorrect gradient value tiles. All
three gradient stores are checked; their host-backed role is still only an
annotation until the Python-to-kernel launch is validated.

Pyrefly requires in-place pointer adjustments to retain their parameter's
declared type. The 4D pointer classes therefore inherit head-local pointer
behavior while `__iadd__` checks the precise batch/head offset; statistics
do likewise. Consequently, an accepted-gap probe demonstrates that an
unadjusted 4D pointer may still load/store a correctly shaped tile: the
types do **not** prove that batch/head adjustment happens *before* access or
that the adjusted pointer points into its stated host allocation. The body
also has no load/store masks; `N_CTX` extents, grid sizing, `pid` coverage,
`BLK_SLICE_FACTOR` divisibility, arbitrary start offsets, dtype/FP8 support,
host allocation/stride origins, and actual helper execution are unverified.
The shared `tl.constexpr` stub is integer-only to preserve symbolic block
quotients; the upstream float `LN2: tl.constexpr` annotation produces one
explicit expected false-positive diagnostic, so its scaling is not validated.

The separate `tests/test_twopass_host_bridge.py` fixture checks Torch
allocation shapes and probes the Python-to-Triton seam. Run it with
`pyrefly-host-bridge.toml`, after installing Torch into the configured `.venv`.
Resolving `torch._C.TensorBase` is necessary:
without it, Tensor assignability can appear to pass incorrectly. With Torch
resolved, wrong scratch axes and output sizes are rejected, including at
`Tensor[...]` annotations. A host Tensor is also rejected as a semantic scratch
pointer, while the actual `kernel[grid]` launch is rejected because the current
`jit` stub does not model it. Thus this fixture does **not** validate the real
Python-to-kernel call or connect Torch storage to the pointer annotations.
Neither dtype nor whether the scratch is initialized is tracked in these
Torch shape types.

The separate `tests/test_vector_add_host_adapter.py` probes a narrower,
positive host contract under the same Torch-enabled configuration. It uses
two **trusted static-only adapters** from `torch.Tensor[[N]]` to the vector
kernel's `InPointer[[N]]` and `OutPointer[[N]]`. A wrapper allocating
`torch.empty((n,))` connects the three host array lengths, `n: Int[N]`, and
the previously body-checked `add_kernel` signature. Negative calls reject
a second input with a different length, a wrong-size output allocation at
the kernel call, and a raw Torch tensor passed as a Triton pointer. This
proves the *declared* host shapes agree with the already checked kernel
body; it does **not** validate the adapters' runtime behavior, the real
`kernel[grid]` launch, contiguity, dtype, device, or grid coverage. It is
a concrete example of the adapter a future out-of-band annotation/launch
hook would need to supply, not a runnable replacement for the tutorial
wrapper. Run the focused check with a Torch-enabled interpreter:

```sh
../../.venv/bin/pyrefly check -c pyrefly-host-bridge.toml --expectations tests/test_vector_add_host_adapter.py
```

The separate `tests/test_gluon_tma_memcpy_host.py` copies the **unchanged
body** of the real Gluon tutorial 04 `memcpy_1d_tma` Python wrapper. Its
static-only annotations require Torch input and output allocations to have
the same one-dimensional length. The real
`TensorDescriptor.from_tensor(input, [XBLOCK], layout)` construction is
modeled as producing a descriptor whose length comes from that Torch tensor
and whose block size comes from `XBLOCK`; `NVMMASharedLayout.get_default_for`
supplies its one-dimensional shared layout. Wrong host output length is
rejected at the wrapper call. Separately, descriptors built from wrong-length
and wrong-block Torch calls are rejected by the already body-checked
`memcpy_1d_tma_kernel` signature. This tests a genuine constructor-to-kernel
shape relationship without a trusted cast from Torch tensors to a kernel
pointer. Check all three host fixtures under `pyrefly-host-bridge.toml` and a
Torch-installed Python interpreter; the default `pyrefly.toml` does not
include Torch's semantic stubs.

The actual `memcpy_1d_tma_kernel[grid](...)` in the copied wrapper has an
explicit expected diagnostic: our `@gluon.jit` signature preserves the
body-checked kernel's direct-call types but does not model a specialized
grid launch. Consequently the experiment does **not** prove the real launch
agrees with descriptors or that the computed grid covers the allocation.
The runtime `TensorDescriptor.from_tensor` checks rank, unit last stride,
positive dimensions, block shape, layout rank and element bitwidth, and base
and row alignment; the static types do not establish the layout's concrete
swizzle value, dtype, device, alignment, or physical tensor identity. A
local `list[Int[Block]]` also loses the block-shape list's rank, so a
two-element list for a one-dimensional tensor is an accepted static gap.
The actual runtime descriptor supports both load and store; our
role-neutral constructor result can satisfy either nominal kernel role, and
an intentional swap of input and output descriptors checks without error.
Thus the source/destination direction needs an additional host-side
annotation or a future validated launch mechanism.
For editor diagnostics on this host fixture, select
`pyrefly-host-bridge.toml` and a Python interpreter with Torch installed.
These three Torch-dependent fixtures are excluded from the regular
`pyrefly.toml` corpus command and checked together with:

```sh
../../.venv/bin/pyrefly check -c pyrefly-host-bridge.toml --expectations tests/test_gluon_tma_memcpy_host.py tests/test_vector_add_host_adapter.py tests/test_twopass_host_bridge.py
```

Tutorial 10's `block_scaled_matmul_kernel_cdna4` has an independent
static-only fixture with an unchanged upstream body. Unlike the NVIDIA
descriptor kernel, CDNA4 receives ordinary pointers to packed FP4 matrix
allocations A `[M, PackedK]` and B `[PackedK, N]`, output `[M, N]`, and
preswizzled scale allocations nominally `[M // 32, 2 * PackedK]` and
`[N // 32, 2 * PackedK]`. Its `BLOCK_K` is in logical FP4 elements, so
the packed operand load and K-pointer increment use `BLOCK_K // 2`; the
scale pointer advances by `BLOCK_K * scale_k_stride`. Separate semantic
roles check the matrix axes and six matrix strides, both scale logical
row identities and four scale strides, and the `[BLOCK_M, BLOCK_N]`
masked output store. The scale reshapes and both MFMA permutations retain
their declared block geometry through `dot_scaled`; negative controls
reject incorrect matrix/output dimensions, scale logical axes or packed
K, scale row/K strides, scale block shape, reshape row extent, scale
K advance, output mask, and output tile. No maskless load overload was
added for ordinary matrices.

Important: an accepted-gap test demonstrates that the scale load has **no
bounds proof**. The body computes physical scale row indices modulo `M`
or `N`, while the preswizzled allocations have only `M // 32` and
`N // 32` rows; those loads are unmasked. For example, at `M=160`,
`BLOCK_M=128`, `pid_m=1`, physical A-scale rows are `0..4` but the
kernel loads rows `4..7`. The callable host wrapper does not require M to
be a multiple of `BLOCK_M`; the script's default CLI path happens to pass
`M=N=8192`, which avoids this particular partial-tile problem. This is a
possible runtime bug, not a GPU-tested failure. Neither a nominal
scale-pointer annotation nor a green fixture validates this interface for
partial last M/N tiles.
The checker also cannot retain the local `SCALE_GROUP_SIZE: tl.constexpr =
32` as a symbolic literal, so the scale `reshape` K-group arguments are
currently plain `int` and their arithmetic is not proved; the resulting
tile shape is a type-level promise of the specialized scale role. The
unmasked matrix K loads require K divisibility/padding not established
here. The input dtype/FP4 packing invariant, scale e8m0 byte dtype and
actual shuffle, output dtype, host origins, grid coverage, pointer offset
provenance, and precise `dot_scaled` physical-versus-logical K relation
remain unverified. `stride_ck` is unused by the upstream kernel. These
gaps matter especially to the host/kernel contract and must not be counted
as proven input safety.

The warp-specialized attention experiment begins with `_attn_fwd_subtile`
from `fused-attention-ws.py`. Its copied body has the upstream Python AST.
The signature shares the Q/K contraction, QK causal-mask tile, rowwise
maximum/sum, V head dimension, and accumulator tile. Negative controls
reject a wrong K head dimension, wrong V/accumulator dimension, and a
different-sized causal mask. The `_fma_f32x2`, `_mul_f32x2`, and
`_reduce_fadd2` helpers have only shape signatures here, not checked bodies.
The enclosing `_attn_fwd_inner_oss_dp` fixture also checks that the K
descriptor produces `[HEAD_DIM, BLOCK_N]` after transposition, the V
descriptor produces `[BLOCK_N, HEAD_DIM]`, and both loaded blocks reach
the same shape-checked subtile signature for both halves. A negative call
rejects a descriptor with a different K block column extent. Descriptor
`load` offsets are still plain integers: the type does not establish that
the K/V descriptors refer to the same batch and head, that the loop stays
within their host allocations, or that its stage selection covers the
intended sequence. The subtile shape signature is declared in this second
fixture; its body is checked separately, not verified as an actual call
across fixture files. The `_attn_fwd_tma_dp` fixture checks both Q
half-block loads, symbolic `[Batch * Heads * Tokens, HEAD_DIM]` descriptor
allocations and `[Batch, Heads, Tokens]` statistics, and both output and
statistics stores. A consistent direct call checks; incorrect output width
and statistics sequence extent are rejected. The incorrect statistics
extent also triggers a collateral `BLOCK_M` inference diagnostic, not a
block-size proof. Actual `ProgramId` multiplication gives the statistics
offset its sequence stride, but neither grid coverage nor batch/head index
bounds are proved. Descriptor offsets remain unrestricted
integers; the inner-loop body is checked in its own fixture, not across
the declared helper call. Its FADD2 auxiliary accumulator is typed as a
tile or scalar zero, as the ordinary variant actually passes zero.
The nonpersistent `_attn_fwd` entry point passes both real program IDs
into the checked TMA-helper signature with an unchanged body. A positive
direct call checks, and a wrong output descriptor width is rejected at
the decorated JIT entry point. Its autotune configuration is a typed
placeholder, not a validated choice of `BLOCK_M` or `BLOCK_N`.
The persistent `_attn_fwd_persist` entry uses a one-dimensional grid and
computes virtual tile indices by adding `num_programs(0)` to a program ID.
The unchanged body yields two expected errors at its TMA-helper call: the
virtual tile remainder and quotient cannot be passed as genuine program
IDs. `ProgramCount` supports the ordinary integer loop arithmetic, but
the stubs do not prove the loop's coverage or `tile_idx` bound. A separate
control checks that incrementing the program ID loses its role. An accepted
gap shows `ProgramId % int` is inferred as `GroupIndex[int]` even for an
ordinary tile count: the general `Int[Groups]` modulo overload also accepts
plain integers. Neither this inferred group role nor the persistent call
validates virtual tile-to-array mapping. Fixing this role distinction will
require a stronger group-count distinction or a checker hook; the attempted
`Int[...]`-derived marker still accepted plain `int` in this prototype.

The final JIT in unnumbered `fused-attention-ws.py`, monolithic
`_attn_bwd`, has an unchanged-body fixture at
`tests/test_attention_ws_monolithic_backward.py`. Its input Q/K/V/dO,
statistics, and dK/dV outputs share the declared attention host shape and
strides. The dQ output instead requires a nominal
`ZeroedAttention4DStridedOutputPointer`: each K/V block atomically adds to
the same dQ allocation, so uninitialized output is unsafe. The original
Python backward does allocate `dq = torch.zeros(..., dtype=torch.float32)`,
but this fixture only **declares** that the host supplied a zeroed buffer;
it does not statically validate the Torch allocation or float32 dtype.
Only zeroed attention tiles have the shape-checked, unmasked `tl.atomic_add`
overload; an annotation-only change to an ordinary dQ output adds an error
at the original atomic, and changing its head width fails at the original
tile-address and atomic calls. A direct negative rejects an ordinary tile.
The preexisting atomic overload for masked scratch matrices remains strict.
The source's float `RCP_LN2: tl.constexpr` triggers an expected stub-level
float/int diagnostic, and its loop triggers a fixpoint warning; loop
coverage, bounds, GPU execution order, and exact pointer element dtype are
not proved. With the previously checked seven JIT definitions (including
the shared `_maybe_make_tensor_desc` body), all eight source JIT bodies in
this unnumbered variant now have an AST-identical checked fixture, without
implying a verified Python-to-Triton launch.

The optional SUBTILING and FADD2 branches retain four expected diagnostics
for missing shape-aware `reshape`, `join`, and tuple `reduce` rules; an
expected `l_ij` uninitialized diagnostic is a checker false positive for
the repeated `not FADD2_REDUCE` condition. Consequently, this is **partial
interior body coverage** plus a typed nonpersistent JIT entry, not a
validated Python-to-kernel launch: descriptor origin and flattened host
`[Z, H, N_CTX, HEAD_DIM]` mapping, unmasked store bounds,
FP8 transposition, variant flags, and grid all remain unverified. The
separate nonzero-start `arange` guard is a prerequisite for typing its
second half-tile without claiming the full block width.
The Triton kernel linter also flags two inherited `tl.dot` calls without
explicit `allow_tf32` (TR011); this fixture preserves the upstream body
rather than changing its numerical or performance behavior.
The enclosing `@triton.jit` helper independently triggers the inherited
TR001 warning for lacking an autotune decorator; the `_attn_fwd_tma_dp`
helper does as well. The actual launch's configuration is outside these
fixtures.

The `fused-attention-ws-device-tma.py` experiment copies the full
`_attn_fwd_tma_dp` body with unchanged Python AST. Its declared Q/K/V/O
descriptors share the flattened `[Batch * Heads * Tokens, Dim]` allocation;
the statistics pointer declares `[Batch, Heads, Tokens]`. The query load,
statistics store, and output descriptor store retain their respective tile
shapes. A consistent direct call is accepted, while a descriptor with a
different output width is rejected. The inner helper has only a four-value
shape signature in this fixture; its body is **not** checked here.

The unchanged body yields expected diagnostics rather than a green pass in
two optional branches: `FADD2_REDUCE` initializes its auxiliary accumulator
with `[BLOCK_M // 2]` elements, but passes it to a helper accepting
`[BLOCK_M]` and later adds it to a full-width accumulator. The `STAGE & 2`
branch unpacks seven values from a helper that returns four. Subsequent
arithmetic diagnostics arise from the merged conditional types; they do not
constitute additional independent shape discoveries. The upstream tutorial
tests instantiate `FADD2_REDUCE=False` and `causal=False` (the host selects
`STAGE=1` for noncausal attention, `STAGE=3` for causal), so these are
source-level findings, **not GPU-confirmed runtime failures**. In particular,
an untyped `STAGE: int` checks both branches without proving either flag's
runtime value. The test's nominal descriptor dimensions are not yet verified
against host allocations, descriptor construction, or grid coverage; the
unmasked descriptor and statistics accesses have no host-bounds proof.
The Triton kernel linter flags the inherited TR001 missing-autotune warning
on this copied helper.
The device data-partition factor is restricted to the source configuration's
literal value `2` in the annotated helper and entry signatures, not treated
as an arbitrary runtime integer. No static check establishes that the
compiler actually partitions the work this way.
The separate `_attn_fwd_inner_oss_dp` fixture checks its entire unchanged
body: K/V descriptor loads produce `[BLOCK_N, HEAD_DIM]`, the K transpose
produces `[HEAD_DIM, BLOCK_N]`, and both values reach the already checked
`_attn_fwd_subtile` body through matching shape signatures. A direct call
with the entry fixture's flattened descriptors and literal-zero auxiliary
accumulator passes; wrong K/V block geometry, Q head width, and a
half-width auxiliary tile are rejected. The source uses the same exact
`_attn_fwd_subtile` body as the sibling warp-specialized fixture, but the
separate Python files do not establish an actual transitive compiler/JIT
call graph. The `tl.range` overlay admits only the configured partition
factor `2`, without proving descriptor load offsets stay within the host
allocation or that a hardware partition follows the declared factor.

The first source-order JIT in
`fused-attention-ws-device-tma-hopper-or-blackwell.py`,
`_attn_fwd_subtile`, is checked with its full unchanged body in
`tests/test_attention_ws_device_subtile_attrs.py`. Its two source-specific
`tl.dot(..., attrs=...)` calls retain the same strict query/key contraction,
value head dimension, and accumulator tiles as the previously checked
unnumbered subtile. The new overloads accept the real optional dot-attribute
keyword but do not relax any operand shape. An annotation-only wrong K head
dimension triggers an additional diagnostic at the original first dot, and a
direct negative tests the same failure through the attribute-bearing
overload. The source's default `FWD_DOT_ATTRS=None` cannot support either
`.get(...)` call; the optional parameter's two expected body diagnostics
remain visible. The original host launch supplies a `FrozenDotAttrs`, but
this fixture does not validate that call, the attribute values or compiler
schedule. Known subtile reshape/reduce gaps remain expected, with no claim
of complete FP8 or hardware safety.
The next source-order device-TMA `_attn_fwd_inner_oss_dp` is checked with its
unchanged full body in `tests/test_attention_ws_device_inner_attrs.py`.
Its `tl.range` uses the actual `merge_epilogue=True`,
`merge_correction=True`, and source configuration's partition factor `2`;
the new overload admits only these values and proves no compiler schedule.
The loaded K tile reaches the checked subtile with a shared Q/K head
dimension, and V has the matching `[BLOCK_N, HEAD_DIM]` tile. Changing
only K's physical block head width or V's row tile adds a new error at the
respective original subtile argument; a direct wrong-K descriptor call is
also rejected. The original auxiliary `l_i0_1` can be literal zero or
a tile, while the checked subtile body requires a row tile, so its call
retains an expected diagnostic: the source's `FADD2_REDUCE` branch is not
proved sound by accepting the other operands. Descriptor host extents,
`FWD_DOT_ATTRS` source/default behavior, memory bounds and launch grid
remain unverified here.

The subsequent `_attn_fwd_tma_dp` body from the same device-TMA variant is
checked in `tests/test_attention_ws_device_tma_attrs.py`. It passes the
source's dot attributes to the checked inner helper, returns four stage-2
values as the source does, and reaches both the statistics `tl.store` and
output descriptor `.store`. Wrong output *tile* head width adds an error
at the original output store; wrong statistics token extent fails in the
original M-pointer address arithmetic. By contrast, changing only the
output descriptor's **full host-column extent** leaves the same three
diagnostics: descriptor `.store` checks the block width, not the entire
host allocation. Full output host width is a declared interface constraint,
not grounded by this body. The three expected original-body diagnostics
at both inner calls and the `FADD2_REDUCE` sum stem from the auxiliary
half-tile versus full-row tile mismatch; an accepted launch or correct
descriptor tile does not fix these. Descriptor origin, FP8 transpose,
host dtype/device, bounds, and launch coverage are likewise unproved.

The next launched `_attn_fwd` in the same device-TMA variant has a full
unchanged-body fixture in `tests/test_attention_ws_device_entry_attrs.py`.
Its source converts Q/K/V/O pointer-or-descriptor inputs using the checked
`_maybe_make_tensor_desc`, with symbolic flattened
`[Batch * Heads * Tokens, Dim]` and source tile shapes. Changing only
`HEAD_DIM` to an unrelated symbolic integer adds original-site errors in
the descriptor `shape`, `strides`, and `block_shape` arguments. Changing
only a pointer's full flattened host-row extent does **not** cause a new
constructor-site diagnostic: the accepted-gap test shows the converted
descriptor retains the wrong extent despite the supplied correct shape.
More importantly, this real constructor yields generic `tensor_descriptor`
for all four operands, erasing the separately declared
`InputMatrixDescriptor`/`OutputMatrixDescriptor` roles used by the already
checked `_attn_fwd_tma_dp`. All four original forwarding arguments retain
expected type errors. No synthetic producer asserts that a generic result
has a more specific input/output role, and these existing errors cannot
ground an end-to-end Python-to-Triton output interface. Shape, dtype,
pointer origin, FP8 layout, descriptor storage, and grid coverage remain
separate obligations beyond the descriptor tile identities described here.

The following device-TMA `_attn_fwd_persist` retains its complete original
grid-stride body in `tests/test_attention_ws_device_persist_attrs.py`. Its
two-positional-argument `tl.range` call is checked by a narrow overload for
`merge_epilogue`, `merge_correction`, and partition factor two; a factor of
three is rejected. Each source descriptor reassignment has an expected
diagnostic because the pointer parameter retains its original type; each of
the four forwarded descriptors has another expected diagnostic because the
generic constructor does not recover input-versus-output descriptor roles. The
loop's `pid` and
`off_hz` are respectively `GroupIndex[int]` and `int`, not trusted
`ProgramId`s; their two forwarding errors remain expected. A mutation of
only the M statistics pointer's token extent adds a new error at the original
`N_CTX` forwarding argument, showing a consistency check at the helper
boundary, **not** a validated descriptor allocation. A separate accepted-gap
probe supplies a wrong pointer host-row extent to the descriptor constructor;
the returned descriptor retains those wrong rows and loses precise block
dimensions despite a correct supplied `shape` and `block_shape`. Neither
this pointer conversion nor the persistent grid-stride scheduler establishes
real Torch allocation extents, descriptor roles, write coverage, or bounds.

The next source-order device-TMA JIT `_split_n` is a distinct recursive
head-width splitter with a complete unchanged-body fixture in
`tests/test_attention_ws_device_split_n.py`. Its narrow nominal split-tile
parameter carries precise two-dimensional `shape`; exact overloads for the
source `reshape([Rows, 2, Cols // 2]).permute(0, 2, 1).split()` preserve
`[Rows, Cols // 2]` for both returned halves. Negative calls reject the
unhalved width, a three-way reshape, a wrong permutation and an ordinary
`tl.tensor[[Rows, Cols]]` argument. The caller's actual `tl.dot` produces
an ordinary tensor, **not** the required nominal split tile: the validated
local transformation is not yet connected to real attention input or output.
Pyrefly reports an expected non-convergent-recursion warning and cannot infer
the complete returned tuple or prove the split factor is a positive power of
two. There is no host allocation contract in this helper. The immediately
following `_attn_bwd_preprocess` has the same complete executable AST as
the existing `tests/test_attention_backward_preprocess.py` fixture and reuses
its shape and negative-control evidence without another copied body.

Gluon's `02-layouts.py` `memcpy_1d_kernel` demonstrates a complementary
explicit-layout idiom. The unchanged body takes an explicitly annotated
one-dimensional layout at `gl.arange`, then applies the same allocation-bound
offset/mask checks as Triton vector add to its input load and output store.
A wrong input or output allocation, layout *rank*, or load-mask bound is
rejected. For a wrong input length the input binds the generic host length,
so both the output pointer and `xnumel` disagree; those two call-site errors
are counted separately.
`gl.BlockedLayout` constructor types check that its four list arguments have
uniform length one or two; `Layout1D` and `Layout2D` otherwise remain trusted
nominal labels, and value, pointer, and mask types forget the chosen
thread/value distribution. In particular, two different
one-dimensional layouts pass interchangeably even if a Gluon operation would
require compatible physical layouts. Stride/order, thread count, warps per
CTA, input/output layout agreement, grid coverage, and whether a host tensor
really supplies the declared length remain unchecked. This is an example of
reusing Triton's host-tile mapping while exposing a distinct layout
conformance obligation, akin to CuTe's fragment layout; it is **not** a claim
that Gluon layouts are fully typed.

The next `02-layouts.py` fixture checks the unchanged `memcpy_2d_kernel` body.
Its input/output pointers declare `[X, Y]` host extents and independent row
and column strides, each bound to the corresponding kernel argument. The
`Layout2D` argument accepts only two-dimensional parent layouts; slicing
dimension 1 produces X-row indices, slicing dimension 0 produces Y-column
indices, and only the expected `[:, None]` / `[None, :]` expansions combine
into a 2D address or mask. Both pointer additions require their own paired
strides, while loads/stores require `[X, Y]` bounds and `[XBLOCK, YBLOCK]`
tiles. Controls reject inconsistent allocation dimensions, an input stride,
a one-dimensional layout, wrong slice dimension or expansion axis, wrong
mask bound, and a wrong output-value tile.

The constructor checks list *rank*, not valid element values, permutations,
warp-product constraints, hardware support, or compatibility between distinct
same-rank parent layouts. The mask types remember the bounds and tile sizes
but do not establish that the masked indices are the actual indices in the
addresses. The kernel's `pid_x`/`pid_y` mapping and the wrapper's
`cdiv(xnumel, XBLOCK)`/`cdiv(ynumel, YBLOCK)` launches do not prove complete
or uniquely in-bounds host coverage. The pointer and stride types describe
host allocation promises; no host-facing torch/JAX tensor validation or
actual runtime Gluon layout validation is implemented here.

The `03-async-copy.py` `memcpy_1d_cpasync_kernel` fixture checks an unchanged
full kernel body, including its async global-to-shared copy. The input and
output base pointers share a symbolic host allocation length with `xnumel`;
their addresses and mask have matching block extents. A distinct 1D
shared-memory descriptor takes `[XBLOCK]` elements and accepts only an input
global-memory tile with the same block width and host-bound mask at
`cp.async_load`. Its `.load` returns `[XBLOCK]` register values, which must
match the output tile width at `gl.store`. Negative controls reject incorrect
host input/output extents, async input mask bound, shared-memory block width,
output value block width, wrong global pointer role, and mixing register and
shared-memory layouts. The shared layout constructor currently recognizes
only the tutorial's one-dimensional `vec=per_phase=max_phase=1` configuration
and checks the rank of its order list, not valid order values.

This static overlay temporarily aliases `gl.constexpr` to `typing.Final`,
which makes Pyrefly infer the distinct types of the *local* register and
shared-memory layouts from their initializers. Without this, a common
`object` annotation loses both layout types and poisons downstream access
checks; a broad `Any` alias would hide incorrect layouts. This workaround
does **not** represent Gluon's normal parameter `gl.constexpr` annotations or
callable `gl.constexpr(...)` constructor. Extending the corpus to such uses
requires an inference-preserving checker hook or a different scoped approach.
The descriptor's float32 allocation is checked, but input/output pointer
element dtypes are not; no host wrapper validates the promised allocation.
The stub does not enforce Ampere's hardware/alignment requirements, capacity
or bank-conflict constraints, per-element mask-to-address provenance, launch
coverage, or completion ordering: `cp.commit_group()` and `cp.wait_group(0)`
are accepted by signature, not tied to a shared-memory typestate transition.

The `04-tma.py` `memcpy_1d_tma_kernel` fixture checks its unchanged full body.
The input/output descriptors declare the same nominal `[Length]` host extent,
`[Block]` transfer size, float32 dtype, and shared-layout identity. The one
parameter annotation `XBLOCK: gl.constexpr` is replaced with the semantic
`Int[Block]` annotation; the source body still uses `gl.constexpr` for the
local descriptor layout, whose precise type survives the previous fixture's
`Final`-alias workaround. The `[XBLOCK]` shared allocation is checked against
both descriptor block widths at `tma.async_load` and `tma.async_store`;
the tile coordinate derives from `program_id(0) * XBLOCK`. The two explicit
`gl.static_assert` comparisons reject distinct block or layout metadata.
Negative controls reject wrong descriptor extent, block size, layout identity,
transfer block, coordinate block, and read/write descriptor roles.

Descriptor host extents remain a **declaration**, not a proof from the TMA
body: neither transfer compares a length argument or masks explicitly. The
runtime descriptors carry physical shapes, strides, dtype, and base pointers,
and the separate Torch host fixture checks that `TensorDescriptor.from_tensor`
preserves the declared `[Length]` from its input allocation. Neither fixture
checks host dtype/physical alignment or validates the wrapper's `cdiv` launch.
Layout equality uses a trusted
nominal identity; it cannot prove two host-created descriptors have equal
physical layouts. The coordinate stub checks tile-start width but currently
accepts lists of any length, not only 1D coordinates. `block_type.nbytes`
is `int`, so `mbarrier.expect` does not prove a matching byte count, and
initialization, wait, invalidation, TMA store visibility, memory ordering,
hardware alignment, and out-of-bounds zero-filling are **not** typestate or
runtime proofs. The `Final` alias is only suitable for local constexpr
annotations; kernels requiring `XBLOCK: gl.constexpr` unchanged in their
parameter lists will still need a dedicated checker rule.

The typing-only `test_launch_protocol_probe.py` isolates a host boundary
limitation; it is not an upstream kernel fixture and does not add to kernel
coverage. Direct calls to a generic kernel **and calls through an identity
decorator** reject mismatched symbolic pointer and length dimensions; a
grid-launched kernel with concrete dimensions does too. Both a `ParamSpec`
wrapper and a grid wrapper preserving the generic function type accept the
mismatched dimensions after `kernel[grid]`. Thus an
ordinary generic `__getitem__` stub does not currently validate Triton's
generic launch contract. The static-only host adapter above checks a direct
call to the kernel signature, **not** the actual JIT launch; a specialized
per-kernel launch protocol or checker support would need its own negative
controls before claiming that boundary is checked.
An attempted typed decorator taking a generic TMA-kernel `Protocol` rejects
both the correctly annotated kernel and a kernel with a wrong block argument;
the two expected errors are reproduced in the probe. A per-kernel grid
protocol can reject mismatched launch arguments, but its signature cannot
currently be established from the generic decorated function: using one
would restate a trusted interface rather than validate the launch boundary.

The next source-order `04-tma.py` fixture checks the unchanged full
`tma_message_passing_kernel` body. The descriptor contract is specifically
for a single int32 message: its nominal host allocation length and TMA
block extent both equal `MESSAGE_SIZE`. `message_desc.block_shape` allocates
a shared-memory tile of that block size; the register offsets from
`MESSAGE_SIZE` must fit its `.store`, and both async TMA transfers require
the shared tile and the same descriptor. The final `gl.store` checks that the
declared output allocation length equals the offset and value tile width;
the output's int32 element dtype is a trusted promise, not a checked store
property. The scalar addition to `gl.arange` preserves its register tile
width without specializing the physical layout.
`ready` has a separate atomic-flag pointer role, with release on exchange
and acquire on polling. Negative controls reject incorrect descriptor host
extent, descriptor block, output extent, flag pointer role, shared tile
width, and reversed atomic semantics. These reject wrong *declared*
interfaces and prove that the kernel body reaches a checked output store.

There is no explicit kernel mask; the TMA descriptor is trusted to handle
bounds. The overlay cannot verify that `TensorDescriptor.from_tensor` was
actually constructed from an int32 torch tensor of the declared shape,
that `ready` is a real one-element initialized flag, or that the wrapper's
two-program launch executes the sender and receiver. The `tma.store_wait`
signature accepts `read_only=False` as the source requests, but deliberately
also accepts `read_only=True`: it does **not** prove that a TMA write became
globally visible before the release flag, nor that the receiver's acquire,
mbarrier, and shared-memory read obey any real synchronization protocol.
The accepted `fence_async_shared()` and barrier calls carry no typestate
or hardware proof. In particular, the model does not check device generation,
descriptor alignment, `block_type.nbytes`, or the layout's physical identity.
`BlockedLayout` currently checks rank but cannot distinguish this kernel's
single-warp `[1]` from the previous async-copy kernel's four-warp `[4]`:
both intentionally produce the same `Layout1D` type.

The next `04-tma.py` fixture checks the unchanged full `issue_loads` helper
body for pipelined 2D TMA reads. Its parameter annotations declare matching
`[Rows, Cols]` A/B input descriptors, `[BlockRows, BlockCols]` TMA blocks,
shared-memory layout identities, matching A/B shared rings, and a barrier
ring with the same symbolic buffer count. The two `tma.async_load` calls
check each descriptor's block and layout against its destination shared tile;
`xoff` carries a row tile-start width. Allocation and direct-call negative
controls reject a write-only descriptor at a read, mismatched host extents
at the helper boundary, block sizes, shared layouts, and ring counts. These
are checks of declared roles and dimensions, not construction of actual
runtime descriptors.

This helper has no checked output store and cannot validate the eventual C
allocation or wrapper. Its body does not compare A/B physical host extents:
equality of `[Rows, Cols]` is enforced by the helper's annotated interface.
`copy_index * YBLOCK` is an ordinary integer, so although `YBLOCK` has the
declared column-block size at the helper boundary, the column coordinate's
actual tile origin is not independently verified. The current 2D TMA
coordinate-list signature does not validate coordinate count or ordering:
the explicit probes `[xoff]` and `[0, xoff]` are accepted. Shared and barrier
ring indexing does not establish bounds or phase; `mbarrier.expect` accepts
an integer byte count but does not prove the two transfers contribute exactly
that count or that the barrier completes. Neither the host descriptor's
physical shape/strides nor the real wrapper and launch are checked here.

The `04-tma.py` `perform_add` fixture checks the unchanged full pipelined
helper body: matching A/B shared rings supply register tiles of symbolic
`[BlockRows, BlockCols]` shape, addition preserves those dimensions, the C
shared tile accepts the result, and the 2D output descriptor's block and
shared layout match that tile at `tma.async_store`. Negative controls reject
wrong input-ring block or count, wrong C tile shape or layout, wrong C
descriptor block or read/write role, mismatched register-addend width, and
wrong output-store tile. This checks a genuine output transfer and the tile
relationship between the input rings and C descriptor, not just an annotated
signature. `xoff` and `YBLOCK` retain row and column block sizes at the
helper boundary; the C descriptor has nominal `[Rows, Cols]` host extents.

The helper receives no A/B descriptors, so C's full host extent cannot be
compared to the input allocations here: a different declared C extent passes
`tma.async_store`. The host wrapper must check that connection.
`read_index * YBLOCK` widens to `int`, and a 2D TMA coordinate list still does
not prove rank, order, or alignment. Barrier `wait` accepts an ordinary
integer phase to accommodate `read_index // num_buffers & 1`, but does not
prove phase parity, a valid ring index, or transfer completion. `store_wait`
and `fence_async_shared` carry no typestate or memory-ordering proof. Physical
descriptor creation, the wrapper's tensor assertions and TMA requirements,
the grid, and the entire outer kernel remain unchecked by this helper.

The `04-tma.py` `elementwise_add_tma_kernel` fixture checks the unchanged
full outer pipelined 2D kernel with real callable signatures imported from
both helper fixtures. Its annotated interface identifies all A/B/C descriptor
host extents as `[Rows, Cols]`, both tile dimensions with `XBLOCK` and
`YBLOCK`, and input versus output descriptor roles. The body allocates A/B
shared rings and a C shared tile of corresponding block shapes and layouts;
both helper calls check those shapes through the async reads, register
addition, and the C async store. The barrier-ring allocation uses the same
symbolic `num_buffers` passed to both helpers. Negative caller controls reject
mismatched A/B/C host extents, tile widths, layout identities, and descriptor
roles. Changing only B's host-column annotation to a separate symbolic
dimension also produces errors on the unchanged `issue_loads` calls, so A/B
descriptor extents are connected by a body-grounded helper call.

The C **host extent is only a declared interface constraint**, not a
body-grounded comparison with A/B. Changing only the annotated C host-row
dimension leaves the unchanged kernel body accepted: `perform_add` receives
the C descriptor and shape-matched tiles but never the A/B descriptors.
TMA tiles can be partially out of bounds, so inventing a required
`coordinate < host extent` constraint would not model TMA accurately. The
`xnumel` argument is unused in this kernel body; `ynumel` only influences
the loop count, not a check against A/B/C descriptor extents. Consequently
the host wrapper must validate that all three tensor shapes really agree,
and that its descriptor construction and launch preserve the declared
signature. Neither `TensorDescriptor.from_tensor` nor the actual JIT grid
launch is checked by this fixture. The `static_range` and `cdiv` signatures
also do not prove positive buffer counts, pipeline fill/drain correctness,
tile coverage, ring bounds, barrier phases, ordering, or transfer completion.

The first `05-wgmma.py` fixture keeps the entire original `small_mma_kernel`
body (AST-identical) and annotates four TMA descriptor parameters. A and B
are declared float16 `[M, K]` and `[K, N]` inputs; C is a float32 `[M, N]`
input and D is a float32 `[M, N]` output. Its TMA loads enforce descriptor
block dimensions and layout against the allocated shared tiles. The
`warpgroup_mma` signature checks A's M/K, B's K/N, and C's M/N register-tile
shapes for both register and shared-memory A operands. The D shared-memory
store then checks the MMA result's `[M, N]` shape, and the final TMA store
checks D's block and layout. Annotation-only mutations of B's K, C's N, and
D's N independently fail at the unchanged MMA or shared store, establishing
those *tile* relationships inside the kernel rather than only at callers.
Negative calls also reject mismatched declared extents, a float32 A
descriptor, or an input descriptor used as an output.

The separate static Torch host probe checks `[M, K]`, `[K, N]`, and `[M, N]`
allocations through fixed-rank block-shape lists into float16 and float32
2D TMA descriptors, then directly calls the kernel's typed signature. It
rejects incorrect B and D host extents. Neither the host probe nor the
kernel body proves the *full* allocation extents from its accesses: each
small-MMA block is loaded at `[0, 0]`, and TMA implicitly handles out-of-
bounds elements. Descriptor `from_tensor` returns a descriptor usable for
both reads and writes; the probe deliberately accepts C and D interchanged.
The factory's float16/float32 overload selects a nominal descriptor dtype
from its layout argument, but Torch shape types do not track `Tensor.dtype`:
a float32 Torch tensor passed through the float16 layout is accepted as an
intended negative control. The actual `small_mma` wrapper and JIT grid launch
are not checked. The `INSTR_SHAPE_N`, number-of-warps, legal WGMMA instruction
shape, shared-memory bytes, synchronization, and Hopper device requirements
also remain unproved.

Both unchanged `gl.static_assert(isinstance(..., gl.NVMMASharedLayout))`
lines currently produce expected Pyrefly diagnostics: the `isinstance`
expression is `bool`, whereas the sound `static_assert` stub requires
`Literal[True]`. The stub intentionally stays strict so `static_assert(False)`
continues to fail in the earlier TMA test. These diagnostics are **not**
evidence that the shared layouts violate the real Triton assertion; the
type checker cannot establish that particular `isinstance` predicate yet.

The next `05-wgmma.py` fixture copies the full original
`blocked_matmul_kernel` body and all three constexpr layout-helper bodies
with identical ASTs. The static parameter contract specializes
`TRANSPOSE_B` to `Literal[False]`: A is float16 `[M, K]` with block
`[BM, BK]`, B is float16 `[K, N]` with block `[BK, BN]`, and output C is
float16 `[M, N]` with block `[BM, BN]`. A 2D descriptor block-shape subtype
exposes the two individual symbolic block widths, letting the body connect
C's block to its `(program_id(0), program_id(1))` tile starts and initialized
accumulator, and A's K block to its loop increment. The TMA transfers
require descriptor/shared-memory block and layout agreement. Changing only
C's annotated M block width produces new errors on the unchanged A TMA
load and final shared-memory store. Direct negative calls reject swapped B
orientation, mismatched B K or C N, wrong output role, and enabling the
transpose branch in this specialized signature. The direct WGMMA probes
accept either an untransposed `[BK, BN]` shared B tile or a `[BN, BK]` tile
after `.permute((1, 0))`, and reject passing the transposed storage directly.

There is a genuine checker limit inside the unchanged body: even with
`TRANSPOSE_B: Literal[False]`, Pyrefly marks both statements in the true
branch as unreachable **but still joins** the unreachable transposed-B tile
with the live `[BK, BN]` tile at `warpgroup_mma`. The sound MMA signature
therefore emits one expected error there. Annotating a second full copy with
`Literal[True]` also left this union and duplicate error, so that copy is not
included. The full `TRANSPOSE_B=True` kernel branch and its correlation
with a host B allocation of `[N, K]` remain **unverified**; the passing
direct permute probe is not a substitute for checking that complete branch.
This cannot be fixed by accepting arbitrary B ranks at WGMMA without losing
the A/B contraction guarantee.

A separate static Torch bridge constructs TMA descriptors from host A
`[M, K]`, B `[K, N]`, and C `[M, N]` and tests the declared direct-call
contract with mismatched B K and C N controls. Actual `kernel[grid]` launch
and the original Python wrapper are not checked. Full descriptor host extents
are not body-proved: annotation-only changes to B's full K extent or C's
full N extent, while preserving their tile shapes, produce **no new error
inside the unchanged kernel body** beyond the preexisting branch-union MMA
error. The body reads A's K dimension for the loop bound but does not
compare B/C extents or check grid coverage, which TMA's implicit
out-of-bounds handling also cannot establish. The coordinate-list overload
checks symbolic tile-start widths but does not track row/column ordering or
prove any K-loop iteration's address, alignment, or bounds. It does not
prove the number of legal warps, instruction shape, `get_instr_shape_n`
termination, device capability, descriptor bytes, mbarrier phase, or MMA/TMA
memory ordering. Torch element dtype remains a trusted nominal promise as
in the first WGMMA fixture.

The following `05-wgmma.py` fixture copies the complete original
`blocked_matmul_pipelined_kernel` body with identical AST. Its float16
descriptors declare host A `[M, K]`, B `[K, N]`, and C `[M, N]`, with
respective shared-memory blocks `[BM, BK]`, `[BK, BN]`, and `[BM, BN]`.
The shared-layout factory and descriptor constructor check that each layout
matches its block shape. Allocation of A and B shared-memory rings obtains
the *tile shape* from their layouts, while the result's buffer count remains
unknown. TMA loads independently check each descriptor's block dimensions
and layout against the indexed ring tile; WGMMA checks A/B contraction and
accumulator dimensions; the output shared-memory store checks `[BM, BN]`.
Changing only B's annotated block K width causes a WGMMA error in the
unchanged body; changing only C's annotated block N width causes errors at
the unchanged TMA load and output store. Negative probes also reject wrong
TMA tile width or layout and a descriptor constructed with a layout for a
different block shape.

The `[2] + descriptor.block_type.shape` expression is inferred as
`list[int]` by Pyrefly's built-in list addition, losing both the buffer count
and block dimensions. The narrow float16 ring allocator therefore trusts
its shape argument and recovers *only* tile dimensions from the independently
checked layout. Explicit accepted-gap probes show that `[3] + block`, `[]`,
and `[2] + an incompatible block` also pass this allocator; **rank, the two
buffers, and agreement of the allocator's actual shape argument are not
verified**. Nor are indexing bounds, phase changes, synchronization, races,
or complete grid coverage. Changing only B's *full host K extent* or C's
*full host N extent* in the signature produces no new errors in the body:
these allocation extents are declared interface obligations, not body-proved.
The static Torch bridge rejects incorrect B/C host dimensions when directly
calling the typed kernel, but does not validate the real wrapper or JIT grid
launch. Torch element dtype and GPU execution remain unverified.

The first kernel in `python/tutorials/gluon/06-tcgen05.py` is
`tmem_example_kernel`; its complete body retains the original AST. The
static-only input and output pointers promise contiguous float32 host arrays
of shape `[M, N]`. The kernel creates row and column ranges of those extents,
and the flattened address `row * N + column` must use the *declared* column
extent as its row stride before either unmasked global-memory access. The
global load produces a `[M, N]` tile; both layout conversions preserve its
shape; the Blackwell tensor-memory allocation has shape `[M, N]`; its store
rejects a different tile shape; its load returns `[M, N]` to the global
output store. Changing only the input pointer's annotated M extent produces
new diagnostics at the unchanged input pointer addition and TMEM store;
changing only the output pointer's annotated N extent produces errors at
the unchanged output store and pointer addition. A direct negative probe
rejects a row stride other than N, and another rejects storing a mismatched
tile into tensor memory. These are kernel-body checks, not just call-site
errors.

A separate static Torch adapter connects `[M, N]` inputs and outputs to the
kernel's declared pointer types and rejects an output with the wrong N.
The adapter **trusts** input contiguity and float32 dtype: neither property
is tracked on `torch.Tensor[[M, N]]`; it does not run the original
`kernel[(1,)]` wrapper. The stubs also do not prove that M and N are legal
power-of-two `arange` extents, that `[64, 64]` and column stride satisfy all
Blackwell tensor-memory hardware restrictions, that `num_warps` is 4 or 8,
or that required device capability and execution constraints hold. Even
though both pointer shapes are grounded in the body, physical allocation
extent and validity of the unmasked accesses rely on those host promises.

The next source-order `06-tcgen05.py` fixture copies the complete original
`small_mma_kernel` body with identical AST. A/B are float16 TMA descriptors
for `[M, K]` and `[K, N]`, C/D are float32 descriptors for `[M, N]`, and
the declared blocks are respectively `[BM, BK]`, `[BK, BN]`, and
`[BM, BN]` for both C and D. The three TMA input loads check block and
shared-layout agreement. In either `LHS_IN_TMEM` branch, the A tile is
checked through its tensor-memory store or remains in shared memory;
both `USE_COMMIT` branches check float16 A/B K contraction and their
`[BM, BN]` float32 tensor-memory accumulator. The C shared-memory load
and accumulator store agree with D's block; the D shared-memory store and
TMA output transfer retain that block and layout. Annotation-only changes
to B's *block K* cause new errors at both original MMA sites; changes to
D's *block N* cause new errors at C-to-accumulator store and both MMA sites.
Independent negative controls reject wrong B K, wrong accumulator N, and
float16 rather than float32 accumulator. A direct static Torch descriptor
bridge rejects wrong B full K, C full M, and D full N at the declared
kernel boundary; it does not check the original launch wrapper.

Full host allocation extents are **not** proved by the kernel body. Changing
only B's full K extent and D's full N extent in semantic annotations while
preserving their block widths yields *no new kernel-body errors*, since TMA
accesses `[0, 0]` and checks tiles rather than full arrays. The body does
not establish launch coverage, descriptor masking or padding, valid MMA
instruction/TMEM layouts, the relation of `tmem_block` to physical tile
shape, legal warp count, float16/float32 Torch backing dtypes, device
capability, barrier phase/count, or ordering of asynchronous operations.
Unlike the float16 layout, the existing float32 layout factory forgets its
source block dimensions: a pinned accepted-gap probe constructs C's
`[BM, BN]` descriptor using a layout created for `[BM, Other]`. Consequently
the declared C/D block shape reaches TMA and MMA checks, but this host-side
float32 layout agreement has not been established.

The following `06-tcgen05.py` fixture copies the complete original
`blocked_matmul_kernel` body with identical AST. For the untransposed
specialization `TRANSPOSE_B: Literal[False]`, its descriptors declare host
A `[M, K]`, B `[K, N]`, C `[M, N]`, and float16 tiles `[BM, BK]`,
`[BK, BN]`, `[BM, BN]`. Block-shaped TMA allocations and transfers check
descriptor/shared-memory tile and layout agreement. C's block dimensions
determine its program-ID tile starts, tensor-memory accumulator shape, and
output shared-memory store shape. In an annotation-only mutation, changing
only C's block-N width produces a *new* diagnostic at B's original TMA
load (`off_n` width no longer matches B's tile), separately from the
preexisting MMA branch error. Direct negative controls reject the wrong B
tile shape and host orientation, a wrong C dtype, and a wrong descriptor
tile start. The static Torch bridge rejects wrong B/C host extents and a
float16 layout constructed for a different descriptor block shape.

One original MMA call has an **expected checker error**. Pyrefly identifies
the `TRANSPOSE_B=True` arm as unreachable for `Literal[False]` but still
joins its `[BN, BK]` permuted B tile with the live `[BK, BN]` tile at
`tcgen05_mma`; the sound K-contraction signature rejects that union.
Changing only B's annotated *block K* modifies this already-existing MMA
diagnostic without adding any new body diagnostic, so B's contraction with
A is **not established by this full-body fixture**. A standalone MMA probe
accepts the correctly permuted `[BN, BK]` storage tile and rejects direct
use of that tile, but cannot prove the complete transposed kernel branch.
The true transpose branch and its required host B `[N, K]` interface remain
unverified; this fixture must not be counted as full transpose coverage.

Changing only B's *full host K extent* and C's *full host N extent* while
preserving block widths produces no new body diagnostics; these are only
declared interface obligations checked at the direct Torch bridge call.
The `Sequence` type for float16 TMA coordinates permits covariance through
the conditional list but cannot prove exactly two coordinates or their
order. Accepted-gap probes confirm that empty and reversed coordinate lists
pass; a foreign tile-start width is rejected. The tensor-memory-layout
factory likewise does not connect its block to the allocation shape:
`[BM, Other]` layout with `[BM, BN]` allocation passes an explicit probe.
No grid-coverage, K-loop bound or step positivity, physical Torch dtype,
Blackwell legality, descriptor padding, barrier phase, or asynchronous
memory-ordering proof is provided.

The final source-order `06-tcgen05.py` kernel is
`blocked_matmul_pipelined_kernel`. Its full original body and the preceding
`get_and_increment` helper have unchanged ASTs, with only kernel parameter
annotations added. The float16 descriptors declare A `[M, K]`, B `[K, N]`,
C `[M, N]`, and shared-memory blocks `[BM, BK]`, `[BK, BN]`, `[BM, BN]`.
The output's `[BM, BN]` block drives two float32 tensor-memory accumulator
allocations, checked against the A/B shared tiles at both upper/lower MMA
sites and against both output shared-memory stores. Annotation-only changing
B's *block K* yields six **new** errors at the original MMA calls. Changing
only C's *block N* yields independent new errors at three B TMA loads and
six MMA calls. These are body-grounded tile relationships, unlike the
preceding conditional-transpose kernel's existing MMA error.

The upper/lower row mapping uses `pid_m * (2 * BLOCK_M)`, which Pyrefly
retains as `GluonTileStart[2 * BM]`. A narrow symbolic half-stride operator
allows `off_m + BLOCK_M` while rejecting addition of an unrelated `Other`
dimension. A wrong destination descriptor row block rejects the doubled
start at TMA; the direct probe verifies both the upper and lower forms.
This arithmetic tracks symbolic *width*, not the numerical tile's location:
for example, `program_id * BM + BM // 2` is also accepted for an ordinary
`GluonTileStart[BM]`. Coordinate order, exact rank, alignment, no overlap,
and launch-grid coverage remain unproved.

The existing float16 ring allocator gets tile dimensions from each checked
descriptor layout, but `[2] + block_type.shape` widens to `list[int]` and
cannot establish the buffer count or actual allocation shape. Pinned
accepted-gap probes show `[3] + block_type.shape` and `[]` also pass.
The loop has an *expected Pyrefly non-convergent-fixpoint warning* because
its repeated K increments produce an expanding union of symbolic multiples
of BK; the type checker does not verify the number of iterations, ring index
or phase bounds, barrier ordering, the K-tail policy, or the exact upper/lower
program mapping. Changing only B's full host K extent or C's full host N
extent adds no body diagnostic. The static Torch bridge rejects wrong B/C
host dimensions on a direct signature call but neither checks the actual
`kernel[grid]` launch nor verifies physical Torch float16 backing: a float32
A tensor passed through a nominal float16 descriptor is an explicit
accepted-gap probe. TMEM layout legality and Blackwell device support also
remain unverified.

The first standalone `@gluon.jit` helper in
`python/tutorials/gluon/07-persistence.py` is `issue_loads`. Its complete
original body has unchanged AST; semantic parameter annotations describe
A `[M, K]` tiled `[BM, BK]`, B `[K, N]` tiled `[BK, BN]`, corresponding
layout-indexed shared-memory rings, M/N tile starts, a barrier ring and a
shared buffer-depth symbol. The original helper's predicate is preserved as
an optional scalar boolean for `mbarrier.expect` and both TMA loads. Changing
*only* B's descriptor block K while keeping its destination ring unchanged
produces a new diagnostic at the original B `tma.async_load`; changing only
the declared N tile-start width produces a new diagnostic at its original B
load coordinates. Wrong B block K, wrong A-ring layout, wrong B full host K,
and wrong N tile-start width also fail at direct calls to the helper. A
static Torch bridge builds A and B float16 descriptors from `[M, K]` and
`[K, N]` tensors and rejects a wrong B K length on a direct helper call.

This is a **JIT helper, not the host-launched persistent matmul kernel**.
The static bridge cannot validate an actual Python launch or produce the
output C shape; those belong to the forthcoming launchable entrypoint.
Changing *only* B's annotated full host K to an unrelated extent produces
no new body diagnostic: this helper loads the two descriptors independently,
and coordinate lists carry no host-bound provenance. A direct accepted-gap
probe loads A and B of different full K extents, and also accepts an empty
coordinate list. The TMA stub does not check coordinate rank or axis order.
The ring `.index` accepts out-of-range and negative indices, so the body
does not prove positive buffer count, phase synchronization, producer-index
bounds, byte-count accuracy, K-loop bounds, tile-start alignment, padding,
or the semantics of the `pred` condition. Torch element dtype remains
unverified: the nominal float16 descriptor constructor also accepts a
float32 tensor of the same shape. The two input blocks' shared BK symbol
is part of the declared interface but only each descriptor-to-ring
relationship, not an A-to-B K contraction, is checked by this helper.

The following `07-persistence.py` JIT helper, `issue_mma`, and the original
`WGMMA` and `MMAv5` aggregate classes retain their original executable-body
ASTs and class field annotations. The helper's parameter annotations
specialize both MMA methods to A `[BM, BK]` and B `[BK, BN]`; an
annotation-only mutation of B's ring K width adds *two new errors* at the
original `mma.issue_async_mma(...)` call, one from each implementation.
Direct negative calls reject the wrong B K for Hopper and Blackwell. The
library's actual aggregate decorator synthesizes the constructors from
class annotations, which the static `@dataclass_transform` stub models.

The strict intrinsic checks in both unchanged aggregate method bodies
produce **expected diagnostics**, rather than silently trusting an
unparameterized accumulator. Upstream's `WGMMA.acc` field is annotated
`Union[warpgroup_mma_accumulator, gl.tensor]`; `MMAv5.acc_tmem` is annotated
`tensor_memory_descriptor`, neither of which retains the accumulator's
`[BM, BN]` shape. A pinned false-positive probe constructs each aggregate
with an accumulator of `[BM, Other]` and passes it to a helper consuming
`[BM, BK] @ [BK, BN]`. These calls pass despite the wrong output width:
the helper validates the declared A/B K relation but **does not** prove that
either actual aggregate implementation has a matching accumulator shape.
The original `initialize` methods lack ordinary Python `@staticmethod`
annotations and produce expected self-binding errors in Pyrefly; making the
stubs or fixture treat these as statically typed factory methods would require
a separately validated representation of Gluon's JIT class semantics.
The aggregate's scalar `use_acc` and barrier descriptor fields also lose
enough information that their original intrinsic calls produce expected
type errors. The static experiment does not check barrier phases, ring
indices, MMA completion, the `num_buffers >= 2` runtime invariant, or any
host output contract. The first host-launched persistent kernel follows
these two independently checked JIT helpers.

The first host-launched kernel in `07-persistence.py`,
`matmul_pipelined_kernel`, has its full original executable-body AST and
semantic descriptor signature A `[M, K]` tiled `[BM, BK]`, B `[K, N]`
tiled `[BK, BN]`, C `[M, N]` tiled `[BM, BN]`. It accepts either original
`WGMMA` or `MMAv5` aggregate class and keeps the original `issue_loads`
and `issue_mma` helper calls. An annotation-only B *block K* mutation adds
six new errors at the two original load-helper and two original MMA-helper
call sites; the MMA helper checks A/B K for both implementation classes.
Changing only C's *block N* produces two new errors at the load-helper
calls: the output descriptor's block sets `off_n`, which no longer matches
B's tile width. Permanent direct-call negatives reject both incorrect
blocks, independently of these source-body mutation checks.

The descriptor's **full host extents are different from its block**.
Changing B's annotated full K produces two errors at `issue_loads` *call
sites*, but the helper's own unchanged body loads A and B independently
and never checks their full K equality. This is a trusted helper signature,
not an independent TMA/body proof. Changing C's annotated full N produces
no new original-kernel diagnostic. The static Torch adapter constructs
float16 descriptors from `[M, K]`, `[K, N]`, `[M, N]` arrays and rejects
wrong B K and C N on direct typed kernel calls for both Hopper and
Blackwell; these are **declared boundary conditions**. The corresponding
float32 A tensor still passes the nominal float16 constructor because
Torch dtype is not tracked. The adapter does not execute the original grid
launch or verify the source wrapper's body.

The original Python `matmul_pipelined` wrapper and its `select_mma_impl`
helper have also been copied without executable-body AST changes. Semantic
annotations reject wrongly shaped B/C tensors at calls *to that wrapper*,
and the original body constructs descriptors with shape-matching block
layouts. But `matmul_pipelined_kernel[grid]` has an explicit expected
not-subscriptable checker error: the actual launch's descriptor arguments
are not statically checked. Changing only B's full K in the wrapper's
parameter annotations adds **no wrapper-body diagnostic** beyond that
unchanged launch error. It would be incorrect to treat the typed wrapper
signature as validation of the Python-to-kernel launch.

The unchanged kernel has two further expected errors. A symbolic
`num_buffers >= 2` comparison produces a `bool`, not a statically proved
`Literal[True]`, and widening `gl.static_assert` to any bool would conceal
invalid buffer counts. The original Hopper accumulator field is annotated
as an unshaped union, so the output `c.to(dtype)` cannot be validated; the
known aggregate accepted-gap probe accepts a wrong output width. Thus
the kernel's output tile width is linked to B's tile start, but the MMA
*result* and C store value shape are **not proved**. Also unproved are
the selected device capability/warp count, legal buffer depth, index and
barrier phases, K-loop scheduling, tile bounds and padding, and coverage
of the two-dimensional launch grid.

The next source-order `07-persistence.py` kernel,
`persistent_matmul_kernel`, retains its full executable-body AST alongside
the original `PersistentTileScheduler` class, including its unmodified
`gl.tensor` field annotations. Its descriptor signature declares host A
`[M, K]`, B `[K, N]`, C `[M, N]` and respective tiles `[BM, BK]`,
`[BK, BN]`, `[BM, BN]`. An annotation-only mutation of B's block K adds
errors in the original kernel at both `issue_loads` and `issue_mma` calls:
the consumer helpers independently enforce the input tile contraction.
Changing only C's block N or full host N adds **no body-site diagnostic**;
the source reads C's host shape to initialize the scheduler, but neither
the scheduler's mapping to B tiles nor the MMA result's shape is proven.
B's full host K is checked only at the `issue_loads` call to a trusted
signature. Direct typed kernel calls reject wrong B/C host dimensions and
tile widths, but those rejections establish the declared interface, not
independent proof that the body computes the correct C array.

The scheduler's original fields are bare `gl.tensor`, so Pyrefly rejects
its constructor's integer arguments, modulo and floor division on its
fields, and use of its tile count in `range`. Its `get_tile` return is a
tuple of `Unknown` coordinates; the original kernel's `off_m` and `off_n`
therefore pass the helper calls without proving that the two tile starts
match `BM` and `BN`. A pinned probe also shows an independent false
positive: `gl.program_id(axis=0) * runtime_block` is accepted as
`GluonTileStart[BM]` even when `runtime_block` is plain `int` unrelated
to `BM`. Neither the source scheduler nor the operator stub currently
grounds the persistent program ID to its host output tile.
The unparameterized accumulator source field still makes the original
`c.to(dtype)` an expected error. The source Python wrapper is copied
without body changes: a typed Torch bridge and calls to its annotated
signature reject wrong B/C extents, while the actual bracketed JIT launch
has an expected not-subscriptable error and does not check those arguments.
The float16 descriptor constructor still accepts an incorrectly typed
float32 Torch input. Grid coverage, legal buffer depth, tile bounds,
device capability, synchronization, and output accumulator shape remain
unverified.

The next distinct JIT helper in `07-persistence.py`, `issue_loads_stealb`,
retains its original full body. The preceding grouped-scheduler factory
uses the same unparameterized `gl.tensor` fields as the basic scheduler:
neither scheduler's tile mapping is proved here. The new load helper
indexes B's ring with `producer % (num_buffers + stealb)` while A and
the barrier use `producer % num_buffers`. Its semantic signature ties A
`[M, K]` tiled `[BM, BK]` to B `[K, N]` tiled `[BK, BN]`, corresponding
shared tiles and typed tile starts. An annotation-only change to B's
block-column width adds two diagnostics at the unchanged B `tma.async_load`
call: the B tile and its typed column start disagree with the descriptor.
Direct calls also reject an incorrect B tile K, B host K, shared B tile
width, or N tile-start width. A Torch descriptor bridge rejects a wrong
B K extent, but its checker accepts a float32 tensor as the nominal
float16 input. There is no source Python wrapper for this helper and it
does not describe or verify an output C array.

The shared B ring type retains its tile shape and layout but **not its
allocation depth**: permanent probes pass arbitrary positive and negative
`stealb` values with exactly the same B ring. Its index operator accepts
indices without proving buffer capacity, bounds, synchronization, or
absence of zero divisors. The helper body's independent TMA loads do not
prove that A and B have equal full host K; that equality is only in its
declared helper signature. The following `issue_mma_stealb` helper and
host-launched `persistent_matmul_pipelined_kernel` are the next distinct
source-order kernels to investigate.

The next `issue_mma_stealb` helper and host-launched
`persistent_matmul_pipelined_kernel` preserve their complete source-body
ASTs; the latter's original Python `persistent_matmul_pipelined` wrapper
also preserves its body. The MMA helper's B buffer now uses
`consumer % (num_buffers + stealb)`, but its shape safety is the same
Hopper/Blackwell A `[BM, BK]` by B `[BK, BN]` contraction as `issue_mma`.
Changing only its B-ring K annotation produces two new errors at the
original `mma.issue_async_mma` call, one per aggregate implementation.
The B ring still has no typed capacity; a negative `stealb` passes even
though indexing and divisor legality are not proved.

The pipelined kernel declares A `[M, K]`, B `[K, N]`, and full and half
output descriptors over the **same** host C `[M, N]`, with tiles
`[BM, BN]` and `[BM, BN // 2]` respectively. Changing only B's annotated
block K adds errors at all original `issue_loads_stealb` and
`issue_mma_stealb` call sites, so the body checks the input contraction
at those helpers. A direct Torch descriptor bridge rejects wrongly shaped
B and half-output C arrays, and the original wrapper's semantic
parameters reject B/C mismatches at its own calls. Those are declared
Python-to-Triton interfaces: its actual `kernel[grid]` launch still has
an expected not-subscriptable error, so its launch arguments are not
checked. An annotation-only mutation of half-output full host N or its
block width causes **no new body-site diagnostic**. The shape of the
half-output view, the two stolen B buffers, and their offsets are not
validated against the output accumulator, despite precise descriptors.

There are expected source-body diagnostics for both `static_assert`
conditions, the original scheduler's erased tile count and predicate,
the unparameterized MMA accumulator's `.to`, use of `c_smem` after a
boolean branch, the mismatched full-output TMA store on the stolen B
buffer, and `reinterpret` on a ring tile whose stub does not model this
operation. Replacing these diagnostics with permissive overloads would
misstate output safety; in particular, no checker evidence grounds the
half-output C value or host bounds. The prototype does not prove buffer
depth at least three, STEALB's relation to extra ring capacity,
capability-dependent branches, barrier phases, or tile schedule coverage.
The Torch bridge still accepts a float32 source for nominally float16 TMA.
The next distinct example is tutorial `08-warp-specialization.py`.

The first source-order `08-warp-specialization.py` worker,
`load_partition`, retains its original executable-body AST. Its parameter
tuples describe input descriptors A/B with declared common host extent
`[Rows, Cols]` and tiles `[BR, BC]`, separately typed C descriptor and
shared ring, four barrier rings, an X tile start, and `YBLOCK`. The stub
models the actual shared ring's `.type.shape[0]` as its typed load-buffer
count, without pretending that an index is in bounds. Changing only B's
annotated block width yields a new error at the original B
`tma.async_load` call because descriptor and shared tile widths disagree.
Direct calls reject a wrong B block, wrong shared-ring/barrier depth, or
wrong X tile-start width. The Torch-to-descriptor bridge rejects a wrong
B host extent, but **that A/B host equality is a declared helper
interface**, not validated by its two independent TMA loads. The
runtime Y offset, derived as `i * YBLOCK`, is typed as a plain integer;
neither alignment to the Y tile nor coordinates staying inside the host
array are proved.

This load worker does not inspect or write C at all. A wrongly sized C
descriptor is an explicit accepted-gap probe both in the direct fixture
and the Torch bridge. Thus it establishes no complete Python-facing
vector-add interface on its own: later `store_partition`,
`compute_partition`, and the host-launched
`elementwise_add_warp_specialized_kernel` need independent checking.
The nominal float32 descriptor constructor still accepts a float16
Torch tensor. Barrier phases, TMA byte-count correctness, buffer-index
bounds, legal depth, and partition scheduling remain unverified.

The next `08-warp-specialization.py` worker, `store_partition`, also
retains its complete original body. Its output descriptor C and output
ring carry the same `[BR, BC]` tile width, and its `numel` tuple declares
the C host dimensions `[Rows, Cols]`. Changing only C's annotated block
width adds an error at the original `tma.async_store`, independently of
the helper's direct-call signature. Direct tests reject a mismatched C
ring tile and X tile-start width. A Torch bridge builds C's descriptor
from an actual output array and rejects a C host extent inconsistent
with the declared `numel` tuple. This host-extent relation is in the
worker's signature: the body loops over `gl.cdiv(ynumel, YBLOCK)` and
stores without checking that X/Y coordinates stay inside C's array.

The store worker never reads either input descriptor. Both A and B can
have the same *wrong* host extent relative to C without a checker error,
as pinned by direct and Torch accepted-gap probes; the output cannot
be claimed correct for the Python vector-add interface until the
compute partition and host-launched kernel connect these operands.
The original source calls `tma.store_wait(outstanding_stores)` with a
nonzero positional count, and the stub now reflects Triton's verified
`store_wait(pendings, read_only=True)` signature. The narrow barrier
`arrive(..., count=1, pred=...)` stub models this worker's source call.
Neither API proves pending-store count bounds, nonnegative buffer depth,
barrier phases, complete grid coverage, or output element dtype: a
float16 Torch output is still accepted as a nominally float32 TMA
descriptor. `compute_partition` is the next source-order kernel.

The original `compute_partition` worker is also checked with an unchanged
executable body. Its tuple of typed shared rings declares A, B and C
tiles with the same `[BR, BC]` shape, while the input and output rings
retain distinct load and store depths. An annotation-only mutation of
B's tile width adds an error at the original `a_val + b_val`; an
independent mutation of C's output-ring width adds an error at the
original `c_buf.store(c_val)`. Those body-site diagnostics validate
the A/B-to-C tile *value* relationship, complementing `load_partition`'s
descriptor-to-input-tile and `store_partition`'s output-tile-to-descriptor
checks. Direct calls reject wrong B width, B ring depth, C width, and
`YBLOCK` width.

This worker receives only shared rings, barriers, `ynumel`, `YBLOCK`
and a layout—**no host tensor or descriptor extents**. A pinned
accepted-gap probe passes a different `ynumel` with the same shared
buffers. The three individually typed workers thus do not by themselves
prove that the Python-visible A/B/C arrays have matching full host
extents; that relation still depends on typing their composition in the
next source-order host-launched
`elementwise_add_warp_specialized_kernel` and its wrapper. Buffer
index bounds, positive capacities, warp scheduling and barrier phases
also remain unverified.

The original `elementwise_add_warp_specialized_kernel` and its Python
`elementwise_add_warp_specialized` wrapper each retain their full executable
AST from `python/tutorials/gluon/08-warp-specialization.py`. Their annotated
interfaces require matching Torch A/B/C extents `[Rows, Cols]` and descriptor
tiles `[BR, BC]`. Direct typed calls reject B's wrong full host width and
C's wrong tile width; the separate Torch-to-descriptor adapter also rejects
wrong B and C host widths. Calls to the original wrapper reject wrong B/C
Torch shapes *at its declared signature*. The original bracketed JIT launch
cannot be checked and retains its expected not-subscriptable diagnostic.

The unchanged launch body checks part of this relationship. A narrow reflected
list-concatenation rule retains the `[BR, BC]` descriptor block shape in
`[num_load_buffers] + desc.block_type.shape`; three shared-ring allocations
therefore retain each descriptor's tile shape. A typed `gl.warp_specialize`
contract pairs each load, compute or store worker with its typed argument
tuple. Changing **only** the launched kernel's B descriptor annotation from
`[Rows, Cols, BR, BC]` to `[Rows, BR, BR, BC]` creates a new error at the
original `gl.warp_specialize` call. So does changing C's full host width to
`BR`, or changing B's block width from `BC` to `BR`. These diagnostics ground
the common descriptor host width and worker tile sizes in the unchanged
launch body; direct-call rejections alone do not establish that proof.

This is not complete launch validation: `gl.warp_specialize` types the
individual `(worker, arguments)` pairs but does not require all three
partitions, their order, or matching partition counts. A positive probe with
only the compute worker passes. The reflected list operation also accepts
`[depth, depth] + desc.block_type.shape` as a *three*-axis shape, even though
it has four runtime entries; buffer depth/rank is therefore not proved.
The original wrapper infers `block_shape = [XBLOCK, YBLOCK]` as a list rather
than a fixed shape, so its layout/descriptor constructors report four
expected diagnostics; the actual `kernel[grid]` launch remains untyped.
A nominally float32 descriptor constructed from a float16 Torch tensor is
an explicit accepted gap. Device, layout/stride, buffer capacity and
synchronization, coordinates staying within arrays, and complete grid
coverage are also unproved.

The first kernel in `python/tutorials/gluon/09-tma-gather-scatter.py`,
`async_gather_kernel`, and its original `async_gather` Python wrapper retain
their complete executable source ASTs. The semantic kernel parameters
declare a Torch source descriptor of host extent `[XMax, YMax]` with TMA
block `[1, BY]`, an offset vector `[BX]`, and an output pointer `[BX, BY]`
with individually named row and column strides. The unchanged kernel loads
the entire offset vector, retains its shape through `convert_layout`, checks
`[1, BY]` at Blackwell `tma.async_gather`, allocates a shared tile `[BX, BY]`,
and requires an unmasked `[BX, BY]` result at its final `gl.store`.
Annotation-only mutations of the *kernel* output row extent cause a new
original `gl.store` diagnostic; changing the offsets extent causes errors at
the original offset load and gather; changing the descriptor's first block
dimension from `1` to `BX` fails at `tma.async_gather`; and changing the
named row stride fails at original output pointer arithmetic. Direct tests
reject mismatched descriptor block shape, offsets length and output shape.

The original wrapper's semantic Torch arguments reject wrong offset-vector
length, and its inferred return is `[BX, BY]`. A separately typed direct
Torch descriptor bridge constructs `[XMax,YMax]` from the actual Torch
source array, and rejects wrong descriptor block axes; the wrapper's real
`kernel[(1,)]` invocation remains untyped with an expected error. There is
no validated Torch-to-`OutMatrixPointer2D` or Torch-to-offset-pointer
conversion in this overlay: the directly typed adapter's pointer types
are trusted declarations, while the original wrapper's output stride comes
from `out.stride()` without a checked Torch stride-to-pointer proof.
The full source host extents `[XMax,YMax]` are likewise declared-only in
this kernel: a descriptor constructed from a different source row count
still passes the unchanged body because gather values may address rows
outside that allocation and yield zeros. The checker does not ensure
correct row offset *values*, Y offset alignment, minimum gather size or
Blackwell-only execution. The `gather4` register/lane layout rule is
unproved: a pinned negative probe accepts the tutorial's invalid layout
`BlockedLayout([4],[32],[4],[0])` at 256 offsets, and accepts a misaligned
Y offset of `2`. Torch element dtypes are not checked by the nominal
float32 descriptor: a float32 offset tensor passes a wrapper expecting
integer offsets.

The following `async_scatter_kernel` and its original `async_scatter` wrapper
also preserve their complete executable ASTs. The destination descriptor
holds a Torch backing array `[XMax,YMax]` and tile `[1,BY]`, while the
offset-pointer length and host source tile rows are both `[BX]`; the source
tile width is `[BY]` with independently named row/column strides. In the
unchanged body, `gl.load` requires a full, unmasked `[BX,BY]` source,
`smem_src.store(src)` checks its value against the descriptor-layout shared
tile, and Blackwell `tma.async_scatter` checks descriptor block `[1,BY]`,
row offsets `[BX]`, and shared tile `[BX,BY]` together. Annotation-only
mutations of source rows or columns fail at the original `gl.load`, of
offset-pointer length at the original pointer addition and scatter call,
of descriptor first block dimension at the original `tma.async_scatter`,
of descriptor tile width at the original source load, and of row stride at
original source-pointer arithmetic. Direct calls reject incompatible
descriptor tiles, source shape and offset length.

The Python wrapper's declared Torch signature rejects wrong source rows,
columns, or offset-array length. A direct typed bridge constructs the
descriptor from an actual Torch destination allocation and rejects both
wrong block axes. The original `kernel[(1,)]` bracket launch still has an
expected untyped diagnostic, so neither its source stride forwarding nor
the Torch-to-`InMatrixPointer2D` or offset-pointer conversion is validated.
Destination full host `[XMax,YMax]` is declared only: a descriptor created
from another Torch row extent still type-checks the kernel, whose block
and row offsets determine scatter accesses. A pinned accepted-gap probe
permits the tutorial's invalid 256-offset `scatter4` layout and a negative
Y offset, which would be illegal for scatter on GPU. Values of the row
offset tensor, including forbidden negative rows, minimum offset count,
Y offset alignment, source/destination dtype compatibility, device
capability, and in-bounds/concurrent writes remain unproved. The wrapper
also accepts float32 row-offset tensors because these stubs track shape,
not element dtype.

The first fused gather/scatter matmul helper in that tutorial, `issue_loads`,
preserves its complete original executable AST. Its annotated inputs give X
host shape `[M,K]`, physical gather-descriptor block `[1,BK]`, and shared X
tile `[BM,BK]`. W has host shape `[K,N]` and regular TMA block `[BK,BN]`;
the offset pointer has declared host length `M`. The unchanged body constructs
an offset tile `[BM]`, indexes the X and W shared-buffer rings, checks the
gathered X tile at `tma.async_gather`, and checks W's block and shared tile at
`tma.async_load`. Annotation-only mutations to X's descriptor block width,
W's descriptor block width, the X ring's row tile, and `off_m` produce new
errors at those original load or offset-expression sites. A wrong W shared
tile width is also rejected by the helper's declared parameter interface.

These descriptors and offset pointers are trusted semantic helper parameters,
not yet derived from the tutorial's Python wrapper: a descriptor constructor
cannot safely assert that a Torch array, a physical TMA block, and the
layout-indexed gathered tile all match without checking their independent
inputs. Direct helper calls reject mismatched W host K or offset-pointer host
M, but changing just either of those helper annotations introduces no new
body error: those full host extents are declaration-only here. The shared
X/W tile contraction is checked by the following `issue_mma` helper; the
Torch host-to-kernel boundary is deferred to a launched-kernel slice. Buffer ring capacity,
the `num_buffers` divisor and index bounds, row-offset values, Blackwell
`gather4` layout restrictions, and barrier synchronization remain unproved.

That `issue_mma` helper also preserves its complete executable source AST,
with semantic types only on its parameters. Its X/W rings provide tiles
`[BM,BK]` and `[BK,BN]` to the *actual* shared-memory MMA method. Changing
only the helper's W-ring K annotation from `BK` to `BM` produces new errors
at the original `mma.issue_async_mma` body line for both Hopper and Blackwell.
Direct callers with a wrong W K tile or a barrier count inconsistent with
`num_buffers` are rejected as well. The source uses separate `index` and
`b_index`, both computed from the same unverified modulo divisor.

The accumulator's output `[BM,BN]` is **not** body-grounded. Existing source
aggregates `WGMMA` and `MMAv5` declare unparameterized fields, erasing the
dimensions of their accumulator; tests with wrong output accumulator width
are accepted. Without changing those original class declarations, parameter
annotations on this helper cannot prove its result shape. Likewise changing
only the helper's X-ring M or W-ring N annotation produces no original MMA
body-site error: the method can instantiate its unconstrained M/N variables
from either supplied tile. Shared-ring capacity has no depth parameter, so
matching a ring's real buffer count to the barrier depth is unverified.
Bounds and positivity of the count, barrier phase, asynchronous completion,
and the relationship to full host K remain outside this helper's proof.

The original launched `matmul_fused_gather_scatter_kernel` and Python
`matmul_fused_gather_scatter` wrapper preserve their complete executable ASTs.
The kernel's semantic inputs declare X host `[M,K]` with physical TMA block
`[1,BK]` and shared gather tile `[BM,BK]`, W `[K,N]` with regular block
`[BK,BN]`, and output host `[M,N]` with physical scatter block `[1,BN]`
and shared tile `[BM,BN]`. Both row-offset pointers declare host length `M`.
An annotation-only mutation of X's full host M gives new errors at the
original `issue_loads` call sites for the gathered offset pointer; changing
W's full host K also fails at those original helper calls. Changing only
the output descriptor's tile width yields a new `tma.async_scatter` error.
The unchanged body checks the gather/load tile sizes via `issue_loads`,
X/W tile contraction via `issue_mma`, and scatter offsets/tile shape at the
original Blackwell operation. Direct calls reject mismatched W host K,
output host width or tile width, and either offset-pointer host length.

That evidence does not prove output *full host* `[M,N]`: changing just the
output descriptor's host N annotation produces no new kernel-body error.
Output offset-pointer full M is also declared rather than grounded by
`async_scatter`, which checks its local `[BM]` tile. As in tutorial07, the
original aggregate scheduler yields a `gl.tensor` instead of a Python loop
count and the MMA accumulator field erases its `.to` method; the kernel
retains these two expected diagnostics. No result matrix dimensions,
buffer capacity, index bounds, barrier phase, or scheduler coverage are
proved by that incomplete accumulator/scheduler path.

The wrapper's declared Torch inputs require X `[M,K]`, W `[K,N]`, gather
and scatter index arrays each `[M]`; its returned allocation is `[M,N]`.
A direct typed host bridge constructs W's regular descriptor from a real
Torch `[K,N]` and rejects incorrect W K, but its X gather descriptor,
output scatter descriptor, and two offset pointers are *trusted inputs*.
The existing `TensorDescriptor.from_tensor` stub cannot soundly construct
the original wrapper's physical `[1,B]` descriptor with a distinct shared
layout `[BM,B]`: with known float16 dtype, both true tutorial calls and
wrong-column-layout controls produce expected constructor errors. A trial
overload that reused one type variable for physical and layout widths
accepted a physical `[1,BK]` paired with layout `[BM,BN]` and falsely
returned a descriptor whose physical width was `BN`; that unsound overload
is not present. The original wrapper obtains its dtype dynamically through
`getattr`, hiding the constructor incompatibility in that body, and its
original `kernel[grid]` call remains untyped. This fixture's tutorial07
import lacks the grouped scheduler factory and reports that expected
missing-attribute diagnostic. Neither the wrapper's descriptor forwarding
nor actual grouped grid coverage is validated. Torch dtype/device,
gather/scatter index values and alignment, and Blackwell execution are
also outside the current shape contracts.
An accepted negative control constructs W's nominal float16 descriptor from
a float32 Torch matrix and passes it to the kernel: Torch element dtype is
not carried by the current shape-only host annotation.

The first kernel in `python/tutorials/gluon/10-tcgen05-copy.py`,
`tcgen05_copy_kernel`, and its original `tcgen05_copy_example` wrapper
preserve their complete executable ASTs. The kernel takes a nominal float32
input pointer and an output pointer with full host extent `[M,N]` and
separately named input/output row and column strides. Its unmasked `gl.load`
requires the entire `[M,N]` input; the source value retains float32 dtype
and shape through the original `(M,N)` shared and tensor-memory allocations.
A narrow `tcgen05_copy` signature requires matching SMEM and TMEM tile
extents; its direct negative rejects a mismatched TMEM width. The tensor
memory `.load` and `gl.convert_layout` retain `[M,N]` up to the final
unmasked `gl.store`. Annotation-only mutations of the kernel's input width
and output width create new errors at the original `gl.load` and `gl.store`,
respectively; changing its output row stride fails at original pointer
arithmetic. Direct calls reject incompatible source/output dimensions and
stride roles.

Both local allocations use the *same* `(M,N)` expression, so their pairing
cannot be broken by a kernel parameter-only shape mutation at the original
`tcgen05_copy` line; the direct intrinsic negative checks that rule instead.
The original `gl.constexpr(mbarrier.MBarrierLayout())` call retains an
expected checker diagnostic because this overlay models `gl.constexpr` as
an annotation but not a callable runtime constructor. The copy's barrier
phase and async completion are therefore unverified. The input's float32
dtype comes from a *trusted pointer role*, not a checked Torch-to-pointer
conversion. An explicit typed adapter links Torch source and destination
extents to trusted pointers and rejects wrong shapes, while accepting a
float16 Torch source with a nominal float32 pointer. The original wrapper
allocates input/output Torch arrays `[M,N]` but its `kernel[(1,)]` launch
remains untyped; it also accepts float16 as `dtype` because the shape-only
host annotations do not constrain Torch element types. The shared layout
parameter is the real `NVMMASharedLayout` base type. The checker retains
copy *dimensions*, not hardware-supported swizzle widths, transposition,
tensor-memory block sizes, device capability, or the runtime relationship
between the shared and tensor-memory layouts.

The next source-order worker in Gluon tutorial 10 is
`matmul_accumulate_load_partition`. The fixture retains its full executable
body, the original unparameterized `PartitionArgs` class, and the executable
method bodies of the tutorial08 `Counter` dependency. Its sole worker
parameter declares A host `[M,K]` and tile `[BM,BK]`, B host `[K,N]` and
tile `[BK,BN]`, and initial C host `[M,N]` and tile `[BM,BN]` through a
separate structural protocol. This protocol is a **trusted declaration**:
the original `PartitionArgs` does not implement it, and a direct negative
call rejects the original aggregate. The C descriptor's original shape
read supplies scheduler extents, and its initial `tma.async_load` checks
the declared C descriptor against the independent C shared tile. Changing
*only* the parameter annotation so that the C shared tile has another
column extent adds a new diagnostic at that unchanged initial load.

A/B descriptor-to-ring pairing is **not proved in this worker body**.
The original `Counter.index` field is bare `gl.tensor`, not a scalar-index
type, so indexing the load rings produces expected checker errors. Changing
only the A ring inner extent in the worker parameter annotation adds **no**
error at the unchanged A `tma.async_load`, because Pyrefly stops checking
the downstream call after the invalid ring index. Separate direct calls to
the strict `tma.async_load` intrinsic reject wrong A inner K, B reduction
K, and C width, but only C has the annotation-only *original-site* proof.
The A/B full host K equality and C full host M/N equality are declared by
the protocol, not checked through the TMA operation, which consumes tile
extents and coordinates rather than whole allocations. The trusted protocol
also does not establish a Torch constructor or an original launcher bridge.
The original persistent scheduler returns an unshaped `gl.tensor` for the
loop count, and the copied `Counter` methods have expected diagnostics for
their scalar-like operations; there is no proof of tile scheduling, ring
capacity, load coordination, barrier phase, or execution on Blackwell.

The next source-order tutorial10 worker is named
`matmul_accmulate_mma_partition` in the original source. Its full executable
body is unchanged. Its parameter protocol extends the preceding trusted
load-worker contract with a float32 tensor-memory accumulator ring
`[AccDepth,BM,BN]` and matching accumulator barrier rings. This ring is
a trusted worker parameter type: a narrow three-dimensional allocation
overload now captures `[AccDepth,BM,BN]` at the launched source call,
but the original unparameterized aggregate does not forward those axes
to this worker's protocol.
The original unparameterized `PartitionArgs` does not implement the worker
protocol; a direct negative call pins that aggregate-to-protocol gap.

The original `tcgen05_copy(p.c_buf, acc_buf)` checks the initial C tile
`[BM,BN]` against the selected accumulator's `[BM,BN]`. The original
`tcgen05_mma(..., acc_buf, use_acc=True)` also checks its accumulator output
tile against the A/B tile's outer dimensions. Changing *only* the worker
parameter annotation so the accumulator ring width becomes `Other` adds
new diagnostics at **both unchanged intrinsic calls**; this proof survives
the separate error for indexing the accumulator ring with the unshaped
`Counter.index`. Direct intrinsic negatives independently reject C tile
width, accumulator row count, B K contraction and accumulator row count
at MMA. These direct probes do not substitute for original-body evidence.

Changing *only* the B descriptor block K in the worker annotation adds
**no new error** at the original MMA: its A/B buffer `.index` operations
already fail on bare `gl.tensor` Counter indexes, suppressing that
contraction check. B's full host K and the A/B block K equality are
declared by the protocol but not grounded by this worker. C's and A's
full host extents similarly do not reach the two compute intrinsics; C
host shape only feeds an unshaped scheduler. The prototype does not prove
that the copied C value precedes MMA execution, that two ring capacities
or phases match, that the scheduler visits all output tiles, or that the
source aggregate forwards these types into the worker. The original
launcher and Torch wrapper remain for the next source-order slice.

The original tutorial10 `matmul_accumulate_epilogue_partition` worker
preserves its complete executable AST. Its semantic parameter protocol
declares C host `[M,N]`, C and accumulator tiles `[BM,BN]`, output pointer
host `[M,N]`, and independent row and column strides. The unchanged
`gl.store` checks the accumulator `.load()` tile `[BM,BN]` against the
pointer's local `[BM,BN]` address, including the intervening
`gl.convert_layout`. Changing *only* the accumulator-ring column extent
produces a new error at that original store. Changing either supplied
output stride produces a new error at its original pointer expression.
These original-site checks require the unmasked tiled-store overload and
a **trusted** `EpilogueSchedulerFactory` return contract: its computed
`GluonTileId` values multiply by typed block extents to create local tile
starts. The original `PersistentTileScheduler` is rejected by a direct
negative assignment to this factory type; its unparameterized source
does not establish the promised tile IDs or iteration count.

The output pointer's *full* `[M,N]` allocation is declared, not checked
by that local store: changing just its host column extent produces no new
original-body diagnostic. The declared wrong-host call is rejected only at
the worker interface. Both scheduler axes share the same nominal tile-ID
type, so swapping their IDs is accepted; neither grid coverage nor output
edge bounds follow from a successful unmasked store. The original Counter
again yields expected errors for the scheduler loop, accumulator ring
index, and barrier phase. Neither its ring depth nor readiness is proved,
and the original unparameterized `PartitionArgs` does not implement the
semantic worker protocol. Torch allocation shape, output dtype, and the
bracketed kernel launch remain for the source-order launcher slice.

The original tutorial10 `matmul_accumulate_kernel` and Python
`matmul_accumulate` wrapper retain their full executable ASTs. The kernel
signature declares A `[M,K]` float16 tiles `[BM,BK]`, B `[K,N]` float16
tiles `[BK,BN]`, C `[M,N]` float32 tiles `[BM,BN]`, and output pointer
`[M,N]` with separately named row and column strides. Direct typed
kernel calls reject wrong B K, C N, output N, and row stride. Its A/B
shared allocations retain descriptor-layout tile types, and its C shared
allocation retains `[BM,BN]`. The narrow three-dimensional tensor-memory
allocator also now gives `[2,BM,BN]` a precise ring type at the original
call. The unchanged `PartitionArgs`
aggregate fields erase every semantic dimension and reject 13 typed
constructor arguments; the original three-worker `warp_specialize` call
also retains an expected signature error. No worker's typed parameter
protocol is therefore established as an output of this aggregate.
Changing *only* the kernel parameter annotation to give B another full
host K or C another full host N adds **no original-body diagnostic**:
the existing aggregate/dispatch errors do not prove a host relationship.
The A/B contraction and C/output full extents remain **declared** at this
launchable boundary, not validated by worker invocation.

The Python wrapper directly declares Torch A `[M,K]`, B `[K,N]`, and
C `[M,N]`; the original `torch.empty((M,N))` derives D from C's shape.
Wrong B K and C N calls are rejected at this declared wrapper signature,
and the inferred result is `[M,N]`. The wrapper resolves its Gluon dtypes
using dynamic `getattr`, so its original descriptor constructors do not
establish a static element dtype or the intended A/B/C shape relationships;
the `kernel[grid]` call is still untyped and reports the expected
not-subscriptable error. Its imported grouped-scheduler factory is also
missing from the minimal tutorial07 module fixture, with an expected
missing-attribute diagnostic. Wrong A or C Torch element dtypes are
accepted negative controls. A *separate*, explicitly trusted host adapter
constructs all three regular A/B/C descriptors using static float16 or
float32 layout choices and rejects bad A/B/C host dimensions when calling
the typed kernel. Its D pointer conversion links a Torch `[M,N]`
allocation to a declared pointer `[M,N]`, but does not check the real
Torch element dtype or stride identities; a mismatched D allocation is
rejected by the typed kernel call, not by the original bracket launch.
Neither that adapter nor the kernel signature validates the original
aggregate's field propagation, scheduler coverage, ring capacity or
phase, Blackwell tile legality, or correctness of the GPU computation.

The first tutorial11 Gluon kernel, `simple_mma_scaled_kernel`, retains its
complete executable source AST (35 body statements); only its parameters
receive semantic annotations. This bounded fixture represents the mxfp8
operand path: A is physically `[M,K]`, transposed B is physically `[N,K]`,
MX scale storage uses `uint8` pointer elements, and the default float16
output descriptor is `[M,N]`. Both scale pointers declare matching K-axis
extent `ScaleK`, but **no constraint proves `ScaleK == K // VEC_SIZE`**. The original
Python wrapper is not included, and no Torch-to-FP8 descriptor constructor
is asserted. Full host B K and the scale and output allocations' host
extents remain signature-only claims, not facts validated by this body.
An accepted negative call passes both scale pointers with a shared but wrong
host K extent, confirming that matching scale signatures do not close this gap.

The unchanged operand TMA loads check each descriptor's block and shared
layout. The original `tcgen05_mma_scaled` call checks A/B physical block K
after B's shared-memory transpose, both scale tensor-memory tile rows and
their *shared tile width*, and accumulator tile `[BM,BN]`; direct negative
intrinsic calls reject incorrect B K, accumulator N, B scale N, and B
scale width. Replacing **only** B's block-K parameter with `2 * BK` adds
an error at that original MMA call; replacing only output block N with
`2 * BN` adds errors there too. Each original `gl.load` preserves the
scale pointer's row/column stride identities and returns the tile shape
that `.store()` checks; replacing only A's row-stride parameter with
`2 * AScaleRowStride` adds an error at that original pointer expression.
The scale K offset arises from `k // VEC_SIZE`, but a scalar offset loses
its relation to the declared scale allocation extent; unmasked loads do
not establish bounds. The output `tma.async_store` checks the descriptor's
block tile against the computed accumulator tile, not output host M/N.

Changing **only** B's full host K to `2 * K` causes caller-signature errors
but none in the original kernel body: TMA consumes block tile and
coordinates, not full descriptor extent. Similarly neither the scale-host
extent nor output full host M/N reaches a checked array-bound comparison.
The body retains both original FP4 branches, but the parameter model is
restricted to mxfp8. Packed FP4's logical `2 * K` relationship, the
branch-dependent format/element packing, the scale-vector numerical
constraint, valid TMA coordinate order and bounds, divisibility of K by
`BK` and `VEC_SIZE`, GPU layout/swizzle eligibility and asynchronous
barrier timing are not proved. The FP8 `dtype == uint8` false branch is
precisely typed; this does not amount to checking either FP4 path.

The next tutorial11 `mma_scaled_contig_kernel` likewise retains all 35
original executable AST statements. Its new relationship is the physical
contiguous scale tile: the mxfp8 `VEC_SIZE=32` variant loads a *one-dimensional*
`[BM * (BK // 32)]` scale tile for A and `[BN * (BK // 32)]` for B, then
reshapes them to `[BM,BK//32]` and `[BN,BK//32]` for their tensor-memory
stores and the same strict scaled MMA. Its scale-pointer semantic types
carry these two tile dimensions; replacing only A's declared tile rows
with `2 * BM` or B's tile K with `2 * (BK // 32)` produces a new error at
the respective original `pointer + gl.arange(...)` expression. Negative
calls also reject mismatched input B host K or tile K, output host N,
either scale tile dimension, and the other scheme's `VEC_SIZE=16`.

The original `relayout_scales_contiguous` produces a four-dimensional
allocation `[MN//BLOCK_MN, (K//VEC_SIZE)//(BLOCK_K//VEC_SIZE), BLOCK_MN,
BLOCK_K//VEC_SIZE]`. The kernel treats it as a flat pointer. The prototype
declares each flattened allocation length independently of M, K, BK and
VEC_SIZE: an accepted negative call passes a wrong A flattened length, and
changing only that parameter annotation adds no kernel-body error. The
unmasked load therefore checks the *local tile length*, not full host
bounds, program-ID base offsets, loop index, 4D axis order, or total
allocation size. The original Python wrapper and physical Torch-to-pointer
constructor are not checked. A generic `tensor.reshape` expression
`Rows * Cols` does not currently infer its variables from the source's
1D tile, so the shape-preserving reshape rule belongs only to this
annotated contiguous-scale tile; it cannot establish arbitrary reshape
or FP4 packing correctness. B full host K and output full host M/N have
the same declaration-only limitations as the preceding kernel.

The next source-order `mma_scaled_packed_block_kernel` and its preceding
`unswizzle_scales_packed_block` helper preserve both complete executable
ASTs. This mxfp8-only static variant declares packed `uint8` scale TMA
descriptors with rank-five block shape
`[1,BM//128,BK//128,2,256]` (or `BN//128` for B) and independently declared
host repeat axes `[1,HostRepM,HostRepK,2,256]`. Allocation carries each
descriptor's physical tile and layout into shared memory; the original
rank-five `tma.async_load` calls accept the matching descriptor/shared
tiles. Direct TMA negative controls reject changing either tile row or K
axis independently, while accepting another full host K repeat extent.
Changing only A's descriptor block row parameter to `2 * BM` adds a new
diagnostic at the original `unswizzle_scales_packed_block` call because the
helper receives `[2*BM,BK]` but its `BLOCK_M` is `[BM]`. Changing only
the descriptor's full host K repeat extent adds no original-body error.

The helper reassigns its annotated `scales` parameter after the first
five-dimensional reshape. Pyrefly retains the parameter's original type,
rejecting that reassignment, the next `.permute`, and the final reshape:
seven expected diagnostics remain in the unchanged helper body. The
inferred helper return is the intermediate rank-five shape, **not** a
validated `[BM,BK//32]` tensor. Both unchanged `gl.set_auto_layout` calls
also retain expected missing-stub diagnostics. A proposed generic 2D
`set_auto_layout` signature was discarded because a direct negative probe
showed it accepted the rank-five intermediate as a 2D tensor-memory tile;
keeping it would falsely imply validation of scale values reaching MMA.
Thus this fixture proves the *TMA input tile* and a helper parameter's
tile relation, not the unswizzle, tensor-memory scale store, scaled MMA
scale compatibility, or numerical validity of packed swizzling. Neither
the physical Torch descriptor constructor nor the original Python wrapper
is modeled. Host allocation extents, divisibility by 128, rank-five
coordinate order, asynchronous synchronization and FP4 remain open.

The next pipeline helper, `async_mma_scaled_impl`, and its prerequisite
`unswizzle_scales_shared_memory` retain their complete executable ASTs;
their parameters describe the mxfp8 variant. The A shared tile is
`[BM,BK]`, B is physically `[BN,BK]` and is transposed at the original
`tcgen05_mma_scaled` call, and the declared accumulator is `[BM,BN]`.
Changing only B's physical K or the accumulator's N parameter creates
an additional error at that original MMA call. The rank-five packed
`uint8` scale inputs carry `[BM,BK]` and `[BN,BK]` logical tile parameters;
changing A's scale-row parameter alone creates an additional diagnostic
at the original unswizzle call. Direct callers also reject independent
B K, scale row/K, and accumulator N mismatches. The `uint8 == uint8`
dtype comparison is precisely true for this mxfp8 scale path, making
the original `VEC_SIZE=32` branch checkable.

The unswizzle helper's first reshape reassigns its annotated parameter,
which Pyrefly still treats as the original rank-five type. Its original
body consequently retains three expected errors, and its inferred return
is an intermediate rank-five shape, **not** a proved two-dimensional scale.
The async helper retains eight expected errors at the original scale
allocations and `tcgen05_copy` calls: the returned intermediate has no
validated `.dtype` or `.type.shape`, nor a compatible copy tile. The
subsequent scaled-MMA acceptance is therefore **not** evidence that
either scale reaches tensor memory with the required shape; inferred
unknown allocation dimensions can satisfy the call vacuously. Scale
value rearrangement, host extents, FP4 format selection, hardware layout
and async synchronization remain outside this helper's validated boundary.

The following pipeline `issue_loads` helper keeps its full 25-statement
executable AST unchanged. For mxfp8, operand TMA descriptors declare
physical A `[M,K]` with block `[BM,BK]` and transposed-storage B `[N,K]`
with block `[BN,BK]`; scale descriptors separately declare five-dimensional
`uint8` host repeat extents and packed blocks `[1,BM//128,BK//128,2,256]`
and `[1,BN//128,BK//128,2,256]`. Its original four `tma.async_load` calls
check each descriptor block against the corresponding indexed shared ring
tile. Annotation-only mismatches of A's ring K, A's scale descriptor block
rows, B's operand block K, or B's scale descriptor block K each add a
diagnostic at their corresponding original TMA call, independent of the
five baseline Counter-index diagnostics. Direct helper callers reject
wrong B full host K, scale block K, and A ring K at the *declaration*;
the original body does not inspect full host extents. Changing the scale
descriptor's full host repeat K or passing one with a different host
repeat K leaves all original-body diagnostics unchanged.

The original producer Counter's `.index` is a `gl.tensor`, not a Python
`int`. All four ring `.index()` calls and the barrier ring lookup retain
expected `tensor[Unknown]` versus `int` errors; no stub claims that the
source has proved a scalar ring index. The returned indexed-tile types
nevertheless let the original TMA calls reject independent physical tile
mismatches. Accepted direct negative controls pass one-coordinate,
three-coordinate, and negative-coordinate lists to a rank-two TMA load:
the sequence-typed coordinates do not prove order, rank, numerical offset
bounds, or that separate tile IDs enumerate the entire host arrays. The
relative host scale extents, allocation ring depth,
barrier phase, FP4 alternatives, and asynchronous completion remain open.

The next `issue_mma` helper retains all five original executable AST
statements. The four shared rings declare FP8 A `[BM,BK]`, physically
transposed-storage B `[BN,BK]`, and rank-five scale tiles parameterized
by `[BM,BK]` and `[BN,BK]`; the tensor-memory accumulator declares
`[BM,BN]`. Their indexed tile types reach the original call to
`async_mma_scaled_impl`: changing only A ring K, B ring K, A scale ring
rows, B scale ring K, or accumulator N independently produces an additional
diagnostic at that call (an A K change is reported against the scale K
arguments). Caller controls reject wrong B K, A scale rows,
and accumulator N at the *declared* helper interface, not as independent
proof of scale values. Both consumer and producer retain the actual
Counter's `gl.tensor` index and phase: six ring-index errors and one
phase-versus-`int` error remain expected at the original barrier, MMA,
and commit calls. No overload pretends that a Counter index is a scalar.
Swapping unrelated barrier ring depths remains accepted; the Counter's
ring capacity, phase progression, asynchronous completion, and the
previous helper's unproved scale unswizzle and tensor-memory copy are
not validated by this helper. It has no direct host-array interface.

The launched `mma_scaled_pipelined_kernel` keeps its complete 35-statement
executable AST; only its parameter annotations are semantic. Its declared
descriptor interface says FP8 A `[M,K]`, physically transposed B `[N,K]`,
FP16 output C `[M,N]`, blocks `[BM,BK]`, `[BN,BK]`, `[BM,BN]`, and packed
`uint8` scale descriptors with independent full host-repeat extents.
The original `[num_buffers] + block_type.shape` expressions now retain
each operand's and scale's physical tile through nominal reflected-shape
types. The associated shared-memory constructors check the shape's
tile parameters against its layout, returning indexed rings without
asserting ring depth. Direct wrong-layout and raw-list shape controls
are rejected. Annotation-only mutation of B's block N adds errors at
the original `issue_mma` calls where its ring no longer agrees with C's
accumulator tile. Mutating only A's scale block K adds errors at original
`issue_loads` and `issue_mma` calls. A narrow three-dimensional float32
tensor-memory allocation retains `[BM,BN]` through the accumulator load,
`acc.to(c_desc.dtype)`, shared-memory `.store`, and original C TMA store.
Independent intrinsic controls reject wrong accumulator/store tile widths;
because both the actual accumulator and shared/output tiles come from
the same C descriptor, **no independent annotation-only C block mutation
triggers a new error at the original store itself**.

The positive ring result does not validate `[num_buffers]` prefix length:
negative controls with `[depth, depth] + block_type.shape` are accepted
despite the extra physical axis. Eleven expected original-body errors
remain for the imported scheduler's tensor-valued tile count, tensor
Counter indices/phases, and tensor-valued predicates. The scheduler
factory's typed tile IDs are trusted, not proved by its upstream body;
tile traversal, ring capacity, async phase, bounds, and hardware resource
constraints therefore remain open. Caller controls reject wrong B full
host K and C full host N, but annotation-only mutation of C full host M
adds no error inside the original body. Its TMA store validates a local
tile, not the output array's full extent or edge coverage. A scale
descriptor with an independent wrong full host-repeat K is accepted;
the numerical scale-to-MMA path still inherits the earlier unswizzle
helper's expected errors. The original Python wrapper and its physical
Torch-to-descriptor constructors remain a separate task.

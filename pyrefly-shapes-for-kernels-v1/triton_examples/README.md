# Triton examples

`test_attention_forward.py` retains the executable bodies of Triton's tutorial
06 descriptor-based forward kernel and its two JIT helpers. The signature
annotations use legacy-style global `IntVar`s because Triton's JIT source
reader does not accept generic `def kernel[T]` syntax; `semantic_jit` removes
the semantic annotations and preserves the bodies. The checked host layout
accepts contiguous FP16 Q/K/V `[B,H,N,D]`, enforcing their matching axes,
head-dimension and block restrictions, and grid `(N/block_m, B*H)`.
`checked_attention_input` and `checked_attention_stats` present validated
Torch allocations as the flattened descriptor and `[B*H,N]` statistics
pointers. The adapter allocates matching output and statistics buffers, then
launches the typed kernel with grid metadata validated by the layout.

The original noncausal, non-FP8 forward body compiles to TTIR for CUDA 90
without a GPU, and CPU tests exercise allocation and grid rejection. This
does **not** prove GPU lowering or execution: no end-to-end GPU numerical
test was run, and Triton's descriptor kernel cannot be executed
in our CPU interpreter. The adapter deliberately requires equal Q and K
sequence lengths and whole blocks, since the kernel makes unmasked descriptor
loads and stores. The `tl.AttentionPointer` and stats-pointer types remain
stub-only views; the conversion functions return the original Torch tensors
after checking shape and contiguity. Direct unvalidated GPU launches bypass
these host checks. Three expected body diagnostics remain suppressed in the
FP8/warp-specialized paths (reshape/join and transposed dot); the v1 stubs
cannot prove those paths. Program-ID arithmetic and descriptor offsets still
need a proof of full grid coverage, not just a checked host grid size.

`test_layer_norm.py` preserves the forward kernel body from Triton's
`python/tutorials/05-layer-norm.py` (the backward kernels remain in v0). The
checked Torch boundary ties input and output to the same `[Rows, Cols]` and
element strides `[Stride, 1]`, weight and bias to `[Cols]`, and mean and
inverse standard deviation to `[Rows]`. Its output allocation deliberately
preserves padded input rows: the unchanged kernel takes **one shared** `stride`
for input and output. `row_output` launches exactly one program per row and
checks that `N`, `stride`, and `BLOCK_SIZE` match the host allocation and
metadata. The block width may be smaller than the column count because the
body loops over successive column blocks. CPU interpretation checks a 5×7
input with row stride 11 and block width 4 against Torch layer norm; the
unchanged frontend body also compiles to TTIR.

Important for a future v2: Pyrefly does not allow an augmented assignment to
change an annotated matrix pointer into a lower-rank row pointer. The 2D
pointer's `+= row * stride` therefore retains its annotated allocation type;
adding 1D column offsets obtains a tile checked against `Cols`. That rule
does **not** prove that the row offset was applied, that it lies inside
`Rows`, or that the mask guards the corresponding row's addresses. The checked
host shape/stride boundary and launch metadata are stronger guarantees than
the row-selection analysis. A future type-system hook for in-place pointer
selection or an explicit checked grid-to-row relation could address this
without editing upstream kernel bodies. The scalar statistics add a second
boundary pattern beyond the previous single-output examples.

`test_low_memory_dropout.py` preserves both executable kernels from Triton's
`python/tutorials/04-low-memory-dropout.py`. Generic unit-stride `[N]` pointer
annotations bind input, keep-mask, and output lengths to `n_elements`. The
explicit-mask boundary checks a boolean keep-mask against the floating-point
input's length and device; the seeded boundary checks an integer seed. Both
check `0 <= p < 1`, retain the input dtype on output, and use `tiled_output`
to bind launch count, length, and block width. Zero-length arrays return
without launching. The `tl.rand` stub gives program-shifted offsets a random
tile of the same width, which the unchanged `tl.where` carries to the store.
Frontend and CPU-interpreter tests cover both forms and partial final tiles.
Neither per-lane mask implication nor RNG equality with Pallas is claimed.
Here `x_keep` is an input *dropout decision*, distinct from Triton's
`mask=offsets < n_elements` bounds guard. The annotations check allocation
lengths and tile widths, but do not prove that every mask protects the exact
pointer expressions used by a load or store. `tl.rand(seed, offsets)` uses
global element offsets; changing tile width need not change the offset
associated with an element.

`test_vector_add.py` keeps the kernel body from Triton's
`python/tutorials/01-vector-add.py` alongside the tutorial's handwritten
`add(x, y)` host wrapper and an explicit checked boundary. The original kernel
statements are unchanged. The handwritten wrapper preserves the tutorial's
`empty_like` allocation, device assertion, `n_elements`, metadata-dependent
grid, and `BLOCK_SIZE=1024` launch. Its one CPU-testing adaptation is to read
the device from the inputs inside the function instead of binding the tutorial's
module-level `DEVICE` to an available GPU at import time. The tests cover JIT
annotation translation, real frontend compilation, interpreter execution,
and agreement between upstream and checked launches at block size 1024.
Only reusable offline TTIR setup lives in
`testing.py`. Run the same unittest module in normal and interpreter modes
from the v1 root; see `../README.md` for commands and the upstream wrapper
audit. Tests requiring the other mode are explicitly skipped in each run.
The checked call is `checked_add(as_host_tensor(x), as_host_tensor(y))`.
`checked_vector` validates unit-stride host inputs and output, preserves
their common symbolic length in its pointer types, and derives `n_elements`.
The evaluated kernel annotations also enable a pre-run check for direct
launches. The checked host function verifies shared lengths and devices,
metadata, and output allocation; kernel bodies remain unchanged. Its
`tiled_output` layout derives the program count from the checked 1D output
length and block size, then checks `n_elements` and `BLOCK_SIZE` at launch.
Empty inputs return without launching a zero-sized grid; the upstream wrapper
remains unchanged for comparison.

`test_fused_softmax.py` keeps the body of Triton's
`python/tutorials/02-fused-softmax.py` and its handwritten launch. Its
`InPointer[[Rows, Cols], [InputStride, 1]]` and
`OutPointer[[Rows, Cols], [OutputStride, 1]]` annotations relate the 2D
allocations to the element row-stride arguments, row index, 1D column tile,
mask, and reduction. The vector-add pointers use `[[N], [1]]`: their raw
offsets do not multiply an input stride. Selecting a row returns an ordinary
lower-rank `InPointer` or `OutPointer`; adding a tile of offsets produces
`InTilePointers` or `OutTilePointers`, whose stride must match the selected
pointer. These rules replace the specialized row-pointer and row-tile-pointer
classes. Both `BLOCK_SIZE` and `num_stages` remain Triton constexprs. The
upstream wrapper chooses launch occupancy using GPU properties and warmup;
the interpreter branch uses four programs because those properties and warmup
are not available on a CPU-only machine. The interpreter test exercises
non-power-of-two columns, padded input rows, and a grid-stride row loop. The
frontend-only test compiles the same body to TTIR. GPU occupancy and final
lowering have not been tested on a device.

The softmax wrapper remains handwritten: its row strides, dimensions, and
occupancy rule are explicit host-side choices. A `grid_stride_output` layout
binds actual output rows and columns and checks `BLOCK_SIZE` at launch. Its
grid may have fewer programs than rows: the unchanged kernel iterates over
rows using `tl.num_programs(0)`. The layout checks launch arguments and grid
range, and checks that its power-of-two block covers every output column.
It does not prove that the row loop matches the grid scheduling.
Its host signature accepts a checked `host_tensor.Tensor[[Rows, Cols],
[InputStride, 1]]` and returns `torch.Tensor[[Rows, Cols]]`. A Torch caller uses
`softmax(as_host_tensor(x))`; the converter checks the 2D shape and unit inner
element stride, and returns the *same* Torch object as a host-layout view.
Its one-argument overload records the row stride as unknown `int`. An optional
explicit target type can also check literal dimensions or element strides.
`checked_matrix` turns the host view into a static `tlt.InPointer` or
`tlt.OutPointer` view, deriving the row stride and dimensions; it also checks
output row non-overlap. The explicit launch no longer casts the kernel to
`Any`; its static signature checks pointer direction and argument count. The
kernel uses real, evaluable `tlt` annotation markers, so this example does
not require postponed annotations.

This is a partially static boundary, not full dependent typing: a generic
`int` row stride returned by the one-argument converter is not a fresh
symbolic stride bound to the allocation. Pyrefly also widens the kernel's
module-level `IntVar` dimensions and strides to `int` at the launch site. A
negative static probe rejected reversed input/output pointers and an
incorrectly shaped host tensor passed to `as_host_tensor`, but
**accepted** swapping the two row-stride arguments at the kernel call.
`@semantic_jit` registers a pre-run check of the evaluated pointer and scalar
annotations: actual array dimensions and element strides bind symbolic values,
and launch scalars must agree with them. The interpreter tests verify that
swapped strides and a mismatched input width fail before running the kernel.
Full *static* binding of the kernel's symbolic call signature remains future
work; no GPU execution or runtime overhead measurement has been performed.

`test_strided_copy.py` probes the general rule without changing either tutorial
body. Its evaluated `tlt.InPointer`/`tlt.OutPointer` annotations use the same
shape-and-stride allocation model as the checked host examples; direct launches
also validate their array dimensions, element strides, and scalar arguments
before running. These tests do not yet provide a separate checked Torch host
wrapper or a checked launch layout. Its 1D kernel reads a sliced array at
`offsets * input_stride`, while its
2D kernel selects a row using its row stride and reads columns using their
independent column stride. CPU tests use contiguous, sliced, and transposed
inputs; both kernels also compile to TTIR without a GPU. `tl.arange` supplies
logical indices and has no stride parameter: the kernels explicitly multiply
by the element stride to construct physical addresses. The stubs type raw
offsets as stride 1 and scaled offsets as their multiplier's stride. Masks
compare logical indices to logical lengths *before* scaling. A Pyrefly negative
probe rejected unscaled offsets into a strided pointer, multiplication by a
different stride, and comparison of physical addresses with a logical length.
These semantic checks do not prove that an arbitrary grid covers the whole
allocation or that all addresses are in bounds; the kernel author supplies the
shape/stride contract, and a safe Python boundary must validate that contract.

For the one-dimensional probes, `Mask` and tiled pointer arrays also carry an
offset-origin tag. `tl.arange(0, BLOCK_SIZE)` is local to the tile, while
adding `program_id * BLOCK_SIZE` marks a program-shifted tile. The tag survives
stride multiplication, so both `tl.load` and `tl.store` reject a mask derived
from unshifted offsets paired with program-shifted addresses. This is a
targeted consistency check, not full mask algebra: distinct expressions with
the same origin tag can still produce incompatible addresses and masks (for
example, using different `program_id` axes), and `tensor` does not retain the
mask after loading. Full validity propagation would need expression-sensitive
index provenance and rules for how tensor operations and store masks combine.

`test_matrix_multiplication.py` preserves Triton's tutorial 03 grouped
matmul body with annotations for full allocation shape, both element strides
per matrix, logical dimensions, block sizes, and grouping. Pyrefly checks the
tile-preserving kernel-to-kernel signature of the unchanged `leaky_relu`
helper. This helper retains `@triton.jit`: quoted
`tl.tensor[[BM, BlockN]]` annotations avoid Python evaluating a Triton type
that cannot be subscripted at runtime, while Pyrefly retains the exact tile
dimensions across the call. The activated matmul branch compiles to TTIR and
runs under the CPU interpreter; a negative type probe rejects a vector input.
The top-level kernel still uses `@semantic_jit` for pointer annotations and
launch checking. The prototype's `semantic_jit` returns a
`SemanticKernel[F]` object whose `__call__` re-infers the callable signature;
for polymorphic helpers it currently widens the return to
`tl.tensor[[int, int]]`. Direct `@triton.jit` preserves the helper's generic
callable type, including its exact tile dimensions. Triton's frontend accepts
this helper's **quoted** annotations because the AST visitor sees a string
literal, but visits and rejects the top-level kernel's unquoted symbolic
annotations such as `tlt.InPointer[[M, K], ...]`. This is a difference in
annotation syntax and wrapper typing, not an inherent restriction on JIT
helpers. Keep raw `@triton.jit` for annotated helpers in v1. The widening in
`SemanticKernel.__call__` is not a fundamental blocker: a dedicated helper
decorator could preserve the callable type if needed. Independently, a checked
launch interface could replace Triton's grid-indexed call with something
easier to type. Neither alternative is implemented or validated here. The
checked host call accepts padded left-hand rows and a transposed right-hand
input, constructs a 2D float16 output, and derives runtime strides; the
evaluated pointer annotations enable a launch hook that checks matching
symbolic scalar dimensions and strides. CPU interpreter tests compare with
Torch matmul; the normal-mode frontend compiles to TTIR.
The allocation pointers use the same generic `InPointer[Shape, Strides]` and
`OutPointer[Shape, Strides]` as the simpler kernels. Their 2D address operators
now return generic `InTilePointers` / `OutTilePointers` with allocation shape,
strides, tile shape, and a tag for which axis wraps. This preserves the
K-mask/load relation and output store shape; a negative type probe rejects
using a row mask where the left input needs a K-column mask. The expressions
that *construct* these address tiles and their masks still use v0-derived
row/column-specific types. Those are remaining prototype debt, not a reason
to keep a matrix-specific pointer type.
The boundary also checks the declared column stride of a transposed input;
the generic host-view overload must not infer unit stride when the caller
explicitly requests unconstrained column stride. A `tiled_output` layout
derives the one-dimensional grid from the checked output allocation and
`(block_m, block_n)`. It preserves the symbolic output shape and tile sizes;
its `launch` checks the actual `M`, `N`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, and
`GROUP_SIZE_M` arguments against the layout before invoking Triton. This
ensures the requested program count matches the output tiles but does not
prove the grouped PID arithmetic inside the unchanged kernel. The K-loop
mask's bound `K - k * BLOCK_SIZE_K` widens to `int`, so mask-provenance and
grouped program-ID coverage remain outside the static proof. No GPU
compilation or execution has been tested.

# Triton examples

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
metadata, and output allocation; kernel bodies remain unchanged.

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
occupancy rule are explicit host-side choices.
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
body. Its 1D kernel reads a sliced array at `offsets * input_stride`, while its
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
per matrix, logical dimensions, block sizes, and grouping. The checked host
call accepts padded left-hand rows and a transposed right-hand input,
constructs a 2D float16 output, and derives
runtime strides; the evaluated pointer annotations enable a launch hook
that checks matching symbolic scalar dimensions and strides. CPU interpreter
tests compare with Torch matmul; the normal-mode frontend compiles to TTIR.
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
explicitly requests unconstrained column stride. The K-loop
mask's bound `K - k * BLOCK_SIZE_K` widens to `int`, so mask-provenance and
grouped program-ID coverage remain outside the static proof. No GPU
compilation or execution has been tested.

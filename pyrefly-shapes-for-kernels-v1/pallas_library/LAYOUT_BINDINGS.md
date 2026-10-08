# Current Pallas layouts as bindings

These are representative bindings assembled by the typed factories in
`layout.py`; other examples have additional patterns and their own typed
factories.
Notation: `I(axes; block; map)` and `O(axes; block; map)` are input and
output bindings; `G(output, axis, block)` is one program-grid axis. Every
binding also carries its host shape and dtype; repeated axis names require
equal dimensions. `None` in a block *removes* that dimension from the kernel
Ref, whereas `block = —` means Pallas's default full-array Ref and no
`BlockSpec`. A grid axis with `exact=True` rejects a partial final tile.

| Kernel | Input bindings | Output bindings | Grid bindings |
| --- | --- | --- | --- |
| Vector add | `x, y: I((length,); (T,); (i) → (i,))` | `out: O((length,); (T,); (i) → (i,))` | `G(out, length, T)` (ceil division) |
| Explicit dropout | `x, keep: I((length,); (T,); (i) → (i,))` (keep may have boolean dtype) | `out: O((length,); (T,); (i) → (i,))` | `G(out, length, T)` |
| Seeded dropout | `x: I((length,); (T,); (i) → (i,))` (seed is captured by the kernel closure) | `out: O((length,); (T,); (i) → (i,))` | `G(out, length, T)` |
| Masked softmax | `row: I((cols,); —; —)` | `row: O((cols,); —; —)` | Empty: `vmap` supplies the outer rows |
| Layer norm | `row, weight, bias: I((features,); —; —)` | `row: O((features,); —; —)`, `mean, rstd: O((); —; —)` | Empty: `vmap` supplies the outer rows |
| Layer-norm input gradient | `x, weight, bias, dout: I((features,); —; —)`; saved `mean, rstd: I((); —; —)` | `dx: O((features,); —; —)` | Empty: one row per call |
| Blocked matmul | `a: I((rows, inner); (RM, K); (i,j) → (i,0))`, `b: I((inner, cols); (K, CN); (i,j) → (0,j))` | `c: O((rows, cols); (RM, CN); (i,j) → (i,j))` | `G(c, rows, RM, exact=True)`, `G(c, cols, CN, exact=True)` |
| Forward attention | `q: I((batch,queries,heads,dim); (None,BQ,None,D); (i,j,h) → (j,i,h,0))`, `k,v: I((batch,keys,heads,dim); (None,K,None,D); (i,j,h) → (j,0,h,0))` | `out: O((batch,queries,heads,dim); (None,BQ,None,D); (i,j,h) → (j,i,h,0))`, `lse: O((batch,heads,queries); (None,None,BQ); (i,j,h) → (j,h,i))` | `G(out, queries, BQ, exact=True)`, `G(out, batch, 1)`, `G(out, heads, 1)` |

For example, the matmul factory creates metadata equivalent to:

```python
binding_layout(
    kernel,
    inputs=(
        InputBinding(ShapeDtypeStruct((rows, inner), dtype),
                     ("rows", "inner"), (row_block, inner), lambda i, j: (i, 0)),
        InputBinding(ShapeDtypeStruct((inner, cols), dtype),
                     ("inner", "cols"), (inner, col_block), lambda i, j: (0, j)),
    ),
    outputs=(
        OutputBinding(ShapeDtypeStruct((rows, cols), dtype),
                      ("rows", "cols"), (row_block, col_block),
                      lambda i, j: (i, j)),
    ),
    grid=(GridBinding(0, 0, row_block, exact=True),
          GridBinding(0, 1, col_block, exact=True)),
)
```

The `rows`/`inner`/`cols` names check input/output host dimension agreement.
The block shapes yield kernel Refs of `[RM, K]`, `[K, CN]`, and `[RM, CN]`.
The index maps express which two-dimensional grid program sees each block;
their *arithmetic* is not checked by the generic assembler. The existing
`matmul_layout` additionally types each index-map lambda and the kernel Ref
signature, then delegates runtime assembly to `binding_layout`.

In attention the input `q` and output `out` have four host axes but their
blocks squeeze the batch and head axes, producing kernel Refs `[BQ, D]`.
The `k`/`v` Refs are `[K, D]`; the reordered statistics output has three host
axes but a Ref of `[BQ]`. The key-block size is used by the kernel's internal
scan, not by its `BlockSpec`: `attention_layout` additionally checks that
`keys` is divisible by that scan block, and checks its head-dimension and
statistics-dtype restrictions. Those checks are *not* inferred merely from
the bindings. For the row kernels, the full-row Ref is handed to the kernel;
masking within softmax is still a kernel-body concern.

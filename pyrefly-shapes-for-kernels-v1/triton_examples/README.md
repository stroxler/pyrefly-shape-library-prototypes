# Triton examples

`test_vector_add.py` keeps the kernel body from Triton's
`python/tutorials/01-vector-add.py` alongside the tutorial's handwritten
`add(x, y)` host wrapper and the generated wrapper. The original kernel
statements are unchanged. The handwritten wrapper preserves the tutorial's
`empty_like` allocation, device assertion, `n_elements`, metadata-dependent
grid, and `BLOCK_SIZE=1024` launch. Its one CPU-testing adaptation is to read
the device from the inputs inside the function instead of binding the tutorial's
module-level `DEVICE` to an available GPU at import time. The tests cover JIT
annotation translation, real frontend compilation, interpreter execution,
and agreement between handwritten and generated wrappers at block size 1024.
Only reusable offline TTIR setup lives in
`testing.py`. Run the same unittest module in normal and interpreter modes
from the v1 root; see `../README.md` for commands and the upstream wrapper
audit. Tests requiring the other mode are explicitly skipped in each run.

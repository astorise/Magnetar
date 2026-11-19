## Context

`NativeCpuExecutor` wraps a `ReferenceCpuExecutor`. The wrapped executor consults the optimized Provider's `PackedWeights` hooks:

- `run`, for a Kernel the Provider computes itself;
- `materialize`, for reads;
- `shape`, for resident shapes.

Only packed weights had a `run` path. The Runtime writes every unpacked weight through `ProviderExecutionApi::write_weight_tensor`, a write that is never repeated while the instance lives, so the Provider may keep any representation that reproduces every value.

## Decisions

### Compact storage, as on CUDA

A weight of at least `COMPACT_WEIGHT_MIN_ELEMENTS` (65,536) values whose every value has zero low 16 bits is stored as its high 16 bits. This is the CUDA Provider's rule. Biases and norm weights stay in f32.

A compact weight lives beside the packed ones. It is forgotten by every write that replaces its resource, as a packed weight is. It reads back unchanged through `materialize`, which counts the expansion like a packed weight's.

### Kernels by weight layout

A matmul's weight is stored either output-major (`[n, k]` used with `transpose_b`, a checkpoint's linear weights) or input-major (`[k, n]`).

- **Output-major.** Each output value is a dot product over a stored row. Rows of `a` are taken four at a time, so one weight load serves four rows. A single row (a decode step) uses four independent accumulators, so consecutive fused multiply-adds do not wait on each other.
- **Input-major.** Each stored row scales and adds into a block of output columns.

Both are written once per weight encoding (f32, bfloat16), so a loop never dispatches per load. The pool shares output columns, in blocks that are multiples of 8.

### Numerics

Products are accumulated in f32 with fused multiply-adds, in another order than the Reference CPU kernel's. Results agree to rounding, not bit for bit, as the packed kernels' do. Tests compare them within the existing tolerance (`2e-5` of the largest value).

## Risks / Trade-offs

- **AVX2 only.** The runner's Zen 5 cores also have AVX-512 and bfloat16 dot products, which llama.cpp may use. A wider kernel is a later step.
- **Non-matmul Kernels.** Norms, RoPE and element-wise Kernels still run on the Reference CPU kernels. They are small against the matmuls at these sizes.

## Why

`add-native-cuda-half-precision-compute`'s Phase 1 landed the `f32<->f16`/`bf16` conversion primitives in `magnetar-runtime`, explicitly inert until a Provider consumed them. This is Phase 2: a real, on-device native half-precision compute capability in `providers/cuda`, closing the README's "native CUDA F16/BF16 compute kernels do not exist yet" line for the first time -- for a narrow, honestly-scoped slice (elementwise `add`/`mul`), not full parity with the `f32` path.

## What Changes

- `providers/cuda` gains its own ported copy of the half-precision conversion primitives (`src/half_precision.rs`, same algorithm as `magnetar-runtime`'s, same "ported, not shared" convention externalized modules already use), plus a new `CudaHalfBuffer` type holding real 2-byte device-resident `F16`/`bfloat16` bit patterns.
- Four new CUDA kernels (`add_f16_kernel`, `mul_f16_kernel`, `add_bf16_kernel`, `mul_bf16_kernel`) compute via manual IEEE 754 bit manipulation inside the kernel (decode to `float`, compute, re-encode), avoiding any dependency on `cuda_fp16.h`/`cuda_bf16.h` -- NVRTC's minimal preprocessor environment does not reliably expose those headers (`kernels.cu`'s own header comment).
- `CudaKernels` gains `upload_half`/`download_half` (convert at the host/device boundary) and `add_half`/`mul_half` (dispatch to the new kernels).
- **Deliberately not done in this change**: `CudaDeviceBuffer` (the existing `f32`-only device buffer every other kernel uses) is untouched; no `KernelAdvertisement` is added, so this capability does not yet participate in the Runtime's planner/Kernel Registry dispatch -- it is reachable only by calling `CudaKernels`' new methods directly, exactly like `concat`/`copy_tensor_admitted` were before any advertisement-driven graph used them.
- **BREAKING**: none. Every existing type, method, and advertised Kernel is unchanged.

## Capabilities

### New Capabilities
(none -- this modifies the existing `cuda-provider` capability)

### Modified Capabilities
- `cuda-provider`: adds a new requirement, "CUDA Provider Offers Native Half-Precision Elementwise Compute", describing this directly-callable (not yet advertised) primitive. The existing "CUDA Provider Layout and DType Support" requirement's "f32 only" declaration is unaffected -- it describes the advertised/planner-selectable surface, which this change does not touch.

## Impact

- `providers/cuda/src/half_precision.rs` (new): ported conversion primitives, with the same exhaustive 65,536-bit-pattern round-trip tests as the `magnetar-runtime` original.
- `providers/cuda/src/kernels.cu`: four new kernels plus their bit-manipulation device helpers.
- `providers/cuda/src/kernels.rs`: `CudaHalfDType`, `CudaHalfBuffer`, `upload_half`/`download_half`/`add_half`/`mul_half`.
- Tests: 8 new tests in `tests_conformance.rs`, verified on real RTX 3070 Ti hardware -- `add_half`/`mul_half` for both `F16` and `bfloat16` are checked against an exact bit-for-bit reference conversion model (not a tolerance-based fudge factor), all passing on the first run; plus shape-mismatch and dtype-mismatch rejection tests. `cargo test`/`clippy --all-targets -- -D warnings`/`fmt --check`/`cargo doc --no-deps` all clean on `providers/cuda`.
- `design.md` documents Phase 3 (wiring an actual compute-dtype choice into the Runtime's planner) as still deferred, separate future work.

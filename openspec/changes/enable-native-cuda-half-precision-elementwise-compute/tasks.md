## 1. Implementation

- [x] 1.1 `providers/cuda/src/half_precision.rs`: ported `f16_to_f32`/`bf16_to_f32`/`f32_to_f16`/`f32_to_bf16`, same algorithm as `magnetar-runtime`'s Phase 1 original.
- [x] 1.2 `providers/cuda/src/kernels.cu`: `half_bits_to_float`/`float_to_half_bits`/`bf16_bits_to_float`/`float_to_bf16_bits` device functions (manual bit manipulation, no `cuda_fp16.h`/`cuda_bf16.h`), plus `add_f16_kernel`/`mul_f16_kernel`/`add_bf16_kernel`/`mul_bf16_kernel`.
- [x] 1.3 `providers/cuda/src/kernels.rs`: `CudaHalfDType`, `CudaHalfBuffer` (real `CudaSlice<u16>`, distinct from `CudaDeviceBuffer`), `CudaKernels::upload_half`/`download_half`/`add_half`/`mul_half`, `same_shape_half` (shape and dtype mismatch rejection).

## 2. Tests

- [x] 2.1 `half_precision.rs`'s own exhaustive round-trip tests (65,536 bit patterns each, `F16` and `bfloat16`) -- passed on the first run.
- [x] 2.2 `tests_conformance.rs`: `add_half_f16_matches_the_real_device_bit_manipulation_exactly`, `mul_half_f16_matches_the_real_device_bit_manipulation_exactly`, `add_half_bf16_matches_the_real_device_bit_manipulation_exactly`, `mul_half_bf16_matches_the_real_device_bit_manipulation_exactly` -- verified against real RTX 3070 Ti hardware, comparing the actual GPU kernel output to an exact reference conversion model (decode-then-re-encode through the same round-to-nearest-even algorithm), not a tolerance band. All 4 passed on the first run.
- [x] 2.3 `add_half_rejects_shape_mismatch`, `add_half_rejects_dtype_mismatch`: host-side validation rejects before any kernel launch.
- [x] 2.4 Full regression on real hardware: `cargo test --lib` (44 pre-existing + 8 new, all passed, 1 pre-existing hardware-gated test still `ignored` as before), `cargo clippy --all-targets -- -D warnings` clean, `cargo fmt -- --check` clean, `cargo doc --no-deps` clean (fixed 4 private-intra-doc-link warnings found on the first `cargo doc` run, by switching to plain-text references for the `pub(crate)` `half_precision` module instead of `[...]` intra-doc links), `cargo deny check` clean (no new dependency added).

## 3. Documentation

- [x] 3.1 `openspec validate enable-native-cuda-half-precision-elementwise-compute --strict` passes.
- [x] 3.2 README and `SUBMODULES.md` updated: the "native CUDA F16/BF16 compute kernels do not exist yet" line is now scoped accurately to `matmul`/`rmsnorm`/`rope`/`attention` specifically (still true for those); the "currently supports"/"Tachyon scope charter reconciliation" sections describe the new `add`/`mul`-only, not-yet-advertised capability honestly. `providers/cuda`'s own README also updated (commit `0cfbdf4`).

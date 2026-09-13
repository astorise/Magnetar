## 1. Implementation

- [x] 1.1 Investigated the two open questions from Phase 2's `design.md`: whether same-Provider multi-Kernel-per-Operator is structurally supported (yes, confirmed by `KernelRegistry`'s key/candidate-matching code and an existing cross-Provider test precedent), and where a compute-dtype request would need to be threaded (nowhere new -- `dtype_requirements` already derives directly from each resource's own declared `TensorDescriptor.dtype`).
- [x] 1.2 `providers/cuda/src/advertisements.rs`: `half_precision_advertisement` helper; `add-half`/`mul-half` entries added to `cuda_kernel_advertisements`, same `OperatorId` as `add`/`mul`, advertising `Float16`/`BrainFloat16`.
- [x] 1.3 `providers/cuda/src/executor.rs`: `half_dtype_from_descriptor` helper; `"add-half"`/`"mul-half"` dispatch arms in `run_invocation` (download f32-stored inputs, `upload_half`, compute via `add_half`/`mul_half`, `download_half`, re-upload as f32). The `f32`-only device allocation table is untouched (deliberate non-goal, see `design.md`).

## 2. Tests

- [x] 2.1 `tests_hardware_hot_path.rs`'s `half_precision_add_and_mul_dispatch_through_the_real_kernel_registry_on_real_hardware`: exercises the *exact* generic dispatch contract (`KernelSelectionRequest` -> `KernelRegistry::select` -> `KernelDispatchPlan::from_selection` -> `KernelDispatcher::revalidate` -> `submit_kernel`/`complete_kernel`), for all 4 combinations of `{Float16, BrainFloat16} x {add, mul}`, verified on real RTX 3070 Ti hardware against the same exact-bit reference conversion model Phase 2 established. Passed on the first run.
- [x] 2.2 `tests.rs`'s `kernel_advertisements_agree_with_availability` updated: advertisement count 11 -> 13, `add-half`/`mul-half` added to the expected name set.
- [x] 2.3 Full regression on real hardware: `cargo test --lib` (45 passed, 1 pre-existing hardware-gated test still `ignored`), `cargo clippy --all-targets -- -D warnings` clean, `cargo fmt -- --check` clean, `cargo doc --no-deps` clean.

## 3. Documentation

- [x] 3.1 `openspec validate wire-cuda-half-precision-into-kernel-registry-dispatch --strict` passes.
- [x] 3.2 README, `SUBMODULES.md`, and `providers/cuda`'s own README updated: the capability's "not yet advertised through the Kernel Registry" caveat is replaced with an accurate description of what is (dispatch through the standard contract, for `add`/`mul` in isolation) and is not (device-resident half-precision storage between calls; any production graph actually requesting it) now true.

## Why

`enable-native-cuda-half-precision-elementwise-compute`'s Phase 2 landed real, hardware-verified `add_half`/`mul_half` on `providers/cuda`, but explicitly deferred making it reachable through the Runtime's actual Kernel Registry/dispatch contract -- it was only callable by invoking `CudaKernels` methods directly, bypassing every real selection/dispatch mechanism a production graph would use. This closes that gap: a `KernelSelectionRequest` declaring a `Float16`/`BrainFloat16` resource now causes the Kernel Registry to select and dispatch to CUDA's native half-precision `add`/`mul`, through the identical generic contract (`KernelSelectionRequest` -> `KernelRegistry::select` -> `KernelDispatchPlan::from_selection` -> `KernelDispatcher::revalidate` -> `ProviderExecutionApi::submit_kernel`/`complete_kernel`) every other Kernel in this codebase already uses.

## What Changes

- `providers/cuda/src/advertisements.rs` gains two new `KernelAdvertisement`s, `add-half`/`mul-half`, distinct `KernelId`s under the *same* `OperatorId` as the existing `f32`-only `add`/`mul` -- an alternative-implementation shape the Kernel Registry already supports for cross-Provider candidates (Reference CPU vs. CUDA); this is the first same-Provider instance of it. They advertise `Float16`/`BrainFloat16`, never `Float32`.
- `providers/cuda/src/executor.rs`'s `run_invocation` gains `"add-half"`/`"mul-half"` dispatch arms: fetch the (`f32`-stored) inputs, convert to real half-precision device buffers via `upload_half`, compute via `add_half`/`mul_half`, convert the result back. The existing `f32`-only device allocation table (`Mutex<BTreeMap<TensorResourceId, CudaDeviceBuffer>>`) is untouched -- a half-precision resource is not persistently device-resident between invocations in this change; see Non-Goals.
- **Deliberately not done in this change**: the device allocation table does not become dtype-polymorphic (a half-precision resource round-trips through the host once per invocation rather than staying resident as half-precision bytes between calls); no Model Component or production graph requests `Float16`/`BrainFloat16` compute -- this remains reachable only by a caller that explicitly builds a `Float16`/`BrainFloat16` `KernelSelectionRequest`, proven here by a new hardware test, not by any real generation path.
- **BREAKING**: none. Every existing advertisement, dispatch arm, and storage behavior for `f32` is unchanged.

## Capabilities

### New Capabilities
(none -- this modifies the existing `cuda-provider` capability)

### Modified Capabilities
- `cuda-provider`: the "CUDA Provider Offers Native Half-Precision Elementwise Compute" requirement (added by Phase 2) is extended: the capability is now also reachable through the Kernel Registry's standard selection/dispatch contract, not only by calling `CudaKernels` methods directly.

## Impact

- `providers/cuda/src/advertisements.rs`: `half_precision_advertisement` helper, two new entries in `cuda_kernel_advertisements`.
- `providers/cuda/src/executor.rs`: `half_dtype_from_descriptor` helper, `"add-half"`/`"mul-half"` arms in `run_invocation`.
- Tests: `tests_hardware_hot_path.rs` gains `half_precision_add_and_mul_dispatch_through_the_real_kernel_registry_on_real_hardware`, exercising the *exact* generic dispatch contract (not a direct `CudaKernels` call) for `F16`/`BrainFloat16` x `add`/`mul` (4 combinations), verified on real RTX 3070 Ti hardware against the same exact-bit reference conversion model Phase 2 established -- passed on the first run. `tests.rs`'s `kernel_advertisements_agree_with_availability` updated for the 2 new advertisements (11 -> 13). Full regression (45 tests), `clippy --all-targets -- -D warnings`, `fmt --check`, `cargo doc --no-deps` all clean.

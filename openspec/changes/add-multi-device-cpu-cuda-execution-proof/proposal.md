## Why

`add-npu-and-tpu-provider-skeletons` closed the additional-Providers portion of the original scope charter item "multi-device execution, and additional Providers (Metal/ROCm/NPU/TPU)"; multi-device execution itself remained untouched. `multi_device_placement.rs` (3530 lines) already defines a real, independently unit-tested data-model/validation library for placement plans, but nothing in this repository had ever driven it from a real execution -- it is formally listed as a deferred capability (`FirstNativeDeferredCapability::MultiDevicePlacement`) and never referenced outside its own module and `conformance.rs`. This repository's tooling has exactly one real GPU, so full production ModelInstance-level multi-device placement (which the existing spec's scenarios describe -- "Model spans two GPUs", peer access, per-GPU memory budgets) cannot be built and verified end to end here. This change draws the same honest line as the Provider skeletons before it: a real, scoped, hardware-verified foundational proof using the two real, heterogeneous Devices this repository does have (Reference CPU + one real NVIDIA GPU), rather than a synthetic-only exercise of the existing data model or an unverifiable full implementation.

## What Changes

- New crate `integration-tests/multi-device-cpu-cuda`: a single `Runtime` registers both `ReferenceCpuProvider` (`magnetar-provider-cpu`) and `CudaProvider` (`magnetar-provider-cuda`) simultaneously and dispatches a real, two-stage `a + b + c` computation across both real Devices (stage 1 `add` on Reference CPU, explicit host-staged movement, stage 2 `add` on the real GPU), using the generic `KernelSelectionRequest` -> `KernelRegistry::select` -> `KernelDispatchPlan::from_selection` -> `KernelDispatcher::revalidate` -> `submit_kernel`/`complete_kernel` contract directly.
- Found and documents a real, previously-unobserved architectural fact: `KernelRegistry::select` ranks every compatible candidate across every registered Provider by fallback rank/pressure/cost -- it does not exclude a candidate merely because its Provider differs from the request's own `ResourceAffinity`. A caller wanting a specific Provider must pick its own candidate out of the full candidate list explicitly. No single-Provider test in this repository's history could have surfaced this, since a single-Provider Runtime's candidate list never has a competing Provider to lose a ranking contest against.
- Ties `magnetar-runtime`'s existing `multi_device_placement` types (`DeviceSet`, `MultiDevicePlacementPlan`, `PipelineStage`, `StageMovementEdge`) to a real execution's own data for the first time: built from the run's real `DeviceMetadata`/`DeviceAvailability`/`TensorResourceId`s after both stages have actually executed, progressed through the real `Building` -> `Validating` -> `Ready` state machine.
- `.github/workflows/quality.yml`: `submodule-integration`'s sweep includes the new crate (gracefully skips without a compatible GPU). `.github/workflows/gpu-runner-smoke.yml`: a new step runs it against the real self-hosted GPU runner; also fixes a real, separate, pre-existing gap found while verifying this (the workflow never checked out `loaders/gguf`/`formats/gguf`, both real dependencies of the already-existing `integration-tests/production-loading` step that runs immediately after).
- **BREAKING**: none. New crate, no existing public contract changes.

## Capabilities

### New Capabilities
(none)

### Modified Capabilities
- `multi-device-placement`: gains two real requirements reflecting what this change actually proved and found -- that `Runtime::register_provider` supports multiple simultaneous Providers with the Kernel Registry ranking across all of them (not filtering by requested Provider), and that `MultiDevicePlacementPlan`'s types can be populated from a real execution's own data.

## Impact

- `integration-tests/multi-device-cpu-cuda`: new crate.
- `.github/workflows/quality.yml`, `.github/workflows/gpu-runner-smoke.yml`: extended; the latter also gets an unrelated, pre-existing checkout-gap fix found along the way.

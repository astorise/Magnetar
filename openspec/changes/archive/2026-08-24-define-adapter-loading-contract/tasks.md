# Tasks

## 1. Adapter Scope

- [x] Define Adapter Artifact as inference data.
- [x] Document Adapter Artifact versus base Model Artifact.
- [x] Document Adapter Loading versus Model Loading.
- [x] Document Adapter Residency.
- [x] Document Adapter Activation.
- [x] Document Adapter versus Provider.
- [x] Document Adapter versus Kernel.
- [x] Document Adapter versus Generation.

## 2. Adapter Module

- [x] Create first-class `adapter` module or equivalent.
- [x] Export canonical adapter types from crate root.
- [x] Keep adapter contract platform-neutral.
- [x] Keep adapter contract independent from direct Provider selection.
- [x] Add module-level documentation.

## 3. Adapter Artifact

- [x] Define AdapterArtifactId.
- [x] Define adapter digest identity.
- [x] Define logical adapter name.
- [x] Define adapter revision.
- [x] Define adapter method.
- [x] Define base model compatibility metadata.
- [x] Define architecture compatibility metadata.
- [x] Define target module metadata.
- [x] Define dtype metadata.
- [x] Define tensor metadata.
- [x] Define quantization metadata where applicable.
- [x] Define license metadata.
- [x] Define provenance metadata.
- [x] Add artifact tests.

## 4. Adapter Methods

- [x] Define lora method.
- [x] Define qlora method.
- [x] Define ia3 placeholder.
- [x] Define prompt-tuning placeholder.
- [x] Define prefix-tuning placeholder.
- [x] Define custom method.
- [x] Reject unsupported methods.
- [x] Add method validation tests.

## 5. Base Model Compatibility

- [x] Validate base model identity.
- [x] Validate base model revision.
- [x] Validate architecture family.
- [x] Validate architecture implementation.
- [x] Validate hidden size.
- [x] Validate layer count.
- [x] Validate target module names.
- [x] Validate tensor shapes.
- [x] Validate tokenizer compatibility where relevant.
- [x] Validate dtype compatibility.
- [x] Validate quantization compatibility.
- [x] Validate Provider Capability compatibility.
- [x] Add compatibility tests.

## 6. Target Module Metadata

- [x] Define target module name.
- [x] Define target module role.
- [x] Define target tensor shape.
- [x] Define target layer selector.
- [x] Validate target module exists.
- [x] Validate target tensor shape.
- [x] Reject missing target module.
- [x] Add target module tests.

## 7. Adapter Loading Request

- [x] Define AdapterLoadingRequest.
- [x] Include request ID.
- [x] Include adapter artifact reference.
- [x] Include base model context reference.
- [x] Include target usage.
- [x] Include adapter method.
- [x] Include requested compute dtype.
- [x] Include residency policy.
- [x] Include activation policy.
- [x] Include merge policy.
- [x] Include memory budget.
- [x] Include required Capabilities.
- [x] Include optional session association.
- [x] Include priority.
- [x] Include timeout.
- [x] Include observability correlation ID.

## 8. Adapter Lifecycle

- [x] Define requested state.
- [x] Define validating state.
- [x] Define planning state.
- [x] Define allocating state.
- [x] Define materializing state.
- [x] Define ready state.
- [x] Define active state.
- [x] Define inactive state.
- [x] Define merging state.
- [x] Define merged state.
- [x] Define unmerging state.
- [x] Define draining state.
- [x] Define unloading state.
- [x] Define unloaded state.
- [x] Define failed state.
- [x] Define invalid state.
- [x] Define allowed transitions.
- [x] Add lifecycle tests.

## 9. Adapter Residency

- [x] Define AdapterResidencyId.
- [x] Define host residency.
- [x] Define pinned host residency.
- [x] Define device residency.
- [x] Define unified/shared residency.
- [x] Define provider-owned opaque residency.
- [x] Define browser linear memory residency.
- [x] Define future WebGPU buffer residency.
- [x] Define sharded residency.
- [x] Define mixed residency.
- [x] Associate residency with Memory Manager.
- [x] Associate residency with Resource Affinity.
- [x] Add residency tests.

## 10. Memory Manager Integration

- [x] Request adapter memory feasibility.
- [x] Account for adapter tensor memory.
- [x] Account for quantized adapter storage.
- [x] Account for compute-ready materialization memory.
- [x] Account for transform workspace.
- [x] Account for merge workspace.
- [x] Account for unmerge workspace where supported.
- [x] Account for transfer staging.
- [x] Account for pinned memory where applicable.
- [x] Account for session adapter budget.
- [x] Support pending allocation.
- [x] Add memory integration tests.

## 11. Activation

- [x] Define activation request.
- [x] Define activation scope operation.
- [x] Define activation scope session.
- [x] Define activation scope model-instance.
- [x] Define activation scope runtime.
- [x] Validate loaded adapter lifecycle.
- [x] Validate base model compatibility.
- [x] Validate session policy.
- [x] Validate memory residency.
- [x] Validate Resource Affinity.
- [x] Validate Provider/Device compatibility.
- [x] Validate batching compatibility.
- [x] Add activation tests.

## 12. Deactivation

- [x] Define deactivation request.
- [x] Deactivate operation-scoped adapter.
- [x] Deactivate session-scoped adapter.
- [x] Deactivate model-instance-scoped adapter.
- [x] Preserve or release residency according to policy.
- [x] Invalidate dependent cache state where required.
- [x] Add deactivation tests.

## 13. Multiple Adapters

- [x] Define single-adapter-only policy.
- [x] Define reject-multiple-adapters policy.
- [x] Define ordered multiple adapters placeholder.
- [x] Define weighted composition placeholder.
- [x] Validate deterministic order.
- [x] Reject unsupported composition.
- [x] Add multiple adapter tests.

## 14. Merge Versus Overlay

- [x] Define overlay strategy.
- [x] Define merge-on-load strategy.
- [x] Define merge-on-activation strategy.
- [x] Define provider-fused strategy.
- [x] Define disabled strategy.
- [x] Validate merge policy.
- [x] Prevent silent base model mutation.
- [x] Add strategy tests.

## 15. Base Model Mutation Tracking

- [x] Track merge source adapter.
- [x] Track affected base residency.
- [x] Track reversible status.
- [x] Track new residency state.
- [x] Track invalidated KV caches.
- [x] Track invalidated Prefix Cache entries.
- [x] Track unload/unmerge policy.
- [x] Add mutation tracking tests.

## 16. KV Cache Integration

- [x] Include active adapter set in KV cache compatibility.
- [x] Reject reuse with incompatible adapter set.
- [x] Invalidate cache on adapter activation where required.
- [x] Invalidate cache on adapter deactivation where required.
- [x] Support adapter-specific cache entries.
- [x] Add KV cache integration tests.

## 17. Prefix Cache Integration

- [x] Include active adapter set in prefix fingerprint where relevant.
- [x] Reject prefix reuse with incompatible adapter set.
- [x] Invalidate prefix entries on adapter change where required.
- [x] Add Prefix Cache integration tests.

## 18. Generation Integration

- [x] Allow GenerationRequest to reference adapter activation where policy allows.
- [x] Apply active adapter context during model forward.
- [x] Reject implicit adapter loading unless policy allows.
- [x] Reject silent adapter activation.
- [x] Add generation integration tests.

## 19. Sampling Integration

- [x] Include adapter set in determinism metadata where relevant.
- [x] Preserve Sampling boundary.
- [x] Add sampling determinism tests with adapter identity metadata.

## 20. Continuous Batching Integration

- [x] Include active adapter set in batch compatibility.
- [x] Validate adapter execution strategy compatibility.
- [x] Validate Provider fused adapter support where needed.
- [x] Validate Resource Affinity.
- [x] Reject incompatible adapter batch placement.
- [x] Add batching integration tests.

## 21. Provider Integration

- [x] Add Provider adapter capability advertisement.
- [x] Validate supported adapter methods.
- [x] Validate maximum adapter rank.
- [x] Validate supported adapter dtypes.
- [x] Validate merge strategies.
- [x] Validate fused adapter kernels.
- [x] Validate target module support.
- [x] Validate quantized adapter formats.
- [x] Keep Provider-owned adapter handles opaque.
- [x] Add Provider integration tests.

## 22. Device Integration

- [x] Track Device-bound adapter residency.
- [x] Reject incompatible Device use.
- [x] Require explicit movement or re-materialization.
- [x] Account for Device memory pressure.
- [x] Add Device integration tests.

## 23. Session Integration

- [x] Define allowed adapters policy.
- [x] Define maximum active adapters.
- [x] Define default adapter.
- [x] Define activation allowed flag.
- [x] Define deactivation allowed flag.
- [x] Define merge allowed flag.
- [x] Define adapter memory budget.
- [x] Define adapter sharing policy.
- [x] Define adapter unload on session close.
- [x] Add session integration tests.

## 24. Browser Compatibility

- [x] Keep Adapter Loading platform-neutral.
- [x] Account for browser linear memory.
- [x] Account for browser memory limits.
- [x] Account for future WebGPU buffers.
- [x] Avoid Wasmtime dependency.
- [x] Avoid native Provider loading requirement.
- [x] Return unsupported browser errors where needed.
- [x] Add wasm32 check where feasible.

## 25. Security And Privacy

- [x] Prevent raw adapter tensor exposure.
- [x] Prevent raw Provider adapter handle exposure.
- [x] Redact adapter metadata according to policy.
- [x] Ensure adapter artifact trust is enforced.
- [x] Ensure revoked adapter cannot load.
- [x] Add security tests.

## 26. Error Model

- [x] Define adapter-artifact-not-found error.
- [x] Define adapter-artifact-invalid error.
- [x] Define adapter-artifact-untrusted error.
- [x] Define adapter-artifact-revoked error.
- [x] Define adapter-method-unsupported error.
- [x] Define base-model-incompatible error.
- [x] Define architecture-incompatible error.
- [x] Define target-module-missing error.
- [x] Define target-tensor-mismatch error.
- [x] Define tokenizer-incompatible error.
- [x] Define storage-dtype-unsupported error.
- [x] Define compute-dtype-unsupported error.
- [x] Define quantization-unsupported error.
- [x] Define adapter-rank-unsupported error.
- [x] Define adapter-shape-mismatch error.
- [x] Define memory-feasibility-failed error.
- [x] Define memory-allocation-failed error.
- [x] Define adapter-loading-queued status.
- [x] Define adapter-loading-timeout error.
- [x] Define Provider-capability-unavailable error.
- [x] Define Provider-adapter-unsupported error.
- [x] Define Provider-not-ready error.
- [x] Define Provider-saturated error.
- [x] Define Device-unavailable error.
- [x] Define Device-memory-insufficient error.
- [x] Define activation-denied error.
- [x] Define activation-conflict error.
- [x] Define multiple-adapters-unsupported error.
- [x] Define merge-unsupported error.
- [x] Define merge-failed error.
- [x] Define unmerge-unsupported error.
- [x] Define unmerge-failed error.
- [x] Define KV-cache-incompatible error.
- [x] Define Prefix-Cache-invalidated status.
- [x] Define unload-failed error.
- [x] Define browser-feature-unsupported error.
- [x] Define internal-adapter error.

## 27. Observability

- [x] Emit adapter loading requested observation.
- [x] Emit adapter artifact validated observation.
- [x] Emit adapter compatibility checked observation.
- [x] Emit adapter loading validation failed observation.
- [x] Emit adapter residency planning started observation.
- [x] Emit adapter residency planning completed observation.
- [x] Emit adapter memory allocation requested observation.
- [x] Emit adapter memory allocation queued observation.
- [x] Emit adapter materialization started observation.
- [x] Emit adapter materialization completed observation.
- [x] Emit adapter ready observation.
- [x] Emit adapter activated observation.
- [x] Emit adapter deactivated observation.
- [x] Emit adapter merge started observation.
- [x] Emit adapter merge completed observation.
- [x] Emit adapter unmerge started observation.
- [x] Emit adapter unmerge completed observation.
- [x] Emit adapter load failed observation.
- [x] Emit adapter unload started observation.
- [x] Emit adapter unloaded observation.
- [x] Emit adapter cache invalidation observation.
- [x] Emit adapter batching compatibility failed observation.
- [x] Avoid raw adapter/model/prompt/handle logging by default.

## 28. Tests

- [x] Test valid adapter artifact.
- [x] Test untrusted adapter fails before allocation.
- [x] Test revoked adapter fails before allocation.
- [x] Test unsupported adapter method.
- [x] Test base model incompatible.
- [x] Test architecture incompatible.
- [x] Test missing target module.
- [x] Test target tensor mismatch.
- [x] Test dtype incompatibility.
- [x] Test quantization unsupported.
- [x] Test memory feasibility failure.
- [x] Test adapter loading lifecycle.
- [x] Test adapter activation.
- [x] Test adapter deactivation.
- [x] Test multiple adapters rejected by policy.
- [x] Test merge strategy unsupported.
- [x] Test silent base mutation forbidden.
- [x] Test KV cache invalidated on adapter change.
- [x] Test Prefix Cache invalidated on adapter change.
- [x] Test batching rejects incompatible adapter sets.
- [x] Test Provider adapter unsupported.
- [x] Test raw adapter tensors not exposed.
- [x] Test raw Provider handles not exposed.

## 29. Documentation

- [x] Document Adapter Loading Contract.
- [x] Document Adapter Artifact.
- [x] Document adapter methods.
- [x] Document base model compatibility.
- [x] Document target modules.
- [x] Document lifecycle.
- [x] Document residency.
- [x] Document activation/deactivation.
- [x] Document multiple adapters.
- [x] Document merge versus overlay.
- [x] Document KV cache relationship.
- [x] Document Prefix Cache relationship.
- [x] Document Generation relationship.
- [x] Document Batching relationship.
- [x] Document Provider relationship.
- [x] Document Session relationship.
- [x] Document browser compatibility.
- [x] Document non-goals.

## 30. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run wasm32 check where feasible.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run Adapter tests.
- [x] Run Model Loading tests.
- [x] Run Session tests where impacted.
- [x] Run Generation tests where impacted.
- [x] Run KV Cache tests where impacted.
- [x] Run Prefix Cache tests where impacted.
- [x] Run Batching tests where impacted.
- [x] Run Provider conformance tests where impacted.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify adapters do not select Provider/Device directly.
- [x] Verify adapter memory is Memory Manager-owned.
- [x] Verify adapter activation is explicit.
- [x] Verify base model mutation is never silent.
- [x] Verify raw adapter tensors are not exposed.
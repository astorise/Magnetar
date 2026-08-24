# Tasks

## 1. Model Loading Scope

- [x] Define Model Loading as Runtime-owned process.
- [x] Document loading versus Model Artifact validation.
- [x] Document loading versus Model Residency.
- [x] Document loading versus Model Instance.
- [x] Document loading versus Provider execution.
- [x] Document loading versus Session creation.
- [x] Document loading versus KV cache creation.

## 2. Model Loading Module

- [x] Create first-class `model_loading` module or equivalent.
- [x] Export canonical model loading types from crate root.
- [x] Keep loading platform-neutral.
- [x] Keep loading independent from direct Provider selection.
- [x] Add module-level documentation.

## 3. Loading Request

- [x] Define ModelLoadingRequest.
- [x] Include request ID.
- [x] Include Model Artifact reference.
- [x] Include target usage.
- [x] Include requested compute dtype.
- [x] Include requested storage handling.
- [x] Include quantization policy.
- [x] Include sharding policy.
- [x] Include residency policy.
- [x] Include memory budget.
- [x] Include placement preference as policy input only.
- [x] Include required Capabilities.
- [x] Include optional session association.
- [x] Include cache policy.
- [x] Include priority.
- [x] Include timeout.
- [x] Include observability correlation ID.

## 4. Artifact Preconditions

- [x] Validate manifest exists.
- [x] Validate manifest schema version.
- [x] Validate artifact digest.
- [x] Validate required parts.
- [x] Validate required shards.
- [x] Validate architecture metadata.
- [x] Validate tokenizer association where required.
- [x] Validate dtype metadata.
- [x] Validate quantization metadata.
- [x] Validate trust policy.
- [x] Validate revocation status.
- [x] Validate license policy where enforced.
- [x] Fail before memory allocation when preconditions fail.

## 5. Loading Lifecycle

- [x] Define requested state.
- [x] Define validating state.
- [x] Define planning state.
- [x] Define allocating state.
- [x] Define materializing state.
- [x] Define ready state.
- [x] Define active state.
- [x] Define draining state.
- [x] Define unloading state.
- [x] Define unloaded state.
- [x] Define failed state.
- [x] Define invalid state.
- [x] Define allowed transitions.
- [x] Add lifecycle tests.

## 6. Architecture Implementation

- [x] Resolve compatible architecture implementation.
- [x] Support Runtime-native implementation.
- [x] Support Component-based implementation.
- [x] Support Provider-assisted implementation.
- [x] Support test fixture implementation.
- [x] Reject missing architecture implementation.
- [x] Ensure architecture does not create Provider identity.
- [x] Add architecture tests.

## 7. Provider Compatibility

- [x] Validate required Capability versions.
- [x] Validate operation family support.
- [x] Validate dtype support.
- [x] Validate layout support.
- [x] Validate quantization support.
- [x] Validate memory placement support.
- [x] Validate data movement support.
- [x] Validate Provider readiness.
- [x] Validate Provider pressure policy.
- [x] Add Provider compatibility tests.

## 8. Device Placement

- [x] Validate Device availability.
- [x] Validate Device memory capacity.
- [x] Validate Device memory pressure.
- [x] Validate Device dtype support where relevant.
- [x] Validate Device placement compatibility.
- [x] Ensure Model Artifact does not select Device directly.
- [x] Add Device placement tests.

## 9. Memory Manager Integration

- [x] Request model loading memory feasibility.
- [x] Account for model weights memory.
- [x] Account for model config memory.
- [x] Account for quantized storage memory.
- [x] Account for compute-ready materialization memory.
- [x] Account for dequantization workspace.
- [x] Account for sharded loading memory.
- [x] Account for transfer staging memory.
- [x] Account for pinned memory where applicable.
- [x] Account for browser memory constraints.
- [x] Support pending allocation.
- [x] Add memory integration tests.

## 10. Residency Plan

- [x] Define ModelResidencyPlan.
- [x] Include artifact reference.
- [x] Include model architecture.
- [x] Include target compute dtype.
- [x] Include storage dtype.
- [x] Include quantization handling.
- [x] Include shard placement.
- [x] Include memory placements.
- [x] Include Provider/Device bindings where resolved.
- [x] Include required data movement.
- [x] Include temporary workspace.
- [x] Include expected resident size.
- [x] Include loading phases.
- [x] Include fallback options.
- [x] Include unload policy.
- [x] Include diagnostics.
- [x] Ensure no raw native handles are exposed.

## 11. Model Residency

- [x] Define ModelResidencyId.
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

## 12. Loading Phases

- [x] Define read-manifest phase.
- [x] Define validate-parts phase.
- [x] Define open-artifact-bytes phase.
- [x] Define validate-shards phase.
- [x] Define plan-memory phase.
- [x] Define allocate-host phase.
- [x] Define allocate-device phase.
- [x] Define materialize-weights phase.
- [x] Define dequantize-or-transform phase.
- [x] Define transfer-to-device phase.
- [x] Define initialize-provider-state phase.
- [x] Define validate-ready phase.
- [x] Define publish-model-context phase.
- [x] Add phase diagnostics tests.

## 13. DType Handling

- [x] Validate storage dtype.
- [x] Validate requested compute dtype.
- [x] Validate allowed conversion.
- [x] Validate temporary workspace.
- [x] Validate Provider compute dtype support.
- [x] Validate Memory Manager storage/compute accounting.
- [x] Record dtype conversion in residency plan.
- [x] Add dtype tests.

## 14. Quantization Handling

- [x] Validate quantization format.
- [x] Validate quantization metadata.
- [x] Decide direct quantized execution.
- [x] Decide load-time dequantization.
- [x] Decide lazy dequantization.
- [x] Decide Provider-specific transform.
- [x] Reject unsupported quantization.
- [x] Record quantization handling in residency plan.
- [x] Add quantization tests.

## 15. Sharded Loading

- [x] Validate shard list.
- [x] Validate each shard digest.
- [x] Validate tensor-to-shard mapping.
- [x] Define sequential loading policy.
- [x] Define parallel loading policy.
- [x] Define single-Device placement policy.
- [x] Define multi-Device split placeholder.
- [x] Define host lazy shard placement.
- [x] Reject unsupported sharding layout.
- [x] Add sharding tests.

## 16. Lazy Loading

- [x] Define lazy loading policy.
- [x] Validate artifact identity before lazy load.
- [x] Validate trust before lazy load.
- [x] Represent pending residency.
- [x] Load required parts on demand.
- [x] Define lazy loading failure behavior.
- [x] Add lazy loading tests or placeholders.

## 17. Partial Loading

- [x] Define partial loading policy.
- [x] Prevent ready state if required parts are missing.
- [x] Allow partial context only for explicit target usage.
- [x] Expose partial status.
- [x] Add partial loading tests.

## 18. Session Integration

- [x] Allow session to reference loaded model context.
- [x] Allow policy-controlled implicit loading during session creation.
- [x] Reject session creation if model unavailable and implicit loading disabled.
- [x] Ensure session close does not necessarily unload model.
- [x] Add session integration tests.

## 19. KV Cache Integration

- [x] Ensure model loading does not create KV cache.
- [x] Invalidate associated KV caches on model unload where policy requires.
- [x] Prevent KV cache reuse after incompatible reload.
- [x] Add KV cache invalidation tests.

## 20. Adapter Preparation

- [x] Allow adapter artifact references.
- [x] Validate adapter/base model compatibility placeholder.
- [x] Prepare adapter residency placeholder.
- [x] Do not define full adapter activation in this change.
- [x] Add adapter placeholder tests.

## 21. Browser Compatibility

- [x] Keep loading contract platform-neutral.
- [x] Account for browser linear memory.
- [x] Account for browser memory limits.
- [x] Account for future WebGPU buffers.
- [x] Reject native pinned memory requirements on browser.
- [x] Reject native Provider loading requirements on browser.
- [x] Avoid Wasmtime dependency.
- [x] Add wasm32 checks where feasible.

## 22. Unload

- [x] Define unload request.
- [x] Prevent new inference use during unload.
- [x] Drain active sessions or operations according to policy.
- [x] Invalidate or release associated KV caches.
- [x] Release Memory Manager resources.
- [x] Release Provider-owned resources.
- [x] Update residency state.
- [x] Emit unload observations.
- [x] Add unload tests.

## 23. Reload

- [x] Define reload request.
- [x] Validate reload as new loading process.
- [x] Support Provider change reload.
- [x] Support Device recovery reload.
- [x] Support dtype change reload.
- [x] Support quantization mode change reload.
- [x] Support artifact update reload.
- [x] Prevent silent mutation of existing context unless policy permits.
- [x] Add reload tests.

## 24. Failure Cleanup

- [x] Clean up after validation failure.
- [x] Clean up after memory allocation failure.
- [x] Clean up after materialization failure.
- [x] Clean up after Provider initialization failure.
- [x] Mark context failed or invalid where appropriate.
- [x] Prevent failed context inference use.
- [x] Add failure cleanup tests.

## 25. Security And Trust

- [x] Ensure loading cannot bypass artifact trust.
- [x] Ensure Provider materialization cannot trust untrusted bytes.
- [x] Prevent raw loaded weights exposure.
- [x] Prevent raw Provider memory handle exposure.
- [x] Prevent Component access to model memory unless authorized inference
      contract allows it.
- [x] Add security tests.

## 26. Error Model

- [x] Define model-artifact-not-found error.
- [x] Define model-artifact-invalid error.
- [x] Define model-artifact-untrusted error.
- [x] Define model-artifact-revoked error.
- [x] Define architecture-unsupported error.
- [x] Define architecture-implementation-missing error.
- [x] Define tokenizer-incompatible error.
- [x] Define required-part-missing error.
- [x] Define shard-missing error.
- [x] Define shard-digest-mismatch error.
- [x] Define storage-dtype-unsupported error.
- [x] Define compute-dtype-unsupported error.
- [x] Define dtype-conversion-unsupported error.
- [x] Define quantization-unsupported error.
- [x] Define quantization-transform-failed error.
- [x] Define memory-feasibility-failed error.
- [x] Define memory-allocation-failed error.
- [x] Define loading-queued status.
- [x] Define loading-timeout error.
- [x] Define provider-capability-unavailable error.
- [x] Define provider-not-ready error.
- [x] Define provider-saturated error.
- [x] Define device-unavailable error.
- [x] Define device-memory-insufficient error.
- [x] Define placement-unsupported error.
- [x] Define data-movement-unsupported error.
- [x] Define materialization-failed error.
- [x] Define provider-initialization-failed error.
- [x] Define unload-failed error.
- [x] Define reload-failed error.
- [x] Define browser-feature-unsupported error.
- [x] Define internal-loading-error.

## 27. Observability

- [x] Emit model loading requested observation.
- [x] Emit artifact preconditions checked observation.
- [x] Emit loading validation failed observation.
- [x] Emit residency planning started observation.
- [x] Emit residency planning completed observation.
- [x] Emit memory allocation requested observation.
- [x] Emit memory allocation queued observation.
- [x] Emit memory allocation failed observation.
- [x] Emit shard loading started observation.
- [x] Emit shard loading completed observation.
- [x] Emit materialization started observation.
- [x] Emit materialization completed observation.
- [x] Emit Provider state initialized observation.
- [x] Emit model ready observation.
- [x] Emit model load failed observation.
- [x] Emit model unloading started observation.
- [x] Emit model unloaded observation.
- [x] Emit model reload requested observation.
- [x] Emit model reload completed observation.
- [x] Emit model residency pressure observation.
- [x] Avoid raw model weights and raw memory handle logging.

## 28. Tests

- [x] Test loading valid model artifact.
- [x] Test loading invalid artifact fails before allocation.
- [x] Test untrusted artifact fails before allocation.
- [x] Test revoked artifact fails before allocation.
- [x] Test missing architecture implementation.
- [x] Test direct Provider selection rejected.
- [x] Test direct Device selection rejected.
- [x] Test unsupported storage dtype.
- [x] Test unsupported compute dtype.
- [x] Test unsupported quantization.
- [x] Test shard digest mismatch.
- [x] Test memory feasibility failure.
- [x] Test Provider capability unavailable.
- [x] Test Provider not ready.
- [x] Test Device unavailable.
- [x] Test materialization failure cleanup.
- [x] Test unload releases memory.
- [x] Test model unload invalidates KV cache.
- [x] Test session creation with implicit loading disabled.
- [x] Test browser unsupported native feature.
- [x] Test raw handles not exposed.

## 29. Documentation

- [x] Document Model Loading Contract.
- [x] Document loading preconditions.
- [x] Document loading lifecycle.
- [x] Document architecture implementation.
- [x] Document Provider compatibility.
- [x] Document Device placement.
- [x] Document Memory Manager relationship.
- [x] Document residency plan.
- [x] Document residency.
- [x] Document dtype handling.
- [x] Document quantization handling.
- [x] Document sharded loading.
- [x] Document lazy loading.
- [x] Document unload.
- [x] Document reload.
- [x] Document Session relationship.
- [x] Document KV cache relationship.
- [x] Document browser compatibility.
- [x] Document non-goals.

## 30. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run wasm32 check where feasible.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run Model Loading tests.
- [x] Run Model Artifact tests.
- [x] Run Memory Manager tests.
- [x] Run Session tests where impacted.
- [x] Run KV cache tests where impacted.
- [x] Run Provider conformance tests where impacted.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify loading is Runtime-owned.
- [x] Verify artifacts do not select Provider/Device.
- [x] Verify Memory Manager tracks residency.
- [x] Verify failed loads clean up resources.
- [x] Verify raw model memory handles are not exposed.
# Tasks

## 1. Model Instance Scope

- [x] Define Model Instance as Runtime-owned loaded inference context.
- [x] Document Model Instance versus Model Artifact.
- [x] Document Model Instance versus Model Loading.
- [x] Document Model Instance versus Model Residency.
- [x] Document Model Instance versus Session.
- [x] Document Model Instance versus Provider resource.
- [x] Document Model Instance versus KV Cache.

## 2. Model Instance Module

- [x] Create first-class `model_instance` module or equivalent.
- [x] Export canonical model instance types from crate root.
- [x] Keep lifecycle platform-neutral.
- [x] Keep Model Instance independent from direct Provider selection by clients.
- [x] Add module-level documentation.

## 3. Model Instance Identity

- [x] Define ModelInstanceId.
- [x] Ensure ID is Runtime-issued.
- [x] Ensure ID is opaque.
- [x] Ensure ID does not encode Provider handles.
- [x] Ensure ID does not encode Device handles.
- [x] Ensure ID does not encode memory pointers.
- [x] Ensure ID does not expose raw model weights.
- [x] Ensure ID alone does not grant authority.
- [x] Add identity tests.

## 4. Instance Definition

- [x] Bind Model Artifact identity.
- [x] Bind architecture implementation identity.
- [x] Bind Model Residency records.
- [x] Bind tokenizer compatibility metadata.
- [x] Bind Provider/Device placement metadata.
- [x] Bind Resource Affinity.
- [x] Bind Runtime policy.
- [x] Bind adapter state.
- [x] Bind associated sessions metadata.
- [x] Bind cache dependency metadata.
- [x] Add definition tests.

## 5. Lifecycle

- [x] Define creating state.
- [x] Define loading state.
- [x] Define warming state.
- [x] Define ready state.
- [x] Define active state.
- [x] Define idle state.
- [x] Define draining state.
- [x] Define suspended state.
- [x] Define reloading state.
- [x] Define unloading state.
- [x] Define unloaded state.
- [x] Define failed state.
- [x] Define invalid state.
- [x] Define removed state.
- [x] Define allowed transitions.
- [x] Add lifecycle transition tests.

## 6. Readiness

- [x] Define not-ready readiness.
- [x] Define ready readiness.
- [x] Define read-only readiness.
- [x] Define draining readiness.
- [x] Define suspended readiness.
- [x] Define failed readiness.
- [x] Keep lifecycle distinct from readiness.
- [x] Validate residency readiness.
- [x] Validate Provider readiness.
- [x] Validate Device readiness.
- [x] Validate adapter readiness.
- [x] Validate memory pressure readiness.
- [x] Add readiness tests.

## 7. Instance Creation

- [x] Require successful Model Loading or explicit loading path.
- [x] Validate Model Artifact identity.
- [x] Validate Model Artifact trust.
- [x] Validate architecture implementation.
- [x] Validate Model Residency Plan.
- [x] Validate Memory Manager admission.
- [x] Validate Provider/Device compatibility.
- [x] Validate tokenizer compatibility metadata.
- [x] Validate Runtime policy.
- [x] Validate browser/native constraints.
- [x] Prevent ready state until checks pass.
- [x] Add creation tests.

## 8. Warmup

- [x] Define warmup policy.
- [x] Support Provider initialization warmup.
- [x] Support kernel preparation placeholder.
- [x] Support operator graph preparation placeholder.
- [x] Support shape plan preparation placeholder.
- [x] Support tokenizer/model metadata validation.
- [x] Support small test execution placeholder.
- [x] Support memory residency verification.
- [x] Support adapter readiness verification.
- [x] Reject readiness on warmup failure.
- [x] Add warmup tests.

## 9. Usage References

- [x] Define usage acquisition.
- [x] Define usage release.
- [x] Track active operation count.
- [x] Track active session count.
- [x] Prevent unload while active unless forced policy applies.
- [x] Release usage on operation completion.
- [x] Release usage on cancellation.
- [x] Add usage reference tests.

## 10. Usage Accounting

- [x] Track active operation count.
- [x] Track active session count.
- [x] Track queued operation count.
- [x] Track total request count.
- [x] Track token counts.
- [x] Track last used timestamp.
- [x] Track residency size.
- [x] Track KV cache dependencies.
- [x] Track adapter dependencies.
- [x] Track failure count.
- [x] Avoid raw prompt/weight/handle exposure.
- [x] Add usage accounting tests.

## 11. Instance Sharing

- [x] Define instance sharing policy.
- [x] Allow Runtime-local sharing where safe.
- [x] Consider tenant/user isolation where available.
- [x] Consider adapter state.
- [x] Consider KV cache privacy.
- [x] Consider Prefix Cache privacy.
- [x] Consider Resource Affinity.
- [x] Reject unsafe sharing.
- [x] Add sharing tests.

## 12. Instance Mutability

- [x] Define semantic mutation.
- [x] Track adapter merge mutation.
- [x] Track Provider-specific preparation.
- [x] Track quantization transform.
- [x] Track residency relocation placeholder.
- [x] Track reload mutation.
- [x] Forbid silent semantic mutation.
- [x] Add mutability tests.

## 13. Adapter Integration

- [x] Track active adapter set.
- [x] Track adapter activation scope.
- [x] Track adapter merge state.
- [x] Validate adapter readiness.
- [x] Invalidate caches on adapter changes where required.
- [x] Include adapter state in batching compatibility.
- [x] Include adapter state in determinism metadata.
- [x] Add adapter integration tests.

## 14. KV Cache Integration

- [x] Bind KV cache compatibility to Model Instance identity where needed.
- [x] Invalidate KV caches on instance unload.
- [x] Invalidate KV caches on incompatible reload.
- [x] Invalidate KV caches on semantic mutation.
- [x] Prevent incompatible cross-instance reuse.
- [x] Add KV cache tests.

## 15. Prefix Cache Integration

- [x] Bind Prefix Cache entries to Model Instance identity where needed.
- [x] Invalidate prefix entries on instance unload.
- [x] Invalidate prefix entries on incompatible reload.
- [x] Invalidate prefix entries on semantic mutation.
- [x] Add Prefix Cache tests.

## 16. Generation Integration

- [x] Require ready Model Instance for generation.
- [x] Allow policy-controlled implicit load path.
- [x] Acquire Model Instance usage before prefill.
- [x] Release usage after generation completion.
- [x] Handle draining instance.
- [x] Handle failed instance.
- [x] Handle invalid instance.
- [x] Add generation integration tests.

## 17. Continuous Batching Integration

- [x] Include Model Instance in batch compatibility.
- [x] Reject incompatible instances in same execution step.
- [x] Consider readiness.
- [x] Consider Resource Affinity.
- [x] Consider active adapter state.
- [x] Consider Provider/Device pressure.
- [x] Add batching integration tests.

## 18. Provider Integration

- [x] Track Provider-owned model resources.
- [x] Keep Provider resources opaque.
- [x] React to Provider health.
- [x] React to Provider readiness.
- [x] React to Provider pressure.
- [x] React to Provider admission state.
- [x] Map Provider failures to instance state.
- [x] Add Provider integration tests.

## 19. Device Integration

- [x] Track Device-bound residency.
- [x] Preserve Device Resource Affinity.
- [x] React to Device loss.
- [x] React to Device reset.
- [x] React to Device pressure.
- [x] React to Device unavailable.
- [x] Add Device integration tests.

## 20. Memory Manager Integration

- [x] Track all instance residency through Memory Manager.
- [x] Update residency on lifecycle changes.
- [x] React to memory pressure.
- [x] Support unload resource release.
- [x] Support suspension placeholder.
- [x] Support relocation placeholder.
- [x] Account for browser memory constraints.
- [x] Add memory integration tests.

## 21. Suspension

- [x] Define suspension policy.
- [x] Suspend on memory pressure where policy allows.
- [x] Suspend on Provider pressure where policy allows.
- [x] Suspend on Device pressure where policy allows.
- [x] Suspend on administrative policy.
- [x] Suspend on browser lifecycle event where relevant.
- [x] Reject new operations while suspended.
- [x] Resume, reload, unload, or fail according to policy.
- [x] Add suspension tests.

## 22. Draining

- [x] Define draining policy.
- [x] Drain on unload request.
- [x] Drain on reload request.
- [x] Drain on policy change.
- [x] Drain on Provider drain.
- [x] Drain on Device pressure.
- [x] Drain on Runtime shutdown.
- [x] Reject new operations while draining.
- [x] Allow active operations to complete according to policy.
- [x] Add draining tests.

## 23. Unload

- [x] Define unload request.
- [x] Stop new operation admission.
- [x] Drain or cancel active operations.
- [x] Invalidate dependent KV caches.
- [x] Invalidate dependent Prefix Cache entries.
- [x] Release adapter associations.
- [x] Release Memory Manager residency.
- [x] Release Provider-owned resources.
- [x] Update lifecycle.
- [x] Avoid dangling session references.
- [x] Add unload tests.

## 24. Reload

- [x] Define reload request.
- [x] Treat reload as validated loading process.
- [x] Support replacement instance.
- [x] Support updated residency.
- [x] Support Provider/Device placement change.
- [x] Support compute dtype change.
- [x] Support quantization handling change.
- [x] Support adapter compatibility change.
- [x] Define session migration policy.
- [x] Prevent silent active semantic mutation.
- [x] Add reload tests.

## 25. Failure Handling

- [x] Define loading failure handling.
- [x] Define warmup failure handling.
- [x] Define Provider initialization failure handling.
- [x] Define Device residency failure handling.
- [x] Define Memory Manager failure handling.
- [x] Define adapter activation failure handling.
- [x] Define generation execution failure handling.
- [x] Define unload failure handling.
- [x] Define reload failure handling.
- [x] Prevent failed/invalid instance from accepting new operations.
- [x] Add failure tests.

## 26. Browser Compatibility

- [x] Keep lifecycle platform-neutral.
- [x] Account for browser memory limits.
- [x] Account for WebAssembly linear memory.
- [x] Account for future WebGPU buffers.
- [x] Avoid Wasmtime dependency.
- [x] Avoid native Provider loading requirement.
- [x] Return unsupported browser errors where needed.
- [x] Add wasm32 check where feasible.

## 27. Error Model

- [x] Define model-instance-not-found error.
- [x] Define model-instance-not-ready error.
- [x] Define model-instance-loading error.
- [x] Define model-instance-warming error.
- [x] Define model-instance-draining error.
- [x] Define model-instance-suspended error.
- [x] Define model-instance-unloading error.
- [x] Define model-instance-unloaded error.
- [x] Define model-instance-failed error.
- [x] Define model-instance-invalid error.
- [x] Define model-instance-removed error.
- [x] Define model-instance-active error.
- [x] Define model-instance-busy error.
- [x] Define model-instance-sharing-denied error.
- [x] Define model-instance-policy-denied error.
- [x] Define model-instance-reload-required error.
- [x] Define model-instance-reload-failed error.
- [x] Define model-instance-unload-failed error.
- [x] Define model-instance-warmup-failed error.
- [x] Define model-instance-Provider-unavailable error.
- [x] Define model-instance-Provider-not-ready error.
- [x] Define model-instance-Provider-failed error.
- [x] Define model-instance-Device-unavailable error.
- [x] Define model-instance-Device-lost error.
- [x] Define model-instance-memory-pressure error.
- [x] Define model-instance-residency-missing error.
- [x] Define model-instance-adapter-incompatible error.
- [x] Define model-instance-KV-cache-invalidated status.
- [x] Define model-instance-Prefix-Cache-invalidated status.
- [x] Define model-instance-browser-feature-unsupported error.
- [x] Define internal-model-instance error.

## 28. Observability

- [x] Emit model instance creation requested observation.
- [x] Emit model instance created observation.
- [x] Emit model instance loading observation.
- [x] Emit model instance warming observation.
- [x] Emit model instance ready observation.
- [x] Emit model instance active observation.
- [x] Emit model instance idle observation.
- [x] Emit model instance draining observation.
- [x] Emit model instance suspended observation.
- [x] Emit model instance reloading observation.
- [x] Emit model instance unloading observation.
- [x] Emit model instance unloaded observation.
- [x] Emit model instance failed observation.
- [x] Emit model instance invalidated observation.
- [x] Emit model instance removed observation.
- [x] Emit model instance usage acquired observation.
- [x] Emit model instance usage released observation.
- [x] Emit model instance sharing denied observation.
- [x] Emit model instance cache invalidation observation.
- [x] Emit model instance memory pressure observation.
- [x] Emit model instance Provider pressure observation.
- [x] Emit model instance Device unavailable observation.
- [x] Avoid raw prompt/weight/cache/handle logging by default.

## 29. Tests

- [x] Test Model Instance creation from loaded model.
- [x] Test opaque ModelInstanceId.
- [x] Test lifecycle transitions.
- [x] Test readiness distinct from lifecycle.
- [x] Test warmup success.
- [x] Test warmup failure prevents ready.
- [x] Test usage acquire/release.
- [x] Test unload blocked while active unless forced policy.
- [x] Test unload invalidates KV cache.
- [x] Test unload invalidates Prefix Cache.
- [x] Test reload creates validated replacement.
- [x] Test adapter activation affects instance compatibility.
- [x] Test generation rejects non-ready instance.
- [x] Test batching rejects incompatible instances.
- [x] Test Provider failure marks instance failed or invalid.
- [x] Test Device loss marks instance suspended/invalid according to policy.
- [x] Test memory pressure suspension.
- [x] Test raw handles not exposed.
- [x] Test raw model weights not exposed.
- [x] Test browser unsupported feature.

## 30. Documentation

- [x] Document Model Instance lifecycle.
- [x] Document Model Instance versus Model Artifact.
- [x] Document Model Instance versus Residency.
- [x] Document lifecycle states.
- [x] Document readiness.
- [x] Document creation.
- [x] Document warmup.
- [x] Document usage references.
- [x] Document sharing policy.
- [x] Document mutability.
- [x] Document Adapter relationship.
- [x] Document KV Cache relationship.
- [x] Document Prefix Cache relationship.
- [x] Document Generation relationship.
- [x] Document Batching relationship.
- [x] Document Provider relationship.
- [x] Document Memory Manager relationship.
- [x] Document suspension.
- [x] Document unload.
- [x] Document reload.
- [x] Document browser compatibility.
- [x] Document non-goals.

## 31. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run wasm32 check where feasible.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run Model Instance tests.
- [x] Run Model Loading tests.
- [x] Run Memory Manager tests.
- [x] Run Adapter tests where impacted.
- [x] Run Session tests where impacted.
- [x] Run Generation tests where impacted.
- [x] Run KV Cache tests where impacted.
- [x] Run Prefix Cache tests where impacted.
- [x] Run Batching tests where impacted.
- [x] Run Provider conformance tests where impacted.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify Model Instance is Runtime-owned.
- [x] Verify Model Instance does not expose raw weights.
- [x] Verify Model Instance does not expose raw Provider/Device handles.
- [x] Verify Generation requires ready Model Instance.
- [x] Verify unload/reload invalidates dependent state.

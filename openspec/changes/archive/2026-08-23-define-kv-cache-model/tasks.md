# Tasks

## 1. KV Cache Scope

- [x] Define KV cache as Runtime-owned inference state.
- [x] Document KV cache versus Inference Session.
- [x] Document KV cache versus Model Artifact.
- [x] Document KV cache versus Provider handle.
- [x] Document KV cache versus Scheduler state.
- [x] Document KV cache versus prefix cache.
- [x] Document KV cache privacy sensitivity.

## 2. KV Cache Module

- [x] Create first-class `kv_cache` module or equivalent.
- [x] Export canonical KV cache types from crate root.
- [x] Keep KV cache platform-neutral.
- [x] Keep KV cache independent from direct Provider selection.
- [x] Add module-level documentation.

## 3. KV Cache Identity

- [x] Define KvCacheId.
- [x] Ensure ID is Runtime-issued.
- [x] Ensure ID is opaque.
- [x] Ensure ID does not encode raw pointer.
- [x] Ensure ID does not encode raw Provider handle.
- [x] Ensure ID does not encode raw Device handle.
- [x] Ensure ID alone does not grant authority.
- [x] Add identity tests.

## 4. KV Cache Scope Kinds

- [x] Define operation scope.
- [x] Define session scope.
- [x] Define model-instance scope.
- [x] Define prefix-cache scope.
- [x] Define batch-slot scope.
- [x] Define runtime-cache scope.
- [x] Add scope tests.

## 5. KV Cache Lifecycle

- [x] Define allocating state.
- [x] Define empty state.
- [x] Define prefilling state.
- [x] Define ready state.
- [x] Define active state.
- [x] Define sealed state.
- [x] Define evicting state.
- [x] Define evicted state.
- [x] Define invalid state.
- [x] Define released state.
- [x] Define failed state.
- [x] Define allowed transitions.
- [x] Add lifecycle transition tests.

## 6. Compatibility Metadata

- [x] Track model identity.
- [x] Track model architecture.
- [x] Track model revision.
- [x] Track tokenizer identity.
- [x] Track tokenizer vocabulary compatibility.
- [x] Track prompt token prefix or fingerprint.
- [x] Track position encoding metadata.
- [x] Track attention implementation metadata.
- [x] Track layer count.
- [x] Track head configuration.
- [x] Track head dimension.
- [x] Track dtype.
- [x] Track quantization mode.
- [x] Track Provider/Device residency.
- [x] Add compatibility tests.

## 7. Prefix Fingerprint

- [x] Define prefix fingerprint.
- [x] Derive fingerprint from validated token IDs.
- [x] Include relevant model configuration.
- [x] Avoid raw prompt text in fingerprint.
- [x] Ensure fingerprint is not authority by itself.
- [x] Add fingerprint tests.

## 8. Layout Metadata

- [x] Define layer count.
- [x] Define head count.
- [x] Define key head count.
- [x] Define value head count.
- [x] Define head dimension.
- [x] Define token capacity.
- [x] Define current token length.
- [x] Define batch dimension.
- [x] Define sequence dimension.
- [x] Define block size placeholder.
- [x] Define page size placeholder.
- [x] Define dtype.
- [x] Define layout format.
- [x] Define contiguous layout metadata.
- [x] Define paged layout metadata.
- [x] Define position range.
- [x] Add layout metadata tests.

## 9. Paged Cache Readiness

- [x] Allow paged or block-based layout metadata.
- [x] Define page identity placeholder.
- [x] Define page occupancy placeholder.
- [x] Define free page reuse placeholder.
- [x] Define prefix sharing placeholder.
- [x] Do not require paged implementation in this change.
- [x] Add tests proving metadata can represent paged layout.

## 10. Quantized KV Cache

- [x] Define KV cache storage dtype.
- [x] Define KV cache compute dtype.
- [x] Define cache quantization metadata.
- [x] Validate Provider compatibility.
- [x] Validate Memory Manager feasibility.
- [x] Add quantized cache metadata tests.

## 11. Memory Manager Integration

- [x] Use KV cache allocation class.
- [x] Request allocation through Memory Manager.
- [x] Track cache residency.
- [x] Track cache memory usage.
- [x] Track cache memory pressure.
- [x] Track pending cache allocation.
- [x] Release cache memory on eviction/release.
- [x] Add memory integration tests.

## 12. Resource Affinity

- [x] Derive Resource Affinity from KV cache residency.
- [x] Track Provider binding where applicable.
- [x] Track Device binding where applicable.
- [x] Reject reuse on incompatible Provider.
- [x] Reject reuse on incompatible Device.
- [x] Require explicit movement, rebuild, or rejection.
- [x] Prevent client-forged cache affinity.
- [x] Add affinity tests.

## 13. Provider Integration

- [x] Define Provider cache creation boundary.
- [x] Define Provider cache update boundary.
- [x] Define Provider cache read boundary.
- [x] Define Provider cache destroy boundary.
- [x] Keep raw Provider cache handles internal.
- [x] Map Provider cache errors to Runtime errors.
- [x] Add Provider integration tests.

## 14. Device Integration

- [x] Track Device-bound cache.
- [x] Reject use on incompatible Device.
- [x] Handle Device unavailable.
- [x] Handle Device memory pressure.
- [x] Handle Device reset/interruption.
- [x] Add Device integration tests.

## 15. Session Integration

- [x] Allow session to own KV cache reference.
- [x] Define session KV cache policy.
- [x] Define max cache tokens.
- [x] Define max cache memory.
- [x] Define cache reuse policy.
- [x] Define cache persistence after session close.
- [x] Define cache privacy policy.
- [x] Release or retain cache on session close according to policy.
- [x] Add session integration tests.

## 16. Generation Integration

- [x] Create or populate cache during prefill.
- [x] Read cache during decode.
- [x] Append cache during decode.
- [x] Validate cache compatibility before reuse.
- [x] Handle cache invalid during generation.
- [x] Add prefill cache tests.
- [x] Add decode cache tests.

## 17. Prefix Cache Preparation

- [x] Define relation to future prefix cache.
- [x] Allow sealed cache for prefix reuse.
- [x] Allow prefix fingerprint metadata.
- [x] Allow cache sharing metadata.
- [x] Defer prefix cache index.
- [x] Add prefix placeholder tests.

## 18. Batching Preparation

- [x] Define batch-slot cache scope.
- [x] Define batch dimension metadata.
- [x] Define token capacity metadata.
- [x] Define layout metadata for future continuous batching.
- [x] Ensure Scheduler does not own cache memory directly.
- [x] Add batching placeholder tests.

## 19. Cache Sharing

- [x] Define cache sharing policy.
- [x] Require explicit sharing permission.
- [x] Consider tenant/session isolation.
- [x] Consider prompt privacy.
- [x] Consider sealed state.
- [x] Reject unsafe mutable sharing.
- [x] Add sharing tests.

## 20. Cache Sealing

- [x] Define sealed state.
- [x] Prevent mutation of sealed cache.
- [x] Allow read-only reuse where policy permits.
- [x] Define fork/copy/rebuild behavior placeholder.
- [x] Add sealed cache tests.

## 21. Eviction

- [x] Define eviction triggers.
- [x] Evict on memory pressure where policy permits.
- [x] Evict on session close where policy requires.
- [x] Evict on idle TTL.
- [x] Evict on total TTL.
- [x] Evict on model unload.
- [x] Evict on Device unavailable.
- [x] Evict on Provider drain where policy requires.
- [x] Release Memory Manager resources.
- [x] Add eviction tests.

## 22. Invalidation

- [x] Invalidate on model mismatch.
- [x] Invalidate on tokenizer mismatch.
- [x] Invalidate on prompt mismatch.
- [x] Invalidate on position mismatch.
- [x] Invalidate on dtype/layout mismatch.
- [x] Invalidate on Provider/Device loss.
- [x] Invalidate on memory corruption detection where available.
- [x] Invalidate on session policy change.
- [x] Add invalidation tests.

## 23. Cancellation Handling

- [x] Define cancellation impact policy.
- [x] Release partial cache by default where conservative.
- [x] Allow retain valid prefix where policy permits.
- [x] Allow seal valid prefix where policy permits.
- [x] Allow quarantine for diagnostics where policy permits.
- [x] Add cancellation cache tests.

## 24. Privacy And Security

- [x] Treat KV cache as sensitive inference state.
- [x] Prevent raw cache content exposure.
- [x] Prevent Component access to KV cache contents.
- [x] Prevent client access without explicit policy.
- [x] Redact cache observability.
- [x] Avoid raw prompt text in cache metadata by default.
- [x] Add privacy tests.

## 25. Browser Compatibility

- [x] Keep KV cache model platform-neutral.
- [x] Add browser-compatible cache capability metadata.
- [x] Account for browser memory limits.
- [x] Account for WebAssembly linear memory.
- [x] Account for future WebGPU buffers.
- [x] Return unsupported errors for unavailable features.
- [x] Add wasm32 compile check where feasible.

## 26. Error Model

- [x] Define cache-allocation-failed error.
- [x] Define cache-admission-denied error.
- [x] Define cache-not-found error.
- [x] Define cache-incompatible error.
- [x] Define cache-invalid error.
- [x] Define cache-evicted error.
- [x] Define cache-released error.
- [x] Define cache-capacity-exceeded error.
- [x] Define cache-position-mismatch error.
- [x] Define cache-prompt-mismatch error.
- [x] Define cache-model-mismatch error.
- [x] Define cache-tokenizer-mismatch error.
- [x] Define cache-dtype-mismatch error.
- [x] Define cache-layout-mismatch error.
- [x] Define cache-provider-mismatch error.
- [x] Define cache-device-mismatch error.
- [x] Define cache-movement-required error.
- [x] Define cache-movement-unsupported error.
- [x] Define cache-sharing-denied error.
- [x] Define cache-sealed error.
- [x] Define cache-mutation-denied error.
- [x] Define cache-memory-pressure error.
- [x] Define cache-provider-failure error.
- [x] Define cache-device-unavailable error.
- [x] Define cache-cancelled error.
- [x] Define cache-internal error.

## 27. Observability

- [x] Emit cache allocation requested observation.
- [x] Emit cache allocation completed observation.
- [x] Emit cache allocation failed observation.
- [x] Emit cache prefill started observation.
- [x] Emit cache prefill completed observation.
- [x] Emit cache decode append observation.
- [x] Emit cache hit observation.
- [x] Emit cache miss observation.
- [x] Emit cache compatibility failed observation.
- [x] Emit cache sealed observation.
- [x] Emit cache evicting observation.
- [x] Emit cache evicted observation.
- [x] Emit cache invalidated observation.
- [x] Emit cache released observation.
- [x] Emit cache memory pressure observation.
- [x] Emit cache movement required observation.
- [x] Emit cache sharing denied observation.
- [x] Avoid raw prompt and raw cache logging by default.

## 28. Tests

- [x] Test cache identity is opaque.
- [x] Test cache lifecycle transitions.
- [x] Test session-owned cache.
- [x] Test operation-owned cache.
- [x] Test prefill creates cache.
- [x] Test decode appends cache.
- [x] Test incompatible model rejects reuse.
- [x] Test incompatible tokenizer rejects reuse.
- [x] Test prompt mismatch rejects reuse.
- [x] Test dtype mismatch rejects reuse.
- [x] Test Provider mismatch rejects reuse.
- [x] Test Device mismatch rejects reuse.
- [x] Test sealed cache rejects mutation.
- [x] Test cache eviction releases memory.
- [x] Test cache invalidation prevents reuse.
- [x] Test cancellation policy releases or preserves cache.
- [x] Test sharing denied by default.
- [x] Test raw cache content not exposed.
- [x] Test raw prompt not in metadata by default.
- [x] Test browser unsupported features return structured errors.

## 29. Documentation

- [x] Document KV Cache model.
- [x] Document lifecycle.
- [x] Document scope.
- [x] Document compatibility.
- [x] Document prefix fingerprint.
- [x] Document memory relationship.
- [x] Document Resource Affinity.
- [x] Document Provider relationship.
- [x] Document Device relationship.
- [x] Document Session relationship.
- [x] Document Generation relationship.
- [x] Document sealing.
- [x] Document eviction.
- [x] Document invalidation.
- [x] Document cancellation policy.
- [x] Document privacy constraints.
- [x] Document browser compatibility.
- [x] Document non-goals.

## 30. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run wasm32 check where feasible.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run KV cache tests.
- [x] Run Session tests.
- [x] Run Generation tests.
- [x] Run Memory Manager tests.
- [x] Run Provider conformance tests where impacted.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify KV cache is Runtime-owned.
- [x] Verify KV cache is not exposed to Components.
- [x] Verify Memory Manager owns cache memory.
- [x] Verify Resource Affinity is preserved.
- [x] Verify raw prompts and raw cache contents are not exposed by default.

# Tasks

## 1. Prefix Cache Scope

- [x] Define Prefix Cache as Runtime-owned prefix reuse index.
- [x] Document Prefix Cache versus KV Cache.
- [x] Document Prefix Cache versus Inference Session.
- [x] Document Prefix Cache versus client conversation.
- [x] Document Prefix Cache versus Scheduler.
- [x] Document Prefix Cache privacy constraints.

## 2. Prefix Cache Module

- [x] Create first-class `prefix_cache` module or equivalent.
- [x] Export canonical prefix cache types from crate root.
- [x] Keep prefix cache platform-neutral.
- [x] Keep prefix cache independent from direct Provider selection.
- [x] Add module-level documentation.

## 3. Prefix Cache Identity

- [x] Define PrefixCacheEntryId.
- [x] Ensure entry ID is Runtime-issued.
- [x] Ensure entry ID is opaque.
- [x] Ensure entry ID does not expose raw prompt text.
- [x] Ensure entry ID does not expose raw token sequence.
- [x] Ensure entry ID does not expose Provider handle.
- [x] Ensure entry ID does not expose Device handle.
- [x] Ensure entry ID alone does not grant authority.

## 4. Prefix Fingerprint

- [x] Define prefix fingerprint.
- [x] Derive fingerprint from validated token IDs.
- [x] Bind fingerprint to model identity.
- [x] Bind fingerprint to model revision.
- [x] Bind fingerprint to tokenizer identity.
- [x] Bind fingerprint to tokenizer revision.
- [x] Bind fingerprint to template identity where relevant.
- [x] Bind fingerprint to position encoding metadata.
- [x] Avoid raw prompt text.
- [x] Avoid reversible prompt representation where practical.
- [x] Add fingerprint tests.

## 5. Prefix Matching

- [x] Define exact prefix match.
- [x] Define partial prefix match.
- [x] Define miss.
- [x] Define incompatible hit.
- [x] Define policy denied hit.
- [x] Define stale hit.
- [x] Define evicted hit.
- [x] Validate fingerprint.
- [x] Validate compatibility metadata.
- [x] Add lookup tests.

## 6. Partial Prefix Reuse

- [x] Define partial prefix reuse policy.
- [x] Validate partial prefix boundary.
- [x] Validate position compatibility.
- [x] Validate attention compatibility.
- [x] Continue prefill from reuse boundary.
- [x] Add partial reuse tests.

## 7. Entry Lifecycle

- [x] Define creating state.
- [x] Define ready state.
- [x] Define sealed state.
- [x] Define active state.
- [x] Define stale state.
- [x] Define invalid state.
- [x] Define evicting state.
- [x] Define evicted state.
- [x] Define released state.
- [x] Define failed state.
- [x] Define allowed transitions.
- [x] Add lifecycle tests.

## 8. Sealed KV Cache Requirement

- [x] Require sealed backing KV cache for shared reuse by default.
- [x] Reject mutable cache sharing by default.
- [x] Allow explicit policy exceptions.
- [x] Validate backing KV cache lifecycle.
- [x] Add sealed cache tests.

## 9. Prefix Cache Scope Kinds

- [x] Define operation scope.
- [x] Define session scope.
- [x] Define model-instance scope.
- [x] Define runtime scope.
- [x] Define tenant scope.
- [x] Define private scope.
- [x] Define shared scope.
- [x] Add scope policy tests.

## 10. Sharing Policy

- [x] Define session-local sharing.
- [x] Define cross-session sharing.
- [x] Define tenant sharing.
- [x] Define runtime sharing.
- [x] Define private-only policy.
- [x] Deny cross-session sharing by default.
- [x] Validate authorization before sharing.
- [x] Add sharing tests.

## 11. Privacy Policy

- [x] Prevent raw prompt text storage by default.
- [x] Prevent raw prompt logging by default.
- [x] Protect raw token sequences.
- [x] Redact cache metadata.
- [x] Deny export by default.
- [x] Define privacy-denied error.
- [x] Add privacy tests.

## 12. Entry Metadata

- [x] Define entry ID.
- [x] Define lifecycle state.
- [x] Define prefix fingerprint.
- [x] Define prefix token length.
- [x] Define model identity.
- [x] Define tokenizer identity.
- [x] Define model instance identity.
- [x] Define template identity where relevant.
- [x] Define backing KV cache reference.
- [x] Define Resource Affinity.
- [x] Define Provider binding metadata.
- [x] Define Device binding metadata.
- [x] Define dtype metadata.
- [x] Define position range.
- [x] Define timestamps.
- [x] Define hit count.
- [x] Define memory estimate.
- [x] Define scope.
- [x] Define sharing policy.
- [x] Define privacy policy.
- [x] Define eviction priority.

## 13. Resource Affinity

- [x] Inherit Resource Affinity from backing KV cache.
- [x] Reject reuse on incompatible Provider.
- [x] Reject reuse on incompatible Device.
- [x] Require explicit movement, rebuild, or miss fallback.
- [x] Prevent client-forged Resource Affinity.
- [x] Add Resource Affinity tests.

## 14. Memory Manager Integration

- [x] Account for backing KV cache memory.
- [x] Account for prefix metadata memory.
- [x] Account for prefix index memory.
- [x] Account for fingerprint storage.
- [x] Account for lookup workspace.
- [x] Account for eviction pressure.
- [x] Add memory integration tests.

## 15. Generation Integration

- [x] Query Prefix Cache before prefill.
- [x] Use hit to reuse sealed KV cache.
- [x] Use miss to run full prefill.
- [x] Use partial hit to continue prefill from boundary.
- [x] Validate lookup result before reuse.
- [x] Add generation integration tests.

## 16. Session Integration

- [x] Allow session-local prefix cache policy.
- [x] Allow session to disable prefix cache.
- [x] Define maximum prefix cache memory.
- [x] Define maximum prefix token length.
- [x] Define sharing scope.
- [x] Define TTL.
- [x] Define persistence after session close.
- [x] Add session integration tests.

## 17. Model Loading Integration

- [x] Bind prefix entries to loaded model context where needed.
- [x] Invalidate entries on model unload where required.
- [x] Invalidate entries on incompatible model reload.
- [x] Add model loading integration tests.

## 18. Tokenizer Integration

- [x] Bind prefix entries to tokenizer identity.
- [x] Bind prefix entries to tokenizer revision.
- [x] Invalidate entries on tokenizer mismatch.
- [x] Invalidate entries on template mismatch.
- [x] Add tokenizer integration tests.

## 19. KV Cache Integration

- [x] Reference sealed KV cache from prefix entry.
- [x] Detect backing KV cache evicted.
- [x] Detect backing KV cache invalid.
- [x] Detect backing KV cache released.
- [x] Mark prefix entry stale or invalid accordingly.
- [x] Add KV cache integration tests.

## 20. Batching Preparation

- [x] Preserve metadata useful for continuous batching.
- [x] Include prefix token length.
- [x] Include position range.
- [x] Include Resource Affinity.
- [x] Include layout metadata through KV cache reference.
- [x] Do not define batching policy in this change.

## 21. Eviction

- [x] Define eviction by memory pressure.
- [x] Define eviction by TTL.
- [x] Define eviction by idle TTL.
- [x] Define eviction by model unload.
- [x] Define eviction by tokenizer update.
- [x] Define eviction by template update.
- [x] Define eviction by Provider drain.
- [x] Define eviction by Device unavailable.
- [x] Define eviction by backing KV cache eviction.
- [x] Define eviction by policy change.
- [x] Release or dereference backing resources.
- [x] Add eviction tests.

## 22. Invalidation

- [x] Invalidate on model mismatch.
- [x] Invalidate on tokenizer mismatch.
- [x] Invalidate on template mismatch.
- [x] Invalidate on prompt prefix mismatch.
- [x] Invalidate on position mismatch.
- [x] Invalidate on backing KV cache invalidation.
- [x] Invalidate on Resource Affinity conflict.
- [x] Invalidate on session policy change.
- [x] Invalidate on privacy policy change.
- [x] Add invalidation tests.

## 23. Browser Compatibility

- [x] Keep Prefix Cache model platform-neutral.
- [x] Account for browser memory limits.
- [x] Account for WebAssembly linear memory.
- [x] Account for future WebGPU buffers.
- [x] Avoid Wasmtime dependency.
- [x] Avoid native Provider loading requirement.
- [x] Return unsupported browser errors where needed.
- [x] Add wasm32 check where feasible.

## 24. Error Model

- [x] Define prefix-cache-disabled error.
- [x] Define prefix-cache-unavailable error.
- [x] Define prefix-entry-not-found error.
- [x] Define prefix-entry-incompatible error.
- [x] Define prefix-fingerprint-mismatch error.
- [x] Define prefix-model-mismatch error.
- [x] Define prefix-tokenizer-mismatch error.
- [x] Define prefix-template-mismatch error.
- [x] Define prefix-position-mismatch error.
- [x] Define prefix-policy-denied error.
- [x] Define prefix-sharing-denied error.
- [x] Define prefix-privacy-denied error.
- [x] Define prefix-stale error.
- [x] Define prefix-invalid error.
- [x] Define prefix-evicted error.
- [x] Define prefix-backing-cache-missing error.
- [x] Define prefix-backing-cache-invalid error.
- [x] Define prefix-resource-affinity-conflict error.
- [x] Define prefix-movement-required error.
- [x] Define prefix-movement-unsupported error.
- [x] Define prefix-memory-pressure error.
- [x] Define prefix-allocation-failed error.
- [x] Define prefix-browser-feature-unsupported error.
- [x] Define prefix-internal error.

## 25. Observability

- [x] Emit prefix cache lookup observation.
- [x] Emit prefix cache hit observation.
- [x] Emit prefix cache miss observation.
- [x] Emit partial prefix hit observation.
- [x] Emit policy denied hit observation.
- [x] Emit incompatible hit observation.
- [x] Emit entry created observation.
- [x] Emit entry sealed observation.
- [x] Emit entry reused observation.
- [x] Emit entry invalidated observation.
- [x] Emit entry evicted observation.
- [x] Emit backing KV cache missing observation.
- [x] Emit sharing denied observation.
- [x] Emit privacy redaction observation.
- [x] Emit memory pressure eviction observation.
- [x] Avoid raw prompt logging by default.
- [x] Avoid raw token sequence logging by default.
- [x] Avoid raw KV cache logging by default.

## 26. Tests

- [x] Test exact prefix hit.
- [x] Test prefix miss.
- [x] Test partial prefix hit.
- [x] Test incompatible model hit.
- [x] Test incompatible tokenizer hit.
- [x] Test template mismatch.
- [x] Test position mismatch.
- [x] Test backing KV cache evicted.
- [x] Test backing KV cache invalid.
- [x] Test policy denied hit.
- [x] Test sharing denied by default.
- [x] Test privacy denied.
- [x] Test session-local reuse.
- [x] Test cross-session reuse disabled by default.
- [x] Test Resource Affinity conflict.
- [x] Test memory pressure eviction.
- [x] Test model unload invalidation.
- [x] Test tokenizer update invalidation.
- [x] Test raw prompt not stored by default.
- [x] Test raw token sequence not exposed by default.
- [x] Test raw KV cache not exposed.

## 27. Documentation

- [x] Document Prefix Cache model.
- [x] Document Prefix Cache versus KV Cache.
- [x] Document prefix fingerprint.
- [x] Document lookup and match kinds.
- [x] Document partial prefix reuse.
- [x] Document lifecycle.
- [x] Document scope.
- [x] Document sharing policy.
- [x] Document privacy policy.
- [x] Document Resource Affinity.
- [x] Document Memory Manager relationship.
- [x] Document Generation relationship.
- [x] Document Session relationship.
- [x] Document KV Cache relationship.
- [x] Document eviction.
- [x] Document invalidation.
- [x] Document browser compatibility.
- [x] Document non-goals.

## 28. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run wasm32 check where feasible.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run Prefix Cache tests.
- [x] Run KV Cache tests.
- [x] Run Generation tests.
- [x] Run Session tests.
- [x] Run Memory Manager tests where impacted.
- [x] Run Provider conformance tests where impacted.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify Prefix Cache is Runtime-owned.
- [x] Verify Prefix Cache does not expose prompt text.
- [x] Verify Prefix Cache does not expose raw token sequences by default.
- [x] Verify Prefix Cache does not expose raw KV cache.
- [x] Verify sharing is policy-controlled.
# Tasks

## 1. Batching Scope

- [x] Define Continuous Batching as Runtime/Scheduler orchestration.
- [x] Document batching versus Generation.
- [x] Document batching versus KV Cache.
- [x] Document batching versus Prefix Cache.
- [x] Document batching versus Memory Manager.
- [x] Document batching versus Provider execution.
- [x] Document batching versus Session state.

## 2. Batching Module

- [x] Create first-class `batching` module or equivalent.
- [x] Export canonical batching types from crate root.
- [x] Keep batching platform-neutral.
- [x] Keep batching independent from direct Provider selection.
- [x] Add module-level documentation.

## 3. Continuous Batch Identity

- [x] Define BatchId.
- [x] Ensure BatchId is Runtime-issued.
- [x] Ensure BatchId is opaque.
- [x] Ensure BatchId does not encode Provider handles.
- [x] Ensure BatchId does not encode Device handles.
- [x] Ensure BatchId does not encode memory pointers.
- [x] Ensure BatchId does not grant authority by itself.

## 4. Operation Lifecycle

- [x] Define admitted state.
- [x] Define queued state.
- [x] Define prefill-pending state.
- [x] Define prefilling state.
- [x] Define decode-pending state.
- [x] Define decoding state.
- [x] Define streaming state.
- [x] Define completed state.
- [x] Define cancelled state.
- [x] Define failed state.
- [x] Define rejected state.
- [x] Define evicted state.
- [x] Define allowed transitions.
- [x] Add operation lifecycle tests.

## 5. Prefill Scheduling

- [x] Define prefill queue.
- [x] Define prefill admission.
- [x] Define prefill batch compatibility.
- [x] Define prefill resource requirements.
- [x] Define prefill memory admission.
- [x] Define prefill cancellation.
- [x] Define prefill failure behavior.
- [x] Add prefill scheduling tests.

## 6. Decode Scheduling

- [x] Define decode queue.
- [x] Define decode step scheduling.
- [x] Define decode batch compatibility.
- [x] Define decode resource requirements.
- [x] Define decode memory admission.
- [x] Define decode cancellation.
- [x] Define decode stop condition handling.
- [x] Add decode scheduling tests.

## 7. Batch Slots

- [x] Define BatchSlotId.
- [x] Define operation binding.
- [x] Define session binding.
- [x] Define model context binding.
- [x] Define tokenizer binding.
- [x] Define current sequence length.
- [x] Define generated token count.
- [x] Define KV cache reference.
- [x] Define Prefix Cache reuse boundary.
- [x] Define Provider/Device placement metadata.
- [x] Define memory reservation reference.
- [x] Define priority.
- [x] Define deadline.
- [x] Define cancellation state.
- [x] Add slot tests.

## 8. Slot Assignment

- [x] Assign slots by Runtime policy.
- [x] Consider model context compatibility.
- [x] Consider tokenizer compatibility.
- [x] Consider Provider compatibility.
- [x] Consider Device compatibility.
- [x] Consider Resource Affinity.
- [x] Consider KV cache residency.
- [x] Consider Prefix Cache reuse.
- [x] Consider sequence length.
- [x] Consider memory budget.
- [x] Consider Provider pressure.
- [x] Consider Device pressure.
- [x] Consider priority.
- [x] Consider fairness.
- [x] Consider latency targets.
- [x] Add slot assignment tests.

## 9. Batch Compatibility

- [x] Validate loaded model context compatibility.
- [x] Validate architecture implementation compatibility.
- [x] Validate compute dtype compatibility.
- [x] Validate tokenizer compatibility.
- [x] Validate Provider/Device compatibility.
- [x] Validate Resource Affinity.
- [x] Validate KV cache layout compatibility.
- [x] Validate sequence length constraints.
- [x] Validate sampling compatibility where Provider-assisted.
- [x] Validate memory placement.
- [x] Add compatibility tests.

## 10. Admission

- [x] Validate model context availability.
- [x] Validate tokenizer compatibility.
- [x] Validate generation parameters.
- [x] Validate sampling parameters.
- [x] Validate session policy.
- [x] Validate memory budget.
- [x] Validate Provider readiness.
- [x] Validate Device readiness.
- [x] Validate queue capacity.
- [x] Validate cancellation state.
- [x] Reject invalid operations before batch entry.
- [x] Add admission tests.

## 11. Memory Manager Integration

- [x] Request memory admission for batch input buffers.
- [x] Request memory admission for batch output buffers.
- [x] Request memory admission for logits buffers.
- [x] Request memory admission for attention masks.
- [x] Request memory admission for position buffers.
- [x] Request memory admission for sampling workspace.
- [x] Request memory admission for KV cache blocks.
- [x] Request memory admission for Prefix Cache lookup workspace.
- [x] Request memory admission for temporary staging.
- [x] Request memory admission for Provider-specific workspace.
- [x] Ensure Scheduler does not allocate directly.
- [x] Add memory integration tests.

## 12. KV Cache Integration

- [x] Assign KV cache to batch slots.
- [x] Grow KV cache through Runtime-managed APIs.
- [x] Seal KV cache where policy requires.
- [x] Release KV cache on operation completion according to policy.
- [x] Respect KV cache Resource Affinity.
- [x] Reject incompatible cache reuse.
- [x] Add KV cache batching tests.

## 13. Prefix Cache Integration

- [x] Query Prefix Cache before prefill where enabled.
- [x] Apply exact prefix hit.
- [x] Apply partial prefix hit.
- [x] Fall back on miss.
- [x] Reject policy-denied hit.
- [x] Preserve prefix privacy.
- [x] Add prefix batching tests.

## 14. Generation Integration

- [x] Schedule Generation prefill steps.
- [x] Schedule Generation decode steps.
- [x] Preserve Generation stop semantics.
- [x] Preserve Generation finish reasons.
- [x] Preserve Generation state updates.
- [x] Preserve Generation streaming semantics.
- [x] Add generation batching tests.

## 15. Sampling Integration

- [x] Allow per-operation Sampling parameters.
- [x] Validate Provider-assisted batched sampling compatibility.
- [x] Preserve per-operation token selection.
- [x] Preserve per-operation RNG state.
- [x] Add sampling batching tests.

## 16. Session Integration

- [x] Apply session concurrency policy.
- [x] Apply session queueing policy.
- [x] Apply session cancellation policy.
- [x] Apply session memory budget.
- [x] Apply session KV cache budget.
- [x] Apply session Prefix Cache policy.
- [x] Apply session timeout.
- [x] Add session batching tests.

## 17. Provider Integration

- [x] Read Provider batch support advertisement.
- [x] Validate max batch size.
- [x] Validate max sequence length.
- [x] Validate max total tokens.
- [x] Validate supported dtypes.
- [x] Validate supported KV cache layout.
- [x] Validate paged attention support where advertised.
- [x] Validate Provider-assisted sampling support.
- [x] Use Provider pressure in batch sizing.
- [x] Add Provider batching tests.

## 18. Device Integration

- [x] Validate Device memory capacity.
- [x] Validate Device memory pressure.
- [x] Validate Device compute capability.
- [x] Validate max batch dimensions.
- [x] Validate dtype support where relevant.
- [x] Validate Resource Affinity.
- [x] Add Device batching tests.

## 19. Scheduling Policy

- [x] Define FIFO policy.
- [x] Define priority policy.
- [x] Define deadline policy.
- [x] Define fairness policy.
- [x] Define weighted fairness placeholder.
- [x] Define latency target policy.
- [x] Define throughput target policy.
- [x] Define decode priority policy.
- [x] Define prefill priority policy.
- [x] Define starvation prevention.
- [x] Define max queue time.
- [x] Define max active operations.
- [x] Define max batch tokens.
- [x] Define max batch sequences.
- [x] Define memory pressure adaptation.
- [x] Add scheduling policy tests.

## 20. Fairness

- [x] Define fairness across sessions.
- [x] Define fairness across priorities.
- [x] Define fairness across operation age.
- [x] Define fairness across model contexts where relevant.
- [x] Prevent starvation where policy requires.
- [x] Add fairness tests.

## 21. Backpressure

- [x] Define queue backpressure.
- [x] Define Provider pressure backpressure.
- [x] Define Device pressure backpressure.
- [x] Define memory pressure backpressure.
- [x] Define streaming consumer backpressure.
- [x] Define session concurrency backpressure.
- [x] Define shutdown backpressure.
- [x] Add backpressure tests.

## 22. Streaming Ordering

- [x] Preserve per-operation token order.
- [x] Isolate streaming consumers.
- [x] Handle slow consumer policy.
- [x] Buffer, block, cancel, or fail according to policy.
- [x] Prevent stream corruption across operations.
- [x] Add streaming ordering tests.

## 23. Cancellation

- [x] Cancel queued operation.
- [x] Cancel active prefill.
- [x] Cancel active decode.
- [x] Cancel streaming operation.
- [x] Cancel entire session.
- [x] Cancel entire batch on shutdown where required.
- [x] Coordinate with Generation.
- [x] Coordinate with KV Cache.
- [x] Coordinate with Memory Manager.
- [x] Coordinate with Provider execution.
- [x] Add cancellation tests.

## 24. Failure Isolation

- [x] Map Provider failure per operation where possible.
- [x] Map Device failure per operation where possible.
- [x] Fail entire batch only when continuation is impossible.
- [x] Preserve unaffected operations.
- [x] Add failure isolation tests.

## 25. Dynamic Batch Resizing

- [x] Resize when operations arrive.
- [x] Resize when operations complete.
- [x] Resize on cancellation.
- [x] Resize on stop condition.
- [x] Resize on memory pressure.
- [x] Resize on Provider pressure.
- [x] Resize on Device pressure.
- [x] Resize on streaming backpressure.
- [x] Resize after Prefix Cache hit/miss.
- [x] Preserve operation correctness.
- [x] Add resizing tests.

## 26. Paged Attention Readiness

- [x] Preserve page/block metadata from KV Cache.
- [x] Avoid assuming contiguous KV cache.
- [x] Validate Provider paged attention advertisement.
- [x] Keep paged implementation optional.
- [x] Add paged readiness tests.

## 27. Browser Compatibility

- [x] Keep batching contract platform-neutral.
- [x] Avoid Wasmtime dependency.
- [x] Avoid native Provider loading requirement.
- [x] Account for browser memory limits.
- [x] Account for WebAssembly linear memory.
- [x] Account for future WebGPU capability.
- [x] Return unsupported browser errors where needed.
- [x] Add wasm32 check where feasible.

## 28. Error Model

- [x] Define batch-unavailable error.
- [x] Define batch-admission-rejected error.
- [x] Define queue-full error.
- [x] Define operation-not-found error.
- [x] Define operation-cancelled error.
- [x] Define operation-timed-out error.
- [x] Define session-concurrency-limit error.
- [x] Define model-incompatible error.
- [x] Define tokenizer-incompatible error.
- [x] Define Provider-unavailable error.
- [x] Define Provider-not-ready error.
- [x] Define Provider-saturated error.
- [x] Define Device-unavailable error.
- [x] Define Device-memory-insufficient error.
- [x] Define memory-admission-failed error.
- [x] Define Resource-Affinity-conflict error.
- [x] Define KV-cache-unavailable error.
- [x] Define KV-cache-incompatible error.
- [x] Define Prefix-Cache-reuse-denied error.
- [x] Define batch-compatibility-failed error.
- [x] Define batch-size-unsupported error.
- [x] Define sequence-length-unsupported error.
- [x] Define streaming-backpressure error.
- [x] Define scheduling-policy-denied error.
- [x] Define runtime-shutdown error.
- [x] Define browser-feature-unsupported error.
- [x] Define internal-batching error.

## 29. Observability

- [x] Emit operation admitted observation.
- [x] Emit operation rejected observation.
- [x] Emit operation queued observation.
- [x] Emit batch formed observation.
- [x] Emit batch resized observation.
- [x] Emit prefill scheduled observation.
- [x] Emit decode scheduled observation.
- [x] Emit batch submitted observation.
- [x] Emit batch completed observation.
- [x] Emit operation completed observation.
- [x] Emit operation cancelled observation.
- [x] Emit operation failed observation.
- [x] Emit queue pressure observation.
- [x] Emit memory pressure observation.
- [x] Emit Provider pressure observation.
- [x] Emit Device pressure observation.
- [x] Emit prefix cache hit in batch observation.
- [x] Emit KV cache assigned observation.
- [x] Emit streaming backpressure observation.
- [x] Emit fairness adjustment observation.
- [x] Emit starvation prevented observation.
- [x] Avoid raw prompt/logits/KV cache/Provider handle logging.

## 30. Tests

- [x] Test operation admission success.
- [x] Test admission rejection.
- [x] Test prefill scheduling.
- [x] Test decode scheduling.
- [x] Test mixed prefill/decode scheduling.
- [x] Test batch compatibility.
- [x] Test incompatible model rejected from same batch.
- [x] Test Resource Affinity conflict.
- [x] Test KV cache slot assignment.
- [x] Test Prefix Cache exact hit reduces prefill.
- [x] Test Prefix Cache miss runs full prefill.
- [x] Test session concurrency limit.
- [x] Test queue full.
- [x] Test cancellation queued operation.
- [x] Test cancellation active decode.
- [x] Test streaming order per operation.
- [x] Test slow streaming consumer policy.
- [x] Test Provider saturated backpressure.
- [x] Test memory pressure batch resize.
- [x] Test failure isolation.
- [x] Test raw handles not exposed.

## 31. Documentation

- [x] Document Continuous Batching.
- [x] Document operation lifecycle.
- [x] Document prefill scheduling.
- [x] Document decode scheduling.
- [x] Document batch slots.
- [x] Document compatibility.
- [x] Document Memory Manager relationship.
- [x] Document KV Cache relationship.
- [x] Document Prefix Cache relationship.
- [x] Document Generation relationship.
- [x] Document Sampling relationship.
- [x] Document Session relationship.
- [x] Document Provider relationship.
- [x] Document scheduling policy.
- [x] Document fairness.
- [x] Document backpressure.
- [x] Document cancellation.
- [x] Document browser compatibility.
- [x] Document non-goals.

## 32. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run wasm32 check where feasible.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run Batching tests.
- [x] Run Scheduler tests.
- [x] Run Generation tests.
- [x] Run KV Cache tests.
- [x] Run Prefix Cache tests.
- [x] Run Memory Manager tests.
- [x] Run Provider conformance tests where impacted.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify batching is Runtime/Scheduler-owned.
- [x] Verify Scheduler does not allocate memory directly.
- [x] Verify batching does not own raw KV cache.
- [x] Verify batching does not select Provider/Device directly.
- [x] Verify per-operation streaming order is preserved.
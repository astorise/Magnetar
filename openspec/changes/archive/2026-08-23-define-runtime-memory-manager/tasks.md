# Tasks

## 1. Current Memory Inventory

- [x] Inventory tensor memory-related types in `compute`.
- [x] Inventory memory planning types in `planning`.
- [x] Inventory Device memory metadata in `device`.
- [x] Inventory Provider-owned resource handling in `provider`.
- [x] Inventory Resource Affinity memory-related bindings.
- [x] Inventory data movement memory requirements.
- [x] Identify allocation logic hidden in unrelated modules.
- [x] Identify staging logic hidden in unrelated modules.
- [x] Identify dtype storage/compute consequences hidden in unrelated modules.
- [x] Identify missing first-class memory concepts.

## 2. Memory Module

- [x] Create a first-class `memory` module.
- [x] Export the module from crate root.
- [x] Add module-level documentation.
- [x] Define Memory Manager ownership.
- [x] Keep Compute contract definitions outside allocator internals.
- [x] Keep Device metadata outside global allocator policy.
- [x] Keep Provider execution outside global memory policy.
- [x] Keep Planning as a consumer of memory feasibility.

## 3. Memory Manager

- [x] Define `MemoryManager` or equivalent Runtime service.
- [x] Define Memory Manager configuration.
- [x] Define Memory Manager lifecycle.
- [x] Define allocation request API.
- [x] Define allocation release API.
- [x] Define memory feasibility API.
- [x] Define memory pressure API.
- [x] Define residency query API.
- [x] Define staging feasibility API.
- [x] Define zero-copy feasibility API.

## 4. Allocation Model

- [x] Define MemoryAllocationId.
- [x] Define MemoryAllocation.
- [x] Define allocation size.
- [x] Define allocation alignment.
- [x] Define allocation class.
- [x] Define allocation placement.
- [x] Define allocation owner.
- [x] Define allocation lifetime.
- [x] Define active allocation state.
- [x] Define released allocation state.
- [x] Define invalid allocation behavior.

## 5. Allocation Classes

- [x] Define tensor allocation class.
- [x] Define model-artifact allocation class.
- [x] Define tokenizer-artifact allocation class.
- [x] Define adapter-artifact allocation class.
- [x] Define quantization-artifact allocation class.
- [x] Define KV cache allocation class.
- [x] Define prefix cache allocation class.
- [x] Define temporary workspace allocation class.
- [x] Define transfer staging allocation class.
- [x] Define pinned host allocation class.
- [x] Define browser linear memory allocation class.
- [x] Define Runtime internal allocation class.

## 6. Caching Allocator

- [x] Define caching allocator behavior.
- [x] Define reusable allocation state.
- [x] Define active allocation state.
- [x] Define reserved arena state.
- [x] Define cache hit behavior.
- [x] Define cache miss behavior.
- [x] Define cache eviction behavior.
- [x] Define memory pressure interaction.
- [x] Define cache limits.
- [x] Add caching allocator tests.

## 7. Arena Model

- [x] Define arena identity.
- [x] Define arena memory class.
- [x] Define arena capacity.
- [x] Define arena growth policy.
- [x] Define arena shrink policy where applicable.
- [x] Define arena ownership.
- [x] Define arena pressure.
- [x] Define arena diagnostics.
- [x] Add arena tests.

## 8. Asynchronous Allocation

- [x] Define asynchronous allocation request.
- [x] Define pending allocation state.
- [x] Define completed allocation state.
- [x] Define failed allocation state.
- [x] Define cancelled allocation state.
- [x] Define allocation timeout.
- [x] Define allocation priority.
- [x] Define allocation deadline.
- [x] Add async allocation tests.

## 9. Pending Queues

- [x] Define pending queue ownership.
- [x] Define pending queue admission.
- [x] Define queue ordering policy.
- [x] Define queue timeout policy.
- [x] Define queue cancellation behavior.
- [x] Define retry behavior after memory pressure changes.
- [x] Define queue diagnostics.
- [x] Add pending queue tests.

## 10. Memory Placement

- [x] Define host ordinary placement.
- [x] Define host pinned placement.
- [x] Define device placement.
- [x] Define unified/shared placement.
- [x] Define provider-owned opaque placement.
- [x] Define external borrowed placement.
- [x] Define browser linear memory placement.
- [x] Define staged temporary placement.
- [x] Validate unsupported placement.
- [x] Add placement tests.

## 11. Pinned Host Memory

- [x] Define pinned host memory semantics.
- [x] Define pinned memory limits.
- [x] Define pinned allocation policy.
- [x] Define pinned memory pressure.
- [x] Define pinned memory release.
- [x] Prevent arbitrary Component pinned allocation.
- [x] Add pinned memory tests.

## 12. Zero-Copy

- [x] Define zero-copy feasibility result.
- [x] Evaluate source residency.
- [x] Evaluate target residency.
- [x] Evaluate Provider support.
- [x] Evaluate Device support.
- [x] Evaluate memory type.
- [x] Evaluate alignment.
- [x] Evaluate dtype compatibility.
- [x] Evaluate layout compatibility.
- [x] Evaluate Resource Affinity.
- [x] Evaluate platform constraints.
- [x] Add zero-copy accepted tests.
- [x] Add zero-copy rejected tests.

## 13. Host Staging

- [x] Integrate HostStagingPolicy with Memory Manager.
- [x] Reject staging when forbidden.
- [x] Allow staging consideration when permitted.
- [x] Reject staging when policy denies it.
- [x] Reject staging when memory pressure prevents it.
- [x] Reject staging when platform does not support it.
- [x] Add host staging tests.

## 14. Storage DType And Compute DType

- [x] Define storage dtype.
- [x] Define compute dtype.
- [x] Define dtype relation.
- [x] Define allocation-size impact.
- [x] Define transfer-size impact.
- [x] Define temporary compute buffer impact.
- [x] Define dequantization workspace impact.
- [x] Define error for unsupported storage dtype.
- [x] Define error for unsupported compute dtype.
- [x] Add storage/compute dtype tests.

## 15. Tensor Residency

- [x] Define TensorResidency.
- [x] Track residency by tensor resource.
- [x] Associate tensor residency with Resource Affinity.
- [x] Associate tensor residency with allocation identity.
- [x] Represent provider-owned tensor residency.
- [x] Represent device-bound tensor residency.
- [x] Represent staged tensor residency.
- [x] Prevent Component-forged residency.
- [x] Add tensor residency tests.

## 16. Model Residency Preparation

- [x] Define placeholder model residency concepts.
- [x] Distinguish model artifact bytes from model resident memory.
- [x] Distinguish compressed storage from compute-ready memory.
- [x] Distinguish quantized storage from compute dtype.
- [x] Prepare for sharded model residency.
- [x] Prepare for adapter overlays.
- [x] Do not define full Model Artifact model in this change.

## 17. Adapter And Quantization Residency

- [x] Define adapter residency placeholder.
- [x] Define quantization artifact residency placeholder.
- [x] Associate residency with inference artifact identity.
- [x] Preserve distinction between Component Artifact and inference data
      artifacts.
- [x] Avoid implementing full adapter contract in this change.

## 18. KV Cache Preparation

- [x] Define KV cache allocation class.
- [x] Define KV cache residency placeholder.
- [x] Define KV cache pressure placeholder.
- [x] Prepare for session-scoped cache.
- [x] Prevent KV cache ownership from being hidden in Scheduler.
- [x] Do not define full KV cache semantics in this change.

## 19. Prefix Cache Preparation

- [x] Define prefix cache allocation class.
- [x] Define prefix cache residency placeholder.
- [x] Define prefix cache pressure placeholder.
- [x] Prepare for session-scoped prefix cache.
- [x] Do not define full prefix cache semantics in this change.

## 20. Memory Pressure

- [x] Define Runtime memory pressure.
- [x] Define Provider memory pressure.
- [x] Define Device memory pressure.
- [x] Define arena pressure.
- [x] Define cache pressure.
- [x] Define KV cache pressure placeholder.
- [x] Define pressure levels.
- [x] Integrate pressure with admission.
- [x] Integrate pressure with Scheduler policy.
- [x] Add memory pressure tests.

## 21. Memory Admission

- [x] Define memory admission decision.
- [x] Include admit.
- [x] Include queue.
- [x] Include reject.
- [x] Include retry-later.
- [x] Include reason.
- [x] Use memory pressure.
- [x] Use allocation class.
- [x] Use Resource Affinity.
- [x] Use Provider/Device status.
- [x] Add admission tests.

## 22. Relationship With Planning

- [x] Update planning to call Memory Manager feasibility.
- [x] Remove allocator ownership from planning where present.
- [x] Keep execution planning separate from allocation internals.
- [x] Ensure planning records memory requirements.
- [x] Ensure planning consumes memory admission result.
- [x] Add planning integration tests.

## 23. Relationship With Scheduler

- [x] Allow Scheduler to consume memory admission.
- [x] Allow Scheduler to queue on memory pressure where policy permits.
- [x] Prevent Scheduler from allocating directly.
- [x] Prevent Scheduler from silently staging memory.
- [x] Add Scheduler-memory tests.

## 24. Relationship With Provider

- [x] Define Provider allocation request boundary.
- [x] Define Provider-owned allocation metadata.
- [x] Define Provider allocation failure mapping.
- [x] Define Provider memory pressure reporting integration.
- [x] Prevent raw Provider memory handle exposure.
- [x] Add Provider-memory integration tests.

## 25. Relationship With Device

- [x] Consume Device memory metadata.
- [x] Consume Device memory pressure.
- [x] Validate allocation placement against Device capability.
- [x] Validate pinned or unified memory support.
- [x] Validate browser/native constraints.
- [x] Add Device-memory tests.

## 26. Browser Memory Constraints

- [x] Define browser linear memory placement.
- [x] Define unsupported native pinned memory on browser targets.
- [x] Define browser memory-limit error.
- [x] Define WebGPU buffer placeholder where relevant.
- [x] Ensure native assumptions are not required on wasm32 target.
- [x] Add cfg-aware memory tests where feasible.

## 27. Memory Error Model

- [x] Define allocation denied error.
- [x] Define allocation pending error.
- [x] Define allocation timeout error.
- [x] Define allocation cancelled error.
- [x] Define out-of-memory error.
- [x] Define saturated pressure error.
- [x] Define unsupported memory class error.
- [x] Define unsupported placement error.
- [x] Define unsupported storage dtype error.
- [x] Define unsupported compute dtype error.
- [x] Define zero-copy unavailable error.
- [x] Define staging forbidden error.
- [x] Define staging denied error.
- [x] Define pinned memory unavailable error.
- [x] Define Provider allocation failed error.
- [x] Define Device memory unavailable error.
- [x] Define browser memory limit exceeded error.
- [x] Define invalid allocation handle error.
- [x] Define Resource Affinity conflict error.

## 28. Observability

- [x] Emit allocation requested observation.
- [x] Emit allocation admitted observation.
- [x] Emit allocation queued observation.
- [x] Emit allocation completed observation.
- [x] Emit allocation failed observation.
- [x] Emit allocation released observation.
- [x] Emit cache hit observation.
- [x] Emit cache miss observation.
- [x] Emit cache eviction observation.
- [x] Emit arena pressure observation.
- [x] Emit pending queue delay observation.
- [x] Emit pinned memory usage observation.
- [x] Emit zero-copy accepted observation.
- [x] Emit zero-copy rejected observation.
- [x] Emit staging inserted observation.
- [x] Emit staging denied observation.
- [x] Emit memory pressure change observation.
- [x] Ensure observability failure does not alter memory decision.

## 29. Public API Audit

- [x] Re-export canonical Memory Manager types from crate root.
- [x] Avoid exposing allocator internals unnecessarily.
- [x] Avoid exposing Provider memory handles publicly.
- [x] Avoid exposing Device native memory handles publicly.
- [x] Avoid exposing raw pointers through Component APIs.
- [x] Keep portable Compute descriptors separate from Runtime memory handles.

## 30. Tests

- [x] Add Memory Manager unit tests.
- [x] Add allocator tests.
- [x] Add arena tests.
- [x] Add pending queue tests.
- [x] Add pinned memory tests.
- [x] Add zero-copy feasibility tests.
- [x] Add staging policy tests.
- [x] Add dtype storage/compute tests.
- [x] Add tensor residency tests.
- [x] Add memory pressure tests.
- [x] Add planning integration tests.
- [x] Add Provider-memory integration tests.
- [x] Add Scheduler-memory integration tests.

## 31. Documentation

- [x] Document Memory Manager ownership.
- [x] Document relationship with Compute.
- [x] Document relationship with Planning.
- [x] Document relationship with Provider.
- [x] Document relationship with Device.
- [x] Document caching allocator.
- [x] Document async arena.
- [x] Document pending queues.
- [x] Document pinned memory.
- [x] Document zero-copy.
- [x] Document staging.
- [x] Document storage dtype versus compute dtype.
- [x] Document memory pressure.
- [x] Document browser memory constraints.

## 32. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run Memory Manager tests.
- [x] Run Compute planning tests.
- [x] Run Provider conformance tests where impacted.
- [x] Run WIT validation.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify Memory Manager is first-class.
- [x] Verify allocator logic is not hidden in Compute.
- [x] Verify memory pressure is observable.
- [x] Verify storage dtype and compute dtype are distinct.


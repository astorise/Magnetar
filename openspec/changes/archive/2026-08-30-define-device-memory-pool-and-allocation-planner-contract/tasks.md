# Tasks

## 1. Device Memory Pool Domain

- [x] Define DeviceMemoryPoolId.
- [x] Define DeviceMemoryPool.
- [x] Define MemoryPoolClass.
- [x] Define PoolCapacity.
- [x] Define pool state.
- [x] Add lifecycle tests.

## 2. Pool Classes

- [x] Add weights.
- [x] Add kv-cache.
- [x] Add workspace.
- [x] Add transient.
- [x] Add persistent.
- [x] Add transfer.
- [x] Add shared.
- [x] Keep vocabulary extensible.

## 3. Capacity Accounting

- [x] Track configured limit.
- [x] Track reserved bytes.
- [x] Track committed bytes.
- [x] Track leased bytes.
- [x] Track reclaimable bytes.
- [x] Track pending reclaim bytes.
- [x] Add accounting invariant tests.

## 4. Reservations

- [x] Define MemoryPoolReservation.
- [x] Add hard reservation.
- [x] Add soft reservation.
- [x] Bind reservation scope.
- [x] Add reservation conflict tests.

## 5. Watermarks

- [x] Define high watermark.
- [x] Define low watermark.
- [x] Define optional critical watermark.
- [x] Trigger pressure state.
- [x] Add watermark transition tests.

## 6. Pool State

- [x] Add initializing.
- [x] Add ready.
- [x] Add pressure.
- [x] Add critical.
- [x] Add draining.
- [x] Add failed.
- [x] Add closed.
- [x] Validate transitions.

## 7. Allocation Request

- [x] Define AllocationRequest.
- [x] Add byte size.
- [x] Add alignment.
- [x] Add allocation class.
- [x] Add memory domain.
- [x] Add lifetime class.
- [x] Add residency requirement.
- [x] Add mutability.
- [x] Add reclaimability.
- [x] Add validation.

## 8. Allocation Classes

- [x] Add model-weight.
- [x] Add adapter-weight.
- [x] Add kv-page.
- [x] Add persistent-cache.
- [x] Add execution-workspace.
- [x] Add intermediate.
- [x] Add transfer-staging.
- [x] Add output.
- [x] Keep extensible.

## 9. Lifetime Classes

- [x] Add model-instance.
- [x] Add session.
- [x] Add execution-plan.
- [x] Add batch-step.
- [x] Add operator.
- [x] Add temporary.
- [x] Add cache-entry.

## 10. Allocation Lease

- [x] Define AllocationLeaseId.
- [x] Define AllocationLease.
- [x] Bind pool.
- [x] Bind logical block/region.
- [x] Add generation.
- [x] Define state.
- [x] Add lifecycle tests.

## 11. Allocation Block

- [x] Define AllocationBlock.
- [x] Keep native backing opaque.
- [x] Track capacity.
- [x] Track free ranges logically or internally.
- [x] Add block lifecycle tests.

## 12. Sub-Allocation

- [x] Support multiple Resource regions per block.
- [x] Validate bounds.
- [x] Validate overlap.
- [x] Validate alignment.
- [x] Integrate lifetime.
- [x] Add sub-allocation tests.

## 13. Arena Support

- [x] Permit arena implementation.
- [x] Keep allocator algorithm non-normative.
- [x] Support persistent arenas.
- [x] Support transient arenas.
- [x] Add architecture tests.

## 14. Persistent Memory

- [x] Reserve weight storage.
- [x] Reserve adapter storage.
- [x] Support Provider-prepared constants.
- [x] Avoid per-request native reallocation.
- [x] Add persistent allocation tests.

## 15. Transient Memory

- [x] Support intermediate allocations.
- [x] Support temporary conversion buffers.
- [x] Support per-operation workspace.
- [x] Allow aggressive safe reuse.
- [x] Add transient tests.

## 16. Workspace Requirements

- [x] Consume Kernel workspace metadata.
- [x] Compute maximum required bytes.
- [x] Preserve alignment.
- [x] Preserve memory domain.
- [x] Add workspace requirement tests.

## 17. Workspace Reuse

- [x] Define workspace reuse groups.
- [x] Track CompletionToken barriers.
- [x] Prevent overlapping reuse.
- [x] Add asynchronous workspace tests.

## 18. Allocation Planner

- [x] Define AllocationPlanner.
- [x] Consume graph/resource lifetimes.
- [x] Consume Kernel workspace requirements.
- [x] Consume residency constraints.
- [x] Consume pool policy.
- [x] Produce AllocationPlan.
- [x] Add planner tests.

## 19. Allocation Plan

- [x] Define AllocationPlanId.
- [x] Define AllocationPlan generation.
- [x] Add scope.
- [x] Add pool bindings.
- [x] Add allocation slots.
- [x] Add lifetime intervals.
- [x] Add reuse groups.
- [x] Add reservation requirements.
- [x] Add guards.

## 20. Allocation Plan Identity

- [x] Include graph fingerprint.
- [x] Include workload envelope.
- [x] Include Kernel workspace fingerprint.
- [x] Include memory-domain requirements.
- [x] Include pool policy version.
- [x] Include allocation policy version.
- [x] Exclude native addresses.
- [x] Add stable identity tests.

## 21. Allocation Slots

- [x] Define AllocationSlot.
- [x] Add minimum bytes.
- [x] Add alignment.
- [x] Add pool class.
- [x] Add lifetime.
- [x] Add reuse group.
- [x] Add slot validation tests.

## 22. Lifetime Analysis

- [x] Derive conservative Tensor lifetimes.
- [x] Account for graph dependencies.
- [x] Account for ExecutionStreams.
- [x] Account for CompletionTokens.
- [x] Add overlapping-lifetime tests.

## 23. Temporal Reuse

- [x] Reuse storage for non-overlapping Resources.
- [x] Preserve semantic Resource identity.
- [x] Keep temporal reuse distinct from aliasing.
- [x] Add reuse tests.

## 24. Asynchronous Reuse Safety

- [x] Require previous completion before reuse.
- [x] Integrate ResourceReadiness.
- [x] Add delayed-completion tests.

## 25. Alignment

- [x] Validate power/non-power alignment as supported.
- [x] Check arithmetic overflow.
- [x] Support compatible over-alignment.
- [x] Add alignment tests.

## 26. Size Classes

- [x] Permit size-class implementation.
- [x] Track logical versus reserved bytes.
- [x] Preserve Tensor logical size.
- [x] Add padding/accounting tests.

## 27. Fragmentation

- [x] Track free bytes.
- [x] Track largest free region.
- [x] Track requested/committed bytes.
- [x] Detect large-allocation fragmentation.
- [x] Add fragmentation tests.

## 28. Compaction

- [x] Define optional compaction operation.
- [x] Identify movable Resources.
- [x] Reject pinned Resources.
- [x] Reject in-flight Resources.
- [x] Preserve mappings/Views.
- [x] Add compaction safety tests.

## 29. Resource Movability

- [x] Define movable.
- [x] Define temporarily pinned.
- [x] Define permanently non-movable.
- [x] Integrate Provider/Plan constraints.
- [x] Add movability tests.

## 30. Relocation

- [x] Allocate new backing.
- [x] Perform explicit internal movement.
- [x] Preserve logical Resource identity where allowed.
- [x] Rebind dependent Plans/Providers.
- [x] Add relocation tests.

## 31. Stable Address Requirements

- [x] Allow Prepared Kernel stable-address requirement.
- [x] Allow Prepared Segment stable-address requirement.
- [x] Pin affected Resource backing.
- [x] Add graph-capture tests.

## 32. Prepared Execution Plan Integration

- [x] Reference AllocationPlan generation.
- [x] Bind Resource slots.
- [x] Validate capacity before ready.
- [x] Invalidate on hard plan incompatibility.
- [x] Add integration tests.

## 33. Stable Slots

- [x] Allocate model-weight slots.
- [x] Allocate adapter slots.
- [x] Allocate persistent graph buffers.
- [x] Add stable-slot tests.

## 34. Dynamic Slots

- [x] Allocate Session slots.
- [x] Allocate batch slots.
- [x] Allocate invocation slots.
- [x] Allocate decode-step slots.
- [x] Add dynamic-slot tests.

## 35. Reservation During Plan Build

- [x] Reserve mandatory capacity.
- [x] Fail Plan readiness if required reservation cannot be satisfied.
- [x] Distinguish reservation from physical commitment.
- [x] Add readiness tests.

## 36. Overcommit

- [x] Define explicit overcommit policy.
- [x] Define maximum budget.
- [x] Define reclaim strategy.
- [x] Disable by default unless configured.
- [x] Add overcommit tests.

## 37. Admission Integration

- [x] Include memory projections in admission.
- [x] Reject predictable OOM before partial execution.
- [x] Add admission tests.

## 38. Model Instance Admission

- [x] Account weights.
- [x] Account mandatory workspace.
- [x] Account minimum KV capacity.
- [x] Account pinned resources.
- [x] Account Provider prepared graph requirements.
- [x] Add model-load tests.

## 39. Session Admission

- [x] Estimate KV demand.
- [x] Reserve initial pages.
- [x] Support bounded maximum reservation.
- [x] Add Session admission tests.

## 40. KV Page Pool

- [x] Define logical KVPagePool.
- [x] Define page size.
- [x] Track total/free/leased pages.
- [x] Track pending reclaim pages.
- [x] Add KV pool tests.

## 41. KV Page Lease

- [x] Bind pages to Session/sequence.
- [x] Integrate CompletionToken lifetime.
- [x] Integrate Prefix Cache ownership.
- [x] Add lease tests.

## 42. KV Page Recycling

- [x] Release only after owner ends.
- [x] Wait for in-flight use.
- [x] Wait for shared references.
- [x] Add recycle safety tests.

## 43. KV Growth

- [x] Support incremental page acquisition.
- [x] Define page-exhaustion behavior.
- [x] Integrate spill policy.
- [x] Integrate backpressure.
- [x] Add long-context tests.

## 44. Continuous Batching Pools

- [x] Add reusable batch workspace.
- [x] Add batch-slot allocations.
- [x] Keep protected KV reservation separate.
- [x] Add batching pressure tests.

## 45. Memory Class Isolation

- [x] Prevent optional autotuning from consuming protected inference memory.
- [x] Protect KV reservation from transient work where configured.
- [x] Add class-isolation tests.

## 46. Pool Borrowing

- [x] Define soft capacity borrowing.
- [x] Track borrowed bytes.
- [x] Define reclaim priority.
- [x] Prevent hard reservation borrowing.
- [x] Add borrowing tests.

## 47. Reclaimability

- [x] Mark reclaimable caches.
- [x] Mark optional replicas.
- [x] Mark active Resources non-reclaimable.
- [x] Add reclaimability tests.

## 48. Reclamation

- [x] Trigger on pressure.
- [x] Respect CompletionTokens.
- [x] Respect mappings.
- [x] Respect pinning.
- [x] Respect aliasing.
- [x] Add reclamation tests.

## 49. Pending Reclaim

- [x] Track logically released but in-flight storage.
- [x] Exclude from immediately free capacity.
- [x] Add accounting tests.

## 50. Asynchronous Free

- [x] Support deferred Provider-native release.
- [x] Track pending physical reclaim.
- [x] Prevent immediate reuse.
- [x] Add async-free tests.

## 51. Provider Pool Capabilities

- [x] Advertise block allocation.
- [x] Advertise async native free.
- [x] Advertise address stability.
- [x] Advertise movability.
- [x] Advertise alignment/granularity.
- [x] Advertise grow/shrink capability.
- [x] Add capability tests.

## 52. Device Capacity

- [x] Consume total memory.
- [x] Consume available estimate.
- [x] Consume pressure metadata.
- [x] Keep Device allocation-free.
- [x] Add Device boundary tests.

## 53. Pool Growth

- [x] Allocate additional backing blocks.
- [x] Respect Device capacity.
- [x] Respect policy limit.
- [x] Add growth tests.

## 54. Pool Shrink

- [x] Release wholly unused blocks.
- [x] Preserve live leases.
- [x] Preserve pending reclaim.
- [x] Add shrink tests.

## 55. Pool Drain

- [x] Stop new leases.
- [x] Preserve existing leases.
- [x] Reclaim safely.
- [x] Close after quiescence.
- [x] Add drain tests.

## 56. OOM Classification

- [x] Add pool-capacity-exceeded.
- [x] Add device-capacity-exceeded.
- [x] Add reservation-conflict.
- [x] Add fragmentation.
- [x] Add alignment failure.
- [x] Add pinned-capacity-exhausted.
- [x] Add kv-page-exhausted.
- [x] Add workspace-exhausted.
- [x] Add Provider allocation failure.

## 57. OOM Retry

- [x] Permit bounded reclaim/retry.
- [x] Define maximum retry count.
- [x] Prevent infinite loops.
- [x] Add retry tests.

## 58. OOM Fallback

- [x] Trim optional caches.
- [x] Drop optional replicas.
- [x] Select lower-workspace Kernel.
- [x] Select alternate Plan.
- [x] Select alternate Device.
- [x] Spill if permitted.
- [x] Reject admission.
- [x] Add fallback tests.

## 59. Kernel Selection Integration

- [x] Feed workspace/pool feasibility into eligibility.
- [x] Prevent infeasible fast Kernel selection.
- [x] Add selection tests.

## 60. Autotuning Integration

- [x] Define tuning memory budget.
- [x] Protect inference reservations.
- [x] Deny tuning under critical pressure.
- [x] Add autotuning pressure tests.

## 61. Performance Feedback

- [x] Observe allocation latency.
- [x] Observe fragmentation.
- [x] Observe pressure.
- [x] Feed compatible context to Performance Model.
- [x] Add feedback tests.

## 62. Allocation Plan Cache

- [x] Define AllocationPlan cache.
- [x] Define cache key.
- [x] Add cache lookup.
- [x] Add stale detection.
- [x] Add cache tests.

## 63. Cached Plan Revalidation

- [x] Check pool availability.
- [x] Check reservations.
- [x] Check Provider/Device compatibility.
- [x] Check workspace requirements.
- [x] Check alignment.
- [x] Add revalidation tests.

## 64. WIT Boundary

- [x] Prevent Component pool creation.
- [x] Prevent allocator-strategy control.
- [x] Prevent native handle exposure.
- [x] Preserve Tensor abstraction.
- [x] Add WIT tests.

## 65. Runtime API Boundary

- [x] Prevent inference request native pool selection.
- [x] Keep memory policy deployment/runtime-controlled.
- [x] Add API tests.

## 66. Error Model

- [x] Add pool errors.
- [x] Add allocation errors.
- [x] Add AllocationPlan errors.
- [x] Add lease errors.
- [x] Add fragmentation errors.
- [x] Add compaction/relocation errors.
- [x] Add KV/workspace OOM errors.
- [x] Add reclamation errors.
- [x] Add internal pool error.

## 67. Observability

- [x] Observe pool lifecycle.
- [x] Observe capacity.
- [x] Observe leases.
- [x] Observe reuse.
- [x] Observe pending reclaim.
- [x] Observe fragmentation.
- [x] Observe reclamation.
- [x] Observe compaction.
- [x] Observe KV pages.
- [x] Observe OOM/fallback.
- [x] Redact native handles/data.

## 68. Conformance

- [x] Prove Memory Manager policy authority.
- [x] Prove Provider native realization.
- [x] Prove Device has no allocator API.
- [x] Prove no native pointer semantics.
- [x] Prove temporal reuse safety.
- [x] Prove async reuse safety.
- [x] Prove alignment.
- [x] Prove reservation isolation.
- [x] Prove watermark reclamation.
- [x] Prove pending reclaim accounting.
- [x] Prove fragmentation classification.
- [x] Prove compaction safety.
- [x] Prove address pinning.
- [x] Prove Plan reservation/readiness.
- [x] Prove KV page lifetime.
- [x] Prove class isolation.
- [x] Prove OOM policy.
- [x] Prove cache revalidation.
- [x] Prove redaction.

## 69. Documentation

- [x] Document DeviceMemoryPool.
- [x] Document reservation/watermarks.
- [x] Document AllocationRequest.
- [x] Document AllocationLease.
- [x] Document AllocationPlan.
- [x] Document temporal reuse.
- [x] Document fragmentation.
- [x] Document KVPagePool.
- [x] Document OOM behavior.
- [x] Document Provider/Memory Manager ownership.

## 70. Final Validation

- [x] Run OpenSpec validation.
- [x] Verify no native allocate/free required on normal token path.
- [x] Verify active asynchronous storage is never reused prematurely.
- [x] Verify protected memory classes remain enforceable.
- [x] Verify Device remains metadata/status-only.
- [x] Verify Runtime can maintain stable Device memory usage across decode.
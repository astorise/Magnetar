# Tasks

## 1. Multi Device Domain

- [x] Define DeviceSet.
- [x] Define DeviceSetId.
- [x] Define PlacementDomain.
- [x] Define PlacementBinding.
- [x] Define MultiDevicePlacementPlanId.
- [x] Define MultiDevicePlacementPlan.
- [x] Define generation/state.
- [x] Add lifecycle tests.

## 2. Device Set

- [x] Support one Device.
- [x] Support multiple same-Provider Devices.
- [x] Support multiple Providers where allowed.
- [x] Preserve heterogeneous capability metadata.
- [x] Add DeviceSet fingerprint.
- [x] Add validation tests.

## 3. Placement Plan Identity

- [x] Include graph fingerprint.
- [x] Include Model Instance revision.
- [x] Include DeviceSet.
- [x] Include Provider versions.
- [x] Include memory-capacity class.
- [x] Include placement-policy version.
- [x] Include partition fingerprint.
- [x] Exclude native handles.

## 4. Placement Granularity

- [x] Support Model Instance placement.
- [x] Support segment placement.
- [x] Support layer-range placement.
- [x] Support Operator-group placement.
- [x] Reserve individual Operator placement.
- [x] Add granularity tests.

## 5. Placement Eligibility

- [x] Check Provider capabilities.
- [x] Check Device capabilities.
- [x] Check Kernel availability.
- [x] Check memory feasibility.
- [x] Check Resource Affinity.
- [x] Check transfer policy.
- [x] Check host-staging policy.
- [x] Add eligibility tests.

## 6. Placement Ranking

- [x] Support latency.
- [x] Support throughput.
- [x] Support memory balance.
- [x] Support transfer cost.
- [x] Support pressure.
- [x] Support stability.
- [x] Keep hard filters ahead of ranking.

## 7. Pipeline Placement

- [x] Define PipelineStage.
- [x] Bind graph nodes.
- [x] Bind Provider/Device.
- [x] Bind input/output requirements.
- [x] Preserve graph order.
- [x] Add stage tests.

## 8. Pipeline Stage Movement

- [x] Detect stage boundary.
- [x] Create explicit movement edge.
- [x] Preserve CompletionToken dependency.
- [x] Preserve destination readiness.
- [x] Add stage-transfer tests.

## 9. Pipeline Overlap

- [x] Permit independent request/stage overlap.
- [x] Respect memory capacity.
- [x] Respect dependencies.
- [x] Respect Scheduler policy.
- [x] Add overlap tests.

## 10. Weight Placement

- [x] Support single-Device weight residency.
- [x] Support weight partition.
- [x] Support weight replication.
- [x] Support hybrid partition/replication.
- [x] Add weight-placement tests.

## 11. Weight Replica

- [x] Preserve Model Artifact identity.
- [x] Preserve dtype/layout.
- [x] Preserve revision.
- [x] Track Device residency.
- [x] Track validity.
- [x] Add replica tests.

## 12. Tensor Partition Descriptor

- [x] Define TensorPartitionDescriptor.
- [x] Bind logical Tensor.
- [x] Define axis.
- [x] Define partition count.
- [x] Define shard list.
- [x] Define reconstruction metadata.
- [x] Add validation tests.

## 13. Partition Axis

- [x] Support dimension-index axis.
- [x] Reserve semantic axes.
- [x] Support head axis.
- [x] Support hidden axis.
- [x] Support vocabulary axis.
- [x] Keep representation extensible.

## 14. Tensor Shard

- [x] Define TensorShardId.
- [x] Define TensorShard.
- [x] Bind parent Tensor.
- [x] Bind logical range.
- [x] Bind shape/dtype/layout.
- [x] Bind placement.
- [x] Bind residency.
- [x] Add shard tests.

## 15. Shard Bounds

- [x] Validate offsets/ranges.
- [x] Validate size arithmetic.
- [x] Reject overflow.
- [x] Reject invalid dimensions.
- [x] Add adversarial tests.

## 16. Partition Completeness

- [x] Detect missing ranges.
- [x] Detect illegal overlaps.
- [x] Support intentional replication.
- [x] Distinguish replica from shard.
- [x] Add coverage tests.

## 17. Hybrid Partition

- [x] Represent partitioned regions.
- [x] Represent replicated regions.
- [x] Preserve explicit relationships.
- [x] Add hybrid fixture.

## 18. Partition-Aware Kernel Compatibility

- [x] Add Kernel partition capability metadata.
- [x] Check input shard semantics.
- [x] Check output shard semantics.
- [x] Reject unsupported shard-as-full Tensor.
- [x] Add Kernel compatibility tests.

## 19. Partition Reconstruction

- [x] Define logical reconstruction.
- [x] Avoid mandatory host reconstruction.
- [x] Make materialization explicit if required.
- [x] Add reconstruction tests.

## 20. No Implicit Collectives

- [x] Prevent automatic all-gather.
- [x] Prevent automatic all-reduce.
- [x] Prevent automatic reduce-scatter.
- [x] Require future explicit collective contract.
- [x] Add boundary tests.

## 21. Cross Device Movement

- [x] Integrate explicit data movement.
- [x] Add Device A to Device B movement.
- [x] Add CompletionToken.
- [x] Preserve source lifetime.
- [x] Preserve destination readiness.
- [x] Add movement tests.

## 22. Peer Access

- [x] Query peer-read.
- [x] Query peer-write.
- [x] Query peer-copy.
- [x] Validate per Device pair.
- [x] Add capability tests.

## 23. Peer Transfer

- [x] Prefer direct transfer when policy chooses.
- [x] Keep movement explicit.
- [x] Add transfer metrics.
- [x] Add peer-transfer tests.

## 24. Host Staging

- [x] Detect required host staging.
- [x] Enforce forbid/permit policy.
- [x] Prevent Provider hidden staging.
- [x] Add denied-staging tests.

## 25. Cross Provider Placement

- [x] Support conservative cross-Provider stage boundary.
- [x] Runtime-mediate synchronization.
- [x] Require explicit movement.
- [x] Prevent native-handle exchange.
- [x] Add cross-Provider tests.

## 26. Per Device Memory Budget

- [x] Define Device memory budget.
- [x] Account weights.
- [x] Account KV.
- [x] Account workspace.
- [x] Account transients.
- [x] Account transfer buffers.
- [x] Add budget tests.

## 27. Per Device Pools

- [x] Bind DeviceMemoryPools to placement.
- [x] Bind AllocationSlots per Device.
- [x] Prevent unspecified global pool use.
- [x] Add pool binding tests.

## 28. Heterogeneous Devices

- [x] Model different capacity.
- [x] Model different Kernels.
- [x] Model different dtype support.
- [x] Model different performance.
- [x] Add heterogeneous fixtures.

## 29. Transfer-Aware Cost

- [x] Include expected transfer bytes.
- [x] Include peer bandwidth class.
- [x] Include synchronization cost.
- [x] Include host staging cost.
- [x] Add cost-ranking tests.

## 30. Placement Hysteresis

- [x] Add stability threshold.
- [x] Add cooldown.
- [x] Prevent minor-pressure flapping.
- [x] Add hysteresis tests.

## 31. Placement Pinning

- [x] Pin Model Instance DeviceSet.
- [x] Pin stage.
- [x] Pin weights.
- [x] Pin Session preference.
- [x] Keep safety/compatibility authoritative.
- [x] Add pin tests.

## 32. Prefill Decode Placement

- [x] Support distinct prefill plan.
- [x] Support distinct decode plan.
- [x] Model phase transition.
- [x] Add phase transition tests.

## 33. Prefill Decode State Movement

- [x] Ensure KV availability.
- [x] Ensure weight availability.
- [x] Ensure completion.
- [x] Perform explicit movement.
- [x] Add transition tests.

## 34. KV Device Ownership

- [x] Bind pages to Device.
- [x] Bind sequence affinity.
- [x] Preserve authoritative ownership.
- [x] Add ownership tests.

## 35. KV Locality

- [x] Prefer decode near KV.
- [x] Avoid per-token Device bouncing.
- [x] Include KV movement cost.
- [x] Add locality tests.

## 36. KV Partitioning Boundary

- [x] Allow only when Attention contract supports it.
- [x] Reject unsupported partition.
- [x] Do not invent collectives.
- [x] Add partition boundary tests.

## 37. KV Replication

- [x] Reserve explicit replica semantics.
- [x] Define authoritative copy.
- [x] Define update/coherency requirement.
- [x] Keep baseline conservative.
- [x] Add replica tests.

## 38. Session Placement Affinity

- [x] Bind preferred Device/Plan.
- [x] Preserve KV locality.
- [x] Allow fallback/migration policy.
- [x] Add Session tests.

## 39. Session Migration

- [x] Define explicit migration.
- [x] Move KV.
- [x] Move adapters.
- [x] Move Session buffers.
- [x] Preserve CompletionTokens.
- [x] Add migration tests.

## 40. Model Instance Placement Plans

- [x] Support multiple plans.
- [x] Support workload-specific plans.
- [x] Support degraded plans.
- [x] Add Model Instance integration tests.

## 41. Prepared Plan Integration

- [x] Record exact Device per segment.
- [x] Record explicit movement nodes.
- [x] Record per-Device AllocationPlan.
- [x] Record placement generation.
- [x] Add Plan tests.

## 42. Placement Guards

- [x] Check Device availability.
- [x] Check Provider readiness.
- [x] Check Kernel preparation.
- [x] Check Resource residency.
- [x] Check memory reservation.
- [x] Check peer path.
- [x] Check host-staging policy.
- [x] Add guard tests.

## 43. Placement Staleness

- [x] Mark stale on pressure shift.
- [x] Mark stale on better placement.
- [x] Mark stale on performance drift.
- [x] Request background re-placement.
- [x] Add stale tests.

## 44. Hard Invalidation

- [x] Invalidate on Device loss.
- [x] Invalidate on Provider loss.
- [x] Invalidate on peer-path loss where required.
- [x] Invalidate on memory infeasibility.
- [x] Invalidate on Kernel revocation/unavailability.
- [x] Add invalidation tests.

## 45. Re Placement

- [x] Define replacement request.
- [x] Build outside hot path.
- [x] Revalidate resources.
- [x] Prepare required Kernels.
- [x] Add replacement tests.

## 46. Atomic Replacement

- [x] Prepare complete new Plan generation.
- [x] Publish atomically.
- [x] Retain old in-flight Plan.
- [x] Add concurrency tests.

## 47. Device Failure

- [x] Detect Device loss.
- [x] Invalidate dependent streams.
- [x] Invalidate Plans.
- [x] Preserve other Devices.
- [x] Add failure tests.

## 48. Degraded Plan

- [x] Support explicit degraded Plan.
- [x] Verify model capacity.
- [x] Verify Kernels.
- [x] Verify memory.
- [x] Verify policy.
- [x] Add degraded tests.

## 49. No Implicit Failover

- [x] Prevent automatic use of arbitrary remaining Device.
- [x] Require ready/built fallback.
- [x] Add failure-without-fallback test.

## 50. Device Recovery

- [x] Re-run health/readiness.
- [x] Rebuild pools.
- [x] Reprepare Kernels.
- [x] Rebuild Placement Plan.
- [x] Add recovery tests.

## 51. Scheduler Integration

- [x] Consume Session affinity.
- [x] Consume Device pressure.
- [x] Consume Plan readiness.
- [x] Keep native handles private.
- [x] Add Scheduler tests.

## 52. Admission

- [x] Check all mandatory Devices.
- [x] Check per-Device memory.
- [x] Check required Kernels.
- [x] Check transfer feasibility.
- [x] Add admission tests.

## 53. Cross Device Concurrency

- [x] Permit independent Device execution.
- [x] Preserve dependency semantics.
- [x] Preserve resource lifetime.
- [x] Add concurrency tests.

## 54. Failure Propagation

- [x] Stop dependent downstream stage after upstream failure.
- [x] Permit explicit fallback.
- [x] Preserve structured reason.
- [x] Add failure-chain tests.

## 55. Replica Eviction

- [x] Evict optional replica independently.
- [x] Preserve authoritative copy.
- [x] Invalidate dependent Plan if needed.
- [x] Add replica eviction tests.

## 56. Kernel Selection Integration

- [x] Couple Device compatibility with Kernel selection.
- [x] Preserve hard eligibility.
- [x] Include transfer/memory cost.
- [x] Add joint-selection tests.

## 57. Autotuning Integration

- [x] Keep tuning Device-specific.
- [x] Revalidate evidence on Device change.
- [x] Add target-specific tests.

## 58. Performance Feedback

- [x] Record Device placement context.
- [x] Detect placement regression.
- [x] Request re-placement.
- [x] Add performance tests.

## 59. Placement Cache

- [x] Define Placement Plan cache.
- [x] Define cache key.
- [x] Add lookup.
- [x] Add invalidation.
- [x] Add cache tests.

## 60. Cached Plan Revalidation

- [x] Check Device availability.
- [x] Check Provider readiness.
- [x] Check memory capacity.
- [x] Check peer capability.
- [x] Check Kernel availability.
- [x] Check policy.
- [x] Add revalidation tests.

## 61. Native Handle Boundary

- [x] Prevent Device pointers.
- [x] Prevent peer handles.
- [x] Prevent native queues.
- [x] Prevent OS interop handles.
- [x] Add handle-leak tests.

## 62. WIT Boundary

- [x] Prevent Component Device selection.
- [x] Prevent topology exposure as authority.
- [x] Preserve portable graph semantics.
- [x] Add WIT tests.

## 63. Runtime API Boundary

- [x] Prevent normal request layer-to-GPU mapping.
- [x] Permit only high-level preferences.
- [x] Keep admin/deployment placement separate.
- [x] Add API tests.

## 64. Error Model

- [x] Add placement errors.
- [x] Add partition errors.
- [x] Add movement errors.
- [x] Add KV placement errors.
- [x] Add Device failure errors.
- [x] Add degraded mode errors.
- [x] Add internal placement error.

## 65. Observability

- [x] Observe placement-plan build.
- [x] Observe stage placement.
- [x] Observe Tensor partition.
- [x] Observe weight replicas.
- [x] Observe Device transfers.
- [x] Observe Session affinity.
- [x] Observe migration.
- [x] Observe stale/invalidation.
- [x] Observe degraded mode.
- [x] Redact native handles/data.

## 66. Conformance

- [x] Prove Runtime placement authority.
- [x] Prove Model Component cannot force Device.
- [x] Prove partition validity.
- [x] Prove replica/partition distinction.
- [x] Prove shard cannot masquerade as full Tensor.
- [x] Prove explicit cross-Device movement.
- [x] Prove host-staging policy.
- [x] Prove peer capability requirement.
- [x] Prove per-Device memory policy.
- [x] Prove heterogeneous Device support.
- [x] Prove transfer-aware selection.
- [x] Prove exact Prepared Plan placement.
- [x] Prove no mid-flight migration.
- [x] Prove KV locality.
- [x] Prove explicit Session migration.
- [x] Prove Device-loss invalidation.
- [x] Prove degraded-plan validation.
- [x] Prove recovery lifecycle.
- [x] Prove cache revalidation.
- [x] Prove handle isolation.
- [x] Prove observability redaction.

## 67. Documentation

- [x] Document DeviceSet.
- [x] Document PlacementDomain.
- [x] Document MultiDevicePlacementPlan.
- [x] Document pipeline stages.
- [x] Document TensorPartitionDescriptor.
- [x] Document TensorShard.
- [x] Document weight replication.
- [x] Document KV locality.
- [x] Document Device failure/degraded plan.
- [x] Document local-only scope.

## 68. Final Validation

- [x] Run OpenSpec validation.
- [x] Verify no distributed collective semantics are introduced.
- [x] Verify Runtime remains placement authority.
- [x] Verify Device remains descriptive.
- [x] Verify all cross-Device movement is explicit.
- [x] Verify one Model Instance can use multiple Devices safely.
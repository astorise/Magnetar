# Tasks

## 1. Residency Domain

- [x] Define MemoryDomain.
- [x] Define ResourceResidency.
- [x] Define ResidencySet.
- [x] Define residency states.
- [x] Bind Provider/Device where applicable.
- [x] Add validation tests.

## 2. Memory Domain Classes

- [x] Add host.
- [x] Add device-local.
- [x] Add host-visible-device.
- [x] Add shared.
- [x] Add managed.
- [x] Reserve external.
- [x] Keep vocabulary extensible.

## 3. Authoritative Storage

- [x] Track authoritative Resource storage.
- [x] Track current replicas.
- [x] Track stale replicas.
- [x] Prevent stale replica reads.
- [x] Add replication-validity tests.

## 4. Device Residency

- [x] Support persistent Device-resident weights.
- [x] Support Device-resident intermediates.
- [x] Support Device-resident KV cache.
- [x] Support Device-resident workspace.
- [x] Add residency lifecycle tests.

## 5. No Host Round Trip

- [x] Prevent mandatory host staging between same-Device compatible Operators.
- [x] Add same-Device pipeline test.
- [x] Add repeated-decode residency test.
- [x] Add host-copy regression test.

## 6. Weight Replication

- [x] Support immutable weight replicas.
- [x] Track Device affinity.
- [x] Preserve Artifact identity.
- [x] Support concurrent reads.
- [x] Add multi-Device weight tests.

## 7. Resource View

- [x] Define ResourceView.
- [x] Add parent Resource.
- [x] Add offset.
- [x] Add shape.
- [x] Add strides.
- [x] Add layout.
- [x] Add View identity.
- [x] Add lifecycle tests.

## 8. View Safety

- [x] Validate bounds.
- [x] Validate integer overflow.
- [x] Validate strides.
- [x] Track overlapping Views.
- [x] Add adversarial View tests.

## 9. Non-Contiguous Views

- [x] Represent non-contiguous View.
- [x] Match Kernel compatibility.
- [x] Prevent hidden contiguous materialization.
- [x] Add non-contiguous tests.

## 10. View Materialization

- [x] Define explicit materialization operation.
- [x] Produce new Tensor Resource.
- [x] Preserve source lifetime.
- [x] Add materialization tests.

## 11. Aliasing

- [x] Track allocation alias relationships.
- [x] Integrate aliasing with ResourceReadiness.
- [x] Integrate aliasing with memory reuse.
- [x] Integrate aliasing with Prepared Plans.
- [x] Add alias tests.

## 12. Zero-Copy Definition

- [x] Define zero-copy semantics.
- [x] Distinguish zero-copy from no synchronization.
- [x] Distinguish zero-copy from no mapping.
- [x] Add semantic tests/documentation.

## 13. Zero-Copy Eligibility

- [x] Check Provider.
- [x] Check Device.
- [x] Check memory domain.
- [x] Check affinity.
- [x] Check dtype.
- [x] Check layout.
- [x] Check alignment.
- [x] Check access mode.
- [x] Check coherency.
- [x] Check readiness.
- [x] Add eligibility tests.

## 14. Resource Mapping

- [x] Define ResourceMappingId.
- [x] Define ResourceMapping.
- [x] Add access mode.
- [x] Add range.
- [x] Add mapped domain.
- [x] Add state.
- [x] Add mapping tests.

## 15. Mapping Readiness

- [x] Wait/order pending Device writes before host read.
- [x] Order host writes before Device read.
- [x] Detect conflicting mapping.
- [x] Add mapping hazard tests.

## 16. Mapping Lifetime

- [x] Pin Resource lifetime while mapped.
- [x] Prevent eviction while mapping active.
- [x] Prevent incompatible reuse.
- [x] Add mapping lifetime tests.

## 17. Mapping Release

- [x] Define unmap/release.
- [x] Establish write visibility on release.
- [x] Release Provider-native mapping safely.
- [x] Add release tests.

## 18. Mapping Pointer Boundary

- [x] Keep native pointer mapping-scoped.
- [x] Prevent pointer serialization.
- [x] Prevent pointer as Resource identity.
- [x] Prevent WIT exposure.
- [x] Add pointer-leak tests.

## 19. Coherent Mapping

- [x] Advertise coherent visibility.
- [x] Preserve execution synchronization.
- [x] Add coherent mapping tests.

## 20. Non-Coherent Mapping

- [x] Advertise non-coherent mapping.
- [x] Perform required host-read visibility operation.
- [x] Perform required Device-read visibility operation.
- [x] Add coherency tests.

## 21. Pinned Host Memory

- [x] Model host allocation optimized for Device access/transfer.
- [x] Keep terminology portable.
- [x] Preserve distinction from Device-local memory.
- [x] Add pinned-memory tests.

## 22. Shared Memory

- [x] Model host/Device shared visibility.
- [x] Define synchronization characteristics.
- [x] Define performance metadata.
- [x] Add shared-memory tests.

## 23. Managed Memory

- [x] Represent managed/unified memory capability.
- [x] Keep vendor-neutral semantics.
- [x] Preserve logical Resource identity across migration.
- [x] Add managed-memory tests.

## 24. Residency Preference

- [x] Define preferred residency intent.
- [x] Support preserve source affinity.
- [x] Support Device-local preference.
- [x] Support host-visible preference.
- [x] Keep preference non-authoritative.
- [x] Add preference tests.

## 25. Hard Residency Requirement

- [x] Allow Kernel/Plan required residency.
- [x] Validate before submission.
- [x] Reject incompatible Resource.
- [x] Add requirement tests.

## 26. Explicit Data Movement

- [x] Represent host-to-Device movement.
- [x] Represent Device-to-host movement.
- [x] Represent Device-to-Device movement.
- [x] Represent cross-Provider movement.
- [x] Preserve host-staging policy.
- [x] Add movement tests.

## 27. No Hidden Host Staging

- [x] Detect Provider-required host staging.
- [x] Report requirement to Runtime.
- [x] Deny when policy forbids.
- [x] Add hidden-staging tests.

## 28. Asynchronous Transfer

- [x] Submit movement on logical ExecutionStream.
- [x] Produce CompletionToken.
- [x] Update destination ResourceReadiness.
- [x] Preserve source lifetime.
- [x] Add async transfer tests.

## 29. Copy Elision

- [x] Detect already compatible residency.
- [x] Remove redundant movement.
- [x] Preserve semantics.
- [x] Add transfer-elision tests.

## 30. Prepared Plan Residency

- [x] Add residency assumptions to Plan.
- [x] Add residency guards.
- [x] Add stable Resource residency binding.
- [x] Add dynamic Resource residency binding.
- [x] Add Plan tests.

## 31. Plan Residency Guard Failure

- [x] Use compatible replica where allowed.
- [x] Use explicit movement where planned/permitted.
- [x] Request rebind/replan.
- [x] Fail safely when no path exists.
- [x] Add guard-failure tests.

## 32. Peer Access Discovery

- [x] Advertise peer-read.
- [x] Advertise peer-write.
- [x] Advertise peer-read-write.
- [x] Advertise peer-copy.
- [x] Keep capability explicit.
- [x] Add discovery tests.

## 33. Peer Zero Copy

- [x] Validate Device pair.
- [x] Validate Provider.
- [x] Validate Resource affinity.
- [x] Validate synchronization.
- [x] Add peer zero-copy tests.

## 34. Peer Transfer

- [x] Support direct Device-to-Device transfer.
- [x] Keep movement explicit.
- [x] Avoid host staging where supported.
- [x] Add peer-transfer tests.

## 35. Cross-Provider Boundary

- [x] Deny implicit zero-copy across Providers.
- [x] Runtime-mediate normal transfers.
- [x] Reserve future interoperability capability.
- [x] Add cross-Provider tests.

## 36. External Memory Handle Boundary

- [x] Prevent DMA-BUF in generic Resource contract.
- [x] Prevent file descriptor exposure.
- [x] Prevent native OS handles.
- [x] Prevent CUDA IPC handles.
- [x] Prevent Vulkan external-memory handles.
- [x] Add handle-leak tests.

## 37. Explicit Resource Import

- [x] Reserve Provider import capability.
- [x] Validate size.
- [x] Validate access rights.
- [x] Validate alignment.
- [x] Validate lifetime.
- [x] Validate Device/Provider compatibility.
- [x] Add import boundary tests.

## 38. Explicit Resource Export

- [x] Reserve policy-controlled export capability.
- [x] Prevent automatic export.
- [x] Prevent inference-client native handle access.
- [x] Add export tests.

## 39. WASM Boundary

- [x] Keep Tensor Resources logical.
- [x] Prevent native pointers.
- [x] Prevent external-memory handles.
- [x] Preserve portable movement semantics.
- [x] Add Component boundary tests.

## 40. Memory Pressure

- [x] Integrate Device residency with pressure.
- [x] Support eviction.
- [x] Support spill.
- [x] Support reduced replication.
- [x] Support alternate Device placement.
- [x] Add pressure tests.

## 41. Eviction Safety

- [x] Prevent eviction while in-flight.
- [x] Preserve ResourceReadiness.
- [x] Preserve mapping lifetime.
- [x] Add eviction race tests.

## 42. Spill Policy

- [x] Make spill explicit.
- [x] Respect host-staging policy.
- [x] Preserve synchronization.
- [x] Add spill-denied tests.

## 43. Residency Pinning

- [x] Support bounded residency pin.
- [x] Integrate admission/capacity.
- [x] Add weight pinning.
- [x] Add KV pinning.
- [x] Add pinning tests.

## 44. KV Cache Residency

- [x] Keep KV pages Device-resident where possible.
- [x] Preserve per-sequence ownership.
- [x] Preserve page readiness.
- [x] Integrate spill/eviction.
- [x] Add long decode tests.

## 45. Prefix Cache Residency

- [x] Support Device-resident prefix entries.
- [x] Support read-only sharing.
- [x] Handle cross-Device replica/transfer.
- [x] Add prefix tests.

## 46. Adapter Residency

- [x] Support persistent adapter weights.
- [x] Preserve adapter revision.
- [x] Integrate Model Instance Plan validity.
- [x] Add adapter tests.

## 47. Quantization And Layout

- [x] Preserve quantization metadata.
- [x] Prevent resident-byte reinterpretation.
- [x] Enforce packing compatibility.
- [x] Add quantization tests.

## 48. Alignment

- [x] Record allocation alignment.
- [x] Check Kernel requirements.
- [x] Trigger explicit materialization/alternate Kernel when needed.
- [x] Add alignment tests.

## 49. Read-Only Sharing

- [x] Mark immutable Resources.
- [x] Allow concurrent Device reads.
- [x] Prevent unauthorized mutation.
- [x] Add sharing tests.

## 50. Provider Capability Discovery

- [x] Advertise memory domains.
- [x] Advertise host mapping.
- [x] Advertise coherency.
- [x] Advertise pinned host allocation.
- [x] Advertise shared memory.
- [x] Advertise managed memory.
- [x] Advertise peer access.
- [x] Advertise peer transfer.
- [x] Add capability tests.

## 51. Device Boundary

- [x] Keep Device descriptive.
- [x] Do not add `allocate`.
- [x] Do not add `map`.
- [x] Do not add peer-copy methods.
- [x] Expose capability metadata only.
- [x] Add architecture tests.

## 52. Memory Manager Authority

- [x] Retain logical allocation authority.
- [x] Retain residency policy.
- [x] Retain eviction policy.
- [x] Retain movement authorization.
- [x] Retain Resource lifetime.
- [x] Add authority tests.

## 53. Provider Authority

- [x] Realize native allocation.
- [x] Realize mapping.
- [x] Realize native transfer.
- [x] Realize coherency operations.
- [x] Keep native handles private.
- [x] Add Provider boundary tests.

## 54. Error Model

- [x] Add residency errors.
- [x] Add zero-copy errors.
- [x] Add View errors.
- [x] Add mapping errors.
- [x] Add coherency errors.
- [x] Add transfer errors.
- [x] Add peer-access errors.
- [x] Add eviction/spill errors.
- [x] Add internal residency error.

## 55. Observability

- [x] Observe residency.
- [x] Observe mapping.
- [x] Observe Views.
- [x] Observe transfer.
- [x] Observe copy elision.
- [x] Observe peer access.
- [x] Observe eviction/spill.
- [x] Observe zero-copy decision.
- [x] Redact native pointers/handles.

## 56. Conformance

- [x] Prove no same-Device host round-trip requirement.
- [x] Prove persistent weights.
- [x] Prove persistent KV.
- [x] Prove intermediate residency.
- [x] Prove zero-copy compatibility gates.
- [x] Prove View no-copy semantics.
- [x] Prove View bounds safety.
- [x] Prove mapping readiness.
- [x] Prove mapping lifetime.
- [x] Prove coherency semantics.
- [x] Prove pointer/handle isolation.
- [x] Prove explicit movement.
- [x] Prove host-staging prohibition.
- [x] Prove async transfer lifetime.
- [x] Prove peer capability requirement.
- [x] Prove cross-Provider isolation.
- [x] Prove Memory Manager authority.
- [x] Prove Plan has no native memory ownership.
- [x] Prove eviction safety.
- [x] Prove observability redaction.

## 57. Documentation

- [x] Document MemoryDomain.
- [x] Document ResourceResidency.
- [x] Document zero-copy meaning.
- [x] Document ResourceView.
- [x] Document ResourceMapping.
- [x] Document coherent/non-coherent mapping.
- [x] Document explicit movement.
- [x] Document peer access.
- [x] Document Device-resident KV.
- [x] Document Provider/Memory Manager boundary.

## 58. Final Validation

- [x] Run OpenSpec validation.
- [x] Verify zero-copy never bypasses semantics.
- [x] Verify no hidden host staging.
- [x] Verify native pointers remain private.
- [x] Verify Device remains metadata/status-only.
- [x] Verify Memory Manager retains residency authority.
- [x] Verify normal decode can remain Device-resident.

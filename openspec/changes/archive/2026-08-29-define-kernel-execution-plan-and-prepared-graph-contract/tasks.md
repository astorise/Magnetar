# Tasks

## 1. Plan Domain

- [x] Define PreparedExecutionPlanId.
- [x] Define PreparedExecutionPlanGeneration.
- [x] Define PreparedExecutionPlan.
- [x] Define Plan state.
- [x] Define Plan scope.
- [x] Define Plan fingerprint.
- [x] Add lifecycle tests.

## 2. Execution Graph Fingerprint

- [x] Define semantic graph fingerprint.
- [x] Include Operator IDs.
- [x] Include Operator versions.
- [x] Include portable attributes.
- [x] Include topology.
- [x] Include logical tensor descriptors.
- [x] Exclude native handles.
- [x] Add deterministic fingerprint tests.

## 3. Model Instance Binding

- [x] Bind Plan to Model Instance.
- [x] Bind Model Instance revision.
- [x] Bind relevant adapter revision.
- [x] Bind execution-policy revision.
- [x] Add revision mismatch tests.

## 4. Plan Scope

- [x] Add execution phase.
- [x] Add workload bucket.
- [x] Add shape envelope.
- [x] Add dtype/layout.
- [x] Add batching mode.
- [x] Add KV-cache mode.
- [x] Add Provider/Device compatibility.
- [x] Add quantization mode where relevant.
- [x] Add scope validation tests.

## 5. Plan Node Binding

- [x] Define PlanNodeBinding.
- [x] Reference graph node/group.
- [x] Reference KernelId.
- [x] Reference Kernel Artifact digest.
- [x] Reference specialization.
- [x] Reference Provider.
- [x] Reference Device.
- [x] Reference opaque PreparedKernelId.
- [x] Reference execution mode.
- [x] Add binding tests.

## 6. Exact Kernel Binding

- [x] Record qualification profile where relevant.
- [x] Record Prepared Kernel generation.
- [x] Record artifact digest.
- [x] Record specialization identity.
- [x] Prevent implicit latest-Kernel substitution.
- [x] Add exact-binding tests.

## 7. Native Handle Boundary

- [x] Prevent function pointers in Plan.
- [x] Prevent Device native handles.
- [x] Prevent Provider pointers.
- [x] Prevent native graph pointers.
- [x] Keep PreparedKernelId opaque.
- [x] Add handle-leak tests.

## 8. Prepared Execution Segments

- [x] Define logical PreparedExecutionSegment.
- [x] Bind segment graph nodes.
- [x] Bind Provider.
- [x] Bind Device.
- [x] Define optional ProviderPreparedSegmentId.
- [x] Add segment validation.

## 9. Provider Segment Handle

- [x] Make ProviderPreparedSegmentId opaque.
- [x] Prevent pointer semantics.
- [x] Keep Provider mapping private.
- [x] Define destruction lifecycle.
- [x] Add opacity tests.

## 10. Segment Semantics

- [x] Preserve graph Operator semantics.
- [x] Validate fused groups.
- [x] Validate node ordering/dependencies.
- [x] Reject hidden semantic changes.
- [x] Add segment equivalence tests.

## 11. Cross-Provider Segments

- [x] Keep Provider/Device binding explicit.
- [x] Require explicit inter-segment data movement.
- [x] Enforce Resource Affinity.
- [x] Enforce host-staging policy.
- [x] Add cross-Provider tests.

## 12. Resource Binding Plan

- [x] Define ResourceBindingPlan.
- [x] Define stable resource slots.
- [x] Define dynamic resource slots.
- [x] Define workspace slots.
- [x] Define KV slots.
- [x] Define adapter slots.
- [x] Define intermediate slots.
- [x] Add resource-plan tests.

## 13. Stable Resources

- [x] Support model-weight resource references.
- [x] Support immutable adapter resources.
- [x] Support Provider-prepared constants.
- [x] Preserve Memory Manager ownership.
- [x] Add stable-resource tests.

## 14. Dynamic Resources

- [x] Support per-request input slots.
- [x] Support output slots.
- [x] Support Session KV slots.
- [x] Support continuous-batch slots.
- [x] Support temporary workspace.
- [x] Prevent Session resource capture.
- [x] Add dynamic-resource tests.

## 15. KV Cache Integration

- [x] Record required KV layout.
- [x] Record KV affinity.
- [x] Record append/read requirements.
- [x] Keep KV contents Session-owned.
- [x] Add KV-plan tests.

## 16. Memory Plan

- [x] Define Plan memory requirements.
- [x] Add workspace upper bounds.
- [x] Add allocation lifetime classes.
- [x] Add aliasing/reuse plan.
- [x] Add placement requirements.
- [x] Preserve Memory Manager authority.
- [x] Add memory-plan tests.

## 17. Memory Revalidation

- [x] Add lightweight feasibility guard.
- [x] Detect hard infeasibility.
- [x] Request re-plan when required.
- [x] Add pressure/invalidation tests.

## 18. Plan Guards

- [x] Define PlanGuard.
- [x] Add shape guard.
- [x] Add dtype guard.
- [x] Add layout guard.
- [x] Add phase guard.
- [x] Add batch guard.
- [x] Add sequence guard.
- [x] Add adapter-revision guard.
- [x] Add KV-layout guard.
- [x] Add affinity guard.
- [x] Add readiness guard.
- [x] Add guard tests.

## 19. Guard Cost

- [x] Keep guard evaluation bounded.
- [x] Prevent candidate discovery during guard.
- [x] Prevent qualification during guard.
- [x] Prevent benchmarking during guard.
- [x] Prevent compilation during guard.
- [x] Add hot-path guard tests.

## 20. Guard Failure

- [x] Support alternate Plan lookup.
- [x] Support re-plan request.
- [x] Support explicit fallback.
- [x] Support structured failure.
- [x] Never execute incompatible Plan.
- [x] Add failure tests.

## 21. Dynamic Shapes

- [x] Define shape envelopes.
- [x] Validate exact/bounded dimensions.
- [x] Reject outside-envelope execution.
- [x] Add dynamic-shape tests.

## 22. Plan Families

- [x] Define Plan family/key.
- [x] Support workload buckets.
- [x] Support multiple shape envelopes.
- [x] Support phase-specific Plans.
- [x] Use bounded Plan lookup.
- [x] Add Plan-family tests.

## 23. Prefill/Decode

- [x] Support distinct prefill Plan.
- [x] Support distinct decode Plan.
- [x] Allow different Kernels.
- [x] Allow different specialization.
- [x] Allow different workspace.
- [x] Add generation-phase tests.

## 24. Continuous Batching

- [x] Add active-sequence constraints.
- [x] Add total-token constraints.
- [x] Add raggedness compatibility.
- [x] Add paged-KV compatibility.
- [x] Add batch-slot binding.
- [x] Avoid request identity capture.
- [x] Add continuous-batching tests.

## 25. Plan Build Pipeline

- [x] Validate graph.
- [x] Query Registry.
- [x] Apply eligibility.
- [x] Apply Kernel Selection Policy.
- [x] Resolve specialization.
- [x] Consume autotuning evidence.
- [x] Build Memory Plan.
- [x] Prepare Kernels.
- [x] Prepare optional segments.
- [x] Validate final Plan.
- [x] Mark ready atomically.
- [x] Add pipeline tests.

## 26. Plan Build Boundary

- [x] Keep Plan build outside normal execute.
- [x] Prevent AI generation.
- [x] Prevent Optimization Campaign launch.
- [x] Permit bounded warmup autotuning.
- [x] Permit cold-path compilation by policy.
- [x] Add boundary tests.

## 27. Plan States

- [x] Add building.
- [x] Add validating.
- [x] Add preparing.
- [x] Add ready.
- [x] Add stale.
- [x] Add invalidated.
- [x] Add retiring.
- [x] Add retired.
- [x] Add failed.
- [x] Validate state transitions.

## 28. Ready State

- [x] Require mandatory Kernel preparation.
- [x] Require hard guards configured.
- [x] Require Memory Plan validity.
- [x] Require current hard policy.
- [x] Add ready-state tests.

## 29. Stale State

- [x] Define optimization-safe staleness.
- [x] Allow temporary execution by policy.
- [x] Trigger re-plan request.
- [x] Keep distinct from invalidated.
- [x] Add stale tests.

## 30. Hard Invalidation

- [x] Invalidate on Kernel revocation.
- [x] Invalidate on required qualification revocation.
- [x] Invalidate on trust denial.
- [x] Invalidate on Provider unavailability.
- [x] Invalidate on Device hard unavailability.
- [x] Invalidate on affinity incompatibility.
- [x] Invalidate on incompatible Model revision.
- [x] Invalidate on missing Prepared Kernel.
- [x] Add invalidation tests.

## 31. Staleness Signals

- [x] Mark stale on better Kernel promotion.
- [x] Mark stale on policy preference update.
- [x] Mark stale on autotuning evidence aging.
- [x] Mark stale on performance regression.
- [x] Mark stale on workload drift.
- [x] Add staleness tests.

## 32. Revalidation

- [x] Define lightweight Plan revalidation.
- [x] Avoid expensive rebuild when still valid.
- [x] Recheck hard dependencies.
- [x] Add revalidation tests.

## 33. Rebuild Request

- [x] Define PlanRebuildRequest.
- [x] Add reason.
- [x] Add desired workload scope.
- [x] Add urgency.
- [x] Deduplicate equivalent requests.
- [x] Add rebuild tests.

## 34. No Hot-Path Rebuild

- [x] Prevent full Registry scan in token loop.
- [x] Prevent compile in Plan execution.
- [x] Prevent autotuning benchmark in Plan execution.
- [x] Prevent memory-plan rebuild in execution.
- [x] Add hot-path conformance tests.

## 35. Safe Plan Switch Boundary

- [x] Define pre-invocation boundary.
- [x] Support decode-step boundary.
- [x] Support batch scheduling boundary.
- [x] Prevent mid-invocation replacement.
- [x] Add concurrency tests.

## 36. Plan Generation Lease

- [x] Define Plan lease/reference.
- [x] Acquire before dispatch.
- [x] Release after completion.
- [x] Prevent destroy while referenced.
- [x] Add lifecycle tests.

## 37. Atomic Plan Replacement

- [x] Prepare complete new generation first.
- [x] Publish atomically.
- [x] Mark old generation retiring.
- [x] Route new work to new generation.
- [x] Preserve old in-flight work.
- [x] Add atomic replacement tests.

## 38. Kernel Hot Swap Integration

- [x] Do not mutate Plan binding in place.
- [x] Build new Plan generation for promoted Kernel.
- [x] Preserve old generation for in-flight work.
- [x] Add Kernel-promotion tests.

## 39. Kernel Revocation Integration

- [x] Locate Plans dependent on revoked Kernel.
- [x] Invalidate for new work.
- [x] Use ready fallback Plan where available.
- [x] Trigger rebuild otherwise.
- [x] Add revocation tests.

## 40. Adaptive Feedback Integration

- [x] Consume Performance Model staleness signal.
- [x] Mark Plan stale.
- [x] Request re-plan.
- [x] Never mutate Plan in place.
- [x] Add adaptive feedback tests.

## 41. Selection Policy Integration

- [x] Record selection policy version.
- [x] Detect material policy change.
- [x] Distinguish preference-only from hard-policy change.
- [x] Add policy invalidation tests.

## 42. Autotuning Integration

- [x] Record Autotuning Record reference.
- [x] Record specialization.
- [x] Mark stale on incompatible tuning evidence.
- [x] Preserve safety when evidence only becomes performance-stale.
- [x] Add tuning integration tests.

## 43. Plan Cache

- [x] Define PreparedExecutionPlanCache.
- [x] Keep distinct from artifact/tuning caches.
- [x] Define cache key.
- [x] Define lookup.
- [x] Define invalidation.
- [x] Add cache tests.

## 44. Plan Cache Key

- [x] Include graph fingerprint.
- [x] Include Model Instance revision.
- [x] Include workload scope.
- [x] Include Kernel artifact digests.
- [x] Include specialization IDs.
- [x] Include Provider version.
- [x] Include Device compatibility.
- [x] Include policy versions.
- [x] Include memory-plan version.
- [x] Include adapter revision.
- [x] Include KV layout.

## 45. Cached Plan Revalidation

- [x] Recheck revocation.
- [x] Recheck trust.
- [x] Recheck qualification.
- [x] Recheck Provider readiness.
- [x] Recheck Device state.
- [x] Rebuild PreparedKernelId state.
- [x] Recheck memory feasibility.
- [x] Add restart tests.

## 46. Plan Persistence Boundary

- [x] Persist logical decisions only where useful.
- [x] Do not persist native handles.
- [x] Treat persisted Plan as recipe until re-prepared.
- [x] Add serialization safety tests.

## 47. Provider Prepared Graph

- [x] Define optional Provider prepared-segment capability.
- [x] Advertise capability.
- [x] Prepare logical segment.
- [x] Return opaque ProviderPreparedSegmentId.
- [x] Define lifecycle.
- [x] Add provider-segment tests.

## 48. Provider Prepared Graph Compatibility

- [x] Bind Provider.
- [x] Bind Device.
- [x] Bind Kernel generations.
- [x] Bind resource model.
- [x] Bind shape envelope.
- [x] Invalidate incompatible segment.
- [x] Add compatibility tests.

## 49. Graph Capture Fallback

- [x] Allow individual Kernel dispatch fallback.
- [x] Make fallback explicit.
- [x] Preserve semantics.
- [x] Add capture failure tests.

## 50. Plan Execution Path

- [x] Lookup compatible Plan.
- [x] Check guards.
- [x] Bind resources.
- [x] Acquire Plan lease.
- [x] Dispatch prepared segments/Kernels.
- [x] Observe completion.
- [x] Release lease.
- [x] Add execution-path tests.

## 51. Error Model

- [x] Add Plan lookup errors.
- [x] Add readiness errors.
- [x] Add build/preparation errors.
- [x] Add guard errors.
- [x] Add compatibility errors.
- [x] Add staleness/invalidation errors.
- [x] Add rebuild errors.
- [x] Add generation lifecycle errors.
- [x] Add segment errors.
- [x] Add hot-path rebuild denial.
- [x] Add internal Plan error.

## 52. Observability

- [x] Observe Plan build.
- [x] Observe node binding.
- [x] Observe segments.
- [x] Observe memory planning.
- [x] Observe ready state.
- [x] Observe cache hit/miss.
- [x] Observe guard failure.
- [x] Observe stale state.
- [x] Observe invalidation.
- [x] Observe rebuild request.
- [x] Observe generation promotion.
- [x] Observe retirement.
- [x] Redact native handles and runtime data.

## 53. Conformance

- [x] Prove Execution Graph remains semantic authority.
- [x] Prove Plan cannot change semantics.
- [x] Prove exact Kernel binding.
- [x] Prove no native-handle exposure.
- [x] Prove dynamic resource-slot safety.
- [x] Prove shape guards.
- [x] Prove stale != invalidated.
- [x] Prove invalidated Plan gets no new work.
- [x] Prove Kernel revocation invalidates Plan.
- [x] Prove Kernel promotion uses new Plan generation.
- [x] Prove Memory Manager authority.
- [x] Prove no full planning in decode hot path.
- [x] Prove atomic Plan replacement.
- [x] Prove in-flight generation lifetime.
- [x] Prove cache cannot bypass eligibility.
- [x] Prove Provider graph capture optionality.
- [x] Prove Provider handles remain opaque.
- [x] Prove adaptive feedback cannot mutate Plan in place.
- [x] Prove redaction.

## 54. Documentation

- [x] Document Execution Graph versus Prepared Plan.
- [x] Document Plan lifecycle.
- [x] Document Plan guards.
- [x] Document resource slots.
- [x] Document stale versus invalidated.
- [x] Document Plan families.
- [x] Document atomic replacement.
- [x] Document Provider prepared segments.
- [x] Document Plan cache/restart behavior.
- [x] Document hot-path objective.

## 55. Final Validation

- [x] Run OpenSpec validation.
- [x] Verify hot path avoids full Kernel resolution.
- [x] Verify Plan does not own native handles.
- [x] Verify Plan does not own Runtime tensor memory.
- [x] Verify invalidation is fail-safe.
- [x] Verify current active execution survives replacement safely.

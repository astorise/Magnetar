# Tasks

## 1. Qualification Domain

- [x] Define QualificationRecord.
- [x] Define qualification identity.
- [x] Define qualification profiles.
- [x] Define qualification status.
- [x] Define qualified-with-limitations.
- [x] Add qualification lifecycle tests.

## 2. Qualified Kernel Artifact

- [x] Define QualifiedKernelArtifact or equivalent metadata relationship.
- [x] Bind it to CompiledKernelArtifact digest.
- [x] Bind qualification profile.
- [x] Bind suite version.
- [x] Bind oracle identity.
- [x] Bind compatibility envelope.
- [x] Keep compiled bytes immutable.
- [x] Add artifact qualification tests.

## 3. Qualification Status

- [x] Add unqualified.
- [x] Add qualifying.
- [x] Add qualified.
- [x] Add qualified-with-limitations.
- [x] Add rejected.
- [x] Add revoked.
- [x] Add expired.
- [x] Validate legal transitions.

## 4. Qualification Profiles

- [x] Add versioned baseline correctness profile.
- [x] Add strict correctness placeholder.
- [x] Add deterministic profile.
- [x] Add approximate-math profile.
- [x] Add quantized profile.
- [x] Add fused profile.
- [x] Allow Provider-specific profiles.
- [x] Prevent stricter-profile inference from weaker evidence.

## 5. Reference Oracle

- [x] Define correctness oracle identity.
- [x] Support Reference CPU as default oracle.
- [x] Record Reference CPU version/fingerprint.
- [x] Support alternative oracle where required.
- [x] Fail if required oracle unavailable.
- [x] Add oracle tests.

## 6. Differential Qualification

- [x] Compare output shape.
- [x] Compare output dtype.
- [x] Compare numerical values.
- [x] Compare NaN/Inf behavior.
- [x] Compare errors.
- [x] Compare aliasing/mutation behavior.
- [x] Add mismatch reporting.
- [x] Add differential tests.

## 7. Tolerance Profiles

- [x] Add absolute tolerance.
- [x] Add relative tolerance.
- [x] Add optional ULP tolerance.
- [x] Bind tolerance to dtype.
- [x] Bind tolerance to accumulation dtype.
- [x] Add quantized tolerances.
- [x] Prevent hidden tolerance widening.
- [x] Add tolerance tests.

## 8. Qualification Matrix

- [x] Define input matrix contract.
- [x] Cover minimum shapes.
- [x] Cover maximum shapes where feasible.
- [x] Cover irregular shapes.
- [x] Cover alignment boundaries.
- [x] Cover batch boundaries.
- [x] Cover sequence boundaries.
- [x] Cover zero values.
- [x] Cover negative values.
- [x] Cover extreme values.
- [x] Cover masks.
- [x] Cover aliasing where supported.
- [x] Add matrix fingerprint.

## 9. Shape Qualification

- [x] Bind evidence to tested shape envelope.
- [x] Prevent automatic full-envelope extrapolation by default.
- [x] Add exact-shape qualification.
- [x] Add range qualification.
- [x] Add head-dimension qualification.
- [x] Add sequence-range qualification.
- [x] Add shape rejection tests.

## 10. Property-Based Qualification

- [x] Define reproducible random seed metadata.
- [x] Bound generated inputs by Kernel constraints.
- [x] Record test generator version where relevant.
- [x] Add property-based qualification fixtures.

## 11. Fused Qualification

- [x] Compare fused implementation against Operator group reference.
- [x] Validate output semantics.
- [x] Validate side effects.
- [x] Validate tolerance profile.
- [x] Add fused conformance tests.

## 12. Quantized Qualification

- [x] Validate scales.
- [x] Validate zero points.
- [x] Validate group size.
- [x] Validate packing.
- [x] Validate storage dtype.
- [x] Validate compute dtype.
- [x] Validate accumulation dtype.
- [x] Validate quantization tolerance.
- [x] Add quantized qualification fixtures.

## 13. Determinism Qualification

- [x] Re-run identical inputs.
- [x] Compare outputs according to determinism contract.
- [x] Record execution mode.
- [x] Record Device dependency.
- [x] Record atomic/reduction dependencies.
- [x] Reject false deterministic claims.
- [x] Add determinism tests.

## 14. Memory Contract Qualification

- [x] Validate input/output arity.
- [x] Validate shapes.
- [x] Validate byte sizes.
- [x] Validate alignment.
- [x] Validate workspace bounds.
- [x] Validate aliasing.
- [x] Validate in-place behavior.
- [x] Validate resource affinity.
- [x] Validate memory classes.
- [x] Add memory contract tests.

## 15. Failure Qualification

- [x] Test invalid input behavior.
- [x] Reject process crashes.
- [x] Reject ABI unwind.
- [x] Reject silent truncation.
- [x] Reject silent layout reinterpretation.
- [x] Reject unstructured corruption.
- [x] Add failure tests.

## 16. Trust Separation

- [x] Keep trust separate from qualification.
- [x] Keep compiler trust separate from artifact trust.
- [x] Keep AI provenance separate from trust.
- [x] Allow policy requiring trusted AND qualified.
- [x] Add state-combination tests.

## 17. Security Qualification

- [x] Validate artifact integrity.
- [x] Validate executable format.
- [x] Validate compiler isolation metadata.
- [x] Validate resource-limit compliance.
- [x] Add Provider-specific safety hooks.
- [x] Ensure security qualification does not authenticate provenance.

## 18. Benchmark Domain

- [x] Define BenchmarkRecord.
- [x] Define benchmark profile identity.
- [x] Define target Device metadata.
- [x] Define workload metadata.
- [x] Define warmup count.
- [x] Define measurement count.
- [x] Define synchronization policy.
- [x] Add benchmark record validation.

## 19. Benchmark Metrics

- [x] Support latency.
- [x] Support throughput.
- [x] Support tail latency.
- [x] Support workspace usage.
- [x] Support memory usage.
- [x] Reserve energy metrics.
- [x] Reserve compile/prepare cost metrics.
- [x] Keep metrics extensible.

## 20. Ranking

- [x] Rank only compatible candidates.
- [x] Rank only policy-approved candidates.
- [x] Rank only qualified candidates when required.
- [x] Rank only trusted candidates when required.
- [x] Use performance after correctness gates.
- [x] Add ranking tests.

## 21. Performance Policies

- [x] Define correctness-only acceptance.
- [x] Define must-beat-current policy.
- [x] Define within-regression-threshold policy.
- [x] Define memory-optimized policy placeholder.
- [x] Prevent incorrect kernel from winning ranking.

## 22. Benchmark Freshness

- [x] Bind results to Provider.
- [x] Bind results to Device architecture.
- [x] Bind results to driver/runtime class.
- [x] Bind results to workload profile.
- [x] Mark stale results.
- [x] Add stale-benchmark tests.

## 23. Kernel Cache Domain

- [x] Define Kernel cache abstraction.
- [x] Keep separate from Model cache.
- [x] Keep separate from Prefix Cache.
- [x] Keep separate from KV Cache.
- [x] Keep separate from Memory Manager residency.

## 24. Kernel Cache Identity

- [x] Include source digest.
- [x] Include compiled digest.
- [x] Include source/compiled format.
- [x] Include compiler identity/version.
- [x] Include compiler options fingerprint.
- [x] Include Provider version.
- [x] Include target architecture.
- [x] Include compatibility metadata.
- [x] Include Operator semantics.
- [x] Include dtype/layout.
- [x] Include shape specialization.
- [x] Add cache-key tests.

## 25. Qualification Cache

- [x] Cache qualification records.
- [x] Include suite version.
- [x] Include oracle version.
- [x] Include profile.
- [x] Include test matrix fingerprint.
- [x] Include tolerance profile.
- [x] Prevent stale evidence reuse.

## 26. Cache Trust

- [x] Ensure cache hit does not imply trust.
- [x] Ensure cache hit does not imply qualification.
- [x] Ensure cache hit does not imply compatibility.
- [x] Ensure cache hit does not imply active status.
- [x] Add cache trust tests.

## 27. Cache Integrity

- [x] Validate entry digest.
- [x] Detect corruption.
- [x] Quarantine/reject corrupt entries.
- [x] Keep unrelated entries usable.
- [x] Add corruption tests.

## 28. Cache Immutability

- [x] Make content-addressed artifacts immutable.
- [x] Use new digest for changed content.
- [x] Separate mutable operational metadata.
- [x] Add immutability tests.

## 29. Cache States

- [x] Add partial.
- [x] Add validating.
- [x] Add ready.
- [x] Add untrusted.
- [x] Add unqualified.
- [x] Add qualified.
- [x] Add rejected.
- [x] Add revoked.
- [x] Add corrupt.
- [x] Add evicting.
- [x] Add evicted.

## 30. Cache Eviction

- [x] Evict persistent artifact independently from Prepared state.
- [x] Protect pinned entries.
- [x] Prevent unsafe eviction.
- [x] Add eviction tests.

## 31. Cache Pinning

- [x] Pin active production Kernels where policy requires.
- [x] Pin rollback candidate where policy requires.
- [x] Pin offline deployment artifacts.
- [x] Add pinning tests.

## 32. Offline Cache

- [x] Reuse compatible cached artifacts offline.
- [x] Avoid recompilation when compatible artifact exists.
- [x] Reject incompatible cached target.
- [x] Add offline tests.

## 33. Candidate Lifecycle

- [x] Define qualified candidate.
- [x] Define candidate state.
- [x] Reserve canary state.
- [x] Define active state.
- [x] Define retiring state.
- [x] Define retired state.
- [x] Define revoked state.
- [x] Validate transitions.

## 34. Promotion

- [x] Make promotion explicit.
- [x] Validate eligibility before promotion.
- [x] Prepare candidate before promotion.
- [x] Prevent partial Registry update.
- [x] Add promotion tests.

## 35. Atomic Registry Promotion

- [x] Define publication generation/epoch.
- [x] Ensure new dispatch sees consistent Registry state.
- [x] Keep previous state intact on failure.
- [x] Add atomicity tests.

## 36. Prepared Kernel Generations

- [x] Add prepared generation number.
- [x] Associate with logical KernelId.
- [x] Associate with artifact digest.
- [x] Associate with Provider/Device.
- [x] Track lifecycle.
- [x] Add generation tests.

## 37. In-Flight Stability

- [x] Pin acquired generation for invocation.
- [x] Prevent mid-flight generation replacement.
- [x] Add concurrent promotion tests.

## 38. Reference Tracking

- [x] Define active reference tracking.
- [x] Support refcount/lease/epoch implementation.
- [x] Prevent destroy while referenced.
- [x] Add lifetime tests.

## 39. Safe Retirement

- [x] Stop routing new work to retiring generation.
- [x] Wait for quiescence.
- [x] Destroy through Provider.
- [x] Add retirement tests.

## 40. Rollback

- [x] Keep known-good rollback candidate.
- [x] Define rollback eligibility.
- [x] Support manual rollback.
- [x] Reserve automatic rollback.
- [x] Stop new work to bad generation.
- [x] Promote old known-good generation.
- [x] Add rollback tests.

## 41. Rollback Window

- [x] Allow retention period for previous generation.
- [x] Prevent immediate destruction when rollback required.
- [x] Add rollback-window tests.

## 42. Revocation

- [x] Revoke artifact.
- [x] Revoke qualification evidence.
- [x] Stop new dispatches.
- [x] Define in-flight revocation policy.
- [x] Add revocation reason.
- [x] Add revocation tests.

## 43. Qualification Expiration

- [x] Expire on incompatible Provider upgrade.
- [x] Expire on incompatible compiler/toolchain change.
- [x] Expire on qualification suite version change where required.
- [x] Expire on Operator semantic change.
- [x] Add expiration tests.

## 44. Hot Swap

- [x] Implement Kernel-level hot swap semantics.
- [x] Keep Provider loaded.
- [x] Do not require `.so` reload.
- [x] Preserve active Device state.
- [x] Add hot-swap tests.

## 45. Model Instance Policy

- [x] Support dynamic Kernel selection.
- [x] Support optional pinned Kernel set.
- [x] Record artifact digest for reproducibility.
- [x] Record qualification profile.
- [x] Add Model Instance tests.

## 46. Session Boundary

- [x] Prevent Session from owning native Kernel state.
- [x] Allow inherited Model Instance Kernel policy.
- [x] Add Session boundary tests.

## 47. Continuous Batching

- [x] Preserve Kernel generation for in-flight batch work.
- [x] Allow new batch work to use new generation.
- [x] Add batching/hot-swap tests.

## 48. Resource Affinity

- [x] Validate candidate Device affinity before promotion.
- [x] Reject incompatible prepared target.
- [x] Preserve tensor residency constraints.
- [x] Add affinity tests.

## 49. Memory Boundary

- [x] Keep Kernel cache separate from Runtime tensor memory.
- [x] Keep executable memory separate from tensor memory.
- [x] Surface optional preparation pressure.
- [x] Preserve Memory Manager tensor authority.
- [x] Add boundary tests.

## 50. Qualification Service Boundary

- [x] Support local qualification.
- [x] Support CI-produced qualification evidence.
- [x] Reserve external qualification services.
- [x] Keep Runtime generator/qualification-service independent.
- [x] Add service-neutral metadata tests.

## 51. Generator Independence

- [x] Support AI-generated Kernels.
- [x] Support human Kernels.
- [x] Support vendor Kernels.
- [x] Support CI-generated Kernels.
- [x] Avoid KernelEvolve-specific Runtime dependencies.

## 52. Failure Atomicity

- [x] Failed qualification leaves active Kernel intact.
- [x] Failed benchmark leaves active Kernel intact.
- [x] Failed preparation leaves active Kernel intact.
- [x] Failed promotion leaves active Kernel intact.
- [x] Add failure atomicity tests.

## 53. Error Model

- [x] Add qualification errors.
- [x] Add benchmark errors.
- [x] Add cache errors.
- [x] Add promotion errors.
- [x] Add hot-swap errors.
- [x] Add retirement errors.
- [x] Add rollback errors.
- [x] Add revocation errors.
- [x] Add internal generated-kernel management error.

## 54. Observability

- [x] Observe qualification start/end.
- [x] Observe differential mismatch.
- [x] Observe qualification rejection.
- [x] Observe qualification revocation.
- [x] Observe benchmark start/end.
- [x] Observe benchmark regression.
- [x] Observe cache hit/miss.
- [x] Observe corruption.
- [x] Observe candidate creation.
- [x] Observe promotion.
- [x] Observe retirement.
- [x] Observe rollback.
- [x] Record generation.
- [x] Redact raw source/binaries/tensors/native handles.

## 55. Conformance

- [x] Validate compiled != qualified.
- [x] Validate qualified != trusted.
- [x] Validate cache hit != eligible.
- [x] Validate Reference CPU oracle.
- [x] Validate mismatch rejection.
- [x] Validate explicit tolerances.
- [x] Validate shape qualification bounds.
- [x] Validate fused equivalence.
- [x] Validate deterministic claims.
- [x] Validate ranking order.
- [x] Validate cache compatibility.
- [x] Validate atomic promotion.
- [x] Validate in-flight old generation.
- [x] Validate safe retirement.
- [x] Validate rollback.
- [x] Validate revocation.
- [x] Validate Provider remains loaded.

## 56. Documentation

- [x] Document qualification pipeline.
- [x] Document oracle model.
- [x] Document trust/qualification distinction.
- [x] Document benchmark ranking.
- [x] Document Kernel cache.
- [x] Document promotion.
- [x] Document hot swap.
- [x] Document rollback.
- [x] Document revocation.
- [x] Document Provider lifetime independence.

## 57. Final Validation

- [x] Run OpenSpec validation.
- [x] Verify generated kernel never becomes active from compilation alone.
- [x] Verify performance never overrides correctness.
- [x] Verify cache never grants implicit trust.
- [x] Verify hot swap never unloads Provider.
- [x] Verify active kernel survives candidate failure.

# Tasks

## 1. Autotuning Domain

- [x] Define KernelAutotuningPlan.
- [x] Define KernelAutotuningSession.
- [x] Define KernelAutotuningRecord.
- [x] Define KernelAutotuningPolicy.
- [x] Define tuning policy version.
- [x] Add lifecycle tests.

## 2. Specialization Template

- [x] Define KernelSpecializationTemplate.
- [x] Bind template to Kernel Artifact.
- [x] Add template version.
- [x] Add specialization axes.
- [x] Add constraints.
- [x] Add qualification coverage.
- [x] Validate boundedness.

## 3. Specialization Axes

- [x] Define specialization axis identity.
- [x] Support enumerated values.
- [x] Support bounded integer range.
- [x] Support bounded powers-of-two.
- [x] Support symbolic enumerations.
- [x] Support Provider-defined bounded set.
- [x] Reject arbitrary string domain.
- [x] Reject unbounded range.

## 4. Axis Constraints

- [x] Define safe declarative constraints.
- [x] Prevent arbitrary scripts.
- [x] Validate cross-axis combinations.
- [x] Detect impossible specialization templates.
- [x] Add constraint tests.

## 5. Specialization Instance

- [x] Define KernelSpecializationInstance.
- [x] Bind to template.
- [x] Store axis/value assignments.
- [x] Define deterministic identity.
- [x] Add stable serialization/fingerprint.
- [x] Add instance tests.

## 6. Semantic Boundary

- [x] Ensure specialization cannot change Operator ID.
- [x] Ensure specialization cannot change Operator version.
- [x] Ensure Runtime-visible semantics stay stable.
- [x] Require distinct Kernel candidate for semantic differences.
- [x] Add boundary tests.

## 7. Source Specialization

- [x] Allow specialization before Provider compilation.
- [x] Reuse Provider Kernel Compilation Capability.
- [x] Keep compilation cold-path.
- [x] Include specialization in compiled artifact identity.
- [x] Add source specialization tests.

## 8. Precompiled Specialization

- [x] Support bundles containing multiple precompiled specializations.
- [x] Match specialization to workload.
- [x] Avoid unnecessary recompilation.
- [x] Add precompiled variant fixture.

## 9. Preparation-Time Specialization

- [x] Support Provider preparation specialization.
- [x] Keep metadata explicit.
- [x] Preserve PreparedKernel opacity.
- [x] Add preparation specialization tests.

## 10. Provider Execution Parameters

- [x] Define bounded Provider execution parameter metadata.
- [x] Require explicit parameter domain.
- [x] Preserve Kernel semantics.
- [x] Add Provider parameter tests.

## 11. Autotuning Plan

- [x] Add candidate set.
- [x] Add specialization space.
- [x] Add workload profile.
- [x] Add benchmark profile.
- [x] Add objective.
- [x] Add resource budget.
- [x] Add qualification policy.
- [x] Add fallback.
- [x] Validate plan.

## 12. Candidate Enumeration

- [x] Compute theoretical candidate bound.
- [x] Enforce maximum evaluated candidates.
- [x] Support deterministic pruning.
- [x] Support Provider hints.
- [x] Prevent domain expansion.
- [x] Add candidate bound tests.

## 13. Search Strategies

- [x] Support exhaustive bounded search.
- [x] Support ordered candidates.
- [x] Support deterministic subset selection.
- [x] Reserve bounded random sampling.
- [x] Keep strategy extensible.
- [x] Add strategy tests.

## 14. Qualification Coverage

- [x] Add ExactInstance.
- [x] Add EnumeratedInstances.
- [x] Add DeclaredEnvelope.
- [x] Add RequiresPerInstanceQualification.
- [x] Validate coverage.
- [x] Add qualification inheritance tests.

## 15. No Implicit Qualification

- [x] Prevent template-level implicit qualification.
- [x] Require explicit coverage evidence.
- [x] Require per-instance qualification where necessary.
- [x] Add unsafe inheritance regression tests.

## 16. Trust Integration

- [x] Preserve source/compiled provenance.
- [x] Re-evaluate specialized compiled artifact trust as required.
- [x] Do not infer trust from successful tuning.
- [x] Add trust tests.

## 17. Autotuning Session States

- [x] Add created.
- [x] Add planning.
- [x] Add preparing.
- [x] Add warming-up.
- [x] Add benchmarking.
- [x] Add evaluating.
- [x] Add completed.
- [x] Add cancelled.
- [x] Add timed-out.
- [x] Add failed.
- [x] Validate transitions.

## 18. Allowed Trigger Points

- [x] Support Model Instance loading.
- [x] Support warmup.
- [x] Support explicit management request.
- [x] Support deployment preparation.
- [x] Reserve idle/background tuning.
- [x] Prevent synchronous decode trigger.

## 19. Hot-Path Boundary

- [x] Add hot-path tuning denial.
- [x] Prevent benchmark start from decode.
- [x] Prevent compilation start from decode.
- [x] Use known-good fallback.
- [x] Add decode boundary tests.

## 20. Model Warmup

- [x] Integrate optional tuning phase.
- [x] Support mandatory tuning policy.
- [x] Support optional tuning policy.
- [x] Expose warming state.
- [x] Add readiness tests.

## 21. Lazy Autotuning

- [x] Support background tuning.
- [x] Preserve active Kernel.
- [x] Bound resource use.
- [x] Publish results atomically.
- [x] Add lazy tuning tests.

## 22. Tuning Fixtures

- [x] Support synthetic fixtures.
- [x] Support deterministic generated fixtures.
- [x] Support authorized benchmark datasets.
- [x] Avoid raw production prompts.
- [x] Add privacy tests.

## 23. Workload Buckets

- [x] Add Operator identity.
- [x] Add shape bucket.
- [x] Add batch bucket.
- [x] Add sequence bucket.
- [x] Add prefill/decode phase.
- [x] Add dtype.
- [x] Add layout.
- [x] Add quantization.
- [x] Add Provider/Device context.
- [x] Add bucket compatibility tests.

## 24. Continuous Batching Context

- [x] Add active sequence count.
- [x] Add total token count.
- [x] Add raggedness.
- [x] Add KV cache mode.
- [x] Ensure tuning does not disturb live batch.
- [x] Add batching tests.

## 25. Benchmark Profile

- [x] Add warmup iteration count.
- [x] Add measurement count.
- [x] Add synchronization policy.
- [x] Add timeout.
- [x] Add metric.
- [x] Add outlier handling metadata.
- [x] Add profile validation.

## 26. Primary Objectives

- [x] Support latency.
- [x] Support throughput.
- [x] Support memory.
- [x] Reserve energy.
- [x] Add secondary objectives.
- [x] Preserve hard eligibility filters.

## 27. Autotuning Record

- [x] Define plan fingerprint.
- [x] Record candidates.
- [x] Record specialization identities.
- [x] Record target.
- [x] Record workload bucket.
- [x] Record benchmark profile.
- [x] Record measurements.
- [x] Record winner.
- [x] Record qualification references.
- [x] Record policy version.
- [x] Record freshness metadata.

## 28. Autotuning Cache

- [x] Define tuning cache.
- [x] Keep distinct from Kernel Artifact Cache.
- [x] Define tuning cache key.
- [x] Add candidate-set fingerprint.
- [x] Add lookup.
- [x] Add invalidation.
- [x] Add cache tests.

## 29. Tuning Cache Context

- [x] Include Operator semantics.
- [x] Include artifact digests.
- [x] Include template version.
- [x] Include Provider version.
- [x] Include Device architecture/features.
- [x] Include driver/runtime compatibility.
- [x] Include dtype/layout.
- [x] Include workload bucket.
- [x] Include objective.
- [x] Include policy version.

## 30. Cache Eligibility Revalidation

- [x] Recheck revocation.
- [x] Recheck trust.
- [x] Recheck qualification.
- [x] Recheck Provider readiness.
- [x] Recheck Device availability.
- [x] Recheck memory feasibility.
- [x] Recheck Prepared Kernel readiness.
- [x] Add stale candidate tests.

## 31. Freshness

- [x] Invalidate on artifact change.
- [x] Invalidate on incompatible Provider change.
- [x] Invalidate on incompatible driver/runtime change.
- [x] Invalidate on template change.
- [x] Invalidate on benchmark profile change.
- [x] Invalidate on policy change.
- [x] Add freshness tests.

## 32. Resource Budgets

- [x] Add max-candidate budget.
- [x] Add compilation-job budget.
- [x] Add preparation budget.
- [x] Add benchmark invocation budget.
- [x] Add wall-clock deadline.
- [x] Add host memory budget.
- [x] Add Device memory/workspace budget.
- [x] Add budget tests.

## 33. Inference Resource Protection

- [x] Lower tuning priority where configured.
- [x] Reject tuning under critical pressure.
- [x] Allow cancellation under pressure.
- [x] Support dedicated tuning Device.
- [x] Add pressure tests.

## 34. Memory Manager Integration

- [x] Query tuning workspace feasibility.
- [x] Reject production-infeasible specialization.
- [x] Release temporary tuning allocations.
- [x] Prevent Tensor Resource leaks.
- [x] Add memory tests.

## 35. Candidate Preparation

- [x] Use Provider.prepare normally.
- [x] Preserve PreparedKernel ownership.
- [x] Track temporary tuning preparation.
- [x] Safely retire tuning-only Prepared state.
- [x] Add preparation tests.

## 36. Candidate Failure Isolation

- [x] Isolate compile failure.
- [x] Isolate prepare failure.
- [x] Isolate qualification failure.
- [x] Isolate benchmark failure.
- [x] Continue remaining candidates where policy permits.
- [x] Add failure tests.

## 37. Known-Good Preservation

- [x] Keep active known-good Kernel available.
- [x] Prevent tuning failure from removing it.
- [x] Add all-candidates-fail test.

## 38. Selection Integration

- [x] Feed tuning evidence to Kernel Selection Policy.
- [x] Keep selection authoritative.
- [x] Keep hysteresis.
- [x] Keep promotion explicit.
- [x] Add integration tests.

## 39. Reproducible Mode

- [x] Allow tuning disablement.
- [x] Allow pinned AutotuningRecord.
- [x] Prevent live retuning from changing pinned Model Instance.
- [x] Add reproducibility tests.

## 40. Provider Defaults And Hints

- [x] Support recommended defaults.
- [x] Support candidate ordering hints.
- [x] Support known-bad combinations.
- [x] Prevent hints expanding domain.
- [x] Prevent hints overriding policy.
- [x] Add hint tests.

## 41. Provider Native Autotuning Boundary

- [x] Define optional bounded Provider autotuning semantics.
- [x] Require declared candidate domain.
- [x] Require cold/warm path execution.
- [x] Preserve Runtime policy.
- [x] Prevent arbitrary code generation.
- [x] Add Provider-boundary tests.

## 42. Specialized Artifact Cache

- [x] Store specialized compiled artifacts by content digest.
- [x] Associate specialization identity.
- [x] Preserve template/source lineage.
- [x] Add cache dedup tests.

## 43. Cross-Device Reuse

- [x] Validate architecture compatibility.
- [x] Validate Device features.
- [x] Validate Provider version.
- [x] Validate driver/runtime class.
- [x] Add incompatible-target tests.

## 44. Offline Deployment

- [x] Support pre-specialized artifacts.
- [x] Support precomputed tuning records.
- [x] Support pinned selection.
- [x] Require no live autotuning.
- [x] Add offline tests.

## 45. Security Boundaries

- [x] Tune accepted artifacts only.
- [x] Exclude quarantined artifacts.
- [x] Reject arbitrary compiler flags.
- [x] Reject arbitrary source mutation.
- [x] Reject arbitrary network authority.
- [x] Add security tests.

## 46. Error Model

- [x] Add autotuning policy errors.
- [x] Add template errors.
- [x] Add axis/domain errors.
- [x] Add specialization errors.
- [x] Add qualification coverage errors.
- [x] Add budget/admission errors.
- [x] Add benchmark errors.
- [x] Add cache/freshness errors.
- [x] Add hot-path denial.
- [x] Add memory/pressure errors.
- [x] Add internal autotuning error.

## 47. Observability

- [x] Observe planning.
- [x] Observe candidate enumeration.
- [x] Observe pruning.
- [x] Observe specialization compilation.
- [x] Observe preparation.
- [x] Observe benchmark.
- [x] Observe candidate failure.
- [x] Observe measurements.
- [x] Observe winner.
- [x] Observe completion.
- [x] Observe cache hit/stale.
- [x] Observe cancellation/timeout.
- [x] Redact sensitive payloads.

## 48. Conformance

- [x] Prove candidate domain bounded.
- [x] Prove no arbitrary source generation.
- [x] Prove no arbitrary compiler args.
- [x] Prove no tuning in decode hot path.
- [x] Prove accepted-artifact requirement.
- [x] Prove qualification coverage.
- [x] Prove no implicit qualification inheritance.
- [x] Prove tuning != qualification.
- [x] Prove context-sensitive tuning cache.
- [x] Prove freshness invalidation.
- [x] Prove Memory Manager authority.
- [x] Prove active Kernel survives tuning failure.
- [x] Prove tuning winner cannot bypass selection.
- [x] Prove Provider hints non-authoritative.
- [x] Prove prefill/decode independence.
- [x] Prove reproducible mode.
- [x] Prove no native handle persistence.
- [x] Prove observability redaction.

## 49. Documentation

- [x] Document optimization vs autotuning.
- [x] Document specialization template.
- [x] Document axis domains.
- [x] Document qualification coverage.
- [x] Document tuning lifecycle.
- [x] Document tuning cache.
- [x] Document hot-path prohibition.
- [x] Document offline/precomputed tuning.
- [x] Document Provider autotuning boundary.

## 50. Final Validation

- [x] Run OpenSpec validation.
- [x] Verify Runtime Autotuning is bounded.
- [x] Verify Runtime does not become code generator.
- [x] Verify specialization cannot change Operator semantics.
- [x] Verify qualification remains authoritative.
- [x] Verify active inference is protected from tuning workload.

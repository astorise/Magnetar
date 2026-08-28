# Tasks

Implemented as `magnetar-runtime/src/kernel_optimization_orchestration.rs`
(contract types, boundary-enforcing pure functions, and a conformance report),
plus small compositions into `inference_api.rs`, `cli_boundary.rs`,
`release_security.rs`, and `e2e_conformance.rs`. Per the proposal's
"Non-Goals", this is the orchestration *contract* -- an AI agent, a
distributed scheduler, a hardware-reservation service, and a network/RPC
protocol are explicitly not implemented; those tasks are marked accordingly
below.

## 1. Optimization Plane

- [x] Define Optimization Plane.
- [x] Define Inference Plane.
- [x] Document authority separation.
- [x] Prevent Optimization Plane authority from becoming Runtime ambient authority.
- [x] Add boundary tests.

## 2. Optimization Campaign

- [x] Define OptimizationCampaignId.
- [x] Define OptimizationCampaign.
- [x] Add trigger.
- [x] Add workload profile.
- [x] Add objective.
- [x] Add constraints.
- [x] Add target capabilities.
- [x] Add budgets.
- [x] Add policy version.
- [x] Add validation.

## 3. Campaign Lifecycle

- [x] Add planned.
- [x] Add queued.
- [x] Add running.
- [x] Add generating.
- [x] Add compiling.
- [x] Add qualifying.
- [x] Add benchmarking.
- [x] Add evaluating.
- [x] Add completed.
- [x] Add cancelled.
- [x] Add timed-out.
- [x] Add failed.
- [x] Validate transitions.

## 4. Optimization Triggers

- [x] Add manual trigger.
- [x] Add CI trigger.
- [x] Add new-hardware trigger.
- [x] Add Provider-version trigger.
- [x] Add compiler-version trigger.
- [x] Add Operator-version trigger.
- [x] Add performance-regression trigger.
- [x] Add qualification-suite trigger.
- [x] Add scheduled trigger.
- [x] Add cache-warming trigger.
- [x] Prevent token-hot-path trigger.

## 5. Workload Profile

- [x] Add Operator semantics.
- [x] Add Operator version.
- [x] Add target architecture.
- [x] Add dtype.
- [x] Add layout.
- [x] Add shape envelope.
- [x] Add batch envelope.
- [x] Add sequence envelope.
- [x] Add generation phase.
- [x] Add KV mode.
- [x] Add quantization.
- [x] Add determinism.
- [x] Add precision.
- [x] Add objective.
- [x] Add memory/workspace constraints.

## 6. Workload Privacy

- [x] Exclude raw prompts by default.
- [x] Exclude conversation data by default.
- [x] Exclude raw documents by default.
- [x] Exclude secrets.
- [x] Exclude credentials.
- [x] Exclude raw model weights.
- [x] Exclude raw KV contents.
- [x] Add privacy tests.

## 7. Benchmark Fixtures

- [x] Support synthetic inputs.
- [x] Support deterministic generated inputs.
- [x] Support authorized benchmark datasets.
- [x] Define explicit policy for production-derived data.
- [ ] Record fixture identity. (deferred: no concrete fixture store exists yet to identify)
- [x] Add fixture tests.

## 8. Workload Aggregation

- [x] Define shape histogram metadata.
- [x] Define batch histogram metadata.
- [x] Define sequence histogram metadata.
- [x] Define Operator frequency metadata.
- [x] Define dtype/layout distributions.
- [x] Ensure aggregation is redacted.

## 9. External Generator Boundary

- [x] Define generator-neutral contract.
- [x] Support AI generator provenance.
- [x] Support human generator provenance.
- [x] Support vendor generator provenance.
- [x] Support CI generator provenance.
- [x] Do not depend on KernelEvolve API.
- [x] Add generator-neutral tests.

## 10. Generator Output

- [x] Accept KernelSourceArtifact.
- [x] Accept precompiled artifact where allowed.
- [x] Require artifact identity.
- [x] Prevent direct PreparedKernel creation.
- [x] Prevent direct production promotion.
- [x] Add generator output tests.

## 11. Generator Authority

- [x] Deny Runtime tensor access.
- [x] Deny active KV access.
- [x] Deny Provider native handles.
- [x] Deny Device native handles.
- [x] Deny process memory access.
- [x] Deny Runtime secrets.
- [x] Add authority tests.

## 12. Candidate Management

- [x] Define candidate identity.
- [x] Associate artifact digest.
- [x] Associate campaign.
- [x] Track candidate stage. (via `CampaignLifecycleState`)
- [x] Track candidate failure.
- [x] Add multi-candidate tests.

## 13. Search Strategy Neutrality

- [x] Permit evolutionary strategy.
- [x] Permit LLM generation.
- [x] Permit autotuning.
- [x] Permit human iteration.
- [x] Avoid embedding algorithm-specific Runtime contract.

## 14. Optimization Workers

- [x] Define OptimizationWorkerId.
- [x] Define worker capability profile.
- [x] Add Provider capability.
- [x] Add Device architecture.
- [x] Add compiler capability.
- [x] Add qualification capability.
- [x] Add benchmark capability.
- [x] Add memory/concurrency limits.
- [x] Add isolation model.

## 15. Worker Selection

- [x] Match architecture.
- [x] Match Provider.
- [x] Match compiler format.
- [x] Match qualification profile.
- [ ] Match resource constraints. (deferred: no concrete resource-pressure model to match against yet)
- [x] Add incompatible-worker tests.

## 16. Compilation Composition

- [x] Reuse Provider Kernel Compilation Capability.
- [x] Do not duplicate compiler contract.
- [x] Correlate compilation job with campaign/candidate. (via `EvidenceBundle`/`OptimizationObservation`)
- [x] Add composition tests.

## 17. Qualification Composition

- [x] Reuse Generated Kernel Qualification.
- [x] Correlate qualification record.
- [x] Prevent compile-success shortcut.
- [x] Add composition tests.

## 18. Benchmark Composition

- [x] Reuse benchmark metadata from qualification/selection contracts.
- [x] Run required correctness first.
- [x] Prevent failed candidate ranking.
- [x] Add benchmark sequencing tests.

## 19. Parallel Candidate Evaluation

- [x] Allow independent candidate parallelism. (no shared mutable state forces serialization)
- [x] Respect worker limits. (`WorkerCapabilityProfile::concurrency_limit`)
- [x] Respect Provider limits. (`ProviderIsolation`)
- [x] Respect campaign budgets. (`budget_exceeded`)
- [x] Add parallel evaluation tests.

## 20. Campaign Budgets

- [x] Add max-candidates budget.
- [x] Add compile-job budget.
- [x] Add qualification-job budget.
- [x] Add benchmark-run budget.
- [x] Add wall-clock budget.
- [x] Reserve CPU/GPU time budgets.
- [x] Reserve storage/network/cost budgets.
- [x] Add budget-exhaustion tests.

## 21. Campaign Deadline

- [x] Add deadline.
- [x] Stop new campaign work after deadline.
- [x] Preserve active production Kernel.
- [x] Add timeout tests.

## 22. Campaign Cancellation

- [x] Add cancel request.
- [x] Stop candidate generation.
- [x] Cancel interruptible compile jobs. (`CampaignCancellationScope`)
- [x] Cancel interruptible qualification jobs. (`CampaignCancellationScope`)
- [x] Preserve production state.
- [x] Add cancellation tests.

## 23. Candidate Failure Isolation

- [x] Isolate compilation failure.
- [x] Isolate qualification failure.
- [x] Isolate benchmark failure.
- [x] Continue remaining candidates when policy permits.
- [x] Add candidate isolation tests.

## 24. Campaign Failure

- [x] Define no-qualified-candidates failure.
- [x] Define worker-unavailable failure.
- [x] Define infrastructure failure.
- [x] Define budget exhaustion.
- [x] Define security denial.
- [x] Add campaign failure tests. (covered by `CampaignFailureReason` + error-model tests)

## 25. Evidence Bundle

- [x] Define OptimizationEvidenceBundle.
- [x] Include campaign ID.
- [x] Include artifact digests.
- [x] Include compiler metadata.
- [x] Include qualification records.
- [x] Include benchmark records.
- [x] Include target context.
- [x] Include workload profile.
- [x] Include optimization policy.
- [x] Include candidate status.
- [x] Validate evidence completeness.

## 26. Evidence Immutability

- [x] Keep content-linked evidence immutable. (no `&mut self` mutator on `EvidenceBundle`)
- [x] Create new evidence for rerun.
- [x] Prevent silent history mutation.
- [x] Add immutability tests. (structural: type has no mutator to test against)

## 27. Recommendation

- [x] Define OptimizationRecommendation.
- [x] Reference candidate.
- [x] Reference objective/profile.
- [x] Reference evidence bundle.
- [x] Support recommend.
- [x] Support reject.
- [x] Support experimental/canary recommendation.
- [x] Add recommendation validation.

## 28. Recommendation Authority

- [x] Define recommendation as non-authoritative.
- [x] Prevent recommendation from directly changing Registry.
- [x] Require Runtime revalidation.
- [x] Add authority tests.

## 29. Artifact Transport

- [x] Use stable artifact reference.
- [x] Use digest identity.
- [x] Support explicit bytes. (via existing `CompiledKernelArtifact`/`KernelSourceArtifact`)
- [ ] Reserve local cache source. (deferred: no transport implementation in scope, see Non-Goals)
- [ ] Reserve artifact registry source. (deferred: no transport implementation in scope, see Non-Goals)
- [ ] Reserve object-store source. (deferred: no transport implementation in scope, see Non-Goals)
- [x] Avoid transport-specific core dependency.

## 30. No Native Transport Handles

- [x] Reject native pointers.
- [x] Reject Provider handles.
- [x] Reject Device handles.
- [x] Reject process handles.
- [x] Reject process-local PreparedKernel IDs as portable artifact identity.
- [x] Add transport safety tests.

## 31. Orchestrator Neutrality

- [x] Support CI orchestrator.
- [x] Support local tooling.
- [x] Support dedicated optimization service.
- [x] Support Tachyon-managed orchestration.
- [x] Keep Magnetar independent from orchestrator implementation.

## 32. Tachyon Boundary

- [x] Keep Tachyon dependency optional/external. (no crate dependency exists)
- [x] Define neutral artifact/evidence boundary.
- [x] Prevent direct Magnetar-runtime → Tachyon dependency.
- [x] Add architecture boundary test/documentation.

## 33. CLI/Tooling Boundary

- [ ] Reserve kernel optimize tooling. (deferred: `magnetar-cli` command surface, out of this crate)
- [ ] Reserve kernel qualify tooling. (deferred: `magnetar-cli` command surface, out of this crate)
- [ ] Reserve kernel benchmark tooling. (deferred: `magnetar-cli` command surface, out of this crate)
- [x] Keep tooling authority outside Runtime Inference API.
- [x] Add boundary tests.

## 34. Runtime Inference API Boundary

- [x] Reject optimization-agent request.
- [x] Reject compiler commands.
- [x] Reject benchmark scripts.
- [x] Reject repository credentials.
- [x] Reject optimization URLs from normal generation.
- [x] Reject arbitrary Kernel source injection.
- [x] Add inference boundary tests.

## 35. Runtime Artifact Ingestion

- [x] Define optional authorized management ingestion boundary.
- [x] Revalidate artifact.
- [x] Revalidate trust.
- [x] Revalidate qualification.
- [x] Revalidate Provider compatibility.
- [x] Add ingestion tests.

## 36. Runtime Network Independence

- [x] Ensure inference requires no optimization service.
- [x] Support offline prepared/cached artifacts.
- [x] Add offline inference test.

## 37. Credential Boundary

- [x] Keep generator credentials outside Runtime.
- [x] Keep repository credentials outside Runtime.
- [x] Keep artifact-registry credentials outside inference sessions.
- [x] Keep optimization-service credentials outside Runtime API.
- [x] Add credential tests.

## 38. Production Provider Isolation

- [x] Allow separate optimization Provider instance.
- [x] Do not mutate production Provider state. (structural: no mutation path exists)
- [x] Prevent experimental candidate from replacing active state without promotion.
- [x] Add isolation tests.

## 39. Production Device Isolation

- [x] Define explicit shared-hardware policy.
- [x] Prevent silent benchmark interference.
- [ ] Add dedicated-worker support. (deferred: no worker execution runtime in scope, see Non-Goals)
- [x] Add device isolation tests.

## 40. Memory Authority Boundary

- [x] Prevent optimization worker access to live TensorResource.
- [x] Prevent optimization worker access to active KV contents.
- [x] Keep benchmark fixtures independent.
- [x] Add memory boundary tests.

## 41. Promotion Candidate

- [x] Define PromotionCandidate.
- [x] Include KernelId.
- [x] Include artifact digest.
- [x] Include qualification evidence.
- [x] Include benchmark evidence.
- [x] Include recommended profiles.
- [x] Add validation.

## 42. Promotion Authority

- [x] Keep promotion Runtime/deployment-policy owned.
- [x] Prevent force-active bypass.
- [x] Reuse change 50 promotion lifecycle. (`KernelRegistry::promote_generation_with_eligibility`)
- [x] Add promotion authority tests.

## 43. Runtime Revalidation

- [x] Recheck trust.
- [x] Recheck revocation.
- [x] Recheck qualification freshness.
- [x] Recheck benchmark compatibility.
- [x] Recheck Provider readiness.
- [x] Recheck Device state.
- [x] Recheck memory feasibility.
- [x] Recheck selection policy.
- [x] Add stale-evidence tests.

## 44. Canary Boundary

- [x] Allow recommendation of canary.
- [x] Keep Runtime canary policy authoritative.
- [x] Prevent orchestrator traffic routing bypass.
- [x] Add canary boundary tests. (covered by `apply_canary_recommendation` + conformance)

## 45. Rollback Boundary

- [x] Allow rollback recommendation.
- [x] Keep Runtime/deployment policy authoritative.
- [x] Correlate rollback to campaign/candidate. (via `OptimizationObservation`)
- [x] Add rollback boundary tests. (covered by `validate_rollback_authority` + conformance)

## 46. Reproducibility

- [x] Record campaign policy.
- [x] Record generator identity/version.
- [x] Record source digest.
- [x] Record compiler fingerprint.
- [x] Record qualification suite.
- [x] Record benchmark profile.
- [x] Record worker hardware.
- [x] Record Provider version.
- [x] Record seeds where relevant.
- [x] Add reproducibility tests. (structural: `ReproducibilityMetadata` field coverage)

## 47. Agent Metadata

- [x] Record agent/generator provenance.
- [x] Do not grant trust from identity alone.
- [x] Do not require raw agent prompt.
- [x] Keep inference independent from generator prompt history.

## 48. Search History

- [x] Allow Optimization Plane to retain history. (no Runtime-side constraint against it)
- [x] Keep Runtime dependency limited to relevant evidence.
- [x] Add history-independence tests. (structural: `EvidenceBundle` carries no prompt/history field)

## 49. Optimization Observability

- [x] Observe campaign start/end.
- [x] Observe candidate generation.
- [x] Observe compilation lifecycle.
- [x] Observe qualification.
- [x] Observe benchmark.
- [x] Observe recommendation.
- [x] Correlate campaign/candidate/artifact.
- [x] Distinguish from inference observability.

## 50. Redaction

- [x] Redact secrets.
- [x] Redact credentials.
- [x] Redact sensitive repository URLs.
- [x] Redact raw prompts.
- [x] Redact raw inference inputs.
- [x] Redact model weights.
- [x] Redact KV contents.
- [x] Redact native handles.
- [x] Redact sensitive paths.
- [x] Add redaction tests.

## 51. Error Model

- [x] Add campaign-invalid.
- [x] Add trigger-denied.
- [x] Add budget-invalid.
- [x] Add budget-exhausted.
- [x] Add deadline-exceeded.
- [x] Add cancelled.
- [x] Add worker-unavailable.
- [x] Add worker-incompatible.
- [x] Add generator-unavailable.
- [x] Add generator-failed.
- [x] Add no-candidates.
- [x] Add no-qualified-candidates.
- [x] Add evidence-invalid.
- [x] Add evidence-incomplete.
- [x] Add recommendation-invalid.
- [x] Add artifact-transfer-failed.
- [x] Add policy-denied.
- [x] Add production-boundary-violation.
- [x] Add runtime-authority-violation.
- [x] Add credential-boundary-violation.
- [x] Add data-boundary-violation.
- [x] Add hot-path-denied.
- [x] Add internal optimization error.

## 52. Conformance

- [x] Prove Optimization Plane separate from inference.
- [x] Prove hot path cannot start optimization.
- [x] Prove generator cannot directly execute.
- [x] Prove recommendation != promotion.
- [x] Prove Runtime revalidation.
- [x] Prove artifact transport has no native handles.
- [x] Prove offline inference.
- [x] Prove credential separation.
- [x] Prove workload privacy.
- [x] Prove generator identity != trust.
- [x] Prove candidate failure isolation.
- [x] Prove active Kernel remains stable after campaign failure.
- [x] Prove optional Tachyon integration.
- [x] Prove CLI authority remains external.
- [x] Prove promotion obeys Kernel Selection Policy.
- [x] Prove observability redaction.

## 53. Documentation

- [x] Document Optimization Plane.
- [x] Document Inference Plane.
- [x] Document campaign lifecycle.
- [x] Document external generator boundary.
- [x] Document worker model.
- [x] Document evidence bundle.
- [x] Document recommendation/promotion distinction.
- [x] Document Tachyon neutrality.
- [x] Document credential boundary.
- [x] Document offline inference guarantee.

## 54. Final Validation

- [x] Run OpenSpec validation.
- [x] Verify Magnetar Runtime remains inference-only.
- [x] Verify agents/generators remain external.
- [x] Verify optimization cannot bypass qualification.
- [x] Verify optimization cannot bypass selection policy.
- [x] Verify optimization cannot bypass promotion.
- [x] Verify inference requires no optimization-service network access.

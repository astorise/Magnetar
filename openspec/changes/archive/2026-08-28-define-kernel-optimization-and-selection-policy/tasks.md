# Tasks

## 1. Selection Policy Domain

- [x] Define KernelSelectionPolicy.
- [x] Define policy version.
- [x] Define optimization profile.
- [x] Define ranking strategy.
- [x] Define fallback policy.
- [x] Define hysteresis policy.
- [x] Define exploration policy.
- [x] Add policy validation tests.

## 2. Eligibility Pipeline

- [x] Separate candidate discovery from eligibility.
- [x] Apply semantic compatibility.
- [x] Apply Operator version compatibility.
- [x] Apply qualification policy.
- [x] Apply trust policy.
- [x] Apply revocation policy.
- [x] Apply dtype compatibility.
- [x] Apply layout compatibility.
- [x] Apply shape compatibility.
- [x] Apply precision compatibility.
- [x] Apply determinism compatibility.
- [x] Apply Resource Affinity.
- [x] Apply Provider readiness.
- [x] Apply Device availability/health.
- [x] Apply memory feasibility.
- [x] Apply workspace feasibility.
- [x] Apply feature requirements.
- [x] Apply Prepared Kernel readiness.
- [x] Add eligibility tests.

## 3. Exclusion Reasons

- [x] Add semantic-incompatible.
- [x] Add operator-version-incompatible.
- [x] Add qualification-required.
- [x] Add qualification-expired.
- [x] Add qualification-revoked.
- [x] Add trust-denied.
- [x] Add dtype-incompatible.
- [x] Add layout-incompatible.
- [x] Add shape-incompatible.
- [x] Add precision-incompatible.
- [x] Add determinism-incompatible.
- [x] Add resource-affinity-incompatible.
- [x] Add provider-unready.
- [x] Add device-unavailable.
- [x] Add device-unhealthy.
- [x] Add memory-infeasible.
- [x] Add workspace-infeasible.
- [x] Add required-feature-missing.
- [x] Add prepared-kernel-unavailable.
- [x] Add benchmark-incompatible.
- [x] Add policy-denied.

## 4. Optimization Profiles

- [x] Add balanced profile.
- [x] Add latency profile.
- [x] Add throughput profile.
- [x] Add memory profile.
- [x] Add deterministic profile.
- [x] Add energy profile.
- [x] Add reproducible profile.
- [x] Make profiles versionable/extensible.

## 5. Ranking Strategies

- [x] Add Lexicographic.
- [x] Add WeightedScore.
- [x] Add PolicyOrdered.
- [x] Add Pinned.
- [x] Validate strategy configuration.
- [x] Add ranking strategy tests.

## 6. Weighted Scoring

- [x] Define normalized metric inputs.
- [x] Define weights.
- [x] Prevent invalid/non-comparable metric combination.
- [x] Handle missing metrics explicitly.
- [x] Add score determinism tests.

## 7. Lexicographic Ranking

- [x] Define ordered objectives.
- [x] Compare first differentiating objective.
- [x] Handle missing values explicitly.
- [x] Add lexicographic tests.

## 8. Tie-Breaking

- [x] Define stable tie-break.
- [x] Avoid HashMap iteration dependence.
- [x] Avoid pointer/process/thread timing dependence.
- [x] Use stable Kernel identity.
- [x] Add deterministic tie tests.

## 9. Benchmark Compatibility

- [x] Validate Provider context.
- [x] Validate Device architecture.
- [x] Validate driver/runtime compatibility.
- [x] Validate Operator version.
- [x] Validate artifact digest.
- [x] Validate dtype/layout.
- [x] Validate shape/workload bucket.
- [x] Validate benchmark profile version.
- [x] Add benchmark compatibility tests.

## 10. Benchmark Freshness

- [x] Represent benchmark age/freshness. (reuses `crate::kernel_benchmark::BenchmarkFreshness`)
- [x] Mark incompatible evidence.
- [x] Support stale acceptance policy.
- [x] Support stale exclusion policy.
- [x] Prevent hot-path rebenchmarking.
- [x] Add stale evidence tests.

## 11. Shape-Aware Ranking

- [x] Index performance by shape envelope.
- [x] Support shape buckets.
- [x] Reject evidence outside compatible envelope.
- [x] Add shape ranking tests.

## 12. Batch-Aware Ranking

- [x] Include active sequence count.
- [x] Include batch width.
- [x] Include total token count.
- [x] Include raggedness metadata.
- [x] Include continuous batching context.
- [x] Add batch ranking tests.

## 13. Prefill/Decode Ranking

- [x] Include generation phase in workload context.
- [x] Permit distinct prefill Kernel.
- [x] Permit distinct decode Kernel.
- [x] Add prefill/decode selection tests.

## 14. Pressure-Aware Ranking

- [x] Consume Provider pressure.
- [x] Consume Device utilization.
- [x] Consume queue pressure.
- [x] Consume memory pressure.
- [x] Consume workspace pressure.
- [x] Keep pressure as optimization input after hard eligibility.
- [x] Add pressure tests.

## 15. Memory-Aware Ranking

- [x] Query Memory Manager feasibility first.
- [x] Incorporate workspace cost.
- [x] Incorporate temporary memory cost.
- [x] Keep Memory Manager authoritative.
- [x] Add memory-selection tests.

## 16. Conversion Costs

- [x] Account for explicit dtype conversion.
- [x] Account for explicit layout conversion.
- [x] Account for explicit data movement.
- [x] Prevent hidden conversion.
- [x] Add total-cost ranking tests.

## 17. Preparation Cost

- [x] Classify one-time preparation cost.
- [x] Classify per-instance cost.
- [x] Avoid charging preparation cost per operation after preparation.
- [x] Add cost-classification tests.

## 18. Compilation Cost

- [x] Preserve compile-cost metadata.
- [x] Use only in cold-path planning.
- [x] Do not add cached compile cost to hot-path ranking.
- [x] Add compile-cost tests.

## 19. Hysteresis

- [x] Define improvement threshold.
- [x] Retain active candidate below threshold.
- [x] Suggest promotion above threshold.
- [x] Add threshold tests.

## 20. Anti-Flapping

- [x] Define cooldown.
- [x] Define minimum-active-duration.
- [x] Support stable measurement window.
- [x] Add rapid-pressure-change tests.

## 21. Selection Versus Promotion

- [x] Keep ranking independent from promotion.
- [x] Allow top candidate without immediate promotion.
- [x] Produce promotion recommendation.
- [x] Integrate with change 50 promotion lifecycle. (`apply_promotion_recommendation` bridges to `KernelRegistry::promote_generation`)
- [x] Add promotion-boundary tests.

## 22. Static Selection

- [x] Select during Model Instance loading. (`static_selection_required_during_warmup` gates it on pinned mode plus a warmup plan reaching `KernelPreparationPlaceholder`)
- [x] Allow lifetime pinning.
- [x] Add static selection tests.

## 23. Dynamic Selection

- [x] Allow policy-controlled re-evaluation.
- [x] Preserve Prepared Kernel lifetime. (via `KernelRegistry::promote_generation`'s existing retire-not-destroy guarantee, exercised through `apply_promotion_recommendation`)
- [x] Preserve in-flight generation.
- [x] Add dynamic selection tests.

## 24. Selection Cache

- [x] Define selection cache key.
- [x] Include Operator.
- [x] Include Provider/Device compatibility.
- [x] Include dtype/layout.
- [x] Include shape bucket.
- [x] Include batch bucket.
- [x] Include sequence bucket.
- [x] Include generation phase.
- [x] Include optimization profile.
- [x] Include policy version.
- [x] Add invalidation rules.
- [x] Add cache tests.

## 25. Model Instance Policy

- [x] Attach selection policy to Model Instance.
- [x] Support dynamic mode.
- [x] Support pinned mode.
- [x] Support deterministic mode.
- [x] Support fallback policy.
- [x] Add Model Instance tests.

## 26. Model Component Boundary

- [x] Prevent Model Component from naming concrete Kernel.
- [x] Prevent Model Component from selecting Provider.
- [x] Prevent Model Component from selecting Device.
- [x] Allow portable Operator requirements only.
- [x] Add boundary tests.

## 27. Session Preferences

- [x] Support optional optimization preference.
- [x] Keep preference non-authoritative.
- [x] Reject unsafe concrete Kernel override.
- [x] Add Session tests.

## 28. Generation Preferences

- [x] Allow high-level profile hint.
- [x] Prevent direct PreparedKernelId.
- [x] Prevent ineligible Kernel forcing.
- [x] Add generation boundary tests.

## 29. CLI Preferences

- [x] Map CLI latency preference.
- [x] Map CLI throughput preference.
- [x] Map deterministic preference.
- [x] Preserve Runtime authority.
- [x] Add CLI boundary tests.

## 30. Provider Boundary

- [x] Consume Provider metadata/metrics.
- [x] Keep final selection in Runtime.
- [x] Define allowed Provider-local private variants.
- [x] Require distinct Kernel identity for Runtime-relevant differences.
- [x] Add Provider boundary tests.

## 31. Fallback Policy

- [x] Define ordered fallback classes.
- [x] Make fallback explicit.
- [x] Support same-Provider fallback.
- [x] Support same-Device fallback.
- [x] Support other Provider fallback.
- [x] Support Reference CPU fallback.
- [x] Support fail-only policy.
- [x] Add fallback tests.

## 32. Cross-Provider Movement

- [x] Reject hidden movement.
- [x] Require explicit data movement.
- [x] Respect host-staging policy.
- [x] Respect Resource Affinity.
- [x] Add cross-Provider fallback tests.

## 33. Reproducible Mode

- [x] Pin KernelId.
- [x] Pin artifact digest.
- [x] Pin qualification profile.
- [x] Pin prepared generation where applicable.
- [x] Define failure if pinned candidate unavailable.
- [x] Add reproducibility tests.

## 34. Exploration

- [x] Define exploration enabled/disabled.
- [x] Restrict to eligible candidates.
- [x] Disable by default for reproducible mode.
- [x] Add exploration tests.

## 35. Canary

- [x] Reserve limited candidate execution.
- [x] Support request-count budget.
- [x] Support time budget.
- [x] Support percentage placeholder.
- [x] Keep distributed canary out of scope.
- [x] Add local canary tests.

## 36. Exploration Failure

- [x] Stop exploration on policy trigger.
- [x] Demote failed candidate.
- [x] Preserve active known-good candidate.
- [x] Integrate rollback where applicable. (`apply_exploration_failure_action` bridges `TriggerRollback` to `KernelRegistry::rollback_generation`)
- [x] Add failure tests.

## 37. Online Measurements

- [x] Associate metrics with Kernel generation.
- [x] Associate Operator.
- [x] Associate workload bucket.
- [x] Associate Provider/Device.
- [x] Prevent raw model data requirement.
- [x] Add measurement tests.

## 38. Policy Precedence

- [x] Define Runtime safety precedence.
- [x] Define deployment policy precedence.
- [x] Define Model Instance policy precedence.
- [x] Define Session preference precedence.
- [x] Define generation preference precedence.
- [x] Define CLI hint precedence.
- [x] Add override-denial tests.

## 39. Explainability

- [x] Record eligible candidates.
- [x] Record exclusions.
- [x] Record rank.
- [x] Record selected candidate.
- [x] Record fallback.
- [x] Record active retention.
- [x] Record promotion recommendation.
- [x] Add redacted explanation API/internal structure.

## 40. Errors

- [x] Add kernel-selection-no-candidates.
- [x] Add kernel-selection-no-eligible-candidates.
- [x] Add kernel-selection-policy-invalid.
- [x] Add kernel-selection-profile-unsupported.
- [x] Add kernel-selection-pinned-kernel-unavailable.
- [x] Add kernel-selection-pinned-kernel-ineligible.
- [x] Add kernel-selection-metric-missing.
- [x] Add kernel-selection-benchmark-stale.
- [x] Add kernel-selection-benchmark-incompatible.
- [x] Add kernel-selection-memory-infeasible.
- [x] Add kernel-selection-affinity-incompatible.
- [x] Add kernel-selection-determinism-unsatisfied.
- [x] Add kernel-selection-fallback-denied.
- [x] Add kernel-selection-fallback-exhausted.
- [x] Add kernel-selection-promotion-threshold-not-met.
- [x] Add kernel-selection-cache-stale.
- [x] Add kernel-selection-exploration-denied.
- [x] Add internal-kernel-selection-error.

## 41. Observability

- [x] Observe selection started.
- [x] Observe candidate discovered.
- [x] Observe candidate excluded.
- [x] Observe candidate eligible.
- [x] Observe candidate ranked.
- [x] Observe Kernel selected.
- [x] Observe active Kernel retained.
- [x] Observe fallback.
- [x] Observe cache hit/miss.
- [x] Observe promotion suggestion.
- [x] Observe hysteresis rejection.
- [x] Observe exploration.
- [x] Redact native handles and model data.

## 42. Conformance

- [x] Prove ineligible fastest Kernel never wins.
- [x] Prove trust filtering happens before ranking.
- [x] Prove qualification filtering happens before ranking.
- [x] Prove Resource Affinity filtering happens before ranking.
- [x] Prove memory rejection cannot be overridden.
- [x] Prove deterministic profile behavior.
- [x] Prove stable tie-breaking.
- [x] Prove stale benchmark handling.
- [x] Prove shape-aware ranking.
- [x] Prove prefill/decode may differ.
- [x] Prove hysteresis.
- [x] Prove fallback explicitness.
- [x] Prove user hints non-authoritative.
- [x] Prove Provider cannot select across Providers.
- [x] Prove exploration only uses eligible candidates.
- [x] Prove explainability redaction.

## 43. Documentation

- [x] Document filter-first/rank-second rule.
- [x] Document optimization profiles.
- [x] Document ranking strategies.
- [x] Document policy precedence.
- [x] Document hysteresis.
- [x] Document fallback.
- [x] Document reproducible mode.
- [x] Document exploration.
- [x] Document explainability.

## 44. Final Validation

- [x] Run OpenSpec validation.
- [x] Verify performance cannot override correctness.
- [x] Verify performance cannot override trust.
- [x] Verify performance cannot override Resource Affinity.
- [x] Verify performance cannot override Memory Manager.
- [x] Verify Model Component remains Kernel-independent.
- [x] Verify selection remains Runtime-owned.

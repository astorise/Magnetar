# Tasks

## 1. Performance Observation Domain

- [x] Define KernelExecutionPerformanceObservation.
- [x] Bind KernelId.
- [x] Bind artifact digest.
- [x] Bind specialization.
- [x] Bind Prepared generation where relevant.
- [x] Bind Operator.
- [x] Bind Provider/Device context.
- [x] Bind workload bucket.
- [x] Bind timestamp.
- [x] Add validation tests.

## 2. Performance Metrics

- [x] Support end-to-end operation latency.
- [x] Support queue delay where available.
- [x] Support Provider submission timing.
- [x] Support Device timing where available.
- [x] Support throughput evidence.
- [x] Support workspace/memory observations.
- [x] Support timeout/failure counters.
- [x] Keep metric vocabulary extensible.

## 3. Timing Capability

- [x] Define timing-method metadata.
- [x] Support host timing.
- [x] Support Provider timing.
- [x] Support Device event timing.
- [x] Allow unavailable metrics.
- [x] Add timing capability tests.

## 4. Workload Bucket

- [x] Define KernelPerformanceWorkloadBucket.
- [x] Add Operator.
- [x] Add shape bucket.
- [x] Add batch bucket.
- [x] Add sequence bucket.
- [x] Add generation phase.
- [x] Add dtype.
- [x] Add layout.
- [x] Add quantization.
- [x] Add Provider/Device compatibility.
- [x] Add bucket identity.

## 5. Bucket Policy

- [x] Define bucket policy version.
- [x] Define shape boundaries.
- [x] Define batch boundaries.
- [x] Define sequence boundaries.
- [x] Define phase categories.
- [x] Define Device grouping.
- [x] Add deterministic mapping tests.

## 6. Privacy Boundary

- [x] Exclude raw prompts.
- [x] Exclude user documents.
- [x] Exclude raw tensor values.
- [x] Exclude model weights.
- [x] Exclude KV contents.
- [x] Avoid unnecessary user/session identifiers.
- [x] Add privacy tests.

## 7. Aggregation

- [x] Define bounded aggregation.
- [x] Add count.
- [x] Add mean.
- [x] Add variance where useful.
- [x] Add min/max.
- [x] Add quantile summaries.
- [x] Add failure count.
- [x] Add timeout count.
- [x] Add pressure summary.
- [x] Add aggregation tests.

## 8. Observation Retention

- [x] Bound raw observation retention.
- [x] Allow aggregation then discard.
- [x] Define time/count limits.
- [x] Add retention tests.

## 9. Performance Model

- [x] Define KernelPerformanceModel.
- [x] Bind candidate.
- [x] Bind workload bucket.
- [x] Bind metric summary.
- [x] Bind sample count.
- [x] Bind evidence quality.
- [x] Bind freshness.
- [x] Bind baseline.
- [x] Bind model version.

## 10. Performance Model Version

- [x] Define model policy version.
- [x] Invalidate incompatible summaries.
- [x] Preserve historical version identity.
- [x] Add version tests.

## 11. Sample Sufficiency

- [x] Define minimum samples.
- [x] Define minimum observation duration.
- [x] Define optional confidence threshold.
- [x] Prevent premature regression action.
- [x] Add low-sample tests.

## 12. Evidence Quality

- [x] Define insufficient.
- [x] Define low.
- [x] Define medium.
- [x] Define high.
- [x] Avoid fabricated statistical confidence.
- [x] Add quality tests.

## 13. Warmup Handling

- [x] Classify warmup samples.
- [x] Exclude or separate where policy says.
- [x] Track steady-state transition.
- [x] Add warmup regression tests.

## 14. Cold-Start Handling

- [x] Track first-use overhead.
- [x] Separate cold-start metrics.
- [x] Prevent cold-start from corrupting steady-state rank where policy excludes it.
- [x] Add cold-start tests.

## 15. Measurement Sampling

- [x] Define sampling policy.
- [x] Support all.
- [x] Support one-in-N.
- [x] Support bounded probabilistic sampling.
- [x] Support per-bucket budgets.
- [x] Add sampling tests.

## 16. Adaptive Sampling

- [x] Increase sampling after promotion.
- [x] Increase sampling for new bucket.
- [x] Increase sampling on suspected regression.
- [x] Reduce sampling after stabilization.
- [x] Keep overhead bounded.
- [x] Add adaptive sampling tests.

## 17. Measurement Overhead

- [x] Define overhead budget.
- [x] Disable/reduce high-cost timing when budget exceeded.
- [x] Preserve inference correctness.
- [x] Add overhead tests.

## 18. Offline Evidence

- [x] Distinguish offline benchmark record.
- [x] Link tuning record.
- [x] Define compatibility checks.
- [x] Add offline evidence tests.

## 19. Online Evidence

- [x] Distinguish production evidence.
- [x] Bind exact Kernel context.
- [x] Aggregate by workload bucket.
- [x] Add online evidence tests.

## 20. Online/Offline Policy

- [x] Add offline-only.
- [x] Add hybrid.
- [x] Add online-preferred-after-sufficient-samples.
- [x] Add pinned-offline.
- [x] Add policy tests.

## 21. Benchmark Drift

- [x] Define benchmark baseline comparison.
- [x] Define relative threshold.
- [x] Define absolute threshold.
- [x] Require sufficient samples.
- [x] Add drift detection tests.

## 22. Workload Drift

- [x] Compare current workload distribution to tuning workload.
- [x] Detect batch shift.
- [x] Detect sequence shift.
- [x] Detect phase shift.
- [x] Detect shape shift.
- [x] Add workload drift tests.

## 23. Performance Regression

- [x] Compare with prior candidate baseline.
- [x] Compare with prior generation.
- [x] Compare with policy SLO.
- [x] Add latency regression.
- [x] Add throughput regression.
- [x] Add p99 regression.
- [x] Add timeout-rate regression.
- [x] Add regression tests.

## 24. Regression Confirmation

- [x] Require minimum samples.
- [x] Require minimum duration where configured.
- [x] Apply hysteresis.
- [x] Avoid single-outlier rollback.
- [x] Add confirmation tests.

## 25. Outlier Handling

- [x] Define outlier policy.
- [x] Preserve tail evidence.
- [x] Prevent silent arbitrary deletion.
- [x] Add outlier tests.

## 26. Device Pressure Correlation

- [x] Correlate Provider pressure.
- [x] Correlate Device utilization.
- [x] Correlate queue depth.
- [x] Correlate memory pressure.
- [x] Avoid false Kernel-specific regression where possible.
- [x] Add pressure correlation tests.

## 27. Selection Feedback

- [x] Feed compatible online metrics to Kernel Selection Policy.
- [x] Preserve hard eligibility.
- [x] Preserve trust requirement.
- [x] Preserve qualification requirement.
- [x] Preserve Resource Affinity.
- [x] Preserve Memory Manager authority.
- [x] Add selection feedback tests.

## 28. Tuning Staleness

- [x] Mark tuning stale on sustained performance drift.
- [x] Mark stale on workload shift.
- [x] Mark stale on candidate-set change.
- [x] Mark stale on Provider/driver incompatibility.
- [x] Add stale tuning tests.

## 29. Retuning Request

- [x] Define KernelRetuningRequest.
- [x] Add reason.
- [x] Add workload bucket.
- [x] Add candidate context.
- [x] Add evidence summary.
- [x] Add urgency.
- [x] Add deduplication/rate limiting.

## 30. Bounded Retuning

- [x] Reuse Runtime Autotuning contract.
- [x] Restrict to authorized templates/candidates.
- [x] Prevent arbitrary source mutation.
- [x] Prevent AI generation.
- [x] Add boundary tests.

## 31. Retuning Admission

- [x] Respect Provider pressure.
- [x] Respect Device pressure.
- [x] Respect inference priority.
- [x] Respect tuning budgets.
- [x] Allow postponement.
- [x] Add admission tests.

## 32. No Hot-Path Retuning

- [x] Detect regression without synchronous benchmark.
- [x] Queue background request.
- [x] Use known-good/fallback behavior.
- [x] Add decode regression tests.

## 33. External Optimization Escalation

- [x] Define optimization escalation signal.
- [x] Trigger only after policy allows.
- [x] Keep external.
- [x] Do not invoke generator from inference Runtime.
- [x] Add escalation tests.

## 34. Demotion Signal

- [x] Define KernelPerformanceDemotionSignal.
- [x] Bind reason/evidence.
- [x] Feed selection/promotion state machine.
- [x] Add demotion tests.

## 35. Rollback Signal

- [x] Define rollback recommendation from confirmed regression.
- [x] Preserve rollback policy authority.
- [x] Add rollback integration tests.

## 36. Post-Promotion Observation

- [x] Define heightened observation window.
- [x] Increase sample rate.
- [x] Compare old/new generations.
- [x] Preserve rollback candidate.
- [x] Add post-promotion tests.

## 37. Performance Health

- [x] Add unknown.
- [x] Add warming.
- [x] Add healthy.
- [x] Add degraded.
- [x] Add regressed.
- [x] Add stale.
- [x] Keep separate from qualification/Provider health.

## 38. Failure Evidence

- [x] Track structured Kernel execution failures.
- [x] Track timeout rate.
- [x] Detect increased failure rates.
- [x] Add failure regression tests.

## 39. Memory Anomaly

- [x] Compare actual workspace to advertised expectation where possible.
- [x] Detect unexpected resource growth.
- [x] Distinguish performance issue from contract violation.
- [x] Add memory anomaly tests.

## 40. Contract Violation

- [x] Escalate contract violation separately.
- [x] Do not classify severe semantic/memory violation as mere slowdown.
- [x] Integrate qualification/revocation workflows.
- [x] Add violation tests.

## 41. Hysteresis

- [x] Define performance action thresholds.
- [x] Define stable duration.
- [x] Define cooldown.
- [x] Prevent selection flapping.
- [x] Add noisy-metric tests.

## 42. Retuning Rate Limit

- [x] Define minimum interval.
- [x] Deduplicate equivalent retuning requests.
- [x] Prevent continuous retuning loops.
- [x] Add rate-limit tests.

## 43. Performance Aging

- [x] Define evidence time window.
- [x] Support decay.
- [x] Support expiration.
- [x] Prevent ancient evidence dominating selection.
- [x] Add aging tests.

## 44. Artifact Binding

- [x] Bind observations to artifact digest.
- [x] Prevent new artifact inheriting old metrics.
- [x] Add replacement tests.

## 45. Specialization Binding

- [x] Bind to Specialization Instance.
- [x] Prevent cross-specialization reuse.
- [x] Add specialization evidence tests.

## 46. Cross-Device Evidence

- [x] Reject incompatible architecture reuse.
- [x] Define optional compatible Device class policy.
- [x] Add cross-device tests.

## 47. Cross-Provider Evidence

- [x] Keep Provider evidence separate.
- [x] Prevent implicit cross-Provider ranking evidence.
- [x] Add cross-Provider tests.

## 48. Model Instance Policy

- [x] Support adaptive mode.
- [x] Support pinned mode.
- [x] Support feedback-disabled mode.
- [x] Add Model Instance tests.

## 49. Reproducible Mode

- [x] Allow observations without adaptation.
- [x] Prevent adaptive selection changes.
- [x] Prevent background tuning changes to pinned instance.
- [x] Add reproducibility tests.

## 50. Generation Integration

- [x] Expose prefill/decode context.
- [x] Expose sequence bucket inputs.
- [x] Expose batch context.
- [x] Do not expose prompt text.
- [x] Add generation privacy tests.

## 51. Continuous Batching

- [x] Define batch-level timing attribution.
- [x] Avoid fake per-session metrics.
- [x] Support batch workload buckets.
- [x] Add batching feedback tests.

## 52. Telemetry Export

- [x] Define aggregated export structure.
- [x] Exclude raw inference content.
- [x] Support external Optimization Plane consumption.
- [x] Add export redaction tests.

## 53. Error Model

- [x] Add observation errors.
- [x] Add bucket errors.
- [x] Add model errors.
- [x] Add sample insufficiency.
- [x] Add measurement errors.
- [x] Add drift errors.
- [x] Add regression errors.
- [x] Add memory anomaly.
- [x] Add retuning errors.
- [x] Add escalation errors.
- [x] Add feedback policy errors.
- [x] Add internal performance error.

## 54. Observability

- [x] Observe sampled measurement.
- [x] Observe model updates.
- [x] Observe insufficient evidence.
- [x] Observe drift.
- [x] Observe workload shift.
- [x] Observe suspected regression.
- [x] Observe confirmed regression.
- [x] Observe recovery.
- [x] Observe retuning request.
- [x] Observe rate limiting.
- [x] Observe rollback recommendation.
- [x] Observe external optimization escalation.
- [x] Redact sensitive data.

## 55. Conformance

- [x] Prove performance != trust.
- [x] Prove performance != qualification.
- [x] Prove ineligible Kernel cannot become eligible from metrics.
- [x] Prove workload/context binding.
- [x] Prove artifact/specialization isolation.
- [x] Prove insufficient samples prevent premature action.
- [x] Prove warmup handling.
- [x] Prove online/offline distinction.
- [x] Prove benchmark drift detection.
- [x] Prove workload drift detection.
- [x] Prove bounded retuning.
- [x] Prove no hot-path retuning.
- [x] Prove no Runtime AI generation.
- [x] Prove hysteresis.
- [x] Prove reproducible mode.
- [x] Prove bounded telemetry retention.
- [x] Prove known-good preservation.
- [x] Prove export redaction.

## 56. Documentation

- [x] Document online versus offline evidence.
- [x] Document workload buckets.
- [x] Document Performance Model.
- [x] Document benchmark drift.
- [x] Document workload drift.
- [x] Document regression detection.
- [x] Document bounded re-tuning.
- [x] Document external escalation.
- [x] Document reproducible mode.
- [x] Document privacy guarantees.

## 57. Final Validation

- [x] Run OpenSpec validation.
- [x] Verify adaptive feedback cannot grant trust.
- [x] Verify adaptive feedback cannot grant qualification.
- [x] Verify adaptive feedback cannot generate code.
- [x] Verify re-tuning stays bounded.
- [x] Verify production inference remains protected from measurement/tuning overhead.
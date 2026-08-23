# Tasks

## 1. Inventory Existing Provider Status

- [x] Inventory existing Provider health fields.
- [x] Inventory existing Device health fields.
- [x] Inventory Provider readiness or availability fields.
- [x] Inventory Scheduler admission checks.
- [x] Inventory Resolution filtering behavior.
- [x] Inventory Provider diagnostics.
- [x] Inventory observability events related to Provider status.
- [x] Identify conflated health/readiness/pressure concepts.

## 2. Lifecycle Model

- [x] Define Provider lifecycle states.
- [x] Include registered.
- [x] Include loading.
- [x] Include initializing.
- [x] Include ready.
- [x] Include draining.
- [x] Include stopped.
- [x] Include failed.
- [x] Include removed.
- [x] Define allowed transitions.
- [x] Add lifecycle transition tests.

## 3. Health Model

- [x] Define Provider health states.
- [x] Include unknown.
- [x] Include healthy.
- [x] Include degraded.
- [x] Include unhealthy.
- [x] Include failed.
- [x] Define health reasons.
- [x] Define severity.
- [x] Define redacted diagnostic detail.
- [x] Add health state tests.

## 4. Readiness Model

- [x] Define readiness states.
- [x] Include not-ready.
- [x] Include ready.
- [x] Include read-only where useful.
- [x] Include draining.
- [x] Define readiness reasons.
- [x] Ensure healthy but not-ready is representable.
- [x] Ensure degraded but ready is representable.
- [x] Add readiness tests.

## 5. Pressure Model

- [x] Define pressure levels.
- [x] Include unknown.
- [x] Include low.
- [x] Include moderate.
- [x] Include high.
- [x] Include saturated.
- [x] Define active operation count signal.
- [x] Define queued operation count signal.
- [x] Define memory pressure signal.
- [x] Define device memory pressure signal.
- [x] Define estimated queue delay signal where possible.
- [x] Define utilization signal where available.
- [x] Add pressure classification tests.

## 6. Admission Model

- [x] Define admission decision.
- [x] Include admit.
- [x] Include prefer-not.
- [x] Include reject.
- [x] Allow admission to be scoped by Provider.
- [x] Allow admission to be scoped by Device.
- [x] Allow admission to be scoped by Capability.
- [x] Allow admission to be scoped by operation family where supported.
- [x] Define admission reason.
- [x] Add admission tests.

## 7. Status Snapshot

- [x] Define Provider status snapshot.
- [x] Include lifecycle.
- [x] Include health.
- [x] Include readiness.
- [x] Include pressure.
- [x] Include admission summary.
- [x] Include timestamp or monotonic freshness.
- [x] Include TTL where applicable.
- [x] Include diagnostic reason.
- [x] Include Device status map.
- [x] Include Capability status map.
- [x] Ensure snapshot is immutable once recorded.
- [x] Add snapshot serialization tests if status is serialized.

## 8. Staleness

- [x] Define status report freshness.
- [x] Define TTL semantics.
- [x] Define stale status behavior.
- [x] Mark expired status as stale.
- [x] Ensure stale status is not treated as fully ready by default.
- [x] Add stale status tests.
- [x] Add policy override test if stale-ready override exists.

## 9. Device-Level Status

- [x] Define Device health state.
- [x] Define Device readiness state.
- [x] Define Device pressure state.
- [x] Define Device availability.
- [x] Define Device memory pressure.
- [x] Define Device interruption state.
- [x] Ensure Provider may be healthy while one Device is unavailable.
- [x] Add Device status tests.

## 10. Capability-Level Status

- [x] Define Capability implementation status per Provider.
- [x] Track readiness by Capability.
- [x] Track health by Capability where useful.
- [x] Track pressure by Capability where useful.
- [x] Support Compute Capability status.
- [x] Prepare for future Generation Capability status.
- [x] Add Capability status tests.

## 11. Operation-Family Status

- [x] Allow optional operation-family status.
- [x] Represent unsupported operation family.
- [x] Represent temporarily unavailable operation family.
- [x] Represent saturated operation family.
- [x] Represent degraded operation family.
- [x] Ensure absence of operation-family status falls back to Capability status.
- [x] Add operation-family tests where implemented.

## 12. Drainage

- [x] Define draining transition.
- [x] Prevent ordinary new unpinned work during drain.
- [x] Allow in-flight work to complete where safe.
- [x] Preserve Resource Affinity for pinned resources.
- [x] Do not silently migrate resources during drain.
- [x] Report drain completion.
- [x] Add draining Provider tests.
- [x] Add pinned-resource-during-drain tests.

## 13. Existing Pinned Resources

- [x] Distinguish new unpinned work from dependent pinned work.
- [x] Allow policy to admit pinned work on draining Provider.
- [x] Reject migration unless explicit data movement exists.
- [x] Preserve Provider-pinned semantics.
- [x] Preserve Device-bound semantics.
- [x] Add tests for pinned resources on not-ready Provider.
- [x] Add tests for pinned resources on draining Provider.

## 14. Interruption State

- [x] Define Provider interruption reasons.
- [x] Include device reset.
- [x] Include driver loss.
- [x] Include device removed.
- [x] Include allocator failure.
- [x] Include OOM recovery.
- [x] Include thermal throttling.
- [x] Include administrative drain.
- [x] Map interruption state to health/readiness/admission.
- [x] Add interruption tests.

## 15. Resolution Integration

- [x] Filter failed Providers.
- [x] Filter unavailable Devices.
- [x] Penalize degraded Providers according to policy.
- [x] Penalize high-pressure Providers according to policy.
- [x] Reject saturated Providers unless policy permits queueing.
- [x] Avoid draining Providers for new unpinned work.
- [x] Preserve Resource Affinity precedence.
- [x] Add Resolution tests for each status dimension.

## 16. Resolution Diagnostics

- [x] Record selected Provider status in diagnostics.
- [x] Record skipped Provider reason.
- [x] Record rejected Provider reason.
- [x] Record stale status reason.
- [x] Record pressure penalty reason.
- [x] Record draining rejection reason.
- [x] Redact unsafe native details.
- [x] Add diagnostic tests.

## 17. Scheduler Integration

- [x] Check selected Provider readiness before submission.
- [x] Check selected Device readiness before submission.
- [x] Check saturation before admission.
- [x] Decide whether to retry resolution, queue, or fail based on policy.
- [x] Ensure Scheduler does not independently choose Provider.
- [x] Add Scheduler status-change tests.
- [x] Add Provider-becomes-not-ready-after-resolution test.

## 18. Provider Execution API Integration

- [x] Ensure Provider execution requests can observe current readiness result.
- [x] Map Provider refusal to structured Runtime error.
- [x] Distinguish Provider refusal from execution failure.
- [x] Distinguish not-ready from unhealthy.
- [x] Distinguish saturated from failed.
- [x] Add Provider refusal tests.

## 19. Error Model

- [x] Add or refine error for Provider not ready.
- [x] Add or refine error for Provider draining.
- [x] Add or refine error for Provider saturated.
- [x] Add or refine error for stale Provider status.
- [x] Add or refine error for Device not ready.
- [x] Add or refine error for Capability not ready.
- [x] Preserve existing Provider unavailable error.
- [x] Map errors to stable diagnostics.

## 20. Observability

- [x] Emit Provider lifecycle observations.
- [x] Emit health change observations.
- [x] Emit readiness change observations.
- [x] Emit pressure change observations.
- [x] Emit stale report observations.
- [x] Emit admission decision observations.
- [x] Emit drain start observations.
- [x] Emit drain completion observations.
- [x] Emit Device status observations.
- [x] Emit Capability status observations.
- [x] Ensure observability failure does not alter status.

## 21. Policy

- [x] Define policy for degraded Provider.
- [x] Define policy for high-pressure Provider.
- [x] Define policy for saturated Provider.
- [x] Define policy for draining Provider.
- [x] Define policy for stale status.
- [x] Define policy for pinned work during drain.
- [x] Define policy for retrying after readiness changes.
- [x] Add policy tests.

## 22. Provider Mock Updates

- [x] Update mock Provider to report lifecycle.
- [x] Update mock Provider to report health.
- [x] Update mock Provider to report readiness.
- [x] Update mock Provider to report pressure.
- [x] Update mock Provider to report Device status.
- [x] Update mock Provider to report Capability status.
- [x] Add mock scenarios for degraded, saturated, draining, stale, and failed.

## 23. Documentation

- [x] Document lifecycle versus health.
- [x] Document health versus readiness.
- [x] Document readiness versus pressure.
- [x] Document admission decisions.
- [x] Document Device-level status.
- [x] Document Capability-level status.
- [x] Document TTL and staleness.
- [x] Document draining behavior.
- [x] Document Resolution interaction.
- [x] Document Scheduler interaction.

## 24. OpenSpec Consistency

- [x] Review Provider specs for ambiguous `available`.
- [x] Review Runtime specs for ambiguous `healthy`.
- [x] Review Scheduler specs for ambiguous admission language.
- [x] Review Resolution specs for ambiguous Provider selection language.
- [x] Replace ambiguous wording with lifecycle/health/readiness/pressure terms.
- [x] Preserve archived changes unchanged.

## 25. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run Provider status tests.
- [x] Run Resolution policy tests.
- [x] Run Scheduler integration tests.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify alive/healthy/ready/pressure are distinct.
- [x] Verify Resource Affinity still overrides policy preference.
- [x] Verify draining does not imply silent migration.

# Tasks

## 1. Conformance Scope

- [x] Define the Provider Conformance Suite scope.
- [x] Limit conformance to Magnetar inference Runtime responsibilities.
- [x] Exclude filesystem, Git, shell, workspace, secrets, network, and general
      tool behavior.
- [x] Define supported Provider target kinds.
- [x] Define built-in Provider target.
- [x] Define dynamic Provider target.
- [x] Define mock/test Provider target.
- [x] Define development Provider target.

## 2. Conformance Profiles

- [x] Define `provider-core` profile.
- [x] Define `provider-compute` profile.
- [x] Define `provider-data-movement` profile.
- [x] Define `provider-cancellation` profile.
- [x] Define `provider-observability` profile.
- [x] Define `provider-dynamic-abi` profile.
- [x] Define optional hardware-specific profiles.
- [x] Define profile selection based on Provider advertisements.
- [x] Define required versus optional profile behavior.

## 3. Conformance Harness

- [x] Create reusable Provider conformance test harness.
- [x] Allow harness to instantiate built-in Providers.
- [x] Allow harness to instantiate mock Providers.
- [x] Allow harness to load dynamic Providers where supported.
- [x] Allow harness to select profiles.
- [x] Allow harness to collect diagnostics.
- [x] Allow harness to produce structured reports.
- [x] Avoid real GPU requirement for default harness tests.

## 4. Conformance Report

- [x] Define report schema.
- [x] Include Provider identity.
- [x] Include Provider version.
- [x] Include Runtime version.
- [x] Include conformance suite version.
- [x] Include selected profiles.
- [x] Include passed tests.
- [x] Include failed tests.
- [x] Include skipped tests.
- [x] Include unsupported optional features.
- [x] Include diagnostic details.
- [x] Include timestamp.
- [x] Prefer machine-readable JSON output where practical.

## 5. Provider Metadata Tests

- [x] Validate ProviderId syntax.
- [x] Validate ProviderId stability.
- [x] Validate Provider name presence.
- [x] Validate Provider version syntax.
- [x] Validate vendor metadata.
- [x] Validate Runtime compatibility metadata.
- [x] Validate feature flags.
- [x] Reject duplicate ProviderId where duplicate policy disallows it.
- [x] Verify metadata does not expose native handles.

## 6. Capability Advertisement Tests

- [x] Validate Capability identifiers.
- [x] Validate Capability versions.
- [x] Reject unsupported Capability major versions.
- [x] Validate operation support metadata.
- [x] Validate data movement support metadata.
- [x] Validate memory requirement metadata.
- [x] Validate Device requirement metadata.
- [x] Verify advertisements match observed behavior.
- [x] Reject malformed advertisements.

## 7. Device Metadata Tests

- [x] Validate DeviceId syntax.
- [x] Validate Device type.
- [x] Validate Provider ownership.
- [x] Validate memory metadata where present.
- [x] Validate Device feature metadata.
- [x] Reject duplicate DeviceId per Provider.
- [x] Verify no raw native handles appear in public Device metadata.
- [x] Verify Device metadata remains stable across equivalent initialization
      where required.

## 8. Provider Status Tests

- [x] Validate lifecycle states.
- [x] Validate health states.
- [x] Validate readiness states.
- [x] Validate pressure states.
- [x] Validate admission decisions.
- [x] Validate freshness/TTL.
- [x] Validate Device-level status.
- [x] Validate Capability-level status.
- [x] Verify healthy is distinct from ready.
- [x] Verify saturated is distinct from failed.
- [x] Verify stale status is handled according to policy.

## 9. Provider Lifecycle Tests

- [x] Test initialization.
- [x] Test registration.
- [x] Test transition to ready.
- [x] Test transition to draining.
- [x] Test drain completion.
- [x] Test transition to failed.
- [x] Test shutdown.
- [x] Test post-shutdown rejection.
- [x] Test resource cleanup.

## 10. Execution Admission Tests

- [x] Test execution admitted when Provider is ready.
- [x] Test execution rejected when Provider not ready.
- [x] Test execution rejected when Provider draining.
- [x] Test execution rejected when Provider saturated and queueing not allowed.
- [x] Test execution admitted or delayed according to explicit queue policy.
- [x] Test Device unavailable rejection.
- [x] Test Capability not-ready rejection.
- [x] Distinguish admission rejection from execution failure.

## 11. Compute Operation Tests

- [x] Test minimal valid Compute operation.
- [x] Test unsupported operation rejection.
- [x] Test invalid input descriptor rejection.
- [x] Test invalid shape rejection.
- [x] Test invalid dtype rejection.
- [x] Test invalid layout rejection.
- [x] Test unsupported operation family rejection.
- [x] Test output descriptor correctness.
- [x] Test deterministic result for simple operation where supported.
- [x] Test numeric tolerance for supported dtype.

## 12. Compute Graph Tests

- [x] Test valid small graph.
- [x] Test graph with unsupported node.
- [x] Test graph with invalid edge.
- [x] Test graph with incompatible shapes.
- [x] Test graph with unsupported dtype.
- [x] Test graph execution plan compatibility.
- [x] Test graph failure maps to stable error.

## 13. Data Movement Tests

- [x] Test upload where advertised.
- [x] Test download where advertised.
- [x] Test copy where advertised.
- [x] Test materialize where advertised.
- [x] Test transfer where advertised.
- [x] Test dtype conversion where advertised.
- [x] Test placement conversion where advertised.
- [x] Test unsupported movement rejected.
- [x] Test host staging forbidden.
- [x] Test host staging permitted but policy denied.
- [x] Test output affinity after movement.

## 14. Resource Affinity Tests

- [x] Test Provider-bound resource remains Provider-bound.
- [x] Test Device-bound resource remains Device-bound.
- [x] Test dependent operation on Provider-bound resource.
- [x] Test dependent operation on Device-bound resource.
- [x] Test incompatible Provider rejected.
- [x] Test explicit data movement required for migration.
- [x] Test draining does not silently migrate resource.
- [x] Test forged affinity rejected.

## 15. Provider-Owned Resource Tests

- [x] Test creation of Provider-owned resource.
- [x] Test use of Provider-owned resource.
- [x] Test invalid resource rejection.
- [x] Test resource lifetime cleanup.
- [x] Test resource after Provider shutdown fails safely.
- [x] Test resource handle is not exposed to Component WIT.
- [x] Test resource handle is not serialized as public stable ID.

## 16. Cancellation Tests

- [x] Test cancellation before execution.
- [x] Test cancellation during execution where supported.
- [x] Test cancellation after completion.
- [x] Test cancellation unsupported error.
- [x] Test cancellation failure error.
- [x] Test cancellation idempotency where required.
- [x] Test cleanup after cancellation.
- [x] Test Provider status remains consistent after cancellation.

## 17. Error Mapping Tests

- [x] Test Provider not ready error.
- [x] Test Provider draining error.
- [x] Test Provider saturated error.
- [x] Test Device unavailable error.
- [x] Test Capability not ready error.
- [x] Test invalid request error.
- [x] Test unsupported operation error.
- [x] Test unsupported dtype error.
- [x] Test unsupported layout error.
- [x] Test allocation failure error.
- [x] Test out-of-memory error.
- [x] Test execution failure error.
- [x] Test resource invalid error.
- [x] Test cancellation unsupported error.
- [x] Test cancellation failed error.
- [x] Test internal Provider error.
- [x] Verify diagnostics are stable and redacted.

## 18. Observability Tests

- [x] Test execution observation emitted.
- [x] Test failure observation emitted.
- [x] Test cancellation observation emitted.
- [x] Test status observation emitted.
- [x] Test Provider lifecycle observation emitted.
- [x] Test observability failure does not change execution result.
- [x] Test no native handles appear in observations.
- [x] Test diagnostics are redacted.

## 19. Dynamic ABI Conformance Tests

- [x] Test factory symbol exists.
- [x] Test ABI version supported.
- [x] Test descriptor structure valid.
- [x] Test required function pointers present.
- [x] Test metadata function behavior.
- [x] Test Capability advertisement function behavior.
- [x] Test Device metadata function behavior.
- [x] Test status function behavior.
- [x] Test execution function behavior.
- [x] Test release functions behavior.
- [x] Test destroy function behavior.
- [x] Test no unwind across ABI.
- [x] Test trait-object factory is not stable conformance path.

## 20. Loading Policy Tests

- [x] Test allowed path loading.
- [x] Test disallowed path rejection.
- [x] Test trusted digest policy.
- [x] Test revoked Provider binary rejection where implemented.
- [x] Test development mode is explicit.
- [x] Test development mode still validates ABI.
- [x] Test invalid dynamic Provider is not registered.

## 21. Hardware-Independent Fixtures

- [x] Provide CPU-only or mock fixtures for default CI.
- [x] Avoid requiring GPU hardware.
- [x] Avoid requiring vendor drivers.
- [x] Avoid requiring network.
- [x] Avoid requiring Tachyon.
- [x] Make fixtures deterministic.
- [x] Document fixture assumptions.

## 22. Optional Hardware Profiles

- [x] Define optional CUDA conformance profile placeholder.
- [x] Define optional Metal conformance profile placeholder.
- [x] Define optional OpenVINO conformance profile placeholder.
- [x] Define optional QNN conformance profile placeholder.
- [x] Mark hardware-dependent tests as optional.
- [x] Document how to run hardware profiles locally.

## 23. Provider Advertisement Versus Behavior

- [x] Verify Provider behavior matches advertised operation support.
- [x] Verify Provider behavior matches advertised data movement support.
- [x] Verify Provider behavior matches advertised dtype support.
- [x] Verify Provider behavior matches advertised layout support.
- [x] Verify Provider behavior matches advertised Device support.
- [x] Fail conformance when advertisement and behavior diverge.

## 24. Conformance CLI Or Test Command

- [x] Define how to run conformance suite locally.
- [x] Provide a cargo test profile or test command.
- [x] Allow selecting Provider target.
- [x] Allow selecting conformance profiles.
- [x] Allow report output path.
- [x] Document examples.

## 25. CI Integration

- [x] Run core Provider conformance in CI.
- [x] Run Compute conformance in CI for mock or CPU Provider.
- [x] Run dynamic ABI fixture conformance where platform supports it.
- [x] Do not require real GPU in default CI.
- [x] Upload or print conformance report where useful.
- [x] Fail CI when required conformance profile fails.

## 26. Documentation

- [x] Document Provider Conformance Suite purpose.
- [x] Document conformance target.
- [x] Document profiles.
- [x] Document required tests per profile.
- [x] Document report format.
- [x] Document hardware-independent behavior.
- [x] Document optional hardware profiles.
- [x] Document non-conformant Provider behavior.
- [x] Document how to run the suite.

## 27. Versioning

- [x] Define conformance suite version.
- [x] Include conformance suite version in report.
- [x] Document version compatibility policy.
- [x] Document that passing one suite version does not imply passing future
      versions.
- [x] Tie Provider compatibility documentation to suite version.

## 28. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run Provider conformance suite.
- [x] Run Provider loading tests.
- [x] Run Compute conformance profile.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify conformance report generation.
- [x] Verify non-conformant Provider fails.
- [x] Verify default CI does not require GPU.
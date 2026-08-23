# Tasks

## 1. Test Inventory

- [x] Inventory existing unit tests.
- [x] Inventory existing integration tests.
- [x] Inventory existing WIT validation tests.
- [x] Inventory existing Component Runtime tests.
- [x] Inventory existing Provider tests.
- [x] Inventory existing Resolution tests.
- [x] Inventory existing Resource Affinity tests.
- [x] Inventory existing Artifact trust tests.
- [x] Inventory existing Provider loading tests.
- [x] Identify critical untested failure paths.

## 2. Test Organization

- [x] Create or update integration test modules.
- [x] Group Provider tests under Provider-related modules.
- [x] Group Runtime orchestration tests under Runtime-related modules.
- [x] Group Component Runtime tests under Component-related modules.
- [x] Group Artifact trust tests under Component Artifact modules.
- [x] Group Distribution tests under Component distribution modules.
- [x] Group Resolution tests under Resolution modules.
- [x] Group Resource Affinity tests under Affinity modules.
- [x] Avoid one oversized miscellaneous test file.

## 3. Mock Provider Infrastructure

- [x] Add configurable mock Provider.
- [x] Support metadata configuration.
- [x] Support Capability advertisement configuration.
- [x] Support Device metadata configuration.
- [x] Support health configuration.
- [x] Support readiness configuration.
- [x] Support pressure configuration.
- [x] Support admission refusal.
- [x] Support execution success.
- [x] Support execution failure.
- [x] Support cancellation unsupported.
- [x] Support cancellation failure.
- [x] Support draining state.
- [x] Support stale status.

## 4. Mock Device Infrastructure

- [x] Add configurable mock Devices.
- [x] Support available Device.
- [x] Support unavailable Device.
- [x] Support degraded Device.
- [x] Support saturated Device.
- [x] Support memory pressure.
- [x] Support Device interruption.
- [x] Support Device-bound resources.

## 5. Mock ComponentEngine Infrastructure

- [x] Add engine-neutral mock ComponentEngine.
- [x] Simulate preparation success.
- [x] Simulate preparation failure.
- [x] Simulate instantiation success.
- [x] Simulate instantiation failure.
- [x] Simulate trap.
- [x] Simulate interruption.
- [x] Simulate resource-limit failure.
- [x] Simulate multiple isolated instances.
- [x] Simulate destroyed instance.

## 6. Mock Distribution Source

- [x] Add fake local distribution source.
- [x] Add fake client-provided distribution source.
- [x] Add fake Tachyon-labelled source metadata.
- [x] Support digest mismatch.
- [x] Support manifest mismatch.
- [x] Support missing artifact.
- [x] Support revoked artifact.
- [x] Support cache corruption.

## 7. Provider Registry Tests

- [x] Test Provider registration success.
- [x] Test duplicate ProviderId rejection or explicit duplicate policy.
- [x] Test invalid metadata rejection.
- [x] Test malformed Capability advertisement rejection.
- [x] Test malformed Device metadata rejection.
- [x] Test Provider unavailable after failed handshake.
- [x] Test Provider Registry contains only successfully registered Providers.

## 8. Provider Status Tests

- [x] Test healthy but not-ready.
- [x] Test degraded but ready.
- [x] Test healthy but saturated.
- [x] Test draining Provider.
- [x] Test stale Provider status.
- [x] Test Device unavailable while Provider healthy.
- [x] Test Capability not-ready while Provider ready.
- [x] Test Provider refusal versus execution failure.
- [x] Test Provider pressure diagnostics.

## 9. Resolution Policy Tests

- [x] Test ready Provider selected over not-ready Provider.
- [x] Test failed Provider rejected.
- [x] Test degraded Provider penalized by policy.
- [x] Test saturated Provider rejected unless queueing allowed.
- [x] Test low-pressure Provider preferred.
- [x] Test draining Provider avoided for new unpinned work.
- [x] Test stale status rejected or penalized by default.
- [x] Test Device-level status affects selection.
- [x] Test Capability-level status affects selection.
- [x] Test Resolution diagnostics include status rejection reasons.

## 10. Resource Affinity Tests

- [x] Test Provider-bound resource remains bound.
- [x] Test Device-bound resource remains bound.
- [x] Test Capability-bound resource remains bound.
- [x] Test Artifact-bound resource remains bound.
- [x] Test Resolution Policy cannot override hard Resource Affinity.
- [x] Test explicit transfer required for incompatible placement.
- [x] Test draining does not silently migrate Provider-owned resource.
- [x] Test forged affinity input rejected.
- [x] Test incompatible affinity returns structured error.

## 11. Compute Boundary Tests

- [x] Test Compute WIT request surface has no Provider target field.
- [x] Test Compute WIT request surface has no Device target field.
- [x] Test Compute WIT request surface has no AffinityGroup target field.
- [x] Test placement intent is portable.
- [x] Test `preserve-source-affinity`.
- [x] Test `runtime-selected`.
- [x] Test `host-accessible`.
- [x] Test host staging forbidden.
- [x] Test host staging permitted but policy denied.
- [x] Test diagnostics may contain selected Provider as output metadata only.
- [x] Test diagnostics may contain selected Device as output metadata only.

## 12. Compute Version Tests

- [x] Test Compute v1.1 does not automatically satisfy Compute v2.0.
- [x] Test Provider advertising only v1.1 rejected for v2 request.
- [x] Test Provider advertising v2 accepted for v2 request.
- [x] Test incompatible Compute major version rejection.
- [x] Test version mismatch diagnostics.

## 13. Execution Planning Tests

- [x] Test valid ComputeExecutionPlan creation.
- [x] Test invalid graph rejected before Provider execution.
- [x] Test unsupported operation family rejected.
- [x] Test unsupported dtype rejected.
- [x] Test unsupported layout rejected.
- [x] Test memory planning failure.
- [x] Test explicit data movement planning.
- [x] Test execution plan stores resolved Provider and Device.
- [x] Test Scheduler consumes plan instead of resolving Provider.

## 14. Scheduler Tests

- [x] Test Scheduler admits valid resolved plan.
- [x] Test selected Provider becomes not-ready before submission.
- [x] Test selected Provider becomes saturated before submission.
- [x] Test selected Device becomes unavailable before submission.
- [x] Test Scheduler retry policy.
- [x] Test Scheduler fail policy.
- [x] Test Scheduler queue policy where applicable.
- [x] Test Scheduler does not independently choose another Provider.
- [x] Test operation cancellation before submission.
- [x] Test operation cancellation during execution.

## 15. Provider Execution Failure Tests

- [x] Test execution success.
- [x] Test execution rejected because Provider not-ready.
- [x] Test execution rejected because Provider draining.
- [x] Test execution rejected because Provider saturated.
- [x] Test execution failed after submission.
- [x] Test Provider panic or internal failure normalized.
- [x] Test cancellation unsupported.
- [x] Test cancellation failure.
- [x] Test output retrieval failure.
- [x] Test Provider-owned resource invalid.

## 16. Provider Loading ABI Tests

- [x] Test missing factory symbol.
- [x] Test unsupported ABI version.
- [x] Test malformed ABI descriptor.
- [x] Test missing metadata function.
- [x] Test missing status function.
- [x] Test missing execution function.
- [x] Test invalid metadata.
- [x] Test invalid Capability advertisement.
- [x] Test invalid Device metadata.
- [x] Test dynamic Provider rejected before registration.
- [x] Test trait-object factory is not stable ABI path.
- [x] Test loading policy rejects disallowed library path.

## 17. Component Runtime Tests

- [x] Test valid Component import validation.
- [x] Test missing import rejection.
- [x] Test unauthorized import rejection.
- [x] Test Link Plan construction.
- [x] Test Capability linking without Provider pinning.
- [x] Test no ambient filesystem.
- [x] Test no ambient network.
- [x] Test no ambient secrets.
- [x] Test no ambient Git.
- [x] Test no broad WASI by default.
- [x] Test multiple Component instances are isolated.
- [x] Test invocation after destruction fails.

## 18. WASM Component Fixture Tests

- [x] Test valid WASM Component fixture preparation.
- [x] Test valid WASM Component fixture instantiation.
- [x] Test valid WASM Component fixture invocation.
- [x] Test fixture with missing import fails.
- [x] Test fixture with unauthorized import fails.
- [x] Test trapping fixture normalizes error.
- [x] Test interruption fixture where feasible.
- [x] Test no ambient WASI fixture.

## 19. Component Artifact Tests

- [x] Test digest validation success.
- [x] Test digest mismatch rejection.
- [x] Test manifest parse failure.
- [x] Test manifest missing required field.
- [x] Test unsupported manifest version.
- [x] Test WIT import mismatch.
- [x] Test WIT export mismatch.
- [x] Test Runtime compatibility failure.
- [x] Test Capability compatibility failure.
- [x] Test untrusted artifact rejected.
- [x] Test revoked artifact rejected.
- [x] Test quarantined artifact rejected.
- [x] Test development mode remains explicit.

## 20. Inference Authority Tests

- [x] Test `model-artifact-read` accepted when trusted.
- [x] Test `tokenizer-artifact-read` accepted when trusted.
- [x] Test `compute-capability` accepted when trusted.
- [x] Test `generation-capability` accepted where supported.
- [x] Test `observability-emit` accepted when trusted.
- [x] Test `filesystem` authority rejected.
- [x] Test `network` authority rejected.
- [x] Test `secrets` authority rejected.
- [x] Test `git` authority rejected.
- [x] Test `workspace` authority rejected.
- [x] Test `process` authority rejected.
- [x] Test trusted digest cannot override forbidden authority.
- [x] Test development mode cannot override forbidden authority.

## 21. Distribution Tests

- [x] Test local-directory package validation.
- [x] Test client-provided package validation.
- [x] Test Tachyon-labelled source metadata does not imply trust.
- [x] Test source-declared digest mismatch.
- [x] Test distributed manifest mismatch.
- [x] Test distributed WIT mismatch.
- [x] Test distributed forbidden authority rejection.
- [x] Test distributed revoked artifact rejection.
- [x] Test cache hit with digest verification.
- [x] Test cache corruption rejection.
- [x] Test offline distribution path.

## 22. Observability Isolation Tests

- [x] Test observability sink failure does not fail Compute execution.
- [x] Test observability sink saturation follows policy.
- [x] Test observability export failure does not alter trust decision.
- [x] Test observability failure does not alter Provider status.
- [x] Test observability cannot grant network authority.
- [x] Test diagnostic redaction.
- [x] Test no native handles in public observations.

## 23. Cancellation and Shutdown Tests

- [x] Test Component invocation cancellation.
- [x] Test Compute operation cancellation.
- [x] Test Provider cancellation unsupported.
- [x] Test Provider cancellation failure.
- [x] Test Runtime shutdown prevents new work.
- [x] Test Runtime shutdown drains or interrupts active work according to policy.
- [x] Test Component instance destruction on shutdown.
- [x] Test Provider drain on shutdown.
- [x] Test resources released according to ownership.

## 24. Concurrency Tests

- [x] Test concurrent Resolution requests.
- [x] Test concurrent Provider status updates.
- [x] Test concurrent Component instances.
- [x] Test same Component instance serialization where required.
- [x] Test Scheduler admission race with Provider readiness change.
- [x] Test cancellation race with completion.
- [x] Test artifact cache concurrent access where cache exists.

## 25. Property or Table-Driven Tests

- [x] Add table-driven tests for Provider status combinations.
- [x] Add table-driven tests for authority validation.
- [x] Add table-driven tests for Compute version compatibility.
- [x] Add table-driven tests for trust policy precedence.
- [x] Add property tests for version range matching if practical.
- [x] Add property tests for digest normalization if practical.

## 26. Fixture Policy

- [x] Ensure fixtures are deterministic.
- [x] Ensure fixtures do not require external network.
- [x] Ensure fixtures do not require real GPU hardware.
- [x] Ensure fixtures do not require Tachyon.
- [x] Document fixture build process.
- [x] Check in fixtures or build them reproducibly in CI.
- [x] Keep malicious/invalid fixtures clearly separated.

## 27. CI Integration

- [x] Ensure new tests run in CI.
- [x] Ensure feature-gated WASM tests run in at least one CI job.
- [x] Ensure Provider ABI tests run in CI where platform supports dynamic libs.
- [x] Skip platform-specific tests explicitly with explanation where needed.
- [x] Ensure coverage includes moved modular Runtime code.
- [x] Ensure failures are deterministic.

## 28. Coverage

- [x] Measure coverage before adding tests.
- [x] Measure coverage after adding tests.
- [x] Identify remaining low-coverage critical modules.
- [x] Add focused tests for low-coverage failure paths.
- [x] Do not increase coverage by testing trivial getters only.
- [x] Preserve existing CI coverage ratchet.
- [x] Raise coverage threshold where justified.

## 29. Documentation

- [x] Document test strategy.
- [x] Document mock Provider usage.
- [x] Document mock ComponentEngine usage.
- [x] Document fixture build process.
- [x] Document failure injection utilities.
- [x] Document how to run feature-gated tests locally.
- [x] Document how tests map to architecture invariants.

## 30. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run Clippy.
- [x] Run all unit tests.
- [x] Run all integration tests.
- [x] Run WASM Component tests.
- [x] Run Provider ABI/loading tests.
- [x] Run artifact trust tests.
- [x] Run distribution tests.
- [x] Run WIT validation.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify architecture invariants are tested.
- [x] Verify failure paths are tested.
- [x] Verify no broad authority is introduced.

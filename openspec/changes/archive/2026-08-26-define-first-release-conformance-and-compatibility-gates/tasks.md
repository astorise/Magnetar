# Tasks

## 1. Gate Scope

- [x] Define first release gate categories.
- [x] Define required gates.
- [x] Define optional gates.
- [x] Define allowed skips.
- [x] Define disallowed skips.
- [x] Define release-blocking failures.
- [x] Define reports.
- [x] Define compatibility matrix.

## 2. Source And Build Gates

- [x] Verify repository state is clean or CI-controlled.
- [x] Verify release tag matches source.
- [x] Verify lockfile reviewed.
- [x] Verify dependency tree resolves.
- [x] Run formatting.
- [x] Run compilation.
- [x] Run Clippy or equivalent linting.
- [x] Run unit tests.
- [x] Run doc tests where applicable.
- [x] Run feature matrix checks.
- [x] Run supported target checks.

## 3. OpenSpec Gates

- [x] Validate OpenSpec.
- [x] List accepted changes.
- [x] Ensure pending changes not included accidentally.
- [x] Declare release baseline.
- [x] Verify release checklist references correct changes.
- [x] Verify no semantic change after freeze.
- [x] Verify no hidden scope expansion.
- [x] Verify specs are internally consistent.
- [x] Block release on failure.

## 4. WIT Gates

- [x] Validate included WIT packages.
- [x] Verify WIT package versions declared.
- [x] Verify breaking WIT changes have major version bump.
- [x] Verify supported WIT version matrix documented.
- [x] Verify generated bindings compile where applicable.
- [x] Verify WIT world boundaries match Runtime contracts.
- [x] Verify Provider/Device targeting not exposed to Components.
- [x] Verify no raw handles exposed through WIT.
- [x] Block release on failure.

## 5. Rust Public API Gates

- [x] Audit exported types.
- [x] Verify public IDs are opaque.
- [x] Verify internal modules not accidentally exported.
- [x] Verify experimental APIs are feature-gated.
- [x] Verify unstable APIs are marked.
- [x] Verify no raw Provider handles exposed.
- [x] Verify no raw Device handles exposed.
- [x] Verify no raw Kernel handles exposed.
- [x] Verify no raw tensor pointers exposed.
- [x] Verify no raw memory pointers exposed.
- [x] Verify no raw KV cache contents exposed.
- [x] Verify no raw model weights exposed.
- [x] Block release on public API safety failure.

## 6. Runtime Contract Gates

- [x] Verify no arbitrary filesystem authority.
- [x] Verify no arbitrary network authority.
- [x] Verify no secret authority.
- [x] Verify no shell/process execution.
- [x] Verify no Git execution.
- [x] Verify no tool execution.
- [x] Verify no agent orchestration.
- [x] Verify Runtime calls stay inference-scoped.
- [x] Verify Runtime rejects boundary violations.
- [x] Block release on Runtime boundary failure.

## 7. Runtime Inference API Gates

- [x] Test model resolution request behavior.
- [x] Test model loading request behavior.
- [x] Test session creation.
- [x] Test session close.
- [x] Test one-shot inference if included.
- [x] Test tokenization API.
- [x] Test generation API.
- [x] Test streaming API.
- [x] Test cancellation API.
- [x] Test diagnostics API.
- [x] Test usage reporting.
- [x] Test structured error preservation.
- [x] Test redaction by default.
- [x] Test no internal handle exposure.

## 8. Provider Gates

- [x] Validate Provider identity.
- [x] Validate Provider status snapshot.
- [x] Validate Device metadata.
- [x] Validate Kernel advertisements.
- [x] Validate readiness/health/pressure separation.
- [x] Validate no native handle exposure.
- [x] Validate Provider execution goes through Kernel Dispatch.
- [x] Validate Provider structured errors.
- [x] Mark non-Reference Providers skipped if out of scope.

## 9. Reference CPU Gates

- [x] Verify Reference CPU Provider registers.
- [x] Verify CPU Device metadata exists.
- [x] Verify required-now Kernels advertised.
- [x] Verify host contiguous f32 path works.
- [x] Verify deterministic fixtures pass.
- [x] Verify correctness prioritized over performance.
- [x] Verify Reference CPU remains baseline.
- [x] Block release on failure.

## 10. Operator First Scope Gates

- [x] Verify embedding coverage.
- [x] Verify matmul coverage.
- [x] Verify RMSNorm coverage.
- [x] Verify RoPE coverage.
- [x] Verify attention coverage.
- [x] Verify softmax coverage.
- [x] Verify SiLU coverage.
- [x] Verify add coverage.
- [x] Verify mul coverage.
- [x] Verify residual-add coverage.
- [x] Verify dtype-conversion coverage where included.
- [x] Verify layout-conversion coverage where included.
- [x] Block release on missing required-now operator coverage.

## 11. Tensor And Memory Gates

- [x] Validate TensorDescriptor behavior.
- [x] Validate TensorResource lifecycle.
- [x] Validate TensorLayout metadata.
- [x] Validate dtype metadata.
- [x] Validate shape metadata.
- [x] Validate host contiguous layout.
- [x] Validate Memory Manager allocation tracking.
- [x] Validate size accounting.
- [x] Validate readiness state.
- [x] Validate cleanup.
- [x] Validate no raw pointer exposure.
- [x] Validate cache storage distinct from memory residency.

## 12. Kernel Registry And Dispatch Gates

- [x] Validate Kernel advertisement validation.
- [x] Validate Kernel candidate lookup.
- [x] Validate candidate filtering.
- [x] Validate Resource Affinity validation.
- [x] Validate Memory Manager feasibility.
- [x] Validate Provider readiness checks.
- [x] Validate Device readiness checks.
- [x] Validate dispatch revalidation.
- [x] Validate structured missing-kernel errors.
- [x] Validate no direct Provider bypass.
- [x] Validate no silent fallback.

## 13. Model Artifact And Loading Gates

- [x] Validate fixture Model Artifact manifest.
- [x] Validate artifact identity.
- [x] Validate trust validation.
- [x] Validate integrity validation.
- [x] Validate tensor inventory validation.
- [x] Validate tokenizer compatibility.
- [x] Validate Model Loading lifecycle.
- [x] Validate Model Instance readiness.
- [x] Validate unload cleanup.
- [x] Validate cache hit does not imply trust.
- [x] Validate recognized format does not imply trust.

## 14. Qwen Baseline Gates

- [x] Validate Qwen config.
- [x] Validate tensor inventory.
- [x] Validate tokenizer compatibility metadata.
- [x] Validate target module metadata.
- [x] Validate prefill graph production.
- [x] Validate decode graph production.
- [x] Validate required operator use.
- [x] Validate no QwenProvider.
- [x] Validate no direct Provider access.
- [x] Validate no direct Kernel access.

## 15. Generation And Sampling Gates

- [x] Validate prefill orchestration.
- [x] Validate decode orchestration.
- [x] Validate greedy sampling.
- [x] Validate stop conditions.
- [x] Validate max new tokens.
- [x] Validate max total tokens.
- [x] Validate finish reason.
- [x] Validate usage accounting.
- [x] Validate cancellation checkpoints.
- [x] Validate streaming events.
- [x] Validate structured generation errors.

## 16. CLI Boundary Gates

- [x] Verify CLI calls Runtime Inference API.
- [x] Verify CLI sends explicit prompt/context.
- [x] Verify Runtime receives no ambient filesystem authority.
- [x] Verify Runtime receives no ambient Git authority.
- [x] Verify Runtime receives no ambient network authority.
- [x] Verify Runtime receives no ambient secret authority.
- [x] Verify Runtime receives no ambient tool authority.
- [x] Verify Runtime receives no ambient shell/process authority.
- [x] Verify Runtime structured errors are preserved.
- [x] Verify CLI diagnostics are redacted.

## 17. E2E Local Inference Gates

- [x] Validate Runtime Inference API entrypoint.
- [x] Validate model resolution.
- [x] Validate model loading.
- [x] Validate Model Instance.
- [x] Validate session.
- [x] Validate tokenizer.
- [x] Validate generation.
- [x] Validate graph production.
- [x] Validate operator validation.
- [x] Validate Kernel Registry.
- [x] Validate Reference CPU dispatch.
- [x] Validate streaming/result.
- [x] Validate cleanup.
- [x] Verify CPU-only.
- [x] Verify local-only.
- [x] Verify deterministic.
- [x] Verify no shortcut.
- [x] Block release on E2E failure.

## 18. Observability And Redaction Gates

- [x] Verify raw prompts absent.
- [x] Verify secrets absent.
- [x] Verify credentials absent.
- [x] Verify raw file contents absent.
- [x] Verify raw model weights absent.
- [x] Verify raw tensor values absent.
- [x] Verify raw KV cache contents absent.
- [x] Verify Provider handles absent.
- [x] Verify Device handles absent.
- [x] Verify Kernel handles absent.
- [x] Verify memory pointers absent.
- [x] Verify raw cache paths absent by default.
- [x] Block release on redaction failure.

## 19. Security And Supply-Chain Gates

- [x] Run dependency audit.
- [x] Run license audit.
- [x] Run secret scanning.
- [x] Generate SBOM or document limitation.
- [x] Generate checksums.
- [x] Generate provenance.
- [x] Review lockfile.
- [x] Review unsafe code.
- [x] Validate artifact integrity.
- [x] Include security notes.
- [x] Document security exceptions.
- [x] Block release on security-blocking failure.

## 20. Compatibility Matrix

- [x] Mark Rust public API status.
- [x] Mark Runtime Inference API status.
- [x] Mark WIT package status.
- [x] Mark Provider ABI status.
- [x] Mark Model Artifact metadata status.
- [x] Mark Tokenizer Artifact metadata status.
- [x] Mark Adapter Artifact metadata status.
- [x] Mark CLI command surface status.
- [x] Mark conformance report format status.
- [x] Mark OpenSpec baseline status.
- [x] Mark supported targets.
- [x] Mark feature flags.
- [x] Use stable-for-v0.1-baseline / preview / experimental / unstable / deferred / unsupported.

## 21. Allowed Skips

- [x] Allow CUDA Provider conformance skip.
- [x] Allow Metal Provider conformance skip.
- [x] Allow OpenVINO Provider conformance skip.
- [x] Allow QNN Provider conformance skip.
- [x] Allow WebGPU Provider conformance skip.
- [x] Allow server API conformance skip.
- [x] Allow model hub source conformance skip.
- [x] Allow production model format conformance skip.
- [x] Record skip reasons.
- [x] Ensure skip does not hide baseline failure.

## 22. Disallowed Skips

- [x] Disallow OpenSpec validation skip.
- [x] Disallow included WIT validation skip.
- [x] Disallow formatting/check/lint skip.
- [x] Disallow required unit/contract test skip.
- [x] Disallow Reference CPU conformance skip.
- [x] Disallow Runtime Inference API baseline skip.
- [x] Disallow CLI boundary test skip.
- [x] Disallow E2E local conformance skip.
- [x] Disallow redaction gate skip.
- [x] Disallow release security gate skip.
- [x] Disallow artifact integrity gate skip.
- [x] Disallow release documentation checklist skip.

## 23. Exception Policy

- [x] Define exception template.
- [x] Require gate name.
- [x] Require failure/deviation.
- [x] Require severity.
- [x] Require affected component.
- [x] Require rationale.
- [x] Require mitigation.
- [x] Require owner.
- [x] Require expiration/follow-up.
- [x] Require release note entry.
- [x] Block undocumented exceptions.

## 24. Release Reports

- [x] Produce machine-readable report.
- [x] Produce human-readable report.
- [x] Include gate name.
- [x] Include gate category.
- [x] Include status.
- [x] Include duration.
- [x] Include version.
- [x] Include target.
- [x] Include feature set.
- [x] Include skip reason.
- [x] Include failure reason.
- [x] Include exception reference.
- [x] Include artifact checksum.
- [x] Include redaction status.
- [x] Ensure reports are redacted.

## 25. Release Candidate Validation

- [x] Run same required gates for release candidates.
- [x] Allow known failures only if documented.
- [x] Mark release candidate as pre-release.
- [x] Block stable release until failures resolved or accepted exception exists.
- [x] Add release candidate report.

## 26. Stable Release Cutover

- [x] Verify all required gates pass.
- [x] Verify allowed skips documented.
- [x] Verify exceptions documented.
- [x] Verify release reports generated.
- [x] Verify compatibility matrix complete.
- [x] Verify release notes complete.
- [x] Verify release artifacts checksummed.
- [x] Verify security notes complete.

## 27. Documentation

- [x] Document release gates.
- [x] Document conformance gates.
- [x] Document compatibility matrix.
- [x] Document allowed skips.
- [x] Document disallowed skips.
- [x] Document exception policy.
- [x] Document release reports.
- [x] Document release candidate process.
- [x] Document stable cutover rules.

## 28. Final Validation

- [x] Run OpenSpec validation.
- [x] Verify all required gates are listed.
- [x] Verify all baseline paths are covered.
- [x] Verify no required baseline gate is skippable.
- [x] Verify compatibility matrix statuses are defined.
- [x] Verify release-blocking policy is explicit.
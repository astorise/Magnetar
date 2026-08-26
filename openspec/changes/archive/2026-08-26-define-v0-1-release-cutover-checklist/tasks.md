# Tasks

## 1. Release Readiness

- [x] Select release branch or commit.
- [x] Select release version.
- [x] Select OpenSpec baseline.
- [x] Confirm release scope.
- [x] Confirm release gates.
- [x] Confirm release artifacts.
- [x] Draft release notes.
- [x] Draft compatibility matrix.
- [x] Draft security notes.

## 2. OpenSpec Freeze

- [x] Freeze accepted changes list.
- [x] Confirm pending changes are not included.
- [x] Confirm no semantic change after freeze.
- [x] Confirm WIT-breaking changes have version bumps.
- [x] Confirm release checklist references correct changes.
- [x] Confirm roadmap items remain deferred unless explicitly included.
- [x] Record freeze status.

## 3. Scope Confirmation

- [x] Confirm Runtime Inference API baseline included.
- [x] Confirm Model Loading baseline included.
- [x] Confirm Model Instance baseline included.
- [x] Confirm Tokenizer fixture path included.
- [x] Confirm Qwen-like baseline fixture path included.
- [x] Confirm Generation and Sampling baseline included.
- [x] Confirm Tensor and Memory baseline included.
- [x] Confirm Operator first scope included.
- [x] Confirm Kernel Registry and Dispatch included.
- [x] Confirm Reference CPU Provider included.
- [x] Confirm CLI boundary harness included.
- [x] Confirm E2E local inference conformance included.
- [x] Confirm CUDA Provider excluded or experimental.
- [x] Confirm Metal Provider excluded or experimental.
- [x] Confirm OpenVINO Provider excluded or experimental.
- [x] Confirm QNN Provider excluded or experimental.
- [x] Confirm WebGPU Provider excluded or experimental.
- [x] Confirm server API implementation excluded.
- [x] Confirm model hub UX excluded.
- [x] Confirm agent/tool runtime excluded.
- [x] Confirm production large model support excluded.

## 4. Version Confirmation

- [x] Confirm release version.
- [x] Confirm crate versions.
- [x] Confirm binary version.
- [x] Confirm WIT package versions.
- [x] Confirm conformance suite versions.
- [x] Confirm OpenSpec baseline version.
- [x] Confirm release candidate lineage where applicable.
- [x] Add version metadata to release report.

## 5. Feature Flag Confirmation

- [x] Confirm default features.
- [x] Confirm Reference CPU required feature.
- [x] Confirm experimental features disabled by default.
- [x] Confirm Provider-specific features outside Reference CPU disabled, absent, or experimental.
- [x] Confirm test-only features not enabled in release build.
- [x] Confirm conformance-only features not enabled in default release build.
- [x] Add feature flag summary.

## 6. Compatibility Matrix

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
- [x] Use approved status vocabulary.
- [x] Verify matrix completeness.

## 7. Required Gate Execution

- [x] Run source/build gates.
- [x] Run OpenSpec gates.
- [x] Run WIT gates.
- [x] Run Rust public API gates.
- [x] Run Runtime contract gates.
- [x] Run Runtime Inference API gates.
- [x] Run Provider gates.
- [x] Run Reference CPU gates.
- [x] Run Operator first scope gates.
- [x] Run Tensor/Memory gates.
- [x] Run Kernel Registry/Dispatch gates.
- [x] Run Model Loading gates.
- [x] Run Qwen baseline gates.
- [x] Run Generation/Sampling gates.
- [x] Run CLI boundary gates.
- [x] Run E2E local inference gates.
- [x] Run observability/redaction gates.
- [x] Run security/supply-chain gates.
- [x] Run release artifact gates.

## 8. Skip Review

- [x] List skipped gates.
- [x] Confirm each skip is outside `v0.1` scope.
- [x] Confirm each skip has reason.
- [x] Confirm no skip hides baseline failure.
- [x] Confirm disallowed skips are absent.
- [x] Include skips in release report.

## 9. Exception Review

- [x] List exceptions.
- [x] Confirm gate name for each exception.
- [x] Confirm issue for each exception.
- [x] Confirm severity for each exception.
- [x] Confirm affected component.
- [x] Confirm rationale.
- [x] Confirm mitigation.
- [x] Confirm owner.
- [x] Confirm expiration or follow-up.
- [x] Confirm release note entry.
- [x] Block release on undocumented exception.

## 10. Security Verification

- [x] Confirm dependency audit completed.
- [x] Confirm license audit completed.
- [x] Confirm secret scan completed.
- [x] Confirm redaction gates passed.
- [x] Confirm native handle checks passed.
- [x] Confirm trust/integrity checks passed.
- [x] Confirm artifact integrity checks passed.
- [x] Confirm checksums generated.
- [x] Confirm SBOM generated or limitation documented.
- [x] Confirm signature status documented.
- [x] Confirm security notes complete.

## 11. Artifact Generation

- [x] Generate source archive.
- [x] Generate Rust crate artifacts where applicable.
- [x] Generate CLI binary where applicable.
- [x] Generate OpenSpec validation report.
- [x] Generate WIT validation report.
- [x] Generate conformance report.
- [x] Generate E2E local inference report.
- [x] Generate release security report.
- [x] Generate compatibility matrix.
- [x] Generate coverage report.
- [x] Generate SBOM or SBOM limitation.
- [x] Generate checksums.
- [x] Generate changelog.
- [x] Generate release notes.
- [x] Mark non-applicable artifacts explicitly.

## 12. Artifact Verification

- [x] Verify checksums generated from final artifacts.
- [x] Verify reports match release commit.
- [x] Verify release notes match compatibility matrix.
- [x] Verify OpenSpec baseline matches tag.
- [x] Verify conformance versions match reports.
- [x] Verify no local paths in artifacts unless explicitly allowed.
- [x] Verify no secrets in artifacts.
- [x] Verify artifact names include version where appropriate.

## 13. Changelog Completion

- [x] Include added contracts.
- [x] Include changed contracts.
- [x] Include removed/deprecated contracts.
- [x] Include release scope.
- [x] Include known limitations.
- [x] Include compatibility status.
- [x] Include security notes.
- [x] Include conformance status.
- [x] Include deferred roadmap items.

## 14. Release Notes Completion

- [x] Explain what `v0.1` is.
- [x] Explain what users can run.
- [x] Explain stable-for-v0.1 status.
- [x] Explain preview status.
- [x] Explain experimental status.
- [x] Explain deferred status.
- [x] Explain unsupported status.
- [x] Explain artifact verification.
- [x] Explain security limitations.
- [x] Explain how to run conformance.
- [x] Include compatibility matrix.
- [x] Include security notes.
- [x] Include known limitations.

## 15. Tagging

- [x] Create semantic version tag only after gates pass.
- [x] Tag release commit.
- [x] Reference OpenSpec baseline.
- [x] Reference artifact checksums.
- [x] Reference release notes.
- [x] Document signed tag status where applicable.

## 16. Publication

- [x] Publish source release.
- [x] Publish crates where applicable.
- [x] Publish binaries where applicable.
- [x] Publish reports.
- [x] Publish documentation.
- [x] Publish release notes.
- [x] Verify deferred roadmap features are not presented as included.
- [x] Verify preview/experimental APIs are not presented as stable.

## 17. Post-Publication Verification

- [x] Verify published artifacts match checksums.
- [x] Verify release notes are visible.
- [x] Verify reports are accessible.
- [x] Verify version command reports expected version.
- [x] Verify documentation links are valid.
- [x] Verify compatibility matrix is visible.
- [x] Verify security notes are visible.
- [x] Verify deferred roadmap clearly separated.

## 18. Rollback And Retraction

- [x] Document how to mark release withdrawn.
- [x] Document how to publish advisory.
- [x] Document how to cut patch release.
- [x] Document how to preserve audit trail.
- [x] Document how to update release notes.

## 19. Post-v0.1 Handoff

- [x] Identify implementation hardening work.
- [x] Identify optimized CPU Provider work.
- [x] Identify model format support work.
- [x] Identify source/cache implementation work.
- [x] Identify server API implementation work.
- [x] Identify production CLI UX work.
- [x] Identify GPU Provider exploration work.
- [x] Identify quantized inference work.
- [x] Identify advanced attention work.
- [x] Confirm post-v0.1 items are not release claims.

## 20. Final Release Statement

- [x] Include final release statement.
- [x] State CPU-local inference runtime baseline.
- [x] State Runtime Inference API validation.
- [x] State Reference CPU Provider validation.
- [x] State E2E local conformance status.
- [x] State post-baseline roadmap features excluded unless marked.

## 21. Final Validation

- [x] Verify readiness complete.
- [x] Verify OpenSpec frozen.
- [x] Verify scope confirmed.
- [x] Verify versions confirmed.
- [x] Verify feature flags confirmed.
- [x] Verify compatibility matrix complete.
- [x] Verify required gates passed.
- [x] Verify skips reviewed.
- [x] Verify exceptions reviewed.
- [x] Verify security verified.
- [x] Verify artifacts generated.
- [x] Verify artifacts verified.
- [x] Verify changelog complete.
- [x] Verify release notes complete.
- [x] Verify tag created after gates.
- [x] Verify publication verified.
- [x] Verify post-v0.1 handoff complete.
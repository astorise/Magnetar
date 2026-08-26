# Tasks

## 1. Release Scope

- [x] Define `v0.1` release scope.
- [x] Include CPU-local baseline.
- [x] Exclude GPU Providers from required scope.
- [x] Exclude production Qwen support from required scope.
- [x] Exclude model hub downloads from required scope.
- [x] Exclude server API implementation from required scope.
- [x] Exclude agent/tool runtime from required scope.
- [x] Document known limitations.

## 2. Versioning Policy

- [x] Define semantic versioning policy.
- [x] Define pre-1.0 breaking change policy.
- [x] Define patch version expectations.
- [x] Define release candidate version policy.
- [x] Add versioning documentation.

## 3. Crate Versioning

- [x] Decide workspace version strategy.
- [x] Define crate version metadata.
- [x] Define internal crate compatibility.
- [x] Define dependency compatibility policy.
- [x] Add crate version checks.

## 4. Binary Versioning

- [x] Add binary version reporting.
- [x] Report binary version.
- [x] Report runtime crate version.
- [x] Report OpenSpec baseline version.
- [x] Report WIT contract versions.
- [x] Report enabled feature flags.
- [x] Report build profile.
- [x] Report commit hash where available.
- [x] Report conformance suite version where available.
- [x] Add version command tests.

## 5. WIT Versioning

- [x] Declare WIT package versions.
- [x] Define breaking WIT change policy.
- [x] Define additive WIT change policy.
- [x] Define documentation-only WIT change policy.
- [x] Document supported WIT versions.
- [x] Add WIT version validation.

## 6. OpenSpec Baseline

- [x] Declare OpenSpec baseline version.
- [x] List accepted changes.
- [x] Record OpenSpec validation status.
- [x] Record compatibility notes.
- [x] Record deferred changes.
- [x] Record conformance status.
- [x] Attach release tag metadata.

## 7. Change Freeze

- [x] Define release freeze point.
- [x] Prevent semantic contract changes after freeze.
- [x] Prevent breaking WIT change without version bump.
- [x] Prevent release gate changes without checklist update.
- [x] Prevent hidden scope expansion.
- [x] Allow non-semantic documentation clarifications.
- [x] Add freeze checklist.

## 8. Feature Flags

- [x] Classify baseline feature flags.
- [x] Classify experimental feature flags.
- [x] Classify Provider-specific feature flags.
- [x] Classify platform-specific feature flags.
- [x] Classify test-only feature flags.
- [x] Classify conformance-only feature flags.
- [x] Ensure experimental features disabled by default.
- [x] Add feature flag tests.

## 9. Provider Feature Flags

- [x] Require Reference CPU Provider for baseline.
- [x] Keep optimized CPU optional or deferred.
- [x] Keep CUDA absent, disabled, or experimental.
- [x] Keep Metal absent, disabled, or experimental.
- [x] Keep OpenVINO absent, disabled, or experimental.
- [x] Keep QNN absent, disabled, or experimental.
- [x] Keep WebGPU absent, disabled, or experimental.
- [x] Add Provider feature checks.

## 10. Component Engine Feature Flags

- [x] Define Wasmtime component engine feature flag.
- [x] Define web component engine placeholder.
- [x] Define test component engine flag.
- [x] Ensure native Wasmtime is feature-gated where needed.
- [x] Ensure browser builds do not require Wasmtime.
- [x] Add component engine feature checks.

## 11. Platform Targets

- [x] Define supported native targets.
- [x] Define CI target set.
- [x] Define wasm32 check-only status if applicable.
- [x] Document unsupported targets.
- [x] Add platform build checks.

## 12. Release Artifacts

- [x] Produce source archive.
- [x] Produce Rust crate artifacts where applicable.
- [x] Produce CLI binary where applicable.
- [x] Produce conformance report.
- [x] Produce E2E report.
- [x] Produce OpenSpec validation report.
- [x] Produce coverage report.
- [x] Produce SBOM placeholder.
- [x] Produce checksums.
- [x] Produce changelog.
- [x] Produce release notes.

## 13. Checksums

- [x] Generate source archive checksum.
- [x] Generate binary checksum.
- [x] Generate report checksums where applicable.
- [x] Document checksum verification.
- [x] Ensure checksums do not replace trust policy.

## 14. Changelog

- [x] Record added contracts.
- [x] Record changed contracts.
- [x] Record removed/deprecated contracts.
- [x] Record fixed issues.
- [x] Record known limitations.
- [x] Record conformance status.
- [x] Record compatibility notes.
- [x] Record security notes.

## 15. Compatibility Policy

- [x] Define Rust public API compatibility status.
- [x] Define WIT compatibility status.
- [x] Define Runtime Inference API compatibility status.
- [x] Define Model Artifact metadata compatibility status.
- [x] Define Provider ABI compatibility status.
- [x] Define OpenSpec baseline compatibility status.
- [x] Define CLI command surface compatibility status.
- [x] Define conformance report format compatibility status.
- [x] Mark unstable areas explicitly.

## 16. Public API Safety

- [x] Verify no raw Provider handle exposure.
- [x] Verify no raw Device handle exposure.
- [x] Verify no raw Kernel handle exposure.
- [x] Verify no raw tensor pointer exposure.
- [x] Verify no raw memory pointer exposure.
- [x] Verify no raw KV cache exposure.
- [x] Verify no raw model weight exposure.
- [x] Add public API safety tests.

## 17. Conformance Versioning

- [x] Version Provider conformance suite.
- [x] Version first operator scope conformance.
- [x] Version Qwen baseline conformance.
- [x] Version Runtime Inference API conformance.
- [x] Version CLI boundary conformance.
- [x] Version E2E local conformance.
- [x] Include conformance versions in release metadata.

## 18. Release Gates

- [x] Run formatting gate.
- [x] Run cargo check gate.
- [x] Run Clippy gate.
- [x] Run unit test gate.
- [x] Run contract test gate.
- [x] Run OpenSpec validation gate.
- [x] Run WIT validation gate.
- [x] Run Reference CPU conformance gate.
- [x] Run Operator first scope conformance gate.
- [x] Run Runtime Inference API test gate.
- [x] Run CLI boundary test gate.
- [x] Run E2E local conformance gate.
- [x] Run coverage gate.
- [x] Run redaction gate.
- [x] Run no raw handle exposure gate.

## 19. Release Failure Policy

- [x] Prevent stable release if required gates fail.
- [x] Allow failed candidate only as marked pre-release.
- [x] Record known failures in release candidate notes.
- [x] Add release gate enforcement.

## 20. Release Candidate Policy

- [x] Define release candidate tag format.
- [x] Require frozen OpenSpec baseline.
- [x] Require conformance report.
- [x] Require known failure list.
- [x] Require release notes draft.
- [x] Prevent RC from being marked stable.

## 21. Build Metadata

- [x] Include commit hash where available.
- [x] Include build timestamp where policy allows.
- [x] Include target triple.
- [x] Include enabled features.
- [x] Include CI run identifier where available.
- [x] Include build profile.
- [x] Include rustc version.
- [x] Redact secrets and local paths.

## 22. Documentation Release

- [x] Publish architecture overview.
- [x] Publish Runtime Inference API overview.
- [x] Publish CLI boundary overview.
- [x] Publish build instructions.
- [x] Publish test instructions.
- [x] Publish conformance instructions.
- [x] Publish feature flag documentation.
- [x] Publish supported targets.
- [x] Publish known limitations.
- [x] Publish post-baseline roadmap.

## 23. Security Notes

- [x] Document sandbox assumptions.
- [x] Document Provider trust model.
- [x] Document no raw handle policy.
- [x] Document default redaction.
- [x] Document source/cache trust boundary.
- [x] Document unsupported security features.
- [x] Document known risks.
- [x] Link to separate security hardening change.

## 24. Publishing Boundary

- [x] Distinguish included baseline.
- [x] Distinguish experimental features.
- [x] Distinguish deferred roadmap.
- [x] Distinguish unsupported features.
- [x] Prevent roadmap features from appearing as release guarantees.
- [x] Add publishing checklist.

## 25. Final Validation

- [x] Run OpenSpec validation.
- [x] Verify release scope is CPU-local baseline.
- [x] Verify release does not require GPU Providers.
- [x] Verify release does not require server API.
- [x] Verify release does not require model hub download.
- [x] Verify all release gates are explicit.
- [x] Verify unstable APIs are marked.
- [x] Verify release artifacts are defined.
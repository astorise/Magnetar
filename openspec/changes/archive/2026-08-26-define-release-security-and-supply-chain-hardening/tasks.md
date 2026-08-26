# Tasks

## 1. Security Scope

- [x] Define `v0.1` security scope.
- [x] Include CPU-local baseline.
- [x] Include workspace dependencies.
- [x] Include release binaries.
- [x] Include release reports.
- [x] Include OpenSpec baseline.
- [x] Include WIT packages.
- [x] Include Reference CPU Provider.
- [x] Include fixture artifacts.
- [x] Exclude future GPU Providers from hardened claims.
- [x] Exclude server API from hardened claims.
- [x] Exclude model hub downloads from hardened claims.
- [x] Document limitations.

## 2. Dependency Audit

- [x] Run dependency advisory audit.
- [x] Check yanked crates.
- [x] Check duplicate high-risk dependencies.
- [x] Check unexpected build scripts.
- [x] Check unexpected native dependencies.
- [x] Check dependency tree drift.
- [x] Define critical advisory blocking rule.
- [x] Document accepted mitigations.

## 3. License Audit

- [x] Collect dependency licenses.
- [x] Identify incompatible licenses.
- [x] Identify unknown licenses.
- [x] Identify missing license metadata.
- [x] Identify license exceptions.
- [x] Generate third-party notices where required.
- [x] Define license blocking rule.
- [x] Document approvals.

## 4. SBOM

- [ ] Generate SBOM if tooling is available. (out of scope: this change defines the SBOM data shape and blocking rule as executable Rust types (`SbomManifest`, `SbomAvailability`) but does not integrate real SBOM-generation tooling -- consistent with the module's non-goal of policy-as-data rather than live tooling integration, matching how `release_packaging::ReleaseArtifactKind::SbomPlaceholder` never generated a real SBOM either.)
- [x] Include package names.
- [x] Include package versions.
- [x] Include dependency list.
- [x] Include dependency versions.
- [x] Include license metadata.
- [x] Include source repository metadata where available.
- [x] Include build target metadata.
- [x] Include feature flags where feasible.
- [x] If incomplete, document limitation.

## 5. Checksums

- [x] Generate source archive checksum.
- [x] Generate binary checksum.
- [x] Generate conformance report checksum.
- [x] Generate E2E report checksum.
- [x] Generate OpenSpec validation report checksum.
- [x] Generate coverage report checksum.
- [x] Generate SBOM checksum where present.
- [x] Verify checksums against final artifacts.

## 6. Signatures

- [x] Decide whether signatures are available for `v0.1`.
- [ ] If available, sign release artifacts. (not applicable for v0.1: signatures are explicitly not implemented -- `SignatureStatus::NotImplementedDocumented` is the expected status, per the proposal's Non-Goals ("does not implement full cryptographic signing"); `validate_signature_status` only enforces that the absence is documented, never that signing itself happens.)
- [x] If unavailable, document signature limitation.
- [x] Ensure absence of signatures is not hidden.
- [x] Add future signing follow-up.

## 7. Provenance

- [x] Record source commit.
- [x] Record release tag.
- [x] Record CI run identifier.
- [x] Record build target.
- [x] Record build profile.
- [x] Record rustc version.
- [x] Record lockfile digest.
- [x] Record OpenSpec baseline digest.
- [x] Record WIT package digest.
- [x] Record conformance report digest.
- [x] Redact secrets and local paths.

## 8. Reproducibility

- [x] Document reproducibility status.
- [x] Document non-reproducible parts if any.
- [x] Preserve verification metadata.
- [x] Add reproducibility notes to release docs.

## 9. Lockfile Policy

- [x] Ensure lockfile is checked in where appropriate.
- [x] Include lockfile digest in provenance.
- [x] Detect lockfile drift.
- [x] Block unreviewed lockfile drift in release candidate.

## 10. Build Script Review

- [x] Identify build scripts in required dependencies.
- [x] Review new build scripts.
- [x] Flag unexpected build scripts.
- [x] Document native build steps.
- [x] Add build script review notes.

## 11. Secret Scanning

- [x] Scan source files.
- [x] Scan generated docs.
- [x] Scan release notes.
- [x] Scan conformance reports.
- [x] Scan E2E reports.
- [x] Scan included logs.
- [x] Scan build metadata.
- [x] Scan packaged artifacts.
- [x] Block release on detected secrets.

## 12. Artifact Integrity

- [x] Verify source state is CI-controlled or clean.
- [x] Verify release tag matches source.
- [x] Verify OpenSpec report matches baseline.
- [x] Verify conformance reports match release commit.
- [x] Verify checksums match final artifacts.
- [x] Add artifact integrity gate.

## 13. Redaction Gates

- [x] Verify raw prompts are absent by default.
- [x] Verify secrets are absent.
- [x] Verify credentials are absent.
- [x] Verify raw file contents are absent.
- [x] Verify raw model weights are absent.
- [x] Verify raw tensor values are absent.
- [x] Verify raw KV cache contents are absent.
- [x] Verify raw Provider handles are absent.
- [x] Verify raw Device handles are absent.
- [x] Verify raw Kernel handles are absent.
- [x] Verify raw memory pointers are absent.
- [x] Verify local filesystem paths are absent unless explicitly allowed.
- [x] Verify raw cache paths are absent by default.

## 14. Provider Trust Boundary

- [x] Document Providers as trusted native code.
- [x] Document Reference CPU Provider status.
- [x] Mark dynamic Provider loading disabled, experimental, or unstable.
- [x] Verify Provider registration does not imply trust beyond policy.
- [x] Add Provider trust tests.

## 15. Native Handle Boundary

- [x] Audit public APIs for native handles.
- [x] Audit diagnostics for native handles.
- [x] Audit reports for native handles.
- [x] Deny CUDA pointer exposure.
- [x] Deny CUDA stream exposure.
- [x] Deny Metal buffer exposure.
- [x] Deny OpenVINO pointer exposure.
- [x] Deny QNN handle exposure.
- [x] Deny raw CPU allocation pointer exposure.
- [x] Add native handle tests.

## 16. Component Artifact Trust

- [x] Validate Component Artifacts before execution.
- [x] Deny unsigned Component Artifacts in production policy unless allowed.
- [x] Document Component trust status.
- [x] Add Component trust tests.

## 17. Model Artifact Trust

- [x] Validate Model Artifacts before loading.
- [x] Validate fixture artifact test trust policy.
- [x] Ensure recognized format does not imply trust.
- [x] Ensure cache presence does not imply trust.
- [x] Add Model Artifact trust tests.

## 18. Source Cache Trust

- [x] Verify cache hit is not trust.
- [x] Verify source kind is not trust.
- [x] Verify alias is not trust.
- [x] Verify local file is not trust.
- [x] Verify fixture requires test policy.
- [x] Verify current policy decides loading.
- [x] Add source/cache trust tests.

## 19. CLI Boundary Security

- [x] Verify CLI authority is not delegated to Runtime.
- [x] Verify filesystem authority is not ambient.
- [x] Verify Git authority is not ambient.
- [x] Verify network authority is not ambient.
- [x] Verify secret authority is not ambient.
- [x] Verify shell/process authority is not ambient.
- [x] Verify tool authority is not ambient.
- [x] Add CLI security boundary tests.

## 20. Runtime Inference API Security

- [x] Verify Runtime has no arbitrary filesystem authority.
- [x] Verify Runtime has no arbitrary network authority.
- [x] Verify Runtime has no secret authority.
- [x] Verify Runtime has no shell/process authority.
- [x] Verify Runtime has no Git authority.
- [x] Verify Runtime has no tool execution authority.
- [x] Verify Runtime has no agent orchestration authority.
- [x] Add Runtime API security tests.

## 21. Unsafe Code Policy

- [x] Detect unsafe Rust usage.
- [x] Review unsafe blocks where present.
- [x] Document unsafe rationale.
- [x] Minimize unsafe in required baseline.
- [x] Deny unreviewed unsafe where policy requires.
- [x] Add unsafe policy check.

## 22. Dependency Feature Policy

- [x] Review enabled dependency features.
- [x] Identify networking-enabling features.
- [x] Identify filesystem-expanding features.
- [x] Identify native plugin/dynamic loading features.
- [x] Identify broad OS capability features.
- [x] Block unexpected capability-expanding features.
- [x] Document accepted features.

## 23. Vulnerability Handling

- [x] Define advisory severity handling.
- [x] Define release blocking criteria.
- [x] Define mitigation documentation.
- [x] Define exception approval.
- [x] Define follow-up tracking.
- [x] Define patch release expectation.
- [x] Add vulnerability policy docs.

## 24. Security Notes

- [x] Document `v0.1` threat model.
- [x] Document trusted native Provider model.
- [x] Document no raw handle policy.
- [x] Document default redaction.
- [x] Document source/cache trust boundary.
- [x] Document Model Artifact trust boundary.
- [x] Document Component Artifact trust boundary.
- [x] Document unsupported security features.
- [x] Document known risks.
- [x] Document reporting process placeholder.

## 25. Release Blocking Criteria

- [x] Block detected secrets.
- [x] Block critical dependency advisory without mitigation.
- [x] Block incompatible required dependency license.
- [x] Block failing redaction gate.
- [x] Block raw internal handle exposure.
- [x] Block failed trust/integrity validation in required fixtures.
- [x] Block E2E conformance bypass.
- [x] Block OpenSpec validation failure.
- [x] Block release artifact checksum mismatch.
- [x] Block undocumented security exception.

## 26. Security Exceptions

- [x] Define exception template.
- [x] Require issue description.
- [x] Require affected component.
- [x] Require severity.
- [x] Require rationale.
- [x] Require mitigation.
- [x] Require owner.
- [x] Require expiration or follow-up.
- [x] Require release note entry.
- [x] Deny undocumented exceptions.

## 27. Observability

- [x] Record dependency audit completed.
- [x] Record license audit completed.
- [x] Record SBOM generated.
- [x] Record checksum generated.
- [x] Record secret scan completed.
- [x] Record redaction gate completed.
- [x] Record provenance generated.
- [x] Record security exception recorded.
- [x] Record release blocked.
- [x] Record release security passed.
- [x] Verify default redaction.

## 28. Final Validation

- [x] Run OpenSpec validation.
- [x] Run dependency audit.
- [x] Run license audit.
- [x] Run secret scanning.
- [x] Run redaction gates.
- [x] Run native handle exposure tests.
- [x] Run trust boundary tests.
- [x] Run artifact integrity checks.
- [x] Verify security notes are included.
- [x] Verify security exceptions are documented.
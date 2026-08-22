# Tasks

## 1. Distribution Scope

- [x] Document that distribution applies only to Magnetar-compatible Inference
      Components.
- [x] Document that general tool Components are outside Magnetar distribution
      scope.
- [x] Document that filesystem, Git, shell, network, secret, and workspace tools
      belong to clients such as `magnetar-cli`.
- [x] Ensure distribution does not weaken the inference-scoped authority model.

## 2. Distribution Unit

- [x] Define Component Artifact Package.
- [x] Include executable Component bytes.
- [x] Include or reference Component manifest.
- [x] Include declared digest.
- [x] Include source identity.
- [x] Include optional publisher identity.
- [x] Include optional signature metadata.
- [x] Include optional provenance metadata.
- [x] Keep package distinct from Model Artifact packages.
- [x] Keep package distinct from Provider binary packages.

## 3. Source Model

- [x] Define Component Distribution Source.
- [x] Define local-directory source.
- [x] Define local-cache source.
- [x] Define client-provided source.
- [x] Define development-fixture source.
- [x] Reserve external-registry source.
- [x] Reserve Tachyon source.
- [x] Treat source kind as metadata only.
- [x] Ensure source kind does not imply trust.

## 4. Pull Flow

- [x] Define artifact resolution by logical identity.
- [x] Define artifact resolution by digest.
- [x] Define version requirement resolution.
- [x] Define candidate artifact list.
- [x] Define fetch by digest.
- [x] Validate fetched bytes locally.
- [x] Reject source results that fail digest validation.
- [x] Add pull-flow tests where a source abstraction exists.

## 5. Push Flow

- [x] Define externally supplied package input.
- [x] Require local validation for pushed packages.
- [x] Reject pushed packages with mismatched digest.
- [x] Reject pushed packages with invalid manifest.
- [x] Reject pushed packages requesting forbidden authority.
- [x] Ensure pushed delivery does not bypass trust policy.
- [x] Add push-flow tests.

## 6. Digest Verification

- [x] Compute digest after receiving bytes.
- [x] Compare computed digest to source-declared digest.
- [x] Compare computed digest to manifest-declared digest.
- [x] Reject mismatch.
- [x] Normalize digest format.
- [x] Add tests for source digest mismatch.
- [x] Add tests for manifest digest mismatch.

## 7. Manifest Validation

- [x] Load distributed manifest.
- [x] Validate schema.
- [x] Validate artifact kind.
- [x] Validate Component identity.
- [x] Validate Component version.
- [x] Validate Runtime compatibility.
- [x] Validate Capability compatibility.
- [x] Validate WIT import declarations.
- [x] Validate WIT export declarations.
- [x] Validate inference authority declarations.
- [x] Reject invalid manifest before preparation.

## 8. WIT Consistency

- [x] Inspect actual WIT imports from received Component bytes.
- [x] Inspect actual WIT exports from received Component bytes.
- [x] Compare actual imports to manifest imports.
- [x] Compare actual exports to manifest exports.
- [x] Reject WIT mismatch.
- [x] Add tests for omitted actual import.
- [x] Add tests for nonexistent declared export.
- [x] Add tests for incompatible version declaration.

## 9. Inference Authority Enforcement

- [x] Reuse inference-scoped authority validation.
- [x] Accept inference authority categories.
- [x] Reject filesystem authority.
- [x] Reject network authority.
- [x] Reject secrets authority.
- [x] Reject Git authority.
- [x] Reject workspace authority.
- [x] Reject shell/process authority.
- [x] Reject tool-execution authority.
- [x] Verify trusted source does not override forbidden authority.
- [x] Verify trusted digest does not override forbidden authority.

## 10. Trust Policy Integration

- [x] Apply trust policy after integrity validation.
- [x] Apply allowlist by digest.
- [x] Apply denylist by digest.
- [x] Apply revoked digest list.
- [x] Apply source policy.
- [x] Apply publisher policy.
- [x] Ensure revoke/deny wins over allow.
- [x] Ensure untrusted packages are not prepared.
- [x] Add trust policy tests for distributed packages.

## 11. Cache Model

- [x] Define optional Component Artifact cache.
- [x] Key cache entries by digest.
- [x] Store source metadata separately from trust decision.
- [x] Verify cached bytes before use.
- [x] Do not trust cache presence.
- [x] Revalidate trust decision when policy requires.
- [x] Reject corrupted cache entries.
- [x] Add cache hit test.
- [x] Add cache integrity failure test.

## 12. Revocation

- [x] Reject revoked digests from all sources.
- [x] Reject revoked cached artifacts.
- [x] Reject revoked pushed packages.
- [x] Reject revoked pull results.
- [x] Support source-provided revocation metadata as advisory input.
- [x] Keep Runtime trust policy authoritative.
- [x] Add revocation tests.

## 13. Version Resolution

- [x] Define logical Component identity lookup.
- [x] Define version requirement format.
- [x] Return candidate digests.
- [x] Do not execute selected candidate without validation.
- [x] Reject candidate whose manifest identity does not match request.
- [x] Reject candidate whose version is incompatible.
- [x] Add version resolution tests where implemented.

## 14. Compatibility Resolution

- [x] Evaluate Magnetar Runtime compatibility locally.
- [x] Evaluate Capability compatibility locally.
- [x] Evaluate Component Engine feature compatibility locally.
- [x] Reject source-selected candidate if incompatible.
- [x] Add compatibility rejection tests.

## 15. Signature Metadata

- [x] Accept optional signature metadata.
- [x] Bind signature metadata to digest.
- [x] Record unsupported signatures as unverified or reject by policy.
- [x] Do not trust signature presence by default.
- [x] Add tests for signature metadata without configured trust root.
- [x] Add tests for signature digest mismatch if metadata supports it.

## 16. Provenance Metadata

- [x] Accept optional provenance metadata.
- [x] Record builder identity where provided.
- [x] Record source repository where provided.
- [x] Record commit digest where provided.
- [x] Record build timestamp where provided.
- [x] Treat provenance as metadata, not automatic trust.
- [x] Add provenance parse tests.

## 17. Offline Operation

- [x] Support local-directory distribution.
- [x] Support local-cache distribution.
- [x] Ensure local trusted artifact can be validated without network.
- [x] Ensure Tachyon is not required.
- [x] Ensure registry access is not required.
- [x] Add offline validation tests.

## 18. Tachyon Boundary

- [x] Document Tachyon as future optional distribution source.
- [x] Ensure Tachyon source does not imply trust.
- [x] Ensure Tachyon source does not permit broad tool authority.
- [x] Ensure Tachyon-provided artifacts are validated locally.
- [x] Preserve "Tachyon distributes; Magnetar validates and executes inference."

## 19. magnetar-cli Boundary

- [x] Document `magnetar-cli` as possible client-provided source.
- [x] Ensure CLI-provided package does not bypass validation.
- [x] Ensure CLI workspace authority does not become Magnetar authority.
- [x] Ensure CLI Git authority does not become Magnetar authority.
- [x] Ensure CLI network authority does not become Magnetar authority.
- [x] Ensure CLI secret authority does not become Magnetar authority.

## 20. Error Model

- [x] Define source-unavailable error.
- [x] Define artifact-not-found error.
- [x] Define version-not-found error.
- [x] Define digest-mismatch error.
- [x] Define manifest-missing error.
- [x] Define manifest-invalid error.
- [x] Define WIT-mismatch error.
- [x] Define compatibility-failure error.
- [x] Define forbidden-authority error.
- [x] Define trust-rejected error.
- [x] Define revoked-artifact error.
- [x] Define cache-integrity-failure error.
- [x] Define unsupported-signature error.
- [x] Define policy-denied error.

## 21. Observability

- [x] Emit source resolution observations.
- [x] Emit package received observations.
- [x] Emit fetch success observations.
- [x] Emit fetch failure observations.
- [x] Emit digest mismatch observations.
- [x] Emit manifest mismatch observations.
- [x] Emit forbidden authority observations.
- [x] Emit trust rejection observations.
- [x] Emit revocation observations.
- [x] Emit cache hit observations.
- [x] Emit cache integrity failure observations.
- [x] Redact credentials and sensitive source URLs.
- [x] Ensure observability failure does not alter validation outcome.

## 22. Documentation

- [x] Document Component Distribution Source.
- [x] Document Component Artifact Package.
- [x] Document pull flow.
- [x] Document push flow.
- [x] Document digest validation.
- [x] Document cache behavior.
- [x] Document source trust limitations.
- [x] Document Tachyon boundary.
- [x] Document magnetar-cli boundary.
- [x] Document out-of-scope tool distribution.

## 23. Security Review

- [x] Verify source identity does not imply trust.
- [x] Verify source-provided digest is verified locally.
- [x] Verify pushed package cannot bypass validation.
- [x] Verify cached package cannot bypass validation.
- [x] Verify broad tool authority is rejected.
- [x] Verify revoked digest rejects across all source kinds.
- [x] Verify manifest cannot grant itself trust.
- [x] Verify package provenance does not imply trust.
- [x] Verify unsupported signature does not imply trust.
- [x] Verify no Tachyon dependency is required.

## 24. Tests

- [x] Test local-directory package validation.
- [x] Test client-provided package validation.
- [x] Test digest mismatch rejection.
- [x] Test invalid manifest rejection.
- [x] Test WIT mismatch rejection.
- [x] Test forbidden authority rejection.
- [x] Test revoked artifact rejection.
- [x] Test untrusted source rejection.
- [x] Test trusted digest acceptance.
- [x] Test trusted source still rejects forbidden authority.
- [x] Test cache integrity verification.
- [x] Test offline operation.
- [x] Test Tachyon source metadata does not imply trust.

## 25. CI

- [ ] Run Component distribution tests in CI.
- [ ] Run artifact validation tests in CI.
- [ ] Run cache integrity tests in CI if cache is implemented.
- [ ] Run offline tests in CI.
- [ ] Run WIT fixture validation in CI.
- [ ] Ensure tests do not require external network.
- [ ] Ensure tests do not require Tachyon.

## 26. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [ ] Run Clippy.
- [x] Run complete tests.
- [x] Run WASM Component Runtime tests.
- [x] Run Component Artifact trust tests.
- [x] Run Component distribution tests.
- [ ] Run WIT validation.
- [x] Run OpenSpec validation.
- [ ] Run coverage validation.
- [x] Verify distributed artifacts are validated locally.
- [x] Verify distribution is inference-scoped.
- [x] Verify Tachyon is optional.
- [x] Verify no general tool Component authority is introduced.

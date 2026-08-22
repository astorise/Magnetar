# Tasks

## 1. Artifact Domain Model

- [x] Define Component Artifact identity.
- [x] Define artifact kind.
- [x] Define digest algorithm.
- [x] Define content digest.
- [x] Define logical Component name.
- [x] Define Component version.
- [x] Define manifest version.
- [x] Define artifact source identity.
- [x] Define optional publisher identity.
- [x] Keep Component Artifact distinct from Model Artifact.
- [x] Keep Component Artifact distinct from Provider binary.

## 2. Manifest Format

- [x] Define the initial manifest format.
- [x] Choose YAML, JSON, or support both.
- [x] Define manifest schema version.
- [x] Define required manifest fields.
- [x] Define optional manifest fields.
- [x] Define Component name field.
- [x] Define Component version field.
- [x] Define Component description field.
- [x] Define Component role field.
- [x] Define WIT import declarations.
- [x] Define WIT export declarations.
- [x] Define Runtime compatibility declaration.
- [x] Define Capability compatibility declaration.
- [x] Define authority requirement declaration.
- [x] Define publisher metadata.
- [x] Define source metadata.
- [x] Define signature metadata placeholder.
- [x] Define provenance metadata placeholder.

## 3. Digest Model

- [x] Select initial digest algorithm.
- [x] Prefer `sha256` unless repository policy chooses otherwise.
- [x] Compute digest over executable Component bytes.
- [x] Record digest in canonical form.
- [x] Validate manifest digest against computed digest.
- [x] Reject digest mismatch.
- [x] Add digest normalization tests.
- [x] Add digest mismatch tests.

## 4. Component Name and Version Validation

- [x] Define valid Component logical name syntax.
- [x] Define version syntax.
- [x] Reject empty names.
- [x] Reject ambiguous names.
- [x] Reject invalid version strings.
- [x] Preserve case-sensitivity policy explicitly.
- [x] Add tests for valid names.
- [x] Add tests for invalid names.

## 5. Manifest Loading

- [x] Load sidecar manifest from local file.
- [x] Report missing manifest as a structured error unless explicitly allowed by
      test/development mode.
- [x] Parse manifest.
- [x] Validate required fields.
- [x] Reject unsupported manifest version.
- [x] Normalize parse errors.
- [x] Avoid executing manifest-defined code.
- [x] Add manifest parse tests.

## 6. Artifact Validation Pipeline

- [x] Implement validation pipeline before Component preparation.
- [x] Compute digest before trusting metadata.
- [x] Load manifest.
- [x] Validate manifest structure.
- [x] Validate digest match.
- [x] Inspect actual WIT imports.
- [x] Inspect actual WIT exports.
- [x] Compare manifest declarations to actual WIT.
- [x] Evaluate Runtime compatibility.
- [x] Evaluate Capability compatibility.
- [x] Evaluate authority declaration syntax.
- [x] Evaluate trust policy.
- [x] Allow ComponentEngine preparation only after successful validation.

## 7. WIT Inspection

- [x] Inspect actual WIT imports from executable Component artifact.
- [x] Inspect actual WIT exports from executable Component artifact.
- [x] Preserve package/interface/version identity.
- [x] Distinguish required imports from optional metadata where supported.
- [x] Normalize WIT inspection errors.
- [x] Add fixture with matching WIT declarations.
- [x] Add fixture with manifest missing actual import.
- [x] Add fixture with manifest claiming nonexistent export.

## 8. WIT Manifest Consistency

- [x] Reject manifest that omits required actual imports.
- [x] Reject manifest that declares incompatible import version.
- [x] Reject manifest that declares nonexistent export.
- [x] Reject manifest with invalid WIT package identity.
- [x] Accept manifest that accurately describes imports and exports.
- [x] Add tests for every mismatch class.

## 9. Runtime Compatibility

- [x] Define Runtime compatibility declaration.
- [x] Support minimum Runtime version.
- [x] Support optional maximum Runtime version if required.
- [x] Reject artifact requiring unsupported future Runtime.
- [x] Reject artifact incompatible with current Runtime when declared.
- [x] Add compatibility tests.
- [x] Avoid coupling compatibility to Tachyon version.

## 10. Capability Compatibility

- [x] Define required Capability compatibility declaration.
- [x] Match declared Capability requirements to WIT imports.
- [x] Reject missing required Capability.
- [x] Reject unsupported major Capability version.
- [x] Reject incompatible version range.
- [x] Preserve semantic version rules.
- [x] Add tests using Compute or a test Capability.

## 11. Authority Requirement Declaration

- [x] Define manifest field for requested authority.
- [x] Support declaration of filesystem need.
- [x] Support declaration of network need.
- [x] Support declaration of environment need.
- [x] Support declaration of process need.
- [x] Support declaration of secret need.
- [x] Support declaration of clock/randomness need where applicable.
- [x] Validate syntax only in this change.
- [x] Do not grant authority in this change.
- [x] Reject unknown authority kind unless policy explicitly allows unknown
      declarations.
- [x] Add tests for unsupported authority declarations.

## 12. Trust Status Model

- [x] Define trust status enum.
- [x] Include unknown.
- [x] Include trusted.
- [x] Include rejected.
- [x] Include quarantined.
- [x] Include revoked.
- [x] Define status transition rules.
- [x] Ensure only trusted artifacts may be prepared.
- [x] Add tests for trust status enforcement.

## 13. Trust Policy

- [x] Define local trust policy model.
- [x] Support allowlist by digest.
- [x] Support denylist by digest.
- [x] Support revoked digest list.
- [x] Support optional publisher allowlist.
- [x] Support optional source allowlist.
- [x] Define precedence between deny/revoke and allow.
- [x] Ensure rejection wins over trust.
- [x] Add trust policy tests.

## 14. Trust Store

- [x] Define file-based trust store format for initial implementation.
- [x] Load trusted digests.
- [x] Load rejected digests.
- [x] Load revoked digests.
- [x] Load trusted publishers where supported.
- [x] Validate trust store schema.
- [x] Reject invalid trust store configuration.
- [x] Avoid storing trust decisions inside the Component manifest itself.
- [x] Add trust store tests.

## 15. Publisher Metadata

- [x] Define publisher metadata field.
- [x] Treat publisher identity as metadata.
- [x] Do not infer trust from publisher identity alone.
- [x] Apply publisher trust only through policy.
- [x] Add tests showing publisher alone does not make artifact trusted.
- [x] Add tests showing trusted publisher policy can influence trust when
      configured.

## 16. Source Metadata

- [x] Define source metadata field.
- [x] Support local source identity.
- [x] Support development fixture source identity.
- [x] Reserve external/Tachyon/registry source identities for future use.
- [x] Do not infer trust from source identity alone.
- [x] Apply source trust only through policy.
- [x] Add tests showing local file presence does not imply trust.

## 17. Signature Metadata Placeholder

- [x] Define optional signature metadata fields.
- [x] Bind signature metadata to artifact digest.
- [x] Treat unsupported signature as unverified.
- [x] Do not treat present signature as trusted unless verification is
      configured.
- [x] Add tests for unsupported signature metadata.
- [x] Add tests that signature mismatch rejects when verification is enabled or
      metadata is structurally invalid.

## 18. Development Mode

- [x] Define explicit development mode.
- [x] Allow unsigned local artifacts only when development mode is enabled.
- [x] Still compute digest in development mode.
- [x] Still validate manifest in development mode.
- [x] Still inspect WIT in development mode.
- [x] Ensure development mode is not silently enabled in production.
- [x] Add development-mode tests.

## 19. Artifact Cache

- [x] Define optional local artifact cache behavior.
- [x] Key cache entries by digest.
- [x] Verify digest when loading from cache.
- [x] Do not trust cache presence.
- [x] Store validation metadata separately from executable trust.
- [x] Add cache integrity tests if cache is implemented in this change.

## 20. Revocation

- [x] Support revoked artifact digests.
- [x] Prevent preparation of revoked artifacts.
- [x] Prevent creation of new instances from revoked artifacts.
- [x] Define active-instance policy for artifacts revoked after instantiation.
- [x] Add tests for revocation before preparation.
- [x] Add tests for revocation after previous trust.

## 21. Quarantine

- [x] Define quarantine status.
- [x] Prevent quarantined artifacts from preparation.
- [x] Preserve diagnostic metadata where appropriate.
- [x] Avoid executing quarantined bytes.
- [x] Add quarantine tests.

## 22. Component Artifact Lifecycle

- [x] Define discovered state.
- [x] Define manifest-loaded state.
- [x] Define digest-verified state.
- [x] Define WIT-validated state.
- [x] Define compatibility-validated state.
- [x] Define trust-evaluated state.
- [x] Define trusted state.
- [x] Define rejected state.
- [x] Define prepared state as separate from artifact state.
- [x] Define instantiated state as separate from artifact state.

## 23. Runtime Integration

- [x] Require artifact validation before ComponentEngine preparation.
- [x] Pass only trusted artifact bytes to preparation.
- [x] Preserve ComponentDefinition identity.
- [x] Attach artifact digest to ComponentDefinition.
- [x] Attach artifact trust decision to ComponentDefinition metadata.
- [x] Prevent direct preparation from arbitrary path outside trusted validation
      except explicit test hooks.
- [x] Update ComponentManager or ComponentRuntime integration accordingly.

## 24. Observability

- [x] Emit artifact discovery observations.
- [x] Emit digest computation observations.
- [x] Emit manifest validation observations.
- [x] Emit WIT mismatch observations.
- [x] Emit compatibility failure observations.
- [x] Emit trust decision observations.
- [x] Emit revocation observations.
- [x] Emit quarantine observations.
- [x] Redact signature/private metadata.
- [x] Ensure observability failures do not alter trust decisions.

## 25. Test Fixtures

- [x] Add valid Component artifact fixture.
- [x] Add valid manifest fixture.
- [x] Add digest mismatch fixture.
- [x] Add WIT import mismatch fixture.
- [x] Add WIT export mismatch fixture.
- [x] Add unsupported Runtime version fixture.
- [x] Add untrusted artifact fixture.
- [x] Add revoked artifact fixture.
- [x] Add development-mode fixture.
- [x] Keep fixture trust separate from production trust defaults.

## 26. Documentation

- [x] Document Component Artifact identity.
- [x] Document manifest format.
- [x] Document digest format.
- [x] Document validation pipeline.
- [x] Document trust statuses.
- [x] Document trust store format.
- [x] Document development mode.
- [x] Document revocation.
- [x] Document quarantine.
- [x] Document separation from Model Artifacts.
- [x] Document separation from Provider binaries.
- [x] Document Tachyon as future optional source, not dependency.

## 27. Security Review

- [x] Verify arbitrary `.wasm` files are not prepared without validation.
- [x] Verify manifest cannot mark itself trusted.
- [x] Verify digest mismatch rejects.
- [x] Verify revoked digest rejects.
- [x] Verify unsupported authority declaration rejects or remains ungranted.
- [x] Verify untrusted publisher does not imply trust.
- [x] Verify local cache does not imply trust.
- [x] Verify development mode is explicit.
- [x] Verify signature metadata is not blindly trusted.

## 28. CI

- [x] Add manifest schema validation to CI where practical.
- [x] Add artifact validation tests to CI.
- [x] Add digest validation tests to CI.
- [x] Add trust policy tests to CI.
- [x] Add WIT consistency tests to CI.
- [x] Ensure fixture artifacts are reproducible or intentionally checked in.
- [x] Run Component artifact tests with the WASM engine feature where required.

## 29. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run WASM Component Runtime tests.
- [x] Run WIT validation.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify only trusted artifacts can be prepared.
- [x] Verify Component Artifact and Model Artifact remain distinct.
- [x] Verify Tachyon is not required.
- [x] Verify arbitrary local `.wasm` execution is not allowed by default.
## ADDED Requirements

### Requirement: Publication Gated On Existing Release Policy

Release publication SHALL NOT proceed unless
`release_packaging::release_may_publish_stable` and
`release_security::evaluate_release_security_blocking` both return `Ok(())`
against gate results and gate inputs collected from the tagged commit's
actual CI run.

#### Scenario: Required gate missing

Given the tagged commit's CI run did not execute `ReferenceCpuConformance`

When the publication pipeline evaluates release readiness

Then `release_may_publish_stable` returns `ReleaseGateMissing` and publication
does not proceed.

#### Scenario: Security gate input triggered

Given the tagged commit's SBOM scan detected an unmitigated critical advisory

When the publication pipeline evaluates release readiness

Then `evaluate_release_security_blocking` returns `ReleaseBlocked` naming
`critical-advisory-unmitigated` and publication does not proceed.

#### Scenario: All gates pass

Given every required `ReleaseGate` passed and every
`ReleaseSecurityGateInputs` field is `false`

When the publication pipeline evaluates release readiness

Then publication proceeds to build release proofs and publish.

### Requirement: Release Proofs Generated From The Tagged Commit's Real Build

Checksums, SBOM entries, and provenance attached to a release SHALL be
computed from the artifacts actually built from the tagged commit, using
`ArtifactChecksum`, `SbomManifest`, and `ReleaseProvenance`, never from
placeholder or previously-cached values.

#### Scenario: Checksum recomputed before upload

Given a release binary has been built from the tagged commit

When the release-proof bundle is assembled

Then `verify_checksum_matches_final_artifact` is evaluated against that
binary's freshly recomputed digest before the binary is uploaded.

#### Scenario: Checksum mismatch blocks upload

Given a recomputed digest does not match its declared `ArtifactChecksum`

When the release-proof bundle is assembled

Then `ChecksumMismatch` is raised, `checksum_mismatch` is recorded as `true`
in `ReleaseSecurityGateInputs`, and the release is blocked.

### Requirement: Crates.io And GitHub Release Are Separate Publication Targets

`magnetar-runtime` SHALL publish to crates.io as its own step, independent of
the GitHub Release step that publishes CLI/binaries and release proofs; a
failure in one SHALL NOT be reported as, or silently treated as, success of
the other.

#### Scenario: Crate publish fails, release step still reports its own outcome

Given the crates.io publish step fails

When the pipeline continues to the GitHub Release step

Then the GitHub Release step's outcome is reported independently, and the
overall pipeline run is marked failed because the crate publish failed.

### Requirement: Credential-Gated Publish Steps Fail Closed

Any pipeline step that performs a real registry write (crates.io publish, GHCR push) SHALL first assert that its required credential secret is present and non-empty, and SHALL fail closed if it is not.

#### Scenario: Missing crates.io token

Given `CARGO_REGISTRY_TOKEN` is not configured in the running environment

When the crates.io publish step begins

Then the step fails immediately with an explicit missing-credential error and
performs no network request to crates.io.

### Requirement: Dry-Run Publication Path Runs Without Credentials

The publication pipeline SHALL include a dry-run validation path (at minimum
`cargo publish --dry-run` and full release-proof generation) that executes
successfully in CI on every run touching publication tooling, regardless of
whether real registry credentials are configured.

#### Scenario: CI run with no secrets configured

Given no repository secrets for crates.io or GHCR are configured

When the publication workflow runs on a pull request touching publication
tooling

Then the dry-run job completes and reports pass/fail on its own merits,
independent of credential availability.

### Requirement: First Real Publication Is Manually Triggered

The first publication of `magnetar-runtime` to crates.io SHALL be initiated by an explicit manual trigger, not by an automatic tag-push event, until that manual run has been confirmed successful at least once.

#### Scenario: Tag pushed before manual confirmation

Given no manual publication run has yet succeeded for this pipeline

When a release tag is pushed

Then the pipeline runs its dry-run and gate-evaluation steps but does not
perform the real credentialed publish step automatically.

### Requirement: Invalid Release Is Withdrawn, Not Deleted

An invalid `magnetar-runtime` crates.io release SHALL be withdrawn using
`cargo yank`, never by attempting to delete or overwrite the published
version; a correction SHALL ship as a new version.

#### Scenario: Published version found to be broken

Given a published crates.io version is later found to violate release
policy

When the release is withdrawn

Then that version is yanked, remains visible as yanked in the crate's version
history, and a corrected version is published under a new version number.

# release-publication Specification

## Purpose
This specification defines the real, credential-gated automation that turns a passing `release-packaging`/`release-security` policy evaluation into an actually published `magnetar-runtime` release: publication is gated on the existing policy, every release proof is computed from the tagged commit's real build, crates.io and GitHub Release are independent targets, credential-gated steps fail closed, a dry-run path runs without credentials, the first real publication is manually triggered, and an invalid release is withdrawn rather than deleted.

## Requirements

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

---

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

---

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

---

### Requirement: Credential-Gated Publish Steps Fail Closed

Any pipeline step that performs a real registry write (crates.io publish, GHCR push) SHALL first assert that its required credential secret is present and non-empty, and SHALL fail closed if it is not.

#### Scenario: Missing crates.io token

Given `CARGO_REGISTRY_TOKEN` is not configured in the running environment

When the crates.io publish step begins

Then the step fails immediately with an explicit missing-credential error and
performs no network request to crates.io.

---

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

---

### Requirement: First Real Publication Is Manually Triggered

The first publication of `magnetar-runtime` to crates.io SHALL be initiated by an explicit manual trigger, not by an automatic tag-push event, until that manual run has been confirmed successful at least once.

#### Scenario: Tag pushed before manual confirmation

Given no manual publication run has yet succeeded for this pipeline

When a release tag is pushed

Then the pipeline runs its dry-run and gate-evaluation steps but does not
perform the real credentialed publish step automatically.

---

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

---

### Requirement: Release Tag Creation Validates Its Inputs

The manual tag-creation workflow SHALL reject tags outside
`vMAJOR.MINOR.PATCH[-PRERELEASE]`, unknown commits, and commits that are not
ancestors of `origin/main` before creating or pushing a tag. It SHALL use the
`release` environment and its workflow-scoped GitHub token.

#### Scenario: Invalid tag or unrelated commit

Given a malformed tag, an unknown commit, or a commit outside `origin/main` history

When the tag-creation workflow validates the request

Then it fails before creating or pushing the tag.

#### Scenario: Tag created with the workflow token

Given the tag-creation workflow pushes a validated tag using `GITHUB_TOKEN`

When the operator proceeds to publication

Then publication requires an explicit dispatch of the publication workflow
against that tag rather than relying on a tag-push workflow trigger.

---

### Requirement: Candidate Releases Retain Prerelease Status

GitHub Release creation and update SHALL mark alpha, beta, and RC tags as
prereleases, including case-insensitive `-RC1` and `-rc.1` suffix forms.
A stable tag such as `v1.0.0` SHALL not be marked as a prerelease.

#### Scenario: Release candidate published or refreshed

Given a tag ending in `-RC1` or `-rc.1`

When the GitHub Release is created or updated

Then the release retains prerelease status.

---

### Requirement: Published Release Includes Real Binaries And Gate Evidence

The GitHub Release SHALL include CLI binaries built for Linux, Windows, and
macOS from the selected tagged commit, alongside checksums, SBOM, provenance,
and the collected release/security gate evidence. Re-running publication for
an existing tag SHALL refresh assets and prerelease status using the current
run's generated outputs. Read-only jobs SHALL retain read-only permissions;
release-write permissions SHALL be scoped to the job that writes the release.

#### Scenario: Existing release refreshed

Given the selected tag already has a GitHub Release

When publication runs again successfully

Then its assets and release status are updated rather than failing solely
because the release already exists.

#### Scenario: Gate evidence accompanies binaries

Given release-gate and security-gate evaluation completed for the tagged commit

When the release bundle is published

Then check-run conclusions, release-gate results, source secret-scan results,
and security-gate inputs accompany the actual built binaries and release proofs.

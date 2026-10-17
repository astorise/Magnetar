## ADDED Requirements

### Requirement: OCI Digest Is The Only Trust Identity

Published Component Artifacts, Kernel Exchange Bundles, and conformance fixtures SHALL be addressed for trust purposes by their OCI content digest (`sha256:...`); a tag or alias SHALL NOT be treated as trust identity by any publishing or consuming step.

#### Scenario: Tag repointed after publication

Given a Component Artifact was published under digest `sha256:aaa...` and
tag `latest`

When a different artifact is later published and `latest` is repointed to
`sha256:bbb...`

Then any consumer that had already pinned `sha256:aaa...` continues to
resolve the original artifact unchanged, and no publishing step treats the
tag move itself as a trust decision.

#### Scenario: Publish step records the digest, not just the tag

Given a Component Artifact push to GHCR succeeds

When the publication pipeline records the result

Then the resulting `sha256:` digest is captured in the release provenance
record, not only the tag used to push it.

### Requirement: Component And Kernel Artifacts Use Versioned OCI Media Types

Component Artifacts, their manifests, Kernel Exchange Bundles, their manifests, and conformance fixtures SHALL be published using explicit versioned OCI media types.

#### Scenario: Component Artifact pushed with correct media type

Given a Component Artifact WASM binary is published to GHCR

When its OCI manifest is constructed

Then the layer's media type is exactly
`application/vnd.magnetar.component.v1+wasm`, and the manifest media type is
`application/vnd.magnetar.component-manifest.v1+json`.

#### Scenario: Kernel Exchange Bundle pushed with correct media type

Given a Kernel Exchange Bundle is published to GHCR

When its OCI manifest is constructed

Then the layer's media type is exactly
`application/vnd.magnetar.kernel-bundle.v1+tar+zstd`, and the manifest media
type is `application/vnd.magnetar.kernel-manifest.v1+json`.

#### Scenario: Conformance fixture pushed with correct media type

Given a conformance fixture is published to GHCR

When its OCI manifest is constructed

Then its media type is exactly `application/vnd.magnetar.conformance.v1+json`.

#### Scenario: Breaking manifest shape change requires a new version

Given a future change needs to alter the Component manifest JSON shape in a
way existing consumers cannot parse

When that change is published

Then it uses `application/vnd.magnetar.component-manifest.v2+json` rather
than silently changing the meaning of the `v1` media type.

### Requirement: GHCR Is The First OCI Backend, Not An Assumed-Only Backend

Component Artifact and Kernel Exchange Bundle publication SHALL target GHCR as the first implemented OCI backend without introducing any requirement elsewhere in Magnetar that only GHCR can supply.

#### Scenario: Ingestion does not hardcode GHCR

Given a Kernel Exchange Bundle is delivered to the ingestion boundary from a
non-GHCR source with an identical digest

When Runtime ingests it

Then the same logical validation applies as a bundle delivered from GHCR,
per the existing `kernel-distribution` "Kernel Distribution Is Transport
Neutral" requirement.

### Requirement: GHCR Push Uses The Workflow's Own Token

The GHCR push step SHALL authenticate using the workflow run's own `GITHUB_TOKEN`, scoped via an explicit `permissions: packages: write` grant on the publishing job, and SHALL NOT require a separately provisioned push credential secret.

#### Scenario: Push authenticates with the workflow's own token

Given the `component-artifact-distribution` job has `permissions: packages: write`

And no externally provisioned GHCR push credential is configured

When the Component Artifact push step runs

Then it authenticates to GHCR using the workflow run's own `GITHUB_TOKEN` and proceeds to push, without requiring any additional secret configuration.

### Requirement: Real OCI Push Only Runs On Manual Dispatch

The GHCR push step SHALL only ever run as part of an explicit `workflow_dispatch` trigger, never automatically off a tag push or any other event, so that a real publish is always a deliberate human action even though no credential gate stands in front of it.

#### Scenario: Workflow is manually triggered

Given a repository collaborator with permission to run workflows triggers `component-artifact-distribution.yml` via `workflow_dispatch`

When the job runs

Then the push step executes using the run's own `GITHUB_TOKEN`, and the `release` GitHub Environment's own protection rules (e.g. required reviewers), if configured by the repository owner, apply before the job starts.

### Requirement: Invalid OCI Publication Is Corrected By New Digest, Not Overwritten

An invalid published Component Artifact or Kernel Exchange Bundle SHALL NOT be deleted or overwritten under its existing digest; a correction SHALL be published as a new artifact with its own new digest.

#### Scenario: Published Component Artifact found to be broken

Given a Component Artifact published at digest `sha256:aaa...` is later
found to be broken

When the artifact is corrected

Then a new artifact is published at a new digest `sha256:ccc...`, the
mutable convenience tag is repointed to `sha256:ccc...`, and `sha256:aaa...`
remains resolvable but is documented as withdrawn.

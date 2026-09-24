## Context

Magnetar has two release-policy modules (`roadmap-contracts::release_packaging`,
`::release_security`) that define, as executable types, what a publishable
release must look like and what must be true before it may ship as stable.
Neither module talks to a registry, a CI credential, or a file on disk beyond
what the caller hands it -- both are pure policy evaluated over caller-supplied
data (`ReleaseGateResult`s, `ArtifactChecksum`s, `SbomManifest`s,
`ReleaseSecurityGateInputs`). Nothing today produces those inputs from a real
build, and nothing today takes a passing evaluation and turns it into bytes on
crates.io, a GitHub Release, or an OCI registry.

Tachyon-Mesh integration audit round 6 (MAG-06) treats this as the last open
production/governance gap and hands down five non-negotiable principles and a
channel-separation table that this design treats as already-decided inputs,
not choices to re-litigate:

- Runtime Core stays transport-neutral -- none of this pipeline's network or
  credential logic may live in or be reachable from `magnetar-runtime`.
- A tag is never trust identity; only a content digest is.
- The distribution channel itself never confers trust (GHCR, a private
  registry, Tachyon, object storage all pass through the same ingestion/trust
  gateway on the consuming side).
- Publication must produce real release proofs generated from the final
  tagged commit's actual build output, not placeholders.
- Cryptographic signatures (MAG-02) are a follow-on chantier; digest pinning
  is the interim trust mechanism this design must not treat as inferior or
  block on.

Channel separation (decided, not open):

| Artifact | Channel | Identity |
|---|---|---|
| `magnetar-runtime` crate | crates.io | SemVer + crate checksum |
| CLI / binaries | GitHub Release | tag + checksum |
| Conformance reports, SBOM, provenance, checksums | GitHub Release | tag + digest |
| Component Artifacts (WASM) | OCI (GHCR first) | `sha256:` digest |
| Kernel Exchange Bundles | OCI (GHCR first) | `sha256:` digest |
| Small conformance fixtures | OCI or Release, by use | digest |
| Model Artifacts | external source (unchanged) | identity+integrity manifest only, never re-hosted |

This session has no crates.io or GHCR credentials, so this design must produce
a pipeline that is fully buildable and testable now, with exactly one
well-defined seam where a human supplies secrets later.

## Goals / Non-Goals

**Goals:**

- A tag-triggered `release-publication` pipeline: build → gate on
  `release_packaging::release_may_publish_stable` and
  `release_security::evaluate_release_security_blocking` → `cargo publish`
  `magnetar-runtime` → create a GitHub Release carrying binaries, checksums,
  SBOM, provenance, and conformance reports generated from that same tagged
  commit's build.
- A `component-artifact-distribution` pipeline: build a Component Artifact or
  Kernel Exchange Bundle → compute its digest → push to GHCR under formalized
  OCI media types → the digest (never the tag) is what downstream ingestion
  (`kernel-distribution`'s existing contract) trusts.
- Every step reachable without secrets (dry-run publish, checksum/SBOM/
  provenance generation, OCI manifest construction and local digest
  verification) is real, tested code in this change -- not a stub.
- Every step that needs a secret (the actual `cargo publish` token push, the
  actual GHCR push) is isolated behind one named-but-unconfigured GitHub
  Actions secret per registry, so the workflow fails closed with a clear
  "secret not configured" error rather than silently no-op'ing or guessing a
  credential.
- A documented withdrawal/correction procedure for an invalid release,
  because neither crates.io nor an OCI digest can be un-published and
  reused under the same identity (crates.io: `cargo yank`; OCI: publish a
  corrected artifact under a new digest and update the mutable alias tag to
  point at it, per the tag-is-not-trust principle).

**Non-Goals:**

- Implementing MAG-02 (artifact signing, Sigstore/cosign or similar). This
  design leaves an explicit extension point (see Decisions) but implements
  no signing.
- Running a real, credentialed publish in this change. That is a follow-up
  action by a human once CI secrets are configured -- see Migration Plan.
- Building a new registry, mirror, or cache. GHCR is the only OCI backend
  implemented; `kernel-distribution`'s transport-neutrality requirement means
  a second backend can be added later without redesigning this pipeline.
- Model Artifact hosting or format changes.

## Decisions

**1. Compose `release_packaging`/`release_security`, don't wrap or duplicate them.**
The publication pipeline's gate step calls `release_may_publish_stable` and
`evaluate_release_security_blocking` directly with `ReleaseGateResult`s and a
`ReleaseSecurityGateInputs` assembled from real CI job outcomes and real
artifact scans. Alternative considered: a new "publication readiness" type
layered on top. Rejected -- it would duplicate exactly the blocking logic
these modules already implement correctly, and would let publication drift
out of sync with the policy modules' own SHALL-strength rules.

**2. `ArtifactChecksum`/`SbomManifest`/`ReleaseProvenance` are the wire format for release proofs.**
The tooling that walks a tagged commit's build output populates these
existing types (not parallel ones) and serializes them into the GitHub
Release. `verify_checksum_matches_final_artifact` is the same function used
both to build the release-proof bundle and, independently, to double-check it
before upload -- so a bug in generation can't silently pass its own check.

**3. New tooling lives as an `xtask`-style crate under `tools/`, following `tools/coverage-ratchet`'s existing pattern.**
Alternative considered: a shell-script pipeline directly in the workflow YAML.
Rejected -- checksum/SBOM/provenance generation and OCI manifest construction
are exactly the kind of logic this repo already prefers as tested Rust
(`tools/coverage-ratchet`, `tools/check_generic_facade_family_isolation.py`
being the one deliberate exception for AST-light static scanning), and a Rust
tool can reuse `roadmap-contracts` types directly instead of re-encoding them
as shell/YAML.

**4. Two separate GitHub Actions workflows, not one.**
`release-publication.yml` triggers on a protected release tag and owns
crates.io + GitHub Release. `component-artifact-distribution.yml` triggers
independently (on its own tag pattern or workflow_dispatch) and owns OCI
publication for Components/Kernel bundles, which may need to ship more often
than a full `magnetar-runtime` release cuts. Alternative considered: one
combined workflow. Rejected -- the audit's own channel table treats these as
independently-cadenced concerns, and Runtime-crate publication SHALL NOT
gate on whether a Component Artifact happens to be ready that day.

**5. Secret seam: one named secret per registry, referenced but not defined here.**
The workflows reference `${{ secrets.CARGO_REGISTRY_TOKEN }}` and
`${{ secrets.GHCR_PUSH_TOKEN }}` (exact names finalized in tasks.md/implementation,
not invented as real values here) and each publish job's first step asserts
the secret is non-empty, failing with a clear message before attempting any
network call. This keeps the distinction between "automation exists" and
"automation is authorized to run for real" mechanical and auditable, rather
than relying on someone remembering which steps are safe to merge before
secrets exist.

**6. OCI media types are versioned from day one.**
`application/vnd.magnetar.component.v1+wasm`,
`application/vnd.magnetar.component-manifest.v1+json`,
`application/vnd.magnetar.kernel-bundle.v1+tar+zstd`,
`application/vnd.magnetar.kernel-manifest.v1+json`,
`application/vnd.magnetar.conformance.v1+json`. The `v1` segment is
deliberate: a breaking change to any manifest shape ships as `v2`, so a
consumer pinned to `v1` never silently receives an incompatible manifest.

**7. Tag-is-not-trust is enforced structurally, not by convention.**
The publish tooling always resolves and records the digest it just pushed and
writes it into the GitHub Release notes / provenance record. The *consuming*
side (already governed by `kernel-distribution`'s existing "External Location
Is Not Identity" requirement) is expected to pin the digest, not the tag; this
change does not modify that consumption-side contract, only ensures the
publish side hands consumers a real digest to pin instead of only a tag.

## Risks / Trade-offs

- [Risk] A workflow merged with the secret seam unfilled looks "done" in CI
  (green) while never having actually published anything, inviting a false
  sense of completion. → Mitigation: the dry-run/local-registry path is a
  required, executed CI job (not skipped), and its output explicitly says
  "dry-run only, no secret configured" so the gap is visible in every run's
  log, not just in this document.
- [Risk] `cargo publish` and OCI push are both irreversible for a given
  identity (yank hides a crate version but does not free it; an OCI digest is
  immutable by definition). A bug in the release-proof generator could ship
  incorrect checksums/SBOM attached to an otherwise-fine artifact. →
  Mitigation: `verify_checksum_matches_final_artifact` runs as a release-blocking
  gate input (`checksum_mismatch` in `ReleaseSecurityGateInputs`) before
  upload, and the withdrawal procedure (Migration Plan) covers exactly this
  case.
  - [Risk] Two independently-cadenced workflows (crate/release vs.
  OCI/Component) could drift out of sync -- e.g. a GitHub Release referencing
  a Component digest that was never actually pushed. → Mitigation: the
  release-proof provenance record links to Component/Kernel digests by value,
  not by assuming a same-day OCI publish; nothing in `release-publication.yml`
  requires `component-artifact-distribution.yml` to have run.
- [Trade-off] Choosing GHCR-only for the first OCI backend is simpler than a
  registry-agnostic client, at the cost of a second backend needing new code
  later. Accepted because `kernel-distribution` already requires transport
  neutrality on the *consuming* side, so a second backend is additive, not a
  breaking redesign.

## Migration Plan

1. Land this change's tooling and both workflows with the secret seam
   present but unfilled; dry-run/local-registry jobs run in CI on every PR
   touching the publication tooling, so regressions are caught immediately
   even before any real secret exists.
2. A human (repo owner) configures `CARGO_REGISTRY_TOKEN` and
   `GHCR_PUSH_TOKEN` (or whatever this change's tasks.md finalizes as their
   exact names) as repository secrets. This step is explicitly outside this
   change's scope to perform.
3. First real publish is a deliberate, manual `workflow_dispatch` (not an
   automatic tag push) so its output can be inspected before any future tag
   push triggers publication unattended.
4. Once the first real `magnetar-runtime` crates.io publish and first real
   GHCR Component push are confirmed, tag-triggered automatic publication is
   enabled.
5. Rollback / correction: a bad crates.io publish is yanked
   (`cargo yank --version <x>`) -- the version stays reserved but stops being
   installable by default; a bad OCI publish is never deleted (digests are
   immutable) -- publish a corrected artifact under its own new digest and
   repoint the mutable convenience tag at it, exactly matching the
   tag-is-not-trust principle this design already enforces structurally.

## Open Questions

- Exact repository-secret names for the crates.io and GHCR tokens -- proposed
  in Decision 5, to be confirmed by the repo owner when they configure them
  (not blocking: the workflow reads whatever name tasks.md finalizes).
- Whether Kernel Exchange Bundle and Component Artifact publication should
  eventually share one workflow keyed by artifact-kind input, once both have
  shipped independently once each -- deferred until there is real usage
  evidence either way.
- Whether the withdrawal procedure needs an automated "supersedes digest
  `sha256:...`" record in the manifest itself, versus living only in GitHub
  Release notes -- left for the signing follow-on chantier (MAG-02) to decide
  alongside trust-policy design, since it touches the same manifest shape.

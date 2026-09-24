# Implement Release Publication Automation

## Why

`release-packaging` and `release-security` already define the *policy* Magnetar
release artifacts must satisfy: versioning rules, required gates, artifact
manifests, checksums, SBOM/provenance/signature-status shape, and the
blocking criteria a stable release must pass. Both changes explicitly deferred
implementing the automation that turns a passing policy evaluation into a
real, publicly fetchable release: `release-packaging`'s own Non-Goals rule out
"release automation" and "publish packages"; `release-security`'s rule out
"remote registry authentication".

That gap is no longer hypothetical. `magnetar-runtime`'s `Cargo.toml` already
carries real `license`/`description`/`repository`/`readme`/`keywords`/
`categories`/`rust-version`/`include` metadata and has zero `path`
dependencies -- it is publish-ready and nobody has run `cargo publish`.
Tachyon-Mesh integration audit round 6 (`docs/audits/audit-magnetar-2026-09-24-revision-publication.md`,
finding MAG-06) confirms this as the last open production/governance gap: Magnetar
has no official, reproducible way to publish the crate, CLI binaries, Component
Artifacts, or Kernel Exchange Bundles, and no documented distribution channel
per artifact kind.

This change defines and implements that publication automation, composing the
existing `release_packaging`/`release_security` policy modules rather than
duplicating their types.

## What Changes

- Define a `release-publication` capability: the tag-triggered pipeline that
  publishes `magnetar-runtime` to crates.io and publishes CLI/binaries plus
  release proofs (checksums, SBOM, provenance, conformance reports) to a
  GitHub Release, gated on `release_packaging::release_may_publish_stable`
  and `release_security::evaluate_release_security_blocking` both passing
  against the artifacts actually built from the tagged commit.
- Define a `component-artifact-distribution` capability: OCI/GHCR publication
  for Component Artifacts, Kernel Exchange Bundles, and small conformance
  fixtures, using digest-based identity as the only trust-relevant identity
  (a tag/alias is a convenience pointer, never itself trusted), with
  formalized OCI media types for each artifact kind.
- Add CI workflow scaffolding for both pipelines, structured so every step
  that needs no registry credentials (dry-run publish, checksum/SBOM/
  provenance generation, OCI manifest construction, digest verification) can
  be built and validated in this change, while steps that need real
  credentials (`cargo publish` token, GHCR push token) are isolated behind a
  documented, not-yet-configured secret so the workflow fails closed and
  loud instead of publishing with an implicit or guessed credential.
- Add a documented invalid-release withdrawal/correction procedure (per
  MAG-06's closure criteria), since a published crates.io version or OCI
  digest cannot simply be deleted and re-published under the same identity.
- **Non-Goals** (explicitly out of scope for this change):
  - Cryptographic signing of any artifact (MAG-02). Digest pinning remains
    the interim trust mechanism; signing is a follow-on change this one must
    not block or duplicate.
  - Actually running a real `cargo publish` or a real OCI push with live
    credentials -- this session has none. This change delivers the
    automation and validates it in dry-run/local-registry form; a human
    configures the CI secrets and triggers the first real publish
    separately.
  - Re-hosting Model Artifact bytes. Large Model Artifacts stay on their
    existing external source; only an identity+integrity manifest is
    published.
  - Changing `kernel-distribution`'s transport-neutrality requirements --
    OCI/GHCR becomes the first standardized backend implementing that
    existing contract, not a replacement for it.

## Capabilities

### New Capabilities

- `release-publication`: the crates.io + GitHub Release publication pipeline
  for `magnetar-runtime` and CLI/binaries, including release-proof generation
  and gating on existing release-packaging/release-security policy.
- `component-artifact-distribution`: the OCI/GHCR publication pipeline for
  Component Artifacts, Kernel Exchange Bundles, and conformance fixtures,
  including media type definitions and digest-identity/tag-is-not-trust
  rules for the publishing side.

### Modified Capabilities

*(none -- this change composes `release-packaging`, `release-security`, and
`kernel-distribution` without changing their existing requirements)*

## Impact

- New CI workflow(s) under `.github/workflows/` for the two pipelines
  (structure only where credentials are required; runnable end-to-end where
  they are not).
- New tooling (likely under a `tools/` or `xtask`-style crate, matching the
  repo's existing `tools/coverage-ratchet` pattern) to generate checksums,
  SBOM, and provenance from a tagged commit's actual build outputs, and to
  construct/push OCI manifests for Component Artifacts and Kernel Exchange
  Bundles.
- `roadmap-contracts::release_packaging` / `::release_security`: consumed
  as-is (their gate/validation functions and artifact types), not modified.
- `docs/release-packaging.md` / `docs/release-security.md`: gain a
  cross-reference to this change once implemented, noting the automation
  gap they each flagged is now filled.
- No change to Runtime Core, Component/Provider/Device semantics, or any
  WIT contract -- this change is entirely about how already-built artifacts
  leave CI and reach a registry.

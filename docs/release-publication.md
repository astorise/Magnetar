# Release Publication (MAG-06)

This document describes the real, credential-gated publication automation
implemented by `openspec/changes/implement-release-publication-automation/`,
built on top of the policy types `docs/release-packaging.md` and
`docs/release-security.md` document. It does not repeat that policy --
only how it is now actually executed.

## Channel separation

| Artifact | Channel | Identity |
|---|---|---|
| `magnetar-runtime` crate | crates.io | SemVer + crate checksum |
| CLI / binaries | GitHub Release | tag + checksum |
| Conformance reports, SBOM, provenance, checksums | GitHub Release | tag + digest |
| Component Artifacts (WASM) | OCI (GHCR) | `sha256:` digest |
| Kernel Exchange Bundles | OCI (GHCR) | `sha256:` digest |
| Model Artifacts | unchanged, external | identity+integrity manifest only, never re-hosted |

A tag is a convenience pointer everywhere in this table. It is never
trust identity; only a digest (a crate's checksum, a release asset's
checksum, or an OCI `sha256:` digest) is. See
`component-artifact-distribution`'s "OCI Digest Is The Only Trust
Identity" requirement.

## Tooling: `tools/release-publish`

A small library + CLI crate (workspace member, no submodule dependency),
invoked the same way `tools/coverage-ratchet` is:
`cargo run --manifest-path tools/release-publish/Cargo.toml -- <command>`.
Every command is credential-free and network-free:

- `checksum-bundle <dir> --out <file>` / `verify-checksums <bundle.json> <dir>`
  -- real SHA-256 `ArtifactChecksum`s over a built directory, and
  re-verification against it before upload.
- `sbom --cargo-metadata <file> --out <file>` -- a real `SbomManifest`
  from `cargo metadata` output, never a hand-maintained list.
- `provenance --repo <dir> --out <file>` -- a real `ReleaseProvenance`
  from this checkout's actual git commit, rustc version, and
  `Cargo.lock`/`openspec/`/WIT directory content digests.
- `release-gate --results <file>` -- evaluates
  `release_packaging::release_may_publish_stable` against a
  `[{"gate": "...", "passed": bool}, ...]` results file.
- `security-gate --cargo-deny-json <file> --out <file> [flags...]` --
  evaluates `release_security::evaluate_release_security_blocking`,
  deriving two of its ten inputs from real `cargo deny -f json check`
  output and taking the rest as explicit flags from the caller's own CI
  job outcomes.
- `oci-manifest --kind ... --input <file> --title <name> --tag <tag> --out-dir <dir>`
  -- builds a real OCI Image Manifest and writes+verifies a local OCI
  Image Layout directory, with every digest computed from the actual
  bytes given.

## Pipeline

### `release-publication.yml` (tag push or manual dispatch)

1. `dry-run-publish` -- `cargo publish --dry-run`. Runs unconditionally,
   no credential required. This is the one job expected to stay green
   with no secrets configured at all.
2. `gate-evaluation` -- fetches this exact commit's own check-run
   conclusions via the GitHub Checks API (never re-runs quality.yml's
   jobs), maps them to `ReleaseGate`s, and evaluates both
   `release_may_publish_stable` and
   `evaluate_release_security_blocking`.
3. `github-release` (tag push only, needs 1+2 green) -- creates a
   GitHub Release with a source archive, checksums, SBOM, and
   provenance. Uses only the workflow's own `GITHUB_TOKEN`; no external
   secret required.
4. `real-publish` (manual `workflow_dispatch` only, needs 1+2 green) --
   the real `cargo publish`. Asserts `CARGO_REGISTRY_TOKEN` is
   configured and fails closed with an explicit error if not, before any
   network call.

### `component-artifact-distribution.yml` (manual dispatch only)

Independent of the above -- Component Artifacts and Kernel Exchange
Bundles may need to ship more often than a full `magnetar-runtime`
release cuts, and Runtime crate publication must never gate on whether
one happens to be ready.

1. Build the OCI manifest and a local OCI Image Layout directory
   (`release-publish oci-manifest`), no network. Always uploaded as a
   workflow artifact, so the exact bytes and digest that would be pushed
   are inspectable even when the next step fails closed.
2. Assert `GHCR_PUSH_TOKEN` is configured (fail closed if not), then push
   to `ghcr.io` by digest with `oras`, and print the resulting digest.
   The convenience tag is repointed alongside it but is never itself
   trust identity.

## The gate-mapping gap, on purpose

`release_may_publish_stable` requires all fifteen `ReleaseGate`s to be
present and passed. `gate-evaluation` maps thirteen of them to
already-existing `quality.yml` check-runs (several roadmap-named gates --
`ContractTests`, `OperatorFirstScopeConformance`,
`RuntimeInferenceApiTests`, `RedactionChecks`, `NoRawHandleExposureChecks`
-- map to `quality / test (ubuntu-latest)`, since the concerns they name
are genuinely exercised there via `cargo test --workspace --all-targets`,
just not as separately named check-runs yet). `CliBoundaryTests` has no
mapping: `magnetar-cli` is deliberately excluded from the root
`[workspace]` and no CI job runs its own test suite today. Until one
exists, `gate-evaluation` -- and therefore `github-release` and
`real-publish` -- correctly fails with `ReleaseGateMissing`. This is the
required fail-closed behavior, not a defect in this change: it makes a
real, previously-invisible release-readiness gap visible and named,
rather than silently skipping it. Adding that CI job is separate,
follow-up work.

## Credential seam

Two repository secrets, read only from the `release` GitHub
Environment (create it and, optionally, add required reviewers to it for
extra hardening -- neither is done by this change):

- `CARGO_REGISTRY_TOKEN` -- crates.io publish token, used only in
  `real-publish`.
- `GHCR_PUSH_TOKEN` -- a token with `packages:write` scope, used only in
  `component-artifact-distribution.yml`'s push step.

Neither is configured by this change. Every step that needs one asserts
it is present and fails with an explicit `::error::` message before any
network call if it is not -- this is deliberate: a workflow merged with
the seam unfilled should never look silently "done."

`oras` (the OCI push tool) is installed from its published GitHub
Releases binary directly, not a pinned third-party GitHub Action -- this
session's GitHub access was scoped to `astorise/Magnetar` only and could
not verify a commit SHA for a third-party action it had not already used
in this repository. Review the pinned `ORAS_VERSION` and the exact
`oras cp --from-oci-layout` invocation against your installed version
before the first real run; this session had no GHCR credentials to
execute and verify it.

## Withdrawal / correction

- **crates.io**: a bad published version is never deleted. Yank it
  (`cargo yank --version <x>`) -- the version stays reserved and visible
  as yanked but stops being installed by default -- and publish a
  corrected version under a new version number.
- **OCI (GHCR)**: a bad published digest is never deleted or
  overwritten. Publish the correction as a new artifact under its own
  new digest, then repoint the mutable convenience tag to it. The old
  digest remains resolvable but should be documented as withdrawn.

## First real publication

Both real-publish paths are manual-`workflow_dispatch`-only until a
human has:

1. Configured `CARGO_REGISTRY_TOKEN` and `GHCR_PUSH_TOKEN` as secrets on
   the `release` environment.
2. Triggered each workflow manually via `workflow_dispatch` and confirmed
   the result before relying on the tag-triggered automatic paths
   (`dry-run-publish`, `gate-evaluation`, `github-release`).

Cryptographic signing (MAG-02) is not implemented by this change; digest
pinning is the trust mechanism until it lands.

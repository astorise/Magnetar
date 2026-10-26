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
- `secret-scan <path>... --out <file>` -- scans real text files (source,
  docs, release notes) for secret-shaped content (cloud/VCS/chat tokens,
  PEM private key headers, generic `key = "..."` assignments) and writes
  a `{"secrets_detected": bool, "findings": [...]}` report. Always
  succeeds (it reports; `security-gate` is what blocks).
- `security-gate --cargo-deny-json <file> --out <file> [--release-gate-results <file>] [--secret-scan-result <file>] [flags...]` --
  evaluates `release_security::evaluate_release_security_blocking`
  against ten inputs, eight of which now come from real evidence (MAG-07,
  below); the remaining two (`checksum_mismatch`,
  `undocumented_security_exception`) stay explicit flags with no
  automated source yet.
- `oci-manifest --kind ... --input <file> --title <name> --tag <tag> --out-dir <dir> [--manifest <file>]`
  -- builds a real OCI Image Manifest and writes+verifies a local OCI
  Image Layout directory, with every digest computed from the actual
  bytes given. `--manifest <file>` embeds a second real layer holding
  that file's bytes, under `COMPONENT_MANIFEST_MEDIA_TYPE_V1`
  (`application/vnd.magnetar.component-manifest.v1+yaml`) -- required
  when `--kind component` (MAG-08, below), rejected for every other
  kind.

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
2. Push to `ghcr.io` by digest with `oras`, authenticated with the
   workflow run's own `GITHUB_TOKEN` (via `permissions: packages: write`
   -- no separately provisioned secret needed), and print the resulting
   digest. The convenience tag is repointed alongside it but is never
   itself trust identity.

## The gate-mapping gap, on purpose

`release_may_publish_stable` requires all fifteen `ReleaseGate`s to be
present and passed. `gate-evaluation` maps thirteen of them to
already-existing `quality.yml` check-runs (several roadmap-named gates --
`ContractTests`, `OperatorFirstScopeConformance`,
`RuntimeInferenceApiTests`, `RedactionChecks`, `NoRawHandleExposureChecks`
-- map to `quality / test (ubuntu-latest)`, since the concerns they name
are genuinely exercised there via `cargo test --workspace --all-targets`,
just not as separately named check-runs yet). `CliBoundaryTests` (now fed by `quality / cli boundary`) has no
mapping: `magnetar-cli` is deliberately excluded from the root
`[workspace]` and no CI job runs its own test suite today. Until one
exists, `gate-evaluation` -- and therefore `github-release` and
`real-publish` -- correctly fails with `ReleaseGateMissing`. This is the
required fail-closed behavior, not a defect in this change: it makes a
real, previously-invisible release-readiness gap visible and named,
rather than silently skipping it. Adding that CI job is separate,
follow-up work.

## Security gate inputs, field by field (MAG-07)

`evaluate_release_security_blocking` takes ten `ReleaseSecurityGateInputs`
booleans. Before this fix, `gate-evaluation`'s `security-gate` invocation
passed none of the eight non-`cargo-deny` flags, so they silently
defaulted to `false` -- the gate recorded "no secret found", "no
redaction failure", etc. for conditions it had never actually checked.
Where each of the ten now comes from:

| Field | Real source |
|---|---|
| `critical_advisory_unmitigated` | `cargo deny -f json check`'s `advisories.errors` |
| `incompatible_license_unapproved` | `cargo deny -f json check`'s `licenses.errors` |
| `secrets_detected` | `secret-scan`'s own result file (`--secret-scan-result`) |
| `redaction_gate_failed` | `release-gate-results.json`'s `RedactionChecks` gate (`--release-gate-results`) |
| `raw_handle_exposed` | `release-gate-results.json`'s `NoRawHandleExposureChecks` gate |
| `e2e_conformance_bypassed` | `release-gate-results.json`'s `E2eLocalConformance` gate |
| `openspec_validation_failed` | `release-gate-results.json`'s `OpenSpecValidation` gate |
| `trust_integrity_failed_in_fixtures` | `release-gate-results.json`'s `ContractTests` gate -- there is no gate of this exact name, but the Component/Model trust fixture tests this field is about live in `magnetar-runtime`'s own test suite, which `ContractTests` already names |
| `checksum_mismatch` | not derived here -- the bundle this would check doesn't exist yet at `gate-evaluation` time; `github-release`/`real-publish`'s own `verify-checksums` step fails the job hard on a real mismatch, a structural block rather than a recorded one |
| `undocumented_security_exception` | not derived here -- no automated exception-registry detector exists; stays an explicit, human-set flag, honestly `false` today only because every known gap is in fact documented (`SECURITY.md`'s "Known gaps"), not because this crate checked |

When `--release-gate-results <file>` is given, a gate the file does not
report counts as **failed** -- missing is blocking, never silently
`false` -- per `release_gate::gate_failed`. The explicit
`--redaction-gate-failed`/`--raw-handle-exposed`/`--e2e-bypassed`/
`--openspec-failed`/`--trust-integrity-failed`/`--secrets-detected`
switches remain as a fallback for a caller with no results file at all,
but `gate-evaluation` itself always passes both `--release-gate-results`
and `--secret-scan-result` now, so in practice eight of the ten fields
are real evidence on every run.

`secret-scan`'s own scope at `gate-evaluation` time is the checked-out
repository (source, docs, release notes) -- bundle-stage artifacts
(conformance/E2E reports, build logs, the packaged release archive named
in `release_security::SecretScanTarget`) don't exist yet at that point
and are not covered by this scan.

## The Component OCI package is self-contained (MAG-08)

Before this fix, `oci-manifest --kind component` published only the
WASM bytes as a single OCI layer. The Component's own
`*.magnetar-component.yaml` manifest -- the file
`magnetar_runtime::component::manifest_path_for_definition` requires
next to the WASM on disk, and the input every WIT/engine/capability/trust
check in `component.rs` actually validates against -- was never part of
the published package. A consumer pulling the Component by its OCI
digest (Tachyon or otherwise) got a WASM binary with no manifest, no
digest tying one to the other, and had to stage a manifest separately,
out of band, from a channel this packaging gave no guarantee about.

`oci-manifest --kind component` now requires `--manifest <file>` and
embeds it as a second real OCI layer, under
`COMPONENT_MANIFEST_MEDIA_TYPE_V1`
(`application/vnd.magnetar.component-manifest.v1+yaml` -- `+yaml`, not
`+json`: the real sidecar on disk is always YAML,
`ComponentManifest::from_yaml_bytes` is what parses it, and this
media type had never been used as a real layer's `mediaType` before
now to name the shape it was publishing). A Component OCI package is
now fully reconstructable from its manifest digest alone: both the WASM
bytes and the manifest a consumer needs to actually load it are present,
each individually digest-verified (`write_oci_layout` recomputes and
checks every layer blob's digest against what the manifest declares
before writing it, so a blob passed in the wrong order is rejected
rather than silently mislabeled). `--kind kernel-bundle`/`conformance`
have no such sidecar and reject `--manifest` if given one.

`component-artifact-distribution.yml`'s `manifest_path` workflow input
is required (checked, fail-closed, before the no-network build step
runs) whenever `kind` is `component`.

## The Kernel Exchange Bundle media type now matches its real bytes (MAG-09)

`KERNEL_BUNDLE_MEDIA_TYPE_V1` was originally defined as
`application/vnd.magnetar.kernel-bundle.v1+tar+zstd`, but nothing in this
workspace has ever produced or consumed zstd: `extract_kernel_exchange_
archive` (`magnetar-runtime`) implements plain tar and tar.gz only, and
no zstd decoder is wired into the ingestion path at all. The one real
Kernel Exchange Bundle published so far (the `cpu_reference_matmul`
fixture) is gzip-compressed, so the published package's declared media
type never matched its actual bytes -- and nothing checked that it did.

Both are now fixed: the constant is `...+tar+gzip`, and
`build_artifact_manifest` rejects `ArtifactKind::KernelExchangeBundle`
bytes that do not start with the gzip magic bytes (`0x1f 0x8b`) before
building a manifest around them, so a declared format can no longer
drift from the real content silently.

The already-published `cpu_reference_matmul` digest predates this fix
and stays resolvable exactly as published, per "Withdrawal /
correction" below -- it is not retroactively rewritten. Republishing it
(or any future Kernel Exchange Bundle) now produces a package whose
declared and actual formats agree.

## Credential seam

One repository secret, read only from the `release` GitHub Environment
(create it and, optionally, add required reviewers to it for extra
hardening -- neither is done by this change):

- `CARGO_REGISTRY_TOKEN` -- crates.io publish token, used only in
  `real-publish`. Not configured by this change; the `real-publish` step
  asserts it is present and fails with an explicit `::error::` message
  before any network call if it is not -- deliberate, so a workflow
  merged with the seam unfilled never looks silently "done."

GHCR push needs no equivalent secret: `component-artifact-distribution.yml`
authenticates with the workflow run's own `GITHUB_TOKEN`, scoped by the
job's `permissions: packages: write` grant -- the same mechanism
`github-release` already uses to create releases. A second,
separately-provisioned push token would have been a redundant credential,
not an additional safety property, since `GITHUB_TOKEN` is always present.
The real-publish gate for GHCR instead comes from the workflow staying
`workflow_dispatch`-only, plus whatever required-reviewer rule the repo
owner optionally attaches to the `release` environment.

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

1. Configured `CARGO_REGISTRY_TOKEN` as a secret on the `release`
   environment (GHCR push needs no equivalent secret -- see "Credential
   seam" above).
2. Triggered each workflow manually via `workflow_dispatch` and confirmed
   the result before relying on the tag-triggered automatic paths
   (`dry-run-publish`, `gate-evaluation`, `github-release`).

Cryptographic signing (MAG-02) is not implemented by this change; digest
pinning is the trust mechanism until it lands.

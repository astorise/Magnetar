## 1. Release-Proof Generation Tooling (no credentials required)

- [ ] 1.1 Create a new `tools/release-publish` (or similarly named) Rust
      crate, following `tools/coverage-ratchet`'s existing pattern (small,
      focused, depends on `magnetar-roadmap-contracts`).
- [ ] 1.2 Implement a build-artifact walker that produces `ArtifactChecksum`
      values (SHA-256) for every declared release artifact from a real build
      output directory.
- [ ] 1.3 Implement `verify_checksum_matches_final_artifact` usage as a
      required step before any artifact is added to the release-proof bundle.
- [ ] 1.4 Implement `SbomManifest` generation from the workspace's actual
      dependency graph (e.g. via `cargo metadata`), including
      `SbomAvailability::Generated` with real entries; wire the
      `PlaceholderDocumented` path with a limitation note for any dependency
      class not yet covered.
- [ ] 1.5 Implement `ReleaseProvenance` population from real CI/build
      context: source commit, release tag, CI run id, build target/profile,
      rustc version, lockfile digest, OpenSpec baseline digest, WIT package
      digest, conformance report digest.
- [ ] 1.6 Implement a `ReleaseSecurityGateInputs` collector that derives each
      of the ten fields from real scan/test outcomes (secret scan, advisory
      scan, license check, redaction check, raw-handle check, trust/integrity
      fixture results, E2E conformance status, OpenSpec validation status,
      checksum verification, documented-exception registry) rather than
      hardcoded `false`.
- [ ] 1.7 Add unit tests for each of the above against fixture build outputs,
      including at least one test proving a checksum mismatch is detected and
      blocks the bundle.

## 2. `release-publication` Pipeline — crates.io + GitHub Release

- [ ] 2.1 Implement the gate-evaluation step: assemble `ReleaseGateResult`s
      and `ReleaseSecurityGateInputs` from CI job outcomes and call
      `release_may_publish_stable` / `evaluate_release_security_blocking`;
      fail the pipeline with the returned error detail if either rejects.
- [ ] 2.2 Add a `cargo publish --dry-run -p magnetar-runtime` step that runs
      unconditionally (no credentials required) and is a required, non-skippable
      CI job.
- [ ] 2.3 Add the real `cargo publish -p magnetar-runtime` step, gated behind
      an explicit non-empty check of a named crates.io token secret; fail
      closed with a clear message when the secret is absent. **Requires a
      human to configure the crates.io token secret before this step can run
      for real — not performed in this change.**
- [ ] 2.4 Add the GitHub Release creation step: attach CLI/binaries,
      checksums, SBOM, provenance, and conformance reports produced by
      section 1's tooling, keyed to the same tagged commit.
- [ ] 2.5 Add workflow-level enforcement that the real-publish steps
      (2.3 and the equivalent OCI push in section 4) only run on an explicit
      `workflow_dispatch`, not automatically on tag push, until a manual run
      has succeeded at least once (tracked via a documented marker, e.g. a
      recorded successful run or a repository variable).
- [ ] 2.6 Add `.github/workflows/release-publication.yml` wiring the above
      steps in order: gate evaluation → dry-run → (manual-gated) real
      publish → GitHub Release.
- [ ] 2.7 Document the crates.io yank-based withdrawal procedure in
      `docs/release-packaging.md` (or a new `docs/release-publication.md`),
      cross-referencing this change.

## 3. Component/Kernel OCI Distribution Tooling (no credentials required)

- [ ] 3.1 Define the five versioned OCI media type constants
      (`application/vnd.magnetar.component.v1+wasm`,
      `application/vnd.magnetar.component-manifest.v1+json`,
      `application/vnd.magnetar.kernel-bundle.v1+tar+zstd`,
      `application/vnd.magnetar.kernel-manifest.v1+json`,
      `application/vnd.magnetar.conformance.v1+json`) in the
      `tools/release-publish` crate (or a shared location it and consuming
      code can both reference).
- [ ] 3.2 Implement OCI manifest construction (layers + media types +
      annotations) for a Component Artifact and for a Kernel Exchange
      Bundle, producing a manifest byte stream and its digest without
      requiring network access.
- [ ] 3.3 Implement local digest computation and a self-check that the
      constructed manifest's digest matches what would be computed after a
      round-trip push/pull, using a local OCI-layout directory (no registry
      required) for the test.
- [ ] 3.4 Add unit tests: correct media type per artifact kind; a `v2`
      media-type constant does not collide with `v1`; manifest digest is
      stable across repeated construction from identical input bytes.

## 4. `component-artifact-distribution` Pipeline — GHCR

- [ ] 4.1 Implement the GHCR push step using the manifests from section 3,
      gated behind an explicit non-empty check of a named GHCR push token
      secret; fail closed with a clear message when the secret is absent.
      **Requires a human to configure the GHCR push token secret before this
      step can run for real — not performed in this change.**
- [ ] 4.2 Implement digest capture and recording into a release provenance
      record after a successful push (or, for the credential-free path, after
      a successful local OCI-layout construction), per the design's
      tag-is-not-trust decision.
- [ ] 4.3 Implement convenience-tag repointing (e.g. `latest`,
      `<component-name>-latest`) as a separate, explicitly non-trust-bearing
      step that never substitutes for digest recording.
- [ ] 4.4 Add `.github/workflows/component-artifact-distribution.yml` wiring
      manifest construction → (manual-gated) GHCR push → digest recording,
      independent of the `release-publication.yml` workflow's schedule.
- [ ] 4.5 Verify no existing `kernel-distribution` ingestion code path
      assumes a GHCR-only source; add a regression test (or extend an
      existing one) proving ingestion of an identical-digest artifact from a
      non-GHCR local source behaves identically, per the
      "Kernel Distribution Is Transport Neutral" requirement.
- [ ] 4.6 Document the OCI digest-based withdrawal/correction procedure
      (new digest + repointed tag, old digest stays resolvable but marked
      withdrawn) alongside the crates.io yank procedure from task 2.7.

## 5. Cross-Cutting Validation

- [ ] 5.1 Run the full verification chain (`cargo fmt --check`, `cargo
      clippy --workspace --all-targets --all-features`, `cargo test
      --workspace`, `cargo doc`, OpenSpec validation, coverage ratchet) with
      the new `tools/release-publish` crate included.
- [ ] 5.2 Add an end-to-end dry-run test (script or CI job) that exercises
      gate evaluation → release-proof generation → dry-run crate publish →
      OCI manifest construction, entirely without network access or
      credentials, and asserts it completes successfully on a clean
      checkout.
- [ ] 5.3 Add a test proving a tag/alias change alone (with no new digest)
      is never sufficient to change what a consumer resolves as trusted,
      per the `component-artifact-distribution` spec's "OCI Digest Is The
      Only Trust Identity" requirement.

## 6. Documentation

- [ ] 6.1 Update `docs/release-packaging.md` and `docs/release-security.md`
      to cross-reference this change and note their previously-declared
      automation gap is now filled.
- [ ] 6.2 Add a new `docs/release-publication.md` (or extend an existing
      release doc) describing the end-to-end pipeline shape, the credential
      seam, and both withdrawal procedures, matching this change's design.md.
- [ ] 6.3 Update `README.md`'s "Magnetar and Tachyon" narrative log noting
      MAG-06 scaffolding is implemented and what remains for a human to
      configure (crates.io token, GHCR push token, first manual publish).

## 7. Deferred To Follow-Up (tracked, not implemented here)

- [ ] 7.1 Record MAG-02 (cryptographic signing) as an explicit follow-on
      chantier referencing this change's digest-pinning interim trust
      mechanism, without designing or implementing signing here.
- [ ] 7.2 Record the human follow-up actions required before real publication
      can occur: configure `CARGO_REGISTRY_TOKEN` and the GHCR push token
      secrets, then trigger the first manual `workflow_dispatch` run for
      each pipeline.

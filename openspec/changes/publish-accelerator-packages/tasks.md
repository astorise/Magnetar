Status: **Proposed (RFC lot 5, optional).** No task is implemented yet. Requires lots 0 to 4. The library repository is created only on the team's decision.

## 1. Publication

- [ ] 1.1 `tools/release-publish`: the accelerator package and kernel evidence media types; evidence as OCI referrers; manual-dispatch publication; tests for the media types and the no-overwrite rule.
- [ ] 1.2 The evidence document format (design D2) and its validation in `magnetar-kernel`; revalidation against the local target.

## 2. Library

- [ ] 2.1 The repository layout and CI (design D1):
  - builds from source per target;
  - qualification on target hardware;
  - benchmarks on Magnetar's campaign jobs;
  - signing and publication.
- [ ] 2.2 The first packages: the lot 1 `silu-mul` and `rmsnorm` CUDA variants and the lot 3 CPU variants, with evidence.

## 3. Third-party ports

- [ ] 3.1 `ported_from` provenance in the manifest; the license allowlist check; attribution blobs; SBOM entries.
- [ ] 3.2 A first port, with its attribution and qualification: `silu_and_mul` from `kernels-community/activation` (Apache-2.0) as a `fused-kernel` variant of `silu-mul`, compared with the built-in and lot 1 kernels.

## 4. Discovery, installation, revocation

- [ ] 4.1 `magnetar kernels list`: read-only, configured sources only, filtered by local hardware and loaded operators, with evidence status; no bundle bytes fetched (tested with a source that fails on any blob read).
- [ ] 4.2 `magnetar kernels install`: through lots 0 to 2, with local qualification, and no promotion unless requested.
- [ ] 4.3 The revocation feed: a signed list with a sequence number, explicit refresh, refusal of a replayed list, local application; offline file import.

## 5. External optimizers

- [ ] 5.1 Document and test C08 outputs (bundle, envelope, evidence) entering through publication or a local source, with Magnetar's promotion rules unchanged.

## 6. Validation and documentation

- [ ] 6.1 `cargo fmt`, `clippy -D warnings`, tests of both workspaces and `tools/release-publish`.
- [ ] 6.2 `docs/architecture/kernel-artifact-sources.md`: publication, evidence, ports, discovery and revocation. `SECURITY.md`: the revocation feed.
- [ ] 6.3 Show, in `magnetar kernels list`, the band a local comparison would need (10% automatic, 5% to 10% manual, lot 2 design D6). Published evidence is labelled as informative only.

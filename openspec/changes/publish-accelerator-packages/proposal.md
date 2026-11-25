## Why

Lot 5 of the [accelerator packages RFC](../../../docs/rfcs/accelerator-packages-2026-11-23.md) is optional. Its target is an ecosystem in which teams publish specialized accelerations independently of model Components, and Magnetar discovers, measures, activates and replaces the best admissible variant without a model code change or a restart.

Lots 0 to 4 make one deployment able to receive, verify, prepare and swap kernels. Several pieces are still missing:

- a library of shared packages and a way to build, qualify, sign and publish them per hardware;
- evidence that travels with an artifact;
- discovery that never installs or executes anything implicitly;
- a revocation channel;
- a clear path for kernels ported from third-party sources.

The Hugging Face Kernel Hub illustrates the last point. Its kernels are PyTorch extensions loaded from Python and built per torch and CUDA version, with no framework-independent ABI, so their binaries cannot be loaded by Magnetar's Providers. Many of their sources are under permissive licenses (`kernels-community/activation` Apache-2.0, `kernels-community/flash-attn3` BSD-3-Clause), so a port is possible. A port needs recorded provenance, license and attribution, and goes through the same qualification as any kernel.

## What Changes

- **A package library repository layout and CI**, outside the Runtime:
  - sources, kernel bundles and `magnetar:accelerator-package@1` envelopes, built from source in CI;
  - qualified per target hardware against the Reference CPU kernels;
  - benchmarked on the same jobs as the built-in kernels;
  - signed with the library's publisher key.
- **OCI publication:**
  - new media types for envelopes (`application/vnd.magnetar.accelerator-package.v1+json`) and evidence (`application/vnd.magnetar.kernel-evidence.v1+json`), beside the existing kernel bundle and manifest types;
  - evidence attached to the artifact it describes by digest, as a referrer;
  - publication by manual dispatch only, and corrections by new digests, as for Components.
- **Third-party ports:**
  - a ported kernel's manifest records `ported_from` (repository, revision, path) and the upstream license;
  - the license is checked against an allowlist, the SBOM lists it, and attribution ships with the bundle;
  - the port is qualified like any kernel, and its provenance grants nothing.
- **Discovery without installation:**
  - a CLI and Runtime catalog query lists the packages in configured sources that match the local hardware and the Model Instances' operators, with their evidence;
  - installing, preparing or promoting stays an explicit operator action under lot 0 trust and permissions.
- **Revocation feed.** A signed revocation list (keys, manifest digests) is published at the same sources and applied locally on explicit refresh. Online checking is not required.
- **External optimizers (C08).** A campaign's output is a bundle and an envelope with evidence. It enters through the same publication or a local source, and Magnetar keeps the last word on promotion.

## Capabilities

### New Capabilities

- `accelerator-package-publication`: the library build and qualification, evidence, third-party ports, discovery without implicit installation, and the revocation feed.

### Modified Capabilities

- `component-artifact-distribution`: media types for accelerator package envelopes and kernel evidence (added requirement).

## Impact

- `tools/release-publish`: the new media types and evidence referrers; manual-dispatch publication.
- `magnetar-cli`: a catalog query (`magnetar kernels list`, read-only) and an explicit install command that goes through lots 0 to 2.
- `magnetar-kernel`: `ported_from` provenance, the license allowlist check, the revocation list format and its local application.
- A new library repository (proposed: `astorise/Magnetar-accelerators`), created only when the team decides to.

## Dependencies

- Lots 0 to 4: [`define-kernel-artifact-sources`](../define-kernel-artifact-sources/proposal.md), [`prepare-prebuilt-cuda-kernels`](../prepare-prebuilt-cuda-kernels/proposal.md), [`serve-kernel-bundles-and-swap-generations`](../serve-kernel-bundles-and-swap-generations/proposal.md), [`load-specialized-cpu-kernels`](../load-specialized-cpu-kernels/proposal.md), [`prepare-compiled-graph-artifacts`](../prepare-compiled-graph-artifacts/proposal.md).
- C08 in [`run-bounded-external-autotuning`](../run-bounded-external-autotuning/proposal.md) for optimizer-produced variants.

## Compatibility and authority

- Discovery is read-only. Nothing is fetched beyond catalog metadata, and nothing is installed, prepared or promoted without an explicit operator action.
- Sources are operator-configured. The Runtime never follows a locator from a package, envelope or evidence document on its own authority.
- Publication keeps the existing rules: OCI digest as the only identity, manual dispatch, no overwrite.

## Non-goals

- A hosted registry service, ratings, or automatic promotion from published evidence.
- Loading PyTorch extensions or any framework-bound binary.
- A mandatory dependency on Tachyon or on any external optimizer.

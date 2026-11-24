## Context

- Release tooling (`tools/release-publish`) already publishes Components, kernel bundles, kernel manifests and conformance fixtures to GHCR, on manual dispatch, under versioned media types.
- Lot 0 adds kernel trust with publisher keys and revocation. Lots 1 to 4 add preparation, hot swap, CPU kernels and envelopes.
- `run-bounded-external-autotuning` (C08) defines the external optimization plane's boundary.

## Goals / Non-Goals

**Goals:**

- A repeatable path from a kernel source to a signed, evidenced, published package.
- Evidence anyone can check against the exact artifact it describes.
- Discovery that informs and never acts.
- Ported third-party kernels with honest provenance and licensing.

**Non-Goals:**

- A hosted service, automatic promotion, framework-bound binaries.

## Decisions

### D1. The library repository

```text
Magnetar-accelerators/
  packages/<name>/
    package.json            # envelope template (accelerator-package@1)
    src/<variant>.(cu|c|wgsl)
    LICENSE, NOTICE         # upstream license and attribution for ports
    qualification.json      # shapes, dtypes, tolerances (generated-kernel-qualification)
  ci/                       # build per target, qualify, benchmark, sign, publish
```

- **Builds** happen from source in CI: NVRTC for CUBIN, the system compiler for CPU variants. No prebuilt binary enters the repository.
- **Qualification** runs on the targeted hardware against the Reference CPU kernels. A variant without a qualification run for a target is not published for it.
- **Benchmarks** use Magnetar's own campaign jobs (the same machine as the built-in kernels) and are published as evidence, never as a selection decision.

### D2. Evidence as referrers

- An evidence document records:
  - the artifact's bundle identity and blob digest;
  - the target (Device, driver, Provider revision);
  - the qualification profile, its result and the oracle identity;
  - optionally a benchmark workload profile and its measurements.
- It is published under `application/vnd.magnetar.kernel-evidence.v1+json`, as an OCI referrer of the bundle.
- The Runtime revalidates evidence against current policy (`Portable Evidence Is Not Automatically Current`). Evidence for another driver or Device informs but does not qualify.

### D3. Third-party ports

- **Manifest provenance:** `ported_from { repository, revision, path, license }`. The license is checked against an allowlist (Apache-2.0, BSD-2/3-Clause, MIT by default). A copyleft or unknown license blocks publication.
- **Packaging.** `NOTICE` and the upstream license ship as bundle blobs with an `attribution` role, and the SBOM lists the upstream component.
- **Example.** A port of `kernels-community/activation`'s `silu_and_mul` CUDA source (Apache-2.0) is a `fused-kernel` variant of Magnetar's `silu-mul`. It is qualified against the Reference CPU `silu-mul` and published only if it passes. Its Hugging Face origin grants no trust.

### D4. Discovery and installation

- `magnetar kernels list [--source <configured>]` reads envelopes and evidence metadata from configured sources. It filters them by local Providers, Devices and features, and by the operators of loaded Model Instances.
- It prints the variants with their evidence status: qualified on this exact target, qualified elsewhere, or none.
- It fetches no bundle bytes.
- `magnetar kernels install <envelope digest> [--variant <id>]` takes the chosen bundles through lots 0 to 2 (source, trust, permission, qualification on this machine, preparation). It does not promote unless the operator also asks to.

### D5. Revocation feed

- **Format:** a signed list of revoked publisher keys and manifest digests, with a monotonic sequence number, published at each configured source.
- **Refresh** is explicit (CLI or a configured schedule). A list with a lower sequence number than the last applied one is refused. Entries apply locally through lot 0's trust store and the registry's revocation path.
- **Offline deployments** apply the list from a file.

## Risks / Trade-offs

- **A library is maintenance.** Each package needs owners and hardware for qualification. The RFC marks this lot optional, and the repository is created only on the team's decision.
- **Ports diverge from upstream.** The recorded revision makes divergence visible. Updating a port is a new bundle, qualified again.

## Open Questions

- How the library's publisher key is rotated. RFC Q4 is decided (lot 0, design D3): the library signs with its own key, and each deployment's operator decides whether to authorize it.
- RFC Q9 is decided (lot 2, design D6): published evidence informs, and promotion needs a comparison on the deployment's own hardware target.

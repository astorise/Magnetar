## Context

The RFC chain is: source → ingestion → validation and qualification → Provider preparation → Kernel Registry → selection and dispatch. In `magnetar-kernel`, every box after the source exists as contract types with conformance tests. None of them is fed by a production source, and none ends in a Provider that prepares an external artifact:

| Step | Exists | Missing |
| --- | --- | --- |
| Bytes | `KernelExchangeBundle`, `extract_kernel_exchange_archive`, `KernelExchangeArchiveLimits` | a source contract other than a local path; Component-served bytes |
| Ingestion | `KernelIngestionTransaction`, `IngestionQuotas`, `QuarantineNamespace`, `ArtifactSourceAuthority`, `RevokedArtifactRegistry` | an observed source for a Component |
| Trust | `KernelSignatureEnvelope` (metadata), `evaluate_artifact_trust(bool)` | signature verification, publisher keys, execution permission |
| Registry | generations, `promote_generation*`, `rollback_generation`, `revoke_kernel*` | candidates backed by externally prepared kernels (lot 1) |

Component and Model Artifacts already verify Ed25519 signatures through `magnetar-core::artifact_signature` (`SignatureRecord`, `key_id_for`, `verify_signature`) against stores with `trusted_publisher_keys` and `revoked_keys`. Kernel trust should reuse that verifier, not invent a second scheme.

## Goals / Non-Goals

**Goals:**

- A source contract that any later transport (Component, OCI, local, cache) satisfies, with bounds enforced by the Runtime rather than promised by the source.
- Distinct, named identities for what is being trusted, what is being transported and what is being executed.
- A kernel trust decision that a cryptographic signature can establish, separate from the permission to execute native code.
- One delivery lifecycle in which delivery, preparation and activation are separate stages with separate authorities.

**Non-Goals:**

- Any Provider-side loading, any WIT change, any OCI client change.

## Decisions

### D1. Four identities

| Identity | Definition | Proves |
| --- | --- | --- |
| Manifest digest | sha256 of the canonical `kernel.manifest.json` bytes (`KernelManifestV1::digest`) | which manifest; what a publisher signs |
| Blob digest | sha256 of one blob (`KernelBlobDigest`) | which bytes a Provider receives |
| Bundle identity | the manifest digest together with the set of blob digests it references, all verified | the logical artifact; the cache and registry key |
| Transport digest | an OCI manifest or archive digest | one packaging of the bundle; never trust |

The RFC warns against conflating them. A repacked bundle keeps its bundle identity (`Repacking Does Not Change Logical Identity`). A transport digest is recorded for audit only.

### D2. A bounded reader the Runtime drives

```text
trait KernelArtifactSource {
    fn descriptors(&self) -> Result<Vec<SourceBundleDescriptor>, SourceError>;   // claims
    fn read_chunk(&self, manifest_digest, offset, max_bytes) -> Result<Vec<u8>, SourceError>;
}
```

The gateway, not the source, owns the loop:

- it caps every chunk (`max_bytes` at most the policy chunk size) and rejects a chunk longer than requested;
- it stops at the policy's cumulative byte quota, whatever the declared size says;
- it hashes progressively, so the archive digest is known when the last chunk arrives;
- it enforces a deadline and a chunk-count limit;
- it hands the staged bytes to `KernelIngestionTransaction`, whose existing validation (archive limits, path safety, manifest parse, blob integrity) is unchanged.

One reader shape serves a Component export (lot 2), an OCI blob fetch through `ArtifactSourceAuthority`, a local file and a cache entry, so each is tested against the same bounds.

### D3. Trust and execution permission are two decisions

- **Trust** answers "were these bytes published by someone we trust?". `KernelTrustStore::evaluate(manifest, envelope)` verifies the envelope with `verify_signature`:
  - the signed digest is the manifest digest;
  - the key comes from `trusted_publisher_keys` and is not in `revoked_keys`;
  - a pinned manifest digest is an alternative for development.

  The outcome is one of `Trusted { publisher }`, `Untrusted`, `Revoked` or `Invalid`, mirroring the Component and Model stores.
- **Execution permission** answers "may Provider P execute format F from publisher X on this deployment?". `KernelExecutionPermission` entries are deployment configuration, denied by default. `nvidia:cubin` on the CUDA Provider and a native shared object on the CPU Provider (lot 3) are different permissions, because their blast radius differs.
- **Preparation eligibility** requires trust, permission, qualification for the requested envelope, and no revocation.

A Component's signature authenticates the Component's code. Bundles embedded in it carry their own envelope and are verified as kernels.

**Decentralized signing, centralized authorization** (RFC Q4, decided):

- **Signing is decentralized.** Each publisher signs its own packages with its own private key, with its own tooling, at publication time.
- **Authorization is centralized** in the deployment. The Runtime verifies signatures only against the public keys that the deployment's operator has authorized in the kernel trust store. A valid signature under a key the operator has not authorized is untrusted.
- **No private key on the Runtime side.** Neither the Runtime nor any Component holds, receives or uses a private key, and no WIT interface carries key material or a signing operation. A kernel-package Component serves bytes and signatures produced elsewhere.
- **A Component confers no trust on the binaries it serves**, whatever its own signature or publisher.

### D4. The delivery lifecycle maps onto existing states

| Stage | Authority | Existing state |
| --- | --- | --- |
| Discovered | source | — (a descriptor claim) |
| Staged | gateway | `IngestionState::Staged` |
| Ingested: accepted, quarantined or rejected | ingestion policy | `IngestionDecisionKind` |
| Qualified | qualification | `QualificationStatus` |
| Prepared | Provider | `PreparedKernelState` |
| Benchmarked | benchmark profile | `kernel_benchmark` evidence |
| Ready | registry | candidate eligible, not selected |
| Active | promotion | `promote_generation` |
| Draining, Retired | registry | retiring and destroyed prepared kernels |
| Rolled back | promotion | `rollback_generation` |

A `KernelDeliveryStage` value records the stage reached per bundle identity and Provider, with the timestamped authority that moved it. Transitions skip no stage: a staged bundle cannot be prepared, and a prepared one cannot be active without promotion. Receiving bytes never implies execution.

### D5. Answers to RFC questions

- **Q1, publication granularity.** One logical bundle per semantic binding, with several compiled variants (manifest v1 already supports multiple compiled variants per artifact list). Distinct semantics are distinct bundles.
- **Q3, association with model Components.** No association in the model WIT. The Runtime resolves bundles from an independent catalog through policy. A model Component's recommendations, if ever added, stay advisory (`Recommendation Is Advisory`).
- **Q4, trust policy.** Decided (D3): decentralized signing by each publisher; centralized authorization of public keys by the deployment's operator; no private key in the Runtime or a Component; no trust conferred by a serving Component; local revocation lists and per-Provider execution permissions.

## Risks / Trade-offs

- **The bounded reader adds copies.** Bundles are small (kernels, not weights), so a copy per chunk is acceptable. A source with memory-mapped bytes can still satisfy the trait.
- **Two decisions where there was one boolean.** A deployment must grant permissions explicitly. That is intended: the RFC's limit is that signatures and tests do not prove native code harmless.
- **The lifecycle record duplicates state** held by ingestion and the registry. It is derived and audited, never authoritative: the authoritative state stays where it is today.

## Open Questions

- Whether kernel revocation lists share a file and format with Component and Model revocations.

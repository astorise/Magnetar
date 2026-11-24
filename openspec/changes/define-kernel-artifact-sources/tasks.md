Status: **Proposed (RFC lot 0).** No task is implemented yet.

## 1. Kernel Artifact Source

- [ ] 1.1 `magnetar-kernel`:
  - `KernelArtifactSource` (descriptors, `read_chunk`), `SourceBundleDescriptor` (manifest digest, declared size as a claim) and `SourceKind` (Component-served, local path, OCI locator, cache);
  - `ObservedIngestionSource::ComponentServed { component_digest }`.
- [ ] 1.2 The gateway read loop: chunk cap, oversized-chunk rejection, cumulative quota, chunk-count limit, deadline, progressive sha256, staging into `KernelIngestionTransaction`. OCI and local sources are reached only through `ArtifactSourceAuthority`.
- [ ] 1.3 `KernelBundleIdentity` (manifest digest plus verified blob digest set) as the key of trust, cache, registry and audit records; the transport digest is recorded for audit only.
- [ ] 1.4 `KernelDeliveryStage` and the transition table of design D4, each transition with its authority; the record is derived from ingestion and registry state, never authoritative.

## 2. Kernel execution trust

- [ ] 2.1 `KernelTrustStore`: trusted publisher keys, revoked keys and pinned manifest digests; envelope verification over the manifest digest with `magnetar_core::artifact_signature::verify_signature`; outcomes trusted, untrusted, revoked, invalid.
- [ ] 2.2 `KernelExecutionPermission` (Provider identity, artifact format, publisher), denied by default, loaded from deployment configuration.
- [ ] 2.3 `preparation_eligibility(bundle, provider, envelope)`: trusted, permitted, qualified, target-compatible, not revoked or quarantined. `evaluate_artifact_trust(bool)` stays for callers without a store.

## 3. Tests

- [ ] 3.1 Sources:
  - an in-memory source and a scripted adversarial source: size lie, oversized chunk, endless reads, past the deadline, wrong digest, corrupt archive;
  - each scenario of `kernel-artifact-source` as a test;
  - a repacked bundle resolving to the same identity.
- [ ] 3.2 Trust:
  - a valid signature, a signature over another digest, a revoked key, envelope metadata alone, a pinned digest;
  - a trusted Component serving an unsigned bundle;
  - a valid signature under a key the operator has not authorized;
  - a Component serving its own public key.

  A check that no WIT or host call carries key material or a signing operation.
- [ ] 3.3 Permission: trusted without permission, permission per format, revocation after qualification.
- [ ] 3.4 The lifecycle: delivery without preparation, a skipped stage refused, `KernelRegistrationAuthority::Component` still refused.
- [ ] 3.5 `cargo fmt`, `clippy -D warnings` and the tests of both workspaces; `magnetar-kernel` on `wasm32-unknown-unknown` (the source and trust types build there; OCI and file sources are native only).

## 4. Documentation

- [ ] 4.1 `docs/architecture/kernel-artifact-sources.md`:
  - the vocabulary (Kernel Exchange Bundle, Kernel Artifact Source, `kernel-package` role, `OperatorKernel` / `FusedKernel` / `CompiledGraph`);
  - the four identities and the delivery lifecycle;
  - trust versus execution permission.
- [ ] 4.2 Link it from `docs/architecture/kernel-registry-dispatch.md` and from `SECURITY.md`'s implemented controls once implemented.
- [ ] 4.3 Record the answers to RFC questions 1, 3 and 4 (design D5) in the RFC's follow-up notes.

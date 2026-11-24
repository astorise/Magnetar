## Why

The [accelerator packages RFC](../../../docs/rfcs/accelerator-packages-2026-11-23.md) proposes kernels distributed independently of models and Providers, served by WebAssembly Components or resolved from OCI. They would run only after Runtime admission, qualification and Provider preparation. This change is its lot 0: the contracts the later lots build on.

`magnetar-kernel` already defines most of the chain:

- the Kernel Exchange Bundle: manifest v1, canonical identity, blob integrity, archive extraction, path safety;
- the ingestion gateway: transactions, quotas, quarantine, revocation;
- registry generations: promotion, rollback, revocation;
- the OCI media type for bundles.

Three gaps keep any external kernel from running:

1. **No source contract.** Bundles are read from local files in tests only. Nothing bounds another source's reads, a Component cannot serve bytes, and manifest, blob, bundle and transport digests are not named apart.
2. **No trust for executable code.** Kernel trust is a policy boolean (`evaluate_artifact_trust`), and the manifest's signature envelope is unverified metadata. Components and Model Artifacts verify Ed25519 signatures; kernel bundles do not. Nothing separates trusting bytes from permitting a Provider to execute them.
3. **No delivery lifecycle.** The RFC separates delivery, preparation and activation. Ingestion and promotion are specified apart, with no rule that one never implies the next.

## What Changes

- **Vocabulary.** A canonical architecture document defines:
  - the Kernel Exchange Bundle (existing);
  - the Kernel Artifact Source;
  - the `kernel-package` Component role (a source of bytes, never a registrant);
  - the accelerator artifact kinds `OperatorKernel`, `FusedKernel` and `CompiledGraph`. `CompiledGraph` is only named here; lot 4 specifies it.
- **Kernel Artifact Source** (`magnetar-kernel`):
  - source kinds: served by a Component, local path, authorized OCI locator, cache;
  - a bounded reader (`read_chunk(bundle, offset, max_bytes)`) and a source descriptor whose declared size is a claim;
  - four distinct identities: manifest digest, blob digests, bundle identity, transport digest;
  - a gateway step that enforces chunk size, cumulative quota, deadline and offset bounds, and hashes as it reads, before handing staged bytes to the existing ingestion transaction.
- **Kernel execution trust** (`magnetar-kernel`, reusing `magnetar-core`'s Ed25519 verification):
  - a `KernelTrustStore` with trusted publisher keys, revoked keys and pinned manifest digests, which verifies a bundle's signature envelope over its manifest digest;
  - a separate, deployment-owned `KernelExecutionPermission` per Provider identity, artifact format and publisher, denied by default;
  - preparation requires trusted bytes, execution permission and qualification. A Component's own signature never extends to the bundles it serves.
- **Delivery lifecycle.** One stage sequence: discovered, staged, ingested (accepted, quarantined or rejected), qualified, prepared, benchmarked, ready, active, draining, retired, rolled back. It maps onto the existing ingestion and registry states, and each transition has its own authority. No stage implies the next.
- **Contract tests** for each acceptance and rejection scenario below: corrupt bytes, wrong digest, size claim exceeded, chunk overflow, an unsigned bundle under the production policy, a revoked key, a missing execution permission, a Component-signed but kernel-unsigned bundle.

No Provider loads anything in this lot. Lot 1 (`prepare-prebuilt-cuda-kernels`) is the first execution.

## Capabilities

### New Capabilities

- `kernel-artifact-source`: bounded, authority-checked sources of Kernel Exchange Bundle bytes, distinct identities, and the delivery lifecycle.
- `kernel-execution-trust`: verified kernel publisher signatures, a kernel trust store, and execution permissions separate from trust.

### Modified Capabilities

- `kernel-artifact-ingestion`: the observed source of a Component-served bundle (added requirement).

## Impact

- Owning layer: Runtime (`magnetar-kernel`: artifact sources, trust store, lifecycle types); `magnetar-core` signature verification is reused, not changed.
- Documentation: `docs/architecture/kernel-artifact-sources.md`; `docs/architecture/kernel-registry-dispatch.md` links to it.
- No WIT change (the `kernel-package` world is lot 2), no Provider change, no Component change.

## Dependencies

None. Lots 1 to 5 depend on this one:

- [`prepare-prebuilt-cuda-kernels`](../prepare-prebuilt-cuda-kernels/proposal.md);
- [`serve-kernel-bundles-and-swap-generations`](../serve-kernel-bundles-and-swap-generations/proposal.md);
- [`load-specialized-cpu-kernels`](../load-specialized-cpu-kernels/proposal.md);
- [`prepare-compiled-graph-artifacts`](../prepare-compiled-graph-artifacts/proposal.md);
- [`publish-accelerator-packages`](../publish-accelerator-packages/proposal.md).

## Compatibility and authority

- Components keep requesting Capabilities. A `kernel-package` Component exports bytes only: it imports nothing, selects no Provider or Device, and receives no native handle. `KernelRegistrationAuthority::Component` keeps `may_register_kernel() == false`.
- The Runtime keeps admission, trust, permission, selection, promotion, rollback and audit. Providers keep loading, native handles, execution and release.
- An OCI source is reachable only through an operator-configured `ArtifactSourceAuthority`. No source grants filesystem, network, process or secret authority to a Component. Manifest locators are never followed on their own authority.
- Existing bundle validation, ingestion and registry behaviour are unchanged. The trust store is additive: a deployment without kernel keys keeps today's fail-closed default (`Untrusted` unless policy approves).

## Non-goals

- Executing an external kernel (lot 1), the `kernel-package` WIT world and generation hot swap (lot 2), a CPU native ABI (lot 3), compiled graphs (lot 4), publication (lot 5).
- Online revocation checking, key distribution infrastructure, or a mandatory Tachyon dependency.
- Treating a digest, a signature or a passed test as proof that native code is harmless: the execution permission is a deployment decision.

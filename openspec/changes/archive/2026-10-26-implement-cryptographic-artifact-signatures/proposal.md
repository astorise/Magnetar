## Why

Component and Model Artifact trust today is digest-pinning and explicit
local-development-policy only (`ComponentTrustStore`, `ModelTrustStore`).
Publisher and source identity are recorded as metadata but never grant trust
on their own -- correct and fail-closed, but it means every new release of a
Component or Model needs its digest re-pinned by hand, and there is no way
to cryptographically authenticate "this artifact was produced by publisher
X" at all.

The design for this (what gets signed, algorithm, key identification,
trust/revocation, verification precedence) was already settled in
[`docs/cryptographic-artifact-signatures.md`](../../../docs/cryptographic-artifact-signatures.md)
(tracks GitHub issue #37), whose own header states implementation belongs in
a separate OpenSpec Change once the design questions were answered. This is
that Change. `SECURITY.md`'s "Known gaps" section and the round-6
architecture audit (MAG-02) both point at this as the remaining
"design only, not implemented" gap in Magnetar's supply-chain trust model.

## What Changes

- Add Ed25519 signature verification as a **second, independent** path to a
  `Trusted` decision in both `ComponentTrustStore::evaluate` and
  `ModelTrustStore::evaluate`, alongside (not replacing) existing digest
  pinning. Every artifact trusted today by digest pinning alone continues to
  be trusted identically after this change ships.
- Add `trusted_publisher_keys` (key id -> public key + bound
  `PublisherIdentity`) and `revoked_keys` to both trust stores, loadable
  from the existing YAML trust-store file format alongside
  `trusted_digests`/`rejected_digests`/`revoked_digests`.
- Add a `SignatureRecord` (key id, Ed25519 signature bytes, signed digest
  string) field to `ComponentArtifactPackage`/manifest and `ModelManifest`
  loading paths. The signature covers the artifact's existing SHA-256
  content digest (`ComponentDigest`/`ModelDigest`), not raw bytes.
- Define fail-closed verification precedence: revoked key -> `Revoked`;
  unknown key -> falls through to digest pinning (not an error); known key
  but broken/mismatched signature -> `Rejected` (stronger negative signal
  than no signature); known key and valid signature -> `Trusted` with an
  **authenticated** `PublisherIdentity` (the identity the operator bound to
  that key, never a self-declared artifact field).
- Add key revocation (`revoked_keys`), symmetric with existing digest
  revocation, checked before signature verification.
- Update `SECURITY.md`'s "Known gaps" entry once implemented, to describe
  what is now verified instead of pointing at the design document.

## Capabilities

### New Capabilities

(none -- this extends existing trust capabilities, it does not introduce a
new one)

### Modified Capabilities

- `component`: `ComponentTrustStore`/trust evaluation gains a signature
  verification path and authenticated publisher identity. Modifies
  "Signature Metadata Is Optional and Non-Authoritative", "File-Based Trust
  Store", and "Revocation"; adds a new requirement for authenticated
  publisher identity via a verified signature.
- `model`: `ModelTrustStore`/trust evaluation gains the same signature
  verification path. Modifies "Model Artifact Trust"; adds new requirements
  for signature-based trust, key revocation, and authenticated publisher
  identity (the model spec today has no dedicated publisher-metadata
  requirement even though `ModelTrustStore.trusted_publishers` already
  exists in code -- this change adds it as the direct contrast point for
  the new authenticated path).

## Impact

- `magnetar-runtime`: `src/component.rs` (`ComponentTrustStore`,
  `ComponentTrustDecision`, manifest/package loading), `src/model.rs`
  (`ModelTrustStore`, `ModelTrustDecision`, manifest loading). New
  dependency: an Ed25519 implementation (crate choice made in design.md;
  `docs/cryptographic-artifact-signatures.md` deferred the exact crate to
  this Change).
- `SECURITY.md`: "Known gaps" entry updated once implemented.
- No breaking change to existing trust store YAML files or existing digest-
  pinning behavior -- new fields are additive and optional.
- MAG-06's release-publication pipeline (`tools/release-publish`,
  `docs/release-publication.md`) is unaffected: it continues to use digest
  pinning as its trust mechanism; this change does not require the release
  pipeline itself to start signing anything, only makes verification
  possible when a publisher does sign.

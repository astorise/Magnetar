# Cryptographic Artifact Signatures And Authenticated Publisher Identity

Status: design only, not implemented. Tracks GitHub issue
[#37](https://github.com/astorise/Magnetar/issues/37). Implementation, once
these questions are settled, belongs in a separate OpenSpec Change -- this
document exists to answer the design questions first, per the issue's own
Non-Goals ("Implementation... should be tracked separately").

## Why this is still open

Component and Model Artifact trust today (`ComponentTrustStore`,
`ModelTrustStore`, `magnetar-runtime/src/component.rs` and `model.rs`) is
digest-pinning and explicit-local-development-policy only. Publisher and
source identity are recorded as metadata but never used to grant trust on
their own (`Requirement: Source-Declared Digest Is A Claim`,
`Requirement: Publisher Metadata Is Not Sufficient Trust`, both already
canonical). This is fail-closed and correct, but it means Magnetar has no
way to trust an artifact it has never seen the exact bytes of before --
every new release of a Component or Model needs its digest re-pinned by
hand, and there is no way to authenticate "this artifact was produced by
publisher X" cryptographically at all.

## What gets signed

Two distinct signable units, matching the two existing trust stores:

- **Component Artifacts** -- the WASM Component binary plus its manifest,
  as a pair (`ComponentArtifactPackage`, `component.rs`). Signing the pair,
  not the binary alone, is required: a manifest swap (different
  capabilities, different distribution source) without a binary change
  must invalidate the signature, since the manifest is part of what a
  publisher is vouching for.
- **Model Artifacts** -- the weight/manifest bundle Model Loading consumes
  (`ModelManifest`/`ModelArtifactId`, `model.rs`). Per-tensor content
  digests (`ModelTensorMetadata.digest`, added by
  `bind-materialized-weight-content-to-model-artifact-digests`) are a
  separate, finer-grained mechanism and are out of scope here -- artifact
  signing operates at the same whole-artifact granularity
  `ModelArtifactId.digest` already does.

In both cases, **the signature is computed over the artifact's existing
SHA-256 content digest, not over the raw bytes directly.** Both
`ComponentDigest::sha256` and `ModelDigest::sha256` already exist and are
already the canonical identity for these artifacts; signing the digest
(a fixed 32-byte value) rather than re-hashing or re-reading potentially
large artifact bytes at verification time reuses that identity instead of
introducing a second, parallel notion of "what this artifact is."

## Algorithm

**Ed25519** (RFC 8032), via the same `sha2`/hashing-adjacent dependency
posture the crate already has (an `ed25519-dalek`-shaped dependency, exact
crate choice deferred to implementation).

Rationale:

- Deterministic signatures (no per-signature randomness requirement that a
  broken RNG could catastrophically undermine, unlike ECDSA/DSA).
- Small keys (32 bytes) and signatures (64 bytes) -- cheap to embed in a
  manifest field, cheap to carry through `ModelManifest`/component
  manifest YAML without meaningfully growing artifact metadata.
- Fast verification, no parameter/curve choices to get wrong (unlike RSA's
  key-size and padding-scheme decisions).
- Already the de facto standard for software-artifact signing in adjacent
  ecosystems (sigstore/cosign, apt/dpkg's newer signing paths, SSH
  certificate signing) -- operators integrating Magnetar into an existing
  signing pipeline are likely to already have Ed25519 keys and tooling.
- No known ecosystem-specific reason to prefer something else: the WASM
  Component Model / WIT ecosystem does not mandate a particular signature
  scheme for this purpose today.

## Key identification

A **key id** is the first 16 bytes (hex-encoded, 32 characters) of
`SHA-256(public_key_bytes)` -- the same truncated-fingerprint convention
SSH and PGP both use, chosen for familiarity rather than inventing a new
one. A signature record carries:

```text
SignatureRecord {
    key_id: String,        // hex fingerprint, as above
    signature: [u8; 64],   // Ed25519 signature over the artifact's digest bytes
    signed_digest: String, // the exact digest value signed, e.g. "sha256:...",
                            // carried alongside rather than assumed, so a
                            // verifier never has to guess which digest
                            // representation (with/without algorithm
                            // prefix, case) produced this signature
}
```

`key_id` is a lookup hint, not itself trust-bearing -- two different public
keys could theoretically collide on a truncated fingerprint (astronomically
unlikely at 128 bits, but not impossible by construction), so verification
always checks the full signature against the full public key the trust
store has on file for that id, never against the id alone.

## Trusted public keys

Both `ComponentTrustStore` and `ModelTrustStore` gain a new field:

```text
trusted_publisher_keys: BTreeMap<KeyId, (PublicKey, PublisherIdentity)>
```

This composes with, and does not replace, the existing
`trusted_digests`/`rejected_digests`/`revoked_digests` (and, for
Components, `quarantined_digests`/`allow_unsigned_local_development`)
fields -- signature verification is a **second, independent way** to reach
a `Trusted` decision, evaluated alongside digest pinning, not instead of
it. An artifact can be trusted by digest pinning alone (no signature
needed, today's existing path, unchanged), by a valid signature from a
trusted key alone (new), or by both (redundant but harmless).

Distribution: **operator-configured only**, the same posture the existing
trust stores already have (`ComponentTrustStore::load_yaml`, a local
policy file). No key registry, discovery protocol, or "well-known keys"
list is in scope -- the deployment authority explicitly adds the public
keys it trusts, the same way it explicitly pins digests today. A future,
separate design could add a registry/TUF-style distribution mechanism on
top of this without changing the verification model itself.

## Authenticated publisher identity

A publisher identity claim becomes **authenticated** exactly when:

1. The artifact carries a `SignatureRecord` whose `signed_digest` matches
   the artifact's actual computed digest (not just the manifest-declared
   one -- the same "declared digest is a claim, computed digest is truth"
   principle `Requirement: Source-Declared Digest Is A Claim` already
   establishes), and
2. `key_id` resolves to a public key in the trust store's
   `trusted_publisher_keys`, and
3. The signature verifies against that public key.

The `PublisherIdentity` bound to that key in the trust store (not any
identity string the artifact itself declares) is what a caller may then
treat as authenticated. This inverts today's model correctly: today,
`ModelManifest`/Component manifest publisher metadata is a self-asserted
string that trust policy explicitly ignores
(`Requirement: Publisher Metadata Is Not Sufficient Trust`); under this
design, an artifact's *self-declared* publisher field is still never
trust-bearing on its own, but a *verified signature from an
operator-trusted key* carries an identity the *operator* attached to that
key, not one the artifact asserts about itself.

## Revocation

Symmetric with the existing `revoke_digest`/`revoked_digests` mechanism:

```text
revoked_keys: BTreeSet<KeyId>
```

A revoked key's signatures are rejected outright, regardless of whether
the digest they cover is independently pinned trusted -- revocation is
checked first, before signature verification even runs, mirroring
`ModelTrustStore::evaluate`'s existing precedence (`revoked_digests`
checked before `rejected_digests` before `trusted_digests`). Revoking a
key does not retroactively distrust artifacts that were *also* trusted by
digest pinning independent of that key -- revocation only removes the
*signature* as a trust path, the same way `revoked_digests` only removes
that specific digest's standing, not every artifact that happens to share
some other property with it.

No online revocation checking (OCSP-style, CRL fetching) is in scope --
revocation lists are operator-configured local policy, the same
distribution model as trusted keys themselves. An operator wanting
timely revocation propagation is responsible for updating their local
trust store's `revoked_keys`, the same way they are already responsible
for updating `revoked_digests` today.

## Verification policy

Fail-closed by default, consistent with every existing trust decision in
this codebase:

- **No signature present**: identical to today -- `Unknown` unless digest
  pinning or explicit local development policy grants trust. Absence of a
  signature is never itself a rejection; it simply does not grant the
  *new* trust path this design adds. Every artifact that is trusted today
  by digest pinning alone continues to be trusted identically after this
  design ships.
- **Signature present but key unknown** (`key_id` not in
  `trusted_publisher_keys`): does not grant trust through the signature
  path; falls through to digest pinning exactly as if no signature were
  present. Not an error -- an artifact may be legitimately signed by a
  publisher this particular deployment simply has not chosen to trust.
- **Signature present, key known, but signature does not verify** (wrong
  key, corrupted signature, `signed_digest` mismatch against the actual
  artifact): `Rejected`, not `Unknown` -- a *present but broken* signature
  is a stronger negative signal than no signature at all (it suggests
  either corruption or an attempted forgery under a claimed key id), so
  this fails closed harder than the "no signature" case rather than
  silently falling back to digest pinning.
- **Signature present, key known and trusted, key revoked**: `Revoked`,
  checked before verification as described above.
- **Signature present, key known and trusted, signature verifies**:
  `Trusted`, with the bound `PublisherIdentity` now authenticated per the
  section above.

This mirrors `ModelTrustStatus`/`ComponentTrustStatus`'s existing
enumeration (`Trusted`, `Rejected`, `Revoked`, `Unknown`, and Components'
`Quarantined`) -- no new status variant is needed, only new *paths* to the
existing ones.

## Non-Goals (of this design, and of the eventual implementation)

- **Key distribution infrastructure** (registries, TUF, well-known-keys
  discovery) -- operator-configured keys only, as with digest pinning
  today.
- **Online revocation checking** -- local, operator-updated revocation
  lists only.
- **Provider signing** -- Providers are trusted native code by
  architectural definition (`SECURITY.md`'s Scope section) and are
  explicitly out of this design's scope, the same as the tracking issue
  states.
- **Per-tensor or per-shard signatures** -- whole-artifact granularity
  only, matching `ComponentDigest`/`ModelArtifactId.digest`'s existing
  granularity; per-tensor content digests remain a separate, already-
  shipped mechanism for a different purpose (content-tamper detection,
  not publisher authentication).
- **Deciding the exact Rust crate for Ed25519** -- a concrete dependency
  choice belongs in the implementing Change, not this design.

## Definition of done for the follow-up implementation Change

- `ComponentTrustStore` and `ModelTrustStore` gain `trusted_publisher_keys`
  and `revoked_keys`, with a `PublisherIdentity` type bound per key.
- `ComponentTrustDecision`/`ModelTrustDecision`'s `evaluate` gain the
  signature-verification path described above, in the precedence order
  specified (revocation before verification; verification failure is
  `Rejected`, not `Unknown`).
- `ComponentArtifactPackage`/`ModelManifest` (or their loading paths) gain
  an optional `SignatureRecord` field/section.
- Tests: valid signature from a trusted key grants `Trusted` with the
  correct authenticated `PublisherIdentity`; unknown key falls through to
  digest pinning; broken signature under a known key is `Rejected`, not
  `Unknown`; revoked key is `Revoked` even when the covered digest is
  separately pinned trusted; absence of a signature behaves identically
  to today for every existing digest-pinning test.
- `SECURITY.md`'s "Known gaps" bullet is updated once implementation
  lands, to describe what is now verified rather than pointing at this
  design document.

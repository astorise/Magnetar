## Context

`ComponentTrustStore` (`magnetar-runtime/src/component.rs`) and
`ModelTrustStore` (`magnetar-runtime/src/model.rs`) are digest-pinning and
explicit-local-development-policy only today. Publisher and source identity
are recorded as metadata but the spec is explicit that they SHALL NOT imply
trust on their own (`Requirement: Publisher Metadata Is Not Sufficient
Trust`, `Requirement: Source Metadata Is Not Sufficient Trust`, both
canonical, component spec). The component spec already has a placeholder
requirement, "Signature Metadata Is Optional and Non-Authoritative", stating
an *unsupported or unverified* signature SHALL NOT grant trust -- this
change makes signatures verifiable, so that requirement's scope narrows from
"never authoritative" to "authoritative only when it actually verifies
against an operator-trusted key."

The design questions (what gets signed, algorithm, key identification,
trust/revocation, verification precedence) were already answered in
[`docs/cryptographic-artifact-signatures.md`](../../../docs/cryptographic-artifact-signatures.md)
before this Change existed. This document is the concrete implementation
design that follows that document's own "Definition of done" section; it
does not re-litigate the design questions already settled there, only makes
the remaining implementation choice (crate) and states how it lands in this
repository's actual module structure.

## Goals / Non-Goals

**Goals:**
- Ed25519 signature verification as a second, independent path to `Trusted`
  in `ComponentTrustStore::evaluate` and `ModelTrustStore::evaluate`,
  composing with existing digest pinning rather than replacing it.
- Authenticated `PublisherIdentity`, bound by the operator to a trusted
  key, distinct from and never derived from an artifact's self-declared
  publisher metadata field.
- Key revocation, symmetric with existing digest revocation.
- Every existing digest-pinning-only trust decision is unchanged after this
  change ships (no regression, no new required fields).

**Non-Goals** (carried over unchanged from
`docs/cryptographic-artifact-signatures.md`'s own Non-Goals section):
- Key distribution infrastructure (registries, TUF, well-known-keys
  discovery) -- operator-configured keys only.
- Online revocation checking -- local, operator-updated revocation lists
  only.
- Provider signing -- Providers are trusted native code by architectural
  definition (`SECURITY.md`'s Scope section), out of scope.
- Per-tensor or per-shard signatures -- whole-artifact granularity only.
- Changing MAG-06's release-publication pipeline to sign anything by
  default -- this change makes verification possible, it does not obligate
  any publisher (including Magnetar's own release pipeline) to start
  signing.

## Decisions

### Signing library: `ed25519-dalek` v3

Verified directly in this Change (not deferred further): `ed25519-dalek`
3.0.0 builds cleanly for both this crate's supported targets --
`x86_64-unknown-linux-gnu` (native, default features) and
`wasm32-unknown-unknown` (`--no-default-features`, matching how
`magnetar-runtime` itself is checked for wasm32 today, since `wasmtime` and
friends are excluded there via `cfg(not(target_arch = "wasm32"))`). Its
`Cargo.toml` categorizes it `no-std`, confirming it has no ambient OS/network
authority -- consistent with Runtime staying transport-neutral. Default
features (`fast`, `zeroize`) are used: `fast` is a pure performance choice
(precomputed tables), `zeroize` zeroes key material on drop, both desirable
for a Runtime that will hold trusted public keys (and, in tests only, private
signing keys) in memory.

Alternatives considered: rolling a hand-written Ed25519 implementation was
rejected outright -- cryptographic primitives are exactly the code a Runtime
should never reimplement. `ring` was considered and rejected: it vendors C
code (BoringSSL-derived assembly) which complicates the wasm32 target and
the crate's existing "no C dependencies for portable code paths" posture;
`ed25519-dalek` is pure Rust, matching every other dependency
`magnetar-runtime` already has in its base `[dependencies]` (`sha2`, `serde`,
`serde_json`, `serde_norway`). Enabled features are `fast`, `zeroize`
(defaults) and `signature` (for the `Signer`/`Verifier` traits used by
test-fixture signing); production verification code uses the inherent,
stricter `VerifyingKey::verify_strict` rather than the `Verifier` trait's
`verify`, so the `signature` feature is only load-bearing for tests.

### Where verification-only code lives vs. test-only signing code

Runtime production code only ever *verifies* signatures (`VerifyingKey`,
`Signature`, `.verify()`/`.verify_strict()`) -- it never generates them; an
artifact's publisher signs offline, outside Magnetar entirely. Test code
needs to construct real `SigningKey`s and sign fixture digests to exercise
the verification path end to end. No new dev-dependency is needed for this:
`ed25519-dalek`'s `SigningKey::from_bytes` accepts a caller-supplied 32-byte
seed directly (no CSPRNG dependency required for deterministic, reproducible
test fixtures), used only under `#[cfg(test)]`.

### `SignatureRecord` placement

Added as an optional field on `ComponentManifest`/`ComponentArtifactPackage`
and `ModelManifest` (mirroring how `ComponentPublisher`/`provenance` are
already optional, additive fields on those same types), not as a separate
side-channel file. This matches `docs/cryptographic-artifact-signatures.md`'s
`SignatureRecord { key_id, signature, signed_digest }` shape exactly:
`signature` stored as `[u8; 64]` (Ed25519's fixed signature size, matching
`ed25519_dalek::Signature`'s own size), `signed_digest` as the full
`"sha256:..."`-prefixed string so a verifier never has to guess digest
representation.

### Key identification and trust store shape

`key_id` = first 16 bytes (32 hex chars) of `SHA-256(public_key_bytes)`,
computed with the crate's existing `sha2` dependency (no new hashing
dependency). `trusted_publisher_keys: BTreeMap<String, (VerifyingKeyBytes,
PublisherIdentity)>` and `revoked_keys: BTreeSet<String>` are added to both
`ComponentTrustStore` and `ModelTrustStore`, loadable from the existing YAML
trust-store format (`TrustStoreYaml`/model equivalent) as new optional
top-level keys, keeping `schema_version: 1` -- these are additive fields, a
trust store file with none of them present parses and behaves exactly as it
does today.

### Verification precedence

Exactly as specified in `docs/cryptographic-artifact-signatures.md`'s
"Verification policy" section, inserted into each store's existing
`evaluate()` precedence chain:

1. Revoked key -> `Revoked` (checked before signature verification, mirrors
   `revoked_digests`'s existing first-checked position).
2. No signature present -> unchanged: falls through to existing digest /
   development-mode / publisher-metadata handling.
3. Signature present, `key_id` not in `trusted_publisher_keys` -> falls
   through to digest pinning, exactly as if unsigned (not an error).
4. Signature present, key known, but `signed_digest` doesn't match the
   artifact's actual computed digest, or the Ed25519 verification fails ->
   `Rejected` (a broken signature under a claimed key id is a stronger
   negative signal than absence).
5. Signature present, key known, signature verifies -> `Trusted`, with the
   store-bound `PublisherIdentity` now the authenticated identity for that
   decision.

For Components this slots in after `quarantined_digests`/`rejected_digests`
(themselves already before `trusted_digests`) and before the existing
unauthenticated-metadata fallback; for Models, after `rejected_digests` and
before `trusted_digests` so a valid signature can grant `Trusted` on its own
even when the digest itself was never separately pinned.

## Risks / Trade-offs

- **[Risk]** A new cryptography dependency increases the audit surface of a
  crate that is otherwise dependency-light and already crates.io-published.
  → **Mitigation**: `ed25519-dalek` is the de facto standard Rust Ed25519
  implementation (used by sigstore/cosign-adjacent tooling, widely audited),
  pure Rust with no C/FFI, and MAG-06's existing `cargo deny`/advisory
  security gate (`tools/release-publish`) will catch any future advisory
  against it the same way it does for every other dependency.
- **[Risk]** Adding a second trust path (signatures) alongside digest
  pinning could be misread as loosening trust. → **Mitigation**: every new
  path in the precedence order above either requires an operator-configured
  trusted key (no different in kind from operator-configured trusted
  digests today) or fails closed harder than today (`Rejected` for a broken
  signature, not silently ignored); no existing digest-pinning-only
  decision changes.
- **[Risk]** `wasm32-unknown-unknown` compatibility could regress on a
  future `ed25519-dalek` upgrade (e.g. a default-feature change pulling in
  `std`). → **Mitigation**: the workspace's own `quality.yml` CI already
  builds `magnetar-runtime` for `wasm32-unknown-unknown`; this is a normal
  CI-caught regression, not a silent one.

## Migration Plan

Purely additive: existing trust store YAML files and existing
`ComponentManifest`/`ModelManifest` instances (real or fixture) need no
changes and continue to evaluate identically. No data migration, no
rollback concern beyond a normal revert.

## Open Questions

None -- the design questions this Change depended on were resolved in
`docs/cryptographic-artifact-signatures.md` before this Change was created;
this document's own Decisions section resolves the one question that
document explicitly deferred (exact crate choice).

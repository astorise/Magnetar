## 1. Dependency and shared types

- [ ] 1.1 Add `ed25519-dalek` (default features: `fast`, `zeroize`) to `magnetar-runtime/Cargo.toml`, verified to build for both `x86_64-unknown-linux-gnu` (native) and `wasm32-unknown-unknown --no-default-features` (matching how `magnetar-runtime` is already checked for wasm32).
- [ ] 1.2 Add a `SignatureRecord { key_id: String, signature: [u8; 64], signed_digest: String }` type, usable from both `component.rs` and `model.rs` (shared helper module or duplicated small type, whichever avoids introducing a cross-module coupling neither file has today).
- [ ] 1.3 Add a `key_id_for(public_key_bytes: &[u8; 32]) -> String` helper: first 16 bytes (32 hex chars) of `SHA-256(public_key_bytes)`, using the crate's existing `sha2` dependency.
- [ ] 1.4 Add a `PublisherIdentity` type (operator-configured identity bound to a trusted key; at minimum a display name, shape driven by what `docs/cryptographic-artifact-signatures.md`'s "Authenticated publisher identity" section needs).

## 2. Component Artifact trust store

- [ ] 2.1 Add `trusted_publisher_keys: BTreeMap<String, (VerifyingKeyBytes, PublisherIdentity)>` and `revoked_keys: BTreeSet<String>` fields to `ComponentTrustStore`, with builder methods mirroring the existing `trust_digest`/`revoke_digest` style.
- [ ] 2.2 Add an optional `signature: Option<SignatureRecord>` field to `ComponentManifest`/`ComponentArtifactPackage` (wherever the manifest is actually carried through the loading path) and its YAML loading (`TrustStoreYaml`/manifest YAML equivalents), matching the additive, backward-compatible posture the design requires.
- [ ] 2.3 Extend `ComponentTrustStore::evaluate` with the signature-verification precedence from design.md: revoked key -> `Revoked` (before verification); no signature -> unchanged; unknown key -> falls through unchanged; known key + broken signature or `signed_digest` mismatch -> `Rejected`; known key + valid signature -> `Trusted`.
- [ ] 2.4 Extend `ComponentTrustDecision` (or add an accessor) so a caller can read the authenticated `PublisherIdentity` when the `Trusted` decision came through the signature path, distinct from the unauthenticated manifest-declared publisher field.
- [ ] 2.5 Update `TrustStoreYaml` (and its `validate`) to parse `trusted_publisher_keys`/`revoked_keys` from the existing `schema: magnetar-component-trust`, `schema_version: 1` YAML format as new optional top-level keys.

## 3. Model Artifact trust store

- [ ] 3.1 Add `trusted_publisher_keys` and `revoked_keys` fields to `ModelTrustStore`, mirroring 2.1.
- [ ] 3.2 Add an optional `signature: Option<SignatureRecord>` field to `ModelManifest`'s loading path, mirroring 2.2.
- [ ] 3.3 Extend `ModelTrustStore::evaluate` with the same signature-verification precedence as 2.3, inserted after `rejected_digests` and before `trusted_digests` per design.md.
- [ ] 3.4 Extend `ModelTrustDecision` with an authenticated-publisher-identity accessor, mirroring 2.4 (respecting `ModelTrustDecision`'s existing `pub(crate)`-constructed, read-only-accessor pattern -- see the struct's own doc comment on why fields are not `pub`).

## 4. Tests

- [ ] 4.1 Component: valid signature from a trusted key grants `Trusted` with the correct authenticated `PublisherIdentity`, even when the digest is not separately pinned.
- [ ] 4.2 Component: unknown signing key falls through to digest pinning (not rejected, not an error).
- [ ] 4.3 Component: broken signature under a known key is `Rejected`, not `Unknown` -- both the "signature bytes don't verify" and the "`signed_digest` doesn't match the actual digest" cases.
- [ ] 4.4 Component: revoked key is `Revoked` even when the covered digest is separately pinned trusted, and revocation is checked before cryptographic verification runs.
- [ ] 4.5 Component: every existing digest-pinning-only test continues to pass unchanged (no regression); absence of a signature behaves identically to today.
- [ ] 4.6 Model: repeat 4.1-4.5 for `ModelTrustStore`/`ModelManifest`.
- [ ] 4.7 Round-trip test: a trust store YAML file containing `trusted_publisher_keys`/`revoked_keys` parses correctly via `ComponentTrustStore::load_yaml`.

## 5. Documentation

- [ ] 5.1 Update `SECURITY.md`'s "Known gaps" entry to describe signature verification as implemented (algorithm, what's verified, what's still operator-configured-only), replacing the "design only" framing, and keep the still-true parts (no key distribution infra, no online revocation checking).
- [ ] 5.2 Add a short "Implemented" note at the top of `docs/cryptographic-artifact-signatures.md` pointing at this Change and at the concrete types/fields that now realize the design, without rewriting the design content itself (it remains the durable design reference).
- [ ] 5.3 Cross-reference from `docs/release-security.md`'s "Publication Automation" section (added by MAG-06) that cryptographic signing is now available for artifacts that choose to use it, while the release pipeline itself still uses digest pinning as its trust mechanism by default.

## MODIFIED Requirements

### Requirement: File-Based Trust Store

The initial Runtime trust store SHALL support YAML with
`schema: magnetar-component-trust` and `schema_version: 1`.

The trust store SHALL support `trusted_digests`, `rejected_digests`,
`revoked_digests`, `quarantined_digests`, `trusted_publishers`,
`trusted_sources`, `development.allow_unsigned_local`,
`trusted_publisher_keys`, and `revoked_keys`.

`trusted_publisher_keys` SHALL map a key id to a public key and a bound publisher identity; `revoked_keys` SHALL list revoked key ids.

The trust store SHALL remain separate from the Component Artifact manifest.

#### Scenario: Trust store with signature fields parses

Given a trust store YAML file declares `trusted_publisher_keys` with one key id mapped to a public key and publisher identity, and `revoked_keys` with one entry

When the trust store is loaded

Then loading succeeds and both fields are available to trust evaluation.

---

## ADDED Requirements

### Requirement: Verified Signature Grants Trust

A Component Artifact whose signature verifies against a public key present in the trust store's `trusted_publisher_keys` SHALL be marked trusted, independent of whether its digest is separately pinned in `trusted_digests`.

A signature SHALL be considered verified only when its `key_id` resolves to a known trusted key, its `signed_digest` matches the artifact's actual computed `ComponentDigest`, and the Ed25519 signature itself verifies against that key's public key bytes.

A Component Artifact digest present in `trusted_digests` SHALL continue to be marked trusted exactly as before this requirement existed, whether or not it also carries a signature.

#### Scenario: Valid signature grants trust without digest pinning

Given a Component Artifact carries a signature whose key id is present in `trusted_publisher_keys`

And the signature verifies against the artifact's actual computed digest

And that digest is not present in `trusted_digests`

When trust is evaluated

Then the artifact is marked trusted through the signature path.

#### Scenario: Digest pinning continues to work unchanged

Given a Component Artifact digest is present in `trusted_digests`

And the artifact carries no signature

When trust is evaluated

Then the artifact is marked trusted exactly as it was before signature verification existed.

---

### Requirement: Unknown Signing Key Falls Through To Digest Pinning

A Component Artifact whose signature's `key_id` is not present in `trusted_publisher_keys` SHALL NOT be rejected on that basis alone; trust evaluation SHALL fall through to digest pinning and other existing trust paths exactly as if no signature were present.

#### Scenario: Signed by an untrusted key

Given a Component Artifact carries a signature whose key id is not present in `trusted_publisher_keys`

And the artifact's digest is not present in `trusted_digests`, `rejected_digests`, or `revoked_digests`

When trust is evaluated

Then the artifact is marked unknown, not rejected, the same result as if it carried no signature at all.

---

### Requirement: Broken Signature Under A Known Key Is Rejected

A Component Artifact whose signature's `key_id` resolves to a known trusted key, but whose signature does not verify against that key or whose `signed_digest` does not match the artifact's actual computed digest, SHALL be marked rejected, not merely unknown.

#### Scenario: Signature does not verify under a known key

Given a Component Artifact carries a signature whose key id is present in `trusted_publisher_keys`

And the Ed25519 signature bytes do not verify against that key's public key for the claimed `signed_digest`

When trust is evaluated

Then the artifact is marked rejected.

#### Scenario: Signed digest does not match the artifact's actual digest

Given a Component Artifact carries a signature whose key id is present in `trusted_publisher_keys`

And the signature's `signed_digest` field does not match the artifact's actual computed digest

When trust is evaluated

Then the artifact is marked rejected, even if the signature bytes would otherwise verify against the claimed digest.

---

### Requirement: Authenticated Publisher Identity Via Signature

A publisher identity SHALL be treated as authenticated only when it is the identity the trust store operator bound to the specific key that produced a verifying signature on that artifact; a publisher identity string declared by the artifact's own manifest SHALL NOT be treated as authenticated regardless of whether a signature is also present.

#### Scenario: Verified signature carries the operator-bound identity

Given a Component Artifact carries a signature that verifies against a trusted key

And the trust store binds that key to a specific publisher identity

When a caller reads the authenticated publisher identity for that trust decision

Then the caller receives the operator-bound identity, not any publisher string the artifact's own manifest declares.

#### Scenario: Manifest publisher field remains unauthenticated on its own

Given a Component Artifact's manifest declares a publisher field

And the artifact carries no signature, or its signature does not verify

When a caller reads the authenticated publisher identity for that trust decision

Then no authenticated publisher identity is available, consistent with `Requirement: Publisher Metadata Is Not Sufficient Trust`.

---

### Requirement: Signature Key Revocation

The Runtime SHALL support revoking a trusted publisher key by key id via `revoked_keys`.

A signature produced under a revoked key SHALL be rejected as revoked, checked before signature verification runs, regardless of whether the covered digest is independently trusted through `trusted_digests`.

Revoking a key SHALL NOT retroactively distrust an artifact that is also trusted independently through digest pinning.

#### Scenario: Revoked key rejects an otherwise-valid signature

Given a Component Artifact carries a signature that would verify against a key present in both `trusted_publisher_keys` and `revoked_keys`

When trust is evaluated

Then the artifact's trust decision is revoked via the signature path, checked before the signature's cryptographic validity is evaluated.

#### Scenario: Revoking a key does not distrust a digest-pinned artifact

Given a Component Artifact's digest is present in `trusted_digests`

And the artifact also carries a signature under a key that is later added to `revoked_keys`

When trust is evaluated after the key revocation

Then the artifact remains trusted through digest pinning, independent of the revoked signature.

//! Ed25519 signature verification shared by Component and Model Artifact
//! trust (`component::ComponentTrustStore`, `model::ModelTrustStore`).
//!
//! Design: `docs/cryptographic-artifact-signatures.md` (tracks issue #37).
//! A signature covers the artifact's existing digest string (e.g.
//! `"sha256:<hex>"`), never the raw artifact bytes -- reusing
//! `ComponentDigest`/`ModelDigest` as the signed identity instead of
//! introducing a second notion of "what this artifact is".

use ed25519_dalek::{Signature, VerifyingKey};
use sha2::{Digest, Sha256};

/// A raw Ed25519 public key, as stored in a trust store's
/// `trusted_publisher_keys`.
pub type VerifyingKeyBytes = [u8; 32];

/// A publisher identity the trust store operator has bound to a specific
/// trusted key. Never derived from an artifact's own self-declared
/// metadata -- only a verified signature under this key authenticates it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PublisherIdentity {
    pub name: String,
}

impl PublisherIdentity {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

/// A signature attached to a Component or Model Artifact, as carried by
/// its manifest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SignatureRecord {
    /// Hex-encoded fingerprint identifying the signing key: see
    /// [`key_id_for`]. A lookup hint, not itself trust-bearing.
    pub key_id: String,
    pub signature: [u8; 64],
    /// The exact digest string (e.g. `"sha256:<hex>"`) that was signed,
    /// carried alongside rather than assumed so a verifier never has to
    /// guess which digest representation produced this signature.
    pub signed_digest: String,
}

impl SignatureRecord {
    pub fn new(
        key_id: impl Into<String>,
        signature: [u8; 64],
        signed_digest: impl Into<String>,
    ) -> Self {
        Self {
            key_id: key_id.into(),
            signature,
            signed_digest: signed_digest.into(),
        }
    }
}

/// First 16 bytes (32 hex chars) of `SHA-256(public_key_bytes)` -- the same
/// truncated-fingerprint convention SSH and PGP both use. Two different
/// public keys colliding on this fingerprint is astronomically unlikely
/// but not impossible by construction, so verification always checks the
/// full signature against the full public key on file for this id, never
/// against the id alone.
pub fn key_id_for(public_key_bytes: &[u8; 32]) -> String {
    let hash = Sha256::digest(public_key_bytes);
    hash[..16]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Decodes a lowercase or uppercase hex string into exactly `N` bytes.
/// Used to parse YAML-carried public keys (32 bytes) and signatures (64
/// bytes) -- neither has a natural YAML scalar representation, so both are
/// carried as hex strings in trust store and manifest files.
pub fn decode_hex<const N: usize>(hex: &str) -> Option<[u8; N]> {
    if hex.len() != N * 2 {
        return None;
    }
    let mut bytes = [0u8; N];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).ok()?;
    }
    Some(bytes)
}

/// The outcome of checking a [`SignatureRecord`] against a specific
/// trusted key's public key bytes and the artifact's actual computed
/// digest.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SignatureVerification {
    /// The signature verifies against the key for the claimed digest, and
    /// the claimed digest matches the artifact's actual computed digest.
    Verified,
    /// `signed_digest` does not match the artifact's actual computed
    /// digest -- the signature may be valid for a different artifact
    /// entirely, so it is never checked cryptographically in this case.
    DigestMismatch,
    /// The key bytes are malformed, or the Ed25519 signature does not
    /// verify against the key for the claimed digest.
    Invalid,
}

/// Verifies `record` against `public_key_bytes`, given the artifact's own
/// actual computed digest string. Never trusts `record.signed_digest` on
/// its own -- it must match `actual_digest` before cryptographic
/// verification is even attempted, per
/// `Requirement: Source-Declared Digest Is A Claim`'s "declared is a claim,
/// computed is truth" principle.
///
/// Uses `verify_strict` (rather than the `Verifier` trait's `verify`) for
/// the stronger of ed25519-dalek's two verification modes, which also
/// rejects malleable/non-canonical signature encodings.
pub fn verify_signature(
    record: &SignatureRecord,
    public_key_bytes: &[u8; 32],
    actual_digest: &str,
) -> SignatureVerification {
    if record.signed_digest != actual_digest {
        return SignatureVerification::DigestMismatch;
    }
    let Ok(verifying_key) = VerifyingKey::from_bytes(public_key_bytes) else {
        return SignatureVerification::Invalid;
    };
    let signature = Signature::from_bytes(&record.signature);
    match verifying_key.verify_strict(record.signed_digest.as_bytes(), &signature) {
        Ok(()) => SignatureVerification::Verified,
        Err(_) => SignatureVerification::Invalid,
    }
}

#[cfg(test)]
mod tests;

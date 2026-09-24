//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use ed25519_dalek::{Signer, SigningKey};

fn keypair(seed_byte: u8) -> (SigningKey, [u8; 32]) {
    let signing_key = SigningKey::from_bytes(&[seed_byte; 32]);
    let public_key_bytes = signing_key.verifying_key().to_bytes();
    (signing_key, public_key_bytes)
}

fn sign(signing_key: &SigningKey, digest: &str) -> [u8; 64] {
    signing_key.sign(digest.as_bytes()).to_bytes()
}

#[test]
fn decode_hex_round_trips_a_public_key() {
    let (_, public_key) = keypair(3);
    let hex: String = public_key
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(decode_hex::<32>(&hex), Some(public_key));
}

#[test]
fn decode_hex_rejects_wrong_length_and_non_hex() {
    assert_eq!(decode_hex::<32>("ab"), None);
    assert_eq!(decode_hex::<32>(&"zz".repeat(32)), None);
}

#[test]
fn key_id_is_stable_and_public_key_sensitive() {
    let (_, key_a) = keypair(1);
    let (_, key_b) = keypair(2);
    assert_eq!(key_id_for(&key_a), key_id_for(&key_a));
    assert_ne!(key_id_for(&key_a), key_id_for(&key_b));
    assert_eq!(key_id_for(&key_a).len(), 32);
}

#[test]
fn valid_signature_over_the_matching_digest_verifies() {
    let (signing_key, public_key) = keypair(7);
    let digest = "sha256:abcd";
    let record = SignatureRecord::new(key_id_for(&public_key), sign(&signing_key, digest), digest);
    assert_eq!(
        verify_signature(&record, &public_key, digest),
        SignatureVerification::Verified
    );
}

#[test]
fn signed_digest_mismatch_is_detected_before_cryptographic_check() {
    let (signing_key, public_key) = keypair(7);
    let record = SignatureRecord::new(
        key_id_for(&public_key),
        sign(&signing_key, "sha256:abcd"),
        "sha256:abcd",
    );
    assert_eq!(
        verify_signature(&record, &public_key, "sha256:different"),
        SignatureVerification::DigestMismatch
    );
}

#[test]
fn signature_from_a_different_key_is_invalid() {
    let (signing_key, _) = keypair(7);
    let (_, other_public_key) = keypair(9);
    let digest = "sha256:abcd";
    let record = SignatureRecord::new(
        key_id_for(&other_public_key),
        sign(&signing_key, digest),
        digest,
    );
    assert_eq!(
        verify_signature(&record, &other_public_key, digest),
        SignatureVerification::Invalid
    );
}

#[test]
fn tampered_signature_bytes_are_invalid() {
    let (signing_key, public_key) = keypair(7);
    let digest = "sha256:abcd";
    let mut signature = sign(&signing_key, digest);
    signature[0] ^= 0xFF;
    let record = SignatureRecord::new(key_id_for(&public_key), signature, digest);
    assert_eq!(
        verify_signature(&record, &public_key, digest),
        SignatureVerification::Invalid
    );
}

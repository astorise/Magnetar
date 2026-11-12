//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use ed25519_dalek::Signer;

fn model_manifest_yaml(digest: &str, signatures_block: &str) -> String {
    format!(
        r#"
schema: magnetar-model-artifact
schema_version: 1
kind: model-bundle
digest: {digest}
model:
  name: magnetar.examples.model
  revision: r1
architecture:
  family: generic
  identifier: generic-v1
artifacts:
  weights:
    kind: model-weights
    digest: {digest}
    size_bytes: 128
{signatures_block}"#
    )
}

fn signatures_yaml(key_id: &str, signed_digest: &str, signature_hex: &str) -> String {
    format!(
        "signatures:\n  - kind: \"ed25519\"\n    key_id: \"{key_id}\"\n    digest: \"{signed_digest}\"\n    signature: \"{signature_hex}\"\n"
    )
}

fn signature_test_keypair(seed: u8) -> (ed25519_dalek::SigningKey, VerifyingKeyBytes) {
    let signing_key = ed25519_dalek::SigningKey::from_bytes(&[seed; 32]);
    let public_key = signing_key.verifying_key().to_bytes();
    (signing_key, public_key)
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn parse_manifest(yaml: &str) -> ModelManifest {
    ModelManifest::from_yaml_str(yaml).expect("manifest parses")
}

fn model_digest() -> String {
    "sha256:0000000000000000000000000000000000000000000000000000000000000001".into()
}

// MAG-02: Ed25519 Model Artifact signature verification
// (docs/cryptographic-artifact-signatures.md). `ModelTrustStore::evaluate`
// is a pure function of `&ModelManifest`, so these mirror
// `component::tests`'s equivalent signature tests exactly.

#[test]
fn model_signature_from_trusted_key_grants_trust_without_digest_pinning() {
    let digest = model_digest();
    let (signing_key, public_key) = signature_test_keypair(11);
    let key_id = key_id_for(&public_key);
    let signature = signing_key.sign(digest.as_bytes()).to_bytes();
    let manifest = parse_manifest(&model_manifest_yaml(
        &digest,
        &signatures_yaml(&key_id, &digest, &hex_encode(&signature)),
    ));

    let trust_store = ModelTrustStore::default()
        .trust_publisher_key(public_key, PublisherIdentity::new("Acme Publishing"));
    let decision = trust_store.evaluate(&manifest);

    assert_eq!(decision.status(), ModelTrustStatus::Trusted);
    assert_eq!(
        decision.authenticated_publisher(),
        Some(&PublisherIdentity::new("Acme Publishing"))
    );
}

#[test]
fn model_digest_pinning_continues_to_work_unchanged_without_a_signature() {
    let digest = model_digest();
    let manifest = parse_manifest(&model_manifest_yaml(&digest, "signatures: []\n"));

    let trust_store = ModelTrustStore::default().trust_digest(digest);
    let decision = trust_store.evaluate(&manifest);

    assert_eq!(decision.status(), ModelTrustStatus::Trusted);
    assert_eq!(decision.reason(), "digest trusted by policy");
    assert_eq!(decision.authenticated_publisher(), None);
}

#[test]
fn model_signature_from_an_untrusted_key_falls_through_to_unknown() {
    let digest = model_digest();
    let (signing_key, public_key) = signature_test_keypair(12);
    let key_id = key_id_for(&public_key);
    let signature = signing_key.sign(digest.as_bytes()).to_bytes();
    let manifest = parse_manifest(&model_manifest_yaml(
        &digest,
        &signatures_yaml(&key_id, &digest, &hex_encode(&signature)),
    ));

    let trust_store = ModelTrustStore::default();
    let decision = trust_store.evaluate(&manifest);

    assert_eq!(decision.status(), ModelTrustStatus::Unknown);
}

#[test]
fn model_signature_with_wrong_bytes_under_a_known_key_is_rejected() {
    let digest = model_digest();
    let (signing_key, public_key) = signature_test_keypair(13);
    let key_id = key_id_for(&public_key);
    let mut signature = signing_key.sign(digest.as_bytes()).to_bytes();
    signature[0] ^= 0xFF;
    let manifest = parse_manifest(&model_manifest_yaml(
        &digest,
        &signatures_yaml(&key_id, &digest, &hex_encode(&signature)),
    ));

    let trust_store = ModelTrustStore::default()
        .trust_publisher_key(public_key, PublisherIdentity::new("Acme Publishing"));
    let decision = trust_store.evaluate(&manifest);

    assert_eq!(decision.status(), ModelTrustStatus::Rejected);
}

#[test]
fn model_signature_with_mismatched_signed_digest_under_a_known_key_is_rejected() {
    let digest = model_digest();
    let (signing_key, public_key) = signature_test_keypair(14);
    let key_id = key_id_for(&public_key);
    let signature = signing_key.sign(digest.as_bytes()).to_bytes();
    let manifest = parse_manifest(&model_manifest_yaml(
        &digest,
        &signatures_yaml(
            &key_id,
            "sha256:0000000000000000000000000000000000000000000000000000000000000002",
            &hex_encode(&signature),
        ),
    ));

    let trust_store = ModelTrustStore::default()
        .trust_publisher_key(public_key, PublisherIdentity::new("Acme Publishing"));
    let decision = trust_store.evaluate(&manifest);

    assert_eq!(decision.status(), ModelTrustStatus::Rejected);
}

#[test]
fn model_revoked_signature_key_is_revoked_even_though_otherwise_valid() {
    let digest = model_digest();
    let (signing_key, public_key) = signature_test_keypair(15);
    let key_id = key_id_for(&public_key);
    let signature = signing_key.sign(digest.as_bytes()).to_bytes();
    let manifest = parse_manifest(&model_manifest_yaml(
        &digest,
        &signatures_yaml(&key_id, &digest, &hex_encode(&signature)),
    ));

    let trust_store = ModelTrustStore::default()
        .trust_publisher_key(public_key, PublisherIdentity::new("Acme Publishing"))
        .revoke_key(key_id);
    let decision = trust_store.evaluate(&manifest);

    assert_eq!(decision.status(), ModelTrustStatus::Revoked);
}

#[test]
fn model_revoking_a_key_does_not_distrust_a_digest_pinned_artifact() {
    let digest = model_digest();
    let (signing_key, public_key) = signature_test_keypair(16);
    let key_id = key_id_for(&public_key);
    let signature = signing_key.sign(digest.as_bytes()).to_bytes();
    let manifest = parse_manifest(&model_manifest_yaml(
        &digest,
        &signatures_yaml(&key_id, &digest, &hex_encode(&signature)),
    ));

    let trust_store = ModelTrustStore::default()
        .trust_digest(digest.clone())
        .trust_publisher_key(public_key, PublisherIdentity::new("Acme Publishing"))
        .revoke_key(key_id);
    let decision = trust_store.evaluate(&manifest);

    assert_eq!(decision.status(), ModelTrustStatus::Trusted);
    assert_eq!(decision.reason(), "digest trusted by policy");
}

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

/// `express-latent-attention-forms`: DeepSeek-V2-Lite's shape validates;
/// each inconsistent 1.7.0 declaration is refused.
#[test]
fn latent_attention_and_expert_declarations_validate_against_the_architecture() {
    let moe = MixtureOfExpertsConfig {
        experts: 64,
        experts_per_token: 6,
        expert_intermediate_size: 1408,
        shared_expert_intermediate_size: Some(2816),
        normalize_top_k: false,
    };
    let valid = ModelArchitectureConfig {
        hidden_size: 2048,
        intermediate_size: 10944,
        num_hidden_layers: 27,
        num_attention_heads: 16,
        num_key_value_heads: 16,
        head_dim: 192,
        vocab_size: 102400,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        mixture_of_experts: Some(moe),
        latent_attention: Some(LatentAttentionConfig {
            q_lora_rank: None,
            kv_lora_rank: 512,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
        }),
        moe_layers: Some((0..27).map(|layer| layer >= 1).collect()),
        expert_routing: Some(ExpertRoutingConfig {
            scoring: ExpertScoring::Softmax,
            groups: Some(ExpertGroups {
                count: 8,
                per_token: 3,
                score: ExpertGroupScore::Max,
            }),
            routed_scaling_factor: 1.0,
            selection_bias: false,
            shared_expert_gated: false,
        }),
        ..Default::default()
    };
    valid.validate().unwrap();
    let refused = |change: fn(&mut ModelArchitectureConfig)| {
        let mut config = valid.clone();
        change(&mut config);
        assert!(config.validate().is_err(), "{config:?}");
    };
    refused(|config| config.hidden_activation = Some("unknown-activation".into()));
    // A query/key head that is not nope + rope wide.
    refused(|config| config.head_dim = 128);
    // Fewer K/V heads than attention heads.
    refused(|config| config.num_key_value_heads = 1);
    // A zero latent width.
    refused(|config| {
        config.latent_attention.as_mut().unwrap().kv_lora_rank = 0;
    });
    // Expert layers for another layer count, or without experts.
    refused(|config| config.moe_layers = Some(vec![true; 26]));
    refused(|config| {
        config.mixture_of_experts = None;
        config.expert_routing = None;
    });
    // Groups that do not divide the experts, or offer too few of them.
    refused(|config| {
        config
            .expert_routing
            .as_mut()
            .unwrap()
            .groups
            .as_mut()
            .unwrap()
            .count = 7;
    });
    refused(|config| {
        config
            .expert_routing
            .as_mut()
            .unwrap()
            .groups
            .as_mut()
            .unwrap()
            .per_token = 0;
    });
    refused(|config| {
        let groups = config
            .expert_routing
            .as_mut()
            .unwrap()
            .groups
            .as_mut()
            .unwrap();
        groups.count = 32;
        groups.per_token = 2;
    });
    refused(|config| {
        config
            .expert_routing
            .as_mut()
            .unwrap()
            .routed_scaling_factor = 0.0
    });
}

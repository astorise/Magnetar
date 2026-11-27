//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use ed25519_dalek::Signer;

#[test]
fn e4m3fn_storage_is_distinct_from_other_one_byte_formats() {
    let fp8 = ModelDType::parse("F8_E4M3").unwrap();
    assert_eq!(fp8, ModelDType::F8E4M3Fn);
    assert_eq!(fp8.descriptor().size_bytes(), 1);
    assert_ne!(fp8.descriptor(), ModelDType::U8.descriptor());
    assert_ne!(fp8.descriptor(), ModelDType::I8.descriptor());
    assert_ne!(fp8.descriptor(), ModelDType::Q8.descriptor());
    assert_eq!(ModelDType::parse("F8_E5M2"), None);
    assert_eq!(ModelDType::parse("F8_E4M3FNUZ"), None);
}

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
            norm_eps: None,
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
            normalization: None,
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
    refused(|config| config.rope_layers = Some(vec![true]));
    // A query/key head that is not nope + rope wide.
    refused(|config| config.head_dim = 128);
    // Fewer K/V heads than attention heads.
    refused(|config| config.num_key_value_heads = 1);
    // A zero latent width.
    refused(|config| {
        config.latent_attention.as_mut().unwrap().kv_lora_rank = 0;
    });
    refused(|config| {
        config.latent_attention.as_mut().unwrap().norm_eps = Some(f64::NAN);
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
    refused(|config| {
        config.expert_routing.as_mut().unwrap().normalization = Some(ExpertNormalization::Always);
    });
    refused(|config| {
        let groups = config
            .expert_routing
            .as_mut()
            .unwrap()
            .groups
            .as_mut()
            .unwrap();
        groups.count = 64;
        groups.per_token = 6;
        groups.score = ExpertGroupScore::TopTwoSum;
    });
}

/// `support-qwen38-flash-next`: the `1.14.0` forms validate against the
/// architecture, and each inconsistent declaration is refused.
#[test]
fn decoder_extensions_validate_against_the_architecture() {
    let valid = ModelArchitectureConfig {
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 16,
        vocab_size: 100,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        decoder_extensions: DecoderExtensions {
            hyper_connections: Some(HyperConnectionConfig {
                streams: 4,
                lowrank: 8,
            }),
            ngram_embeddings: Some(NgramEmbeddingConfig {
                layers: vec![false, true, false, false],
                embedding_dim: 32,
                conv_kernel: 4,
                ngram_size: 3,
                heads_per_ngram: 8,
                vocabulary_base: 1000,
                vocabulary_multiple: 128,
                seed: 1234,
                eos_token: 2,
            }),
            token_selection: Some(TokenSelectionConfig {
                heads: 2,
                key_heads: 1,
                head_dim: 16,
                budget: 32,
                compress_ratio: 4,
            }),
            sigmoid_linear_gate: true,
            ..Default::default()
        },
        ..Default::default()
    };
    valid.validate().unwrap();
    assert!(valid.decoder_extensions.is_declared());
    assert!(!DecoderExtensions::default().is_declared());
    let refused = |change: fn(&mut DecoderExtensions)| {
        let mut config = valid.clone();
        change(&mut config.decoder_extensions);
        assert!(
            config.validate().is_err(),
            "{:?}",
            config.decoder_extensions
        );
    };
    // A single stream is no hyper-connection, and the rank is positive.
    refused(|extensions| extensions.hyper_connections.as_mut().unwrap().streams = 1);
    refused(|extensions| extensions.hyper_connections.as_mut().unwrap().lowrank = 0);
    // One entry per layer; the embedding splits evenly over its heads.
    refused(|extensions| {
        extensions
            .ngram_embeddings
            .as_mut()
            .unwrap()
            .layers
            .pop()
            .map(drop)
            .unwrap()
    });
    refused(|extensions| extensions.ngram_embeddings.as_mut().unwrap().embedding_dim = 30);
    refused(|extensions| extensions.ngram_embeddings.as_mut().unwrap().conv_kernel = 0);
    // The indexer has one key head and a budget of whole blocks.
    refused(|extensions| extensions.token_selection.as_mut().unwrap().key_heads = 2);
    refused(|extensions| extensions.token_selection.as_mut().unwrap().budget = 30);
    refused(|extensions| extensions.token_selection.as_mut().unwrap().compress_ratio = 0);
}

#[test]
fn mamba2_hybrid_requires_bounded_geometry_and_explicit_block_semantics() {
    let valid = ModelArchitectureConfig {
        hidden_size: 8,
        intermediate_size: 12,
        num_hidden_layers: 4,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        vocab_size: 24,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        rope_layers: Some(vec![false; 4]),
        hidden_activation: Some("relu_squared".into()),
        mixture_of_experts: Some(MixtureOfExpertsConfig {
            experts: 4,
            experts_per_token: 2,
            expert_intermediate_size: 3,
            shared_expert_intermediate_size: Some(6),
            normalize_top_k: true,
        }),
        moe_layers: Some(vec![false, false, true, false]),
        expert_routing: Some(ExpertRoutingConfig {
            scoring: ExpertScoring::Sigmoid,
            groups: Some(ExpertGroups {
                count: 1,
                per_token: 1,
                score: ExpertGroupScore::TopTwoSum,
            }),
            routed_scaling_factor: 2.5,
            selection_bias: true,
            normalization: Some(ExpertNormalization::Always),
            shared_expert_gated: false,
        }),
        decoder_extensions: DecoderExtensions {
            mamba2_hybrid: Some(Mamba2HybridConfig {
                layers: vec![
                    HybridDecoderBlock::Mamba2,
                    HybridDecoderBlock::Attention,
                    HybridDecoderBlock::RoutedMoe,
                    HybridDecoderBlock::Mamba2,
                ],
                heads: 4,
                head_dim: 2,
                groups: 2,
                state_dim: 3,
                conv_kernel: 3,
                epsilon: 1e-5,
                dt_min: 0.0,
                dt_max: None,
                conv_bias: true,
                linear_bias: false,
            }),
            ..Default::default()
        },
        ..Default::default()
    };
    valid.validate().unwrap();
    assert!(valid.decoder_extensions.is_declared());
    let invalid = |change: fn(&mut ModelArchitectureConfig)| {
        let mut candidate = valid.clone();
        change(&mut candidate);
        assert!(candidate.validate().is_err(), "{candidate:?}");
    };
    invalid(|config| {
        config
            .decoder_extensions
            .mamba2_hybrid
            .as_mut()
            .unwrap()
            .layers
            .pop();
    });
    invalid(|config| {
        config
            .decoder_extensions
            .mamba2_hybrid
            .as_mut()
            .unwrap()
            .groups = 3
    });
    invalid(|config| {
        config
            .decoder_extensions
            .mamba2_hybrid
            .as_mut()
            .unwrap()
            .state_dim = u64::MAX
    });
    invalid(|config| {
        config
            .decoder_extensions
            .mamba2_hybrid
            .as_mut()
            .unwrap()
            .epsilon = f64::NAN
    });
    invalid(|config| {
        config
            .decoder_extensions
            .mamba2_hybrid
            .as_mut()
            .unwrap()
            .dt_min = -1.0
    });
    invalid(|config| {
        config
            .decoder_extensions
            .mamba2_hybrid
            .as_mut()
            .unwrap()
            .dt_max = Some(f64::INFINITY)
    });
    invalid(|config| config.rope_layers = None);
    invalid(|config| config.rope_layers.as_mut().unwrap()[0] = true);
    invalid(|config| config.moe_layers.as_mut().unwrap()[1] = true);
    invalid(|config| config.expert_routing = None);
    invalid(|config| config.hidden_activation = None);
    invalid(|config| config.attention_output_gate = true);
}

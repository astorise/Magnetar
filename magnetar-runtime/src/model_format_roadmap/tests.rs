//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::model::{
    MODEL_ARTIFACT_SCHEMA_VERSION, ModelArchitecture, ModelArtifactId, ModelArtifactKind,
    ModelDType, ModelDigest, ModelLicenseMetadata, ModelManifest, ModelName, ModelRevision,
};
use std::collections::BTreeMap;
use std::collections::BTreeSet;

fn fixture_model_manifest() -> ModelManifest {
    let digest = ModelDigest::parse(format!("sha256:{}", "2".repeat(64))).unwrap();
    let id = ModelArtifactId::new(
        ModelArtifactKind::ModelBundle,
        ModelName::new("fixture-model").unwrap(),
        ModelRevision::new("v1").unwrap(),
        digest,
    );
    ModelManifest {
        schema_version: MODEL_ARTIFACT_SCHEMA_VERSION,
        id,
        architecture: ModelArchitecture::new("qwen", "qwen2"),
        parts: BTreeMap::new(),
        storage_dtype: None,
        compute_dtype: None,
        supported_compute_dtypes: BTreeSet::new(),
        tensors: Vec::new(),
        tokenizer: None,
        tokenizer_config: None,
        chat_template: None,
        prompt_template: None,
        generation: None,
        quantization: None,
        shards: Vec::new(),
        runtime_features: BTreeSet::new(),
        memory_features: BTreeSet::new(),
        provider_capabilities: Vec::new(),
        component: None,
        license: None,
        provenance: None,
        signatures: Vec::new(),
        source: None,
        architecture_config: None,
    }
}

#[test]
fn model_format_roadmap_rejects_empty_provider_name() {
    assert!(matches!(
        reject_model_format_provider_name("   "),
        Err(ModelFormatRoadmapError::InternalModelFormatError { .. })
    ));
}

#[test]
fn model_format_roadmap_normalized_manifest_coverage_tracks_present_fields() {
    let mut manifest = fixture_model_manifest();
    let coverage = NormalizedManifestCoverage::from_manifest(&manifest);
    assert!(coverage.identity);
    assert!(coverage.digest);
    assert!(coverage.architecture_family);
    assert!(!coverage.tokenizer);
    assert!(!coverage.license);
    assert!(coverage.covers_required_fields() || manifest.parts.is_empty());

    manifest.tokenizer = Some("tokenizer.json".into());
    manifest.license = Some(ModelLicenseMetadata {
        identifier: "apache-2.0".into(),
        url: None,
        usage_restrictions: Vec::new(),
    });
    let coverage = NormalizedManifestCoverage::from_manifest(&manifest);
    assert!(coverage.tokenizer);
    assert!(coverage.license);
}

#[test]
fn model_format_roadmap_safetensors_manifest_validates_and_normalizes() {
    let manifest = SafetensorsManifest {
        tensors: vec![SafetensorsTensorEntry {
            name: "layer.0.weight".into(),
            shape: vec![4, 4],
            dtype: ModelDType::F32,
            byte_offset: 0,
            byte_length: 64,
        }],
        header_metadata: BTreeMap::new(),
    };
    assert!(manifest.validate().is_ok());
    let tensors = manifest.into_tensor_metadata();
    assert_eq!(tensors.len(), 1);
    assert_eq!(tensors[0].name, "layer.0.weight");
    assert_eq!(tensors[0].offset_bytes, Some(0));
    assert_eq!(tensors[0].size_bytes, Some(64));

    let degenerate = SafetensorsManifest {
        tensors: vec![SafetensorsTensorEntry {
            name: "bad".into(),
            shape: vec![0],
            dtype: ModelDType::F32,
            byte_offset: 0,
            byte_length: 4,
        }],
        header_metadata: BTreeMap::new(),
    };
    assert!(matches!(
        degenerate.validate(),
        Err(ModelFormatRoadmapError::SafetensorsInvalid { .. })
    ));
}

#[test]
fn model_format_roadmap_generation_config_as_defaults_and_override() {
    let parsed = GenerationConfigMetadata {
        temperature: Some(0.7),
        max_new_tokens: Some(256),
        stop_strings: vec!["<eos>".into()],
        ..Default::default()
    };
    let defaults = parsed.as_defaults();
    assert_eq!(defaults.temperature, Some(0.7));
    assert_eq!(defaults.max_tokens, Some(256));
    assert_eq!(defaults.stop_tokens, vec!["<eos>".to_string()]);

    assert_eq!(apply_generation_override(Some(0.7), Some(1.5)), Some(1.5));
    assert_eq!(apply_generation_override(Some(0.7), None), Some(0.7));
    assert_eq!(apply_generation_override::<f32>(None, None), None);
}

#[test]
fn model_format_roadmap_chat_template_requires_compatibility_and_variables() {
    let metadata = ChatTemplateMetadata {
        identity: "qwen-chat".into(),
        source: ChatTemplateSourceKind::EmbeddedInManifest,
        tokenizer_compatible: true,
        model_family_compatible: true,
        required_variables: BTreeSet::from(["messages".to_string()]),
        special_token_interaction: BTreeSet::new(),
    };
    assert!(validate_chat_template(&metadata, &BTreeSet::new()).is_err());
    assert!(validate_chat_template(&metadata, &BTreeSet::from(["messages".to_string()])).is_ok());

    let incompatible = ChatTemplateMetadata {
        tokenizer_compatible: false,
        ..metadata
    };
    assert!(matches!(
        validate_chat_template(&incompatible, &BTreeSet::new()),
        Err(ModelFormatRoadmapError::ChatTemplateInvalid { .. })
    ));

    assert_eq!(
        redact_chat_template_diagnostic("plain message"),
        "plain message"
    );
}

#[test]
fn model_format_roadmap_sentencepiece_unsupported_feature_fails_explicitly() {
    let metadata = SentencePieceMetadata {
        model_identity: "spm-1".into(),
        vocabulary_size: 32000,
        special_tokens: Vec::new(),
        normalization: None,
        browser_supported: false,
        license: None,
        supported_features: BTreeSet::from(["bpe".to_string()]),
    };
    assert!(reject_unsupported_sentencepiece_feature(&metadata, "bpe").is_ok());
    assert!(matches!(
        reject_unsupported_sentencepiece_feature(&metadata, "byte-fallback"),
        Err(ModelFormatRoadmapError::SentencePieceUnsupported { .. })
    ));
}

#[test]
fn model_format_roadmap_phases_are_ordered_1_through_12() {
    let mut ordinals: Vec<u8> = MODEL_FORMAT_ROADMAP_PHASES
        .iter()
        .map(|phase| phase.ordinal())
        .collect();
    ordinals.sort_unstable();
    assert_eq!(ordinals, (1..=12).collect::<Vec<_>>());
    for phase in MODEL_FORMAT_ROADMAP_PHASES {
        assert!(!phase.id().is_empty());
        assert!(phase.normalizes_into_existing_contract());
    }
}

#[test]
fn model_format_roadmap_rejects_format_shaped_provider_names() {
    for name in [
        "GGUFProvider",
        "SafetensorsProvider",
        "QwenSafetensorsProvider",
        "sentencepiece-provider",
        "tokenizer-json-provider",
    ] {
        assert!(
            reject_model_format_provider_name(name).is_err(),
            "{name} must be rejected"
        );
    }
}

#[test]
fn model_format_roadmap_format_parsers_cannot_supply_execution_graphs() {
    assert!(reject_format_execution_graph(true).is_err());
    assert!(reject_format_execution_graph(false).is_ok());
}

#[test]
fn model_format_roadmap_tokenizer_config_requires_explicit_runtime_validation() {
    assert!(reject_silent_tokenizer_config_override(false).is_err());
    assert!(reject_silent_tokenizer_config_override(true).is_ok());
}

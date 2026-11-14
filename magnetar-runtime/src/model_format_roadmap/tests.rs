//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::adapter::*;
use crate::model::*;
use crate::model::{
    ArtifactFormat, MODEL_ARTIFACT_SCHEMA_VERSION, ModelArchitecture, ModelArtifactId,
    ModelArtifactKind, ModelDType, ModelDigest, ModelLicenseMetadata, ModelManifest, ModelName,
    ModelRevision,
};
use crate::tests::*;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

use crate::compute_types::ComputeDType;
use crate::kernel::{
    KernelDequantizationBehavior, KernelQuantizationMetadata, KernelQuantizationMethod,
};
use crate::model::{
    ModelArtifactSource, ModelQuantization, ModelQuantizationFormat, ModelTrustStatus,
    ModelTrustStore,
};
use crate::operator::{OperatorFamily, OperatorId, TensorLayoutKind};
use crate::tokenizer::{
    SpecialToken, SpecialTokenKind, TokenizerArtifactId, TokenizerFamily, TokenizerId,
    TokenizerRevision,
};
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
        artifact_format: ArtifactFormat::HuggingFace,
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

#[test]
fn model_format_roadmap_memory_mapping_policy_rejects_raw_pointer_exposure() {
    let policy = MemoryMappingPolicy {
        mapping_allowed: true,
        streaming_read_allowed: true,
        exposes_raw_pointer: true,
    };
    assert!(policy.validate().is_err());
    let safe = MemoryMappingPolicy {
        exposes_raw_pointer: false,
        ..policy
    };
    assert!(safe.validate().is_ok());
}

#[test]
fn model_format_roadmap_allows_hardware_and_optimized_provider_names() {
    for name in [
        "ReferenceCpuProvider",
        "CudaProvider",
        "OptimizedCpuProvider",
    ] {
        assert!(
            reject_model_format_provider_name(name).is_ok(),
            "{name} must be allowed"
        );
    }
}

#[test]
fn model_format_roadmap_torch_dtype_never_forces_compute_dtype() {
    assert_eq!(
        torch_dtype_does_not_force_compute_dtype(Some("bfloat16"), ModelDType::F32),
        ModelDType::F32
    );
    assert_eq!(
        torch_dtype_does_not_force_compute_dtype(None, ModelDType::Bf16),
        ModelDType::Bf16
    );
}

#[test]
fn model_format_roadmap_normalizes_tokenizer_json() {
    let parsed = TokenizerJsonMetadata {
        vocabulary_size: 32000,
        added_tokens: Vec::new(),
        special_tokens: vec![SpecialToken::new(SpecialTokenKind::Bos, "<s>", 1)],
        normalizer: Some("nfc".into()),
        pre_tokenizer: Some("byte-level".into()),
        decoder: Some("byte-level".into()),
        supports_offsets: true,
    };
    let metadata = normalize_tokenizer_json(
        TokenizerId::new("tok-1").unwrap(),
        TokenizerArtifactId::new("tokenizer.json").unwrap(),
        ModelDigest::parse(format!("sha256:{}", "6".repeat(64))).unwrap(),
        TokenizerFamily::new("qwen").unwrap(),
        TokenizerRevision::new("v1").unwrap(),
        &parsed,
    )
    .unwrap();
    assert_eq!(metadata.vocabulary_size, 32000);
    assert!(metadata.supports_offsets);
    assert_eq!(metadata.special_tokens.len(), 1);

    let empty = TokenizerJsonMetadata {
        vocabulary_size: 0,
        ..parsed
    };
    assert!(matches!(
        normalize_tokenizer_json(
            TokenizerId::new("tok-2").unwrap(),
            TokenizerArtifactId::new("tokenizer.json").unwrap(),
            ModelDigest::parse(format!("sha256:{}", "7".repeat(64))).unwrap(),
            TokenizerFamily::new("qwen").unwrap(),
            TokenizerRevision::new("v1").unwrap(),
            &empty,
        ),
        Err(ModelFormatRoadmapError::TokenizerJsonInvalid { .. })
    ));
}

#[test]
fn model_format_roadmap_gguf_metadata_validates_and_normalizes_quantized_tensors() {
    let quantization = ModelQuantization {
        format: ModelQuantizationFormat::GgufQ4K,
        group_size: Some(32),
        block_size: None,
        scale_dtype: Some(ModelDType::F16),
        zero_point_dtype: None,
        per_channel: false,
        workspace_bytes: None,
        required_capabilities: Vec::new(),
    };
    let gguf = GgufMetadata {
        architecture: "qwen2".into(),
        alignment: 32,
        tensors: vec![GgufTensorEntry {
            name: "layer.0.weight".into(),
            shape: vec![4, 4],
            dtype: ModelDType::Q4K,
            quantization: Some(quantization),
        }],
        tokenizer_embedded: None,
        key_values: BTreeMap::new(),
    };
    assert!(gguf.validate().is_ok());
    let tensors = gguf.into_tensor_metadata();
    assert_eq!(tensors.len(), 1);
    assert!(tensors[0].quantization.is_some());
    assert_eq!(tensors[0].layout.as_deref(), Some("quantized-packed"));

    let empty = GgufMetadata {
        tensors: Vec::new(),
        ..gguf
    };
    assert!(matches!(
        empty.validate(),
        Err(ModelFormatRoadmapError::GgufInvalid { .. })
    ));

    assert!(reject_model_format_provider_name("GGUFProvider").is_err());
}

#[test]
fn model_format_roadmap_quantization_declaration_requires_scale_dtype_and_rejects_hidden_dequant() {
    let missing_scale = ModelFormatQuantizationDeclaration {
        model_quantization: ModelQuantization {
            format: ModelQuantizationFormat::Gptq,
            group_size: Some(64),
            block_size: None,
            scale_dtype: None,
            zero_point_dtype: None,
            per_channel: false,
            workspace_bytes: None,
            required_capabilities: Vec::new(),
        },
        kernel_compatibility: None,
    };
    assert!(matches!(
        validate_model_format_quantization(&missing_scale, true),
        Err(ModelFormatRoadmapError::QuantizationMetadataInvalid { .. })
    ));

    let with_kernel = ModelFormatQuantizationDeclaration {
        model_quantization: ModelQuantization {
            scale_dtype: Some(ModelDType::F16),
            ..missing_scale.model_quantization.clone()
        },
        kernel_compatibility: Some(KernelQuantizationMetadata {
            method: KernelQuantizationMethod::Int8,
            storage_dtype: ComputeDType::SInt8,
            compute_dtype: ComputeDType::Float32,
            accumulation_dtype: ComputeDType::Float32,
            scale_dtype: ComputeDType::Float32,
            zero_point_dtype: None,
            group_size: None,
            packing_layout: TensorLayoutKind::QuantizedPacked,
            dequantization: KernelDequantizationBehavior::ExplicitBeforeOperator,
            supported_operators: BTreeSet::from([OperatorId::magnetar(
                "matmul",
                1,
                OperatorFamily::LinearAlgebra,
            )]),
            conformance_tolerance_profile: "operator-default".into(),
        }),
    };
    assert!(validate_model_format_quantization(&with_kernel, true).is_ok());
    assert!(matches!(
        validate_model_format_quantization(&with_kernel, false),
        Err(ModelFormatRoadmapError::QuantizationMetadataInvalid { .. })
    ));
}

#[test]
fn model_format_roadmap_source_and_local_file_and_network_boundaries() {
    for source in [
        ModelArtifactSource::LocalPath("/models/qwen".into()),
        ModelArtifactSource::LocalCache("cache-1".into()),
        ModelArtifactSource::ClientProvided("client-1".into()),
        ModelArtifactSource::Registry("registry-1".into()),
        ModelArtifactSource::HuggingFace("qwen/qwen2".into()),
        ModelArtifactSource::Oci("oci://image".into()),
        ModelArtifactSource::Tachyon("tachyon-1".into()),
    ] {
        assert!(reject_arbitrary_model_download(&source).is_ok());
    }

    let local = ModelArtifactSource::LocalPath("/models/qwen".into());
    assert!(matches!(
        validate_local_file_boundary(&local, false),
        Err(ModelFormatRoadmapError::ModelFormatLocalFileDenied { .. })
    ));
    assert!(validate_local_file_boundary(&local, true).is_ok());

    assert!(reject_raw_network_model_reference("https://example.com/model.gguf").is_err());
    assert!(reject_raw_network_model_reference("qwen/qwen2").is_ok());
}

#[test]
fn model_format_roadmap_format_alone_does_not_grant_trust() {
    let store = ModelTrustStore::default();
    let manifest = fixture_model_manifest();
    let decision = model_format_grants_no_trust(&store, &manifest);
    assert_eq!(decision.status(), ModelTrustStatus::Unknown);

    let trusted_store = ModelTrustStore::default().trust_digest(manifest.id.digest.value.clone());
    let trusted_decision = model_format_grants_no_trust(&trusted_store, &manifest);
    assert_eq!(trusted_decision.status(), ModelTrustStatus::Trusted);
}

#[test]
fn model_format_roadmap_conformance_report_is_conformant() {
    let report = run_model_format_roadmap_conformance();
    assert!(!report.results.is_empty());
    for result in &report.results {
        assert!(
            result.passed,
            "{} failed: {:?}",
            result.requirement, result.diagnostic
        );
    }
    assert!(report.is_conformant());
}

#[test]
fn model_format_roadmap_detects_missing_and_duplicate_shard_tensors() {
    let mut index = ShardIndex::default();
    index.shards.push(ModelShard {
        id: ModelShardId::new("shard-0").unwrap(),
        digest: ModelDigest::parse(format!("sha256:{}", "3".repeat(64))).unwrap(),
        size_bytes: 1024,
        order: 0,
    });
    index.tensor_shard_map.insert(
        "layer.0.weight".into(),
        ModelShardId::new("shard-0").unwrap(),
    );
    assert!(detect_missing_shards(&index).is_ok());

    index.tensor_shard_map.insert(
        "layer.1.weight".into(),
        ModelShardId::new("shard-missing").unwrap(),
    );
    assert!(matches!(
        detect_missing_shards(&index),
        Err(ModelFormatRoadmapError::ShardMissing { .. })
    ));

    let duplicate = vec![
        tensor("layer.0.weight", vec![4, 4]),
        tensor("layer.0.weight", vec![4, 4]),
    ];
    assert!(detect_duplicate_tensor_names(&duplicate).is_err());

    let inconsistent = vec![
        tensor("layer.0.weight", vec![4, 4]),
        tensor("layer.0.weight", vec![8, 8]),
    ];
    assert!(matches!(
        validate_shard_tensor_shape_consistency(&inconsistent),
        Err(ModelFormatRoadmapError::ShardIndexInvalid { .. })
    ));

    let ordered = vec![
        ModelShard {
            id: ModelShardId::new("shard-0").unwrap(),
            digest: ModelDigest::parse(format!("sha256:{}", "4".repeat(64))).unwrap(),
            size_bytes: 1,
            order: 0,
        },
        ModelShard {
            id: ModelShardId::new("shard-1").unwrap(),
            digest: ModelDigest::parse(format!("sha256:{}", "5".repeat(64))).unwrap(),
            size_bytes: 1,
            order: 0,
        },
    ];
    assert!(matches!(
        validate_shard_loading_order(&ordered),
        Err(ModelFormatRoadmapError::ShardIndexInvalid { .. })
    ));
}

#[test]
fn model_format_roadmap_normalizes_lora_adapter_without_activating_or_trusting_it() {
    let base_model = AdapterBaseModelCompatibility {
        model_name: ModelName::new("qwen").unwrap(),
        model_revision: ModelRevision::new("v1").unwrap(),
        model_artifact: None,
        tokenizer: None,
        architecture: AdapterArchitectureCompatibility {
            family: "qwen".into(),
            implementation: "qwen2".into(),
            hidden_size: Some(4096),
            layer_count: Some(32),
            position_encoding: None,
            target_modules: BTreeSet::from(["q_proj".to_string()]),
            supported_storage_dtypes: BTreeSet::from([ModelDType::F16]),
            supported_compute_dtypes: BTreeSet::from([ComputeDType::Float16]),
            supported_quantization_formats: BTreeSet::new(),
        },
    };
    let metadata = LoraAdapterFormatMetadata {
        target_modules: vec!["q_proj".into()],
        rank: 8,
        alpha: 16,
        scaling: Some(2.0),
        dropout: Some(0.1),
        base_model,
        tensors: vec![tensor("q_proj.lora_a", vec![4096, 8])],
        storage_dtype: ModelDType::F16,
        compute_dtype: Some(ComputeDType::Float16),
        quantization: None,
        required_capabilities: Vec::new(),
        license: None,
        provenance: None,
    };
    let id = AdapterArtifactId::new(
        AdapterName::new("support-lora").unwrap(),
        AdapterRevision::new("r1").unwrap(),
        AdapterDigest::parse(format!("sha256:{}", "8".repeat(64))).unwrap(),
    );
    let artifact = normalize_lora_adapter(id, &metadata);
    assert_eq!(artifact.method, AdapterMethod::Lora);
    assert_eq!(artifact.rank, Some(8));
    assert_eq!(artifact.alpha, Some(16));
    assert_eq!(artifact.trust, AdapterTrustStatus::Unknown);
    assert_eq!(artifact.targets.len(), 1);
}

#[test]
fn model_format_roadmap_conformance_fixture_kinds_cover_twelve_categories() {
    assert_eq!(MODEL_FORMAT_CONFORMANCE_FIXTURES.len(), 12);
    for fixture in MODEL_FORMAT_CONFORMANCE_FIXTURES {
        assert!(!fixture.id().is_empty());
    }
}

#[test]
fn model_format_roadmap_observation_redacts_metadata() {
    let observation =
        ModelFormatRoadmapObservation::new(ModelFormatRoadmapObservationKind::ManifestNormalized)
            .with_artifact("fixture-model")
            .with_redacted_metadata("path", "/etc/secret/model.safetensors");
    assert_eq!(observation.artifact.as_deref(), Some("fixture-model"));
    assert!(observation.redacted_metadata.contains_key("path"));
}

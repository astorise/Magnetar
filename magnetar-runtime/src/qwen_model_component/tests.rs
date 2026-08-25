//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::{
    AdapterSetId, FallbackClass, GenerationModelReference, MemoryManager, ModelArchitecture,
    ModelArtifactId, ModelArtifactKind, ModelArtifactSource, ModelDigest, ModelInstanceDefinition,
    ModelLoadingCoordinator, ModelLoadingRequest, ModelLoadingRequestId, ModelName,
    ModelQuantizationPolicy, ModelRevision, ModelTrustDecision, ModelTrustStatus, ResourceAffinity,
    TokenizerId,
};
use std::collections::BTreeMap;

fn small_architecture() -> ModelComponentArchitectureMetadata {
    qwen_architecture_metadata(8, 2, 2, 2, 4, 16, 32, 64)
}

fn small_config() -> QwenConfig {
    QwenConfig::new(small_architecture(), QwenRopeConfig::standard(4))
}

fn small_identity() -> ModelComponentIdentity {
    qwen_component_identity(
        ModelComponentId::new("qwen-baseline").unwrap(),
        ModelComponentVersion::new(1, 0, 0),
        ModelComponentImplementationKind::RuntimeNative,
    )
}

#[test]
fn valid_minimal_config_builds_descriptor() {
    let descriptor = qwen_component_descriptor(small_identity(), &small_config()).unwrap();
    assert_eq!(descriptor.architecture.family, QWEN_ARCHITECTURE_FAMILY);
    assert_eq!(
        descriptor.target_modules.len(),
        QWEN_TARGET_MODULE_ROLES.len()
    );
}

#[test]
fn invalid_architecture_family_is_rejected() {
    let mut architecture = small_architecture();
    architecture.family = "llama".into();
    let config = QwenConfig::new(architecture, QwenRopeConfig::standard(4));
    assert_eq!(
        config.validate(&small_identity()),
        Err(QwenComponentError::ArchitectureUnsupported)
    );
}

#[test]
fn invalid_hidden_head_configuration_is_rejected() {
    let mut architecture = small_architecture();
    architecture.hidden_size = 9; // not attention_head_count * head_dimension
    let config = QwenConfig::new(architecture, QwenRopeConfig::standard(4));
    assert!(matches!(
        config.validate(&small_identity()),
        Err(QwenComponentError::ConfigInvalid {
            field: "head dimension",
            ..
        })
    ));
}

#[test]
fn missing_tensor_inventory_is_detected() {
    let config = small_config();
    let tensors = vec![ModelTensorMetadata {
        name: "token_embedding".into(),
        shape: vec![32, 8],
        storage_dtype: ModelDType::F32,
        layout: None,
        shard: None,
        offset_bytes: None,
        size_bytes: None,
        quantization: None,
        expected_compute_dtype: None,
    }];
    assert!(matches!(
        qwen_validate_tensor_inventory(&config, &tensors),
        Err(QwenComponentError::TensorInventoryMissing { .. })
    ));
}

#[test]
fn invalid_tensor_shape_is_detected() {
    let config = small_config();
    let tensor = ModelTensorMetadata {
        name: "lm_head".into(),
        shape: vec![1, 1],
        storage_dtype: ModelDType::F32,
        layout: None,
        shard: None,
        offset_bytes: None,
        size_bytes: None,
        quantization: None,
        expected_compute_dtype: None,
    };
    assert!(matches!(
        qwen_validate_tensor_shapes(&config, std::slice::from_ref(&tensor)),
        Err(QwenComponentError::TensorShapeMismatch { .. })
    ));
}

#[test]
fn target_modules_are_exposed() {
    let modules = qwen_target_modules();
    assert!(
        modules
            .iter()
            .any(|module| module.role == TargetModuleRole::QProj)
    );
    assert!(
        modules
            .iter()
            .any(|module| module.role == TargetModuleRole::LmHead)
    );
}

#[test]
fn prefill_graph_production_succeeds() {
    let config = small_config();
    let identity = small_identity();
    let result = qwen_prefill_graph(&config, &identity, 4, true).unwrap();
    assert_eq!(result.graph.phase, ExecutionGraphPhase::Prefill);
    assert!(result.validation.is_some());
}

#[test]
fn decode_graph_production_includes_kv_cache_append() {
    let config = small_config();
    let identity = small_identity();
    let result = qwen_decode_graph(&config, &identity).unwrap();
    assert_eq!(result.graph.phase, ExecutionGraphPhase::Decode);
    let appended = result
            .graph
            .edges
            .values()
            .any(|edge| matches!(&edge.kv_cache, Some(metadata) if metadata.behavior == GraphKvCacheBehavior::Append));
    assert!(appended);
}

#[test]
fn required_operator_scope_validation_passes_for_baseline_requirements() {
    validate_model_component_first_scope_requirements(&qwen_operator_requirements()).unwrap();
}

#[test]
fn out_of_scope_operator_is_rejected_by_first_scope() {
    let out_of_scope = OperatorRequirement::new(OperatorId::magnetar(
        "flash-attention",
        1,
        OperatorFamily::Attention,
    ));
    assert!(validate_model_component_first_scope_requirements(&[out_of_scope]).is_err());
}

#[test]
fn tokenizer_compatibility_declares_vocabulary_size() {
    let config = small_config();
    let compatibility = qwen_tokenizer_compatibility(&config);
    assert_eq!(
        compatibility.vocabulary_size,
        config.architecture.vocabulary_size
    );
}

#[test]
fn tokenizer_vocabulary_mismatch_is_rejected() {
    let config = small_config();
    let tokenizer = crate::TokenizerMetadata {
        id: crate::TokenizerId::new("qwen-tokenizer").unwrap(),
        artifact: crate::TokenizerArtifactId::new("qwen-tokenizer-artifact").unwrap(),
        digest: ModelDigest::sha256(b"tokenizer"),
        family: crate::TokenizerFamily::new("bpe").unwrap(),
        revision: crate::TokenizerRevision::new("rev1").unwrap(),
        vocabulary_size: config.architecture.vocabulary_size as u32 + 1,
        added_token_count: 0,
        token_id_range: crate::TokenIdRange::new(0, config.architecture.vocabulary_size as u32),
        model_max_length: None,
        special_tokens: Vec::new(),
        additional_special_tokens: Vec::new(),
        byte_fallback: false,
        normalization: None,
        pre_tokenizer: None,
        supports_offsets: false,
        supports_token_type_ids: false,
        supports_browser: true,
    };
    assert_eq!(
        qwen_validate_tokenizer_compatibility(&config, &tokenizer),
        Err(QwenComponentError::TokenizerIncompatible)
    );
}

#[test]
fn generation_defaults_exceeding_context_length_are_rejected() {
    let config = small_config();
    let defaults = crate::ModelGenerationDefaults {
        max_tokens: Some(1_000),
        ..crate::ModelGenerationDefaults::default()
    };
    assert!(matches!(
        qwen_validate_generation_defaults(&config, &defaults, None),
        Err(QwenComponentError::GenerationMetadataInvalid { .. })
    ));
}

#[test]
fn kv_cache_metadata_matches_architecture() {
    let config = small_config();
    let metadata = qwen_kv_cache_metadata(&config);
    assert_eq!(metadata.layer_count, config.architecture.layer_count);
    assert_eq!(metadata.head_dimension, config.architecture.head_dimension);
    assert!(!metadata.paged);
}

#[test]
fn adapter_target_validation_exposes_expected_modules() {
    let config = small_config();
    let compatibility = qwen_adapter_architecture_compatibility(&config, "qwen-baseline");
    assert!(compatibility.target_modules.contains("q_proj"));
    assert_eq!(
        compatibility.hidden_size,
        Some(config.architecture.hidden_size)
    );
}

#[test]
fn unsupported_quantization_is_rejected() {
    let config = small_config();
    let identity = small_identity();
    let descriptor = qwen_component_descriptor(identity, &config).unwrap();
    let mut manifest_architecture = ModelArchitecture::new(QWEN_ARCHITECTURE_FAMILY, "qwen-test");
    manifest_architecture.required_component_role = None;
    let manifest = ModelManifest {
        schema_version: MODEL_ARTIFACT_SCHEMA_VERSION,
        id: ModelArtifactId::new(
            ModelArtifactKind::ModelBundle,
            ModelName::new("qwen-test").unwrap(),
            ModelRevision::new("rev1").unwrap(),
            ModelDigest::sha256(b"qwen-test"),
        ),
        architecture: manifest_architecture,
        parts: BTreeMap::new(),
        storage_dtype: Some(ModelDType::F32),
        compute_dtype: Some(ModelDType::F32),
        supported_compute_dtypes: BTreeSet::from([ModelDType::F32]),
        tensors: Vec::new(),
        tokenizer: None,
        tokenizer_config: None,
        chat_template: None,
        prompt_template: None,
        generation: None,
        quantization: Some(crate::ModelQuantization {
            format: crate::ModelQuantizationFormat::GgufQ4K,
            group_size: None,
            block_size: None,
            scale_dtype: None,
            zero_point_dtype: None,
            per_channel: false,
            workspace_bytes: None,
            required_capabilities: Vec::new(),
        }),
        shards: Vec::new(),
        runtime_features: BTreeSet::new(),
        memory_features: BTreeSet::new(),
        provider_capabilities: Vec::new(),
        component: None,
        license: None,
        provenance: None,
        signatures: Vec::new(),
        source: Some(ModelArtifactSource::LocalCache("qwen-test".into())),
    };
    assert_eq!(
        qwen_validate_model_artifact(&descriptor, &config, &manifest),
        Err(QwenComponentError::QuantizationUnsupported)
    );
}

#[test]
fn authority_denies_forbidden_authorities() {
    let authority = qwen_authority();
    assert!(authority.contains(&ModelComponentAuthority::ModelArtifactRead));
    assert!(crate::validate_model_component_authority(["network"]).is_err());
    assert!(crate::validate_model_component_authority(["filesystem"]).is_err());
}

#[test]
fn no_provider_device_kernel_handle_exposure() {
    assert_eq!(
        crate::provider_handle_access_error(),
        ModelComponentError::ProviderAccessDenied
    );
    assert_eq!(
        crate::device_handle_access_error(),
        ModelComponentError::DeviceAccessDenied
    );
    assert_eq!(
        crate::kernel_handle_access_error(),
        ModelComponentError::KernelAccessDenied
    );
}

#[test]
fn rope_dynamic_scaling_unsupported_is_rejected() {
    let mut rope = QwenRopeConfig::standard(4);
    rope.scale = Some(2.0);
    assert!(matches!(
        rope.validate(4),
        Err(QwenComponentError::RopeUnsupported { .. })
    ));
}

#[test]
fn reference_cpu_covers_required_now_operators() {
    qwen_validate_reference_cpu_coverage().unwrap();
}

#[test]
fn browser_support_matches_implementation_kind() {
    assert!(qwen_browser_supported(ModelComponentImplementationKind::RuntimeNative).is_ok());
}

#[test]
fn contract_versions_accept_self_and_reject_newer_major() {
    qwen_validate_contract_versions(
        QWEN_TENSOR_CONTRACT_VERSION,
        QWEN_TOKENIZER_CONTRACT_VERSION,
        QWEN_KV_CACHE_CONTRACT_VERSION,
        QWEN_ADAPTER_CONTRACT_VERSION,
    )
    .unwrap();
    let newer_tensor = crate::CapabilityVersion::new(QWEN_TENSOR_CONTRACT_VERSION.major + 1, 0, 0);
    assert_eq!(
        qwen_validate_contract_versions(
            newer_tensor,
            QWEN_TOKENIZER_CONTRACT_VERSION,
            QWEN_KV_CACHE_CONTRACT_VERSION,
            QWEN_ADAPTER_CONTRACT_VERSION,
        ),
        Err(QwenComponentError::ComponentUnsupportedVersion)
    );
}

#[test]
fn tied_embedding_shape_must_match_vocabulary_and_hidden_size() {
    let mut config = small_config();
    config.tied_embeddings = true;
    let mut tensors = Vec::new();
    for name in qwen_expected_tensor_names(config.architecture.layer_count, true) {
        let shape = qwen_expected_tensor_shape(&name, &config).unwrap_or_default();
        tensors.push(ModelTensorMetadata {
            name,
            shape,
            storage_dtype: ModelDType::F32,
            layout: None,
            shard: None,
            offset_bytes: None,
            size_bytes: None,
            quantization: None,
            expected_compute_dtype: None,
        });
    }
    assert!(qwen_validate_tied_embedding_shape(&config, &tensors).is_ok());

    for tensor in &mut tensors {
        if tensor.name == "token_embedding" {
            tensor.shape = vec![1, 1];
        }
    }
    assert!(matches!(
        qwen_validate_tied_embedding_shape(&config, &tensors),
        Err(QwenComponentError::TensorShapeMismatch { .. })
    ));
}

#[test]
fn tied_embeddings_alias_lm_head_weight_in_graph() {
    let mut config = small_config();
    config.tied_embeddings = true;
    let identity = small_identity();
    let result = qwen_prefill_graph(&config, &identity, 4, false).unwrap();
    let lm_head_edge = result
        .graph
        .edges
        .get(&TensorEdgeId::new("weight.lm_head"))
        .unwrap();
    assert_eq!(
        lm_head_edge.aliasing,
        TensorAliasing::MayAlias(TensorEdgeId::new("weight.token_embedding"))
    );
}

#[test]
fn untied_lm_head_weight_does_not_alias() {
    let config = small_config();
    let identity = small_identity();
    let result = qwen_prefill_graph(&config, &identity, 4, false).unwrap();
    let lm_head_edge = result
        .graph
        .edges
        .get(&TensorEdgeId::new("weight.lm_head"))
        .unwrap();
    assert_eq!(lm_head_edge.aliasing, TensorAliasing::None);
}

#[test]
fn target_module_details_expose_layer_selector_and_insertion_point() {
    let details = qwen_target_module_details(4);
    assert_eq!(details.len(), QWEN_TARGET_MODULE_ROLES.len());
    let q_proj = details
        .iter()
        .find(|detail| detail.module.role == TargetModuleRole::QProj)
        .unwrap();
    assert_eq!(
        q_proj.layer_selector,
        AdapterLayerSelector::RangeInclusive { start: 0, end: 3 }
    );
    assert_eq!(
        q_proj.graph_insertion_point,
        QwenGraphInsertionPoint::AttentionProjection
    );
    assert!(
        q_proj
            .supported_adapter_methods
            .contains(&AdapterMethod::Lora)
    );
    let embedding = details
        .iter()
        .find(|detail| detail.module.role == TargetModuleRole::Embedding)
        .unwrap();
    assert_eq!(embedding.layer_selector, AdapterLayerSelector::All);
    assert_eq!(
        embedding.graph_insertion_point,
        QwenGraphInsertionPoint::Embedding
    );
    let lm_head = details
        .iter()
        .find(|detail| detail.module.role == TargetModuleRole::LmHead)
        .unwrap();
    assert_eq!(
        lm_head.graph_insertion_point,
        QwenGraphInsertionPoint::LogitsProjection
    );
}

#[test]
fn tokenizer_missing_required_bos_is_rejected() {
    let mut config = small_config();
    config.require_bos = true;
    let tokenizer = crate::TokenizerMetadata {
        id: crate::TokenizerId::new("qwen-tokenizer").unwrap(),
        artifact: crate::TokenizerArtifactId::new("qwen-tokenizer-artifact").unwrap(),
        digest: ModelDigest::sha256(b"tokenizer"),
        family: crate::TokenizerFamily::new("bpe").unwrap(),
        revision: crate::TokenizerRevision::new("rev1").unwrap(),
        vocabulary_size: config.architecture.vocabulary_size as u32,
        added_token_count: 0,
        token_id_range: crate::TokenIdRange::new(0, config.architecture.vocabulary_size as u32 - 1),
        model_max_length: None,
        special_tokens: vec![crate::SpecialToken::new(
            crate::SpecialTokenKind::Eos,
            "<eos>",
            0,
        )],
        additional_special_tokens: Vec::new(),
        byte_fallback: false,
        normalization: None,
        pre_tokenizer: None,
        supports_offsets: false,
        supports_token_type_ids: false,
        supports_browser: true,
    };
    assert_eq!(
        qwen_validate_tokenizer_compatibility(&config, &tokenizer),
        Err(QwenComponentError::TokenizerIncompatible)
    );
}

#[test]
fn chat_template_required_but_missing_is_rejected() {
    let mut config = small_config();
    config.chat_template_required = true;
    let identity = small_identity();
    let descriptor = qwen_component_descriptor(identity, &config).unwrap();
    let manifest_architecture = ModelArchitecture::new(QWEN_ARCHITECTURE_FAMILY, "qwen-test");
    let mut tensors = Vec::new();
    for name in qwen_expected_tensor_names(config.architecture.layer_count, false) {
        let shape = qwen_expected_tensor_shape(&name, &config).unwrap_or_default();
        tensors.push(ModelTensorMetadata {
            name,
            shape,
            storage_dtype: ModelDType::F32,
            layout: None,
            shard: None,
            offset_bytes: None,
            size_bytes: None,
            quantization: None,
            expected_compute_dtype: None,
        });
    }
    let manifest = ModelManifest {
        schema_version: MODEL_ARTIFACT_SCHEMA_VERSION,
        id: ModelArtifactId::new(
            ModelArtifactKind::ModelBundle,
            ModelName::new("qwen-test").unwrap(),
            ModelRevision::new("rev1").unwrap(),
            ModelDigest::sha256(b"qwen-test"),
        ),
        architecture: manifest_architecture,
        parts: BTreeMap::new(),
        storage_dtype: Some(ModelDType::F32),
        compute_dtype: Some(ModelDType::F32),
        supported_compute_dtypes: BTreeSet::from([ModelDType::F32]),
        tensors,
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
        source: Some(ModelArtifactSource::LocalCache("qwen-test".into())),
    };
    assert!(matches!(
        qwen_validate_model_artifact(&descriptor, &config, &manifest),
        Err(QwenComponentError::ComponentInvalid { .. })
    ));
}

#[test]
fn generation_defaults_non_authoritative_sampling_values_are_never_rejected() {
    let config = small_config();
    for temperature in [-5.0, 0.0, 1.0, 1_000.0] {
        let defaults = crate::ModelGenerationDefaults {
            temperature: Some(temperature),
            top_p: Some(-1.0),
            top_k: Some(u32::MAX),
            ..crate::ModelGenerationDefaults::default()
        };
        assert!(qwen_validate_generation_defaults(&config, &defaults, None).is_ok());
    }
}

#[test]
fn adapter_activation_rejected_when_baseline_lacks_graph_support() {
    assert_eq!(
        qwen_validate_adapter_activation_supported(QwenAdapterGraphSupport::baseline()),
        Err(QwenComponentError::AdapterUnsupported)
    );
    assert!(
        qwen_validate_adapter_activation_supported(QwenAdapterGraphSupport {
            overlay_supported: true,
            merge_supported: false,
        })
        .is_ok()
    );
}

#[test]
fn adapter_target_shape_mismatch_is_rejected() {
    let config = small_config();
    let target = AdapterTargetModule {
        name: "q_proj".into(),
        role: AdapterTargetModuleRole::QueryProjection,
        layer_selector: None,
        expected_shape: vec![1, 1],
    };
    assert!(matches!(
        qwen_validate_adapter_target_shapes(&config, std::slice::from_ref(&target)),
        Err(QwenComponentError::TensorShapeMismatch { .. })
    ));
    let a = &config.architecture;
    let matching = AdapterTargetModule {
        expected_shape: vec![a.hidden_size, a.attention_head_count * a.head_dimension],
        ..target
    };
    assert!(qwen_validate_adapter_target_shapes(&config, &[matching]).is_ok());
}

#[test]
fn tensor_scope_rejects_placeholder_dtype_and_non_contiguous_layout() {
    let f32_contiguous = TensorDescriptor::new(
        ShapeDescriptor::new(vec![1, 1]),
        DTypeDescriptor::portable(ComputeDType::Float32),
        LayoutDescriptor::Contiguous,
    );
    assert!(qwen_validate_tensor_scope(&f32_contiguous).is_ok());

    let float16 = TensorDescriptor::new(
        ShapeDescriptor::new(vec![1, 1]),
        DTypeDescriptor::portable(ComputeDType::Float16),
        LayoutDescriptor::Contiguous,
    );
    assert_eq!(
        qwen_validate_tensor_scope(&float16),
        Err(QwenComponentError::DTypeUnsupported)
    );

    let strided = TensorDescriptor::new(
        ShapeDescriptor::new(vec![1, 1]),
        DTypeDescriptor::portable(ComputeDType::Float32),
        LayoutDescriptor::Strided {
            strides_elements: vec![1, 1],
            offset_elements: 0,
        },
    );
    assert_eq!(
        qwen_validate_tensor_scope(&strided),
        Err(QwenComponentError::LayoutUnsupported)
    );
}

#[test]
fn explicit_dtype_and_layout_conversion_nodes_validate() {
    let raw_input = TensorEdge::new(
        TensorEdgeId::new("raw.input"),
        TensorDescriptor::new(
            ShapeDescriptor::new(vec![1, 2]),
            DTypeDescriptor::portable(ComputeDType::Float16),
            LayoutDescriptor::Strided {
                strides_elements: vec![1, 1],
                offset_elements: 0,
            },
        ),
    );
    let converted_dtype = f32_edge("converted.dtype", vec![1, 2]);
    let converted_layout = f32_edge("converted.layout", vec![1, 2]);
    let mut graph = ExecutionGraph::new(
        ExecutionGraphId::new("conversion-test"),
        ExecutionGraphPhase::Test,
    )
    .with_edge(raw_input);
    graph = qwen_insert_dtype_conversion(
        graph,
        "dtype-fix",
        TensorEdgeId::new("raw.input"),
        converted_dtype,
    );
    graph = qwen_insert_layout_conversion(
        graph,
        "layout-fix",
        TensorEdgeId::new("converted.dtype"),
        converted_layout,
    );
    graph.validate(&default_graph_catalog()).unwrap();
}

#[test]
fn reference_cpu_rejects_incompatible_kv_head_count() {
    let q = crate::HostTensor::new(vec![1, 4], vec![0.0; 4]).unwrap();
    let k = crate::HostTensor::new(vec![1, 4], vec![0.0; 4]).unwrap();
    let v = crate::HostTensor::new(vec![1, 4], vec![0.0; 4]).unwrap();
    // head_count=2 is not a multiple of kv_head_count=3: unsupported variant.
    assert!(crate::attention(&q, &k, &v, 2, 2, Some(3), None, true).is_err());
    // head_count=2 dividing kv_head_count=2 (standard multi-head) succeeds.
    assert!(crate::attention(&q, &k, &v, 2, 2, Some(2), None, true).is_ok());
}

#[test]
fn kv_cache_compatibility_embeds_component_version() {
    let identity_v1 = small_identity();
    let identity_v2 = qwen_component_identity(
        ModelComponentId::new("qwen-baseline").unwrap(),
        ModelComponentVersion::new(2, 0, 0),
        ModelComponentImplementationKind::RuntimeNative,
    );
    let model = GenerationModelReference::LoadedModelContext("ctx".into());
    let tokenizer = TokenizerId::new("qwen-tokenizer").unwrap();
    let compatibility_v1 =
        qwen_kv_cache_compatibility(&identity_v1, model.clone(), tokenizer.clone());
    let compatibility_v2 = qwen_kv_cache_compatibility(&identity_v2, model, tokenizer);
    assert_ne!(
        compatibility_v1.model_architecture,
        compatibility_v2.model_architecture
    );
}

#[test]
fn prefix_cache_compatibility_rejects_cross_version_and_cross_adapter_reuse() {
    let identity_v1 = small_identity();
    let identity_v2 = qwen_component_identity(
        ModelComponentId::new("qwen-baseline").unwrap(),
        ModelComponentVersion::new(2, 0, 0),
        ModelComponentImplementationKind::RuntimeNative,
    );
    let config = small_config();
    let model = GenerationModelReference::LoadedModelContext("ctx".into());
    let tokenizer = TokenizerId::new("qwen-tokenizer").unwrap();

    let base = qwen_prefix_cache_compatibility(
        &identity_v1,
        &config,
        model.clone(),
        tokenizer.clone(),
        None,
        None,
    );
    let same = qwen_prefix_cache_compatibility(
        &identity_v1,
        &config,
        model.clone(),
        tokenizer.clone(),
        None,
        None,
    );
    assert!(base.validate_reuse(&same).is_ok());

    let different_version = qwen_prefix_cache_compatibility(
        &identity_v2,
        &config,
        model.clone(),
        tokenizer.clone(),
        None,
        None,
    );
    assert!(base.validate_reuse(&different_version).is_err());

    let adapter_set = AdapterSetId::empty();
    let with_adapter = qwen_prefix_cache_compatibility(
        &identity_v1,
        &config,
        model,
        tokenizer,
        None,
        Some(&adapter_set),
    );
    assert!(base.validate_reuse(&with_adapter).is_err());
}

#[test]
fn observability_functions_preserve_component_identity_and_tag_events() {
    let component = ModelComponentId::new("qwen-baseline").unwrap();
    let observations = vec![
        qwen_observation_component_resolved(&component),
        qwen_observation_component_validated(&component),
        qwen_observation_component_rejected(&component, "unsupported"),
        qwen_observation_config_validated(&component),
        qwen_observation_tensor_inventory_checked(&component),
        qwen_observation_target_modules_exposed(&component),
        qwen_observation_tokenizer_compatibility_checked(&component),
        qwen_observation_kv_metadata_produced(&component),
        qwen_observation_prefill_graph_produced(&component),
        qwen_observation_decode_graph_produced(&component),
        qwen_observation_graph_validation_failed(&component, "bad graph"),
        qwen_observation_required_operator_missing(&component, "flash-attention"),
        qwen_observation_reference_cpu_coverage_missing(&component),
        qwen_observation_authority_denied(&component, "network"),
        qwen_observation_conformance_result(&component, true),
    ];
    assert_eq!(observations.len(), 15);
    for observation in &observations {
        assert_eq!(observation.component, Some(component.clone()));
        assert!(observation.redacted_metadata.contains_key("qwen-event"));
    }
}

#[test]
fn conformance_report_is_conformant_for_valid_config() {
    let report = qwen_conformance_report(&small_config(), &small_identity());
    assert!(report.is_conformant(), "failing checks: {report:?}");
    let names: BTreeSet<&str> = report.checks.iter().map(|check| check.name).collect();
    assert!(names.contains("valid-minimal-config"));
    assert!(names.contains("prefill-graph-production"));
    assert!(names.contains("decode-graph-production"));
}

fn integration_config() -> QwenConfig {
    QwenConfig::new(
        qwen_architecture_metadata(4, 1, 2, 2, 2, 8, 16, 32),
        QwenRopeConfig::standard(2),
    )
}

fn integration_manifest(architecture: ModelArchitecture) -> ModelManifest {
    let config = integration_config();
    let digest = "sha256:0000000000000000000000000000000000000000000000000000000000000001";
    let mut tensor_yaml = String::new();
    for name in qwen_expected_tensor_names(config.architecture.layer_count, false) {
        let shape = qwen_expected_tensor_shape(&name, &config).unwrap();
        let shape_text = shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        tensor_yaml.push_str(&format!(
            "  - name: {name}\n    shape: [{shape_text}]\n    storage_dtype: f32\n"
        ));
    }
    ModelManifest::from_yaml_str(&format!(
        r#"
schema: magnetar-model-artifact
schema_version: 1
kind: model-bundle
digest: {digest}
model:
  name: qwen-integration-model
  revision: r1
architecture:
  family: {family}
  identifier: {identifier}
storage_dtype: f32
compute_dtype: f32
supported_compute_dtypes: [f32]
artifacts:
  weights:
    kind: model-weights
    digest: {digest}
    size_bytes: 128
  config:
    kind: model-config
    digest: {digest}
    size_bytes: 16
tensors:
{tensor_yaml}"#,
        family = architecture.family,
        identifier = architecture.identifier,
    ))
    .unwrap()
}

/// End-to-end integration: a Qwen Component identity/config drives real
/// Model Loading (trust, precondition validation, memory admission,
/// architecture resolution, residency planning), Qwen artifact/tensor
/// validation against the exact loaded manifest, Model Instance creation
/// referencing the Qwen architecture implementation, and finally Runtime
/// graph planning/execution of the produced prefill graph. This is the
/// concrete mechanism by which Model Loading, Model Instance, and (via
/// the Execution Graph boundary) Generation use the Qwen Component today;
/// `generation::prefill`/`decode_step` remain pre-existing
/// architecture-agnostic placeholders pending a future Runtime inference
/// API change.
#[test]
fn qwen_component_integrates_with_model_loading_and_instance_and_graph_execution() {
    let config = integration_config();
    let identity = qwen_component_identity(
        ModelComponentId::new("qwen-integration").unwrap(),
        ModelComponentVersion::new(1, 0, 0),
        ModelComponentImplementationKind::RuntimeNative,
    );
    let architecture_implementation = qwen_architecture_implementation(
        &identity,
        crate::ModelArchitectureImplementationKind::ComponentBased,
    );
    let manifest = integration_manifest(architecture_implementation.architecture.clone());

    let descriptor = qwen_component_descriptor(identity.clone(), &config).unwrap();
    qwen_validate_model_artifact(&descriptor, &config, &manifest).unwrap();
    assert_eq!(qwen_target_modules().len(), QWEN_TARGET_MODULE_ROLES.len());

    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(architecture_implementation.clone());
    let mut memory = MemoryManager::default();
    let mut request = ModelLoadingRequest::new(
        ModelLoadingRequestId::new("qwen-integration-load"),
        manifest.id.clone(),
    );
    request.quantization_policy = ModelQuantizationPolicy::RejectUnsupported;
    let loaded = coordinator
        .load(
            request,
            &manifest,
            &ModelTrustDecision::new(ModelTrustStatus::Trusted, "trusted fixture"),
            &mut memory,
        )
        .unwrap();

    let instance = ModelInstanceDefinition::from_loaded_context(
        &loaded,
        architecture_implementation.clone(),
        ResourceAffinity::new(FallbackClass::Transparent),
    );
    assert_eq!(
        instance.architecture.architecture,
        architecture_implementation.architecture
    );
    assert_eq!(
        instance.architecture.architecture.identifier,
        qwen_component_compatibility_key(&identity)
    );

    let prefill = qwen_prefill_graph(&config, &identity, 4, true).unwrap();
    assert!(prefill.validation.is_some());
    let policy = crate::GraphPlanningPolicy::default();
    let plan = crate::plan_execution_graph(&prefill.graph, &default_graph_catalog(), &policy, None)
        .unwrap();
    assert!(!plan.execution_order.is_empty());
    crate::execute_graph_boundary(&prefill.graph, &default_graph_catalog(), &policy).unwrap();

    let kv_cache_compatibility = qwen_kv_cache_compatibility(
        &identity,
        GenerationModelReference::ModelInstance(
            crate::ModelInstanceId::new("qwen-instance").unwrap(),
        ),
        TokenizerId::new("qwen-tokenizer").unwrap(),
    );
    assert_eq!(
        kv_cache_compatibility.model_architecture,
        Some(qwen_component_compatibility_key(&identity))
    );
}

use magnetar_runtime::{
    ActivationKind, AttentionVariant, CapabilityId, CapabilityVersion, ComponentDigest,
    ComponentEngineRequirements, ComponentManifest, ComponentMetadata, ComponentSource,
    ComponentTrustStatus, DTypeDescriptor, ExecutionGraph, ExecutionGraphId, ExecutionGraphPhase,
    LayoutDescriptor, ModelComponentArchitectureMetadata, ModelComponentAuthority,
    ModelComponentCapabilityKind, ModelComponentCapabilityRequirement, ModelComponentDescriptor,
    ModelComponentError, ModelComponentId, ModelComponentIdentity,
    ModelComponentImplementationKind, ModelComponentKvCacheMetadata, ModelComponentModelType,
    ModelComponentQuantizationCompatibility, ModelComponentSignatureState,
    ModelComponentTokenizerCompatibility, ModelComponentTrustStatus, ModelComponentVersion,
    ModelQuantizationFormat, NormalizationKind, OperatorFamily, OperatorId, OperatorRequirement,
    PositionEncodingKind, ShapeDescriptor, TargetModuleMetadata, TargetModuleRole,
    TensorDescriptor, TensorEdge, TensorEdgeId, device_handle_access_error,
    kernel_handle_access_error, memory_pointer_access_error, provider_handle_access_error,
    provider_owned_resource_access_error, validate_model_component_authority,
    validate_model_component_config_data, validate_model_component_role,
};
use std::collections::BTreeSet;

fn component_id() -> ModelComponentId {
    ModelComponentId::new("qwen.component").unwrap()
}

fn identity() -> ModelComponentIdentity {
    ModelComponentIdentity::new(
        component_id(),
        ModelComponentVersion::new(1, 0, 0),
        ModelComponentImplementationKind::TestFixture,
    )
    .trusted()
    .with_architecture_family("qwen")
    .with_architecture_revision("qwen2")
    .with_model_artifact_schema_version(1)
}

fn architecture() -> ModelComponentArchitectureMetadata {
    ModelComponentArchitectureMetadata {
        family: "qwen".into(),
        model_type: ModelComponentModelType::CausalLanguageModel,
        hidden_size: 16,
        layer_count: 2,
        attention_head_count: 4,
        kv_head_count: 2,
        head_dimension: 4,
        intermediate_size: 64,
        vocabulary_size: 32000,
        context_length: 128,
        position_encoding: PositionEncodingKind::Rotary,
        normalization: NormalizationKind::RmsNorm,
        activation: ActivationKind::Silu,
        attention: AttentionVariant::GroupedQuery,
        quantization: Some(ModelQuantizationFormat::GgufQ4K),
        tokenizer_family: None,
        adapter_target_modules: [TargetModuleRole::QProj, TargetModuleRole::VProj].into(),
    }
}

fn descriptor() -> ModelComponentDescriptor {
    let supported_methods = [ModelQuantizationFormat::GgufQ4K].into();
    ModelComponentDescriptor {
        identity: identity(),
        architecture: architecture(),
        target_modules: TargetModuleRole::all()
            .into_iter()
            .map(TargetModuleMetadata::canonical)
            .collect(),
        graph_phases: [ExecutionGraphPhase::Warmup, ExecutionGraphPhase::Decode].into(),
        operator_requirements: vec![OperatorRequirement::new(OperatorId::magnetar(
            "matmul",
            1,
            OperatorFamily::LinearAlgebra,
        ))],
        capability_requirements: vec![ModelComponentCapabilityRequirement {
            kind: ModelComponentCapabilityKind::GraphProduction,
            id: CapabilityId::new("magnetar:model-component/graph-production"),
            min_version: CapabilityVersion::new(1, 0, 0),
        }],
        authority: [
            ModelComponentAuthority::GraphProduction,
            ModelComponentAuthority::OperatorCatalogRead,
            ModelComponentAuthority::ModelArtifactRead,
        ]
        .into(),
        kv_cache: Some(ModelComponentKvCacheMetadata {
            layer_count: 2,
            head_count: 4,
            kv_head_count: 2,
            head_dimension: 4,
            cache_dtype: "bf16".into(),
            layout_preference: "paged".into(),
            paged: true,
            append_semantics: "append-position".into(),
            position_behavior: "rotary".into(),
        }),
        tokenizer: Some(ModelComponentTokenizerCompatibility {
            vocabulary_size: 32000,
            special_tokens: BTreeSet::from(["eos".into(), "bos".into()]),
            family: None,
            chat_template_required: true,
            added_token_behavior: Some("append-only".into()),
        }),
        quantization: Some(ModelComponentQuantizationCompatibility {
            supported_methods,
            tensor_grouping: Some("group-size-32".into()),
            scale_metadata_required: true,
            zero_point_metadata_required: false,
            packed_layout: Some("q4_k".into()),
            dequantization_operators: BTreeSet::new(),
            quantized_operators: BTreeSet::new(),
        }),
    }
}

#[test]
fn model_component_identity_role_and_architecture_are_validated() {
    let descriptor = descriptor();

    descriptor.validate().unwrap();
    assert_eq!(
        descriptor.identity.trust,
        ModelComponentTrustStatus::Trusted
    );
    assert!(descriptor.supports_graph_phase(ExecutionGraphPhase::Decode));
    assert_eq!(
        TargetModuleRole::QProj.canonical_name(),
        descriptor.target_modules[0].architecture_name
    );
}

#[test]
fn model_component_rejects_untrusted_invalid_or_unsupported_identity() {
    let mut rejected_identity = identity();
    rejected_identity.trust = ModelComponentTrustStatus::Rejected;
    assert_eq!(
        rejected_identity.validate(),
        Err(ModelComponentError::ModelComponentUntrusted)
    );

    assert!(matches!(
        ModelComponentId::new("../provider"),
        Err(ModelComponentError::ModelComponentInvalid { .. })
    ));

    let mut unsupported_identity = identity();
    unsupported_identity.version = ModelComponentVersion::new(2, 0, 0);
    assert_eq!(
        unsupported_identity.validate(),
        Err(ModelComponentError::ModelComponentUnsupportedVersion)
    );
}

#[test]
fn architecture_metadata_rejects_invalid_shapes_and_unsupported_family() {
    let mut metadata = architecture();
    metadata.hidden_size = 15;
    assert!(matches!(
        metadata.validate(&identity()),
        Err(ModelComponentError::ArchitectureMetadataInvalid {
            field: "head dimension",
            ..
        })
    ));

    let mut metadata = architecture();
    metadata.family = "llama".into();
    assert_eq!(
        metadata.validate(&identity()),
        Err(ModelComponentError::ArchitectureUnsupported)
    );
}

#[test]
fn model_component_authority_is_inference_scoped() {
    let allowed =
        validate_model_component_authority(["model-artifact-read", "graph-production"]).unwrap();
    assert!(allowed.contains(&ModelComponentAuthority::ModelArtifactRead));

    assert_eq!(
        validate_model_component_authority(["filesystem"]),
        Err(ModelComponentError::AuthorityDenied {
            authority: "filesystem".into()
        })
    );
    assert_eq!(
        validate_model_component_config_data("file:///tmp/model.json"),
        Err(ModelComponentError::AuthorityDenied {
            authority: "filesystem".into()
        })
    );
}

#[test]
fn component_manifest_role_must_be_model_component() {
    let manifest = ComponentManifest {
        component: ComponentMetadata::new("qwen.component", "1.0.0", "fixture"),
        role: "tokenizer".into(),
        digest: ComponentDigest::sha256(b"component"),
        runtime_min_version: "0.1.0".into(),
        runtime_max_version: None,
        imports: BTreeSet::new(),
        optional_imports: BTreeSet::new(),
        exports: BTreeSet::new(),
        capabilities: Vec::new(),
        engine: ComponentEngineRequirements::default(),
        authority: Vec::new(),
        publisher: None,
        source: ComponentSource {
            kind: "local".into(),
            uri: "fixture".into(),
        },
        signatures: Vec::new(),
    };

    assert!(matches!(
        validate_model_component_role(&manifest),
        Err(ModelComponentError::ModelComponentInvalid { .. })
    ));
}

#[test]
fn operator_requirements_are_portable_not_provider_kernel_names() {
    let portable = OperatorRequirement::new(OperatorId::magnetar(
        "attention",
        1,
        OperatorFamily::Attention,
    ));
    portable.validate().unwrap();

    let provider_specific = OperatorRequirement::new(OperatorId::new(
        "cuda",
        "flash_attention",
        1,
        OperatorFamily::Attention,
    ));
    assert_eq!(
        provider_specific.validate(),
        Err(ModelComponentError::OperatorCatalogIncompatible)
    );
}

#[test]
fn component_produced_graph_is_validated_by_runtime_graph_contract() {
    let graph = ExecutionGraph::new(
        ExecutionGraphId::new("empty-decode"),
        ExecutionGraphPhase::Decode,
    );
    let result = magnetar_runtime::GraphProductionResult::validated(
        graph,
        &component_id(),
        &magnetar_runtime::default_graph_catalog(),
    )
    .unwrap();

    assert_eq!(result.validation.unwrap().validated_nodes, 0);
    assert!(!result.graph.producer.component_has_raw_provider_access());
}

#[test]
fn graph_validation_failure_is_reported_as_model_component_error() {
    let edge = TensorEdgeId::new("missing-producer-node");
    let graph = ExecutionGraph::new(ExecutionGraphId::new("bad"), ExecutionGraphPhase::Decode)
        .with_edge(TensorEdge::new(
            edge,
            TensorDescriptor::new(
                ShapeDescriptor::new([1, 1]),
                DTypeDescriptor::portable(magnetar_runtime::ComputeDType::Float32),
                LayoutDescriptor::Contiguous,
            ),
        ));

    let result = magnetar_runtime::GraphProductionResult::validated(
        graph,
        &component_id(),
        &magnetar_runtime::default_graph_catalog(),
    );

    assert!(matches!(
        result,
        Ok(_) | Err(ModelComponentError::GraphValidationFailed(_))
    ));
}

#[test]
fn model_component_metadata_covers_cache_tokenizer_quantization_and_signatures() {
    let descriptor = descriptor();

    assert!(descriptor.kv_cache.unwrap().paged);
    assert_eq!(descriptor.tokenizer.unwrap().vocabulary_size, 32000);
    assert!(
        descriptor
            .quantization
            .unwrap()
            .supported_methods
            .contains(&ModelQuantizationFormat::GgufQ4K)
    );
    assert_eq!(
        ModelComponentSignatureState::Verified,
        ModelComponentSignatureState::Verified
    );
    assert_eq!(ComponentTrustStatus::Trusted, ComponentTrustStatus::Trusted);
}

#[test]
fn raw_provider_device_kernel_memory_and_provider_resources_are_denied() {
    assert_eq!(
        provider_handle_access_error(),
        ModelComponentError::ProviderAccessDenied
    );
    assert_eq!(
        device_handle_access_error(),
        ModelComponentError::DeviceAccessDenied
    );
    assert_eq!(
        kernel_handle_access_error(),
        ModelComponentError::KernelAccessDenied
    );
    assert_eq!(
        memory_pointer_access_error(),
        ModelComponentError::MemoryPointerAccessDenied
    );
    assert_eq!(
        provider_owned_resource_access_error(),
        ModelComponentError::ProviderOwnedResourceAccessDenied
    );
}

use magnetar_runtime::{
    MemoryManager, MemoryManagerConfig, ModelArchitecture, ModelArchitectureImplementation,
    ModelArchitectureImplementationKind, ModelDType, ModelLoadingCachePolicy,
    ModelLoadingCoordinator, ModelLoadingErrorCode, ModelLoadingObservationKind,
    ModelLoadingRequest, ModelLoadingRequestId, ModelLoadingState, ModelQuantizationHandling,
    ModelQuantizationPolicy, ModelResidencyLocation, ModelShardingPolicy, ModelTrustDecision,
    ModelTrustStatus, ModelUnloadPolicy, compute_dtype_supported, invalidates_kv_cache_on_unload,
    reload_is_new_loading_process,
};

fn digest() -> String {
    "sha256:0000000000000000000000000000000000000000000000000000000000000001".into()
}

fn valid_manifest() -> magnetar_runtime::ModelManifest {
    magnetar_runtime::ModelManifest::from_yaml_str(&format!(
        r#"
schema: magnetar-model-artifact
schema_version: 1
kind: model-bundle
digest: {}
model:
  name: qwen.example
  revision: r1
architecture:
  family: qwen
  identifier: qwen2
storage_dtype: int8
compute_dtype: bf16
supported_compute_dtypes: [bf16, fp16]
artifacts:
  weights:
    kind: model-weights
    digest: {}
    size_bytes: 128
  config:
    kind: model-config
    digest: {}
    size_bytes: 16
quantization:
  format: q4_k
  workspace_bytes: 64
shards:
  - id: shard0
    digest: {}
    size_bytes: 128
    order: 0
tensors:
  - name: transformer.wte.weight
    shape: [4, 8]
    storage_dtype: int8
    shard: shard0
"#,
        digest(),
        digest(),
        digest(),
        digest()
    ))
    .unwrap()
}

fn trusted() -> ModelTrustDecision {
    ModelTrustDecision::new(ModelTrustStatus::Trusted, "trusted fixture")
}

fn coordinator() -> ModelLoadingCoordinator {
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(ModelArchitectureImplementation {
        architecture: ModelArchitecture::new("qwen", "qwen2"),
        kind: ModelArchitectureImplementationKind::TestFixture,
        required_capabilities: Vec::new(),
    });
    coordinator
}

#[test]
fn loading_rejects_untrusted_artifact_before_memory_allocation() {
    let manifest = valid_manifest();
    let mut coordinator = coordinator();
    let mut memory = MemoryManager::default();
    let request =
        ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    let untrusted = ModelTrustDecision::new(ModelTrustStatus::Rejected, "policy denied");

    let error = coordinator
        .load(request, &manifest, &untrusted, &mut memory)
        .unwrap_err();

    assert_eq!(error.code, ModelLoadingErrorCode::ModelArtifactUntrusted);
    assert_eq!(memory.allocations().count(), 0);
}

#[test]
fn loading_resolves_architecture_before_planning() {
    let manifest = valid_manifest();
    let coordinator = ModelLoadingCoordinator::new();

    let error = coordinator
        .resolve_architecture(&manifest.architecture)
        .unwrap_err();

    assert_eq!(
        error.code,
        ModelLoadingErrorCode::ArchitectureImplementationMissing
    );
}

#[test]
fn loading_creates_runtime_owned_ready_context_without_raw_handles() {
    let manifest = valid_manifest();
    let mut coordinator = coordinator();
    let mut memory = MemoryManager::default();
    let mut request =
        ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    request.quantization_policy = ModelQuantizationPolicy::DequantizeAtLoad;
    request.sharding_policy = ModelShardingPolicy::Sequential;

    let context = coordinator
        .load(request, &manifest, &trusted(), &mut memory)
        .unwrap();

    assert_eq!(context.state, ModelLoadingState::Ready);
    assert!(context.can_start_inference());
    assert!(!context.plan.has_raw_native_handles());
    assert_eq!(
        context.plan.quantization_handling,
        ModelQuantizationHandling::DequantizeAtLoad(
            magnetar_runtime::ModelQuantizationFormat::GgufQ4K
        )
    );
    assert_eq!(
        context.plan.memory_placements,
        vec![ModelResidencyLocation::Host]
    );
    assert_eq!(memory.allocations().count(), 1);
    assert!(
        coordinator
            .observations()
            .iter()
            .any(|observation| observation.kind == ModelLoadingObservationKind::ModelReady)
    );
}

#[test]
fn lifecycle_transitions_reject_invalid_jump() {
    assert!(ModelLoadingState::Requested.can_transition_to(ModelLoadingState::Validating));
    assert!(!ModelLoadingState::Requested.can_transition_to(ModelLoadingState::Ready));
    assert!(ModelLoadingState::Ready.can_transition_to(ModelLoadingState::Unloading));
    assert!(ModelLoadingState::Unloading.can_transition_to(ModelLoadingState::Unloaded));
}

#[test]
fn memory_budget_failure_does_not_allocate() {
    let manifest = valid_manifest();
    let mut coordinator = coordinator();
    let mut memory = MemoryManager::default();
    let mut request =
        ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    request.quantization_policy = ModelQuantizationPolicy::DequantizeAtLoad;
    request.memory_budget_bytes = Some(1);

    let error = coordinator
        .load(request, &manifest, &trusted(), &mut memory)
        .unwrap_err();

    assert_eq!(error.code, ModelLoadingErrorCode::MemoryFeasibilityFailed);
    assert_eq!(memory.allocations().count(), 0);
}

#[test]
fn unsupported_quantization_requires_explicit_policy() {
    let manifest = valid_manifest();
    let mut coordinator = coordinator();
    let mut memory = MemoryManager::default();
    let request =
        ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());

    let error = coordinator
        .load(request, &manifest, &trusted(), &mut memory)
        .unwrap_err();

    assert_eq!(error.code, ModelLoadingErrorCode::QuantizationUnsupported);
}

#[test]
fn memory_manager_rejection_maps_to_loading_error() {
    let manifest = valid_manifest();
    let mut coordinator = coordinator();
    let mut memory = MemoryManager::new(MemoryManagerConfig {
        max_runtime_bytes: Some(1),
        ..MemoryManagerConfig::default()
    });
    let mut request =
        ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    request.quantization_policy = ModelQuantizationPolicy::DequantizeAtLoad;

    let error = coordinator
        .load(request, &manifest, &trusted(), &mut memory)
        .unwrap_err();

    assert_eq!(error.code, ModelLoadingErrorCode::MemoryAllocationFailed);
}

#[test]
fn session_kv_cache_reload_and_dtype_policies_are_explicit() {
    let manifest = valid_manifest();
    assert!(compute_dtype_supported(&manifest, ModelDType::Bf16));
    assert!(!compute_dtype_supported(&manifest, ModelDType::F32));
    assert!(invalidates_kv_cache_on_unload(
        ModelLoadingCachePolicy::InvalidateKvCacheOnUnload
    ));

    let request = ModelLoadingRequest::new(ModelLoadingRequestId::new("load-2"), manifest.id);
    let reload = magnetar_runtime::ModelReloadRequest {
        previous_residency: magnetar_runtime::ModelResidencyId::new(1),
        request,
        allow_context_mutation: false,
    };
    assert!(reload_is_new_loading_process(&reload));
    assert_eq!(
        ModelUnloadPolicy::DrainActiveUse,
        ModelUnloadPolicy::DrainActiveUse
    );
}

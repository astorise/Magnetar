use magnetar_runtime::{
    ModelDType, ModelLoadingCachePolicy, ModelLoadingCoordinator, ModelLoadingErrorCode,
    ModelLoadingState, ModelUnloadPolicy, compute_dtype_supported, invalidates_kv_cache_on_unload,
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

/// `ModelLoadingCoordinator::load` became `pub(crate)`
/// (`seal-model-loading-and-instance-creation-primitives`), so every test
/// that called it directly moved into `magnetar-runtime/src/tests.rs`,
/// where `pub(crate)` access still applies. What remains here genuinely
/// does not depend on `load`.
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
fn lifecycle_transitions_reject_invalid_jump() {
    assert!(ModelLoadingState::Requested.can_transition_to(ModelLoadingState::Validating));
    assert!(!ModelLoadingState::Requested.can_transition_to(ModelLoadingState::Ready));
    assert!(ModelLoadingState::Ready.can_transition_to(ModelLoadingState::Unloading));
    assert!(ModelLoadingState::Unloading.can_transition_to(ModelLoadingState::Unloaded));
}

#[test]
fn session_kv_cache_reload_and_dtype_policies_are_explicit() {
    let manifest = valid_manifest();
    assert!(compute_dtype_supported(&manifest, ModelDType::Bf16));
    assert!(!compute_dtype_supported(&manifest, ModelDType::F32));
    assert!(invalidates_kv_cache_on_unload(
        ModelLoadingCachePolicy::InvalidateKvCacheOnUnload
    ));

    let request = magnetar_runtime::ModelLoadingRequest::new(
        magnetar_runtime::ModelLoadingRequestId::new("load-2"),
        manifest.id,
    );
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

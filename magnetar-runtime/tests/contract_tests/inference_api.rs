use magnetar_runtime::{
    InferenceApiError, ModelArtifactId, ModelArtifactKind, ModelDigest, ModelName, ModelRef,
    ModelRegistry, ModelResolutionRequest, ModelResolutionSource, ModelRevision,
};

#[test]
fn inference_api_model_reference_resolves_through_local_registry() {
    let reference = ModelRef::new("qwen-test").unwrap();
    let artifact = ModelArtifactId::new(
        ModelArtifactKind::ModelWeights,
        ModelName::new("qwen").unwrap(),
        ModelRevision::new("1").unwrap(),
        ModelDigest::sha256(b"weights"),
    );
    let mut registry = ModelRegistry::new();
    registry.register(reference.clone(), artifact.clone());

    let result = registry
        .resolve(&ModelResolutionRequest::new(reference))
        .unwrap();
    assert_eq!(result.artifact, artifact);
}

#[test]
fn inference_api_model_reference_resolution_fails_for_unregistered_reference() {
    let registry = ModelRegistry::new();
    let reference = ModelRef::new("unknown-model").unwrap();
    let error = registry
        .resolve(&ModelResolutionRequest::new(reference))
        .unwrap_err();
    assert!(matches!(
        error,
        InferenceApiError::ModelResolutionFailed { .. }
    ));
}

#[test]
fn inference_api_model_resolution_source_placeholders_fail_structured() {
    let mut registry = ModelRegistry::new();
    let reference = ModelRef::new("qwen-test").unwrap();
    registry.register(
        reference.clone(),
        ModelArtifactId::new(
            ModelArtifactKind::ModelWeights,
            ModelName::new("qwen").unwrap(),
            ModelRevision::new("1").unwrap(),
            ModelDigest::sha256(b"weights"),
        ),
    );

    for source in [
        ModelResolutionSource::FutureExternalSource,
        ModelResolutionSource::FutureTachyonSource,
    ] {
        let mut request = ModelResolutionRequest::new(reference.clone());
        request.source = source;
        let error = registry.resolve(&request).unwrap_err();
        assert!(matches!(
            error,
            InferenceApiError::ModelResolutionFailed { .. }
        ));
    }
}

#[test]
fn inference_api_model_resolution_local_registry_source_still_resolves() {
    let reference = ModelRef::new("qwen-test").unwrap();
    let artifact = ModelArtifactId::new(
        ModelArtifactKind::ModelWeights,
        ModelName::new("qwen").unwrap(),
        ModelRevision::new("1").unwrap(),
        ModelDigest::sha256(b"weights"),
    );
    let mut registry = ModelRegistry::new();
    registry.register(reference.clone(), artifact.clone());

    let mut request = ModelResolutionRequest::new(reference);
    request.source = ModelResolutionSource::LocalRegistry;
    assert_eq!(registry.resolve(&request).unwrap().artifact, artifact);
}

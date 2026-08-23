use magnetar_runtime::{
    ComponentDigest, ComponentError, InferenceArtifactKind, InferenceArtifactReference,
};

#[test]
fn component_inference_artifact_rejects_path_like_identity() {
    let digest = ComponentDigest::sha256(b"model");

    assert!(matches!(
        InferenceArtifactReference::new(InferenceArtifactKind::Model, "../model", digest),
        Err(ComponentError::ArtifactRejected { .. })
    ));
}

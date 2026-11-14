use magnetar_runtime::{
    ComponentDigest, ComponentError, InferenceArtifactKind, InferenceArtifactReference,
};
use magnetar_runtime::{ComponentManager, InferenceArtifactRegistry};
use std::fs;

#[test]
fn component_inference_artifact_rejects_path_like_identity() {
    let digest = ComponentDigest::sha256(b"model");

    assert!(matches!(
        InferenceArtifactReference::new(InferenceArtifactKind::Model, "../model", digest),
        Err(ComponentError::ArtifactRejected { .. })
    ));
}

#[test]
fn inference_artifact_registry_handles_tokenizer_template_adapter_and_quantization() {
    let mut registry = InferenceArtifactRegistry::default();
    for kind in [
        InferenceArtifactKind::Tokenizer,
        InferenceArtifactKind::PromptTemplate,
        InferenceArtifactKind::Adapter,
        InferenceArtifactKind::Quantization,
    ] {
        let id = format!("{kind:?}").to_ascii_lowercase();
        registry
            .register(
                InferenceArtifactReference::new(kind, &id, ComponentDigest::sha256(id.as_bytes()))
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(registry.resolve(kind, &id, None).unwrap().kind, kind);
    }
    assert!(matches!(
        registry.resolve(InferenceArtifactKind::Tokenizer, "C:\\tokenizer.json", None),
        Err(ComponentError::ArtifactRejected { .. })
    ));
}

#[test]
fn component_discovery_returns_only_wasm_artifacts() {
    let directory =
        std::env::temp_dir().join(format!("magnetar-components-{}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    fs::write(directory.join("valid.wasm"), []).unwrap();
    fs::write(directory.join("ignored.txt"), []).unwrap();

    let discovered = ComponentManager::discover([&directory]).unwrap();
    fs::remove_dir_all(&directory).unwrap();
    assert_eq!(discovered, vec![directory.join("valid.wasm")]);
}

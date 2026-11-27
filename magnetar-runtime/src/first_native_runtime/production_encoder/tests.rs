use super::super::graph_execution::modernbert_graph_tests::{Case, graph};
use super::super::tests::ProductionIntegrationPayloadSource;
use super::*;

#[test]
fn resident_encoder_releases_resources_after_success_failure_and_unload() {
    let cases: Vec<Case> = serde_json::from_str(include_str!(
        "../../../fixtures/modernbert-encoder-reference.json"
    ))
    .unwrap();
    let case = cases.iter().find(|case| !case.bias).unwrap();
    let (_, bindings) = graph(case);
    let bytes =
        include_bytes!("../../../fixtures/components/modernbert-encoder.component.wasm").to_vec();
    let component = ComponentDigest::sha256(&bytes);
    let component_manifest = include_str!(
        "../../../fixtures/components/qwen-real.component.wasm.magnetar-component.yaml"
    )
    .replace(
        "sha256:321b42b83773544e1ba553cc7196d3d742cd9ef1e2af3f04da72b66e402a730f",
        &component.value,
    )
    .replace("magnetar.qwen.real", "magnetar.modernbert.resident-test")
    .replace("qwen-model-component", "modernbert-model-component")
    .replace("    - qwen2\r\n", "    - modernbert\r\n")
    .replace("    - qwen2\n", "    - modernbert\n")
    .replace("1.14.0", "1.12.0");
    register_inference_component_artifact(
        bytes,
        component_manifest.into_bytes(),
        &ComponentTrustStore::default().trust_digest(&component.value),
    )
    .unwrap();

    let fixture = e2e_fixture().unwrap();
    let mut architecture = architecture_config_from_first_native_model_config(&fixture.config);
    let mut manifest = fixture.manifest;
    manifest.id = ModelArtifactId::new(
        ModelArtifactKind::ModelBundle,
        ModelName::new("resident-modernbert-test").unwrap(),
        ModelRevision::new("r1").unwrap(),
        ModelDigest::sha256(b"resident modernbert test"),
    );
    manifest.architecture = ModelArchitecture::new("modernbert", "resident-modernbert-test");
    manifest.artifact_format = ArtifactFormat::HuggingFace;
    architecture.hidden_size = 8;
    architecture.intermediate_size = 12;
    architecture.num_hidden_layers = 3;
    architecture.num_attention_heads = 2;
    architecture.num_key_value_heads = 2;
    architecture.head_dim = 4;
    architecture.vocab_size = 24;
    architecture.hidden_activation = Some("gelu".into());
    architecture.attention_bias = false;
    manifest.architecture_config = Some(architecture);
    let encoder = EncoderGraphConfig {
        norm_epsilon: 1e-5,
        max_positions: 64,
        norm_bias: false,
        attention_bias: false,
        mlp_bias: false,
        layers: (0..3)
            .map(|index| EncoderLayerConfig {
                local_radius: (index == 1).then_some(2),
                rope_theta: if index == 1 { 10_000.0 } else { 160_000.0 },
                attention_normalization: index != 0,
            })
            .collect(),
    };
    let mut bytes_by_name = BTreeMap::new();
    manifest.tensors = bindings
        .into_iter()
        .filter_map(|(edge, tensor)| {
            let name = edge.as_str().strip_prefix("input.").unwrap();
            if matches!(name, "token_ids" | "mask") {
                return None;
            }
            let bytes = tensor.content_bytes();
            let metadata = ModelTensorMetadata {
                name: name.into(),
                shape: tensor.shape,
                storage_dtype: ModelDType::F32,
                layout: None,
                shard: None,
                offset_bytes: Some(0),
                size_bytes: Some(bytes.len() as u64),
                quantization: None,
                expected_compute_dtype: None,
                digest: Some(ModelDigest::sha256(&bytes)),
            };
            bytes_by_name.insert(name.into(), bytes);
            Some(metadata)
        })
        .collect();
    let provider = Arc::new(ReferenceCpuProvider::new());
    let executor = provider.executor();
    let mut loaded = ProductionLoadedEncoder::load(
        manifest.clone(),
        &ProductionIntegrationPayloadSource { bytes_by_name },
        ModelTrustStore::default().trust_digest(&manifest.id.digest.value),
        provider,
        component,
        encoder,
    )
    .unwrap();
    let mask: Vec<bool> = case.mask.iter().map(|&value| value != 0).collect();
    let stored = executor.stored_tensor_count();
    let tracked = executor.tracked_allocations().len();
    for _ in 0..3 {
        let output = loaded.encode(&case.ids, &mask).unwrap();
        for (&actual, &expected) in output.data.iter().zip(&case.output) {
            assert!((actual - expected).abs() <= 3e-6 + 1e-6 * expected.abs());
        }
        loaded
            .runtime
            .memory()
            .assert_allocation_totals_consistent();
        assert_eq!(executor.stored_tensor_count(), stored);
        assert_eq!(executor.tracked_allocations().len(), tracked);
    }

    // Removing a late weight forces a failure after intermediate allocations.
    let weight = loaded
        .runtime
        .model_instances()
        .instance(&loaded.instance)
        .unwrap()
        .definition()
        .resource_bindings
        .weights["layers.0.mlp.Wo.weight"]
        .clone();
    let saved = executor.read_tensor(&weight).unwrap();
    assert!(executor.release_tensor(&weight));
    let stored_after_removal = executor.stored_tensor_count();
    assert!(loaded.encode(&case.ids, &mask).is_err());
    assert_eq!(executor.stored_tensor_count(), stored_after_removal);
    assert_eq!(executor.tracked_allocations().len(), tracked);
    loaded
        .runtime
        .memory()
        .assert_allocation_totals_consistent();
    executor.write_tensor(weight, saved);
    loaded.encode(&case.ids, &mask).unwrap();
    loaded.unload().unwrap();
    assert_eq!(executor.stored_tensor_count(), 0);
    assert_eq!(executor.tracked_allocations().len(), 0);
}

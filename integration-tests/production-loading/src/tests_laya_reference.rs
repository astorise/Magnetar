//! Synthetic ModernBERT -> Laya-head comparison with pinned PyTorch classes.
//! The numerical tests read synthetic Safetensors, not published checkpoint
//! weights. They compare both a reference encoder and the Component graph
//! executor, followed by the reference decision head, to the upstream oracle.

use magnetar_component_laya::{
    Answer, Calibrator, QuestionType,
    head::{CheckpointTensor, DecisionHead, checkpoint_tensor_shapes},
};
use magnetar_component_modernbert::{
    checkpoint::{
        CheckpointNamespace, CheckpointProfile, DecodedTensor, StorageDType, TensorDescriptor,
    },
    encoder_reference::Encoder,
};
use magnetar_format_safetensors::{TensorToSerialize, serialize};
use magnetar_loader_huggingface::discover_raw_safetensors;
use magnetar_runtime::{
    DecisionPrediction, DecisionResult, DecisionScore, HostTensor, ModelArtifactSource, ModelDType,
    ModelDigest, ProductionArtifactPayloadSource, ProductionModelSource, ProductionPayloadRange,
    host_tensors_from_artifact_bytes,
};
use std::collections::BTreeMap;

fn pattern(name: &str, count: usize) -> Vec<f32> {
    let seed = name
        .bytes()
        .enumerate()
        .map(|(index, byte)| (index + 1) * byte as usize)
        .sum::<usize>()
        % 97;
    (0..count)
        .map(|index| {
            let mut value = ((index * 17 + seed) % 101) as f32 - 50.0;
            value /= 200.0;
            if name.contains("norm") && name.ends_with(".weight") {
                value += 1.0;
            }
            value
        })
        .collect()
}

fn models_from_synthetic_safetensors() -> (
    Encoder,
    DecisionHead,
    CheckpointProfile,
    BTreeMap<String, HostTensor>,
) {
    let config = serde_json::json!({
        "model_type": "modernbert", "hidden_size": 8, "intermediate_size": 12,
        "vocab_size": 24, "num_attention_heads": 2, "num_hidden_layers": 3,
        "max_position_embeddings": 64, "norm_eps": 1e-5,
        "norm_bias": false, "attention_bias": false, "mlp_bias": false,
        "hidden_activation": "gelu", "local_attention": 4,
        "layer_types": ["full_attention", "sliding_attention", "full_attention"]
    });
    let profile = CheckpointProfile::from_huggingface_config(
        &serde_json::to_vec(&config).unwrap(),
        CheckpointNamespace::Encoder,
    )
    .unwrap();
    // The Python oracle's wrapper adds another `encoder.` level to parameter
    // names when generating its synthetic values; it is not part of the
    // actual Laya checkpoint namespace.
    let mut weights: Vec<_> = profile
        .weights()
        .iter()
        .map(|spec| {
            let values = pattern(
                &format!("encoder.{}", spec.checkpoint_name),
                spec.checkpoint_shape.iter().product(),
            );
            (
                spec.checkpoint_name.clone(),
                spec.checkpoint_shape.clone(),
                values,
            )
        })
        .collect();
    weights.extend(
        checkpoint_tensor_shapes(8, 2, 2)
            .unwrap()
            .into_iter()
            .map(|(name, shape)| {
                let values = pattern(&name, shape.iter().product());
                (name, shape, values)
            }),
    );
    let bytes: Vec<Vec<u8>> = weights
        .iter()
        .map(|(_, _, values)| {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        })
        .collect();
    let descriptors: Vec<_> = weights
        .iter()
        .zip(&bytes)
        .map(|((name, shape, _), bytes)| TensorToSerialize {
            name,
            dtype: ModelDType::F32,
            shape: shape.iter().map(|&size| size as u64).collect(),
            data: bytes,
        })
        .collect();
    let file = serialize(&descriptors, &BTreeMap::new()).unwrap();
    let expected_digest = ModelDigest::sha256(&file);
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("model.safetensors"), file).unwrap();
    let source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::LocalPath(directory.path().to_path_buf()),
        directory.path().to_path_buf(),
    );
    let raw = discover_raw_safetensors(&source).unwrap();
    assert_eq!(raw.single_file_digest.as_ref(), Some(&expected_digest));
    let inventory = raw
        .tensors
        .iter()
        .map(|tensor| {
            (
                tensor.name.clone(),
                TensorDescriptor {
                    shape: tensor.shape.iter().map(|&size| size as usize).collect(),
                    dtype: match tensor.storage_dtype {
                        ModelDType::F16 => StorageDType::F16,
                        ModelDType::Bf16 => StorageDType::BF16,
                        ModelDType::F32 => StorageDType::F32,
                        _ => StorageDType::Unsupported,
                    },
                },
            )
        })
        .collect();
    profile.validate_inventory(&inventory).unwrap();
    let mut encoder_tensors = BTreeMap::new();
    let mut head_tensors = BTreeMap::new();
    for metadata in raw.tensors {
        let range = ProductionPayloadRange {
            identity: metadata.name.clone(),
            offset: metadata.offset_bytes.unwrap(),
            length: metadata.size_bytes.unwrap(),
            digest: metadata.digest.clone(),
        };
        let bytes = raw.payloads.read_payload(&range).unwrap();
        let mut bounded = metadata.clone();
        bounded.offset_bytes = Some(0);
        bounded.shard = None;
        let tensor = host_tensors_from_artifact_bytes(&[bounded], &bytes, 0)
            .unwrap()
            .remove(&metadata.name)
            .unwrap();
        let shape = tensor.shape.iter().map(|&size| size as usize).collect();
        if metadata.name.starts_with("encoder.") {
            encoder_tensors.insert(
                metadata.name,
                DecodedTensor {
                    shape,
                    values: tensor.data,
                },
            );
        } else {
            head_tensors.insert(
                metadata.name,
                CheckpointTensor {
                    shape,
                    values: tensor.data,
                },
            );
        }
    }
    let graph_weights = profile
        .weights()
        .iter()
        .map(|spec| {
            let values = spec
                .graph_values(&encoder_tensors[&spec.checkpoint_name])
                .unwrap();
            let shape = spec
                .graph_shape
                .iter()
                .map(|&size| size as u64)
                .collect::<Vec<_>>();
            (
                spec.graph_name.clone(),
                HostTensor::new(shape, values).unwrap(),
            )
        })
        .collect();
    (
        profile.clone().into_reference(encoder_tensors).unwrap(),
        DecisionHead::from_checkpoint_tensors(8, 2, 2, head_tensors).unwrap(),
        profile,
        graph_weights,
    )
}

pub(crate) fn encoder_graph_request(
    profile: &CheckpointProfile,
    sequence_length: u64,
) -> (
    magnetar_runtime::ComponentDigest,
    magnetar_runtime::FirstNativeEncoderGraphRequest,
) {
    use magnetar_runtime::{
        ComponentDigest, ComponentTrustStore, EncoderGraphConfig, EncoderLayerConfig,
        FirstNativeEncoderGraphRequest, ModelArchitectureConfig, ModelComponentId,
        ModelComponentIdentity, ModelComponentImplementationKind, ModelComponentVersion,
        register_inference_component_artifact,
    };

    let settings = profile.graph_settings();
    let bytes = include_bytes!(
        "../../../magnetar-runtime/fixtures/components/modernbert-encoder.component.wasm"
    )
    .to_vec();
    let digest = ComponentDigest::sha256(&bytes);
    let manifest = include_str!(
        "../../../magnetar-runtime/fixtures/components/qwen-real.component.wasm.magnetar-component.yaml"
    )
    .replace(
        "sha256:321b42b83773544e1ba553cc7196d3d742cd9ef1e2af3f04da72b66e402a730f",
        &digest.value,
    )
    .replace("magnetar.qwen.real", "magnetar.modernbert.laya-profile")
    .replace("qwen-model-component", "modernbert-model-component")
    .replace("    - qwen2\r\n", "    - modernbert\r\n")
    .replace("    - qwen2\n", "    - modernbert\n")
    .replace("1.14.0", "1.12.0");
    let trust = ComponentTrustStore::default().trust_digest(&digest.value);
    register_inference_component_artifact(bytes, manifest.into_bytes(), &trust).unwrap();

    let architecture = ModelArchitectureConfig {
        hidden_size: settings.hidden_size as u64,
        intermediate_size: settings.intermediate_size as u64,
        num_hidden_layers: settings.layers.len() as u32,
        num_attention_heads: settings.attention_heads as u32,
        num_key_value_heads: settings.attention_heads as u32,
        head_dim: (settings.hidden_size / settings.attention_heads) as u64,
        vocab_size: settings.vocabulary_size as u64,
        rms_norm_eps: settings.norm_epsilon as f32,
        rope_theta: settings.layers[0].rope_theta,
        hidden_activation: Some("gelu".into()),
        attention_bias: settings.attention_bias,
        ..Default::default()
    };
    let encoder = EncoderGraphConfig {
        norm_epsilon: settings.norm_epsilon,
        max_positions: settings.max_positions as u64,
        norm_bias: settings.norm_bias,
        attention_bias: settings.attention_bias,
        mlp_bias: settings.mlp_bias,
        layers: settings
            .layers
            .iter()
            .map(|layer| EncoderLayerConfig {
                local_radius: layer.local_radius.map(|radius| radius.try_into().unwrap()),
                rope_theta: layer.rope_theta,
                attention_normalization: layer.attention_normalization,
            })
            .collect(),
    };
    let weight_shapes = profile
        .weights()
        .iter()
        .map(|weight| {
            (
                weight.graph_name.clone(),
                weight
                    .graph_shape
                    .iter()
                    .map(|&dimension| dimension as u64)
                    .collect(),
            )
        })
        .collect();
    let request = FirstNativeEncoderGraphRequest {
        architecture,
        encoder,
        weight_shapes,
        identity: ModelComponentIdentity::new(
            ModelComponentId::new("modernbert-laya-profile").unwrap(),
            ModelComponentVersion::new(1, 0, 0),
            ModelComponentImplementationKind::WebAssemblyComponent,
        )
        .trusted()
        .with_architecture_family("modernbert"),
        sequence_length,
    };
    (digest, request)
}

#[test]
fn laya_checkpoint_profile_builds_a_real_modernbert_component_graph() {
    use magnetar_runtime::{
        build_first_native_encoder_graph_from_named_component, initial_operator_catalog,
    };

    let profile = CheckpointProfile::from_huggingface_config(
        include_bytes!("../../../components/modernbert/tests/fixtures/laya-encoder.config.json"),
        CheckpointNamespace::Encoder,
    )
    .unwrap();
    let settings = profile.graph_settings();
    let (digest, request) = encoder_graph_request(&profile, 5);
    let graph = build_first_native_encoder_graph_from_named_component(&digest, &request).unwrap();
    graph.validate(&initial_operator_catalog()).unwrap();
    assert_eq!(graph.phase, magnetar_runtime::ExecutionGraphPhase::Encode);
    assert!(graph.nodes.len() > settings.layers.len());

    let mut malformed = request;
    malformed
        .weight_shapes
        .get_mut("layers.0.attn.Wqkv.weight")
        .unwrap()
        .swap(0, 1);
    assert!(build_first_native_encoder_graph_from_named_component(&digest, &malformed).is_err());
}

#[test]
fn raw_half_precision_checkpoint_payloads_decode_to_f32() {
    let f16 = [0x00, 0x3c, 0x00, 0xc0];
    let bf16 = [0x80, 0x3f, 0x00, 0xc0];
    let file = serialize(
        &[
            TensorToSerialize {
                name: "encoder.embeddings.norm.weight",
                dtype: ModelDType::F16,
                shape: vec![2],
                data: &f16,
            },
            TensorToSerialize {
                name: "head.layers.0.norm1.weight",
                dtype: ModelDType::Bf16,
                shape: vec![2],
                data: &bf16,
            },
        ],
        &BTreeMap::new(),
    )
    .unwrap();
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("model.safetensors"), file).unwrap();
    let source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::LocalPath(directory.path().to_path_buf()),
        directory.path().to_path_buf(),
    );
    let raw = discover_raw_safetensors(&source).unwrap();
    for metadata in raw.tensors {
        let range = ProductionPayloadRange {
            identity: metadata.name.clone(),
            offset: metadata.offset_bytes.unwrap(),
            length: metadata.size_bytes.unwrap(),
            digest: metadata.digest.clone(),
        };
        let bytes = raw.payloads.read_payload(&range).unwrap();
        let mut bounded = metadata.clone();
        bounded.offset_bytes = Some(0);
        let tensor = host_tensors_from_artifact_bytes(&[bounded], &bytes, 0)
            .unwrap()
            .remove(&metadata.name)
            .unwrap();
        assert_eq!(tensor.shape, [2]);
        assert_eq!(tensor.data, [1.0, -2.0]);
    }
}

fn numbers(value: &serde_json::Value, key: &str) -> Vec<f32> {
    value[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item.as_f64().unwrap() as f32)
        .collect()
}

fn integers(value: &serde_json::Value, key: &str) -> Vec<usize> {
    value[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item.as_u64().unwrap() as usize)
        .collect()
}

fn close(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() <= 2e-6 + 1e-6 * b.abs(),
            "{what}[{index}]: {a} != {b}"
        );
    }
}

#[derive(Clone)]
struct EncoderPayloads(BTreeMap<String, Vec<u8>>);

impl ProductionArtifactPayloadSource for EncoderPayloads {
    fn read_payload(
        &self,
        range: &ProductionPayloadRange,
    ) -> Result<Vec<u8>, magnetar_runtime::ProductionIngestionError> {
        let missing = || magnetar_runtime::ProductionIngestionError::PayloadUnavailable {
            identity: range.identity.clone(),
        };
        let bytes = self.0.get(&range.identity).ok_or_else(missing)?;
        if range.offset != 0 || range.length != bytes.len() as u64 {
            return Err(
                magnetar_runtime::ProductionIngestionError::PayloadOutOfBounds {
                    identity: range.identity.clone(),
                },
            );
        }
        if range
            .digest
            .as_ref()
            .is_some_and(|digest| *digest != ModelDigest::sha256(bytes))
        {
            return Err(
                magnetar_runtime::ProductionIngestionError::IntegrityMismatch {
                    identity: range.identity.clone(),
                },
            );
        }
        Ok(bytes.clone())
    }
}

#[test]
fn resident_encoder_loads_authenticated_weights_and_matches_laya_reference() {
    use magnetar_runtime::{
        ArtifactFormat, ModelArchitecture, ModelArtifactId, ModelArtifactKind, ModelName,
        ModelRevision, ModelTensorMetadata, ModelTrustStore, ProductionLoadedEncoder,
    };
    let (encoder, head, profile, weights) = models_from_synthetic_safetensors();
    let (component, request) = encoder_graph_request(&profile, 1);
    let mut manifest = magnetar_runtime::e2e_fixture().unwrap().manifest;
    manifest.id = ModelArtifactId::new(
        ModelArtifactKind::ModelBundle,
        ModelName::new("synthetic-modernbert").unwrap(),
        ModelRevision::new("r1").unwrap(),
        ModelDigest::sha256(b"synthetic modernbert artifact"),
    );
    manifest.architecture = ModelArchitecture::new("modernbert", "synthetic-modernbert");
    manifest.architecture_config = Some(request.architecture.clone());
    manifest.artifact_format = ArtifactFormat::HuggingFace;
    let mut payloads = BTreeMap::new();
    manifest.tensors = weights
        .into_iter()
        .map(|(name, tensor)| {
            let bytes = tensor.content_bytes();
            let metadata = ModelTensorMetadata {
                name: name.clone(),
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
            payloads.insert(name, bytes);
            metadata
        })
        .collect();
    let payloads = EncoderPayloads(payloads);
    let provider = std::sync::Arc::new(magnetar_runtime::ReferenceCpuProvider::new());
    let load = |payloads: &EncoderPayloads, trust| {
        ProductionLoadedEncoder::load(
            manifest.clone(),
            payloads,
            trust,
            provider.clone(),
            component.clone(),
            request.encoder.clone(),
        )
    };
    assert!(load(&payloads, ModelTrustStore::default()).is_err());
    let mut corrupted = payloads.clone();
    corrupted.0.values_mut().next().unwrap()[0] ^= 1;
    let trusted = || ModelTrustStore::default().trust_digest(manifest.id.digest.value.clone());
    assert!(load(&corrupted, trusted()).is_err());
    let mut loaded = load(&payloads, trusted()).unwrap();
    assert!(loaded.encode(&[], &[]).is_err());
    assert!(loaded.encode(&[1, 2], &[true]).is_err());
    assert!(loaded.encode(&[1, 2], &[false, false]).is_err());
    assert!(loaded.encode(&[24], &[true]).is_err());
    let cases: Vec<serde_json::Value> =
        serde_json::from_str(include_str!("../fixtures/laya-reference.json")).unwrap();
    for case in cases {
        let ids = integers(&case, "ids")
            .into_iter()
            .map(|id| id as u32)
            .collect::<Vec<_>>();
        let mask = integers(&case, "mask")
            .into_iter()
            .map(|id| id != 0)
            .collect::<Vec<_>>();
        let markers = integers(&case, "markers");
        let qtype = match case["qtype"].as_u64().unwrap() {
            0 => QuestionType::Choice,
            1 => QuestionType::Score,
            2 => QuestionType::Noul,
            _ => panic!("unknown question type"),
        };
        let hidden = loaded.encode(&ids, &mask).unwrap();
        let expected_hidden = encoder.forward(&ids, &mask, None).unwrap();
        close(&hidden.data, &expected_hidden, "resident encoder hidden");
        let output = head
            .forward(&hidden.data, ids.len(), &mask, &markers, qtype)
            .unwrap();
        close(
            &output.option_logits,
            &numbers(&case, "logits"),
            "resident option logits",
        );
        close(
            &output.action_logits,
            &numbers(&case, "action_logits"),
            "resident action logits",
        );
    }
    loaded.unload().unwrap();
}

#[test]
fn laya_encoder_and_head_match_pinned_upstream_probabilities() {
    let cases: Vec<serde_json::Value> =
        serde_json::from_str(include_str!("../fixtures/laya-reference.json")).unwrap();
    let calibrator = Calibrator::from_config(include_bytes!(
        "../../../components/laya/tests/fixtures/rl_agent_config.json"
    ))
    .unwrap();
    let (encoder, head, profile, graph_weights) = models_from_synthetic_safetensors();
    for case in cases {
        let ids = integers(&case, "ids")
            .into_iter()
            .map(|value| value as u32)
            .collect::<Vec<_>>();
        let mask = integers(&case, "mask")
            .into_iter()
            .map(|value| value != 0)
            .collect::<Vec<_>>();
        let markers = integers(&case, "markers");
        let qtype = match case["qtype"].as_u64().unwrap() {
            0 => QuestionType::Choice,
            1 => QuestionType::Score,
            2 => QuestionType::Noul,
            _ => panic!("unknown question type"),
        };
        let hidden = encoder.forward(&ids, &mask, None).unwrap();
        let (digest, request) = encoder_graph_request(&profile, ids.len() as u64);
        let graph = magnetar_runtime::build_first_native_encoder_graph_from_named_component(
            &digest, &request,
        )
        .unwrap();
        let mut bindings: BTreeMap<_, _> = graph_weights
            .iter()
            .map(|(name, tensor)| {
                (
                    magnetar_runtime::TensorEdgeId::new(format!("weight.{name}")),
                    tensor.clone(),
                )
            })
            .collect();
        bindings.insert(
            magnetar_runtime::TensorEdgeId::new("input.token_ids"),
            HostTensor::new(
                [ids.len() as u64],
                ids.iter().map(|&id| id as f32).collect::<Vec<_>>(),
            )
            .unwrap(),
        );
        bindings.insert(
            magnetar_runtime::TensorEdgeId::new("input.attention_mask"),
            HostTensor::new(
                [mask.len() as u64],
                mask.iter()
                    .map(|&valid| if valid { 1.0 } else { 0.0 })
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
        );
        let outcome = magnetar_runtime::run_first_native_graph_with_provider_and_bindings(
            std::sync::Arc::new(magnetar_runtime::ReferenceCpuProvider::new()),
            &magnetar_runtime::e2e_fixture().unwrap(),
            &graph,
            &magnetar_runtime::KvCacheId::new("laya-encoder-conformance").unwrap(),
            bindings,
            &["hidden_states"],
        )
        .unwrap();
        let graph_hidden =
            &outcome.bindings[&magnetar_runtime::TensorEdgeId::new("hidden_states")].data;
        close(graph_hidden, &hidden, "encoder hidden states");
        let result = head
            .forward(graph_hidden, ids.len(), &mask, &markers, qtype)
            .unwrap();
        close(
            &result.option_logits,
            &numbers(&case, "logits"),
            "option logits",
        );
        close(
            &result.action_logits,
            &numbers(&case, "action_logits"),
            "action logits",
        );
        let decision = result.decide(&calibrator, qtype).unwrap();
        close(
            &decision
                .decision
                .probabilities
                .iter()
                .map(|&value| value as f32)
                .collect::<Vec<_>>(),
            &numbers(&case, "probabilities"),
            "probabilities",
        );
        let action = decision.action_probabilities[0];
        close(
            &[action as f32],
            &[case["action_probability"].as_f64().unwrap() as f32],
            "action probability",
        );
        assert_eq!(
            decision.option_scores.len(),
            decision.decision.probabilities.len()
        );
        assert_eq!(
            decision.action_scores.len(),
            decision.action_probabilities.len()
        );
        let output = DecisionResult {
            options: decision
                .option_scores
                .iter()
                .zip(&decision.decision.probabilities)
                .map(|(&score, &probability)| DecisionScore {
                    score: score.into(),
                    probability,
                })
                .collect(),
            prediction: match decision.decision.answer {
                Answer::Choice(index) => {
                    DecisionPrediction::SelectedOption(index.try_into().unwrap())
                }
                Answer::Score(value) | Answer::Noul(value) => {
                    DecisionPrediction::NumericValue(value)
                }
            },
            confidence: decision.decision.confidence,
            actions: Some(
                decision
                    .action_scores
                    .iter()
                    .zip(&decision.action_probabilities)
                    .map(|(&score, &probability)| DecisionScore {
                        score: score.into(),
                        probability,
                    })
                    .collect(),
            ),
        };
        output.validate().unwrap();
        assert_eq!(output.options.len(), markers.len());
    }
}

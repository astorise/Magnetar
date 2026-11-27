//! Synthetic ModernBERT -> Laya-head comparison with pinned PyTorch classes.
//! This test does not use checkpoint weights or the Magnetar runtime graph.

use magnetar_component_laya::{
    Answer, Calibrator, QuestionType,
    head::{CheckpointTensor, DecisionHead, checkpoint_tensor_shapes},
};
use magnetar_component_modernbert::{
    checkpoint::{CheckpointNamespace, CheckpointProfile, DecodedTensor},
    encoder_reference::Encoder,
};
use magnetar_runtime::{DecisionPrediction, DecisionResult, DecisionScore};
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

fn encoder() -> Encoder {
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
    let tensors: BTreeMap<_, _> = profile
        .weights()
        .iter()
        .map(|spec| {
            (
                spec.checkpoint_name.clone(),
                DecodedTensor {
                    shape: spec.checkpoint_shape.clone(),
                    values: pattern(
                        &format!("encoder.{}", spec.checkpoint_name),
                        spec.checkpoint_shape.iter().product(),
                    ),
                },
            )
        })
        .collect();
    profile.into_reference(tensors).unwrap()
}

#[test]
fn laya_checkpoint_profile_builds_a_real_modernbert_component_graph() {
    use magnetar_runtime::{
        ComponentDigest, ComponentTrustStore, EncoderGraphConfig, EncoderLayerConfig,
        FirstNativeEncoderGraphRequest, ModelArchitectureConfig, ModelComponentId,
        ModelComponentIdentity, ModelComponentImplementationKind, ModelComponentVersion,
        build_first_native_encoder_graph_from_named_component, initial_operator_catalog,
        register_inference_component_artifact,
    };

    let profile = CheckpointProfile::from_huggingface_config(
        include_bytes!("../../../components/modernbert/tests/fixtures/laya-encoder.config.json"),
        CheckpointNamespace::Encoder,
    )
    .unwrap();
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
        sequence_length: 5,
    };
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

fn head() -> DecisionHead {
    let tensors = checkpoint_tensor_shapes(8, 2, 2)
        .unwrap()
        .into_iter()
        .map(|(name, shape)| {
            let values = pattern(&name, shape.iter().product());
            (name, CheckpointTensor { shape, values })
        })
        .collect();
    DecisionHead::from_checkpoint_tensors(8, 2, 2, tensors).unwrap()
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

#[test]
fn laya_encoder_and_head_match_pinned_upstream_probabilities() {
    let cases: Vec<serde_json::Value> =
        serde_json::from_str(include_str!("../fixtures/laya-reference.json")).unwrap();
    let calibrator = Calibrator::from_config(include_bytes!(
        "../../../components/laya/tests/fixtures/rl_agent_config.json"
    ))
    .unwrap();
    let encoder = encoder();
    let head = head();
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
        let result = head
            .forward(&hidden, ids.len(), &mask, &markers, qtype)
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

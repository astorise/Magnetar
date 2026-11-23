//! Synthetic ModernBERT -> Laya-head comparison with pinned PyTorch classes.
//! This test does not use checkpoint weights or the Magnetar runtime graph.

use magnetar_component_laya::{
    Answer, Calibrator, QuestionType,
    head::{DecisionHead, LayerNorm as HeadNorm, Linear as HeadLinear, TransformerLayer},
};
use magnetar_component_modernbert::{
    AttentionKind,
    encoder_reference::{Encoder, EncoderLayer, LayerNorm as EncoderNorm, Linear as EncoderLinear},
};
use magnetar_runtime::{DecisionPrediction, DecisionResult, DecisionScore};

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

fn encoder_linear(name: &str, input: usize, output: usize) -> EncoderLinear {
    EncoderLinear {
        weight: pattern(&format!("{name}.weight"), input * output),
        bias: None,
    }
}

fn encoder_norm(name: &str, hidden: usize) -> EncoderNorm {
    EncoderNorm {
        weight: pattern(&format!("{name}.weight"), hidden),
        bias: None,
    }
}

fn encoder() -> Encoder {
    let hidden = 8;
    let intermediate = 12;
    let prefix = "encoder.encoder";
    let layers = (0..3)
        .map(|index| {
            let prefix = format!("{prefix}.layers.{index}");
            let local = index == 1;
            EncoderLayer {
                attention: if local {
                    AttentionKind::Local
                } else {
                    AttentionKind::Global
                },
                rope_theta: if local { 10_000.0 } else { 160_000.0 },
                attention_norm: (index != 0)
                    .then(|| encoder_norm(&format!("{prefix}.attn_norm"), hidden)),
                qkv: encoder_linear(&format!("{prefix}.attn.Wqkv"), hidden, 3 * hidden),
                attention_output: encoder_linear(&format!("{prefix}.attn.Wo"), hidden, hidden),
                mlp_norm: encoder_norm(&format!("{prefix}.mlp_norm"), hidden),
                mlp_in: encoder_linear(&format!("{prefix}.mlp.Wi"), hidden, 2 * intermediate),
                mlp_out: encoder_linear(&format!("{prefix}.mlp.Wo"), intermediate, hidden),
            }
        })
        .collect();
    Encoder {
        vocab_size: 24,
        hidden,
        intermediate,
        heads: 2,
        max_positions: 64,
        local_radius: 2,
        epsilon: 1e-5,
        token_embedding: pattern(
            &format!("{prefix}.embeddings.tok_embeddings.weight"),
            24 * hidden,
        ),
        embedding_norm: encoder_norm(&format!("{prefix}.embeddings.norm"), hidden),
        layers,
        final_norm: encoder_norm(&format!("{prefix}.final_norm"), hidden),
    }
}

fn head_linear(name: &str, input: usize, output: usize) -> HeadLinear {
    HeadLinear {
        weight: pattern(&format!("{name}.weight"), input * output),
        bias: pattern(&format!("{name}.bias"), output),
    }
}

fn head_norm(name: &str, hidden: usize) -> HeadNorm {
    HeadNorm {
        weight: pattern(&format!("{name}.weight"), hidden),
        bias: pattern(&format!("{name}.bias"), hidden),
    }
}

fn head() -> DecisionHead {
    let d = 8;
    let layers = (0..2)
        .map(|index| {
            let prefix = format!("head.layers.{index}");
            TransformerLayer {
                norm1: head_norm(&format!("{prefix}.norm1"), d),
                norm2: head_norm(&format!("{prefix}.norm2"), d),
                qkv: HeadLinear {
                    weight: pattern(&format!("{prefix}.self_attn.in_proj_weight"), 3 * d * d),
                    bias: pattern(&format!("{prefix}.self_attn.in_proj_bias"), 3 * d),
                },
                attention_output: head_linear(&format!("{prefix}.self_attn.out_proj"), d, d),
                feed_forward_in: head_linear(&format!("{prefix}.linear1"), d, 4 * d),
                feed_forward_out: head_linear(&format!("{prefix}.linear2"), 4 * d, d),
            }
        })
        .collect();
    DecisionHead {
        hidden: d,
        type_embedding: pattern("type_emb.weight", 3 * d),
        layers,
        scorer_norm: head_norm("scorer.0", d),
        scorer_in: head_linear("scorer.1", d, d),
        scorer_out: head_linear("scorer.3", d, 1),
        action_in: head_linear("act_head.0", d + 4, 256),
        action_out: head_linear("act_head.2", 256, 2),
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

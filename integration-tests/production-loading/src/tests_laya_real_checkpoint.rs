//! Optional numerical comparison against the complete pinned Laya checkpoint.
//!
//! The encoder Component graph uses explicit test weight bindings; the Rust
//! reference head consumes its output. This is not the production inference API.

use std::{collections::BTreeMap, path::PathBuf, sync::Arc};

use magnetar_component_laya::{
    Calibrator, QuestionType,
    head::{CheckpointTensor, DecisionHead, checkpoint_tensor_shapes},
};
use magnetar_component_modernbert::checkpoint::{
    CheckpointNamespace, CheckpointProfile, DecodedTensor, StorageDType, TensorDescriptor,
};
use magnetar_loader_huggingface::inspect_safetensors_file;
use magnetar_provider_cpu::native::{NativeCpuExecutor, NativeCpuProvider};
use magnetar_runtime::{
    HostTensor, KvCacheId, ModelArtifactSource, ModelDType, ModelDigest,
    ProductionArtifactPayloadSource, ProductionModelSource, ProductionPayloadRange, TensorEdgeId,
    build_first_native_encoder_graph_from_named_component, e2e_fixture,
    host_tensors_from_artifact_bytes, run_first_native_graph_with_provider_and_bindings,
};

fn numbers(value: &serde_json::Value, key: &str) -> Vec<f32> {
    value[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_f64().unwrap() as f32)
        .collect()
}

fn integers(value: &serde_json::Value, key: &str) -> Vec<usize> {
    value[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_u64().unwrap() as usize)
        .collect()
}

fn close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    let mut max_error = 0.0_f32;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        max_error = max_error.max(error);
        assert!(
            error <= 5e-5 + 1e-5 * expected.abs(),
            "{label}[{index}]: {actual} != {expected} (error {error})"
        );
    }
    eprintln!("{label} maximum absolute error: {max_error}");
}

#[test]
#[ignore = "requires the pinned 843 MB Laya checkpoint and sufficient host memory"]
fn resident_laya_checkpoint_decisions_match_pytorch() {
    use magnetar_component_laya::{ChoiceOption, Question};
    use magnetar_inference_component::laya::LoadedLaya;
    use magnetar_loader_huggingface::HuggingFaceTokenizer;
    use magnetar_runtime::ModelTrustStore;

    let path = PathBuf::from(
        std::env::var_os("MAGNETAR_LAYA_REAL_CHECKPOINT")
            .expect("set MAGNETAR_LAYA_REAL_CHECKPOINT to the pinned model.safetensors"),
    );
    let encoder_config =
        include_bytes!("../../../components/modernbert/tests/fixtures/laya-encoder.config.json");
    let decision_config =
        include_bytes!("../../../components/laya/tests/fixtures/rl_agent_config.json");
    let profile =
        CheckpointProfile::from_huggingface_config(encoder_config, CheckpointNamespace::Encoder)
            .unwrap();
    let (component, _) = super::tests_laya_reference::encoder_graph_request(&profile, 1);
    let provider = Arc::new(NativeCpuProvider::with_executor(
        NativeCpuExecutor::with_settings(true, false, 4),
    ));
    let mut model = LoadedLaya::load_pinned(
        &path,
        encoder_config,
        decision_config,
        component,
        ModelTrustStore::default().trust_digest(
            "sha256:891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c",
        ),
        provider,
    )
    .unwrap();
    let cases: Vec<serde_json::Value> = serde_json::from_str(include_str!(
        "../fixtures/laya-real-checkpoint-reference.json"
    ))
    .unwrap();
    for case in cases {
        let ids: Vec<u32> = integers(&case, "ids")
            .into_iter()
            .map(|n| n as u32)
            .collect();
        let mask: Vec<bool> = integers(&case, "mask")
            .into_iter()
            .map(|n| n != 0)
            .collect();
        let markers = integers(&case, "markers");
        let kind = match case["qtype"].as_u64().unwrap() {
            0 => QuestionType::Choice,
            1 => QuestionType::Score,
            _ => panic!("unsupported reference question type"),
        };
        assert!(model.decide_tokenized(&ids, &mask, &[0], kind).is_err());
        let actual = model.decide_tokenized(&ids, &mask, &markers, kind).unwrap();
        close(
            &actual
                .options
                .iter()
                .map(|item| item.score as f32)
                .collect::<Vec<_>>(),
            &numbers(&case, "logits"),
            "resident option logits",
        );
        close(
            &actual
                .options
                .iter()
                .map(|item| item.probability as f32)
                .collect::<Vec<_>>(),
            &numbers(&case, "probabilities"),
            "resident option probabilities",
        );
        let actions = actual.actions.unwrap();
        close(
            &actions
                .iter()
                .map(|item| item.score as f32)
                .collect::<Vec<_>>(),
            &numbers(&case, "action_logits"),
            "resident action logits",
        );
        close(
            &actions
                .iter()
                .map(|item| item.probability as f32)
                .collect::<Vec<_>>(),
            &numbers(&case, "action_probabilities"),
            "resident action probabilities",
        );
    }
    let tokenizer_path = PathBuf::from(
        std::env::var_os("MAGNETAR_LAYA_TOKENIZER")
            .expect("set MAGNETAR_LAYA_TOKENIZER to the pinned tokenizer.json"),
    );
    let tokenizer = HuggingFaceTokenizer::from_bytes(
        &std::fs::read(tokenizer_path).unwrap(),
        None,
        "laya-tokenizer",
        Some(50_368),
    )
    .unwrap();
    let text_cases: Vec<serde_json::Value> =
        serde_json::from_str(include_str!("../fixtures/laya-text-reference.json")).unwrap();
    for case in text_cases {
        let question = &case["question"];
        let instructions = question["ins"].as_str().unwrap();
        let state = case["state"].as_str().unwrap();
        let (actual, kind) = match question["t"].as_str().unwrap() {
            "choice" => {
                let choices: Vec<_> = case["option_order"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|label| {
                        let label = label.as_str().unwrap();
                        ChoiceOption {
                            label,
                            description: question["crit"][label].as_str(),
                        }
                    })
                    .collect();
                (
                    model
                        .decide_text(
                            &tokenizer,
                            state,
                            &Question::Choice {
                                instructions,
                                options: &choices,
                            },
                        )
                        .unwrap(),
                    QuestionType::Choice,
                )
            }
            "score" => {
                let levels: Vec<_> = question["crit"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_str().unwrap())
                    .collect();
                (
                    model
                        .decide_text(
                            &tokenizer,
                            state,
                            &Question::Score {
                                instructions,
                                levels: &levels,
                            },
                        )
                        .unwrap(),
                    QuestionType::Score,
                )
            }
            "noul" => (
                model
                    .decide_text(
                        &tokenizer,
                        state,
                        &Question::Noul {
                            instructions,
                            when_false: question["crit"]["false"].as_str(),
                            when_true: question["crit"]["true"].as_str(),
                        },
                    )
                    .unwrap(),
                QuestionType::Noul,
            ),
            _ => panic!("unknown reference question type"),
        };
        let ids: Vec<u32> = integers(&case, "ids")
            .into_iter()
            .map(|n| n as u32)
            .collect();
        let expected = model
            .decide_tokenized(
                &ids,
                &vec![true; ids.len()],
                &integers(&case, "markers"),
                kind,
            )
            .unwrap();
        assert_eq!(
            actual, expected,
            "text input differs from upstream tokenization"
        );
        eprintln!(
            "validated Laya text question: {}",
            question["t"].as_str().unwrap()
        );
    }
    model.unload().unwrap();
}

#[test]
#[ignore = "requires the pinned 843 MB Laya checkpoint and sufficient host memory"]
fn laya_real_checkpoint_component_and_head_match_pytorch() {
    let path = PathBuf::from(
        std::env::var_os("MAGNETAR_LAYA_REAL_CHECKPOINT")
            .expect("set MAGNETAR_LAYA_REAL_CHECKPOINT to the pinned model.safetensors"),
    );
    let source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::LocalPath(path.clone()),
        path.parent().unwrap().to_path_buf(),
    );
    let raw =
        inspect_safetensors_file(&source, path.file_name().unwrap().to_str().unwrap()).unwrap();
    assert_eq!(
        raw.single_file_digest,
        Some(
            ModelDigest::parse(
                "sha256:891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"
            )
            .unwrap()
        )
    );
    let profile = CheckpointProfile::from_huggingface_config(
        include_bytes!("../../../components/modernbert/tests/fixtures/laya-encoder.config.json"),
        CheckpointNamespace::Encoder,
    )
    .unwrap();
    let inventory: BTreeMap<_, _> = raw
        .tensors
        .iter()
        .map(|tensor| {
            (
                tensor.name.clone(),
                TensorDescriptor {
                    shape: tensor
                        .shape
                        .iter()
                        .map(|&n| usize::try_from(n).unwrap())
                        .collect(),
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
    let specs: BTreeMap<_, _> = profile
        .weights()
        .iter()
        .map(|spec| (spec.checkpoint_name.clone(), spec))
        .collect();
    let head_shapes = checkpoint_tensor_shapes(1024, 2, 2).unwrap();
    assert_eq!(inventory.len(), specs.len() + head_shapes.len() + 1);
    let mut graph_weights = BTreeMap::new();
    let mut head_weights = BTreeMap::new();
    for metadata in raw.tensors {
        if metadata.name == "temperature" {
            assert_eq!(metadata.storage_dtype, ModelDType::F32);
            assert_eq!(metadata.shape, [3]);
            continue;
        }
        assert!(specs.contains_key(&metadata.name) || head_shapes.contains_key(&metadata.name));
        let bytes = raw
            .payloads
            .read_payload(&ProductionPayloadRange {
                identity: metadata.name.clone(),
                offset: metadata.offset_bytes.unwrap(),
                length: metadata.size_bytes.unwrap(),
                digest: metadata.digest.clone(),
            })
            .unwrap();
        let mut bounded = metadata.clone();
        bounded.offset_bytes = Some(0);
        bounded.shard = None;
        let tensor = host_tensors_from_artifact_bytes(&[bounded], &bytes, 0)
            .unwrap()
            .remove(&metadata.name)
            .unwrap();
        let shape = tensor
            .shape
            .iter()
            .map(|&n| usize::try_from(n).unwrap())
            .collect();
        if let Some(spec) = specs.get(&metadata.name) {
            let decoded = DecodedTensor {
                shape,
                values: tensor.data,
            };
            let values = spec.graph_values(&decoded).unwrap();
            graph_weights.insert(
                spec.graph_name.clone(),
                HostTensor::new(
                    spec.graph_shape
                        .iter()
                        .map(|&n| n as u64)
                        .collect::<Vec<_>>(),
                    values,
                )
                .unwrap(),
            );
        } else {
            assert_eq!(head_shapes[&metadata.name], shape);
            head_weights.insert(
                metadata.name,
                CheckpointTensor {
                    shape,
                    values: tensor.data,
                },
            );
        }
    }
    assert_eq!(graph_weights.len(), specs.len());
    let head = DecisionHead::from_checkpoint_tensors(1024, 2, 2, head_weights).unwrap();
    let calibrator = Calibrator::from_config(include_bytes!(
        "../../../components/laya/tests/fixtures/rl_agent_config.json"
    ))
    .unwrap();
    let cases: Vec<serde_json::Value> = serde_json::from_str(include_str!(
        "../fixtures/laya-real-checkpoint-reference.json"
    ))
    .unwrap();
    for case in cases {
        let ids = integers(&case, "ids");
        let mask: Vec<_> = integers(&case, "mask")
            .into_iter()
            .map(|n| n != 0)
            .collect();
        let markers = integers(&case, "markers");
        let qtype = match case["qtype"].as_u64().unwrap() {
            0 => QuestionType::Choice,
            1 => QuestionType::Score,
            _ => panic!("unsupported reference question type"),
        };
        let (digest, request) =
            super::tests_laya_reference::encoder_graph_request(&profile, ids.len() as u64);
        let graph =
            build_first_native_encoder_graph_from_named_component(&digest, &request).unwrap();
        let mut bindings: BTreeMap<_, _> = graph_weights
            .iter()
            .map(|(name, tensor)| (TensorEdgeId::new(format!("weight.{name}")), tensor.clone()))
            .collect();
        bindings.insert(
            TensorEdgeId::new("input.token_ids"),
            HostTensor::new(
                [ids.len() as u64],
                ids.iter().map(|&n| n as f32).collect::<Vec<_>>(),
            )
            .unwrap(),
        );
        bindings.insert(
            TensorEdgeId::new("input.attention_mask"),
            HostTensor::new(
                [mask.len() as u64],
                mask.iter()
                    .map(|&valid| if valid { 1.0 } else { 0.0 })
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
        );
        let outcome = run_first_native_graph_with_provider_and_bindings(
            Arc::new(NativeCpuProvider::with_executor(
                NativeCpuExecutor::with_settings(true, false, 4),
            )),
            &e2e_fixture().unwrap(),
            &graph,
            &KvCacheId::new("laya-real-checkpoint").unwrap(),
            bindings,
            &["hidden_states"],
        )
        .unwrap();
        let hidden = &outcome.bindings[&TensorEdgeId::new("hidden_states")].data;
        close(hidden, &numbers(&case, "hidden"), "encoder hidden states");
        let scores = head
            .forward(hidden, ids.len(), &mask, &markers, qtype)
            .unwrap();
        close(
            &scores.option_logits,
            &numbers(&case, "logits"),
            "option logits",
        );
        close(
            &scores.action_logits,
            &numbers(&case, "action_logits"),
            "action logits",
        );
        let decision = scores.decide(&calibrator, qtype).unwrap();
        close(
            &decision
                .decision
                .probabilities
                .iter()
                .map(|&n| n as f32)
                .collect::<Vec<_>>(),
            &numbers(&case, "probabilities"),
            "option probabilities",
        );
        close(
            &decision
                .action_probabilities
                .iter()
                .map(|&n| n as f32)
                .collect::<Vec<_>>(),
            &numbers(&case, "action_probabilities"),
            "action probabilities",
        );
    }
}

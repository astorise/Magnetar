//! Full Kev pointer head and Qwen3.5+LoRA references from PyTorch.

use std::{collections::BTreeMap, path::PathBuf};

use magnetar_component_kev::{
    CheckpointTensor, PointerHead,
    encoding::{Delimiters, EncodedRequest, Question, encode_rows},
};
use magnetar_loader_huggingface::inspect_safetensors_file;
use magnetar_runtime::{
    ModelArtifactSource, ModelDigest, ProductionArtifactPayloadSource, ProductionModelSource,
    ProductionPayloadRange, host_tensors_from_artifact_bytes,
};

fn integers(value: &serde_json::Value, name: &str) -> Vec<usize> {
    value[name]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_u64().unwrap() as usize)
        .collect()
}

fn floats(value: &serde_json::Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_f64().unwrap() as f32)
        .collect()
}

fn load_real_head() -> PointerHead {
    let path = PathBuf::from(
        std::env::var_os("MAGNETAR_KEV_HEAD_SAFETENSORS")
            .expect("set MAGNETAR_KEV_HEAD_SAFETENSORS to the exported head"),
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
                "sha256:ab7c75921e79ec27ca97d8bf7b928e277dac843a6f4a528f9e8b7ab0fe7cd47d"
            )
            .unwrap()
        )
    );
    let mut tensors = BTreeMap::new();
    for metadata in raw.tensors {
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
        tensors.insert(
            metadata.name,
            CheckpointTensor {
                shape: tensor.shape.iter().map(|&n| n as usize).collect(),
                values: tensor.data,
            },
        );
    }
    PointerHead::from_checkpoint_tensors(tensors, 2.351_096).unwrap()
}

fn reference_request() -> EncodedRequest {
    let first = vec![vec![20], vec![21]];
    let second = vec![vec![22], vec![23], vec![24]];
    let questions = [
        Question {
            instruction: &[10],
            options: &first,
        },
        Question {
            instruction: &[11],
            options: &second,
        },
    ];
    encode_rows(
        &[271],
        &questions,
        Delimiters {
            state: 248060,
            question: 248061,
            option_start: 248049,
            option_end: 248050,
            decide: 248062,
        },
        16,
        128,
    )
    .unwrap()
}

#[test]
#[ignore = "requires the safely exported pinned Kev pointer-head weights"]
fn full_kev_head_scores_real_adapted_qwen_hidden_states() {
    let head = load_real_head();
    let request = reference_request();
    let cases: Vec<serde_json::Value> = serde_json::from_str(include_str!(
        "../fixtures/kev-real-checkpoint-reference.json"
    ))
    .unwrap();
    assert_eq!(request.rows.len(), cases.len());
    let mut max_error = 0.0_f32;
    for (row, case) in request.rows.iter().zip(cases) {
        assert_eq!(
            request
                .prefix
                .iter()
                .map(|&n| n as usize)
                .collect::<Vec<_>>(),
            integers(&case, "prefix")
        );
        assert_eq!(
            row.token_ids
                .iter()
                .map(|&n| n as usize)
                .collect::<Vec<_>>(),
            integers(&case, "row")
        );
        assert_eq!(
            row.option_end_indices,
            integers(&case, "option_end_indices")
        );
        assert_eq!(
            row.decision_index,
            case["decision_index"].as_u64().unwrap() as usize
        );
        let mut hidden = vec![0.0; row.token_ids.len() * 1024];
        hidden[row.decision_index * 1024..(row.decision_index + 1) * 1024]
            .copy_from_slice(&floats(&case["decision"]));
        for (&index, option) in row
            .option_end_indices
            .iter()
            .zip(case["options"].as_array().unwrap())
        {
            hidden[index * 1024..(index + 1) * 1024].copy_from_slice(&floats(option));
        }
        let result = head.score_row(row, &hidden).unwrap();
        for (actual, expected) in [
            (&result.logits, floats(&case["logits"])),
            (&result.probabilities, floats(&case["probabilities"])),
        ] {
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                let error = (actual - expected).abs();
                max_error = max_error.max(error);
                assert!(error <= 2e-5, "{actual} != {expected}");
            }
        }
    }
    eprintln!("maximum absolute score/probability error: {max_error}");
}

/// Qualifies the derived Qwen backbone and the full Kev head together. The
/// adapter has been merged externally; this does not test prefix-cache reuse.
#[test]
#[ignore = "requires the merged Kev/Qwen checkpoint and safely exported pointer head"]
fn merged_kev_backbone_production_logits_match_pytorch() {
    use std::{fs, io::Read, ops::ControlFlow, sync::Arc};

    use magnetar_loader_huggingface::{
        HuggingFaceIngestor, HuggingFaceTokenizer, parse_tokenizer_config,
    };
    use magnetar_provider_cpu::native::{NativeCpuExecutor, NativeCpuProvider};
    use magnetar_runtime::{
        GenerationParameters, ModelTrustStore, ProductionGenerationRequest, ProductionLoadedModel,
        ProductionModelArtifactIngestor, PromptInput, StopConditions, Tokenizer,
        production_model_fixture,
    };
    use sha2::{Digest, Sha256};

    let directory = PathBuf::from(
        std::env::var_os("MAGNETAR_KEV_MERGED_DIR")
            .expect("set MAGNETAR_KEV_MERGED_DIR to the prepared derived bundle"),
    );
    let mut file = fs::File::open(directory.join("model.safetensors")).unwrap();
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 8 << 20];
    loop {
        let count = file.read(&mut buffer).unwrap();
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    let actual: String = hash
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(
        actual,
        "a7aeec3ce2e2b0ab61fed3598fc89b80668eb774e7d0df6ddfabad8931bca51f"
    );
    let source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::LocalPath(directory.clone()),
        directory.clone(),
    );
    let ingested = HuggingFaceIngestor::new().ingest(&source).unwrap();
    let tokenizer_config =
        parse_tokenizer_config(&fs::read(directory.join("tokenizer_config.json")).unwrap())
            .unwrap();
    let tokenizer = HuggingFaceTokenizer::from_bytes(
        &fs::read(directory.join("tokenizer.json")).unwrap(),
        Some(&tokenizer_config),
        "kev-qwen35-merged",
        Some(248_320),
    )
    .unwrap();
    let metadata = tokenizer.metadata().clone();
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(tokenizer);
    let trust = ModelTrustStore::default().trust_digest(ingested.manifest.id.digest.value.clone());
    let fixture =
        production_model_fixture(ingested.manifest.clone(), metadata, tokenizer.clone()).unwrap();
    super::test_support::register_real_qwen_component();
    let provider = Arc::new(NativeCpuProvider::with_executor(
        NativeCpuExecutor::with_settings(true, false, 4),
    ));
    let mut model =
        ProductionLoadedModel::load(fixture, ingested.payload_source.as_ref(), trust, provider)
            .unwrap();
    model.set_prefill_replay(false);
    let cases: Vec<serde_json::Value> = serde_json::from_str(include_str!(
        "../fixtures/kev-real-checkpoint-reference.json"
    ))
    .unwrap();
    let request = reference_request();
    let head = load_real_head();
    assert_eq!(cases.len(), request.rows.len());
    let mut max_error = 0.0_f32;
    let mut max_hidden_error = 0.0_f32;
    let mut max_score_error = 0.0_f32;
    for (case, row) in cases.iter().zip(&request.rows) {
        let tokens: Vec<u32> = integers(case, "prefix")
            .into_iter()
            .chain(integers(case, "row"))
            .map(|n| n as u32)
            .collect();
        let hidden = model
            .observe_prefill_edge(&tokens, "edge.final_norm")
            .unwrap();
        assert_eq!(hidden.shape, vec![tokens.len() as u64, 1024]);
        let row_hidden = &hidden.data[request.prefix.len() * 1024..];
        let selected = std::iter::once((row.decision_index, floats(&case["decision"]))).chain(
            row.option_end_indices
                .iter()
                .zip(case["options"].as_array().unwrap())
                .map(|(&index, values)| (index, floats(values))),
        );
        for (index, expected) in selected {
            for (&actual, expected) in row_hidden[index * 1024..(index + 1) * 1024]
                .iter()
                .zip(expected)
            {
                let error = (actual - expected).abs();
                max_hidden_error = max_hidden_error.max(error);
                assert!(error <= 1e-3, "hidden state {actual} != {expected}");
            }
        }
        let scored = head.score_row(row, row_hidden).unwrap();
        for (actual, expected) in [
            (&scored.logits, floats(&case["logits"])),
            (&scored.probabilities, floats(&case["probabilities"])),
        ] {
            assert_eq!(actual.len(), expected.len());
            for (&actual, expected) in actual.iter().zip(expected) {
                let error = (actual - expected).abs();
                max_score_error = max_score_error.max(error);
                assert!(error <= 1e-4, "score {actual} != {expected}");
            }
        }
        let mut logits = Vec::new();
        let outcome = model
            .generate_streaming_with_logits(
                ProductionGenerationRequest {
                    prompt: PromptInput::TokenIds(tokens),
                    parameters: GenerationParameters::greedy(),
                    stop_conditions: StopConditions::default(),
                    max_new_tokens: Some(1),
                    max_generation_millis: None,
                },
                None,
                &mut |_| ControlFlow::Continue(()),
                &mut |values| logits.extend_from_slice(values),
            )
            .unwrap();
        assert_eq!(
            outcome.result.output.generated_token_ids[0] as u64,
            case["vocab_argmax"].as_u64().unwrap()
        );
        for (index, expected) in integers(case, "vocab_probe_ids")
            .into_iter()
            .zip(floats(&case["vocab_probe_logits"]))
        {
            let error = (logits[index] - expected).abs();
            max_error = max_error.max(error);
            assert!(
                error <= 0.05,
                "token {index}: {} != {expected}",
                logits[index]
            );
        }
    }
    let first_options = vec![vec![20], vec![21]];
    let second_options = vec![vec![22], vec![23], vec![24]];
    let questions = [
        magnetar_inference_component::kev::TokenizedQuestion {
            instruction: &[10],
            options: &first_options,
        },
        magnetar_inference_component::kev::TokenizedQuestion {
            instruction: &[11],
            options: &second_options,
        },
    ];
    let head_path = PathBuf::from(std::env::var_os("MAGNETAR_KEV_HEAD_SAFETENSORS").unwrap());
    let product_head =
        magnetar_inference_component::kev::load_pinned_head_safetensors(&head_path).unwrap();
    let decisions = magnetar_inference_component::kev::decide_tokenized(
        &mut model,
        &product_head,
        &[271],
        &questions,
    )
    .unwrap();
    // These strings tokenize to the exact IDs used by the independent PyTorch
    // fixture with the pinned Kev tokenizer ("\n\n" is token 271).
    let text_questions = [
        magnetar_inference_component::kev::TextQuestion {
            instruction: "+",
            options: &["5", "6"],
        },
        magnetar_inference_component::kev::TextQuestion {
            instruction: ",",
            options: &["7", "8", "9"],
        },
    ];
    let text_decisions = magnetar_inference_component::kev::decide_text(
        &mut model,
        tokenizer.as_ref(),
        &product_head,
        "\n\n",
        &text_questions,
    )
    .unwrap();
    assert_eq!(decisions, text_decisions);
    for (decision, case) in decisions.iter().zip(&cases) {
        let expected_logits = floats(&case["logits"]);
        let expected_probabilities = floats(&case["probabilities"]);
        assert_eq!(decision.options.len(), expected_logits.len());
        for ((option, expected_logit), expected_probability) in decision
            .options
            .iter()
            .zip(expected_logits)
            .zip(expected_probabilities)
        {
            assert!((option.score - f64::from(expected_logit)).abs() <= 1e-4);
            assert!((option.probability - f64::from(expected_probability)).abs() <= 1e-4);
        }
    }
    eprintln!("merged Kev backbone maximum absolute logit error: {max_error}");
    eprintln!(
        "maximum hidden-state error: {max_hidden_error}; score/probability error: {max_score_error}"
    );
    model.unload().unwrap();
}

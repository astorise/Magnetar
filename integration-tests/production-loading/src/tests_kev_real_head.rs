//! Full Kev pointer head and Qwen3.5+LoRA hidden states from PyTorch.
//! The backbone itself is not run by the Runtime in this test.

use std::{collections::BTreeMap, path::PathBuf};

use magnetar_component_kev::{
    CheckpointTensor, PointerHead,
    encoding::{Delimiters, Question, encode_rows},
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

#[test]
#[ignore = "requires the safely exported pinned Kev pointer-head weights"]
fn full_kev_head_scores_real_adapted_qwen_hidden_states() {
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
    let head = PointerHead::from_checkpoint_tensors(tensors, 2.351_096).unwrap();
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
    let request = encode_rows(
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
    .unwrap();
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

/// Qualifies the adapted Qwen backbone's generation logits. The derived
/// checkpoint is pre-merged externally; Runtime adapter loading and Kev's
/// pointer-head inference are not exercised by this test.
#[test]
#[ignore = "requires the separately prepared 2.75 GB merged Kev/Qwen checkpoint"]
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
    let fixture = production_model_fixture(ingested.manifest.clone(), metadata, tokenizer).unwrap();
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
    let mut max_error = 0.0_f32;
    for case in cases {
        let tokens: Vec<u32> = integers(&case, "prefix")
            .into_iter()
            .chain(integers(&case, "row"))
            .map(|n| n as u32)
            .collect();
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
        for (index, expected) in integers(&case, "vocab_probe_ids")
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
    eprintln!("merged Kev backbone maximum absolute logit error: {max_error}");
    model.unload().unwrap();
}

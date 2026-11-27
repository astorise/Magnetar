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

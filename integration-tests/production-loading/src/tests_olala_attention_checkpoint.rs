//! Optional real-weight Olala differential-attention comparison.

use magnetar_component_olala::attention_profile::AttentionProfile;
use magnetar_runtime::{
    HostTensor, ModelDType, ModelTensorMetadata, OperatorAttributeValue, operator_semantics,
    tensor_from_raw_bytes,
};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path};

fn checked(path: &Path, expected: &str) -> Vec<u8> {
    let bytes = std::fs::read(path).unwrap();
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(digest, expected);
    bytes
}

fn values(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    let (chunks, remainder) = bytes.as_chunks::<4>();
    assert!(remainder.is_empty());
    chunks
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect()
}

fn close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    let mut maximum = 0.0_f32;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let difference = (actual - expected).abs();
        maximum = maximum.max(difference);
        assert!(
            difference <= 7e-5 + 1e-5 * expected.abs(),
            "{label}[{index}]: {actual} != {expected} (error {difference})"
        );
    }
    eprintln!("{label}: maximum absolute error {maximum}");
}

/// No checkpoint data is checked in; see the fixture README for generation.
#[test]
#[ignore = "requires pinned Olala attention byte ranges and PyTorch reference"]
fn real_olala_attention_weights_match_pytorch_in_f32() {
    let root = std::env::var_os("MAGNETAR_OLALA_ATTENTION_DIR")
        .expect("set MAGNETAR_OLALA_ATTENTION_DIR to the reference directory");
    let root = Path::new(&root);
    let header = checked(
        &root.join("olala-header.bin"),
        "81df2fdc92cc4faa6c66b9786d2103045c6b466a575b31f97151d64f653882f5",
    );
    let f32_range = checked(
        &root.join("olala-attention-layer4-f32.bin"),
        "589d98523668c2226beea57b4e69d073b8bd88d686672a32b1798d07c03366eb",
    );
    let bf16_range = checked(
        &root.join("olala-attention-layer4-bf16.bin"),
        "90c8e3ac8d85c7e17d0bb8370d9d41cd66fa656996ab0322a42b467a8e94d03f",
    );
    let header: serde_json::Value = serde_json::from_slice(&header[8..]).unwrap();
    let profile = AttentionProfile::from_huggingface_config(include_bytes!(
        "../../../components/olala/tests/fixtures/olala-mini.config.json"
    ))
    .unwrap();
    let plan = profile.plan_sequence(4, 2, 0).unwrap();
    let mut weights = Vec::new();
    for spec in &plan.weights {
        let entry = &header[&spec.checkpoint_name];
        let shape: Vec<usize> = serde_json::from_value(entry["shape"].clone()).unwrap();
        assert_eq!(shape, spec.checkpoint_shape, "{}", spec.checkpoint_name);
        let (base, region, dtype, expected_dtype) =
            if spec.checkpoint_name.ends_with("softmax_scaler") {
                (55_829_544, &f32_range, ModelDType::F32, "F32")
            } else {
                (11_481_542_688, &bf16_range, ModelDType::Bf16, "BF16")
            };
        assert_eq!(entry["dtype"], expected_dtype, "{}", spec.checkpoint_name);
        let start = entry["data_offsets"][0].as_u64().unwrap() as usize;
        let end = entry["data_offsets"][1].as_u64().unwrap() as usize;
        let raw = &region[start - base..end - base];
        let metadata = ModelTensorMetadata {
            name: spec.checkpoint_name.clone(),
            shape: shape.iter().map(|&value| value as u64).collect(),
            storage_dtype: dtype,
            layout: None,
            shard: None,
            offset_bytes: None,
            size_bytes: Some(raw.len() as u64),
            quantization: None,
            expected_compute_dtype: None,
            digest: None,
        };
        let decoded = tensor_from_raw_bytes(&metadata, raw).unwrap();
        let reordered = spec.reorder_f32(&decoded.data).unwrap();
        weights.push(
            HostTensor::new(
                spec.operator_shape
                    .iter()
                    .map(|&n| n as u64)
                    .collect::<Vec<_>>(),
                reordered,
            )
            .unwrap(),
        );
    }
    let reference = root.join("olala-attention-reference");
    let hidden = values(&reference.join("input.f32"));
    assert_eq!(hidden.len(), 4 * profile.hidden_size);
    let run = |start: usize, tokens: usize, state: Option<&[HostTensor]>| {
        let plan = profile.plan_sequence(4, tokens, start).unwrap();
        let mut tensors = vec![
            HostTensor::new(
                [tokens as u64, profile.hidden_size as u64],
                hidden[start * profile.hidden_size..(start + tokens) * profile.hidden_size]
                    .to_vec(),
            )
            .unwrap(),
        ];
        if let Some(state) = state {
            tensors.extend_from_slice(state);
        } else {
            for shape in &plan.input_shapes[1..5] {
                tensors.push(
                    HostTensor::new(
                        shape.iter().map(|&n| n as u64).collect::<Vec<_>>(),
                        vec![0.0; shape.iter().product()],
                    )
                    .unwrap(),
                );
            }
        }
        tensors.push(
            HostTensor::new(
                [tokens as u64],
                (start..start + tokens)
                    .map(|position| position as f32)
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
        );
        tensors.extend_from_slice(&weights);
        let mut attrs = BTreeMap::new();
        for (name, value) in [
            ("query_heads", profile.query_heads),
            ("kv_heads", profile.kv_heads),
            ("head_dim", profile.head_dim),
            ("rank", profile.rank),
            ("past_length", start),
            ("window_left", plan.window_left.unwrap()),
        ] {
            attrs.insert(name.into(), OperatorAttributeValue::Integer(value as i64));
        }
        for (name, value) in [
            ("epsilon", plan.epsilon),
            ("scale", plan.score_scale),
            ("softcap", plan.softcap),
        ] {
            attrs.insert(name.into(), OperatorAttributeValue::Float(value));
        }
        attrs.insert(
            "zero_centered_norm".into(),
            OperatorAttributeValue::Boolean(plan.zero_centered_norm),
        );
        attrs.insert(
            "round_bfloat16".into(),
            OperatorAttributeValue::Boolean(plan.round_bfloat16),
        );
        let config =
            operator_semantics::projected_differential_attention_from_attributes(&attrs).unwrap();
        operator_semantics::projected_differential_attention(&tensors, config).unwrap()
    };
    let compare = |name: &str, actual: &[HostTensor; 5]| {
        for (index, suffix) in ["output", "raw_key", "raw_value", "keys", "values"]
            .iter()
            .enumerate()
        {
            close(
                &actual[index].data,
                &values(&reference.join(format!("{name}.{suffix}.f32"))),
                &format!("{name} {suffix}"),
            );
        }
    };
    compare("full", &run(0, 4, None));
    let prefill = run(0, 2, None);
    compare("prefill", &prefill);
    let mut state = prefill[1..].to_vec();
    for index in 0..2 {
        let output = run(index + 2, 1, Some(&state));
        compare(&format!("decode{index}"), &output);
        state = output[1..].to_vec();
    }
}

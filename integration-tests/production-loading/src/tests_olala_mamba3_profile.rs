//! Check Olala's pinned checkpoint plan against the portable operator contract.

use magnetar_component_olala::mamba3_profile::Mamba3Profile;
use magnetar_runtime::{
    ComputeDType, DTypeDescriptor, HostTensor, ModelDType, ModelTensorMetadata,
    OperatorAttributeValue, OperatorFamily, OperatorId, ShapeDescriptor, TensorDescriptor,
    initial_operator_catalog, operator_semantics, tensor_from_raw_bytes,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

#[test]
fn pinned_olala_mamba3_plan_matches_the_runtime_operator_interface() {
    let profile = Mamba3Profile::from_huggingface_config(include_bytes!(
        "../../../components/olala/tests/fixtures/olala-mini.config.json"
    ))
    .unwrap();
    let catalog = initial_operator_catalog();
    let operator = catalog
        .get(&OperatorId::magnetar(
            "mamba3-mixer",
            1,
            OperatorFamily::LinearAlgebra,
        ))
        .unwrap();
    for tokens in [1, 7, profile.max_tokens()] {
        let plan = profile.plan_sequence(0, tokens).unwrap();
        let descriptor = |shape: &Vec<usize>| {
            TensorDescriptor::materialized(
                ShapeDescriptor::new(shape.iter().map(|&n| n as u64).collect::<Vec<_>>()),
                DTypeDescriptor::portable(ComputeDType::Float32),
            )
        };
        let inputs: Vec<_> = plan.input_shapes.iter().map(descriptor).collect();
        let outputs: Vec<_> = plan.output_shapes.iter().map(descriptor).collect();
        let mut attributes = BTreeMap::new();
        for (name, value) in [
            ("heads", plan.heads),
            ("groups", plan.groups),
            ("rank", plan.rank),
            ("head_dim", plan.head_dim),
            ("state_dim", plan.state_dim),
            ("angles", plan.angles),
        ] {
            attributes.insert(name.into(), OperatorAttributeValue::Integer(value as i64));
        }
        for (name, value) in [("epsilon", plan.epsilon), ("decay_floor", plan.decay_floor)] {
            attributes.insert(name.into(), OperatorAttributeValue::Float(value));
        }
        attributes.insert(
            "zero_centered_norm".into(),
            OperatorAttributeValue::Boolean(plan.zero_centered_norm),
        );
        operator
            .validate_invocation(&inputs, &outputs, &attributes)
            .unwrap();
        let semantics = operator_semantics::mamba3_mixer_from_attributes(&attributes).unwrap();
        assert_eq!(semantics.geometry.heads, plan.heads);
        assert_eq!(semantics.geometry.rank, plan.rank);
        let mut invalid_inputs = inputs.clone();
        invalid_inputs[3] = descriptor(&vec![1, 1, 1]);
        assert!(
            operator
                .validate_invocation(&invalid_inputs, &outputs, &attributes)
                .is_err()
        );
    }
}

/// Runs only when the separately downloaded, hash-checked weights and
/// PyTorch reference files are supplied. No checkpoint data is checked in.
#[test]
#[ignore = "requires pinned Olala mixer byte ranges and PyTorch reference"]
fn real_olala_mamba3_weights_match_pytorch_in_f32() {
    use std::path::Path;

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
                difference <= 5e-5 + 1e-4 * expected.abs(),
                "{label}[{index}]: {actual} != {expected} (error {difference})"
            );
        }
        eprintln!("{label}: maximum absolute error {maximum}");
    }

    let root = std::env::var_os("MAGNETAR_OLALA_MAMBA3_DIR")
        .expect("set MAGNETAR_OLALA_MAMBA3_DIR to the reference directory");
    let root = Path::new(&root);
    let header = checked(
        &root.join("olala-header.bin"),
        "81df2fdc92cc4faa6c66b9786d2103045c6b466a575b31f97151d64f653882f5",
    );
    let f32_range = checked(
        &root.join("olala-mamba3-layer0-f32.bin"),
        "63bc9130e58e93d8f2682b8ca8d676b474ec070dd91a1a5b9bce9c0f8edd7bc7",
    );
    let bf16_range = checked(
        &root.join("olala-mamba3-layer0-bf16.bin"),
        "37becb7237e9f3d731b106eeda23aab33a83e3ee315cb8c616da94b0bf642f0f",
    );
    let header: serde_json::Value = serde_json::from_slice(&header[8..]).unwrap();
    let profile = Mamba3Profile::from_huggingface_config(include_bytes!(
        "../../../components/olala/tests/fixtures/olala-mini.config.json"
    ))
    .unwrap();
    let plan = profile.plan_sequence(0, 2).unwrap();
    let mut weights = Vec::new();
    for spec in &plan.weights {
        let entry = &header[&spec.checkpoint_name];
        let shape: Vec<usize> = serde_json::from_value(entry["shape"].clone()).unwrap();
        assert_eq!(shape, spec.checkpoint_shape, "{}", spec.checkpoint_name);
        assert_eq!(
            entry["dtype"], spec.checkpoint_dtype,
            "{}",
            spec.checkpoint_name
        );
        let start = entry["data_offsets"][0].as_u64().unwrap() as usize;
        let end = entry["data_offsets"][1].as_u64().unwrap() as usize;
        let (base, region, dtype) = match spec.checkpoint_dtype {
            "F32" => (24, &f32_range, ModelDType::F32),
            "BF16" => (1_000_145_952, &bf16_range, ModelDType::Bf16),
            _ => panic!("unexpected checkpoint dtype"),
        };
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
    let mut attributes = BTreeMap::new();
    for (name, value) in [
        ("heads", plan.heads),
        ("groups", plan.groups),
        ("rank", plan.rank),
        ("head_dim", plan.head_dim),
        ("state_dim", plan.state_dim),
        ("angles", plan.angles),
    ] {
        attributes.insert(name.into(), OperatorAttributeValue::Integer(value as i64));
    }
    attributes.insert(
        "epsilon".into(),
        OperatorAttributeValue::Float(plan.epsilon),
    );
    attributes.insert(
        "decay_floor".into(),
        OperatorAttributeValue::Float(plan.decay_floor),
    );
    attributes.insert(
        "zero_centered_norm".into(),
        OperatorAttributeValue::Boolean(plan.zero_centered_norm),
    );
    let config = operator_semantics::mamba3_mixer_from_attributes(&attributes).unwrap();
    let reference = root.join("olala-mamba3-reference");
    let input = values(&reference.join("input.f32"));
    assert_eq!(input.len(), 4 * 1536);
    let run = |start: usize, tokens: usize, state: Option<&[HostTensor]>| {
        let mut tensors = vec![
            HostTensor::new(
                [tokens as u64, 1536],
                input[start * 1536..(start + tokens) * 1536].to_vec(),
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
        tensors.extend_from_slice(&weights);
        operator_semantics::mamba3_mixer(&tensors, config).unwrap()
    };
    let full = run(0, 4, None);
    close(
        &full[0].data,
        &values(&reference.join("full.f32")),
        "full output",
    );
    let mut prefill = run(0, 2, None);
    for angle in &mut prefill[1].data {
        *angle = angle.rem_euclid(std::f32::consts::TAU);
    }
    for (index, suffix) in ["output", "angle", "ssm", "key", "input"]
        .iter()
        .enumerate()
    {
        close(
            &prefill[index].data,
            &values(&reference.join(format!("prefill.{suffix}.f32"))),
            &format!("prefill {suffix}"),
        );
    }
    let mut state = prefill[1..].to_vec();
    for index in 0..2 {
        let result = run(index + 2, 1, Some(&state));
        for (i, suffix) in ["output", "angle", "ssm", "key", "input"]
            .iter()
            .enumerate()
        {
            close(
                &result[i].data,
                &values(&reference.join(format!("decode{index}.{suffix}.f32"))),
                &format!("decode {index} {suffix}"),
            );
        }
        state = result[1..].to_vec();
    }
}

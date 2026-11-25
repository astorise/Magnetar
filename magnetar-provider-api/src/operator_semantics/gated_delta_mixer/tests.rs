use super::*;

fn values(count: u64, seed: u32, scale: f32) -> Vec<f32> {
    (0..count)
        .map(|i| {
            let mixed = (i as u32).wrapping_mul(2_246_822_519).wrapping_add(seed) % 2001;
            (mixed as f32 - 1000.0) / 1000.0 * scale
        })
        .collect()
}

fn tensor(shape: &[u64], seed: u32, scale: f32) -> HostTensor {
    HostTensor::new(shape.to_vec(), values(shape.iter().product(), seed, scale)).unwrap()
}

/// `[rows, in] x [out, in]ᵀ`, each sum in input order (as the
/// Reference's own projections).
fn project(x: &[f32], weight: &HostTensor) -> Vec<f32> {
    let width = weight.shape[1] as usize;
    x.chunks_exact(width)
        .flat_map(|row| {
            weight
                .data
                .chunks_exact(width)
                .map(|w| w.iter().zip(row).map(|(w, x)| w * x).sum::<f32>())
                .collect::<Vec<_>>()
        })
        .collect()
}

/// A Provider projecting the inputs itself and passing the projections
/// to `gated_delta_mixer_from_projections` gets exactly the
/// Reference's outputs, masked tokens and next states included.
#[test]
fn the_mixer_from_its_projections_equals_the_whole_mixer() {
    let config = GatedDeltaMixer {
        key_heads: 2,
        value_heads: 4,
        key_dim: 8,
        value_dim: 4,
        conv_kernel: 4,
        epsilon: 1e-6,
        output_gate: MixerOutputGate::Silu,
    };
    let (tokens, hidden, channels, value_width) = (3u64, 12u64, 48u64, 16u64);
    let mask = HostTensor::new([tokens], vec![1.0, 0.0, 1.0]).unwrap();
    let inputs = vec![
        tensor(&[tokens, hidden], 1, 1.0),
        tensor(&[channels, 4], 2, 0.5),
        tensor(&[4, 8, 4], 3, 0.2),
        tensor(&[channels, hidden], 4, 0.3),
        tensor(&[value_width, hidden], 5, 0.3),
        tensor(&[4, hidden], 6, 0.3),
        tensor(&[4, hidden], 7, 0.3),
        tensor(&[channels, 4], 8, 0.5),
        tensor(&[4], 9, 0.5),
        tensor(&[4], 10, 0.5),
        tensor(&[4], 11, 0.2),
        tensor(&[hidden, value_width], 12, 0.3),
        mask.clone(),
    ];
    let whole = gated_delta_mixer(&inputs, config).unwrap();

    let mut x = inputs[0].data.clone();
    for (row, &keep) in x.chunks_exact_mut(hidden as usize).zip(&mask.data) {
        if keep == 0.0 {
            row.fill(0.0);
        }
    }
    let projections = [3, 4, 5, 6].map(|index| project(&x, &inputs[index]));
    let [normalized, convolution, recurrent] = gated_delta_mixer_from_projections(
        [
            &projections[0],
            &projections[1],
            &projections[2],
            &projections[3],
        ],
        tokens,
        [&inputs[1], &inputs[2]],
        [&inputs[7], &inputs[8], &inputs[9], &inputs[10]],
        config,
    )
    .unwrap();
    assert_eq!(normalized.shape, [tokens, value_width]);
    assert_eq!(project(&normalized.data, &inputs[11]), whole[0].data);
    assert_eq!(convolution, whole[1]);
    assert_eq!(recurrent, whole[2]);
    // Mismatched projections are refused.
    assert!(
        gated_delta_mixer_from_projections(
            [
                &projections[0][1..],
                &projections[1],
                &projections[2],
                &projections[3],
            ],
            tokens,
            [&inputs[1], &inputs[2]],
            [&inputs[7], &inputs[8], &inputs[9], &inputs[10]],
            config,
        )
        .is_err()
    );
}

/// `output_gate_type` picks the gate's activation (`support-qwen38-flash-next`,
/// Qwen4-Exp): with `sigmoid` the normalized output is `silu`'s divided by
/// `z`, element by element, and nothing else changes (states included).
#[test]
fn the_output_gate_activation_follows_output_gate_type() {
    let silu_config = GatedDeltaMixer {
        key_heads: 2,
        value_heads: 4,
        key_dim: 8,
        value_dim: 4,
        conv_kernel: 4,
        epsilon: 1e-6,
        output_gate: MixerOutputGate::Silu,
    };
    let sigmoid_config = GatedDeltaMixer {
        output_gate: MixerOutputGate::Sigmoid,
        ..silu_config
    };
    let (tokens, hidden, channels, value_width) = (3u64, 12u64, 48u64, 16u64);
    let x = values(tokens * hidden, 21, 1.0);
    let weights = [
        tensor(&[channels, hidden], 22, 0.3),
        tensor(&[value_width, hidden], 23, 0.3),
        tensor(&[4, hidden], 24, 0.3),
        tensor(&[4, hidden], 25, 0.3),
    ];
    let projections = [0, 1, 2, 3].map(|index| project(&x, &weights[index]));
    let states = [tensor(&[channels, 4], 26, 0.5), tensor(&[4, 8, 4], 27, 0.2)];
    let norm_weights = [
        tensor(&[channels, 4], 28, 0.5),
        tensor(&[4], 29, 0.5),
        tensor(&[4], 30, 0.5),
        tensor(&[4], 31, 0.2),
    ];
    let run = |config| {
        gated_delta_mixer_from_projections(
            [
                &projections[0],
                &projections[1],
                &projections[2],
                &projections[3],
            ],
            tokens,
            [&states[0], &states[1]],
            [
                &norm_weights[0],
                &norm_weights[1],
                &norm_weights[2],
                &norm_weights[3],
            ],
            config,
        )
        .unwrap()
    };
    let silu = run(silu_config);
    let sigmoid = run(sigmoid_config);
    let z = &projections[1];
    for ((silu, sigmoid), z) in silu[0].data.iter().zip(&sigmoid[0].data).zip(z) {
        assert!(
            (sigmoid * z - silu).abs() <= 1e-5 * (1.0 + silu.abs()),
            "sigmoid {sigmoid} * z {z} should be silu {silu}"
        );
    }
    assert_ne!(silu[0], sigmoid[0], "the two gates differ");
    assert_eq!(silu[1], sigmoid[1], "the convolution state is untouched");
    assert_eq!(silu[2], sigmoid[2], "the recurrent state is untouched");

    // The attribute selects the activation; an unknown one is refused.
    let attributes = |gate: Option<&str>| {
        let mut attributes = BTreeMap::from([
            ("key_heads".to_string(), OperatorAttributeValue::Integer(2)),
            (
                "value_heads".to_string(),
                OperatorAttributeValue::Integer(4),
            ),
            ("key_dim".to_string(), OperatorAttributeValue::Integer(8)),
            ("value_dim".to_string(), OperatorAttributeValue::Integer(4)),
            (
                "conv_kernel".to_string(),
                OperatorAttributeValue::Integer(4),
            ),
            ("epsilon".to_string(), OperatorAttributeValue::Float(1e-6)),
        ]);
        if let Some(gate) = gate {
            attributes.insert(
                "output_gate_type".into(),
                OperatorAttributeValue::String(gate.into()),
            );
        }
        attributes
    };
    assert_eq!(
        gated_delta_mixer_from_attributes(&attributes(None))
            .unwrap()
            .output_gate,
        MixerOutputGate::Silu
    );
    assert_eq!(
        gated_delta_mixer_from_attributes(&attributes(Some("silu")))
            .unwrap()
            .output_gate,
        MixerOutputGate::Silu
    );
    assert_eq!(
        gated_delta_mixer_from_attributes(&attributes(Some("sigmoid")))
            .unwrap()
            .output_gate,
        MixerOutputGate::Sigmoid
    );
    assert!(gated_delta_mixer_from_attributes(&attributes(Some("gelu"))).is_err());
}

//! `support-qwen38-flash-next` lot 3: the Reference semantics of Qwen4-Exp's
//! blocks against fixtures produced by running Transformers' `qwen4_exp`
//! (pinned in design D7) on synthetic weights:
//! `fixtures/generate-qwen4-exp-reference.py`.

use magnetar_provider_api::{
    GatedResidual, HostTensor, gated_residual, gated_residual_inject, repeat_features,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Flat {
    shape: Vec<u64>,
    data: Vec<f32>,
}

impl Flat {
    fn tensor(&self) -> HostTensor {
        HostTensor::new(self.shape.clone(), self.data.clone()).unwrap()
    }
}

#[derive(Deserialize)]
struct GatedResidualCase {
    norm_weight: Flat,
    down: Flat,
    up: Flat,
    inject: Flat,
    input: Flat,
    mixed: Flat,
    injection: Flat,
    block_output: Flat,
    after_block: Flat,
    final_mixed: Flat,
    embedding: Flat,
    repeated: Flat,
}

#[derive(Deserialize)]
struct Fixture {
    hc_count: usize,
    epsilon: f32,
    gated_residual: GatedResidualCase,
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!("../fixtures/qwen4-exp-reference.json")).unwrap()
}

fn assert_matches(actual: &HostTensor, expected: &Flat, what: &str) {
    assert_eq!(actual.shape, expected.shape, "{what} shape");
    for (index, (a, e)) in actual.data.iter().zip(&expected.data).enumerate() {
        assert!(
            (a - e).abs() <= 1e-5 * (1.0 + e.abs()),
            "{what}[{index}]: {a} vs transformers {e}"
        );
    }
}

/// The gated residual (with and without the write weights), the closing
/// injection of a block's output into every stream, and the embedding's
/// repetition into streams, against the pinned Transformers module.
#[test]
fn the_gated_residual_matches_the_pinned_transformers_module() {
    let fixture = fixture();
    let case = &fixture.gated_residual;
    let config = GatedResidual {
        hc_count: fixture.hc_count,
        epsilon: fixture.epsilon,
        // Qwen4ExpTextRMSNorm scales by `1 + weight`.
        weight_offset: 1.0,
    };
    let mut inputs = vec![
        case.input.tensor(),
        case.norm_weight.tensor(),
        case.down.tensor(),
        case.up.tensor(),
    ];
    let without_write = gated_residual(&inputs, config).unwrap();
    assert_eq!(without_write.len(), 1);
    assert_matches(&without_write[0], &case.final_mixed, "final mixer");

    inputs.push(case.inject.tensor());
    let with_write = gated_residual(&inputs, config).unwrap();
    assert_eq!(with_write.len(), 2);
    assert_matches(&with_write[0], &case.mixed, "mixed input");
    assert_matches(&with_write[1], &case.injection, "write weights");

    let after = gated_residual_inject(&[
        case.input.tensor(),
        case.block_output.tensor(),
        with_write[1].clone(),
    ])
    .unwrap();
    assert_matches(&after, &case.after_block, "residual after the block");

    let repeated = repeat_features(&case.embedding.tensor(), fixture.hc_count).unwrap();
    assert_matches(&repeated, &case.repeated, "repeated embedding");

    // Malformed inputs are refused, not computed.
    assert!(gated_residual(&inputs[..3], config).is_err());
    let mut wrong = inputs.clone();
    wrong[1] = HostTensor::new([3], vec![0.0; 3]).unwrap();
    assert!(gated_residual(&wrong, config).is_err());
    assert!(
        gated_residual(
            &inputs,
            GatedResidual {
                hc_count: 1,
                ..config
            }
        )
        .is_err()
    );
    assert!(repeat_features(&case.embedding.tensor(), 0).is_err());
}

//! `support-qwen38-flash-next` lot 3: the Reference semantics of Qwen4-Exp's
//! blocks against fixtures produced by running Transformers' `qwen4_exp`
//! (pinned in design D7) on synthetic weights:
//! `fixtures/generate-qwen4-exp-reference.py`.

use magnetar_provider_api::{
    GatedResidual, HostTensor, QsaSelect, gated_residual, gated_residual_inject, qsa_select,
    repeat_features,
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
struct QsaCase {
    head_count: usize,
    head_dim: usize,
    budget: usize,
    compress_ratio: usize,
    rotary_dim: usize,
    rope_base: f32,
    key_norm_weight: Flat,
    queries: Flat,
    keys: Flat,
    /// The `[T, K]` mask the module builds, one per query.
    selected: Flat,
}

#[derive(Deserialize)]
struct Fixture {
    hc_count: usize,
    epsilon: f32,
    gated_residual: GatedResidualCase,
    qsa_indexer: QsaCase,
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

/// The QSA selection against the pinned indexer: for every one of 20
/// queries, the set of tokens kept equals the module's mask, where the
/// 8-token budget in blocks of 2 prunes from the tenth query on. A query
/// with fewer visible tokens than the budget keeps them all.
#[test]
fn the_qsa_selection_matches_the_pinned_transformers_indexer() {
    let fixture = fixture();
    let case = &fixture.qsa_indexer;
    let config = QsaSelect {
        head_count: case.head_count,
        head_dim: case.head_dim,
        compress_ratio: case.compress_ratio,
        budget: case.budget,
        epsilon: fixture.epsilon,
        weight_offset: 1.0,
        rope_base: case.rope_base,
        rotary_dim: case.rotary_dim,
    };
    let inputs = [
        case.queries.tensor(),
        case.keys.tensor(),
        case.key_norm_weight.tensor(),
    ];
    let selection = qsa_select(&inputs, config).unwrap();
    let tokens = case.keys.shape[0] as usize;
    assert_eq!(selection.shape, [tokens as u64, config.slots() as u64]);
    let mut pruned = 0;
    for query in 0..tokens {
        let kept: Vec<usize> = selection.data[query * config.slots()..(query + 1) * config.slots()]
            .iter()
            .filter(|&&index| index >= 0.0)
            .map(|&index| index as usize)
            .collect();
        assert!(kept.windows(2).all(|pair| pair[0] < pair[1]), "ascending");
        let expected: Vec<usize> = (0..tokens)
            .filter(|&token| case.selected.data[query * tokens + token] != 0.0)
            .collect();
        assert_eq!(kept, expected, "query {query}");
        pruned += usize::from(kept.len() < query + 1);
    }
    // The fixture would prove nothing if it never pruned.
    assert!(pruned >= 8, "{pruned} pruned queries");

    // Malformed geometry and inputs are refused.
    assert!(qsa_select(&inputs[..2], config).is_err());
    assert!(
        qsa_select(
            &inputs,
            QsaSelect {
                budget: 7,
                ..config
            }
        )
        .is_err()
    );
    assert!(
        qsa_select(
            &inputs,
            QsaSelect {
                rotary_dim: config.head_dim + 2,
                ..config
            }
        )
        .is_err()
    );
}

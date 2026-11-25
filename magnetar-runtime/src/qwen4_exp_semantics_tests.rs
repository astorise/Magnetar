//! `support-qwen38-flash-next` lot 3: the Reference semantics of Qwen4-Exp's
//! blocks against fixtures produced by running Transformers' `qwen4_exp`
//! (pinned in design D7) on synthetic weights:
//! `fixtures/generate-qwen4-exp-reference.py`.

use magnetar_provider_api::{
    GatedResidual, HostTensor, NgramEmbedding, NgramPle, QsaSelect, gated_residual,
    gated_residual_inject, ngram_embedding, ngram_ple, qsa_select, repeat_features,
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
struct PleCase {
    ngram_size: usize,
    heads_per_ngram: usize,
    vocab_size: u64,
    vocab_base: u64,
    layer_index: u64,
    seed: u64,
    eos_token_id: i64,
    conv_kernel: usize,
    conv_dilation: usize,
    token_ids: Vec<f32>,
    table: Flat,
    head_vocab_sizes: Vec<u64>,
    multipliers: Vec<i64>,
    embeddings: Flat,
    hidden: Flat,
    key_proj: Flat,
    value_proj: Flat,
    norm_key: Flat,
    norm_query: Flat,
    norm_conv: Flat,
    conv: Flat,
    output: Flat,
    chunked_output: Flat,
}

#[derive(Deserialize)]
struct Fixture {
    hc_count: usize,
    epsilon: f32,
    gated_residual: GatedResidualCase,
    qsa_indexer: QsaCase,
    ngram_ple: PleCase,
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

/// The hashed n-gram lookup and the PLE layer against the pinned modules,
/// on a second PLE layer (so its primes and multipliers differ from the
/// first's) over a sequence with EOS tokens that end n-gram segments; then
/// the same sequence in two chunks, the context and the convolution history
/// carried between them.
#[test]
fn the_ngram_ple_matches_the_pinned_transformers_layer() {
    let fixture = fixture();
    let case = &fixture.ngram_ple;
    let hash = NgramEmbedding {
        ngram_size: case.ngram_size,
        heads_per_ngram: case.heads_per_ngram,
        vocab_size: case.vocab_size,
        vocab_base: case.vocab_base,
        layer_index: case.layer_index,
        seed: case.seed,
        eos_token_id: case.eos_token_id,
    };
    assert_eq!(
        hash.head_vocab_sizes(),
        case.head_vocab_sizes,
        "head primes"
    );
    assert_eq!(hash.multipliers(), case.multipliers, "hash multipliers");

    let tokens = case.token_ids.len();
    let ids = HostTensor::new([tokens as u64], case.token_ids.clone()).unwrap();
    let start = HostTensor::new(
        [hash.context_len() as u64],
        vec![case.eos_token_id as f32; hash.context_len()],
    )
    .unwrap();
    let [embeddings, context] =
        ngram_embedding(&[ids.clone(), start.clone(), case.table.tensor()], hash).unwrap();
    assert_matches(&embeddings, &case.embeddings, "n-gram embeddings");
    assert_eq!(
        context.data,
        &case.token_ids[tokens - hash.context_len()..],
        "the next context is the last tokens"
    );

    let ple = NgramPle {
        hc_count: fixture.hc_count,
        epsilon: fixture.epsilon,
        weight_offset: 1.0,
        conv_kernel: case.conv_kernel,
        conv_dilation: case.conv_dilation,
    };
    let channels = case.hidden.shape[1];
    let history = || {
        HostTensor::new(
            [channels, ple.state_len() as u64],
            vec![0.0; channels as usize * ple.state_len()],
        )
        .unwrap()
    };
    let weights = |hidden: HostTensor, embeddings: HostTensor, history: HostTensor| {
        vec![
            hidden,
            embeddings,
            case.key_proj.tensor(),
            case.value_proj.tensor(),
            case.norm_key.tensor(),
            case.norm_query.tensor(),
            case.norm_conv.tensor(),
            case.conv.tensor(),
            history,
        ]
    };
    let [output, _] = ngram_ple(
        &weights(case.hidden.tensor(), embeddings.clone(), history()),
        ple,
    )
    .unwrap();
    assert_matches(&output, &case.output, "PLE output");

    // Two chunks of 5 and 7 tokens, state carried in both layers.
    let rows = |tensor: &HostTensor, from: usize, to: usize| {
        let width = tensor.shape[1] as usize;
        HostTensor::new(
            [(to - from) as u64, width as u64],
            tensor.data[from * width..to * width].to_vec(),
        )
        .unwrap()
    };
    let hidden = case.hidden.tensor();
    let mut carried_context = start.clone();
    let mut carried_history = history();
    let mut chunked = Vec::new();
    for (from, to) in [(0, 5), (5, tokens)] {
        let chunk_ids =
            HostTensor::new([(to - from) as u64], case.token_ids[from..to].to_vec()).unwrap();
        let [chunk_embeddings, next_context] =
            ngram_embedding(&[chunk_ids, carried_context, case.table.tensor()], hash).unwrap();
        let [chunk_output, next_history] = ngram_ple(
            &weights(rows(&hidden, from, to), chunk_embeddings, carried_history),
            ple,
        )
        .unwrap();
        chunked.extend(chunk_output.data);
        carried_context = next_context;
        carried_history = next_history;
    }
    let chunked = HostTensor::new(case.output.shape.clone(), chunked).unwrap();
    assert_matches(&chunked, &case.chunked_output, "chunked PLE output");
    assert_matches(&chunked, &case.output, "chunked equals whole");

    // Malformed inputs are refused.
    assert!(
        ngram_embedding(
            &[ids, start, case.table.tensor()],
            NgramEmbedding {
                vocab_base: 10_000_000,
                ..hash
            }
        )
        .is_err()
    );
    let mut short = weights(case.hidden.tensor(), embeddings, history());
    short.pop();
    assert!(ngram_ple(&short, ple).is_err());
}

//! `support-qwen38-flash-next` lot 3: the Reference semantics of Qwen4-Exp's
//! blocks against fixtures produced by running Transformers' `qwen4_exp`
//! (pinned in design D7) on synthetic weights:
//! `fixtures/generate-qwen4-exp-reference.py`.

use magnetar_provider_api::{
    GatedResidual, HostTensor, MoeRouting, NgramEmbedding, NgramPle, QsaSelect, gated_residual,
    gated_residual_inject, moe_experts, ngram_embedding, ngram_ple, qsa_select, repeat_features,
};
use serde::Deserialize;

#[derive(Deserialize)]
pub(crate) struct Flat {
    pub(crate) shape: Vec<u64>,
    pub(crate) data: Vec<f32>,
}

impl Flat {
    pub(crate) fn tensor(&self) -> HostTensor {
        HostTensor::new(self.shape.clone(), self.data.clone()).unwrap()
    }
}

#[derive(Deserialize)]
pub(crate) struct GatedResidualCase {
    pub(crate) norm_weight: Flat,
    pub(crate) down: Flat,
    pub(crate) up: Flat,
    pub(crate) inject: Flat,
    pub(crate) input: Flat,
    pub(crate) mixed: Flat,
    pub(crate) injection: Flat,
    pub(crate) block_output: Flat,
    pub(crate) after_block: Flat,
    pub(crate) final_mixed: Flat,
    pub(crate) embedding: Flat,
    pub(crate) repeated: Flat,
}

#[derive(Deserialize)]
pub(crate) struct QsaCase {
    pub(crate) head_count: usize,
    pub(crate) head_dim: usize,
    pub(crate) budget: usize,
    pub(crate) compress_ratio: usize,
    pub(crate) rotary_dim: usize,
    pub(crate) rope_base: f32,
    pub(crate) key_norm_weight: Flat,
    pub(crate) queries: Flat,
    pub(crate) keys: Flat,
    /// The `[T, K]` mask the module builds, one per query.
    pub(crate) selected: Flat,
}

#[derive(Deserialize)]
pub(crate) struct PleCase {
    pub(crate) ngram_size: usize,
    pub(crate) heads_per_ngram: usize,
    pub(crate) vocab_size: u64,
    pub(crate) vocab_base: u64,
    pub(crate) layer_index: u64,
    pub(crate) seed: u64,
    pub(crate) eos_token_id: i64,
    pub(crate) conv_kernel: usize,
    pub(crate) conv_dilation: usize,
    pub(crate) token_ids: Vec<f32>,
    pub(crate) table: Flat,
    pub(crate) head_vocab_sizes: Vec<u64>,
    pub(crate) multipliers: Vec<i64>,
    pub(crate) embeddings: Flat,
    pub(crate) hidden: Flat,
    pub(crate) key_proj: Flat,
    pub(crate) value_proj: Flat,
    pub(crate) norm_key: Flat,
    pub(crate) norm_query: Flat,
    pub(crate) norm_conv: Flat,
    pub(crate) conv: Flat,
    pub(crate) output: Flat,
    pub(crate) chunked_output: Flat,
}

#[derive(Deserialize)]
pub(crate) struct MoeCase {
    pub(crate) experts: u64,
    pub(crate) top_k: u64,
    pub(crate) norm_topk_prob: bool,
    pub(crate) input: Flat,
    pub(crate) router: Flat,
    pub(crate) gate_up_proj: Flat,
    pub(crate) down_proj: Flat,
    pub(crate) shared_gate_proj: Flat,
    pub(crate) shared_up_proj: Flat,
    pub(crate) shared_down_proj: Flat,
    pub(crate) shared_expert_gate: Flat,
    pub(crate) output: Flat,
}

#[derive(Deserialize)]
pub(crate) struct Fixture {
    pub(crate) moe_block: MoeCase,
    pub(crate) hc_count: usize,
    pub(crate) epsilon: f32,
    pub(crate) gated_residual: GatedResidualCase,
    pub(crate) qsa_indexer: QsaCase,
    pub(crate) ngram_ple: PleCase,
}

pub(crate) fn fixture() -> Fixture {
    serde_json::from_str(include_str!("../fixtures/qwen4-exp-reference.json")).unwrap()
}

pub(crate) fn assert_matches(actual: &HostTensor, expected: &Flat, what: &str) {
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

/// `x [rows, in] · weightᵀ` for a checkpoint `[out, in]` weight.
fn linear(x: &[f32], width: usize, weight: &Flat) -> Vec<f32> {
    let out = weight.shape[0] as usize;
    x.chunks_exact(width)
        .flat_map(|row| {
            (0..out).map(move |o| {
                weight.data[o * width..(o + 1) * width]
                    .iter()
                    .zip(row)
                    .map(|(w, x)| w * x)
                    .sum::<f32>()
            })
        })
        .collect()
}

/// Qwen4-Exp's MoE block needs no new operator: the existing `moe-experts`
/// routing (softmax over every expert, top-k, renormalized, a tie to the
/// lower expert) and combination, plus a shared expert behind a sigmoid
/// gate, equal the pinned `Qwen4ExpTextSparseMoeBlock`. The checkpoint's
/// stacked `gate_up_proj [E, 2I, H]` and `down_proj [E, H, I]` map onto the
/// operator's `gate`/`up [H, E*I]` and `down [I, E*H]`.
#[test]
fn the_moe_block_matches_through_the_existing_operator() {
    let case = fixture().moe_block;
    assert!(case.norm_topk_prob, "Qwen4-Exp renormalizes its top-k");
    let (experts, top_k) = (case.experts as usize, case.top_k as usize);
    let (tokens, hidden) = (case.input.shape[0] as usize, case.input.shape[1] as usize);
    let inter = case.gate_up_proj.shape[1] as usize / 2;
    let (mut gate, mut up) = (
        vec![0.0; hidden * experts * inter],
        vec![0.0; hidden * experts * inter],
    );
    let mut down = vec![0.0; inter * experts * hidden];
    for e in 0..experts {
        for i in 0..inter {
            for h in 0..hidden {
                gate[h * experts * inter + e * inter + i] =
                    case.gate_up_proj.data[(e * 2 * inter + i) * hidden + h];
                up[h * experts * inter + e * inter + i] =
                    case.gate_up_proj.data[(e * 2 * inter + inter + i) * hidden + h];
            }
        }
        for h in 0..hidden {
            for i in 0..inter {
                down[i * experts * hidden + e * hidden + h] =
                    case.down_proj.data[(e * hidden + h) * inter + i];
            }
        }
    }
    let tensor = |shape: [usize; 2], data: Vec<f32>| {
        HostTensor::new([shape[0] as u64, shape[1] as u64], data).unwrap()
    };
    let input = case.input.tensor();
    let logits = tensor(
        [tokens, experts],
        linear(&case.input.data, hidden, &case.router),
    );
    let routed = moe_experts(
        &input,
        &logits,
        [
            &tensor([hidden, experts * inter], gate),
            &tensor([hidden, experts * inter], up),
            &tensor([inter, experts * hidden], down),
        ],
        MoeRouting {
            experts: case.experts,
            top_k: case.top_k,
            normalize: case.norm_topk_prob,
        },
    )
    .unwrap();
    assert_eq!(top_k, 2, "the fixture routes to two of four experts");

    // The shared expert: silu(gate) * up, down, scaled by sigmoid(gate_w · x).
    let shared_gate = linear(&case.input.data, hidden, &case.shared_gate_proj);
    let shared_up = linear(&case.input.data, hidden, &case.shared_up_proj);
    let activated: Vec<f32> = shared_gate
        .iter()
        .zip(&shared_up)
        .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
        .collect();
    let shared = linear(
        &activated,
        case.shared_gate_proj.shape[0] as usize,
        &case.shared_down_proj,
    );
    let scale = linear(&case.input.data, hidden, &case.shared_expert_gate);
    let mut out = routed.data.clone();
    for (token, row) in out.chunks_exact_mut(hidden).enumerate() {
        let gate = 1.0 / (1.0 + (-scale[token]).exp());
        for (feature, value) in row.iter_mut().enumerate() {
            *value += gate * shared[token * hidden + feature];
        }
    }
    let actual = HostTensor::new(case.output.shape.clone(), out).unwrap();
    assert_matches(&actual, &case.output, "MoE block output");
}

//! Unit tests for the parent module.
//!
//! Shared fixtures live in `first_native_runtime/tests.rs`.

use super::super::tests::*;
use super::*;

#[test]
fn chunk_lengths_parse_and_fall_back_to_the_default() {
    assert_eq!(parse_prefill_chunk(None), DEFAULT_PREFILL_CHUNK);
    assert_eq!(parse_prefill_chunk(Some(" 512 ")), 512);
    assert_eq!(parse_prefill_chunk(Some("0")), 0);
    assert_eq!(parse_prefill_chunk(Some("-1")), DEFAULT_PREFILL_CHUNK);
    assert_eq!(parse_prefill_chunk(Some("many")), DEFAULT_PREFILL_CHUNK);
}

#[test]
fn passes_cover_the_prompt_and_only_the_last_is_short() {
    assert_eq!(chunk_spans(11, 4), vec![(0, 4), (4, 4), (8, 3)]);
    assert_eq!(chunk_spans(8, 4), vec![(0, 4), (4, 4)]);
    assert_eq!(chunk_spans(3, 4), vec![(0, 3)]);
    assert!(chunk_spans(0, 4).is_empty());
    assert!(chunk_spans(5, 0).is_empty());
    // The last pass's padding rows are written too.
    assert_eq!(chunked_rows(11, 4), 12);
    assert_eq!(chunked_rows(8, 4), 8);
}

/// A decoder's prefill reads K and V from its KV history; one whose
/// attention reads V from an edge outside it cannot continue a history.
#[test]
fn a_prefill_continues_history_only_when_attention_reads_it() {
    let fixture = e2e_fixture().expect("fixture builds");
    let graph = first_native_component_graphs_for_prompt(&fixture, 6)
        .expect("graphs build")
        .prefill;
    assert!(prefill_continues_history(&graph));

    let mut detached = graph.clone();
    let value = detached
        .nodes
        .values()
        .find(|node| node.operator.name() == "attention")
        .map(|node| node.inputs[2].clone())
        .expect("the graph has an attention node");
    detached
        .edges
        .get_mut(&value)
        .expect("the value edge exists")
        .kv_cache = None;
    assert!(!prefill_continues_history(&detached));
}

/// A production model's chunked prefills generate the tokens and the first
/// logits of its one-pass prefills, every pass sharing the chunk's graph.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn a_production_model_prefills_long_prompts_in_chunks_of_one_graph() {
    let (fixture, payload_source) = production_shaped_canonical_fixture();
    let mut model = ProductionLoadedModel::load(
        fixture.clone(),
        &payload_source,
        ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone()),
        Arc::new(ReferenceCpuProvider::new()),
    )
    .expect("production loading succeeds");
    model.set_prefill_buckets(Vec::new());
    let generate = |model: &mut ProductionLoadedModel, prompt: &str| {
        let mut first_logits: Vec<f32> = Vec::new();
        let generation = model
            .generate_streaming_with_logits(
                ProductionGenerationRequest {
                    prompt: PromptInput::PlainText(prompt.into()),
                    parameters: GenerationParameters::greedy(),
                    stop_conditions: StopConditions::default(),
                    max_new_tokens: Some(4),
                    max_generation_millis: None,
                },
                None,
                &mut |_| std::ops::ControlFlow::Continue(()),
                &mut |logits| {
                    if first_logits.is_empty() {
                        first_logits = logits.to_vec();
                    }
                },
            )
            .unwrap();
        (
            generation.result.output.usage.prompt_tokens as u64,
            generation.result.output.generated_token_ids,
            first_logits,
        )
    };
    let prompts = ["abcdefghijk", "abcdefgh", "hey you"];
    model.set_prefill_chunk(0);
    let one_pass: Vec<_> = prompts
        .iter()
        .map(|prompt| generate(&mut model, prompt))
        .collect();
    model.set_prefill_chunk(4);
    assert_eq!(model.prefill_chunk(), 4);
    for (prompt, (prompt_tokens, tokens, logits)) in prompts.iter().zip(&one_pass) {
        let (chunked_tokens, chunked_ids, chunked_logits) = generate(&mut model, prompt);
        assert_eq!(chunked_tokens, *prompt_tokens);
        assert_eq!(&chunked_ids, tokens, "{prompt}");
        assert_eq!(chunked_logits.len(), logits.len());
        let drift = chunked_logits
            .iter()
            .zip(logits)
            .map(|(chunked, whole)| (chunked - whole).abs())
            .fold(0.0_f32, f32::max);
        assert!(drift < 1e-5, "{prompt}: logits drift {drift}");
    }
    // Every prompt longer than the chunk ran the chunk's graph: no graph
    // longer than the chunk was built.
    assert!(one_pass[0].0 > 4 && one_pass[1].0 > 4, "{one_pass:?}");
    let lengths = model.graph_cache.prompt_lengths();
    assert!(lengths.contains(&4), "{lengths:?}");
    assert!(lengths.iter().all(|length| *length <= 4), "{lengths:?}");

    // A prompt of exactly the chunk's length runs the chunk's graph in one
    // pass, with stores of its own: only a bucket's prefills are pooled.
    let (exact_tokens, exact_ids, exact_logits) = &one_pass[1];
    model.set_prefill_chunk(*exact_tokens);
    let (tokens, ids, logits) = generate(&mut model, prompts[1]);
    assert_eq!(tokens, *exact_tokens);
    assert_eq!(&ids, exact_ids);
    assert_eq!(&logits, exact_logits);
    assert_eq!(model.kv_pool_bytes(), 0);
}

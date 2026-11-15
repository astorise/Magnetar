//! Unit tests for the parent module.
//!
//! Shared fixtures live in `first_native_runtime/tests.rs`.

use super::super::tests::*;
use super::*;

#[test]
fn bucket_lists_parse_sort_and_fall_back_to_the_defaults() {
    assert_eq!(
        parse_prefill_buckets(None),
        DEFAULT_PREFILL_BUCKETS.to_vec()
    );
    assert_eq!(parse_prefill_buckets(Some("64, 8,8,32")), vec![8, 32, 64]);
    assert_eq!(
        parse_prefill_buckets(Some("8,zero")),
        DEFAULT_PREFILL_BUCKETS.to_vec()
    );
    assert_eq!(
        parse_prefill_buckets(Some("0,8")),
        DEFAULT_PREFILL_BUCKETS.to_vec()
    );
    assert_eq!(
        parse_prefill_buckets(Some("")),
        DEFAULT_PREFILL_BUCKETS.to_vec()
    );
}

#[test]
fn buckets_never_exceed_the_context_length() {
    assert_eq!(
        buckets_within_context(vec![64, 0, 16, 16, 32], 32),
        vec![16, 32]
    );
    assert_eq!(buckets_within_context(vec![64, 16], 0), vec![16, 64]);
}

#[test]
fn a_prompt_takes_the_smallest_bucket_holding_it() {
    let buckets = [16, 32];
    assert_eq!(prefill_bucket(&buckets, 1), Some(16));
    assert_eq!(prefill_bucket(&buckets, 16), Some(16));
    assert_eq!(prefill_bucket(&buckets, 17), Some(32));
    assert_eq!(prefill_bucket(&buckets, 33), None);
    assert_eq!(prefill_bucket(&buckets, 0), None);
    assert_eq!(prefill_bucket(&[], 4), None);
}

#[test]
fn padding_repeats_the_last_prompt_token() {
    assert_eq!(pad_prompt_ids(&[5, 7], 4), vec![5.0, 7.0, 7.0, 7.0]);
    assert_eq!(pad_prompt_ids(&[5, 7], 2), vec![5.0, 7.0]);
}

#[test]
fn the_logits_row_gather_feeds_the_lm_head_one_row() {
    let fixture = e2e_fixture().expect("fixture builds");
    let graphs = first_native_component_graphs_for_prompt(&fixture, 8).unwrap();
    let gathered = insert_logits_row_gather(&graphs.prefill).expect("the LM head is a matmul");
    let logits = &gathered.edges[&TensorEdgeId::new("logits")];
    assert_eq!(
        logits.descriptor.shape.dimensions,
        vec![1, E2E_FIXTURE_VOCAB]
    );
    let head = &gathered.nodes[logits.producer.as_ref().unwrap()];
    assert_eq!(head.inputs[0].as_str(), LOGITS_ROW_HIDDEN);
    let select = &gathered.nodes[&ExecutionNodeId::new(LOGITS_ROW_NODE)];
    assert_eq!(select.operator.name(), "embedding");
    assert_eq!(select.inputs[0].as_str(), LOGITS_ROW_INPUT);
    assert_eq!(gathered.nodes.len(), graphs.prefill.nodes.len() + 1);
    first_native_graph_execution_order(&gathered).expect("the rewritten graph is acyclic");
    // A second gather is refused rather than stacked.
    assert!(insert_logits_row_gather(&gathered).is_none());
}

/// Runs a prefill of `prompt` and `decode_steps` decode steps on the
/// Reference CPU, bucketed with `buckets` or at the prompt's exact length
/// with `None`. Returns each step's logits and each layer's valid KV rows
/// after the prefill.
fn run(
    prompt: &[TokenId],
    buckets: Option<Vec<u64>>,
    decode_steps: usize,
) -> (Vec<Vec<f32>>, Vec<Option<u64>>) {
    run_with(prompt, buckets, true, decode_steps)
}

/// [`run`]; without `gather`, a bucket's graph keeps every row's logits.
fn run_with(
    prompt: &[TokenId],
    buckets: Option<Vec<u64>>,
    gather: bool,
    decode_steps: usize,
) -> (Vec<Vec<f32>>, Vec<Option<u64>>) {
    let fixture = e2e_fixture().expect("fixture builds");
    let mut runtime = build_runtime_with_model_execution_engine(&fixture);
    let (instance, _memory) = load_fixture_instance(&fixture, &mut runtime).unwrap();
    let tokenized = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::TokenIds(prompt.to_vec())),
        None,
    )
    .unwrap();
    let request = prepare_generation(
        &runtime,
        build_generation_request(
            GenerationRequestId::new("bucketed-prefill").unwrap(),
            None,
            GenerationModelReference::ModelInstance(instance.clone()),
            generation_tokenizer_reference(&fixture),
            tokenized,
            decode_steps + 1,
            GenerationParameters::greedy(),
            StopConditions::default(),
            StreamingMode::Disabled,
        ),
    )
    .unwrap();
    let prompt_tokens = request.input_token_ids.len() as u64;
    let mut engine = new_kv_lifecycle_engine(&fixture);
    let mut plans = match buckets {
        Some(buckets) => {
            let cache = ComponentGraphCache::default();
            cache.set_prefill_buckets(buckets);
            let length = cache.prefill_graph_length(prompt_tokens);
            let graphs = first_native_component_graphs_for_prompt(&fixture, length).unwrap();
            let graphs = Arc::new(if gather {
                cache.prepare(&runtime, graphs, length)
            } else {
                CachedGraphs::new(graphs)
            });
            assert_eq!(graphs.logits_row, gather && cache.is_prefill_bucket(length));
            let plans = graphs
                .prepared_plans(
                    &runtime,
                    &instance,
                    length,
                    &ProviderBinding::new(crate::REFERENCE_CPU_PROVIDER_NAME),
                )
                .unwrap();
            cache.insert(length, graphs);
            engine.graph_cache = Some(cache);
            plans
        }
        None => {
            first_native_plans_for_prompt(&runtime, &fixture, &instance, prompt_tokens).unwrap()
        }
    };
    let mut steps = Vec::new();
    let prefill = engine
        .execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))
        .unwrap();
    steps.push(prefill.logits.clone());
    engine
        .commit_generation_step(&mut runtime, &request, &[], 1, &prefill)
        .unwrap();
    let rows = engine.kv_states.lock().unwrap()
        [&E2eRuntimeModelExecutionEngine::kv_state_key(&request)]
        .layer_kv
        .values()
        .map(|layer| layer.rows)
        .collect();
    let mut generated = vec![1];
    for index in 1..=decode_steps {
        let step = engine
            .execute_generation_step(&mut runtime, &request, &generated, Some(&mut plans.decode))
            .unwrap();
        steps.push(step.logits.clone());
        let token = (index % 3 + 1) as TokenId;
        engine
            .commit_generation_step(&mut runtime, &request, &generated, token, &step)
            .unwrap();
        generated.push(token);
    }
    engine
        .release_request_state(&mut runtime, &request)
        .unwrap();
    (steps, rows)
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
        .unwrap()
}

/// A bucketed prefill's logits and every later decode step's match the
/// exact-length prefill's: its padding rows are neither sampled nor read,
/// its committed rows are the prompt's, and decoding writes over them.
/// Without the logits row gather, the engine reads row `n - 1` instead.
#[test]
fn a_bucketed_prefill_matches_the_exact_length_prefill() {
    let decode_steps = 5;
    let prompts = [&[3, 9, 4][..], &[3, 9, 4, 1, 1, 7, 2, 8], &[5; 11]];
    for (prompt, gather) in prompts
        .iter()
        .flat_map(|prompt| [(prompt, true), (prompt, false)])
    {
        let (expected, exact_rows) = run(prompt, None, decode_steps);
        let (bucketed, rows) = run_with(prompt, Some(vec![8, 16]), gather, decode_steps);
        assert_eq!(rows, exact_rows, "prompt {prompt:?}");
        assert!(rows.iter().all(|rows| *rows == Some(prompt.len() as u64)));
        assert_eq!(bucketed.len(), expected.len());
        for (step, (actual, expected)) in bucketed.iter().zip(&expected).enumerate() {
            assert_eq!(actual.len(), expected.len());
            assert_eq!(
                argmax(actual),
                argmax(expected),
                "prompt {prompt:?}, step {step}"
            );
            for (actual, expected) in actual.iter().zip(expected) {
                assert!(
                    (actual - expected).abs() <= 1e-4,
                    "prompt {prompt:?}, step {step}: {actual} vs {expected}"
                );
            }
        }
    }
}

/// A prompt longer than the largest bucket runs at its exact length.
#[test]
fn a_prompt_above_the_largest_bucket_runs_at_its_exact_length() {
    let cache = ComponentGraphCache::default();
    cache.set_prefill_buckets(vec![4, 8]);
    assert_eq!(cache.prefill_graph_length(3), 4);
    assert_eq!(cache.prefill_graph_length(9), 9);
    assert!(!cache.is_prefill_bucket(9));
    let prompt = [3, 9, 4, 1, 1, 7, 2, 8, 6];
    let (expected, _) = run(&prompt, None, 2);
    let (unbucketed, _) = run(&prompt, Some(vec![4, 8]), 2);
    assert_eq!(unbucketed, expected);
}

/// A production model's bucketed prefills generate the tokens of its
/// exact-length ones, and prompts of different lengths share their
/// bucket's graphs.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn a_production_model_shares_one_bucket_graph_across_prompt_lengths() {
    let (fixture, payload_source) = production_shaped_canonical_fixture();
    let mut model = ProductionLoadedModel::load(
        fixture.clone(),
        &payload_source,
        ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone()),
        Arc::new(ReferenceCpuProvider::new()),
    )
    .expect("production loading succeeds");
    let generate = |model: &mut ProductionLoadedModel, prompt: &str| {
        let generation = model
            .generate(
                ProductionGenerationRequest {
                    prompt: PromptInput::PlainText(prompt.into()),
                    parameters: GenerationParameters::greedy(),
                    stop_conditions: StopConditions::default(),
                    max_new_tokens: Some(4),
                    max_generation_millis: None,
                },
                None,
            )
            .unwrap();
        (
            generation.result.output.usage.prompt_tokens as u64,
            generation.result.output.generated_token_ids,
        )
    };
    let prompts = ["hi", "hey you", "abcdefghijk"];
    model.set_prefill_buckets(Vec::new());
    let exact: Vec<_> = prompts
        .iter()
        .map(|prompt| generate(&mut model, prompt))
        .collect();
    model.set_prefill_buckets(vec![16]);
    for (prompt, (prompt_tokens, tokens)) in prompts.iter().zip(&exact) {
        assert!(*prompt_tokens <= 16, "{prompt}: {prompt_tokens} tokens");
        assert_eq!(&generate(&mut model, prompt).1, tokens, "{prompt}");
    }
    assert_eq!(model.graph_cache.prompt_lengths(), vec![16]);
}

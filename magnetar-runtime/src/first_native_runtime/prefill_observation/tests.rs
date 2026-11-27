use super::super::tests::*;
use super::*;

#[test]
fn observed_prefill_matches_reference_and_releases_its_resources() {
    let (fixture, payload) = production_shaped_canonical_fixture();
    let mut reference = fixture.clone();
    reference.weights = e2e_fixture_weights(&reference.config).unwrap();
    let provider = Arc::new(ReferenceCpuProvider::new());
    let mut model = ProductionLoadedModel::load(
        fixture.clone(),
        &payload,
        ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone()),
        provider.clone(),
    )
    .unwrap();
    let stored = provider.executor().stored_tensor_count();
    let tracked = provider.executor().tracked_allocations().len();
    let caches = model.runtime.kv_caches().caches().count();

    for tokens in [&[5, 6, 7][..], &[8, 9][..], &[5, 6, 7][..]] {
        let expected = e2e_forward_hidden_states(&reference, tokens).unwrap();
        let actual = model
            .observe_prefill_edge(tokens, "edge.final_norm")
            .unwrap();
        assert_eq!(actual.shape, expected.shape);
        for (actual, expected) in actual.data.iter().zip(&expected.data) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
        model.runtime.memory().assert_allocation_totals_consistent();
        assert_eq!(provider.executor().stored_tensor_count(), stored);
        assert_eq!(provider.executor().tracked_allocations().len(), tracked);
        assert_eq!(model.runtime.kv_caches().caches().count(), caches);
    }

    model
        .generate(
            ProductionGenerationRequest {
                prompt: PromptInput::TokenIds(vec![5, 6, 7]),
                parameters: GenerationParameters::greedy(),
                stop_conditions: StopConditions::default(),
                max_new_tokens: Some(1),
                max_generation_millis: None,
            },
            None,
        )
        .expect("observation preserves the resident model for generation");
}

#[test]
fn observed_prefill_rejects_invalid_inputs_without_allocating() {
    let (fixture, payload) = production_shaped_canonical_fixture();
    let provider = Arc::new(ReferenceCpuProvider::new());
    let mut model = ProductionLoadedModel::load(
        fixture.clone(),
        &payload,
        ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone()),
        provider.clone(),
    )
    .unwrap();
    let stored = provider.executor().stored_tensor_count();
    let tracked = provider.executor().tracked_allocations().len();
    let caches = model.runtime.kv_caches().caches().count();
    for (tokens, edge) in [
        (vec![], "edge.final_norm"),
        (
            vec![fixture.config.architecture.vocabulary_size as u32],
            "edge.final_norm",
        ),
        (vec![1; 513], "edge.final_norm"),
        (vec![1], "edge.does_not_exist"),
        (vec![1], "weight.token_embedding"),
    ] {
        assert!(model.observe_prefill_edge(&tokens, edge).is_err());
        assert_eq!(provider.executor().stored_tensor_count(), stored);
        assert_eq!(provider.executor().tracked_allocations().len(), tracked);
        assert_eq!(model.runtime.kv_caches().caches().count(), caches);
    }
}

#[test]
fn observed_prefill_does_not_execute_the_later_generation_head() {
    let (fixture, payload) = production_shaped_canonical_fixture();
    let mut reference = fixture.clone();
    reference.weights = e2e_fixture_weights(&reference.config).unwrap();
    let provider = Arc::new(ReferenceCpuProvider::new());
    let mut model = ProductionLoadedModel::load(
        fixture.clone(),
        &payload,
        ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone()),
        provider.clone(),
    )
    .unwrap();
    let weight = model
        .runtime
        .model_instances()
        .instance(&model.instance)
        .unwrap()
        .definition()
        .resource_bindings
        .weights["lm_head"]
        .clone();
    let executor = provider.executor();
    let saved = executor.read_tensor(&weight).unwrap();
    assert!(executor.release_tensor(&weight));
    let expected = e2e_forward_hidden_states(&reference, &[5, 6, 7]).unwrap();
    let actual = model
        .observe_prefill_edge(&[5, 6, 7], "edge.final_norm")
        .unwrap();
    for (actual, expected) in actual.data.iter().zip(&expected.data) {
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }
    executor.write_tensor(weight, saved);
}

#[test]
fn failed_observed_prefill_releases_partial_execution() {
    let (fixture, payload) = production_shaped_canonical_fixture();
    let provider = Arc::new(ReferenceCpuProvider::new());
    let mut model = ProductionLoadedModel::load(
        fixture.clone(),
        &payload,
        ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone()),
        provider.clone(),
    )
    .unwrap();
    let weight = model
        .runtime
        .model_instances()
        .instance(&model.instance)
        .unwrap()
        .definition()
        .resource_bindings
        .weights["layers.0.mlp.down_proj"]
        .clone();
    let executor = provider.executor();
    let saved = executor.read_tensor(&weight).unwrap();
    assert!(executor.release_tensor(&weight));
    let stored = executor.stored_tensor_count();
    let tracked = executor.tracked_allocations().len();
    let caches = model.runtime.kv_caches().caches().count();
    let live = model
        .runtime
        .memory()
        .allocations()
        .filter(|allocation| model.runtime.memory().is_live_allocation(allocation.id))
        .count();

    assert!(
        model
            .observe_prefill_edge(&[5, 6, 7], "edge.final_norm")
            .is_err()
    );
    assert_eq!(executor.stored_tensor_count(), stored);
    assert_eq!(executor.tracked_allocations().len(), tracked);
    assert_eq!(model.runtime.kv_caches().caches().count(), caches);
    assert_eq!(
        model
            .runtime
            .memory()
            .allocations()
            .filter(|allocation| model.runtime.memory().is_live_allocation(allocation.id))
            .count(),
        live
    );
    model.runtime.memory().assert_allocation_totals_consistent();
    executor.write_tensor(weight, saved);
    model
        .observe_prefill_edge(&[5, 6, 7], "edge.final_norm")
        .unwrap();
}

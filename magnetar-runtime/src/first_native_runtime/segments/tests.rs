//! Unit tests for the parent module.
//!
//! Shared mock executors and fixtures live in `first_native_runtime/tests.rs`.

use super::super::tests::*;
use super::*;

/// `audit-complet-cuda-hot-path-2026-09-08` P1-1's generalized entrypoint
/// (`run_first_native_graph_with_provider`), sanity-checked in-crate
/// against Reference CPU before trusting it for an external CUDA
/// integration test -- if the generalization introduced a bug, this is
/// the fast place to catch it, not the slower cross-repo GPU loop.
/// Confirms: the dispatch actually succeeds through the fully generalized
/// path (`build_runtime_with_model_execution_engine_and_provider` ->
/// `load_fixture_instance_for_provider` -> `prepare_first_native_plan_
/// for_graph` with an explicit Provider -> `execute_first_native_graph`), the
/// resolved Provider is genuinely Reference CPU (not silently defaulted
/// there by some untouched hardcoded path), and every weight/output
/// resource's own recorded `ResourceAffinity`/`MemoryPlacement` agrees.
#[test]
fn run_first_native_graph_with_provider_matches_reference_cpu_e2e_dispatch() {
    let fixture = e2e_fixture().expect("fixture builds");
    let graphs =
        first_native_component_graphs_for_prompt(&fixture, 2).expect("component graphs build");
    let cache_id =
        KvCacheId::new("first-native-generalized-entrypoint-cache").expect("cache id is valid");
    let outcome = run_first_native_graph_with_provider(
        Arc::new(ReferenceCpuProvider::new()),
        &fixture,
        &graphs.prefill,
        &cache_id,
        &[1, 2],
    )
    .expect("a real prefill dispatch against Reference CPU through the generalized path succeeds");

    assert_eq!(outcome.dispatch.status, KernelResultStatus::Succeeded);
    assert_eq!(
        outcome.resolved_provider.as_str(),
        REFERENCE_CPU_PROVIDER_NAME,
        "the generalized entrypoint must resolve to the Provider it was actually given"
    );
    assert!(
        !outcome.bindings.is_empty(),
        "a real prefill dispatch must bind at least the 'logits' output edge"
    );

    // Every graph edge with a recorded residency (RoPE's KV-cache-tied
    // outputs, `rope_k` in particular, legitimately bypass
    // `admit_kernel_output` for their own separate KV Append write path
    // and so have none -- not a regression here, and not what this test is
    // about) must have a `MemoryPlacement` naming Reference CPU
    // specifically (`resolved_output_placement`'s provider-aware lookup,
    // not the empty-affinity placeholder `admit_kernel_output` separately
    // records for every output regardless of Provider) -- not left
    // unresolved, and not silently defaulted to some other Provider by an
    // untouched hardcoded path. At least one edge (`embedding`, the first
    // node, never KV-tied) must actually be checked, so this cannot pass
    // vacuously.
    let mut checked_edges = 0;
    for edge_id in outcome.bindings.keys() {
        let resource_id = TensorResourceId::new(format!("edge.{edge_id}"));
        let Some(residency) = outcome.runtime.memory().tensor_residency(&resource_id) else {
            continue;
        };
        checked_edges += 1;
        assert_eq!(
            residency.placement,
            MemoryPlacement::ProviderOwnedOpaque(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)),
            "edge '{edge_id}' must be Reference-CPU-placed"
        );
    }
    assert!(
        checked_edges > 0,
        "expected at least one bound edge to have a recorded residency"
    );
}

/// `audit-complet-cuda-hot-path-2026-09-08` P1-1: proves a fully custom,
/// genuinely grouped-query-shaped `FirstNativeModelConfig`
/// (`attention_head_count=4, kv_head_count=2`, which the one canonical E2E
/// fixture this crate ships is not) can run through the real first-native
/// pipeline end to end -- own manifest, own synthetic weights, own
/// Rust-constructed graph (`qwen_prefill_graph`, the same graph-building
/// recipe the Component-based path itself calls into) -- via
/// `run_first_native_graph_with_provider_and_weights`, sanity-checked
/// in-crate against Reference CPU before trusting the same pattern for an
/// external CUDA integration test. Confirms the graph's own `rope_q`/
/// `rope_k` nodes actually carry *different* `head_count` values (the GQA
/// shape itself, not just that dispatch succeeds), and that dispatch
/// against this shape actually succeeds -- proving the generalized
/// Prepared Plan/weight-materialization path handles a non-canonical
/// configuration, not just the one fixture it was written against.
#[test]
fn run_first_native_graph_with_provider_and_weights_handles_a_genuinely_gqa_shaped_config() {
    let architecture = first_native_architecture_metadata(8, 1, 4, 2, 2, 16, 258, 32);
    let identity = first_native_component_identity(
        ModelComponentId::new("gqa-integration-fixture").expect("static id is valid"),
        ModelComponentVersion::new(1, 0, 0),
        ModelComponentImplementationKind::WebAssemblyComponent,
    );
    let config = FirstNativeModelConfig::new(architecture, FirstNativeRopeConfig::standard(2));
    config.validate(&identity).expect("GQA config validates");
    let architecture_implementation = qwen_architecture_implementation(
        &identity,
        ModelArchitectureImplementationKind::ComponentBased,
    );
    let weights = e2e_fixture_weights(&config).expect("GQA fixture weights build");
    let manifest = e2e_fixture_manifest_from_weights(
        &config,
        &architecture_implementation.architecture,
        &weights,
    )
    .expect("GQA fixture manifest builds");
    let tokenizer = e2e_fixture_tokenizer().expect("fixture tokenizer builds");
    let descriptor = first_native_component_descriptor(identity.clone(), &config)
        .expect("GQA component descriptor builds");
    validate_first_native_model_artifact(&descriptor, &config, &manifest)
        .expect("GQA manifest matches its own descriptor");
    let fixture = E2eFixture {
        config,
        identity,
        architecture_implementation,
        manifest,
        tokenizer,
        weights,
    };

    let graph = qwen_prefill_graph(&fixture.config, &fixture.identity, 2, true)
        .expect("GQA prefill graph builds")
        .graph;
    let q_head_count = graph
        .nodes
        .get(&ExecutionNodeId::new("layer0.rope_q"))
        .and_then(|node| node.attributes.get("head_count"))
        .cloned();
    let k_head_count = graph
        .nodes
        .get(&ExecutionNodeId::new("layer0.rope_k"))
        .and_then(|node| node.attributes.get("head_count"))
        .cloned();
    assert_eq!(
        q_head_count,
        Some(OperatorAttributeValue::Integer(4)),
        "rope_q must carry the attention head count"
    );
    assert_eq!(
        k_head_count,
        Some(OperatorAttributeValue::Integer(2)),
        "rope_k must carry the (smaller) key/value head count -- the actual GQA shape"
    );
    assert_ne!(
        q_head_count, k_head_count,
        "this configuration is only a genuine GQA proof if Q and K actually differ"
    );

    let cache_id = KvCacheId::new("gqa-integration-fixture-cache").expect("cache id is valid");
    let outcome = run_first_native_graph_with_provider_and_weights(
        Arc::new(ReferenceCpuProvider::new()),
        &fixture,
        &fixture.weights,
        &graph,
        &cache_id,
        &[1, 2],
    )
    .expect("a real GQA-shaped prefill dispatch through the generalized path succeeds");
    assert_eq!(outcome.dispatch.status, KernelResultStatus::Succeeded);
    assert_eq!(
        outcome.resolved_provider.as_str(),
        REFERENCE_CPU_PROVIDER_NAME
    );
}

/// `implement-production-qwen-model-loading` task 8.8: at least two
/// materially different Qwen configurations must produce config-correct
/// graphs through the *real, compiled* Qwen Component -- not only the one
/// canonical tiny E2E fixture dimensions, and not only through
/// `qwen_prefill_graph`'s Rust-synthesized oracle path (which the test
/// directly above already covers for this same GQA shape). This proves the
/// Component itself -- via the `model-config` Capability -- derives a
/// different, still-correct graph for a materially different
/// architecture, matching `qwen_prefill_graph`'s independent oracle exactly
/// in node count and RoPE head-count attributes.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn real_qwen_component_produces_a_config_correct_graph_for_a_genuinely_gqa_shaped_config() {
    let architecture = first_native_architecture_metadata(8, 1, 4, 2, 2, 16, 258, 32);
    let identity = first_native_component_identity(
        ModelComponentId::new("gqa-real-component-fixture").expect("static id is valid"),
        ModelComponentVersion::new(1, 0, 0),
        ModelComponentImplementationKind::WebAssemblyComponent,
    );
    let config = FirstNativeModelConfig::new(architecture, FirstNativeRopeConfig::standard(2));
    config.validate(&identity).expect("GQA config validates");
    let architecture_implementation = qwen_architecture_implementation(
        &identity,
        ModelArchitectureImplementationKind::ComponentBased,
    );
    let weights = e2e_fixture_weights(&config).expect("GQA fixture weights build");
    let manifest = e2e_fixture_manifest_from_weights(
        &config,
        &architecture_implementation.architecture,
        &weights,
    )
    .expect("GQA fixture manifest builds");
    let tokenizer = e2e_fixture_tokenizer().expect("fixture tokenizer builds");
    let descriptor = first_native_component_descriptor(identity.clone(), &config)
        .expect("GQA component descriptor builds");
    validate_first_native_model_artifact(&descriptor, &config, &manifest)
        .expect("GQA manifest matches its own descriptor");
    let fixture = E2eFixture {
        config,
        identity,
        architecture_implementation,
        manifest,
        tokenizer,
        weights,
    };

    // The independent oracle -- same as the hand-built-graph test above --
    // is what the real Component's output is compared against.
    let oracle_graph = qwen_prefill_graph(&fixture.config, &fixture.identity, 2, true)
        .expect("GQA oracle prefill graph builds")
        .graph;

    let (component_graphs, _definition, _instance) =
        build_first_native_graphs_from_real_qwen_component(&fixture, 2)
            .expect("the real Component produces graphs for a GQA-shaped config");

    assert_eq!(
        component_graphs.prefill.nodes.len(),
        oracle_graph.nodes.len(),
        "the real Component's prefill graph must have the same node count as the oracle \
         for this architecture"
    );
    let component_q_head_count = component_graphs
        .prefill
        .nodes
        .get(&ExecutionNodeId::new("layer0.rope_q"))
        .and_then(|node| node.attributes.get("head_count"))
        .cloned();
    let component_k_head_count = component_graphs
        .prefill
        .nodes
        .get(&ExecutionNodeId::new("layer0.rope_k"))
        .and_then(|node| node.attributes.get("head_count"))
        .cloned();
    assert_eq!(
        component_q_head_count,
        Some(OperatorAttributeValue::Integer(4)),
        "the real Component's rope_q must carry the attention head count from model-config"
    );
    assert_eq!(
        component_k_head_count,
        Some(OperatorAttributeValue::Integer(2)),
        "the real Component's rope_k must carry the (smaller) key/value head count -- \
         the actual GQA shape, derived from model-config rather than a compiled-in constant"
    );

    let cache_id = KvCacheId::new("gqa-real-component-fixture-cache").expect("cache id is valid");
    let outcome = run_first_native_graph_with_provider_and_weights(
        Arc::new(ReferenceCpuProvider::new()),
        &fixture,
        &fixture.weights,
        &component_graphs.prefill,
        &cache_id,
        &[1, 2],
    )
    .expect("a real GQA-shaped prefill dispatch through the real Component's own graph succeeds");
    assert_eq!(outcome.dispatch.status, KernelResultStatus::Succeeded);
    assert_eq!(
        outcome.resolved_provider.as_str(),
        REFERENCE_CPU_PROVIDER_NAME
    );
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn e2e_two_segment_split_decode_step_matches_full_graph_decode() {
    check_two_segment_split_decode_step_matches_full_graph_decode().expect(
        "a real decode step split across two segment Model Instances, each dispatched twice \
         (prefill then decode) with its own KV state threaded forward, must bit-for-bit match \
         the full, unsegmented graph's own decode step",
    );
}

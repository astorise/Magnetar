//! Unit tests for the parent module.
//!
//! Shared mock executors and fixtures live in `first_native_runtime/tests.rs`.

use super::super::tests::*;
use super::*;

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn e2e_two_segment_split_produces_identical_output_to_full_graph() {
    check_two_segment_split_produces_identical_output_to_full_graph().expect(
        "splitting a real forward pass into two segment Model Instances, bridged \
                  through a real hidden-state hand-off, must bit-for-bit match running the \
                  same prompt through the one full graph",
    );
}

/// Consolidates every assertion that depends on `QWEN_REAL_COMPONENT_BYTES`'s
/// digest into one sequential test, deliberately -- not split across
/// several `#[test]` functions. `REGISTERED_COMPONENT_RUNTIMES` is one
/// process-wide registry shared by every test in this binary, keyed by real
/// content digest; every one of these assertions registers (or attempts to
/// register) the *same* fixture bytes, so the same digest.
///
/// Splitting the untrusted-rejection assertions into their own `#[test]`
/// functions used to be genuinely racy under Rust's default parallel test
/// execution: whichever test's *trusted* registration ran first populated
/// the shared cache entry, after which a *later*, deliberately-untrusted
/// registration attempt for that same digest hit the idempotent-return fast
/// path before ever re-evaluating trust, making the "untrusted artifacts
/// are rejected" assertion pass or fail based on test scheduling instead of
/// behavior -- caught by a real CI failure (this exact race), not found by
/// inspection. That fast path turned out not to be only a test-scheduling
/// hazard: it was the identical bug in production code (Tachyon
/// integration audit MAG-03, tracked as #71) -- `register_inference_component_artifact`'s
/// cache hit returned `Ok(digest)` without ever consulting *this* caller's
/// own `trust`, so any caller could receive an artifact only some other,
/// earlier caller had actually authorized. Fixed by re-evaluating trust
/// against the cache hit on every call, which also means the scheduling
/// race described above can no longer happen -- the mid-test
/// untrusted-after-cached assertion below now passes deterministically
/// regardless of the order these tests run in. Assertions stay
/// consolidated here anyway, both for locality (everything about this one
/// digest's lifecycle in one place) and because splitting them still
/// wouldn't be free of shared-state ordering to reason about.
///
/// This same reasoning is why the Tachyon integration audit MAG-07 proof
/// (two distinct, independently-registered Components served at once) lives
/// at the end of this function instead of its own `#[test]`: a second test
/// that trusts and registers `QWEN_REAL_COMPONENT_BYTES` under its own
/// fresh `ComponentTrustStore` would still share this function's own
/// digest-keyed cache entry.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn register_inference_component_artifact_enforces_trust_is_idempotent_and_matches_the_singleton_path()
 {
    let expected_digest = ComponentDigest::sha256(QWEN_REAL_COMPONENT_BYTES);

    // No trust granted at all: real bytes, real (matching) declared digest,
    // but an empty ComponentTrustStore -- must be rejected, never silently
    // accepted just because the artifact is well-formed. Must run before
    // any trusted registration below (see this test's own doc comment).
    let untrusted = register_inference_component_artifact(
        QWEN_REAL_COMPONENT_BYTES.to_vec(),
        QWEN_REAL_COMPONENT_MANIFEST_BYTES.to_vec(),
        &ComponentTrustStore::default(),
    );
    assert!(
        untrusted.is_err(),
        "an artifact trusted by nothing must not register: {untrusted:?}"
    );

    // The caller's own trust store, naming this artifact's real digest --
    // never `QWEN_REAL_COMPONENT_DIGEST` or any other hardcoded constant.
    let trust = ComponentTrustStore::default().trust_digest(&expected_digest.value);
    let digest = register_inference_component_artifact(
        QWEN_REAL_COMPONENT_BYTES.to_vec(),
        QWEN_REAL_COMPONENT_MANIFEST_BYTES.to_vec(),
        &trust,
    )
    .expect("a caller-trusted, well-formed artifact must register")
    .digest;
    assert_eq!(
        digest, expected_digest,
        "the registry must key by the artifact's own real sha256, not a claim"
    );

    // Idempotent: re-registering the identical bytes under the same trust
    // is a harmless no-op, not an error, and returns the same digest.
    let second = register_inference_component_artifact(
        QWEN_REAL_COMPONENT_BYTES.to_vec(),
        QWEN_REAL_COMPONENT_MANIFEST_BYTES.to_vec(),
        &trust,
    )
    .expect("re-registering the same bytes is a harmless no-op, not an error")
    .digest;
    assert_eq!(digest, second);

    // Tachyon integration audit MAG-03 / #71: this digest is now cached
    // (registered and trusted above), so this call exercises exactly the
    // cache-hit path -- and must still be rejected for a caller whose own
    // trust store does not name it, the same as the untrusted call at the
    // top of this test did before anything was cached. Before the MAG-03
    // fix, the cache-hit fast path returned `Ok(digest)` unconditionally
    // without ever consulting `trust`, so a caller who authorized nothing
    // would silently receive an artifact only some *other*, earlier caller
    // had actually authorized -- an authorization bypass, not merely the
    // test-scheduling hazard this function's own doc comment above
    // describes. Reject-then-cache-then-reject-again in one sequential
    // test (like every other assertion here) rather than splitting the
    // second rejection into its own `#[test]`, for the identical
    // shared-registry reason.
    let untrusted_after_cached = register_inference_component_artifact(
        QWEN_REAL_COMPONENT_BYTES.to_vec(),
        QWEN_REAL_COMPONENT_MANIFEST_BYTES.to_vec(),
        &ComponentTrustStore::default(),
    );
    assert!(
        untrusted_after_cached.is_err(),
        "a caller-trusted digest must not become implicitly trusted for a \
         later caller whose own trust store authorizes nothing, just \
         because the compiled runtime is already cached: {untrusted_after_cached:?}"
    );

    // The load-bearing correctness proof for this generic registry: a
    // Component registered above (a caller-supplied digest/trust, no
    // hardcoded Qwen constant anywhere in the call) produces *exactly* the
    // same graphs -- same operator-sequence hashes, not just the same node
    // counts -- as the pre-existing, hardcoded single-Qwen-singleton path
    // (`build_first_native_graphs_from_real_qwen_component`) does for the
    // identical underlying Component bytes and the identical fixture
    // config/identity. Graph production itself was never Qwen-specific;
    // this proves the generic entry point reaches the exact same real
    // behavior, not a parallel implementation that merely looks similar.
    let fixture = e2e_fixture().expect("fixture builds");
    let prompt_token_count = 2u64;

    let (via_named, _definition, _instance) = build_first_native_graphs_from_named_component(
        &digest,
        &architecture_config_from_first_native_model_config(&fixture.config),
        &fixture.identity,
        prompt_token_count,
    )
    .expect("named-component graph production succeeds");
    let (via_singleton, _definition, _instance) =
        build_first_native_graphs_from_real_qwen_component(&fixture, prompt_token_count)
            .expect("singleton-path graph production succeeds");

    assert_eq!(
        via_named.prefill_node_count,
        via_singleton.prefill_node_count
    );
    assert_eq!(via_named.decode_node_count, via_singleton.decode_node_count);
    let named_prefill_hash =
        qwen_operator_sequence_hash(&qwen_graph_operator_codes(&via_named.prefill).unwrap());
    let singleton_prefill_hash =
        qwen_operator_sequence_hash(&qwen_graph_operator_codes(&via_singleton.prefill).unwrap());
    assert_eq!(named_prefill_hash, singleton_prefill_hash);
    let named_decode_hash =
        qwen_operator_sequence_hash(&qwen_graph_operator_codes(&via_named.decode).unwrap());
    let singleton_decode_hash =
        qwen_operator_sequence_hash(&qwen_graph_operator_codes(&via_singleton.decode).unwrap());
    assert_eq!(named_decode_hash, singleton_decode_hash);

    // Tachyon integration audit MAG-07: the registry must genuinely serve a
    // *second*, structurally unrelated Component at the same time as the
    // real Qwen Component registered above -- not merely accept a
    // caller-supplied digest that happens to resolve to the one hardcoded
    // Qwen singleton. `SYNTHETIC_MINIMAL_COMPONENT_BYTES` implements the
    // same `model-component-graph-producer` world but with every decoder
    // layer omitted (`embedding -> rmsnorm -> matmul`), so its node count
    // and operator-sequence hash can never coincide with Qwen's.
    let synthetic_digest = ComponentDigest::sha256(SYNTHETIC_MINIMAL_COMPONENT_BYTES);
    assert_ne!(
        synthetic_digest, expected_digest,
        "the two fixtures must be genuinely distinct artifacts"
    );
    let synthetic_trust = ComponentTrustStore::default().trust_digest(&synthetic_digest.value);
    let registered_synthetic_digest = register_inference_component_artifact(
        SYNTHETIC_MINIMAL_COMPONENT_BYTES.to_vec(),
        SYNTHETIC_MINIMAL_COMPONENT_MANIFEST_BYTES.to_vec(),
        &synthetic_trust,
    )
    .expect("the synthetic minimal Component must also register, alongside Qwen")
    .digest;
    assert_eq!(registered_synthetic_digest, synthetic_digest);

    let (synthetic_graphs, _definition, _instance) =
        build_first_native_graphs_from_named_component(
            &synthetic_digest,
            &architecture_config_from_first_native_model_config(&fixture.config),
            &fixture.identity,
            prompt_token_count,
        )
        .expect("named-component graph production succeeds for the synthetic Component too");
    assert_ne!(
        via_named.prefill_node_count, synthetic_graphs.prefill_node_count,
        "a full Qwen layer's worth of nodes must never coincide with the layer-less synthetic graph"
    );
    assert_ne!(
        via_named.decode_node_count,
        synthetic_graphs.decode_node_count
    );
    let synthetic_prefill_hash =
        qwen_operator_sequence_hash(&qwen_graph_operator_codes(&synthetic_graphs.prefill).unwrap());
    assert_ne!(named_prefill_hash, synthetic_prefill_hash);

    // Building the synthetic Component's graphs must not disturb the real
    // Qwen Component's own registered runtime: building from the Qwen
    // digest again still succeeds and still matches the singleton path.
    let (qwen_graphs_again, _definition, _instance) = build_first_native_graphs_from_named_component(
        &digest,
        &architecture_config_from_first_native_model_config(&fixture.config),
        &fixture.identity,
        prompt_token_count,
    )
    .expect("the real Qwen Component's own runtime remains usable after another Component registers");
    let qwen_again_prefill_hash = qwen_operator_sequence_hash(
        &qwen_graph_operator_codes(&qwen_graphs_again.prefill).unwrap(),
    );
    assert_eq!(qwen_again_prefill_hash, singleton_prefill_hash);
}

/// Proves the generic Component registry genuinely serves a second,
/// *real* production model architecture family -- not merely a synthetic
/// fixture (see the registry test above) or a caller-supplied digest that
/// always resolves back to the one hardcoded Qwen singleton. The real
/// Llama Component (`components/llama`) implements graph-building logic
/// structurally identical to the real Qwen Component's own: both are real
/// instances of the same pre-norm/RoPE/grouped-query-attention/SwiGLU
/// decoder block (Qwen2's architecture is Llama's with an added QKV bias
/// term, not a different block shape), so this test proves two distinct
/// claims that together justify that design: (1) for the *same*
/// `architecture-config` the independently-compiled Llama binary produces
/// byte-identical graphs (node counts and operator-sequence hashes) to the
/// Qwen binary -- the contract's genericity is real, not merely a
/// documentation claim; and (2) the same Llama binary produces genuinely
/// different graphs when `attention-bias` differs, driven entirely by
/// `model-config`, never by a hardcoded branch in either Component --
/// a real Llama checkpoint's own bias-free config and a real Qwen2
/// checkpoint's own bias-bearing config reach this same code path
/// correctly.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn build_first_native_graphs_from_named_component_serves_a_real_second_architecture_family() {
    let llama_digest = ComponentDigest::sha256(LLAMA_REAL_COMPONENT_BYTES);
    let qwen_digest = ComponentDigest::sha256(QWEN_REAL_COMPONENT_BYTES);
    assert_ne!(
        llama_digest, qwen_digest,
        "independently-compiled Llama and Qwen Component binaries must have distinct real digests"
    );
    let trust = ComponentTrustStore::default().trust_digest(&llama_digest.value);
    let registered = register_inference_component_artifact(
        LLAMA_REAL_COMPONENT_BYTES.to_vec(),
        LLAMA_REAL_COMPONENT_MANIFEST_BYTES.to_vec(),
        &trust,
    )
    .expect("the real Llama Component must register")
    .digest;
    assert_eq!(registered, llama_digest);

    // e2e_fixture()'s own config has `attention_bias: false` (`FirstNativeModelConfig::
    // new`'s default) -- the same value a real, bias-free Llama checkpoint
    // would carry, and (not coincidentally) what the pre-existing Qwen
    // singleton path itself already builds graphs against elsewhere in
    // this file.
    let fixture = e2e_fixture().expect("fixture builds");
    assert!(
        !fixture.config.attention_bias,
        "this comparison assumes the fixture's own default (no QKV bias)"
    );
    let prompt_token_count = 2u64;

    let (llama_graphs, _definition, _instance) = build_first_native_graphs_from_named_component(
        &llama_digest,
        &architecture_config_from_first_native_model_config(&fixture.config),
        &fixture.identity,
        prompt_token_count,
    )
    .expect("named-component graph production succeeds for the real Llama Component");
    let (qwen_singleton_graphs, _definition, _instance) =
        build_first_native_graphs_from_real_qwen_component(&fixture, prompt_token_count)
            .expect("singleton-path graph production succeeds");

    assert_eq!(
        llama_graphs.prefill_node_count, qwen_singleton_graphs.prefill_node_count,
        "for the same bias-free config, the independently-compiled Llama and Qwen binaries must \
         produce the same node count -- their graph-building logic is the same generic decoder"
    );
    assert_eq!(
        llama_graphs.decode_node_count,
        qwen_singleton_graphs.decode_node_count
    );
    let llama_prefill_hash =
        qwen_operator_sequence_hash(&qwen_graph_operator_codes(&llama_graphs.prefill).unwrap());
    let qwen_singleton_prefill_hash = qwen_operator_sequence_hash(
        &qwen_graph_operator_codes(&qwen_singleton_graphs.prefill).unwrap(),
    );
    assert_eq!(
        llama_prefill_hash, qwen_singleton_prefill_hash,
        "the same operator sequence, not just the same node count"
    );

    // Now prove the same real Llama binary correctly reacts to a
    // bias-bearing config (what a real Qwen2 checkpoint would carry): more
    // nodes (one `add` per q/k/v bias), a different operator sequence --
    // driven entirely by `model-config`, not a hardcoded per-family branch
    // in the Component.
    let mut biased_config = fixture.config.clone();
    biased_config.attention_bias = true;
    let (llama_biased_graphs, _definition, _instance) =
        build_first_native_graphs_from_named_component(
            &llama_digest,
            &architecture_config_from_first_native_model_config(&biased_config),
            &fixture.identity,
            prompt_token_count,
        )
        .expect(
            "the same real Llama Component binary also handles a bias-bearing config correctly",
        );
    assert_ne!(
        llama_graphs.prefill_node_count, llama_biased_graphs.prefill_node_count,
        "attention_bias must add one bias node per q/k/v projection, per layer"
    );
    let llama_biased_prefill_hash = qwen_operator_sequence_hash(
        &qwen_graph_operator_codes(&llama_biased_graphs.prefill).unwrap(),
    );
    assert_ne!(llama_prefill_hash, llama_biased_prefill_hash);
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn build_first_native_graphs_from_named_component_fails_closed_for_an_unregistered_digest() {
    let never_registered = ComponentDigest::parse(
        "sha256",
        "0000000000000000000000000000000000000000000000000000000000000000",
    );
    let fixture = e2e_fixture().expect("fixture builds");
    let result = build_first_native_graphs_from_named_component(
        &never_registered,
        &architecture_config_from_first_native_model_config(&fixture.config),
        &fixture.identity,
        2,
    );
    match result {
        Err(E2eConformanceError::ModelComponentFailed { .. }) => {}
        Err(other) => panic!(
            "an unregistered digest must fail with ModelComponentFailed, got a different error: {other:?}"
        ),
        Ok(_) => {
            panic!("an unregistered digest must fail closed, not silently build a fallback graph")
        }
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn e2e_qwen_component_wasm_reported_wrong_operator_hash_fails_before_planning() {
    let wat = qwen_component_fixture_wat(
        r#"(func (export "prefill-node-count") (result i32) i32.const 19)"#,
        r#"(func (export "decode-node-count") (result i32) i32.const 19)"#,
        r#"(func (export "provider-authority-count") (result i32) i32.const 0)"#,
        r#"(func (export "prefill-operator-hash") (result i32) i32.const -559038737)"#,
        &format!(
            r#"(func (export "decode-operator-hash") (result i32) i32.const {})"#,
            QWEN_COMPONENT_FIXTURE_OPERATOR_HASH as i32
        ),
    );
    let (component_package, digest) = qwen_component_preflight_package(&wat, None);
    let request = trusted_preflight_request_for_temp_component(component_package, &digest);

    let fixture = e2e_fixture().expect("fixture builds");
    let mut runtime = build_runtime_with_model_execution_engine(&fixture);
    let (instance, _memory) =
        load_fixture_instance(&fixture, &mut runtime).expect("fixture instance loads");

    let preflight = validate_and_instantiate_qwen_component_before_first_native_planning(request)
        .expect("component with valid counts and one wrong hash still instantiates");
    let result =
        build_first_native_graphs_from_component_output(&fixture, 2, preflight.graph_semantics)
            .and_then(|graphs| {
                prepare_first_native_execution_plans(&runtime, &instance, graphs, 2)
            });

    assert!(
        matches!(
            result,
            Err(E2eConformanceError::GraphValidationFailed { .. })
        ),
        "Qwen Component's own wrong operator-hash export was not rejected"
    );
}

/// `model-config` 1.4.0's `rope-scaling`: the real Llama Component reads a
/// checkpoint's Llama 3 frequency scaling and declares it on every `rope`
/// node itself (the Runtime only forwards node attributes), while a
/// Component that does not apply it (the real Qwen Component) refuses to
/// build a graph for a scaled checkpoint -- it traps on a declared
/// `rope-scaling` (1.5.0), and a Component predating 1.4.0 would be
/// refused by the Runtime's own check -- instead of silently building an
/// unscaled graph.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn llama3_rope_scaling_is_declared_by_the_component_and_required_of_it() {
    let llama_digest = ComponentDigest::sha256(LLAMA_REAL_COMPONENT_BYTES);
    let trust = ComponentTrustStore::default().trust_digest(&llama_digest.value);
    register_inference_component_artifact(
        LLAMA_REAL_COMPONENT_BYTES.to_vec(),
        LLAMA_REAL_COMPONENT_MANIFEST_BYTES.to_vec(),
        &trust,
    )
    .expect("the real Llama Component must register");
    let scaling = crate::operator::RopeBandedFrequencyScaling {
        factor: 32.0,
        low_freq_factor: 1.0,
        high_freq_factor: 4.0,
        original_context_length: 8192,
    };
    let mut fixture = e2e_fixture().expect("fixture builds");
    fixture.config.rope.frequency_scaling = Some(crate::RopeFrequencyScaling::Banded(scaling));
    let architecture = architecture_config_from_first_native_model_config(&fixture.config);
    assert_eq!(
        architecture.rope_frequency_scaling, fixture.config.rope.frequency_scaling,
        "the configuration conversion keeps the scaling"
    );

    let (graphs, _definition, _instance) = build_first_native_graphs_from_named_component(
        &llama_digest,
        &architecture,
        &fixture.identity,
        2,
    )
    .expect("the Llama Component builds graphs for a Llama 3 scaled checkpoint");
    let mut rope_nodes = 0;
    for graph in [&graphs.prefill, &graphs.decode] {
        for node in graph.nodes.values() {
            if node.operator.name() == "rope" {
                rope_nodes += 1;
                assert_eq!(
                    crate::operator::RopeBandedFrequencyScaling::from_attributes(&node.attributes),
                    Ok(Some(scaling))
                );
            }
        }
    }
    assert!(rope_nodes > 0);

    let Err(error) = build_first_native_graphs_from_real_qwen_component(&fixture, 2) else {
        panic!("a Component that does not declare the scaling must be refused");
    };
    assert!(
        error.to_string().contains("frequency scaling")
            || error.to_string().starts_with("e2e-model-component-failed"),
        "unexpected error: {error}"
    );
}

/// `model-config` 1.5.0's `mixture-of-experts` (`execute-and-cache-moe-
/// experts` F02): for a checkpoint that declares one, the real Qwen
/// Component replaces every layer's dense MLP with `moe-experts` over the
/// router logits and stacked expert weights, plus the shared expert scaled
/// by `sigmoid` through `row-scale`; a dense checkpoint keeps its dense
/// MLP and no MoE node.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn the_qwen_component_builds_a_mixture_of_experts_graph_when_the_config_declares_one() {
    let digest = ComponentDigest::sha256(QWEN_REAL_COMPONENT_BYTES);
    let trust = ComponentTrustStore::default().trust_digest(&digest.value);
    register_inference_component_artifact(
        QWEN_REAL_COMPONENT_BYTES.to_vec(),
        QWEN_REAL_COMPONENT_MANIFEST_BYTES.to_vec(),
        &trust,
    )
    .expect("the real Qwen Component must register");
    let fixture = e2e_fixture().expect("fixture builds");
    let dense = architecture_config_from_first_native_model_config(&fixture.config);
    let moe = crate::MixtureOfExpertsConfig {
        experts: 6,
        experts_per_token: 2,
        expert_intermediate_size: 4,
        shared_expert_intermediate_size: Some(8),
        normalize_top_k: false,
    };
    let routed = ModelArchitectureConfig {
        mixture_of_experts: Some(moe),
        ..dense.clone()
    };
    let operators = |config: &ModelArchitectureConfig| {
        let (graphs, _definition, _instance) =
            build_first_native_graphs_from_named_component(&digest, config, &fixture.identity, 2)
                .expect("the Qwen Component builds graphs");
        let mut nodes = Vec::new();
        for graph in [&graphs.prefill, &graphs.decode] {
            nodes.extend(graph.nodes.values().cloned());
        }
        nodes
    };

    let dense_nodes = operators(&dense);
    assert!(
        dense_nodes
            .iter()
            .all(|node| node.operator.name() != "moe-experts")
    );

    let nodes = operators(&routed);
    let layers = u64::from(dense.num_hidden_layers);
    let moe_nodes: Vec<_> = nodes
        .iter()
        .filter(|node| node.operator.name() == "moe-experts")
        .collect();
    assert_eq!(
        moe_nodes.len() as u64,
        2 * layers,
        "one per layer and phase"
    );
    for node in &moe_nodes {
        assert_eq!(node.inputs.len(), 5);
        assert_eq!(
            node.attributes.get("experts"),
            Some(&OperatorAttributeValue::Integer(6))
        );
        assert_eq!(
            node.attributes.get("top_k"),
            Some(&OperatorAttributeValue::Integer(2))
        );
        assert_eq!(
            node.attributes.get("normalize_top_k"),
            Some(&OperatorAttributeValue::Boolean(false))
        );
        assert!(node.inputs[2].as_str().ends_with("moe.experts_gate_proj"));
        assert!(node.inputs[4].as_str().ends_with("moe.experts_down_proj"));
    }
    let count = |operator: &str| {
        nodes
            .iter()
            .filter(|node| node.operator.name() == operator)
            .count() as u64
    };
    assert_eq!(count("row-scale"), 2 * layers);
    assert!(nodes.iter().any(|node| {
        node.operator.name() == "activation"
            && node.attributes.get("kind")
                == Some(&OperatorAttributeValue::String("sigmoid".into()))
    }));

    // Tensor parallelism: whole experts split, the shared expert split like
    // a dense MLP, the router and the shared expert's gate replicated.
    let (graphs, _definition, _instance) =
        build_first_native_graphs_from_named_component(&digest, &routed, &fixture.identity, 2)
            .expect("the Qwen Component builds graphs");
    let plan = crate::tensor_parallel::TensorParallelPlan::from_graphs(
        [&graphs.prefill, &graphs.decode],
        vec![DeviceId::new("a"), DeviceId::new("b")],
    )
    .expect("the plan is consistent");
    use crate::tensor_parallel::WeightShard;
    let shard = |suffix: &str| plan.shard_for_weight(&format!("layers.0.{suffix}"));
    let experts = Some(WeightShard::Experts { experts: 6 });
    let named = |suffix: &str| {
        moe_nodes[0]
            .inputs
            .iter()
            .chain(nodes.iter().flat_map(|node| &node.inputs))
            .filter_map(|edge| edge.as_str().strip_prefix("weight."))
            .find(|name| name.starts_with("layers.0.") && name.ends_with(suffix))
            .map(str::to_string)
            .unwrap_or_else(|| panic!("no layer 0 weight ending in {suffix}"))
    };
    for (suffix, expected) in [
        ("experts_gate_proj", experts),
        ("experts_up_proj", experts),
        ("experts_down_proj", experts),
        ("shared_gate_proj", Some(WeightShard::Columns)),
        ("shared_up_proj", Some(WeightShard::Columns)),
        ("shared_down_proj", Some(WeightShard::Rows)),
        ("router_proj", None),
        ("shared_expert_gate", None),
    ] {
        let name = named(suffix);
        let canonical = name.strip_prefix("layers.0.").unwrap();
        assert_eq!(shard(canonical), expected, "{name}");
    }
}

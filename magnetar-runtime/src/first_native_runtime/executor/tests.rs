//! Unit tests for the parent module.
//!
//! Shared mock executors and fixtures live in `first_native_runtime/tests.rs`.

use super::super::tests::*;
use super::*;

#[test]
fn e2e_graph_production_and_execution_succeeds_for_valid_fixture() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_graph_production_and_execution(&fixture).expect("prefill/decode graphs execute");
}

#[test]
fn e2e_qwen_graph_nodes_have_prepared_kernel_bindings() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_qwen_graph_nodes_have_prepared_kernel_bindings(&fixture)
        .expect("Qwen graph nodes are bound to prepared kernels");
}

#[test]
fn e2e_graph_dispatch_rejects_unregistered_provider() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_graph_dispatch_rejects_unregistered_provider(&fixture)
        .expect("graph dispatch rejects a plan binding naming an unregistered provider");
}

#[test]
fn e2e_graph_dispatch_uses_registered_provider_instance() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_graph_dispatch_uses_registered_provider_instance(&fixture)
        .expect("graph dispatch executes through Runtime's registered provider instance");
}

#[test]
fn e2e_graph_dispatch_rejects_revoked_prepared_kernel() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_graph_dispatch_rejects_revoked_prepared_kernel(&fixture)
        .expect("graph dispatch refuses a revoked PreparedKernel");
}

#[test]
fn e2e_graph_dispatch_rejects_stale_prepared_kernel_generation() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_graph_dispatch_rejects_stale_prepared_kernel_generation(&fixture)
        .expect("graph dispatch refuses a stale PreparedKernel generation");
}

#[test]
fn e2e_graph_dispatch_rejects_provider_binding_mismatch() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_graph_dispatch_rejects_provider_binding_mismatch(&fixture)
        .expect("graph dispatch refuses a Provider binding mismatch");
}

#[test]
fn e2e_graph_dispatch_accounts_outputs_through_runtime_memory_manager() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_graph_dispatch_accounts_outputs_through_runtime_memory_manager(&fixture)
        .expect("graph dispatch accounts outputs through Runtime's MemoryManager");
}

#[test]
fn e2e_two_output_split_dispatch_produces_independently_resolvable_resources() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_two_output_split_dispatch_produces_independently_resolvable_resources(&fixture).expect(
        "a two-output 'split' Kernel dispatch produces two independently resolvable Tensor \
         Resources, each holding the correct half of its pre-split input",
    );
}

#[test]
fn e2e_graph_dispatch_releases_workspace_after_use() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_graph_dispatch_releases_workspace_after_use(&fixture)
        .expect("graph dispatch releases workspace allocations after use");
}

#[test]
fn e2e_graph_dispatch_records_memory_feasibility_failure_under_tight_budget() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_graph_dispatch_records_memory_feasibility_failure_under_tight_budget(&fixture)
        .expect("tight memory budget is recorded as a feasibility failure");
}

#[test]
fn e2e_weight_binding_rejects_tampered_artifact_bytes() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_weight_binding_rejects_tampered_artifact_bytes(&fixture)
        .expect("model loading rejects a weight artifact with tampered bytes");
}

#[test]
fn e2e_weight_byte_change_alters_generated_logits() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_weight_byte_change_alters_generated_logits(&fixture)
        .expect("changing one weight byte changes the generated logits");
}

#[test]
fn e2e_graph_execution_fails_closed_on_missing_weight() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_graph_execution_fails_closed_on_missing_weight(&fixture)
        .expect("graph execution fails closed when a required weight is missing");
}

#[test]
fn e2e_weight_resources_are_isolated_per_model_instance() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_weight_resources_are_isolated_per_model_instance(&fixture)
        .expect("weight resources are isolated per Model Instance");
}

#[test]
fn e2e_unload_releases_weight_resource_allocations() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_unload_releases_weight_resource_allocations(&fixture)
        .expect("unloading a Model Instance releases its weight resource allocations");
}

#[test]
fn e2e_repeated_load_unload_does_not_accumulate_weight_storage() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_repeated_load_unload_does_not_accumulate_weight_storage(&fixture)
        .expect("repeated load/unload cycles do not accumulate Provider-owned weight storage");
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn e2e_qwen_component_artifact_trust_is_validated_before_planning() {
    validate_and_instantiate_trusted_qwen_component_before_first_native_planning()
        .expect("trusted Qwen Component fixture validates before planning");
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn e2e_qwen_component_instantiates_with_wasmtime_limits_before_planning() {
    let preflight = validate_and_instantiate_trusted_qwen_component_before_first_native_planning()
        .expect("trusted Qwen Component fixture instantiates before planning");
    assert!(preflight.definition.get() > 0);
    assert!(preflight.instance.get() > 0);
    let fixture = e2e_fixture().expect("fixture builds");
    assert_eq!(
        preflight.graph_semantics,
        qwen_component_graph_semantics_for_prompt(&fixture.config, &fixture.identity, 2)
            .expect("fixture graph semantics derive")
    );
    assert!(preflight.observations.iter().any(|observation| {
        observation.kind == ComponentObservationKind::Instantiation
            && observation.message.contains("component instance ready")
    }));
    assert!(preflight.observations.iter().any(|observation| {
        observation.kind == ComponentObservationKind::Invocation
            && observation
                .message
                .contains("component invocation completed")
    }));
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn e2e_qwen_component_artifact_trust_rejection_fails_before_planning() {
    let mut request = QwenComponentPreflightRequest::default_trusted();
    request.trust_store = ComponentTrustStore::default();
    match validate_and_instantiate_qwen_component_before_first_native_planning(request) {
        Err(E2eConformanceError::ModelComponentFailed { reason })
            if reason.contains("artifact rejected") || reason.contains("no trust policy") =>
        {
            Ok(())
        }
        Err(error) => Err(error),
        Ok(_) => Err(E2eConformanceError::ModelComponentFailed {
            reason: "untrusted Qwen Component fixture was accepted".into(),
        }),
    }
    .expect("untrusted Qwen Component fixture is rejected before planning");
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn e2e_qwen_component_missing_artifact_fails_before_planning() {
    let (component_package, digest) = qwen_component_preflight_package("", None);
    let request = QwenComponentPreflightRequest {
        component_package,
        trust_store: ComponentTrustStore::default().trust_digest(&digest),
        limits: inference_component_runtime_limits(),
    };

    let result = validate_and_instantiate_qwen_component_before_first_native_planning(request);

    assert!(
        matches!(
            result,
            Err(E2eConformanceError::ModelComponentFailed { .. })
        ),
        "missing Qwen Component artifact was not rejected: {result:?}"
    );
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn e2e_qwen_component_digest_mismatch_fails_before_planning() {
    let wat = qwen_component_count_fixture_wat(19, 19, 0);
    let wrong_digest = "sha256:0000000000000000000000000000000000000000000000000000000000000000";
    let (component_package, _digest) = qwen_component_preflight_package(&wat, Some(wrong_digest));
    let request = trusted_preflight_request_for_temp_component(component_package, wrong_digest);

    let result = validate_and_instantiate_qwen_component_before_first_native_planning(request);

    assert!(
        matches!(
            result,
            Err(E2eConformanceError::ModelComponentFailed { .. })
        ),
        "digest-mismatched Qwen Component artifact was not rejected: {result:?}"
    );
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn e2e_qwen_component_deadline_fails_before_planning() {
    let wat = qwen_component_count_fixture_wat(19, 19, 0);
    let (component_package, digest) = qwen_component_preflight_package(&wat, None);
    let mut request = trusted_preflight_request_for_temp_component(component_package, &digest);
    request.limits.execution_deadline_millis = Some(0);

    let result = validate_and_instantiate_qwen_component_before_first_native_planning(request);

    assert!(
        matches!(
            result,
            Err(E2eConformanceError::ModelComponentFailed { .. })
        ),
        "expired Qwen Component deadline was not rejected: {result:?}"
    );
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn e2e_qwen_component_invalid_output_fails_before_planning() {
    let wat = r#"(component
(core module $m
    (func (export "prefill-node-count"))
    (func (export "decode-node-count") (result i32) i32.const 12)
    (func (export "provider-authority-count") (result i32) i32.const 0))
(core instance $i (instantiate $m))
(func (export "prefill-node-count")
    (canon lift (core func $i "prefill-node-count")))
(func (export "decode-node-count") (result u32)
    (canon lift (core func $i "decode-node-count")))
(func (export "provider-authority-count") (result u32)
    (canon lift (core func $i "provider-authority-count")))
(func $prefill-node-count
    (canon lift (core func $i "prefill-node-count")))
(func $decode-node-count (result u32)
    (canon lift (core func $i "decode-node-count")))
(func $provider-authority-count (result u32)
    (canon lift (core func $i "provider-authority-count")))
(instance $qwen-graph-fixture
    (export "prefill-node-count" (func $prefill-node-count))
    (export "decode-node-count" (func $decode-node-count))
    (export "provider-authority-count" (func $provider-authority-count)))
(export "magnetar:qwen/graph-fixture@1.0.0" (instance $qwen-graph-fixture)))
"#;
    let (component_package, digest) = qwen_component_preflight_package(wat, None);
    let request = trusted_preflight_request_for_temp_component(component_package, &digest);

    let result = validate_and_instantiate_qwen_component_before_first_native_planning(request);

    assert!(
        matches!(
            result,
            Err(E2eConformanceError::GraphValidationFailed { .. })
        ),
        "Qwen Component invalid output was not rejected: {result:?}"
    );
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn e2e_qwen_component_incompatible_graph_fails_before_planning() {
    let fixture = e2e_fixture().expect("fixture builds");
    let mut runtime = build_runtime_with_model_execution_engine(&fixture);
    let (instance, _memory) =
        load_fixture_instance(&fixture, &mut runtime).expect("fixture instance loads");
    let component_graph_semantics = QwenComponentGraphSemantics {
        prefill_node_count: 99,
        ..qwen_component_graph_semantics_for_prompt(&fixture.config, &fixture.identity, 2)
            .expect("fixture graph semantics derive")
    };

    let result =
        build_first_native_graphs_from_component_output(&fixture, 2, component_graph_semantics)
            .and_then(|graphs| {
                prepare_first_native_execution_plans(&runtime, &instance, graphs, 2)
            });

    assert!(
        matches!(
            result,
            Err(E2eConformanceError::GraphValidationFailed { .. })
        ),
        "Qwen Component/runtime graph mismatch was not rejected"
    );
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn e2e_qwen_component_matching_node_count_but_wrong_operator_sequence_fails_before_planning() {
    let fixture = e2e_fixture().expect("fixture builds");
    let mut runtime = build_runtime_with_model_execution_engine(&fixture);
    let (instance, _memory) =
        load_fixture_instance(&fixture, &mut runtime).expect("fixture instance loads");
    // Same node counts as the real graphs (so a count-only proof would
    // accept this), but a declared operator-sequence hash that does not
    // match any real graph -- proving semantic comparison, not just
    // size, gates plan preparation.
    let component_graph_semantics = QwenComponentGraphSemantics {
        prefill_operator_hash: 0xdead_beef,
        ..qwen_component_graph_semantics_for_prompt(&fixture.config, &fixture.identity, 2)
            .expect("fixture graph semantics derive")
    };

    let result =
        build_first_native_graphs_from_component_output(&fixture, 2, component_graph_semantics)
            .and_then(|graphs| {
                prepare_first_native_execution_plans(&runtime, &instance, graphs, 2)
            });

    assert!(
        matches!(
            result,
            Err(E2eConformanceError::GraphValidationFailed { .. })
        ),
        "Qwen Component operator-sequence mismatch with matching node counts was not rejected"
    );
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn e2e_qwen_component_provider_authority_fails_before_planning() {
    let wat = qwen_component_count_fixture_wat(19, 19, 1);
    let (component_package, digest) = qwen_component_preflight_package(&wat, None);
    let request = trusted_preflight_request_for_temp_component(component_package, &digest);

    let result = validate_and_instantiate_qwen_component_before_first_native_planning(request);

    assert!(
        matches!(result, Err(E2eConformanceError::BoundaryViolation { .. })),
        "Qwen Component Provider authority was not rejected: {result:?}"
    );
}

#[test]
fn e2e_graph_executor_rejects_missing_plan_binding() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_graph_executor_rejects_missing_plan_binding(&fixture)
        .expect("graph executor rejects a node with no published plan binding");
}

#[test]
fn e2e_graph_executor_rejects_unsupported_operator() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_graph_executor_rejects_unsupported_operator(&fixture)
        .expect("graph executor rejects an operator it does not implement");
}

#[test]
fn e2e_graph_executor_rejects_removed_producer_node() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_graph_executor_rejects_removed_producer_node(&fixture)
        .expect("graph executor rejects a graph with a removed producer node");
}

#[test]
fn e2e_error_categories_use_structured_codes() {
    let expected = [
        (
            "e2e-suite-unavailable",
            E2eConformanceError::SuiteUnavailable {
                reason: String::new(),
            },
        ),
        (
            "e2e-fixture-invalid",
            E2eConformanceError::FixtureInvalid {
                reason: String::new(),
            },
        ),
        (
            "e2e-model-resolution-failed",
            E2eConformanceError::ModelResolutionFailed {
                reason: String::new(),
            },
        ),
        (
            "e2e-model-loading-failed",
            E2eConformanceError::ModelLoadingFailed {
                reason: String::new(),
            },
        ),
        (
            "e2e-model-component-failed",
            E2eConformanceError::ModelComponentFailed {
                reason: String::new(),
            },
        ),
        (
            "e2e-tokenizer-failed",
            E2eConformanceError::TokenizerFailed {
                reason: String::new(),
            },
        ),
        (
            "e2e-session-failed",
            E2eConformanceError::SessionFailed {
                reason: String::new(),
            },
        ),
        (
            "e2e-generation-failed",
            E2eConformanceError::GenerationFailed {
                reason: String::new(),
            },
        ),
        (
            "e2e-sampling-failed",
            E2eConformanceError::SamplingFailed {
                reason: String::new(),
            },
        ),
        (
            "e2e-streaming-failed",
            E2eConformanceError::StreamingFailed {
                reason: String::new(),
            },
        ),
        (
            "e2e-graph-validation-failed",
            E2eConformanceError::GraphValidationFailed {
                reason: String::new(),
            },
        ),
        (
            "e2e-operator-coverage-missing",
            E2eConformanceError::OperatorCoverageMissing {
                reason: String::new(),
            },
        ),
        (
            "e2e-kernel-coverage-missing",
            E2eConformanceError::KernelCoverageMissing {
                reason: String::new(),
            },
        ),
        (
            "e2e-memory-validation-failed",
            E2eConformanceError::MemoryValidationFailed {
                reason: String::new(),
            },
        ),
        (
            "e2e-redaction-failed",
            E2eConformanceError::RedactionFailed {
                reason: String::new(),
            },
        ),
        (
            "e2e-boundary-violation",
            E2eConformanceError::BoundaryViolation {
                reason: String::new(),
            },
        ),
        (
            "e2e-determinism-failed",
            E2eConformanceError::DeterminismFailed {
                reason: String::new(),
            },
        ),
        (
            "internal-e2e-conformance",
            E2eConformanceError::Internal {
                reason: String::new(),
            },
        ),
    ];
    for (code, error) in expected {
        assert_eq!(error.code(), code);
    }
}

#[test]
fn e2e_kv_provider_failure_stores_no_pending_state() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_kv_provider_failure_stores_no_pending_state(&fixture)
        .expect("a generation step that fails Provider dispatch stores no pending KV state");
}

#[test]
fn e2e_generation_observations_never_carry_raw_prompt_or_handles() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_generation_observations_never_carry_raw_prompt_or_handles(&fixture)
        .expect("generation observations never carry raw prompt text or native handles");
}

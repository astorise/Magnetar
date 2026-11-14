//! Unit tests for the parent module.
//!
//! Shared mock executors and fixtures live in `first_native_runtime/tests.rs`.

use super::super::tests::*;
use super::*;

#[test]
fn e2e_success_path_resolves_loads_generates_and_cleans_up() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_success_path(&fixture).expect("Runtime success path completes");
}

#[test]
fn e2e_runs_without_gpu_network_or_tachyon() {
    // Structural: the fixture and success path never reference GPU,
    // network, or Tachyon primitives, and CLI-owned authorities are
    // explicitly denied to Runtime.
    check_cli_boundary_denials().expect("CLI-owned authorities are denied");
}

#[test]
fn e2e_fixture_tokenizer_produces_deterministic_tokens() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_fixture_tokenizer_deterministic(&fixture).expect("tokenization is deterministic");
}

#[test]
fn e2e_already_tokenized_prompt_path_bypasses_text_tokenization() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_already_tokenized_prompt_path(&fixture).expect("already-tokenized path is preserved");
}

#[test]
fn e2e_no_shortcut_direct_provider_invocation_is_rejected() {
    check_no_shortcut_direct_provider_rejected().expect("direct-invocation shortcut rejected");
}

#[test]
fn e2e_no_shortcuts_rejects_incomplete_per_node_causal_chain() {
    check_e2e_no_shortcuts_rejects_incomplete_per_node_causal_chain()
        .expect("incomplete per-node causal chain is rejected");
}

#[test]
fn e2e_reference_cpu_selected_through_kernel_registry() {
    check_reference_cpu_selected_through_kernel_registry()
        .expect("Reference CPU selected through Kernel Registry");
}

#[test]
fn e2e_operator_coverage_report_lists_required_operators() {
    let fixture = e2e_fixture().expect("fixture builds");
    let operators = check_operator_coverage(&fixture).expect("operator coverage computed");
    for expected in E2E_EXERCISED_OPERATORS {
        assert!(operators.contains(expected), "missing operator {expected}");
    }
}

#[test]
fn e2e_invalid_graph_fixture_fails_validation() {
    check_invalid_graph_fixture().expect("invalid graph fixture is rejected");
}

#[test]
fn e2e_max_new_tokens_reached_stops_generation() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_max_new_tokens_stops_generation(&fixture).expect("max token stop is honored");
}

#[test]
fn e2e_generation_cancelled_stops_with_cancelled_finish_reason() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_generation_cancelled(&fixture).expect("cancellation is honored");
}

#[test]
fn e2e_sampling_greedy_selects_deterministic_token() {
    check_sampling_greedy_deterministic().expect("greedy sampling is deterministic");
}

#[test]
fn e2e_streaming_events_are_ordered() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_streaming_order(&fixture).expect("streaming events are ordered");
}

#[test]
fn e2e_closed_session_rejects_generation() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_closed_session_rejects_generation(&fixture).expect("closed session is rejected");
}

#[test]
fn e2e_kv_cache_diagnostics_redact_raw_contents() {
    check_kv_cache_diagnostics_redacted().expect("cache usage carries no raw contents");
}

#[test]
fn e2e_cli_boundary_rejects_workspace_file_access() {
    check_cli_boundary_denials().expect("CLI boundary denials hold");
}

#[test]
fn e2e_diagnostics_redact_raw_values_on_failure() {
    check_diagnostics_redaction_on_failure().expect("diagnostics redact native handles");
}

#[test]
fn e2e_failure_cases_report_structured_errors() {
    check_invalid_model_reference().expect("invalid model reference rejected");
    check_untrusted_artifact(&e2e_fixture().unwrap()).expect("untrusted artifact rejected");
    check_incompatible_tokenizer(&e2e_fixture().unwrap()).expect("incompatible tokenizer rejected");
    check_unsupported_operator().expect("unsupported operator rejected");
    check_missing_kernel().expect("missing kernel rejected");
    check_required_kernel_removal_fails_coverage().expect("required kernel removal fails coverage");
    check_invalid_tensor_shape(&e2e_fixture().unwrap()).expect("invalid tensor shape rejected");
    check_memory_admission_failure().expect("memory admission failure rejected");
    check_closed_session_rejects_generation(&e2e_fixture().unwrap())
        .expect("closed session rejected");
    check_first_native_generation_requires_ready_model_instance(&e2e_fixture().unwrap())
        .expect("non-ready model instance rejected");
    check_missing_prepared_plan_fails_closed().expect("missing prepared plan rejected");
    check_invalidated_prepared_plan_rejects_new_work(&e2e_fixture().unwrap())
        .expect("invalidated prepared plan rejected");
    check_qwen_graph_nodes_have_prepared_kernel_bindings(&e2e_fixture().unwrap())
        .expect("Qwen graph nodes bound to prepared kernels");
    check_generation_cancelled(&e2e_fixture().unwrap()).expect("cancellation reported");
    check_cli_boundary_denials().expect("policy denial reported");
    check_raw_handle_access_denied().expect("raw handle access denied");
}

#[test]
fn e2e_determinism_repeated_runs_produce_matching_tokens() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_determinism(&fixture).expect("generation is deterministic");
}

#[test]
fn e2e_report_contains_required_metadata_fields() {
    let report = run_e2e_local_inference_conformance();
    check_report_metadata(&report).expect("report has required metadata");
    assert!(report.redacted);
    let no_shortcut_success = report
        .test_cases
        .iter()
        .find(|test| test.name == "success-path-no-shortcut-validated")
        .expect("success-path no-shortcut validation is reported");
    assert_eq!(no_shortcut_success.status, E2eTestStatus::Passed);
    assert!(no_shortcut_success.diagnostic.is_none());
    assert!(report.is_conformant());
}

#[test]
fn e2e_ci_can_run_without_gpu_and_reports_only_expected_required_failure() {
    let report = run_e2e_local_inference_conformance();
    let failed: Vec<_> = report
        .test_cases
        .iter()
        .filter(|test| test.status == E2eTestStatus::Failed)
        .map(|test| test.name.as_str())
        .collect();
    assert_eq!(failed, Vec::<&str>::new());
}

#[test]
fn e2e_observability_emits_only_redacted_report_metadata() {
    let report = run_e2e_local_inference_conformance();
    let json = e2e_conformance_report_json(&report).expect("report serializes");
    assert!(!json.contains("0x"));
    assert!(!json.contains("native_handle"));
    assert!(report.redacted);
}

#[test]
fn e2e_report_round_trips_through_json() {
    let report = run_e2e_local_inference_conformance();
    let json = e2e_conformance_report_json(&report).expect("serializes");
    let restored: E2eConformanceReport = serde_json::from_str(&json).expect("deserializes");
    assert_eq!(restored.suite_version, report.suite_version);
    assert_eq!(restored.test_cases.len(), report.test_cases.len());
}

#[test]
fn e2e_one_shot_session_exercises_normal_generation_sampling_and_kernel_path() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_one_shot_session_normal_paths(&fixture)
        .expect("one-shot session uses normal generation path");
}

#[test]
fn e2e_chat_message_prompt_path_uses_formatter_and_tokenizer_contract() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_chat_message_prompt_path(&fixture).expect("chat message prompt tokenizes");
}

#[test]
fn e2e_no_shortcut_direct_kernel_invocation_is_rejected() {
    check_no_shortcut_direct_kernel_invocation_rejected()
        .expect("fabricated incompatible kernel candidate rejected");
}

#[test]
fn e2e_no_shortcut_model_loading_bypass_is_detected() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_no_shortcut_model_loading_bypass_detected(&fixture)
        .expect("Model Loading bypass is detected");
}

#[test]
fn e2e_no_shortcut_model_component_bypass_is_detected() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_no_shortcut_model_component_bypass_detected(&fixture)
        .expect("Model Component bypass is detected");
}

#[test]
fn e2e_no_shortcut_memory_manager_bypass_is_detected() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_no_shortcut_memory_manager_bypass_detected(&fixture)
        .expect("Memory Manager bypass is detected");
}

#[test]
fn e2e_dtype_and_layout_conversion_are_never_silent() {
    check_dtype_and_layout_conversion_are_explicit()
        .expect("dtype/layout conversion is explicit, never silent");
}

#[test]
fn e2e_max_total_tokens_reached_stops_generation() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_max_total_tokens_stops_generation(&fixture).expect("max_total_tokens stops generation");
}

#[test]
fn e2e_stochastic_sampling_is_seed_deterministic() {
    check_stochastic_sampling_is_seed_deterministic()
        .expect("seeded stochastic sampling is reproducible");
}

#[test]
fn e2e_kv_and_prefix_cache_lifecycle_redacts_raw_contents() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_kv_and_prefix_cache_lifecycle(&fixture).expect("KV/Prefix cache lifecycle completes");
}

#[test]
fn e2e_tensor_resource_lifecycle_reaches_ready_and_released() {
    check_tensor_resource_lifecycle().expect("Tensor Resource lifecycle completes");
}

#[test]
fn e2e_memory_operator_output_accounting_leaves_no_untracked_allocation() {
    check_memory_operator_output_accounting()
        .expect("operator output and workspace allocations are tracked and released");
}

#[test]
fn e2e_generation_timeout_maps_to_structured_error() {
    check_generation_timeout_maps_to_structured_error()
        .expect("generation timeout maps to a structured error deterministically");
}

#[test]
fn e2e_run_emits_lifecycle_observation_markers() {
    let report = run_e2e_local_inference_conformance();
    for marker in [
        "observation-suite-started",
        "observation-fixture-loaded",
        "observation-success-path-started",
        "observation-success-path-completed",
        "observation-failure-case-started",
        "observation-failure-case-completed",
        "observation-redaction-failure",
        "observation-boundary-violation",
        "observation-report-generated",
    ] {
        assert!(
            report.test_cases.iter().any(|test| test.name == marker),
            "missing lifecycle observation marker: {marker}"
        );
    }
}

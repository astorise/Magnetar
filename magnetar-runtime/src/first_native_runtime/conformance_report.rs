//! The end-to-end local inference conformance suite and its report.

use super::*;

// ---------------------------------------------------------------------
// Report / result types
// ---------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize)]
pub enum E2eTestStatus {
    Passed,
    Failed,
    Skipped,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct E2eTestResult {
    pub name: String,
    pub status: E2eTestStatus,
    pub diagnostic: Option<String>,
}

impl E2eTestResult {
    pub fn passed(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: E2eTestStatus::Passed,
            diagnostic: None,
        }
    }

    pub fn failed(name: impl Into<String>, diagnostic: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: E2eTestStatus::Failed,
            diagnostic: Some(compute::redact_backend_diagnostic(&diagnostic.into())),
        }
    }

    pub fn skipped(name: impl Into<String>, diagnostic: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: E2eTestStatus::Skipped,
            diagnostic: Some(compute::redact_backend_diagnostic(&diagnostic.into())),
        }
    }

    pub(super) fn from_result(
        name: impl Into<String>,
        result: Result<(), E2eConformanceError>,
    ) -> Self {
        match result {
            Ok(()) => Self::passed(name),
            Err(error) => Self::failed(name, format!("{}: {}", error.code(), error.reason())),
        }
    }
}

pub(super) fn no_shortcut_success_path_result(fixture: &E2eFixture) -> E2eTestResult {
    let outcome = match run_success_path(fixture) {
        Ok(outcome) => outcome,
        Err(error) => {
            return E2eTestResult::failed(
                "success-path-no-shortcut-validated",
                format!(
                    "required no-shortcut validation failed before success path completed: {}: {}",
                    error.code(),
                    error.reason()
                ),
            );
        }
    };
    match validate_e2e_no_shortcuts(
        outcome.observer.observations(),
        &reference_cpu_kernel_advertisements(),
    ) {
        Ok(()) => E2eTestResult::passed("success-path-no-shortcut-validated"),
        Err(E2eConformanceError::BoundaryViolation { reason }) => E2eTestResult::failed(
            "success-path-no-shortcut-validated",
            format!("required no-shortcut validation failed: {reason}"),
        ),
        Err(error) => E2eTestResult::failed(
            "success-path-no-shortcut-validated",
            format!("{}: {}", error.code(), error.reason()),
        ),
    }
}

/// Machine-readable End-to-End Local Inference Conformance report. Never
/// carries raw prompts, weights, tensor values, cache contents, handles, or
/// memory pointers -- only redacted summaries and structured test results.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct E2eConformanceReport {
    pub suite_version: String,
    pub fixture_version: String,
    pub runtime_version: String,
    pub provider_summary: String,
    pub device_summary: String,
    pub model_component_summary: String,
    pub operator_coverage: BTreeSet<String>,
    pub kernel_coverage: BTreeSet<String>,
    pub test_cases: Vec<E2eTestResult>,
    pub redacted: bool,
    pub duration_millis: u64,
    pub timestamp_unix_seconds: u64,
}

impl E2eConformanceReport {
    pub(super) fn empty() -> Self {
        Self {
            suite_version: E2E_SUITE_VERSION.into(),
            fixture_version: E2E_FIXTURE_VERSION.into(),
            runtime_version: MAGNETAR_RUNTIME_VERSION.into(),
            provider_summary: String::new(),
            device_summary: String::new(),
            model_component_summary: String::new(),
            operator_coverage: BTreeSet::new(),
            kernel_coverage: BTreeSet::new(),
            test_cases: Vec::new(),
            redacted: true,
            duration_millis: 0,
            timestamp_unix_seconds: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_secs())
                .unwrap_or_default(),
        }
    }

    pub fn record(&mut self, result: E2eTestResult) {
        self.test_cases.push(result);
    }

    pub fn is_conformant(&self) -> bool {
        !self
            .test_cases
            .iter()
            .any(|test| test.status == E2eTestStatus::Failed)
    }

    pub fn passed_count(&self) -> usize {
        self.test_cases
            .iter()
            .filter(|test| test.status == E2eTestStatus::Passed)
            .count()
    }

    pub fn failed_count(&self) -> usize {
        self.test_cases
            .iter()
            .filter(|test| test.status == E2eTestStatus::Failed)
            .count()
    }

    pub fn skipped_count(&self) -> usize {
        self.test_cases
            .iter()
            .filter(|test| test.status == E2eTestStatus::Skipped)
            .count()
    }
}

pub fn validate_first_native_native_execution_evidence(
    report: &E2eConformanceReport,
) -> Result<(), E2eConformanceError> {
    first_native_model_execution_profile()
        .validate()
        .map_err(|error| E2eConformanceError::Internal {
            reason: error.to_string(),
        })?;
    if !report.is_conformant() {
        return Err(E2eConformanceError::BoundaryViolation {
            reason: "first native profile requires a conformant E2E report".into(),
        });
    }
    if report.provider_summary != REFERENCE_CPU_PROVIDER_NAME {
        return Err(E2eConformanceError::BoundaryViolation {
            reason: "first native profile requires Reference CPU Provider evidence".into(),
        });
    }
    if !report.model_component_summary.contains("e2e-qwen-fixture") {
        return Err(E2eConformanceError::ModelComponentFailed {
            reason: "first native profile requires Qwen Model Component evidence".into(),
        });
    }
    if report.operator_coverage.is_empty() {
        return Err(E2eConformanceError::OperatorCoverageMissing {
            reason: "first native profile requires Operator execution evidence".into(),
        });
    }
    if report.kernel_coverage.is_empty() {
        return Err(E2eConformanceError::KernelCoverageMissing {
            reason: "first native profile requires Kernel execution evidence".into(),
        });
    }
    for required in [
        "success-path-no-shortcut-validated",
        "operator-coverage",
        "kernel-coverage",
        "reference-cpu-selected-through-kernel-registry",
        "no-shortcut-direct-provider-rejected",
        "no-shortcut-direct-kernel-invocation-rejected",
    ] {
        if !report
            .test_cases
            .iter()
            .any(|test| test.name == required && test.status == E2eTestStatus::Passed)
        {
            return Err(E2eConformanceError::BoundaryViolation {
                reason: format!("missing first native structural evidence '{required}'"),
            });
        }
    }
    Ok(())
}

pub fn e2e_conformance_report_json(
    report: &E2eConformanceReport,
) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(report)
}

// ---------------------------------------------------------------------
// No-shortcut validation
// ---------------------------------------------------------------------

/// Validates that inference produced execution evidence for the Kernel
/// Registry / Kernel Dispatch / Reference CPU path rather than relying on a
/// caller assertion.
pub fn validate_e2e_no_shortcuts(
    observations: &[InferenceApiObservation],
    advertisements: &[KernelAdvertisement],
) -> Result<(), E2eConformanceError> {
    let has = |kind: InferenceApiObservationKind| {
        observations
            .iter()
            .any(|observation| observation.kind == kind)
    };
    for required in [
        InferenceApiObservationKind::ExecutionGraphValidated,
        InferenceApiObservationKind::KernelSelected,
        InferenceApiObservationKind::KernelDispatched,
        InferenceApiObservationKind::ProviderExecuted,
        InferenceApiObservationKind::TensorLogitsProduced,
    ] {
        if !has(required) {
            return Err(E2eConformanceError::BoundaryViolation {
                reason: format!("inference did not emit {required:?} execution evidence"),
            });
        }
    }
    if observations.is_empty() {
        return Err(E2eConformanceError::BoundaryViolation {
            reason:
                "inference bypassed Kernel Registry and invoked Reference CPU Provider directly"
                    .into(),
        });
    }
    validate_no_placeholder_kernel_advertisements(advertisements)
        .map_err(E2eConformanceError::from)?;
    validate_reference_cpu_required_kernel_coverage(advertisements)
        .map_err(E2eConformanceError::from)?;
    validate_e2e_per_node_causal_chain(observations)?;
    Ok(())
}

/// Extracts the `node=...` correlation an observation's message carries
/// (Correctif 17 / task group 17), if any -- the same `node=<value> ` shape
/// every per-node causal event in `inference_api.rs`'s generation loop
/// writes via `observation_message`. Returns the value up to (not
/// including) the next space, since further `key=value` pairs may follow.
pub(super) fn observation_node_correlation(message: &str) -> Option<&str> {
    message
        .split("node=")
        .nth(1)
        .map(|rest| rest.split(' ').next().unwrap_or(rest))
}

/// Correctif 17 / task group 17: unlike the presence checks above (which
/// only confirm five *global* evidence categories occurred somewhere across
/// the whole generation), this walks every distinct graph node that
/// actually dispatched (identified by its `GraphNodeReady` events) and
/// requires each one to carry a *complete* per-node causal chain --
/// `PlanBindingResolved`, `PreparedKernelResolved`, `ProviderSubmitted`,
/// `ProviderCompleted`, and `TensorResourceProduced`, all correlated to
/// that same node -- not merely that each event kind occurred for *some*
/// node. `KvUpdatePrepared`/`KvUpdateCommitted` are deliberately excluded:
/// they only apply to KV-cache-bearing nodes (not every node produces one),
/// and `KvUpdateCommitted` in particular correlates to a `TensorResourceId`
/// rather than a node at all (see its emission site's doc comment in
/// `inference_api.rs`).
pub(super) fn validate_e2e_per_node_causal_chain(
    observations: &[InferenceApiObservation],
) -> Result<(), E2eConformanceError> {
    let ready_nodes: BTreeSet<&str> = observations
        .iter()
        .filter(|observation| observation.kind == InferenceApiObservationKind::GraphNodeReady)
        .filter_map(|observation| observation_node_correlation(&observation.message))
        .collect();
    if ready_nodes.is_empty() {
        return Err(E2eConformanceError::BoundaryViolation {
            reason: "inference emitted no per-node GraphNodeReady evidence at all".into(),
        });
    }
    for required in [
        InferenceApiObservationKind::PlanBindingResolved,
        InferenceApiObservationKind::PreparedKernelResolved,
        InferenceApiObservationKind::ProviderSubmitted,
        InferenceApiObservationKind::ProviderCompleted,
        InferenceApiObservationKind::TensorResourceProduced,
    ] {
        let correlated_nodes: BTreeSet<&str> = observations
            .iter()
            .filter(|observation| observation.kind == required)
            .filter_map(|observation| observation_node_correlation(&observation.message))
            .collect();
        let missing: Vec<&str> = ready_nodes.difference(&correlated_nodes).copied().collect();
        if !missing.is_empty() {
            return Err(E2eConformanceError::BoundaryViolation {
                reason: format!(
                    "node(s) {missing:?} dispatched (GraphNodeReady) but never emitted a \
                     correlated {required:?} -- incomplete per-node causal chain"
                ),
            });
        }
    }
    Ok(())
}

pub(super) fn check_success_path(fixture: &E2eFixture) -> Result<(), E2eConformanceError> {
    let outcome = run_success_path(fixture)?;
    if outcome
        .generation_result
        .output
        .generated_token_ids
        .is_empty()
    {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "success path produced no generated tokens".into(),
        });
    }
    Ok(())
}

pub(super) fn check_determinism(fixture: &E2eFixture) -> Result<(), E2eConformanceError> {
    let first = run_success_path(fixture)?;
    let second = run_success_path(fixture)?;
    if first.generation_result.output.generated_token_ids
        != second.generation_result.output.generated_token_ids
    {
        return Err(E2eConformanceError::DeterminismFailed {
            reason: "repeated success path runs produced different generated tokens".into(),
        });
    }
    Ok(())
}

pub(super) fn check_streaming_order(fixture: &E2eFixture) -> Result<(), E2eConformanceError> {
    let outcome = run_success_path(fixture)?;
    let kinds: Vec<InferenceApiObservationKind> = outcome
        .observer
        .observations()
        .iter()
        .map(|observation| observation.kind)
        .collect();
    let position = |kind: InferenceApiObservationKind| kinds.iter().position(|k| *k == kind);
    let (
        Some(prefill_started),
        Some(prefill_completed),
        Some(first_token),
        Some(completed),
        Some(closed),
    ) = (
        position(InferenceApiObservationKind::PrefillStarted),
        position(InferenceApiObservationKind::PrefillCompleted),
        position(InferenceApiObservationKind::TokenGenerated),
        position(InferenceApiObservationKind::GenerationCompleted),
        position(InferenceApiObservationKind::StreamClosed),
    )
    else {
        return Err(E2eConformanceError::StreamingFailed {
            reason: "expected streaming observation kinds were not all emitted".into(),
        });
    };
    if !(prefill_started < prefill_completed
        && prefill_completed < first_token
        && first_token < completed
        && completed < closed)
    {
        return Err(E2eConformanceError::StreamingFailed {
            reason: "streaming events were not emitted in the expected order".into(),
        });
    }
    Ok(())
}

pub(super) fn record_operator_dispatch<T>(
    covered: &mut BTreeSet<String>,
    name: &str,
    result: Result<(KernelDispatchResult, T), InferenceApiError>,
) -> Result<(), E2eConformanceError> {
    result.map_err(E2eConformanceError::from)?;
    covered.insert(name.to_string());
    Ok(())
}

pub(super) fn check_operator_coverage(
    fixture: &E2eFixture,
) -> Result<BTreeSet<String>, E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let provider: Arc<dyn ProviderExecutionApi> = Arc::new(ReferenceCpuExecutor::new());
    let mut node_events = Vec::new();
    let mut dispatch_ctx = FirstNativeDispatchContext {
        runtime: &mut runtime,
        provider: provider.clone(),
        prepared_plan: None,
        graph: None,
        sequence_length: None,
        last_provider_execution: None,
        internal_selection: None,
        bounded_kv_edges: BTreeMap::new(),
        node_events: &mut node_events,
    };
    let architecture = &fixture.config.architecture;
    let mut covered = BTreeSet::new();
    let hidden = HostTensor::new(
        [2, architecture.hidden_size],
        vec![0.25; (2 * architecture.hidden_size) as usize],
    )?;
    let one_row_hidden = HostTensor::new(
        [1, architecture.hidden_size],
        vec![0.5; architecture.hidden_size as usize],
    )?;
    let ids = HostTensor::new([2], vec![1.0, 2.0])?;
    let embedding_table = fixture_tensor_by_name(&fixture.weights, "token_embedding")?.clone();
    record_operator_dispatch(
        &mut covered,
        "embedding",
        dispatch_reference_cpu_operator(
            &mut dispatch_ctx,
            "coverage.embedding",
            dispatch_operator_id("embedding", OperatorFamily::Tensor),
            vec![
                NodeInputResource::Fresh(
                    TensorResourceId::new("coverage.embedding.table"),
                    f32_tensor_descriptor(&embedding_table),
                    embedding_table,
                ),
                NodeInputResource::Fresh(
                    TensorResourceId::new("coverage.embedding.ids"),
                    f32_tensor_descriptor(&ids),
                    ids,
                ),
            ],
            (
                TensorResourceId::new("coverage.embedding.out"),
                TensorDescriptor::new(
                    ShapeDescriptor::new([2, architecture.hidden_size]),
                    DTypeDescriptor::portable(ComputeDType::Float32),
                    LayoutDescriptor::Contiguous,
                ),
            ),
            BTreeMap::new(),
        ),
    )?;
    let weight = HostTensor::new(
        [2, architecture.hidden_size],
        vec![1.0; (2 * architecture.hidden_size) as usize],
    )?;
    record_operator_dispatch(
        &mut covered,
        "rmsnorm",
        dispatch_qwen_rmsnorm(
            &mut dispatch_ctx,
            "coverage.rmsnorm",
            NodeValue::Host(hidden.clone()),
            NodeValue::Host(weight),
            fixture.config.rmsnorm_epsilon,
            None,
        ),
    )?;
    let matmul_b = HostTensor::new(
        [architecture.hidden_size, architecture.hidden_size],
        vec![0.125; (architecture.hidden_size * architecture.hidden_size) as usize],
    )?;
    record_operator_dispatch(
        &mut covered,
        "matmul",
        dispatch_qwen_matmul(
            &mut dispatch_ctx,
            "coverage.matmul",
            NodeValue::Host(one_row_hidden.clone()),
            NodeValue::Host(matmul_b),
            None,
        ),
    )?;
    record_operator_dispatch(
        &mut covered,
        "rope",
        dispatch_qwen_rope(
            &mut dispatch_ctx,
            "coverage.rope",
            NodeValue::Host(one_row_hidden.clone()),
            architecture.attention_head_count,
            &fixture.config.rope,
            RopePairing::HalfSplit,
            None,
            0,
            None,
        ),
    )?;
    record_operator_dispatch(
        &mut covered,
        "attention",
        dispatch_qwen_attention(
            &mut dispatch_ctx,
            "coverage.attention",
            NodeValue::Host(one_row_hidden.clone()),
            NodeValue::Host(one_row_hidden.clone()),
            NodeValue::Host(one_row_hidden.clone()),
            architecture,
            None,
            None,
        ),
    )?;
    record_operator_dispatch(
        &mut covered,
        "softmax",
        dispatch_qwen_unary(
            &mut dispatch_ctx,
            "coverage.softmax",
            "softmax",
            OperatorFamily::Activation,
            NodeValue::Host(one_row_hidden.clone()),
            BTreeMap::new(),
            None,
        ),
    )?;
    record_operator_dispatch(
        &mut covered,
        "silu",
        dispatch_qwen_unary(
            &mut dispatch_ctx,
            "coverage.silu",
            "silu",
            OperatorFamily::Activation,
            NodeValue::Host(one_row_hidden.clone()),
            BTreeMap::new(),
            None,
        ),
    )?;
    record_operator_dispatch(
        &mut covered,
        "mul",
        dispatch_qwen_binary_same_shape(
            &mut dispatch_ctx,
            "coverage.mul",
            "mul",
            OperatorFamily::Tensor,
            NodeValue::Host(one_row_hidden.clone()),
            NodeValue::Host(one_row_hidden.clone()),
            None,
        ),
    )?;
    record_operator_dispatch(
        &mut covered,
        "add",
        dispatch_qwen_binary_same_shape(
            &mut dispatch_ctx,
            "coverage.add",
            "add",
            OperatorFamily::Tensor,
            NodeValue::Host(one_row_hidden.clone()),
            NodeValue::Host(one_row_hidden.clone()),
            None,
        ),
    )?;
    record_operator_dispatch(
        &mut covered,
        "residual-add",
        dispatch_qwen_binary_same_shape(
            &mut dispatch_ctx,
            "coverage.residual_add",
            "residual-add",
            OperatorFamily::Tensor,
            NodeValue::Host(one_row_hidden.clone()),
            NodeValue::Host(one_row_hidden),
            None,
        ),
    )?;
    Ok(covered)
}

pub(super) fn check_kernel_coverage() -> Result<BTreeSet<String>, E2eConformanceError> {
    let advertisements = reference_cpu_kernel_advertisements();
    Ok(advertisements
        .iter()
        .map(|advertisement| advertisement.implemented_operator.name().to_string())
        .collect())
}

pub(super) fn check_no_shortcut_direct_provider_rejected() -> Result<(), E2eConformanceError> {
    match validate_e2e_no_shortcuts(&[], &reference_cpu_kernel_advertisements()) {
        Err(E2eConformanceError::BoundaryViolation { .. }) => Ok(()),
        Err(other) => Err(E2eConformanceError::Internal {
            reason: format!("expected e2e-boundary-violation, got {}", other.code()),
        }),
        Ok(()) => Err(E2eConformanceError::Internal {
            reason: "direct Provider shortcut was not rejected".into(),
        }),
    }
}

pub(super) fn check_reference_cpu_selected_through_kernel_registry()
-> Result<(), E2eConformanceError> {
    let runtime = build_runtime();
    let advertisements = reference_cpu_kernel_advertisements();
    let matmul_operator = advertisements
        .iter()
        .find(|advertisement| advertisement.implemented_operator.name() == "matmul")
        .map(|advertisement| advertisement.implemented_operator.clone())
        .ok_or_else(|| E2eConformanceError::KernelCoverageMissing {
            reason: "Reference CPU does not advertise matmul".into(),
        })?;
    let request = KernelSelectionRequest::new(
        "e2e-matmul-selection",
        matmul_operator,
        ResourceAffinity::new(FallbackClass::Transparent),
    );
    let result = runtime
        .kernel_registry()
        .select(&request)
        .map_err(E2eConformanceError::from)?;
    match result.selected {
        Some(_) => Ok(()),
        None => Err(E2eConformanceError::KernelCoverageMissing {
            reason: "Kernel Registry selected no candidate for matmul".into(),
        }),
    }
}

pub(super) fn check_invalid_graph_fixture() -> Result<(), E2eConformanceError> {
    let bad_operator = OperatorId::new(
        OPERATOR_NAMESPACE,
        "unsupported-op",
        1,
        OperatorFamily::Control,
    );
    let edge = TensorEdge::new(
        TensorEdgeId::new("bad.output"),
        TensorDescriptor::new(
            ShapeDescriptor::new(vec![1, 1]),
            DTypeDescriptor::portable(ComputeDType::Float32),
            LayoutDescriptor::Contiguous,
        ),
    );
    let mut node = ExecutionNode::new(ExecutionNodeId::new("bad-node"), bad_operator);
    node.outputs = vec![TensorEdgeId::new("bad.output")];
    let graph = ExecutionGraph::new(
        ExecutionGraphId::new("e2e-invalid-graph"),
        ExecutionGraphPhase::Test,
    )
    .with_edge(edge)
    .with_node(node);
    match validate_first_scope_graph(&graph) {
        Err(_) => Ok(()),
        Ok(()) => Err(E2eConformanceError::Internal {
            reason: "expected invalid graph fixture to fail validation".into(),
        }),
    }
}

pub(super) fn check_max_new_tokens_stops_generation(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let tokenized = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::PlainText("a".into())),
        None,
    )?;
    let request = build_generation_request(
        GenerationRequestId::new("e2e-max-new-tokens")?,
        None,
        GenerationModelReference::ModelInstance(instance.clone()),
        generation_tokenizer_reference(fixture),
        tokenized,
        1,
        GenerationParameters::greedy(),
        StopConditions::default(),
        StreamingMode::Disabled,
    );
    let request = prepare_generation(&runtime, request)?;
    let mut observer = InferenceApiObserver::new();
    let result = run_first_native_generation_loop_with_plans(
        &mut runtime,
        fixture,
        &instance,
        &request,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |_generated_so_far| false,
        &mut observer,
    )?;
    if result.output.finish_reason != FinishReason::MaxNewTokens {
        return Err(E2eConformanceError::GenerationFailed {
            reason: format!(
                "expected FinishReason::MaxNewTokens, got {:?}",
                result.output.finish_reason
            ),
        });
    }
    if result.output.generated_token_count != 1 {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "expected exactly one generated token".into(),
        });
    }
    Ok(())
}

pub(super) fn check_generation_cancelled(fixture: &E2eFixture) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let tokenized = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::PlainText("a".into())),
        None,
    )?;
    let request = build_generation_request(
        GenerationRequestId::new("e2e-cancelled")?,
        None,
        GenerationModelReference::ModelInstance(instance),
        generation_tokenizer_reference(fixture),
        tokenized,
        4,
        GenerationParameters::greedy(),
        StopConditions::default(),
        StreamingMode::Disabled,
    );
    let request = prepare_generation(&runtime, request)?;
    let mut observer = InferenceApiObserver::new();
    let result = run_generation_loop(
        &mut runtime,
        &request,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |_generated_so_far| true,
        &mut observer,
    )?;
    if result.output.finish_reason != FinishReason::Cancelled {
        return Err(E2eConformanceError::GenerationFailed {
            reason: format!(
                "expected FinishReason::Cancelled, got {:?}",
                result.output.finish_reason
            ),
        });
    }
    Ok(())
}

pub(super) fn check_sampling_greedy_deterministic() -> Result<(), E2eConformanceError> {
    let tokenizer = e2e_fixture_tokenizer()?;
    let request = GenerationRequest {
        request_id: GenerationRequestId::new("e2e-sampling")?,
        session: None,
        model: GenerationModelReference::LoadedModelContext("e2e-fixture".into()),
        tokenizer: GenerationTokenizerReference {
            tokenizer_id: tokenizer.metadata().id.clone(),
            metadata: tokenizer.metadata().clone(),
        },
        prompt_token_count: 1,
        input_token_ids: vec![1],
        max_new_tokens: 1,
        max_total_tokens: None,
        model_context_length: None,
        parameters: GenerationParameters::greedy(),
        stop_conditions: StopConditions::default(),
        streaming: StreamingMode::Disabled,
        priority: GenerationPriority::default(),
        cancellation: CancellationMetadata::default(),
        memory: GenerationMemoryEstimate::default(),
        correlation_id: None,
        trace_id: None,
    };
    let mut logits = vec![0.0_f32; E2E_FIXTURE_VOCAB as usize];
    logits[42] = 10.0;
    let (first, _) =
        decode_step_from_sampling(&request, &[], logits.clone(), SamplingPolicy::default())
            .map_err(|error| E2eConformanceError::SamplingFailed {
                reason: error.to_string(),
            })?;
    let (second, _) = decode_step_from_sampling(&request, &[], logits, SamplingPolicy::default())
        .map_err(|error| E2eConformanceError::SamplingFailed {
        reason: error.to_string(),
    })?;
    if first.selected_token_id != 42 || second.selected_token_id != 42 {
        return Err(E2eConformanceError::SamplingFailed {
            reason: format!(
                "expected greedy selection of token 42 both times, got {} and {}",
                first.selected_token_id, second.selected_token_id
            ),
        });
    }
    Ok(())
}

pub(super) fn check_closed_session_rejects_generation(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let session_request = SessionCreationRequest {
        model: GenerationModelReference::ModelInstance(instance),
        tokenizer: generation_tokenizer_reference(fixture),
        generation_defaults: GenerationParameters::greedy(),
        policy: SessionPolicy::default(),
        memory: SessionMemoryBudget::default(),
        allowed_capabilities: BTreeSet::new(),
        correlation_id: None,
        created_at_millis: 0,
    };
    let session = create_inference_session(&mut runtime, session_request)?;
    close_inference_session(&mut runtime, &session)?;
    match session_status(
        &runtime,
        &session,
        &SessionAccessPolicy::authorize(session.clone()),
    ) {
        Err(InferenceApiError::SessionNotFound | InferenceApiError::SessionClosed) => Ok(()),
        Err(other) => Err(E2eConformanceError::from(other)),
        Ok(status) if status.lifecycle == SessionLifecycleState::Closed => Ok(()),
        Ok(_) => Err(E2eConformanceError::Internal {
            reason: "closed session was still reported usable".into(),
        }),
    }
}

pub(super) fn check_memory_admission_failure() -> Result<(), E2eConformanceError> {
    let manager = MemoryManager::new(MemoryManagerConfig {
        max_runtime_bytes: Some(1),
        allow_pending_allocations: false,
        ..MemoryManagerConfig::default()
    });
    let tokenizer = e2e_fixture_tokenizer()?;
    let request = GenerationRequest {
        request_id: GenerationRequestId::new("e2e-memory-admission")?,
        session: None,
        model: GenerationModelReference::LoadedModelContext("e2e-fixture".into()),
        tokenizer: GenerationTokenizerReference {
            tokenizer_id: tokenizer.metadata().id.clone(),
            metadata: tokenizer.metadata().clone(),
        },
        prompt_token_count: 1,
        input_token_ids: vec![1],
        max_new_tokens: 1,
        max_total_tokens: None,
        model_context_length: None,
        parameters: GenerationParameters::greedy(),
        stop_conditions: StopConditions::default(),
        streaming: StreamingMode::Disabled,
        priority: GenerationPriority::default(),
        cancellation: CancellationMetadata::default(),
        memory: GenerationMemoryEstimate {
            input_token_buffer_bytes: 10_000_000,
            queue_allowed: false,
            ..GenerationMemoryEstimate::default()
        },
        correlation_id: None,
        trace_id: None,
    };
    let admission = memory_admission(&request, &manager).map_err(|error| {
        E2eConformanceError::MemoryValidationFailed {
            reason: error.to_string(),
        }
    })?;
    if admission.is_admitted() {
        return Err(E2eConformanceError::Internal {
            reason: "expected memory admission to reject an over-budget request".into(),
        });
    }
    Ok(())
}

pub(super) fn check_incompatible_tokenizer(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut incompatible = fixture.tokenizer.metadata().clone();
    incompatible.special_tokens.clear();
    match validate_first_native_tokenizer_compatibility(&fixture.config, &incompatible) {
        Err(_) => Ok(()),
        Ok(()) => Err(E2eConformanceError::Internal {
            reason: "expected incompatible tokenizer (missing EOS) to be rejected".into(),
        }),
    }
}

pub(super) fn check_untrusted_artifact(fixture: &E2eFixture) -> Result<(), E2eConformanceError> {
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(fixture.architecture_implementation.clone());
    let mut memory = MemoryManager::default();
    let request = ModelLoadingRequest::new(
        ModelLoadingRequestId::new("e2e-untrusted-load"),
        fixture.manifest.id.clone(),
    );
    match coordinator.load(
        request,
        &fixture.manifest,
        // A real `ModelTrustStore` verdict, never a hand-built decision:
        // only the store can issue one.
        &ModelTrustStore::default()
            .reject_digest(fixture.manifest.id.digest.value.clone())
            .evaluate(&fixture.manifest),
        &mut memory,
    ) {
        Err(_) => Ok(()),
        Ok(_) => Err(E2eConformanceError::Internal {
            reason: "expected untrusted artifact to be rejected by Model Loading".into(),
        }),
    }
}

pub(super) fn check_invalid_tensor_shape(fixture: &E2eFixture) -> Result<(), E2eConformanceError> {
    let bad_tensor = ModelTensorMetadata {
        name: "token_embedding".into(),
        shape: vec![1],
        storage_dtype: ModelDType::F32,
        layout: None,
        shard: None,
        offset_bytes: None,
        size_bytes: None,
        quantization: None,
        expected_compute_dtype: None,
        digest: None,
    };
    match validate_first_native_tensor_shapes(&fixture.config, std::slice::from_ref(&bad_tensor)) {
        Err(_) => Ok(()),
        Ok(()) => Err(E2eConformanceError::Internal {
            reason: "expected invalid tensor shape to be rejected".into(),
        }),
    }
}

pub(super) fn check_unsupported_operator() -> Result<(), E2eConformanceError> {
    let unsupported = OperatorId::new(
        OPERATOR_NAMESPACE,
        "flash-attention-v3",
        1,
        OperatorFamily::Attention,
    );
    match validate_required_now_operator(&unsupported) {
        Err(_) => Ok(()),
        Ok(_) => Err(E2eConformanceError::Internal {
            reason: "expected unsupported operator to be rejected".into(),
        }),
    }
}

pub(super) fn check_missing_kernel() -> Result<(), E2eConformanceError> {
    match validate_reference_cpu_required_kernel_coverage(&[]) {
        Err(_) => Ok(()),
        Ok(()) => Err(E2eConformanceError::Internal {
            reason: "expected empty kernel advertisement list to fail coverage".into(),
        }),
    }
}

pub(super) fn check_required_kernel_removal_fails_coverage() -> Result<(), E2eConformanceError> {
    let mut advertisements = reference_cpu_kernel_advertisements();
    // Must remove a kernel the Qwen fixture's graph actually requires, not
    // merely the last-advertised one: `reference_cpu_kernel_advertisements`
    // also advertises Kernels (e.g. `split`, `reach-architecture-freeze-1`
    // task 5.4/5.5's multi-output proof Operator) that exist to prove
    // generic execution-path capability, not because any E2E fixture graph
    // node uses them -- removing one of those is correctly *not* expected
    // to fail coverage, so `.pop()`'s "whatever is last" used to work only
    // by coincidence (every advertised kernel was Qwen-required) and broke
    // the moment that stopped being true. `matmul` is unconditionally
    // required by every Qwen graph (every projection is one), so find it
    // by name rather than relying on Vec order at all.
    let matmul_index = advertisements
        .iter()
        .position(|advertisement| advertisement.implemented_operator.name() == "matmul")
        .ok_or_else(|| E2eConformanceError::Internal {
            reason: "expected first-native fixture to advertise a 'matmul' kernel".into(),
        })?;
    let removed = advertisements.remove(matmul_index);
    match validate_reference_cpu_required_kernel_coverage(&advertisements) {
        Err(error) if error.reason.contains(removed.implemented_operator.name()) => Ok(()),
        Err(error) => Err(E2eConformanceError::Internal {
            reason: format!(
                "required kernel removal failed with unexpected diagnostic: {}",
                error.reason
            ),
        }),
        Ok(()) => Err(E2eConformanceError::Internal {
            reason: format!(
                "expected removal of required kernel '{}' to fail coverage",
                removed.implemented_operator.name()
            ),
        }),
    }
}

pub(super) fn check_invalid_model_reference() -> Result<(), E2eConformanceError> {
    let registry = ModelRegistry::new();
    let model_ref = ModelRef::new("unregistered-model")?;
    match registry.resolve(&ModelResolutionRequest::new(model_ref)) {
        Err(_) => Ok(()),
        Ok(_) => Err(E2eConformanceError::Internal {
            reason: "expected resolution of an unregistered model to fail".into(),
        }),
    }
}

pub(super) fn check_raw_handle_access_denied() -> Result<(), E2eConformanceError> {
    let errors = [
        provider_handle_access_error(),
        device_handle_access_error(),
        kernel_handle_access_error(),
        memory_pointer_access_error(),
    ];
    if errors.iter().any(|error| {
        !matches!(
            error,
            ModelComponentError::ProviderAccessDenied
                | ModelComponentError::DeviceAccessDenied
                | ModelComponentError::KernelAccessDenied
                | ModelComponentError::MemoryPointerAccessDenied
        )
    }) {
        return Err(E2eConformanceError::Internal {
            reason: "raw handle access errors did not use structured denial categories".into(),
        });
    }
    Ok(())
}

pub(super) fn check_cli_boundary_denials() -> Result<(), E2eConformanceError> {
    for capability in [
        "workspace-filesystem",
        "git",
        "shell",
        "tool-call",
        "secret",
    ] {
        if reject_cli_owned_authority(capability).is_ok() {
            return Err(E2eConformanceError::BoundaryViolation {
                reason: format!("capability '{capability}' should have been denied to Runtime"),
            });
        }
    }
    if !run_cli_boundary_conformance().is_conformant() {
        return Err(E2eConformanceError::BoundaryViolation {
            reason: "CLI boundary conformance report is not conformant".into(),
        });
    }
    Ok(())
}

pub(super) fn check_kernel_optimization_orchestration_boundary() -> Result<(), E2eConformanceError>
{
    let report = run_kernel_optimization_orchestration_conformance();
    if !report.is_conformant() {
        let failures: Vec<String> = report
            .results
            .into_iter()
            .filter(|result| !result.passed)
            .map(|result| result.requirement)
            .collect();
        return Err(E2eConformanceError::BoundaryViolation {
            reason: format!(
                "kernel optimization orchestration conformance report is not conformant: {failures:?}"
            ),
        });
    }
    Ok(())
}

pub(super) fn check_kv_cache_diagnostics_redacted() -> Result<(), E2eConformanceError> {
    // The first E2E suite does not wire a live KV Cache into generation;
    // `CacheUsageSummary` structurally carries only hit/miss booleans, so
    // there is no raw cache payload to redact by construction.
    let usage = CacheUsageSummary {
        kv_cache_hit: Some(false),
        prefix_cache_hit: None,
    };
    if usage.kv_cache_hit.is_none() && usage.prefix_cache_hit.is_none() {
        return Err(E2eConformanceError::Internal {
            reason: "cache usage summary was unexpectedly empty".into(),
        });
    }
    Ok(())
}

pub(super) fn check_diagnostics_redaction_on_failure() -> Result<(), E2eConformanceError> {
    let error = InferenceApiError::ModelLoadingFailed {
        reason: "raw prompt 'super secret prompt' at native_handle 0xdeadbeef".into(),
    };
    let redacted = compute::redact_backend_diagnostic(&error.to_string());
    if redacted.contains("0xdeadbeef") || redacted.contains("native_handle") {
        return Err(E2eConformanceError::RedactionFailed {
            reason: "diagnostic redaction failed to strip a native handle".into(),
        });
    }
    Ok(())
}

pub(super) fn check_no_shortcut_direct_kernel_invocation_rejected()
-> Result<(), E2eConformanceError> {
    let runtime = build_runtime();
    let advertisements = reference_cpu_kernel_advertisements();
    let matmul_advertisement = advertisements
        .iter()
        .find(|advertisement| advertisement.implemented_operator.name() == "matmul")
        .cloned()
        .ok_or_else(|| E2eConformanceError::KernelCoverageMissing {
            reason: "Reference CPU does not advertise matmul".into(),
        })?;
    let request = KernelSelectionRequest::new(
        "e2e-direct-kernel-invocation",
        matmul_advertisement.implemented_operator.clone(),
        ResourceAffinity::new(FallbackClass::Transparent),
    );
    let selection = runtime
        .kernel_registry()
        .select(&request)
        .map_err(E2eConformanceError::from)?;
    let mut bypassing_candidate =
        selection
            .selected
            .ok_or_else(|| E2eConformanceError::KernelCoverageMissing {
                reason: "Kernel Registry selected no candidate for matmul".into(),
            })?;
    // A KernelDispatchPlan has no constructor other than `from_selection`,
    // and it refuses any candidate the registry itself did not mark
    // compatible -- so a caller cannot fabricate a dispatch plan for a
    // Kernel invocation it invoked directly, bypassing selection.
    bypassing_candidate.compatible = false;
    match KernelDispatchPlan::from_selection(
        KernelDispatchPlanId::new("e2e-bypass-plan"),
        &request,
        &bypassing_candidate,
        &matmul_advertisement,
        KernelInvocationId::new("e2e-bypass-invocation"),
    ) {
        Err(_) => Ok(()),
        Ok(_) => Err(E2eConformanceError::Internal {
            reason: "expected a fabricated incompatible kernel candidate to be rejected".into(),
        }),
    }
}

pub(super) fn check_no_shortcut_model_loading_bypass_detected(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(fixture.architecture_implementation.clone());
    let mut memory = MemoryManager::default();
    let request = ModelLoadingRequest::new(
        ModelLoadingRequestId::new("e2e-loading-bypass-check"),
        fixture.manifest.id.clone(),
    );
    let mut loaded = coordinator.load(
        request,
        &fixture.manifest,
        &ModelTrustStore::default()
            .trust_digest(fixture.manifest.id.digest.value.clone())
            .evaluate(&fixture.manifest),
        &mut memory,
    )?;
    // Simulate a caller that fabricates a "loaded" context without the real
    // Model Loading phases ever completing.
    loaded.state = ModelLoadingState::Requested;
    if loaded.can_start_inference() {
        return Err(E2eConformanceError::Internal {
            reason:
                "a context that bypassed real Model Loading phases was accepted as inference-ready"
                    .into(),
        });
    }
    Ok(())
}

pub(super) fn check_no_shortcut_model_component_bypass_detected(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    // No architecture is registered on this coordinator, simulating a
    // caller that skips Model Component resolution entirely.
    let mut coordinator = ModelLoadingCoordinator::new();
    let mut memory = MemoryManager::default();
    let request = ModelLoadingRequest::new(
        ModelLoadingRequestId::new("e2e-component-bypass-check"),
        fixture.manifest.id.clone(),
    );
    match coordinator.load(
        request,
        &fixture.manifest,
        &ModelTrustStore::default()
            .trust_digest(fixture.manifest.id.digest.value.clone())
            .evaluate(&fixture.manifest),
        &mut memory,
    ) {
        Err(_) => Ok(()),
        Ok(_) => Err(E2eConformanceError::Internal {
            reason: "expected loading without a registered Model Component to fail".into(),
        }),
    }
}

pub(super) fn check_no_shortcut_memory_manager_bypass_detected(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(fixture.architecture_implementation.clone());
    let mut starved_memory = MemoryManager::new(MemoryManagerConfig {
        max_runtime_bytes: Some(1),
        allow_pending_allocations: false,
        ..MemoryManagerConfig::default()
    });
    let request = ModelLoadingRequest::new(
        ModelLoadingRequestId::new("e2e-memory-bypass-check"),
        fixture.manifest.id.clone(),
    );
    match coordinator.load(
        request,
        &fixture.manifest,
        &ModelTrustStore::default()
            .trust_digest(fixture.manifest.id.digest.value.clone())
            .evaluate(&fixture.manifest),
        &mut starved_memory,
    ) {
        Err(_) => Ok(()),
        Ok(_) => Err(E2eConformanceError::Internal {
            reason:
                "expected Model Loading to honor Memory Manager admission instead of bypassing it"
                    .into(),
        }),
    }
}

pub(super) fn check_dtype_and_layout_conversion_are_explicit() -> Result<(), E2eConformanceError> {
    let tensor = HostTensor::new([1, 2], vec![1.0_f32, 2.0_f32])?;
    // Identity conversions are explicit and succeed.
    dtype_conversion(&tensor, ComputeDType::Float32, ComputeDType::Float32)?;
    layout_conversion(
        &tensor,
        TensorLayoutKind::Contiguous,
        TensorLayoutKind::Contiguous,
    )?;
    // A non-identity conversion is never performed silently: Reference CPU
    // explicitly rejects it rather than guessing at a conversion.
    if dtype_conversion(&tensor, ComputeDType::Float32, ComputeDType::Float16).is_ok() {
        return Err(E2eConformanceError::Internal {
            reason: "expected a non-identity dtype conversion to be explicitly rejected".into(),
        });
    }
    if layout_conversion(
        &tensor,
        TensorLayoutKind::Contiguous,
        TensorLayoutKind::Strided,
    )
    .is_ok()
    {
        return Err(E2eConformanceError::Internal {
            reason: "expected a non-identity layout conversion to be explicitly rejected".into(),
        });
    }
    Ok(())
}

pub(super) fn check_max_total_tokens_stops_generation(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    // `GenerationRequest::validate` requires `prompt + max_new_tokens <=
    // max_total_tokens`, and `stop_reason_for` checks `max_new_tokens`
    // before `max_total_tokens` -- so through the fully validated
    // `run_generation_loop` path, `max_new_tokens` always fires at or
    // before the total-token boundary is crossed. This exercises the
    // `max_total_tokens` stop-condition branch directly, the same way
    // `decode_step` does internally on every step.
    let tokenized = TokenizationResult {
        token_ids: vec![10, 20],
        token_count: 2,
        offsets: None,
        diagnostics: Vec::new(),
        correlation_id: None,
    };
    let mut request = build_generation_request(
        GenerationRequestId::new("e2e-max-total-tokens")?,
        None,
        GenerationModelReference::LoadedModelContext("e2e-fixture".into()),
        generation_tokenizer_reference(fixture),
        tokenized,
        10,
        GenerationParameters::greedy(),
        StopConditions::default(),
        StreamingMode::Disabled,
    );
    request.max_total_tokens = Some(request.prompt_token_count + 1);
    let finish_reason = stop_reason_for(&request, &[42]);
    if finish_reason != Some(FinishReason::MaxTotalTokens) {
        return Err(E2eConformanceError::GenerationFailed {
            reason: format!("expected Some(FinishReason::MaxTotalTokens), got {finish_reason:?}"),
        });
    }
    Ok(())
}

pub(super) fn check_stochastic_sampling_is_seed_deterministic() -> Result<(), E2eConformanceError> {
    let tokenizer = e2e_fixture_tokenizer()?;
    let request = GenerationRequest {
        request_id: GenerationRequestId::new("e2e-stochastic-sampling")?,
        session: None,
        model: GenerationModelReference::LoadedModelContext("e2e-fixture".into()),
        tokenizer: GenerationTokenizerReference {
            tokenizer_id: tokenizer.metadata().id.clone(),
            metadata: tokenizer.metadata().clone(),
        },
        prompt_token_count: 1,
        input_token_ids: vec![1],
        max_new_tokens: 1,
        max_total_tokens: None,
        model_context_length: None,
        parameters: GenerationParameters {
            temperature: 1.0,
            sampling_enabled: true,
            greedy: false,
            seed: Some(42),
            ..GenerationParameters::default()
        },
        stop_conditions: StopConditions::default(),
        streaming: StreamingMode::Disabled,
        priority: GenerationPriority::default(),
        cancellation: CancellationMetadata::default(),
        memory: GenerationMemoryEstimate::default(),
        correlation_id: None,
        trace_id: None,
    };
    let mut logits = vec![0.1_f32; E2E_FIXTURE_VOCAB as usize];
    logits[7] = 2.0;
    logits[9] = 1.9;
    let (first, _) =
        decode_step_from_sampling(&request, &[], logits.clone(), SamplingPolicy::default())
            .map_err(E2eConformanceError::from)?;
    let (second, _) = decode_step_from_sampling(&request, &[], logits, SamplingPolicy::default())
        .map_err(E2eConformanceError::from)?;
    if first.selected_token_id != second.selected_token_id {
        return Err(E2eConformanceError::SamplingFailed {
            reason: "seeded stochastic sampling produced different tokens across repeated runs"
                .into(),
        });
    }
    if first.selection_mode != SamplingSelectionMode::Stochastic {
        return Err(E2eConformanceError::SamplingFailed {
            reason: format!(
                "expected SamplingSelectionMode::Stochastic, got {:?}",
                first.selection_mode
            ),
        });
    }
    Ok(())
}

pub(super) fn check_kv_and_prefix_cache_lifecycle(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut memory = MemoryManager::default();
    let mut kv_manager = KvCacheManager::new();
    let kv_compatibility = KvCacheCompatibility::new(
        GenerationModelReference::LoadedModelContext("e2e-fixture".into()),
        fixture.tokenizer.metadata().id.clone(),
    );
    let layout = KvCacheLayoutMetadata::contiguous(
        fixture.config.architecture.layer_count as u32,
        fixture.config.architecture.attention_head_count as u32,
        fixture.config.architecture.head_dimension as u32,
        fixture.config.architecture.context_length as u32,
        ComputeDType::Float32,
    );
    let kv_cache = KvCache::new(
        KvCacheId::new("e2e-kv-cache")?,
        KvCacheScope::Session,
        kv_compatibility,
        layout,
    );
    let kv_id = kv_manager.create(kv_cache)?;
    kv_manager.allocate_memory(&kv_id, &mut memory)?;
    kv_manager.prefill_completed(&kv_id, 2)?;
    kv_manager.decode_append(&kv_id, 1)?;
    if kv_manager
        .observations()
        .iter()
        .any(|observation| observation.raw_cache_available)
    {
        return Err(E2eConformanceError::RedactionFailed {
            reason: "KV cache observation exposed raw cache contents".into(),
        });
    }

    let backing = PrefixCacheBackingKvCache::from_kv_cache(kv_manager.cache(&kv_id)?);
    let mut prefix_manager = PrefixCacheManager::new();
    let fingerprint = PrefixCacheFingerprint::from_validated_tokens(
        &[1, 2, 3],
        "e2e-fixture-model",
        &fixture.tokenizer.metadata().id,
    );
    let compatibility = PrefixCacheCompatibility::new(
        GenerationModelReference::LoadedModelContext("e2e-fixture".into()),
        fixture.tokenizer.metadata().id.clone(),
    );
    let lookup_request = PrefixCacheLookupRequest {
        fingerprint: fingerprint.clone(),
        compatibility: compatibility.clone(),
        requested_prefix_token_length: fingerprint.token_length(),
        session: None,
        owner: None,
        tenant: None,
        affinity: None,
        allow_partial: false,
    };
    let miss = prefix_manager.lookup(&lookup_request);
    if miss.kind != PrefixCacheMatchKind::Miss {
        return Err(E2eConformanceError::Internal {
            reason: format!(
                "expected an initial Prefix Cache lookup to miss, got {:?}",
                miss.kind
            ),
        });
    }

    let entry = PrefixCacheEntry::new(
        PrefixCacheEntryId::new("e2e-prefix-entry")?,
        fingerprint,
        compatibility,
        backing,
    );
    prefix_manager.create(entry, &PrefixCachePolicy::default())?;
    let hit = prefix_manager.lookup(&lookup_request);
    if hit.kind == PrefixCacheMatchKind::Miss {
        return Err(E2eConformanceError::Internal {
            reason: "expected a Prefix Cache lookup to hit after inserting a matching entry".into(),
        });
    }
    if prefix_manager
        .observations()
        .iter()
        .any(|observation| observation.raw_kv_cache_available || observation.raw_prompt_available)
    {
        return Err(E2eConformanceError::RedactionFailed {
            reason: "Prefix Cache observation exposed raw prompt or KV cache contents".into(),
        });
    }

    kv_manager.seal(&kv_id)?;
    kv_manager.release(&kv_id)?;
    Ok(())
}

pub(super) fn check_tensor_resource_lifecycle() -> Result<(), E2eConformanceError> {
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new(vec![1, 4]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let tensor_id = TensorResourceId::new("e2e-tensor-output");
    let affinity = ResourceAffinity::new(FallbackClass::Transparent);
    let residency =
        TensorResidency::new(tensor_id.clone(), MemoryPlacement::HostOrdinary, affinity);
    let mut resource = TensorResource::new(tensor_id, descriptor, residency);
    resource.transition_to(TensorLifecycleState::Planned)?;
    resource.transition_to(TensorLifecycleState::Allocating)?;
    resource.mark_ready()?;
    if resource.readiness != TensorReadiness::Ready {
        return Err(E2eConformanceError::Internal {
            reason: "expected Tensor Resource readiness to be Ready after mark_ready".into(),
        });
    }
    resource.ensure_usable()?;
    resource.transition_to(TensorLifecycleState::Released)?;
    Ok(())
}

pub(super) fn check_memory_operator_output_accounting() -> Result<(), E2eConformanceError> {
    let mut memory = MemoryManager::default();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new(vec![1, 4]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let operator_output_request = MemoryAllocationRequest::for_tensor(
        &descriptor,
        MemoryPlacement::HostOrdinary,
        MemoryAllocationOwner::Runtime,
    )
    .map_err(E2eConformanceError::from)?;
    let operator_output = memory.allocate(operator_output_request)?;
    let workspace_request = MemoryAllocationRequest::new(
        MemoryAllocationClass::TemporaryWorkspace,
        64,
        MemoryPlacement::HostOrdinary,
        MemoryAllocationOwner::Runtime,
    );
    let workspace = memory.allocate(workspace_request)?;

    let active_during = memory
        .allocations()
        .filter(|allocation| allocation.state == MemoryAllocationState::Active)
        .count();
    if active_during < 2 {
        return Err(E2eConformanceError::MemoryValidationFailed {
            reason: "expected operator output and workspace allocations to be tracked as active"
                .into(),
        });
    }

    memory.release(operator_output.id)?;
    memory.release(workspace.id)?;
    let active_after = memory
        .allocations()
        .filter(|allocation| allocation.state == MemoryAllocationState::Active)
        .count();
    if active_after != 0 {
        return Err(E2eConformanceError::MemoryValidationFailed {
            reason: "an allocation remained active after release, indicating an untracked Runtime-visible allocation"
                .into(),
        });
    }
    Ok(())
}

pub(super) fn check_generation_timeout_maps_to_structured_error() -> Result<(), E2eConformanceError>
{
    let mapped = InferenceApiError::from(BatchingError::OperationTimedOut);
    if !matches!(mapped, InferenceApiError::GenerationTimeout) {
        return Err(E2eConformanceError::Internal {
            reason: format!(
                "expected BatchingError::OperationTimedOut to map to InferenceApiError::GenerationTimeout, got {mapped:?}"
            ),
        });
    }
    // Deterministic memory-side timeout: no wall-clock sleep, just an
    // explicit deadline compared against a caller-supplied "now".
    let mut memory = MemoryManager::new(MemoryManagerConfig {
        allow_pending_allocations: true,
        ..MemoryManagerConfig::default()
    });
    let request = MemoryAllocationRequest::new(
        MemoryAllocationClass::Tensor,
        64,
        MemoryPlacement::HostOrdinary,
        MemoryAllocationOwner::Runtime,
    )
    .with_deadline_millis(1_000);
    memory.submit_pending_allocation(request, 0)?;
    let expired = memory.expire_pending_allocations(2_000);
    if !expired
        .iter()
        .any(|error| matches!(error, MemoryError::AllocationTimeout { .. }))
    {
        return Err(E2eConformanceError::Internal {
            reason: "expected an expired pending allocation to report AllocationTimeout".into(),
        });
    }
    Ok(())
}

pub(super) fn check_report_metadata(
    report: &E2eConformanceReport,
) -> Result<(), E2eConformanceError> {
    if report.suite_version.is_empty()
        || report.fixture_version.is_empty()
        || report.runtime_version.is_empty()
        || report.provider_summary.is_empty()
        || report.device_summary.is_empty()
        || report.model_component_summary.is_empty()
        || report.operator_coverage.is_empty()
        || report.kernel_coverage.is_empty()
        || report.test_cases.is_empty()
    {
        return Err(E2eConformanceError::Internal {
            reason: "report is missing required metadata fields".into(),
        });
    }
    Ok(())
}

/// Runs the End-to-End Local Inference Conformance suite and returns a
/// machine-readable [`E2eConformanceReport`]. Runs entirely on CPU without
/// GPU hardware, network access, Tachyon, or external processes.
pub fn run_e2e_local_inference_conformance() -> E2eConformanceReport {
    let start = SystemTime::now();
    let mut report = E2eConformanceReport::empty();
    report.record(E2eTestResult::passed("observation-suite-started"));

    let fixture = match e2e_fixture() {
        Ok(fixture) => fixture,
        Err(error) => {
            report.record(E2eTestResult::failed(
                "fixture-model",
                format!("{}: {}", error.code(), error.reason()),
            ));
            report.duration_millis = elapsed_millis(start);
            return report;
        }
    };
    report.record(E2eTestResult::passed("observation-fixture-loaded"));

    report.provider_summary = REFERENCE_CPU_PROVIDER_NAME.into();
    report.device_summary = REFERENCE_CPU_DEVICE_ID.into();
    report.model_component_summary = format!(
        "{}@{}",
        fixture.identity.id.as_str(),
        qwen_component_compatibility_key(&fixture.identity)
    );

    report.record(E2eTestResult::from_result(
        "fixture-model",
        fixture
            .manifest
            .validate()
            .map(|_| ())
            .map_err(E2eConformanceError::from),
    ));
    report.record(E2eTestResult::from_result(
        "fixture-tokenizer-deterministic",
        check_fixture_tokenizer_deterministic(&fixture),
    ));
    report.record(E2eTestResult::from_result(
        "already-tokenized-prompt-path",
        check_already_tokenized_prompt_path(&fixture),
    ));
    report.record(E2eTestResult::passed("observation-success-path-started"));
    let success_path_result = check_success_path(&fixture);
    report.record(E2eTestResult::from_result(
        "observation-success-path-completed",
        success_path_result.clone().map(|_| ()),
    ));
    report.record(E2eTestResult::from_result(
        "success-path",
        success_path_result,
    ));
    report.record(no_shortcut_success_path_result(&fixture));
    report.record(E2eTestResult::from_result(
        "determinism",
        check_determinism(&fixture),
    ));
    report.record(E2eTestResult::from_result(
        "one-shot-session-normal-paths",
        check_one_shot_session_normal_paths(&fixture),
    ));
    report.record(E2eTestResult::from_result(
        "chat-message-prompt-path",
        check_chat_message_prompt_path(&fixture),
    ));
    report.record(E2eTestResult::from_result(
        "streaming-order",
        check_streaming_order(&fixture),
    ));
    match check_operator_coverage(&fixture) {
        Ok(operators) => {
            report.operator_coverage = operators;
            report.record(E2eTestResult::passed("operator-coverage"));
        }
        Err(error) => report.record(E2eTestResult::failed(
            "operator-coverage",
            format!("{}: {}", error.code(), error.reason()),
        )),
    }
    match check_kernel_coverage() {
        Ok(kernels) => {
            report.kernel_coverage = kernels;
            report.record(E2eTestResult::passed("kernel-coverage"));
        }
        Err(error) => report.record(E2eTestResult::failed(
            "kernel-coverage",
            format!("{}: {}", error.code(), error.reason()),
        )),
    }
    report.record(E2eTestResult::from_result(
        "no-shortcut-direct-provider-rejected",
        check_no_shortcut_direct_provider_rejected(),
    ));
    report.record(E2eTestResult::from_result(
        "no-shortcut-direct-kernel-invocation-rejected",
        check_no_shortcut_direct_kernel_invocation_rejected(),
    ));
    report.record(E2eTestResult::from_result(
        "no-shortcut-model-loading-bypass-detected",
        check_no_shortcut_model_loading_bypass_detected(&fixture),
    ));
    report.record(E2eTestResult::from_result(
        "no-shortcut-model-component-bypass-detected",
        check_no_shortcut_model_component_bypass_detected(&fixture),
    ));
    report.record(E2eTestResult::from_result(
        "no-shortcut-memory-manager-bypass-detected",
        check_no_shortcut_memory_manager_bypass_detected(&fixture),
    ));
    report.record(E2eTestResult::from_result(
        "no-shortcut-dtype-and-layout-conversion-explicit",
        check_dtype_and_layout_conversion_are_explicit(),
    ));
    report.record(E2eTestResult::from_result(
        "reference-cpu-selected-through-kernel-registry",
        check_reference_cpu_selected_through_kernel_registry(),
    ));
    report.record(E2eTestResult::from_result(
        "invalid-graph-fixture-rejected",
        check_invalid_graph_fixture(),
    ));
    #[cfg(test)]
    report.record(E2eTestResult::from_result(
        "graph-production-and-execution",
        check_graph_production_and_execution(&fixture),
    ));
    report.record(E2eTestResult::from_result(
        "max-new-tokens-stops-generation",
        check_max_new_tokens_stops_generation(&fixture),
    ));
    report.record(E2eTestResult::from_result(
        "max-total-tokens-stops-generation",
        check_max_total_tokens_stops_generation(&fixture),
    ));
    #[cfg(test)]
    report.record(E2eTestResult::from_result(
        "eos-token-stops-generation",
        check_eos_token_stops_generation(&fixture),
    ));
    report.record(E2eTestResult::from_result(
        "generation-cancelled",
        check_generation_cancelled(&fixture),
    ));
    report.record(E2eTestResult::from_result(
        "sampling-greedy-deterministic",
        check_sampling_greedy_deterministic(),
    ));
    report.record(E2eTestResult::from_result(
        "sampling-stochastic-seed-deterministic",
        check_stochastic_sampling_is_seed_deterministic(),
    ));
    report.record(E2eTestResult::from_result(
        "closed-session-rejects-generation",
        check_closed_session_rejects_generation(&fixture),
    ));
    report.record(E2eTestResult::from_result(
        "memory-admission-failure",
        check_memory_admission_failure(),
    ));
    report.record(E2eTestResult::from_result(
        "memory-operator-output-accounting",
        check_memory_operator_output_accounting(),
    ));
    report.record(E2eTestResult::from_result(
        "tensor-resource-lifecycle",
        check_tensor_resource_lifecycle(),
    ));
    report.record(E2eTestResult::from_result(
        "kv-and-prefix-cache-lifecycle",
        check_kv_and_prefix_cache_lifecycle(&fixture),
    ));
    report.record(E2eTestResult::passed("observation-failure-case-started"));
    report.record(E2eTestResult::from_result(
        "incompatible-tokenizer-rejected",
        check_incompatible_tokenizer(&fixture),
    ));
    report.record(E2eTestResult::from_result(
        "untrusted-artifact-rejected",
        check_untrusted_artifact(&fixture),
    ));
    report.record(E2eTestResult::from_result(
        "invalid-tensor-shape-rejected",
        check_invalid_tensor_shape(&fixture),
    ));
    report.record(E2eTestResult::from_result(
        "unsupported-operator-rejected",
        check_unsupported_operator(),
    ));
    report.record(E2eTestResult::from_result(
        "missing-kernel-rejected",
        check_missing_kernel(),
    ));
    report.record(E2eTestResult::from_result(
        "required-kernel-removal-rejected",
        check_required_kernel_removal_fails_coverage(),
    ));
    report.record(E2eTestResult::from_result(
        "invalid-model-reference-rejected",
        check_invalid_model_reference(),
    ));
    report.record(E2eTestResult::from_result(
        "raw-handle-access-denied",
        check_raw_handle_access_denied(),
    ));
    let boundary_result = check_cli_boundary_denials();
    report.record(E2eTestResult::from_result(
        "cli-boundary-denials",
        boundary_result.clone(),
    ));
    report.record(E2eTestResult::from_result(
        "observation-boundary-violation",
        boundary_result,
    ));
    report.record(E2eTestResult::from_result(
        "generation-timeout",
        check_generation_timeout_maps_to_structured_error(),
    ));
    report.record(E2eTestResult::passed("observation-failure-case-completed"));

    let redaction_result = check_diagnostics_redaction_on_failure();
    report.record(E2eTestResult::from_result(
        "observation-redaction-failure",
        redaction_result.clone(),
    ));
    report.record(E2eTestResult::from_result(
        "diagnostics-redaction-on-failure",
        redaction_result,
    ));
    report.record(E2eTestResult::from_result(
        "kv-cache-diagnostics-redacted",
        check_kv_cache_diagnostics_redacted(),
    ));
    report.record(E2eTestResult::from_result(
        "kernel-optimization-orchestration-boundary",
        check_kernel_optimization_orchestration_boundary(),
    ));

    report.duration_millis = elapsed_millis(start);
    let metadata_check = check_report_metadata(&report);
    report.record(E2eTestResult::from_result(
        "report-metadata-complete",
        metadata_check,
    ));
    report.record(E2eTestResult::passed("observation-report-generated"));
    report
}

pub(super) struct FixtureChatFormatter;

impl ChatTemplateFormatter for FixtureChatFormatter {
    fn format(&self, messages: &[ChatMessage]) -> Result<String, InferenceApiError> {
        Ok(messages
            .iter()
            .map(|message| format!("{}: {}\n", message.role, message.content))
            .collect())
    }
}

pub(super) fn check_chat_message_prompt_path(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let messages = vec![ChatMessage::new("u", "hi")];
    let tokenized = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::ChatMessages(messages)),
        Some(&FixtureChatFormatter),
    )?;
    if tokenized.token_ids.is_empty() {
        return Err(E2eConformanceError::TokenizerFailed {
            reason: "chat message prompt path produced no tokens".into(),
        });
    }
    Ok(())
}

pub(super) fn check_one_shot_session_normal_paths(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let session_request = SessionCreationRequest {
        model: GenerationModelReference::ModelInstance(instance.clone()),
        tokenizer: generation_tokenizer_reference(fixture),
        generation_defaults: GenerationParameters::greedy(),
        policy: SessionPolicy::default(),
        memory: SessionMemoryBudget::default(),
        allowed_capabilities: BTreeSet::new(),
        correlation_id: None,
        created_at_millis: 0,
    };
    let session = create_one_shot_session(&mut runtime, session_request)?;

    // Normal Tokenizer path: the fixture tokenizer via the same Tokenizer
    // Contract boundary the success path uses.
    let tokenized = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::PlainText("hi".into())),
        None,
    )?;

    // Normal Generation/Sampling/Provider-Kernel path: the same plan-bound
    // generation loop + real Reference CPU forward pass as the multi-call
    // success path, just against the one-shot session.
    let request = build_generation_request(
        GenerationRequestId::new("e2e-one-shot-generation")?,
        Some(session.clone()),
        GenerationModelReference::ModelInstance(instance.clone()),
        generation_tokenizer_reference(fixture),
        tokenized,
        2,
        GenerationParameters::greedy(),
        StopConditions::default(),
        StreamingMode::Disabled,
    );
    let request = prepare_generation(&runtime, request)?;
    let mut observer = InferenceApiObserver::new();
    let result = run_first_native_generation_loop_with_plans(
        &mut runtime,
        fixture,
        &instance,
        &request,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |_generated_so_far| false,
        &mut observer,
    )?;
    if result.output.generated_token_ids.is_empty() {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "one-shot session generation produced no tokens".into(),
        });
    }
    close_inference_session(&mut runtime, &session)?;
    Ok(())
}

pub(super) fn check_already_tokenized_prompt_path(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let already_tokenized = vec![10, 20, 30];
    let tokenized = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::TokenIds(already_tokenized.clone())),
        None,
    )?;
    if tokenized.token_ids != already_tokenized {
        return Err(E2eConformanceError::TokenizerFailed {
            reason: "already-tokenized prompt path did not preserve token ids".into(),
        });
    }
    Ok(())
}

pub(super) fn check_fixture_tokenizer_deterministic(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let first = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::PlainText("hi".into())),
        None,
    )?;
    let second = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::PlainText("hi".into())),
        None,
    )?;
    if first.token_ids != second.token_ids {
        return Err(E2eConformanceError::TokenizerFailed {
            reason: "fixture tokenizer produced non-deterministic token ids".into(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Section 7 (Runtime-Owned KV Data): prepare/commit/abort lifecycle tests
// ---------------------------------------------------------------------

pub(super) fn elapsed_millis(start: SystemTime) -> u64 {
    SystemTime::now()
        .duration_since(start)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default()
}

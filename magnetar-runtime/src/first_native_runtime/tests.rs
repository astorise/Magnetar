//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test source
//! rather than Runtime implementation source (task 9.1).

use super::*;
use crate::planning::*;
use crate::scheduler::*;

/// The correct prefill/decode Operator-sequence hash for the E2E fixture
/// architecture (see `qwen_operator_sequence_hash`), computed once and
/// hard-coded here the same way the fixture's node counts are: this
/// component is a fixed stand-in for one exact architecture shape, not a
/// dynamic graph compiler.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
const QWEN_COMPONENT_FIXTURE_OPERATOR_HASH: u32 = 0x655b_1541;

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
fn qwen_component_fixture_wat(
    prefill_export: &str,
    decode_export: &str,
    authority_export: &str,
    prefill_hash_export: &str,
    decode_hash_export: &str,
) -> String {
    format!(
        r#"(component
(core module $m
    {prefill_export}
    {decode_export}
    {authority_export}
    {prefill_hash_export}
    {decode_hash_export})
(core instance $i (instantiate $m))
(func (export "prefill-node-count") (result u32)
    (canon lift (core func $i "prefill-node-count")))
(func (export "decode-node-count") (result u32)
    (canon lift (core func $i "decode-node-count")))
(func (export "provider-authority-count") (result u32)
    (canon lift (core func $i "provider-authority-count")))
(func (export "prefill-operator-hash") (result u32)
    (canon lift (core func $i "prefill-operator-hash")))
(func (export "decode-operator-hash") (result u32)
    (canon lift (core func $i "decode-operator-hash")))
(func $prefill-node-count (result u32)
    (canon lift (core func $i "prefill-node-count")))
(func $decode-node-count (result u32)
    (canon lift (core func $i "decode-node-count")))
(func $provider-authority-count (result u32)
    (canon lift (core func $i "provider-authority-count")))
(func $prefill-operator-hash (result u32)
    (canon lift (core func $i "prefill-operator-hash")))
(func $decode-operator-hash (result u32)
    (canon lift (core func $i "decode-operator-hash")))
(instance $qwen-graph-fixture
    (export "prefill-node-count" (func $prefill-node-count))
    (export "decode-node-count" (func $decode-node-count))
    (export "provider-authority-count" (func $provider-authority-count))
    (export "prefill-operator-hash" (func $prefill-operator-hash))
    (export "decode-operator-hash" (func $decode-operator-hash)))
(export "magnetar:qwen/graph-fixture@1.0.0" (instance $qwen-graph-fixture)))
"#
    )
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
fn qwen_component_count_fixture_wat(prefill: u32, decode: u32, authority: u32) -> String {
    qwen_component_fixture_wat(
        &format!(r#"(func (export "prefill-node-count") (result i32) i32.const {prefill})"#),
        &format!(r#"(func (export "decode-node-count") (result i32) i32.const {decode})"#),
        &format!(
            r#"(func (export "provider-authority-count") (result i32) i32.const {authority})"#
        ),
        &format!(
            r#"(func (export "prefill-operator-hash") (result i32) i32.const {})"#,
            QWEN_COMPONENT_FIXTURE_OPERATOR_HASH as i32
        ),
        &format!(
            r#"(func (export "decode-operator-hash") (result i32) i32.const {})"#,
            QWEN_COMPONENT_FIXTURE_OPERATOR_HASH as i32
        ),
    )
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
fn qwen_component_manifest(digest: &str) -> String {
    format!(
        r#"schema: magnetar-component-artifact
schema_version: 1
artifact:
  kind: component
  digest:
    algorithm: sha256
    value: "{digest}"
component:
  name: "magnetar.qwen.graph-fixture"
  version: "0.1.0"
  description: "Executable Qwen graph fixture component"
  role: "qwen-graph-fixture"
runtime:
  magnetar:
    min_version: "0.1.0"
wit:
  imports: []
  exports:
    - package: "magnetar:qwen"
      interface: "graph-fixture"
      version: "1.0.0"
capabilities:
  requires: []
authority:
  requires: []
engine:
  profile: "native"
  features:
    - component-model
    - resource-limits
publisher:
  id: "local-dev"
  name: "Local Development"
source:
  kind: "local"
  uri: "./qwen-graph.component.wat"
signatures: []
"#
    )
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
fn sha256_component_digest(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::from("sha256:");
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in digest {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
fn qwen_component_preflight_package(
    wat: &str,
    manifest_digest: Option<&str>,
) -> (ComponentArtifactPackage, String) {
    let digest = sha256_component_digest(wat.as_bytes());
    let package = ComponentArtifactPackage::new(
        wat.as_bytes().to_vec(),
        qwen_component_manifest(manifest_digest.unwrap_or(&digest)).into_bytes(),
        ComponentDigest::parse("sha256", manifest_digest.unwrap_or(&digest)),
        ComponentDistributionSource::new(
            ComponentDistributionSourceKind::DevelopmentFixture,
            QWEN_GRAPH_COMPONENT_NAME,
        ),
    );
    (package, digest)
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
fn trusted_preflight_request_for_temp_component(
    component_package: ComponentArtifactPackage,
    digest: &str,
) -> QwenComponentPreflightRequest {
    QwenComponentPreflightRequest {
        component_package,
        trust_store: ComponentTrustStore::default().trust_digest(digest),
        limits: qwen_component_runtime_limits(),
    }
}

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
fn e2e_fixture_model_passes_validation() {
    let fixture = e2e_fixture().expect("fixture builds and validates");
    fixture.manifest.validate().expect("manifest re-validates");
    assert_eq!(
        fixture.identity.implementation,
        ModelComponentImplementationKind::WebAssemblyComponent
    );
    assert_eq!(
        fixture.config.architecture.vocabulary_size,
        E2E_FIXTURE_VOCAB
    );
    assert_eq!(fixture.config.architecture.hidden_size, E2E_FIXTURE_HIDDEN);
    assert_eq!(fixture.config.architecture.layer_count, E2E_FIXTURE_LAYERS);
}

#[test]
fn e2e_fixture_weight_digest_is_stable() {
    let fixture = e2e_fixture().expect("fixture builds");
    let digest = e2e_fixture_weight_digest(&fixture.weights);
    assert_eq!(digest, E2E_FIXTURE_WEIGHT_DIGEST);
    assert_eq!(digest, e2e_fixture_weight_digest(&fixture.weights));
}

#[test]
fn e2e_fixture_tokenizer_produces_deterministic_tokens() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_fixture_tokenizer_deterministic(&fixture).expect("tokenization is deterministic");
}

#[test]
fn e2e_graph_nodes_transport_stays_tensor_value_typed() {
    check_execute_qwen_graph_nodes_transport_has_no_host_tensor_typed_calls().expect(
        "execute_qwen_graph_nodes's per-node transport has no direct HostTensor-typed calls",
    );
}

#[test]
fn e2e_already_tokenized_prompt_path_bypasses_text_tokenization() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_already_tokenized_prompt_path(&fixture).expect("already-tokenized path is preserved");
}

#[test]
fn e2e_raw_prompt_logging_is_disabled_by_default() {
    assert!(!SessionPolicy::default().raw_prompt_logging_allowed);
}

#[test]
fn e2e_required_path_returns_usage_and_cleans_up() {
    let fixture = e2e_fixture().expect("fixture builds");
    let result = run_success_path(&fixture).expect("success path returns output");
    assert!(result.generation_result.output.usage.generated_tokens > 0);
    assert!(!result.observer.observations().is_empty());
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
fn e2e_generation_step_logits_are_produced_by_the_evidence_bearing_dispatch() {
    let fixture = e2e_fixture().expect("fixture builds");
    let runtime = build_runtime();
    let sequence = vec![1u32, 2u32];

    let normed_final =
        e2e_forward_hidden_states(&fixture, &sequence).expect("hidden states computed");
    let token_embedding = fixture_tensor_by_name(&fixture.weights, "token_embedding")
        .expect("token embedding present");
    let token_embedding_transposed =
        transpose_rows_cols(token_embedding).expect("token embedding transposes");

    let (dispatch_result, dispatched_output) =
        dispatch_matmul(&runtime, &normed_final, &token_embedding_transposed)
            .expect("real matmul dispatch succeeds");
    assert_eq!(dispatch_result.status, KernelResultStatus::Succeeded);

    let vocab = fixture.config.architecture.vocabulary_size as usize;
    let last_row_start = (sequence.len() - 1) * vocab;
    let dispatched_logits = &dispatched_output.data[last_row_start..last_row_start + vocab];

    // What `E2eRuntimeModelExecutionEngine::execute_generation_step` returns
    // for this sequence must equal the dispatch's own output exactly --
    // it is read directly from `dispatched_output`, never recomputed
    // separately -- so this also confirms the dispatch path is numerically
    // correct against the independent `e2e_forward` ground truth.
    let expected = e2e_forward(&fixture, &sequence).expect("forward pass produces logits");
    assert_eq!(dispatched_logits, expected.as_slice());

    // Tampering with the dispatch's actual input changes its output,
    // proving the returned data is causally produced by this dispatch --
    // not decorated onto an unrelated proof computation whose result is
    // discarded, which is the shortcut this test guards against.
    let corrupted_embedding = HostTensor::new(
        token_embedding_transposed.shape.clone(),
        vec![0.0_f32; token_embedding_transposed.data.len()],
    )
    .expect("zeroed tensor constructs");
    let (corrupted_result, corrupted_output) =
        dispatch_matmul(&runtime, &normed_final, &corrupted_embedding)
            .expect("corrupted dispatch still succeeds");
    assert_eq!(corrupted_result.status, KernelResultStatus::Succeeded);
    assert_ne!(
        &corrupted_output.data[last_row_start..last_row_start + vocab],
        dispatched_logits
    );
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
fn e2e_graph_production_and_execution_succeeds_for_valid_fixture() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_graph_production_and_execution(&fixture).expect("prefill/decode graphs execute");
}

#[test]
fn e2e_max_new_tokens_reached_stops_generation() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_max_new_tokens_stops_generation(&fixture).expect("max token stop is honored");
}

#[test]
fn e2e_eos_token_stops_generation() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_eos_token_stops_generation(&fixture).expect("EOS stop is honored");
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
fn e2e_first_native_generation_requires_ready_model_instance() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_first_native_generation_requires_ready_model_instance(&fixture)
        .expect("non-ready model instance is rejected");
}

#[test]
fn e2e_missing_prepared_plan_fails_closed() {
    check_missing_prepared_plan_fails_closed().expect("missing prepared plan is rejected");
}

#[test]
fn e2e_invalidated_prepared_plan_rejects_new_work() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_invalidated_prepared_plan_rejects_new_work(&fixture)
        .expect("invalidated prepared plan is rejected");
}

#[test]
fn e2e_stale_plan_outside_policy_fails_closed() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_stale_plan_outside_policy_fails_closed(&fixture)
        .expect("plan stale outside its rebuild policy is rejected");
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
fn e2e_graph_dispatch_ignores_kernel_registry_preference_change_after_plan_publication() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_graph_dispatch_ignores_kernel_registry_preference_change_after_plan_publication(&fixture)
        .expect(
            "a Kernel Registry preference change after Plan publication does not affect an \
             already-published, ready Plan",
        );
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

/// Non-Reference-CPU Kernel-level `ProviderExecutionApi` implementation used
/// only to prove that generic resolution (Correctif 3) reaches whatever
/// Provider Runtime has registered, not specifically `ReferenceCpuExecutor`.
/// Its "kernel" is a minimal, deterministic copy of the sole input resource
/// to the sole output resource -- not a real Kernel catalog implementation.
struct MockKernelExecutor {
    storage: Mutex<BTreeMap<TensorResourceId, HostTensor>>,
}
impl MockKernelExecutor {
    fn new() -> Self {
        Self {
            storage: Mutex::new(BTreeMap::new()),
        }
    }
}
impl ProviderExecutionApi for MockKernelExecutor {
    fn submit(
        &self,
        request: ProviderExecutionRequest,
    ) -> Result<ProviderExecutionHandle, ProviderExecutionError> {
        Ok(ProviderExecutionHandle::new(
            request.operation,
            request.plan.id.clone(),
            request.provider.clone(),
            request.device.clone(),
        ))
    }
    fn status(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderExecutionStatus, ProviderExecutionError> {
        Ok(ProviderExecutionStatus::new(
            handle.clone(),
            SchedulingState::Completed,
        ))
    }
    fn cancel(
        &self,
        _handle: &ProviderExecutionHandle,
    ) -> Result<ProviderCancellationOutcome, ProviderExecutionError> {
        Ok(ProviderCancellationOutcome::Unsupported)
    }
    fn complete(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderExecutionResult, ProviderExecutionError> {
        Ok(ProviderExecutionResult::completed(
            handle.clone(),
            Vec::new(),
        ))
    }
    fn release(&self, _handle: ProviderExecutionHandle) -> Result<(), ProviderExecutionError> {
        Ok(())
    }
    fn submit_kernel(
        &self,
        _advertisement: &KernelAdvertisement,
        _operator: &OperatorSpec,
        invocation: &KernelInvocation,
        _memory: &mut MemoryManager,
    ) -> Result<ProviderExecutionHandle, ProviderExecutionError> {
        if let (Some(input), Some(output)) = (invocation.inputs.first(), invocation.outputs.first())
        {
            let tensor = self
                .storage
                .lock()
                .unwrap()
                .get(&input.resource.id)
                .cloned();
            if let Some(tensor) = tensor {
                self.storage
                    .lock()
                    .unwrap()
                    .insert(output.resource.id.clone(), tensor);
            }
        }
        Ok(ProviderExecutionHandle::new(
            ScheduledOperationId::new(1),
            ExecutionPlanId::new(invocation.id.as_str().to_string()),
            ProviderBinding::new("magnetar:provider/mock-kernel"),
            None,
        ))
    }
    fn complete_kernel(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<KernelResult, ProviderExecutionError> {
        Ok(KernelResult::success(KernelInvocationId::new(
            handle.plan.as_str(),
        )))
    }
    fn write_tensor(&self, id: TensorResourceId, tensor: HostTensor) {
        self.storage.lock().unwrap().insert(id, tensor);
    }
    fn read_tensor(&self, id: &TensorResourceId) -> Option<HostTensor> {
        self.storage.lock().unwrap().get(id).cloned()
    }
    fn write_tensor_value(&self, id: TensorResourceId, value: TensorValue) {
        if let TensorValue::Host(tensor) = value {
            self.write_tensor(id, tensor);
        }
    }
    fn read_tensor_value(&self, id: &TensorResourceId) -> Option<TensorValue> {
        self.read_tensor(id).map(TensorValue::Host)
    }
}

/// A minimal, non-Reference-CPU Provider execution API implementation that
/// never exposes host-visible bytes for any resource
/// (`define-provider-prepared-kernel-execution-contract`): every
/// `read_tensor_value` answers [`TensorValue::Opaque`], and the
/// `HostTensor`-typed `read_tensor`/`write_tensor` pair (which this
/// contract deliberately leaves in place for callers that want it, see
/// that trait's documentation) is simply never implemented, defaulting to
/// "nothing". Exists to prove [`TensorValue::into_host`]'s
/// residency-unavailable error fires against a real, independent
/// implementation of the Provider-agnostic contract, not only against
/// Reference CPU (which never produces `Opaque`).
struct DeviceResidentOnlyExecutor {
    resources: Mutex<BTreeSet<TensorResourceId>>,
}
impl DeviceResidentOnlyExecutor {
    fn new() -> Self {
        Self {
            resources: Mutex::new(BTreeSet::new()),
        }
    }
}
impl ProviderExecutionApi for DeviceResidentOnlyExecutor {
    fn submit(
        &self,
        request: ProviderExecutionRequest,
    ) -> Result<ProviderExecutionHandle, ProviderExecutionError> {
        Ok(ProviderExecutionHandle::new(
            request.operation,
            request.plan.id.clone(),
            request.provider.clone(),
            request.device.clone(),
        ))
    }
    fn status(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderExecutionStatus, ProviderExecutionError> {
        Ok(ProviderExecutionStatus::new(
            handle.clone(),
            SchedulingState::Completed,
        ))
    }
    fn cancel(
        &self,
        _handle: &ProviderExecutionHandle,
    ) -> Result<ProviderCancellationOutcome, ProviderExecutionError> {
        Ok(ProviderCancellationOutcome::Unsupported)
    }
    fn complete(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderExecutionResult, ProviderExecutionError> {
        Ok(ProviderExecutionResult::completed(
            handle.clone(),
            Vec::new(),
        ))
    }
    fn release(&self, _handle: ProviderExecutionHandle) -> Result<(), ProviderExecutionError> {
        Ok(())
    }
    fn write_tensor_value(&self, id: TensorResourceId, _value: TensorValue) {
        self.resources.lock().unwrap().insert(id);
    }
    fn read_tensor_value(&self, id: &TensorResourceId) -> Option<TensorValue> {
        if self.resources.lock().unwrap().contains(id) {
            Some(TensorValue::Opaque)
        } else {
            None
        }
    }
}

struct MockKernelProvider {
    executor: Arc<MockKernelExecutor>,
}
impl MockKernelProvider {
    fn new() -> Self {
        Self {
            executor: Arc::new(MockKernelExecutor::new()),
        }
    }
}
impl Provider for MockKernelProvider {
    fn metadata(&self) -> ProviderMetadata {
        ProviderMetadata::new(
            "magnetar:provider/mock-kernel",
            "0.0.0",
            "test",
            "Non-Reference-CPU mock Provider proving generic Kernel dispatch (Correctif 3)",
        )
    }
    fn register(&self, _registry: &mut ProviderRegistry) -> Result<(), ProviderError> {
        Ok(())
    }
    fn execution_api(&self) -> Option<Arc<dyn ProviderExecutionApi>> {
        Some(self.executor.clone())
    }
}

#[test]
fn provider_execution_generic_resolution_reaches_non_reference_cpu_provider() {
    let provider_binding = ProviderBinding::new("magnetar:provider/mock-kernel");
    let runtime = Runtime::builder()
        .register_provider(Arc::new(MockKernelProvider::new()))
        .build()
        .expect("mock provider registers cleanly");

    let api = resolve_kernel_execution_provider(&runtime, &provider_binding).expect(
        "generic resolution finds the registered mock provider without downcasting to a \
         concrete Provider type",
    );

    // Borrow an arbitrary, valid advertisement/operator pair from Reference
    // CPU purely as inert filler: the mock ignores their content entirely,
    // and constructing one from scratch is not the point of this test.
    let reference_cpu = ReferenceCpuProvider::new();
    let advertisement = reference_cpu
        .kernel_advertisements()
        .into_iter()
        .next()
        .expect("Reference CPU advertises at least one Kernel");
    let catalog = initial_operator_catalog();
    let operator = catalog.get(&advertisement.implemented_operator).unwrap();

    let input_id = TensorResourceId::new("mock-input");
    let output_id = TensorResourceId::new("mock-output");
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let affinity = ResourceAffinity::new(FallbackClass::Transparent);
    api.write_tensor(
        input_id.clone(),
        HostTensor::new([2], vec![1.0, 2.0]).unwrap(),
    );
    let invocation = KernelInvocation::new(
        KernelInvocationId::new("mock-invocation"),
        advertisement.implemented_operator.clone(),
        advertisement.id.clone(),
        provider_binding.clone(),
        affinity.clone(),
    )
    .with_input(KernelResource::new(
        TensorResourceDescriptor::new(input_id, descriptor.clone(), affinity.clone()),
        KernelMemoryClass::Host,
    ))
    .with_output(KernelResource::new(
        TensorResourceDescriptor::new(output_id.clone(), descriptor, affinity),
        KernelMemoryClass::Host,
    ));

    let mut memory = MemoryManager::new(MemoryManagerConfig::default());
    let handle = api
        .submit_kernel(&advertisement, operator, &invocation, &mut memory)
        .expect("mock Provider implements Kernel-level submission");
    let result = api
        .complete_kernel(&handle)
        .expect("mock Provider implements Kernel-level completion");
    assert_eq!(result.status, KernelResultStatus::Succeeded);
    let output = api
        .read_tensor(&output_id)
        .expect("mock Provider's minimal kernel copied input to output");
    assert_eq!(output.data, vec![1.0, 2.0]);
}

/// `define-provider-prepared-kernel-execution-contract`: a Provider that
/// only ever answers [`TensorValue::Opaque`] (never `Host`) is a valid,
/// independent implementation of the Provider-agnostic tensor value
/// contract -- resolved the same generic way as any other Provider
/// (`Provider::execution_api`, no downcasting), and
/// [`TensorValue::into_host`] SHALL fail with a structured,
/// resource-naming error for it rather than panicking or silently
/// fabricating bytes.
#[test]
fn tensor_value_into_host_fails_structurally_for_a_device_resident_only_provider() {
    let executor = DeviceResidentOnlyExecutor::new();
    let id = TensorResourceId::new("device-only-resource");

    // Nothing written yet: genuinely absent, distinct from "present but
    // opaque".
    assert!(executor.read_tensor_value(&id).is_none());

    executor.write_tensor_value(id.clone(), TensorValue::Opaque);
    let value = executor
        .read_tensor_value(&id)
        .expect("resource is present after being written");
    assert!(matches!(value, TensorValue::Opaque));

    match value.into_host(&id) {
        Err(TensorError::ResidencyUnavailable { reason }) => {
            assert!(
                reason.contains(id.as_str()),
                "expected the residency-unavailable error to name the resource id; got: {reason}"
            );
        }
        Err(other) => panic!("expected ResidencyUnavailable, got {other:?}"),
        Ok(_) => panic!("into_host must not succeed for an Opaque value"),
    }
}

/// Static check (Correctif 3 Definition of Done): first-native dispatch
/// must never recover a concrete Provider type from `&dyn Provider` via
/// `downcast_ref`. Reads this module's own source text rather than
/// asserting on run-time behavior, since the property being enforced is the
/// absence of a coding pattern, not a computed result.
#[test]
fn first_native_dispatch_source_contains_no_provider_downcast() {
    let source = include_str!("../first_native_runtime.rs");
    assert!(
        !source.contains("downcast_ref"),
        "first_native_runtime.rs must not recover a concrete Provider type via \
         downcast_ref; resolve through ProviderExecutionApi (Provider::execution_api) instead"
    );
    assert!(
        !source.contains("as &dyn std::any::Any") && !source.contains("as &dyn Any"),
        "first_native_runtime.rs must not cast a Provider reference to dyn Any"
    );
}

/// Static guard (Correctif 7 / task group 10; superseded at task 12.6, per
/// explicit direction to remove the `non-strict-fixture-fallback` opt-in
/// entirely rather than keep it as a production escape hatch). The prior
/// version of this guard verified that the Rust-synthesized Qwen graph
/// fallback required an explicit Cargo feature to even *compile* into a
/// production build -- `cargo check --no-default-features` failed to
/// compile without it, a real but blunt fail-closed mechanism (an
/// environment that cannot get a real Component engine could not produce a
/// runnable binary at all, only a compile error). That opt-in is gone now:
/// production has exactly one graph-producing branch of
/// `first_native_component_graphs_for_prompt` (and the sibling inline
/// blocks in `run_success_path_with_prompt`/`FirstNativeChatSession::turn`)
/// when a strict Component engine is unavailable, and it is an unconditional,
/// structured runtime error -- not a second, unattested Rust-synthesized
/// graph, and not a compile failure either. This guard checks, by source
/// inspection (the only way to check something a `not(test)` cfg makes
/// impossible to invoke directly from a `#[test]`, and the same technique
/// the guard it replaces already used), that: (1) every production,
/// non-test cfg branch that used to require `feature =
/// "non-strict-fixture-fallback"` is gone from `first_native_runtime.rs`
/// entirely -- the feature no longer exists in `Cargo.toml`, so if this
/// string reappears here it means someone reintroduced a production
/// opt-in fallback; and (2) the fail-closed production stub's error
/// message is still present, at the expected count of call sites
/// (`first_native_component_graphs_for_prompt`, `run_success_path_with_prompt`,
/// `FirstNativeChatSession::turn`).
#[test]
fn first_native_dispatch_has_no_production_fallback_and_fails_closed_instead() {
    let source = include_str!("../first_native_runtime.rs");
    assert!(
        !source.contains("non-strict-fixture-fallback"),
        "the non-strict-fixture-fallback Cargo feature was deliberately removed (task 12.6); \
         its reappearance in first_native_runtime.rs means a production opt-in fallback to the \
         unattested Rust-synthesized graph was reintroduced"
    );
    let fail_closed_occurrences = source
        .matches("no Component engine is available on this build target")
        .count();
    assert_eq!(
        fail_closed_occurrences, 3,
        "expected exactly the three documented production fail-closed stubs \
         (first_native_component_graphs_for_prompt, run_success_path_with_prompt, \
         FirstNativeChatSession::turn) to share this exact structured error message when no \
         strict Component engine is available; found {fail_closed_occurrences}. If a call site \
         was added or removed, update this count."
    );
}

/// Static guard (`reach-architecture-freeze-1` task 12.4): production's
/// Qwen Component loader (`qwen_real_component_package`'s `not(test)`
/// branch) must never embed the real Component binary via `include_bytes!`
/// or claim `ComponentDistributionSourceKind::DevelopmentFixture` -- those
/// are only legitimate in the `#[cfg(test)]` branch, checked separately
/// below. `not(test)` code cannot be invoked from a `#[test]` at all (that
/// is the whole point of the cfg), so this is a source-text check, the same
/// technique the sibling guards above already use for the same structural
/// reason.
#[test]
fn qwen_component_production_loader_has_no_embedded_fixture() {
    let source = include_str!("../first_native_runtime.rs");
    let production_loader_start = source
        .find("not(test)\n))]\nfn qwen_real_component_package()")
        .expect("the production (not(test)) qwen_real_component_package overload exists");
    let production_loader_end = production_loader_start
        + source[production_loader_start..]
            .find("\n}\n")
            .expect("the production loader function has a closing brace");
    let production_loader_source = &source[production_loader_start..production_loader_end];
    assert!(
        !production_loader_source.contains("include_bytes!"),
        "production's Qwen Component loader must never embed the Component binary via \
         include_bytes! -- that is a test-fixture-only mechanism (task 12.4)"
    );
    assert!(
        !production_loader_source.contains("DevelopmentFixture"),
        "production's Qwen Component loader must never claim \
         ComponentDistributionSourceKind::DevelopmentFixture -- it resolves bytes externally \
         and should report LocalDirectory (or another real source kind), not a fixture"
    );
    assert!(
        production_loader_source.contains("std::env::var")
            && production_loader_source.contains("std::fs::read"),
        "expected production's Qwen Component loader to resolve bytes from an external, \
         caller-configured source (env var + local file read) rather than an embedded fixture"
    );
}

/// Static guard (Correctif 13 / task group 7): `execute_qwen_graph` used to
/// require `std::mem::take(runtime.memory_mut())` to get an independent
/// `&mut MemoryManager` alongside a `&Runtime`; that gap is closed now that
/// `QwenDispatchContext` holds a single `&mut Runtime` instead of separate
/// `runtime`/`memory` fields (see its doc comment), so no first-native
/// dispatch code should call `std::mem::take` on the Runtime memory service
/// at all. This fails if the pattern reappears -- prefer cloning the
/// specific value borrowed from `runtime` (e.g. a `KernelAdvertisement`)
/// instead of taking the whole `MemoryManager`.
#[test]
fn first_native_dispatch_never_takes_runtime_memory_manager() {
    let source = include_str!("../first_native_runtime.rs");
    let occurrences = source
        .matches("std::mem::take(runtime.memory_mut())")
        .count();
    assert_eq!(
        occurrences, 0,
        "expected zero std::mem::take(runtime.memory_mut()) calls in first_native_runtime.rs; \
         found {occurrences}. Hold a single `&mut Runtime` (see `QwenDispatchContext`) and clone \
         out any value that must outlive a later `memory_mut()` call instead of taking the whole \
         MemoryManager."
    );
}

#[test]
fn e2e_graph_dispatch_accounts_outputs_through_runtime_memory_manager() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_graph_dispatch_accounts_outputs_through_runtime_memory_manager(&fixture)
        .expect("graph dispatch accounts outputs through Runtime's MemoryManager");
}

#[test]
fn e2e_graph_dispatch_intermediate_edge_is_resolvable_from_provider_storage() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_graph_dispatch_intermediate_edge_is_resolvable_from_provider_storage(&fixture)
        .expect("intermediate graph edges resolve from Provider storage, not a private cache");
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

/// `materialize-weights-from-real-model-artifact` task 3.1: proves the
/// real-artifact-bytes path (`e2e_fixture_weights_from_real_artifact`,
/// reading the checked-in `E2E_FIXTURE_SAFETENSORS_BYTES` at real offsets)
/// produces the exact same materialized tensors as the pre-existing
/// in-memory construction (`e2e_fixture_weights`) -- before
/// `bind_qwen_fixture_weights`'s production call site is allowed to switch
/// from one to the other. Equivalence proven, not assumed, matching this
/// session's own working pattern for prior real-Component/real-artifact
/// cutovers.
#[test]
fn e2e_fixture_real_artifact_weights_match_in_memory_weights() {
    let config = e2e_fixture_config();
    let in_memory = e2e_fixture_weights(&config).expect("in-memory fixture weights build");
    let from_real_artifact =
        e2e_fixture_weights_from_real_artifact(&config).expect("real-artifact weights materialize");

    assert_eq!(
        in_memory.keys().collect::<Vec<_>>(),
        from_real_artifact.keys().collect::<Vec<_>>(),
        "real-artifact and in-memory weight maps must cover the same tensor names"
    );
    for (name, expected_tensor) in &in_memory {
        let actual_tensor = &from_real_artifact[name];
        assert_eq!(
            actual_tensor.shape, expected_tensor.shape,
            "tensor '{name}' shape mismatch between real-artifact and in-memory paths"
        );
        assert_eq!(
            actual_tensor.data, expected_tensor.data,
            "tensor '{name}' data mismatch between real-artifact and in-memory paths"
        );
    }
}

#[test]
fn e2e_weight_binding_rejects_tampered_artifact_bytes() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_weight_binding_rejects_tampered_artifact_bytes(&fixture)
        .expect("model loading rejects a weight artifact with tampered bytes");
}

#[test]
fn e2e_weight_materialization_failure_demotes_instance_from_ready() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_weight_materialization_failure_demotes_instance_from_ready(&fixture)
        .expect("a Model Instance is demoted away from Ready when weight materialization fails");
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
        limits: qwen_component_runtime_limits(),
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
fn e2e_qwen_component_fuel_exhaustion_fails_before_planning() {
    let wat = qwen_component_fixture_wat(
        r#"(func (export "prefill-node-count") (result i32)
            (loop $again br $again)
            i32.const 13)"#,
        r#"(func (export "decode-node-count") (result i32) i32.const 19)"#,
        r#"(func (export "provider-authority-count") (result i32) i32.const 0)"#,
        &format!(
            r#"(func (export "prefill-operator-hash") (result i32) i32.const {})"#,
            QWEN_COMPONENT_FIXTURE_OPERATOR_HASH as i32
        ),
        &format!(
            r#"(func (export "decode-operator-hash") (result i32) i32.const {})"#,
            QWEN_COMPONENT_FIXTURE_OPERATOR_HASH as i32
        ),
    );
    let (component_package, digest) = qwen_component_preflight_package(&wat, None);
    let mut request = trusted_preflight_request_for_temp_component(component_package, &digest);
    request.limits.engine_execution_budget = Some(1_000);

    let result = validate_and_instantiate_qwen_component_before_first_native_planning(request);

    assert!(
        matches!(
            result,
            Err(E2eConformanceError::ModelComponentFailed { .. })
        ),
        "runaway Qwen Component was not stopped by fuel: {result:?}"
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
fn e2e_kv_cache_diagnostics_redact_raw_contents() {
    check_kv_cache_diagnostics_redacted().expect("cache usage carries no raw contents");
}

#[test]
fn e2e_incremental_decode_uses_existing_kv_and_matches_full_sequence_oracle() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_incremental_decode_matches_full_sequence_oracle(&fixture)
        .expect("incremental decode matches full-sequence oracle");
}

#[test]
fn e2e_graph_executor_matches_full_sequence_oracle() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_graph_executor_matches_full_sequence_oracle(&fixture)
        .expect("graph executor logits and KV state match full-sequence oracle");
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
fn e2e_graph_executor_rejects_cyclic_graph() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_graph_executor_rejects_cyclic_graph(&fixture).expect("graph executor rejects a cycle");
}

#[test]
fn e2e_graph_executor_rejects_removed_producer_node() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_graph_executor_rejects_removed_producer_node(&fixture)
        .expect("graph executor rejects a graph with a removed producer node");
}

#[test]
fn e2e_graph_executor_logits_provenance_requires_declared_output_edge() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_graph_executor_logits_provenance_requires_declared_output_edge(&fixture)
        .expect("graph executor never fabricates a 'logits' binding");
}

#[test]
fn e2e_generation_loop_decode_positions_follow_generated_tokens() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_generation_loop_decode_positions_follow_generated_tokens(&fixture)
        .expect("generation loop decode positions match the position oracle");
}

#[test]
fn e2e_generation_loop_executes_published_plan_bindings() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_generation_loop_executes_published_plan_bindings(&fixture)
        .expect("generation loop fails closed when a published plan binding is missing");
}

#[test]
fn e2e_incremental_decode_rejects_missing_layer_kv() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_incremental_decode_rejects_missing_layer_kv(&fixture)
        .expect("decode requires existing layer KV state");
}

#[test]
fn e2e_tensor_output_updates_readiness_without_raw_pointer() {
    let fixture = e2e_fixture().expect("fixture builds");
    let logits = e2e_forward(&fixture, &[1, 2]).expect("forward pass produces logits");
    assert_eq!(logits.len(), E2E_FIXTURE_VOCAB as usize);
    assert!(logits.iter().all(|value| value.is_finite()));
}

#[test]
fn e2e_resource_cleanup_after_generation_and_session_close() {
    let fixture = e2e_fixture().expect("fixture builds");
    let mut runtime = build_runtime();
    let (instance, _memory) = load_fixture_instance(&fixture, &mut runtime).expect("loads");
    let report = unload_model_instance(
        &mut runtime,
        &instance,
        ModelInstanceUnloadPolicy::DrainActiveUse,
    )
    .expect("unload succeeds");
    assert!(!report.dangling_session_references);
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
fn e2e_local_suite_does_not_require_tachyon_or_browser() {
    // Browser support is explicit and structured, never assumed.
    let _ = qwen_browser_supported(ModelComponentImplementationKind::RuntimeNative);
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
fn e2e_fixture_tokenizer_streams_decode_across_multiple_chunks() {
    let fixture = e2e_fixture().expect("fixture builds");
    let full = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::PlainText("hi!".into())),
        None,
    )
    .expect("tokenizes");
    assert!(full.token_ids.len() >= 2);

    let mut state = StreamingDecodeState::default();
    let mut decoded = String::new();
    for token_id in &full.token_ids {
        let output = fixture
            .tokenizer
            .streaming_decode(state, vec![*token_id], false)
            .expect("streaming decode step succeeds");
        decoded.push_str(&output.text);
        state = output.pending_partial_state.unwrap_or_default();
    }
    assert_eq!(decoded, "hi!");
}

#[test]
fn e2e_one_shot_session_uses_normal_model_instance_and_tokenizer_path() {
    let fixture = e2e_fixture().expect("fixture builds");
    let mut runtime = build_runtime();
    let (instance, _memory) = load_fixture_instance(&fixture, &mut runtime).expect("loads");
    let session_request = SessionCreationRequest {
        model: GenerationModelReference::ModelInstance(instance),
        tokenizer: generation_tokenizer_reference(&fixture),
        generation_defaults: GenerationParameters::greedy(),
        policy: SessionPolicy::default(),
        memory: SessionMemoryBudget::default(),
        allowed_capabilities: BTreeSet::new(),
        correlation_id: None,
        created_at_millis: 0,
    };
    let session = create_one_shot_session(&mut runtime, session_request).expect("creates");
    let status = session_status(
        &runtime,
        &session,
        &SessionAccessPolicy::authorize(session.clone()),
    )
    .expect("status is readable");
    assert_eq!(status.lifecycle, SessionLifecycleState::Ready);
    close_inference_session(&mut runtime, &session).expect("closes");
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
fn e2e_chat_message_prompt_without_formatter_is_policy_denied() {
    let fixture = e2e_fixture().expect("fixture builds");
    let messages = vec![ChatMessage::new("user", "hi")];
    let result = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::ChatMessages(messages)),
        None,
    );
    assert!(matches!(
        result,
        Err(InferenceApiError::PolicyDenied { .. })
    ));
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

#[test]
fn e2e_authoritative_path_collects_correlated_runtime_observations() {
    let fixture = e2e_fixture().expect("fixture builds");
    let outcome = run_success_path(&fixture).expect("success path runs");
    let observations = outcome.observer.observations();
    // ComponentValidated/ComponentInstantiated are only emitted by the real
    // Qwen Component's strict-path preflight (`build_first_native_graphs_
    // from_real_qwen_component`); a test build without a strict Component
    // engine takes the test-oracle fallback instead (task 12.6), which
    // never claims to validate/instantiate a real Component, so asserting
    // this evidence only makes sense when the strict engine is what this
    // build actually exercises.
    #[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
    for kind in [
        InferenceApiObservationKind::ComponentValidated,
        InferenceApiObservationKind::ComponentInstantiated,
    ] {
        assert!(
            observations
                .iter()
                .any(|observation| observation.kind == kind),
            "missing authoritative observation {kind:?}"
        );
    }
    for kind in [
        InferenceApiObservationKind::ModelInstanceReady,
        InferenceApiObservationKind::GraphValidationCompleted,
        InferenceApiObservationKind::PlanSelected,
        InferenceApiObservationKind::PlanGuardAccepted,
        InferenceApiObservationKind::KernelResolved,
        InferenceApiObservationKind::KernelPrepared,
        InferenceApiObservationKind::ProviderSubmitted,
        InferenceApiObservationKind::ProviderCompleted,
        InferenceApiObservationKind::KvCacheCommitted,
        InferenceApiObservationKind::LogitsProduced,
        InferenceApiObservationKind::SamplingCompleted,
        InferenceApiObservationKind::TokenCommitted,
        InferenceApiObservationKind::GraphNodeReady,
        InferenceApiObservationKind::PlanBindingResolved,
        InferenceApiObservationKind::PreparedKernelResolved,
        InferenceApiObservationKind::TensorResourceProduced,
        InferenceApiObservationKind::KvUpdatePrepared,
        InferenceApiObservationKind::KvUpdateCommitted,
    ] {
        assert!(
            observations
                .iter()
                .any(|observation| observation.kind == kind),
            "missing authoritative observation {kind:?}"
        );
    }
    // Correctif 17 / task group 17: these six are per-*node*, not global --
    // a real multi-node graph run must produce more than one, each
    // correlated to a different node, not one repeated event.
    let distinct_ready_nodes: std::collections::BTreeSet<&str> = observations
        .iter()
        .filter(|observation| observation.kind == InferenceApiObservationKind::GraphNodeReady)
        .filter_map(|observation| {
            observation
                .message
                .split("node=")
                .nth(1)
                .map(|rest| rest.split(' ').next().unwrap_or(rest))
        })
        .collect();
    assert!(
        distinct_ready_nodes.len() > 1,
        "expected GraphNodeReady for more than one distinct graph node, found: \
         {distinct_ready_nodes:?}"
    );
    assert!(observations.iter().any(|observation| {
        observation.kind == InferenceApiObservationKind::TensorResourceProduced
            && observation.message.contains("node=")
            && observation.message.contains("resource=")
    }));
    assert!(observations.iter().any(|observation| {
        observation.kind == InferenceApiObservationKind::PlanSelected
            && observation.message.contains("request=e2e-success-path")
            && observation.message.contains("plan_generation=")
    }));
    assert!(observations.iter().any(|observation| {
        observation.kind == InferenceApiObservationKind::KernelResolved
            && observation.message.contains("kernel=")
            && observation.message.contains("provider=")
            && observation.message.contains("model_instance=")
    }));
    assert!(observations.iter().any(|observation| {
        observation.kind == InferenceApiObservationKind::PlanSelected
            && observation.message.contains("phase=decode")
            && observation.message.contains("kv_position=")
    }));
    assert!(
        outcome
            .kv_observations
            .iter()
            .any(|observation| observation.kind == KvCacheObservationKind::PrefillCompleted)
    );
    assert!(
        outcome
            .kv_observations
            .iter()
            .any(|observation| observation.kind == KvCacheObservationKind::DecodeAppend)
    );
    assert!(outcome.kv_observations.iter().all(|observation| {
        !observation.raw_prompt_available
            && !observation.raw_cache_available
            && !observation.raw_provider_handle_available
    }));
}

#[test]
fn e2e_kv_sampling_failure_leaves_cache_uncommitted() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_kv_sampling_failure_leaves_cache_uncommitted(&fixture)
        .expect("a KV write with no commit call stays pending, never promoted");
}

#[test]
fn e2e_kv_provider_failure_stores_no_pending_state() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_kv_provider_failure_stores_no_pending_state(&fixture)
        .expect("a generation step that fails Provider dispatch stores no pending KV state");
}

#[test]
fn e2e_kv_cancelled_decode_does_not_corrupt_committed_cache() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_kv_cancelled_decode_does_not_corrupt_committed_cache(&fixture)
        .expect("a cancelled decode's pending KV write does not alter the committed cache");
}

#[test]
fn e2e_kv_pending_write_is_memory_admitted_for_its_concatenated_size() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_kv_pending_write_is_memory_admitted_for_its_concatenated_size(&fixture)
        .expect("decode's concatenated pending KV write is memory-admitted at its real size");
}

#[test]
fn e2e_kv_pending_write_allocation_is_released_on_discard() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_kv_pending_write_allocation_is_released_on_discard(&fixture)
        .expect("discarding a pending KV state releases its admitted allocations");
}

#[test]
fn e2e_kv_partial_layer_failure_during_commit_rolls_back_cleanly() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_kv_partial_layer_failure_during_commit_rolls_back_cleanly(&fixture).expect(
        "a partial-layer commit failure rolls back cleanly, without a mixed-generation cache",
    );
}

#[test]
fn e2e_kv_double_commit_second_call_is_rejected() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_kv_double_commit_second_call_is_rejected(&fixture)
        .expect("a second commit for an already-committed generation step is rejected");
}

#[test]
fn e2e_kv_double_abort_is_idempotent() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_kv_double_abort_is_idempotent(&fixture)
        .expect("discarding a pending KV state twice in a row is idempotent");
}

#[test]
fn e2e_kv_stale_pending_state_does_not_survive_a_failed_retry() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_kv_stale_pending_state_does_not_survive_a_failed_retry(&fixture)
        .expect("a stale pending KV write is discarded before a failed retry, not left to be wrongly committed");
}

#[test]
fn e2e_kv_wrong_session_reuse_is_rejected_by_compatibility() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_kv_wrong_session_reuse_is_rejected_by_compatibility(&fixture)
        .expect("reusing one session's committed KV cache under another session's compatibility is rejected");
}

#[test]
fn e2e_generation_step_rechecks_model_instance_readiness() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_generation_step_rechecks_model_instance_readiness(&fixture)
        .expect("a generation step re-checks model instance readiness for itself");
}

#[test]
fn e2e_generation_observations_never_carry_raw_prompt_or_handles() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_generation_observations_never_carry_raw_prompt_or_handles(&fixture)
        .expect("generation observations never carry raw prompt text or native handles");
}

#[test]
fn e2e_chat_session_close_releases_kv_cache_and_model_instance() {
    check_chat_session_close_releases_kv_cache_and_model_instance()
        .expect("closing a chat session releases its KV cache and model instance");
}

#[test]
fn e2e_chat_sessions_are_isolated_from_each_other() {
    check_chat_sessions_are_isolated_from_each_other()
        .expect("two chat sessions for the same model are isolated from each other");
}

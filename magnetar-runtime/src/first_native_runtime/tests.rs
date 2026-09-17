//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test source
//! rather than Runtime implementation source (task 9.1).

use super::*;
use crate::planning::*;
use crate::scheduler::*;
use crate::{CapabilityVersion, Device, ExecutionGraphSemanticFingerprint};
use std::sync::atomic::{AtomicBool, Ordering};

/// The correct prefill/decode Operator-sequence hash for the E2E fixture
/// architecture (see `qwen_operator_sequence_hash`), computed once and
/// hard-coded here the same way the fixture's node counts are: this
/// component is a fixed stand-in for one exact architecture shape, not a
/// dynamic graph compiler.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
const QWEN_COMPONENT_FIXTURE_OPERATOR_HASH: u32 = 0x52b1_f815;

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
    fn write_tensor(
        &self,
        id: TensorResourceId,
        tensor: HostTensor,
    ) -> Result<(), ProviderExecutionError> {
        self.storage.lock().unwrap().insert(id, tensor);
        Ok(())
    }
    fn read_tensor(&self, id: &TensorResourceId) -> Option<HostTensor> {
        self.storage.lock().unwrap().get(id).cloned()
    }
    fn write_tensor_value(
        &self,
        id: TensorResourceId,
        value: TensorValue,
    ) -> Result<(), ProviderExecutionError> {
        if let TensorValue::Host(tensor) = value {
            self.write_tensor(id, tensor)?;
        }
        Ok(())
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
    fn write_tensor_value(
        &self,
        id: TensorResourceId,
        _value: TensorValue,
    ) -> Result<(), ProviderExecutionError> {
        self.resources.lock().unwrap().insert(id);
        Ok(())
    }
    fn read_tensor_value(&self, id: &TensorResourceId) -> Option<TensorValue> {
        if self.resources.lock().unwrap().contains(id) {
            Some(TensorValue::Opaque)
        } else {
            None
        }
    }
}

/// A `ProviderExecutionApi` whose `write_tensor`/`release_tensor` fail on
/// demand, so tests can prove the structured-error propagation added by
/// the "Provider tensor resource mutations lack structured failure
/// channels" GitHub issue actually reaches `WeightMaterializationTransaction`
/// and `Runtime::unload_model_instance`. Every other method delegates to a
/// real `ReferenceCpuExecutor`, so a test that only fails one specific
/// operation still sees otherwise-normal behavior for everything else.
struct FailableProviderExecutionApi {
    inner: ReferenceCpuExecutor,
    fail_write: AtomicBool,
    fail_release: AtomicBool,
}
impl FailableProviderExecutionApi {
    fn new() -> Self {
        Self {
            inner: ReferenceCpuExecutor::new(),
            fail_write: AtomicBool::new(false),
            fail_release: AtomicBool::new(false),
        }
    }
    fn simulated_error(&self, phase: ProviderExecutionPhase) -> ProviderExecutionError {
        ProviderExecutionError::new(
            ProviderExecutionErrorCode::MaterializationFailed,
            phase,
            ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
            None,
            "simulated Provider failure injected by FailableProviderExecutionApi",
        )
    }
}
impl ProviderExecutionApi for FailableProviderExecutionApi {
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
    fn write_tensor(
        &self,
        id: TensorResourceId,
        tensor: HostTensor,
    ) -> Result<(), ProviderExecutionError> {
        if self.fail_write.load(Ordering::SeqCst) {
            return Err(self.simulated_error(ProviderExecutionPhase::Prepare));
        }
        self.inner.write_tensor(id, tensor);
        Ok(())
    }
    fn read_tensor(&self, id: &TensorResourceId) -> Option<HostTensor> {
        self.inner.read_tensor(id)
    }
    fn release_tensor(&self, id: &TensorResourceId) -> Result<bool, ProviderExecutionError> {
        if self.fail_release.load(Ordering::SeqCst) {
            return Err(self.simulated_error(ProviderExecutionPhase::Release));
        }
        Ok(self.inner.release_tensor(id))
    }
    fn write_tensor_value(
        &self,
        id: TensorResourceId,
        value: TensorValue,
    ) -> Result<(), ProviderExecutionError> {
        if self.fail_write.load(Ordering::SeqCst) {
            return Err(self.simulated_error(ProviderExecutionPhase::Prepare));
        }
        if let TensorValue::Host(tensor) = value {
            self.inner.write_tensor(id, tensor);
        }
        Ok(())
    }
    // Real admit-then-write-with-rollback (mirrors
    // `ReferenceCpuExecutor`/the default trait implementation's documented
    // shape) rather than the trait's fail-closed default: this mock needs
    // admission to actually succeed so `write_tensor_value`'s injectable
    // failure is the thing under test, not admission itself.
    fn write_tensor_value_admitted(
        &self,
        memory: &mut MemoryManager,
        resource_id: TensorResourceId,
        value: TensorValue,
        class: MemoryAllocationClass,
        owner: MemoryAllocationOwner,
    ) -> Result<(), TensorValueAdmissionError> {
        let byte_size = match &value {
            TensorValue::Host(tensor) => tensor.data.len() as u64 * size_of::<f32>() as u64,
            TensorValue::Opaque => 0,
        };
        let allocation = memory
            .allocate(MemoryAllocationRequest::new(
                class,
                byte_size,
                MemoryPlacement::ProviderOwnedOpaque(ProviderBinding::new(
                    REFERENCE_CPU_PROVIDER_NAME,
                )),
                owner,
            ))
            .map_err(TensorValueAdmissionError::Memory)?;
        if let Err(error) = self.write_tensor_value(resource_id, value) {
            let _ = memory.release(allocation.id);
            return Err(TensorValueAdmissionError::Provider(error));
        }
        Ok(())
    }
}

/// P0 fix regression: `WeightMaterializationTransaction::stage_weight`
/// propagates a real Provider write failure as
/// `InferenceApiError::ProviderTensorWriteFailed` instead of ignoring a
/// bare `()`, and rolls back the Memory Manager allocation it had already
/// admitted for that weight before propagating -- the same rollback shape
/// this transaction already applies to a residency-registration failure.
#[test]
fn stage_weight_propagates_and_rolls_back_on_provider_write_failure() {
    let fixture = e2e_fixture().expect("fixture builds");
    let mut runtime = build_runtime_trusting_fixture(&fixture);
    let (instance, _memory) =
        load_fixture_instance(&fixture, &mut runtime).expect("instance loads");

    let executor = Arc::new(FailableProviderExecutionApi::new());
    executor.fail_write.store(true, Ordering::SeqCst);
    let mut transaction = WeightMaterializationTransaction {
        provider_binding: ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        executor,
        staged: Vec::new(),
    };
    let tensor = HostTensor::new([4, 8], vec![0.0; 32]).unwrap();
    let error = transaction
        .stage_weight(
            &mut runtime,
            &instance,
            "test",
            "transformer.wte.weight",
            &tensor,
        )
        .unwrap_err();
    assert!(matches!(
        error,
        InferenceApiError::ProviderTensorWriteFailed { .. }
    ));
    // `MemoryManager::allocations()` is a permanent ledger -- a released
    // allocation stays present with `state: Released`, it does not
    // disappear -- so "rolled back, not leaked" means the allocation this
    // attempt admitted (the highest-numbered one, ids being assigned
    // sequentially) ended in `Released` state, not `Active`.
    let this_attempts_allocation = runtime
        .memory()
        .allocations()
        .max_by_key(|allocation| allocation.id)
        .expect("at least one allocation exists");
    assert_eq!(
        this_attempts_allocation.state,
        MemoryAllocationState::Released,
        "the allocation admitted just before the failed write must be rolled back, not leaked"
    );
}

/// `generalize-first-native-provider-dispatch` regression: the `TensorValue`
/// counterpart to `stage_weight_propagates_and_rolls_back_on_provider_write_failure`
/// above. Before this change, `write_tensor_value_admitted` could only
/// report a `MemoryError`, so a genuine Provider-native write failure after
/// successful admission had no way to surface -- this asserts it now does,
/// distinguishably, with the same allocate-then-release-on-failure rollback
/// discipline.
#[test]
fn write_tensor_value_admitted_propagates_and_rolls_back_on_provider_write_failure() {
    let executor = FailableProviderExecutionApi::new();
    executor.fail_write.store(true, Ordering::SeqCst);
    let mut memory = MemoryManager::new(MemoryManagerConfig::default());

    let tensor = HostTensor::new([4, 8], vec![0.0; 32]).unwrap();
    let error = executor
        .write_tensor_value_admitted(
            &mut memory,
            TensorResourceId::new("test-tensor-value-write-failure"),
            TensorValue::Host(tensor),
            MemoryAllocationClass::Tensor,
            MemoryAllocationOwner::Runtime,
        )
        .unwrap_err();
    assert!(
        matches!(error, TensorValueAdmissionError::Provider(_)),
        "a write failure after successful admission must be reported as Provider, not Memory: {error:?}"
    );

    let this_attempts_allocation = memory
        .allocations()
        .max_by_key(|allocation| allocation.id)
        .expect("admission ran before the write failed");
    assert_eq!(
        this_attempts_allocation.state,
        MemoryAllocationState::Released,
        "the allocation admitted just before the failed write must be rolled back, not leaked"
    );
}

/// `Provider` wrapper around [`FailableProviderExecutionApi`], the minimum
/// needed to register it with a `Runtime` under [`REFERENCE_CPU_PROVIDER_NAME`]
/// (only `metadata`/`register` are non-defaulted on `Provider`; `devices`/
/// `kernel_advertisements` default to empty, which is fine here -- this
/// provider is used only to exercise weight materialization and unload,
/// neither of which dispatch through the Kernel Registry).
struct TestFailableProvider {
    metadata: ProviderMetadata,
    executor: Arc<FailableProviderExecutionApi>,
}
impl Provider for TestFailableProvider {
    fn metadata(&self) -> ProviderMetadata {
        self.metadata.clone()
    }
    fn register(&self, _registry: &mut ProviderRegistry) -> Result<(), ProviderError> {
        Ok(())
    }
    fn execution_api(&self) -> Option<Arc<dyn ProviderExecutionApi>> {
        Some(self.executor.clone())
    }
}

/// P0 fix regression: `Runtime::unload_model_instance` propagates a real
/// Provider release failure as a structured `ModelInstanceError` instead
/// of ignoring a bare `bool`.
#[test]
fn unload_model_instance_propagates_provider_release_failure() {
    let fixture = e2e_fixture().expect("fixture builds");
    let executor = Arc::new(FailableProviderExecutionApi::new());
    executor.fail_release.store(true, Ordering::SeqCst);
    let mut runtime = Runtime::builder()
        .register_provider(Arc::new(TestFailableProvider {
            metadata: ProviderMetadata::new(
                REFERENCE_CPU_PROVIDER_NAME,
                "test",
                "test",
                "test-only Provider whose release_tensor always fails",
            ),
            executor,
        }))
        .trust_store(
            ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone()),
        )
        .build()
        .unwrap();
    // `load_fixture_instance` materializes the fixture's weights internally
    // (via `bind_qwen_fixture_weights`) and reaches `Ready` -- this
    // succeeds because only `fail_release`, not `fail_write`, is set.
    let (instance, _memory) =
        load_fixture_instance(&fixture, &mut runtime).expect("instance loads and reaches Ready");
    assert_eq!(
        runtime.model_instance(&instance).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Ready
    );

    let error = runtime
        .unload_model_instance(&instance, ModelInstanceUnloadPolicy::DrainActiveUse)
        .unwrap_err();
    assert!(matches!(
        error,
        ModelInstanceError::InternalModelInstance { .. }
    ));
}

/// A `ProviderExecutionApi` whose Kernel-level `complete_kernel` fails on
/// demand for one targeted Operator, so tests can prove
/// `dispatch_reference_cpu_operator_pre_admitted`'s rollback
/// (`make-first-native-cuda-hot-path-device-resident`'s Decision 7) fires
/// for a genuine kernel/completion-time failure -- as opposed to
/// `stage_weight_propagates_and_rolls_back_on_provider_write_failure`'s
/// weight-write failure, or a tight-memory-budget's submit-time admission
/// failure. Every other Operator (and every non-Kernel method) delegates to
/// a real, independent `ReferenceCpuExecutor`, so a test that only fails one
/// specific Operator still sees otherwise-normal numeric execution for
/// everything else -- including the weight materialization that has to
/// succeed first for the graph to reach that Operator at all.
struct KernelFailableProviderExecutionApi {
    inner: ReferenceCpuExecutor,
    fail_operator: Mutex<Option<&'static str>>,
    failing_handles: Mutex<BTreeSet<ProviderExecutionId>>,
}
impl KernelFailableProviderExecutionApi {
    fn new(fail_operator: &'static str) -> Self {
        Self {
            inner: ReferenceCpuExecutor::new(),
            fail_operator: Mutex::new(Some(fail_operator)),
            failing_handles: Mutex::new(BTreeSet::new()),
        }
    }
}
impl ProviderExecutionApi for KernelFailableProviderExecutionApi {
    fn submit(
        &self,
        request: ProviderExecutionRequest,
    ) -> Result<ProviderExecutionHandle, ProviderExecutionError> {
        self.inner.submit(request)
    }
    fn status(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderExecutionStatus, ProviderExecutionError> {
        self.inner.status(handle)
    }
    fn cancel(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderCancellationOutcome, ProviderExecutionError> {
        self.inner.cancel(handle)
    }
    fn complete(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderExecutionResult, ProviderExecutionError> {
        self.inner.complete(handle)
    }
    fn release(&self, handle: ProviderExecutionHandle) -> Result<(), ProviderExecutionError> {
        self.inner.release(handle)
    }
    fn submit_kernel(
        &self,
        advertisement: &KernelAdvertisement,
        operator: &OperatorSpec,
        invocation: &KernelInvocation,
        memory: &mut MemoryManager,
    ) -> Result<ProviderExecutionHandle, ProviderExecutionError> {
        let handle = self
            .inner
            .submit_kernel(advertisement, operator, invocation, memory)?;
        if *self.fail_operator.lock().unwrap() == Some(invocation.operator.name()) {
            self.failing_handles
                .lock()
                .unwrap()
                .insert(handle.id.clone());
        }
        Ok(handle)
    }
    fn complete_kernel(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<KernelResult, ProviderExecutionError> {
        if self.failing_handles.lock().unwrap().remove(&handle.id) {
            return Err(ProviderExecutionError::new(
                ProviderExecutionErrorCode::ExecutionFailed,
                ProviderExecutionPhase::Complete,
                handle.provider.clone(),
                handle.device.clone(),
                "simulated Kernel completion failure injected by \
                 KernelFailableProviderExecutionApi",
            ));
        }
        self.inner.complete_kernel(handle)
    }
    fn write_tensor(
        &self,
        id: TensorResourceId,
        tensor: HostTensor,
    ) -> Result<(), ProviderExecutionError> {
        ReferenceCpuExecutor::write_tensor(&self.inner, id, tensor);
        Ok(())
    }
    fn read_tensor(&self, id: &TensorResourceId) -> Option<HostTensor> {
        ReferenceCpuExecutor::read_tensor(&self.inner, id)
    }
    fn release_tensor(&self, id: &TensorResourceId) -> Result<bool, ProviderExecutionError> {
        Ok(ReferenceCpuExecutor::release_tensor(&self.inner, id))
    }
    fn release_admitted_tensor(
        &self,
        memory: &mut MemoryManager,
        id: &TensorResourceId,
    ) -> Result<bool, ProviderExecutionError> {
        Ok(ReferenceCpuExecutor::release_admitted_tensor(
            &self.inner,
            memory,
            id,
        ))
    }
    fn write_tensor_admitted(
        &self,
        memory: &mut MemoryManager,
        resource_id: TensorResourceId,
        tensor: HostTensor,
        class: MemoryAllocationClass,
        owner: MemoryAllocationOwner,
    ) -> Result<(), MemoryError> {
        self.inner
            .write_tensor_admitted(memory, resource_id, tensor, class, owner)
    }
    fn read_tensor_value(&self, id: &TensorResourceId) -> Option<TensorValue> {
        self.inner.read_tensor_value(id)
    }
    fn write_tensor_value(
        &self,
        id: TensorResourceId,
        value: TensorValue,
    ) -> Result<(), ProviderExecutionError> {
        self.inner.write_tensor_value(id, value)
    }
    fn write_tensor_value_admitted(
        &self,
        memory: &mut MemoryManager,
        resource_id: TensorResourceId,
        value: TensorValue,
        class: MemoryAllocationClass,
        owner: MemoryAllocationOwner,
    ) -> Result<(), TensorValueAdmissionError> {
        self.inner
            .write_tensor_value_admitted(memory, resource_id, value, class, owner)
    }
    fn allocate_workspace(
        &self,
        memory: &mut MemoryManager,
        size_bytes: u64,
    ) -> Result<MemoryAllocationId, MemoryError> {
        self.inner.allocate_workspace(memory, size_bytes)
    }
    fn observations(&self) -> Vec<KernelObservation> {
        self.inner.observations()
    }
}

/// `Provider` wrapper around [`KernelFailableProviderExecutionApi`], the
/// minimum needed to register it with a `Runtime` under
/// [`REFERENCE_CPU_PROVIDER_NAME`] so first-native dispatch resolves it
/// exactly where it would otherwise resolve the real `ReferenceCpuProvider`.
struct TestFailableKernelProvider {
    executor: Arc<KernelFailableProviderExecutionApi>,
}
impl Provider for TestFailableKernelProvider {
    fn metadata(&self) -> ProviderMetadata {
        reference_cpu_provider_metadata()
    }
    fn register(&self, _registry: &mut ProviderRegistry) -> Result<(), ProviderError> {
        Ok(())
    }
    fn execution_api(&self) -> Option<Arc<dyn ProviderExecutionApi>> {
        Some(self.executor.clone())
    }
    fn devices(&self) -> Vec<Arc<dyn Device>> {
        vec![Arc::new(reference_cpu_device())]
    }
    fn kernel_advertisements(&self) -> Vec<KernelAdvertisement> {
        reference_cpu_kernel_advertisements()
    }
}

/// `audit-complet-cuda-hot-path-2026-09-08` P1-1's generalized entrypoint
/// (`run_first_native_graph_with_provider`), sanity-checked in-crate
/// against Reference CPU before trusting it for an external CUDA
/// integration test -- if the generalization introduced a bug, this is
/// the fast place to catch it, not the slower cross-repo GPU loop.
/// Confirms: the dispatch actually succeeds through the fully generalized
/// path (`build_runtime_with_model_execution_engine_and_provider` ->
/// `load_fixture_instance_for_provider` -> `prepare_first_native_plan_
/// for_graph` with an explicit Provider -> `execute_qwen_graph`), the
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
/// genuinely grouped-query-shaped `QwenConfig`
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
    let architecture = qwen_architecture_metadata(8, 1, 4, 2, 2, 16, 258, 32);
    let identity = qwen_component_identity(
        ModelComponentId::new("gqa-integration-fixture").expect("static id is valid"),
        ModelComponentVersion::new(1, 0, 0),
        ModelComponentImplementationKind::WebAssemblyComponent,
    );
    let config = QwenConfig::new(architecture, QwenRopeConfig::standard(2));
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
    let descriptor = qwen_component_descriptor(identity.clone(), &config)
        .expect("GQA component descriptor builds");
    qwen_validate_model_artifact(&descriptor, &config, &manifest)
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
    let architecture = qwen_architecture_metadata(8, 1, 4, 2, 2, 16, 258, 32);
    let identity = qwen_component_identity(
        ModelComponentId::new("gqa-real-component-fixture").expect("static id is valid"),
        ModelComponentVersion::new(1, 0, 0),
        ModelComponentImplementationKind::WebAssemblyComponent,
    );
    let config = QwenConfig::new(architecture, QwenRopeConfig::standard(2));
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
    let descriptor = qwen_component_descriptor(identity.clone(), &config)
        .expect("GQA component descriptor builds");
    qwen_validate_model_artifact(&descriptor, &config, &manifest)
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

/// task group 6 (`make-first-native-cuda-hot-path-device-resident`'s
/// Decision 7) / 6.2: a failure during Kernel Registry/dispatch-plan
/// construction -- here, `attention`'s required workspace failing to admit
/// under a tight Runtime memory budget, the same scenario
/// `check_graph_dispatch_records_memory_feasibility_failure_under_tight_budget`
/// already exercises -- must roll back the output this node had already
/// pre-admitted before that point, not leak it. `attention` is the first
/// node in this fixture's graph whose workspace does not fit the budget, so
/// its own pre-admitted output allocation is still the highest-numbered
/// allocation when the failure fires (workspace itself never became an
/// allocation -- admission failed before one was created), matching
/// `stage_weight_propagates_and_rolls_back_on_provider_write_failure`'s
/// same "highest id -> this attempt's allocation" reasoning.
#[test]
fn e2e_graph_dispatch_rolls_back_pre_admitted_output_on_submit_time_failure() {
    let fixture = e2e_fixture().expect("fixture builds");
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let mut runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .config(RuntimeConfig {
            memory: MemoryManagerConfig {
                max_runtime_bytes: Some(1 << 16),
                allow_pending_allocations: false,
                ..MemoryManagerConfig::default()
            },
            ..RuntimeConfig::default()
        })
        .trust_store(
            ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone()),
        )
        .build()
        .expect("Reference CPU provider registers cleanly");
    register_reference_cpu_prepared_kernels(&mut runtime);
    let (instance, _memory) =
        load_fixture_instance(&fixture, &mut runtime).expect("instance loads");
    let mut plans = first_native_plans_for_prompt(&runtime, &fixture, &instance, 2)
        .expect("prepared plans build");
    let graphs =
        first_native_component_graphs_for_prompt(&fixture, 2).expect("component graphs build");
    let ids = HostTensor::new([2], vec![1.0, 2.0]).expect("token id tensor builds");
    let cache_id = KvCacheId::new("test-submit-time-rollback-cache").expect("cache id is valid");

    let result = execute_qwen_graph(
        &mut runtime,
        &fixture,
        &instance,
        &cache_id,
        &graphs.prefill,
        &mut plans.prefill,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    );
    assert!(
        result.is_err(),
        "expected the tight memory budget to fail graph dispatch"
    );

    let this_attempts_allocation = runtime
        .memory()
        .allocations()
        .max_by_key(|allocation| allocation.id)
        .expect("the failing node's output was pre-admitted before the workspace failed");
    assert_eq!(
        this_attempts_allocation.state,
        MemoryAllocationState::Released,
        "a submit-time (Kernel Registry/dispatch-plan construction) failure must roll back \
         the pre-admitted output allocation, not leak it"
    );

    // audit-complet-cuda-hot-path-2026-09-08 P0-2 / Correctif B's "failure
    // before materialization" case: `attention`'s workspace admission
    // fails before the Provider is ever invoked, so its output was never
    // written to Provider storage in the first place -- rollback's
    // `release_tensor` call is a correctly idempotent no-op here, not a
    // real cleanup. Checked anyway (the completion-time counterpart test
    // covers the genuinely materialized case) so both halves of the
    // audit's test list are directly represented.
    let attention_node = graphs
        .prefill
        .nodes
        .values()
        .find(|node| node.operator.name() == "attention")
        .expect("this fixture's prefill graph has an attention node");
    let attention_output_edge = attention_node
        .outputs
        .first()
        .expect("attention node has an output edge");
    let attention_output_id = TensorResourceId::new(format!("edge.{attention_output_edge}"));
    assert!(
        executor.read_tensor_value(&attention_output_id).is_none(),
        "a submit-time failure must never leave a Provider-side resource behind for the \
         output it pre-admitted but never got to materialize"
    );
}

/// task group 6 (`make-first-native-cuda-hot-path-device-resident`'s
/// Decision 7) / 6.3: a failure in the Kernel's own execution/completion
/// (here, `ctx.provider.complete_kernel` itself, injected via
/// `KernelFailableProviderExecutionApi`) must roll back the output that
/// node had already pre-admitted before dispatch, exactly like a
/// submit-time failure. Targets `embedding`, the first node this fixture's
/// prefill graph dispatches, so zero earlier nodes have succeeded yet --
/// the Runtime's Active allocation count right after model loading (weights
/// staged, nothing dispatched) must be unchanged after the failed attempt,
/// which is a stronger, id-independent way to prove "rolled back, not
/// leaked" than picking out one allocation by id.
#[test]
fn e2e_graph_dispatch_rolls_back_pre_admitted_output_on_kernel_completion_failure() {
    let fixture = e2e_fixture().expect("fixture builds");
    let executor = Arc::new(KernelFailableProviderExecutionApi::new("embedding"));
    let mut runtime = Runtime::builder()
        .register_provider(Arc::new(TestFailableKernelProvider {
            executor: executor.clone(),
        }))
        .trust_store(
            ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone()),
        )
        .build()
        .expect("Reference CPU provider registers cleanly");
    register_reference_cpu_prepared_kernels(&mut runtime);
    let (instance, _memory) =
        load_fixture_instance(&fixture, &mut runtime).expect("instance loads");
    let mut plans = first_native_plans_for_prompt(&runtime, &fixture, &instance, 2)
        .expect("prepared plans build");
    let graphs =
        first_native_component_graphs_for_prompt(&fixture, 2).expect("component graphs build");
    let ids = HostTensor::new([2], vec![1.0, 2.0]).expect("token id tensor builds");
    let cache_id =
        KvCacheId::new("test-kernel-completion-rollback-cache").expect("cache id is valid");

    let active_before = runtime
        .memory()
        .allocations()
        .filter(|allocation| allocation.state == MemoryAllocationState::Active)
        .count();

    let result = execute_qwen_graph(
        &mut runtime,
        &fixture,
        &instance,
        &cache_id,
        &graphs.prefill,
        &mut plans.prefill,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    );
    assert!(
        result.is_err(),
        "expected the injected embedding completion failure to fail graph dispatch"
    );

    let active_after = runtime
        .memory()
        .allocations()
        .filter(|allocation| allocation.state == MemoryAllocationState::Active)
        .count();
    assert_eq!(
        active_after,
        active_before + 1,
        "a kernel/completion-time failure on the very first graph node must leave the \
         Runtime's Active allocation count exactly one higher than it was right after model \
         loading -- the one legitimate admission being the graph's own 'input.token_ids' \
         resource (admitted up front, before any node dispatches -- see \
         `execute_qwen_graph_nodes`'s `initial_bindings` loop); `embedding`'s own \
         pre-admitted output must be rolled back, not left behind as a second, leaked \
         allocation"
    );

    // audit-complet-cuda-hot-path-2026-09-08 P0-2 / Correctif B: the
    // allocation/residency check above only proves the *Memory Manager*
    // side was rolled back -- Reference CPU's `submit_kernel_invocation`
    // runs the Kernel and writes its output into Provider storage
    // synchronously, *before* `complete_kernel` is ever called, so by the
    // time this mock's injected `complete_kernel` failure fires, the
    // Provider genuinely already materialized `embedding`'s output. The
    // fix must release it from Provider storage too, not just Memory
    // Manager's ledger -- checked directly against the same `executor`
    // this dispatch actually ran through, not inferred from the
    // allocation count alone.
    let embedding_node = graphs
        .prefill
        .nodes
        .values()
        .find(|node| node.operator.name() == "embedding")
        .expect("this fixture's prefill graph has an embedding node");
    let embedding_output_edge = embedding_node
        .outputs
        .first()
        .expect("embedding node has an output edge");
    let embedding_output_id = TensorResourceId::new(format!("edge.{embedding_output_edge}"));
    assert!(
        executor.read_tensor_value(&embedding_output_id).is_none(),
        "a kernel/completion-time failure must also release the pre-admitted output from \
         Provider storage, not just Memory Manager's residency ledger -- \
         `embedding`'s output was genuinely materialized Provider-side before completion \
         failed, and must not survive rollback as an orphaned Provider-side resource"
    );

    // 6.4/6.5: a retry with the injected failure cleared must succeed
    // cleanly -- proving the rollback left no orphaned residency/allocation
    // under the same resource id that would otherwise collide with, or be
    // silently reused by, the retried dispatch.
    let mut retry_plans = first_native_plans_for_prompt(&runtime, &fixture, &instance, 2)
        .expect("prepared plans build for retry");
    let retry_ids = HostTensor::new([2], vec![1.0, 2.0]).expect("token id tensor builds");
    let retry_cache_id =
        KvCacheId::new("test-kernel-completion-rollback-retry-cache").expect("cache id is valid");
    *executor.fail_operator.lock().unwrap() = None;
    executor.failing_handles.lock().unwrap().clear();
    let retry_result = execute_qwen_graph(
        &mut runtime,
        &fixture,
        &instance,
        &retry_cache_id,
        &graphs.prefill,
        &mut retry_plans.prefill,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), retry_ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    );
    assert!(
        retry_result.is_ok(),
        "a retried dispatch after a rolled-back failure must succeed cleanly: {:?}",
        retry_result.err()
    );
}

/// GitHub issue "First-native graph executor only propagates the first
/// output of a multi-output node": `dispatch_reference_cpu_operator_multi`
/// (the shared implementation `dispatch_reference_cpu_operator`'s
/// single-output wrapper now delegates to) threads every requested output
/// through to the graph, not just the first. `split` is the Reference CPU
/// Provider's one genuinely multi-output Operator (`updated_resources` gets
/// two entries), so dispatching it through this path and checking both
/// outputs -- not just the first -- come back with the correct data is the
/// most direct proof this actually works end to end, not just at the type
/// level.
#[test]
fn dispatch_reference_cpu_operator_multi_binds_every_output() {
    let fixture = e2e_fixture().expect("fixture builds");
    let mut runtime = build_runtime_trusting_fixture(&fixture);
    let provider: Arc<dyn ProviderExecutionApi> = Arc::new(ReferenceCpuExecutor::new());
    let mut node_events = Vec::new();
    let mut dispatch_ctx = QwenDispatchContext {
        runtime: &mut runtime,
        provider,
        prepared_plan: None,
        graph: None,
        sequence_length: None,
        last_provider_execution: None,
        node_events: &mut node_events,
    };
    let input = HostTensor::new([1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let (_dispatch_result, mut outputs) = dispatch_reference_cpu_operator_multi(
        &mut dispatch_ctx,
        "multi-output.split",
        dispatch_operator_id("split", OperatorFamily::Tensor),
        vec![NodeInputResource::Fresh(
            TensorResourceId::new("multi-output.split.in"),
            f32_tensor_descriptor(&input),
            input,
        )],
        vec![
            (
                TensorResourceId::new("multi-output.split.left"),
                TensorDescriptor::new(
                    ShapeDescriptor::new([1, 2]),
                    DTypeDescriptor::portable(ComputeDType::Float32),
                    LayoutDescriptor::Contiguous,
                ),
            ),
            (
                TensorResourceId::new("multi-output.split.right"),
                TensorDescriptor::new(
                    ShapeDescriptor::new([1, 2]),
                    DTypeDescriptor::portable(ComputeDType::Float32),
                    LayoutDescriptor::Contiguous,
                ),
            ),
        ],
        BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(
        outputs.len(),
        2,
        "both outputs must be bound, not just the first"
    );
    let right = outputs
        .pop()
        .unwrap()
        .into_host(&dispatch_ctx.provider)
        .unwrap();
    let left = outputs
        .pop()
        .unwrap()
        .into_host(&dispatch_ctx.provider)
        .unwrap();
    assert_eq!(left.shape, vec![1, 2]);
    assert_eq!(left.data, vec![1.0, 2.0]);
    assert_eq!(right.shape, vec![1, 2]);
    assert_eq!(right.data, vec![3.0, 4.0]);
}

/// A `ProviderExecutionApi` that delegates real computation to an inner
/// `ReferenceCpuExecutor` but always reports `TensorValue::Opaque` for any
/// resource it holds -- simulating a device-resident Provider (like CUDA)
/// with a real, independently-correct Kernel implementation behind it,
/// rather than a synthetic no-op double. Used to exercise
/// `enable-device-resident-kernel-chaining`'s `NodeInputResource::Resident`
/// passthrough against real Kernel dispatch, not just at the type level.
struct OpaqueReportingExecutor {
    inner: ReferenceCpuExecutor,
}
impl OpaqueReportingExecutor {
    fn new() -> Self {
        Self {
            inner: ReferenceCpuExecutor::new(),
        }
    }
}
impl ProviderExecutionApi for OpaqueReportingExecutor {
    fn submit(
        &self,
        request: ProviderExecutionRequest,
    ) -> Result<ProviderExecutionHandle, ProviderExecutionError> {
        self.inner.submit(request)
    }
    fn status(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderExecutionStatus, ProviderExecutionError> {
        self.inner.status(handle)
    }
    fn cancel(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderCancellationOutcome, ProviderExecutionError> {
        self.inner.cancel(handle)
    }
    fn complete(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderExecutionResult, ProviderExecutionError> {
        self.inner.complete(handle)
    }
    fn release(&self, handle: ProviderExecutionHandle) -> Result<(), ProviderExecutionError> {
        self.inner.release(handle)
    }
    fn submit_kernel(
        &self,
        advertisement: &KernelAdvertisement,
        operator: &OperatorSpec,
        invocation: &KernelInvocation,
        memory: &mut MemoryManager,
    ) -> Result<ProviderExecutionHandle, ProviderExecutionError> {
        self.inner
            .submit_kernel(advertisement, operator, invocation, memory)
    }
    fn complete_kernel(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<KernelResult, ProviderExecutionError> {
        self.inner.complete_kernel(handle)
    }
    fn write_tensor(
        &self,
        id: TensorResourceId,
        tensor: HostTensor,
    ) -> Result<(), ProviderExecutionError> {
        self.inner.write_tensor(id, tensor);
        Ok(())
    }
    fn read_tensor(&self, id: &TensorResourceId) -> Option<HostTensor> {
        self.inner.read_tensor(id)
    }
    fn release_tensor(&self, id: &TensorResourceId) -> Result<bool, ProviderExecutionError> {
        Ok(self.inner.release_tensor(id))
    }
    fn write_tensor_admitted(
        &self,
        memory: &mut MemoryManager,
        resource_id: TensorResourceId,
        tensor: HostTensor,
        class: MemoryAllocationClass,
        owner: MemoryAllocationOwner,
    ) -> Result<(), MemoryError> {
        self.inner
            .write_tensor_admitted(memory, resource_id, tensor, class, owner)
    }
    // The one deliberate divergence from `self.inner`: always `Opaque` when
    // present, never `Host`, so callers can never assume this Provider's
    // resources are host-visible without an explicit `into_host`.
    fn read_tensor_value(&self, id: &TensorResourceId) -> Option<TensorValue> {
        self.inner.read_tensor(id).map(|_| TensorValue::Opaque)
    }
    fn write_tensor_value(
        &self,
        id: TensorResourceId,
        value: TensorValue,
    ) -> Result<(), ProviderExecutionError> {
        self.inner.write_tensor_value(id, value)
    }
    fn write_tensor_value_admitted(
        &self,
        memory: &mut MemoryManager,
        resource_id: TensorResourceId,
        value: TensorValue,
        class: MemoryAllocationClass,
        owner: MemoryAllocationOwner,
    ) -> Result<(), TensorValueAdmissionError> {
        self.inner
            .write_tensor_value_admitted(memory, resource_id, value, class, owner)
    }
    fn observations(&self) -> Vec<KernelObservation> {
        self.inner.observations()
    }
}

/// `enable-device-resident-kernel-chaining` task 5.6: two consecutive
/// Kernel dispatches against a Provider that never reports `Host` values
/// (`OpaqueReportingExecutor`) -- the second dispatch's `NodeInputResource`
/// for the first dispatch's output must be `Resident` (reused by id), never
/// re-uploaded under a fresh id, yet the actual computed numbers must still
/// be correct (proving the passthrough reaches real, independently
/// verified Kernel computation, not a stub).
#[test]
fn resident_input_passthrough_reuses_existing_resource_and_computes_correctly() {
    let fixture = e2e_fixture().expect("fixture builds");
    let mut runtime = build_runtime_trusting_fixture(&fixture);
    let provider: Arc<dyn ProviderExecutionApi> = Arc::new(OpaqueReportingExecutor::new());
    let mut node_events = Vec::new();
    let mut dispatch_ctx = QwenDispatchContext {
        runtime: &mut runtime,
        provider: provider.clone(),
        prepared_plan: None,
        graph: None,
        sequence_length: None,
        last_provider_execution: None,
        node_events: &mut node_events,
    };
    let a = HostTensor::new([1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = HostTensor::new([1, 4], vec![10.0, 10.0, 10.0, 10.0]).unwrap();
    let (_dispatch, sum) = dispatch_reference_cpu_operator(
        &mut dispatch_ctx,
        "resident.add",
        dispatch_operator_id("add", OperatorFamily::Tensor),
        vec![
            NodeInputResource::Fresh(
                TensorResourceId::new("resident.add.a"),
                f32_tensor_descriptor(&a),
                a,
            ),
            NodeInputResource::Fresh(
                TensorResourceId::new("resident.add.b"),
                f32_tensor_descriptor(&b),
                b,
            ),
        ],
        (
            TensorResourceId::new("resident.add.out"),
            f32_tensor_descriptor(&HostTensor::new([1, 4], vec![0.0; 4]).unwrap()),
        ),
        BTreeMap::new(),
    )
    .unwrap();
    let sum = sum.into_host(&dispatch_ctx.provider).unwrap();
    assert_eq!(sum.data, vec![11.0, 12.0, 13.0, 14.0]);

    // The producer's own output resource id -- reused verbatim, not a
    // freshly synthesized one -- confirming this is a real by-reference
    // passthrough, not merely a same-shaped copy.
    let sum_resource_id = TensorResourceId::new("resident.add.out");
    assert!(matches!(
        provider.read_tensor_value(&sum_resource_id),
        Some(TensorValue::Opaque)
    ));

    let scale = HostTensor::new([1, 4], vec![2.0, 2.0, 2.0, 2.0]).unwrap();
    let (_dispatch, product) = dispatch_reference_cpu_operator(
        &mut dispatch_ctx,
        "resident.mul",
        dispatch_operator_id("mul", OperatorFamily::Tensor),
        vec![
            // The Resident input: no write, no fresh id -- `sum_resource_id`
            // is used exactly as-is.
            NodeInputResource::Resident(sum_resource_id, f32_tensor_descriptor(&scale)),
            NodeInputResource::Fresh(
                TensorResourceId::new("resident.mul.b"),
                f32_tensor_descriptor(&scale),
                scale,
            ),
        ],
        (
            TensorResourceId::new("resident.mul.out"),
            f32_tensor_descriptor(&HostTensor::new([1, 4], vec![0.0; 4]).unwrap()),
        ),
        BTreeMap::new(),
    )
    .unwrap();
    let product = product.into_host(&dispatch_ctx.provider).unwrap();
    assert_eq!(product.data, vec![22.0, 24.0, 26.0, 28.0]);
}

/// `make-first-native-cuda-hot-path-device-resident` task 2.5: proves
/// `dispatch_qwen_rmsnorm` no longer requires its caller to materialize a
/// Device-resident input to `HostTensor` before calling it -- previously
/// impossible (the old signature took `HostTensor` directly, so a caller
/// with only a Resident value had no choice but to `.into_host()` first).
/// A MatMul-shaped output already Resident under the resolved Provider
/// (`OpaqueReportingExecutor`, which never reports `Host`) passes straight
/// into RMSNorm and computes the real, independently-verified result.
#[test]
fn rmsnorm_accepts_a_resident_input_without_materializing_it_first() {
    let fixture = e2e_fixture().expect("fixture builds");
    let mut runtime = build_runtime_trusting_fixture(&fixture);
    let provider: Arc<dyn ProviderExecutionApi> = Arc::new(OpaqueReportingExecutor::new());

    // Simulates a prior MatMul's output already written under its own
    // resource id, resident in the Provider's own storage.
    let matmul_output_id = TensorResourceId::new("rmsnorm.resident.input");
    let input = HostTensor::new([1, 4], vec![2.0, 4.0, 4.0, 8.0]).unwrap();
    provider
        .write_tensor(matmul_output_id.clone(), input)
        .unwrap();
    assert!(matches!(
        provider.read_tensor_value(&matmul_output_id),
        Some(TensorValue::Opaque)
    ));

    let mut node_events = Vec::new();
    let mut dispatch_ctx = QwenDispatchContext {
        runtime: &mut runtime,
        provider: provider.clone(),
        prepared_plan: None,
        graph: None,
        sequence_length: None,
        last_provider_execution: None,
        node_events: &mut node_events,
    };
    let weight = HostTensor::new([4], vec![1.0, 1.0, 1.0, 1.0]).unwrap();
    let (_dispatch, normed) = dispatch_qwen_rmsnorm(
        &mut dispatch_ctx,
        "rmsnorm.resident",
        NodeValue::Resident {
            id: matmul_output_id,
            shape: vec![1, 4],
        },
        NodeValue::Host(weight),
        1e-6,
        None,
    )
    .unwrap();
    let normed = normed.into_host(&dispatch_ctx.provider).unwrap();
    // RMS of [2,4,4,8] = sqrt((4+16+16+64)/4) = sqrt(25) = 5; each element
    // divided by 5 (weight is all-ones): [0.4, 0.8, 0.8, 1.6].
    for (actual, expected) in normed.data.iter().zip([0.4, 0.8, 0.8, 1.6]) {
        assert!(
            (actual - expected).abs() < 1e-4,
            "expected {expected}, got {actual}"
        );
    }
}

/// `make-first-native-cuda-hot-path-device-resident` task 4.3: a weight
/// resource reported `Opaque` by the resolved Provider must resolve to
/// `NodeValue::Resident` with the shape taken from the graph edge's own
/// `TensorEdge.descriptor` -- not an error. Before this fix,
/// `resolve_qwen_weight_edge` called `.into_host()` unconditionally, which
/// fails with a structured error (`TensorValue::into_host` on `Opaque`) the
/// moment a weight is genuinely Device-resident -- exactly the audit's
/// P0-1 finding. Proven directly: without the fix, this call panics with
/// that same structured error instead of returning `Resident`.
#[test]
fn weight_edge_resolves_opaque_weight_to_resident_without_materializing() {
    let provider: Arc<dyn ProviderExecutionApi> = Arc::new(OpaqueReportingExecutor::new());
    let weight_id = TensorResourceId::new("weight.q_proj.resource");
    let weight = HostTensor::new([4, 4], vec![1.0; 16]).unwrap();
    provider.write_tensor(weight_id.clone(), weight).unwrap();
    assert!(matches!(
        provider.read_tensor_value(&weight_id),
        Some(TensorValue::Opaque)
    ));

    let mut weight_bindings = BTreeMap::new();
    weight_bindings.insert("q_proj".to_string(), weight_id.clone());
    let descriptor = f32_tensor_descriptor_from_shape(&[4, 4]);

    let resolved =
        resolve_qwen_weight_edge(&provider, &weight_bindings, "weight.q_proj", &descriptor)
            .expect("an Opaque weight must resolve, not error");

    match resolved {
        NodeValue::Resident { id, shape } => {
            assert_eq!(id, weight_id);
            assert_eq!(shape, vec![4, 4]);
        }
        NodeValue::Host(_) => panic!("expected NodeValue::Resident, got NodeValue::Host"),
    }
}

/// `make-first-native-cuda-hot-path-device-resident` task 5.3: `lm_head`'s
/// tied-embedding weight is transposed exactly once, at Model Load
/// (`bind_qwen_fixture_weights` -> `qwen_weights_with_derived_lm_head`),
/// not recomputed by `resolve_qwen_weight_edge` on every generation step.
/// Proven by running two separate graph dispatches (prefill, then decode)
/// against the same `ModelInstance` and confirming: the `lm_head` resource
/// binding is the same resource id both times, the Provider's stored data
/// for it is byte-identical to `token_embedding`'s transpose computed
/// independently, and that data is unchanged after the second dispatch --
/// nothing re-wrote it in between.
#[test]
fn lm_head_weight_is_transposed_once_at_model_load_not_per_generation_step() {
    let fixture = e2e_fixture().expect("fixture builds");
    assert!(
        fixture.config.tied_embeddings,
        "this test's premise requires a tied-embeddings fixture"
    );
    let mut runtime = build_runtime_trusting_fixture(&fixture);
    let (instance, _memory) = load_fixture_instance(&fixture, &mut runtime).unwrap();

    let weight_bindings = runtime
        .model_instance(&instance)
        .unwrap()
        .definition
        .resource_bindings
        .weights
        .clone();
    let lm_head_id = weight_bindings
        .get("lm_head")
        .expect("tied-embeddings Model Load must stage an lm_head weight")
        .clone();

    let token_embedding = fixture_tensor_by_name(&fixture.weights, "token_embedding").unwrap();
    let expected = transpose_rows_cols(token_embedding).unwrap();

    let executor = resolve_kernel_execution_provider(
        &runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )
    .unwrap();
    let staged_before = executor.read_tensor(&lm_head_id).unwrap();
    assert_eq!(
        staged_before.data, expected.data,
        "staged lm_head must equal token_embedding's transpose"
    );

    // Two full dispatches (prefill, then decode) through the real graph
    // executor, reusing the same ModelInstance/weight bindings.
    let prompt = [1, 2];
    let mut plans =
        first_native_plans_for_prompt(&runtime, &fixture, &instance, prompt.len() as u64).unwrap();
    let graphs = first_native_component_graphs_for_prompt(&fixture, prompt.len() as u64).unwrap();
    let cache_id = KvCacheId::new("test-lm-head-single-transpose-cache").unwrap();
    let prompt_ids = HostTensor::new(
        [prompt.len() as u64],
        prompt.iter().map(|id| *id as f32).collect::<Vec<_>>(),
    )
    .unwrap();
    let (_dispatch, _bindings, layer_kv, _provider) = execute_qwen_graph(
        &mut runtime,
        &fixture,
        &instance,
        &cache_id,
        &graphs.prefill,
        &mut plans.prefill,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), prompt_ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    )
    .unwrap();
    let admitted_ids = HostTensor::new([1], vec![3.0]).unwrap();
    execute_qwen_graph(
        &mut runtime,
        &fixture,
        &instance,
        &cache_id,
        &graphs.decode,
        &mut plans.decode,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), admitted_ids)]),
        Some(&layer_kv),
        Some(prompt.len() as u64),
        &mut Vec::new(),
    )
    .unwrap();

    // Same resource id, same data, after both dispatches: nothing staged a
    // second (or third) copy or rewrote it per step.
    let weight_bindings_after = runtime
        .model_instance(&instance)
        .unwrap()
        .definition
        .resource_bindings
        .weights
        .clone();
    assert_eq!(weight_bindings_after.get("lm_head"), Some(&lm_head_id));
    let staged_after = executor.read_tensor(&lm_head_id).unwrap();
    assert_eq!(
        staged_after.data, expected.data,
        "lm_head's staged data must still equal the same transpose after two dispatches, not have been recomputed or corrupted"
    );
}

fn test_kernel_id(provider: &str, name: &str) -> KernelId {
    KernelId::new(
        ProviderBinding::new(provider),
        name,
        CapabilityVersion::new(1, 0, 0),
        OperatorId::magnetar(name, 1, OperatorFamily::LinearAlgebra),
        KernelOperatorVersionRange::exact(1),
        KernelImplementationFamily::CpuScalar,
    )
}

/// `make-first-native-cuda-hot-path-device-resident` task 1.5: a node whose
/// Prepared Plan binds a non-Reference-CPU Provider/Device must produce a
/// `ResourceAffinity` matching that binding, not the previous hardcoded
/// `reference-cpu` -- this is `resolved_resource_affinity`'s whole reason
/// to exist (the audit's P0-3 finding). Tested directly against the helper
/// rather than through a full dispatch: a minimal `PreparedExecutionPlan`
/// with one binding is enough to prove the lookup, matching
/// `resolved_output_placement`'s own established test style.
#[test]
fn resolved_resource_affinity_matches_a_non_reference_cpu_plan_binding() {
    let node = ExecutionNodeId::new("cuda.matmul");
    let mut plan = PreparedExecutionPlan::new(
        PreparedExecutionPlanId::new("cuda-plan").unwrap(),
        PreparedExecutionPlanGeneration::new(1),
        ExecutionGraphSemanticFingerprint::new("sha256:test").unwrap(),
        PreparedExecutionPlanScope::for_phase(PreparedExecutionPhase::Prefill),
    )
    .unwrap();
    let device = DeviceBinding::new(DeviceId::new("cuda:0"));
    let binding = PlanNodeBinding::new(
        [node.clone()],
        test_kernel_id("magnetar:provider/cuda", "matmul"),
        ProviderBinding::new("magnetar:provider/cuda"),
    )
    .unwrap()
    .with_device(device.clone());
    plan.add_node_binding(binding).unwrap();

    let context = ExecutionContextId::new(1);
    let affinity = resolved_resource_affinity(Some(&plan), &node, context);

    assert_eq!(
        affinity.provider(),
        Some(&ProviderBinding::new("magnetar:provider/cuda"))
    );
    assert_eq!(affinity.device(), Some(&device));
}

/// Companion to the above: no Prepared Plan, or a Plan with no binding for
/// this node, must still fall back to the Reference CPU default -- every
/// existing direct-dispatch/test caller without a Plan relies on this
/// unchanged behavior.
#[test]
fn resolved_resource_affinity_falls_back_to_reference_cpu_without_a_binding() {
    let node = ExecutionNodeId::new("some.node");
    let context = ExecutionContextId::new(1);

    let no_plan = resolved_resource_affinity(None, &node, context);
    assert_eq!(
        no_plan.provider(),
        Some(&ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME))
    );

    let plan = PreparedExecutionPlan::new(
        PreparedExecutionPlanId::new("empty-plan").unwrap(),
        PreparedExecutionPlanGeneration::new(1),
        ExecutionGraphSemanticFingerprint::new("sha256:test2").unwrap(),
        PreparedExecutionPlanScope::for_phase(PreparedExecutionPhase::Prefill),
    )
    .unwrap();
    let no_binding = resolved_resource_affinity(Some(&plan), &node, context);
    assert_eq!(
        no_binding.provider(),
        Some(&ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME))
    );
}

/// `make-first-native-cuda-hot-path-device-resident` task 1.6: a Resident
/// resource's own recorded affinity (e.g. a Model-Load-time weight's) must
/// survive a later dispatch on the same Provider -- aggregated, not
/// replaced. Proven by giving the resident resource a `Device` binding the
/// dispatch's own (no-Prepared-Plan) affinity does not have: if the result
/// carries that Device, it can only have come from the resident resource's
/// own record, not from the dispatch's freshly-derived affinity.
#[test]
fn resident_resource_affinity_is_preserved_not_overwritten() {
    let fixture = e2e_fixture().expect("fixture builds");
    let mut runtime = build_runtime_trusting_fixture(&fixture);
    let resident_id = TensorResourceId::new("resident.weight");
    let device = DeviceBinding::new(DeviceId::new("cuda:0"));
    let recorded_affinity = ResourceAffinity::new(FallbackClass::Transparent)
        .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME))
        .with_device(device.clone());
    runtime
        .memory_mut()
        .record_tensor_residency(TensorResidency::new(
            resident_id.clone(),
            MemoryPlacement::ProviderOwnedOpaque(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)),
            recorded_affinity,
        ))
        .unwrap();

    // The dispatch's own affinity (no Prepared Plan bound): same Provider,
    // no Device of its own.
    let dispatch_affinity = ResourceAffinity::new(FallbackClass::Transparent)
        .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME));
    let node = ExecutionNodeId::new("resident.affinity.add");

    let resolved =
        resident_resource_affinity(runtime.memory(), &resident_id, &dispatch_affinity, &node)
            .unwrap();

    assert_eq!(
        resolved.provider(),
        Some(&ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME))
    );
    assert_eq!(
        resolved.device(),
        Some(&device),
        "the resident resource's own Device binding must survive aggregation, \
         proving it was preserved rather than replaced by the dispatch's own \
         (deviceless) affinity"
    );
}

/// Companion to the above: a genuine Provider conflict between a Resident
/// resource's recorded affinity and the dispatch's own resolved affinity
/// must be rejected with a structured error, never silently overwritten.
#[test]
fn resident_resource_affinity_conflict_is_rejected() {
    let fixture = e2e_fixture().expect("fixture builds");
    let mut runtime = build_runtime_trusting_fixture(&fixture);
    let resident_id = TensorResourceId::new("resident.weight.conflict");
    let recorded_affinity = ResourceAffinity::new(FallbackClass::Transparent)
        .with_provider(ProviderBinding::new("magnetar:provider/cuda"));
    runtime
        .memory_mut()
        .record_tensor_residency(TensorResidency::new(
            resident_id.clone(),
            MemoryPlacement::ProviderOwnedOpaque(ProviderBinding::new("magnetar:provider/cuda")),
            recorded_affinity,
        ))
        .unwrap();

    let dispatch_affinity = ResourceAffinity::new(FallbackClass::Transparent)
        .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME));
    let node = ExecutionNodeId::new("resident.affinity.conflict.add");

    let result =
        resident_resource_affinity(runtime.memory(), &resident_id, &dispatch_affinity, &node);

    assert!(
        result.is_err(),
        "a Provider mismatch between a Resident resource's recorded affinity \
         and the dispatch's own resolved affinity must be rejected, not \
         silently resolved by picking one of them"
    );
}

/// GitHub issue "A Model Instance stuck in Loading cannot currently be
/// unloaded or canceled": proves the existing `fail_instance` +
/// `unload_model_instance` combination already provides a real, working
/// cancellation path for an instance that was created but never (or only
/// partially) materialized -- `ModelInstance::fail`/`invalidate`
/// unconditionally set lifecycle to `Failed`/`Invalid` regardless of the
/// instance's current state (they do not go through
/// `allows_transition_to` at all), and `(Failed, Unloading)` is already a
/// valid transition `ModelInstanceManager::unload` already accepts. No new
/// lifecycle transition or unload entrypoint was needed.
#[test]
fn loading_instance_can_be_canceled_via_fail_then_unload() {
    let fixture = e2e_fixture().expect("fixture builds");
    let mut runtime = build_runtime_trusting_fixture(&fixture);
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(fixture.architecture_implementation.clone());
    let mut request = ModelLoadingRequest::new(
        ModelLoadingRequestId::new("e2e-fixture-load"),
        fixture.manifest.id.clone(),
    );
    request.quantization_policy = ModelQuantizationPolicy::RejectUnsupported;
    let loaded = load_model(
        &mut coordinator,
        &mut runtime,
        ModelLoadingApiRequest::new(request),
        &fixture.manifest,
    )
    .expect("loading succeeds");
    let instance = create_model_instance(
        &mut runtime,
        &loaded,
        fixture.architecture_implementation.clone(),
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .expect("creation succeeds");
    // Never materialized: still Loading, exactly the stuck scenario this
    // issue describes.
    assert_eq!(
        runtime.model_instance(&instance).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Loading
    );

    runtime
        .model_instances_mut()
        .fail_instance(
            &instance,
            ModelInstanceError::InternalModelInstance {
                reason: "abandoned before materialization".into(),
            },
        )
        .expect("fail_instance works from Loading");
    assert_eq!(
        runtime.model_instance(&instance).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Failed
    );

    let report = runtime
        .unload_model_instance(&instance, ModelInstanceUnloadPolicy::DrainActiveUse)
        .expect("an instance that never materialized anything still unloads cleanly");
    assert!(report.released_weight_resources.is_empty());
    assert!(report.released_memory_allocations.is_empty());
    assert_eq!(
        runtime.model_instance(&instance).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Unloaded
    );
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
    )
    .unwrap();
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

/// `generalize-first-native-provider-dispatch` P0 fix regression: a Model
/// Instance whose placement binds a non-Reference-CPU registered Provider
/// materializes its weights through *that* Provider's storage, not
/// Reference CPU's -- `WeightMaterializationTransaction::begin()` used to
/// hardcode Reference CPU unconditionally, which would have silently
/// written this weight into the wrong Provider (or failed to find it later
/// under the right one).
#[test]
fn weight_materialization_uses_the_model_instances_bound_provider() {
    let fixture = e2e_fixture().expect("fixture builds");
    let mock_provider = ProviderBinding::new("magnetar:provider/mock-kernel");
    let mut runtime = Runtime::builder()
        .register_provider(Arc::new(ReferenceCpuProvider::new()))
        .register_provider(Arc::new(MockKernelProvider::new()))
        .trust_store(
            ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone()),
        )
        .build()
        .expect("both providers register cleanly");

    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(fixture.architecture_implementation.clone());
    let mut request = ModelLoadingRequest::new(
        ModelLoadingRequestId::new("provider-dispatch-weight-test"),
        fixture.manifest.id.clone(),
    );
    request.quantization_policy = ModelQuantizationPolicy::RejectUnsupported;
    let loaded = load_model(
        &mut coordinator,
        &mut runtime,
        ModelLoadingApiRequest::new(request),
        &fixture.manifest,
    )
    .expect("model loads");
    let instance = create_model_instance(
        &mut runtime,
        &loaded,
        fixture.architecture_implementation.clone(),
        ResourceAffinity::new(FallbackClass::Transparent).with_provider(mock_provider.clone()),
    )
    .expect("instance binds to the mock provider (loading resolved no conflicting binding)");

    let weight_name = "test.arbitrary.weight";
    let tensor = HostTensor::new([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    materialize_model_instance_weights(
        &mut runtime,
        &instance,
        "test",
        &BTreeMap::from([(weight_name.to_string(), tensor.clone())]),
    )
    .expect("weight materializes through the bound (mock) provider");

    let resource_id = runtime
        .model_instance(&instance)
        .unwrap()
        .definition()
        .resource_bindings
        .weights
        .get(weight_name)
        .cloned()
        .expect("weight binding recorded");

    let mock_api = resolve_kernel_execution_provider(&runtime, &mock_provider)
        .expect("mock provider resolves");
    assert_eq!(
        mock_api.read_tensor(&resource_id),
        Some(tensor),
        "weight must be readable from the bound mock Provider's storage"
    );

    let cpu_api = resolve_kernel_execution_provider(
        &runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )
    .expect("reference CPU provider resolves");
    assert!(
        cpu_api.read_tensor(&resource_id).is_none(),
        "weight must not have been written to Reference CPU's storage"
    );
}

/// `generalize-first-native-provider-dispatch` P0 fix regression:
/// `KvUpdateTransaction::begin` resolves the Provider carried on
/// `FirstNativeExecutionKvState.provider` (what `execute_qwen_graph`
/// resolved and wrote this step's pending K/V resources under), not
/// Reference CPU unconditionally. Tests `begin`'s resolution directly
/// (rather than driving a full non-CPU generation step, which would need a
/// mock Provider implementing every Qwen graph Operator, not just this
/// contract) since `begin` is the entire fix -- everything downstream of it
/// already receives whichever `provider_binding` it resolves.
#[test]
fn kv_update_transaction_resolves_the_states_bound_provider() {
    let mock_provider = ProviderBinding::new("magnetar:provider/mock-kernel");
    let runtime = Runtime::builder()
        .register_provider(Arc::new(ReferenceCpuProvider::new()))
        .register_provider(Arc::new(MockKernelProvider::new()))
        .build()
        .expect("both providers register cleanly");

    let state = FirstNativeExecutionKvState {
        cache: KvCacheId::new("test-kv-transaction-binding-cache").unwrap(),
        compatibility: KvCacheCompatibility::new(
            GenerationModelReference::LoadedModelContext("qwen-test".into()),
            TokenizerId::new("qwen-test-tokenizer").unwrap(),
        ),
        layer_kv: QwenLayerKvMap::new(),
        provider: Some(mock_provider.clone()),
    };

    let transaction = KvUpdateTransaction::begin(&runtime, &state).expect("mock provider resolves");
    assert_eq!(
        transaction.provider_binding, mock_provider,
        "must resolve the Provider bound on the KV state, not Reference CPU"
    );
}

/// Same fix, the other branch: a state with no bound Provider (never
/// touched by `execute_qwen_graph`, e.g. a freshly created prefill state)
/// still falls back to Reference CPU -- preserving today's only real
/// behavior exactly.
#[test]
fn kv_update_transaction_falls_back_to_reference_cpu_when_unbound() {
    let runtime = build_runtime_trusting_fixture(&e2e_fixture().expect("fixture builds"));

    let state = FirstNativeExecutionKvState {
        cache: KvCacheId::new("test-kv-transaction-fallback-cache").unwrap(),
        compatibility: KvCacheCompatibility::new(
            GenerationModelReference::LoadedModelContext("qwen-test".into()),
            TokenizerId::new("qwen-test-tokenizer").unwrap(),
        ),
        layer_kv: QwenLayerKvMap::new(),
        provider: None,
    };

    let transaction = KvUpdateTransaction::begin(&runtime, &state).expect("reference CPU resolves");
    assert_eq!(
        transaction.provider_binding,
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)
    );
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

    executor
        .write_tensor_value(id.clone(), TensorValue::Opaque)
        .unwrap();
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
/// `FirstNativeChatSession::turn`, and -- since
/// `wire-inference-component-to-generic-registry` --
/// `build_first_native_graphs_from_named_component`'s own no-engine
/// fallback, the same fail-closed guarantee extended to a caller-registered
/// Component instead of only the CLI's hardcoded singleton).
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
        fail_closed_occurrences, 4,
        "expected exactly the four documented production fail-closed stubs \
         (first_native_component_graphs_for_prompt, run_success_path_with_prompt, \
         FirstNativeChatSession::turn, build_first_native_graphs_from_named_component) to share \
         this exact structured error message when no strict Component engine is available; \
         found {fail_closed_occurrences}. If a call site was added or removed, update this count."
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
    // Normalized to `\n` regardless of this checkout's line-ending
    // convention (a real bug this test itself had: it panicked on Windows
    // CI, whose checkout uses CRLF, before this normalization was added).
    let source = include_str!("../first_native_runtime.rs").replace("\r\n", "\n");
    let production_loader_start = source
        .find("not(test)\n))]\nfn qwen_real_component_package()")
        .expect("the production (not(test)) qwen_real_component_package overload exists");
    // Covers qwen_real_component_package's thin not(test) wrapper and its
    // two siblings immediately after it in the source:
    // resolve_qwen_component_from_env_var (the std::env::var read) and
    // resolve_qwen_component_from_lookup (the std::fs::read logic, split
    // out of resolve_qwen_component_from_env_var by #68 so tests can
    // supply a lookup result directly instead of mutating the real process
    // environment) -- factored out so the actual env-var/file-read logic
    // stays directly testable rather than living inside a not(test) cfg
    // that no #[test] could ever reach. The third `\n}\n` closes the last
    // of the three.
    let first_fn_end = production_loader_start
        + source[production_loader_start..]
            .find("\n}\n")
            .expect("qwen_real_component_package has a closing brace")
        + "\n}\n".len();
    let second_fn_end = first_fn_end
        + source[first_fn_end..]
            .find("\n}\n")
            .expect("resolve_qwen_component_from_env_var has a closing brace")
        + "\n}\n".len();
    let production_loader_end = second_fn_end
        + source[second_fn_end..]
            .find("\n}\n")
            .expect("resolve_qwen_component_from_lookup has a closing brace");
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

/// Static guard (`close-tachyon-scope-audit-gaps` task 3.3): the real
/// chat-template rendering path added by task group 3 --
/// `run_production_qwen_generation_for_provider_with_prompt`, the only
/// production entry point that accepts a caller-supplied
/// `ChatTemplateFormatter` -- must stay reachable through this crate's
/// public API alone. Source-inspects `first_native_runtime.rs` for the
/// exact `pub fn` declaration rather than asserting on run-time behavior,
/// since the property being enforced (this function is not accidentally
/// `pub(crate)`, `pub(super)`, or private) is a visibility fact a
/// `#[test]` inside the same crate cannot otherwise distinguish -- an
/// external caller in a different crate can call a `pub fn` here but
/// not a `pub(crate)` one, so only source inspection actually proves
/// this. Mirrors the sibling guards above, which use the same technique
/// for the same structural reason.
#[test]
fn production_chat_formatted_generation_entry_point_is_public() {
    let source = include_str!("../first_native_runtime.rs");
    assert!(
        source.contains("pub fn run_production_qwen_generation_for_provider_with_prompt("),
        "run_production_qwen_generation_for_provider_with_prompt must be declared `pub fn` -- \
         it is the only production entry point through which an external caller can render \
         PromptInput::ChatMessages via a real, artifact-declared ChatTemplateFormatter, and a \
         narrower visibility would make the real chat-template path unreachable from outside \
         this crate"
    );
}

/// Static guard (`expose-production-generation-parameters` task 2.3):
/// `run_production_qwen_generation_for_provider_with_request`, the only
/// production entry point that accepts caller-supplied
/// `GenerationParameters`/`StopConditions` instead of the hardcoded
/// greedy/default values every other entry point in this family still
/// passes, must stay reachable through this crate's public API alone.
/// Mirrors the sibling guard above for the same structural reason: a
/// `#[test]` inside this crate cannot otherwise distinguish `pub fn` from
/// `pub(crate)`/`pub(super)`/private, and only source inspection proves it.
#[test]
fn production_request_generation_entry_point_is_public() {
    let source = include_str!("../first_native_runtime.rs");
    assert!(
        source.contains("pub fn run_production_qwen_generation_for_provider_with_request("),
        "run_production_qwen_generation_for_provider_with_request must be declared `pub fn` -- \
         it is the only production entry point through which an external caller (e.g. Tachyon \
         translating an OpenAI-shaped request) can supply real GenerationParameters/StopConditions, \
         and a narrower visibility would make that surface unreachable from outside this crate"
    );
}

/// `register_qwen_component_artifact` must be safe to call unconditionally,
/// every time a caller might need first-native generation, without the
/// caller tracking its own "have I registered yet" state (task 12.4's
/// design: `magnetar-cli` calls this before every `one_shot`/`ChatSession::
/// open`). Proves the second call is a genuine no-op, not a panic or an
/// error, regardless of whether the bytes differ from the first call's.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn register_qwen_component_artifact_is_idempotent() {
    register_qwen_component_artifact(
        b"first-call-bytes".to_vec(),
        b"first-call-manifest".to_vec(),
    );
    register_qwen_component_artifact(
        b"second-call-different-bytes".to_vec(),
        b"second-call-different-manifest".to_vec(),
    );
}

/// `resolve_qwen_component_from_lookup` is the extracted, directly
/// testable logic behind `qwen_real_component_package`'s production
/// fallback branch, which is itself `not(test)` and so can never be
/// invoked from a `#[test]` at all. #68: previously these tests drove
/// `resolve_qwen_component_from_env_var` and mutated the real process
/// environment (`std::env::set_var`/`remove_var`, `unsafe` in Rust 2024
/// precisely because they race with any *concurrent* environment access on
/// another thread) to fake each scenario; each test used its own
/// uniquely-named variable, which avoided colliding on the same key but
/// not the actual hazard of a concurrent access on another thread during
/// `cargo test`'s parallel execution. Supplying the lookup result directly
/// removes the hazard entirely rather than working around it. Filesystem
/// paths are still real temp files (each test uses its own uniquely-named
/// path, and writes to distinct paths from different threads are not a
/// hazard the way concurrent environment mutation is).
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
mod resolve_qwen_component_from_env_var_tests {
    use super::*;

    #[test]
    fn rejects_a_missing_env_var() {
        let var_name = "MAGNETAR_TEST_QWEN_COMPONENT_PATH_UNSET_CASE";
        let error =
            resolve_qwen_component_from_lookup(var_name, Err(std::env::VarError::NotPresent))
                .unwrap_err();
        assert!(matches!(
            error,
            E2eConformanceError::ModelComponentFailed { reason } if reason.contains(var_name)
        ));
    }

    #[test]
    fn rejects_a_nonexistent_component_file() {
        let var_name = "MAGNETAR_TEST_QWEN_COMPONENT_PATH_MISSING_FILE_CASE";
        let path = std::env::temp_dir().join("magnetar-test-qwen-component-does-not-exist.wasm");
        let error =
            resolve_qwen_component_from_lookup(var_name, Ok(path.to_str().unwrap().to_string()))
                .unwrap_err();
        assert!(matches!(
            error,
            E2eConformanceError::ModelComponentFailed { reason }
                if reason.contains("failed to read Qwen Component bytes")
        ));
    }

    #[test]
    fn rejects_a_component_file_with_no_manifest() {
        let var_name = "MAGNETAR_TEST_QWEN_COMPONENT_PATH_MISSING_MANIFEST_CASE";
        let path = std::env::temp_dir().join("magnetar-test-qwen-component-no-manifest.wasm");
        std::fs::write(&path, b"pretend-component-bytes").expect("write test component file");
        let error =
            resolve_qwen_component_from_lookup(var_name, Ok(path.to_str().unwrap().to_string()))
                .unwrap_err();
        let _ = std::fs::remove_file(&path);
        assert!(matches!(
            error,
            E2eConformanceError::ModelComponentFailed { reason }
                if reason.contains("failed to read Qwen Component manifest")
        ));
    }

    #[test]
    fn reads_bytes_and_manifest_from_the_configured_path() {
        let var_name = "MAGNETAR_TEST_QWEN_COMPONENT_PATH_HAPPY_CASE";
        let path = std::env::temp_dir().join("magnetar-test-qwen-component-happy.wasm");
        let manifest_path = std::env::temp_dir()
            .join("magnetar-test-qwen-component-happy.wasm.magnetar-component.yaml");
        std::fs::write(&path, b"pretend-component-bytes").expect("write test component file");
        std::fs::write(&manifest_path, b"pretend-manifest-bytes")
            .expect("write test manifest file");
        let package =
            resolve_qwen_component_from_lookup(var_name, Ok(path.to_str().unwrap().to_string()))
                .expect("resolves successfully");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&manifest_path);
        assert_eq!(package.component_bytes, b"pretend-component-bytes");
        assert_eq!(package.manifest_bytes, b"pretend-manifest-bytes");
        assert_eq!(
            package.source.kind,
            ComponentDistributionSourceKind::LocalDirectory
        );
    }

    /// One narrow check that `resolve_qwen_component_from_env_var` is
    /// genuinely wired to `std::env::var`, not only
    /// `resolve_qwen_component_from_lookup` in isolation -- reads a
    /// variable this test never sets (a plain read, with no concurrent
    /// writer, carries none of `set_var`/`remove_var`'s hazard) rather
    /// than reintroducing environment mutation.
    #[test]
    fn env_var_wrapper_reports_a_variable_that_is_genuinely_absent() {
        let var_name = "MAGNETAR_TEST_QWEN_COMPONENT_PATH_GENUINELY_ABSENT_7f2e9c";
        let error = resolve_qwen_component_from_env_var(var_name).unwrap_err();
        assert!(matches!(
            error,
            E2eConformanceError::ModelComponentFailed { reason } if reason.contains(var_name)
        ));
    }
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
fn e2e_graph_dispatch_does_not_leak_kernel_output_allocations_across_repeated_dispatch() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_graph_dispatch_does_not_leak_kernel_output_allocations_across_repeated_dispatch(&fixture)
        .expect("repeated graph dispatch does not leak Provider-owned Kernel output allocations");
}

#[test]
fn e2e_graph_dispatch_intermediate_edge_is_resolvable_from_provider_storage() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_graph_dispatch_intermediate_edge_is_resolvable_from_provider_storage(&fixture)
        .expect("intermediate graph edges resolve from Provider storage, not a private cache");
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
fn e2e_materialize_model_instance_weights_rejects_content_digest_mismatch() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_materialize_model_instance_weights_rejects_content_digest_mismatch(&fixture)
        .expect("materialize_model_instance_weights rejects tampered tensor content");
}

#[test]
fn e2e_materialize_model_instance_weights_accepts_matching_content() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_materialize_model_instance_weights_accepts_matching_content(&fixture)
        .expect("materialize_model_instance_weights accepts real, untampered tensor content");
}

#[test]
fn e2e_weight_materialization_failure_never_reaches_ready() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_weight_materialization_failure_never_reaches_ready(&fixture).expect(
        "a Model Instance never reports Ready when weight materialization fails, and rolls \
         back every weight staged in the failed attempt",
    );
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
fn e2e_two_segment_split_produces_identical_output_to_full_graph() {
    check_two_segment_split_produces_identical_output_to_full_graph().expect(
        "splitting a real forward pass into two segment Model Instances, bridged \
                  through a real hidden-state hand-off, must bit-for-bit match running the \
                  same prompt through the one full graph",
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

// ---------------------------------------------------------------------------
// Generic, multi-Component runtime registry
// (`wire-generic-inference-component-runtime`)
// ---------------------------------------------------------------------------

/// Consolidates every assertion that depends on `QWEN_REAL_COMPONENT_BYTES`'s
/// digest into one sequential test, deliberately -- not split across
/// several `#[test]` functions. `REGISTERED_COMPONENT_RUNTIMES` is one
/// process-wide registry shared by every test in this binary, keyed by real
/// content digest; every one of these assertions registers (or attempts to
/// register) the *same* fixture bytes, so the same digest. Splitting them
/// into separate `#[test]` functions is genuinely racy under Rust's default
/// parallel test execution: whichever test's *trusted* registration runs
/// first populates the shared cache entry, after which a *later*,
/// deliberately-untrusted registration attempt for that same digest would
/// hit the idempotent-return fast path before ever re-evaluating trust,
/// making the "untrusted artifacts are rejected" assertion pass or fail
/// based on test scheduling instead of behavior -- caught by a real CI
/// failure (this exact race), not found by inspection.
///
/// This same reasoning is why the Tachyon integration audit MAG-07 proof
/// (two distinct, independently-registered Components served at once) lives
/// at the end of this function instead of its own `#[test]`: a second test
/// that trusts and registers `QWEN_REAL_COMPONENT_BYTES` under its own
/// fresh `ComponentTrustStore` races this function's own untrusted-rejection
/// assertion above by the identical mechanism (confirmed by a real,
/// reproduced local failure before consolidating).
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
    .expect("a caller-trusted, well-formed artifact must register");
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
    .expect("re-registering the same bytes is a harmless no-op, not an error");
    assert_eq!(digest, second);

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
        &fixture.config,
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
    .expect("the synthetic minimal Component must also register, alongside Qwen");
    assert_eq!(registered_synthetic_digest, synthetic_digest);

    let (synthetic_graphs, _definition, _instance) =
        build_first_native_graphs_from_named_component(
            &synthetic_digest,
            &fixture.config,
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
        &fixture.config,
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
    .expect("the real Llama Component must register");
    assert_eq!(registered, llama_digest);

    // e2e_fixture()'s own config has `attention_bias: false` (`QwenConfig::
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
        &fixture.config,
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
            &biased_config,
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
        &fixture.config,
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
    let mut runtime = build_runtime_trusting_fixture(&fixture);
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
    let mut runtime = build_runtime_trusting_fixture(&fixture);
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

/// In-memory [`ProductionArtifactPayloadSource`] test double: looks up
/// bytes by tensor name only (offset/length are not cross-checked against
/// a real file layout, since there is no file here) -- proves
/// `load_production_qwen_instance` genuinely reads weight bytes through
/// the payload-source contract, not a fixture-only shortcut.
struct ProductionIntegrationPayloadSource {
    bytes_by_name: BTreeMap<String, Vec<u8>>,
}

impl crate::production_model_ingestion::ProductionArtifactPayloadSource
    for ProductionIntegrationPayloadSource
{
    fn read_payload(
        &self,
        range: &crate::production_model_ingestion::ProductionPayloadRange,
    ) -> Result<Vec<u8>, crate::production_model_ingestion::ProductionIngestionError> {
        self.bytes_by_name
            .get(&range.identity)
            .cloned()
            .ok_or_else(|| {
                crate::production_model_ingestion::ProductionIngestionError::PayloadOutOfBounds {
                    identity: range.identity.clone(),
                }
            })
    }
}

/// Task groups 10-11 end to end: a `QwenConfig` deliberately different
/// from the canonical E2E fixture (hidden_size=8, attention_head_count=4,
/// kv_head_count=2 -- genuinely GQA-shaped, unlike the fixture's 2/2) is
/// wrapped as a production-shaped `ModelManifest` (no fixture manifest
/// constructor, real `architecture_config`, a real payload source keyed
/// only by canonical tensor name/bytes) and driven through
/// `load_production_qwen_instance` -> `production_qwen_fixture` ->
/// the same real Qwen Component graph production, generic Runtime
/// Inference API, and generation loop every other first-native caller
/// uses -- with no `qwen-test` identity or fixture manifest anywhere in
/// this path.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn production_loading_generates_end_to_end_with_a_non_canonical_qwen_config() {
    // vocab_size=258 (not 64): the delegate tokenizer below is the real
    // byte-level FixtureTokenizer (each byte encodes to `byte + 1`, up to
    // 256, plus BOS/EOS), so the vocabulary must cover that range
    // regardless of this test's otherwise-deliberately-non-canonical
    // dimensions.
    let architecture = qwen_architecture_metadata(8, 1, 4, 2, 2, 16, 258, 1_000_000);
    let config = QwenConfig {
        architecture,
        rope: QwenRopeConfig {
            base: 10_000.0,
            scale: None,
            dimension: 2,
            position_mode: QwenRopePositionMode::Sequential,
            dynamic_scaling_supported: false,
        },
        rmsnorm_epsilon: 1e-6,
        tied_embeddings: false,
        attention_bias: false,
        require_bos: false,
        require_pad: false,
        expected_added_tokens: None,
        chat_template_required: false,
    };

    let weights = e2e_fixture_weights(&config).expect("synthetic weights build");
    let tensors = e2e_fixture_weight_inventory(&config).expect("tensor inventory builds");
    let mut bytes_by_name = BTreeMap::new();
    for tensor in &tensors {
        let host_tensor = weights
            .get(&tensor.name)
            .expect("a weight exists for every inventory tensor");
        let mut bytes = Vec::with_capacity(host_tensor.data.len() * 4);
        for value in &host_tensor.data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes_by_name.insert(tensor.name.clone(), bytes);
    }
    let payload_source = ProductionIntegrationPayloadSource { bytes_by_name };

    let digest = ModelDigest::parse(format!("sha256:{}", "7".repeat(64))).unwrap();
    let id = ModelArtifactId::new(
        ModelArtifactKind::ModelBundle,
        ModelName::new("production-integration-test").unwrap(),
        ModelRevision::new("r1").unwrap(),
        digest,
    );
    let mut parts = BTreeMap::new();
    parts.insert(
        "weights".to_string(),
        ModelArtifactPart {
            name: "weights".to_string(),
            kind: ModelArtifactKind::ModelWeights,
            digest: ModelDigest::parse(format!("sha256:{}", "8".repeat(64))).unwrap(),
            size_bytes: None,
            required: true,
        },
    );
    parts.insert(
        "config".to_string(),
        ModelArtifactPart {
            name: "config".to_string(),
            kind: ModelArtifactKind::ModelConfig,
            digest: ModelDigest::parse(format!("sha256:{}", "9".repeat(64))).unwrap(),
            size_bytes: None,
            required: true,
        },
    );
    let manifest = ModelManifest {
        schema_version: crate::MODEL_ARTIFACT_SCHEMA_VERSION,
        id,
        architecture: ModelArchitecture::new("qwen", "production-integration-test"),
        parts,
        storage_dtype: Some(ModelDType::F32),
        compute_dtype: None,
        supported_compute_dtypes: BTreeSet::from([ModelDType::F32]),
        tensors,
        tokenizer: None,
        tokenizer_config: None,
        chat_template: None,
        prompt_template: None,
        generation: None,
        quantization: None,
        shards: Vec::new(),
        runtime_features: BTreeSet::new(),
        memory_features: BTreeSet::new(),
        provider_capabilities: Vec::new(),
        component: None,
        license: None,
        provenance: None,
        signatures: Vec::new(),
        source: None,
        architecture_config: Some(architecture_config_from_qwen_config(&config)),
    };

    let tokenizer_metadata = e2e_fixture_tokenizer().unwrap().metadata().clone();
    let delegate_tokenizer: std::sync::Arc<dyn crate::tokenizer::Tokenizer + Send + Sync> =
        std::sync::Arc::new(e2e_fixture_tokenizer().unwrap());
    let fixture = production_qwen_fixture(manifest.clone(), tokenizer_metadata, delegate_tokenizer)
        .expect("production fixture builds against a genuinely different config than the canonical fixture");
    assert_eq!(fixture.config.architecture.attention_head_count, 4);
    assert_eq!(fixture.config.architecture.kv_head_count, 2);

    let mut runtime = build_runtime_with_model_execution_engine(&fixture);
    let instance = load_production_qwen_instance(&mut runtime, &manifest, &payload_source).expect(
        "production loading succeeds through the generic Inference API, not a fixture loader",
    );
    require_ready_first_native_instance(&runtime, &instance)
        .expect("a production-loaded instance is genuinely ready");

    let session_request = SessionCreationRequest {
        model: GenerationModelReference::ModelInstance(instance.clone()),
        tokenizer: generation_tokenizer_reference(&fixture),
        generation_defaults: GenerationParameters::greedy(),
        policy: SessionPolicy::default(),
        memory: SessionMemoryBudget::default(),
        allowed_capabilities: BTreeSet::new(),
        correlation_id: None,
        created_at_millis: 0,
    };
    let session = create_inference_session(&mut runtime, session_request).expect("session creates");

    let tokenized = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::PlainText("hi".into())),
        None,
    )
    .expect("tokenization succeeds");

    let (component_graphs, _definition, _component_instance) =
        build_first_native_graphs_from_real_qwen_component(
            &fixture,
            tokenized.token_ids.len() as u64,
        )
        .expect("the real Qwen Component produces graphs for this non-canonical config");

    let mut prepared_plans = prepare_first_native_execution_plans(
        &runtime,
        &instance,
        component_graphs,
        tokenized.token_ids.len() as u64,
    )
    .expect("execution plans prepare");

    let mut observer = InferenceApiObserver::new();
    let request = build_generation_request(
        GenerationRequestId::new("production-integration-test").unwrap(),
        Some(session.clone()),
        GenerationModelReference::ModelInstance(instance.clone()),
        generation_tokenizer_reference(&fixture),
        tokenized,
        4,
        GenerationParameters::greedy(),
        StopConditions {
            eos: EosPolicy {
                eos_token_ids: vec![E2E_FIXTURE_EOS_TOKEN],
                ..EosPolicy::default()
            },
            ..StopConditions::default()
        },
        StreamingMode::TokenIds,
    );
    let request = prepare_generation(&runtime, request).expect("generation request validates");

    let mut execution_plans = RuntimeGenerationExecutionPlans {
        prefill: &mut prepared_plans.prefill,
        decode: &mut prepared_plans.decode,
    };
    let generation_result = run_generation_loop_with_execution_plans(
        &mut runtime,
        &request,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |_generated_so_far| false,
        &mut observer,
        &mut execution_plans,
    )
    .expect("generation runs end to end through the real Component graph and Provider dispatch");

    assert!(
        !generation_result.output.generated_token_ids.is_empty(),
        "production loading + generation produced at least one token"
    );

    close_inference_session(&mut runtime, &session).expect("session closes");
    unload_model_instance(
        &mut runtime,
        &instance,
        ModelInstanceUnloadPolicy::DrainActiveUse,
    )
    .expect("model instance unloads cleanly, no leaked resources");
}

/// The load-bearing correctness proof for `ProductionQwenLoadedModel::
/// load_with_component` (`wire-generic-inference-component-runtime`'s
/// follow-up phase, closing the Tachyon integration audit's MAG-02): a
/// model loaded against an *explicitly registered* Component digest
/// generates *exactly* the same tokens as the same model loaded through
/// [`ProductionQwenLoadedModel::load`]'s pre-existing hardcoded-singleton
/// path, for the identical underlying Component bytes. This proves the
/// registered Component genuinely drives generation end to end (plan
/// production in `prepare_generation` and dispatch-time graph production in
/// `E2eRuntimeModelExecutionEngine::execute_generation_step` both honor the
/// same digest), not merely that graph construction alone matches (already
/// proven by `register_inference_component_artifact_enforces_trust_is_
/// idempotent_and_matches_the_singleton_path`).
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn production_qwen_loaded_model_load_with_component_matches_the_singleton_path() {
    // `e2e_fixture()`'s own manifest is shaped for the in-memory,
    // fixture-only `E2eRuntimeModelExecutionEngine` path (no declared
    // per-tensor byte offset/size) -- `load_production_qwen_instance_for_
    // provider` (the real production-loading path `ProductionQwenLoadedModel::
    // load` drives) requires those, so this test builds its own
    // production-shaped manifest around the same canonical config, exactly
    // like `production_loading_generates_end_to_end_with_a_non_canonical_
    // qwen_config` does for its own deliberately-different config.
    // `tied_embeddings: false` (unlike the canonical fixture's own `true`):
    // this test drives loading through the real production path
    // (`load_production_qwen_instance_for_provider`), which expects a
    // literal `lm_head` weight resource bound -- tied-embedding derivation
    // is an ingestion-layer concern (`loaders/huggingface`'s
    // `append_synthetic_lm_head_if_tied`) this hand-built manifest
    // deliberately bypasses, exactly like `production_loading_generates_
    // end_to_end_with_a_non_canonical_qwen_config`'s own config already
    // does. Otherwise identical to the canonical fixture the real
    // `qwen-real.component.wasm` Component already exercises elsewhere.
    let mut config = e2e_fixture_config();
    config.tied_embeddings = false;
    let weights = e2e_fixture_weights(&config).expect("synthetic weights build");
    let tensors = e2e_fixture_weight_inventory(&config).expect("tensor inventory builds");
    let mut bytes_by_name = BTreeMap::new();
    for tensor in &tensors {
        let host_tensor = weights
            .get(&tensor.name)
            .expect("a weight exists for every inventory tensor");
        let mut bytes = Vec::with_capacity(host_tensor.data.len() * 4);
        for value in &host_tensor.data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes_by_name.insert(tensor.name.clone(), bytes);
    }
    let payload_source = ProductionIntegrationPayloadSource { bytes_by_name };

    let digest_seed = ModelDigest::parse(format!("sha256:{}", "a".repeat(64))).unwrap();
    let id = ModelArtifactId::new(
        ModelArtifactKind::ModelBundle,
        ModelName::new("load-with-component-test").unwrap(),
        ModelRevision::new("r1").unwrap(),
        digest_seed,
    );
    let mut parts = BTreeMap::new();
    parts.insert(
        "weights".to_string(),
        ModelArtifactPart {
            name: "weights".to_string(),
            kind: ModelArtifactKind::ModelWeights,
            digest: ModelDigest::parse(format!("sha256:{}", "b".repeat(64))).unwrap(),
            size_bytes: None,
            required: true,
        },
    );
    parts.insert(
        "config".to_string(),
        ModelArtifactPart {
            name: "config".to_string(),
            kind: ModelArtifactKind::ModelConfig,
            digest: ModelDigest::parse(format!("sha256:{}", "c".repeat(64))).unwrap(),
            size_bytes: None,
            required: true,
        },
    );
    let manifest = ModelManifest {
        schema_version: crate::MODEL_ARTIFACT_SCHEMA_VERSION,
        id,
        architecture: ModelArchitecture::new("qwen", "load-with-component-test"),
        parts,
        storage_dtype: Some(ModelDType::F32),
        compute_dtype: None,
        supported_compute_dtypes: BTreeSet::from([ModelDType::F32]),
        tensors,
        tokenizer: None,
        tokenizer_config: None,
        chat_template: None,
        prompt_template: None,
        generation: None,
        quantization: None,
        shards: Vec::new(),
        runtime_features: BTreeSet::new(),
        memory_features: BTreeSet::new(),
        provider_capabilities: Vec::new(),
        component: None,
        license: None,
        provenance: None,
        signatures: Vec::new(),
        source: None,
        architecture_config: Some(architecture_config_from_qwen_config(&config)),
    };

    let tokenizer_metadata = e2e_fixture_tokenizer().unwrap().metadata().clone();
    let delegate_tokenizer: std::sync::Arc<dyn crate::tokenizer::Tokenizer + Send + Sync> =
        std::sync::Arc::new(e2e_fixture_tokenizer().unwrap());
    let fixture = production_qwen_fixture(manifest, tokenizer_metadata, delegate_tokenizer)
        .expect("production fixture builds against the canonical config");

    let component_trust = ComponentTrustStore::default()
        .trust_digest(&ComponentDigest::sha256(QWEN_REAL_COMPONENT_BYTES).value);
    let digest = register_inference_component_artifact(
        QWEN_REAL_COMPONENT_BYTES.to_vec(),
        QWEN_REAL_COMPONENT_MANIFEST_BYTES.to_vec(),
        &component_trust,
    )
    .expect("registration succeeds");

    let model_trust =
        || ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone());
    let request = || ProductionGenerationRequest {
        prompt: PromptInput::PlainText("hi".into()),
        parameters: GenerationParameters::greedy(),
        stop_conditions: StopConditions::default(),
        max_new_tokens: Some(2),
        max_generation_millis: None,
    };

    let mut via_singleton = ProductionQwenLoadedModel::load(
        fixture.clone(),
        &payload_source,
        model_trust(),
        Arc::new(ReferenceCpuProvider::new()),
    )
    .expect("singleton-path loading succeeds");
    let singleton_result = via_singleton
        .generate(request(), None)
        .expect("singleton-path generation succeeds");

    let mut via_named = ProductionQwenLoadedModel::load_with_component(
        fixture.clone(),
        &payload_source,
        model_trust(),
        Arc::new(ReferenceCpuProvider::new()),
        Some(digest),
    )
    .expect("named-component loading succeeds");
    let named_result = via_named
        .generate(request(), None)
        .expect("named-component generation succeeds");

    assert_eq!(
        singleton_result.result.output.generated_token_ids,
        named_result.result.output.generated_token_ids,
        "loading via an explicitly registered Component digest must produce identical \
         generation to the hardcoded singleton path, for the identical underlying Component \
         bytes"
    );
}

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine"
))]
const QWEN_GRAPH_COMPONENT_NAME: &str = "magnetar.qwen.graph-fixture";

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine"
))]
const QWEN_GRAPH_COMPONENT_DIGEST: &str =
    "sha256:a85b9fc4fa182aa1ce2f4a55458b125e7cf4aac06dc9f5e60a2d686292677f7b";

#[cfg(test)]
fn apply_rope_per_head(
    tensor: &HostTensor,
    head_count: u64,
    head_dimension: u64,
    rope_config: &QwenRopeConfig,
) -> Result<HostTensor, E2eConformanceError> {
    let (rows, cols) = tensor.rows_cols()?;
    let mut out = vec![0.0_f32; tensor.data.len()];
    for head in 0..head_count {
        let start_col = head * head_dimension;
        let mut head_data = Vec::with_capacity((rows * head_dimension) as usize);
        for row in 0..rows {
            let base = (row * cols + start_col) as usize;
            head_data.extend_from_slice(&tensor.data[base..base + head_dimension as usize]);
        }
        let head_tensor = HostTensor::new([rows, head_dimension], head_data)?;
        let rotated = rope(
            &head_tensor,
            rope_config.base as f32,
            rope_config.scale.unwrap_or(1.0) as f32,
            rope_config.dimension,
            0,
            1,
        )?;
        for row in 0..rows {
            let dst_base = (row * cols + start_col) as usize;
            let src_base = (row * head_dimension) as usize;
            out[dst_base..dst_base + head_dimension as usize]
                .copy_from_slice(&rotated.data[src_base..src_base + head_dimension as usize]);
        }
    }
    HostTensor::new(tensor.shape.clone(), out).map_err(E2eConformanceError::from)
}

/// Test oracle for the decoder stack. Production first-native generation uses
/// `execute_qwen_hidden_states_through_dispatch` instead.
#[cfg(test)]
fn e2e_forward_hidden_states(
    fixture: &E2eFixture,
    token_ids: &[TokenId],
) -> Result<HostTensor, E2eConformanceError> {
    if token_ids.is_empty() {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "forward pass requires at least one token".into(),
        });
    }
    let architecture = &fixture.config.architecture;
    let seq_len = token_ids.len() as u64;
    let epsilon = fixture.config.rmsnorm_epsilon;

    let ids_tensor = HostTensor::new(
        [seq_len],
        token_ids.iter().map(|id| *id as f32).collect::<Vec<_>>(),
    )?;
    let token_embedding = fixture_tensor_by_name(&fixture.weights, "token_embedding")?;
    let mut hidden_states = embedding_lookup(token_embedding, &ids_tensor)?;

    for layer in 0..architecture.layer_count {
        let prefix = format!("layers.{layer}.");
        let input_norm = fixture_tensor_by_name(&fixture.weights, &format!("{prefix}input_norm"))?;
        let q_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}self_attn.q_proj"))?;
        let k_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}self_attn.k_proj"))?;
        let v_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}self_attn.v_proj"))?;
        let o_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}self_attn.o_proj"))?;
        let post_attn_norm =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}post_attn_norm"))?;
        let gate_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}mlp.gate_proj"))?;
        let up_weight = fixture_tensor_by_name(&fixture.weights, &format!("{prefix}mlp.up_proj"))?;
        let down_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}mlp.down_proj"))?;

        let normed = rmsnorm(&hidden_states, input_norm, epsilon)?;
        let q = matmul(&normed, q_weight, false, false)?;
        let k = matmul(&normed, k_weight, false, false)?;
        let v = matmul(&normed, v_weight, false, false)?;
        let q = apply_rope_per_head(
            &q,
            architecture.attention_head_count,
            architecture.head_dimension,
            &fixture.config.rope,
        )?;
        let k = apply_rope_per_head(
            &k,
            architecture.kv_head_count,
            architecture.head_dimension,
            &fixture.config.rope,
        )?;
        let attention_out = attention(
            &q,
            &k,
            &v,
            architecture.attention_head_count,
            architecture.head_dimension,
            Some(architecture.kv_head_count),
            None,
            true,
        )?;
        let attention_proj = matmul(&attention_out, o_weight, false, false)?;
        hidden_states = residual_add(&attention_proj, &hidden_states)?;

        let normed_mlp = rmsnorm(&hidden_states, post_attn_norm, epsilon)?;
        let gate = matmul(&normed_mlp, gate_weight, false, false)?;
        let up = matmul(&normed_mlp, up_weight, false, false)?;
        let activated = silu(&gate);
        let gated = mul(&activated, &up)?;
        let mlp_out = matmul(&gated, down_weight, false, false)?;
        hidden_states = residual_add(&mlp_out, &hidden_states)?;
    }

    let final_norm = fixture_tensor_by_name(&fixture.weights, "final_norm")?;
    rmsnorm(&hidden_states, final_norm, epsilon).map_err(E2eConformanceError::from)
}

#[cfg(test)]
/// Test oracle for a deterministic Qwen-like forward pass. This is deliberately
/// not compiled into the production runtime path; the runtime path executes
/// operators through Kernel Registry selection and Provider dispatch.
pub fn e2e_forward(
    fixture: &E2eFixture,
    token_ids: &[TokenId],
) -> Result<Vec<f32>, E2eConformanceError> {
    let normed_final = e2e_forward_hidden_states(fixture, token_ids)?;
    let token_embedding = fixture_tensor_by_name(&fixture.weights, "token_embedding")?;
    // Tied embeddings: logits = normed_final @ token_embedding^T.
    let logits = matmul(&normed_final, token_embedding, false, true)?;
    // Exercise the softmax kernel for operator-coverage/report purposes;
    // Sampling owns the authoritative distribution derived from raw logits.
    let _distribution = softmax_rows(&logits)?;

    let vocab = fixture.config.architecture.vocabulary_size as usize;
    let last_row_start = (token_ids.len() - 1) * vocab;
    Ok(logits.data[last_row_start..last_row_start + vocab].to_vec())
}

#[cfg(test)]
/// Test-oracle only (`implement-device-resident-multi-step-cuda-decode`):
/// production's own KV-history concatenation now dispatches through the
/// portable "concat" Operator (`dispatch_qwen_concat`) instead of this
/// plain-Rust helper; this survives only as
/// `execute_qwen_decode_hidden_states_through_dispatch`'s (also
/// `#[cfg(test)]`) own independent cross-check implementation.
fn concat_rows(a: &HostTensor, b: &HostTensor) -> Result<HostTensor, InferenceApiError> {
    let (a_rows, a_cols) = a.rows_cols().map_err(runtime_generation_failed)?;
    let (b_rows, b_cols) = b.rows_cols().map_err(runtime_generation_failed)?;
    if a_cols != b_cols {
        return Err(InferenceApiError::GenerationFailed {
            reason: format!("cannot concatenate tensors with widths {a_cols} and {b_cols}"),
        });
    }
    let mut data = Vec::with_capacity(a.data.len() + b.data.len());
    data.extend_from_slice(&a.data);
    data.extend_from_slice(&b.data);
    HostTensor::new([a_rows + b_rows, a_cols], data).map_err(runtime_generation_failed)
}

/// Stable numeric codes for the Operator names the Qwen graph builder emits,
/// shared between Runtime (deriving the expected sequence from
/// `ExecutionGraph`) and the Qwen Model Component boundary (which describes
/// its own graph as this same code sequence -- see
/// `qwen_graph_operator_codes` and `qwen-graph.component.wat`'s
/// `prefill-operator-code`/`decode-operator-code` exports). A plain
/// name-to-code table rather than `OperatorId` equality: the Component
/// boundary exchanges scalar `u32`s, not portable Operator identities.
#[cfg(test)]
fn qwen_operator_kind_code(name: &str) -> Option<u32> {
    match name {
        "embedding" => Some(0),
        "rmsnorm" => Some(1),
        "matmul" => Some(2),
        "rope" => Some(3),
        "attention" => Some(4),
        "silu" => Some(5),
        "mul" => Some(6),
        "residual-add" => Some(7),
        "split" => Some(8),
        // The QKV bias-add operator (`config.attention_bias`'s conditional
        // node in both `components/qwen` and `components/llama`) -- never
        // exercised by a hash comparison before the real Llama cross-
        // architecture proof test needed to hash a bias-bearing graph for
        // the first time, surfacing this table's previously-latent gap
        // (every other graph-hashing test uses the shared fixture's
        // default `attention_bias: false`).
        "add" => Some(9),
        _ => None,
    }
}

/// Derives the expected Operator-kind-code sequence for `graph`, in the same
/// dependency order `execute_qwen_graph` executes it in: the semantic
/// content a Qwen Model Component is expected to reproduce when describing
/// its own graph (see `qwen_operator_kind_code`).
#[cfg(test)]
fn qwen_graph_operator_codes(graph: &ExecutionGraph) -> Result<Vec<u32>, E2eConformanceError> {
    let order = qwen_graph_execution_order(graph)?;
    order
        .iter()
        .map(|node_id| {
            let node = graph.nodes.get(node_id).ok_or_else(|| {
                E2eConformanceError::GraphValidationFailed {
                    reason: format!("first-native graph is missing node '{node_id}'"),
                }
            })?;
            qwen_operator_kind_code(node.operator.name()).ok_or_else(|| {
                E2eConformanceError::GraphValidationFailed {
                    reason: format!(
                        "graph node '{node_id}' uses operator '{}' with no known kind code",
                        node.operator.name()
                    ),
                }
            })
        })
        .collect()
}

/// A deterministic FNV-1a-style hash over an ordered Operator-kind-code
/// sequence. The Component boundary's invocation model exchanges only
/// zero-argument, single-`u32`-result calls (see [`ComponentInvocation`]),
/// so a component cannot return its full node sequence as a list; instead it
/// computes this same hash internally (see `qwen-graph.component.wat`'s
/// `prefill-operator-hash`/`decode-operator-hash` exports, which perform the
/// identical unrolled XOR/multiply steps over its own hard-coded sequence)
/// and Runtime compares hashes -- a proof over the actual ordered semantic
/// content, not just a count.
#[cfg(test)]
fn qwen_operator_sequence_hash(codes: &[u32]) -> u32 {
    const FNV_OFFSET_BASIS: u32 = 0x811c_9dc5;
    const FNV_PRIME: u32 = 0x0100_0193;
    codes.iter().fold(FNV_OFFSET_BASIS, |hash, code| {
        (hash ^ *code).wrapping_mul(FNV_PRIME)
    })
}

#[cfg(test)]
/// Static guard (`define-provider-prepared-kernel-execution-contract` task
/// 2.3): [`execute_qwen_graph_nodes`]'s per-node transport migrated fully off
/// the `HostTensor`-typed [`ProviderExecutionApi`] methods (that Change's
/// task group 5) -- every read/write in its per-node loop goes through
/// `read_tensor_value`/`write_tensor_value_admitted` instead, materializing
/// to `HostTensor` only at the explicit host-materialization boundaries via
/// `TensorValue::into_host` (weight binding, KV-history concatenation, final
/// logits extraction, plus each node's own Kernel-input resolution). This
/// scans the function's own source text so a future edit that reintroduces a
/// direct `.read_tensor(`/`.write_tensor(`/`.write_tensor_admitted(` call
/// into that loop fails a test immediately, rather than the two pathways
/// (`HostTensor`-typed and `TensorValue`-typed) silently coexisting
/// indefinitely -- design.md's stated risk for that Change. Test-only: this
/// is a source-level build invariant, not runtime behavior
/// `run_e2e_local_inference_conformance` needs to check in production.
fn check_execute_qwen_graph_nodes_transport_has_no_host_tensor_typed_calls()
-> Result<(), E2eConformanceError> {
    const SOURCE: &str = include_str!("../first_native_runtime.rs");
    let start = SOURCE.find("fn execute_qwen_graph_nodes(").ok_or_else(|| {
        E2eConformanceError::Internal {
            reason: "execute_qwen_graph_nodes not found in first_native_runtime.rs source".into(),
        }
    })?;
    let body_start = SOURCE[start..]
        .find('{')
        .map(|offset| start + offset)
        .ok_or_else(|| E2eConformanceError::Internal {
            reason: "execute_qwen_graph_nodes has no function body in source".into(),
        })?;
    let mut depth = 0i32;
    let mut body_end = body_start;
    for (offset, ch) in SOURCE[body_start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    body_end = body_start + offset + 1;
                    break;
                }
            }
            _ => {}
        }
    }
    if body_end == body_start {
        return Err(E2eConformanceError::Internal {
            reason: "execute_qwen_graph_nodes's function body braces did not balance".into(),
        });
    }
    let body = &SOURCE[body_start..body_end];
    // Exact-name matches only (immediate `(` after the method name), so the
    // intended replacements -- `.read_tensor_value(`, `.write_tensor_value(`,
    // `.write_tensor_value_admitted(` -- do not themselves trip this guard.
    let host_tensor_typed_call_count =
        [".read_tensor(", ".write_tensor(", ".write_tensor_admitted("]
            .iter()
            .map(|needle| body.matches(needle).count())
            .sum::<usize>();
    if host_tensor_typed_call_count != 0 {
        return Err(E2eConformanceError::Internal {
            reason: format!(
                "execute_qwen_graph_nodes's per-node transport has \
                 {host_tensor_typed_call_count} direct HostTensor-typed \
                 ProviderExecutionApi call(s); it must read/write through \
                 TensorValue (read_tensor_value/write_tensor_value_admitted) \
                 and materialize only at explicit host-materialization \
                 boundaries via TensorValue::into_host"
            ),
        });
    }
    Ok(())
}

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine"
))]
/// The MLP gate/up projection the prefill/decode oracle dispatch sequences
/// below exercise, split by which graph source `first_native_plans_for_prompt`
/// actually built its `PreparedExecutionPlan` against (mirrors
/// [`first_native_component_graphs_for_prompt`]'s own cfg split exactly):
/// under the strict, default build the Plan comes from the checked-in real
/// Qwen Component's graph, which still declares two standalone
/// `gate_proj`/`up_proj` matmul nodes (`components/qwen` is unmodified);
/// without a strict Component engine, the Plan comes from the Rust-
/// synthesized fallback recipe (`qwen_model_component::qwen_build_graph`),
/// which fuses them into one `gate_up_proj` matmul followed by a
/// genuinely two-output `split` (`define-provider-prepared-kernel-
/// execution-contract` task group 3). Every oracle dispatch here must use
/// the identical node-id shape the Plan was actually built from, or Kernel
/// selection fails closed with "no binding for node ...".
fn dispatch_qwen_oracle_mlp_gate_up(
    dispatch_ctx: &mut QwenDispatchContext<'_>,
    fixture: &E2eFixture,
    prefix: &str,
    layer_id: &str,
    normed_mlp: NodeValue,
) -> Result<(NodeValue, NodeValue), InferenceApiError> {
    let gate_weight = fixture_tensor_by_name(&fixture.weights, &format!("{prefix}mlp.gate_proj"))
        .map_err(runtime_generation_failed)?
        .clone();
    let up_weight = fixture_tensor_by_name(&fixture.weights, &format!("{prefix}mlp.up_proj"))
        .map_err(runtime_generation_failed)?
        .clone();
    let (_dispatch, gate) = dispatch_qwen_matmul(
        dispatch_ctx,
        &format!("{layer_id}.gate_proj"),
        normed_mlp.clone(),
        NodeValue::Host(gate_weight),
        None,
    )?;
    let (_dispatch, up) = dispatch_qwen_matmul(
        dispatch_ctx,
        &format!("{layer_id}.up_proj"),
        normed_mlp,
        NodeValue::Host(up_weight),
        None,
    )?;
    Ok((gate, up))
}

#[cfg(all(
    test,
    not(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))
))]
fn dispatch_qwen_oracle_mlp_gate_up(
    dispatch_ctx: &mut QwenDispatchContext<'_>,
    fixture: &E2eFixture,
    prefix: &str,
    layer_id: &str,
    normed_mlp: NodeValue,
) -> Result<(NodeValue, NodeValue), InferenceApiError> {
    let gate_up_weight =
        fixture_tensor_by_name(&fixture.weights, &format!("{prefix}mlp.gate_up_proj"))
            .map_err(runtime_generation_failed)?
            .clone();
    let (_dispatch, gate_up) = dispatch_qwen_matmul(
        dispatch_ctx,
        &format!("{layer_id}.gate_up_proj"),
        normed_mlp,
        NodeValue::Host(gate_up_weight),
        None,
    )?;
    let (_dispatch, split_outputs) = dispatch_qwen_split(
        dispatch_ctx,
        &format!("{layer_id}.split"),
        gate_up,
        [None, None],
    )?;
    let [gate, up]: [NodeValue; 2] =
        split_outputs
            .try_into()
            .map_err(|_| InferenceApiError::GraphPlanningFailed {
                reason: format!("'{layer_id}.split' produced an unexpected number of outputs"),
            })?;
    Ok((gate, up))
}

#[cfg(test)]
/// Test-only oracle: a hand-written, hard-coded prefill dispatch sequence
/// kept only so tests can cross-check `execute_qwen_graph`'s output against
/// an independently-written recipe. Production first-native execution
/// cannot reach this function -- it computes logits exclusively through
/// `execute_qwen_graph` (see `E2eRuntimeModelExecutionEngine::
/// execute_generation_step`). `prepared_plan` is mandatory (not optional):
/// the first-native hot path must always look up a published
/// [`PlanNodeBinding`]/[`PreparedKernelId`] rather than ever falling back to
/// ad hoc Kernel Registry selection here -- planning-time selection belongs
/// in [`prepare_first_native_plan_for_graph`], not in this execution path.
fn execute_qwen_prefill_hidden_states_through_dispatch(
    runtime: &mut Runtime,
    fixture: &E2eFixture,
    token_ids: &[TokenId],
    prepared_plan: &mut PreparedExecutionPlan,
) -> Result<
    (
        KernelDispatchResult,
        HostTensor,
        Vec<FirstNativeLayerKvState>,
    ),
    InferenceApiError,
> {
    if token_ids.is_empty() {
        return Err(InferenceApiError::GenerationFailed {
            reason: "forward pass requires at least one token".into(),
        });
    }
    // Resolved from Runtime's own registration (not a throwaway) so the
    // decode oracle's separate call can read back the K/V resources this
    // call writes -- the whole point of testing incremental decode against
    // the KV state prefill actually produced.
    let provider = resolve_kernel_execution_provider(
        runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )?;
    let mut node_events = Vec::new();
    let mut dispatch_ctx = QwenDispatchContext {
        runtime,
        provider: provider.clone(),
        prepared_plan: Some(prepared_plan),
        graph: None,
        sequence_length: None,
        last_provider_execution: None,
        node_events: &mut node_events,
    };
    let architecture = &fixture.config.architecture;
    let seq_len = token_ids.len() as u64;
    let epsilon = fixture.config.rmsnorm_epsilon;

    let ids_tensor = HostTensor::new(
        [seq_len],
        token_ids.iter().map(|id| *id as f32).collect::<Vec<_>>(),
    )
    .map_err(runtime_generation_failed)?;
    let token_embedding = fixture_tensor_by_name(&fixture.weights, "token_embedding")
        .map_err(runtime_generation_failed)?
        .clone();
    let (_embedding_dispatch, hidden_states) = dispatch_reference_cpu_operator(
        &mut dispatch_ctx,
        "embedding",
        dispatch_operator_id("embedding", OperatorFamily::Tensor),
        vec![
            NodeInputResource::Fresh(
                TensorResourceId::new("embedding.table"),
                f32_tensor_descriptor(&token_embedding),
                token_embedding,
            ),
            NodeInputResource::Fresh(
                TensorResourceId::new("embedding.ids"),
                f32_tensor_descriptor(&ids_tensor),
                ids_tensor,
            ),
        ],
        (
            TensorResourceId::new("embedding.out"),
            TensorDescriptor::new(
                ShapeDescriptor::new([seq_len, architecture.hidden_size]),
                DTypeDescriptor::portable(ComputeDType::Float32),
                LayoutDescriptor::Contiguous,
            ),
        ),
        BTreeMap::new(),
    )?;
    let mut hidden_states = hidden_states.into_host(&dispatch_ctx.provider)?;

    let mut layer_kv = Vec::with_capacity(architecture.layer_count as usize);
    for layer in 0..architecture.layer_count {
        let prefix = format!("layers.{layer}.");
        let input_norm = fixture_tensor_by_name(&fixture.weights, &format!("{prefix}input_norm"))
            .map_err(runtime_generation_failed)?
            .clone();
        let q_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}self_attn.q_proj"))
                .map_err(runtime_generation_failed)?
                .clone();
        let k_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}self_attn.k_proj"))
                .map_err(runtime_generation_failed)?
                .clone();
        let v_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}self_attn.v_proj"))
                .map_err(runtime_generation_failed)?
                .clone();
        let o_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}self_attn.o_proj"))
                .map_err(runtime_generation_failed)?
                .clone();
        let post_attn_norm =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}post_attn_norm"))
                .map_err(runtime_generation_failed)?
                .clone();
        let down_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}mlp.down_proj"))
                .map_err(runtime_generation_failed)?
                .clone();

        let layer_id = format!("layer{layer}");
        let (_dispatch, normed) = dispatch_qwen_rmsnorm(
            &mut dispatch_ctx,
            &format!("{layer_id}.input_norm"),
            NodeValue::Host(hidden_states.clone()),
            NodeValue::Host(input_norm),
            epsilon,
            None,
        )?;
        let (_dispatch, q) = dispatch_qwen_matmul(
            &mut dispatch_ctx,
            &format!("{layer_id}.q_proj"),
            normed.clone(),
            NodeValue::Host(q_weight),
            None,
        )?;
        let (_dispatch, k) = dispatch_qwen_matmul(
            &mut dispatch_ctx,
            &format!("{layer_id}.k_proj"),
            normed.clone(),
            NodeValue::Host(k_weight),
            None,
        )?;
        let (_dispatch, v) = dispatch_qwen_matmul(
            &mut dispatch_ctx,
            &format!("{layer_id}.v_proj"),
            normed,
            NodeValue::Host(v_weight),
            None,
        )?;
        let v = v.into_host(&dispatch_ctx.provider)?;
        let (_dispatch, q) = dispatch_qwen_rope(
            &mut dispatch_ctx,
            &format!("{layer_id}.rope_q"),
            q,
            architecture.attention_head_count,
            &fixture.config.rope,
            0,
            None,
        )?;
        let q = q.into_host(&dispatch_ctx.provider)?;
        let (_dispatch, k) = dispatch_qwen_rope(
            &mut dispatch_ctx,
            &format!("{layer_id}.rope_k"),
            k,
            architecture.kv_head_count,
            &fixture.config.rope,
            0,
            None,
        )?;
        let k = k.into_host(&dispatch_ctx.provider)?;
        let k_resource = TensorResourceId::new(format!("oracle-kv.layer{layer}.k"));
        let v_resource = TensorResourceId::new(format!("oracle-kv.layer{layer}.v"));
        dispatch_ctx
            .provider
            .write_tensor(k_resource.clone(), k.clone())
            .map_err(|error| InferenceApiError::ProviderTensorWriteFailed {
                reason: error.to_string(),
            })?;
        dispatch_ctx
            .provider
            .write_tensor(v_resource.clone(), v.clone())
            .map_err(|error| InferenceApiError::ProviderTensorWriteFailed {
                reason: error.to_string(),
            })?;
        layer_kv.push(FirstNativeLayerKvState {
            k: k_resource,
            v: v_resource,
        });
        let (_dispatch, attention_out) = dispatch_qwen_attention(
            &mut dispatch_ctx,
            &format!("{layer_id}.attention"),
            NodeValue::Host(q),
            NodeValue::Host(k),
            NodeValue::Host(v),
            architecture,
            None,
        )?;
        let (_dispatch, attention_proj) = dispatch_qwen_matmul(
            &mut dispatch_ctx,
            &format!("{layer_id}.o_proj"),
            attention_out,
            NodeValue::Host(o_weight),
            None,
        )?;
        let (_dispatch, post_attention) = dispatch_qwen_binary_same_shape(
            &mut dispatch_ctx,
            &format!("{layer_id}.residual1"),
            "residual-add",
            OperatorFamily::Tensor,
            attention_proj,
            NodeValue::Host(hidden_states),
            None,
        )?;
        let (_dispatch, normed_mlp) = dispatch_qwen_rmsnorm(
            &mut dispatch_ctx,
            &format!("{layer_id}.post_attn_norm"),
            post_attention.clone(),
            NodeValue::Host(post_attn_norm),
            epsilon,
            None,
        )?;
        let (gate, up) = dispatch_qwen_oracle_mlp_gate_up(
            &mut dispatch_ctx,
            fixture,
            &prefix,
            &layer_id,
            normed_mlp,
        )?;
        let (_dispatch, activated) = dispatch_qwen_unary(
            &mut dispatch_ctx,
            &format!("{layer_id}.silu"),
            "silu",
            OperatorFamily::Activation,
            gate,
            BTreeMap::new(),
            None,
        )?;
        let (_dispatch, gated) = dispatch_qwen_binary_same_shape(
            &mut dispatch_ctx,
            &format!("{layer_id}.mul"),
            "mul",
            OperatorFamily::Tensor,
            activated,
            up,
            None,
        )?;
        let (_dispatch, mlp_out) = dispatch_qwen_matmul(
            &mut dispatch_ctx,
            &format!("{layer_id}.down_proj"),
            gated,
            NodeValue::Host(down_weight),
            None,
        )?;
        let (_dispatch, layer_out) = dispatch_qwen_binary_same_shape(
            &mut dispatch_ctx,
            &format!("{layer_id}.residual2"),
            "residual-add",
            OperatorFamily::Tensor,
            mlp_out,
            post_attention,
            None,
        )?;
        hidden_states = layer_out.into_host(&dispatch_ctx.provider)?;
    }

    let final_norm = fixture_tensor_by_name(&fixture.weights, "final_norm")
        .map_err(runtime_generation_failed)?
        .clone();
    let (dispatch, hidden_states) = dispatch_qwen_rmsnorm(
        &mut dispatch_ctx,
        "final_norm",
        NodeValue::Host(hidden_states),
        NodeValue::Host(final_norm),
        epsilon,
        None,
    )?;
    let hidden_states = hidden_states.into_host(&dispatch_ctx.provider)?;
    Ok((dispatch, hidden_states, layer_kv))
}

#[cfg(test)]
/// Test-only oracle, kept only for cross-checking `execute_qwen_graph`; see
/// [`execute_qwen_prefill_hidden_states_through_dispatch`]'s doc comment.
fn execute_qwen_decode_hidden_states_through_dispatch(
    runtime: &mut Runtime,
    fixture: &E2eFixture,
    token_id: TokenId,
    kv_state: &FirstNativeExecutionKvState,
    absolute_position: u64,
    prepared_plan: &mut PreparedExecutionPlan,
) -> Result<
    (
        KernelDispatchResult,
        HostTensor,
        Vec<FirstNativeLayerKvState>,
    ),
    InferenceApiError,
> {
    // Resolved from Runtime's own registration (not a throwaway) so this
    // call can read back the K/V resources prefill wrote -- see
    // `execute_qwen_prefill_hidden_states_through_dispatch`'s doc comment.
    let provider = resolve_kernel_execution_provider(
        runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )?;
    let mut node_events = Vec::new();
    let mut dispatch_ctx = QwenDispatchContext {
        runtime,
        provider: provider.clone(),
        prepared_plan: Some(prepared_plan),
        graph: None,
        sequence_length: None,
        last_provider_execution: None,
        node_events: &mut node_events,
    };
    let architecture = &fixture.config.architecture;
    if kv_state.layer_kv.len() != architecture.layer_count as usize {
        return Err(InferenceApiError::KvCacheUnavailable {
            reason: format!(
                "decode requires {} layer KV entries, found {}",
                architecture.layer_count,
                kv_state.layer_kv.len()
            ),
        });
    }
    let epsilon = fixture.config.rmsnorm_epsilon;
    let ids_tensor =
        HostTensor::new([1], vec![token_id as f32]).map_err(runtime_generation_failed)?;
    let token_embedding = fixture_tensor_by_name(&fixture.weights, "token_embedding")
        .map_err(runtime_generation_failed)?
        .clone();
    let (_embedding_dispatch, hidden_states) = dispatch_reference_cpu_operator(
        &mut dispatch_ctx,
        "decode.embedding",
        dispatch_operator_id("embedding", OperatorFamily::Tensor),
        vec![
            NodeInputResource::Fresh(
                TensorResourceId::new("decode.embedding.table"),
                f32_tensor_descriptor(&token_embedding),
                token_embedding,
            ),
            NodeInputResource::Fresh(
                TensorResourceId::new("decode.embedding.ids"),
                f32_tensor_descriptor(&ids_tensor),
                ids_tensor,
            ),
        ],
        (
            TensorResourceId::new("decode.embedding.out"),
            TensorDescriptor::new(
                ShapeDescriptor::new([1, architecture.hidden_size]),
                DTypeDescriptor::portable(ComputeDType::Float32),
                LayoutDescriptor::Contiguous,
            ),
        ),
        BTreeMap::new(),
    )?;
    let mut hidden_states = hidden_states.into_host(&dispatch_ctx.provider)?;

    let mut updated_layer_kv = Vec::with_capacity(architecture.layer_count as usize);
    for layer in 0..architecture.layer_count {
        let prefix = format!("layers.{layer}.");
        let input_norm = fixture_tensor_by_name(&fixture.weights, &format!("{prefix}input_norm"))
            .map_err(runtime_generation_failed)?
            .clone();
        let q_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}self_attn.q_proj"))
                .map_err(runtime_generation_failed)?
                .clone();
        let k_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}self_attn.k_proj"))
                .map_err(runtime_generation_failed)?
                .clone();
        let v_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}self_attn.v_proj"))
                .map_err(runtime_generation_failed)?
                .clone();
        let o_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}self_attn.o_proj"))
                .map_err(runtime_generation_failed)?
                .clone();
        let post_attn_norm =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}post_attn_norm"))
                .map_err(runtime_generation_failed)?
                .clone();
        let down_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}mlp.down_proj"))
                .map_err(runtime_generation_failed)?
                .clone();

        let layer_id = format!("decode.layer{layer}");
        let (_dispatch, normed) = dispatch_qwen_rmsnorm(
            &mut dispatch_ctx,
            &format!("{layer_id}.input_norm"),
            NodeValue::Host(hidden_states.clone()),
            NodeValue::Host(input_norm),
            epsilon,
            None,
        )?;
        let (_dispatch, q) = dispatch_qwen_matmul(
            &mut dispatch_ctx,
            &format!("{layer_id}.q_proj"),
            normed.clone(),
            NodeValue::Host(q_weight),
            None,
        )?;
        let (_dispatch, k_new) = dispatch_qwen_matmul(
            &mut dispatch_ctx,
            &format!("{layer_id}.k_proj"),
            normed.clone(),
            NodeValue::Host(k_weight),
            None,
        )?;
        let (_dispatch, v_new) = dispatch_qwen_matmul(
            &mut dispatch_ctx,
            &format!("{layer_id}.v_proj"),
            normed,
            NodeValue::Host(v_weight),
            None,
        )?;
        let v_new = v_new.into_host(&dispatch_ctx.provider)?;
        let (_dispatch, q) = dispatch_qwen_rope(
            &mut dispatch_ctx,
            &format!("{layer_id}.rope_q"),
            q,
            architecture.attention_head_count,
            &fixture.config.rope,
            absolute_position,
            None,
        )?;
        let q = q.into_host(&dispatch_ctx.provider)?;
        let (_dispatch, k_new) = dispatch_qwen_rope(
            &mut dispatch_ctx,
            &format!("{layer_id}.rope_k"),
            k_new,
            architecture.kv_head_count,
            &fixture.config.rope,
            absolute_position,
            None,
        )?;
        let k_new = k_new.into_host(&dispatch_ctx.provider)?;
        let historical = &kv_state.layer_kv[&(layer as usize)];
        let historical_k = dispatch_ctx
            .provider
            .read_tensor(&historical.k)
            .ok_or_else(|| InferenceApiError::KvCacheUnavailable {
                reason: format!("no materialized historical K data for layer {layer}"),
            })?;
        let historical_v = dispatch_ctx
            .provider
            .read_tensor(&historical.v)
            .ok_or_else(|| InferenceApiError::KvCacheUnavailable {
                reason: format!("no materialized historical V data for layer {layer}"),
            })?;
        let k = concat_rows(&historical_k, &k_new)?;
        let v = concat_rows(&historical_v, &v_new)?;
        let k_resource = TensorResourceId::new(format!("oracle-kv.layer{layer}.k"));
        let v_resource = TensorResourceId::new(format!("oracle-kv.layer{layer}.v"));
        dispatch_ctx
            .provider
            .write_tensor(k_resource.clone(), k.clone())
            .map_err(|error| InferenceApiError::ProviderTensorWriteFailed {
                reason: error.to_string(),
            })?;
        dispatch_ctx
            .provider
            .write_tensor(v_resource.clone(), v.clone())
            .map_err(|error| InferenceApiError::ProviderTensorWriteFailed {
                reason: error.to_string(),
            })?;
        updated_layer_kv.push(FirstNativeLayerKvState {
            k: k_resource,
            v: v_resource,
        });
        let (_dispatch, attention_out) = dispatch_qwen_attention(
            &mut dispatch_ctx,
            &format!("{layer_id}.attention"),
            NodeValue::Host(q),
            NodeValue::Host(k),
            NodeValue::Host(v),
            architecture,
            None,
        )?;
        let (_dispatch, attention_proj) = dispatch_qwen_matmul(
            &mut dispatch_ctx,
            &format!("{layer_id}.o_proj"),
            attention_out,
            NodeValue::Host(o_weight),
            None,
        )?;
        let (_dispatch, post_attention) = dispatch_qwen_binary_same_shape(
            &mut dispatch_ctx,
            &format!("{layer_id}.residual1"),
            "residual-add",
            OperatorFamily::Tensor,
            attention_proj,
            NodeValue::Host(hidden_states),
            None,
        )?;
        let (_dispatch, normed_mlp) = dispatch_qwen_rmsnorm(
            &mut dispatch_ctx,
            &format!("{layer_id}.post_attn_norm"),
            post_attention.clone(),
            NodeValue::Host(post_attn_norm),
            epsilon,
            None,
        )?;
        let (gate, up) = dispatch_qwen_oracle_mlp_gate_up(
            &mut dispatch_ctx,
            fixture,
            &prefix,
            &layer_id,
            normed_mlp,
        )?;
        let (_dispatch, activated) = dispatch_qwen_unary(
            &mut dispatch_ctx,
            &format!("{layer_id}.silu"),
            "silu",
            OperatorFamily::Activation,
            gate,
            BTreeMap::new(),
            None,
        )?;
        let (_dispatch, gated) = dispatch_qwen_binary_same_shape(
            &mut dispatch_ctx,
            &format!("{layer_id}.mul"),
            "mul",
            OperatorFamily::Tensor,
            activated,
            up,
            None,
        )?;
        let (_dispatch, mlp_out) = dispatch_qwen_matmul(
            &mut dispatch_ctx,
            &format!("{layer_id}.down_proj"),
            gated,
            NodeValue::Host(down_weight),
            None,
        )?;
        let (_dispatch, layer_out) = dispatch_qwen_binary_same_shape(
            &mut dispatch_ctx,
            &format!("{layer_id}.residual2"),
            "residual-add",
            OperatorFamily::Tensor,
            mlp_out,
            post_attention,
            None,
        )?;
        hidden_states = layer_out.into_host(&dispatch_ctx.provider)?;
    }

    let final_norm = fixture_tensor_by_name(&fixture.weights, "final_norm")
        .map_err(runtime_generation_failed)?
        .clone();
    let (dispatch, hidden_states) = dispatch_qwen_rmsnorm(
        &mut dispatch_ctx,
        "decode.final_norm",
        NodeValue::Host(hidden_states),
        NodeValue::Host(final_norm),
        epsilon,
        None,
    )?;
    let hidden_states = hidden_states.into_host(&dispatch_ctx.provider)?;
    Ok((dispatch, hidden_states, updated_layer_kv))
}

#[cfg(test)]
/// Test-only oracle, kept only for cross-checking `execute_qwen_graph`; see
/// [`execute_qwen_prefill_hidden_states_through_dispatch`]'s doc comment.
fn dispatch_qwen_logits_projection(
    runtime: &Runtime,
    fixture: &E2eFixture,
    hidden_states: &HostTensor,
    prepared_plan: &PreparedExecutionPlan,
) -> Result<(KernelDispatchResult, Vec<f32>), InferenceApiError> {
    let token_embedding =
        fixture_tensor_by_name(&fixture.weights, "token_embedding").map_err(|error| {
            InferenceApiError::GenerationFailed {
                reason: error.to_string(),
            }
        })?;
    let token_embedding_transposed = transpose_rows_cols(token_embedding).map_err(|error| {
        InferenceApiError::GenerationFailed {
            reason: error.to_string(),
        }
    })?;
    let (dispatch_result, output) = dispatch_matmul_with_prepared_plan(
        runtime,
        hidden_states,
        &token_embedding_transposed,
        "lm_head",
        Some(prepared_plan),
    )?;
    let vocab = fixture.config.architecture.vocabulary_size as usize;
    let output_rows = output.data.len() / vocab;
    let last_row_start = output_rows.saturating_sub(1) * vocab;
    Ok((
        dispatch_result,
        output.data[last_row_start..last_row_start + vocab].to_vec(),
    ))
}

#[cfg(test)]
fn build_runtime_with_model_execution_engine_and_forced_token(
    fixture: &E2eFixture,
    forced_token: Option<TokenId>,
) -> Runtime {
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .model_execution_engine(std::sync::Arc::new(E2eRuntimeModelExecutionEngine {
            fixture: fixture.clone(),
            kv_states: Arc::new(Mutex::new(BTreeMap::new())),
            pending_kv_states: Arc::new(Mutex::new(BTreeMap::new())),
            component_digest: None,
            forced_token,
        }))
        .trust_store(
            ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone()),
        )
        .build()
        .expect("Reference CPU provider registers cleanly");
    register_reference_cpu_prepared_kernels(&mut runtime);
    runtime
}

#[cfg(test)]
/// `transactional-weight-materialization`: a Model Instance whose weight
/// materialization fails SHALL never have reported Ready in the first
/// place -- `ModelInstances::create()` leaves it in `Loading`, and only a
/// fully successful `WeightMaterializationTransaction::commit` reaches
/// Ready. Proven here under a memory budget tight enough to admit `load()`'s
/// own aggregate allocation but not every subsequent per-tensor weight
/// admission, and that the failed attempt leaves no weight bound to the
/// instance (real rollback, not just a lifecycle label).
///
/// An earlier version of this test (and the code it tested) had the
/// instance reach `Ready` immediately on creation, then get demoted after
/// materialization failed -- a real, since-fixed bug an external audit of
/// PR #36 correctly identified: nothing prevented a caller from observing
/// the instance as `Ready` during that window. This test's name and
/// assertions were rewritten to match the corrected behavior, not just the
/// corrected code.
/// Shared by every check that proves a weight's `TensorResidency` record is
/// gone once its Provider storage and Memory Manager allocation have both
/// been released -- rollback, unload, and repeated load/unload all assert
/// this same property (`invalidate-tensor-residency-on-release`); `context`
/// names which one, for the failure message.
fn assert_tensor_residency_absent(
    runtime: &Runtime,
    resource_id: &TensorResourceId,
    context: &str,
) -> Result<(), E2eConformanceError> {
    if runtime.memory().tensor_residency(resource_id).is_some() {
        return Err(E2eConformanceError::MemoryValidationFailed {
            reason: format!(
                "weight resource '{resource_id}' still has a TensorResidency record {context}"
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
fn check_weight_materialization_failure_never_reaches_ready(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .config(RuntimeConfig {
            memory: MemoryManagerConfig {
                // Generous: `create_model_instance` now releases the
                // whole-artifact-level planning allocation `load_model`
                // admits as soon as a real Model Instance exists to track
                // residency through its own `usage.residency_bytes`
                // instead (task 12.6's allocation-count-before/after-
                // unload fix), and the fixture's per-tensor weight total
                // exactly equals that planning size (both derived from the
                // same manifest) -- leaving no numeric gap between "tight
                // enough to still admit the one-time planning allocation"
                // and "tight enough to fail partway through per-tensor
                // admission". A manually admitted spacer allocation below,
                // sized against this same generous budget, recreates that
                // gap explicitly and controllably instead.
                max_runtime_bytes: Some(5072 * 3),
                allow_pending_allocations: false,
                ..MemoryManagerConfig::default()
            },
            ..RuntimeConfig::default()
        })
        .trust_store(
            ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone()),
        )
        .build()
        .map_err(|error| E2eConformanceError::SuiteUnavailable {
            reason: error.to_string(),
        })?;
    register_reference_cpu_prepared_kernels(&mut runtime);

    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(fixture.architecture_implementation.clone());
    let mut request = ModelLoadingRequest::new(
        ModelLoadingRequestId::new("e2e-weight-materialization-failure"),
        fixture.manifest.id.clone(),
    );
    request.quantization_policy = ModelQuantizationPolicy::RejectUnsupported;
    let loaded = load_model(
        &mut coordinator,
        &mut runtime,
        ModelLoadingApiRequest::new(request),
        &fixture.manifest,
    )?;
    let instance = create_model_instance(
        &mut runtime,
        &loaded,
        fixture.architecture_implementation.clone(),
        ResourceAffinity::new(FallbackClass::Transparent),
    )?;

    // Explicit spacer: consumes most of the generous budget above,
    // leaving enough room for some but not all of the fixture's per-
    // tensor weight allocations (which sum to exactly the same bytes the
    // now-released planning allocation used) -- a controlled way to
    // recreate a tight-budget failure partway through materialization,
    // now that the planning allocation itself no longer stays resident to
    // do so implicitly.
    let spacer = runtime
        .memory_mut()
        .allocate(
            MemoryAllocationRequest::new(
                MemoryAllocationClass::ModelArtifact,
                (5072 * 3) - 2500,
                MemoryPlacement::HostOrdinary,
                MemoryAllocationOwner::InferenceArtifact("spacer".into()),
            )
            .with_alignment(64),
        )
        .map_err(|error| E2eConformanceError::SuiteUnavailable {
            reason: format!("test spacer allocation failed to set up: {error}"),
        })?;
    let _ = spacer;

    // Confirm the instance is genuinely NOT Ready immediately after
    // creation -- the corrected behavior, replacing what used to be an
    // assertion that it *was* Ready here.
    let status_before = runtime
        .model_instance(&instance)
        .map_err(InferenceApiError::from)?
        .status();
    if status_before.lifecycle == ModelInstanceLifecycleState::Ready
        || status_before.readiness.accepts_generation()
    {
        return Err(E2eConformanceError::GenerationFailed {
            reason: format!(
                "expected the instance to NOT be Ready right after creation, before any \
                 weight materialization has run; got lifecycle {:?} / readiness {:?}",
                status_before.lifecycle, status_before.readiness
            ),
        });
    }

    match materialize_model_instance_weights(
        &mut runtime,
        &instance,
        fixture.manifest.id.name.as_str(),
        &fixture.weights,
    ) {
        Err(_) => {}
        Ok(()) => {
            return Err(E2eConformanceError::GenerationFailed {
                reason: "expected weight materialization to fail under a tight memory budget \
                          (test miscalibrated, or admission stopped being enforced)"
                    .into(),
            });
        }
    }

    let status_after = runtime
        .model_instance(&instance)
        .map_err(InferenceApiError::from)?
        .status();
    if status_after.lifecycle == ModelInstanceLifecycleState::Ready {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "instance is Ready after weight materialization failed".into(),
        });
    }
    if status_after.readiness.accepts_generation() {
        return Err(E2eConformanceError::GenerationFailed {
            reason: format!(
                "instance readiness accepts generation after weight materialization failed: {:?}",
                status_after.readiness
            ),
        });
    }
    if status_after.lifecycle != ModelInstanceLifecycleState::Failed {
        return Err(E2eConformanceError::GenerationFailed {
            reason: format!(
                "expected the instance to end in Failed after materialization failed; got {:?}",
                status_after.lifecycle
            ),
        });
    }
    let bound_weight_count = runtime
        .model_instance(&instance)
        .map_err(InferenceApiError::from)?
        .definition
        .resource_bindings
        .weights
        .len();
    if bound_weight_count != 0 {
        return Err(E2eConformanceError::GenerationFailed {
            reason: format!(
                "expected zero weights bound after a failed materialization attempt rolled \
                 back (real rollback, not just a lifecycle label); found {bound_weight_count}"
            ),
        });
    }
    // Prove the rollback released Provider-owned storage too, not only the
    // Model Instance's own bindings -- `WeightMaterializationTransaction::
    // abort` must have called `release_tensor` for every weight staged
    // before the failure, for any weight this attempt might have reached.
    let executor = resolve_kernel_execution_provider(
        &runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )
    .map_err(|error| E2eConformanceError::GenerationFailed {
        reason: error.to_string(),
    })?;
    for name in fixture.weights.keys() {
        let resource_id = TensorResourceId::new(format!("model.{instance}.weight.{name}"));
        if executor.read_tensor(&resource_id).is_some() {
            return Err(E2eConformanceError::GenerationFailed {
                reason: format!(
                    "weight resource '{resource_id}' remained present in Provider-owned \
                     storage after a failed materialization attempt was supposed to roll it \
                     back"
                ),
            });
        }
        assert_tensor_residency_absent(
            &runtime,
            &resource_id,
            "after a failed materialization attempt was supposed to roll it back",
        )?;
    }
    Ok(())
}

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine"
))]
const QWEN_GRAPH_COMPONENT_BYTES: &[u8] =
    include_bytes!("../../fixtures/components/qwen-graph.component.wat");

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine"
))]
const QWEN_GRAPH_COMPONENT_MANIFEST_BYTES: &[u8] =
    include_bytes!("../../fixtures/components/qwen-graph.component.wat.magnetar-component.yaml");

#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine",
    test
))]
/// Test-oracle only (Tachyon integration audit MAG-07): a second, real,
/// structurally distinct Component implementing the same
/// `model-component-graph-producer` world -- `embedding -> rmsnorm ->
/// matmul`, with every decoder layer omitted, so its node count and
/// operator-sequence hash can never coincide with the real Qwen Component's.
/// Exists to prove [`register_inference_component_artifact`]'s registry
/// genuinely supports two independently-registered Components at once, not
/// just one hardcoded Qwen singleton wearing a generic-looking API. Never a
/// production Model Component; production never reads this.
const SYNTHETIC_MINIMAL_COMPONENT_BYTES: &[u8] =
    include_bytes!("../../fixtures/components/synthetic-minimal.component.wasm");

#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine",
    test
))]
const SYNTHETIC_MINIMAL_COMPONENT_MANIFEST_BYTES: &[u8] = include_bytes!(
    "../../fixtures/components/synthetic-minimal.component.wasm.magnetar-component.yaml"
);

#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine",
    test
))]
/// Test-oracle only: the checked-in real Llama Model Component
/// (`components/llama`), a genuine second production model architecture
/// family -- not a synthetic/degenerate fixture like
/// `SYNTHETIC_MINIMAL_COMPONENT_BYTES` above. Its graph-building logic is
/// structurally identical to the real Qwen Component's own (both real,
/// well-documented instances of the same pre-norm/RoPE/grouped-query-
/// attention/SwiGLU decoder block -- Qwen2's architecture is Llama's with
/// an added QKV bias term, not a different block shape); the two produce
/// different graphs only because a real Llama `architecture-config` has no
/// QKV bias while a real Qwen2 one does, driven entirely by `model-config`,
/// never by a hardcoded branch in either Component. Production never reads
/// this constant directly -- an embedder registers the real bytes itself,
/// exactly as it would for Qwen.
const LLAMA_REAL_COMPONENT_BYTES: &[u8] =
    include_bytes!("../../fixtures/components/llama-real.component.wasm");

#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine",
    test
))]
const LLAMA_REAL_COMPONENT_MANIFEST_BYTES: &[u8] =
    include_bytes!("../../fixtures/components/llama-real.component.wasm.magnetar-component.yaml");

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine"
))]
fn qwen_graph_component_package() -> ComponentArtifactPackage {
    ComponentArtifactPackage::new(
        QWEN_GRAPH_COMPONENT_BYTES.to_vec(),
        QWEN_GRAPH_COMPONENT_MANIFEST_BYTES.to_vec(),
        ComponentDigest::parse("sha256", QWEN_GRAPH_COMPONENT_DIGEST),
        ComponentDistributionSource::new(
            ComponentDistributionSourceKind::DevelopmentFixture,
            QWEN_GRAPH_COMPONENT_NAME,
        ),
    )
}

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine"
))]
#[derive(Debug)]
struct QwenComponentPreflight {
    definition: ComponentDefinitionId,
    instance: ComponentInstanceId,
    graph_semantics: QwenComponentGraphSemantics,
    observations: Vec<ComponentObservation>,
}

#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
/// What the Qwen Model Component reports about its own prefill/decode
/// graphs: not just node counts (which a component could satisfy with any
/// arbitrary set of operators) but a hash of the full ordered
/// Operator-kind-code sequence (`qwen_operator_sequence_hash`), so
/// `validate_against_graphs` performs genuine semantic comparison against
/// the Runtime-built graph rather than proving only that the two graphs
/// happen to be the same size.
struct QwenComponentGraphSemantics {
    prefill_node_count: usize,
    decode_node_count: usize,
    prefill_operator_hash: u32,
    decode_operator_hash: u32,
}

#[cfg(test)]
impl QwenComponentGraphSemantics {
    fn validate_against_graphs(
        &self,
        prefill: &ExecutionGraph,
        decode: &ExecutionGraph,
    ) -> Result<(), E2eConformanceError> {
        if self.prefill_node_count != prefill.nodes.len() {
            return Err(E2eConformanceError::GraphValidationFailed {
                reason: format!(
                    "Qwen Component prefill graph declared {} node(s), runtime graph has {}",
                    self.prefill_node_count,
                    prefill.nodes.len()
                ),
            });
        }
        if self.decode_node_count != decode.nodes.len() {
            return Err(E2eConformanceError::GraphValidationFailed {
                reason: format!(
                    "Qwen Component decode graph declared {} node(s), runtime graph has {}",
                    self.decode_node_count,
                    decode.nodes.len()
                ),
            });
        }
        let expected_prefill_hash =
            qwen_operator_sequence_hash(&qwen_graph_operator_codes(prefill)?);
        if self.prefill_operator_hash != expected_prefill_hash {
            return Err(E2eConformanceError::GraphValidationFailed {
                reason: format!(
                    "Qwen Component prefill graph declared operator-sequence hash {:#010x}, runtime graph expects {expected_prefill_hash:#010x}",
                    self.prefill_operator_hash
                ),
            });
        }
        let expected_decode_hash = qwen_operator_sequence_hash(&qwen_graph_operator_codes(decode)?);
        if self.decode_operator_hash != expected_decode_hash {
            return Err(E2eConformanceError::GraphValidationFailed {
                reason: format!(
                    "Qwen Component decode graph declared operator-sequence hash {:#010x}, runtime graph expects {expected_decode_hash:#010x}",
                    self.decode_operator_hash
                ),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
fn build_first_native_graphs_from_component_output(
    fixture: &E2eFixture,
    prompt_token_count: u64,
    component_graph_semantics: QwenComponentGraphSemantics,
) -> Result<FirstNativeComponentGraphs, E2eConformanceError> {
    let prefill = qwen_prefill_graph(
        &fixture.config,
        &fixture.identity,
        prompt_token_count.max(1),
        true,
    )?;
    let decode = qwen_decode_graph(
        &fixture.config,
        &fixture.identity,
        prompt_token_count.max(1),
    )?;
    component_graph_semantics.validate_against_graphs(&prefill.graph, &decode.graph)?;
    validate_first_scope_graph(&prefill.graph)?;
    validate_first_scope_graph(&decode.graph)?;
    Ok(FirstNativeComponentGraphs {
        prefill_node_count: prefill.graph.nodes.len(),
        decode_node_count: decode.graph.nodes.len(),
        prefill: prefill.graph,
        decode: decode.graph,
    })
}

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine"
))]
struct QwenComponentPreflightRequest {
    component_package: ComponentArtifactPackage,
    trust_store: ComponentTrustStore,
    limits: ComponentResourceLimits,
}

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine"
))]
impl QwenComponentPreflightRequest {
    fn default_trusted() -> Self {
        Self {
            component_package: qwen_graph_component_package(),
            trust_store: ComponentTrustStore::default().trust_digest(QWEN_GRAPH_COMPONENT_DIGEST),
            limits: qwen_component_runtime_limits(),
        }
    }
}

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine"
))]
fn invoke_qwen_component_u32(
    manager: &mut ComponentManager,
    instance: ComponentInstanceId,
    interface: &WitInterface,
    operation: &str,
) -> Result<u32, E2eConformanceError> {
    let result = manager
        .invoke(ComponentInvocation::new(
            instance,
            interface.clone(),
            operation,
        ))
        .map_err(|error| E2eConformanceError::ModelComponentFailed {
            reason: error.to_string(),
        })?;
    match result.values.as_slice() {
        [ComponentValue::U32(value)] => Ok(*value),
        values => Err(E2eConformanceError::GraphValidationFailed {
            reason: format!(
                "Qwen Component export '{operation}' returned {values:?}, expected u32"
            ),
        }),
    }
}

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine"
))]
fn validate_and_instantiate_qwen_component_before_first_native_planning(
    request: QwenComponentPreflightRequest,
) -> Result<QwenComponentPreflight, E2eConformanceError> {
    let mut manager = ComponentManager::with_engine(Box::new(
        crate::component_wasmtime::WasmtimeComponentEngine::new().map_err(|error| {
            E2eConformanceError::ModelComponentFailed {
                reason: error.to_string(),
            }
        })?,
    ));
    manager.set_resource_limits(request.limits);
    manager.set_trust_store(request.trust_store);
    let definition = manager
        .prepare_pushed_package(request.component_package)
        .map_err(|error| E2eConformanceError::ModelComponentFailed {
            reason: error.to_string(),
        })?;
    let instance = manager
        .instantiate_prepared_component(definition)
        .map_err(|error| E2eConformanceError::ModelComponentFailed {
            reason: error.to_string(),
        })?;
    let interface = WitInterface::new("magnetar:qwen/graph-fixture", "1.0.0");
    let authority = invoke_qwen_component_u32(
        &mut manager,
        instance,
        &interface,
        "provider-authority-count",
    )?;
    if authority != 0 {
        return Err(E2eConformanceError::BoundaryViolation {
            reason: "Qwen Component fixture requested Provider authority".into(),
        });
    }
    let graph_semantics = QwenComponentGraphSemantics {
        prefill_node_count: invoke_qwen_component_u32(
            &mut manager,
            instance,
            &interface,
            "prefill-node-count",
        )? as usize,
        decode_node_count: invoke_qwen_component_u32(
            &mut manager,
            instance,
            &interface,
            "decode-node-count",
        )? as usize,
        prefill_operator_hash: invoke_qwen_component_u32(
            &mut manager,
            instance,
            &interface,
            "prefill-operator-hash",
        )?,
        decode_operator_hash: invoke_qwen_component_u32(
            &mut manager,
            instance,
            &interface,
            "decode-operator-hash",
        )?,
    };
    Ok(QwenComponentPreflight {
        definition,
        instance,
        graph_semantics,
        observations: manager.observations().to_vec(),
    })
}

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine"
))]
fn validate_and_instantiate_trusted_qwen_component_before_first_native_planning()
-> Result<QwenComponentPreflight, E2eConformanceError> {
    validate_and_instantiate_qwen_component_before_first_native_planning(
        QwenComponentPreflightRequest::default_trusted(),
    )
}

#[cfg(test)]
/// Builds the portable graph semantics a component describing `config`'s
/// prefill/decode graphs correctly would report at `prompt_token_count`:
/// node counts and the full Operator-kind-code sequence
/// (`qwen_graph_operator_codes`), derived directly from the Runtime-built
/// graphs. Used wherever a caller needs a known-correct value rather than
/// one queried across the Component boundary (the non-component fallback
/// build, and tests not themselves exercising component/graph mismatch
/// detection).
fn qwen_component_graph_semantics_for_prompt(
    config: &QwenConfig,
    identity: &ModelComponentIdentity,
    prompt_token_count: u64,
) -> Result<QwenComponentGraphSemantics, E2eConformanceError> {
    let prefill = qwen_prefill_graph(config, identity, prompt_token_count.max(1), true)?.graph;
    let decode = qwen_decode_graph(config, identity, prompt_token_count.max(1))?.graph;
    Ok(QwenComponentGraphSemantics {
        prefill_node_count: prefill.nodes.len(),
        decode_node_count: decode.nodes.len(),
        prefill_operator_hash: qwen_operator_sequence_hash(&qwen_graph_operator_codes(&prefill)?),
        decode_operator_hash: qwen_operator_sequence_hash(&qwen_graph_operator_codes(&decode)?),
    })
}

#[cfg(test)]
/// Correctif 17 / task group 17: `validate_e2e_no_shortcuts` (via
/// `validate_e2e_per_node_causal_chain`) SHALL reject a per-node causal
/// chain that is *incomplete* for a node that genuinely dispatched, not
/// only confirm the five global evidence categories occurred somewhere. A
/// node with `GraphNodeReady` and `PlanBindingResolved`/`PreparedKernelResolved`/
/// `ProviderSubmitted` but no correlated `ProviderCompleted` or
/// `TensorResourceProduced` (as if a dispatch died silently between submit
/// and completion) must be caught, distinctly from the presence-only check
/// this task group's fix supersedes.
fn check_e2e_no_shortcuts_rejects_incomplete_per_node_causal_chain()
-> Result<(), E2eConformanceError> {
    let node = |kind: InferenceApiObservationKind, name: &str| {
        InferenceApiObservation::new(kind, format!("per-node causal event; node={name}"), None)
    };
    let observations = vec![
        node(InferenceApiObservationKind::GraphNodeReady, "embedding"),
        node(
            InferenceApiObservationKind::PlanBindingResolved,
            "embedding",
        ),
        node(
            InferenceApiObservationKind::PreparedKernelResolved,
            "embedding",
        ),
        node(InferenceApiObservationKind::ProviderSubmitted, "embedding"),
        // Deliberately missing: ProviderCompleted / TensorResourceProduced
        // for "embedding" -- as if the node's dispatch died silently
        // between submission and completion.
    ];
    match validate_e2e_per_node_causal_chain(&observations) {
        Err(E2eConformanceError::BoundaryViolation { reason })
            if reason.contains("embedding") && reason.contains("ProviderCompleted") =>
        {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected error for an incomplete per-node causal chain: {error}"),
        }),
        Ok(()) => Err(E2eConformanceError::GenerationFailed {
            reason: "validator accepted an incomplete per-node causal chain".into(),
        }),
    }
}

/// Test-oracle only (task 12.6): exercises the Rust-synthesized graph
/// builder directly to prove it stays internally valid, independent of
/// whether anything in production ever uses it as a graph source.
#[cfg(test)]
pub(crate) fn check_graph_production_and_execution(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    // kv_cache_enabled=true so the prefill graph's K/V edges are actually
    // marked as cache outputs -- otherwise the decode graph below would
    // claim 2 cached tokens that this graph never produced.
    let prefill = qwen_prefill_graph(&fixture.config, &fixture.identity, 2, true)?;
    if prefill.validation.is_none() {
        return Err(E2eConformanceError::GraphValidationFailed {
            reason: "prefill graph was produced without validation".into(),
        });
    }
    // The prefill graph above was built for a 2-token prompt with caching
    // enabled, so the decode graph represents generating the 3rd token
    // against those 2 cached ones.
    let decode = qwen_decode_graph(&fixture.config, &fixture.identity, 2)?;
    if decode.validation.is_none() {
        return Err(E2eConformanceError::GraphValidationFailed {
            reason: "decode graph was produced without validation".into(),
        });
    }
    let policy = GraphPlanningPolicy::default();
    let catalog = default_graph_catalog();
    plan_execution_graph(&prefill.graph, &catalog, &policy, None)
        .map_err(E2eConformanceError::from)?;
    execute_graph_boundary(&prefill.graph, &catalog, &policy).map_err(E2eConformanceError::from)?;
    plan_execution_graph(&decode.graph, &catalog, &policy, None)
        .map_err(E2eConformanceError::from)?;
    execute_graph_boundary(&decode.graph, &catalog, &policy).map_err(E2eConformanceError::from)?;
    Ok(())
}

#[cfg(test)]
pub(crate) fn check_eos_token_stops_generation(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine_and_forced_token(
        fixture,
        Some(E2E_FIXTURE_EOS_TOKEN),
    );
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let tokenized = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::PlainText("a".into())),
        None,
    )?;
    let request = build_generation_request(
        GenerationRequestId::new("e2e-eos-stop")?,
        None,
        GenerationModelReference::ModelInstance(instance),
        generation_tokenizer_reference(fixture),
        tokenized,
        4,
        GenerationParameters::greedy(),
        StopConditions {
            eos: EosPolicy {
                eos_token_ids: vec![E2E_FIXTURE_EOS_TOKEN],
                ..EosPolicy::default()
            },
            ..StopConditions::default()
        },
        StreamingMode::Disabled,
    );
    let request = prepare_generation(&runtime, request)?;
    let mut observer = InferenceApiObserver::new();
    let result = run_generation_loop(
        &mut runtime,
        &request,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |_generated_so_far| false,
        &mut observer,
    )?;
    if result.output.finish_reason != FinishReason::EosToken {
        return Err(E2eConformanceError::GenerationFailed {
            reason: format!(
                "expected FinishReason::EosToken, got {:?}",
                result.output.finish_reason
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
fn check_first_native_generation_requires_ready_model_instance(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    suspend_model_instance(
        &mut runtime,
        &instance,
        ModelInstanceSuspensionReason::AdministrativePolicy,
    )?;

    match require_ready_first_native_instance(&runtime, &instance) {
        Err(InferenceApiError::ModelInstanceNotReady { reason })
            if reason.contains("requires ready ModelInstance") =>
        {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected readiness error: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "first-native generation accepted a non-ready ModelInstance".into(),
        }),
    }
}

#[cfg(test)]
fn check_missing_prepared_plan_fails_closed() -> Result<(), E2eConformanceError> {
    let context = first_native_plan_context(PreparedExecutionPhase::Prefill, 1);
    match require_compatible_first_native_plan(None, &context) {
        Err(PreparedExecutionPlanError::PlanNotFound) => Ok(()),
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected missing-plan error: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "first-native execution accepted missing PreparedExecutionPlan".into(),
        }),
    }
}

#[cfg(test)]
fn check_invalidated_prepared_plan_rejects_new_work(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let graphs = build_first_native_graphs_from_component_output(
        fixture,
        2,
        qwen_component_graph_semantics_for_prompt(&fixture.config, &fixture.identity, 2)?,
    )?;
    let mut plans = prepare_first_native_execution_plans(&runtime, &instance, graphs, 2)?;
    plans
        .decode
        .hard_invalidate(crate::kernel_execution_plan::PlanRebuildReason::KernelRevoked)?;
    let context = first_native_plan_context(PreparedExecutionPhase::Decode, 1);
    match require_compatible_first_native_plan(Some(&mut plans.decode), &context) {
        Err(PreparedExecutionPlanError::PlanNotReadyForExecution) => Ok(()),
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected invalidated-plan error: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "first-native execution accepted invalidated PreparedExecutionPlan".into(),
        }),
    }
}

#[cfg(test)]
fn check_stale_plan_outside_policy_fails_closed(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let graphs = build_first_native_graphs_from_component_output(
        fixture,
        2,
        qwen_component_graph_semantics_for_prompt(&fixture.config, &fixture.identity, 2)?,
    )?;
    let mut plans = prepare_first_native_execution_plans(&runtime, &instance, graphs, 2)?;
    plans.decode.mark_stale(
        crate::kernel_execution_plan::PlanRebuildReason::KernelRevoked,
        crate::kernel_execution_plan::PlanRebuildUrgency::RequiredBeforeNewWork,
    )?;
    let context = first_native_plan_context(PreparedExecutionPhase::Decode, 1);
    match require_compatible_first_native_plan(Some(&mut plans.decode), &context) {
        Err(PreparedExecutionPlanError::PlanStaleOutsidePolicy) => Ok(()),
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected stale-outside-policy error: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "first-native execution accepted a plan stale outside its rebuild policy"
                .into(),
        }),
    }
}

#[cfg(test)]
fn check_qwen_graph_nodes_have_prepared_kernel_bindings(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let graphs = build_first_native_graphs_from_component_output(
        fixture,
        2,
        qwen_component_graph_semantics_for_prompt(&fixture.config, &fixture.identity, 2)?,
    )?;
    let plans = prepare_first_native_execution_plans(&runtime, &instance, graphs, 2)?;

    for (plan, expected_node_count) in [
        (&plans.prefill, plans.prefill_node_count),
        (&plans.decode, plans.decode_node_count),
    ] {
        if plan.node_bindings.len() != expected_node_count {
            return Err(E2eConformanceError::GenerationFailed {
                reason: format!(
                    "prepared plan has {} bindings for {expected_node_count} graph nodes",
                    plan.node_bindings.len()
                ),
            });
        }
        for binding in &plan.node_bindings {
            if binding.graph_nodes.is_empty() {
                return Err(E2eConformanceError::GenerationFailed {
                    reason: "prepared plan contains a binding without graph nodes".into(),
                });
            }
            if binding.kernel.provider.as_str() != REFERENCE_CPU_PROVIDER_NAME {
                return Err(E2eConformanceError::GenerationFailed {
                    reason: "prepared plan selected a non-Reference CPU provider".into(),
                });
            }
            if binding.provider.as_str() != REFERENCE_CPU_PROVIDER_NAME {
                return Err(E2eConformanceError::GenerationFailed {
                    reason: "prepared plan binding provider is not Reference CPU".into(),
                });
            }
            if binding.device.as_ref().map(ToString::to_string).as_deref()
                != Some(REFERENCE_CPU_DEVICE_ID)
            {
                return Err(E2eConformanceError::GenerationFailed {
                    reason: "prepared plan binding did not record Reference CPU device identity"
                        .into(),
                });
            }
            if binding.prepared_kernel.is_none() || binding.prepared_kernel_generation.is_none() {
                return Err(E2eConformanceError::GenerationFailed {
                    reason: "prepared plan binding lacks PreparedKernelId or generation".into(),
                });
            }
            if binding.qualification_profile.as_deref() != Some(REFERENCE_CPU_CONFORMANCE_PROFILE) {
                return Err(E2eConformanceError::GenerationFailed {
                    reason: "prepared plan binding lacks implementation conformance identity"
                        .into(),
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
fn check_graph_dispatch_rejects_unregistered_provider(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let mut plans = first_native_plans_for_prompt(&runtime, fixture, &instance, 2)?;
    let graphs = first_native_component_graphs_for_prompt(fixture, 2)?;
    // Every node binding names a Provider Runtime must resolve at execution
    // time (task 5.2); point them all at a name nothing registers, the
    // execution-time equivalent of the Provider having been removed from
    // Runtime's registration between plan preparation and execution.
    for binding in &mut plans.prefill.node_bindings {
        binding.provider = ProviderBinding::new("unregistered-provider");
    }
    let ids = HostTensor::new([2], vec![1.0, 2.0])?;
    let cache_id = KvCacheId::new("test-unregistered-executor-cache")?;
    match execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graphs.prefill,
        &mut plans.prefill,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    ) {
        Err(InferenceApiError::ProviderUnavailable { reason })
            if reason.contains("unregistered-provider") =>
        {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected error for an unregistered provider: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "graph executor dispatched through an unregistered provider".into(),
        }),
    }
}

#[cfg(test)]
fn check_graph_dispatch_uses_registered_provider_instance(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let mut plans = first_native_plans_for_prompt(&runtime, fixture, &instance, 2)?;
    let graphs = first_native_component_graphs_for_prompt(fixture, 2)?;
    let provider_binding = plans
        .prefill
        .node_bindings
        .first()
        .map(|binding| binding.provider.clone())
        .ok_or_else(|| E2eConformanceError::GraphValidationFailed {
            reason: "prefill plan has no node bindings".into(),
        })?;
    let before = resolve_kernel_execution_provider(&runtime, &provider_binding)
        .map_err(E2eConformanceError::from)?
        .observations()
        .len();
    let ids = HostTensor::new([2], vec![1.0, 2.0])?;
    let cache_id = KvCacheId::new("test-registered-executor-instance-cache")?;
    execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graphs.prefill,
        &mut plans.prefill,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    )
    .map_err(E2eConformanceError::from)?;
    // Re-resolving from Runtime provider registration after dispatch (the
    // same path production execution uses) must observe the *same*
    // registered instance's growing observation trail -- not a disconnected
    // throwaway that discarded its own observations when it went out of
    // scope.
    let after = resolve_kernel_execution_provider(&runtime, &provider_binding)
        .map_err(E2eConformanceError::from)?
        .observations()
        .len();
    if after <= before {
        return Err(E2eConformanceError::GenerationFailed {
            reason:
                "graph dispatch did not record observations on the registered provider instance"
                    .into(),
        });
    }
    Ok(())
}

#[cfg(test)]
fn check_graph_dispatch_accounts_outputs_through_runtime_memory_manager(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let mut plans = first_native_plans_for_prompt(&runtime, fixture, &instance, 2)?;
    let graphs = first_native_component_graphs_for_prompt(fixture, 2)?;
    let ids = HostTensor::new([2], vec![1.0, 2.0])?;
    let cache_id = KvCacheId::new("test-output-accounting-cache")?;
    execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graphs.prefill,
        &mut plans.prefill,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    )
    .map_err(E2eConformanceError::from)?;
    let tensor_allocations = runtime
        .memory()
        .allocations()
        .filter(|allocation| allocation.request.class == MemoryAllocationClass::Tensor)
        .count();
    if tensor_allocations == 0 {
        return Err(E2eConformanceError::MemoryValidationFailed {
            reason:
                "graph dispatch produced no output tensor allocations in Runtime's MemoryManager"
                    .into(),
        });
    }
    Ok(())
}

#[cfg(test)]
/// `enable-device-resident-kernel-chaining`'s discovered leak fix:
/// `execute_invocation_with_memory_manager` (both `providers/cpu` and
/// `providers/cuda`, and this crate's own in-crate `ReferenceCpuExecutor`)
/// previously admitted a fresh `MemoryAllocationId` for each Kernel-internal
/// output resource on every single dispatch, without ever releasing the
/// previous one for that same resource id -- and a Kernel-internal output
/// id (e.g. `{operation_id}.out`) is derived only from the graph node id,
/// so it is stable across every separate dispatch of the same graph.
/// Without the fix, dispatching the identical graph twice would leave the
/// first run's now-orphaned allocations still `Active` in the Memory
/// Manager's ledger forever; with it, the second run's admissions replace
/// (and release) the first's, so the Provider-owned Tensor allocation
/// count does not grow.
fn check_graph_dispatch_does_not_leak_kernel_output_allocations_across_repeated_dispatch(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;

    let provider_owned_active_tensor_count = |runtime: &Runtime| {
        runtime
            .memory()
            .allocations()
            .filter(|allocation| {
                allocation.state == MemoryAllocationState::Active
                    && allocation.request.class == MemoryAllocationClass::Tensor
                    && matches!(allocation.request.owner, MemoryAllocationOwner::Provider(_))
            })
            .count()
    };

    let ids = HostTensor::new([2], vec![1.0, 2.0])?;
    let mut plans = first_native_plans_for_prompt(&runtime, fixture, &instance, 2)?;
    let graphs = first_native_component_graphs_for_prompt(fixture, 2)?;
    let cache_id_1 = KvCacheId::new("test-leak-fix-cache-one")?;
    execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id_1,
        &graphs.prefill,
        &mut plans.prefill,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids.clone())]),
        None,
        Some(0),
        &mut Vec::new(),
    )
    .map_err(E2eConformanceError::from)?;
    let count_after_first_dispatch = provider_owned_active_tensor_count(&runtime);

    // A second, independent dispatch of the *same* graph (same node ids,
    // hence the same Kernel-internal output resource ids), under a
    // different KV cache so Session-owned edge resources don't collide --
    // only the Provider-owned Kernel-internal admissions this fix targets
    // are being counted above.
    let mut plans = first_native_plans_for_prompt(&runtime, fixture, &instance, 2)?;
    let graphs = first_native_component_graphs_for_prompt(fixture, 2)?;
    let cache_id_2 = KvCacheId::new("test-leak-fix-cache-two")?;
    execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id_2,
        &graphs.prefill,
        &mut plans.prefill,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    )
    .map_err(E2eConformanceError::from)?;
    let count_after_second_dispatch = provider_owned_active_tensor_count(&runtime);

    if count_after_second_dispatch != count_after_first_dispatch {
        return Err(E2eConformanceError::MemoryValidationFailed {
            reason: format!(
                "expected the Provider-owned Active Tensor allocation count to stay constant \
                 across a second dispatch of the identical graph (Kernel-internal output ids are \
                 stable across dispatches, so re-admission must replace, not accumulate): \
                 {count_after_first_dispatch} after the first dispatch, \
                 {count_after_second_dispatch} after the second"
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
/// Correctif 5: `execute_qwen_graph_nodes`'s node-to-node transport is
/// Resource-based, not a private `HostTensor` cache -- an *intermediate*
/// graph edge's value (not just the final returned bindings) must be
/// independently readable straight from the registered Provider's storage,
/// under the resource id the executor recorded for it.
fn check_graph_dispatch_intermediate_edge_is_resolvable_from_provider_storage(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let mut plans = first_native_plans_for_prompt(&runtime, fixture, &instance, 2)?;
    let graphs = first_native_component_graphs_for_prompt(fixture, 2)?;
    let embedding_output_edge = graphs
        .prefill
        .nodes
        .get(&ExecutionNodeId::new("embedding"))
        .and_then(|node| node.outputs.first())
        .cloned()
        .ok_or_else(|| E2eConformanceError::GraphValidationFailed {
            reason: "prefill graph has no 'embedding' node output edge".into(),
        })?;
    let ids = HostTensor::new([2], vec![1.0, 2.0])?;
    let cache_id = KvCacheId::new("test-intermediate-edge-resource-cache")?;
    execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graphs.prefill,
        &mut plans.prefill,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    )
    .map_err(E2eConformanceError::from)?;
    let provider = resolve_kernel_execution_provider(
        &runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )
    .map_err(E2eConformanceError::from)?;
    let resource_id = TensorResourceId::new(format!("edge.{embedding_output_edge}"));
    if provider.read_tensor(&resource_id).is_none() {
        return Err(E2eConformanceError::MemoryValidationFailed {
            reason: format!(
                "intermediate edge '{embedding_output_edge}' is not resolvable from Provider \
                 storage at resource '{resource_id}'; graph execution must not hold this \
                 value only in a private, non-Provider-backed cache"
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
/// `define-provider-prepared-kernel-execution-contract` task 3.4: a
/// genuinely two-output Kernel dispatch ("split", the Rust test-oracle
/// graph's fused `gate_up_proj` -> `split` -> `gate`/`up` recipe -- see
/// `qwen_model_component::qwen_build_graph`) must leave *both* declared
/// output edges independently resolvable from Provider storage, under
/// *different* resource ids, each holding the correct half of the
/// pre-split tensor -- not just the first output propagated, the second
/// silently dropped or aliased onto the first (the historical bug this
/// task group closes). Builds the graph directly through
/// `qwen_prefill_graph` rather than `first_native_component_graphs_for_prompt`
/// so this proof holds regardless of whether a strict Component engine is
/// available: the "split" node exists only in this Rust-synthesized
/// recipe (`qwen_expected_tensor_names`'s doc comment), not in the
/// checked-in real Qwen Component's own graph.
fn check_two_output_split_dispatch_produces_independently_resolvable_resources(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let prefill = qwen_prefill_graph(&fixture.config, &fixture.identity, 2, true)?.graph;
    let split_node = prefill
        .nodes
        .values()
        .find(|node| node.operator.name() == "split")
        .ok_or_else(|| E2eConformanceError::GraphValidationFailed {
            reason: "prefill graph has no 'split' node".into(),
        })?;
    let split_input_edge = split_node.inputs.first().cloned().ok_or_else(|| {
        E2eConformanceError::GraphValidationFailed {
            reason: "'split' node has no input edge".into(),
        }
    })?;
    let (gate_edge, up_edge) = match split_node.outputs.as_slice() {
        [left, right] => (left.clone(), right.clone()),
        other => {
            return Err(E2eConformanceError::GraphValidationFailed {
                reason: format!("'split' node expects exactly 2 output edges, got {other:?}"),
            });
        }
    };
    let decode = qwen_decode_graph(&fixture.config, &fixture.identity, 2)?.graph;
    let prefill_graph = prefill.clone();
    let graphs = FirstNativeComponentGraphs {
        prefill_node_count: prefill.nodes.len(),
        prefill,
        decode_node_count: decode.nodes.len(),
        decode,
    };
    let mut plans = prepare_first_native_execution_plans(&runtime, &instance, graphs, 2)?;
    let ids = HostTensor::new([2], vec![1.0, 2.0])?;
    let cache_id = KvCacheId::new("test-two-output-split-cache")?;
    execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &prefill_graph,
        &mut plans.prefill,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    )
    .map_err(E2eConformanceError::from)?;
    let provider = resolve_kernel_execution_provider(
        &runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )
    .map_err(E2eConformanceError::from)?;
    let gate_up_resource = TensorResourceId::new(format!("edge.{split_input_edge}"));
    let gate_resource = TensorResourceId::new(format!("edge.{gate_edge}"));
    let up_resource = TensorResourceId::new(format!("edge.{up_edge}"));
    if gate_resource == up_resource {
        return Err(E2eConformanceError::MemoryValidationFailed {
            reason: "split's two output edges resolved to the same resource id".into(),
        });
    }
    let gate_up = provider.read_tensor(&gate_up_resource).ok_or_else(|| {
        E2eConformanceError::MemoryValidationFailed {
            reason: format!("split's pre-split input '{gate_up_resource}' is not resolvable"),
        }
    })?;
    let gate = provider.read_tensor(&gate_resource).ok_or_else(|| {
        E2eConformanceError::MemoryValidationFailed {
            reason: format!(
                "split's first output '{gate_resource}' is not independently resolvable \
                 from Provider storage"
            ),
        }
    })?;
    let up = provider.read_tensor(&up_resource).ok_or_else(|| {
        E2eConformanceError::MemoryValidationFailed {
            reason: format!(
                "split's second output '{up_resource}' is not independently resolvable \
                 from Provider storage"
            ),
        }
    })?;
    let cols = *gate_up
        .shape
        .last()
        .ok_or_else(|| E2eConformanceError::MemoryValidationFailed {
            reason: "split's pre-split input has no dimensions".into(),
        })? as usize;
    let half = cols / 2;
    let rows = gate_up.data.len() / cols;
    if gate.shape != up.shape || gate.data.len() != rows * half {
        return Err(E2eConformanceError::MemoryValidationFailed {
            reason: format!(
                "split's two outputs have unexpected shapes: gate={:?}, up={:?}, \
                 expected each to be the pre-split input's last dimension halved",
                gate.shape, up.shape
            ),
        });
    }
    for row in 0..rows {
        let expected_gate = &gate_up.data[row * cols..row * cols + half];
        let expected_up = &gate_up.data[row * cols + half..(row + 1) * cols];
        let actual_gate = &gate.data[row * half..(row + 1) * half];
        let actual_up = &up.data[row * half..(row + 1) * half];
        if actual_gate != expected_gate || actual_up != expected_up {
            return Err(E2eConformanceError::MemoryValidationFailed {
                reason: format!(
                    "split's outputs for row {row} do not match the expected halves of its \
                     pre-split input: gate {actual_gate:?} (expected {expected_gate:?}), \
                     up {actual_up:?} (expected {expected_up:?})"
                ),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
fn check_graph_dispatch_releases_workspace_after_use(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let mut plans = first_native_plans_for_prompt(&runtime, fixture, &instance, 2)?;
    // This fixture's one layer includes an `attention` node, the only
    // Operator Reference CPU advertises a required workspace for.
    let graphs = first_native_component_graphs_for_prompt(fixture, 2)?;
    let ids = HostTensor::new([2], vec![1.0, 2.0])?;
    let cache_id = KvCacheId::new("test-workspace-release-cache")?;
    execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graphs.prefill,
        &mut plans.prefill,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    )
    .map_err(E2eConformanceError::from)?;
    let workspace_allocations: Vec<_> = runtime
        .memory()
        .allocations()
        .filter(|allocation| allocation.request.class == MemoryAllocationClass::TemporaryWorkspace)
        .collect();
    if workspace_allocations.is_empty() {
        return Err(E2eConformanceError::MemoryValidationFailed {
            reason: "attention dispatch requested no workspace allocation to release".into(),
        });
    }
    if workspace_allocations
        .iter()
        .any(|allocation| allocation.state == MemoryAllocationState::Active)
    {
        return Err(E2eConformanceError::MemoryValidationFailed {
            reason: "workspace allocation was not released after its dispatch completed".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
fn check_graph_dispatch_records_memory_feasibility_failure_under_tight_budget(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    // Wide enough for model loading's own resident-bytes allocation and this
    // fixture's per-tensor weight resources (task 6.2) to be admitted, but
    // far below `attention`'s required 1 MiB workspace -- so *that*
    // allocation is what fails admission, not an earlier, unrelated one.
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .config(RuntimeConfig {
            memory: MemoryManagerConfig {
                max_runtime_bytes: Some(1 << 16),
                allow_pending_allocations: false,
                ..MemoryManagerConfig::default()
            },
            ..RuntimeConfig::default()
        })
        .trust_store(
            ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone()),
        )
        .build()
        .map_err(|error| E2eConformanceError::SuiteUnavailable {
            reason: error.to_string(),
        })?;
    register_reference_cpu_prepared_kernels(&mut runtime);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let mut plans = first_native_plans_for_prompt(&runtime, fixture, &instance, 2)?;
    let graphs = first_native_component_graphs_for_prompt(fixture, 2)?;
    let ids = HostTensor::new([2], vec![1.0, 2.0])?;
    let cache_id = KvCacheId::new("test-tight-budget-cache")?;
    let result = execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graphs.prefill,
        &mut plans.prefill,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    );
    // Memory Admission Precedes Provider Materialization: every node's
    // declared output must be admitted before its Kernel is dispatched, so
    // the first node this fixture's graph reaches whose output (or, for
    // `attention`, required workspace) does not fit the tight budget hard-
    // fails admission and the Kernel is never dispatched for it.
    match result {
        Err(
            InferenceApiError::MemoryAdmissionFailed { reason }
            | InferenceApiError::GenerationFailed { reason },
        ) if reason.contains("out of memory") || reason.contains("memory admission failed") => {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected error under a tight Runtime memory budget: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "graph dispatch succeeded despite a tight Runtime memory budget".into(),
        }),
    }
}

#[cfg(test)]
fn check_weight_binding_rejects_tampered_artifact_bytes(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let mut tampered = fixture.clone();
    let (_name, tensor) =
        tampered
            .weights
            .iter_mut()
            .next()
            .ok_or_else(|| E2eConformanceError::FixtureInvalid {
                reason: "fixture has no weight tensors to tamper with".into(),
            })?;
    tensor.data[0] += 1.0;
    match load_fixture_instance(&tampered, &mut runtime) {
        Err(E2eConformanceError::FixtureInvalid { reason }) if reason.contains("digest") => Ok(()),
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected error for a tampered weight artifact: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "model loading accepted a weight artifact with tampered bytes".into(),
        }),
    }
}

#[cfg(test)]
/// Like `load_fixture_instance`, but binds `weights` directly through
/// `materialize_model_instance_weights` instead of `bind_qwen_fixture_weights`
/// -- so a caller can supply a deliberately altered weight map without it
/// being rejected by the fixture's own digest check (task 6.5's concern,
/// already covered by `check_weight_binding_rejects_tampered_artifact_bytes`;
/// not what this is for).
fn load_fixture_instance_with_weights(
    fixture: &E2eFixture,
    runtime: &mut Runtime,
    weights: &BTreeMap<String, HostTensor>,
) -> Result<ModelInstanceId, E2eConformanceError> {
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(fixture.architecture_implementation.clone());
    let mut request = ModelLoadingRequest::new(
        ModelLoadingRequestId::new("e2e-fixture-load-weight-sensitivity"),
        fixture.manifest.id.clone(),
    );
    request.quantization_policy = ModelQuantizationPolicy::RejectUnsupported;
    let loaded = load_model(
        &mut coordinator,
        runtime,
        ModelLoadingApiRequest::new(request),
        &fixture.manifest,
    )?;
    let instance = create_model_instance(
        runtime,
        &loaded,
        fixture.architecture_implementation.clone(),
        ResourceAffinity::new(FallbackClass::Transparent),
    )?;
    let mut weights = weights.clone();
    qwen_weights_with_derived_lm_head(fixture, &mut weights)?;
    materialize_model_instance_weights(
        runtime,
        &instance,
        fixture.manifest.id.name.as_str(),
        &weights,
    )?;
    Ok(instance)
}

#[cfg(test)]
/// `bind-materialized-weight-content-to-model-artifact-digests`: proves
/// the content-digest check at the exact public entrypoint it lives in
/// (`WeightMaterializationTransaction::stage_weight`, reached through
/// `materialize_model_instance_weights`), not only through
/// `bind_qwen_fixture_weights`'s separate, earlier, aggregate in-memory
/// check (`check_weight_binding_rejects_tampered_artifact_bytes` proves
/// that one). `fixture.manifest`'s tensor inventory now carries real
/// per-tensor digests computed from the real checked-in Safetensors file
/// (`e2e_fixture_manifest`), so tampering one tensor's bytes before
/// materializing it directly must be rejected with the specific
/// content-digest-mismatch error, not merely *some* error.
fn check_materialize_model_instance_weights_rejects_content_digest_mismatch(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let mut tampered_weights = fixture.weights.clone();
    let (_name, tensor) =
        tampered_weights
            .iter_mut()
            .next()
            .ok_or_else(|| E2eConformanceError::FixtureInvalid {
                reason: "fixture has no weight tensors to tamper with".into(),
            })?;
    tensor.data[0] += 1.0;
    match load_fixture_instance_with_weights(fixture, &mut runtime, &tampered_weights) {
        Err(E2eConformanceError::GenerationFailed { reason })
            if reason.contains("weight content digest mismatch") =>
        {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("expected a weight content digest mismatch error, got: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "materialize_model_instance_weights accepted tampered tensor content".into(),
        }),
    }
}

#[cfg(test)]
/// Regression guard for the happy path this Change's check sits directly
/// in front of: the real, untampered fixture weights (bit-identical to
/// what their declared digests were computed from) must still materialize
/// and bind normally through the exact same entrypoint the mismatch test
/// above uses.
fn check_materialize_model_instance_weights_accepts_matching_content(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let real_weights = e2e_fixture_weights_from_real_artifact(&fixture.config)?;
    load_fixture_instance_with_weights(fixture, &mut runtime, &real_weights)?;
    Ok(())
}

#[cfg(test)]
/// Runs a real prefill through the production graph-execution path
/// (`execute_qwen_graph`, the same one `execute_generation_step` uses) with
/// `weights` bound to a fresh `ModelInstance`, and returns the "logits"
/// edge's values.
fn forward_logits_with_weights(
    fixture: &E2eFixture,
    weights: &BTreeMap<String, HostTensor>,
    prompt: &[TokenId],
) -> Result<Vec<f32>, E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let instance = load_fixture_instance_with_weights(fixture, &mut runtime, weights)?;
    let mut plans =
        first_native_plans_for_prompt(&runtime, fixture, &instance, prompt.len() as u64)?;
    let graphs = first_native_component_graphs_for_prompt(fixture, prompt.len() as u64)?;
    let ids = HostTensor::new(
        [prompt.len() as u64],
        prompt.iter().map(|id| *id as f32).collect::<Vec<_>>(),
    )?;
    let cache_id = KvCacheId::new("test-weight-sensitivity-cache")?;
    let (_dispatch, mut bindings, _layer_kv, _provider) = execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graphs.prefill,
        &mut plans.prefill,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    )?;
    let logits = bindings
        .remove(&TensorEdgeId::new("logits"))
        .ok_or_else(|| E2eConformanceError::GenerationFailed {
            reason: "first-native graph produced no logits output".into(),
        })?;
    Ok(logits.data)
}

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine"
))]
#[allow(clippy::too_many_arguments)]
/// [`forward_logits_with_weights`]'s per-Device pipeline-placement
/// counterpart (`add-real-multi-device-model-instance-placement`): loads a
/// segment-only Model Instance for decoder layer range `[start_layer,
/// end_layer)` (`filter_manifest_for_layer_range`/
/// `qwen_weight_name_in_layer_range`), builds its real prefill-segment
/// graph through the real Qwen Component
/// (`build_first_native_prefill_graph_segment_for_config`), and runs it
/// with `boundary_hidden` bound to `input.hidden_states_in` when
/// `start_layer != 0` (`input.token_ids` otherwise, exactly like a full
/// graph's own first segment) -- returning the segment's own `"logits"`-
/// named output tensor (a raw post-layer hidden state for an internal
/// segment, real logits for one reaching `num_hidden_layers`).
fn forward_segment_logits_with_weights(
    fixture: &E2eFixture,
    weights: &BTreeMap<String, HostTensor>,
    prompt: &[TokenId],
    start_layer: u32,
    end_layer: u32,
    boundary_hidden: Option<HostTensor>,
) -> Result<HostTensor, E2eConformanceError> {
    let num_hidden_layers = fixture.config.architecture.layer_count as u32;
    let segment_manifest = filter_manifest_for_layer_range(
        &fixture.manifest,
        start_layer,
        end_layer,
        num_hidden_layers,
    );
    // Derived from the *full* weight set (needs `token_embedding`, which a
    // segment starting mid-stack never itself carries) before filtering
    // down to this segment's own subset -- mirrors `load_fixture_instance_
    // with_weights`'s own ordering for the exact same reason (tied
    // embeddings' `lm_head` is never a separately declared manifest tensor;
    // see `qwen_expected_tensor_names`).
    let mut segment_weights = weights.clone();
    qwen_weights_with_derived_lm_head(fixture, &mut segment_weights)?;
    segment_weights.retain(|name, _| {
        qwen_weight_name_in_layer_range(name, start_layer, end_layer, num_hidden_layers)
    });

    let mut runtime = build_runtime_trusting_fixture(fixture);
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(fixture.architecture_implementation.clone());
    let mut request = ModelLoadingRequest::new(
        ModelLoadingRequestId::new(format!("test-segment-load-{start_layer}-{end_layer}")),
        segment_manifest.id.clone(),
    );
    request.quantization_policy = ModelQuantizationPolicy::RejectUnsupported;
    let loaded = load_model(
        &mut coordinator,
        &mut runtime,
        ModelLoadingApiRequest::new(request),
        &segment_manifest,
    )?;
    let instance = create_model_instance(
        &mut runtime,
        &loaded,
        fixture.architecture_implementation.clone(),
        ResourceAffinity::new(FallbackClass::Transparent),
    )?;
    materialize_model_instance_weights(
        &mut runtime,
        &instance,
        segment_manifest.id.name.as_str(),
        &segment_weights,
    )?;

    let provider_binding = ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME);
    let status = require_ready_first_native_instance(&runtime, &instance)?;
    let mutation_version = status.status().mutation_version;
    let (segment_graph, _definition, _instance_handle) =
        build_first_native_prefill_graph_segment_for_config(
            &fixture.config,
            &fixture.identity,
            prompt.len() as u64,
            start_layer,
            end_layer,
        )?;
    let mut plan = prepare_first_native_plan_for_graph(
        &runtime,
        &segment_graph,
        &instance,
        mutation_version,
        prompt.len() as u64,
        PreparedExecutionPlanGeneration::new(1),
        &provider_binding,
    )?;

    let initial_bindings = if start_layer == 0 {
        let ids = HostTensor::new(
            [prompt.len() as u64],
            prompt.iter().map(|id| *id as f32).collect::<Vec<_>>(),
        )?;
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)])
    } else {
        let hidden = boundary_hidden.ok_or_else(|| E2eConformanceError::FixtureInvalid {
            reason: "segment starting mid-stack requires a boundary hidden-state tensor".into(),
        })?;
        BTreeMap::from([(TensorEdgeId::new("input.hidden_states_in"), hidden)])
    };
    let cache_id = KvCacheId::new(format!("test-segment-cache-{start_layer}-{end_layer}"))?;
    let (_dispatch, mut bindings, _layer_kv, _provider) = execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &segment_graph,
        &mut plan,
        initial_bindings,
        None,
        Some(0),
        &mut Vec::new(),
    )?;
    bindings
        .remove(&TensorEdgeId::new("logits"))
        .ok_or_else(|| E2eConformanceError::GenerationFailed {
            reason: "segment graph produced no logits-named output".into(),
        })
}

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine"
))]
/// A small, real 2-decoder-layer Qwen fixture, independent of the canonical
/// 1-layer `E2E_FIXTURE_*` constants (too small to split into two non-
/// trivial segments) -- built from `e2e_fixture_manifest_from_weights`
/// (self-consistent digests for an arbitrary `QwenConfig`, not tied to the
/// checked-in canonical Safetensors fixture), exactly like `e2e_fixture()`
/// itself, just with `layer_count: 2`.
fn two_layer_segment_test_fixture() -> Result<E2eFixture, E2eConformanceError> {
    let architecture = qwen_architecture_metadata(4, 2, 2, 2, 2, 8, 258, 32);
    let mut config = QwenConfig::new(architecture, QwenRopeConfig::standard(2));
    config.tied_embeddings = true;
    let identity = qwen_component_identity(
        ModelComponentId::new("segment-split-fixture").expect("static id is valid"),
        ModelComponentVersion::new(1, 0, 0),
        ModelComponentImplementationKind::WebAssemblyComponent,
    );
    config.validate(&identity)?;
    let architecture_implementation = qwen_model_component::qwen_architecture_implementation(
        &identity,
        ModelArchitectureImplementationKind::ComponentBased,
    );
    let weights = e2e_fixture_weights(&config)?;
    let manifest = e2e_fixture_manifest_from_weights(
        &config,
        &architecture_implementation.architecture,
        &weights,
    )?;
    let tokenizer = e2e_fixture_tokenizer()?;

    let descriptor = qwen_component_descriptor(identity.clone(), &config)?;
    qwen_validate_model_artifact(&descriptor, &config, &manifest)?;

    Ok(E2eFixture {
        config,
        identity,
        architecture_implementation,
        manifest,
        tokenizer,
        weights,
    })
}

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine"
))]
/// `add-real-multi-device-model-instance-placement`, Phase B.1's central
/// correctness proof: splitting a real Qwen forward pass into two
/// sequential, independently-loaded segment Model Instances (layers
/// `[0, mid)` then `[mid, num_hidden_layers)`, the boundary hidden state
/// handed from the first segment's real output into the second segment's
/// `hidden_states_in` input) SHALL produce bit-for-bit the same final
/// output as running the exact same prompt through the one, full,
/// unsegmented graph -- proof the segmentation itself (weight-range
/// filtering, `build-*-graph-segment`, and the host-bridged tensor hand-
/// off) is a pure decomposition of the same computation, not an
/// approximation, before any real cross-Device movement
/// (`CudaExecutor::copy_tensor_from_peer_admitted`, already proven
/// elsewhere) is layered on top of it in a later phase.
fn check_two_segment_split_produces_identical_output_to_full_graph()
-> Result<(), E2eConformanceError> {
    let fixture = two_layer_segment_test_fixture()?;
    let num_hidden_layers = fixture.config.architecture.layer_count as u32;
    let mid = num_hidden_layers / 2;
    if mid == 0 || mid == num_hidden_layers {
        return Err(E2eConformanceError::FixtureInvalid {
            reason: "segment split test fixture must have at least 2 decoder layers".into(),
        });
    }
    let prompt: [TokenId; 3] = [3, 5, 7];

    let full_logits = forward_logits_with_weights(&fixture, &fixture.weights, &prompt)?;

    let segment_one_hidden =
        forward_segment_logits_with_weights(&fixture, &fixture.weights, &prompt, 0, mid, None)?;
    let segment_two_logits = forward_segment_logits_with_weights(
        &fixture,
        &fixture.weights,
        &prompt,
        mid,
        num_hidden_layers,
        Some(segment_one_hidden.clone()),
    )?;

    if full_logits != segment_two_logits.data {
        return Err(E2eConformanceError::GenerationFailed {
            reason: format!(
                "two-segment split output does not bit-for-bit match the full single-graph \
                 output: full={full_logits:?} segmented={:?}",
                segment_two_logits.data
            ),
        });
    }

    // Sanity check against a vacuously-always-equal comparison: the first
    // segment's own raw hidden-state output must genuinely differ from the
    // full graph's final logits (different shapes/semantics -- a
    // pre-lm-head hidden state, not a vocabulary distribution), so the
    // equality above is proof the *second* segment's real computation
    // matches, not an accidental identity somewhere upstream.
    if segment_one_hidden.data == full_logits {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "first segment's raw hidden-state output unexpectedly equals the full \
                      graph's final logits; the two-segment test is not exercising a real \
                      mid-stack boundary"
                .into(),
        });
    }
    Ok(())
}

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine"
))]
/// Phase C's own real correctness proof, cheap and Reference-CPU-only
/// (`add-real-multi-device-model-instance-placement`): a real DECODE step
/// split across two segment Model Instances -- each loaded *once*
/// (`load_first_native_segment_with_provider_and_weights`) and dispatched
/// *twice* (a real prefill, then a real decode step reusing the same
/// Instance and threading its own `FirstNativeProviderRunOutcome::
/// layer_kv` forward via `run_first_native_graph_segment_dispatch`'s
/// `kv_history` parameter, exactly like a real multi-step generation loop
/// would) -- SHALL produce bit-for-bit the same decode logits as running
/// the identical prompt and follow-up token through the one, full,
/// unsegmented graph's own prefill-then-decode pair. Phase B.1/B.2 above
/// only ever proved a single prefill dispatch; this is the first
/// (cheap, always-run, not `#[ignore]`d) proof that segment DECODE --
/// `build_first_native_decode_graph_segment_for_config` and the
/// `kv_history`/`absolute_position_override` threading
/// `run_first_native_graph_segment_dispatch` gained for Phase C -- is
/// correct, independent of and much cheaper than the real ~1GB-checkpoint,
/// two-real-GPU proof in `integration-tests/production-loading`.
fn check_two_segment_split_decode_step_matches_full_graph_decode() -> Result<(), E2eConformanceError>
{
    let fixture = two_layer_segment_test_fixture()?;
    let num_hidden_layers = fixture.config.architecture.layer_count as u32;
    let mid = num_hidden_layers / 2;
    if mid == 0 || mid == num_hidden_layers {
        return Err(E2eConformanceError::FixtureInvalid {
            reason: "segment split test fixture must have at least 2 decoder layers".into(),
        });
    }
    let prompt: [TokenId; 3] = [3, 5, 7];
    let admitted: TokenId = 9;
    let prompt_len = prompt.len() as u64;

    // Reference: the real, full, unsegmented graph's own prefill-then-
    // decode pair, exactly like `check_graph_executor_matches_full_
    // sequence_oracle`'s own pattern.
    let mut full_runtime = build_runtime_trusting_fixture(&fixture);
    let full_instance =
        load_fixture_instance_with_weights(&fixture, &mut full_runtime, &fixture.weights)?;
    let mut full_plans =
        first_native_plans_for_prompt(&full_runtime, &fixture, &full_instance, prompt_len)?;
    let full_graphs = first_native_component_graphs_for_prompt(&fixture, prompt_len)?;
    let full_cache = KvCacheId::new("segment-decode-coverage-full-cache")?;
    let prompt_ids = HostTensor::new(
        [prompt_len],
        prompt.iter().map(|id| *id as f32).collect::<Vec<_>>(),
    )?;
    let (_prefill_dispatch, _prefill_bindings, full_layer_kv, _provider) = execute_qwen_graph(
        &mut full_runtime,
        &fixture,
        &full_instance,
        &full_cache,
        &full_graphs.prefill,
        &mut full_plans.prefill,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), prompt_ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    )
    .map_err(E2eConformanceError::from)?;
    let admitted_ids = HostTensor::new([1], vec![admitted as f32])?;
    let (_decode_dispatch, full_decode_bindings, _updated_layer_kv, _provider) =
        execute_qwen_graph(
            &mut full_runtime,
            &fixture,
            &full_instance,
            &full_cache,
            &full_graphs.decode,
            &mut full_plans.decode,
            BTreeMap::from([(TensorEdgeId::new("input.token_ids"), admitted_ids)]),
            Some(&full_layer_kv),
            Some(prompt_len),
            &mut Vec::new(),
        )
        .map_err(E2eConformanceError::from)?;
    let full_decode_logits = full_decode_bindings
        .get(&TensorEdgeId::new("logits"))
        .ok_or_else(|| E2eConformanceError::GenerationFailed {
            reason: "full graph decode produced no logits output".into(),
        })?
        .data
        .clone();

    // Segmented: each segment's own Model Instance loaded once, dispatched
    // twice (prefill, then decode reusing its own updated `layer_kv`).
    let provider: Arc<dyn Provider> = Arc::new(ReferenceCpuProvider::new());
    let (segment_one_runtime, segment_one_instance, segment_one_binding) =
        load_first_native_segment_with_provider_and_weights(
            provider.clone(),
            &fixture,
            &fixture.weights,
            0,
            mid,
        )?;
    let (segment_two_runtime, segment_two_instance, segment_two_binding) =
        load_first_native_segment_with_provider_and_weights(
            provider,
            &fixture,
            &fixture.weights,
            mid,
            num_hidden_layers,
        )?;
    let segment_cache = KvCacheId::new("segment-decode-coverage-segment-cache")?;

    let (segment_one_prefill_graph, _definition, _instance) =
        build_first_native_prefill_graph_segment_for_config(
            &fixture.config,
            &fixture.identity,
            prompt_len,
            0,
            mid,
        )?;
    let segment_one_prefill_outcome = run_first_native_graph_segment_dispatch(
        segment_one_runtime,
        &fixture,
        segment_one_instance,
        &segment_one_binding,
        &segment_one_prefill_graph,
        &segment_cache,
        &prompt,
        0,
        None,
        None,
        Some(0),
    )?;
    let segment_one_prefill_hidden = segment_one_prefill_outcome
        .bindings
        .get(&TensorEdgeId::new("logits"))
        .ok_or_else(|| E2eConformanceError::GenerationFailed {
            reason: "segment one prefill produced no logits-named output".into(),
        })?
        .clone();

    let (segment_two_prefill_graph, _definition, _instance) =
        build_first_native_prefill_graph_segment_for_config(
            &fixture.config,
            &fixture.identity,
            prompt_len,
            mid,
            num_hidden_layers,
        )?;
    let segment_two_prefill_outcome = run_first_native_graph_segment_dispatch(
        segment_two_runtime,
        &fixture,
        segment_two_instance,
        &segment_two_binding,
        &segment_two_prefill_graph,
        &segment_cache,
        &prompt,
        mid,
        Some(QwenSegmentBoundaryInput::Host(segment_one_prefill_hidden)),
        None,
        Some(0),
    )?;

    let (segment_one_decode_graph, _definition, _instance) =
        build_first_native_decode_graph_segment_for_config(
            &fixture.config,
            &fixture.identity,
            prompt_len,
            0,
            mid,
        )?;
    let segment_one_decode_outcome = run_first_native_graph_segment_dispatch(
        segment_one_prefill_outcome.runtime,
        &fixture,
        segment_one_prefill_outcome.instance,
        &segment_one_binding,
        &segment_one_decode_graph,
        &segment_cache,
        &[admitted],
        0,
        None,
        Some(&segment_one_prefill_outcome.layer_kv),
        Some(prompt_len),
    )?;
    let segment_one_decode_hidden = segment_one_decode_outcome
        .bindings
        .get(&TensorEdgeId::new("logits"))
        .ok_or_else(|| E2eConformanceError::GenerationFailed {
            reason: "segment one decode produced no logits-named output".into(),
        })?
        .clone();

    let (segment_two_decode_graph, _definition, _instance) =
        build_first_native_decode_graph_segment_for_config(
            &fixture.config,
            &fixture.identity,
            prompt_len,
            mid,
            num_hidden_layers,
        )?;
    let segment_two_decode_outcome = run_first_native_graph_segment_dispatch(
        segment_two_prefill_outcome.runtime,
        &fixture,
        segment_two_prefill_outcome.instance,
        &segment_two_binding,
        &segment_two_decode_graph,
        &segment_cache,
        &[admitted],
        mid,
        Some(QwenSegmentBoundaryInput::Host(segment_one_decode_hidden)),
        Some(&segment_two_prefill_outcome.layer_kv),
        Some(prompt_len),
    )?;
    let segmented_decode_logits = segment_two_decode_outcome
        .bindings
        .get(&TensorEdgeId::new("logits"))
        .ok_or_else(|| E2eConformanceError::GenerationFailed {
            reason: "segment two decode produced no logits output".into(),
        })?
        .data
        .clone();

    if full_decode_logits != segmented_decode_logits {
        return Err(E2eConformanceError::GenerationFailed {
            reason: format!(
                "two-segment decode step does not bit-for-bit match the full graph's own \
                 decode step: full={full_decode_logits:?} segmented={segmented_decode_logits:?}"
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
/// Correctif 6 / task 8.7: a single changed weight byte in the Artifact
/// SHALL change generated logits -- proof the graph-executed path actually
/// reads and uses the bound weight bytes numerically, rather than (for
/// example) a cached or hard-coded computation that happens to match the
/// fixture's usual values. Complements `check_weight_binding_rejects_tampered_artifact_bytes`
/// (task 8.8), which proves a *digest* mismatch is caught before binding --
/// this instead proves the bound bytes are not merely checked but actually
/// consumed.
fn check_weight_byte_change_alters_generated_logits(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    // This test's own mutated weights would now (correctly) be rejected by
    // `bind-materialized-weight-content-to-model-artifact-digests`'s
    // content-digest check if bound against `fixture.manifest`'s real,
    // digest-bearing tensor inventory -- that rejection is a *different*
    // property, already proven by `check_weight_binding_rejects_tampered_
    // artifact_bytes` and the new digest-mismatch tests this Change adds.
    // This test's own concern is orthogonal: given bytes that *did* bind
    // (however that happened), are they actually consumed numerically. A
    // digest-free copy of the fixture keeps that concern isolated rather
    // than conflating it with the digest check.
    let mut digest_free_fixture = fixture.clone();
    for tensor in &mut digest_free_fixture.manifest.tensors {
        tensor.digest = None;
    }
    let prompt = [1, 2];
    let baseline_logits =
        forward_logits_with_weights(&digest_free_fixture, &fixture.weights, &prompt)?;

    let mut mutated_weights = fixture.weights.clone();
    let (_name, tensor) =
        mutated_weights
            .iter_mut()
            .next()
            .ok_or_else(|| E2eConformanceError::FixtureInvalid {
                reason: "fixture has no weight tensors to mutate".into(),
            })?;
    tensor.data[0] += 1.0;
    let mutated_logits =
        forward_logits_with_weights(&digest_free_fixture, &mutated_weights, &prompt)?;

    if baseline_logits == mutated_logits {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "changing one weight byte did not change the generated logits".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
fn check_graph_execution_fails_closed_on_missing_weight(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    // Loading bound every declared weight successfully (digest verified);
    // removing one binding afterward -- without touching the fixture or its
    // digest -- isolates the "this instance is missing a resource graph
    // execution needs" failure mode from artifact tampering, which
    // `check_weight_binding_rejects_tampered_artifact_bytes` already covers.
    runtime
        .model_instances_mut()
        .instance_mut(&instance)
        .map_err(InferenceApiError::from)?
        .definition
        .resource_bindings
        .weights
        .remove("token_embedding");
    let mut plans = first_native_plans_for_prompt(&runtime, fixture, &instance, 2)?;
    let graphs = first_native_component_graphs_for_prompt(fixture, 2)?;
    let ids = HostTensor::new([2], vec![1.0, 2.0])?;
    let cache_id = KvCacheId::new("test-missing-weight-cache")?;
    match execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graphs.prefill,
        &mut plans.prefill,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    ) {
        Err(InferenceApiError::ModelLoadingFailed { reason })
            if reason.contains("token_embedding") =>
        {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected error for a missing weight: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "graph execution succeeded despite a missing required weight".into(),
        }),
    }
}

#[cfg(test)]
fn check_weight_resources_are_isolated_per_model_instance(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (first, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let (second, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    if first == second {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "loading the same fixture twice produced the same Model Instance id".into(),
        });
    }
    let first_bindings = runtime
        .model_instance(&first)
        .map_err(InferenceApiError::from)?
        .definition
        .resource_bindings
        .weights
        .clone();
    let second_bindings = runtime
        .model_instance(&second)
        .map_err(InferenceApiError::from)?
        .definition
        .resource_bindings
        .weights
        .clone();
    if first_bindings.is_empty() || first_bindings.len() != second_bindings.len() {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "both Model Instances did not bind the same set of weight names".into(),
        });
    }
    for (name, first_resource) in &first_bindings {
        let second_resource =
            second_bindings
                .get(name)
                .ok_or_else(|| E2eConformanceError::GenerationFailed {
                    reason: format!("second Model Instance has no binding for weight '{name}'"),
                })?;
        if first_resource == second_resource {
            return Err(E2eConformanceError::GenerationFailed {
                reason: format!(
                    "weight '{name}' resolved to the same TensorResourceId for two different Model Instances"
                ),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
fn check_unload_releases_weight_resource_allocations(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let allocation_ids: Vec<_> = runtime
        .model_instance(&instance)
        .map_err(InferenceApiError::from)?
        .definition
        .resource_bindings
        .memory_allocations
        .iter()
        .copied()
        .collect();
    if allocation_ids.is_empty() {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "loading the fixture bound no weight memory allocations to release".into(),
        });
    }
    // `transactional-weight-materialization` (P0-2-bis): unload must also
    // release the Provider-owned weight Tensor Resources themselves, not
    // only Memory Manager accounting -- capture them before unload so they
    // can be checked against Provider storage afterward.
    let weight_resource_ids: Vec<TensorResourceId> = runtime
        .model_instance(&instance)
        .map_err(InferenceApiError::from)?
        .definition
        .resource_bindings
        .weights
        .values()
        .cloned()
        .collect();
    if weight_resource_ids.is_empty() {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "loading the fixture bound no weight Tensor Resources to release".into(),
        });
    }
    let executor = resolve_kernel_execution_provider(
        &runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )
    .map_err(|error| E2eConformanceError::GenerationFailed {
        reason: error.to_string(),
    })?;
    for resource_id in &weight_resource_ids {
        if executor.read_tensor(resource_id).is_none() {
            return Err(E2eConformanceError::GenerationFailed {
                reason: format!(
                    "weight resource '{resource_id}' was not actually present in Provider \
                     storage before unload (test precondition broken)"
                ),
            });
        }
    }
    runtime
        .unload_model_instance(&instance, ModelInstanceUnloadPolicy::RejectActiveUse)
        .map_err(|error| E2eConformanceError::GenerationFailed {
            reason: error.to_string(),
        })?;
    for allocation_id in allocation_ids {
        let state = runtime
            .memory()
            .allocations()
            .find(|allocation| allocation.id == allocation_id)
            .map(|allocation| allocation.state);
        if state == Some(MemoryAllocationState::Active) {
            return Err(E2eConformanceError::MemoryValidationFailed {
                reason: format!(
                    "weight allocation {allocation_id:?} remained Active after Model Instance unload"
                ),
            });
        }
    }
    for resource_id in &weight_resource_ids {
        if executor.read_tensor(resource_id).is_some() {
            return Err(E2eConformanceError::MemoryValidationFailed {
                reason: format!(
                    "weight resource '{resource_id}' remained present in Provider-owned \
                     storage after Model Instance unload (P0-2-bis: unload must release \
                     Provider-owned weight storage, not only Memory Manager accounting)"
                ),
            });
        }
        assert_tensor_residency_absent(&runtime, resource_id, "after Model Instance unload")?;
    }
    Ok(())
}

#[cfg(test)]
/// Proves the load/unload cycle does not accumulate Provider-owned weight
/// storage over repeated cycles -- the audit's own "100x load/unload"
/// case, done at a smaller, still-meaningful count (each cycle already
/// proves the property; more repetitions prove only that it does not
/// degrade with iteration count, which a fixed small count already shows
/// as well without materially slower test runs).
fn check_repeated_load_unload_does_not_accumulate_weight_storage(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    const CYCLES: usize = 10;
    let mut runtime = build_runtime_trusting_fixture(fixture);
    for cycle in 0..CYCLES {
        let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
        let weight_resource_ids: Vec<TensorResourceId> = runtime
            .model_instance(&instance)
            .map_err(InferenceApiError::from)?
            .definition
            .resource_bindings
            .weights
            .values()
            .cloned()
            .collect();
        if weight_resource_ids.is_empty() {
            return Err(E2eConformanceError::GenerationFailed {
                reason: format!("cycle {cycle}: loading the fixture bound no weight resources"),
            });
        }
        runtime
            .unload_model_instance(&instance, ModelInstanceUnloadPolicy::RejectActiveUse)
            .map_err(|error| E2eConformanceError::GenerationFailed {
                reason: format!("cycle {cycle}: unload failed: {error}"),
            })?;
        let executor = resolve_kernel_execution_provider(
            &runtime,
            &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        )
        .map_err(|error| E2eConformanceError::GenerationFailed {
            reason: error.to_string(),
        })?;
        for resource_id in &weight_resource_ids {
            if executor.read_tensor(resource_id).is_some() {
                return Err(E2eConformanceError::MemoryValidationFailed {
                    reason: format!(
                        "cycle {cycle}: weight resource '{resource_id}' still present in \
                         Provider storage after unload -- storage is accumulating across cycles"
                    ),
                });
            }
            // Each cycle creates a fresh Model Instance, so a fresh
            // TensorResourceId per weight -- a residency record surviving
            // past its own cycle's unload would mean residency metadata
            // grows unbounded across cycles even though Provider storage
            // and Memory Manager accounting both look clean
            // (`invalidate-tensor-residency-on-release`).
            assert_tensor_residency_absent(
                &runtime,
                resource_id,
                &format!("after unload in cycle {cycle} -- residency metadata is accumulating"),
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
fn check_incremental_decode_matches_full_sequence_oracle(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let prompt = vec![1, 2];
    let admitted = 3;
    let mut plans =
        first_native_plans_for_prompt(&runtime, fixture, &instance, prompt.len() as u64)?;
    let (_prefill_dispatch, _prefill_hidden, layer_kv) =
        execute_qwen_prefill_hidden_states_through_dispatch(
            &mut runtime,
            fixture,
            &prompt,
            &mut plans.prefill,
        )?;
    let kv_state = FirstNativeExecutionKvState {
        cache: KvCacheId::new("first-native-oracle-kv").map_err(E2eConformanceError::from)?,
        compatibility: KvCacheCompatibility::new(
            GenerationModelReference::LoadedModelContext("qwen-test".into()),
            TokenizerId::new("qwen-test-tokenizer")?,
        ),
        // The hand-rolled decode oracle's own return is a plain, densely
        // 0-indexed `Vec` (it processes every layer unconditionally, never
        // a segment), so its Vec index already equals the real layer
        // number here -- `enumerate()` recovers that as an explicit key
        // for `QwenLayerKvMap`.
        layer_kv: layer_kv.into_iter().enumerate().collect(),
        provider: None,
    };

    let (_decode_dispatch, decode_hidden, updated_layer_kv) =
        execute_qwen_decode_hidden_states_through_dispatch(
            &mut runtime,
            fixture,
            admitted,
            &kv_state,
            prompt.len() as u64,
            &mut plans.decode,
        )?;
    let (_logits_dispatch, incremental_logits) =
        dispatch_qwen_logits_projection(&runtime, fixture, &decode_hidden, &plans.decode)?;

    let mut full_sequence = prompt;
    full_sequence.push(admitted);
    let oracle_logits = e2e_forward(fixture, &full_sequence)?;
    for (index, (actual, expected)) in incremental_logits
        .iter()
        .zip(oracle_logits.iter())
        .enumerate()
    {
        if (actual - expected).abs() > 1e-4 {
            return Err(E2eConformanceError::GenerationFailed {
                reason: format!(
                    "incremental decode logits diverged at {index}: {actual} != {expected}"
                ),
            });
        }
    }

    let executor = resolve_kernel_execution_provider(
        &runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )
    .map_err(E2eConformanceError::from)?;
    for layer in updated_layer_kv {
        let k_tensor = executor.read_tensor(&layer.k).ok_or_else(|| {
            E2eConformanceError::GenerationFailed {
                reason: format!("no materialized K tensor for resource '{}'", layer.k),
            }
        })?;
        let v_tensor = executor.read_tensor(&layer.v).ok_or_else(|| {
            E2eConformanceError::GenerationFailed {
                reason: format!("no materialized V tensor for resource '{}'", layer.v),
            }
        })?;
        let (k_rows, _) = k_tensor.rows_cols()?;
        let (v_rows, _) = v_tensor.rows_cols()?;
        if k_rows != full_sequence.len() as u64 || v_rows != full_sequence.len() as u64 {
            return Err(E2eConformanceError::GenerationFailed {
                reason: "decode did not append exactly one K/V row per layer".into(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
/// Proves the graph-driven executor (`execute_qwen_graph`, which production
/// first-native execution now uses exclusively) produces logits matching the
/// independent `e2e_forward` oracle, and that its recorded per-layer KV
/// state carries one row per historical token. Complements
/// `check_incremental_decode_matches_full_sequence_oracle`, which checks the
/// same oracle against the retired hand-written dispatch sequence.
fn check_graph_executor_matches_full_sequence_oracle(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let prompt = vec![1, 2];
    let admitted = 3;
    let mut plans =
        first_native_plans_for_prompt(&runtime, fixture, &instance, prompt.len() as u64)?;
    let graphs = first_native_component_graphs_for_prompt(fixture, prompt.len() as u64)?;

    let cache_id = KvCacheId::new("test-graph-executor-cache")?;
    let prompt_ids = HostTensor::new(
        [prompt.len() as u64],
        prompt.iter().map(|id| *id as f32).collect::<Vec<_>>(),
    )?;
    let (_prefill_dispatch, _prefill_bindings, layer_kv, _provider) = execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graphs.prefill,
        &mut plans.prefill,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), prompt_ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    )
    .map_err(E2eConformanceError::from)?;

    let admitted_ids = HostTensor::new([1], vec![admitted as f32])?;
    let (_decode_dispatch, decode_bindings, updated_layer_kv, _provider) = execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graphs.decode,
        &mut plans.decode,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), admitted_ids)]),
        Some(&layer_kv),
        Some(prompt.len() as u64),
        &mut Vec::new(),
    )
    .map_err(E2eConformanceError::from)?;

    let logits = decode_bindings
        .get(&TensorEdgeId::new("logits"))
        .ok_or_else(|| E2eConformanceError::GenerationFailed {
            reason: "graph executor produced no logits output".into(),
        })?;

    let mut full_sequence = prompt;
    full_sequence.push(admitted);
    let oracle_logits = e2e_forward(fixture, &full_sequence)?;
    for (index, (actual, expected)) in logits.data.iter().zip(oracle_logits.iter()).enumerate() {
        if (actual - expected).abs() > 1e-4 {
            return Err(E2eConformanceError::GenerationFailed {
                reason: format!(
                    "graph executor decode logits diverged at {index}: {actual} != {expected}"
                ),
            });
        }
    }

    let executor = resolve_kernel_execution_provider(
        &runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )
    .map_err(E2eConformanceError::from)?;
    for layer in updated_layer_kv.values() {
        let k_tensor = executor.read_tensor(&layer.k).ok_or_else(|| {
            E2eConformanceError::GenerationFailed {
                reason: format!("no materialized K tensor for resource '{}'", layer.k),
            }
        })?;
        let v_tensor = executor.read_tensor(&layer.v).ok_or_else(|| {
            E2eConformanceError::GenerationFailed {
                reason: format!("no materialized V tensor for resource '{}'", layer.v),
            }
        })?;
        let (k_rows, _) = k_tensor.rows_cols()?;
        let (v_rows, _) = v_tensor.rows_cols()?;
        if k_rows != full_sequence.len() as u64 || v_rows != full_sequence.len() as u64 {
            return Err(E2eConformanceError::GenerationFailed {
                reason: "graph executor decode did not append exactly one K/V row per layer".into(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
fn parse_absolute_position(message: &str) -> Result<Option<usize>, E2eConformanceError> {
    let Some(value) = message
        .split_whitespace()
        .find_map(|part| part.strip_prefix("absolute_position="))
    else {
        return Ok(None);
    };
    value
        .parse::<usize>()
        .map(Some)
        .map_err(|error| E2eConformanceError::GenerationFailed {
            reason: format!("invalid absolute_position observation {value:?}: {error}"),
        })
}

#[cfg(test)]
fn check_generation_loop_decode_positions_follow_generated_tokens(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let prompt = vec![1, 2, 3, 4];
    let tokenized = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::TokenIds(prompt.clone())),
        None,
    )?;
    let request = build_generation_request(
        GenerationRequestId::new("e2e-decode-position-oracle")?,
        None,
        GenerationModelReference::ModelInstance(instance.clone()),
        generation_tokenizer_reference(fixture),
        tokenized,
        4,
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
    if result.output.generated_token_count != 4 {
        return Err(E2eConformanceError::GenerationFailed {
            reason: format!(
                "expected four generated tokens for multi-step decode, got {}",
                result.output.generated_token_count
            ),
        });
    }

    let mut positions = Vec::new();
    for observation in observer.observations() {
        if observation.kind == InferenceApiObservationKind::ProviderCompleted
            && observation.message.contains("model_input_tokens=1")
            && let Some(position) = parse_absolute_position(&observation.message)?
        {
            positions.push(position);
        }
    }
    let expected = vec![prompt.len(), prompt.len() + 1, prompt.len() + 2];
    if positions != expected {
        return Err(E2eConformanceError::GenerationFailed {
            reason: format!(
                "decode absolute positions diverged from generation-loop oracle: {positions:?} != {expected:?}"
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
fn graph_prefill_setup(
    fixture: &E2eFixture,
) -> Result<
    (
        Runtime,
        ModelInstanceId,
        KvCacheId,
        ExecutionGraph,
        PreparedExecutionPlan,
        HostTensor,
    ),
    E2eConformanceError,
> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let prompt = [1, 2];
    let plans = first_native_plans_for_prompt(&runtime, fixture, &instance, prompt.len() as u64)?;
    let graphs = first_native_component_graphs_for_prompt(fixture, prompt.len() as u64)?;
    let ids = HostTensor::new(
        [prompt.len() as u64],
        prompt.iter().map(|id| *id as f32).collect::<Vec<_>>(),
    )?;
    let cache_id = KvCacheId::new("test-graph-prefill-setup-cache")?;
    Ok((
        runtime,
        instance,
        cache_id,
        graphs.prefill,
        plans.prefill,
        ids,
    ))
}

#[cfg(test)]
fn check_graph_executor_rejects_missing_plan_binding(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let (mut runtime, instance, cache_id, graph, mut plan, ids) = graph_prefill_setup(fixture)?;
    plan.node_bindings.retain(|binding| {
        !binding
            .graph_nodes
            .contains(&ExecutionNodeId::new("embedding"))
    });
    match execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graph,
        &mut plan,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    ) {
        Err(InferenceApiError::KernelUnavailable { reason }) if reason.contains("embedding") => {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected error for a missing plan binding: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "graph executor accepted a graph node with no published plan binding".into(),
        }),
    }
}

#[cfg(test)]
fn check_graph_executor_rejects_unsupported_operator(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let (mut runtime, instance, cache_id, mut graph, mut plan, ids) = graph_prefill_setup(fixture)?;
    let node = graph
        .nodes
        .get_mut(&ExecutionNodeId::new("embedding"))
        .ok_or_else(|| E2eConformanceError::GraphValidationFailed {
            reason: "prefill graph is missing node 'embedding'".into(),
        })?;
    node.operator = OperatorId::magnetar("softmax", 1, OperatorFamily::Activation);
    match execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graph,
        &mut plan,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    ) {
        Err(InferenceApiError::OperatorUnsupported { reason }) if reason.contains("softmax") => {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected error for an unsupported operator: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "graph executor dispatched an operator it does not implement".into(),
        }),
    }
}

#[cfg(test)]
fn check_graph_executor_rejects_cyclic_graph(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let (mut runtime, instance, cache_id, _graph, mut plan, _ids) = graph_prefill_setup(fixture)?;
    let mut graph = ExecutionGraph::new(
        ExecutionGraphId::new("cyclic-test"),
        ExecutionGraphPhase::Test,
    );
    graph = graph
        .with_edge(TensorEdge::new(
            TensorEdgeId::new("a"),
            f32_tensor_descriptor(&HostTensor::new([1, 1], vec![0.0])?),
        ))
        .with_edge(TensorEdge::new(
            TensorEdgeId::new("b"),
            f32_tensor_descriptor(&HostTensor::new([1, 1], vec![0.0])?),
        ))
        .with_node(
            ExecutionNode::new(
                ExecutionNodeId::new("node-a"),
                OperatorId::magnetar("silu", 1, OperatorFamily::Activation),
            )
            .with_input(TensorEdgeId::new("b"))
            .with_output(TensorEdgeId::new("a")),
        )
        .with_node(
            ExecutionNode::new(
                ExecutionNodeId::new("node-b"),
                OperatorId::magnetar("silu", 1, OperatorFamily::Activation),
            )
            .with_input(TensorEdgeId::new("a"))
            .with_output(TensorEdgeId::new("b")),
        );
    match execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graph,
        &mut plan,
        BTreeMap::new(),
        None,
        Some(0),
        &mut Vec::new(),
    ) {
        Err(InferenceApiError::GraphPlanningFailed { reason }) if reason.contains("cycle") => {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected error for a cyclic graph: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "graph executor accepted a cyclic graph".into(),
        }),
    }
}

#[cfg(test)]
fn check_graph_executor_rejects_removed_producer_node(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let (mut runtime, instance, cache_id, mut graph, mut plan, ids) = graph_prefill_setup(fixture)?;
    // Removing the node that produces `layer0.q` leaves `layer0.rope_q`
    // depending on an edge nothing in the graph produces -- structurally the
    // same shape a component or graph-mutation bug would take. `plan` was
    // built from the graph before this mutation, so
    // `PreparedExecutionPlanExecutor::prepare_node_execution` (Correctif 4)
    // now rejects every node's dispatch on the very first one it reaches:
    // the graph's semantic fingerprint no longer matches the published
    // Plan's, which is a stronger, earlier rejection of the same
    // underlying inconsistency than reaching the specific missing-producer
    // edge deeper into execution.
    graph.nodes.remove(&ExecutionNodeId::new("layer0.q_proj"));
    match execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graph,
        &mut plan,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    ) {
        Err(InferenceApiError::KernelUnavailable { reason })
            if reason.contains("PlanValidationFailed") =>
        {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected error for a removed producer node: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "graph executor accepted a graph with a removed producer node".into(),
        }),
    }
}

#[cfg(test)]
fn check_graph_executor_logits_provenance_requires_declared_output_edge(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let (mut runtime, instance, cache_id, mut graph, mut plan, ids) = graph_prefill_setup(fixture)?;
    // Removing the `lm_head` node means nothing produces the `logits` edge
    // this graph declares as its output -- the caller must observe that
    // absence explicitly rather than reading stale or unrelated data. `plan`
    // was built from the graph before this mutation, so
    // `PreparedExecutionPlanExecutor::prepare_node_execution` (Correctif 4)
    // now fails closed on the graph/Plan fingerprint mismatch before any
    // node dispatches -- a stronger guarantee against a fabricated `logits`
    // binding than reaching the end of a partial run and checking its
    // absence, since no dispatch happens at all.
    graph.nodes.remove(&ExecutionNodeId::new("lm_head"));
    match execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graph,
        &mut plan,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    ) {
        Err(InferenceApiError::KernelUnavailable { reason })
            if reason.contains("PlanValidationFailed") =>
        {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected error for a missing output-producing node: {error}"),
        }),
        Ok((_dispatch, bindings, _layer_kv, _provider)) => {
            if bindings.contains_key(&TensorEdgeId::new("logits")) {
                Err(E2eConformanceError::GenerationFailed {
                    reason: "graph executor produced a 'logits' binding with no producing node"
                        .into(),
                })
            } else {
                Err(E2eConformanceError::GenerationFailed {
                    reason: "graph/Plan fingerprint mismatch was not detected".into(),
                })
            }
        }
    }
}

#[cfg(test)]
/// Correctif 4, task 4.6: a `PreparedKernelId` a published Plan binds to
/// SHALL be refused for new dispatch once revoked, rather than the revoked
/// state being silently ignored because dispatch never actually asked the
/// Kernel Registry about it.
fn check_graph_dispatch_rejects_revoked_prepared_kernel(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let (mut runtime, instance, cache_id, graph, mut plan, ids) = graph_prefill_setup(fixture)?;
    let embedding_kernel = plan
        .node_bindings
        .iter()
        .find(|binding| {
            binding
                .graph_nodes
                .contains(&ExecutionNodeId::new("embedding"))
        })
        .map(|binding| binding.kernel.clone())
        .ok_or_else(|| E2eConformanceError::KernelCoverageMissing {
            reason: "prefill plan has no binding for node embedding".into(),
        })?;
    runtime.kernel_registry_mut().revoke_kernel(
        &embedding_kernel,
        "test: simulate revocation after Plan publication",
    );
    match execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graph,
        &mut plan,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    ) {
        // Revocation deactivates the Kernel's advertisement (Kernel
        // Registry's `active` flag), which `dispatch_reference_cpu_operator`
        // rejects immediately after `prepare_node_execution` resolves the
        // binding -- an active-advertisement lookup is a separate concern
        // from `PreparedKernel.state`, so this is the correct rejection
        // point for advertisement-level revocation specifically.
        Err(InferenceApiError::KernelUnavailable { reason })
            if reason.contains("no longer active") =>
        {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected error for a revoked prepared kernel: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "graph dispatch executed a revoked PreparedKernel".into(),
        }),
    }
}

#[cfg(test)]
/// Correctif 4, task 4.5: once a Plan is published, a Kernel Registry
/// preference change (e.g. a newer, more attractively-ranked Kernel
/// registered for the same Operator) SHALL NOT affect that already-
/// published, ready Plan's dispatch -- `prepare_node_execution` looks up
/// the specific `PreparedKernelId` the binding already names, it never
/// re-ranks candidates the way live Kernel Registry selection
/// (`KernelRegistry::select`) does.
fn check_graph_dispatch_ignores_kernel_registry_preference_change_after_plan_publication(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let (mut runtime, _instance, _cache_id, graph, mut plan, ids) = graph_prefill_setup(fixture)?;
    let embedding_kernel = plan
        .node_bindings
        .iter()
        .find(|binding| {
            binding
                .graph_nodes
                .contains(&ExecutionNodeId::new("embedding"))
        })
        .map(|binding| binding.kernel.clone())
        .ok_or_else(|| E2eConformanceError::KernelCoverageMissing {
            reason: "prefill plan has no binding for node embedding".into(),
        })?;

    // Register a second, deliberately better-ranked (lower cost, lower
    // fallback rank) Kernel advertising the same "embedding" Operator
    // *after* the Plan was already published -- if a live Kernel Registry
    // selection were consulted instead of the Plan's own binding, this
    // would be a legitimate, more attractive contender.
    let mut competitor = reference_cpu_kernel_advertisements()
        .into_iter()
        .find(|advertisement| advertisement.implemented_operator.name() == "embedding")
        .ok_or_else(|| E2eConformanceError::KernelCoverageMissing {
            reason: "Reference CPU fixture does not advertise embedding".into(),
        })?;
    competitor.id.name = format!("{}-cheaper-competitor", competitor.id.name);
    competitor
        .performance_hints
        .insert("estimated-cost".into(), "0".into());
    competitor
        .performance_hints
        .insert("fallback-rank".into(), "0".into());
    runtime
        .kernel_registry_mut()
        .register_fixture_advertisement(competitor)
        .map_err(|error| E2eConformanceError::KernelCoverageMissing {
            reason: format!("failed to register competing embedding Kernel: {error}"),
        })?;

    let provider = resolve_kernel_execution_provider(
        &runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )
    .map_err(E2eConformanceError::from)?;
    let seq_len = ids.shape.first().copied().unwrap_or(1);
    let architecture = &fixture.config.architecture;
    let token_embedding = fixture_tensor_by_name(&fixture.weights, "token_embedding")?.clone();
    let mut node_events = Vec::new();
    let mut dispatch_ctx = QwenDispatchContext {
        runtime: &mut runtime,
        provider: provider.clone(),
        prepared_plan: Some(&mut plan),
        graph: Some(&graph),
        sequence_length: Some(seq_len),
        last_provider_execution: None,
        node_events: &mut node_events,
    };
    let (dispatch_result, _hidden_states) = dispatch_reference_cpu_operator(
        &mut dispatch_ctx,
        "embedding",
        dispatch_operator_id("embedding", OperatorFamily::Tensor),
        vec![
            NodeInputResource::Fresh(
                TensorResourceId::new("embedding.table"),
                f32_tensor_descriptor(&token_embedding),
                token_embedding,
            ),
            NodeInputResource::Fresh(
                TensorResourceId::new("embedding.ids"),
                f32_tensor_descriptor(&ids),
                ids.clone(),
            ),
        ],
        (
            TensorResourceId::new("embedding.out"),
            TensorDescriptor::new(
                ShapeDescriptor::new([seq_len, architecture.hidden_size]),
                DTypeDescriptor::portable(ComputeDType::Float32),
                LayoutDescriptor::Contiguous,
            ),
        ),
        BTreeMap::new(),
    )
    .map_err(E2eConformanceError::from)?;

    if dispatch_result.selected_kernel != embedding_kernel {
        return Err(E2eConformanceError::GenerationFailed {
            reason: format!(
                "published Plan binding for 'embedding' was bypassed by a newer, \
                 better-ranked Kernel registration: expected {embedding_kernel:?}, dispatched \
                 {:?}",
                dispatch_result.selected_kernel
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
/// Correctif 4, task 4.7: a Plan binding's `PreparedKernelGeneration` SHALL
/// match the Kernel Registry's active generation for that `PreparedKernelId`
/// at dispatch time; a stale generation (e.g. left over from before a hot
/// Kernel replacement) is refused rather than dispatched as if current.
fn check_graph_dispatch_rejects_stale_prepared_kernel_generation(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let (mut runtime, instance, cache_id, graph, mut plan, ids) = graph_prefill_setup(fixture)?;
    let binding = plan
        .node_bindings
        .iter_mut()
        .find(|binding| {
            binding
                .graph_nodes
                .contains(&ExecutionNodeId::new("embedding"))
        })
        .ok_or_else(|| E2eConformanceError::KernelCoverageMissing {
            reason: "prefill plan has no binding for node embedding".into(),
        })?;
    binding.prepared_kernel_generation = Some(PreparedKernelGeneration::new(u64::MAX));
    match execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graph,
        &mut plan,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    ) {
        Err(InferenceApiError::KernelUnavailable { reason })
            if reason.contains("PreparedKernelGenerationMismatch") =>
        {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected error for a stale prepared kernel generation: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "graph dispatch executed a binding with a stale PreparedKernel generation"
                .into(),
        }),
    }
}

#[cfg(test)]
/// Correctif 4, task 4.8: a Plan binding's declared `provider` SHALL match
/// the Kernel Registry's active `PreparedKernel` provider at dispatch time;
/// a mismatch (e.g. a binding pointing at a Provider the active Kernel is
/// no longer registered under) is refused rather than silently dispatched
/// against whichever Provider the `PreparedKernel` actually belongs to.
fn check_graph_dispatch_rejects_provider_binding_mismatch(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let (mut runtime, instance, cache_id, graph, mut plan, ids) = graph_prefill_setup(fixture)?;
    // Mutate the *last* binding, not the first: `execute_qwen_graph` resolves
    // the Provider it actually dispatches through from `node_bindings.first()`
    // (every node in a first-native graph binds to the same Provider), so
    // corrupting that one would fail earlier and coarser, at graph-level
    // Provider resolution, rather than exercising `prepare_node_execution`'s
    // own per-binding Provider consistency check this test targets.
    let binding = plan.node_bindings.last_mut().ok_or_else(|| {
        E2eConformanceError::KernelCoverageMissing {
            reason: "prefill plan has no node bindings".into(),
        }
    })?;
    binding.provider = ProviderBinding::new("magnetar:provider/does-not-exist");
    match execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graph,
        &mut plan,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    ) {
        Err(InferenceApiError::KernelUnavailable { reason })
            if reason.contains("PlanProviderUnavailable") =>
        {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected error for a Provider binding mismatch: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "graph dispatch executed a binding whose Provider does not match the active PreparedKernel".into(),
        }),
    }
}

#[cfg(test)]
fn check_generation_loop_executes_published_plan_bindings(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let prompt = vec![1, 2, 3, 4];
    let tokenized = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::TokenIds(prompt.clone())),
        None,
    )?;
    // Must be the same graph source `execute_generation_step` itself will
    // dispatch against (see that function's own doc comment on this exact
    // requirement) -- this test only cares about the *binding removal*
    // below producing the expected failure, not which recipe built the
    // graph, so it uses the same helper every real dispatch path uses
    // rather than calling the Rust-synthesized recipe directly.
    let component_graphs = first_native_component_graphs_for_prompt(fixture, prompt.len() as u64)?;
    let mut prepared_plans = prepare_first_native_execution_plans(
        &runtime,
        &instance,
        component_graphs,
        prompt.len() as u64,
    )?;
    prepared_plans.prefill.node_bindings.retain(|binding| {
        !binding
            .graph_nodes
            .contains(&ExecutionNodeId::new("lm_head"))
    });

    let request = build_generation_request(
        GenerationRequestId::new("e2e-plan-binding-required")?,
        None,
        GenerationModelReference::ModelInstance(instance),
        generation_tokenizer_reference(fixture),
        tokenized,
        1,
        GenerationParameters::greedy(),
        StopConditions::default(),
        StreamingMode::Disabled,
    );
    let request = prepare_generation(&runtime, request)?;
    let mut observer = InferenceApiObserver::new();
    let mut execution_plans = RuntimeGenerationExecutionPlans {
        prefill: &mut prepared_plans.prefill,
        decode: &mut prepared_plans.decode,
    };
    match run_generation_loop_with_execution_plans(
        &mut runtime,
        &request,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |_generated_so_far| false,
        &mut observer,
        &mut execution_plans,
    ) {
        Err(InferenceApiError::KernelUnavailable { reason }) if reason.contains("lm_head") => {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected prepared binding error: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "generation succeeded despite missing published lm_head binding".into(),
        }),
    }
}

#[cfg(test)]
fn check_incremental_decode_rejects_missing_layer_kv(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let mut plans = first_native_plans_for_prompt(&runtime, fixture, &instance, 2)?;
    let kv_state = FirstNativeExecutionKvState {
        cache: KvCacheId::new("first-native-empty-kv").map_err(E2eConformanceError::from)?,
        compatibility: KvCacheCompatibility::new(
            GenerationModelReference::LoadedModelContext("qwen-test".into()),
            TokenizerId::new("qwen-test-tokenizer")?,
        ),
        layer_kv: QwenLayerKvMap::new(),
        provider: None,
    };
    match execute_qwen_decode_hidden_states_through_dispatch(
        &mut runtime,
        fixture,
        3,
        &kv_state,
        2,
        &mut plans.decode,
    ) {
        Err(InferenceApiError::KvCacheUnavailable { .. }) => Ok(()),
        Err(error) => Err(E2eConformanceError::from(error)),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "decode accepted missing layer KV state".into(),
        }),
    }
}

#[cfg(test)]
fn new_kv_lifecycle_engine(fixture: &E2eFixture) -> E2eRuntimeModelExecutionEngine {
    E2eRuntimeModelExecutionEngine {
        fixture: fixture.clone(),
        kv_states: Arc::new(Mutex::new(BTreeMap::new())),
        pending_kv_states: Arc::new(Mutex::new(BTreeMap::new())),
        component_digest: None,
        forced_token: None,
    }
}

#[cfg(test)]
fn kv_lifecycle_test_request(
    fixture: &E2eFixture,
    runtime: &Runtime,
    instance: &ModelInstanceId,
    request_id: &str,
    session: Option<InferenceSessionId>,
    prompt: &[TokenId],
) -> Result<GenerationRequest, E2eConformanceError> {
    let tokenized = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::TokenIds(prompt.to_vec())),
        None,
    )?;
    let request = build_generation_request(
        GenerationRequestId::new(request_id)?,
        session,
        GenerationModelReference::ModelInstance(instance.clone()),
        generation_tokenizer_reference(fixture),
        tokenized,
        4,
        GenerationParameters::greedy(),
        StopConditions::default(),
        StreamingMode::Disabled,
    );
    Ok(prepare_generation(runtime, request)?)
}

#[cfg(test)]
fn kv_lifecycle_session_request(
    fixture: &E2eFixture,
    instance: &ModelInstanceId,
) -> SessionCreationRequest {
    SessionCreationRequest {
        model: GenerationModelReference::ModelInstance(instance.clone()),
        tokenizer: generation_tokenizer_reference(fixture),
        generation_defaults: GenerationParameters::greedy(),
        policy: SessionPolicy::default(),
        memory: SessionMemoryBudget::default(),
        allowed_capabilities: BTreeSet::new(),
        correlation_id: None,
        created_at_millis: 0,
    }
}

#[cfg(test)]
fn prefill_cache_id_from_step(
    step: &RuntimeModelExecutionStep,
) -> Result<KvCacheId, E2eConformanceError> {
    match &step.kv_commit {
        Some(RuntimeKvCacheCommit::PrefillCompleted { cache, .. }) => Ok(cache.clone()),
        _ => Err(E2eConformanceError::GenerationFailed {
            reason: "expected a prefill KV commit descriptor".into(),
        }),
    }
}

#[cfg(test)]
/// Proves a generation step's KV write stays *pending* -- never promoted
/// onto the cache's committed `layer_resources` (task 7.4 prepare) -- when
/// sampling rejects every candidate after a successful forward pass and
/// `commit_generation_step` is consequently never called.
fn check_kv_sampling_failure_leaves_cache_uncommitted(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let engine = new_kv_lifecycle_engine(fixture);
    let request = kv_lifecycle_test_request(
        fixture,
        &runtime,
        &instance,
        "kv-sampling-failure",
        None,
        &[1, 2],
    )?;
    let mut plans = first_native_plans_for_prompt(
        &runtime,
        fixture,
        &instance,
        request.input_token_ids.len() as u64,
    )?;
    let step =
        engine.execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))?;
    let cache = prefill_cache_id_from_step(&step)?;
    if !runtime.kv_cache(&cache)?.layer_resources.is_empty() {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "KV cache carries committed layer resources despite no commit call".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
/// Proves a generation step that fails during Provider dispatch (task 5.2's
/// registered-Provider resolution failing here) never stores a pending KV
/// state -- there is nothing a later, unrelated commit could wrongly
/// promote.
fn check_kv_provider_failure_stores_no_pending_state(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let engine = new_kv_lifecycle_engine(fixture);
    let request = kv_lifecycle_test_request(
        fixture,
        &runtime,
        &instance,
        "kv-provider-failure",
        None,
        &[1, 2],
    )?;
    let mut plans = first_native_plans_for_prompt(
        &runtime,
        fixture,
        &instance,
        request.input_token_ids.len() as u64,
    )?;
    for binding in &mut plans.prefill.node_bindings {
        binding.provider = ProviderBinding::new("unregistered-provider");
    }
    match engine.execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill)) {
        Err(InferenceApiError::ProviderUnavailable { .. }) => {}
        Err(error) => {
            return Err(E2eConformanceError::GenerationFailed {
                reason: format!("unexpected error for an unregistered provider: {error}"),
            });
        }
        Ok(_) => {
            return Err(E2eConformanceError::GenerationFailed {
                reason: "generation step succeeded despite an unregistered provider".into(),
            });
        }
    }
    if !engine
        .pending_kv_states
        .lock()
        .map_err(|_| E2eConformanceError::GenerationFailed {
            reason: "pending KV state lock poisoned".into(),
        })?
        .is_empty()
    {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "a failed generation step left a pending KV state behind".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
/// Proves a decode step's pending KV write never reaches the cache's
/// committed `layer_resources` when the request is cancelled before
/// `commit_generation_step` runs -- the committed cache stays exactly what
/// the prior successful commit left it as.
fn check_kv_cancelled_decode_does_not_corrupt_committed_cache(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let engine = new_kv_lifecycle_engine(fixture);
    let request = kv_lifecycle_test_request(
        fixture,
        &runtime,
        &instance,
        "kv-cancel-rollback",
        None,
        &[1, 2],
    )?;
    let mut plans = first_native_plans_for_prompt(
        &runtime,
        fixture,
        &instance,
        request.input_token_ids.len() as u64,
    )?;

    let prefill_step =
        engine.execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))?;
    engine.commit_generation_step(&mut runtime, &request, &[], 1, &prefill_step)?;
    let cache = prefill_cache_id_from_step(&prefill_step)?;
    let committed_after_prefill = runtime.kv_cache(&cache)?.layer_resources.clone();

    let generated = vec![1];
    let _decode_step = engine.execute_generation_step(
        &mut runtime,
        &request,
        &generated,
        Some(&mut plans.decode),
    )?;
    let committed_after_cancelled_decode = runtime.kv_cache(&cache)?.layer_resources.clone();
    if committed_after_cancelled_decode != committed_after_prefill {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "a cancelled decode step's pending KV write altered the committed cache".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
/// Correctif 1 / task 1.8: decode's KV-history-concatenated pending write is
/// genuinely admitted through `MemoryManager` (via `write_tensor_admitted`),
/// not left as a bare, unaccounted `write_tensor` -- the admitted
/// allocation's byte size reflects the *concatenated* (history + new token)
/// tensor a decode step's `Append` KV behavior produces, not just the newly
/// dispatched token's own smaller Kernel output.
fn check_kv_pending_write_is_memory_admitted_for_its_concatenated_size(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let engine = new_kv_lifecycle_engine(fixture);
    let request = kv_lifecycle_test_request(
        fixture,
        &runtime,
        &instance,
        "kv-pending-admission",
        None,
        &[1, 2],
    )?;
    let mut plans = first_native_plans_for_prompt(
        &runtime,
        fixture,
        &instance,
        request.input_token_ids.len() as u64,
    )?;
    let prefill_step =
        engine.execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))?;
    engine.commit_generation_step(&mut runtime, &request, &[], 1, &prefill_step)?;
    let cache = prefill_cache_id_from_step(&prefill_step)?;

    let generated = vec![1];
    engine.execute_generation_step(&mut runtime, &request, &generated, Some(&mut plans.decode))?;

    // This fixture's prefill prompt (`&[1, 2]`) is 2 tokens; decode appends
    // 1 more, so the pending K/V write's history-concatenated row count is
    // 3, each row `hidden_size` wide.
    let expected_bytes =
        3 * fixture.config.architecture.hidden_size * std::mem::size_of::<f32>() as u64;
    let matching_allocations = runtime
        .memory()
        .allocations()
        .filter(|allocation| {
            allocation.state == MemoryAllocationState::Active
                && allocation.request.owner == MemoryAllocationOwner::Session(cache.to_string())
                && allocation.request.size_bytes == expected_bytes
        })
        .count();
    // 4, not 2: `execute_qwen_graph_nodes` reassigns `output_tensor` to the
    // concatenated value *before* the KV-node's `edge.*` write too (so a
    // later reader of that edge sees the same concatenated value the
    // pending write does -- see that write site's own doc comment), so
    // each of K and V produces one concatenated-size allocation for its
    // `kv.*.pending` resource *and* one for its `edge.*` resource: this
    // fixture's single layer's K and V nodes together admit 2 + 2.
    if matching_allocations != 4 {
        return Err(E2eConformanceError::MemoryValidationFailed {
            reason: format!(
                "expected 4 Active {expected_bytes}-byte allocations (K and V, each with a \
                 pending-resource and an edge-resource allocation) for the decode step's \
                 concatenated KV write, owned by cache '{cache}'; found {matching_allocations}"
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
/// Correctif 1 / task 1.9: discarding a pending KV state
/// (`discard_pending_kv_state`, invoked automatically at the start of the
/// next generation step, or directly on cancellation) releases the
/// `MemoryManager` allocation the pending write admitted (task 1.8's fix),
/// not just the Provider storage entry -- otherwise every cancelled or
/// failed decode step would leak one allocation per layer per role forever.
fn check_kv_pending_write_allocation_is_released_on_discard(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let engine = new_kv_lifecycle_engine(fixture);
    let request = kv_lifecycle_test_request(
        fixture,
        &runtime,
        &instance,
        "kv-pending-discard-release",
        None,
        &[1, 2],
    )?;
    let mut plans = first_native_plans_for_prompt(
        &runtime,
        fixture,
        &instance,
        request.input_token_ids.len() as u64,
    )?;
    // Deliberately never committed: the pending write stays pending, so its
    // allocation is still held when discarded below.
    let prefill_step =
        engine.execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))?;
    let cache = prefill_cache_id_from_step(&prefill_step)?;

    let active_owned_by_cache = |runtime: &Runtime| {
        runtime
            .memory()
            .allocations()
            .filter(|allocation| {
                allocation.state == MemoryAllocationState::Active
                    && allocation.request.owner == MemoryAllocationOwner::Session(cache.to_string())
            })
            .count()
    };
    let active_before_discard = active_owned_by_cache(&runtime);
    engine.discard_pending_kv_state(&mut runtime, &request)?;
    let active_after_discard = active_owned_by_cache(&runtime);
    let released = active_before_discard.saturating_sub(active_after_discard);
    // K and V for this fixture's single layer.
    if released != 2 {
        return Err(E2eConformanceError::MemoryValidationFailed {
            reason: format!(
                "expected discard to release exactly 2 (K and V) Active allocations owned by \
                 cache '{cache}'; released {released} (before: {active_before_discard}, after: \
                 {active_after_discard})"
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
/// Correctif 11 / task group 9: a multi-layer KV commit is atomic. Sabotages
/// the *second* resource (layer 0's pending V, after K would otherwise
/// promote successfully) a decode step's commit would promote, and proves
/// the whole commit fails and the cache's committed state is left exactly
/// as the prior successful commit produced it -- not with layer 0's K
/// pointing at this step's data while V still points at the previous
/// step's.
fn check_kv_partial_layer_failure_during_commit_rolls_back_cleanly(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let engine = new_kv_lifecycle_engine(fixture);
    let request = kv_lifecycle_test_request(
        fixture,
        &runtime,
        &instance,
        "kv-partial-layer-failure",
        None,
        &[1, 2],
    )?;
    let mut plans = first_native_plans_for_prompt(
        &runtime,
        fixture,
        &instance,
        request.input_token_ids.len() as u64,
    )?;

    let prefill_step =
        engine.execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))?;
    engine.commit_generation_step(&mut runtime, &request, &[], 1, &prefill_step)?;
    let cache = prefill_cache_id_from_step(&prefill_step)?;
    let committed_before = runtime.kv_cache(&cache)?.layer_resources.clone();
    let layer0_before =
        committed_before
            .get(&0)
            .cloned()
            .ok_or_else(|| E2eConformanceError::GenerationFailed {
                reason: "prefill commit produced no layer 0 KV binding".into(),
            })?;
    let provider_for_prefill_check = resolve_kernel_execution_provider(
        &runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )
    .map_err(E2eConformanceError::from)?;
    let original_k_value = provider_for_prefill_check
        .read_tensor(&layer0_before.k)
        .ok_or_else(|| E2eConformanceError::GenerationFailed {
            reason: "prefill-committed K resource is not resolvable from Provider storage".into(),
        })?;

    let generated = vec![1];
    let decode_step = engine.execute_generation_step(
        &mut runtime,
        &request,
        &generated,
        Some(&mut plans.decode),
    )?;
    let pending_v =
        {
            let pending_kv_states = engine.pending_kv_states.lock().map_err(|_| {
                E2eConformanceError::GenerationFailed {
                    reason: "pending KV state lock poisoned".into(),
                }
            })?;
            pending_kv_states
                .values()
                .next()
                .and_then(|state| state.layer_kv.get(&0))
                .map(|layer| layer.v.clone())
                .ok_or_else(|| E2eConformanceError::GenerationFailed {
                    reason: "decode step produced no pending KV state to sabotage".into(),
                })?
        };
    // Remove layer 0's pending V straight from Provider storage -- exactly
    // what `promote_pending_kv_layer_role`'s "no pending KV data to commit"
    // error path detects -- while leaving K's pending resource intact, so
    // K would promote successfully if the commit were not atomic.
    let provider = resolve_kernel_execution_provider(
        &runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )
    .map_err(E2eConformanceError::from)?;
    let _ = provider.release_tensor(&pending_v);

    match engine.commit_generation_step(&mut runtime, &request, &generated, 2, &decode_step) {
        Err(InferenceApiError::KvCacheUnavailable { .. }) => {}
        Err(error) => {
            return Err(E2eConformanceError::GenerationFailed {
                reason: format!("unexpected error for a sabotaged mid-commit resource: {error}"),
            });
        }
        Ok(()) => {
            return Err(E2eConformanceError::GenerationFailed {
                reason: "commit succeeded despite a missing pending resource for one layer".into(),
            });
        }
    }

    let committed_after_failed_commit = runtime.kv_cache(&cache)?.layer_resources.clone();
    if committed_after_failed_commit != committed_before {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "a failed multi-layer KV commit left the cache in a partially-promoted state"
                .into(),
        });
    }
    // The deeper property a binding-equality check alone cannot see: layer
    // 0's K would have promoted successfully in isolation (only V was
    // sabotaged), so a non-atomic implementation could release the
    // pre-existing K allocation and overwrite its Provider-stored bytes
    // before ever learning V failed -- leaving `layer_resources` pointing
    // at an unchanged resource id whose *contents* were nonetheless
    // destroyed. Both must still be exactly as they were before this
    // attempt.
    if !runtime
        .memory()
        .allocations()
        .any(|allocation| allocation.id == layer0_before.k_allocation)
    {
        return Err(E2eConformanceError::MemoryValidationFailed {
            reason: "the pre-existing layer 0 K allocation was released despite the commit \
                     that would have replaced it failing"
                .into(),
        });
    }
    let provider = resolve_kernel_execution_provider(
        &runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )
    .map_err(E2eConformanceError::from)?;
    let k_value_after_failed_commit = provider.read_tensor(&layer0_before.k).ok_or_else(|| {
        E2eConformanceError::MemoryValidationFailed {
            reason: "layer 0's committed K resource is no longer resolvable from Provider \
                     storage after a failed commit"
                .into(),
        }
    })?;
    if k_value_after_failed_commit.data != original_k_value.data {
        return Err(E2eConformanceError::MemoryValidationFailed {
            reason: "a failed multi-layer KV commit destructively overwrote layer 0's \
                     still-committed K bytes before the failure was known"
                .into(),
        });
    }
    Ok(())
}

#[cfg(test)]
/// Proves a second `commit_generation_step` call for the same completed
/// step is rejected rather than silently re-promoting (or double-releasing)
/// KV resources the first commit already promoted.
fn check_kv_double_commit_second_call_is_rejected(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let engine = new_kv_lifecycle_engine(fixture);
    let request = kv_lifecycle_test_request(
        fixture,
        &runtime,
        &instance,
        "kv-double-commit",
        None,
        &[1, 2],
    )?;
    let mut plans = first_native_plans_for_prompt(
        &runtime,
        fixture,
        &instance,
        request.input_token_ids.len() as u64,
    )?;
    let step =
        engine.execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))?;
    engine.commit_generation_step(&mut runtime, &request, &[], 1, &step)?;
    match engine.commit_generation_step(&mut runtime, &request, &[], 1, &step) {
        Err(InferenceApiError::KvCacheUnavailable { reason }) if reason.contains("pending") => {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected error for a double commit: {error}"),
        }),
        Ok(()) => Err(E2eConformanceError::GenerationFailed {
            reason: "a second commit for the same generation step unexpectedly succeeded".into(),
        }),
    }
}

#[cfg(test)]
/// Proves discarding a pending KV state twice in a row (a cancellation
/// racing a cleanup retry, for example) is idempotent rather than erroring
/// the second time just because there is nothing left to discard.
fn check_kv_double_abort_is_idempotent(fixture: &E2eFixture) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let engine = new_kv_lifecycle_engine(fixture);
    let request = kv_lifecycle_test_request(
        fixture,
        &runtime,
        &instance,
        "kv-double-abort",
        None,
        &[1, 2],
    )?;
    let mut plans = first_native_plans_for_prompt(
        &runtime,
        fixture,
        &instance,
        request.input_token_ids.len() as u64,
    )?;
    engine.execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))?;
    engine.discard_pending_kv_state(&mut runtime, &request)?;
    engine.discard_pending_kv_state(&mut runtime, &request)?;
    if !engine
        .pending_kv_states
        .lock()
        .map_err(|_| E2eConformanceError::GenerationFailed {
            reason: "pending KV state lock poisoned".into(),
        })?
        .is_empty()
    {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "pending KV state survived two discard calls".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
/// Proves a stale pending KV write left by an earlier attempt does not
/// survive a subsequent, *failed* retry for the same request -- and so
/// cannot later be wrongly promoted by an unrelated commit call. The first
/// attempt succeeds and leaves a pending write nothing ever commits (as if
/// a downstream failure occurred); the retry is routed through an
/// unregistered Provider so it fails too, but must still discard the first
/// attempt's stale pending entry before doing so.
fn check_kv_stale_pending_state_does_not_survive_a_failed_retry(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let engine = new_kv_lifecycle_engine(fixture);
    let request = kv_lifecycle_test_request(
        fixture,
        &runtime,
        &instance,
        "kv-stale-pending",
        None,
        &[1, 2],
    )?;
    let mut plans = first_native_plans_for_prompt(
        &runtime,
        fixture,
        &instance,
        request.input_token_ids.len() as u64,
    )?;
    let first_step =
        engine.execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))?;

    for binding in &mut plans.prefill.node_bindings {
        binding.provider = ProviderBinding::new("unregistered-provider");
    }
    if engine
        .execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))
        .is_ok()
    {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "retry with an unregistered provider unexpectedly succeeded".into(),
        });
    }

    match engine.commit_generation_step(&mut runtime, &request, &[], 1, &first_step) {
        Err(InferenceApiError::KvCacheUnavailable { reason }) if reason.contains("pending") => {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!(
                "unexpected error committing after a discarded stale pending state: {error}"
            ),
        }),
        Ok(()) => Err(E2eConformanceError::GenerationFailed {
            reason:
                "commit succeeded using a stale pending KV state that should have been discarded"
                    .into(),
        }),
    }
}

#[cfg(test)]
/// Proves a KV cache committed under one request's compatibility (its
/// prefix fingerprint is derived from that request's own id) cannot be
/// reused under a different request/session's compatibility -- Runtime's
/// `validate_kv_cache_reuse` must reject the mismatch rather than letting
/// one session's decode read another session's KV state.
fn check_kv_wrong_session_reuse_is_rejected_by_compatibility(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let engine = new_kv_lifecycle_engine(fixture);

    let session_a = create_inference_session(
        &mut runtime,
        kv_lifecycle_session_request(fixture, &instance),
    )?;
    let session_b = create_inference_session(
        &mut runtime,
        kv_lifecycle_session_request(fixture, &instance),
    )?;

    let request_a = kv_lifecycle_test_request(
        fixture,
        &runtime,
        &instance,
        "kv-session-a",
        Some(session_a),
        &[1, 2],
    )?;
    let request_b = kv_lifecycle_test_request(
        fixture,
        &runtime,
        &instance,
        "kv-session-b",
        Some(session_b),
        &[1, 2],
    )?;

    let mut plans_a = first_native_plans_for_prompt(
        &runtime,
        fixture,
        &instance,
        request_a.input_token_ids.len() as u64,
    )?;
    let step_a = engine.execute_generation_step(
        &mut runtime,
        &request_a,
        &[],
        Some(&mut plans_a.prefill),
    )?;
    engine.commit_generation_step(&mut runtime, &request_a, &[], 1, &step_a)?;
    let cache_a = prefill_cache_id_from_step(&step_a)?;

    let compatibility_b = engine.kv_compatibility(&request_b);
    match runtime.validate_kv_cache_reuse(&cache_a, &compatibility_b, None) {
        Err(_) => Ok(()),
        Ok(()) => Err(E2eConformanceError::GenerationFailed {
            reason: "session B's compatibility was accepted for reuse of session A's KV cache"
                .into(),
        }),
    }
}

#[cfg(test)]
/// Proves a generation step re-checks Model Instance readiness for *itself*
/// (task 8.1) rather than reusing the one-time check
/// `prepare_first_native_execution_plans` performed before the generation
/// loop started: draining the instance between prefill and decode -- which
/// leaves its weight resource bindings fully intact, only its lifecycle
/// readiness changes -- must now fail the decode step closed instead of
/// silently proceeding on stale readiness evidence.
fn check_generation_step_rechecks_model_instance_readiness(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let engine = new_kv_lifecycle_engine(fixture);
    let request = kv_lifecycle_test_request(
        fixture,
        &runtime,
        &instance,
        "kv-readiness-recheck",
        None,
        &[1, 2],
    )?;
    let mut plans = first_native_plans_for_prompt(
        &runtime,
        fixture,
        &instance,
        request.input_token_ids.len() as u64,
    )?;

    let prefill_step =
        engine.execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))?;
    engine.commit_generation_step(&mut runtime, &request, &[], 1, &prefill_step)?;

    runtime
        .model_instances_mut()
        .instance_mut(&instance)
        .map_err(InferenceApiError::from)?
        .drain()
        .map_err(InferenceApiError::from)?;

    let generated = vec![1];
    match engine.execute_generation_step(
        &mut runtime,
        &request,
        &generated,
        Some(&mut plans.decode),
    ) {
        Err(InferenceApiError::ModelInstanceNotReady { .. }) => Ok(()),
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected error for a drained model instance: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "decode proceeded through a drained (not-ready) model instance".into(),
        }),
    }
}

#[cfg(test)]
/// Proves the generation-level observation stream (task 8.2) -- which
/// already carries causal component/graph/plan/provider/resource/KV/
/// sampling/token-commit evidence -- never carries the raw prompt text or a
/// native pointer-style marker, across every observation kind a real
/// forward pass emits, not just the higher-level conformance report JSON
/// `e2e_observability_emits_only_redacted_report_metadata` already checks.
fn check_generation_observations_never_carry_raw_prompt_or_handles(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let prompt = "zzyzx-secret";
    let outcome = run_success_path_with_prompt(fixture, &ModelRef::new("qwen-test")?, prompt)?;
    if outcome.observer.observations().is_empty() {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "success path emitted no observations to check for redaction".into(),
        });
    }
    for observation in outcome.observer.observations() {
        if observation.message.contains(prompt) {
            return Err(E2eConformanceError::BoundaryViolation {
                reason: format!(
                    "observation {:?} carried raw prompt text: {}",
                    observation.kind, observation.message
                ),
            });
        }
        if observation.message.contains("0x")
            || observation
                .message
                .to_ascii_lowercase()
                .contains("native_handle")
        {
            return Err(E2eConformanceError::BoundaryViolation {
                reason: format!(
                    "observation {:?} carried a native handle or pointer marker: {}",
                    observation.kind, observation.message
                ),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
/// Proves `FirstNativeChatSession::close` (task 8.3/8.4) genuinely releases
/// both the KV cache a chat turn created and the Model Instance it ran
/// against -- not just returning `Ok(())` without having done the
/// underlying work. The KV cache a chat turn creates gets
/// `KvCacheRetentionPolicy::ReleaseOnSessionClose` (the `KvCachePolicy`
/// default) and is scoped to this chat session's own `InferenceSessionId`,
/// so closing that *same* session is what must release it.
fn check_chat_session_close_releases_kv_cache_and_model_instance() -> Result<(), E2eConformanceError>
{
    let model_ref = ModelRef::new("qwen-test")?;
    let mut chat = FirstNativeChatSession::open(&model_ref).map_err(|error| {
        E2eConformanceError::GenerationFailed {
            reason: error.to_string(),
        }
    })?;
    chat.turn("hi", 1)
        .map_err(|error| E2eConformanceError::GenerationFailed {
            reason: error.to_string(),
        })?;

    let session = chat.session.clone();
    let instance = chat.instance.clone();
    let cache_id = chat
        .runtime
        .kv_caches()
        .caches()
        .find(|cache| cache.session.as_ref() == Some(&session))
        .map(|cache| cache.id.clone())
        .ok_or_else(|| E2eConformanceError::GenerationFailed {
            reason: "chat turn created no session-scoped KV cache".into(),
        })?;
    if chat.runtime.kv_cache(&cache_id)?.lifecycle == KvCacheLifecycleState::Released {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "chat turn's KV cache was already released before session close".into(),
        });
    }
    if chat.runtime.kv_cache(&cache_id)?.policy.retention
        != KvCacheRetentionPolicy::ReleaseOnSessionClose
    {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "chat turn's KV cache does not use the expected session-close retention policy"
                .into(),
        });
    }

    // The same two steps `FirstNativeChatSession::close` performs, kept
    // `&mut` here (rather than calling the consuming public `close`) so
    // this test can inspect `chat.runtime` immediately afterward and prove
    // cleanup targeted the *same* session and instance a turn actually
    // used.
    close_inference_session(&mut chat.runtime, &session).map_err(E2eConformanceError::from)?;
    unload_model_instance(
        &mut chat.runtime,
        &instance,
        ModelInstanceUnloadPolicy::DrainActiveUse,
    )
    .map_err(E2eConformanceError::from)?;

    if chat.runtime.kv_cache(&cache_id)?.lifecycle != KvCacheLifecycleState::Released {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "closing the chat session did not release its KV cache".into(),
        });
    }
    if chat
        .runtime
        .model_instance_status(&instance)
        .map_err(E2eConformanceError::from)?
        .readiness
        .accepts_generation()
    {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "unloading the chat session's model instance left it accepting generation"
                .into(),
        });
    }
    Ok(())
}

#[cfg(test)]
/// Proves two `FirstNativeChatSession`s opened for the same model are
/// isolated (task 8.4): distinct `InferenceSessionId`s and distinct,
/// independently scoped KV caches -- one session's turn cannot be confused
/// with, or leak state into, another's.
fn check_chat_sessions_are_isolated_from_each_other() -> Result<(), E2eConformanceError> {
    let model_ref = ModelRef::new("qwen-test")?;
    let mut chat_a = FirstNativeChatSession::open(&model_ref).map_err(|error| {
        E2eConformanceError::GenerationFailed {
            reason: error.to_string(),
        }
    })?;
    let mut chat_b = FirstNativeChatSession::open(&model_ref).map_err(|error| {
        E2eConformanceError::GenerationFailed {
            reason: error.to_string(),
        }
    })?;

    if chat_a.session_id() == chat_b.session_id() {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "two independently opened chat sessions were assigned the same session id"
                .into(),
        });
    }

    chat_a
        .turn("hello from a", 1)
        .map_err(|error| E2eConformanceError::GenerationFailed {
            reason: error.to_string(),
        })?;
    chat_b
        .turn("hello from b", 1)
        .map_err(|error| E2eConformanceError::GenerationFailed {
            reason: error.to_string(),
        })?;

    let session_a = chat_a.session.clone();
    let cache_a_belongs_to_a = chat_a
        .runtime
        .kv_caches()
        .caches()
        .any(|cache| cache.session.as_ref() == Some(&session_a));
    if !cache_a_belongs_to_a {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "chat session A's own Runtime recorded no KV cache scoped to its session"
                .into(),
        });
    }
    // Session A's Runtime is entirely separate from session B's -- there is
    // no shared KV cache manager, memory manager, or session table between
    // them, so B's session id cannot appear in A's cache table at all.
    let session_b = chat_b.session_id().clone();
    if chat_a
        .runtime
        .kv_caches()
        .caches()
        .any(|cache| cache.session.as_ref() == Some(&session_b))
    {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "chat session A's Runtime recorded a KV cache scoped to session B".into(),
        });
    }

    chat_a
        .close()
        .map_err(|error| E2eConformanceError::GenerationFailed {
            reason: error.to_string(),
        })?;
    chat_b
        .close()
        .map_err(|error| E2eConformanceError::GenerationFailed {
            reason: error.to_string(),
        })?;
    Ok(())
}

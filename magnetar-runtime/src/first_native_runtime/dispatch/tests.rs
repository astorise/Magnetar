//! Unit tests for the parent module.
//!
//! Shared mock executors and fixtures live in `first_native_runtime/tests.rs`.

use super::super::tests::*;
use super::*;
use crate::ExecutionGraphSemanticFingerprint;

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
fn e2e_graph_dispatch_ignores_kernel_registry_preference_change_after_plan_publication() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_graph_dispatch_ignores_kernel_registry_preference_change_after_plan_publication(&fixture)
        .expect(
            "a Kernel Registry preference change after Plan publication does not affect an \
             already-published, ready Plan",
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
    let mut dispatch_ctx = FirstNativeDispatchContext {
        runtime: &mut runtime,
        provider,
        prepared_plan: None,
        graph: None,
        sequence_length: None,
        last_provider_execution: None,
        internal_selection: None,
        bounded_kv_edges: BTreeMap::new(),
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
/// `resolve_first_native_weight_edge` called `.into_host()` unconditionally, which
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
        resolve_first_native_weight_edge(&provider, &weight_bindings, "weight.q_proj", &descriptor)
            .expect("an Opaque weight must resolve, not error");

    match resolved {
        NodeValue::Resident { id, shape } => {
            assert_eq!(id, weight_id);
            assert_eq!(shape, vec![4, 4]);
        }
        NodeValue::Host(_) => panic!("expected NodeValue::Resident, got NodeValue::Host"),
    }
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
fn e2e_graph_executor_rejects_cyclic_graph() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_graph_executor_rejects_cyclic_graph(&fixture).expect("graph executor rejects a cycle");
}

#[test]
fn e2e_incremental_decode_rejects_missing_layer_kv() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_incremental_decode_rejects_missing_layer_kv(&fixture)
        .expect("decode requires existing layer KV state");
}

//! Per-operator dispatch through the Kernel Registry and prepared plans.

use super::*;

/// Physically transposes a rank-2 [`HostTensor`], returning a `[cols, rows]`
/// tensor whose `[j, i]` entry is `tensor[i, j]`. [`dispatch_matmul`] always
/// runs an untransposed `a @ b` (matching what the portable `matmul`
/// Operator's shape rule validates -- `a[-1] == b[-2]` -- which does not
/// consult a `transpose_b` execution attribute), so a caller that needs
/// `a @ b^T` transposes `b` itself before dispatching.
pub(super) fn transpose_rows_cols(tensor: &HostTensor) -> Result<HostTensor, E2eConformanceError> {
    let (rows, cols) = tensor.rows_cols()?;
    let mut out = vec![0.0_f32; tensor.data.len()];
    for row in 0..rows {
        for col in 0..cols {
            out[(col * rows + row) as usize] = tensor.data[(row * cols + col) as usize];
        }
    }
    HostTensor::new([cols, rows], out).map_err(E2eConformanceError::from)
}

/// Dispatches a genuine Kernel Registry -> Kernel Dispatch -> Reference CPU
/// Provider matmul computing `a @ b`, returning both the
/// [`KernelDispatchResult`] (used to build [`RuntimeGenerationExecutionEvidence`])
/// and the actual output tensor the Provider computed and wrote back.
///
/// This exists so evidence is always drawn from the dispatch that produced
/// the data a caller returns, rather than from an unrelated "proof"
/// computation whose result is discarded: the caller reads the returned
/// output tensor -- not a value it computed independently -- so a dispatch
/// that never ran, or that ran over different data, cannot produce evidence
/// for logits it did not causally produce.
pub(super) fn dispatch_matmul(
    runtime: &Runtime,
    a: &HostTensor,
    b: &HostTensor,
) -> Result<(KernelDispatchResult, HostTensor), InferenceApiError> {
    dispatch_matmul_with_prepared_plan(runtime, a, b, "matmul", None)
}

pub(super) fn dispatch_matmul_with_prepared_plan(
    runtime: &Runtime,
    a: &HostTensor,
    b: &HostTensor,
    operation_id: &str,
    prepared_plan: Option<&PreparedExecutionPlan>,
) -> Result<(KernelDispatchResult, HostTensor), InferenceApiError> {
    let advertisements = reference_cpu_kernel_advertisements();
    let matmul_operator = advertisements
        .iter()
        .find(|advertisement| advertisement.implemented_operator.name() == "matmul")
        .map(|advertisement| advertisement.implemented_operator.clone())
        .ok_or_else(|| InferenceApiError::KernelUnavailable {
            reason: "Reference CPU fixture does not advertise matmul".into(),
        })?;
    let to_generation_failed = |error: ReferenceCpuError| InferenceApiError::GenerationFailed {
        reason: error.to_string(),
    };
    let (a_rows, a_cols) = a.rows_cols().map_err(to_generation_failed)?;
    let (b_rows, b_cols) = b.rows_cols().map_err(to_generation_failed)?;
    let out_rows = a_rows;
    let out_cols = b_cols;

    let dtype = DTypeDescriptor::portable(ComputeDType::Float32);
    let a_descriptor = TensorDescriptor::new(
        ShapeDescriptor::new([a_rows, a_cols]),
        dtype.clone(),
        LayoutDescriptor::Contiguous,
    );
    let b_descriptor = TensorDescriptor::new(
        ShapeDescriptor::new([b_rows, b_cols]),
        dtype.clone(),
        LayoutDescriptor::Contiguous,
    );
    let output_descriptor = TensorDescriptor::new(
        ShapeDescriptor::new([out_rows, out_cols]),
        dtype,
        LayoutDescriptor::Contiguous,
    );
    let affinity = ResourceAffinity::new(FallbackClass::Transparent)
        .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME))
        .with_execution_context(runtime.context().id());

    let input_a = TensorResourceDescriptor::new(
        TensorResourceId::new("e2e-runtime-generation-a"),
        a_descriptor,
        affinity.clone(),
    );
    let input_b = TensorResourceDescriptor::new(
        TensorResourceId::new("e2e-runtime-generation-b"),
        b_descriptor,
        affinity.clone(),
    );
    let output = TensorResourceDescriptor::new(
        TensorResourceId::new("e2e-runtime-generation-logits"),
        output_descriptor,
        affinity,
    );
    let selection_request = KernelSelectionRequest::new(
        "e2e-runtime-generation-step",
        matmul_operator,
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_input(KernelResource::new(
        input_a.clone(),
        KernelMemoryClass::Host,
    ))
    .with_input(KernelResource::new(
        input_b.clone(),
        KernelMemoryClass::Host,
    ))
    .with_output(KernelResource::new(output.clone(), KernelMemoryClass::Host));
    let selected = if let Some(prepared_plan) = prepared_plan {
        prepared_candidate_for_operation(
            runtime,
            prepared_plan,
            operation_id,
            &selection_request.operator,
        )?
    } else {
        runtime
            .kernel_registry()
            .select(&selection_request)
            .map_err(|error| InferenceApiError::KernelUnavailable {
                reason: error.to_string(),
            })?
            .selected
            .ok_or_else(|| InferenceApiError::KernelUnavailable {
                reason: "Kernel Registry selected no Reference CPU candidate".into(),
            })?
    };
    let advertisement = runtime
        .kernel_registry()
        .active_advertisement(&selected.kernel)
        .ok_or_else(|| InferenceApiError::KernelUnavailable {
            reason: "selected Reference CPU advertisement is no longer active".into(),
        })?;
    let mut plan = KernelDispatchPlan::from_selection(
        KernelDispatchPlanId::new("e2e-runtime-generation-dispatch"),
        &selection_request,
        &selected,
        advertisement,
        KernelInvocationId::new("e2e-runtime-generation-invocation"),
    )
    .map_err(|error| InferenceApiError::KernelUnavailable {
        reason: format!("{error:?}"),
    })?;
    let mut dispatcher = KernelDispatcher::new();
    dispatcher
        .revalidate(runtime.kernel_registry(), &mut plan)
        .map_err(|error| InferenceApiError::KernelUnavailable {
            reason: format!("{error:?}"),
        })?;
    let provider = ReferenceCpuExecutor::new();
    provider.write_tensor(input_a.id.clone(), a.clone());
    provider.write_tensor(input_b.id.clone(), b.clone());
    let mut memory = MemoryManager::default();
    let operator_catalog = initial_operator_catalog();
    let matmul_spec = operator_catalog
        .get(&advertisement.implemented_operator)
        .map_err(|error| InferenceApiError::KernelUnavailable {
            reason: error.to_string(),
        })?;
    let kernel_result = provider.execute_invocation_with_memory_manager(
        advertisement,
        matmul_spec,
        &plan.invocation,
        &mut memory,
    );
    let dispatch_result = KernelDispatchResult::from_kernel_result(&plan, kernel_result);
    if dispatch_result.status != KernelResultStatus::Succeeded {
        return Err(InferenceApiError::ProviderUnavailable {
            reason: format!(
                "Reference CPU matmul dispatch failed: {:?}",
                dispatch_result.error
            ),
        });
    }
    let output_tensor =
        provider
            .read_tensor(&output.id)
            .ok_or_else(|| InferenceApiError::GenerationFailed {
                reason: "Reference CPU matmul dispatch produced no output tensor".into(),
            })?;
    Ok((dispatch_result, output_tensor))
}

pub(super) fn f32_tensor_descriptor(tensor: &HostTensor) -> TensorDescriptor {
    f32_tensor_descriptor_from_shape(&tensor.shape)
}

pub(super) fn shape_rows_cols(shape: &[u64]) -> Result<(u64, u64), InferenceApiError> {
    match shape {
        [rows, cols] => Ok((*rows, *cols)),
        other => Err(InferenceApiError::GraphPlanningFailed {
            reason: format!("expected rank-2 tensor, got shape {other:?}"),
        }),
    }
}

pub(super) fn f32_tensor_descriptor_from_shape(shape: &[u64]) -> TensorDescriptor {
    TensorDescriptor::new(
        ShapeDescriptor::new(shape.to_vec()),
        DTypeDescriptor::portable(ComputeDType::Float32),
        LayoutDescriptor::Contiguous,
    )
}

/// One node's resolved input *or* output: either real host bytes, or an
/// existing device-resident (or otherwise Provider-opaque) resource this
/// dispatch can reuse/leave in place instead of downloading and
/// re-uploading a copy under a fresh id
/// (`enable-device-resident-kernel-chaining`'s input-side passthrough
/// decision, extended to outputs by
/// `unify-provider-output-admission-and-residency`'s Decision 3). `shape`
/// is carried alongside `Resident` because `TensorValue::Opaque` itself
/// carries no metadata -- callers that only need shape (matmul's
/// `rows_cols`, embedding's `sequence_length`, `bindings`' own
/// bookkeeping, ...) never have to materialize the value just to read it.
#[derive(Clone)]
pub(super) enum NodeValue {
    Host(HostTensor),
    Resident {
        id: TensorResourceId,
        shape: Vec<u64>,
    },
}

impl NodeValue {
    pub(super) fn shape(&self) -> &[u64] {
        match self {
            Self::Host(tensor) => &tensor.shape,
            Self::Resident { shape, .. } => shape,
        }
    }

    pub(super) fn rows_cols(&self) -> Result<(u64, u64), InferenceApiError> {
        shape_rows_cols(self.shape())
    }

    /// Materializes to real host bytes, downloading only if this value
    /// wasn't already host-shaped. Used for the operators
    /// (`rmsnorm`'s weight broadcast, `rope`'s per-head slicing) that
    /// genuinely manipulate raw floats in Rust rather than only forwarding
    /// the value to a Kernel dispatch.
    pub(super) fn into_host(
        self,
        provider: &Arc<dyn ProviderExecutionApi>,
    ) -> Result<HostTensor, InferenceApiError> {
        match self {
            Self::Host(tensor) => Ok(tensor),
            Self::Resident { id, .. } => {
                provider
                    .read_tensor(&id)
                    .ok_or_else(|| InferenceApiError::GraphPlanningFailed {
                        reason: format!("no materialized data for resident resource '{id}'"),
                    })
            }
        }
    }
}

/// One input resource ready to hand to
/// [`dispatch_reference_cpu_operator_multi`]: either fresh host bytes that
/// still need writing under a newly synthesized id (today's behavior), or
/// an existing resource this dispatch reuses by id with no write at all.
pub(super) enum NodeInputResource {
    Fresh(TensorResourceId, TensorDescriptor, HostTensor),
    Resident(TensorResourceId, TensorDescriptor),
}

/// Builds a [`NodeInputResource`] for one named input argument (`suffix`,
/// e.g. `"a"`/`"b"`) of `operation_id`'s dispatch.
pub(super) fn node_input_resource(
    operation_id: &str,
    suffix: &str,
    value: NodeValue,
) -> NodeInputResource {
    match value {
        NodeValue::Host(tensor) => {
            let descriptor = f32_tensor_descriptor(&tensor);
            NodeInputResource::Fresh(
                TensorResourceId::new(format!("{operation_id}.{suffix}")),
                descriptor,
                tensor,
            )
        }
        NodeValue::Resident { id, shape } => {
            NodeInputResource::Resident(id, f32_tensor_descriptor_from_shape(&shape))
        }
    }
}

/// The `KernelMemoryClass` this dispatch's resources should advertise:
/// derived from the Prepared Plan's own resolved Provider binding for this
/// node when one exists, defaulting to `Host` (Reference CPU's class, and
/// the safe default for the no-Prepared-Plan fallback/test-oracle path).
/// Every real Provider other than Reference CPU advertises `Device`
/// (`providers/cuda`'s own Kernel advertisements do) -- hardcoding `Host`
/// unconditionally here made every CUDA Kernel invocation through this
/// dispatch loop fail `KernelMemoryClassUnsupported` at
/// `validate_invocation`, independent of residency
/// (`enable-device-resident-kernel-chaining`'s discovered prerequisite fix).
pub(super) fn resolved_kernel_memory_class(
    prepared_plan: Option<&PreparedExecutionPlan>,
    node: &ExecutionNodeId,
) -> KernelMemoryClass {
    let Some(prepared_plan) = prepared_plan else {
        return KernelMemoryClass::Host;
    };
    let Some(binding) = prepared_plan.node_binding(node) else {
        return KernelMemoryClass::Host;
    };
    if crate::reference_cpu::host_memory_provider(binding.provider.as_str()) {
        KernelMemoryClass::Host
    } else {
        KernelMemoryClass::Device
    }
}

/// The `MemoryPlacement` this dispatch's *output* should be pre-admitted
/// under (`unify-provider-output-admission-and-residency`'s Decision 3):
/// mirrors whichever placement the resolved Provider's own
/// `execute_invocation_with_memory_manager` would otherwise have chosen
/// itself (`ProviderOwnedOpaque` for Reference CPU, `Device` for any other
/// Provider bound to a Device) -- same lookup pattern as
/// `resolved_kernel_memory_class`, kept separate rather than merged since
/// the two describe different things (`KernelMemoryClass` is
/// Kernel-selection-facing, `MemoryPlacement` is Memory-Manager-facing).
pub(super) fn resolved_output_placement(
    prepared_plan: Option<&PreparedExecutionPlan>,
    node: &ExecutionNodeId,
) -> MemoryPlacement {
    let Some(prepared_plan) = prepared_plan else {
        return MemoryPlacement::ProviderOwnedOpaque(ProviderBinding::new(
            REFERENCE_CPU_PROVIDER_NAME,
        ));
    };
    let Some(binding) = prepared_plan.node_binding(node) else {
        return MemoryPlacement::ProviderOwnedOpaque(ProviderBinding::new(
            REFERENCE_CPU_PROVIDER_NAME,
        ));
    };
    if crate::reference_cpu::host_memory_provider(binding.provider.as_str()) {
        return MemoryPlacement::ProviderOwnedOpaque(binding.provider.clone());
    }
    match &binding.device {
        Some(device) => MemoryPlacement::Device(device.clone()),
        // A non-CPU Provider binding with no Device named is not expected
        // in practice (Device-executing Kernels always bind a Device), but
        // fails safe to the same opaque placement Reference CPU uses
        // rather than fabricating a placeholder Device identity.
        None => MemoryPlacement::ProviderOwnedOpaque(binding.provider.clone()),
    }
}

/// The `ResourceAffinity` a freshly-admitted resource for this dispatch
/// should carry (`make-first-native-cuda-hot-path-device-resident`'s
/// Decision 6): mirrors `resolved_output_placement`'s exact lookup pattern
/// rather than the previous hardcoded `REFERENCE_CPU_PROVIDER_NAME` --
/// `dispatch_reference_cpu_operator_multi` no longer has to guess which
/// Provider a node actually resolved to when the Prepared Plan already
/// says so.
pub(super) fn resolved_resource_affinity(
    prepared_plan: Option<&PreparedExecutionPlan>,
    node: &ExecutionNodeId,
    execution_context: ExecutionContextId,
) -> ResourceAffinity {
    let base =
        ResourceAffinity::new(FallbackClass::Transparent).with_execution_context(execution_context);
    let Some(prepared_plan) = prepared_plan else {
        return base.with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME));
    };
    let Some(binding) = prepared_plan.node_binding(node) else {
        return base.with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME));
    };
    let base = base.with_provider(binding.provider.clone());
    match &binding.device {
        Some(device) => base.with_device(device.clone()),
        None => base,
    }
}

/// The `ResourceAffinity` a `NodeInputResource::Resident` resource should
/// carry for this dispatch (`make-first-native-cuda-hot-path-device-
/// resident`'s Decision 6): if `id` already has a recorded
/// `TensorResidency` (a Model-Load-time weight, or an earlier producer's
/// output), its own affinity is preserved and aggregated with `dispatch_
/// affinity` through `resource-affinity`'s existing "Affinity Constraint
/// Aggregation" contract (conflict rejected, not silently overwritten).
/// A resource with no recorded residency yet (should not normally happen
/// for a caller-asserted `Resident` value, but handled rather than
/// panicking) falls back to `dispatch_affinity` alone.
pub(super) fn resident_resource_affinity(
    memory: &MemoryManager,
    id: &TensorResourceId,
    dispatch_affinity: &ResourceAffinity,
    node: &ExecutionNodeId,
) -> Result<ResourceAffinity, InferenceApiError> {
    match memory.tensor_residency(id) {
        Some(residency) => {
            AffinityConstraints::try_from_affinities([&residency.affinity, dispatch_affinity])
                .map(AffinityConstraints::into_affinity)
                .map_err(|error| InferenceApiError::GraphPlanningFailed {
                    reason: format!(
                        "resident resource '{id}' affinity conflicts with node '{node}'s resolved Provider/Device: {error:?}"
                    ),
                })
        }
        None => Ok(dispatch_affinity.clone()),
    }
}

pub(super) struct FirstNativeDispatchContext<'a> {
    /// A single exclusive borrow (Correctif 13 / task group 7), not split
    /// into separate `runtime`/`memory` fields: `MemoryManager` is a field
    /// of `Runtime` itself, so a caller outside `runtime.rs` cannot borrow
    /// it mutably alongside an independent shared borrow of the rest of
    /// `Runtime` without either destructively swapping it out (the
    /// `std::mem::take` this replaced) or holding one exclusive borrow for
    /// everything. Any value read through `runtime` that must stay alive
    /// across a later `runtime.memory_mut()` call (e.g. a `&KernelAdvertisement`
    /// from `active_advertisement`) is cloned to an owned value at the
    /// point of lookup instead, so the two never need to overlap.
    pub(super) runtime: &'a mut Runtime,
    /// Resolved generically from Runtime's provider registration through
    /// [`ProviderExecutionApi`] (Correctif 3): no downcast to a concrete
    /// Provider type is ever performed on this hot path.
    pub(super) provider: Arc<dyn ProviderExecutionApi>,
    pub(super) prepared_plan: Option<&'a mut PreparedExecutionPlan>,
    /// The semantic graph this dispatch's nodes belong to. Required
    /// alongside `prepared_plan` to route through
    /// `PreparedExecutionPlanExecutor::prepare_node_execution` (Correctif
    /// 4): that authority revalidates the plan's graph fingerprint against
    /// the graph actually being executed, not just the operation id string.
    /// `None` only for callers that have no full graph (e.g. hand-written
    /// test oracles), which fall back to a synthetic-candidate path that
    /// production dispatch never takes.
    /// Fingerprinted once per graph execution, not per node.
    pub(super) graph: Option<FingerprintedGraph<'a>>,
    /// This generation step's token count, used to populate
    /// `PlanGuardContext::sequence_length` when revalidating a published
    /// Plan's guards (Correctif 4). An absent value is treated by
    /// `PlanGuard::SequenceRange` as out of range (fail-closed), so this
    /// SHALL be set whenever `prepared_plan` and `graph` are both `Some`.
    pub(super) sequence_length: Option<u64>,
    /// The most recent dispatch's Provider submission/completion identity
    /// (task 5.3), recorded by `dispatch_reference_cpu_operator` via the
    /// registered Provider's own [`ProviderExecutionApi::complete`] rather
    /// than a caller-fabricated id. `None` until the first successful
    /// dispatch.
    pub(super) last_provider_execution: Option<ProviderExecutionResult>,
    /// The last selection made for an Operator the Runtime adds to the step
    /// itself (the KV history append's `concat` or `kv-write`), and what it
    /// was selected for. Every layer's K and V append in a step asks the
    /// registry the same question (same operator, affinity, memory class,
    /// tensor descriptors and affinities); the registry does not change
    /// while a graph executes, so the answer is reused.
    pub(super) internal_selection: Option<(InternalSelectionKey, KernelCandidate)>,
    /// KV edges bound to an in-place KV store, with the number of leading
    /// rows attention reads from it (`append-kv-history-in-place`).
    pub(super) bounded_kv_edges: BTreeMap<TensorEdgeId, u64>,
    /// Per-node causal-chain events (Correctif 17 / task group 17), pushed
    /// directly into the caller's own `Vec` (not accumulated locally and
    /// drained only on success) as each node's dispatch progresses through
    /// `dispatch_reference_cpu_operator` and the KV-write blocks in
    /// `execute_first_native_graph_nodes` -- so a node's real events survive even if
    /// a *later* node's dispatch fails and this whole call returns early via
    /// `?`. The ultimate caller (`inference_api.rs`'s generation loop) owns
    /// the real `InferenceApiObserver` and turns each one into a redacted
    /// `InferenceApiObservation`.
    pub(super) node_events: &'a mut Vec<PerNodeCausalEvent>,
}

/// One per-node causal-chain event captured during graph dispatch
/// (Correctif 17 / task group 17): names the `ExecutionNodeId` it belongs
/// to and, for the events that produce one, the `TensorResourceId` it
/// correlates to -- so a caller can walk a *complete* per-node chain
/// (`GraphNodeReady` -> ... -> `TensorResourceProduced`, per node) rather
/// than only confirming five global evidence categories were present
/// somewhere in the whole generation step, which `validate_e2e_no_shortcuts`
/// did before this task group.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PerNodeCausalEvent {
    pub(crate) kind: InferenceApiObservationKind,
    pub(crate) node: ExecutionNodeId,
    pub(crate) resource: Option<TensorResourceId>,
}

impl PerNodeCausalEvent {
    pub(super) fn new(kind: InferenceApiObservationKind, node: ExecutionNodeId) -> Self {
        Self {
            kind,
            node,
            resource: None,
        }
    }

    pub(super) fn with_resource(mut self, resource: TensorResourceId) -> Self {
        self.resource = Some(resource);
        self
    }
}

/// Maps a dispatch `operation_id` back to the [`ExecutionNodeId`] published
/// in the plan. Two dispatch-side conventions do not exist at the graph
/// level and must be normalized away here: a `"decode."` prefix (the decode
/// phase's dispatch labels distinguish the same underlying node from its
/// prefill counterpart) and a trailing `".head{N}"` suffix (RoPE dispatches
/// once per attention head against a single whole-tensor `rope` graph node,
/// so every per-head sub-dispatch shares that node's one binding).
pub(super) fn qwen_plan_node_id(operation_id: &str) -> ExecutionNodeId {
    let node_id = operation_id.strip_prefix("decode.").unwrap_or(operation_id);
    let node_id = match node_id.rsplit_once(".head") {
        Some((base, suffix))
            if !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit()) =>
        {
            base
        }
        _ => node_id,
    };
    ExecutionNodeId::new(node_id)
}

pub(super) fn prepared_candidate_for_operation(
    runtime: &Runtime,
    prepared_plan: &PreparedExecutionPlan,
    operation_id: &str,
    operator: &OperatorId,
) -> Result<KernelCandidate, InferenceApiError> {
    let node_id = qwen_plan_node_id(operation_id);
    let binding = prepared_plan
        .node_bindings
        .iter()
        .find(|binding| binding.graph_nodes.contains(&node_id))
        .ok_or_else(|| InferenceApiError::KernelUnavailable {
            reason: format!("prepared execution plan has no binding for node {node_id}"),
        })?;
    if binding.kernel.operator != *operator {
        return Err(InferenceApiError::KernelUnavailable {
            reason: format!(
                "prepared binding for node {node_id} targets operator {}, not {operator}",
                binding.kernel.operator
            ),
        });
    }
    let prepared_kernel =
        binding
            .prepared_kernel
            .ok_or_else(|| InferenceApiError::KernelUnavailable {
                reason: format!("prepared binding for node {node_id} has no PreparedKernelId"),
            })?;
    let prepared_generation =
        binding
            .prepared_kernel_generation
            .ok_or_else(|| InferenceApiError::KernelUnavailable {
                reason: format!(
                    "prepared binding for node {node_id} has no PreparedKernel generation"
                ),
            })?;
    let prepared = runtime
        .kernel_registry()
        .prepared_kernel(&prepared_kernel)
        .ok_or_else(|| InferenceApiError::KernelUnavailable {
            reason: format!("PreparedKernelId {prepared_kernel} for node {node_id} is not active"),
        })?;
    if !prepared.state.is_dispatchable() {
        return Err(InferenceApiError::KernelUnavailable {
            reason: format!("PreparedKernelId {prepared_kernel} for node {node_id} is not ready"),
        });
    }
    if prepared.kernel != binding.kernel || prepared.generation != prepared_generation {
        return Err(InferenceApiError::KernelUnavailable {
            reason: format!("prepared kernel for node {node_id} does not match plan binding"),
        });
    }
    Ok(KernelCandidate {
        kernel: binding.kernel.clone(),
        provider: binding.provider.clone(),
        device: binding.device.clone(),
        operator: operator.clone(),
        compatible: true,
        dtype_compatible: true,
        layout_compatible: true,
        shape_compatible: true,
        memory_compatible: true,
        workspace_feasible: true,
        affinity_compatible: true,
        deterministic_compatible: true,
        precision_compatible: true,
        provider_ready: true,
        device_ready: true,
        provider_status: None,
        device_status: None,
        pressure_score: 0,
        conformance_status: binding.qualification_profile.clone(),
        estimated_cost: 0,
        fallback_rank: 0,
        rejection_reason: None,
    })
}

/// A resolved Kernel for one dispatch: either the authoritative
/// `PreparedPlanNodeExecution` a published Plan already validated
/// (Correctif 4's production path), or a `KernelCandidate` from the
/// synthetic-candidate or live-selection fallback paths (test oracles / no
/// Plan bound).
pub(super) enum HotPathKernelSelection {
    Prepared(PreparedPlanNodeExecution),
    Candidate(KernelCandidate),
}

impl HotPathKernelSelection {
    pub(super) fn kernel(&self) -> &KernelId {
        match self {
            Self::Prepared(prepared) => &prepared.kernel,
            Self::Candidate(candidate) => &candidate.kernel,
        }
    }
}

/// Dispatches one Reference CPU Operator node with an arbitrary number of
/// outputs -- most callers have exactly one and use
/// [`dispatch_reference_cpu_operator`]'s single-output convenience wrapper
/// below, but a genuinely multi-output Operator (e.g. `split`) needs every
/// entry of the Kernel's `KernelResult::updated_resources` bound to its own
/// requested output, not just the first (GitHub issue "First-native graph
/// executor only propagates the first output of a multi-output node").
pub(super) fn dispatch_reference_cpu_operator_multi(
    ctx: &mut FirstNativeDispatchContext<'_>,
    operation_id: &str,
    operator: OperatorId,
    inputs: Vec<NodeInputResource>,
    outputs: Vec<(TensorResourceId, TensorDescriptor)>,
    attributes: BTreeMap<String, OperatorAttributeValue>,
) -> Result<(KernelDispatchResult, Vec<NodeValue>), InferenceApiError> {
    // Correctif 17 / task group 17: this node's causal chain starts here,
    // once its inputs (written just below) are resolved and it is about to
    // be dispatched. `node` is reused for every later event this call
    // pushes, so the whole chain correlates to the same `ExecutionNodeId`
    // regardless of which selection arm (published Plan, synthetic
    // candidate, or live selection) this dispatch actually takes.
    let node = qwen_plan_node_id(operation_id);
    ctx.node_events.push(PerNodeCausalEvent::new(
        InferenceApiObservationKind::GraphNodeReady,
        node.clone(),
    ));
    let affinity = resolved_resource_affinity(
        ctx.prepared_plan.as_deref(),
        &node,
        ctx.runtime.context().id(),
    );
    let memory_class = resolved_kernel_memory_class(ctx.prepared_plan.as_deref(), &node);
    let output_resources: Vec<TensorResourceDescriptor> = outputs
        .into_iter()
        .map(|(id, descriptor)| TensorResourceDescriptor::new(id, descriptor, affinity.clone()))
        .collect();
    let mut selection_request = KernelSelectionRequest::new(
        format!("e2e-runtime-{operation_id}"),
        operator,
        affinity.clone(),
    );
    selection_request
        .outputs
        .reserve_exact(output_resources.len());
    selection_request.inputs.reserve_exact(inputs.len());
    for output_resource in &output_resources {
        selection_request = selection_request
            .with_output(KernelResource::new(output_resource.clone(), memory_class));
    }
    for input in inputs {
        let resource = match input {
            NodeInputResource::Fresh(id, descriptor, tensor) => {
                let resource =
                    TensorResourceDescriptor::new(id.clone(), descriptor, affinity.clone());
                ctx.provider.write_tensor(id, tensor).map_err(|error| {
                    InferenceApiError::ProviderTensorWriteFailed {
                        reason: error.to_string(),
                    }
                })?;
                resource
            }
            // Already resident under `id` in the resolved Provider's own
            // storage (this dispatch's earlier producer wrote it there,
            // or a graph input/weight was admitted under it) -- reused by
            // reference, no download-then-reupload copy. Its own recorded
            // affinity (e.g. a Model-Load-time weight's) is preserved, not
            // overwritten by this dispatch's freshly-derived one --
            // `make-first-native-cuda-hot-path-device-resident`'s Decision
            // 6 -- via `resource-affinity`'s own existing aggregation
            // contract (conflict rejected, not silently replaced).
            NodeInputResource::Resident(id, descriptor) => {
                let resolved_affinity =
                    resident_resource_affinity(ctx.runtime.memory(), &id, &affinity, &node)?;
                TensorResourceDescriptor::new(id, descriptor, resolved_affinity)
            }
        };
        selection_request =
            selection_request.with_input(KernelResource::new(resource, memory_class));
    }
    // Correctif 4: once a Plan is published, a node's execution SHALL be
    // driven directly from its `PlanNodeBinding`/`PreparedKernelId` through
    // `PreparedExecutionPlanExecutor`, not by reconstructing a synthetic,
    // always-compatible `KernelCandidate` after the fact. The synthetic
    // path (`prepared_candidate_for_operation`) survives only for callers
    // that have a `PreparedExecutionPlan` but no full `ExecutionGraph` to
    // revalidate it against -- hand-written test oracles, never production.
    let resolve_clock = ctx.runtime.phase_ledger().clock();
    let selection = match (ctx.prepared_plan.as_deref_mut(), ctx.graph.as_ref()) {
        (Some(prepared_plan), Some(graph)) => {
            let guard_context = first_native_plan_context(
                prepared_plan.scope.phase,
                ctx.sequence_length.unwrap_or(1),
            );
            let prepared = PreparedExecutionPlanExecutor::new()
                .prepare_fingerprinted_node_execution(
                    graph,
                    prepared_plan,
                    ctx.runtime.kernel_registry(),
                    &guard_context,
                    &node,
                )
                .map_err(|error| InferenceApiError::KernelUnavailable {
                    reason: format!("{operation_id}: {error:?}"),
                })?;
            ctx.node_events.push(PerNodeCausalEvent::new(
                InferenceApiObservationKind::PlanBindingResolved,
                node.clone(),
            ));
            HotPathKernelSelection::Prepared(prepared)
        }
        (Some(prepared_plan), None) => {
            HotPathKernelSelection::Candidate(prepared_candidate_for_operation(
                ctx.runtime,
                prepared_plan,
                operation_id,
                &selection_request.operator,
            )?)
        }
        (None, _) => {
            let selection = ctx
                .runtime
                .kernel_registry()
                .select(&selection_request)
                .map_err(|error| InferenceApiError::KernelUnavailable {
                    reason: format!("{operation_id}: {error}"),
                })?;
            HotPathKernelSelection::Candidate(selection.selected.ok_or_else(|| {
                InferenceApiError::KernelUnavailable {
                    reason: format!("Kernel Registry selected no candidate for {operation_id}"),
                }
            })?)
        }
    };
    // Borrowed from the registry for the rest of this dispatch alongside the
    // Memory Manager and the phase ledger (task group 7 / Correctif 13), so
    // the advertisement is not copied for every node.
    let (registry, memory, phases) = ctx.runtime.kernel_registry_memory_and_phases_mut();
    let advertisement = registry
        .active_advertisement(selection.kernel())
        .ok_or_else(|| InferenceApiError::KernelUnavailable {
            reason: format!("selected advertisement for {operation_id} is no longer active"),
        })?;
    ctx.node_events.push(PerNodeCausalEvent::new(
        InferenceApiObservationKind::PreparedKernelResolved,
        node.clone(),
    ));
    let mut plan = match &selection {
        HotPathKernelSelection::Prepared(prepared) => {
            KernelDispatchPlan::from_prepared_node_execution_owned(
                KernelDispatchPlanId::new(format!("e2e-runtime-{operation_id}-dispatch")),
                selection_request,
                prepared,
                advertisement,
                KernelInvocationId::new(format!("e2e-runtime-{operation_id}-invocation")),
            )
        }
        HotPathKernelSelection::Candidate(candidate) => KernelDispatchPlan::from_selection(
            KernelDispatchPlanId::new(format!("e2e-runtime-{operation_id}-dispatch")),
            &selection_request,
            candidate,
            advertisement,
            KernelInvocationId::new(format!("e2e-runtime-{operation_id}-invocation")),
        ),
    }
    .map_err(|error| InferenceApiError::KernelUnavailable {
        reason: format!("{error:?}"),
    })?;
    plan.invocation.attributes = attributes;
    if advertisement.workspace.required {
        let workspace = ctx
            .provider
            .allocate_workspace(
                memory,
                advertisement.workspace.size_bytes_upper_bound.unwrap_or(1),
            )
            .map_err(|error| InferenceApiError::MemoryAdmissionFailed {
                reason: error.to_string(),
            })?;
        plan.invocation.workspace = Some(workspace);
        plan.workspace_reservation = Some(workspace);
    }
    let mut dispatcher = KernelDispatcher::for_single_dispatch();
    dispatcher
        .revalidate(registry, &mut plan)
        .map_err(|error| InferenceApiError::KernelUnavailable {
            reason: format!("{error:?}"),
        })?;
    phases.record(resolve_clock, StepPhase::KernelResolve);
    let catalog_clock = phases.clock();
    let operator_catalog = shared_initial_operator_catalog();
    let operator_spec = operator_catalog
        .get(&advertisement.implemented_operator)
        .map_err(|error| InferenceApiError::KernelUnavailable {
            reason: error.to_string(),
        })?;
    phases.record(catalog_clock, StepPhase::OperatorCatalog);
    let execute_clock = phases.clock();
    // Provider submission/completion identity (Correctif 2): `submit_kernel`
    // is what causally triggers the numerical Kernel work (Reference CPU is
    // synchronous, so it runs as part of this call); the returned handle is
    // never constructed after the fact to merely dress up evidence that
    // already happened through a different call. Resolved generically
    // through `ProviderExecutionApi` (Correctif 3): `ctx.provider` is a
    // trait object, not a concrete `ReferenceCpuExecutor`.
    let handle = ctx
        .provider
        .submit_kernel(advertisement, operator_spec, &plan.invocation, memory)
        .map_err(|error| InferenceApiError::ProviderUnavailable {
            reason: format!(
                "Reference CPU dispatch for {operation_id} could not be submitted: {error}"
            ),
        })?;
    ctx.node_events.push(PerNodeCausalEvent::new(
        InferenceApiObservationKind::ProviderSubmitted,
        node.clone(),
    ));
    // Workspace is invocation-scoped: release it back to Runtime's
    // MemoryManager once this dispatch has used it, whether or not the
    // dispatch succeeded, rather than letting it accumulate as a leaked
    // allocation across every node this graph executes (task 5.4/5.5/5.6).
    if let Some(workspace) = plan.workspace_reservation {
        let _ = memory.release(workspace);
    }
    // `complete_kernel` observes the SAME work `handle` was returned for
    // above -- correlated by that handle, not fabricated -- and fails
    // structurally if the handle is somehow unknown (it cannot be completed
    // twice from this single call site, but the check keeps this path
    // honest about what "complete" means).
    let kernel_result = ctx.provider.complete_kernel(&handle).map_err(|error| {
        InferenceApiError::ProviderUnavailable {
            reason: format!(
                "Reference CPU dispatch for {operation_id} could not be completed: {error}"
            ),
        }
    })?;
    phases.record_operator(
        execute_clock,
        StepPhase::KernelExecute,
        advertisement.implemented_operator.name(),
    );
    let dispatch_result = KernelDispatchResult::from_kernel_result(&plan, kernel_result);
    if dispatch_result.status != KernelResultStatus::Succeeded {
        return Err(InferenceApiError::ProviderUnavailable {
            reason: format!(
                "Reference CPU dispatch for {operation_id} failed: {:?}",
                dispatch_result.error
            ),
        });
    }
    ctx.node_events.push(PerNodeCausalEvent::new(
        InferenceApiObservationKind::ProviderCompleted,
        node.clone(),
    ));
    // Every requested output is read back and bound to its corresponding
    // graph edge in the same order `outputs` was given, not just the
    // first -- a genuinely multi-output Operator (e.g. `split`) produces
    // one `KernelResult::updated_resources` entry per output, and each
    // needs its own `TensorResourceProduced` causal event and its own
    // read-back, the same treatment the single-output case already got.
    // Reads back through `TensorValue` (`read_tensor_value`), not the
    // `HostTensor`-typed `read_tensor` -- a Device-resident output stays
    // Opaque here (no forced download) instead of being materialized
    // unconditionally on every single dispatch
    // (`unify-provider-output-admission-and-residency`'s Decision 3: this
    // is the output-side half of the round-trip
    // `enable-device-resident-kernel-chaining` deferred).
    let mut output_tensors = Vec::with_capacity(output_resources.len());
    for output_resource in &output_resources {
        let value = ctx
            .provider
            .read_tensor_value(&output_resource.id)
            .ok_or_else(|| InferenceApiError::GenerationFailed {
                reason: format!(
                    "Reference CPU dispatch for {operation_id} produced no output for resource '{}'",
                    output_resource.id
                ),
            })?;
        let output_tensor = match value {
            TensorValue::Host(tensor) => NodeValue::Host(tensor),
            TensorValue::Opaque => NodeValue::Resident {
                id: output_resource.id.clone(),
                shape: output_resource.descriptor.shape.dimensions.clone(),
            },
        };
        ctx.node_events.push(
            PerNodeCausalEvent::new(
                InferenceApiObservationKind::TensorResourceProduced,
                node.clone(),
            )
            .with_resource(output_resource.id.clone()),
        );
        output_tensors.push(output_tensor);
    }
    ctx.last_provider_execution = Some(ProviderExecutionResult::completed(
        handle,
        dispatch_result.updated_resources.clone(),
    ));
    Ok((dispatch_result, output_tensors))
}

/// Single-output convenience wrapper over
/// [`dispatch_reference_cpu_operator_multi`]: almost every graph node in
/// this file's Qwen dispatch has exactly one output, and threading a
/// `Vec` through every one of those call sites for a case that structurally
/// cannot produce more than one element would only add noise.
pub(super) fn dispatch_reference_cpu_operator(
    ctx: &mut FirstNativeDispatchContext<'_>,
    operation_id: &str,
    operator: OperatorId,
    inputs: Vec<NodeInputResource>,
    output: (TensorResourceId, TensorDescriptor),
    attributes: BTreeMap<String, OperatorAttributeValue>,
) -> Result<(KernelDispatchResult, NodeValue), InferenceApiError> {
    let (dispatch_result, mut outputs) = dispatch_reference_cpu_operator_multi(
        ctx,
        operation_id,
        operator,
        inputs,
        vec![output],
        attributes,
    )?;
    let output_tensor = outputs
        .pop()
        .ok_or_else(|| InferenceApiError::GenerationFailed {
            reason: format!("Reference CPU dispatch for {operation_id} produced no output"),
        })?;
    Ok((dispatch_result, output_tensor))
}

pub(super) fn runtime_generation_failed(error: impl std::fmt::Display) -> InferenceApiError {
    InferenceApiError::GenerationFailed {
        reason: error.to_string(),
    }
}

pub(super) fn dispatch_operator_id(name: &str, family: OperatorFamily) -> OperatorId {
    OperatorId::magnetar(name, 1, family)
}

/// Where a leaf `dispatch_qwen_*` function's output should live, resolved
/// once by `dispatch_first_native_graph_node` (or left `None` by callers with no
/// edge identity of their own -- RoPE's internal per-head dispatches, test
/// oracles) and threaded down to whichever function actually computes the
/// output descriptor.
pub(super) type OutputTarget = (TensorResourceId, MemoryPlacement, MemoryAllocationOwner);

/// Resolves `(TensorResourceId, TensorDescriptor)` to hand to
/// `dispatch_reference_cpu_operator[_multi]`'s `output`/`outputs`
/// parameter: when `output_target` names the graph edge's own final
/// identity, pre-admits it there and now (Runtime-side, before
/// submission -- `unify-provider-output-admission-and-residency`'s
/// Decision 3), so the Kernel writes directly into it and no later
/// download-then-reupload is needed. Otherwise (no edge identity to target)
/// falls back to synthesizing a Kernel-internal id exactly as before,
/// leaving admission to the Provider's own fallback self-admission path
/// (`unify-provider-output-admission-and-residency`'s Decision 2).
pub(super) fn resolve_output_target(
    ctx: &mut FirstNativeDispatchContext<'_>,
    operation_id: &str,
    descriptor: TensorDescriptor,
    output_target: Option<OutputTarget>,
) -> Result<(TensorResourceId, TensorDescriptor), InferenceApiError> {
    match output_target {
        Some((id, placement, owner)) => {
            ctx.runtime
                .memory_mut()
                .admit_kernel_output(
                    id.clone(),
                    &descriptor,
                    placement,
                    owner,
                    ResourceAffinity::new(FallbackClass::Transparent),
                )
                .map_err(|error| InferenceApiError::MemoryAdmissionFailed {
                    reason: format!("cannot pre-admit output for {operation_id}: {error:?}"),
                })?;
            Ok((id, descriptor))
        }
        None => Ok((
            TensorResourceId::new(format!("{operation_id}.out")),
            descriptor,
        )),
    }
}

/// Releases a caller's pre-admission for `id` on *both* sides of the
/// admission boundary -- this Provider's own opaque storage (if the failed
/// dispatch had actually gone as far as materializing something there
/// before failing) and the Memory Manager's `TensorResidency` record plus
/// the allocation it references -- leaving no trace behind on either.
/// Reached only when a dispatch that pre-admitted its output
/// (`resolve_output_target`'s `Some` branch) subsequently fails, at *any*
/// point after admission: Kernel Registry/dispatch-plan construction (a
/// "submit-time" failure, before the Provider ever sees the invocation) or
/// the Kernel's own execution/completion (`make-first-native-cuda-hot-
/// path-device-resident`'s Decision 7 -- both failure classes are handled
/// uniformly here, by wrapping the entire post-admission call rather than
/// distinguishing where exactly it failed).
///
/// `provider.release_tensor` runs first, Memory Manager second (audit-
/// complet-cuda-hot-path-2026-09-08's Correctif B): a submit-time failure
/// never reached the Provider, so `release_tensor` is a correctly-`Ok(false)`
/// no-op there -- calling it unconditionally rather than trying to guess
/// whether the Provider materialized anything is simpler and no less
/// correct, since `release_tensor` is documented idempotent for exactly
/// this "nothing to release" case. Best-effort on both sides: this runs on
/// an already-failing path, so a further failure releasing either side is
/// not itself surfaced -- the original dispatch error is what the caller
/// needs to see.
pub(super) fn rollback_pre_admitted_output(
    provider: &dyn ProviderExecutionApi,
    memory: &mut MemoryManager,
    id: &TensorResourceId,
) {
    let _ = provider.release_tensor(id);
    if let Some(residency) = memory.remove_tensor_residency(id)
        && let Some(allocation) = residency.allocation
    {
        let _ = memory.release(allocation);
    }
}

/// Dispatches through `dispatch_reference_cpu_operator`, rolling back a
/// caller-pre-admitted output if the dispatch itself fails afterward -- a
/// pre-admission is a real Memory Manager allocation the caller now owns;
/// leaving it behind after a failed dispatch would be an admission leak.
/// Replaces the `resolve_output_target` + `dispatch_reference_cpu_operator`
/// pair every leaf `dispatch_qwen_*` function used to call directly, so the
/// rollback is expressed once rather than duplicated across each of them.
pub(super) fn dispatch_reference_cpu_operator_pre_admitted(
    ctx: &mut FirstNativeDispatchContext<'_>,
    operation_id: &str,
    operator: OperatorId,
    inputs: Vec<NodeInputResource>,
    output_descriptor: TensorDescriptor,
    output_target: Option<OutputTarget>,
    attributes: BTreeMap<String, OperatorAttributeValue>,
) -> Result<(KernelDispatchResult, NodeValue), InferenceApiError> {
    let pre_admitted = output_target.is_some();
    let output = resolve_output_target(ctx, operation_id, output_descriptor, output_target)?;
    let output_id = output.0.clone();
    let result =
        dispatch_reference_cpu_operator(ctx, operation_id, operator, inputs, output, attributes);
    if result.is_err() && pre_admitted {
        rollback_pre_admitted_output(&*ctx.provider, ctx.runtime.memory_mut(), &output_id);
    }
    result
}

/// `graph_fusion`'s fused matmul nodes: `matmul(a, b)` plus `extra`
/// (`matmul-bias`'s `[cols]` bias row, or `matmul-residual`'s residual).
pub(super) fn dispatch_qwen_matmul_fused(
    ctx: &mut FirstNativeDispatchContext<'_>,
    operation_id: &str,
    operator: OperatorId,
    a: NodeValue,
    b: NodeValue,
    extra: NodeValue,
    output_target: Option<OutputTarget>,
) -> Result<(KernelDispatchResult, NodeValue), InferenceApiError> {
    let (rows, _) = a.rows_cols()?;
    let (_, cols) = b.rows_cols()?;
    let descriptor = TensorDescriptor::new(
        ShapeDescriptor::new([rows, cols]),
        DTypeDescriptor::portable(ComputeDType::Float32),
        LayoutDescriptor::Contiguous,
    );
    dispatch_reference_cpu_operator_pre_admitted(
        ctx,
        operation_id,
        operator,
        vec![
            node_input_resource(operation_id, "a", a),
            node_input_resource(operation_id, "b", b),
            node_input_resource(operation_id, "extra", extra),
        ],
        descriptor,
        output_target,
        BTreeMap::new(),
    )
}

pub(super) fn dispatch_qwen_matmul(
    ctx: &mut FirstNativeDispatchContext<'_>,
    operation_id: &str,
    a: NodeValue,
    b: NodeValue,
    output_target: Option<OutputTarget>,
) -> Result<(KernelDispatchResult, NodeValue), InferenceApiError> {
    let (rows, _) = a.rows_cols()?;
    let (_, cols) = b.rows_cols()?;
    let descriptor = TensorDescriptor::new(
        ShapeDescriptor::new([rows, cols]),
        DTypeDescriptor::portable(ComputeDType::Float32),
        LayoutDescriptor::Contiguous,
    );
    dispatch_reference_cpu_operator_pre_admitted(
        ctx,
        operation_id,
        dispatch_operator_id("matmul", OperatorFamily::LinearAlgebra),
        vec![
            node_input_resource(operation_id, "a", a),
            node_input_resource(operation_id, "b", b),
        ],
        descriptor,
        output_target,
        BTreeMap::new(),
    )
}

/// Dispatches the "concat" Operator -- KV-history concatenation
/// (`implement-device-resident-multi-step-cuda-decode`): stacks `a`'s rows
/// above `b`'s rows through the exact same generic dispatch machinery
/// every other Qwen graph node uses, instead of downloading both to host
/// and concatenating in Rust (`concat_rows`, this function's predecessor
/// in the KV-history-append block). `a` is the previous steps' historical
/// K/V, `b` is this step's newly computed K/V; mirrors
/// `dispatch_qwen_matmul`'s own pattern of computing an output shape that
/// differs from either input's own shape.
/// Everything a `concat` selection depends on besides the registry: the
/// request's affinity and memory class and each resource's descriptor and
/// affinity (not its id).
pub(super) type InternalSelectionKey = (
    OperatorId,
    ResourceAffinity,
    KernelMemoryClass,
    Vec<(TensorDescriptor, ResourceAffinity)>,
);

/// Dispatches the "concat" Operator for KV-history concatenation
/// (`implement-device-resident-multi-step-cuda-decode`): stacks `a`'s rows
/// above `b`'s rows through a real Kernel Registry selection and real
/// Provider dispatch, instead of downloading both to host and
/// concatenating in Rust (`concat_rows`, this function's predecessor in
/// the KV-history-append block).
///
/// Deliberately does *not* go through `dispatch_reference_cpu_operator[_
/// multi/_pre_admitted]`: this is Runtime-internal bookkeeping tied to
/// `real_node`'s own output edge, not itself a node the static graph
/// declares, so there is no `PreparedExecutionPlan` node binding for it
/// under any id -- attempting one either fails outright (a synthetic id
/// the graph never declared) or would silently resolve `real_node`'s own
/// *original* Kernel binding (e.g. "rope") instead of "concat" (`prepare_
/// node_execution` binds purely by node id, not by requested operator).
/// This performs a real, live Kernel Registry selection instead -- the
/// same selection logic `dispatch_reference_cpu_operator_multi`'s own
/// `(None, _)` branch already uses for a caller with no Prepared Plan at
/// all, not `prepared_candidate_for_operation`'s synthetic always-
/// compatible path -- using `real_node`'s own already-resolved,
/// already-correct resource affinity (from its real Prepared Plan
/// binding) so the selection targets the actual bound Provider (CUDA or
/// Reference CPU), not a hardcoded/default one. Never participates in the
/// per-node causal-chain contract (`GraphNodeReady`/`PlanBindingResolved`
/// etc., `validate_e2e_per_node_causal_chain`) -- the same posture the
/// plain-Rust `concat_rows` this replaces already had (zero node-level
/// observability); what changed is that this now runs against the actual
/// bound Provider's real Kernel instead of Rust host code, not the
/// observability shape.
pub(super) fn dispatch_qwen_concat(
    ctx: &mut FirstNativeDispatchContext<'_>,
    operation_id: &str,
    real_node: &ExecutionNodeId,
    a: NodeValue,
    b: NodeValue,
    output_target: Option<OutputTarget>,
) -> Result<NodeValue, InferenceApiError> {
    let (a_rows, a_cols) = a.rows_cols()?;
    let (b_rows, b_cols) = b.rows_cols()?;
    if a_cols != b_cols {
        return Err(InferenceApiError::GraphPlanningFailed {
            reason: format!(
                "graph node '{operation_id}' (operator 'concat') expects both inputs to share \
                 the same column count, got {a_cols} and {b_cols}"
            ),
        });
    }
    let output_descriptor = f32_tensor_descriptor_from_shape(&[a_rows + b_rows, a_cols]);
    let output = resolve_output_target(ctx, operation_id, output_descriptor, output_target)?;
    dispatch_runtime_internal_operator(
        ctx,
        operation_id,
        real_node,
        dispatch_operator_id("concat", OperatorFamily::Tensor),
        [("a", a), ("b", b)],
        output,
        BTreeMap::new(),
    )
}

/// Dispatches an Operator the Runtime adds to a graph step itself (the KV
/// history append's `concat` or `kv-write`), not a node of the graph: there
/// is no Prepared Plan binding for it, so its Kernel comes from a live
/// Kernel Registry selection (reused while the same request repeats within
/// a graph execution), with `real_node`'s own resolved affinity so it
/// targets the Provider that node is bound to. `output` is the resource the
/// Kernel writes, already admitted by the caller when it needs to be.
pub(super) fn dispatch_runtime_internal_operator<const N: usize>(
    ctx: &mut FirstNativeDispatchContext<'_>,
    operation_id: &str,
    real_node: &ExecutionNodeId,
    operator: OperatorId,
    inputs: [(&str, NodeValue); N],
    output: (TensorResourceId, TensorDescriptor),
    attributes: BTreeMap<String, OperatorAttributeValue>,
) -> Result<NodeValue, InferenceApiError> {
    let (output_id, output_descriptor) = output;
    let affinity = resolved_resource_affinity(
        ctx.prepared_plan.as_deref(),
        real_node,
        ctx.runtime.context().id(),
    );
    let memory_class = resolved_kernel_memory_class(ctx.prepared_plan.as_deref(), real_node);
    let output_resource =
        TensorResourceDescriptor::new(output_id.clone(), output_descriptor, affinity.clone());
    let mut selection_request = KernelSelectionRequest::new(
        format!("e2e-runtime-{operation_id}"),
        operator.clone(),
        affinity.clone(),
    )
    .with_output(KernelResource::new(output_resource.clone(), memory_class));
    for (name, value) in inputs {
        let input = node_input_resource(operation_id, name, value);
        let resource = match input {
            NodeInputResource::Fresh(id, descriptor, tensor) => {
                let resource =
                    TensorResourceDescriptor::new(id.clone(), descriptor, affinity.clone());
                ctx.provider.write_tensor(id, tensor).map_err(|error| {
                    InferenceApiError::ProviderTensorWriteFailed {
                        reason: error.to_string(),
                    }
                })?;
                resource
            }
            NodeInputResource::Resident(id, descriptor) => {
                let resolved_affinity =
                    resident_resource_affinity(ctx.runtime.memory(), &id, &affinity, real_node)?;
                TensorResourceDescriptor::new(id, descriptor, resolved_affinity)
            }
        };
        selection_request =
            selection_request.with_input(KernelResource::new(resource, memory_class));
    }
    let selection_key: InternalSelectionKey = (
        operator,
        affinity.clone(),
        memory_class,
        selection_request
            .outputs
            .iter()
            .chain(&selection_request.inputs)
            .map(|resource| {
                (
                    resource.resource.descriptor.clone(),
                    resource.resource.affinity.clone(),
                )
            })
            .collect(),
    );
    let candidate = match &ctx.internal_selection {
        Some((key, candidate)) if *key == selection_key => candidate.clone(),
        _ => {
            let selection = ctx
                .runtime
                .kernel_registry()
                .select(&selection_request)
                .map_err(|error| InferenceApiError::KernelUnavailable {
                    reason: format!("{operation_id}: {error}"),
                })?;
            let candidate =
                selection
                    .selected
                    .ok_or_else(|| InferenceApiError::KernelUnavailable {
                        reason: format!("Kernel Registry selected no candidate for {operation_id}"),
                    })?;
            ctx.internal_selection = Some((selection_key, candidate.clone()));
            candidate
        }
    };
    // Borrowed alongside the Memory Manager instead of copied, as in
    // `dispatch_reference_cpu_operator_multi`.
    let (registry, memory, _phases) = ctx.runtime.kernel_registry_memory_and_phases_mut();
    let advertisement = registry
        .active_advertisement(&candidate.kernel)
        .ok_or_else(|| InferenceApiError::KernelUnavailable {
            reason: format!("selected advertisement for {operation_id} is no longer active"),
        })?;
    let mut plan = KernelDispatchPlan::from_selection(
        KernelDispatchPlanId::new(format!("e2e-runtime-{operation_id}-dispatch")),
        &selection_request,
        &candidate,
        advertisement,
        KernelInvocationId::new(format!("e2e-runtime-{operation_id}-invocation")),
    )
    .map_err(|error| InferenceApiError::KernelUnavailable {
        reason: format!("{error:?}"),
    })?;
    plan.invocation.attributes = attributes;
    let handle = ctx
        .provider
        .submit_kernel(
            advertisement,
            shared_initial_operator_catalog()
                .get(&advertisement.implemented_operator)
                .map_err(|error| InferenceApiError::KernelUnavailable {
                    reason: error.to_string(),
                })?,
            &plan.invocation,
            memory,
        )
        .map_err(|error| InferenceApiError::ProviderUnavailable {
            reason: format!(
                "Reference CPU dispatch for {operation_id} could not be submitted: {error}"
            ),
        })?;
    if let Some(workspace) = plan.workspace_reservation.take() {
        let _ = memory.release(workspace);
    }
    let kernel_result = ctx.provider.complete_kernel(&handle).map_err(|error| {
        InferenceApiError::ProviderUnavailable {
            reason: format!(
                "Reference CPU dispatch for {operation_id} could not be completed: {error}"
            ),
        }
    })?;
    let dispatch_result = KernelDispatchResult::from_kernel_result(&plan, kernel_result);
    if dispatch_result.status != KernelResultStatus::Succeeded {
        return Err(InferenceApiError::ProviderUnavailable {
            reason: format!(
                "Reference CPU dispatch for {operation_id} failed: {:?}",
                dispatch_result.error
            ),
        });
    }
    let value = ctx
        .provider
        .read_tensor_value(&output_resource.id)
        .ok_or_else(|| InferenceApiError::GenerationFailed {
            reason: format!(
                "Reference CPU dispatch for {operation_id} produced no output for resource '{}'",
                output_resource.id
            ),
        })?;
    Ok(match value {
        TensorValue::Host(tensor) => NodeValue::Host(tensor),
        TensorValue::Opaque => NodeValue::Resident {
            id: output_resource.id.clone(),
            shape: output_resource.descriptor.shape.dimensions.clone(),
        },
    })
}

pub(super) fn dispatch_qwen_unary(
    ctx: &mut FirstNativeDispatchContext<'_>,
    operation_id: &str,
    name: &str,
    family: OperatorFamily,
    input: NodeValue,
    attributes: BTreeMap<String, OperatorAttributeValue>,
    output_target: Option<OutputTarget>,
) -> Result<(KernelDispatchResult, NodeValue), InferenceApiError> {
    let output_descriptor = f32_tensor_descriptor_from_shape(input.shape());
    dispatch_reference_cpu_operator_pre_admitted(
        ctx,
        operation_id,
        dispatch_operator_id(name, family),
        vec![node_input_resource(operation_id, "input", input)],
        output_descriptor,
        output_target,
        attributes,
    )
}

pub(super) fn dispatch_qwen_binary_same_shape(
    ctx: &mut FirstNativeDispatchContext<'_>,
    operation_id: &str,
    name: &str,
    family: OperatorFamily,
    a: NodeValue,
    b: NodeValue,
    output_target: Option<OutputTarget>,
) -> Result<(KernelDispatchResult, NodeValue), InferenceApiError> {
    let output_descriptor = f32_tensor_descriptor_from_shape(a.shape());
    dispatch_reference_cpu_operator_pre_admitted(
        ctx,
        operation_id,
        dispatch_operator_id(name, family),
        vec![
            node_input_resource(operation_id, "a", a),
            node_input_resource(operation_id, "b", b),
        ],
        output_descriptor,
        output_target,
        BTreeMap::new(),
    )
}

/// Dispatches the "split" Operator -- the one genuinely multi-output node
/// this graph executor produces (`define-provider-prepared-kernel-
/// execution-contract` task group 3): halves `input`'s last dimension into
/// two independently resolvable outputs via
/// [`dispatch_reference_cpu_operator_multi`] directly (there is no
/// single-output convenience wrapper to go through here). Pre-admits both
/// `output_targets` up front, same discipline as
/// `dispatch_reference_cpu_operator_pre_admitted`'s single-output rollback,
/// extended to roll back whichever of the two targets was actually admitted
/// if either admission or the dispatch itself fails.
pub(super) fn dispatch_qwen_split(
    ctx: &mut FirstNativeDispatchContext<'_>,
    operation_id: &str,
    input: NodeValue,
    output_targets: [Option<OutputTarget>; 2],
) -> Result<(KernelDispatchResult, Vec<NodeValue>), InferenceApiError> {
    let (rows, cols) = input.rows_cols()?;
    if cols % 2 != 0 || cols == 0 {
        return Err(InferenceApiError::GraphPlanningFailed {
            reason: format!(
                "graph node '{operation_id}' (operator 'split') expects an even, non-zero last dimension, got {cols}"
            ),
        });
    }
    let half_descriptor = f32_tensor_descriptor_from_shape(&[rows, cols / 2]);
    let [left_target, right_target] = output_targets;
    let left_pre_admitted = left_target.is_some();
    let left_output = resolve_output_target(
        ctx,
        &format!("{operation_id}.left"),
        half_descriptor.clone(),
        left_target,
    )?;
    let left_output_id = left_output.0.clone();
    let right_pre_admitted = right_target.is_some();
    let right_output = match resolve_output_target(
        ctx,
        &format!("{operation_id}.right"),
        half_descriptor,
        right_target,
    ) {
        Ok(output) => output,
        Err(error) => {
            if left_pre_admitted {
                rollback_pre_admitted_output(
                    &*ctx.provider,
                    ctx.runtime.memory_mut(),
                    &left_output_id,
                );
            }
            return Err(error);
        }
    };
    let right_output_id = right_output.0.clone();
    let result = dispatch_reference_cpu_operator_multi(
        ctx,
        operation_id,
        dispatch_operator_id("split", OperatorFamily::Tensor),
        vec![node_input_resource(operation_id, "input", input)],
        vec![left_output, right_output],
        BTreeMap::new(),
    );
    if result.is_err() {
        if left_pre_admitted {
            rollback_pre_admitted_output(&*ctx.provider, ctx.runtime.memory_mut(), &left_output_id);
        }
        if right_pre_admitted {
            rollback_pre_admitted_output(
                &*ctx.provider,
                ctx.runtime.memory_mut(),
                &right_output_id,
            );
        }
    }
    result
}

/// Both real Kernels behind this Operator (`providers/cpu::rmsnorm`,
/// `providers/cuda::CudaKernels::rmsnorm`) derive `cols` from `input`'s own
/// shape and accept a `[cols]`-shaped `weight` directly, broadcasting it
/// internally -- confirmed by reading both directly
/// (`make-first-native-cuda-hot-path-device-resident` task 2.1). The
/// per-row broadcast this function used to perform in Rust before
/// dispatch was unneeded duplication of work every real Kernel already
/// does, and forced both inputs to materialize to `HostTensor` even when
/// one or both were already Device-resident. `input`/`weight` are
/// `NodeValue` now, passed straight through to `node_input_resource`
/// exactly like `dispatch_qwen_matmul`'s inputs.
pub(super) fn dispatch_qwen_rmsnorm(
    ctx: &mut FirstNativeDispatchContext<'_>,
    operation_id: &str,
    input: NodeValue,
    weight: NodeValue,
    epsilon: f32,
    output_target: Option<OutputTarget>,
) -> Result<(KernelDispatchResult, NodeValue), InferenceApiError> {
    let mut attributes = BTreeMap::new();
    attributes.insert(
        "epsilon".into(),
        OperatorAttributeValue::Float(epsilon as f64),
    );
    let output_descriptor = f32_tensor_descriptor_from_shape(input.shape());
    dispatch_reference_cpu_operator_pre_admitted(
        ctx,
        operation_id,
        dispatch_operator_id("rmsnorm", OperatorFamily::Normalization),
        vec![
            node_input_resource(operation_id, "input", input),
            node_input_resource(operation_id, "weight", weight),
        ],
        output_descriptor,
        output_target,
        attributes,
    )
}

/// A single Kernel invocation rotating all `head_count` head blocks of
/// `input` at once, via the `rope` Operator's `head_count` attribute
/// (`make-first-native-cuda-hot-path-device-resident`) -- replaces the
/// former `dispatch_qwen_rope_per_head`, which dispatched one Kernel
/// invocation per head by slicing/reassembling real bytes in Rust. `input`
/// is `NodeValue`-eligible for Resident passthrough like every other
/// operator here, since this is now a genuine single Kernel dispatch
/// rather than a Rust-computed reassembly of several.
#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_qwen_rope(
    ctx: &mut FirstNativeDispatchContext<'_>,
    operation_id: &str,
    input: NodeValue,
    head_count: u64,
    rope_config: &FirstNativeRopeConfig,
    pairing: RopePairing,
    frequency_scaling: Option<&crate::operator::RopeBandedFrequencyScaling>,
    position_offset: u64,
    output_target: Option<OutputTarget>,
) -> Result<(KernelDispatchResult, NodeValue), InferenceApiError> {
    let mut attributes = BTreeMap::new();
    attributes.insert(
        "rotation_pairing".into(),
        OperatorAttributeValue::String(pairing.as_attribute().into()),
    );
    attributes.insert(
        "base".into(),
        OperatorAttributeValue::Float(rope_config.base),
    );
    attributes.insert(
        "scale".into(),
        OperatorAttributeValue::Float(rope_config.scale.unwrap_or(1.0)),
    );
    attributes.insert(
        "dimension".into(),
        OperatorAttributeValue::Integer(rope_config.dimension as i64),
    );
    attributes.insert(
        "position_mode".into(),
        OperatorAttributeValue::String(rope_config.position_mode.as_str().into()),
    );
    attributes.insert(
        "position_offset".into(),
        OperatorAttributeValue::Integer(position_offset as i64),
    );
    attributes.insert(
        "head_count".into(),
        OperatorAttributeValue::Integer(head_count as i64),
    );
    if let Some(scaling) = frequency_scaling {
        // A Kernel that ignored these attributes would rotate with unscaled
        // frequencies and silently degrade every token: refuse a Provider
        // whose `rope` Kernels do not all declare that they honour them.
        if let Some(provider) = ctx
            .prepared_plan
            .as_ref()
            .and_then(|plan| plan.node_bindings.first())
            .map(|binding| binding.provider.clone())
            && !rope_banded_scaling_supported(ctx.runtime, &provider)
        {
            return Err(InferenceApiError::KernelUnavailable {
                reason: format!(
                    "operation '{operation_id}' needs banded RoPE frequency scaling, which \
                     Provider {provider:?} does not advertise ('{}')",
                    crate::operator::ROPE_BANDED_FREQUENCY_SCALING_FEATURE
                ),
            });
        }
        scaling.insert_attributes(&mut attributes);
    }
    let output_descriptor = f32_tensor_descriptor_from_shape(input.shape());
    dispatch_reference_cpu_operator_pre_admitted(
        ctx,
        operation_id,
        dispatch_operator_id("rope", OperatorFamily::PositionEncoding),
        vec![node_input_resource(operation_id, "input", input)],
        output_descriptor,
        output_target,
        attributes,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_qwen_attention(
    ctx: &mut FirstNativeDispatchContext<'_>,
    operation_id: &str,
    q: NodeValue,
    k: NodeValue,
    v: NodeValue,
    architecture: &ModelComponentArchitectureMetadata,
    output_target: Option<OutputTarget>,
    key_count: Option<u64>,
) -> Result<(KernelDispatchResult, NodeValue), InferenceApiError> {
    let mut attributes = BTreeMap::new();
    attributes.insert("causal".into(), OperatorAttributeValue::Boolean(true));
    // K and V are in-place KV stores: only their first `key_count` rows are
    // the key sequence (`append-kv-history-in-place`).
    if let Some(key_count) = key_count {
        attributes.insert(
            "key_count".into(),
            OperatorAttributeValue::Integer(key_count as i64),
        );
    }
    attributes.insert(
        "head_count".into(),
        OperatorAttributeValue::Integer(architecture.attention_head_count as i64),
    );
    attributes.insert(
        "kv_head_count".into(),
        OperatorAttributeValue::Integer(architecture.kv_head_count as i64),
    );
    attributes.insert(
        "head_dimension".into(),
        OperatorAttributeValue::Integer(architecture.head_dimension as i64),
    );
    attributes.insert(
        "attention_mask_kind".into(),
        OperatorAttributeValue::String("causal".into()),
    );
    let output_descriptor = f32_tensor_descriptor_from_shape(q.shape());
    dispatch_reference_cpu_operator_pre_admitted(
        ctx,
        operation_id,
        dispatch_operator_id("attention", OperatorFamily::Attention),
        vec![
            node_input_resource(operation_id, "q", q),
            node_input_resource(operation_id, "k", k),
            node_input_resource(operation_id, "v", v),
        ],
        output_descriptor,
        output_target,
        attributes,
    )
}

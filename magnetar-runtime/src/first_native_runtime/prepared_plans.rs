//! Prepared execution plans for first-native graphs.

use super::*;

/// Confirms `instance` is genuinely ready to generate *right now* (not
/// merely inferred from a past check). Public for the same reason
/// [`prepare_first_native_execution_plans`] is
/// (`implement-production-qwen-model-loading` task group 12): an embedder
/// driving a production-loaded instance through the real generation loop
/// needs this same readiness confirmation before every step, exactly as
/// every in-crate caller already does.
pub fn require_ready_first_native_instance<'a>(
    runtime: &'a Runtime,
    instance: &ModelInstanceId,
) -> Result<&'a ModelInstance, InferenceApiError> {
    let model_instance = runtime
        .model_instance(instance)
        .map_err(InferenceApiError::from)?;
    let status = model_instance.status();
    if !status.readiness.accepts_generation() {
        return Err(InferenceApiError::ModelInstanceNotReady {
            reason: format!(
                "first-native generation requires ready ModelInstance '{instance}', got lifecycle {:?} / readiness {:?}",
                status.lifecycle, status.readiness
            ),
        });
    }
    Ok(model_instance)
}

/// The prepared prefill/decode plans [`prepare_first_native_execution_plans`]
/// publishes. Public alongside it (task group 12) so an embedder can hold
/// `&mut prefill`/`&mut decode` to build a
/// [`crate::inference_api::RuntimeGenerationExecutionPlans`].
#[derive(Clone)]
pub struct FirstNativePreparedPlans {
    pub prefill: PreparedExecutionPlan,
    pub prefill_node_count: usize,
    pub decode: PreparedExecutionPlan,
    pub decode_node_count: usize,
}

pub(super) fn first_native_plan_context(
    phase: PreparedExecutionPhase,
    token_count: u64,
) -> PlanGuardContext {
    let mut context = PlanGuardContext::for_phase(phase);
    context.sequence_length = Some(token_count.max(1));
    context.total_tokens = Some(token_count.max(1));
    context.affinity = Some(ResourceAffinity::new(FallbackClass::Transparent));
    context.provider_ready = true;
    context.device_ready = true;
    context.memory_feasible = true;
    context
}

pub(super) fn require_compatible_first_native_plan(
    plan: Option<&mut PreparedExecutionPlan>,
    context: &PlanGuardContext,
) -> Result<PreparedExecutionPlanGeneration, PreparedExecutionPlanError> {
    let plan = plan.ok_or(PreparedExecutionPlanError::PlanNotFound)?;
    plan.execute_ready_path(context)?;
    Ok(plan.generation)
}

pub(super) fn prepare_first_native_plan_for_graph(
    runtime: &Runtime,
    graph: &ExecutionGraph,
    instance: &ModelInstanceId,
    model_instance_revision: u64,
    token_count: u64,
    generation: PreparedExecutionPlanGeneration,
    provider: &ProviderBinding,
) -> Result<PreparedExecutionPlan, E2eConformanceError> {
    let phase = PreparedExecutionPhase::from(graph.phase);
    let mut scope = PreparedExecutionPlanScope::for_phase(phase)
        .with_model_instance(instance.clone(), model_instance_revision)
        .with_workload_bucket(format!("{:?}-{}-tokens", phase, token_count.max(1)));
    scope.provider = Some(provider.clone());

    let mut plan = PreparedExecutionPlan::new(
        PreparedExecutionPlanId::new(format!("first-native-{phase:?}-plan"))?,
        generation,
        semantic_graph_fingerprint(graph),
        scope,
    )?;
    let affinity = ResourceAffinity::new(FallbackClass::Transparent)
        .with_provider(provider.clone())
        .with_execution_context(runtime.context().id());

    let mut bindings = Vec::with_capacity(graph.nodes.len());
    for (node_id, node) in &graph.nodes {
        let node_affinity = node
            .resource_affinity
            .clone()
            .unwrap_or_else(|| affinity.clone());
        let mut selection_request = KernelSelectionRequest::new(
            format!("first-native-plan-{phase:?}-{}", node_id.as_str()),
            node.operator.clone(),
            node_affinity.clone(),
        );
        selection_request.graph_plan = Some(graph.id.clone());
        selection_request.model_instance = Some(instance.clone());
        selection_request.observability_correlation =
            Some(format!("first-native-plan:{phase:?}:{node_id}"));
        for input in &node.inputs {
            let resource = graph_kernel_resource(graph, input, &node_affinity, provider)?;
            merge_graph_edge_requirements(&mut selection_request, &resource);
            if let Some(kv) = graph
                .edges
                .get(input)
                .and_then(|edge| edge.kv_cache.as_ref())
            {
                note_graph_kv_requirement(&mut selection_request, kv);
                if matches!(kv.behavior, GraphKvCacheBehavior::Input) {
                    merge_kernel_managed_kv_requirement(&mut selection_request, kv, &resource);
                }
            }
            selection_request = selection_request.with_input(resource);
        }
        for output in &node.outputs {
            let resource = graph_kernel_resource(graph, output, &node_affinity, provider)?;
            merge_graph_edge_requirements(&mut selection_request, &resource);
            if let Some(kv) = graph
                .edges
                .get(output)
                .and_then(|edge| edge.kv_cache.as_ref())
            {
                note_graph_kv_requirement(&mut selection_request, kv);
            }
            selection_request = selection_request.with_output(resource);
        }
        let selection = runtime
            .kernel_registry()
            .select(&selection_request)
            .map_err(|error| E2eConformanceError::KernelCoverageMissing {
                reason: format!(
                    "Kernel Registry selection failed for node {node_id} ({operator}): {error}",
                    operator = node.operator,
                ),
            })?;
        let candidate =
            selection
                .selected
                .ok_or_else(|| E2eConformanceError::KernelCoverageMissing {
                    reason: format!("no Kernel Registry candidate selected for node {node_id}"),
                })?;

        let mut binding = PlanNodeBinding::new(
            [node_id.clone()],
            candidate.kernel.clone(),
            candidate.provider.clone(),
        )?
        .with_specialization(format!("first-native-{phase:?}-{}", node_id.as_str()));
        let prepared_kernel = runtime
            .kernel_registry()
            .active_prepared_kernel(&candidate.kernel)
            .ok_or_else(|| E2eConformanceError::KernelCoverageMissing {
                reason: format!("no active PreparedKernel registered for node {node_id}"),
            })?;
        binding = binding.with_prepared_kernel(prepared_kernel.id, prepared_kernel.generation);
        if let Some(device) = candidate.device {
            binding = binding.with_device(device);
        }
        if let Some(profile) = candidate.kernel.conformance_profile {
            binding = binding.with_qualification_profile(profile);
        }
        bindings.push(binding);
    }
    plan.add_node_bindings(bindings)?;

    plan.set_resource_plan(ResourceBindingPlan::default())?;
    plan.set_memory_requirements(PlanMemoryRequirements::default())?;
    plan.add_guard(PlanGuard::Phase(phase));
    // `max` must be at least `token_count`: this Plan's own node bindings
    // and edge shapes were built for exactly `token_count` tokens (see the
    // node-binding loop above), so a workload of that same size must
    // always be self-consistent. `E2E_FIXTURE_CONTEXT` is only a floor,
    // preserving every existing tiny-synthetic-fixture test's behavior
    // (built for well under 32 tokens); a real production prompt (e.g. one
    // rendered through a real chat template, routinely 30-100+ tokens)
    // needs the ceiling to actually cover its own real length instead of
    // being rejected by an unrelated fixture-sized cap
    // (`close-tachyon-scope-audit-gaps` task 3.2: found by this change's
    // own end-to-end real-chat-template test, which failed here before
    // this fix even though the chat-template rendering itself was
    // already correct).
    plan.add_guard(PlanGuard::SequenceRange {
        min: 1,
        max: token_count.max(E2E_FIXTURE_CONTEXT),
    });
    plan.add_guard(PlanGuard::Readiness);
    plan.add_guard(PlanGuard::AffinityRequired);
    plan.add_guard(PlanGuard::MemoryFeasible);
    plan.mark_ready_atomically()?;

    let mut plan_for_validation = plan.clone();
    let context = first_native_plan_context(phase, token_count);
    require_compatible_first_native_plan(Some(&mut plan_for_validation), &context)?;
    Ok(plan)
}

/// `provider`-aware (`audit-complet-cuda-hot-path-2026-09-08` P1-1,
/// discovered as a real blocker, not just the audit's own separately-
/// flagged P2-1 architecture debt in `resolved_kernel_memory_class`):
/// every graph edge this crate produces -- Rust-constructed and real
/// Component-produced alike -- declares `TensorResidencyConstraint::Host`
/// by default (a graph is Provider-agnostic declarative metadata; it does
/// not know in advance which Provider will execute it), so the unqualified
/// `Host` case previously mapped unconditionally to `KernelMemoryClass::
/// Host`. That happened to match Reference CPU's own kernels (the only
/// Provider this ever ran against before `run_first_native_graph_with_
/// provider`), so the mismatch was never exercised -- CUDA's kernels
/// require `KernelMemoryClass::Device` and were rejected outright
/// (`MemoryClassUnsupported`) the first time a Prepared Plan was actually
/// built for a non-Reference-CPU Provider. An edge that *explicitly*
/// declares a residency constraint (`Device`, `BrowserLinearMemory`,
/// `ProviderOwnedOpaque`) still wins outright -- only the default,
/// unconstrained `Host` case is resolved from `provider` instead of
/// assumed.
pub(super) fn kernel_memory_class_for_edge(
    edge: &TensorEdge,
    provider: &ProviderBinding,
) -> KernelMemoryClass {
    match edge.residency {
        TensorResidencyConstraint::Device => KernelMemoryClass::Device,
        TensorResidencyConstraint::BrowserLinearMemory => KernelMemoryClass::BrowserLinearMemory,
        TensorResidencyConstraint::ProviderOwnedOpaque => KernelMemoryClass::ProviderOwned,
        TensorResidencyConstraint::Host => match edge.memory_class {
            MemoryAllocationClass::HostPinned | MemoryAllocationClass::TransferStaging => {
                KernelMemoryClass::PinnedHost
            }
            MemoryAllocationClass::BrowserLinearMemory => KernelMemoryClass::BrowserLinearMemory,
            _ if crate::reference_cpu::host_memory_provider(provider.as_str()) => {
                KernelMemoryClass::Host
            }
            _ => KernelMemoryClass::Device,
        },
    }
}

pub(super) fn layout_kind_for_descriptor(descriptor: &TensorDescriptor) -> TensorLayoutKind {
    match descriptor.layout {
        LayoutDescriptor::Contiguous => TensorLayoutKind::Contiguous,
        LayoutDescriptor::Strided { .. } => TensorLayoutKind::Strided,
        LayoutDescriptor::Blocked { .. } => TensorLayoutKind::Blocked,
        LayoutDescriptor::Paged { .. } => TensorLayoutKind::Paged,
        LayoutDescriptor::PackedQuantized { .. } => TensorLayoutKind::QuantizedPacked,
        LayoutDescriptor::AttentionSpecific { .. } => TensorLayoutKind::AttentionSpecific,
        LayoutDescriptor::BrowserCompatible { .. } => TensorLayoutKind::BrowserCompatible,
        LayoutDescriptor::ProviderOpaque { .. } => TensorLayoutKind::ProviderOpaque,
    }
}

pub(super) fn graph_kernel_resource(
    graph: &ExecutionGraph,
    edge_id: &TensorEdgeId,
    default_affinity: &ResourceAffinity,
    provider: &ProviderBinding,
) -> Result<KernelResource, E2eConformanceError> {
    let edge =
        graph
            .edges
            .get(edge_id)
            .ok_or_else(|| E2eConformanceError::GraphValidationFailed {
                reason: format!("graph node references missing tensor edge '{edge_id}'"),
            })?;
    let affinity = edge
        .affinity
        .clone()
        .unwrap_or_else(|| default_affinity.clone());
    Ok(KernelResource::new(
        TensorResourceDescriptor::new(
            TensorResourceId::new(edge.id.as_str()),
            edge.descriptor.clone(),
            affinity,
        ),
        kernel_memory_class_for_edge(edge, provider),
    ))
}

pub(super) fn merge_graph_edge_requirements(
    request: &mut KernelSelectionRequest,
    resource: &KernelResource,
) {
    if let DTypeDescriptor::Portable(dtype) = &resource.resource.descriptor.dtype {
        request.dtype_requirements.insert(*dtype);
    }
    request
        .layout_requirements
        .insert(layout_kind_for_descriptor(&resource.resource.descriptor));
    request
        .memory_class_requirements
        .insert(resource.memory_class);
}

pub(super) fn note_graph_kv_requirement(
    request: &mut KernelSelectionRequest,
    kv: &GraphKvCacheMetadata,
) {
    let behavior = match kv.behavior {
        GraphKvCacheBehavior::Input => "input",
        GraphKvCacheBehavior::Output => "output",
        GraphKvCacheBehavior::Append => "append",
    };
    let cache_kind = if kv.paged { "paged" } else { "contiguous" };
    request.policy.insert(
        format!("graph-kv-cache:{}", kv.cache_id),
        format!("{behavior}:{cache_kind}"),
    );
}

pub(super) fn merge_kernel_managed_kv_requirement(
    request: &mut KernelSelectionRequest,
    kv: &GraphKvCacheMetadata,
    resource: &KernelResource,
) {
    let metadata = request
        .kv_cache
        .get_or_insert_with(|| KernelKvCacheMetadata {
            layouts: BTreeSet::new(),
            paged_cache: kv.paged,
            append: false,
            read: false,
            dtypes: BTreeSet::new(),
            memory_classes: BTreeSet::new(),
            affinity: Some(resource.resource.affinity.clone()),
        });
    metadata.paged_cache |= kv.paged;
    metadata.read = true;
    metadata.layouts.insert(if kv.paged {
        "paged".into()
    } else {
        "contiguous".into()
    });
    if let DTypeDescriptor::Portable(dtype) = &resource.resource.descriptor.dtype {
        metadata.dtypes.insert(*dtype);
    }
    metadata.memory_classes.insert(resource.memory_class);
}

/// Builds real published prepared execution plans for `instance` at
/// `prompt_token_count` against the Reference CPU Provider. Public
/// (task group 12): the one supported way to prepare a production-loaded
/// Model Instance's plans before driving
/// [`crate::inference_api::run_generation_loop_with_execution_plans`].
/// Reference-CPU-only; see
/// [`prepare_first_native_execution_plans_for_provider`] for an arbitrary
/// registered Provider (e.g. CUDA).
pub fn prepare_first_native_execution_plans(
    runtime: &Runtime,
    instance: &ModelInstanceId,
    graphs: FirstNativeComponentGraphs,
    prompt_token_count: u64,
) -> Result<FirstNativePreparedPlans, E2eConformanceError> {
    prepare_first_native_execution_plans_for_provider(
        runtime,
        instance,
        graphs,
        prompt_token_count,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )
}

/// [`prepare_first_native_execution_plans`], generalized to bind the
/// resulting Prepared Execution Plans to an arbitrary Provider instead of
/// hardcoding Reference CPU -- `prepare_first_native_plan_for_graph`'s own
/// `scope.provider`/fallback `ResourceAffinity` are the actual source of
/// the "every first-native Plan targets Reference CPU regardless of what
/// is registered" behavior (`audit-complet-cuda-hot-path-2026-09-08`'s
/// P1-1): a graph node with no `resource_affinity` of its own (true of
/// every Qwen graph builder, Rust- and Component-produced alike) falls
/// back to whatever this function passes as `provider`, and Kernel
/// Registry selection then fails closed for any Provider that isn't the
/// one named there. Never imports or references any concrete non-
/// Reference-CPU Provider type: `provider` arrives as a plain
/// `ProviderBinding` (a name), resolved against whatever the caller
/// actually registered.
pub fn prepare_first_native_execution_plans_for_provider(
    runtime: &Runtime,
    instance: &ModelInstanceId,
    graphs: FirstNativeComponentGraphs,
    prompt_token_count: u64,
    provider: &ProviderBinding,
) -> Result<FirstNativePreparedPlans, E2eConformanceError> {
    let status = require_ready_first_native_instance(runtime, instance)?;
    let mutation_version = status.status().mutation_version;
    Ok(FirstNativePreparedPlans {
        prefill: prepare_first_native_plan_for_graph(
            runtime,
            &graphs.prefill,
            instance,
            mutation_version,
            prompt_token_count,
            PreparedExecutionPlanGeneration::new(1),
            provider,
        )?,
        prefill_node_count: graphs.prefill_node_count,
        decode: prepare_first_native_plan_for_graph(
            runtime,
            &graphs.decode,
            instance,
            mutation_version,
            1,
            PreparedExecutionPlanGeneration::new(1),
            provider,
        )?,
        decode_node_count: graphs.decode_node_count,
    })
}

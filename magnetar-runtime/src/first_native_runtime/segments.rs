//! Provider runs of the first-native graph, whole and by multi-device segment.

use super::*;

/// Registers Kernel Registry `PreparedKernel` entries for every one of
/// `provider`'s own advertised Kernels, bound to `provider`'s own metadata
/// name and its (first) discovered Device -- the Provider-agnostic core
/// [`register_reference_cpu_prepared_kernels`] is a thin wrapper over, so
/// a first-native run against *any* registered Provider (not only
/// Reference CPU) can reach a `Ready` `PreparedKernel` generation for
/// every Kernel it advertises. Never imports or references any concrete
/// non-Reference-CPU Provider type: resolved entirely through the
/// `Provider`/`Device` trait objects this crate already depends on.
/// [`register_reference_cpu_prepared_kernels`], generalized to an
/// arbitrary registered Provider instead of hardcoding Reference CPU.
/// Public alongside it (task group 12): an embedder targeting a Provider
/// other than Reference CPU (e.g. CUDA) for production generation needs
/// this same registration step.
pub fn register_prepared_kernels_for_provider(
    runtime: &mut Runtime,
    provider: &dyn Provider,
) -> Result<(), E2eConformanceError> {
    let provider_binding = ProviderBinding::new(provider.metadata().name.clone());
    let device_binding = provider
        .devices()
        .first()
        .map(|device| DeviceBinding::new(device.id().clone()))
        .ok_or_else(|| E2eConformanceError::ModelComponentFailed {
            reason: format!(
                "provider '{provider_binding}' has no discovered Device to bind prepared kernels to"
            ),
        })?;
    let mut prepared_ids = PreparedKernelIdAllocator::default();
    for advertisement in provider.kernel_advertisements() {
        let id = prepared_ids.allocate();
        let mut prepared = PreparedKernel::new(
            id,
            advertisement.id.clone(),
            CompiledKernelArtifactId::from_digest(format!(
                "builtin:{}",
                advertisement.id.stable_key()
            )),
            provider_binding.clone(),
            device_binding.clone(),
            PreparedKernelGeneration::new(1),
        );
        prepared
            .mark_ready()
            .map_err(|error| E2eConformanceError::ModelComponentFailed {
                reason: format!(
                    "prepared kernel for '{}' could not become Ready: {error}",
                    advertisement.id.stable_key()
                ),
            })?;
        runtime
            .kernel_registry_mut()
            .register_prepared_kernel(prepared);
        runtime
            .kernel_registry_mut()
            .promote_generation(&advertisement.id, id)
            .map_err(|error| E2eConformanceError::ModelComponentFailed {
                reason: format!(
                    "prepared kernel generation for '{}' could not be promoted: {error}",
                    advertisement.id.stable_key()
                ),
            })?;
    }
    Ok(())
}

/// Everything an external caller needs to verify Device residency,
/// Provider/Device `ResourceAffinity`, and Kernel provenance directly
/// after one real first-native dispatch run through
/// [`run_first_native_graph_with_provider`]. `runtime` is deliberately
/// still alive and owned here (not torn down the way the full generate-
/// and-cleanup entrypoints in this file are), so its
/// `MemoryManager`/`TensorResidency` state can be inspected before
/// anything is unloaded.
pub struct FirstNativeProviderRunOutcome {
    pub runtime: Runtime,
    pub instance: ModelInstanceId,
    pub dispatch: KernelDispatchResult,
    pub bindings: BTreeMap<TensorEdgeId, HostTensor>,
    pub resolved_provider: ProviderBinding,
    /// This dispatch's own updated per-layer KV state, keyed by real layer
    /// number (`LayerKvMap`) -- ignored by every pre-existing single-
    /// dispatch caller in this family, and consumed by a caller threading
    /// it into a later decode step against the *same* Model Instance/
    /// Provider (`add-real-multi-device-model-instance-placement`'s
    /// segment decode loop).
    pub layer_kv: LayerKvMap,
}

/// Runs one real first-native dispatch -- real Model Loading, real weight
/// materialization, a real Prepared Execution Plan, and the actual
/// `execute_first_native_graph` entrypoint every production first-native call
/// goes through -- against `provider`, an arbitrary registered Provider,
/// instead of the Reference CPU every other E2E helper in this file
/// hardcodes (`audit-complet-cuda-hot-path-2026-09-08` P1-1). `graph` is
/// the caller's own choice of Execution Graph: the real Qwen Model
/// Component's own output (via this crate's Component Engine), or one
/// directly Rust-constructed via the already-public
/// `qwen_prefill_graph`/`qwen_decode_graph` -- both are equally real
/// graph-building recipes, and this function does not care which produced
/// `graph`. `kv_cache_id` names the KV Cache session this dispatch
/// participates in (a fresh, caller-chosen id for a standalone run).
///
/// Never imports or references any concrete non-Reference-CPU Provider
/// type: `provider` arrives as a `Provider` trait object this crate
/// already depends on, and every Provider/Device resolution downstream
/// (`execute_first_native_graph`, `resolve_first_native_weight_edge`,
/// `resident_resource_affinity`, `resolved_resource_affinity`, ...) was
/// already generic before this change -- the two pieces that did hardcode
/// Reference CPU (`build_runtime_with_model_execution_engine`'s
/// registration and Kernel Registry seeding, `prepare_first_native_plan_
/// for_graph`'s Plan scope/fallback affinity) are generalized above this
/// function via their own `_for_provider`/`_and_provider` siblings, not
/// worked around here.
pub fn run_first_native_graph_with_provider(
    provider: Arc<dyn Provider>,
    fixture: &E2eFixture,
    graph: &ExecutionGraph,
    kv_cache_id: &KvCacheId,
    token_ids: &[u32],
) -> Result<FirstNativeProviderRunOutcome, E2eConformanceError> {
    let provider_binding = ProviderBinding::new(provider.metadata().name.clone());
    let mut runtime = build_runtime_with_model_execution_engine_and_provider(fixture, provider)?;
    let (instance, _memory) =
        load_fixture_instance_for_provider(fixture, &mut runtime, &provider_binding)?;
    run_first_native_graph_dispatch(
        runtime,
        fixture,
        instance,
        &provider_binding,
        graph,
        kv_cache_id,
        token_ids,
    )
}

/// [`run_first_native_graph_with_provider`], generalized to materialize
/// caller-supplied `weights` (through
/// `load_fixture_instance_with_weights_for_provider`) instead of the one
/// canonical E2E fixture's own digest-checked weights -- lets a caller run
/// the real first-native pipeline against a `fixture`/`FirstNativeModelConfig`/`graph`
/// of their own choosing end to end, for example a genuinely grouped-
/// query-shaped configuration this crate's own fixture is not.
pub fn run_first_native_graph_with_provider_and_weights(
    provider: Arc<dyn Provider>,
    fixture: &E2eFixture,
    weights: &BTreeMap<String, HostTensor>,
    graph: &ExecutionGraph,
    kv_cache_id: &KvCacheId,
    token_ids: &[u32],
) -> Result<FirstNativeProviderRunOutcome, E2eConformanceError> {
    let provider_binding = ProviderBinding::new(provider.metadata().name.clone());
    let mut runtime = build_runtime_with_model_execution_engine_and_provider(fixture, provider)?;
    let instance = load_fixture_instance_with_weights_for_provider(
        fixture,
        &mut runtime,
        &provider_binding,
        weights,
    )?;
    run_first_native_graph_dispatch(
        runtime,
        fixture,
        instance,
        &provider_binding,
        graph,
        kv_cache_id,
        token_ids,
    )
}

/// Shared dispatch tail for [`run_first_native_graph_with_provider`] and
/// [`run_first_native_graph_with_provider_and_weights`]: builds a real
/// Prepared Execution Plan for `graph` bound to `provider_binding`, then
/// runs the actual `execute_first_native_graph` entrypoint every production
/// first-native call goes through. The two callers differ only in how
/// `instance`'s weights were materialized before reaching here.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_first_native_graph_dispatch(
    mut runtime: Runtime,
    fixture: &E2eFixture,
    instance: ModelInstanceId,
    provider_binding: &ProviderBinding,
    graph: &ExecutionGraph,
    kv_cache_id: &KvCacheId,
    token_ids: &[u32],
) -> Result<FirstNativeProviderRunOutcome, E2eConformanceError> {
    let status = require_ready_first_native_instance(&runtime, &instance)?;
    let mutation_version = status.status().mutation_version;
    let mut plan = prepare_first_native_plan_for_graph(
        &runtime,
        graph,
        &instance,
        mutation_version,
        token_ids.len() as u64,
        PreparedExecutionPlanGeneration::new(1),
        provider_binding,
    )?;
    let ids_tensor = HostTensor::new(
        [token_ids.len() as u64],
        token_ids.iter().map(|id| *id as f32).collect::<Vec<_>>(),
    )
    .map_err(|error| E2eConformanceError::GenerationFailed {
        reason: error.to_string(),
    })?;
    let mut node_events = Vec::new();
    let (dispatch, bindings, layer_kv, resolved_provider) = execute_first_native_graph(
        &mut runtime,
        fixture,
        &instance,
        kv_cache_id,
        graph,
        &mut plan,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids_tensor)]),
        None,
        Some(0),
        &mut node_events,
    )
    .map_err(E2eConformanceError::from)?;
    Ok(FirstNativeProviderRunOutcome {
        runtime,
        instance,
        dispatch,
        bindings,
        resolved_provider,
        layer_kv,
    })
}

/// [`load_fixture_instance_with_weights_for_provider`], restricted to
/// decoder layer range `[start_layer, end_layer)`
/// (`add-real-multi-device-model-instance-placement`): filters both
/// `fixture.manifest`'s tensor inventory (so `load_model`'s
/// `required_weight_names` reflects only this segment's own subset,
/// exactly like [`filter_manifest_for_layer_range`]'s own doc comment)
/// and `weights` itself (after deriving `lm_head` from the *full* set,
/// same ordering reason as [`load_fixture_instance_with_weights`]) down
/// to that same subset before materializing.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub(super) fn load_fixture_instance_segment_with_weights_for_provider(
    fixture: &E2eFixture,
    runtime: &mut Runtime,
    provider_binding: &ProviderBinding,
    weights: &BTreeMap<String, HostTensor>,
    start_layer: u32,
    end_layer: u32,
) -> Result<ModelInstanceId, E2eConformanceError> {
    let num_hidden_layers = fixture.config.architecture.layer_count as u32;
    let segment_manifest = filter_manifest_for_layer_range(
        &fixture.manifest,
        start_layer,
        end_layer,
        num_hidden_layers,
    );
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(fixture.architecture_implementation.clone());
    let mut request = ModelLoadingRequest::new(
        ModelLoadingRequestId::new(format!(
            "first-native-provider-segment-load-{start_layer}-{end_layer}"
        )),
        segment_manifest.id.clone(),
    );
    request.quantization_policy = ModelQuantizationPolicy::RejectUnsupported;
    let loaded = load_model(
        &mut coordinator,
        runtime,
        ModelLoadingApiRequest::new(request),
        &segment_manifest,
    )?;
    let instance = create_model_instance(
        runtime,
        &loaded,
        fixture.architecture_implementation.clone(),
        ResourceAffinity::new(FallbackClass::Transparent).with_provider(provider_binding.clone()),
    )?;
    let mut segment_weights = weights.clone();
    qwen_weights_with_derived_lm_head(fixture, &mut segment_weights)?;
    segment_weights.retain(|name, _| {
        qwen_weight_name_in_layer_range(name, start_layer, end_layer, num_hidden_layers)
    });
    materialize_model_instance_weights(
        runtime,
        &instance,
        segment_manifest.id.name.as_str(),
        &segment_weights,
    )
    .map_err(E2eConformanceError::from)?;
    Ok(instance)
}

/// A segment graph's own starting boundary input for a segment not
/// beginning at layer 0 (`input.hidden_states_in`, `model-component-
/// graph.wit` 1.3.0) -- `add-real-multi-device-model-instance-placement`'s
/// real, zero-Host-round-trip counterpart of the original Host-only
/// boundary.
pub enum QwenSegmentBoundaryInput {
    /// A Host-resident tensor, written into this segment's own Provider
    /// storage the same way any other graph input is -- a real, explicit
    /// data movement step, but one that stages through the Host.
    Host(HostTensor),
    /// A tensor ALREADY resident in this segment's own Provider storage
    /// under `resource_id`, real shape `shape` -- e.g. written directly by
    /// a real, zero-Host-round-trip Device-to-Device move
    /// (`CudaExecutor::copy_tensor_from_peer_admitted`, `add-real-peer-to-
    /// peer-gpu-movement`) before this dispatch. No write happens here at
    /// all; the resource id is simply recorded as this edge's own binding.
    Resident {
        resource_id: TensorResourceId,
        shape: Vec<u64>,
    },
}

/// `run_first_native_graph_dispatch`, restricted to decoder layer range
/// `[start_layer, end_layer)` (`add-real-multi-device-model-instance-
/// placement`): binds `token_ids` to `input.token_ids` when `start_layer
/// == 0` (a segment's own first, leftmost range), exactly like a full
/// graph; otherwise binds `boundary_input` (the *previous* segment's own
/// real output, either Host-staged or already Device-resident -- see
/// [`QwenSegmentBoundaryInput`]) to `input.hidden_states_in`, matching
/// `model-component-graph.wit` 1.3.0's segment boundary semantics.
/// `kv_history` is this SAME segment's own prior-step KV state (`None` for
/// a prefill-segment graph, `Some` for a decode-segment graph continuing a
/// prior prefill or decode step against this segment); `absolute_position_
/// override` is the real absolute token position this dispatch's RoPE
/// terms must use (`0` for prefill, the real count of already-committed
/// tokens for decode) -- both threaded straight through to
/// `execute_first_native_graph`, exactly like the full-model dispatch family
/// already does.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[allow(clippy::too_many_arguments)]
pub fn run_first_native_graph_segment_dispatch(
    mut runtime: Runtime,
    fixture: &E2eFixture,
    instance: ModelInstanceId,
    provider_binding: &ProviderBinding,
    graph: &ExecutionGraph,
    kv_cache_id: &KvCacheId,
    token_ids: &[u32],
    start_layer: u32,
    boundary_input: Option<QwenSegmentBoundaryInput>,
    kv_history: Option<&LayerKvMap>,
    absolute_position_override: Option<u64>,
) -> Result<FirstNativeProviderRunOutcome, E2eConformanceError> {
    let status = require_ready_first_native_instance(&runtime, &instance)?;
    let mutation_version = status.status().mutation_version;
    let sequence_length = if start_layer == 0 {
        token_ids.len() as u64
    } else {
        let shape = match &boundary_input {
            Some(QwenSegmentBoundaryInput::Host(tensor)) => Some(&tensor.shape),
            Some(QwenSegmentBoundaryInput::Resident { shape, .. }) => Some(shape),
            None => None,
        };
        shape
            .and_then(|shape| shape.first().copied())
            .ok_or_else(|| E2eConformanceError::FixtureInvalid {
                reason: "segment starting mid-stack requires a boundary hidden-state tensor \
                          with a non-empty shape"
                    .into(),
            })?
    };
    let mut plan = prepare_first_native_plan_for_graph(
        &runtime,
        graph,
        &instance,
        mutation_version,
        sequence_length,
        PreparedExecutionPlanGeneration::new(1),
        provider_binding,
    )?;
    let (initial_bindings, resident_bindings) = if start_layer == 0 {
        let ids_tensor = HostTensor::new(
            [token_ids.len() as u64],
            token_ids.iter().map(|id| *id as f32).collect::<Vec<_>>(),
        )
        .map_err(|error| E2eConformanceError::GenerationFailed {
            reason: error.to_string(),
        })?;
        (
            BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids_tensor)]),
            BTreeMap::new(),
        )
    } else {
        match boundary_input.ok_or_else(|| E2eConformanceError::FixtureInvalid {
            reason: "segment starting mid-stack requires a boundary hidden-state tensor".into(),
        })? {
            QwenSegmentBoundaryInput::Host(hidden) => (
                BTreeMap::from([(TensorEdgeId::new("input.hidden_states_in"), hidden)]),
                BTreeMap::new(),
            ),
            QwenSegmentBoundaryInput::Resident { resource_id, shape } => (
                BTreeMap::new(),
                BTreeMap::from([(
                    TensorEdgeId::new("input.hidden_states_in"),
                    (resource_id, shape),
                )]),
            ),
        }
    };
    let mut node_events = Vec::new();
    let (dispatch, bindings, layer_kv, resolved_provider) =
        execute_first_native_graph_with_resident_input(
            &mut runtime,
            fixture,
            &instance,
            kv_cache_id,
            graph,
            &mut plan,
            initial_bindings,
            resident_bindings,
            kv_history,
            absolute_position_override,
            &mut node_events,
            MaterializedEdges::All,
            None,
            None,
        )
        .map_err(E2eConformanceError::from)?;
    Ok(FirstNativeProviderRunOutcome {
        runtime,
        instance,
        dispatch,
        bindings,
        resolved_provider,
        layer_kv,
    })
}

/// [`run_first_native_graph_with_provider_and_weights`]'s per-Device
/// pipeline-placement counterpart (`add-real-multi-device-model-instance-
/// placement`, the "two `ModelInstance`s + explicit movement" design):
/// runs one decoder-layer-range SEGMENT's graph
/// (`build_first_native_prefill_graph_segment_for_config`) against a real
/// `provider`, loading only that segment's own weight subset
/// (`load_fixture_instance_segment_with_weights_for_provider`).
///
/// `boundary_input` is `None` for a segment starting at layer 0 (reads
/// `token_ids` instead, like a full graph's own first segment); for a
/// segment starting mid-stack, it is the *previous* segment's own real
/// `"logits"`-named output tensor (a raw post-layer hidden state, not
/// real logits -- see `build_first_native_prefill_graph_segment_for_
/// config`'s doc comment), read from `FirstNativeProviderRunOutcome::
/// bindings` after `execute_first_native_graph`'s own Host-materialization
/// boundary.
///
/// This single-shot convenience wrapper always crosses through the Host
/// between two segments -- fine for a quick prefill-only proof (Phase
/// B.1/B.2's own single-dispatch tests). A caller wanting the real,
/// zero-Host-round-trip Device-to-Device move
/// (`CudaExecutor::copy_tensor_from_peer_admitted`, `add-real-peer-to-
/// peer-gpu-movement`) for this boundary instead should call
/// [`run_first_native_graph_segment_dispatch`] directly with
/// [`QwenSegmentBoundaryInput::Resident`], after writing the peer-copied
/// data into this segment's own Provider storage themselves -- the
/// generalization this wrapper's own single-`HostTensor` shape does not
/// expose.
///
/// A single-shot convenience wrapper (loads a fresh segment Model
/// Instance, then runs exactly one dispatch): fine for a one-off prefill,
/// but a real multi-step decode loop across many steps needs to load
/// *once* and dispatch *repeatedly* against the same Instance instead --
/// see [`load_first_native_segment_with_provider_and_weights`] (load
/// only) and [`run_first_native_graph_segment_dispatch`] (pub, reusable
/// per-step dispatch that hands its own `runtime`/`instance` back on
/// [`FirstNativeProviderRunOutcome`] for the next call).
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[allow(clippy::too_many_arguments)]
pub fn run_first_native_graph_segment_with_provider_and_weights(
    provider: Arc<dyn Provider>,
    fixture: &E2eFixture,
    weights: &BTreeMap<String, HostTensor>,
    graph: &ExecutionGraph,
    kv_cache_id: &KvCacheId,
    token_ids: &[u32],
    start_layer: u32,
    end_layer: u32,
    boundary_input: Option<HostTensor>,
) -> Result<FirstNativeProviderRunOutcome, E2eConformanceError> {
    let (runtime, instance, provider_binding) =
        load_first_native_segment_with_provider_and_weights(
            provider,
            fixture,
            weights,
            start_layer,
            end_layer,
        )?;
    run_first_native_graph_segment_dispatch(
        runtime,
        fixture,
        instance,
        &provider_binding,
        graph,
        kv_cache_id,
        token_ids,
        start_layer,
        boundary_input.map(QwenSegmentBoundaryInput::Host),
        None,
        Some(0),
    )
}

/// Loads a segment-only Model Instance (`load_fixture_instance_segment_
/// with_weights_for_provider`) against a real, caller-constructed
/// `provider`, without dispatching anything yet -- the "load once" half
/// [`run_first_native_graph_segment_with_provider_and_weights`]'s single-
/// shot convenience wrapper conflates with its one dispatch. A real
/// multi-step decode loop across many steps calls this once, then
/// [`run_first_native_graph_segment_dispatch`] repeatedly against the
/// returned `(Runtime, ModelInstanceId, ProviderBinding)`, threading each
/// step's own `FirstNativeProviderRunOutcome::{runtime, layer_kv}`
/// forward into the next call.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub fn load_first_native_segment_with_provider_and_weights(
    provider: Arc<dyn Provider>,
    fixture: &E2eFixture,
    weights: &BTreeMap<String, HostTensor>,
    start_layer: u32,
    end_layer: u32,
) -> Result<(Runtime, ModelInstanceId, ProviderBinding), E2eConformanceError> {
    let provider_binding = ProviderBinding::new(provider.metadata().name.clone());
    let mut runtime = build_runtime_with_model_execution_engine_and_provider(fixture, provider)?;
    let instance = load_fixture_instance_segment_with_weights_for_provider(
        fixture,
        &mut runtime,
        &provider_binding,
        weights,
        start_layer,
        end_layer,
    )?;
    Ok((runtime, instance, provider_binding))
}

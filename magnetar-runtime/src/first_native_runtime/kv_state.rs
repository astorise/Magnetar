//! KV state of a first-native request: pending layers, commit, discard and release.

use super::*;

/// One promoted layer's committed binding, and the binding it replaces (if
/// any), pending this transaction's `commit`.
pub(super) struct PromotedKvLayer {
    pub(super) layer: u32,
    pub(super) binding: KvLayerResourceBinding,
    pub(super) previous: Option<KvLayerResourceBinding>,
    /// Valid rows of an in-place KV store (`append-kv-history-in-place`);
    /// `None` for a copy-based generation.
    pub(super) rows: Option<u64>,
}

/// The KV commit lifecycle, named and typed explicitly (Correctif 11 / task
/// group 9): admission, pending-Resource-ID lookup, Provider materialization
/// under an attempt-unique resource id, then either `commit` (publish every
/// layer's binding to the cache and release the resources each one
/// replaces) or `abort` (release everything this attempt admitted, leaving
/// the cache's prior committed state completely untouched). A transaction
/// promotes every layer under a *new* resource id first, without touching
/// the cache's existing committed bindings at all -- only a fully successful
/// `commit` publishes anything, so a decode step's failed commit cannot
/// leave some layers pointing at this step's data and others at the
/// previous step's.
pub(super) struct KvUpdateTransaction {
    pub(super) provider_binding: ProviderBinding,
    pub(super) executor: Arc<dyn ProviderExecutionApi>,
    pub(super) promoted: Vec<PromotedKvLayer>,
}

impl KvUpdateTransaction {
    /// Resolves the Provider `execute_first_native_graph` actually wrote this step's
    /// pending K/V resources under (carried on `state.provider`), falling
    /// back to Reference CPU only when unset -- `generalize-first-native-
    /// provider-dispatch`'s fix: this used to hardcode Reference CPU
    /// unconditionally, so a non-CPU step's pending writes would be looked
    /// for under the wrong Provider binding at commit time.
    pub(super) fn begin(
        runtime: &Runtime,
        state: &FirstNativeExecutionKvState,
    ) -> Result<Self, InferenceApiError> {
        let provider_binding = state
            .provider
            .clone()
            .unwrap_or_else(|| ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME));
        let executor = resolve_kernel_execution_provider(runtime, &provider_binding)?;
        Ok(Self {
            provider_binding,
            executor,
            promoted: Vec::new(),
        })
    }

    /// Admits and writes one layer's pending K/V under attempt-unique
    /// resource ids. Does not touch the cache's existing committed
    /// bindings; on failure, the caller should call `abort` to release
    /// everything already promoted earlier this attempt.
    pub(super) fn promote_layer(
        &mut self,
        runtime: &mut Runtime,
        cache: &KvCacheId,
        layer: usize,
        pending: &FirstNativeLayerKvState,
    ) -> Result<(), InferenceApiError> {
        let previous = runtime
            .kv_cache(cache)?
            .layer_resources
            .get(&(layer as u32))
            .cloned();
        // In-place KV (`append-kv-history-in-place`): the step wrote its
        // rows into the layer's stores beyond the committed rows, so
        // committing only adopts the new row count; nothing is copied.
        if let Some(rows) = pending.rows {
            let allocation = |id: &TensorResourceId| {
                runtime
                    .memory()
                    .tensor_residency(id)
                    .and_then(|residency| residency.allocation)
                    .ok_or_else(|| InferenceApiError::KvCacheUnavailable {
                        reason: format!("in-place KV store '{id}' has no admitted allocation"),
                    })
            };
            let binding = KvLayerResourceBinding {
                k: pending.k.clone(),
                v: pending.v.clone(),
                k_allocation: allocation(&pending.k)?,
                v_allocation: allocation(&pending.v)?,
            };
            self.promoted.push(PromotedKvLayer {
                layer: layer as u32,
                binding,
                previous,
                rows: Some(rows),
            });
            return Ok(());
        }
        let binding = promote_pending_kv_layer(
            runtime,
            &self.executor,
            &self.provider_binding,
            cache,
            layer,
            pending,
        )?;
        self.promoted.push(PromotedKvLayer {
            layer: layer as u32,
            binding,
            previous,
            rows: None,
        });
        Ok(())
    }

    /// Releases every resource admitted so far this attempt, leaving the
    /// cache's prior committed state completely untouched.
    pub(super) fn abort(self, runtime: &mut Runtime) {
        // In-place stores are shared with the committed history and were
        // not created by this transaction: nothing of theirs to release.
        for already in self.promoted.iter().filter(|layer| layer.rows.is_none()) {
            let _ = runtime.memory_mut().release(already.binding.k_allocation);
            let _ = runtime.memory_mut().release(already.binding.v_allocation);
            // Best-effort, like the memory releases above: this is already
            // an unwind over a different failure, with nothing further to
            // roll back to if the Provider release itself fails.
            let _ = self
                .executor
                .release_admitted_tensor(runtime.memory_mut(), &already.binding.k);
            let _ = self
                .executor
                .release_admitted_tensor(runtime.memory_mut(), &already.binding.v);
        }
    }

    /// Publishes every promoted layer's binding to the cache and releases
    /// the resources each one replaces, reached only once every layer in
    /// this transaction has been successfully promoted.
    pub(super) fn commit(
        self,
        runtime: &mut Runtime,
        cache: &KvCacheId,
    ) -> Result<LayerKvMap, InferenceApiError> {
        let mut committed = LayerKvMap::new();
        for promoted_layer in self.promoted {
            if let Some(previous) = promoted_layer.previous
                // The same in-place store, still in use.
                && previous.k != promoted_layer.binding.k
            {
                let _ = runtime.memory_mut().release(previous.k_allocation);
                let _ = runtime.memory_mut().release(previous.v_allocation);
                // Best-effort: the new binding above already promoted
                // successfully, so a failure releasing the superseded one
                // is a leak to surface separately, not a reason to fail
                // this otherwise-successful commit.
                // `release_admitted_tensor`, not `release_tensor`: the
                // superseded generation was admitted (and is tracked) by
                // the Provider, so dropping only its storage would leave a
                // stale tracking entry per layer and role per decode step
                // (`bound-memory-manager-bookkeeping`).
                let _ = self
                    .executor
                    .release_admitted_tensor(runtime.memory_mut(), &previous.k);
                let _ = self
                    .executor
                    .release_admitted_tensor(runtime.memory_mut(), &previous.v);
            }
            runtime
                .kv_caches_mut()
                .cache_mut(cache)?
                .layer_resources
                .insert(promoted_layer.layer, promoted_layer.binding.clone());
            committed.insert(
                promoted_layer.layer as usize,
                FirstNativeLayerKvState {
                    k: promoted_layer.binding.k,
                    v: promoted_layer.binding.v,
                    rows: promoted_layer.rows,
                },
            );
        }
        Ok(committed)
    }
}

/// Promotes one layer's pending K and V together, admitting and writing
/// both under attempt-unique resource ids. Neither publishes anything to
/// the cache nor releases any prior allocation -- both are
/// [`KvUpdateTransaction::commit`]'s responsibility once it knows the
/// *entire* multi-layer commit succeeded.
pub(super) fn promote_pending_kv_layer(
    runtime: &mut Runtime,
    executor: &Arc<dyn ProviderExecutionApi>,
    provider_binding: &ProviderBinding,
    cache: &KvCacheId,
    layer: usize,
    pending: &FirstNativeLayerKvState,
) -> Result<KvLayerResourceBinding, InferenceApiError> {
    let k = promote_pending_kv_layer_role(
        runtime,
        executor,
        provider_binding,
        cache,
        layer,
        "k",
        &pending.k,
    )?;
    let v = promote_pending_kv_layer_role(
        runtime,
        executor,
        provider_binding,
        cache,
        layer,
        "v",
        &pending.v,
    )?;
    Ok(KvLayerResourceBinding {
        k: k.0,
        v: v.0,
        k_allocation: k.1,
        v_allocation: v.1,
    })
}

/// Monotonic source of attempt-unique KV commit resource-id suffixes
/// (`implement-device-resident-multi-step-cuda-decode`): previously this
/// promotion used the freshly-created `MemoryAllocationId` for uniqueness,
/// but `copy_tensor_admitted` (used below to avoid a host round trip)
/// must be told the destination resource id *before* it admits anything,
/// so no allocation id exists yet to derive a name from. Global and
/// process-wide rather than per-cache/per-layer is deliberate: the only
/// property this needs is "never repeats," which a single counter
/// trivially guarantees, and every real caller commits layers one at a
/// time anyway (no cross-layer parallelism to race).
pub(super) static KV_COMMIT_GENERATION: AtomicU64 = AtomicU64::new(0);

pub(super) fn next_kv_commit_generation() -> u64 {
    KV_COMMIT_GENERATION.fetch_add(1, Ordering::Relaxed)
}

#[allow(clippy::too_many_arguments)]
/// Releases a step's pending KV resource: the Provider's storage and the
/// allocation it tracks for it (a pending write admitted by the Provider,
/// `copy_tensor_admitted`), and a Runtime pre-admission recorded for it (a
/// KV append writes its pending resource through `admit_kernel_output`,
/// which the Provider does not track).
pub(super) fn release_pending_kv_resource(
    runtime: &mut Runtime,
    provider: &Arc<dyn ProviderExecutionApi>,
    id: &TensorResourceId,
) {
    let _ = provider.release_admitted_tensor(runtime.memory_mut(), id);
    if let Some(allocation) = runtime
        .memory_mut()
        .remove_tensor_residency(id)
        .and_then(|residency| residency.allocation)
    {
        let _ = runtime.memory_mut().release(allocation);
    }
}

pub(super) fn promote_pending_kv_layer_role(
    runtime: &mut Runtime,
    executor: &Arc<dyn ProviderExecutionApi>,
    provider_binding: &ProviderBinding,
    cache: &KvCacheId,
    layer: usize,
    role: &str,
    pending_resource: &TensorResourceId,
) -> Result<(TensorResourceId, MemoryAllocationId), InferenceApiError> {
    // The resource id is unique to this promotion attempt (keyed by a
    // monotonic generation counter), not the stable
    // `kv.{cache}.layer{N}.{role}` name a naive implementation might reuse
    // across decode steps -- reusing a stable id would mean this write
    // destructively overwrites the previous step's still-valid committed
    // bytes before the caller knows whether the *whole* multi-layer commit
    // succeeds, making rollback impossible to do correctly (Correctif 11).
    let committed_resource = TensorResourceId::new(format!(
        "kv.{cache}.layer{layer}.{role}.gen{}",
        next_kv_commit_generation()
    ));
    // Checked explicitly, before attempting the copy: a missing pending
    // resource (already committed, aborted, or sabotaged mid-commit by a
    // partial-layer-failure test) is a `KvCacheUnavailable` case this
    // caller already has a specific, tested contract for -- not a
    // `copy_tensor_admitted` failure, whose own error shape does not
    // reliably distinguish "missing" from a genuine device-copy failure
    // across every Provider. `read_tensor_value` is a pure presence check
    // (never downloads, per its own contract), so this costs nothing on a
    // device-resident Provider.
    if executor.read_tensor_value(pending_resource).is_none() {
        return Err(InferenceApiError::KvCacheUnavailable {
            reason: format!(
                "no pending KV data to commit for layer {layer} ({role}); already committed or aborted?"
            ),
        });
    }
    // Real Provider-side copy: admits and copies in one call (mirroring
    // `write_tensor_admitted`'s own admit-then-write shape), never
    // downloading the pending resource to host merely to change its
    // identity (`implement-device-resident-multi-step-cuda-decode`).
    // `provider_binding` is threaded through purely for its own error-
    // reason formatting below; `copy_tensor_admitted` derives the real
    // Memory Placement (Device or ProviderOwnedOpaque) from the resolved
    // Provider itself, exactly like `write_tensor_admitted` already does.
    let allocation_id = executor
        .copy_tensor_admitted(
            runtime.memory_mut(),
            pending_resource,
            committed_resource.clone(),
            MemoryAllocationClass::Tensor,
            MemoryAllocationOwner::Session(cache.to_string()),
        )
        .map_err(|error| match error {
            TensorValueAdmissionError::Memory(error) => InferenceApiError::MemoryAdmissionFailed {
                reason: format!(
                    "failed to account committed KV resource for layer {layer} ({role}) on \
                     provider '{provider_binding}': {error}"
                ),
            },
            TensorValueAdmissionError::Provider(error) => {
                InferenceApiError::ProviderTensorWriteFailed {
                    reason: format!(
                        "failed to commit KV resource for layer {layer} ({role}) on provider \
                         '{provider_binding}': {error}"
                    ),
                }
            }
        })?;
    Ok((committed_resource, allocation_id))
}

impl E2eRuntimeModelExecutionEngine {
    /// One decode step's graph execution: a replay of the request's
    /// captured step when it has one, a captured per-node execution at
    /// [`STEP_CAPTURE_DECODE_INDEX`], and a per-node execution otherwise or
    /// whenever capture or replay cannot be used
    /// (`capture-and-replay-cuda-graphs`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute_decode_graph(
        &self,
        runtime: &mut Runtime,
        request: &GenerationRequest,
        kv_state: &FirstNativeExecutionKvState,
        execution_plan: &mut PreparedExecutionPlan,
        graphs: &CachedGraphs,
        model_instance: &ModelInstanceId,
        token: TokenId,
        absolute_position: u64,
        decode_index: usize,
        node_events: &mut Vec<PerNodeCausalEvent>,
        kv_store_capacity: Option<u64>,
        greedy: Option<&DeviceGreedyPlan>,
    ) -> Result<(FirstNativeGraphExecutionOutput, Option<u64>), InferenceApiError> {
        let ids_tensor =
            HostTensor::new([1], vec![token as f32]).map_err(runtime_generation_failed)?;
        let key = Self::kv_state_key(request);
        // Only steps whose KV history is kept in place: their shapes and
        // resources are the same from one step to the next.
        let replayable = self.step_replay.is_enabled()
            && kv_store_capacity.is_some()
            && !kv_state.layer_kv.is_empty()
            && kv_state.layer_kv.values().all(|layer| layer.rows.is_some());
        if replayable {
            if let Some(recorded) = self.step_replay.recorded(&key) {
                if let Some(output) = replay_decode_step(
                    runtime,
                    &recorded,
                    ids_tensor.clone(),
                    absolute_position,
                    greedy,
                )? {
                    return Ok(output);
                }
                // Invalidated: this step runs per node, a later one may
                // capture again.
                self.step_replay.release(&key);
            } else if decode_index >= STEP_CAPTURE_DECODE_INDEX
                && !self.step_replay.is_declined(&key)
            {
                match self.capture_decode_step(
                    runtime,
                    kv_state,
                    execution_plan,
                    graphs,
                    model_instance,
                    ids_tensor.clone(),
                    absolute_position,
                    node_events,
                    kv_store_capacity,
                )? {
                    Some((recorded, output)) => {
                        trace_step_capture("captured", format!("decode step {decode_index}"));
                        self.step_replay.record(key, recorded);
                        return Ok((output, None));
                    }
                    None => self.step_replay.decline(key),
                }
            }
        }
        execute_first_native_graph_materializing(
            runtime,
            &self.fixture,
            model_instance,
            &kv_state.cache,
            &graphs.decode,
            execution_plan,
            BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids_tensor)]),
            Some(&kv_state.layer_kv),
            Some(absolute_position),
            node_events,
            MaterializedEdges::Only(&["logits"]),
            Some(&graphs.decode_memo),
            kv_store_capacity,
        )
        .map(|output| (output, None))
    }

    /// Runs one decode step per node inside a step capture. `None` when the
    /// Provider cannot capture it: nothing is recorded, and the caller runs
    /// the step again per node (a step is idempotent: its KV rows land
    /// beyond the committed rows and every output is replaced).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn capture_decode_step(
        &self,
        runtime: &mut Runtime,
        kv_state: &FirstNativeExecutionKvState,
        execution_plan: &mut PreparedExecutionPlan,
        graphs: &CachedGraphs,
        model_instance: &ModelInstanceId,
        ids_tensor: HostTensor,
        absolute_position: u64,
        node_events: &mut Vec<PerNodeCausalEvent>,
        kv_store_capacity: Option<u64>,
    ) -> Result<Option<(RecordedDecodeStep, FirstNativeGraphExecutionOutput)>, InferenceApiError>
    {
        let Some(provider_binding) = execution_plan
            .node_bindings
            .first()
            .map(|binding| binding.provider.clone())
        else {
            return Ok(None);
        };
        let executor = resolve_kernel_execution_provider(runtime, &provider_binding)?;
        if !executor.supports_step_capture() {
            trace_step_capture("capture skipped", "the Provider does not support it");
            return Ok(None);
        }
        // The input is written before the capture starts, as the per-node
        // path writes it: a replay writes each step's token into it.
        let input = TensorResourceId::new("input.token_ids");
        let input_shape = ids_tensor.shape.clone();
        let upload_clock = runtime.phase_ledger().clock();
        executor
            .write_tensor_value_admitted(
                runtime.memory_mut(),
                input.clone(),
                TensorValue::Host(ids_tensor),
                MemoryAllocationClass::Tensor,
                MemoryAllocationOwner::Session(kv_state.cache.to_string()),
            )
            .map_err(|error| InferenceApiError::ProviderTensorWriteFailed {
                reason: format!("failed to write graph input resource '{input}': {error}"),
            })?;
        runtime
            .phase_ledger_mut()
            .record(upload_clock, StepPhase::InputUpload);
        let capture_clock = runtime.phase_ledger().clock();
        let request = StepCaptureRequest {
            advancing_attributes: STEP_ADVANCING_ATTRIBUTES
                .iter()
                .map(|name| (*name).to_string())
                .collect(),
        };
        let begun = executor.begin_step_capture(&request);
        runtime
            .phase_ledger_mut()
            .record(capture_clock, StepPhase::StepCapture);
        if let Err(error) = begun {
            trace_step_capture("capture not begun", error);
            return Ok(None);
        }
        let events_before = node_events.len();
        // Nothing is downloaded while capturing.
        let executed = execute_first_native_graph_with_resident_input(
            runtime,
            &self.fixture,
            model_instance,
            &kv_state.cache,
            &graphs.decode,
            execution_plan,
            BTreeMap::new(),
            BTreeMap::from([(
                TensorEdgeId::new("input.token_ids"),
                (input.clone(), input_shape),
            )]),
            Some(&kv_state.layer_kv),
            Some(absolute_position),
            node_events,
            MaterializedEdges::Only(&[]),
            Some(&graphs.decode_memo),
            kv_store_capacity,
        );
        let end_clock = runtime.phase_ledger().clock();
        let captured = match executed {
            Ok(output) => match executor.end_step_capture() {
                Ok(capture) => Some((capture, output)),
                Err(error) => {
                    trace_step_capture("capture not ended", error);
                    None
                }
            },
            Err(error) => {
                trace_step_capture("captured step failed", format!("{error:?}"));
                executor.abort_step_capture();
                None
            }
        };
        runtime
            .phase_ledger_mut()
            .record(end_clock, StepPhase::StepCapture);
        let Some((capture, (dispatch, _, layer_kv, provider))) = captured else {
            node_events.truncate(events_before);
            return Ok(None);
        };
        let logits = graph_edge_resource_id("logits");
        let Some(logits_shape) = executor.resident_shape(&logits) else {
            trace_step_capture("capture dropped", "the logits are not resident");
            executor.release_step_capture(&capture);
            node_events.truncate(events_before);
            return Ok(None);
        };
        let materialize_clock = runtime.phase_ledger().clock();
        let bindings = materialize_first_native_graph_bindings(
            &*executor,
            &BTreeMap::from([(
                TensorEdgeId::new("logits"),
                (logits.clone(), logits_shape.clone()),
            )]),
        )?;
        runtime
            .phase_ledger_mut()
            .record(materialize_clock, StepPhase::OutputMaterialize);
        let recorded = RecordedDecodeStep {
            capture,
            executor: executor.clone(),
            provider: provider.clone(),
            position: absolute_position,
            input,
            logits: (logits, logits_shape),
            layer_kv: layer_kv.clone(),
            dispatch: dispatch.clone(),
        };
        Ok(Some((recorded, (dispatch, bindings, layer_kv, provider))))
    }

    /// Promotes each layer's *pending* K/V resource (written by
    /// `execute_first_native_graph_nodes` during this step) to the cache's
    /// *committed* resource, atomically across every layer, via
    /// [`KvUpdateTransaction`].
    pub(super) fn promote_pending_kv_resources(
        &self,
        runtime: &mut Runtime,
        state: &FirstNativeExecutionKvState,
    ) -> Result<LayerKvMap, InferenceApiError> {
        let mut transaction = KvUpdateTransaction::begin(runtime, state)?;
        for (&layer, pending) in state.layer_kv.iter() {
            if let Err(error) = transaction.promote_layer(runtime, &state.cache, layer, pending) {
                transaction.abort(runtime);
                return Err(error);
            }
        }
        transaction.commit(runtime, &state.cache)
    }
}

#[cfg(test)]
mod tests;

//! Recurrent state of a first-native session (`support-qwen38-flash-next`).
//!
//! A graph's recurrent bindings (`ExecutionGraph::recurrent_state_bindings`)
//! read and replace fixed-size state carried from one step to the next,
//! beside the KV history. The state belongs to the step's KV cache:
//! - a step reads the value pending in the current attempt, else the
//!   committed value, else zeros;
//! - each step's output is copied under an attempt-unique resource and
//!   recorded as pending;
//! - accepting the token commits the pending values together with the KV
//!   layers ([`KvUpdateTransaction::commit`]), and discarding the attempt
//!   releases them, leaving the committed state untouched.

use super::*;

/// Monotonic suffix of recurrent state resources, so a step never
/// overwrites a value a commit or a rollback still needs.
static RECURRENT_STATE_GENERATION: AtomicU64 = AtomicU64::new(0);

/// The graph's recurrent bindings, or none without recurrent edges.
pub(super) fn recurrent_bindings(
    graph: &ExecutionGraph,
) -> Result<Vec<GraphRecurrentStateBinding>, InferenceApiError> {
    if !graph
        .edges
        .values()
        .any(|edge| edge.recurrent_state.is_some())
    {
        return Ok(Vec::new());
    }
    graph
        .recurrent_state_bindings()
        .map_err(|error| InferenceApiError::GraphPlanningFailed {
            reason: format!("invalid recurrent state bindings: {error}"),
        })
}

/// Whether the graph reads or writes recurrent state.
pub(super) fn has_recurrent_state(graph: &ExecutionGraph) -> bool {
    graph
        .edges
        .values()
        .any(|edge| edge.recurrent_state.is_some())
}

/// Binds each recurrent input edge to the cache's state. The caller never
/// supplies these inputs: the state is Runtime-owned.
pub(super) fn bind_recurrent_inputs(
    runtime: &Runtime,
    cache: &KvCacheId,
    bindings: &[GraphRecurrentStateBinding],
    initial: &mut BTreeMap<TensorEdgeId, HostTensor>,
    resident: &mut BTreeMap<TensorEdgeId, (TensorResourceId, Vec<u64>)>,
) -> Result<(), InferenceApiError> {
    if bindings.is_empty() {
        return Ok(());
    }
    let kv_cache =
        runtime
            .kv_cache(cache)
            .map_err(|error| InferenceApiError::KvCacheUnavailable {
                reason: format!("recurrent state needs the step's KV cache '{cache}': {error}"),
            })?;
    for binding in bindings {
        let input = &binding.input_edge;
        if initial.contains_key(input) || resident.contains_key(input) {
            return Err(InferenceApiError::GraphPlanningFailed {
                reason: format!(
                    "recurrent state input '{input}' is bound by the Runtime, not the caller"
                ),
            });
        }
        let layout = &binding.layout;
        let shape = layout.descriptor.shape.dimensions.clone();
        let stored = kv_cache
            .pending_recurrent_resources
            .get(&layout.resource_id)
            .or_else(|| kv_cache.recurrent_resources.get(&layout.resource_id));
        match stored {
            Some(stored) => {
                if stored.shape != shape || stored.compatibility_key != layout.compatibility_key {
                    return Err(InferenceApiError::KvCacheUnavailable {
                        reason: format!(
                            "recurrent state '{}' was written with another layout",
                            layout.resource_id
                        ),
                    });
                }
                resident.insert(input.clone(), (stored.resource.clone(), shape));
            }
            None => {
                let elements = shape.iter().product::<u64>() as usize;
                let zeros = HostTensor::new(shape, vec![0.0; elements])
                    .map_err(runtime_generation_failed)?;
                initial.insert(input.clone(), zeros);
            }
        }
    }
    Ok(())
}

/// Records each recurrent output's value as the cache's pending state,
/// copied under an attempt-unique resource, and releases the pending value
/// it replaces.
pub(super) fn stage_recurrent_outputs(
    runtime: &mut Runtime,
    executor: &Arc<dyn ProviderExecutionApi>,
    cache: &KvCacheId,
    bindings: &[GraphRecurrentStateBinding],
    outputs: &BTreeMap<TensorEdgeId, (TensorResourceId, Vec<u64>)>,
) -> Result<(), InferenceApiError> {
    for binding in bindings {
        let layout = &binding.layout;
        let (source, shape) = outputs.get(&binding.output_edge).ok_or_else(|| {
            InferenceApiError::GraphPlanningFailed {
                reason: format!(
                    "graph produced no value for recurrent state '{}'",
                    layout.resource_id
                ),
            }
        })?;
        let resource = TensorResourceId::new(format!(
            "recurrent.{cache}.{}.gen{}",
            layout.resource_id,
            RECURRENT_STATE_GENERATION.fetch_add(1, Ordering::Relaxed)
        ));
        let allocation = executor
            .copy_tensor_admitted(
                runtime.memory_mut(),
                source,
                resource.clone(),
                MemoryAllocationClass::Tensor,
                MemoryAllocationOwner::Session(cache.to_string()),
            )
            .map_err(|error| InferenceApiError::ProviderTensorWriteFailed {
                reason: format!(
                    "failed to keep recurrent state '{}': {error:?}",
                    layout.resource_id
                ),
            })?;
        let staged = RecurrentStateResourceBinding {
            resource,
            allocation,
            shape: shape.clone(),
            compatibility_key: layout.compatibility_key.clone(),
        };
        let replaced = match runtime.kv_caches_mut().cache_mut(cache) {
            Ok(kv_cache) => kv_cache
                .pending_recurrent_resources
                .insert(layout.resource_id.clone(), staged),
            Err(error) => {
                release_recurrent_resource(runtime, executor, &staged);
                return Err(error.into());
            }
        };
        if let Some(replaced) = replaced {
            release_recurrent_resource(runtime, executor, &replaced);
        }
    }
    Ok(())
}

/// Makes the cache's pending recurrent values its committed state and
/// releases the values they replace. Called once every KV layer of the
/// step has been promoted.
pub(super) fn commit_recurrent_state(
    runtime: &mut Runtime,
    executor: &Arc<dyn ProviderExecutionApi>,
    cache: &KvCacheId,
) -> Result<(), InferenceApiError> {
    let kv_cache = runtime.kv_caches_mut().cache_mut(cache)?;
    let pending = std::mem::take(&mut kv_cache.pending_recurrent_resources);
    let mut replaced = Vec::new();
    for (resource_id, binding) in pending {
        if let Some(previous) = kv_cache.recurrent_resources.insert(resource_id, binding) {
            replaced.push(previous);
        }
    }
    for previous in replaced {
        release_recurrent_resource(runtime, executor, &previous);
    }
    Ok(())
}

/// Releases the cache's pending recurrent values, keeping its committed
/// state. A cache that no longer exists has none.
pub(super) fn discard_recurrent_state(
    runtime: &mut Runtime,
    executor: &Arc<dyn ProviderExecutionApi>,
    cache: &KvCacheId,
) {
    let pending = match runtime.kv_caches_mut().cache_mut(cache) {
        Ok(kv_cache) => std::mem::take(&mut kv_cache.pending_recurrent_resources),
        Err(_) => return,
    };
    for binding in pending.values() {
        release_recurrent_resource(runtime, executor, binding);
    }
}

/// Releases the cache's committed and pending recurrent values (its
/// request's end).
pub(super) fn release_recurrent_state(
    runtime: &mut Runtime,
    executor: &Arc<dyn ProviderExecutionApi>,
    cache: &KvCacheId,
) {
    discard_recurrent_state(runtime, executor, cache);
    let committed = match runtime.kv_caches_mut().cache_mut(cache) {
        Ok(kv_cache) => std::mem::take(&mut kv_cache.recurrent_resources),
        Err(_) => return,
    };
    for binding in committed.values() {
        release_recurrent_resource(runtime, executor, binding);
    }
}

fn release_recurrent_resource(
    runtime: &mut Runtime,
    executor: &Arc<dyn ProviderExecutionApi>,
    binding: &RecurrentStateResourceBinding,
) {
    // Best-effort, like a superseded KV generation's release.
    let _ = executor.release_admitted_tensor(runtime.memory_mut(), &binding.resource);
    if runtime.memory().is_live_allocation(binding.allocation) {
        let _ = runtime.memory_mut().release(binding.allocation);
    }
}

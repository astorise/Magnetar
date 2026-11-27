//! Copy-based KV histories bound by the Runtime as ordinary graph tensors.

use super::graph_execution::{KvRole, parse_kv_cache_id};
use super::*;

type HistorySchema = [(String, String); 2];

pub(super) fn has_history_inputs(graph: &ExecutionGraph) -> bool {
    graph.edges.values().any(|edge| {
        edge.kv_cache
            .as_ref()
            .is_some_and(|metadata| metadata.behavior == GraphKvCacheBehavior::HistoryInput)
    })
}

/// Bind each explicitly declared history input to the request's own KV
/// resources. The matching complete-history output is staged by the normal KV
/// transaction. Components may declare it via the experimental 1.16 ABI.
#[allow(clippy::too_many_arguments)]
pub(super) fn bind_history_inputs(
    runtime: &Runtime,
    executor: &Arc<dyn ProviderExecutionApi>,
    cache: &KvCacheId,
    graph: &ExecutionGraph,
    layer_count: usize,
    history: Option<&LayerKvMap>,
    position: Option<u64>,
    initial: &mut BTreeMap<TensorEdgeId, HostTensor>,
    resident: &mut BTreeMap<TensorEdgeId, (TensorResourceId, Vec<u64>)>,
) -> Result<BTreeMap<usize, HistorySchema>, InferenceApiError> {
    if !has_history_inputs(graph) {
        return Ok(BTreeMap::new());
    }
    let malformed = |reason: String| InferenceApiError::GraphPlanningFailed { reason };
    let mut inputs = Vec::new();
    let mut schemas: BTreeMap<usize, [Option<(String, String)>; 2]> = BTreeMap::new();
    let bindings = graph
        .kv_history_bindings()
        .map_err(|error| malformed(format!("invalid graph history bindings: {error}")))?;
    for binding in &bindings {
        let id = &binding.input_edge;
        let shape = &graph.edges[id].descriptor.shape.dimensions;
        let (layer, role) = parse_kv_cache_id(&binding.cache_id)?;
        let index = match role {
            KvRole::K => 0,
            KvRole::V => 1,
        };
        if layer >= layer_count || binding.cache_id.starts_with(".layer") {
            return Err(malformed(format!("invalid history input '{id}'")));
        }
        // Two different logical names must not write the same physical
        // layer/role: the normal KV transaction has one slot for each.
        for candidate in graph
            .edges
            .values()
            .filter_map(|candidate| candidate.kv_cache.as_ref())
        {
            if matches!(
                candidate.behavior,
                GraphKvCacheBehavior::Output | GraphKvCacheBehavior::Append
            ) {
                let (candidate_layer, candidate_role) = parse_kv_cache_id(&candidate.cache_id)?;
                if candidate_layer == layer
                    && candidate_role == role
                    && (candidate.cache_id != binding.cache_id
                        || candidate.behavior != GraphKvCacheBehavior::Output)
                {
                    return Err(malformed(format!(
                        "conflicting KV output for history input '{id}'"
                    )));
                }
            }
        }
        let schema = (binding.cache_id.clone(), binding.compatibility_key.clone());
        if schemas.entry(layer).or_default()[index]
            .replace(schema)
            .is_some()
        {
            return Err(malformed(format!(
                "duplicate history input for layer {layer}"
            )));
        }
        inputs.push((id, shape, layer, index));
    }
    if inputs.is_empty() {
        return Ok(BTreeMap::new());
    }
    let schemas: BTreeMap<usize, HistorySchema> = schemas
        .into_iter()
        .map(|(layer, [key, value])| {
            Ok((
                layer,
                [
                    key.ok_or_else(|| malformed(format!("layer {layer} has no key history")))?,
                    value
                        .ok_or_else(|| malformed(format!("layer {layer} has no value history")))?,
                ],
            ))
        })
        .collect::<Result<_, InferenceApiError>>()?;
    let position =
        position.ok_or_else(|| malformed("history input needs an explicit position".into()))?;
    let kv_cache =
        runtime
            .kv_cache(cache)
            .map_err(|error| InferenceApiError::KvCacheUnavailable {
                reason: format!("history input needs cache '{cache}': {error}"),
            })?;
    if !matches!(
        kv_cache.lifecycle,
        KvCacheLifecycleState::Empty
            | KvCacheLifecycleState::Prefilling
            | KvCacheLifecycleState::Ready
            | KvCacheLifecycleState::Active
    ) {
        return Err(InferenceApiError::KvCacheUnavailable {
            reason: format!("cache '{cache}' cannot accept a history update"),
        });
    }
    if position == 0 {
        let recurrent = graph
            .recurrent_state_bindings()
            .map_err(|error| malformed(format!("invalid recurrent state bindings: {error}")))?;
        if recurrent.iter().any(|binding| {
            kv_cache
                .recurrent_resources
                .contains_key(&binding.layout.resource_id)
                || kv_cache
                    .pending_recurrent_resources
                    .contains_key(&binding.layout.resource_id)
        }) {
            return Err(InferenceApiError::KvCacheUnavailable {
                reason: "a new history requires reset recurrent state".into(),
            });
        }
    }
    for (id, shape, layer, index) in inputs {
        if initial.contains_key(id) || resident.contains_key(id) || shape[0] != position.max(1) {
            return Err(malformed(format!(
                "invalid caller binding or history length for '{id}'"
            )));
        }
        if position == 0 {
            if history.is_some() || kv_cache.layer_resources.contains_key(&(layer as u32)) {
                return Err(InferenceApiError::KvCacheUnavailable {
                    reason: format!("a new history for layer {layer} requires a fresh cache"),
                });
            }
            let elements = shape
                .iter()
                .try_fold(1u64, |n, &dimension| {
                    n.checked_mul(dimension)
                        .filter(|&n| dimension > 0 && n <= 67_108_864)
                })
                .ok_or_else(|| malformed(format!("invalid history shape for '{id}'")))?;
            initial.insert(
                id.clone(),
                HostTensor::new(shape.clone(), vec![0.0; elements as usize])
                    .map_err(runtime_generation_failed)?,
            );
        } else {
            let stored = history
                .and_then(|history| history.get(&layer))
                .ok_or_else(|| InferenceApiError::KvCacheUnavailable {
                    reason: format!("missing history for layer {layer}"),
                })?;
            if stored.rows.is_some()
                || stored.history_schema.as_ref() != schemas.get(&layer)
                || stored.history_owner.as_ref() != Some(cache)
            {
                return Err(InferenceApiError::KvCacheUnavailable {
                    reason: format!("incompatible history for layer {layer}"),
                });
            }
            let resource = if index == 0 { &stored.k } else { &stored.v };
            let actual_shape = executor.resident_shape(resource);
            if actual_shape.as_ref() != Some(shape) {
                return Err(InferenceApiError::KvCacheUnavailable {
                    reason: format!(
                        "history resource for '{id}' has shape {actual_shape:?}, expected {shape:?}"
                    ),
                });
            }
            resident.insert(id.clone(), (resource.clone(), shape.clone()));
        }
    }
    Ok(schemas)
}

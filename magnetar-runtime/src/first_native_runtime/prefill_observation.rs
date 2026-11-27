//! Scoped observation of a trusted decoder Component's prefill graph.

use super::*;

/// This first implementation retains intermediate resources for the duration
/// of a call. Bound its work until scoped executions can release them eagerly.
const MAX_OBSERVED_PREFILL_TOKENS: usize = 512;
static OBSERVATION_GENERATION: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
mod tests;

impl ProductionLoadedModel {
    /// Execute an independent, uncached prefill and return one Component graph
    /// edge shaped `[tokens, hidden_size]`. The caller selects an edge whose
    /// meaning is documented by the trusted Component; Runtime does not infer
    /// model-specific semantics from its name. The token sequence starts at
    /// position zero, and all temporary KV, recurrent and graph resources are
    /// discarded after the call. This does not return a reusable prefix cache.
    ///
    /// Inputs are bounded to 512 tokens in this first profile. A Component
    /// without the requested edge, an unsupported Provider, or a tensor-parallel
    /// model is rejected. The method does not apply a task-specific decision head.
    pub fn observe_prefill_edge(
        &mut self,
        token_ids: &[u32],
        edge_name: &str,
    ) -> Result<HostTensor, InferenceApiError> {
        let invalid = |reason: &str| InferenceApiError::GraphPlanningFailed {
            reason: reason.into(),
        };
        let architecture = &self.fixture.config.architecture;
        if token_ids.is_empty()
            || token_ids.len() > MAX_OBSERVED_PREFILL_TOKENS
            || token_ids.len() as u64 > architecture.context_length
            || token_ids
                .iter()
                .any(|&id| u64::from(id) >= architecture.vocabulary_size || id >= (1 << 24))
        {
            return Err(invalid("observed prefill has invalid token IDs or length"));
        }
        if edge_name.is_empty() || edge_name.len() > 256 {
            return Err(invalid("observed edge name is invalid"));
        }
        if self.graph_cache.tensor_parallel() {
            return Err(InferenceApiError::Unsupported {
                reason: "prefill observation does not yet support tensor parallelism".into(),
            });
        }
        let tokens = token_ids.len() as u64;
        let (graphs, _, _) = match &self.component_digest {
            Some(digest) => build_first_native_graphs_from_named_component_for_artifact(
                digest,
                &architecture_config_from_first_native_model_config(&self.fixture.config),
                &self.fixture.manifest.tensors,
                &self.fixture.identity,
                tokens,
            ),
            None => build_first_native_graphs_from_real_qwen_component(&self.fixture, tokens),
        }
        .map_err(|error| invalid(&format!("could not build observed prefill graph: {error}")))?;
        let graph = &graphs.prefill;
        let recurrent = super::recurrent_state::recurrent_bindings(graph)?;
        let edge_id = TensorEdgeId::new(edge_name);
        let shape = [tokens, architecture.hidden_size];
        if !graph
            .edges
            .get(&edge_id)
            .is_some_and(|edge| edge.descriptor.shape.dimensions == shape)
            || !graph
                .nodes
                .values()
                .any(|node| node.outputs.contains(&edge_id))
        {
            return Err(invalid(
                "Component does not produce the requested token representation edge",
            ));
        }
        let revision = require_ready_first_native_instance(&self.runtime, &self.instance)?
            .status()
            .mutation_version;
        let mut plan = prepare_first_native_plan_for_graph(
            &self.runtime,
            graph,
            &self.instance,
            revision,
            tokens,
            PreparedExecutionPlanGeneration::new(1),
            &self.provider_binding,
        )
        .map_err(|error| invalid(&format!("could not plan observed prefill: {error}")))?;
        let executor = resolve_kernel_execution_provider(&self.runtime, &self.provider_binding)?;
        let unique = OBSERVATION_GENERATION.fetch_add(1, Ordering::Relaxed);
        let namespace = format!("prefill-observation-{}-{unique}", std::process::id());
        let cache = self
            .runtime
            .create_kv_cache(KvCache::new(
                KvCacheId::new(&namespace).map_err(InferenceApiError::from)?,
                KvCacheScope::Operation,
                KvCacheCompatibility::new(
                    GenerationModelReference::ModelInstance(self.instance.clone()),
                    self.fixture.tokenizer.metadata().id.clone(),
                ),
                KvCacheLayoutMetadata::contiguous(
                    self.fixture.config.kv_layer_count() as u32,
                    architecture.kv_head_count as u32,
                    architecture.head_dimension as u32,
                    token_ids.len() as u32,
                    ComputeDType::Float32,
                )
                .with_layer_row_widths(self.fixture.config.kv_row_widths())
                .with_layer_value_row_widths(self.fixture.config.kv_value_row_widths()),
            ))
            .map_err(InferenceApiError::from)?;
        let input_resource = TensorResourceId::new(format!("{namespace}.input.token_ids"));
        let mut events = Vec::new();
        let result = (|| {
            let ids = HostTensor::new(
                [tokens],
                token_ids.iter().map(|&id| id as f32).collect::<Vec<_>>(),
            )
            .map_err(|error| invalid(&error.to_string()))?;
            executor
                .write_tensor_value_admitted(
                    self.runtime.memory_mut(),
                    input_resource.clone(),
                    TensorValue::Host(ids),
                    MemoryAllocationClass::Tensor,
                    MemoryAllocationOwner::Session(namespace.clone()),
                )
                .map_err(|error| InferenceApiError::ProviderTensorWriteFailed {
                    reason: format!("could not upload observed prefill input: {error:?}"),
                })?;
            let (_, mut outputs, _, _) = execute_first_native_graph_with_resident_input(
                &mut self.runtime,
                &self.fixture,
                &self.instance,
                &cache,
                graph,
                &mut plan,
                BTreeMap::new(),
                BTreeMap::from([(
                    TensorEdgeId::new("input.token_ids"),
                    (input_resource.clone(), vec![tokens]),
                )]),
                None,
                Some(0),
                &mut events,
                MaterializedEdges::Only(&[edge_name]),
                None,
                None,
                Some(&namespace),
                false,
                Some(&namespace),
            )?;
            outputs
                .remove(&edge_id)
                .ok_or_else(|| invalid("Component did not materialize the requested edge"))
        })();
        // The observed pass never commits a generation step. Release pending
        // KV and recurrent state and every namespace-owned temporary, including
        // partially produced outputs when an operator fails.
        let mut cleanup_error = super::recurrent_state::release_observed_recurrent_state(
            &mut self.runtime,
            &executor,
            &cache,
        )
        .err();
        let mut resources: BTreeSet<_> = graph
            .nodes
            .values()
            .flat_map(|node| node.outputs.iter())
            .map(|edge| TensorResourceId::new(format!("{namespace}.edge.{}", edge.as_str())))
            .collect();
        resources.insert(input_resource);
        for binding in recurrent {
            resources.insert(TensorResourceId::new(format!(
                "{namespace}.input.{}",
                binding.input_edge.as_str()
            )));
        }
        // RoPE on a KV edge may first dispatch to a synthesized node output
        // before the executor copies it into the pending KV resource.
        for node in graph.nodes.values() {
            resources.insert(TensorResourceId::new(format!(
                "{namespace}.{}.out",
                node.id
            )));
        }
        for layer in 0..self.fixture.config.kv_layer_count() {
            for role in ["k", "v"] {
                resources.insert(TensorResourceId::new(format!(
                    "{namespace}.kv.{cache}.layer{layer}.{role}.pending"
                )));
            }
        }
        for resource in resources {
            let released = executor
                .release_admitted_tensor(self.runtime.memory_mut(), &resource)
                .map_err(|error| InferenceApiError::ProviderTensorReleaseFailed {
                    reason: format!("could not release observed resource '{resource}': {error}"),
                });
            match released {
                Ok(_) => {
                    if let Some(allocation) = self
                        .runtime
                        .memory_mut()
                        .remove_tensor_residency(&resource)
                        .and_then(|residency| residency.allocation)
                        && self.runtime.memory().is_live_allocation(allocation)
                        && let Err(error) = self.runtime.memory_mut().release(allocation)
                    {
                        cleanup_error.get_or_insert_with(|| {
                            InferenceApiError::ProviderTensorReleaseFailed {
                                reason: format!("could not release observed allocation: {error}"),
                            }
                        });
                    }
                }
                Err(error) => {
                    cleanup_error.get_or_insert(error);
                }
            }
        }
        let cleanup = self
            .runtime
            .release_kv_cache(&cache)
            .map_err(InferenceApiError::from)
            .and_then(|()| {
                self.runtime
                    .kv_caches_mut()
                    .forget_released(&cache)
                    .map(|_| ())
                    .map_err(InferenceApiError::from)
            });
        let cleanup = cleanup.and_then(|()| cleanup_error.map_or(Ok(()), Err));
        match (result, cleanup) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(error), _) | (_, Err(error)) => Err(error),
        }
    }
}

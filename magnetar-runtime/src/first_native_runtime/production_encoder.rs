//! Resident, Component-produced single-sequence encoder execution.

use super::*;

const MAX_ENCODER_TOKENS: usize = 512;
static ENCODER_EXECUTION: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
mod tests;

/// A trusted encoder artifact whose graph-facing weights are resident in a
/// Provider. The caller supplies an authenticated manifest and payload source,
/// and must derive `encoder` from the same checkpoint configuration.
///
/// This initial single-sequence profile accepts explicit token IDs and a
/// padding mask, not text or a task-specific decision head. Call [`Self::unload`]
/// when the resident model is no longer needed.
pub struct ProductionLoadedEncoder {
    runtime: Runtime,
    instance: ModelInstanceId,
    provider_binding: ProviderBinding,
    component: ComponentDigest,
    request: FirstNativeEncoderGraphRequest,
    profile: NativeGraphExecutionProfile,
}

impl ProductionLoadedEncoder {
    /// Load and validate the graph-facing encoder tensor inventory through the
    /// normal model-loading transaction. The Component must already have been
    /// registered with its own independent trust policy.
    pub fn load(
        manifest: ModelManifest,
        payloads: &dyn ProductionArtifactPayloadSource,
        trust: ModelTrustStore,
        provider: Arc<dyn Provider>,
        component: ComponentDigest,
        encoder: EncoderGraphConfig,
    ) -> Result<Self, InferenceApiError> {
        let invalid = |reason: String| InferenceApiError::ModelComponentUnavailable { reason };
        let registered =
            named_component_runtime(&component).map_err(|error| invalid(error.to_string()))?;
        if (!registered.supported_architecture_families.is_empty()
            && !registered
                .supported_architecture_families
                .contains(&manifest.architecture.family))
            || (!registered.supported_artifact_formats.is_empty()
                && !registered
                    .supported_artifact_formats
                    .contains(manifest.artifact_format.as_str()))
        {
            return Err(invalid(
                "encoder Component is incompatible with the model artifact".into(),
            ));
        }
        let architecture = manifest.architecture_config.clone().ok_or_else(|| {
            invalid("encoder model artifact has no architecture configuration".into())
        })?;
        let weight_shapes = manifest
            .tensors
            .iter()
            .map(|tensor| (tensor.name.clone(), tensor.shape.clone()))
            .collect();
        let request = FirstNativeEncoderGraphRequest {
            architecture: architecture.clone(),
            encoder,
            weight_shapes,
            identity: production_model_component_identity(),
            sequence_length: 1,
        };
        let graph = build_first_native_encoder_graph_from_named_component(&component, &request)
            .map_err(|error| invalid(format!("encoder graph rejected the artifact: {error}")))?;
        reject_encoder_state(&graph)?;
        let profile = NativeGraphExecutionProfile {
            hidden_size: architecture.hidden_size,
            attention: super::dispatch::AttentionGeometry {
                heads: u64::from(architecture.num_attention_heads),
                kv_heads: u64::from(architecture.num_key_value_heads),
                head_dimension: architecture.head_dim,
            },
            rope: FirstNativeRopeConfig {
                base: architecture.rope_theta,
                scale: architecture.rope_scaling_factor.map(f64::from),
                dimension: architecture.head_dim,
                position_mode: FirstNativeRopePositionMode::Sequential,
                dynamic_scaling_supported: false,
                frequency_scaling: architecture.rope_frequency_scaling,
            },
            kv_layer_count: 0,
            decoder_layer_count: 0,
            kv_head_dimensions: Vec::new(),
            has_latent_attention: false,
        };
        let provider_binding = ProviderBinding::new(provider.metadata().name.clone());
        let mut runtime = Runtime::builder()
            .register_provider(provider.clone())
            .trust_store(trust)
            .build()
            .map_err(|error| InferenceApiError::InferenceApiUnavailable {
                reason: error.to_string(),
            })?;
        register_prepared_kernels_for_provider(&mut runtime, &*provider).map_err(|error| {
            InferenceApiError::KernelUnavailable {
                reason: error.to_string(),
            }
        })?;
        let instance = load_production_model_instance_for_provider(
            &mut runtime,
            &manifest,
            payloads,
            &provider_binding,
        )?;
        require_ready_first_native_instance(&runtime, &instance)?;
        Ok(Self {
            runtime,
            instance,
            provider_binding,
            component,
            request,
            profile,
        })
    }

    /// Encode a nonempty sequence with an explicit validity mask. This first
    /// profile rejects more than 512 tokens and returns `[tokens, hidden_size]`.
    /// It allocates no reusable KV or recurrent state.
    pub fn encode(
        &mut self,
        token_ids: &[u32],
        valid: &[bool],
    ) -> Result<HostTensor, InferenceApiError> {
        let invalid = |reason: &str| InferenceApiError::GraphPlanningFailed {
            reason: reason.into(),
        };
        if token_ids.is_empty()
            || token_ids.len() > MAX_ENCODER_TOKENS
            || token_ids.len() as u64 > self.request.encoder.max_positions
            || token_ids.len() != valid.len()
            || !valid.iter().any(|&item| item)
            || token_ids
                .iter()
                .any(|&id| u64::from(id) >= self.request.architecture.vocab_size || id >= (1 << 24))
        {
            return Err(invalid("invalid encoder tokens or padding mask"));
        }
        let mut request = self.request.clone();
        request.sequence_length = token_ids.len() as u64;
        let graph =
            build_first_native_encoder_graph_from_named_component(&self.component, &request)
                .map_err(|error| invalid(&format!("could not build encoder graph: {error}")))?;
        reject_encoder_state(&graph)?;
        let revision = require_ready_first_native_instance(&self.runtime, &self.instance)?
            .status()
            .mutation_version;
        let mut plan = prepare_first_native_plan_for_graph(
            &self.runtime,
            &graph,
            &self.instance,
            revision,
            request.sequence_length,
            PreparedExecutionPlanGeneration::new(1),
            &self.provider_binding,
        )
        .map_err(|error| invalid(&format!("could not plan encoder graph: {error}")))?;
        let executor = resolve_kernel_execution_provider(&self.runtime, &self.provider_binding)?;
        let unique = ENCODER_EXECUTION.fetch_add(1, Ordering::Relaxed);
        let namespace = format!("encoder-execution-{}-{unique}", std::process::id());
        let cache = KvCacheId::new(&namespace).map_err(InferenceApiError::from)?;
        let ids_resource = TensorResourceId::new(format!("{namespace}.input.token_ids"));
        let mask_resource = TensorResourceId::new(format!("{namespace}.input.attention_mask"));
        let result = (|| {
            let inputs: [(TensorResourceId, Vec<f32>); 2] = [
                (
                    ids_resource.clone(),
                    token_ids.iter().map(|&id| id as f32).collect(),
                ),
                (
                    mask_resource.clone(),
                    valid
                        .iter()
                        .map(|&item| if item { 1.0 } else { 0.0 })
                        .collect(),
                ),
            ];
            for (resource, data) in inputs {
                let tensor = HostTensor::new([request.sequence_length], data)
                    .map_err(|error| invalid(&error.to_string()))?;
                executor
                    .write_tensor_value_admitted(
                        self.runtime.memory_mut(),
                        resource,
                        TensorValue::Host(tensor),
                        MemoryAllocationClass::Tensor,
                        MemoryAllocationOwner::Session(namespace.clone()),
                    )
                    .map_err(|error| InferenceApiError::ProviderTensorWriteFailed {
                        reason: format!("could not upload encoder input: {error:?}"),
                    })?;
            }
            let (_, mut outputs, _, _) = execute_first_native_graph_with_profile(
                &mut self.runtime,
                &self.profile,
                &self.instance,
                &cache,
                &graph,
                &mut plan,
                BTreeMap::new(),
                BTreeMap::from([
                    (
                        TensorEdgeId::new("input.token_ids"),
                        (ids_resource.clone(), vec![request.sequence_length]),
                    ),
                    (
                        TensorEdgeId::new("input.attention_mask"),
                        (mask_resource.clone(), vec![request.sequence_length]),
                    ),
                ]),
                None,
                Some(0),
                &mut Vec::new(),
                MaterializedEdges::Only(&["hidden_states"]),
                None,
                None,
                Some(&namespace),
                false,
                Some(&namespace),
            )?;
            outputs
                .remove(&TensorEdgeId::new("hidden_states"))
                .ok_or_else(|| invalid("encoder did not return hidden states"))
        })();
        let mut resources: BTreeSet<_> = graph
            .nodes
            .values()
            .flat_map(|node| node.outputs.iter())
            .map(|edge| TensorResourceId::new(format!("{namespace}.edge.{}", edge.as_str())))
            .collect();
        resources.insert(ids_resource);
        resources.insert(mask_resource);
        for node in graph.nodes.values() {
            resources.insert(TensorResourceId::new(format!(
                "{namespace}.{}.out",
                node.id
            )));
        }
        let mut cleanup_error = None;
        for resource in resources {
            if let Err(error) = super::prefill_observation::release_private_resource_checked(
                &mut self.runtime,
                &executor,
                &resource,
            ) {
                cleanup_error.get_or_insert(error);
            }
        }
        match (result, cleanup_error) {
            (Ok(value), None) => Ok(value),
            (Err(error), _) | (_, Some(error)) => Err(error),
        }
    }

    /// Release the resident model's Provider resources.
    pub fn unload(mut self) -> Result<(), InferenceApiError> {
        unload_model_instance(
            &mut self.runtime,
            &self.instance,
            ModelInstanceUnloadPolicy::DrainActiveUse,
        )
        .map(|_| ())
    }
}

fn reject_encoder_state(graph: &ExecutionGraph) -> Result<(), InferenceApiError> {
    if graph.edges.values().any(|edge| {
        edge.kv_cache.is_some() || edge.recurrent_state.is_some() || edge.prefix_cache.is_some()
    }) {
        return Err(InferenceApiError::Unsupported {
            reason: "encoder execution does not accept decoder cache or recurrent state".into(),
        });
    }
    Ok(())
}

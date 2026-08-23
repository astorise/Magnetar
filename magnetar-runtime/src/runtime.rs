use crate::compute::{
    effective_compute_advertisement, ensure_non_empty_id, insert_unique,
    resolve_compute_value_descriptor, validate_compute_operation_schema,
};
use crate::planning::{
    classify_execution_plan, execution_phase_from_step_kind, execution_plan_id,
    execution_step_kind_from_memory_decision, graph_output_uses, last_use_for_input,
    last_use_for_node_output, memory_bytes, memory_error_from_compute_validation,
    planning_error_from_affinity, planning_error_from_validation, provider_memory_limit,
};
use crate::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
static NEXT_EXECUTION_CONTEXT_ID: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1);
static NEXT_AFFINITY_GROUP_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
static NEXT_SCHEDULED_OPERATION_ID: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1);
static NEXT_RUNTIME_OBSERVATION_SEQUENCE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1);

pub(crate) fn next_execution_context_id() -> ExecutionContextId {
    ExecutionContextId::new(
        NEXT_EXECUTION_CONTEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
    )
}
pub(crate) fn next_affinity_group_id() -> AffinityGroupId {
    AffinityGroupId::new(NEXT_AFFINITY_GROUP_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}
pub(crate) fn next_scheduled_operation_id() -> ScheduledOperationId {
    ScheduledOperationId::new(
        NEXT_SCHEDULED_OPERATION_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
    )
}
pub(crate) fn next_runtime_observation_sequence() -> u64 {
    NEXT_RUNTIME_OBSERVATION_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RuntimeConfig {
    pub resolution_policy: BuiltInResolutionPolicy,
    pub memory: MemoryManagerConfig,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionContext {
    id: ExecutionContextId,
    config: RuntimeConfig,
}
impl Default for ExecutionContext {
    fn default() -> Self {
        Self {
            id: next_execution_context_id(),
            config: RuntimeConfig::default(),
        }
    }
}
impl ExecutionContext {
    pub const fn id(&self) -> ExecutionContextId {
        self.id
    }
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }
}
#[derive(Default)]
pub struct RuntimeBuilder {
    config: RuntimeConfig,
    providers: Vec<Arc<dyn Provider>>,
}
impl RuntimeBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn config(mut self, x: RuntimeConfig) -> Self {
        self.config = x;
        self
    }
    pub fn register_provider(mut self, x: Arc<dyn Provider>) -> Self {
        self.providers.push(x);
        self
    }
    pub fn build(self) -> Result<Runtime, ProviderError> {
        let mut providers = ProviderLoader::new();
        for x in self.providers {
            providers.register_provider_isolated(x);
        }
        let runtime = Runtime {
            context: ExecutionContext {
                id: next_execution_context_id(),
                config: self.config.clone(),
            },
            memory: MemoryManager::new(self.config.memory),
            providers,
            initialized: true,
        };
        Ok(runtime)
    }
}
pub struct Runtime {
    context: ExecutionContext,
    memory: MemoryManager,
    providers: ProviderLoader,
    initialized: bool,
}
impl Runtime {
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder::new()
    }
    pub fn initialize(config: RuntimeConfig) -> Self {
        Self::builder()
            .config(config)
            .build()
            .expect("valid configuration")
    }
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
    pub fn context(&self) -> &ExecutionContext {
        &self.context
    }
    pub fn memory(&self) -> &MemoryManager {
        &self.memory
    }
    pub fn memory_mut(&mut self) -> &mut MemoryManager {
        &mut self.memory
    }
    pub fn providers(&self) -> &ProviderLoader {
        &self.providers
    }
    pub fn register_provider(&mut self, x: Arc<dyn Provider>) -> Result<(), ProviderError> {
        self.providers.register_provider(x)
    }
    /// Returns every registered execution target in deterministic ID order.
    pub fn devices(&self) -> impl Iterator<Item = &dyn Device> {
        self.providers.registry().devices()
    }
    pub fn device(&self, id: &DeviceId) -> Option<&dyn Device> {
        self.providers.registry().device(id)
    }
    /// Resolves all compatible providers, ordered for deterministic fallback.
    pub fn resolve_providers(&self, capability: &Capability) -> Vec<&dyn Provider> {
        self.try_resolve_providers(capability).unwrap_or_default()
    }
    /// Resolves providers while reporting invalid capability dependencies.
    pub fn try_resolve_providers(
        &self,
        capability: &Capability,
    ) -> Result<Vec<&dyn Provider>, ProviderError> {
        self.providers
            .try_resolve_providers_with_policy(capability, self.context.config.resolution_policy)
    }
    /// Resolves one coherent Provider for resources that already carry state.
    ///
    /// A group is created only when `dependencies` is non-empty and no input
    /// group exists. Independent resources therefore remain shareable until a
    /// dependent operation creates a grouped resource from them.
    pub fn resolve_with_affinity<'a>(
        &'a self,
        capability: &Capability,
        dependencies: &[&ResourceAffinity],
        fallback: FallbackClass,
    ) -> Result<AffinityResolution<'a>, AffinityError> {
        if !self.initialized {
            return Err(AffinityError::RuntimeNotInitialized);
        }
        let mut constraints =
            AffinityConstraints::try_from_affinities(dependencies.iter().copied())?;
        constraints.require_fallback(fallback);
        constraints.merge(
            &ResourceAffinity::new(FallbackClass::Transparent)
                .with_execution_context(self.context.id),
        )?;
        if !dependencies.is_empty() && constraints.affinity().group().is_none() {
            constraints.merge(
                &ResourceAffinity::new(FallbackClass::Transparent)
                    .with_group(next_affinity_group_id()),
            )?;
        }

        let (provider, selected, decision) = self.providers.resolve_with_constraints(
            capability,
            &constraints,
            self.context.config.resolution_policy,
            ExecutionPhase::BeforeResourceCreation,
            true,
        )?;
        let provider_binding = ProviderBinding::new(provider.metadata().name);
        let affinity = constraints
            .into_affinity()
            .with_provider(provider_binding)
            .with_capability(CapabilityBinding::new(
                selected.id.clone(),
                selected.version,
            ));
        Ok(AffinityResolution {
            provider,
            capability: selected,
            affinity,
            decision,
        })
    }
    pub fn resolve_with_affinity_at_phase<'a>(
        &'a self,
        capability: &Capability,
        dependencies: &[&ResourceAffinity],
        fallback: FallbackClass,
        execution_phase: ExecutionPhase,
        replayable_input: bool,
    ) -> Result<AffinityResolution<'a>, AffinityError> {
        if !self.initialized {
            return Err(AffinityError::RuntimeNotInitialized);
        }
        let mut constraints =
            AffinityConstraints::try_from_affinities(dependencies.iter().copied())?;
        constraints.require_fallback(fallback);
        constraints.merge(
            &ResourceAffinity::new(FallbackClass::Transparent)
                .with_execution_context(self.context.id),
        )?;
        if !dependencies.is_empty() && constraints.affinity().group().is_none() {
            constraints.merge(
                &ResourceAffinity::new(FallbackClass::Transparent)
                    .with_group(next_affinity_group_id()),
            )?;
        }

        let (provider, selected, decision) = self.providers.resolve_with_constraints(
            capability,
            &constraints,
            self.context.config.resolution_policy,
            execution_phase,
            replayable_input,
        )?;
        let provider_binding = ProviderBinding::new(provider.metadata().name);
        let affinity = constraints
            .into_affinity()
            .with_provider(provider_binding)
            .with_capability(CapabilityBinding::new(
                selected.id.clone(),
                selected.version,
            ));
        Ok(AffinityResolution {
            provider,
            capability: selected,
            affinity,
            decision,
        })
    }
    /// Resolves providers for a component's WIT import without exposing a
    /// provider dependency to the component itself.
    pub fn resolve_component_import(
        &self,
        interface: &WitInterface,
    ) -> Result<Vec<&dyn Provider>, ProviderError> {
        self.providers.try_resolve_providers_with_policy(
            &Capability::from_wit(interface.clone())?,
            self.context.config.resolution_policy,
        )
    }
    pub fn validate_compute_operation_requests(
        &self,
        provider: &str,
        operations: &[ComputeOperationRequest],
    ) -> Result<Vec<ComputeOperationDescriptor>, ComputeValidationError> {
        let descriptors = operations
            .iter()
            .map(|operation| {
                let family =
                    ComputeOperationFamily::from_id(&operation.family_id).ok_or_else(|| {
                        ComputeValidationError::UnknownOperationFamily(operation.family_id.clone())
                    })?;
                Ok(ComputeOperationDescriptor {
                    schema_id: None,
                    family,
                    dtype: operation.dtype,
                    layout: operation.layout,
                    precision: operation.precision,
                    attributes: BTreeMap::new(),
                    tensors: operation.tensors.clone(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.validate_compute_operations(provider, &descriptors)?;
        Ok(descriptors)
    }
    pub fn validate_compute_operations(
        &self,
        provider: &str,
        operations: &[ComputeOperationDescriptor],
    ) -> Result<(), ComputeValidationError> {
        let metadata = self
            .providers
            .provider(provider)
            .map(Provider::metadata)
            .ok_or_else(|| {
                ComputeValidationError::ProviderUnavailable(ProviderBinding::new(provider))
            })?;
        let advertisement = effective_compute_advertisement(&metadata);
        if !advertisement.supports_capability_version(COMPUTE_CAPABILITY_VERSION) {
            return Err(ComputeValidationError::UnsupportedAdvertisement {
                provider: ProviderBinding::new(&metadata.name),
                reason: format!(
                    "provider does not advertise compatible '{}' version {}",
                    COMPUTE_CAPABILITY_ID, COMPUTE_CAPABILITY_VERSION
                ),
            });
        }
        for operation in operations {
            if let Some(schema_id) = &operation.schema_id
                && advertisement
                    .unsupported_operation_schemas
                    .contains(schema_id)
            {
                return Err(ComputeValidationError::UnsupportedOperationSchema {
                    provider: ProviderBinding::new(&metadata.name),
                    operation: schema_id.clone(),
                });
            }
            let schema_result = validate_compute_operation_schema(operation)?;
            let support = if let Some(schema_id) = &operation.schema_id {
                advertisement
                    .operation_schemas
                    .get(schema_id)
                    .map(OperationSchemaSupport::operation_support)
                    .or_else(|| {
                        advertisement
                            .operation_families
                            .get(&operation.family)
                            .map(OperationFamilySupport::operation_support)
                    })
                    .ok_or_else(|| ComputeValidationError::UnsupportedOperationFamily {
                        provider: ProviderBinding::new(&metadata.name),
                        family: operation.family,
                    })?
            } else {
                advertisement
                    .operation_families
                    .get(&operation.family)
                    .map(OperationFamilySupport::operation_support)
                    .ok_or_else(|| ComputeValidationError::UnsupportedOperationFamily {
                        provider: ProviderBinding::new(&metadata.name),
                        family: operation.family,
                    })?
            };
            support.supports(operation)?;
            drop(schema_result);
        }
        Ok(())
    }
    pub fn validate_compute_tensor_resources(
        &self,
        provider: &str,
        resources: &[TensorResourceDescriptor],
    ) -> Result<(), ComputeValidationError> {
        let target = ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(ProviderBinding::new(provider))
            .with_capability(CapabilityBinding::new(
                CapabilityId::new(COMPUTE_CAPABILITY_ID),
                COMPUTE_CAPABILITY_VERSION,
            ));
        for resource in resources {
            target
                .validate_with(&resource.affinity)
                .map_err(ComputeValidationError::IncompatibleResourceAffinity)?;
        }
        Ok(())
    }
    pub fn validate_compute_data_movement(
        &self,
        provider: &str,
        movements: &[ComputeDataMovementDescriptor],
    ) -> Result<(), ComputeValidationError> {
        let metadata = self
            .providers
            .provider(provider)
            .map(Provider::metadata)
            .ok_or_else(|| {
                ComputeValidationError::ProviderUnavailable(ProviderBinding::new(provider))
            })?;
        let provider_binding = ProviderBinding::new(&metadata.name);
        let advertisement = effective_compute_advertisement(&metadata);
        let target = ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(provider_binding.clone())
            .with_capability(CapabilityBinding::new(
                CapabilityId::new(COMPUTE_CAPABILITY_ID),
                COMPUTE_CAPABILITY_VERSION,
            ));
        for movement in movements {
            let support = advertisement
                .data_movement
                .get(&movement.kind)
                .map(DataMovementSupport::movement_support)
                .ok_or_else(|| ComputeValidationError::UnsupportedDataMovement {
                    provider: provider_binding.clone(),
                    kind: movement.kind,
                })?;
            support.supports(&provider_binding, movement)?;
            if let Some(source) = movement.source.tensor() {
                let permits_explicit_replacement = matches!(
                    movement.placement,
                    ComputePlacementIntent::RuntimeSelected
                        | ComputePlacementIntent::HostAccessible
                ) && matches!(
                    movement.kind,
                    ComputeDataMovementKind::Transfer
                        | ComputeDataMovementKind::PlacementConversion
                        | ComputeDataMovementKind::Download
                );
                if !permits_explicit_replacement
                    || source
                        .affinity
                        .provider()
                        .is_none_or(|source_provider| source_provider.as_str() == provider)
                {
                    target
                        .validate_with(&source.affinity)
                        .map_err(ComputeValidationError::IncompatibleResourceAffinity)?;
                }
            }
        }
        Ok(())
    }
    pub fn memory_admission_for_plan(
        &self,
        plan: &MemoryPlan,
        queue_allowed: bool,
    ) -> MemoryAdmissionDecision {
        let request = MemoryAllocationRequest::new(
            MemoryAllocationClass::TemporaryWorkspace,
            plan.pressure.estimated_peak_bytes,
            MemoryPlacement::ProviderOwnedOpaque(plan.provider.clone()),
            MemoryAllocationOwner::Provider(plan.provider.clone()),
        )
        .with_affinity(plan.output_affinity.clone());
        self.memory.admit(MemoryAdmissionRequest {
            allocation: request,
            pressure: self.memory.pressure_snapshot(),
            queue_allowed,
        })
    }
    pub fn plan_compute_data_movement_memory(
        &self,
        provider: &str,
        movements: &[ComputeDataMovementDescriptor],
    ) -> Result<MemoryPlan, MemoryPlanningError> {
        let metadata = self
            .providers
            .provider(provider)
            .map(Provider::metadata)
            .ok_or_else(|| MemoryPlanningError::MemoryPlanningFailed {
                reason: format!("provider '{provider}' is unavailable"),
                report: MemoryPressureReport::default(),
            })?;
        let provider_binding = ProviderBinding::new(&metadata.name);
        let mut plan = MemoryPlan::new(provider_binding.clone(), None, self.context.id);
        let advertisement = effective_compute_advertisement(&metadata);

        for (index, movement) in movements.iter().enumerate() {
            let support = advertisement
                .data_movement
                .get(&movement.kind)
                .ok_or_else(|| MemoryPlanningError::TransferRequired {
                    reason: format!("provider does not advertise '{}'", movement.kind.id()),
                    report: plan.pressure.clone(),
                })?;
            let host_staging_permitted = movement.host_staging == HostStagingPolicy::Permit;
            if host_staging_permitted && !support.allow_host_staging {
                return Err(MemoryPlanningError::TransferRequired {
                    reason: "host staging must be permitted by the component and advertised".into(),
                    report: plan.pressure.clone(),
                });
            }
            let output_bytes = memory_bytes(&movement.output, &plan.pressure)?;
            let staging = self
                .memory
                .staging_feasibility(movement.host_staging, output_bytes);
            if host_staging_permitted && !staging.feasible {
                return Err(MemoryPlanningError::TransferRequired {
                    reason: staging.reason,
                    report: plan.pressure.clone(),
                });
            }
            let mut affinity = ResourceAffinity::new(FallbackClass::ProviderPinned)
                .with_provider(provider_binding.clone())
                .with_capability(CapabilityBinding::new(
                    CapabilityId::new(COMPUTE_CAPABILITY_ID),
                    COMPUTE_CAPABILITY_VERSION,
                ))
                .with_execution_context(self.context.id);
            if movement.placement == ComputePlacementIntent::PreserveSourceAffinity
                && let Some(source) = movement.source.tensor()
                && let Some(device) = source.affinity.device()
            {
                affinity = affinity.with_device(device.clone());
            }
            let requirement_id = format!("movement:{index}:{}", movement.kind.id());
            let region = match movement.kind {
                ComputeDataMovementKind::Materialize => MemoryRegionKind::Materialization,
                ComputeDataMovementKind::Transfer => MemoryRegionKind::Transfer,
                ComputeDataMovementKind::Upload
                | ComputeDataMovementKind::Download
                | ComputeDataMovementKind::Copy
                | ComputeDataMovementKind::DTypeConversion
                | ComputeDataMovementKind::PlacementConversion => MemoryRegionKind::Transfer,
            };
            plan.add_requirement(MemoryRequirement::new(
                requirement_id.clone(),
                region,
                output_bytes,
                affinity.clone(),
            ))?;
            plan.decisions.push(match movement.kind {
                ComputeDataMovementKind::Materialize => {
                    plan.pressure.materialization_cost_bytes = plan
                        .pressure
                        .materialization_cost_bytes
                        .checked_add(output_bytes)
                        .ok_or_else(|| MemoryPlanningError::SizeOverflow {
                            reason: "materialization memory cost overflows u64".into(),
                            report: plan.pressure.clone(),
                        })?;
                    MemoryPlanningDecision::RequireMaterialization {
                        requirement: requirement_id.clone(),
                    }
                }
                _ if host_staging_permitted => {
                    plan.pressure.transfer_buffer_cost_bytes = plan
                        .pressure
                        .transfer_buffer_cost_bytes
                        .checked_add(output_bytes)
                        .ok_or_else(|| MemoryPlanningError::SizeOverflow {
                            reason: "transfer memory cost overflows u64".into(),
                            report: plan.pressure.clone(),
                        })?;
                    MemoryPlanningDecision::AccountHostStaging {
                        requirement: requirement_id.clone(),
                    }
                }
                _ => MemoryPlanningDecision::RequireTransfer {
                    requirement: requirement_id.clone(),
                },
            });
            plan.tensor_lifetimes.push(TensorLifetime {
                id: requirement_id,
                first_step: index,
                last_step: index,
                byte_size: output_bytes,
                affinity,
            });
        }
        self.validate_memory_plan(&metadata, &mut plan)?;
        Ok(plan)
    }
    pub fn plan_compute_graph_memory(
        &self,
        provider: &str,
        graph: &ComputeGraph,
    ) -> Result<MemoryPlan, MemoryPlanningError> {
        let metadata = self
            .providers
            .provider(provider)
            .map(Provider::metadata)
            .ok_or_else(|| MemoryPlanningError::MemoryPlanningFailed {
                reason: format!("provider '{provider}' is unavailable"),
                report: MemoryPressureReport::default(),
            })?;
        let provider_binding = ProviderBinding::new(&metadata.name);
        let mut plan = MemoryPlan::new(
            provider_binding.clone(),
            Some(graph.id.clone()),
            self.context.id,
        );
        let mut input_descriptors = BTreeMap::new();
        let mut input_affinities = BTreeMap::new();
        let mut output_descriptors = BTreeMap::new();
        let mut completed_nodes = BTreeSet::new();
        let graph_end = graph.nodes.len() + 1;
        let target = ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(provider_binding.clone())
            .with_capability(CapabilityBinding::new(
                CapabilityId::new(COMPUTE_CAPABILITY_ID),
                COMPUTE_CAPABILITY_VERSION,
            ))
            .with_execution_context(self.context.id);

        for input in &graph.inputs {
            input_descriptors.insert(input.id.clone(), input.value.descriptor().clone());
            let affinity = input
                .value
                .affinity()
                .cloned()
                .unwrap_or_else(|| target.clone());
            if let ComputeInputValue::TensorResource(resource) = &input.value {
                if let Some(source_provider) = resource.affinity.provider()
                    && source_provider.as_str() != provider
                {
                    return Err(MemoryPlanningError::TransferRequired {
                        reason: format!(
                            "input resource '{}' is bound to provider '{source_provider}'",
                            resource.id
                        ),
                        report: plan.pressure.clone(),
                    });
                }
                if resource.affinity.fallback() == FallbackClass::ProviderPinned {
                    plan.decisions
                        .push(MemoryPlanningDecision::PreservePinnedResource {
                            resource: resource.id.clone(),
                        });
                }
            }
            let bytes = memory_bytes(input.value.descriptor(), &plan.pressure)?;
            let id = format!("input:{}", input.id);
            plan.add_requirement(MemoryRequirement::new(
                id.clone(),
                MemoryRegionKind::GraphInput,
                bytes,
                affinity.clone(),
            ))?;
            plan.tensor_lifetimes.push(TensorLifetime {
                id,
                first_step: 0,
                last_step: last_use_for_input(graph, &input.id).unwrap_or(graph_end),
                byte_size: bytes,
                affinity: affinity.clone(),
            });
            input_affinities.insert(input.id.clone(), affinity);
        }

        for (node_index, node) in graph.nodes.iter().enumerate() {
            let step = node_index + 1;
            for input in &node.inputs {
                let descriptor = resolve_compute_value_descriptor(
                    Some(&node.id),
                    input,
                    &input_descriptors,
                    &output_descriptors,
                    &completed_nodes,
                )
                .map_err(memory_error_from_compute_validation)?;
                if descriptor.view.is_some() && descriptor.layout.kind() != ComputeLayout::Dense {
                    let bytes = memory_bytes(descriptor, &plan.pressure)?;
                    let requirement_id = format!("materialize:{}:{step}", node.id);
                    plan.add_requirement(MemoryRequirement::new(
                        requirement_id.clone(),
                        MemoryRegionKind::Materialization,
                        bytes,
                        target.clone(),
                    ))?;
                    plan.pressure.materialization_cost_bytes = plan
                        .pressure
                        .materialization_cost_bytes
                        .checked_add(bytes)
                        .ok_or_else(|| MemoryPlanningError::SizeOverflow {
                            reason: "materialization memory cost overflows u64".into(),
                            report: plan.pressure.clone(),
                        })?;
                    plan.decisions
                        .push(MemoryPlanningDecision::RequireMaterialization {
                            requirement: requirement_id,
                        });
                }
            }
            for output in &node.outputs {
                let bytes = memory_bytes(&output.descriptor, &plan.pressure)?;
                let id = format!("node:{}:{}", node.id, output.id);
                let lifetime = TensorLifetime {
                    id: id.clone(),
                    first_step: step,
                    last_step: last_use_for_node_output(graph, &node.id, &output.id)
                        .unwrap_or(step),
                    byte_size: bytes,
                    affinity: target.clone(),
                };
                let reuses = plan.find_reusable_buffer(&lifetime);
                let mut requirement = MemoryRequirement::new(
                    id.clone(),
                    if graph_output_uses(graph, &node.id, &output.id) {
                        MemoryRegionKind::GraphOutput
                    } else {
                        MemoryRegionKind::Intermediate
                    },
                    bytes,
                    target.clone(),
                );
                if !graph_output_uses(graph, &node.id, &output.id) {
                    requirement = requirement.reusable();
                }
                plan.add_requirement(requirement)?;
                plan.buffer_lifetimes.push(BufferLifetime {
                    id: format!("buffer:{id}"),
                    source: id.clone(),
                    first_step: lifetime.first_step,
                    last_step: lifetime.last_step,
                    byte_size: bytes,
                    affinity: target.clone(),
                    reuses: reuses.clone(),
                });
                plan.decisions.push(match reuses {
                    Some(buffer) => MemoryPlanningDecision::Reuse {
                        requirement: id.clone(),
                        buffer,
                    },
                    None => MemoryPlanningDecision::Allocate {
                        requirement: id.clone(),
                    },
                });
                output_descriptors.insert(
                    (node.id.clone(), output.id.clone()),
                    output.descriptor.clone(),
                );
                plan.tensor_lifetimes.push(lifetime);
            }
            completed_nodes.insert(node.id.clone());
        }

        for output in &graph.outputs {
            match &output.source {
                ComputeValueRef::Input(input) => {
                    if let Some(affinity) = input_affinities.get(input) {
                        plan.output_affinity = affinity.clone();
                    }
                }
                ComputeValueRef::NodeOutput { .. } => {
                    plan.output_affinity = target.clone();
                }
            }
        }
        self.validate_memory_plan(&metadata, &mut plan)?;
        Ok(plan)
    }
    pub fn validate_memory_plan(
        &self,
        metadata: &ProviderMetadata,
        plan: &mut MemoryPlan,
    ) -> Result<(), MemoryPlanningError> {
        let provider_limit = provider_memory_limit(metadata);
        if provider_limit != u64::MAX {
            plan.diagnostics
                .push(MemoryPlanningDiagnostic::ProviderLimit {
                    provider: plan.provider.clone(),
                    max_bytes: provider_limit,
                });
        }
        for requirement in &plan.requirements {
            if requirement.byte_size > provider_limit {
                return Err(MemoryPlanningError::ProviderMemoryLimitExceeded {
                    provider: plan.provider.clone(),
                    required: requirement.byte_size,
                    limit: provider_limit,
                    report: plan.pressure.clone(),
                });
            }
        }
        let selected_device = plan
            .requirements
            .iter()
            .find_map(|requirement| requirement.affinity.device().cloned())
            .or_else(|| {
                self.providers
                    .registry()
                    .devices_for_provider(plan.provider.as_str())
                    .find(|device| device.metadata().memory_capacity > 0)
                    .map(|device| DeviceBinding::new(device.id().clone()))
            });
        if let Some(device) = selected_device
            && let Some(runtime_device) = self.device(device.id())
        {
            let limit = runtime_device.metadata().memory_capacity;
            if limit > 0 {
                plan.pressure.selected_device = Some(device.clone());
                plan.diagnostics
                    .push(MemoryPlanningDiagnostic::DeviceLimit {
                        device: device.clone(),
                        max_bytes: limit,
                    });
                if plan.pressure.estimated_peak_bytes > limit {
                    plan.pressure.rejected_device_limit = Some(limit);
                    return Err(MemoryPlanningError::DeviceMemoryLimitExceeded {
                        device,
                        required: plan.pressure.estimated_peak_bytes,
                        limit,
                        report: plan.pressure.clone(),
                    });
                }
            }
        }
        Ok(())
    }
    pub fn plan_compute_execution(
        &self,
        graph: &ComputeGraph,
    ) -> Result<ComputeExecutionPlan, ComputePlanningError> {
        if !self.initialized {
            return Err(ComputePlanningError::PlanningFailed {
                reason: "runtime is not initialized".into(),
            });
        }

        let dependencies = graph
            .inputs
            .iter()
            .filter_map(|input| input.value.affinity())
            .collect::<Vec<_>>();
        let mut constraints = AffinityConstraints::try_from_affinities(dependencies)
            .map_err(ComputePlanningError::IncompatibleResourceAffinity)?;
        constraints.require_fallback(FallbackClass::ProviderPinned);
        constraints
            .merge(
                &ResourceAffinity::new(FallbackClass::ProviderPinned)
                    .with_capability(CapabilityBinding::new(
                        CapabilityId::new(COMPUTE_CAPABILITY_ID),
                        COMPUTE_CAPABILITY_VERSION,
                    ))
                    .with_execution_context(self.context.id),
            )
            .map_err(ComputePlanningError::IncompatibleResourceAffinity)?;
        if !graph.inputs.is_empty() && constraints.affinity().group().is_none() {
            constraints
                .merge(
                    &ResourceAffinity::new(FallbackClass::Transparent)
                        .with_group(next_affinity_group_id()),
                )
                .map_err(ComputePlanningError::IncompatibleResourceAffinity)?;
        }

        let compute = compute_capability();
        let (provider, capability, decision) = self
            .providers
            .resolve_with_constraints(
                &compute,
                &constraints,
                self.context.config.resolution_policy,
                ExecutionPhase::BeforeResourceCreation,
                true,
            )
            .map_err(planning_error_from_affinity)?;
        let metadata = provider.metadata();
        let provider_binding = ProviderBinding::new(&metadata.name);
        let selected_device = decision.selected_device.clone().or_else(|| {
            constraints.affinity().device().cloned().or_else(|| {
                self.providers
                    .registry()
                    .devices_for_provider(provider_binding.as_str())
                    .find(|device| device.availability().accepts_new_work_by_default())
                    .map(|device| DeviceBinding::new(device.id().clone()))
            })
        });
        if let Some(device) = &selected_device
            && self.device(device.id()).is_none()
        {
            return Err(ComputePlanningError::DeviceUnavailable(device.clone()));
        }

        self.validate_compute_graph(provider_binding.as_str(), graph)
            .map_err(planning_error_from_validation)?;
        let memory_plan = self
            .plan_compute_graph_memory(provider_binding.as_str(), graph)
            .map_err(ComputePlanningError::MemoryPlanFailed)?;

        let target_affinity = ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(provider_binding.clone())
            .with_capability(CapabilityBinding::new(
                capability.id.clone(),
                capability.version,
            ))
            .with_execution_context(self.context.id);
        let mut input_descriptors = BTreeMap::new();
        let mut output_descriptors = BTreeMap::new();
        let mut completed_nodes = BTreeSet::new();
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut constraints_out = vec![
            ExecutionConstraint::ResolutionPolicy(decision.policy_id.clone()),
            ExecutionConstraint::Provider(provider_binding.clone()),
            ExecutionConstraint::CapabilityVersion(CapabilityBinding::new(
                capability.id.clone(),
                capability.version,
            )),
            ExecutionConstraint::NoHiddenCpuStaging,
            ExecutionConstraint::NoImplicitProviderMigration,
            ExecutionConstraint::DeterministicBehavior,
        ];
        if let Some(device) = &selected_device {
            constraints_out.push(ExecutionConstraint::Device(device.clone()));
        }

        for input in &graph.inputs {
            let descriptor = input.value.descriptor().clone();
            input_descriptors.insert(input.id.clone(), descriptor.clone());
            let affinity = input
                .value
                .affinity()
                .cloned()
                .unwrap_or_else(|| target_affinity.clone());
            if let Some(group) = affinity.group() {
                constraints_out.push(ExecutionConstraint::AffinityGroup(group));
            }
            constraints_out.push(ExecutionConstraint::ResourceAffinity(affinity.clone()));
            let resource = match &input.value {
                ComputeInputValue::TensorResource(resource) => Some(resource.id.clone()),
                ComputeInputValue::TensorDescriptor(_) | ComputeInputValue::Constant(_) => None,
            };
            inputs.push(ExecutionInput {
                id: input.id.clone(),
                descriptor,
                resource,
                affinity,
            });
        }

        for node in &graph.nodes {
            if let Some(schema_id) = &node.operation.schema_id {
                constraints_out.push(ExecutionConstraint::OperationSchema(schema_id.clone()));
            }
            if let Some(dtype) = node.operation.dtype {
                constraints_out.push(ExecutionConstraint::DType(dtype));
            }
            if let Some(layout) = node.operation.layout {
                constraints_out.push(ExecutionConstraint::Layout(layout));
            }
            if let Some(precision) = node.operation.precision {
                constraints_out.push(ExecutionConstraint::PrecisionPolicy(precision));
            }
            for output in &node.outputs {
                output_descriptors.insert(
                    (node.id.clone(), output.id.clone()),
                    output.descriptor.clone(),
                );
            }
            completed_nodes.insert(node.id.clone());
        }

        for requirement in &memory_plan.requirements {
            constraints_out.push(ExecutionConstraint::MemoryRequirement(
                requirement.id.clone(),
            ));
        }
        for decision in &memory_plan.decisions {
            match decision {
                MemoryPlanningDecision::RequireTransfer { requirement }
                | MemoryPlanningDecision::AccountHostStaging { requirement } => {
                    constraints_out.push(ExecutionConstraint::ExplicitTransferRequirement(
                        requirement.clone(),
                    ));
                }
                MemoryPlanningDecision::RequireMaterialization { requirement } => {
                    constraints_out.push(ExecutionConstraint::ExplicitMaterializationRequired(
                        requirement.clone(),
                    ));
                }
                MemoryPlanningDecision::PreservePinnedResource { .. } => {}
                MemoryPlanningDecision::Allocate { .. } | MemoryPlanningDecision::Reuse { .. } => {}
            }
        }

        for output in &graph.outputs {
            let descriptor = resolve_compute_value_descriptor(
                None,
                &output.source,
                &input_descriptors,
                &output_descriptors,
                &completed_nodes,
            )
            .map_err(planning_error_from_validation)?
            .clone();
            let affinity = match &output.source {
                ComputeValueRef::Input(input) => inputs
                    .iter()
                    .find(|candidate| &candidate.id == input)
                    .map(|input| input.affinity.clone())
                    .unwrap_or_else(|| target_affinity.clone()),
                ComputeValueRef::NodeOutput { .. } => memory_plan.output_affinity.clone(),
            };
            outputs.push(ExecutionOutput {
                id: output.id.clone(),
                descriptor,
                affinity,
            });
        }

        let mut steps = vec![
            ExecutionStep::new(
                "validate:graph",
                ComputeExecutionPhase::Validation,
                ExecutionStepKind::ValidateGraph,
                provider_binding.clone(),
            )
            .with_device(selected_device.clone()),
            ExecutionStep::new(
                "resolve:provider",
                ComputeExecutionPhase::Resolution,
                ExecutionStepKind::ResolveProvider,
                provider_binding.clone(),
            )
            .with_device(selected_device.clone())
            .depends_on("validate:graph"),
        ];
        if selected_device.is_some() {
            steps.push(
                ExecutionStep::new(
                    "resolve:device",
                    ComputeExecutionPhase::Resolution,
                    ExecutionStepKind::ResolveDevice,
                    provider_binding.clone(),
                )
                .with_device(selected_device.clone())
                .depends_on("resolve:provider"),
            );
        }
        for input in &inputs {
            let kind = if input.affinity.fallback() == FallbackClass::ProviderPinned {
                ExecutionStepKind::PreserveProviderPinnedAffinity
            } else if input.affinity.device().is_some() {
                ExecutionStepKind::PreserveDeviceBoundAffinity
            } else if input.affinity.group().is_some() {
                ExecutionStepKind::PreserveAffinityGroup
            } else {
                ExecutionStepKind::BindInputResource
            };
            steps.push(
                ExecutionStep::new(
                    format!("bind:input:{}", input.id),
                    ComputeExecutionPhase::Planning,
                    kind,
                    provider_binding.clone(),
                )
                .with_device(selected_device.clone())
                .depends_on("resolve:provider"),
            );
        }
        for decision in &memory_plan.decisions {
            let kind = execution_step_kind_from_memory_decision(decision);
            steps.push(
                ExecutionStep::new(
                    format!("memory:{decision:?}"),
                    execution_phase_from_step_kind(&kind),
                    kind,
                    provider_binding.clone(),
                )
                .with_device(selected_device.clone())
                .depends_on("resolve:provider"),
            );
        }
        steps.push(
            ExecutionStep::new(
                "validate:memory",
                ComputeExecutionPhase::MemoryAllocation,
                ExecutionStepKind::ValidateMemory,
                provider_binding.clone(),
            )
            .with_device(selected_device.clone())
            .depends_on("resolve:provider"),
        );
        for output in &outputs {
            steps.push(
                ExecutionStep::new(
                    format!("bind:output:{}", output.id),
                    ComputeExecutionPhase::Planning,
                    ExecutionStepKind::BindOutputResource,
                    provider_binding.clone(),
                )
                .with_device(selected_device.clone())
                .depends_on("validate:memory"),
            );
        }
        steps.push(
            ExecutionStep::new(
                "submit:provider",
                ComputeExecutionPhase::ProviderSubmission,
                ExecutionStepKind::SubmitToProvider,
                provider_binding.clone(),
            )
            .with_device(selected_device.clone())
            .depends_on("validate:memory"),
        );

        let mut diagnostics = vec![
            ExecutionDiagnostic::SelectedProvider(provider_binding.clone()),
            ExecutionDiagnostic::SelectedCapability(CapabilityBinding::new(
                capability.id.clone(),
                capability.version,
            )),
            ExecutionDiagnostic::PolicyDecisionReason(decision.reason.clone()),
            ExecutionDiagnostic::ResolutionDecision(decision.clone()),
        ];
        if let Some(device) = &selected_device {
            diagnostics.push(ExecutionDiagnostic::SelectedDevice(device.clone()));
        }
        diagnostics.extend(decision.rejected_candidates.iter().map(|rejection| {
            ExecutionDiagnostic::RejectedProviderCandidate {
                provider: rejection.provider.clone(),
                reason: rejection.reason.clone(),
            }
        }));
        diagnostics.extend(
            memory_plan
                .diagnostics
                .iter()
                .cloned()
                .map(ExecutionDiagnostic::Memory),
        );

        let mut plan = ComputeExecutionPlan {
            id: execution_plan_id(&graph.id, &provider_binding),
            trace_id: TraceId::new(format!("trace:{}", graph.id)),
            graph: graph.id.clone(),
            provider: provider_binding,
            device: selected_device,
            capability: CapabilityBinding::new(capability.id.clone(), capability.version),
            policy: decision.policy_id,
            classification: classify_execution_plan(&inputs),
            inputs,
            outputs,
            constraints: constraints_out,
            steps,
            memory_plan,
            diagnostics,
            validated: false,
        };
        self.validate_compute_execution_plan(&plan)?;
        plan.validated = true;
        Ok(plan)
    }
    pub fn validate_compute_execution_plan(
        &self,
        plan: &ComputeExecutionPlan,
    ) -> Result<(), ComputePlanningError> {
        if plan.graph
            != plan.memory_plan.graph.clone().ok_or_else(|| {
                ComputePlanningError::InvalidExecutionPlan {
                    reason: "execution plan memory plan is not tied to a graph".into(),
                }
            })?
        {
            return Err(ComputePlanningError::InvalidExecutionPlan {
                reason: "execution plan graph does not match memory plan graph".into(),
            });
        }
        if self.providers.provider(plan.provider.as_str()).is_none() {
            return Err(ComputePlanningError::ProviderUnavailable(
                plan.provider.clone(),
            ));
        }
        if let Some(device) = &plan.device
            && self.device(device.id()).is_none()
        {
            return Err(ComputePlanningError::DeviceUnavailable(device.clone()));
        }
        let step_ids = plan
            .steps
            .iter()
            .map(|step| step.id.as_str())
            .collect::<BTreeSet<_>>();
        for step in &plan.steps {
            for dependency in &step.dependencies {
                if !step_ids.contains(dependency.as_str()) {
                    return Err(ComputePlanningError::InvalidExecutionPlan {
                        reason: format!(
                            "step '{}' has unresolved dependency '{}'",
                            step.id, dependency
                        ),
                    });
                }
            }
            if step.provider != plan.provider {
                return Err(ComputePlanningError::InvalidExecutionPlan {
                    reason: format!(
                        "step '{}' migrates provider from '{}' to '{}'",
                        step.id, plan.provider, step.provider
                    ),
                });
            }
        }
        if plan
            .inputs
            .iter()
            .any(|input| input.resource.is_some() && input.affinity.provider().is_none())
        {
            return Err(ComputePlanningError::InvalidExecutionPlan {
                reason: "tensor resource inputs must retain Provider affinity".into(),
            });
        }
        Ok(())
    }
    pub fn scheduler(&self, capacity: usize) -> Scheduler {
        Scheduler::new(SchedulingPolicy::Fifo, capacity)
    }
    pub fn validate_scheduler_plan(
        &self,
        plan: &ComputeExecutionPlan,
    ) -> Result<(), ComputePlanningError> {
        if !plan.is_validated() {
            return Err(ComputePlanningError::InvalidExecutionPlan {
                reason: "scheduler accepts only validated execution plans".into(),
            });
        }
        self.validate_compute_execution_plan(plan)?;
        if !plan.constraints.iter().any(|constraint| {
            matches!(constraint, ExecutionConstraint::NoImplicitProviderMigration)
        }) {
            return Err(ComputePlanningError::InvalidExecutionPlan {
                reason: "execution plan must forbid implicit Provider migration".into(),
            });
        }
        Ok(())
    }
    pub fn schedule_compute_execution(
        &self,
        scheduler: &mut Scheduler,
        plan: ComputeExecutionPlan,
    ) -> Result<ScheduledOperationId, SchedulerError> {
        scheduler.schedule(self, plan)
    }
    pub fn prepare_provider_execution(
        &self,
        operation: &ScheduledOperation,
    ) -> Result<ProviderExecutionRequest, ProviderExecutionError> {
        self.validate_scheduler_plan(&operation.plan)
            .map_err(|error| {
                ProviderExecutionError::new(
                    ProviderExecutionErrorCode::InvalidExecutionPlan,
                    ProviderExecutionPhase::Prepare,
                    operation.plan.provider.clone(),
                    operation.plan.device.clone(),
                    error.to_string(),
                )
            })?;
        self.validate_provider_execution_bindings(
            &operation.plan.provider,
            operation.plan.device.as_ref(),
            ProviderExecutionPhase::Prepare,
        )?;
        if operation.plan.provider != operation.plan.memory_plan.provider {
            return Err(ProviderExecutionError::new(
                ProviderExecutionErrorCode::MemoryPlanRejected,
                ProviderExecutionPhase::Prepare,
                operation.plan.provider.clone(),
                operation.plan.device.clone(),
                "memory plan provider does not match selected Provider",
            ));
        }
        let provider_affinity = ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(operation.plan.provider.clone())
            .with_capability(operation.plan.capability.clone());
        provider_affinity
            .validate_with(&operation.plan.memory_plan.output_affinity)
            .map_err(|error| {
                ProviderExecutionError::new(
                    ProviderExecutionErrorCode::IncompatibleResourceAffinity,
                    ProviderExecutionPhase::Prepare,
                    operation.plan.provider.clone(),
                    operation.plan.device.clone(),
                    error.to_string(),
                )
            })?;
        if !operation
            .plan
            .steps
            .iter()
            .any(|step| step.kind == ExecutionStepKind::SubmitToProvider)
        {
            return Err(ProviderExecutionError::new(
                ProviderExecutionErrorCode::InvalidExecutionPlan,
                ProviderExecutionPhase::Prepare,
                operation.plan.provider.clone(),
                operation.plan.device.clone(),
                "execution plan does not contain a Provider submission step",
            ));
        }
        Ok(ProviderExecutionRequest::from_operation(operation))
    }
    pub fn submit_provider_execution(
        &self,
        request: ProviderExecutionRequest,
    ) -> Result<ProviderExecutionHandle, ProviderExecutionError> {
        self.validate_compute_execution_plan(&request.plan)
            .map_err(|error| {
                ProviderExecutionError::new(
                    ProviderExecutionErrorCode::InvalidExecutionPlan,
                    ProviderExecutionPhase::Submit,
                    request.provider.clone(),
                    request.device.clone(),
                    error.to_string(),
                )
            })?;
        if !request.plan.is_validated() {
            return Err(ProviderExecutionError::new(
                ProviderExecutionErrorCode::InvalidExecutionPlan,
                ProviderExecutionPhase::Submit,
                request.provider.clone(),
                request.device.clone(),
                "Provider execution accepts only validated execution plans",
            ));
        }
        if request.provider != request.plan.provider || request.device != request.plan.device {
            return Err(ProviderExecutionError::new(
                ProviderExecutionErrorCode::InvalidExecutionPlan,
                ProviderExecutionPhase::Submit,
                request.provider.clone(),
                request.device.clone(),
                "Provider execution request changed selected Provider or Device",
            ));
        }
        self.validate_provider_execution_bindings(
            &request.provider,
            request.device.as_ref(),
            ProviderExecutionPhase::Submit,
        )?;
        let api = self.provider_execution_api(
            &request.provider,
            request.device.as_ref(),
            ProviderExecutionPhase::Submit,
        )?;
        let handle = api.submit(request.clone())?;
        if handle.operation != request.operation
            || handle.plan != request.plan.id
            || handle.provider != request.provider
            || handle.device != request.device
        {
            return Err(ProviderExecutionError::new(
                ProviderExecutionErrorCode::SubmissionFailed,
                ProviderExecutionPhase::Submit,
                request.provider,
                request.device,
                "Provider returned a handle that does not match the scheduled operation",
            ));
        }
        Ok(handle)
    }
    pub fn observe_provider_execution(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderExecutionStatus, ProviderExecutionError> {
        let api = self.provider_execution_api(
            &handle.provider,
            handle.device.as_ref(),
            ProviderExecutionPhase::Observe,
        )?;
        let status = api.status(handle)?;
        if status.handle != *handle {
            return Err(ProviderExecutionError::new(
                ProviderExecutionErrorCode::ExecutionFailed,
                ProviderExecutionPhase::Observe,
                handle.provider.clone(),
                handle.device.clone(),
                "Provider returned status for a different execution handle",
            ));
        }
        Ok(status)
    }
    pub fn cancel_provider_execution(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderCancellationOutcome, ProviderExecutionError> {
        let api = self.provider_execution_api(
            &handle.provider,
            handle.device.as_ref(),
            ProviderExecutionPhase::Cancel,
        )?;
        api.cancel(handle)
    }
    pub fn complete_provider_execution(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderExecutionResult, ProviderExecutionError> {
        let api = self.provider_execution_api(
            &handle.provider,
            handle.device.as_ref(),
            ProviderExecutionPhase::Complete,
        )?;
        let result = api.complete(handle)?;
        if result.handle != *handle {
            return Err(ProviderExecutionError::new(
                ProviderExecutionErrorCode::ExecutionFailed,
                ProviderExecutionPhase::Complete,
                handle.provider.clone(),
                handle.device.clone(),
                "Provider returned result for a different execution handle",
            ));
        }
        if !result.state.is_terminal() {
            return Err(ProviderExecutionError::new(
                ProviderExecutionErrorCode::ExecutionFailed,
                ProviderExecutionPhase::Complete,
                handle.provider.clone(),
                handle.device.clone(),
                "Provider completion result must be terminal",
            ));
        }
        let expected_affinity = ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(handle.provider.clone())
            .with_capability(CapabilityBinding::new(
                CapabilityId::new(COMPUTE_CAPABILITY_ID),
                COMPUTE_CAPABILITY_VERSION,
            ));
        for output in &result.outputs {
            expected_affinity
                .validate_with(&output.affinity)
                .map_err(|error| {
                    ProviderExecutionError::new(
                        ProviderExecutionErrorCode::IncompatibleResourceAffinity,
                        ProviderExecutionPhase::Complete,
                        handle.provider.clone(),
                        handle.device.clone(),
                        error.to_string(),
                    )
                })?;
        }
        Ok(result)
    }
    pub fn release_provider_execution(
        &self,
        handle: ProviderExecutionHandle,
    ) -> Result<(), ProviderExecutionError> {
        let api = self.provider_execution_api(
            &handle.provider,
            handle.device.as_ref(),
            ProviderExecutionPhase::Release,
        )?;
        api.release(handle)
    }
    fn provider_execution_api(
        &self,
        provider: &ProviderBinding,
        device: Option<&DeviceBinding>,
        phase: ProviderExecutionPhase,
    ) -> Result<&dyn ProviderExecutionApi, ProviderExecutionError> {
        self.validate_provider_execution_bindings(provider, device, phase)?;
        let provider_ref = self.providers.provider(provider.as_str()).ok_or_else(|| {
            ProviderExecutionError::new(
                ProviderExecutionErrorCode::ProviderUnavailable,
                phase,
                provider.clone(),
                device.cloned(),
                "selected Provider is unavailable",
            )
        })?;
        provider_ref.execution_api().ok_or_else(|| {
            ProviderExecutionError::new(
                ProviderExecutionErrorCode::UnsupportedOperation,
                phase,
                provider.clone(),
                device.cloned(),
                "selected Provider does not implement ProviderExecutionApi",
            )
        })
    }
    fn validate_provider_execution_bindings(
        &self,
        provider: &ProviderBinding,
        device: Option<&DeviceBinding>,
        phase: ProviderExecutionPhase,
    ) -> Result<(), ProviderExecutionError> {
        let provider_ref = self.providers.provider(provider.as_str()).ok_or_else(|| {
            ProviderExecutionError::new(
                ProviderExecutionErrorCode::ProviderUnavailable,
                phase,
                provider.clone(),
                device.cloned(),
                "selected Provider is unavailable",
            )
        })?;
        let provider_status = provider_ref.status_snapshot();
        if matches!(provider_status.health_reason, ProviderStatusReason::Stale) {
            return Err(ProviderExecutionError::new(
                ProviderExecutionErrorCode::StaleHealthReport,
                phase,
                provider.clone(),
                device.cloned(),
                "selected Provider status is stale",
            ));
        }
        if let Some(code) =
            provider_execution_error_for_health(provider_status.provider_health_compat())
        {
            return Err(ProviderExecutionError::new(
                code,
                phase,
                provider.clone(),
                device.cloned(),
                format!(
                    "selected Provider status is {:?}/{:?}/{:?}/{:?}",
                    provider_status.lifecycle,
                    provider_status.health,
                    provider_status.readiness,
                    provider_status.pressure
                ),
            ));
        }
        if let Some(device) = device {
            let Some(runtime_device) = self.device(device.id()) else {
                return Err(ProviderExecutionError::new(
                    ProviderExecutionErrorCode::DeviceUnavailable,
                    phase,
                    provider.clone(),
                    Some(device.clone()),
                    "selected Device is unavailable",
                ));
            };
            if let Some(code) = device_execution_error_for_health(runtime_device.availability()) {
                return Err(ProviderExecutionError::new(
                    code,
                    phase,
                    provider.clone(),
                    Some(device.clone()),
                    format!(
                        "selected Device health is {:?}",
                        runtime_device.availability()
                    ),
                ));
            }
            if runtime_device.metadata().provider != provider.as_str() {
                return Err(ProviderExecutionError::new(
                    ProviderExecutionErrorCode::IncompatibleResourceAffinity,
                    phase,
                    provider.clone(),
                    Some(device.clone()),
                    "selected Device is owned by a different Provider",
                ));
            }
        }
        Ok(())
    }
    pub fn validate_compute_graph(
        &self,
        provider: &str,
        graph: &ComputeGraph,
    ) -> Result<ComputeGraphValidationReport, ComputeValidationError> {
        ensure_non_empty_id("graph", graph.id.as_str())?;

        let mut input_ids = BTreeSet::new();
        let mut input_descriptors = BTreeMap::new();
        let mut resource_affinities = Vec::new();
        for input in &graph.inputs {
            ensure_non_empty_id("input", input.id.as_str())?;
            insert_unique(&mut input_ids, "input", &input.id)?;
            input
                .value
                .descriptor()
                .validate(&TensorDescriptorLimits::default())?;
            input_descriptors.insert(input.id.clone(), input.value.descriptor().clone());
            if let Some(affinity) = input.value.affinity() {
                resource_affinities.push(affinity);
            }
        }

        let mut node_ids = BTreeSet::new();
        let mut completed_nodes = BTreeSet::new();
        let mut output_descriptors = BTreeMap::new();
        let mut operations = Vec::new();
        for node in &graph.nodes {
            ensure_non_empty_id("node", node.id.as_str())?;
            insert_unique(&mut node_ids, "node", &node.id)?;

            let mut node_output_ids = BTreeSet::new();
            for output in &node.outputs {
                ensure_non_empty_id("node output", output.id.as_str())?;
                insert_unique(&mut node_output_ids, "node output", &output.id)?;
                output
                    .descriptor
                    .validate(&TensorDescriptorLimits::default())?;
                output_descriptors.insert(
                    (node.id.clone(), output.id.clone()),
                    output.descriptor.clone(),
                );
            }

            let mut operation = node.operation.clone();
            for input in &node.inputs {
                operation.tensors.push(
                    resolve_compute_value_descriptor(
                        Some(&node.id),
                        input,
                        &input_descriptors,
                        &output_descriptors,
                        &completed_nodes,
                    )?
                    .clone(),
                );
            }
            operation
                .tensors
                .extend(node.outputs.iter().map(|output| output.descriptor.clone()));
            operations.push(operation);
            completed_nodes.insert(node.id.clone());
        }

        let mut graph_output_ids = BTreeSet::new();
        for output in &graph.outputs {
            ensure_non_empty_id("graph output", output.id.as_str())?;
            insert_unique(&mut graph_output_ids, "graph output", &output.id)?;
            resolve_compute_value_descriptor(
                None,
                &output.source,
                &input_descriptors,
                &output_descriptors,
                &completed_nodes,
            )?;
        }

        self.validate_compute_operations(provider, &operations)?;
        self.plan_compute_graph_memory(provider, graph)
            .map_err(ComputeValidationError::MemoryPlanning)?;
        if !resource_affinities.is_empty() {
            let resources = graph
                .inputs
                .iter()
                .filter_map(|input| match &input.value {
                    ComputeInputValue::TensorResource(resource) => Some(resource.clone()),
                    ComputeInputValue::TensorDescriptor(_) | ComputeInputValue::Constant(_) => None,
                })
                .collect::<Vec<_>>();
            self.validate_compute_tensor_resources(provider, &resources)?;
        }

        Ok(ComputeGraphValidationReport {
            provider: ProviderBinding::new(provider),
            graph: graph.id.clone(),
            node_count: graph.nodes.len(),
            input_count: graph.inputs.len(),
            output_count: graph.outputs.len(),
        })
    }
    pub fn submit_validated_compute_graph(
        &self,
        provider: &str,
        graph: &ComputeGraph,
    ) -> Result<ComputeSubmission, ComputeValidationError> {
        self.validate_compute_graph(provider, graph)?;
        let dependencies = graph
            .inputs
            .iter()
            .filter_map(|input| input.value.affinity())
            .collect::<Vec<_>>();
        let mut constraints = AffinityConstraints::try_from_affinities(dependencies)
            .map_err(ComputeValidationError::IncompatibleResourceAffinity)?;
        constraints.require_fallback(FallbackClass::ProviderPinned);
        constraints
            .merge(
                &ResourceAffinity::new(FallbackClass::ProviderPinned)
                    .with_provider(ProviderBinding::new(provider))
                    .with_capability(CapabilityBinding::new(
                        CapabilityId::new(COMPUTE_CAPABILITY_ID),
                        COMPUTE_CAPABILITY_VERSION,
                    ))
                    .with_execution_context(self.context.id),
            )
            .map_err(ComputeValidationError::IncompatibleResourceAffinity)?;
        if constraints.affinity().group().is_none() {
            constraints
                .merge(
                    &ResourceAffinity::new(FallbackClass::Transparent)
                        .with_group(next_affinity_group_id()),
                )
                .map_err(ComputeValidationError::IncompatibleResourceAffinity)?;
        }
        let affinity = constraints.into_affinity();
        Ok(ComputeSubmission::new(
            graph.id.clone(),
            ProviderBinding::new(provider),
            affinity,
        ))
    }
    pub fn wrap_compute_outputs(
        &self,
        submission: &ComputeSubmission,
        outputs: impl IntoIterator<Item = (TensorResourceId, TensorDescriptor)>,
    ) -> Vec<TensorResourceDescriptor> {
        outputs
            .into_iter()
            .map(|(id, descriptor)| {
                TensorResourceDescriptor::new(id, descriptor, submission.affinity.clone())
            })
            .collect()
    }
    pub fn wrap_compute_data_movement_output(
        &self,
        provider: &str,
        movement: &ComputeDataMovementDescriptor,
        id: TensorResourceId,
    ) -> Result<TensorResourceDescriptor, ComputeValidationError> {
        self.validate_compute_data_movement(provider, std::slice::from_ref(movement))?;
        let mut affinity = ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(ProviderBinding::new(provider))
            .with_capability(CapabilityBinding::new(
                CapabilityId::new(COMPUTE_CAPABILITY_ID),
                COMPUTE_CAPABILITY_VERSION,
            ))
            .with_execution_context(self.context.id);
        if let Some(source) = movement.source.tensor() {
            if movement.placement == ComputePlacementIntent::PreserveSourceAffinity
                && let Some(device) = source.affinity.device()
            {
                affinity = affinity.with_device(device.clone());
            }
            if let Some(group) = source.affinity.group() {
                affinity = affinity.with_group(group);
            }
        }
        Ok(TensorResourceDescriptor::new(
            id,
            movement.output.clone(),
            affinity,
        ))
    }
    pub fn shutdown(&mut self) {
        let _ = self.providers.shutdown();
        self.initialized = false;
    }
}

pub(crate) fn provider_execution_error_for_health(
    health: ProviderHealth,
) -> Option<ProviderExecutionErrorCode> {
    match health {
        HealthState::Unknown => Some(ProviderExecutionErrorCode::ProviderHealthUnknown),
        HealthState::Initializing => Some(ProviderExecutionErrorCode::ProviderInitializing),
        HealthState::Available | HealthState::Degraded => None,
        HealthState::Saturated => Some(ProviderExecutionErrorCode::ProviderSaturated),
        HealthState::Draining => Some(ProviderExecutionErrorCode::ProviderDraining),
        HealthState::Unavailable => Some(ProviderExecutionErrorCode::ProviderUnavailable),
        HealthState::Interrupted => Some(ProviderExecutionErrorCode::ProviderInterrupted),
    }
}

pub(crate) fn device_execution_error_for_health(
    health: DeviceAvailability,
) -> Option<ProviderExecutionErrorCode> {
    match health {
        HealthState::Unknown | HealthState::Initializing => {
            Some(ProviderExecutionErrorCode::DeviceHealthUnknown)
        }
        HealthState::Available | HealthState::Degraded | HealthState::Draining => None,
        HealthState::Saturated => Some(ProviderExecutionErrorCode::DeviceSaturated),
        HealthState::Unavailable | HealthState::Interrupted => {
            Some(ProviderExecutionErrorCode::DeviceUnavailable)
        }
    }
}

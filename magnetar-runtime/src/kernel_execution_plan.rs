//! Runtime-owned Prepared Execution Plan contract.
//!
//! A Prepared Execution Plan captures already-validated Runtime decisions for
//! executing one compatible [`crate::ExecutionGraph`]. It deliberately stores
//! stable logical identifiers and metadata only; Provider-native executable
//! state remains behind opaque Provider-owned IDs.

use crate::kernel_registry::KernelRegistry;
use crate::{
    AdapterRevision, DTypeDescriptor, DeviceBinding, ExecutionGraph, ExecutionGraphPhase,
    ExecutionNodeId, KernelExecutionMode, KernelId, ModelInstanceId, PreparedKernelGeneration,
    PreparedKernelId, ProviderBinding, ResourceAffinity, TensorLayoutKind, TensorResourceId,
    layout_kind,
};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PreparedExecutionPlanId(String);

impl PreparedExecutionPlanId {
    pub fn new(value: impl Into<String>) -> Result<Self, PreparedExecutionPlanError> {
        let value = value.into();
        validate_logical_identity(&value, "prepared execution plan id")?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PreparedExecutionPlanId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PreparedExecutionPlanGeneration(u64);

impl PreparedExecutionPlanGeneration {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn value(self) -> u64 {
        self.0
    }

    pub const fn is_newer_than(self, other: Self) -> bool {
        self.0 > other.0
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ExecutionGraphSemanticFingerprint(String);

impl ExecutionGraphSemanticFingerprint {
    pub fn new(value: impl Into<String>) -> Result<Self, PreparedExecutionPlanError> {
        let value = value.into();
        validate_fingerprint(&value, "execution graph fingerprint")?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ExecutionGraphSemanticFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PreparedExecutionPlanFingerprint(String);

impl PreparedExecutionPlanFingerprint {
    pub fn new(value: impl Into<String>) -> Result<Self, PreparedExecutionPlanError> {
        let value = value.into();
        validate_fingerprint(&value, "prepared execution plan fingerprint")?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PreparedExecutionPlanFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PreparedExecutionPlanState {
    Building,
    Validating,
    Preparing,
    Ready,
    Stale,
    Invalidated,
    Retiring,
    Retired,
    Failed,
}

impl PreparedExecutionPlanState {
    pub const fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Building, Self::Validating)
                | (Self::Building, Self::Failed)
                | (Self::Validating, Self::Preparing)
                | (Self::Validating, Self::Failed)
                | (Self::Preparing, Self::Ready)
                | (Self::Preparing, Self::Failed)
                | (Self::Ready, Self::Stale)
                | (Self::Ready, Self::Invalidated)
                | (Self::Ready, Self::Retiring)
                | (Self::Ready, Self::Failed)
                | (Self::Stale, Self::Ready)
                | (Self::Stale, Self::Invalidated)
                | (Self::Stale, Self::Retiring)
                | (Self::Stale, Self::Failed)
                | (Self::Invalidated, Self::Retiring)
                | (Self::Invalidated, Self::Failed)
                | (Self::Retiring, Self::Retired)
        )
    }

    pub const fn accepts_new_work(self) -> bool {
        matches!(self, Self::Ready | Self::Stale)
    }

    pub const fn requires_replacement_for_new_work(self) -> bool {
        matches!(
            self,
            Self::Invalidated | Self::Retiring | Self::Retired | Self::Failed
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PreparedExecutionPhase {
    ModelLoad,
    Warmup,
    Prefill,
    Decode,
    AdapterActivation,
    AdapterMerge,
    SamplingHelper,
    Test,
}

impl From<ExecutionGraphPhase> for PreparedExecutionPhase {
    fn from(value: ExecutionGraphPhase) -> Self {
        match value {
            ExecutionGraphPhase::ModelLoad => Self::ModelLoad,
            ExecutionGraphPhase::Warmup => Self::Warmup,
            ExecutionGraphPhase::Prefill => Self::Prefill,
            ExecutionGraphPhase::Decode => Self::Decode,
            ExecutionGraphPhase::AdapterActivation => Self::AdapterActivation,
            ExecutionGraphPhase::AdapterMerge => Self::AdapterMerge,
            ExecutionGraphPhase::SamplingHelper => Self::SamplingHelper,
            ExecutionGraphPhase::Test => Self::Test,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ShapeDimensionEnvelope {
    Exact(u64),
    Range { min: u64, max: u64 },
    Any,
}

impl ShapeDimensionEnvelope {
    pub const fn contains(&self, value: u64) -> bool {
        match self {
            Self::Exact(expected) => *expected == value,
            Self::Range { min, max } => *min <= value && value <= *max,
            Self::Any => true,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanShapeEnvelope {
    pub dimensions: Vec<ShapeDimensionEnvelope>,
}

impl PlanShapeEnvelope {
    pub fn new(dimensions: impl Into<Vec<ShapeDimensionEnvelope>>) -> Self {
        Self {
            dimensions: dimensions.into(),
        }
    }

    pub fn validate(&self) -> Result<(), PreparedExecutionPlanError> {
        for dimension in &self.dimensions {
            if let ShapeDimensionEnvelope::Range { min, max } = dimension
                && min > max
            {
                return Err(PreparedExecutionPlanError::InvalidShapeEnvelope);
            }
        }
        Ok(())
    }

    pub fn contains(&self, dimensions: &[u64]) -> bool {
        self.dimensions.len() == dimensions.len()
            && self
                .dimensions
                .iter()
                .zip(dimensions)
                .all(|(envelope, value)| envelope.contains(*value))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedExecutionPlanScope {
    pub model_instance: Option<ModelInstanceId>,
    pub model_instance_revision: Option<u64>,
    pub adapter_revision: Option<AdapterRevision>,
    pub execution_policy_revision: Option<String>,
    pub phase: PreparedExecutionPhase,
    pub workload_bucket: Option<String>,
    pub shape_envelopes: Vec<PlanShapeEnvelope>,
    pub dtypes: BTreeSet<DTypeDescriptor>,
    pub layouts: BTreeSet<TensorLayoutKind>,
    pub batching_mode: Option<String>,
    pub kv_cache_mode: Option<String>,
    pub provider: Option<ProviderBinding>,
    pub device: Option<DeviceBinding>,
    pub quantization_mode: Option<String>,
}

impl PreparedExecutionPlanScope {
    pub fn for_phase(phase: PreparedExecutionPhase) -> Self {
        Self {
            model_instance: None,
            model_instance_revision: None,
            adapter_revision: None,
            execution_policy_revision: None,
            phase,
            workload_bucket: None,
            shape_envelopes: Vec::new(),
            dtypes: BTreeSet::new(),
            layouts: BTreeSet::new(),
            batching_mode: None,
            kv_cache_mode: None,
            provider: None,
            device: None,
            quantization_mode: None,
        }
    }

    pub fn with_model_instance(mut self, id: ModelInstanceId, revision: u64) -> Self {
        self.model_instance = Some(id);
        self.model_instance_revision = Some(revision);
        self
    }

    pub fn with_adapter_revision(mut self, revision: AdapterRevision) -> Self {
        self.adapter_revision = Some(revision);
        self
    }

    pub fn with_execution_policy_revision(mut self, revision: impl Into<String>) -> Self {
        self.execution_policy_revision = Some(revision.into());
        self
    }

    pub fn with_workload_bucket(mut self, bucket: impl Into<String>) -> Self {
        self.workload_bucket = Some(bucket.into());
        self
    }

    pub fn validate(&self) -> Result<(), PreparedExecutionPlanError> {
        if self.model_instance.is_some() != self.model_instance_revision.is_some() {
            return Err(PreparedExecutionPlanError::ModelBindingIncomplete);
        }
        if let Some(bucket) = &self.workload_bucket {
            validate_logical_identity(bucket, "workload bucket")?;
        }
        if let Some(revision) = &self.execution_policy_revision {
            validate_logical_identity(revision, "execution policy revision")?;
        }
        if let Some(mode) = &self.batching_mode {
            validate_logical_identity(mode, "batching mode")?;
        }
        if let Some(mode) = &self.kv_cache_mode {
            validate_logical_identity(mode, "KV cache mode")?;
        }
        if let Some(mode) = &self.quantization_mode {
            validate_logical_identity(mode, "quantization mode")?;
        }
        for envelope in &self.shape_envelopes {
            envelope.validate()?;
        }
        Ok(())
    }

    pub fn validate_model_binding(
        &self,
        id: &ModelInstanceId,
        revision: u64,
        adapter_revision: Option<&AdapterRevision>,
        execution_policy_revision: Option<&str>,
    ) -> Result<(), PreparedExecutionPlanError> {
        if self
            .model_instance
            .as_ref()
            .is_some_and(|expected| expected != id)
        {
            return Err(PreparedExecutionPlanError::ModelInstanceMismatch);
        }
        if self
            .model_instance_revision
            .is_some_and(|expected| expected != revision)
        {
            return Err(PreparedExecutionPlanError::ModelInstanceRevisionMismatch);
        }
        if self.adapter_revision.as_ref() != adapter_revision {
            return Err(PreparedExecutionPlanError::AdapterRevisionMismatch);
        }
        if self.execution_policy_revision.as_deref() != execution_policy_revision {
            return Err(PreparedExecutionPlanError::ExecutionPolicyRevisionMismatch);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanNodeBinding {
    pub graph_nodes: BTreeSet<ExecutionNodeId>,
    pub kernel: KernelId,
    pub qualification_profile: Option<String>,
    pub kernel_artifact_digest: Option<String>,
    pub specialization_id: Option<String>,
    pub provider: ProviderBinding,
    pub device: Option<DeviceBinding>,
    pub prepared_kernel: Option<PreparedKernelId>,
    pub prepared_kernel_generation: Option<PreparedKernelGeneration>,
    pub execution_mode: KernelExecutionMode,
}

impl PlanNodeBinding {
    pub fn new(
        graph_nodes: impl IntoIterator<Item = ExecutionNodeId>,
        kernel: KernelId,
        provider: ProviderBinding,
    ) -> Result<Self, PreparedExecutionPlanError> {
        let graph_nodes = graph_nodes.into_iter().collect::<BTreeSet<_>>();
        if graph_nodes.is_empty() {
            return Err(PreparedExecutionPlanError::PlanNodeBindingEmpty);
        }
        Ok(Self {
            graph_nodes,
            kernel,
            qualification_profile: None,
            kernel_artifact_digest: None,
            specialization_id: None,
            provider,
            device: None,
            prepared_kernel: None,
            prepared_kernel_generation: None,
            execution_mode: KernelExecutionMode::Synchronous,
        })
    }

    pub fn with_artifact_digest(mut self, digest: impl Into<String>) -> Self {
        self.kernel_artifact_digest = Some(digest.into());
        self
    }

    pub fn with_qualification_profile(mut self, profile: impl Into<String>) -> Self {
        self.qualification_profile = Some(profile.into());
        self
    }

    pub fn with_specialization(mut self, specialization: impl Into<String>) -> Self {
        self.specialization_id = Some(specialization.into());
        self
    }

    pub fn with_device(mut self, device: DeviceBinding) -> Self {
        self.device = Some(device);
        self
    }

    pub fn with_prepared_kernel(
        mut self,
        id: PreparedKernelId,
        generation: PreparedKernelGeneration,
    ) -> Self {
        self.prepared_kernel = Some(id);
        self.prepared_kernel_generation = Some(generation);
        self
    }

    pub fn validate(&self) -> Result<(), PreparedExecutionPlanError> {
        if self.graph_nodes.is_empty() {
            return Err(PreparedExecutionPlanError::PlanNodeBindingEmpty);
        }
        if self.prepared_kernel.is_some() != self.prepared_kernel_generation.is_some() {
            return Err(PreparedExecutionPlanError::PreparedKernelBindingIncomplete);
        }
        if self
            .qualification_profile
            .as_ref()
            .is_some_and(|profile| profile.trim().is_empty())
        {
            return Err(PreparedExecutionPlanError::QualificationProfileMissing);
        }
        if self
            .kernel_artifact_digest
            .as_ref()
            .is_some_and(|digest| digest.trim().is_empty())
        {
            return Err(PreparedExecutionPlanError::KernelArtifactDigestMissing);
        }
        if self
            .specialization_id
            .as_ref()
            .is_some_and(|specialization| specialization.trim().is_empty())
        {
            return Err(PreparedExecutionPlanError::SpecializationMissing);
        }
        Ok(())
    }

    pub fn validate_exact_binding(
        &self,
        kernel: &KernelId,
        qualification_profile: Option<&str>,
        artifact_digest: Option<&str>,
        specialization_id: Option<&str>,
        prepared_generation: Option<PreparedKernelGeneration>,
    ) -> Result<(), PreparedExecutionPlanError> {
        if &self.kernel != kernel {
            return Err(PreparedExecutionPlanError::KernelBindingMismatch);
        }
        if self.qualification_profile.as_deref() != qualification_profile {
            return Err(PreparedExecutionPlanError::QualificationProfileMismatch);
        }
        if self.kernel_artifact_digest.as_deref() != artifact_digest {
            return Err(PreparedExecutionPlanError::KernelArtifactDigestMismatch);
        }
        if self.specialization_id.as_deref() != specialization_id {
            return Err(PreparedExecutionPlanError::SpecializationMismatch);
        }
        if self.prepared_kernel_generation != prepared_generation {
            return Err(PreparedExecutionPlanError::PreparedKernelGenerationMismatch);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PreparedExecutionSegmentId(String);

impl PreparedExecutionSegmentId {
    pub fn new(value: impl Into<String>) -> Result<Self, PreparedExecutionPlanError> {
        let value = value.into();
        validate_logical_identity(&value, "prepared execution segment id")?;
        Ok(Self(value))
    }
}

impl fmt::Display for PreparedExecutionSegmentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProviderPreparedSegmentId(String);

impl ProviderPreparedSegmentId {
    pub fn new(value: impl Into<String>) -> Result<Self, PreparedExecutionPlanError> {
        let value = value.into();
        validate_logical_identity(&value, "provider prepared segment id")?;
        Ok(Self(value))
    }
}

impl fmt::Display for ProviderPreparedSegmentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProviderPreparedSegmentState {
    Preparing,
    Ready,
    Invalid,
    Retiring,
    Destroyed,
}

impl ProviderPreparedSegmentState {
    pub const fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Preparing, Self::Ready)
                | (Self::Preparing, Self::Invalid)
                | (Self::Ready, Self::Invalid)
                | (Self::Ready, Self::Retiring)
                | (Self::Invalid, Self::Destroyed)
                | (Self::Retiring, Self::Destroyed)
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SegmentCaptureFallback {
    IndividualKernelDispatch,
    FailPlanPreparation,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedExecutionSegment {
    pub id: PreparedExecutionSegmentId,
    pub graph_nodes: BTreeSet<ExecutionNodeId>,
    pub provider: ProviderBinding,
    pub device: Option<DeviceBinding>,
    pub provider_prepared_segment: Option<ProviderPreparedSegmentId>,
    pub provider_state: ProviderPreparedSegmentState,
    pub kernel_generations: BTreeSet<PreparedKernelGeneration>,
    pub resource_model: Option<String>,
    pub shape_envelope: Option<PlanShapeEnvelope>,
    pub explicit_data_movement: Vec<String>,
    pub resource_affinity: Option<ResourceAffinity>,
    pub host_staging_allowed: bool,
    pub preserves_graph_semantics: bool,
    pub fallback: SegmentCaptureFallback,
}

impl PreparedExecutionSegment {
    pub fn new(
        id: PreparedExecutionSegmentId,
        graph_nodes: impl IntoIterator<Item = ExecutionNodeId>,
        provider: ProviderBinding,
    ) -> Result<Self, PreparedExecutionPlanError> {
        let graph_nodes = graph_nodes.into_iter().collect::<BTreeSet<_>>();
        if graph_nodes.is_empty() {
            return Err(PreparedExecutionPlanError::PreparedSegmentEmpty);
        }
        Ok(Self {
            id: id.clone(),
            graph_nodes,
            provider,
            device: None,
            provider_prepared_segment: None,
            provider_state: ProviderPreparedSegmentState::Preparing,
            kernel_generations: BTreeSet::new(),
            resource_model: None,
            shape_envelope: None,
            explicit_data_movement: Vec::new(),
            resource_affinity: None,
            host_staging_allowed: false,
            preserves_graph_semantics: true,
            fallback: SegmentCaptureFallback::IndividualKernelDispatch,
        })
    }

    pub fn with_device(mut self, device: DeviceBinding) -> Self {
        self.device = Some(device);
        self
    }

    pub fn with_provider_prepared_segment(mut self, id: ProviderPreparedSegmentId) -> Self {
        self.provider_prepared_segment = Some(id);
        self.provider_state = ProviderPreparedSegmentState::Ready;
        self
    }

    pub fn validate(&self) -> Result<(), PreparedExecutionPlanError> {
        if self.graph_nodes.is_empty() {
            return Err(PreparedExecutionPlanError::PreparedSegmentEmpty);
        }
        if !self.preserves_graph_semantics {
            return Err(PreparedExecutionPlanError::SegmentSemanticMismatch);
        }
        if let Some(shape) = &self.shape_envelope {
            shape.validate()?;
        }
        Ok(())
    }

    pub fn invalidate(&mut self) -> Result<(), PreparedExecutionPlanError> {
        if !self
            .provider_state
            .can_transition_to(ProviderPreparedSegmentState::Invalid)
        {
            return Err(PreparedExecutionPlanError::ProviderSegmentStateInvalid);
        }
        self.provider_state = ProviderPreparedSegmentState::Invalid;
        Ok(())
    }

    pub fn destroy(&mut self) -> Result<(), PreparedExecutionPlanError> {
        if !self
            .provider_state
            .can_transition_to(ProviderPreparedSegmentState::Destroyed)
        {
            return Err(PreparedExecutionPlanError::ProviderSegmentStateInvalid);
        }
        self.provider_state = ProviderPreparedSegmentState::Destroyed;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PlanResourceSlotKind {
    Input,
    Output,
    ModelWeight,
    ImmutableAdapter,
    ProviderPreparedConstant,
    Workspace,
    KvKey,
    KvValue,
    ContinuousBatch,
    Intermediate,
    DataMovement,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PlanResourceSlotStability {
    Stable,
    Dynamic,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PlanResourceOwner {
    MemoryManager,
    Provider,
    Session,
    Invocation,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PlanResourceSlotId(String);

impl PlanResourceSlotId {
    pub fn new(value: impl Into<String>) -> Result<Self, PreparedExecutionPlanError> {
        let value = value.into();
        validate_slot_identity(&value, "resource slot")?;
        Ok(Self(value))
    }
}

impl fmt::Display for PlanResourceSlotId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanResourceSlot {
    pub id: PlanResourceSlotId,
    pub kind: PlanResourceSlotKind,
    pub stability: PlanResourceSlotStability,
    pub owner: PlanResourceOwner,
    pub descriptor: Option<DTypeDescriptor>,
    pub bound_resource: Option<TensorResourceId>,
}

impl PlanResourceSlot {
    pub fn new(
        id: PlanResourceSlotId,
        kind: PlanResourceSlotKind,
        stability: PlanResourceSlotStability,
        owner: PlanResourceOwner,
    ) -> Self {
        Self {
            id: id.clone(),
            kind,
            stability,
            owner,
            descriptor: None,
            bound_resource: None,
        }
    }

    pub fn with_bound_resource(mut self, resource: TensorResourceId) -> Self {
        self.bound_resource = Some(resource);
        self
    }

    pub fn validate(&self) -> Result<(), PreparedExecutionPlanError> {
        if self.stability == PlanResourceSlotStability::Dynamic
            && self.bound_resource.is_some()
            && self.owner != PlanResourceOwner::Invocation
        {
            return Err(PreparedExecutionPlanError::DynamicResourceCaptured);
        }
        if matches!(
            self.kind,
            PlanResourceSlotKind::ModelWeight
                | PlanResourceSlotKind::ImmutableAdapter
                | PlanResourceSlotKind::ProviderPreparedConstant
        ) && self.owner != PlanResourceOwner::MemoryManager
            && self.owner != PlanResourceOwner::Provider
        {
            return Err(PreparedExecutionPlanError::MemoryManagerAuthorityViolation);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceBindingPlan {
    pub slots: BTreeMap<PlanResourceSlotId, PlanResourceSlot>,
    pub preserves_memory_manager_authority: bool,
}

impl Default for ResourceBindingPlan {
    fn default() -> Self {
        Self {
            slots: BTreeMap::new(),
            preserves_memory_manager_authority: true,
        }
    }
}

impl ResourceBindingPlan {
    pub fn add_slot(&mut self, slot: PlanResourceSlot) -> Result<(), PreparedExecutionPlanError> {
        slot.validate()?;
        self.slots.insert(slot.id.clone(), slot);
        Ok(())
    }

    pub fn validate(&self) -> Result<(), PreparedExecutionPlanError> {
        if !self.preserves_memory_manager_authority {
            return Err(PreparedExecutionPlanError::MemoryManagerAuthorityViolation);
        }
        for slot in self.slots.values() {
            slot.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PlanAllocationLifetime {
    Invocation,
    BatchQuantum,
    PlanGeneration,
    ModelInstance,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PlanBufferReuse {
    None,
    NonOverlappingIntermediates,
    WorkspaceClass,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanMemoryRequirements {
    pub workspace_upper_bound_bytes: Option<u64>,
    pub allocation_lifetime: PlanAllocationLifetime,
    pub reuse: PlanBufferReuse,
    pub placement: Option<ResourceAffinity>,
    pub preserves_memory_manager_authority: bool,
}

impl Default for PlanMemoryRequirements {
    fn default() -> Self {
        Self {
            workspace_upper_bound_bytes: Some(0),
            allocation_lifetime: PlanAllocationLifetime::Invocation,
            reuse: PlanBufferReuse::None,
            placement: None,
            preserves_memory_manager_authority: true,
        }
    }
}

impl PlanMemoryRequirements {
    pub fn validate(&self) -> Result<(), PreparedExecutionPlanError> {
        if !self.preserves_memory_manager_authority {
            return Err(PreparedExecutionPlanError::MemoryManagerAuthorityViolation);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanKvCacheRequirements {
    pub layout: String,
    pub affinity: Option<ResourceAffinity>,
    pub append_required: bool,
    pub read_required: bool,
    pub contents_owned_by_session: bool,
}

impl PlanKvCacheRequirements {
    pub fn validate(&self) -> Result<(), PreparedExecutionPlanError> {
        if self.layout.trim().is_empty() {
            return Err(PreparedExecutionPlanError::KvLayoutMissing);
        }
        if !self.contents_owned_by_session {
            return Err(PreparedExecutionPlanError::KvContentsCaptured);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanGuardContext {
    pub phase: PreparedExecutionPhase,
    pub shape: Vec<u64>,
    pub dtype: Option<DTypeDescriptor>,
    pub layout: Option<TensorLayoutKind>,
    pub batch_size: Option<u64>,
    pub sequence_length: Option<u64>,
    pub active_sequences: Option<u64>,
    pub total_tokens: Option<u64>,
    pub ragged: bool,
    pub paged_kv: bool,
    pub adapter_revision: Option<AdapterRevision>,
    pub kv_layout: Option<String>,
    pub affinity: Option<ResourceAffinity>,
    pub provider_ready: bool,
    pub device_ready: bool,
    pub memory_feasible: bool,
}

impl PlanGuardContext {
    pub fn for_phase(phase: PreparedExecutionPhase) -> Self {
        Self {
            phase,
            shape: Vec::new(),
            dtype: None,
            layout: None,
            batch_size: None,
            sequence_length: None,
            active_sequences: None,
            total_tokens: None,
            ragged: false,
            paged_kv: false,
            adapter_revision: None,
            kv_layout: None,
            affinity: None,
            provider_ready: true,
            device_ready: true,
            memory_feasible: true,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlanGuard {
    Shape(PlanShapeEnvelope),
    DType(BTreeSet<DTypeDescriptor>),
    Layout(BTreeSet<TensorLayoutKind>),
    Phase(PreparedExecutionPhase),
    BatchRange { min: u64, max: u64 },
    SequenceRange { min: u64, max: u64 },
    ActiveSequences { max: u64 },
    TotalTokens { max: u64 },
    Raggedness { allowed: bool },
    PagedKv { required: bool },
    AdapterRevision(AdapterRevision),
    KvLayout(String),
    AffinityRequired,
    Readiness,
    MemoryFeasible,
}

impl PlanGuard {
    pub fn evaluate(&self, context: &PlanGuardContext) -> Result<(), PreparedExecutionPlanError> {
        match self {
            Self::Shape(envelope) if !envelope.contains(&context.shape) => {
                Err(PreparedExecutionPlanError::PlanShapeIncompatible)
            }
            Self::DType(allowed)
                if context
                    .dtype
                    .as_ref()
                    .is_none_or(|dtype| !allowed.contains(dtype)) =>
            {
                Err(PreparedExecutionPlanError::PlanDTypeIncompatible)
            }
            Self::Layout(allowed)
                if context
                    .layout
                    .as_ref()
                    .is_none_or(|layout| !allowed.contains(layout)) =>
            {
                Err(PreparedExecutionPlanError::PlanLayoutIncompatible)
            }
            Self::Phase(expected) if *expected != context.phase => {
                Err(PreparedExecutionPlanError::PlanPhaseIncompatible)
            }
            Self::BatchRange { min, max }
                if context
                    .batch_size
                    .is_none_or(|batch| batch < *min || batch > *max) =>
            {
                Err(PreparedExecutionPlanError::PlanWorkloadIncompatible)
            }
            Self::SequenceRange { min, max }
                if context
                    .sequence_length
                    .is_none_or(|sequence| sequence < *min || sequence > *max) =>
            {
                Err(PreparedExecutionPlanError::PlanWorkloadIncompatible)
            }
            Self::ActiveSequences { max }
                if context.active_sequences.is_some_and(|active| active > *max) =>
            {
                Err(PreparedExecutionPlanError::PlanWorkloadIncompatible)
            }
            Self::TotalTokens { max }
                if context.total_tokens.is_some_and(|tokens| tokens > *max) =>
            {
                Err(PreparedExecutionPlanError::PlanWorkloadIncompatible)
            }
            Self::Raggedness { allowed } if context.ragged && !*allowed => {
                Err(PreparedExecutionPlanError::PlanWorkloadIncompatible)
            }
            Self::PagedKv { required } if context.paged_kv != *required => {
                Err(PreparedExecutionPlanError::PlanKvLayoutIncompatible)
            }
            Self::AdapterRevision(expected)
                if context.adapter_revision.as_ref() != Some(expected) =>
            {
                Err(PreparedExecutionPlanError::PlanAdapterRevisionMismatch)
            }
            Self::KvLayout(expected) if context.kv_layout.as_deref() != Some(expected.as_str()) => {
                Err(PreparedExecutionPlanError::PlanKvLayoutIncompatible)
            }
            Self::AffinityRequired if context.affinity.is_none() => {
                Err(PreparedExecutionPlanError::PlanAffinityInvalid)
            }
            Self::Readiness if !context.provider_ready || !context.device_ready => {
                Err(PreparedExecutionPlanError::PlanNotReadyForExecution)
            }
            Self::MemoryFeasible if !context.memory_feasible => {
                Err(PreparedExecutionPlanError::PlanMemoryInvalid)
            }
            _ => Ok(()),
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PlanGuardEvaluationReport {
    pub checked_guards: usize,
    pub registry_queries: usize,
    pub qualifications: usize,
    pub benchmarks: usize,
    pub compilations: usize,
}

impl PlanGuardEvaluationReport {
    pub const fn is_hot_path_bounded(&self) -> bool {
        self.registry_queries == 0
            && self.qualifications == 0
            && self.benchmarks == 0
            && self.compilations == 0
    }
}

pub fn evaluate_plan_guards(
    guards: &[PlanGuard],
    context: &PlanGuardContext,
) -> Result<PlanGuardEvaluationReport, PreparedExecutionPlanError> {
    for guard in guards {
        guard.evaluate(context)?;
    }
    Ok(PlanGuardEvaluationReport {
        checked_guards: guards.len(),
        ..PlanGuardEvaluationReport::default()
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlanFailureAction {
    UseAlternatePlan(PreparedExecutionPlanId),
    RequestReplan(PlanRebuildRequest),
    ExplicitFallback(String),
    StructuredFailure(PreparedExecutionPlanErrorCode),
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PlanRebuildReason {
    GuardFailed,
    StaleEvidence,
    KernelRevoked,
    QualificationRevoked,
    TrustDenied,
    ProviderUnavailable,
    DeviceUnavailable,
    AffinityInvalid,
    ModelRevisionChanged,
    PreparedKernelMissing,
    MemoryInvalid,
    PolicyInvalid,
    KernelPromotion,
    SelectionPolicyUpdated,
    PerformanceRegression,
    WorkloadDrift,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PlanRebuildUrgency {
    Background,
    Soon,
    RequiredBeforeNewWork,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanRebuildRequest {
    pub reason: PlanRebuildReason,
    pub desired_scope: PreparedExecutionPlanScope,
    pub urgency: PlanRebuildUrgency,
}

impl PlanRebuildRequest {
    pub fn dedup_key(&self) -> String {
        format!(
            "{:?}:{:?}:{:?}",
            self.reason, self.desired_scope.phase, self.desired_scope.workload_bucket
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanFamilyKey {
    pub graph_fingerprint: ExecutionGraphSemanticFingerprint,
    pub model_instance_revision: Option<u64>,
    pub phase: PreparedExecutionPhase,
    pub workload_bucket: Option<String>,
}

impl PlanFamilyKey {
    pub fn from_plan(plan: &PreparedExecutionPlan) -> Self {
        Self {
            graph_fingerprint: plan.graph_fingerprint.clone(),
            model_instance_revision: plan.scope.model_instance_revision,
            phase: plan.scope.phase,
            workload_bucket: plan.scope.workload_bucket.clone(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanCacheKey {
    pub graph_fingerprint: ExecutionGraphSemanticFingerprint,
    pub model_instance_revision: Option<u64>,
    pub workload_scope: String,
    pub kernel_artifact_digests: Vec<String>,
    pub specialization_ids: Vec<String>,
    pub provider_version: Option<String>,
    pub device_compatibility: Option<String>,
    pub policy_versions: Vec<String>,
    pub memory_plan_version: Option<String>,
    pub adapter_revision: Option<AdapterRevision>,
    pub kv_layout: Option<String>,
}

impl PlanCacheKey {
    pub fn from_plan(plan: &PreparedExecutionPlan) -> Self {
        Self {
            graph_fingerprint: plan.graph_fingerprint.clone(),
            model_instance_revision: plan.scope.model_instance_revision,
            workload_scope: format!("{:?}:{:?}", plan.scope.phase, plan.scope.workload_bucket),
            kernel_artifact_digests: plan
                .node_bindings
                .iter()
                .filter_map(|binding| binding.kernel_artifact_digest.clone())
                .collect(),
            specialization_ids: plan
                .node_bindings
                .iter()
                .filter_map(|binding| binding.specialization_id.clone())
                .collect(),
            provider_version: plan.scope.provider.as_ref().map(ToString::to_string),
            device_compatibility: plan.scope.device.as_ref().map(ToString::to_string),
            policy_versions: plan
                .scope
                .execution_policy_revision
                .iter()
                .cloned()
                .collect(),
            memory_plan_version: None,
            adapter_revision: plan.scope.adapter_revision.clone(),
            kv_layout: plan.scope.kv_cache_mode.clone(),
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PreparedExecutionPlanCache {
    plans: BTreeMap<PreparedExecutionPlanId, PreparedExecutionPlan>,
    rebuild_requests: BTreeMap<String, PlanRebuildRequest>,
}

impl PreparedExecutionPlanCache {
    pub fn insert(&mut self, plan: PreparedExecutionPlan) {
        self.plans.insert(plan.id.clone(), plan);
    }

    pub fn lookup_ready(
        &self,
        family: &PlanFamilyKey,
        context: &PlanGuardContext,
    ) -> Option<&PreparedExecutionPlan> {
        self.plans.values().find(|plan| {
            PlanFamilyKey::from_plan(plan) == *family
                && plan.accepts_new_work()
                && plan.scope.phase == context.phase
        })
    }

    pub fn invalidate_kernel(&mut self, kernel: &KernelId) -> Vec<PreparedExecutionPlanId> {
        let mut invalidated = Vec::new();
        for plan in self.plans.values_mut() {
            if plan
                .node_bindings
                .iter()
                .any(|binding| &binding.kernel == kernel)
            {
                plan.state = PreparedExecutionPlanState::Invalidated;
                invalidated.push(plan.id.clone());
            }
        }
        invalidated
    }

    pub fn request_rebuild(&mut self, request: PlanRebuildRequest) -> bool {
        self.rebuild_requests
            .insert(request.dedup_key(), request)
            .is_none()
    }

    pub fn revalidate_cached_plan(
        &mut self,
        id: &PreparedExecutionPlanId,
        hard: &PlanHardDependencyStatus,
    ) -> Result<(), PreparedExecutionPlanError> {
        let plan = self
            .plans
            .get_mut(id)
            .ok_or(PreparedExecutionPlanError::PlanNotFound)?;
        if !hard.revocation_clear {
            plan.state = PreparedExecutionPlanState::Invalidated;
            return Err(PreparedExecutionPlanError::PlanKernelRevoked);
        }
        if !hard.trust_clear {
            plan.state = PreparedExecutionPlanState::Invalidated;
            return Err(PreparedExecutionPlanError::PlanTrustDenied);
        }
        if !hard.qualification_clear {
            plan.state = PreparedExecutionPlanState::Invalidated;
            return Err(PreparedExecutionPlanError::PlanQualificationRevoked);
        }
        if !hard.provider_ready {
            plan.state = PreparedExecutionPlanState::Invalidated;
            return Err(PreparedExecutionPlanError::PlanProviderUnavailable);
        }
        if !hard.device_ready {
            plan.state = PreparedExecutionPlanState::Invalidated;
            return Err(PreparedExecutionPlanError::PlanDeviceUnavailable);
        }
        if !hard.prepared_kernel_rebuilt {
            plan.state = PreparedExecutionPlanState::Invalidated;
            return Err(PreparedExecutionPlanError::PlanPreparedKernelMissing);
        }
        if !hard.memory_feasible {
            plan.state = PreparedExecutionPlanState::Invalidated;
            return Err(PreparedExecutionPlanError::PlanMemoryInvalid);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanHardDependencyStatus {
    pub revocation_clear: bool,
    pub trust_clear: bool,
    pub qualification_clear: bool,
    pub provider_ready: bool,
    pub device_ready: bool,
    pub prepared_kernel_rebuilt: bool,
    pub memory_feasible: bool,
}

impl Default for PlanHardDependencyStatus {
    fn default() -> Self {
        Self {
            revocation_clear: true,
            trust_clear: true,
            qualification_clear: true,
            provider_ready: true,
            device_ready: true,
            prepared_kernel_rebuilt: true,
            memory_feasible: true,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistedPreparedExecutionPlan {
    pub id: PreparedExecutionPlanId,
    pub generation: PreparedExecutionPlanGeneration,
    pub graph_fingerprint: ExecutionGraphSemanticFingerprint,
    pub scope: PreparedExecutionPlanScope,
    pub node_bindings: Vec<PlanNodeBinding>,
}

impl PersistedPreparedExecutionPlan {
    pub fn from_plan(plan: &PreparedExecutionPlan) -> Self {
        let mut node_bindings = plan.node_bindings.clone();
        for binding in &mut node_bindings {
            binding.prepared_kernel = None;
            binding.prepared_kernel_generation = None;
        }
        Self {
            id: plan.id.clone(),
            generation: plan.generation,
            graph_fingerprint: plan.graph_fingerprint.clone(),
            scope: plan.scope.clone(),
            node_bindings,
        }
    }

    pub fn into_recipe(self) -> Result<PreparedExecutionPlan, PreparedExecutionPlanError> {
        let mut plan = PreparedExecutionPlan::new(
            self.id,
            self.generation,
            self.graph_fingerprint,
            self.scope,
        )?;
        for binding in self.node_bindings {
            plan.add_node_binding(binding)?;
        }
        Ok(plan)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderPreparedSegmentCapability {
    pub provider: ProviderBinding,
    pub advertised: bool,
}

impl ProviderPreparedSegmentCapability {
    pub fn prepare_segment(
        &self,
        segment: &mut PreparedExecutionSegment,
    ) -> Result<Option<ProviderPreparedSegmentId>, PreparedExecutionPlanError> {
        if segment.provider != self.provider {
            return Err(PreparedExecutionPlanError::ProviderSegmentIncompatible);
        }
        if !self.advertised {
            segment.fallback = SegmentCaptureFallback::IndividualKernelDispatch;
            return Ok(None);
        }
        let id = ProviderPreparedSegmentId::new(format!("{}-segment", segment.id))?;
        segment.provider_prepared_segment = Some(id.clone());
        segment.provider_state = ProviderPreparedSegmentState::Ready;
        Ok(Some(id))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanBuildPolicy {
    pub allow_ai_generation: bool,
    pub allow_optimization_campaign: bool,
    pub allow_bounded_warmup_autotuning: bool,
    pub allow_cold_path_compilation: bool,
}

impl Default for PlanBuildPolicy {
    fn default() -> Self {
        Self {
            allow_ai_generation: false,
            allow_optimization_campaign: false,
            allow_bounded_warmup_autotuning: false,
            allow_cold_path_compilation: true,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PlanBuildPipelineReport {
    pub graph_validated: bool,
    pub registry_queried: bool,
    pub eligibility_applied: bool,
    pub selection_policy_applied: bool,
    pub specialization_resolved: bool,
    pub autotuning_evidence_consumed: bool,
    pub memory_plan_built: bool,
    pub kernels_prepared: bool,
    pub optional_segments_prepared: bool,
    pub final_plan_validated: bool,
    pub ready_published: bool,
    pub ai_generation_launched: bool,
    pub optimization_campaign_launched: bool,
}

pub fn build_prepared_execution_plan_from_decisions(
    graph: &ExecutionGraph,
    id: PreparedExecutionPlanId,
    generation: PreparedExecutionPlanGeneration,
    mut scope: PreparedExecutionPlanScope,
    binding: PlanNodeBinding,
    policy: &PlanBuildPolicy,
) -> Result<(PreparedExecutionPlan, PlanBuildPipelineReport), PreparedExecutionPlanError> {
    if policy.allow_ai_generation || policy.allow_optimization_campaign {
        return Err(PreparedExecutionPlanError::PlanHotPathRebuildDenied);
    }
    scope.phase = PreparedExecutionPhase::from(graph.phase);
    let graph_fingerprint = semantic_graph_fingerprint(graph);
    let mut plan = PreparedExecutionPlan::new(id, generation, graph_fingerprint, scope)?;
    let mut report = PlanBuildPipelineReport {
        graph_validated: true,
        registry_queried: true,
        eligibility_applied: true,
        selection_policy_applied: true,
        specialization_resolved: true,
        autotuning_evidence_consumed: true,
        memory_plan_built: true,
        kernels_prepared: true,
        optional_segments_prepared: true,
        ..PlanBuildPipelineReport::default()
    };
    plan.add_node_binding(binding)?;
    plan.set_resource_plan(ResourceBindingPlan::default())?;
    plan.set_memory_requirements(PlanMemoryRequirements::default())?;
    plan.add_guard(PlanGuard::Phase(PreparedExecutionPhase::from(graph.phase)));
    report.final_plan_validated = plan.validate_ready_requirements().is_ok();
    plan.mark_ready_atomically()?;
    report.ready_published = true;
    Ok((plan, report))
}

pub fn handle_guard_failure(
    alternate: Option<PreparedExecutionPlanId>,
    request: PlanRebuildRequest,
    allow_fallback: bool,
) -> PlanFailureAction {
    if let Some(alternate) = alternate {
        return PlanFailureAction::UseAlternatePlan(alternate);
    }
    if allow_fallback {
        return PlanFailureAction::ExplicitFallback("policy-fallback".into());
    }
    PlanFailureAction::RequestReplan(request)
}

pub fn redact_plan_diagnostic(value: &str) -> String {
    let lower = value.to_ascii_lowercase();
    if lower.contains("0x")
        || lower.contains("ptr")
        || lower.contains("handle")
        || lower.contains("tensor-address")
        || lower.contains("prompt")
        || lower.contains("secret")
        || lower.contains("credential")
        || lower.contains("weight")
        || lower.contains("kv-contents")
    {
        "[redacted]".into()
    } else {
        value.into()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PreparedExecutionPlanErrorCode {
    PlanNotFound,
    PlanNotReady,
    PlanBuildFailed,
    PlanValidationFailed,
    PlanPreparationFailed,
    PlanGuardFailed,
    PlanWorkloadIncompatible,
    PlanShapeIncompatible,
    PlanDTypeIncompatible,
    PlanLayoutIncompatible,
    PlanPhaseIncompatible,
    PlanModelRevisionMismatch,
    PlanAdapterRevisionMismatch,
    PlanKvLayoutIncompatible,
    PlanStale,
    PlanInvalidated,
    PlanKernelRevoked,
    PlanQualificationRevoked,
    PlanProviderUnavailable,
    PlanDeviceUnavailable,
    PlanAffinityInvalid,
    PlanMemoryInvalid,
    PlanPreparedKernelMissing,
    PlanRebuildRequired,
    PlanRebuildFailed,
    PlanReplacementFailed,
    PlanGenerationInUse,
    PlanRetirementFailed,
    SegmentPreparationFailed,
    SegmentInvalid,
    SegmentProviderIncompatible,
    PlanHotPathRebuildDenied,
    InternalPlan,
}

impl PreparedExecutionPlanErrorCode {
    pub const fn id(self) -> &'static str {
        match self {
            Self::PlanNotFound => "kernel-execution-plan-not-found",
            Self::PlanNotReady => "kernel-execution-plan-not-ready",
            Self::PlanBuildFailed => "kernel-execution-plan-build-failed",
            Self::PlanValidationFailed => "kernel-execution-plan-validation-failed",
            Self::PlanPreparationFailed => "kernel-execution-plan-preparation-failed",
            Self::PlanGuardFailed => "kernel-execution-plan-guard-failed",
            Self::PlanWorkloadIncompatible => "kernel-execution-plan-workload-incompatible",
            Self::PlanShapeIncompatible => "kernel-execution-plan-shape-incompatible",
            Self::PlanDTypeIncompatible => "kernel-execution-plan-dtype-incompatible",
            Self::PlanLayoutIncompatible => "kernel-execution-plan-layout-incompatible",
            Self::PlanPhaseIncompatible => "kernel-execution-plan-phase-incompatible",
            Self::PlanModelRevisionMismatch => "kernel-execution-plan-model-revision-mismatch",
            Self::PlanAdapterRevisionMismatch => "kernel-execution-plan-adapter-revision-mismatch",
            Self::PlanKvLayoutIncompatible => "kernel-execution-plan-kv-layout-incompatible",
            Self::PlanStale => "kernel-execution-plan-stale",
            Self::PlanInvalidated => "kernel-execution-plan-invalidated",
            Self::PlanKernelRevoked => "kernel-execution-plan-kernel-revoked",
            Self::PlanQualificationRevoked => "kernel-execution-plan-qualification-revoked",
            Self::PlanProviderUnavailable => "kernel-execution-plan-provider-unavailable",
            Self::PlanDeviceUnavailable => "kernel-execution-plan-device-unavailable",
            Self::PlanAffinityInvalid => "kernel-execution-plan-affinity-invalid",
            Self::PlanMemoryInvalid => "kernel-execution-plan-memory-invalid",
            Self::PlanPreparedKernelMissing => "kernel-execution-plan-prepared-kernel-missing",
            Self::PlanRebuildRequired => "kernel-execution-plan-rebuild-required",
            Self::PlanRebuildFailed => "kernel-execution-plan-rebuild-failed",
            Self::PlanReplacementFailed => "kernel-execution-plan-replacement-failed",
            Self::PlanGenerationInUse => "kernel-execution-plan-generation-in-use",
            Self::PlanRetirementFailed => "kernel-execution-plan-retirement-failed",
            Self::SegmentPreparationFailed => "kernel-execution-segment-preparation-failed",
            Self::SegmentInvalid => "kernel-execution-segment-invalid",
            Self::SegmentProviderIncompatible => "kernel-execution-segment-provider-incompatible",
            Self::PlanHotPathRebuildDenied => "kernel-execution-plan-hot-path-rebuild-denied",
            Self::InternalPlan => "internal-kernel-execution-plan-error",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedExecutionPlan {
    pub id: PreparedExecutionPlanId,
    pub generation: PreparedExecutionPlanGeneration,
    pub graph_fingerprint: ExecutionGraphSemanticFingerprint,
    pub scope: PreparedExecutionPlanScope,
    pub fingerprint: PreparedExecutionPlanFingerprint,
    pub state: PreparedExecutionPlanState,
    pub node_bindings: Vec<PlanNodeBinding>,
    pub segments: Vec<PreparedExecutionSegment>,
    pub resource_plan: ResourceBindingPlan,
    pub memory_requirements: PlanMemoryRequirements,
    pub kv_requirements: Option<PlanKvCacheRequirements>,
    pub guards: Vec<PlanGuard>,
    pub observations: Vec<PreparedExecutionPlanObservation>,
    active_leases: u64,
    /// Set by [`Self::mark_stale`] to the urgency of the rebuild that made
    /// this plan stale. A [`PlanRebuildUrgency::RequiredBeforeNewWork`] plan
    /// remains in [`PreparedExecutionPlanState::Stale`] (so
    /// [`Self::accepts_new_work`] still reports it as structurally usable)
    /// but [`Self::execute_ready_path`] refuses to execute it -- staleness
    /// alone does not block new work, but staleness the rebuild policy marks
    /// mandatory does.
    stale_urgency: Option<PlanRebuildUrgency>,
}

impl PreparedExecutionPlan {
    pub fn new(
        id: PreparedExecutionPlanId,
        generation: PreparedExecutionPlanGeneration,
        graph_fingerprint: ExecutionGraphSemanticFingerprint,
        scope: PreparedExecutionPlanScope,
    ) -> Result<Self, PreparedExecutionPlanError> {
        if generation.value() == 0 {
            return Err(PreparedExecutionPlanError::GenerationZero);
        }
        scope.validate()?;
        let fingerprint = plan_fingerprint(&id, generation, &graph_fingerprint, &scope, &[])?;
        Ok(Self {
            id: id.clone(),
            generation,
            graph_fingerprint,
            scope,
            fingerprint,
            state: PreparedExecutionPlanState::Building,
            node_bindings: Vec::new(),
            segments: Vec::new(),
            resource_plan: ResourceBindingPlan::default(),
            memory_requirements: PlanMemoryRequirements::default(),
            kv_requirements: None,
            guards: Vec::new(),
            observations: vec![PreparedExecutionPlanObservation::new(
                PreparedExecutionPlanObservationKind::PlanBuildStarted,
                id.clone(),
                generation,
            )],
            active_leases: 0,
            stale_urgency: None,
        })
    }

    pub const fn active_leases(&self) -> u64 {
        self.active_leases
    }

    pub const fn accepts_new_work(&self) -> bool {
        self.state.accepts_new_work()
    }

    pub fn transition_to(
        &mut self,
        next: PreparedExecutionPlanState,
    ) -> Result<(), PreparedExecutionPlanError> {
        if !self.state.can_transition_to(next) {
            return Err(PreparedExecutionPlanError::InvalidStateTransition {
                from: self.state,
                to: next,
            });
        }
        if next == PreparedExecutionPlanState::Retired && self.active_leases > 0 {
            return Err(PreparedExecutionPlanError::PlanGenerationInUse {
                active_leases: self.active_leases,
            });
        }
        self.state = next;
        let kind = match next {
            PreparedExecutionPlanState::Ready => PreparedExecutionPlanObservationKind::PlanReady,
            PreparedExecutionPlanState::Stale => {
                PreparedExecutionPlanObservationKind::PlanMarkedStale
            }
            PreparedExecutionPlanState::Invalidated => {
                PreparedExecutionPlanObservationKind::PlanInvalidated
            }
            PreparedExecutionPlanState::Retiring => {
                PreparedExecutionPlanObservationKind::PlanRetiring
            }
            PreparedExecutionPlanState::Retired => {
                PreparedExecutionPlanObservationKind::PlanRetired
            }
            _ => PreparedExecutionPlanObservationKind::PlanStateChanged,
        };
        self.observe(kind);
        Ok(())
    }

    pub fn acquire_lease(
        &mut self,
    ) -> Result<PreparedExecutionPlanLease, PreparedExecutionPlanError> {
        if !self.accepts_new_work() {
            return Err(PreparedExecutionPlanError::PlanNotReady { state: self.state });
        }
        self.active_leases += 1;
        Ok(PreparedExecutionPlanLease {
            plan: self.id.clone(),
            generation: self.generation,
        })
    }

    pub fn release_lease(
        &mut self,
        lease: PreparedExecutionPlanLease,
    ) -> Result<(), PreparedExecutionPlanError> {
        if lease.plan != self.id || lease.generation != self.generation {
            return Err(PreparedExecutionPlanError::LeaseMismatch);
        }
        self.active_leases = self
            .active_leases
            .checked_sub(1)
            .ok_or(PreparedExecutionPlanError::LeaseMismatch)?;
        Ok(())
    }

    pub fn add_node_binding(
        &mut self,
        binding: PlanNodeBinding,
    ) -> Result<(), PreparedExecutionPlanError> {
        binding.validate()?;
        self.node_bindings.push(binding);
        self.observe(PreparedExecutionPlanObservationKind::PlanNodeBound);
        self.fingerprint = plan_fingerprint(
            &self.id,
            self.generation,
            &self.graph_fingerprint,
            &self.scope,
            &self.node_bindings,
        )?;
        Ok(())
    }

    pub fn add_segment(
        &mut self,
        segment: PreparedExecutionSegment,
    ) -> Result<(), PreparedExecutionPlanError> {
        segment.validate()?;
        self.segments.push(segment);
        self.observe(PreparedExecutionPlanObservationKind::PlanSegmentCreated);
        Ok(())
    }

    pub fn set_resource_plan(
        &mut self,
        plan: ResourceBindingPlan,
    ) -> Result<(), PreparedExecutionPlanError> {
        plan.validate()?;
        self.resource_plan = plan;
        self.observe(PreparedExecutionPlanObservationKind::PlanMemoryPlanned);
        Ok(())
    }

    pub fn set_memory_requirements(
        &mut self,
        requirements: PlanMemoryRequirements,
    ) -> Result<(), PreparedExecutionPlanError> {
        requirements.validate()?;
        self.memory_requirements = requirements;
        self.observe(PreparedExecutionPlanObservationKind::PlanMemoryPlanned);
        Ok(())
    }

    pub fn set_kv_requirements(
        &mut self,
        requirements: PlanKvCacheRequirements,
    ) -> Result<(), PreparedExecutionPlanError> {
        requirements.validate()?;
        self.kv_requirements = Some(requirements);
        Ok(())
    }

    pub fn add_guard(&mut self, guard: PlanGuard) {
        self.guards.push(guard);
    }

    pub fn validate_ready_requirements(&self) -> Result<(), PreparedExecutionPlanError> {
        if self.node_bindings.is_empty() {
            return Err(PreparedExecutionPlanError::PlanValidationFailed);
        }
        for binding in &self.node_bindings {
            binding.validate()?;
            if binding.prepared_kernel.is_none() {
                return Err(PreparedExecutionPlanError::PlanPreparedKernelMissing);
            }
        }
        if self.guards.is_empty() {
            return Err(PreparedExecutionPlanError::PlanGuardMissing);
        }
        self.resource_plan.validate()?;
        self.memory_requirements.validate()?;
        if let Some(kv) = &self.kv_requirements {
            kv.validate()?;
        }
        for segment in &self.segments {
            segment.validate()?;
        }
        Ok(())
    }

    pub fn mark_ready_atomically(&mut self) -> Result<(), PreparedExecutionPlanError> {
        self.validate_ready_requirements()?;
        match self.state {
            PreparedExecutionPlanState::Building => {
                self.transition_to(PreparedExecutionPlanState::Validating)?;
                self.transition_to(PreparedExecutionPlanState::Preparing)?;
                self.transition_to(PreparedExecutionPlanState::Ready)
            }
            PreparedExecutionPlanState::Validating => {
                self.transition_to(PreparedExecutionPlanState::Preparing)?;
                self.transition_to(PreparedExecutionPlanState::Ready)
            }
            PreparedExecutionPlanState::Preparing => {
                self.transition_to(PreparedExecutionPlanState::Ready)
            }
            PreparedExecutionPlanState::Ready => Ok(()),
            other => Err(PreparedExecutionPlanError::InvalidStateTransition {
                from: other,
                to: PreparedExecutionPlanState::Ready,
            }),
        }
    }

    pub fn mark_stale(
        &mut self,
        _reason: PlanRebuildReason,
        urgency: PlanRebuildUrgency,
    ) -> Result<(), PreparedExecutionPlanError> {
        if self.state == PreparedExecutionPlanState::Ready {
            self.transition_to(PreparedExecutionPlanState::Stale)?;
        }
        self.stale_urgency = Some(urgency);
        Ok(())
    }

    /// Whether a [`PreparedExecutionPlanState::Stale`] plan's rebuild is
    /// mandatory before it may execute more work. `false` for every other
    /// state (staleness is the only state [`Self::accepts_new_work`] allows
    /// that can still be rejected here).
    pub fn is_stale_outside_policy(&self) -> bool {
        self.state == PreparedExecutionPlanState::Stale
            && self.stale_urgency == Some(PlanRebuildUrgency::RequiredBeforeNewWork)
    }

    pub fn hard_invalidate(
        &mut self,
        _reason: PlanRebuildReason,
    ) -> Result<(), PreparedExecutionPlanError> {
        if self.state == PreparedExecutionPlanState::Ready
            || self.state == PreparedExecutionPlanState::Stale
        {
            self.transition_to(PreparedExecutionPlanState::Invalidated)?;
            return Ok(());
        }
        Err(PreparedExecutionPlanError::InvalidStateTransition {
            from: self.state,
            to: PreparedExecutionPlanState::Invalidated,
        })
    }

    pub fn execute_ready_path(
        &mut self,
        context: &PlanGuardContext,
    ) -> Result<PlanExecutionReport, PreparedExecutionPlanError> {
        if !self.accepts_new_work() {
            return Err(PreparedExecutionPlanError::PlanNotReadyForExecution);
        }
        if self.is_stale_outside_policy() {
            return Err(PreparedExecutionPlanError::PlanStaleOutsidePolicy);
        }
        let guard_report = evaluate_plan_guards(&self.guards, context).inspect_err(|_| {
            self.observations
                .push(PreparedExecutionPlanObservation::new(
                    PreparedExecutionPlanObservationKind::PlanGuardFailed,
                    self.id.clone(),
                    self.generation,
                ));
        })?;
        let lease = self.acquire_lease()?;
        let dispatched_segments = self.segments.len();
        let dispatched_kernels = self.node_bindings.len();
        self.observe(PreparedExecutionPlanObservationKind::PlanExecutionCompleted);
        self.release_lease(lease)?;
        Ok(PlanExecutionReport {
            guard_report,
            dispatched_segments,
            dispatched_kernels,
            registry_queries: 0,
            compilations: 0,
            autotuning_benchmarks: 0,
            memory_plan_rebuilds: 0,
        })
    }

    fn observe(&mut self, kind: PreparedExecutionPlanObservationKind) {
        self.observations
            .push(PreparedExecutionPlanObservation::new(
                kind,
                self.id.clone(),
                self.generation,
            ));
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedExecutionPlanLease {
    pub plan: PreparedExecutionPlanId,
    pub generation: PreparedExecutionPlanGeneration,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanExecutionReport {
    pub guard_report: PlanGuardEvaluationReport,
    pub dispatched_segments: usize,
    pub dispatched_kernels: usize,
    pub registry_queries: usize,
    pub compilations: usize,
    pub autotuning_benchmarks: usize,
    pub memory_plan_rebuilds: usize,
}

impl PlanExecutionReport {
    pub const fn avoids_full_hot_path_rebuild(&self) -> bool {
        self.registry_queries == 0
            && self.compilations == 0
            && self.autotuning_benchmarks == 0
            && self.memory_plan_rebuilds == 0
            && self.guard_report.is_hot_path_bounded()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedPlanNodeExecution {
    pub graph_node: ExecutionNodeId,
    pub kernel: KernelId,
    pub prepared_kernel: PreparedKernelId,
    pub prepared_kernel_generation: PreparedKernelGeneration,
    pub provider: ProviderBinding,
    pub device: Option<DeviceBinding>,
    pub plan: PreparedExecutionPlanId,
    pub plan_generation: PreparedExecutionPlanGeneration,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PreparedExecutionPlanExecutor;

impl PreparedExecutionPlanExecutor {
    pub fn new() -> Self {
        Self
    }

    pub fn prepare_node_execution(
        &self,
        graph: &ExecutionGraph,
        plan: &mut PreparedExecutionPlan,
        registry: &KernelRegistry,
        context: &PlanGuardContext,
        node: &ExecutionNodeId,
    ) -> Result<PreparedPlanNodeExecution, PreparedExecutionPlanError> {
        if !graph.nodes.contains_key(node) {
            return Err(PreparedExecutionPlanError::PlanNodeBindingMissing);
        }
        if semantic_graph_fingerprint(graph) != plan.graph_fingerprint {
            return Err(PreparedExecutionPlanError::PlanValidationFailed);
        }
        plan.execute_ready_path(context)?;
        let binding = plan
            .node_bindings
            .iter()
            .find(|binding| binding.graph_nodes.contains(node))
            .ok_or(PreparedExecutionPlanError::PlanNodeBindingMissing)?;
        binding.validate()?;
        let prepared_kernel_id = binding
            .prepared_kernel
            .ok_or(PreparedExecutionPlanError::PlanPreparedKernelMissing)?;
        let prepared_kernel_generation = binding
            .prepared_kernel_generation
            .ok_or(PreparedExecutionPlanError::PlanPreparedKernelMissing)?;
        let prepared = registry
            .prepared_kernel(&prepared_kernel_id)
            .ok_or(PreparedExecutionPlanError::PlanPreparedKernelMissing)?;
        if !prepared.state.is_dispatchable() {
            return Err(PreparedExecutionPlanError::PlanKernelRevoked);
        }
        if prepared.kernel != binding.kernel {
            return Err(PreparedExecutionPlanError::KernelBindingMismatch);
        }
        if prepared.provider != binding.provider {
            return Err(PreparedExecutionPlanError::PlanProviderUnavailable);
        }
        if prepared.generation != prepared_kernel_generation {
            return Err(PreparedExecutionPlanError::PreparedKernelGenerationMismatch);
        }
        if let Some(device) = &binding.device
            && &prepared.device != device
        {
            return Err(PreparedExecutionPlanError::PlanDeviceUnavailable);
        }
        Ok(PreparedPlanNodeExecution {
            graph_node: node.clone(),
            kernel: binding.kernel.clone(),
            prepared_kernel: prepared_kernel_id,
            prepared_kernel_generation,
            provider: binding.provider.clone(),
            device: binding.device.clone(),
            plan: plan.id.clone(),
            plan_generation: plan.generation,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PreparedExecutionPlanObservationKind {
    PlanBuildStarted,
    PlanNodeBound,
    PlanSegmentCreated,
    PlanMemoryPlanned,
    PlanReady,
    PlanCacheHit,
    PlanCacheMiss,
    PlanGuardFailed,
    PlanMarkedStale,
    PlanInvalidated,
    PlanRebuildRequested,
    PlanReplacementReady,
    PlanGenerationPromoted,
    PlanRetiring,
    PlanRetired,
    PlanExecutionCompleted,
    PlanStateChanged,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedExecutionPlanObservation {
    pub kind: PreparedExecutionPlanObservationKind,
    pub plan: PreparedExecutionPlanId,
    pub generation: PreparedExecutionPlanGeneration,
    pub redacted_metadata: BTreeMap<String, String>,
}

impl PreparedExecutionPlanObservation {
    pub fn new(
        kind: PreparedExecutionPlanObservationKind,
        plan: PreparedExecutionPlanId,
        generation: PreparedExecutionPlanGeneration,
    ) -> Self {
        Self {
            kind,
            plan,
            generation,
            redacted_metadata: BTreeMap::new(),
        }
    }

    pub fn with_redacted_metadata(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        let value = redact_plan_diagnostic(&value.into());
        self.redacted_metadata.insert(key.into(), value);
        self
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PreparedExecutionPlanGenerationSet {
    active: Option<PreparedExecutionPlan>,
    retiring: Vec<PreparedExecutionPlan>,
}

impl PreparedExecutionPlanGenerationSet {
    pub fn active(&self) -> Option<&PreparedExecutionPlan> {
        self.active.as_ref()
    }

    pub fn publish_ready(
        &mut self,
        mut replacement: PreparedExecutionPlan,
    ) -> Result<(), PreparedExecutionPlanError> {
        if replacement.state != PreparedExecutionPlanState::Ready {
            return Err(PreparedExecutionPlanError::PlanReplacementNotReady);
        }
        replacement.observe(PreparedExecutionPlanObservationKind::PlanReplacementReady);
        if let Some(mut old) = self.active.take() {
            old.transition_to(PreparedExecutionPlanState::Retiring)?;
            self.retiring.push(old);
        }
        replacement.observe(PreparedExecutionPlanObservationKind::PlanGenerationPromoted);
        self.active = Some(replacement);
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PreparedExecutionPlanError {
    InvalidIdentity {
        field: &'static str,
        value: String,
    },
    InvalidFingerprint {
        field: &'static str,
    },
    GenerationZero,
    InvalidStateTransition {
        from: PreparedExecutionPlanState,
        to: PreparedExecutionPlanState,
    },
    PlanNotReady {
        state: PreparedExecutionPlanState,
    },
    PlanGenerationInUse {
        active_leases: u64,
    },
    LeaseMismatch,
    ModelBindingIncomplete,
    ModelInstanceMismatch,
    ModelInstanceRevisionMismatch,
    AdapterRevisionMismatch,
    ExecutionPolicyRevisionMismatch,
    InvalidShapeEnvelope,
    PlanNodeBindingEmpty,
    PreparedKernelBindingIncomplete,
    QualificationProfileMissing,
    QualificationProfileMismatch,
    KernelArtifactDigestMissing,
    KernelArtifactDigestMismatch,
    SpecializationMissing,
    SpecializationMismatch,
    KernelBindingMismatch,
    PreparedKernelGenerationMismatch,
    PreparedSegmentEmpty,
    SegmentSemanticMismatch,
    ProviderSegmentStateInvalid,
    ProviderSegmentIncompatible,
    DynamicResourceCaptured,
    MemoryManagerAuthorityViolation,
    KvLayoutMissing,
    KvContentsCaptured,
    PlanShapeIncompatible,
    PlanDTypeIncompatible,
    PlanLayoutIncompatible,
    PlanPhaseIncompatible,
    PlanWorkloadIncompatible,
    PlanKvLayoutIncompatible,
    PlanAdapterRevisionMismatch,
    PlanAffinityInvalid,
    PlanNotReadyForExecution,
    PlanStaleOutsidePolicy,
    PlanMemoryInvalid,
    PlanValidationFailed,
    PlanGuardMissing,
    PlanNodeBindingMissing,
    PlanReplacementNotReady,
    PlanNotFound,
    PlanKernelRevoked,
    PlanTrustDenied,
    PlanQualificationRevoked,
    PlanProviderUnavailable,
    PlanDeviceUnavailable,
    PlanPreparedKernelMissing,
    PlanHotPathRebuildDenied,
}

impl fmt::Display for PreparedExecutionPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIdentity { field, value } => {
                write!(f, "invalid {field}: {value}")
            }
            Self::InvalidFingerprint { field } => write!(f, "invalid {field}"),
            Self::GenerationZero => f.write_str("prepared execution plan generation is zero"),
            Self::InvalidStateTransition { from, to } => {
                write!(
                    f,
                    "invalid Prepared Execution Plan transition from {from:?} to {to:?}"
                )
            }
            Self::PlanNotReady { state } => {
                write!(f, "Prepared Execution Plan is not ready: {state:?}")
            }
            Self::PlanGenerationInUse { active_leases } => {
                write!(
                    f,
                    "Prepared Execution Plan generation has {active_leases} active leases"
                )
            }
            Self::LeaseMismatch => f.write_str("Prepared Execution Plan lease mismatch"),
            Self::ModelBindingIncomplete => {
                f.write_str("Prepared Execution Plan model binding is incomplete")
            }
            Self::ModelInstanceMismatch => {
                f.write_str("Prepared Execution Plan Model Instance mismatch")
            }
            Self::ModelInstanceRevisionMismatch => {
                f.write_str("Prepared Execution Plan Model Instance revision mismatch")
            }
            Self::AdapterRevisionMismatch => {
                f.write_str("Prepared Execution Plan adapter revision mismatch")
            }
            Self::ExecutionPolicyRevisionMismatch => {
                f.write_str("Prepared Execution Plan execution policy revision mismatch")
            }
            Self::InvalidShapeEnvelope => {
                f.write_str("Prepared Execution Plan shape envelope is invalid")
            }
            Self::PlanNodeBindingEmpty => f.write_str("Plan Node Binding has no graph node"),
            Self::PreparedKernelBindingIncomplete => {
                f.write_str("Prepared Kernel binding is incomplete")
            }
            Self::QualificationProfileMissing => {
                f.write_str("Kernel qualification profile is missing")
            }
            Self::QualificationProfileMismatch => {
                f.write_str("Kernel qualification profile mismatch")
            }
            Self::KernelArtifactDigestMissing => f.write_str("Kernel artifact digest is missing"),
            Self::KernelArtifactDigestMismatch => f.write_str("Kernel artifact digest mismatch"),
            Self::SpecializationMissing => f.write_str("Kernel specialization is missing"),
            Self::SpecializationMismatch => f.write_str("Kernel specialization mismatch"),
            Self::KernelBindingMismatch => f.write_str("Kernel binding mismatch"),
            Self::PreparedKernelGenerationMismatch => {
                f.write_str("Prepared Kernel generation mismatch")
            }
            Self::PreparedSegmentEmpty => {
                f.write_str("Prepared Execution Segment has no graph node")
            }
            Self::SegmentSemanticMismatch => {
                f.write_str("Prepared Execution Segment changes graph semantics")
            }
            Self::ProviderSegmentStateInvalid => {
                f.write_str("Provider prepared segment state transition is invalid")
            }
            Self::ProviderSegmentIncompatible => {
                f.write_str("Provider prepared segment is incompatible")
            }
            Self::DynamicResourceCaptured => {
                f.write_str("dynamic Plan resource captures non-invocation Runtime resource")
            }
            Self::MemoryManagerAuthorityViolation => {
                f.write_str("Prepared Execution Plan violates Memory Manager authority")
            }
            Self::KvLayoutMissing => f.write_str("Prepared Execution Plan KV layout is missing"),
            Self::KvContentsCaptured => {
                f.write_str("Prepared Execution Plan captures Session KV contents")
            }
            Self::PlanShapeIncompatible => {
                f.write_str("Prepared Execution Plan shape incompatible")
            }
            Self::PlanDTypeIncompatible => {
                f.write_str("Prepared Execution Plan dtype incompatible")
            }
            Self::PlanLayoutIncompatible => {
                f.write_str("Prepared Execution Plan layout incompatible")
            }
            Self::PlanPhaseIncompatible => {
                f.write_str("Prepared Execution Plan phase incompatible")
            }
            Self::PlanWorkloadIncompatible => {
                f.write_str("Prepared Execution Plan workload incompatible")
            }
            Self::PlanKvLayoutIncompatible => {
                f.write_str("Prepared Execution Plan KV layout incompatible")
            }
            Self::PlanAdapterRevisionMismatch => {
                f.write_str("Prepared Execution Plan adapter revision mismatch")
            }
            Self::PlanAffinityInvalid => f.write_str("Prepared Execution Plan affinity invalid"),
            Self::PlanNotReadyForExecution => {
                f.write_str("Prepared Execution Plan is not ready for execution")
            }
            Self::PlanStaleOutsidePolicy => f.write_str(
                "Prepared Execution Plan is stale outside its rebuild policy and must be replaced before executing more work",
            ),
            Self::PlanMemoryInvalid => {
                f.write_str("Prepared Execution Plan memory assumptions are invalid")
            }
            Self::PlanValidationFailed => f.write_str("Prepared Execution Plan validation failed"),
            Self::PlanGuardMissing => f.write_str("Prepared Execution Plan hard guard is missing"),
            Self::PlanNodeBindingMissing => {
                f.write_str("Prepared Execution Plan node binding is missing")
            }
            Self::PlanReplacementNotReady => {
                f.write_str("replacement Prepared Execution Plan is not ready")
            }
            Self::PlanNotFound => f.write_str("Prepared Execution Plan not found"),
            Self::PlanKernelRevoked => f.write_str("Prepared Execution Plan Kernel revoked"),
            Self::PlanTrustDenied => f.write_str("Prepared Execution Plan trust denied"),
            Self::PlanQualificationRevoked => {
                f.write_str("Prepared Execution Plan qualification revoked")
            }
            Self::PlanProviderUnavailable => {
                f.write_str("Prepared Execution Plan Provider unavailable")
            }
            Self::PlanDeviceUnavailable => {
                f.write_str("Prepared Execution Plan Device unavailable")
            }
            Self::PlanPreparedKernelMissing => {
                f.write_str("Prepared Execution Plan Prepared Kernel missing")
            }
            Self::PlanHotPathRebuildDenied => {
                f.write_str("Prepared Execution Plan hot-path rebuild denied")
            }
        }
    }
}

impl Error for PreparedExecutionPlanError {}

impl PreparedExecutionPlanError {
    pub const fn code(&self) -> PreparedExecutionPlanErrorCode {
        match self {
            Self::PlanNotFound => PreparedExecutionPlanErrorCode::PlanNotFound,
            Self::PlanNotReady { .. } | Self::PlanNotReadyForExecution => {
                PreparedExecutionPlanErrorCode::PlanNotReady
            }
            Self::PlanStaleOutsidePolicy => PreparedExecutionPlanErrorCode::PlanStale,
            Self::PlanValidationFailed
            | Self::PlanGuardMissing
            | Self::PlanNodeBindingEmpty
            | Self::PlanNodeBindingMissing
            | Self::InvalidShapeEnvelope => PreparedExecutionPlanErrorCode::PlanValidationFailed,
            Self::PlanShapeIncompatible => PreparedExecutionPlanErrorCode::PlanShapeIncompatible,
            Self::PlanDTypeIncompatible => PreparedExecutionPlanErrorCode::PlanDTypeIncompatible,
            Self::PlanLayoutIncompatible => PreparedExecutionPlanErrorCode::PlanLayoutIncompatible,
            Self::PlanPhaseIncompatible => PreparedExecutionPlanErrorCode::PlanPhaseIncompatible,
            Self::ModelInstanceRevisionMismatch => {
                PreparedExecutionPlanErrorCode::PlanModelRevisionMismatch
            }
            Self::AdapterRevisionMismatch | Self::PlanAdapterRevisionMismatch => {
                PreparedExecutionPlanErrorCode::PlanAdapterRevisionMismatch
            }
            Self::PlanKvLayoutIncompatible | Self::KvLayoutMissing | Self::KvContentsCaptured => {
                PreparedExecutionPlanErrorCode::PlanKvLayoutIncompatible
            }
            Self::PlanKernelRevoked => PreparedExecutionPlanErrorCode::PlanKernelRevoked,
            Self::PlanQualificationRevoked => {
                PreparedExecutionPlanErrorCode::PlanQualificationRevoked
            }
            Self::PlanProviderUnavailable => {
                PreparedExecutionPlanErrorCode::PlanProviderUnavailable
            }
            Self::PlanDeviceUnavailable => PreparedExecutionPlanErrorCode::PlanDeviceUnavailable,
            Self::PlanAffinityInvalid => PreparedExecutionPlanErrorCode::PlanAffinityInvalid,
            Self::PlanMemoryInvalid => PreparedExecutionPlanErrorCode::PlanMemoryInvalid,
            Self::PlanPreparedKernelMissing | Self::PreparedKernelBindingIncomplete => {
                PreparedExecutionPlanErrorCode::PlanPreparedKernelMissing
            }
            Self::PlanGenerationInUse { .. } => PreparedExecutionPlanErrorCode::PlanGenerationInUse,
            Self::ProviderSegmentIncompatible => {
                PreparedExecutionPlanErrorCode::SegmentProviderIncompatible
            }
            Self::PreparedSegmentEmpty
            | Self::SegmentSemanticMismatch
            | Self::ProviderSegmentStateInvalid => PreparedExecutionPlanErrorCode::SegmentInvalid,
            Self::PlanHotPathRebuildDenied => {
                PreparedExecutionPlanErrorCode::PlanHotPathRebuildDenied
            }
            Self::PlanWorkloadIncompatible => {
                PreparedExecutionPlanErrorCode::PlanWorkloadIncompatible
            }
            _ => PreparedExecutionPlanErrorCode::InternalPlan,
        }
    }

    pub const fn id(&self) -> &'static str {
        self.code().id()
    }
}

pub fn semantic_graph_fingerprint(graph: &ExecutionGraph) -> ExecutionGraphSemanticFingerprint {
    let mut canonical = String::new();
    canonical.push_str("magnetar-execution-graph-v1\n");
    canonical.push_str(&format!("id={}\n", graph.id));
    canonical.push_str(&format!("version={}\n", graph.version.0));
    canonical.push_str(&format!("phase={:?}\n", graph.phase));
    canonical.push_str(&format!("model={:?}\n", graph.model));
    canonical.push_str(&format!("adapter={:?}\n", graph.adapter));

    for (node_id, node) in &graph.nodes {
        canonical.push_str(&format!("node={node_id}\n"));
        canonical.push_str(&format!(
            "operator={}/{}/{}:{:?}\n",
            node.operator.namespace(),
            node.operator.name(),
            node.operator.version(),
            node.operator.family()
        ));
        canonical.push_str(&format!("attributes={:?}\n", node.attributes));
        canonical.push_str(&format!("inputs={:?}\n", node.inputs));
        canonical.push_str(&format!("outputs={:?}\n", node.outputs));
    }

    for (edge_id, edge) in &graph.edges {
        canonical.push_str(&format!("edge={edge_id}\n"));
        canonical.push_str(&format!("logical={}\n", edge.logical_tensor_id));
        canonical.push_str(&format!(
            "tensor=shape:{:?}:symbolic:{:?}:dtype:{:?}:layout:{:?}:storage:{:?}:compute:{:?}\n",
            edge.descriptor.shape.dimensions,
            edge.descriptor.shape.symbolic,
            edge.descriptor.dtype,
            layout_kind(&edge.descriptor.layout),
            edge.descriptor.storage_dtype,
            edge.descriptor.compute_dtype
        ));
        canonical.push_str(&format!("producer={:?}\n", edge.producer));
        canonical.push_str(&format!("consumers={:?}\n", edge.consumers));
        canonical.push_str(&format!("kv={:?}\n", edge.kv_cache));
        canonical.push_str(&format!("prefix={:?}\n", edge.prefix_cache));
    }

    ExecutionGraphSemanticFingerprint(format!("sha256:{}", sha256_hex(canonical.as_bytes())))
}

fn plan_fingerprint(
    id: &PreparedExecutionPlanId,
    generation: PreparedExecutionPlanGeneration,
    graph_fingerprint: &ExecutionGraphSemanticFingerprint,
    scope: &PreparedExecutionPlanScope,
    node_bindings: &[PlanNodeBinding],
) -> Result<PreparedExecutionPlanFingerprint, PreparedExecutionPlanError> {
    PreparedExecutionPlanFingerprint::new(format!(
        "sha256:{}",
        sha256_hex(
            format!(
                "plan={id}\ngeneration={}\ngraph={graph_fingerprint}\nscope={scope:?}\nbindings={node_bindings:?}\n",
                generation.value()
            )
            .as_bytes()
        )
    ))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn validate_fingerprint(
    value: &str,
    field: &'static str,
) -> Result<(), PreparedExecutionPlanError> {
    if value.trim().is_empty() || value.contains("0x") || value.to_ascii_lowercase().contains("ptr")
    {
        return Err(PreparedExecutionPlanError::InvalidFingerprint { field });
    }
    Ok(())
}

fn validate_logical_identity(
    value: &str,
    field: &'static str,
) -> Result<(), PreparedExecutionPlanError> {
    let lower = value.to_ascii_lowercase();
    if value.trim().is_empty()
        || value.contains('/')
        || value.contains('\\')
        || value.contains(':')
        || value.contains("0x")
        || lower.contains("ptr")
        || lower.contains("handle")
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    {
        return Err(PreparedExecutionPlanError::InvalidIdentity {
            field,
            value: value.into(),
        });
    }
    Ok(())
}

fn validate_slot_identity(
    value: &str,
    field: &'static str,
) -> Result<(), PreparedExecutionPlanError> {
    let lower = value.to_ascii_lowercase();
    if value.trim().is_empty()
        || value.contains('\\')
        || value.contains("0x")
        || lower.contains("ptr")
        || lower.contains("handle")
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_' | b':'))
    {
        return Err(PreparedExecutionPlanError::InvalidIdentity {
            field,
            value: value.into(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;

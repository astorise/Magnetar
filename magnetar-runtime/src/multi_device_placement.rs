//! Local multi-Device placement contracts.
//!
//! The Runtime owns these structures. They describe stable logical placement
//! decisions and deliberately avoid Provider-native handles.

use crate::{
    DTypeDescriptor, DeviceAvailability, DeviceBinding, DeviceMetadata, ExecutionNodeId,
    HostStagingPolicy, KernelId, KernelMemoryClass, MemoryAllocationClass, MemoryDomain,
    ModelInstanceId, ProviderBinding, ProviderPressureLevel, ResourceAffinity, ShapeDescriptor,
    TensorLayoutKind, TensorResourceId,
};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DeviceSetId(String);

impl DeviceSetId {
    pub fn new(value: impl Into<String>) -> Result<Self, MultiDevicePlacementError> {
        let value = value.into();
        validate_logical_id(&value, "device set id")?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for DeviceSetId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MultiDevicePlacementPlanId(String);

impl MultiDevicePlacementPlanId {
    pub fn new(value: impl Into<String>) -> Result<Self, MultiDevicePlacementError> {
        let value = value.into();
        validate_logical_id(&value, "placement plan id")?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for MultiDevicePlacementPlanId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MultiDevicePlacementGeneration(u64);

impl MultiDevicePlacementGeneration {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn value(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MultiDevicePlacementFingerprint(String);

impl MultiDevicePlacementFingerprint {
    pub fn new(value: impl Into<String>) -> Result<Self, MultiDevicePlacementError> {
        let value = value.into();
        validate_fingerprint(&value, "placement fingerprint")?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for MultiDevicePlacementFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MultiDevicePlacementState {
    Building,
    Validating,
    Ready,
    Stale,
    Invalidated,
    Retiring,
    Retired,
    Failed,
}

impl MultiDevicePlacementState {
    pub const fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Building, Self::Validating)
                | (Self::Building, Self::Failed)
                | (Self::Validating, Self::Ready)
                | (Self::Validating, Self::Failed)
                | (Self::Ready, Self::Stale)
                | (Self::Ready, Self::Invalidated)
                | (Self::Ready, Self::Retiring)
                | (Self::Stale, Self::Ready)
                | (Self::Stale, Self::Invalidated)
                | (Self::Stale, Self::Retiring)
                | (Self::Invalidated, Self::Retiring)
                | (Self::Retiring, Self::Retired)
        )
    }

    pub const fn accepts_new_work(self) -> bool {
        matches!(self, Self::Ready | Self::Stale)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceSetMember {
    pub provider: ProviderBinding,
    pub device: DeviceBinding,
    pub metadata: DeviceMetadata,
    pub availability: DeviceAvailability,
}

impl DeviceSetMember {
    pub fn new(metadata: DeviceMetadata, availability: DeviceAvailability) -> Self {
        Self {
            provider: ProviderBinding::new(&metadata.provider),
            device: DeviceBinding::new(metadata.id.clone()),
            metadata,
            availability,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceSet {
    pub id: DeviceSetId,
    members: Vec<DeviceSetMember>,
}

impl DeviceSet {
    pub fn new(
        id: DeviceSetId,
        members: impl IntoIterator<Item = DeviceSetMember>,
    ) -> Result<Self, MultiDevicePlacementError> {
        let members = members.into_iter().collect::<Vec<_>>();
        if members.is_empty() {
            return Err(MultiDevicePlacementError::NoDevices);
        }
        let mut seen = BTreeSet::new();
        for member in &members {
            if member.metadata.provider != member.provider.as_str() {
                return Err(MultiDevicePlacementError::ProviderDeviceMismatch);
            }
            if !seen.insert(member.device.clone()) {
                return Err(MultiDevicePlacementError::DuplicateDevice(
                    member.device.clone(),
                ));
            }
        }
        Ok(Self { id, members })
    }

    pub fn members(&self) -> &[DeviceSetMember] {
        &self.members
    }

    pub fn providers(&self) -> BTreeSet<ProviderBinding> {
        self.members
            .iter()
            .map(|member| member.provider.clone())
            .collect()
    }

    pub fn contains_device(&self, device: &DeviceBinding) -> bool {
        self.members.iter().any(|member| &member.device == device)
    }

    pub fn fingerprint(&self) -> MultiDevicePlacementFingerprint {
        let mut lines = vec![format!("device-set={}", self.id)];
        for member in &self.members {
            lines.push(format!(
                "provider={};device={};arch={};mem={};dtype={:?};layout={:?};memclass={:?};pressure={:?};availability={:?}",
                member.provider,
                member.device,
                member.metadata.architecture,
                member.metadata.memory_capacity,
                member.metadata.dtype_support,
                member.metadata.layout_support,
                member.metadata.memory_class_support,
                member.metadata.pressure,
                member.availability
            ));
        }
        sha256_fingerprint(&lines.join("\n"))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlacementDomain {
    pub name: String,
    pub devices: BTreeSet<DeviceBinding>,
    pub allow_cross_provider: bool,
    pub host_staging_policy: HostStagingPolicy,
}

impl PlacementDomain {
    pub fn new(
        name: impl Into<String>,
        devices: impl IntoIterator<Item = DeviceBinding>,
    ) -> Result<Self, MultiDevicePlacementError> {
        let name = name.into();
        validate_logical_id(&name, "placement domain")?;
        let devices = devices.into_iter().collect::<BTreeSet<_>>();
        if devices.is_empty() {
            return Err(MultiDevicePlacementError::NoDevices);
        }
        Ok(Self {
            name,
            devices,
            allow_cross_provider: false,
            host_staging_policy: HostStagingPolicy::Forbid,
        })
    }

    pub const fn with_cross_provider(mut self, allowed: bool) -> Self {
        self.allow_cross_provider = allowed;
        self
    }

    pub const fn with_host_staging_policy(mut self, policy: HostStagingPolicy) -> Self {
        self.host_staging_policy = policy;
        self
    }

    pub fn validate_against(
        &self,
        device_set: &DeviceSet,
    ) -> Result<(), MultiDevicePlacementError> {
        for device in &self.devices {
            if !device_set.contains_device(device) {
                return Err(MultiDevicePlacementError::DeviceOutsideSet(device.clone()));
            }
        }
        let providers = device_set
            .members()
            .iter()
            .filter(|member| self.devices.contains(&member.device))
            .map(|member| member.provider.clone())
            .collect::<BTreeSet<_>>();
        if providers.len() > 1 && !self.allow_cross_provider {
            return Err(MultiDevicePlacementError::CrossProviderDenied);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlacementBinding {
    pub provider: ProviderBinding,
    pub device: DeviceBinding,
    pub memory_domain: MemoryDomain,
    pub affinity: Option<ResourceAffinity>,
    pub constraints: BTreeSet<String>,
}

impl PlacementBinding {
    pub fn new(
        provider: ProviderBinding,
        device: DeviceBinding,
        memory_domain: MemoryDomain,
    ) -> Result<Self, MultiDevicePlacementError> {
        match &memory_domain {
            MemoryDomain::DeviceLocal(bound) | MemoryDomain::HostVisibleDevice(bound)
                if bound != &device =>
            {
                return Err(MultiDevicePlacementError::MemoryDomainDeviceMismatch);
            }
            MemoryDomain::Shared {
                device: Some(bound),
                ..
            }
            | MemoryDomain::Managed {
                device: Some(bound),
                ..
            } if bound != &device => {
                return Err(MultiDevicePlacementError::MemoryDomainDeviceMismatch);
            }
            _ => {}
        }
        Ok(Self {
            provider,
            device,
            memory_domain,
            affinity: None,
            constraints: BTreeSet::new(),
        })
    }

    pub fn with_constraint(
        mut self,
        constraint: impl Into<String>,
    ) -> Result<Self, MultiDevicePlacementError> {
        let constraint = constraint.into();
        validate_no_native_handle(&constraint, "placement constraint")?;
        self.constraints.insert(constraint);
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PlacementGranularity {
    ModelInstance,
    GraphSegment,
    LayerRange,
    OperatorGroup,
    IndividualOperatorReserved,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlacementScope {
    pub granularity: PlacementGranularity,
    pub model_instance: Option<ModelInstanceId>,
    pub graph_nodes: BTreeSet<ExecutionNodeId>,
    pub layer_range: Option<(u32, u32)>,
    pub operator_group: Option<String>,
}

impl PlacementScope {
    pub fn model_instance(model_instance: ModelInstanceId) -> Self {
        Self {
            granularity: PlacementGranularity::ModelInstance,
            model_instance: Some(model_instance),
            graph_nodes: BTreeSet::new(),
            layer_range: None,
            operator_group: None,
        }
    }

    pub fn layer_range(start: u32, end: u32) -> Result<Self, MultiDevicePlacementError> {
        if start > end {
            return Err(MultiDevicePlacementError::InvalidLayerRange);
        }
        Ok(Self {
            granularity: PlacementGranularity::LayerRange,
            model_instance: None,
            graph_nodes: BTreeSet::new(),
            layer_range: Some((start, end)),
            operator_group: None,
        })
    }

    pub fn graph_segment(
        nodes: impl IntoIterator<Item = ExecutionNodeId>,
    ) -> Result<Self, MultiDevicePlacementError> {
        let graph_nodes = nodes.into_iter().collect::<BTreeSet<_>>();
        if graph_nodes.is_empty() {
            return Err(MultiDevicePlacementError::EmptyPlacementScope);
        }
        Ok(Self {
            granularity: PlacementGranularity::GraphSegment,
            model_instance: None,
            graph_nodes,
            layer_range: None,
            operator_group: None,
        })
    }

    pub fn operator_group(name: impl Into<String>) -> Result<Self, MultiDevicePlacementError> {
        let name = name.into();
        validate_logical_id(&name, "operator group")?;
        Ok(Self {
            granularity: PlacementGranularity::OperatorGroup,
            model_instance: None,
            graph_nodes: BTreeSet::new(),
            layer_range: None,
            operator_group: Some(name),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PipelineStage {
    pub stage_id: String,
    pub graph_nodes: BTreeSet<ExecutionNodeId>,
    pub binding: PlacementBinding,
    pub input_requirements: BTreeSet<TensorResourceId>,
    pub output_requirements: BTreeSet<TensorResourceId>,
    pub order: u32,
}

impl PipelineStage {
    pub fn new(
        stage_id: impl Into<String>,
        graph_nodes: impl IntoIterator<Item = ExecutionNodeId>,
        binding: PlacementBinding,
        order: u32,
    ) -> Result<Self, MultiDevicePlacementError> {
        let stage_id = stage_id.into();
        validate_logical_id(&stage_id, "pipeline stage id")?;
        let graph_nodes = graph_nodes.into_iter().collect::<BTreeSet<_>>();
        if graph_nodes.is_empty() {
            return Err(MultiDevicePlacementError::EmptyPlacementScope);
        }
        Ok(Self {
            stage_id,
            graph_nodes,
            binding,
            input_requirements: BTreeSet::new(),
            output_requirements: BTreeSet::new(),
            order,
        })
    }

    pub fn with_input(mut self, resource: TensorResourceId) -> Self {
        self.input_requirements.insert(resource);
        self
    }

    pub fn with_output(mut self, resource: TensorResourceId) -> Self {
        self.output_requirements.insert(resource);
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageMovementEdge {
    pub from_stage: String,
    pub to_stage: String,
    pub resource: TensorResourceId,
    pub source: PlacementBinding,
    pub destination: PlacementBinding,
    pub host_staging_policy: HostStagingPolicy,
    pub preserves_source_lifetime: bool,
    pub preserves_destination_readiness: bool,
}

impl StageMovementEdge {
    pub fn new(
        from_stage: impl Into<String>,
        to_stage: impl Into<String>,
        resource: TensorResourceId,
        source: PlacementBinding,
        destination: PlacementBinding,
        host_staging_policy: HostStagingPolicy,
    ) -> Result<Self, MultiDevicePlacementError> {
        let from_stage = from_stage.into();
        let to_stage = to_stage.into();
        validate_logical_id(&from_stage, "source stage id")?;
        validate_logical_id(&to_stage, "destination stage id")?;
        if source.device == destination.device {
            return Err(MultiDevicePlacementError::MovementNotRequired);
        }
        Ok(Self {
            from_stage,
            to_stage,
            resource,
            source,
            destination,
            host_staging_policy,
            preserves_source_lifetime: true,
            preserves_destination_readiness: true,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PipelineOverlapMode {
    Serial,
    IndependentStages,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PipelineOverlapPolicy {
    pub mode: PipelineOverlapMode,
    pub memory_capacity_ok: bool,
    pub dependencies_satisfied: bool,
    pub scheduler_policy_allows: bool,
}

impl PipelineOverlapPolicy {
    pub const fn independent() -> Self {
        Self {
            mode: PipelineOverlapMode::IndependentStages,
            memory_capacity_ok: true,
            dependencies_satisfied: true,
            scheduler_policy_allows: true,
        }
    }

    pub const fn permits_overlap(&self) -> bool {
        matches!(self.mode, PipelineOverlapMode::IndependentStages)
            && self.memory_capacity_ok
            && self.dependencies_satisfied
            && self.scheduler_policy_allows
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum WeightPlacementKind {
    SingleDevice,
    Partitioned,
    Replicated,
    Hybrid,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightPlacement {
    pub artifact_identity: String,
    pub revision: String,
    pub dtype: DTypeDescriptor,
    pub layout: TensorLayoutKind,
    pub kind: WeightPlacementKind,
    pub devices: BTreeSet<DeviceBinding>,
    pub partition: Option<TensorPartitionDescriptor>,
    pub replicas: Vec<WeightReplica>,
}

impl WeightPlacement {
    pub fn new(
        artifact_identity: impl Into<String>,
        revision: impl Into<String>,
        dtype: DTypeDescriptor,
        layout: TensorLayoutKind,
        kind: WeightPlacementKind,
        devices: impl IntoIterator<Item = DeviceBinding>,
    ) -> Result<Self, MultiDevicePlacementError> {
        let artifact_identity = artifact_identity.into();
        let revision = revision.into();
        validate_logical_id(&artifact_identity, "weight artifact identity")?;
        validate_logical_id(&revision, "weight revision")?;
        let devices = devices.into_iter().collect::<BTreeSet<_>>();
        if devices.is_empty() {
            return Err(MultiDevicePlacementError::NoDevices);
        }
        Ok(Self {
            artifact_identity,
            revision,
            dtype,
            layout,
            kind,
            devices,
            partition: None,
            replicas: Vec::new(),
        })
    }

    pub fn with_partition(mut self, partition: TensorPartitionDescriptor) -> Self {
        self.partition = Some(partition);
        self
    }

    pub fn with_replica(mut self, replica: WeightReplica) -> Self {
        self.replicas.push(replica);
        self
    }

    pub fn validate(&self) -> Result<(), MultiDevicePlacementError> {
        match self.kind {
            WeightPlacementKind::SingleDevice if self.devices.len() != 1 => {
                return Err(MultiDevicePlacementError::WeightPlacementInvalid);
            }
            WeightPlacementKind::Partitioned if self.partition.is_none() => {
                return Err(MultiDevicePlacementError::WeightPlacementInvalid);
            }
            WeightPlacementKind::Replicated if self.replicas.is_empty() => {
                return Err(MultiDevicePlacementError::WeightPlacementInvalid);
            }
            WeightPlacementKind::Hybrid if self.partition.is_none() || self.replicas.is_empty() => {
                return Err(MultiDevicePlacementError::WeightPlacementInvalid);
            }
            _ => {}
        }
        if let Some(partition) = &self.partition {
            partition.validate()?;
        }
        for replica in &self.replicas {
            if replica.artifact_identity != self.artifact_identity
                || replica.revision != self.revision
                || replica.dtype != self.dtype
                || replica.layout != self.layout
            {
                return Err(MultiDevicePlacementError::ReplicaIncompatible);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightReplica {
    pub artifact_identity: String,
    pub revision: String,
    pub dtype: DTypeDescriptor,
    pub layout: TensorLayoutKind,
    pub residency: PlacementBinding,
    pub valid: bool,
    pub authoritative: bool,
}

impl WeightReplica {
    pub fn new(
        artifact_identity: impl Into<String>,
        revision: impl Into<String>,
        dtype: DTypeDescriptor,
        layout: TensorLayoutKind,
        residency: PlacementBinding,
    ) -> Result<Self, MultiDevicePlacementError> {
        let artifact_identity = artifact_identity.into();
        let revision = revision.into();
        validate_logical_id(&artifact_identity, "replica artifact identity")?;
        validate_logical_id(&revision, "replica revision")?;
        Ok(Self {
            artifact_identity,
            revision,
            dtype,
            layout,
            residency,
            valid: true,
            authoritative: false,
        })
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TensorShardId(String);

impl TensorShardId {
    pub fn new(value: impl Into<String>) -> Result<Self, MultiDevicePlacementError> {
        let value = value.into();
        validate_logical_id(&value, "tensor shard id")?;
        Ok(Self(value))
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TensorPartitionAxis {
    Dimension(u32),
    Semantic(String),
    Head,
    Hidden,
    Vocabulary,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LogicalRange {
    pub axis: TensorPartitionAxis,
    pub start: u64,
    pub end: u64,
}

impl LogicalRange {
    pub fn new(axis: TensorPartitionAxis, start: u64, end: u64) -> Self {
        Self { axis, start, end }
    }

    pub const fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    pub fn overlaps(&self, other: &Self) -> bool {
        self.axis == other.axis && self.start < other.end && other.start < self.end
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorShard {
    pub id: TensorShardId,
    pub parent: TensorResourceId,
    pub logical_range: LogicalRange,
    pub shape: ShapeDescriptor,
    pub dtype: DTypeDescriptor,
    pub layout: TensorLayoutKind,
    pub placement: PlacementBinding,
    pub residency: MemoryDomain,
    pub replica: bool,
}

impl TensorShard {
    pub fn new(
        id: TensorShardId,
        parent: TensorResourceId,
        logical_range: LogicalRange,
        shape: ShapeDescriptor,
        dtype: DTypeDescriptor,
        layout: TensorLayoutKind,
        placement: PlacementBinding,
    ) -> Self {
        let residency = placement.memory_domain.clone();
        Self {
            id,
            parent,
            logical_range,
            shape,
            dtype,
            layout,
            placement,
            residency,
            replica: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TensorReconstructionPolicy {
    LogicalOnly,
    ExplicitMaterializationRequired,
    UnsupportedWithoutFutureCollective,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorPartitionDescriptor {
    pub logical_resource: TensorResourceId,
    pub partition_axis: TensorPartitionAxis,
    pub partition_count: u32,
    pub shards: Vec<TensorShard>,
    pub reconstruction: TensorReconstructionPolicy,
    pub allow_replicated_ranges: bool,
}

impl TensorPartitionDescriptor {
    pub fn new(
        logical_resource: TensorResourceId,
        partition_axis: TensorPartitionAxis,
        partition_count: u32,
        reconstruction: TensorReconstructionPolicy,
    ) -> Result<Self, MultiDevicePlacementError> {
        if partition_count == 0 {
            return Err(MultiDevicePlacementError::PartitionInvalid);
        }
        Ok(Self {
            logical_resource,
            partition_axis,
            partition_count,
            shards: Vec::new(),
            reconstruction,
            allow_replicated_ranges: false,
        })
    }

    pub fn with_shard(mut self, shard: TensorShard) -> Self {
        self.shards.push(shard);
        self
    }

    pub const fn allow_replicas(mut self) -> Self {
        self.allow_replicated_ranges = true;
        self
    }

    pub fn validate(&self) -> Result<(), MultiDevicePlacementError> {
        if self.shards.len() != self.partition_count as usize {
            return Err(MultiDevicePlacementError::PartitionMissingShard);
        }
        let mut ranges = Vec::new();
        for shard in &self.shards {
            if shard.parent != self.logical_resource
                || shard.logical_range.axis != self.partition_axis
                || shard.logical_range.is_empty()
                || shard.shape.dimensions.contains(&0)
            {
                return Err(MultiDevicePlacementError::PartitionInvalid);
            }
            let _ = shard
                .shape
                .element_count()
                .map_err(|_| MultiDevicePlacementError::PartitionOverflow)?;
            for existing in &ranges {
                if shard.logical_range.overlaps(existing) && !self.allow_replicated_ranges {
                    return Err(MultiDevicePlacementError::PartitionOverlap);
                }
            }
            ranges.push(shard.logical_range.clone());
        }
        ranges.sort_by_key(|range| range.start);
        for pair in ranges.windows(2) {
            if pair[0].end != pair[1].start && !self.allow_replicated_ranges {
                return Err(MultiDevicePlacementError::PartitionGap);
            }
        }
        Ok(())
    }

    pub const fn implies_collective(&self) -> bool {
        false
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelPartitionCompatibility {
    pub kernel: KernelId,
    pub input_axes: BTreeSet<TensorPartitionAxis>,
    pub output_axes: BTreeSet<TensorPartitionAxis>,
}

impl KernelPartitionCompatibility {
    pub fn supports_input(&self, descriptor: &TensorPartitionDescriptor) -> bool {
        self.input_axes.contains(&descriptor.partition_axis)
    }

    pub fn validate_input(
        &self,
        descriptor: &TensorPartitionDescriptor,
    ) -> Result<(), MultiDevicePlacementError> {
        if self.supports_input(descriptor) {
            Ok(())
        } else {
            Err(MultiDevicePlacementError::ShardAsFullTensorUnsupported)
        }
    }

    pub fn validate_output(
        &self,
        descriptor: &TensorPartitionDescriptor,
    ) -> Result<(), MultiDevicePlacementError> {
        if self.output_axes.contains(&descriptor.partition_axis) {
            Ok(())
        } else {
            Err(MultiDevicePlacementError::ShardAsFullTensorUnsupported)
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DevicePairAccessMode {
    PeerRead,
    PeerWrite,
    PeerCopy,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DevicePairTransferCapability {
    pub provider: ProviderBinding,
    pub source: DeviceBinding,
    pub destination: DeviceBinding,
    pub modes: BTreeSet<DevicePairAccessMode>,
    pub bandwidth_class: String,
    pub requires_host_staging: bool,
    pub exposes_native_handle: bool,
}

impl DevicePairTransferCapability {
    pub fn new(
        provider: ProviderBinding,
        source: DeviceBinding,
        destination: DeviceBinding,
        modes: impl IntoIterator<Item = DevicePairAccessMode>,
    ) -> Self {
        Self {
            provider,
            source,
            destination,
            modes: modes.into_iter().collect(),
            bandwidth_class: "unknown".into(),
            requires_host_staging: false,
            exposes_native_handle: false,
        }
    }

    pub fn validate(&self) -> Result<(), MultiDevicePlacementError> {
        validate_no_native_handle(&self.bandwidth_class, "peer bandwidth class")?;
        if self.exposes_native_handle {
            return Err(MultiDevicePlacementError::NativeHandleForbidden {
                field: "peer transfer capability",
            });
        }
        Ok(())
    }

    pub fn supports(&self, mode: DevicePairAccessMode) -> bool {
        self.modes.contains(&mode)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DeviceTransferKind {
    DirectPeer,
    ProviderLocalCopy,
    HostStaged,
    CrossProviderBoundary,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceTransferPlan {
    pub kind: DeviceTransferKind,
    pub movement: StageMovementEdge,
    pub expected_bytes: u64,
    pub peer_bandwidth_class: Option<String>,
    pub synchronization_cost_micros: u64,
    pub host_staging_cost_micros: u64,
}

impl DeviceTransferPlan {
    pub fn new(
        movement: StageMovementEdge,
        expected_bytes: u64,
        capability: Option<&DevicePairTransferCapability>,
    ) -> Result<Self, MultiDevicePlacementError> {
        let cross_provider = movement.source.provider != movement.destination.provider;
        let Some(capability) = capability else {
            if movement.host_staging_policy == HostStagingPolicy::Forbid {
                return Err(MultiDevicePlacementError::PeerTransferUnavailable);
            }
            return Ok(Self {
                kind: if cross_provider {
                    DeviceTransferKind::CrossProviderBoundary
                } else {
                    DeviceTransferKind::HostStaged
                },
                movement,
                expected_bytes,
                peer_bandwidth_class: None,
                synchronization_cost_micros: 100,
                host_staging_cost_micros: 1_000,
            });
        };
        capability.validate()?;
        if capability.source != movement.source.device
            || capability.destination != movement.destination.device
        {
            return Err(MultiDevicePlacementError::PeerTransferUnavailable);
        }
        if capability.requires_host_staging
            && movement.host_staging_policy == HostStagingPolicy::Forbid
        {
            return Err(MultiDevicePlacementError::HostStagingDenied);
        }
        if capability.supports(DevicePairAccessMode::PeerCopy) {
            Ok(Self {
                kind: DeviceTransferKind::DirectPeer,
                movement,
                expected_bytes,
                peer_bandwidth_class: Some(capability.bandwidth_class.clone()),
                synchronization_cost_micros: 25,
                host_staging_cost_micros: 0,
            })
        } else if capability.requires_host_staging {
            Ok(Self {
                kind: DeviceTransferKind::HostStaged,
                movement,
                expected_bytes,
                peer_bandwidth_class: Some(capability.bandwidth_class.clone()),
                synchronization_cost_micros: 100,
                host_staging_cost_micros: 1_000,
            })
        } else {
            Ok(Self {
                kind: DeviceTransferKind::ProviderLocalCopy,
                movement,
                expected_bytes,
                peer_bandwidth_class: Some(capability.bandwidth_class.clone()),
                synchronization_cost_micros: 50,
                host_staging_cost_micros: 0,
            })
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceMemoryBudget {
    pub device: DeviceBinding,
    pub capacity_bytes: u64,
    pub weights_bytes: u64,
    pub kv_bytes: u64,
    pub workspace_bytes: u64,
    pub transient_bytes: u64,
    pub transfer_buffer_bytes: u64,
    pub reserved_headroom_bytes: u64,
}

impl DeviceMemoryBudget {
    pub fn new(device: DeviceBinding, capacity_bytes: u64) -> Self {
        Self {
            device,
            capacity_bytes,
            weights_bytes: 0,
            kv_bytes: 0,
            workspace_bytes: 0,
            transient_bytes: 0,
            transfer_buffer_bytes: 0,
            reserved_headroom_bytes: 0,
        }
    }

    pub fn used_bytes(&self) -> Result<u64, MultiDevicePlacementError> {
        [
            self.weights_bytes,
            self.kv_bytes,
            self.workspace_bytes,
            self.transient_bytes,
            self.transfer_buffer_bytes,
            self.reserved_headroom_bytes,
        ]
        .into_iter()
        .try_fold(0_u64, |acc, value| {
            acc.checked_add(value)
                .ok_or(MultiDevicePlacementError::MemoryBudgetOverflow)
        })
    }

    pub fn validate(&self) -> Result<(), MultiDevicePlacementError> {
        if self.used_bytes()? > self.capacity_bytes {
            return Err(MultiDevicePlacementError::MemoryBudgetExceeded);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AllocationSlotBinding {
    pub slot_id: String,
    pub device: DeviceBinding,
    pub class: MemoryAllocationClass,
    pub bytes: u64,
}

impl AllocationSlotBinding {
    pub fn new(
        slot_id: impl Into<String>,
        device: DeviceBinding,
        class: MemoryAllocationClass,
        bytes: u64,
    ) -> Result<Self, MultiDevicePlacementError> {
        let slot_id = slot_id.into();
        validate_logical_id(&slot_id, "allocation slot id")?;
        Ok(Self {
            slot_id,
            device,
            class,
            bytes,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DevicePoolBinding {
    pub device: DeviceBinding,
    pub pool_id: String,
    pub slots: Vec<AllocationSlotBinding>,
}

impl DevicePoolBinding {
    pub fn new(
        device: DeviceBinding,
        pool_id: impl Into<String>,
    ) -> Result<Self, MultiDevicePlacementError> {
        let pool_id = pool_id.into();
        validate_logical_id(&pool_id, "device memory pool id")?;
        Ok(Self {
            device,
            pool_id,
            slots: Vec::new(),
        })
    }

    pub fn with_slot(
        mut self,
        slot: AllocationSlotBinding,
    ) -> Result<Self, MultiDevicePlacementError> {
        if slot.device != self.device {
            return Err(MultiDevicePlacementError::AllocationSlotDeviceMismatch);
        }
        self.slots.push(slot);
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), MultiDevicePlacementError> {
        if self.slots.is_empty() {
            return Err(MultiDevicePlacementError::UnspecifiedGlobalPoolUse);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlacementHysteresisPolicy {
    pub minimum_improvement_micros: u64,
    pub cooldown_millis: u64,
}

impl PlacementHysteresisPolicy {
    pub const fn new(minimum_improvement_micros: u64, cooldown_millis: u64) -> Self {
        Self {
            minimum_improvement_micros,
            cooldown_millis,
        }
    }

    pub const fn should_replace(
        &self,
        current_cost_micros: u64,
        candidate_cost_micros: u64,
        millis_since_last_change: u64,
    ) -> bool {
        candidate_cost_micros.saturating_add(self.minimum_improvement_micros) < current_cost_micros
            && millis_since_last_change >= self.cooldown_millis
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PlacementPinKind {
    ModelInstanceDeviceSet,
    Stage,
    Weights,
    Kv,
    SessionPreference,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlacementPin {
    pub kind: PlacementPinKind,
    pub model_instance: Option<ModelInstanceId>,
    pub stage_id: Option<String>,
    pub weight_artifact: Option<String>,
    pub session_id: Option<String>,
    pub device_set: Option<DeviceSetId>,
    pub binding: Option<PlacementBinding>,
    pub compatibility_authoritative: bool,
    pub device_available: bool,
}

impl PlacementPin {
    pub fn validate(&self) -> Result<(), MultiDevicePlacementError> {
        if !self.compatibility_authoritative || !self.device_available {
            return Err(MultiDevicePlacementError::PlacementPinInvalid);
        }
        if let Some(stage_id) = &self.stage_id {
            validate_logical_id(stage_id, "stage pin")?;
        }
        if let Some(weight_artifact) = &self.weight_artifact {
            validate_logical_id(weight_artifact, "weight pin")?;
        }
        if let Some(session_id) = &self.session_id {
            validate_logical_id(session_id, "session pin")?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PlacementPhase {
    Prefill,
    Decode,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhasePlacementPlans {
    pub prefill_plan: MultiDevicePlacementPlanId,
    pub decode_plan: MultiDevicePlacementPlanId,
}

impl PhasePlacementPlans {
    pub fn plan_for(&self, phase: PlacementPhase) -> &MultiDevicePlacementPlanId {
        match phase {
            PlacementPhase::Prefill => &self.prefill_plan,
            PlacementPhase::Decode => &self.decode_plan,
        }
    }

    pub fn uses_distinct_plans(&self) -> bool {
        self.prefill_plan != self.decode_plan
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhaseTransitionReadiness {
    pub from: PlacementPhase,
    pub to: PlacementPhase,
    pub kv_available: bool,
    pub weights_available: bool,
    pub upstream_completion_observed: bool,
    pub explicit_movements_completed: bool,
    pub decode_guards_passed: bool,
}

impl PhaseTransitionReadiness {
    pub const fn permits_transition(&self) -> bool {
        self.kv_available
            && self.weights_available
            && self.upstream_completion_observed
            && self.explicit_movements_completed
            && self.decode_guards_passed
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvPagePlacement {
    pub page_id: String,
    pub session_id: String,
    pub sequence_id: String,
    pub owner: PlacementBinding,
    pub authoritative: bool,
}

impl KvPagePlacement {
    pub fn new(
        page_id: impl Into<String>,
        session_id: impl Into<String>,
        sequence_id: impl Into<String>,
        owner: PlacementBinding,
    ) -> Result<Self, MultiDevicePlacementError> {
        let page_id = page_id.into();
        let session_id = session_id.into();
        let sequence_id = sequence_id.into();
        validate_logical_id(&page_id, "kv page id")?;
        validate_logical_id(&session_id, "kv session id")?;
        validate_logical_id(&sequence_id, "kv sequence id")?;
        Ok(Self {
            page_id,
            session_id,
            sequence_id,
            owner,
            authoritative: true,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvLocalityDecision {
    pub decode_binding: PlacementBinding,
    pub authoritative_kv_device: DeviceBinding,
    pub kv_movement_cost_micros: u64,
    pub permits_per_token_bounce: bool,
}

impl KvLocalityDecision {
    pub fn favors_kv_locality(&self) -> bool {
        self.decode_binding.device == self.authoritative_kv_device
            || (!self.permits_per_token_bounce && self.kv_movement_cost_micros > 0)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvPartitionBoundary {
    pub attention_contract_supports_partition: bool,
    pub required_collectives: BTreeSet<String>,
}

impl KvPartitionBoundary {
    pub fn validate(&self) -> Result<(), MultiDevicePlacementError> {
        if !self.attention_contract_supports_partition {
            return Err(MultiDevicePlacementError::KvPlacementInvalid);
        }
        if !self.required_collectives.is_empty() {
            return Err(MultiDevicePlacementError::ImplicitCollective);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvReplicaPolicy {
    pub authoritative_device: DeviceBinding,
    pub replicas: BTreeSet<DeviceBinding>,
    pub update_coherency_explicit: bool,
    pub baseline_prefers_single_authority: bool,
}

impl KvReplicaPolicy {
    pub const fn validate(&self) -> Result<(), MultiDevicePlacementError> {
        if !self.update_coherency_explicit || !self.baseline_prefers_single_authority {
            return Err(MultiDevicePlacementError::KvPlacementInvalid);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SessionPlacementAffinity {
    pub session_id: String,
    pub preferred_device: DeviceBinding,
    pub preferred_plan: MultiDevicePlacementPlanId,
    pub preserves_kv_locality: bool,
    pub migration_allowed: bool,
}

impl SessionPlacementAffinity {
    pub fn new(
        session_id: impl Into<String>,
        preferred_device: DeviceBinding,
        preferred_plan: MultiDevicePlacementPlanId,
    ) -> Result<Self, MultiDevicePlacementError> {
        let session_id = session_id.into();
        validate_logical_id(&session_id, "session id")?;
        Ok(Self {
            session_id,
            preferred_device,
            preferred_plan,
            preserves_kv_locality: true,
            migration_allowed: false,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SessionMigrationPlan {
    pub session_id: String,
    pub source: PlacementBinding,
    pub destination: PlacementBinding,
    pub moves_kv: bool,
    pub moves_adapters: bool,
    pub moves_session_buffers: bool,
    pub preserves_completion_tokens: bool,
}

impl SessionMigrationPlan {
    pub const fn is_explicit_and_complete(&self) -> bool {
        self.moves_kv
            && self.moves_adapters
            && self.moves_session_buffers
            && self.preserves_completion_tokens
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PlacementPlanRole {
    Default,
    WorkloadSpecific,
    PhaseSpecific(PlacementPhase),
    Degraded,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelInstancePlacementPlans {
    pub model_instance: ModelInstanceId,
    pub plans: BTreeMap<PlacementPlanRole, MultiDevicePlacementPlanId>,
}

impl ModelInstancePlacementPlans {
    pub fn add_plan(&mut self, role: PlacementPlanRole, plan: MultiDevicePlacementPlanId) {
        self.plans.insert(role, plan);
    }

    pub fn supports_role(&self, role: PlacementPlanRole) -> bool {
        self.plans.contains_key(&role)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedExecutionPlacement {
    pub placement_plan: MultiDevicePlacementPlanId,
    pub generation: MultiDevicePlacementGeneration,
    pub exact_segment_bindings: Vec<(String, PlacementBinding)>,
    pub movement_nodes: Vec<StageMovementEdge>,
    pub per_device_allocation_plans: BTreeMap<DeviceBinding, String>,
}

impl PreparedExecutionPlacement {
    pub fn validate(&self) -> Result<(), MultiDevicePlacementError> {
        if self.generation.value() == 0 || self.exact_segment_bindings.is_empty() {
            return Err(MultiDevicePlacementError::NoPlacementBindings);
        }
        for plan_id in self.per_device_allocation_plans.values() {
            validate_logical_id(plan_id, "allocation plan id")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlacementGuardSnapshot {
    pub device_available: bool,
    pub provider_ready: bool,
    pub kernel_prepared: bool,
    pub resource_resident: bool,
    pub memory_reserved: bool,
    pub peer_path_available: bool,
    pub host_staging_policy_valid: bool,
}

impl PlacementGuardSnapshot {
    pub const fn all_pass(&self) -> bool {
        self.device_available
            && self.provider_ready
            && self.kernel_prepared
            && self.resource_resident
            && self.memory_reserved
            && self.peer_path_available
            && self.host_staging_policy_valid
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PlacementStalenessReason {
    PressureShift,
    BetterPlacementAvailable,
    PerformanceDrift,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlacementStaleness {
    pub reason: PlacementStalenessReason,
    pub request_background_replacement: bool,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PlacementInvalidationReason {
    DeviceLost,
    ProviderLost,
    PeerPathLost,
    MemoryInfeasible,
    KernelRevoked,
    ResourceAffinityInvalid,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlacementInvalidation {
    pub reason: PlacementInvalidationReason,
    pub invalid_for_new_work: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlacementReplacementRequest {
    pub current_plan: MultiDevicePlacementPlanId,
    pub build_outside_hot_path: bool,
    pub revalidate_resources: bool,
    pub prepare_required_kernels: bool,
}

impl PlacementReplacementRequest {
    pub const fn can_build(&self) -> bool {
        self.build_outside_hot_path && self.revalidate_resources && self.prepare_required_kernels
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AtomicPlacementReplacement {
    pub old_plan: MultiDevicePlacementPlanId,
    pub old_generation: MultiDevicePlacementGeneration,
    pub new_plan: MultiDevicePlacementPlanId,
    pub new_generation: MultiDevicePlacementGeneration,
    pub new_plan_complete: bool,
    pub old_in_flight_retained: bool,
}

impl AtomicPlacementReplacement {
    pub const fn can_publish(&self) -> bool {
        self.new_plan_complete
            && self.old_in_flight_retained
            && self.new_generation.value() > self.old_generation.value()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceFailureImpact {
    pub lost_device: DeviceBinding,
    pub invalidates_streams: bool,
    pub invalidates_plans: bool,
    pub preserves_other_devices: bool,
}

impl DeviceFailureImpact {
    pub const fn is_isolated_failure_domain(&self) -> bool {
        self.invalidates_streams && self.invalidates_plans && self.preserves_other_devices
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DegradedPlanValidation {
    pub plan: MultiDevicePlacementPlanId,
    pub explicit_degraded_plan: bool,
    pub model_capacity_ok: bool,
    pub kernels_ok: bool,
    pub memory_ok: bool,
    pub policy_ok: bool,
}

impl DegradedPlanValidation {
    pub const fn valid(&self) -> bool {
        self.explicit_degraded_plan
            && self.model_capacity_ok
            && self.kernels_ok
            && self.memory_ok
            && self.policy_ok
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FailoverPolicy {
    pub arbitrary_remaining_device_forbidden: bool,
    pub ready_fallback_plan: Option<MultiDevicePlacementPlanId>,
}

impl FailoverPolicy {
    pub const fn can_fail_over(&self) -> bool {
        self.arbitrary_remaining_device_forbidden && self.ready_fallback_plan.is_some()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceRecoveryChecklist {
    pub health_readiness_checked: bool,
    pub pools_rebuilt: bool,
    pub kernels_reprepared: bool,
    pub placement_plan_rebuilt: bool,
}

impl DeviceRecoveryChecklist {
    pub const fn ready_for_new_work(&self) -> bool {
        self.health_readiness_checked
            && self.pools_rebuilt
            && self.kernels_reprepared
            && self.placement_plan_rebuilt
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SchedulerPlacementInput {
    pub session_affinity: Option<SessionPlacementAffinity>,
    pub device_pressure: BTreeMap<DeviceBinding, ProviderPressureLevel>,
    pub plan_ready: bool,
    pub exposes_native_handles: bool,
}

impl SchedulerPlacementInput {
    pub const fn can_admit(&self) -> bool {
        self.plan_ready && !self.exposes_native_handles
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlacementAdmissionCheck {
    pub mandatory_devices_available: bool,
    pub per_device_memory_ok: bool,
    pub required_kernels_available: bool,
    pub transfers_feasible: bool,
}

impl PlacementAdmissionCheck {
    pub const fn admits(&self) -> bool {
        self.mandatory_devices_available
            && self.per_device_memory_ok
            && self.required_kernels_available
            && self.transfers_feasible
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CrossDeviceConcurrencyContract {
    pub independent_device_execution: bool,
    pub dependencies_preserved: bool,
    pub resource_lifetime_preserved: bool,
}

impl CrossDeviceConcurrencyContract {
    pub const fn permits_concurrency(&self) -> bool {
        self.independent_device_execution
            && self.dependencies_preserved
            && self.resource_lifetime_preserved
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FailurePropagationDecision {
    pub upstream_failed: bool,
    pub downstream_stopped: bool,
    pub explicit_fallback: Option<MultiDevicePlacementPlanId>,
    pub structured_reason: MultiDevicePlacementErrorCode,
}

impl FailurePropagationDecision {
    pub const fn downstream_allowed(&self) -> bool {
        !self.upstream_failed || self.explicit_fallback.is_some()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplicaEvictionDecision {
    pub optional_replica: bool,
    pub authoritative_copy_remains: bool,
    pub no_in_flight_references: bool,
    pub dependent_plan_invalidated: bool,
}

impl ReplicaEvictionDecision {
    pub const fn can_evict(&self) -> bool {
        self.optional_replica && self.authoritative_copy_remains && self.no_in_flight_references
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JointKernelPlacementDecision {
    pub binding: PlacementBinding,
    pub kernel: KernelId,
    pub hard_eligible: bool,
    pub transfer_cost_micros: u64,
    pub memory_cost_bytes: u64,
}

impl JointKernelPlacementDecision {
    pub const fn selectable(&self) -> bool {
        self.hard_eligible
    }

    pub const fn total_cost(&self) -> u64 {
        self.transfer_cost_micros
            .saturating_add(self.memory_cost_bytes)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceSpecificTuningEvidence {
    pub device: DeviceBinding,
    pub kernel: KernelId,
    pub performance_context: String,
}

impl DeviceSpecificTuningEvidence {
    pub fn valid_for(&self, device: &DeviceBinding) -> Result<(), MultiDevicePlacementError> {
        validate_no_native_handle(&self.performance_context, "autotuning context")?;
        if &self.device == device {
            Ok(())
        } else {
            Err(MultiDevicePlacementError::KernelUnavailable)
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlacementPerformanceFeedback {
    pub plan: MultiDevicePlacementPlanId,
    pub device_context: DeviceBinding,
    pub segment_id: String,
    pub baseline_micros: u64,
    pub observed_micros: u64,
}

impl PlacementPerformanceFeedback {
    pub const fn regressed(&self) -> bool {
        self.observed_micros
            > self
                .baseline_micros
                .saturating_add(self.baseline_micros / 10)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PlacementPlanCacheKey {
    pub graph_fingerprint: MultiDevicePlacementFingerprint,
    pub model_instance_revision: u64,
    pub device_set_fingerprint: MultiDevicePlacementFingerprint,
    pub provider_versions: Vec<(ProviderBinding, String)>,
    pub memory_budget_class: String,
    pub workload_scope: String,
    pub placement_policy_version: String,
    pub partition_fingerprint: Option<MultiDevicePlacementFingerprint>,
}

impl PlacementPlanCacheKey {
    pub fn validate(&self) -> Result<(), MultiDevicePlacementError> {
        validate_logical_id(&self.memory_budget_class, "memory budget class")?;
        validate_logical_id(&self.workload_scope, "workload scope")?;
        validate_logical_id(&self.placement_policy_version, "placement policy version")?;
        for (_, version) in &self.provider_versions {
            validate_no_native_handle(version, "provider version")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PlacementPlanCache {
    plans: BTreeMap<PlacementPlanCacheKey, MultiDevicePlacementPlanId>,
    invalidated: BTreeSet<PlacementPlanCacheKey>,
}

impl PlacementPlanCache {
    pub fn insert(&mut self, key: PlacementPlanCacheKey, plan: MultiDevicePlacementPlanId) {
        self.invalidated.remove(&key);
        self.plans.insert(key, plan);
    }

    pub fn lookup(&self, key: &PlacementPlanCacheKey) -> Option<&MultiDevicePlacementPlanId> {
        if self.invalidated.contains(key) {
            None
        } else {
            self.plans.get(key)
        }
    }

    pub fn invalidate(&mut self, key: &PlacementPlanCacheKey) {
        self.plans.remove(key);
        self.invalidated.insert(key.clone());
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CachedPlanRevalidation {
    pub device_available: bool,
    pub provider_ready: bool,
    pub memory_capacity_ok: bool,
    pub peer_capability_ok: bool,
    pub kernel_available: bool,
    pub policy_ok: bool,
    pub resource_residency_ok: bool,
}

impl CachedPlanRevalidation {
    pub const fn valid(&self) -> bool {
        self.device_available
            && self.provider_ready
            && self.memory_capacity_ok
            && self.peer_capability_ok
            && self.kernel_available
            && self.policy_ok
            && self.resource_residency_ok
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeHandlePrivacyCheck {
    pub device_pointer: Option<String>,
    pub peer_handle: Option<String>,
    pub native_queue: Option<String>,
    pub os_handle: Option<String>,
}

impl NativeHandlePrivacyCheck {
    pub fn validate(&self) -> Result<(), MultiDevicePlacementError> {
        for (field, value) in [
            ("device pointer", &self.device_pointer),
            ("peer handle", &self.peer_handle),
            ("native queue", &self.native_queue),
            ("os handle", &self.os_handle),
        ] {
            if let Some(value) = value {
                validate_no_native_handle(value, field)?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentPlacementRequest {
    pub portable_requirements: BTreeSet<String>,
    pub requested_device: Option<DeviceBinding>,
    pub topology_authority: bool,
}

impl ComponentPlacementRequest {
    pub fn validate_wit_boundary(&self) -> Result<(), MultiDevicePlacementError> {
        if self.requested_device.is_some() || self.topology_authority {
            return Err(MultiDevicePlacementError::PolicyAuthorityViolation);
        }
        for requirement in &self.portable_requirements {
            validate_no_native_handle(requirement, "portable component requirement")?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RuntimePreference {
    LowLatency,
    Deterministic,
    MemoryConstrained,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeInferencePlacementRequest {
    pub preferences: BTreeSet<RuntimePreference>,
    pub layer_to_device_mapping: Vec<(u32, DeviceBinding)>,
    pub admin_policy_binding: Option<MultiDevicePlacementPlanId>,
}

impl RuntimeInferencePlacementRequest {
    pub const fn normal_request_allowed(&self) -> bool {
        self.layer_to_device_mapping.is_empty()
    }

    pub const fn admin_policy_allowed(&self) -> bool {
        self.admin_policy_binding.is_some()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PlacementObservationKind {
    PlanBuildStarted,
    PlacementCandidateEvaluated,
    PlacementCandidateExcluded,
    PlacementPlanReady,
    StagePlaced,
    TensorPartitionCreated,
    WeightReplicaCreated,
    WeightReplicaEvicted,
    CrossDeviceTransferStarted,
    CrossDeviceTransferCompleted,
    SessionPlacementBound,
    SessionPlacementMigrated,
    PlacementPlanStale,
    PlacementPlanInvalidated,
    PlacementRebuildRequested,
    DegradedPlacementActivated,
    DeviceLost,
    DeviceRecovered,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlacementObservation {
    pub kind: PlacementObservationKind,
    pub plan: Option<MultiDevicePlacementPlanId>,
    pub generation: Option<MultiDevicePlacementGeneration>,
    pub provider: Option<ProviderBinding>,
    pub device: Option<DeviceBinding>,
    pub stage_id: Option<String>,
    pub tensor_partition_id: Option<String>,
    pub movement_bytes: Option<u64>,
    pub reason: Option<MultiDevicePlacementErrorCode>,
    pub detail: Option<String>,
}

impl PlacementObservation {
    pub fn validate_redacted(&self) -> Result<(), MultiDevicePlacementError> {
        for (field, value) in [
            ("stage id", &self.stage_id),
            ("tensor partition id", &self.tensor_partition_id),
            ("observability detail", &self.detail),
        ] {
            if let Some(value) = value {
                validate_no_native_handle(value, field)?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultiDeviceConformanceReport {
    pub runtime_placement_authority: bool,
    pub component_cannot_force_device: bool,
    pub partition_validity: bool,
    pub replica_partition_distinct: bool,
    pub shard_not_full_tensor: bool,
    pub explicit_cross_device_movement: bool,
    pub host_staging_policy_preserved: bool,
    pub peer_capability_required: bool,
    pub per_device_memory_policy: bool,
    pub heterogeneous_device_support: bool,
    pub transfer_aware_selection: bool,
    pub exact_prepared_placement: bool,
    pub no_mid_flight_migration: bool,
    pub kv_locality: bool,
    pub explicit_session_migration: bool,
    pub device_loss_invalidation: bool,
    pub degraded_plan_validation: bool,
    pub recovery_lifecycle: bool,
    pub cache_revalidation: bool,
    pub handle_isolation: bool,
    pub observability_redaction: bool,
}

impl MultiDeviceConformanceReport {
    pub const fn passes(&self) -> bool {
        self.runtime_placement_authority
            && self.component_cannot_force_device
            && self.partition_validity
            && self.replica_partition_distinct
            && self.shard_not_full_tensor
            && self.explicit_cross_device_movement
            && self.host_staging_policy_preserved
            && self.peer_capability_required
            && self.per_device_memory_policy
            && self.heterogeneous_device_support
            && self.transfer_aware_selection
            && self.exact_prepared_placement
            && self.no_mid_flight_migration
            && self.kv_locality
            && self.explicit_session_migration
            && self.device_loss_invalidation
            && self.degraded_plan_validation
            && self.recovery_lifecycle
            && self.cache_revalidation
            && self.handle_isolation
            && self.observability_redaction
    }
}

pub const MULTI_DEVICE_PLACEMENT_DOCUMENTATION_TOPICS: &[&str] = &[
    "DeviceSet",
    "PlacementDomain",
    "MultiDevicePlacementPlan",
    "pipeline stages",
    "TensorPartitionDescriptor",
    "TensorShard",
    "weight replication",
    "KV locality",
    "Device failure and degraded plan",
    "local-only scope",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlacementCandidate {
    pub scope: PlacementScope,
    pub binding: PlacementBinding,
    pub required_kernel: Option<KernelId>,
    pub required_memory_bytes: u64,
    pub available_memory_bytes: u64,
    pub required_memory_class: KernelMemoryClass,
    pub provider_capable: bool,
    pub device_capable: bool,
    pub kernel_available: bool,
    pub resource_affinity_valid: bool,
    pub transfer_permitted: bool,
    pub host_staging_permitted: bool,
    pub latency_micros: u64,
    pub throughput_units: u64,
    pub transfer_cost_micros: u64,
    pub pressure: ProviderPressureLevel,
    pub stability_penalty_micros: u64,
}

impl PlacementCandidate {
    pub fn is_eligible(&self) -> bool {
        self.provider_capable
            && self.device_capable
            && self.kernel_available
            && self.required_memory_bytes <= self.available_memory_bytes
            && self.resource_affinity_valid
            && self.transfer_permitted
            && self.host_staging_permitted
    }

    pub fn cost(&self) -> u64 {
        let pressure_penalty = match self.pressure {
            ProviderPressureLevel::Unknown | ProviderPressureLevel::Low => 0,
            ProviderPressureLevel::Moderate => 25,
            ProviderPressureLevel::High => 100,
            ProviderPressureLevel::Saturated => 1_000_000,
        };
        self.latency_micros
            .saturating_add(self.transfer_cost_micros)
            .saturating_add(self.stability_penalty_micros)
            .saturating_add(pressure_penalty)
            .saturating_sub(self.throughput_units.min(1_000))
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PlacementDecisionReport {
    pub evaluated: usize,
    pub rejected: Vec<PlacementRejection>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlacementRejection {
    pub device: DeviceBinding,
    pub reason: MultiDevicePlacementErrorCode,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultiDevicePlacementPlan {
    pub id: MultiDevicePlacementPlanId,
    pub generation: MultiDevicePlacementGeneration,
    pub state: MultiDevicePlacementState,
    pub graph_fingerprint: MultiDevicePlacementFingerprint,
    pub model_instance_revision: u64,
    pub device_set: DeviceSet,
    pub provider_versions: BTreeMap<ProviderBinding, String>,
    pub memory_capacity_class: String,
    pub placement_policy_version: String,
    pub partition_fingerprint: Option<MultiDevicePlacementFingerprint>,
    pub bindings: Vec<(PlacementScope, PlacementBinding)>,
    pub stages: Vec<PipelineStage>,
    pub movement_edges: Vec<StageMovementEdge>,
    pub guards: BTreeSet<String>,
}

impl MultiDevicePlacementPlan {
    pub fn new(
        id: MultiDevicePlacementPlanId,
        generation: MultiDevicePlacementGeneration,
        graph_fingerprint: MultiDevicePlacementFingerprint,
        model_instance_revision: u64,
        device_set: DeviceSet,
        placement_policy_version: impl Into<String>,
    ) -> Result<Self, MultiDevicePlacementError> {
        if generation.value() == 0 {
            return Err(MultiDevicePlacementError::GenerationZero);
        }
        let placement_policy_version = placement_policy_version.into();
        validate_logical_id(&placement_policy_version, "placement policy version")?;
        Ok(Self {
            id,
            generation,
            state: MultiDevicePlacementState::Building,
            graph_fingerprint,
            model_instance_revision,
            device_set,
            provider_versions: BTreeMap::new(),
            memory_capacity_class: "unspecified".into(),
            placement_policy_version,
            partition_fingerprint: None,
            bindings: Vec::new(),
            stages: Vec::new(),
            movement_edges: Vec::new(),
            guards: BTreeSet::new(),
        })
    }

    pub fn fingerprint(&self) -> MultiDevicePlacementFingerprint {
        let mut lines = vec![
            format!("plan={}", self.id),
            format!("generation={}", self.generation.value()),
            format!("graph={}", self.graph_fingerprint),
            format!("model-revision={}", self.model_instance_revision),
            format!("device-set={}", self.device_set.fingerprint()),
            format!("memory-capacity-class={}", self.memory_capacity_class),
            format!("policy={}", self.placement_policy_version),
        ];
        for (provider, version) in &self.provider_versions {
            lines.push(format!("provider={provider};version={version}"));
        }
        if let Some(partition) = &self.partition_fingerprint {
            lines.push(format!("partition={partition}"));
        }
        for (scope, binding) in &self.bindings {
            lines.push(format!(
                "binding={:?};provider={};device={};memory={:?}",
                scope.granularity, binding.provider, binding.device, binding.memory_domain
            ));
        }
        sha256_fingerprint(&lines.join("\n"))
    }

    pub fn add_binding(
        &mut self,
        scope: PlacementScope,
        binding: PlacementBinding,
    ) -> Result<(), MultiDevicePlacementError> {
        if !self.device_set.contains_device(&binding.device) {
            return Err(MultiDevicePlacementError::DeviceOutsideSet(binding.device));
        }
        self.bindings.push((scope, binding));
        Ok(())
    }

    pub fn transition_to(
        &mut self,
        next: MultiDevicePlacementState,
    ) -> Result<(), MultiDevicePlacementError> {
        if !self.state.can_transition_to(next) {
            return Err(MultiDevicePlacementError::InvalidStateTransition {
                from: self.state,
                to: next,
            });
        }
        self.state = next;
        Ok(())
    }

    pub fn mark_ready(&mut self) -> Result<(), MultiDevicePlacementError> {
        if self.bindings.is_empty() && self.stages.is_empty() {
            return Err(MultiDevicePlacementError::NoPlacementBindings);
        }
        self.transition_to(MultiDevicePlacementState::Validating)?;
        self.transition_to(MultiDevicePlacementState::Ready)
    }
}

pub fn select_lowest_cost_eligible(
    candidates: impl IntoIterator<Item = PlacementCandidate>,
) -> Result<(PlacementCandidate, PlacementDecisionReport), MultiDevicePlacementError> {
    let mut report = PlacementDecisionReport::default();
    let mut eligible = Vec::new();
    for candidate in candidates {
        report.evaluated += 1;
        if candidate.is_eligible() {
            eligible.push(candidate);
        } else {
            report.rejected.push(PlacementRejection {
                device: candidate.binding.device.clone(),
                reason: candidate_rejection_reason(&candidate),
            });
        }
    }
    eligible
        .into_iter()
        .min_by_key(PlacementCandidate::cost)
        .map(|candidate| (candidate, report))
        .ok_or(MultiDevicePlacementError::NoFeasiblePlan)
}

fn candidate_rejection_reason(candidate: &PlacementCandidate) -> MultiDevicePlacementErrorCode {
    if !candidate.provider_capable {
        MultiDevicePlacementErrorCode::ProviderIncompatible
    } else if !candidate.device_capable {
        MultiDevicePlacementErrorCode::DeviceIncompatible
    } else if !candidate.kernel_available {
        MultiDevicePlacementErrorCode::KernelUnavailable
    } else if candidate.required_memory_bytes > candidate.available_memory_bytes {
        MultiDevicePlacementErrorCode::MemoryInfeasible
    } else if !candidate.resource_affinity_valid {
        MultiDevicePlacementErrorCode::AffinityInvalid
    } else if !candidate.transfer_permitted {
        MultiDevicePlacementErrorCode::TransferDenied
    } else if !candidate.host_staging_permitted {
        MultiDevicePlacementErrorCode::HostStagingDenied
    } else {
        MultiDevicePlacementErrorCode::NoFeasiblePlan
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MultiDevicePlacementErrorCode {
    PlacementDisabled,
    PolicyInvalid,
    NoDevices,
    NoFeasiblePlan,
    DeviceIncompatible,
    ProviderIncompatible,
    MemoryInfeasible,
    KernelUnavailable,
    AffinityInvalid,
    TransferDenied,
    PeerAccessUnavailable,
    PeerTransferUnavailable,
    HostStagingDenied,
    PlanStale,
    PlanInvalidated,
    PlanBuildFailed,
    PlanCacheStale,
    TensorPartitionInvalid,
    TensorPartitionAxisInvalid,
    TensorPartitionGap,
    TensorPartitionOverlap,
    TensorPartitionShardMissing,
    TensorPartitionShardIncompatible,
    TensorPartitionKernelUnsupported,
    KvPlacementInvalid,
    KvMigrationFailed,
    StageInvalid,
    StageDependencyInvalid,
    StageTransferFailed,
    DeviceLost,
    DegradedPlanUnavailable,
    ReplacementFailed,
    Internal,
}

impl MultiDevicePlacementErrorCode {
    pub const fn id(self) -> &'static str {
        match self {
            Self::PlacementDisabled => "multi-device-placement-disabled",
            Self::PolicyInvalid => "multi-device-placement-policy-invalid",
            Self::NoDevices => "multi-device-placement-no-devices",
            Self::NoFeasiblePlan => "multi-device-placement-no-feasible-plan",
            Self::DeviceIncompatible => "multi-device-placement-device-incompatible",
            Self::ProviderIncompatible => "multi-device-placement-provider-incompatible",
            Self::MemoryInfeasible => "multi-device-placement-memory-infeasible",
            Self::KernelUnavailable => "multi-device-placement-kernel-unavailable",
            Self::AffinityInvalid => "multi-device-placement-affinity-invalid",
            Self::TransferDenied => "multi-device-placement-transfer-denied",
            Self::PeerAccessUnavailable => "multi-device-placement-peer-access-unavailable",
            Self::PeerTransferUnavailable => "multi-device-placement-peer-transfer-unavailable",
            Self::HostStagingDenied => "multi-device-placement-host-staging-denied",
            Self::PlanStale => "multi-device-placement-plan-stale",
            Self::PlanInvalidated => "multi-device-placement-plan-invalidated",
            Self::PlanBuildFailed => "multi-device-placement-plan-build-failed",
            Self::PlanCacheStale => "multi-device-placement-plan-cache-stale",
            Self::TensorPartitionInvalid => "tensor-partition-invalid",
            Self::TensorPartitionAxisInvalid => "tensor-partition-axis-invalid",
            Self::TensorPartitionGap => "tensor-partition-gap",
            Self::TensorPartitionOverlap => "tensor-partition-overlap",
            Self::TensorPartitionShardMissing => "tensor-partition-shard-missing",
            Self::TensorPartitionShardIncompatible => "tensor-partition-shard-incompatible",
            Self::TensorPartitionKernelUnsupported => "tensor-partition-kernel-unsupported",
            Self::KvPlacementInvalid => "multi-device-kv-placement-invalid",
            Self::KvMigrationFailed => "multi-device-kv-migration-failed",
            Self::StageInvalid => "multi-device-stage-invalid",
            Self::StageDependencyInvalid => "multi-device-stage-dependency-invalid",
            Self::StageTransferFailed => "multi-device-stage-transfer-failed",
            Self::DeviceLost => "multi-device-device-lost",
            Self::DegradedPlanUnavailable => "multi-device-degraded-plan-unavailable",
            Self::ReplacementFailed => "multi-device-replacement-failed",
            Self::Internal => "internal-multi-device-placement-error",
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum MultiDevicePlacementError {
    InvalidIdentity {
        field: &'static str,
        value: String,
    },
    InvalidFingerprint {
        field: &'static str,
    },
    NativeHandleForbidden {
        field: &'static str,
    },
    GenerationZero,
    NoDevices,
    DuplicateDevice(DeviceBinding),
    ProviderDeviceMismatch,
    DeviceOutsideSet(DeviceBinding),
    CrossProviderDenied,
    MemoryDomainDeviceMismatch,
    InvalidLayerRange,
    EmptyPlacementScope,
    MovementNotRequired,
    NoPlacementBindings,
    NoFeasiblePlan,
    WeightPlacementInvalid,
    ReplicaIncompatible,
    PartitionInvalid,
    PartitionMissingShard,
    PartitionGap,
    PartitionOverlap,
    PartitionOverflow,
    ShardAsFullTensorUnsupported,
    PeerTransferUnavailable,
    HostStagingDenied,
    KernelUnavailable,
    MemoryBudgetOverflow,
    MemoryBudgetExceeded,
    AllocationSlotDeviceMismatch,
    UnspecifiedGlobalPoolUse,
    PlacementPinInvalid,
    KvPlacementInvalid,
    ImplicitCollective,
    PolicyAuthorityViolation,
    PlacementInvalidated(PlacementInvalidationReason),
    DegradedPlanUnavailable,
    ReplacementFailed,
    DeviceLost(DeviceBinding),
    InvalidStateTransition {
        from: MultiDevicePlacementState,
        to: MultiDevicePlacementState,
    },
}

impl MultiDevicePlacementError {
    pub const fn code(&self) -> MultiDevicePlacementErrorCode {
        match self {
            Self::NoDevices => MultiDevicePlacementErrorCode::NoDevices,
            Self::CrossProviderDenied => MultiDevicePlacementErrorCode::TransferDenied,
            Self::NoFeasiblePlan => MultiDevicePlacementErrorCode::NoFeasiblePlan,
            Self::WeightPlacementInvalid | Self::ReplicaIncompatible => {
                MultiDevicePlacementErrorCode::PolicyInvalid
            }
            Self::PartitionInvalid
            | Self::PartitionMissingShard
            | Self::PartitionGap
            | Self::PartitionOverlap
            | Self::PartitionOverflow
            | Self::ShardAsFullTensorUnsupported => MultiDevicePlacementErrorCode::PolicyInvalid,
            Self::PeerTransferUnavailable => MultiDevicePlacementErrorCode::PeerTransferUnavailable,
            Self::HostStagingDenied => MultiDevicePlacementErrorCode::HostStagingDenied,
            Self::KernelUnavailable => MultiDevicePlacementErrorCode::KernelUnavailable,
            Self::MemoryBudgetOverflow | Self::MemoryBudgetExceeded => {
                MultiDevicePlacementErrorCode::MemoryInfeasible
            }
            Self::AllocationSlotDeviceMismatch | Self::UnspecifiedGlobalPoolUse => {
                MultiDevicePlacementErrorCode::PolicyInvalid
            }
            Self::PlacementPinInvalid | Self::PolicyAuthorityViolation => {
                MultiDevicePlacementErrorCode::PolicyInvalid
            }
            Self::KvPlacementInvalid => MultiDevicePlacementErrorCode::KvPlacementInvalid,
            Self::ImplicitCollective => MultiDevicePlacementErrorCode::TensorPartitionInvalid,
            Self::PlacementInvalidated(_) => MultiDevicePlacementErrorCode::PlanInvalidated,
            Self::DegradedPlanUnavailable => MultiDevicePlacementErrorCode::DegradedPlanUnavailable,
            Self::ReplacementFailed => MultiDevicePlacementErrorCode::ReplacementFailed,
            Self::DeviceLost(_) => MultiDevicePlacementErrorCode::DeviceLost,
            Self::DeviceOutsideSet(_) | Self::MemoryDomainDeviceMismatch => {
                MultiDevicePlacementErrorCode::DeviceIncompatible
            }
            Self::NativeHandleForbidden { .. } => MultiDevicePlacementErrorCode::Internal,
            _ => MultiDevicePlacementErrorCode::PolicyInvalid,
        }
    }
}

impl fmt::Display for MultiDevicePlacementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIdentity { field, value } => {
                write!(f, "invalid {field} '{value}'")
            }
            Self::InvalidFingerprint { field } => write!(f, "invalid {field}"),
            Self::NativeHandleForbidden { field } => {
                write!(f, "{field} must not contain native handles")
            }
            Self::GenerationZero => write!(f, "placement generation must be non-zero"),
            Self::NoDevices => write!(f, "device set must contain at least one Device"),
            Self::DuplicateDevice(device) => write!(f, "duplicate Device '{device}'"),
            Self::ProviderDeviceMismatch => {
                write!(f, "Device provider metadata does not match binding")
            }
            Self::DeviceOutsideSet(device) => write!(f, "Device '{device}' is outside DeviceSet"),
            Self::CrossProviderDenied => {
                write!(f, "cross-Provider placement is denied by domain policy")
            }
            Self::MemoryDomainDeviceMismatch => {
                write!(f, "memory domain Device does not match placement binding")
            }
            Self::InvalidLayerRange => write!(f, "layer range start must be <= end"),
            Self::EmptyPlacementScope => write!(f, "placement scope must include work"),
            Self::MovementNotRequired => write!(f, "stage movement requires distinct Devices"),
            Self::NoPlacementBindings => write!(f, "placement plan has no bindings or stages"),
            Self::NoFeasiblePlan => write!(f, "no feasible multi-Device placement plan exists"),
            Self::WeightPlacementInvalid => write!(f, "weight placement is invalid"),
            Self::ReplicaIncompatible => write!(f, "weight replica is incompatible"),
            Self::PartitionInvalid => write!(f, "tensor partition is invalid"),
            Self::PartitionMissingShard => write!(f, "tensor partition is missing shards"),
            Self::PartitionGap => write!(f, "tensor partition has a gap"),
            Self::PartitionOverlap => write!(f, "tensor partition has an illegal overlap"),
            Self::PartitionOverflow => write!(f, "tensor partition size arithmetic overflowed"),
            Self::ShardAsFullTensorUnsupported => {
                write!(f, "kernel cannot consume TensorShard as a full Tensor")
            }
            Self::PeerTransferUnavailable => {
                write!(f, "peer transfer is unavailable for Device pair")
            }
            Self::HostStagingDenied => write!(f, "host staging is denied by placement policy"),
            Self::KernelUnavailable => write!(f, "required kernel is unavailable"),
            Self::MemoryBudgetOverflow => write!(f, "Device memory budget overflowed"),
            Self::MemoryBudgetExceeded => write!(f, "Device memory budget exceeded capacity"),
            Self::AllocationSlotDeviceMismatch => {
                write!(f, "allocation slot is bound to a different Device")
            }
            Self::UnspecifiedGlobalPoolUse => {
                write!(
                    f,
                    "multi-Device allocation cannot use an unspecified global pool"
                )
            }
            Self::PlacementPinInvalid => {
                write!(
                    f,
                    "placement pin cannot bypass compatibility or Device availability"
                )
            }
            Self::KvPlacementInvalid => write!(f, "KV placement is invalid"),
            Self::ImplicitCollective => {
                write!(f, "partition contract cannot invent implicit collectives")
            }
            Self::PolicyAuthorityViolation => {
                write!(f, "caller cannot authoritatively select concrete Devices")
            }
            Self::PlacementInvalidated(reason) => {
                write!(f, "placement invalidated by {reason:?}")
            }
            Self::DegradedPlanUnavailable => write!(f, "degraded placement plan is unavailable"),
            Self::ReplacementFailed => write!(f, "replacement placement plan failed"),
            Self::DeviceLost(device) => write!(f, "Device '{device}' was lost"),
            Self::InvalidStateTransition { from, to } => {
                write!(
                    f,
                    "invalid placement state transition from {from:?} to {to:?}"
                )
            }
        }
    }
}

impl Error for MultiDevicePlacementError {}

fn validate_logical_id(value: &str, field: &'static str) -> Result<(), MultiDevicePlacementError> {
    if value.trim().is_empty()
        || value.len() > 128
        || value.trim() != value
        || value.contains('/')
        || value.contains('\\')
        || value.contains(':')
        || contains_native_handle_marker(value)
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    {
        return Err(MultiDevicePlacementError::InvalidIdentity {
            field,
            value: value.into(),
        });
    }
    Ok(())
}

fn validate_fingerprint(value: &str, field: &'static str) -> Result<(), MultiDevicePlacementError> {
    let Some(digest) = value.strip_prefix("sha256:") else {
        return Err(MultiDevicePlacementError::InvalidFingerprint { field });
    };
    if digest.len() < 8 || contains_native_handle_marker(value) {
        return Err(MultiDevicePlacementError::InvalidFingerprint { field });
    }
    Ok(())
}

fn validate_no_native_handle(
    value: &str,
    field: &'static str,
) -> Result<(), MultiDevicePlacementError> {
    if contains_native_handle_marker(value) {
        return Err(MultiDevicePlacementError::NativeHandleForbidden { field });
    }
    Ok(())
}

fn sha256_fingerprint(value: &str) -> MultiDevicePlacementFingerprint {
    let digest = Sha256::digest(value.as_bytes());
    let mut encoded = String::with_capacity("sha256:".len() + digest.len() * 2);
    encoded.push_str("sha256:");
    for byte in digest {
        encoded.push_str(&format!("{byte:02x}"));
    }
    MultiDevicePlacementFingerprint(encoded)
}

fn contains_native_handle_marker(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    [
        "0x",
        "ptr",
        "pointer",
        "cuda ipc",
        "dma-buf",
        "dmabuf",
        "fd=",
        "native handle",
        "native queue",
        "peer handle",
        "os handle",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CapabilityVersion, ComputeDType, DeviceId, DeviceType, KernelImplementationFamily,
        KernelOperatorVersionRange, OperatorFamily, OperatorId,
    };

    fn device(name: &str, provider: &str, memory_capacity: u64) -> DeviceSetMember {
        let mut metadata =
            DeviceMetadata::new(DeviceId::new(name), name, DeviceType::Gpu, provider);
        metadata.architecture = format!("arch-{name}");
        metadata.memory_capacity = memory_capacity;
        metadata
            .memory_class_support
            .insert(KernelMemoryClass::Device);
        DeviceSetMember::new(metadata, DeviceAvailability::Available)
    }

    fn binding(provider: &str, name: &str) -> PlacementBinding {
        let device = DeviceBinding::new(DeviceId::new(name));
        PlacementBinding::new(
            ProviderBinding::new(provider),
            device.clone(),
            MemoryDomain::DeviceLocal(device),
        )
        .unwrap()
    }

    fn shard(name: &str, parent: &str, start: u64, end: u64, device_name: &str) -> TensorShard {
        TensorShard::new(
            TensorShardId::new(name).unwrap(),
            TensorResourceId::new(parent),
            LogicalRange::new(TensorPartitionAxis::Hidden, start, end),
            ShapeDescriptor::new([end - start]),
            DTypeDescriptor::portable(ComputeDType::Float32),
            TensorLayoutKind::Contiguous,
            binding("cuda", device_name),
        )
    }

    fn test_kernel(name: &str) -> KernelId {
        KernelId::new(
            ProviderBinding::new("cuda"),
            name,
            CapabilityVersion::new(1, 0, 0),
            OperatorId::magnetar(name, 1, OperatorFamily::Attention),
            KernelOperatorVersionRange::exact(1),
            KernelImplementationFamily::Cuda,
        )
    }

    #[test]
    fn device_set_supports_single_multi_provider_and_heterogeneous_fingerprint() {
        let single = DeviceSet::new(
            DeviceSetId::new("single").unwrap(),
            [device("gpu0", "cuda", 24)],
        )
        .unwrap();
        assert_eq!(single.members().len(), 1);

        let multi = DeviceSet::new(
            DeviceSetId::new("heterogeneous").unwrap(),
            [
                device("gpu0", "cuda", 24),
                device("gpu1", "cuda", 8),
                device("cpu0", "cpu", 64),
            ],
        )
        .unwrap();
        assert_eq!(multi.providers().len(), 2);
        assert_ne!(single.fingerprint(), multi.fingerprint());

        assert!(matches!(
            DeviceSet::new(
                DeviceSetId::new("dup").unwrap(),
                [device("gpu0", "cuda", 24), device("gpu0", "cuda", 24)]
            ),
            Err(MultiDevicePlacementError::DuplicateDevice(_))
        ));
    }

    #[test]
    fn placement_domain_and_binding_reject_cross_provider_and_native_handles() {
        let set = DeviceSet::new(
            DeviceSetId::new("set").unwrap(),
            [device("gpu0", "cuda", 24), device("cpu0", "cpu", 64)],
        )
        .unwrap();
        let domain = PlacementDomain::new(
            "local",
            [
                DeviceBinding::new(DeviceId::new("gpu0")),
                DeviceBinding::new(DeviceId::new("cpu0")),
            ],
        )
        .unwrap();
        assert!(matches!(
            domain.validate_against(&set),
            Err(MultiDevicePlacementError::CrossProviderDenied)
        ));
        assert!(
            domain
                .with_cross_provider(true)
                .validate_against(&set)
                .is_ok()
        );
        assert!(
            binding("cuda", "gpu0")
                .with_constraint("cuda pointer 0xdeadbeef")
                .is_err()
        );
    }

    #[test]
    fn placement_plan_identity_includes_graph_revision_devices_providers_policy_and_partitions() {
        let set = DeviceSet::new(
            DeviceSetId::new("set").unwrap(),
            [device("gpu0", "cuda", 24), device("gpu1", "cuda", 24)],
        )
        .unwrap();
        let mut plan = MultiDevicePlacementPlan::new(
            MultiDevicePlacementPlanId::new("plan").unwrap(),
            MultiDevicePlacementGeneration::new(1),
            MultiDevicePlacementFingerprint::new("sha256:graph0001").unwrap(),
            7,
            set,
            "policy-v1",
        )
        .unwrap();
        plan.provider_versions
            .insert(ProviderBinding::new("cuda"), "12.4".into());
        plan.memory_capacity_class = "24gib".into();
        plan.partition_fingerprint =
            Some(MultiDevicePlacementFingerprint::new("sha256:partition0001").unwrap());
        plan.add_binding(
            PlacementScope::layer_range(0, 15).unwrap(),
            binding("cuda", "gpu0"),
        )
        .unwrap();
        let first = plan.fingerprint();
        plan.model_instance_revision = 8;
        assert_ne!(first, plan.fingerprint());
    }

    #[test]
    fn placement_scope_supports_model_segment_layer_operator_group_and_reserved_operator() {
        assert_eq!(
            PlacementScope::model_instance(ModelInstanceId::new("model-a").unwrap()).granularity,
            PlacementGranularity::ModelInstance
        );
        assert_eq!(
            PlacementScope::graph_segment([ExecutionNodeId::new("node0")])
                .unwrap()
                .granularity,
            PlacementGranularity::GraphSegment
        );
        assert_eq!(
            PlacementScope::layer_range(0, 7).unwrap().granularity,
            PlacementGranularity::LayerRange
        );
        assert_eq!(
            PlacementScope::operator_group("attention")
                .unwrap()
                .granularity,
            PlacementGranularity::OperatorGroup
        );
        assert!(
            PlacementGranularity::IndividualOperatorReserved > PlacementGranularity::OperatorGroup
        );
    }

    #[test]
    fn eligibility_applies_hard_filters_before_transfer_aware_ranking() {
        let scope = PlacementScope::layer_range(0, 1).unwrap();
        let fast_but_impossible = PlacementCandidate {
            scope: scope.clone(),
            binding: binding("cuda", "gpu1"),
            required_kernel: None,
            required_memory_bytes: 32,
            available_memory_bytes: 16,
            required_memory_class: KernelMemoryClass::Device,
            provider_capable: true,
            device_capable: true,
            kernel_available: true,
            resource_affinity_valid: true,
            transfer_permitted: true,
            host_staging_permitted: true,
            latency_micros: 1,
            throughput_units: 1_000,
            transfer_cost_micros: 0,
            pressure: ProviderPressureLevel::Low,
            stability_penalty_micros: 0,
        };
        let slower_without_transfer = PlacementCandidate {
            required_memory_bytes: 8,
            available_memory_bytes: 16,
            latency_micros: 40,
            transfer_cost_micros: 0,
            binding: binding("cuda", "gpu0"),
            scope,
            ..fast_but_impossible.clone()
        };

        let (selected, report) =
            select_lowest_cost_eligible([fast_but_impossible, slower_without_transfer]).unwrap();
        assert_eq!(
            selected.binding.device,
            DeviceBinding::new(DeviceId::new("gpu0"))
        );
        assert_eq!(
            report.rejected[0].reason,
            MultiDevicePlacementErrorCode::MemoryInfeasible
        );
    }

    #[test]
    fn pipeline_stages_preserve_order_and_create_explicit_cross_device_movement() {
        let gpu0 = binding("cuda", "gpu0");
        let gpu1 = binding("cuda", "gpu1");
        let stage0 =
            PipelineStage::new("stage0", [ExecutionNodeId::new("block0")], gpu0.clone(), 0)
                .unwrap()
                .with_output(TensorResourceId::new("activation"));
        let stage1 =
            PipelineStage::new("stage1", [ExecutionNodeId::new("block1")], gpu1.clone(), 1)
                .unwrap()
                .with_input(TensorResourceId::new("activation"));
        let movement = StageMovementEdge::new(
            &stage0.stage_id,
            &stage1.stage_id,
            TensorResourceId::new("activation"),
            gpu0,
            gpu1,
            HostStagingPolicy::Forbid,
        )
        .unwrap();

        assert!(stage0.order < stage1.order);
        assert!(
            stage0
                .output_requirements
                .contains(&TensorResourceId::new("activation"))
        );
        assert!(
            stage1
                .input_requirements
                .contains(&TensorResourceId::new("activation"))
        );
        assert!(movement.preserves_source_lifetime);
        assert!(movement.preserves_destination_readiness);
        assert_eq!(movement.host_staging_policy, HostStagingPolicy::Forbid);
    }

    #[test]
    fn pipeline_overlap_requires_memory_dependencies_and_scheduler_policy() {
        assert!(PipelineOverlapPolicy::independent().permits_overlap());
        assert!(
            !PipelineOverlapPolicy {
                memory_capacity_ok: false,
                ..PipelineOverlapPolicy::independent()
            }
            .permits_overlap()
        );
        assert!(
            !PipelineOverlapPolicy {
                dependencies_satisfied: false,
                ..PipelineOverlapPolicy::independent()
            }
            .permits_overlap()
        );
        assert!(
            !PipelineOverlapPolicy {
                scheduler_policy_allows: false,
                ..PipelineOverlapPolicy::independent()
            }
            .permits_overlap()
        );
    }

    #[test]
    fn weight_placement_supports_single_partition_replica_and_hybrid_relationships() {
        let dtype = DTypeDescriptor::portable(ComputeDType::Float32);
        let partition = TensorPartitionDescriptor::new(
            TensorResourceId::new("wq"),
            TensorPartitionAxis::Hidden,
            2,
            TensorReconstructionPolicy::LogicalOnly,
        )
        .unwrap()
        .with_shard(shard("wq-0", "wq", 0, 4096, "gpu0"))
        .with_shard(shard("wq-1", "wq", 4096, 8192, "gpu1"));
        let replica = WeightReplica::new(
            "artifact-wq",
            "r1",
            dtype.clone(),
            TensorLayoutKind::Contiguous,
            binding("cuda", "gpu0"),
        )
        .unwrap();

        WeightPlacement::new(
            "artifact-wq",
            "r1",
            dtype.clone(),
            TensorLayoutKind::Contiguous,
            WeightPlacementKind::SingleDevice,
            [DeviceBinding::new(DeviceId::new("gpu0"))],
        )
        .unwrap()
        .validate()
        .unwrap();
        WeightPlacement::new(
            "artifact-wq",
            "r1",
            dtype.clone(),
            TensorLayoutKind::Contiguous,
            WeightPlacementKind::Partitioned,
            [
                DeviceBinding::new(DeviceId::new("gpu0")),
                DeviceBinding::new(DeviceId::new("gpu1")),
            ],
        )
        .unwrap()
        .with_partition(partition.clone())
        .validate()
        .unwrap();
        WeightPlacement::new(
            "artifact-wq",
            "r1",
            dtype,
            TensorLayoutKind::Contiguous,
            WeightPlacementKind::Hybrid,
            [
                DeviceBinding::new(DeviceId::new("gpu0")),
                DeviceBinding::new(DeviceId::new("gpu1")),
            ],
        )
        .unwrap()
        .with_partition(partition)
        .with_replica(replica)
        .validate()
        .unwrap();
    }

    #[test]
    fn tensor_partition_validates_bounds_gaps_overlaps_replicas_and_collective_boundary() {
        let complete = TensorPartitionDescriptor::new(
            TensorResourceId::new("hidden"),
            TensorPartitionAxis::Hidden,
            2,
            TensorReconstructionPolicy::LogicalOnly,
        )
        .unwrap()
        .with_shard(shard("hidden-0", "hidden", 0, 4, "gpu0"))
        .with_shard(shard("hidden-1", "hidden", 4, 8, "gpu1"));
        assert!(complete.validate().is_ok());
        assert!(!complete.implies_collective());

        let gap = TensorPartitionDescriptor::new(
            TensorResourceId::new("hidden"),
            TensorPartitionAxis::Hidden,
            2,
            TensorReconstructionPolicy::LogicalOnly,
        )
        .unwrap()
        .with_shard(shard("hidden-0", "hidden", 0, 4, "gpu0"))
        .with_shard(shard("hidden-1", "hidden", 5, 8, "gpu1"));
        assert!(matches!(
            gap.validate(),
            Err(MultiDevicePlacementError::PartitionGap)
        ));

        let overlap = TensorPartitionDescriptor::new(
            TensorResourceId::new("hidden"),
            TensorPartitionAxis::Hidden,
            2,
            TensorReconstructionPolicy::LogicalOnly,
        )
        .unwrap()
        .with_shard(shard("hidden-0", "hidden", 0, 4, "gpu0"))
        .with_shard(shard("hidden-1", "hidden", 3, 8, "gpu1"));
        assert!(matches!(
            overlap.validate(),
            Err(MultiDevicePlacementError::PartitionOverlap)
        ));

        let replicated = overlap.allow_replicas();
        assert!(replicated.validate().is_ok());
    }

    #[test]
    fn kernel_partition_compatibility_rejects_unsupported_shard_as_full_tensor() {
        let descriptor = TensorPartitionDescriptor::new(
            TensorResourceId::new("hidden"),
            TensorPartitionAxis::Hidden,
            2,
            TensorReconstructionPolicy::ExplicitMaterializationRequired,
        )
        .unwrap()
        .with_shard(shard("hidden-0", "hidden", 0, 4, "gpu0"))
        .with_shard(shard("hidden-1", "hidden", 4, 8, "gpu1"));
        let supported = KernelPartitionCompatibility {
            kernel: test_kernel("partitioned-attention"),
            input_axes: [TensorPartitionAxis::Hidden].into_iter().collect(),
            output_axes: [TensorPartitionAxis::Hidden].into_iter().collect(),
        };
        let unsupported = KernelPartitionCompatibility {
            kernel: test_kernel("full-attention"),
            input_axes: BTreeSet::new(),
            output_axes: BTreeSet::new(),
        };

        supported.validate_input(&descriptor).unwrap();
        supported.validate_output(&descriptor).unwrap();
        assert!(matches!(
            unsupported.validate_input(&descriptor),
            Err(MultiDevicePlacementError::ShardAsFullTensorUnsupported)
        ));
    }

    #[test]
    fn peer_transfer_keeps_movement_explicit_metrics_redacted_and_policy_checked() {
        let movement = StageMovementEdge::new(
            "stage0",
            "stage1",
            TensorResourceId::new("activation"),
            binding("cuda", "gpu0"),
            binding("cuda", "gpu1"),
            HostStagingPolicy::Forbid,
        )
        .unwrap();
        let capability = DevicePairTransferCapability {
            bandwidth_class: "nvlink-class".into(),
            ..DevicePairTransferCapability::new(
                ProviderBinding::new("cuda"),
                DeviceBinding::new(DeviceId::new("gpu0")),
                DeviceBinding::new(DeviceId::new("gpu1")),
                [
                    DevicePairAccessMode::PeerRead,
                    DevicePairAccessMode::PeerCopy,
                ],
            )
        };

        let plan = DeviceTransferPlan::new(movement.clone(), 4096, Some(&capability)).unwrap();
        assert_eq!(plan.kind, DeviceTransferKind::DirectPeer);
        assert_eq!(plan.expected_bytes, 4096);
        assert_eq!(plan.peer_bandwidth_class.as_deref(), Some("nvlink-class"));
        assert_eq!(plan.host_staging_cost_micros, 0);

        let staged = DevicePairTransferCapability {
            requires_host_staging: true,
            ..capability
        };
        assert!(matches!(
            DeviceTransferPlan::new(movement, 4096, Some(&staged)),
            Err(MultiDevicePlacementError::HostStagingDenied)
        ));
    }

    #[test]
    fn cross_provider_transfer_requires_explicit_boundary_and_no_native_handles() {
        let movement = StageMovementEdge::new(
            "stage0",
            "stage1",
            TensorResourceId::new("activation"),
            binding("cuda", "gpu0"),
            binding("cpu", "cpu0"),
            HostStagingPolicy::Permit,
        )
        .unwrap();
        let transfer = DeviceTransferPlan::new(movement, 1024, None).unwrap();
        assert_eq!(transfer.kind, DeviceTransferKind::CrossProviderBoundary);
        assert!(transfer.host_staging_cost_micros > 0);

        let native = DevicePairTransferCapability {
            bandwidth_class: "cuda ipc 0xdeadbeef".into(),
            ..DevicePairTransferCapability::new(
                ProviderBinding::new("cuda"),
                DeviceBinding::new(DeviceId::new("gpu0")),
                DeviceBinding::new(DeviceId::new("gpu1")),
                [DevicePairAccessMode::PeerCopy],
            )
        };
        assert!(matches!(
            native.validate(),
            Err(MultiDevicePlacementError::NativeHandleForbidden { .. })
        ));
    }

    #[test]
    fn per_device_memory_budget_accounts_all_runtime_owned_classes() {
        let mut budget = DeviceMemoryBudget::new(DeviceBinding::new(DeviceId::new("gpu0")), 1024);
        budget.weights_bytes = 256;
        budget.kv_bytes = 128;
        budget.workspace_bytes = 128;
        budget.transient_bytes = 64;
        budget.transfer_buffer_bytes = 64;
        budget.reserved_headroom_bytes = 64;
        assert_eq!(budget.used_bytes().unwrap(), 704);
        budget.validate().unwrap();

        budget.transfer_buffer_bytes = 512;
        assert!(matches!(
            budget.validate(),
            Err(MultiDevicePlacementError::MemoryBudgetExceeded)
        ));
    }

    #[test]
    fn device_pool_binding_assigns_allocation_slots_to_concrete_device() {
        let gpu0 = DeviceBinding::new(DeviceId::new("gpu0"));
        let gpu1 = DeviceBinding::new(DeviceId::new("gpu1"));
        let pool = DevicePoolBinding::new(gpu0.clone(), "gpu0-pool")
            .unwrap()
            .with_slot(
                AllocationSlotBinding::new(
                    "weights",
                    gpu0.clone(),
                    MemoryAllocationClass::ModelArtifact,
                    256,
                )
                .unwrap(),
            )
            .unwrap()
            .with_slot(
                AllocationSlotBinding::new("kv", gpu0.clone(), MemoryAllocationClass::KvCache, 128)
                    .unwrap(),
            )
            .unwrap();
        pool.validate().unwrap();

        assert!(matches!(
            DevicePoolBinding::new(gpu0, "empty-pool")
                .unwrap()
                .validate(),
            Err(MultiDevicePlacementError::UnspecifiedGlobalPoolUse)
        ));
        assert!(matches!(
            DevicePoolBinding::new(gpu1.clone(), "gpu1-pool")
                .unwrap()
                .with_slot(
                    AllocationSlotBinding::new(
                        "wrong-device",
                        DeviceBinding::new(DeviceId::new("gpu0")),
                        MemoryAllocationClass::TemporaryWorkspace,
                        32,
                    )
                    .unwrap(),
                ),
            Err(MultiDevicePlacementError::AllocationSlotDeviceMismatch)
        ));
    }

    #[test]
    fn hysteresis_requires_material_improvement_and_cooldown_before_replacement() {
        let policy = PlacementHysteresisPolicy::new(100, 1_000);
        assert!(!policy.should_replace(1_000, 950, 2_000));
        assert!(!policy.should_replace(1_000, 800, 999));
        assert!(policy.should_replace(1_000, 800, 1_000));
    }

    #[test]
    fn plan_lifecycle_uses_generation_and_blocks_empty_ready_state() {
        let set = DeviceSet::new(
            DeviceSetId::new("set").unwrap(),
            [device("gpu0", "cuda", 24)],
        )
        .unwrap();
        let mut plan = MultiDevicePlacementPlan::new(
            MultiDevicePlacementPlanId::new("plan").unwrap(),
            MultiDevicePlacementGeneration::new(1),
            MultiDevicePlacementFingerprint::new("sha256:graph0001").unwrap(),
            1,
            set,
            "policy-v1",
        )
        .unwrap();
        assert!(matches!(
            plan.mark_ready(),
            Err(MultiDevicePlacementError::NoPlacementBindings)
        ));
        plan.add_binding(
            PlacementScope::graph_segment([ExecutionNodeId::new("node0")]).unwrap(),
            binding("cuda", "gpu0"),
        )
        .unwrap();
        plan.mark_ready().unwrap();
        assert!(plan.state.accepts_new_work());
        assert!(matches!(
            plan.transition_to(MultiDevicePlacementState::Building),
            Err(MultiDevicePlacementError::InvalidStateTransition { .. })
        ));
    }

    #[test]
    fn placement_pins_do_not_override_safety_or_compatibility() {
        let pin = PlacementPin {
            kind: PlacementPinKind::ModelInstanceDeviceSet,
            model_instance: Some(ModelInstanceId::new("model-a").unwrap()),
            stage_id: Some("stage0".into()),
            weight_artifact: Some("weights-a".into()),
            session_id: Some("session-a".into()),
            device_set: Some(DeviceSetId::new("set-a").unwrap()),
            binding: Some(binding("cuda", "gpu0")),
            compatibility_authoritative: true,
            device_available: true,
        };
        pin.validate().unwrap();

        let unsafe_pin = PlacementPin {
            compatibility_authoritative: false,
            ..pin
        };
        assert!(matches!(
            unsafe_pin.validate(),
            Err(MultiDevicePlacementError::PlacementPinInvalid)
        ));
    }

    #[test]
    fn prefill_decode_transition_requires_state_and_explicit_movement() {
        let phases = PhasePlacementPlans {
            prefill_plan: MultiDevicePlacementPlanId::new("prefill").unwrap(),
            decode_plan: MultiDevicePlacementPlanId::new("decode").unwrap(),
        };
        assert!(phases.uses_distinct_plans());
        assert_eq!(
            phases.plan_for(PlacementPhase::Decode),
            &MultiDevicePlacementPlanId::new("decode").unwrap()
        );

        let ready = PhaseTransitionReadiness {
            from: PlacementPhase::Prefill,
            to: PlacementPhase::Decode,
            kv_available: true,
            weights_available: true,
            upstream_completion_observed: true,
            explicit_movements_completed: true,
            decode_guards_passed: true,
        };
        assert!(ready.permits_transition());
        assert!(
            !PhaseTransitionReadiness {
                kv_available: false,
                ..ready
            }
            .permits_transition()
        );
    }

    #[test]
    fn kv_ownership_locality_partition_replication_and_session_migration_are_explicit() {
        let kv = KvPagePlacement::new("page0", "session0", "sequence0", binding("cuda", "gpu1"))
            .unwrap();
        assert!(kv.authoritative);
        assert_eq!(kv.owner.device, DeviceBinding::new(DeviceId::new("gpu1")));

        let locality = KvLocalityDecision {
            decode_binding: binding("cuda", "gpu1"),
            authoritative_kv_device: DeviceBinding::new(DeviceId::new("gpu1")),
            kv_movement_cost_micros: 80,
            permits_per_token_bounce: false,
        };
        assert!(locality.favors_kv_locality());

        KvPartitionBoundary {
            attention_contract_supports_partition: true,
            required_collectives: BTreeSet::new(),
        }
        .validate()
        .unwrap();
        assert!(matches!(
            (KvPartitionBoundary {
                attention_contract_supports_partition: false,
                required_collectives: BTreeSet::new(),
            })
            .validate(),
            Err(MultiDevicePlacementError::KvPlacementInvalid)
        ));

        KvReplicaPolicy {
            authoritative_device: DeviceBinding::new(DeviceId::new("gpu1")),
            replicas: [DeviceBinding::new(DeviceId::new("gpu0"))]
                .into_iter()
                .collect(),
            update_coherency_explicit: true,
            baseline_prefers_single_authority: true,
        }
        .validate()
        .unwrap();

        let affinity = SessionPlacementAffinity::new(
            "session0",
            DeviceBinding::new(DeviceId::new("gpu1")),
            MultiDevicePlacementPlanId::new("decode").unwrap(),
        )
        .unwrap();
        assert!(affinity.preserves_kv_locality);

        assert!(
            SessionMigrationPlan {
                session_id: "session0".into(),
                source: binding("cuda", "gpu1"),
                destination: binding("cuda", "gpu0"),
                moves_kv: true,
                moves_adapters: true,
                moves_session_buffers: true,
                preserves_completion_tokens: true,
            }
            .is_explicit_and_complete()
        );
    }

    #[test]
    fn model_instance_prepared_plan_guards_stale_invalidation_and_atomic_replacement_are_explicit()
    {
        let mut plans = ModelInstancePlacementPlans {
            model_instance: ModelInstanceId::new("model-a").unwrap(),
            plans: BTreeMap::new(),
        };
        plans.add_plan(
            PlacementPlanRole::Default,
            MultiDevicePlacementPlanId::new("default").unwrap(),
        );
        plans.add_plan(
            PlacementPlanRole::WorkloadSpecific,
            MultiDevicePlacementPlanId::new("batch-large").unwrap(),
        );
        plans.add_plan(
            PlacementPlanRole::Degraded,
            MultiDevicePlacementPlanId::new("degraded").unwrap(),
        );
        assert!(plans.supports_role(PlacementPlanRole::WorkloadSpecific));
        assert!(plans.supports_role(PlacementPlanRole::Degraded));

        let movement = StageMovementEdge::new(
            "stage0",
            "stage1",
            TensorResourceId::new("activation"),
            binding("cuda", "gpu0"),
            binding("cuda", "gpu1"),
            HostStagingPolicy::Forbid,
        )
        .unwrap();
        let prepared = PreparedExecutionPlacement {
            placement_plan: MultiDevicePlacementPlanId::new("default").unwrap(),
            generation: MultiDevicePlacementGeneration::new(4),
            exact_segment_bindings: vec![
                ("stage0".into(), binding("cuda", "gpu0")),
                ("stage1".into(), binding("cuda", "gpu1")),
            ],
            movement_nodes: vec![movement],
            per_device_allocation_plans: [
                (DeviceBinding::new(DeviceId::new("gpu0")), "alloc0".into()),
                (DeviceBinding::new(DeviceId::new("gpu1")), "alloc1".into()),
            ]
            .into_iter()
            .collect(),
        };
        prepared.validate().unwrap();

        assert!(
            PlacementGuardSnapshot {
                device_available: true,
                provider_ready: true,
                kernel_prepared: true,
                resource_resident: true,
                memory_reserved: true,
                peer_path_available: true,
                host_staging_policy_valid: true,
            }
            .all_pass()
        );
        assert!(
            PlacementStaleness {
                reason: PlacementStalenessReason::PressureShift,
                request_background_replacement: true,
            }
            .request_background_replacement
        );
        assert!(
            PlacementInvalidation {
                reason: PlacementInvalidationReason::DeviceLost,
                invalid_for_new_work: true,
            }
            .invalid_for_new_work
        );
        assert!(
            PlacementReplacementRequest {
                current_plan: MultiDevicePlacementPlanId::new("default").unwrap(),
                build_outside_hot_path: true,
                revalidate_resources: true,
                prepare_required_kernels: true,
            }
            .can_build()
        );
        assert!(
            AtomicPlacementReplacement {
                old_plan: MultiDevicePlacementPlanId::new("default").unwrap(),
                old_generation: MultiDevicePlacementGeneration::new(4),
                new_plan: MultiDevicePlacementPlanId::new("default-v2").unwrap(),
                new_generation: MultiDevicePlacementGeneration::new(5),
                new_plan_complete: true,
                old_in_flight_retained: true,
            }
            .can_publish()
        );
    }

    #[test]
    fn failure_degraded_recovery_scheduler_admission_concurrency_and_replica_eviction_are_guarded()
    {
        assert!(
            DeviceFailureImpact {
                lost_device: DeviceBinding::new(DeviceId::new("gpu1")),
                invalidates_streams: true,
                invalidates_plans: true,
                preserves_other_devices: true,
            }
            .is_isolated_failure_domain()
        );
        assert!(
            DegradedPlanValidation {
                plan: MultiDevicePlacementPlanId::new("degraded").unwrap(),
                explicit_degraded_plan: true,
                model_capacity_ok: true,
                kernels_ok: true,
                memory_ok: true,
                policy_ok: true,
            }
            .valid()
        );
        assert!(
            !FailoverPolicy {
                arbitrary_remaining_device_forbidden: true,
                ready_fallback_plan: None,
            }
            .can_fail_over()
        );
        assert!(
            DeviceRecoveryChecklist {
                health_readiness_checked: true,
                pools_rebuilt: true,
                kernels_reprepared: true,
                placement_plan_rebuilt: true,
            }
            .ready_for_new_work()
        );
        assert!(
            SchedulerPlacementInput {
                session_affinity: None,
                device_pressure: BTreeMap::from([(
                    DeviceBinding::new(DeviceId::new("gpu0")),
                    ProviderPressureLevel::Low,
                )]),
                plan_ready: true,
                exposes_native_handles: false,
            }
            .can_admit()
        );
        assert!(
            PlacementAdmissionCheck {
                mandatory_devices_available: true,
                per_device_memory_ok: true,
                required_kernels_available: true,
                transfers_feasible: true,
            }
            .admits()
        );
        assert!(
            CrossDeviceConcurrencyContract {
                independent_device_execution: true,
                dependencies_preserved: true,
                resource_lifetime_preserved: true,
            }
            .permits_concurrency()
        );
        let failure = FailurePropagationDecision {
            upstream_failed: true,
            downstream_stopped: true,
            explicit_fallback: None,
            structured_reason: MultiDevicePlacementErrorCode::DeviceLost,
        };
        assert!(!failure.downstream_allowed());
        assert!(failure.downstream_stopped);
        assert!(
            ReplicaEvictionDecision {
                optional_replica: true,
                authoritative_copy_remains: true,
                no_in_flight_references: true,
                dependent_plan_invalidated: true,
            }
            .can_evict()
        );
    }

    #[test]
    fn kernel_tuning_performance_cache_revalidation_native_wit_and_api_boundaries_hold() {
        let joint = JointKernelPlacementDecision {
            binding: binding("cuda", "gpu0"),
            kernel: test_kernel("attention"),
            hard_eligible: true,
            transfer_cost_micros: 40,
            memory_cost_bytes: 16,
        };
        assert!(joint.selectable());
        assert_eq!(joint.total_cost(), 56);

        let evidence = DeviceSpecificTuningEvidence {
            device: DeviceBinding::new(DeviceId::new("gpu0")),
            kernel: test_kernel("attention"),
            performance_context: "sm90-low-pressure".into(),
        };
        evidence
            .valid_for(&DeviceBinding::new(DeviceId::new("gpu0")))
            .unwrap();
        assert!(
            evidence
                .valid_for(&DeviceBinding::new(DeviceId::new("gpu1")))
                .is_err()
        );
        assert!(
            PlacementPerformanceFeedback {
                plan: MultiDevicePlacementPlanId::new("plan").unwrap(),
                device_context: DeviceBinding::new(DeviceId::new("gpu0")),
                segment_id: "stage0".into(),
                baseline_micros: 100,
                observed_micros: 112,
            }
            .regressed()
        );

        let key = PlacementPlanCacheKey {
            graph_fingerprint: MultiDevicePlacementFingerprint::new("sha256:graph0001").unwrap(),
            model_instance_revision: 1,
            device_set_fingerprint: MultiDevicePlacementFingerprint::new("sha256:devices1")
                .unwrap(),
            provider_versions: vec![(ProviderBinding::new("cuda"), "12.4".into())],
            memory_budget_class: "24gib".into(),
            workload_scope: "decode".into(),
            placement_policy_version: "policy-v1".into(),
            partition_fingerprint: None,
        };
        key.validate().unwrap();
        let mut cache = PlacementPlanCache::default();
        cache.insert(
            key.clone(),
            MultiDevicePlacementPlanId::new("cached").unwrap(),
        );
        assert!(cache.lookup(&key).is_some());
        cache.invalidate(&key);
        assert!(cache.lookup(&key).is_none());
        assert!(
            CachedPlanRevalidation {
                device_available: true,
                provider_ready: true,
                memory_capacity_ok: true,
                peer_capability_ok: true,
                kernel_available: true,
                policy_ok: true,
                resource_residency_ok: true,
            }
            .valid()
        );

        assert!(matches!(
            NativeHandlePrivacyCheck {
                device_pointer: Some("0xdeadbeef".into()),
                peer_handle: None,
                native_queue: None,
                os_handle: None,
            }
            .validate(),
            Err(MultiDevicePlacementError::NativeHandleForbidden { .. })
        ));
        ComponentPlacementRequest {
            portable_requirements: ["attention".into()].into_iter().collect(),
            requested_device: None,
            topology_authority: false,
        }
        .validate_wit_boundary()
        .unwrap();
        assert!(matches!(
            ComponentPlacementRequest {
                portable_requirements: BTreeSet::new(),
                requested_device: Some(DeviceBinding::new(DeviceId::new("gpu0"))),
                topology_authority: false,
            }
            .validate_wit_boundary(),
            Err(MultiDevicePlacementError::PolicyAuthorityViolation)
        ));
        assert!(
            !RuntimeInferencePlacementRequest {
                preferences: [RuntimePreference::LowLatency].into_iter().collect(),
                layer_to_device_mapping: vec![(0, DeviceBinding::new(DeviceId::new("gpu0")))],
                admin_policy_binding: None,
            }
            .normal_request_allowed()
        );
    }

    #[test]
    fn errors_observability_conformance_and_documentation_cover_final_contract() {
        let expected_error_ids = [
            MultiDevicePlacementErrorCode::NoFeasiblePlan.id(),
            MultiDevicePlacementErrorCode::TensorPartitionInvalid.id(),
            MultiDevicePlacementErrorCode::StageTransferFailed.id(),
            MultiDevicePlacementErrorCode::KvPlacementInvalid.id(),
            MultiDevicePlacementErrorCode::DeviceLost.id(),
            MultiDevicePlacementErrorCode::DegradedPlanUnavailable.id(),
            MultiDevicePlacementErrorCode::Internal.id(),
        ];
        assert!(expected_error_ids.contains(&"multi-device-kv-placement-invalid"));
        assert!(expected_error_ids.contains(&"internal-multi-device-placement-error"));

        PlacementObservation {
            kind: PlacementObservationKind::CrossDeviceTransferCompleted,
            plan: Some(MultiDevicePlacementPlanId::new("plan").unwrap()),
            generation: Some(MultiDevicePlacementGeneration::new(1)),
            provider: Some(ProviderBinding::new("cuda")),
            device: Some(DeviceBinding::new(DeviceId::new("gpu0"))),
            stage_id: Some("stage0".into()),
            tensor_partition_id: Some("hidden-split".into()),
            movement_bytes: Some(4096),
            reason: Some(MultiDevicePlacementErrorCode::NoFeasiblePlan),
            detail: Some("redacted-logical-detail".into()),
        }
        .validate_redacted()
        .unwrap();
        assert!(matches!(
            PlacementObservation {
                kind: PlacementObservationKind::CrossDeviceTransferStarted,
                plan: None,
                generation: None,
                provider: None,
                device: None,
                stage_id: None,
                tensor_partition_id: None,
                movement_bytes: None,
                reason: None,
                detail: Some("native queue 0xdeadbeef".into()),
            }
            .validate_redacted(),
            Err(MultiDevicePlacementError::NativeHandleForbidden { .. })
        ));

        assert!(
            MultiDeviceConformanceReport {
                runtime_placement_authority: true,
                component_cannot_force_device: true,
                partition_validity: true,
                replica_partition_distinct: true,
                shard_not_full_tensor: true,
                explicit_cross_device_movement: true,
                host_staging_policy_preserved: true,
                peer_capability_required: true,
                per_device_memory_policy: true,
                heterogeneous_device_support: true,
                transfer_aware_selection: true,
                exact_prepared_placement: true,
                no_mid_flight_migration: true,
                kv_locality: true,
                explicit_session_migration: true,
                device_loss_invalidation: true,
                degraded_plan_validation: true,
                recovery_lifecycle: true,
                cache_revalidation: true,
                handle_isolation: true,
                observability_redaction: true,
            }
            .passes()
        );
        for topic in [
            "DeviceSet",
            "PlacementDomain",
            "MultiDevicePlacementPlan",
            "TensorShard",
            "KV locality",
            "local-only scope",
        ] {
            assert!(MULTI_DEVICE_PLACEMENT_DOCUMENTATION_TOPICS.contains(&topic));
        }
    }
}

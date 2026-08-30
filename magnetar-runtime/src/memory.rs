//! Runtime-owned memory management contracts.
//!
//! Compute describes portable tensors and operations. The memory manager owns
//! the Runtime interpretation of allocation classes, placement, residency,
//! staging feasibility, zero-copy feasibility, pressure, and memory admission.

use crate::{
    AffinityError, CompletionTokenId, DTypeDescriptor, DeviceAvailability, DeviceBinding,
    DeviceMetadata, HostStagingPolicy, ProviderAdmissionDecision, ProviderBinding,
    ProviderPressureLevel, ProviderStatusSnapshot, ResourceAffinity, ShapeDescriptor,
    TensorDescriptor, TensorResourceId,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
    ops::Range,
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MemoryAllocationId(u64);

impl MemoryAllocationId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn value(self) -> u64 {
        self.0
    }
}

impl fmt::Display for MemoryAllocationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "memory-allocation:{}", self.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MemoryArenaId(u64);

impl MemoryArenaId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn value(self) -> u64 {
        self.0
    }
}

impl fmt::Display for MemoryArenaId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "memory-arena:{}", self.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryAllocationClass {
    Tensor,
    ModelArtifact,
    TokenizerArtifact,
    AdapterArtifact,
    QuantizationArtifact,
    KvCache,
    PrefixCache,
    TemporaryWorkspace,
    TransferStaging,
    HostPinned,
    BrowserLinearMemory,
    RuntimeInternal,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MemoryPlacement {
    HostOrdinary,
    HostPinned,
    Device(DeviceBinding),
    UnifiedShared,
    ProviderOwnedOpaque(ProviderBinding),
    ExternalBorrowed,
    BrowserLinearMemory,
    StagedTemporary(Box<MemoryPlacement>),
}

impl MemoryPlacement {
    pub const fn requires_pinned_host(&self) -> bool {
        matches!(self, Self::HostPinned)
    }

    pub const fn is_browser_only(&self) -> bool {
        matches!(self, Self::BrowserLinearMemory)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ResourceMappingId(u64);

impl ResourceMappingId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn value(self) -> u64 {
        self.0
    }
}

impl fmt::Display for ResourceMappingId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "resource-mapping:{}", self.0)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MemoryDomain {
    Host,
    DeviceLocal(DeviceBinding),
    HostVisibleDevice(DeviceBinding),
    Shared {
        provider: Option<ProviderBinding>,
        device: Option<DeviceBinding>,
    },
    Managed {
        provider: Option<ProviderBinding>,
        device: Option<DeviceBinding>,
    },
    External(String),
}

impl MemoryDomain {
    pub fn from_placement(placement: &MemoryPlacement) -> Self {
        match placement {
            MemoryPlacement::HostOrdinary | MemoryPlacement::HostPinned => Self::Host,
            MemoryPlacement::Device(device) => Self::DeviceLocal(device.clone()),
            MemoryPlacement::UnifiedShared => Self::Shared {
                provider: None,
                device: None,
            },
            MemoryPlacement::ProviderOwnedOpaque(provider) => Self::Managed {
                provider: Some(provider.clone()),
                device: None,
            },
            MemoryPlacement::ExternalBorrowed => Self::External("borrowed".into()),
            MemoryPlacement::BrowserLinearMemory => Self::Host,
            MemoryPlacement::StagedTemporary(inner) => Self::from_placement(inner),
        }
    }

    pub const fn is_host_visible(&self) -> bool {
        matches!(
            self,
            Self::Host | Self::HostVisibleDevice(_) | Self::Shared { .. } | Self::Managed { .. }
        )
    }

    pub const fn is_device_resident(&self) -> bool {
        matches!(
            self,
            Self::DeviceLocal(_)
                | Self::HostVisibleDevice(_)
                | Self::Shared {
                    device: Some(_),
                    ..
                }
                | Self::Managed {
                    device: Some(_),
                    ..
                }
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceResidencyState {
    Resident,
    MappingPending,
    TransferPending,
    Replicated,
    Evicting,
    Evicted,
    Invalid,
}

impl ResourceResidencyState {
    pub const fn is_readable(self) -> bool {
        matches!(self, Self::Resident | Self::Replicated)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceResidency {
    pub resource: TensorResourceId,
    pub memory_domain: MemoryDomain,
    pub provider: Option<ProviderBinding>,
    pub device: Option<DeviceBinding>,
    pub allocation: Option<MemoryAllocationId>,
    pub state: ResourceResidencyState,
    pub authoritative: bool,
    pub current: bool,
}

impl ResourceResidency {
    pub fn new(resource: TensorResourceId, memory_domain: MemoryDomain) -> Self {
        let device = match &memory_domain {
            MemoryDomain::DeviceLocal(device)
            | MemoryDomain::HostVisibleDevice(device)
            | MemoryDomain::Shared {
                device: Some(device),
                ..
            }
            | MemoryDomain::Managed {
                device: Some(device),
                ..
            } => Some(device.clone()),
            _ => None,
        };
        let provider = match &memory_domain {
            MemoryDomain::Shared {
                provider: Some(provider),
                ..
            }
            | MemoryDomain::Managed {
                provider: Some(provider),
                ..
            } => Some(provider.clone()),
            _ => None,
        };
        Self {
            resource,
            memory_domain,
            provider,
            device,
            allocation: None,
            state: ResourceResidencyState::Resident,
            authoritative: true,
            current: true,
        }
    }

    pub const fn with_allocation(mut self, allocation: MemoryAllocationId) -> Self {
        self.allocation = Some(allocation);
        self
    }

    pub fn with_provider(mut self, provider: ProviderBinding) -> Self {
        self.provider = Some(provider);
        self
    }

    pub fn with_device(mut self, device: DeviceBinding) -> Self {
        self.device = Some(device);
        self
    }

    pub fn with_provider_if_some(mut self, provider: Option<ProviderBinding>) -> Self {
        if let Some(provider) = provider {
            self.provider = Some(provider);
        }
        self
    }

    pub fn with_device_if_some(mut self, device: Option<DeviceBinding>) -> Self {
        if let Some(device) = device {
            self.device = Some(device);
        }
        self
    }

    pub const fn with_state(mut self, state: ResourceResidencyState) -> Self {
        self.state = state;
        self
    }

    pub const fn stale(mut self) -> Self {
        self.current = false;
        self.authoritative = false;
        self
    }

    pub const fn replicated(mut self) -> Self {
        self.state = ResourceResidencyState::Replicated;
        self.authoritative = false;
        self.current = true;
        self
    }

    pub const fn is_device_resident(&self) -> bool {
        self.memory_domain.is_device_resident()
    }

    pub const fn is_readable_current(&self) -> bool {
        self.current && self.state.is_readable()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidencySet {
    pub resource: TensorResourceId,
    replicas: Vec<ResourceResidency>,
}

impl ResidencySet {
    pub fn new(resource: TensorResourceId) -> Self {
        Self {
            resource,
            replicas: Vec::new(),
        }
    }

    pub fn with_residency(mut self, residency: ResourceResidency) -> Result<Self, MemoryError> {
        self.add(residency)?;
        Ok(self)
    }

    pub fn add(&mut self, residency: ResourceResidency) -> Result<(), MemoryError> {
        if residency.resource != self.resource {
            return Err(MemoryError::ResidencyInvalid {
                reason: "residency resource does not match set resource".into(),
            });
        }
        if residency.authoritative {
            for existing in &mut self.replicas {
                existing.authoritative = false;
            }
        }
        self.replicas.push(residency);
        Ok(())
    }

    pub fn replicas(&self) -> &[ResourceResidency] {
        &self.replicas
    }

    pub fn current_replicas(&self) -> impl Iterator<Item = &ResourceResidency> {
        self.replicas.iter().filter(|replica| replica.current)
    }

    pub fn stale_replicas(&self) -> impl Iterator<Item = &ResourceResidency> {
        self.replicas.iter().filter(|replica| !replica.current)
    }

    pub fn authoritative(&self) -> Option<&ResourceResidency> {
        self.replicas
            .iter()
            .find(|replica| replica.authoritative && replica.current)
    }

    pub fn readable_for(&self, domain: &MemoryDomain) -> Result<&ResourceResidency, MemoryError> {
        self.replicas
            .iter()
            .find(|replica| &replica.memory_domain == domain && replica.is_readable_current())
            .ok_or_else(|| MemoryError::ResidencyUnavailable {
                reason: "no current readable replica exists for requested memory domain".into(),
            })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceMappingAccess {
    Read,
    Write,
    ReadWrite,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceMappingState {
    Pending,
    Active,
    Releasing,
    Released,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceMapping {
    pub id: ResourceMappingId,
    pub resource: TensorResourceId,
    pub access: ResourceMappingAccess,
    pub range: Range<u64>,
    pub mapped_domain: MemoryDomain,
    pub state: ResourceMappingState,
    pub coherency: MappingCoherency,
    pub visibility_transition_required: bool,
}

impl ResourceMapping {
    pub fn new(
        id: ResourceMappingId,
        resource: TensorResourceId,
        access: ResourceMappingAccess,
        range: Range<u64>,
        mapped_domain: MemoryDomain,
    ) -> Result<Self, MemoryError> {
        if range.start >= range.end {
            return Err(MemoryError::MappingRangeInvalid {
                reason: "mapping range must be non-empty".into(),
            });
        }
        Ok(Self {
            id,
            resource,
            access,
            range,
            mapped_domain,
            state: ResourceMappingState::Pending,
            coherency: MappingCoherency::Unknown,
            visibility_transition_required: false,
        })
    }

    pub const fn activate(mut self) -> Self {
        self.state = ResourceMappingState::Active;
        self
    }

    pub const fn with_coherency(mut self, coherency: MappingCoherency) -> Self {
        self.visibility_transition_required = matches!(coherency, MappingCoherency::NonCoherent);
        self.coherency = coherency;
        self
    }

    pub fn overlaps(&self, other: &Self) -> bool {
        self.resource == other.resource
            && self.range.start < other.range.end
            && other.range.start < self.range.end
    }

    pub const fn conflicts_with(&self, other: &Self) -> bool {
        matches!(
            (self.access, other.access),
            (ResourceMappingAccess::Write, _)
                | (ResourceMappingAccess::ReadWrite, _)
                | (_, ResourceMappingAccess::Write)
                | (_, ResourceMappingAccess::ReadWrite)
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ResourceViewId(u64);

impl ResourceViewId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn value(self) -> u64 {
        self.0
    }
}

impl fmt::Display for ResourceViewId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "resource-view:{}", self.0)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceView {
    pub id: ResourceViewId,
    pub parent: TensorResourceId,
    pub offset_elements: u64,
    pub shape: ShapeDescriptor,
    pub strides_elements: Vec<i64>,
    pub layout: String,
}

impl ResourceView {
    pub fn new(
        id: ResourceViewId,
        parent: TensorResourceId,
        offset_elements: u64,
        shape: ShapeDescriptor,
        strides_elements: impl Into<Vec<i64>>,
        layout: impl Into<String>,
    ) -> Result<Self, MemoryError> {
        let strides_elements = strides_elements.into();
        if shape.dimensions.len() != strides_elements.len() {
            return Err(MemoryError::ViewInvalid {
                reason: "view stride rank must match shape rank".into(),
            });
        }
        if strides_elements.contains(&0) {
            return Err(MemoryError::ViewInvalid {
                reason: "view strides must be non-zero".into(),
            });
        }
        Ok(Self {
            id,
            parent,
            offset_elements,
            shape,
            strides_elements,
            layout: layout.into(),
        })
    }

    pub fn validate_bounds(&self, parent_elements: u64) -> Result<(), MemoryError> {
        let max_offset = self.max_linear_element_offset()?;
        let exclusive_end = self
            .offset_elements
            .checked_add(max_offset)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| MemoryError::ViewOverflow {
                reason: "view end offset overflows u64".into(),
            })?;
        if exclusive_end > parent_elements {
            return Err(MemoryError::ViewOutOfBounds {
                reason: "view exceeds parent resource bounds".into(),
            });
        }
        Ok(())
    }

    pub fn linear_span(&self) -> Result<Range<u64>, MemoryError> {
        let max = self
            .offset_elements
            .checked_add(self.max_linear_element_offset()?)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| MemoryError::ViewOverflow {
                reason: "view span overflows u64".into(),
            })?;
        Ok(self.offset_elements..max)
    }

    pub fn overlaps(&self, other: &Self) -> Result<bool, MemoryError> {
        if self.parent != other.parent {
            return Ok(false);
        }
        let left = self.linear_span()?;
        let right = other.linear_span()?;
        Ok(left.start < right.end && right.start < left.end)
    }

    pub fn requires_materialization(&self, supports_strided: bool) -> bool {
        !supports_strided && !self.is_contiguous_row_major()
    }

    fn max_linear_element_offset(&self) -> Result<u64, MemoryError> {
        self.shape
            .dimensions
            .iter()
            .zip(&self.strides_elements)
            .try_fold(0_u64, |max, (dimension, stride)| {
                if *dimension == 0 {
                    return Ok(max);
                }
                let span = dimension
                    .checked_sub(1)
                    .and_then(|extent| extent.checked_mul(stride.unsigned_abs()))
                    .ok_or_else(|| MemoryError::ViewOverflow {
                        reason: "view stride span overflows u64".into(),
                    })?;
                max.checked_add(span)
                    .ok_or_else(|| MemoryError::ViewOverflow {
                        reason: "view maximum offset overflows u64".into(),
                    })
            })
    }

    fn is_contiguous_row_major(&self) -> bool {
        self.shape.row_major_strides() == self.strides_elements
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceMaterialization {
    pub source_view: ResourceViewId,
    pub result: TensorResourceId,
    pub reason: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MappingCoherency {
    Unknown,
    Coherent,
    NonCoherent,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ZeroCopyAccessMode {
    Read,
    Write,
    ReadWrite,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ZeroCopyEligibility {
    pub provider_compatible: bool,
    pub device_compatible: bool,
    pub memory_domain_compatible: bool,
    pub affinity_compatible: bool,
    pub dtype_compatible: bool,
    pub layout_compatible: bool,
    pub alignment_compatible: bool,
    pub access_compatible: bool,
    pub coherency_compatible: bool,
    pub readiness_compatible: bool,
}

impl ZeroCopyEligibility {
    pub const fn compatible() -> Self {
        Self {
            provider_compatible: true,
            device_compatible: true,
            memory_domain_compatible: true,
            affinity_compatible: true,
            dtype_compatible: true,
            layout_compatible: true,
            alignment_compatible: true,
            access_compatible: true,
            coherency_compatible: true,
            readiness_compatible: true,
        }
    }

    pub const fn is_eligible(&self) -> bool {
        self.provider_compatible
            && self.device_compatible
            && self.memory_domain_compatible
            && self.affinity_compatible
            && self.dtype_compatible
            && self.layout_compatible
            && self.alignment_compatible
            && self.access_compatible
            && self.coherency_compatible
            && self.readiness_compatible
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceMovementKind {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
    CrossProvider,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceMovement {
    pub source: TensorResourceId,
    pub destination_domain: MemoryDomain,
    pub destination_provider: Option<ProviderBinding>,
    pub destination_device: Option<DeviceBinding>,
    pub kind: ResourceMovementKind,
    pub host_staging_required: bool,
    pub host_staging_policy: HostStagingPolicy,
    pub completion: Option<CompletionTokenId>,
}

impl ResourceMovement {
    pub fn new(
        source: TensorResourceId,
        source_domain: &MemoryDomain,
        destination_domain: MemoryDomain,
        host_staging_policy: HostStagingPolicy,
    ) -> Self {
        let kind = match (
            source_domain.is_device_resident(),
            destination_domain.is_device_resident(),
        ) {
            (false, true) => ResourceMovementKind::HostToDevice,
            (true, false) => ResourceMovementKind::DeviceToHost,
            (true, true) => ResourceMovementKind::DeviceToDevice,
            (false, false) => ResourceMovementKind::CrossProvider,
        };
        let destination_provider = match &destination_domain {
            MemoryDomain::Shared { provider, .. } | MemoryDomain::Managed { provider, .. } => {
                provider.clone()
            }
            _ => None,
        };
        let destination_device = match &destination_domain {
            MemoryDomain::DeviceLocal(device) | MemoryDomain::HostVisibleDevice(device) => {
                Some(device.clone())
            }
            MemoryDomain::Shared { device, .. } | MemoryDomain::Managed { device, .. } => {
                device.clone()
            }
            _ => None,
        };
        Self {
            source,
            destination_domain,
            destination_provider,
            destination_device,
            kind,
            host_staging_required: false,
            host_staging_policy,
            completion: None,
        }
    }

    pub const fn requiring_host_staging(mut self) -> Self {
        self.host_staging_required = true;
        self
    }

    pub const fn with_completion(mut self, completion: CompletionTokenId) -> Self {
        self.completion = Some(completion);
        self
    }

    pub fn validate(&self) -> Result<(), MemoryError> {
        if self.host_staging_required && self.host_staging_policy == HostStagingPolicy::Forbid {
            return Err(MemoryError::TransferHostStagingDenied {
                reason: "movement requires forbidden host staging".into(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PeerAccessMode {
    PeerRead,
    PeerWrite,
    PeerReadWrite,
    PeerCopy,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PeerAccessCapability {
    pub provider: ProviderBinding,
    pub source: DeviceBinding,
    pub target: DeviceBinding,
    pub modes: BTreeSet<PeerAccessMode>,
}

impl PeerAccessCapability {
    pub fn new(
        provider: ProviderBinding,
        source: DeviceBinding,
        target: DeviceBinding,
        modes: impl IntoIterator<Item = PeerAccessMode>,
    ) -> Self {
        Self {
            provider,
            source,
            target,
            modes: modes.into_iter().collect(),
        }
    }

    pub fn allows(
        &self,
        source: &DeviceBinding,
        target: &DeviceBinding,
        mode: PeerAccessMode,
    ) -> bool {
        &self.source == source && &self.target == target && self.modes.contains(&mode)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MemoryCapabilityDescriptor {
    pub memory_domains: BTreeSet<MemoryDomain>,
    pub host_mapping: bool,
    pub coherent_mapping: bool,
    pub non_coherent_mapping: bool,
    pub pinned_host_allocation: bool,
    pub shared_memory: bool,
    pub managed_memory: bool,
    pub peer_access: Vec<PeerAccessCapability>,
    pub peer_transfer: bool,
}

impl MemoryCapabilityDescriptor {
    pub fn supports_domain(&self, domain: &MemoryDomain) -> bool {
        self.memory_domains.contains(domain)
    }

    pub fn allows_peer_access(
        &self,
        source: &DeviceBinding,
        target: &DeviceBinding,
        mode: PeerAccessMode,
    ) -> bool {
        self.peer_access
            .iter()
            .any(|capability| capability.allows(source, target, mode))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResidencyPreference {
    PreserveSourceAffinity,
    PreferDeviceLocal(DeviceBinding),
    PreferHostVisible(DeviceBinding),
    PreferDomain(MemoryDomain),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidencyRequirement {
    pub required_domain: MemoryDomain,
    pub required_provider: Option<ProviderBinding>,
    pub required_device: Option<DeviceBinding>,
}

impl ResidencyRequirement {
    pub fn satisfied_by(&self, residency: &ResourceResidency) -> bool {
        residency.memory_domain == self.required_domain
            && self
                .required_provider
                .as_ref()
                .is_none_or(|provider| residency.provider.as_ref() == Some(provider))
            && self
                .required_device
                .as_ref()
                .is_none_or(|device| residency.device.as_ref() == Some(device))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidencyPin {
    pub resource: TensorResourceId,
    pub domain: MemoryDomain,
    pub capacity_class: MemoryAllocationClass,
    pub bounded_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MemoryPressureAction {
    Evict(ResourceMovement),
    Spill(ResourceMovement),
    ReduceReplication(TensorResourceId),
    AlternateDevice(DeviceBinding),
    RejectAdmission(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceImportDescriptor {
    pub size_bytes: u64,
    pub alignment_bytes: u64,
    pub access: ResourceMappingAccess,
    pub provider: ProviderBinding,
    pub device: Option<DeviceBinding>,
    pub lifetime_description: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceExportPolicy {
    pub allowed: bool,
    pub policy_id: String,
    pub exposes_native_handle: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidencyObservation {
    pub kind: MemoryObservationKind,
    pub resource: Option<TensorResourceId>,
    pub domain: Option<MemoryDomain>,
    pub message: String,
}

impl ResidencyObservation {
    pub fn redacted(
        kind: MemoryObservationKind,
        resource: Option<TensorResourceId>,
        domain: Option<MemoryDomain>,
        message: impl AsRef<str>,
    ) -> Self {
        Self {
            kind,
            resource,
            domain,
            message: redact_memory_diagnostic(message.as_ref()),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MemoryAllocationOwner {
    Runtime,
    Provider(ProviderBinding),
    Device(DeviceBinding),
    ComponentRuntime,
    InferenceArtifact(String),
    Session(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryAllocationState {
    Pending,
    Active,
    Released,
    Reusable,
    ReservedArena,
    Failed,
    Cancelled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryAllocationLifetime {
    Operation,
    ExecutionContext,
    Session,
    Runtime,
    ExternalBorrow,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryDTypeRelation {
    pub storage_dtype: DTypeDescriptor,
    pub compute_dtype: DTypeDescriptor,
}

impl MemoryDTypeRelation {
    pub fn new(storage_dtype: DTypeDescriptor, compute_dtype: DTypeDescriptor) -> Self {
        Self {
            storage_dtype,
            compute_dtype,
        }
    }

    pub fn storage_size_bytes(&self, elements: u64) -> Result<u64, MemoryError> {
        elements
            .checked_mul(self.storage_dtype.size_bytes())
            .ok_or(MemoryError::SizeOverflow)
    }

    pub fn compute_workspace_bytes(&self, elements: u64) -> Result<u64, MemoryError> {
        if self.storage_dtype == self.compute_dtype {
            return Ok(0);
        }
        elements
            .checked_mul(self.compute_dtype.size_bytes())
            .ok_or(MemoryError::SizeOverflow)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryAllocationRequest {
    pub class: MemoryAllocationClass,
    pub size_bytes: u64,
    pub alignment_bytes: u64,
    pub placement: MemoryPlacement,
    pub owner: MemoryAllocationOwner,
    pub lifetime: MemoryAllocationLifetime,
    pub affinity: Option<ResourceAffinity>,
    pub dtype: Option<MemoryDTypeRelation>,
    pub priority: u8,
    pub deadline_millis: Option<u64>,
}

impl MemoryAllocationRequest {
    pub fn new(
        class: MemoryAllocationClass,
        size_bytes: u64,
        placement: MemoryPlacement,
        owner: MemoryAllocationOwner,
    ) -> Self {
        Self {
            class,
            size_bytes,
            alignment_bytes: 1,
            placement,
            owner,
            lifetime: MemoryAllocationLifetime::Operation,
            affinity: None,
            dtype: None,
            priority: 0,
            deadline_millis: None,
        }
    }

    pub const fn with_alignment(mut self, alignment_bytes: u64) -> Self {
        self.alignment_bytes = alignment_bytes;
        self
    }

    pub fn with_affinity(mut self, affinity: ResourceAffinity) -> Self {
        self.affinity = Some(affinity);
        self
    }

    pub fn with_dtype_relation(mut self, dtype: MemoryDTypeRelation) -> Self {
        self.dtype = Some(dtype);
        self
    }

    pub const fn with_deadline_millis(mut self, deadline_millis: u64) -> Self {
        self.deadline_millis = Some(deadline_millis);
        self
    }

    pub const fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Build a Tensor allocation request from a descriptor's estimated
    /// size, so callers never hand-compute tensor byte sizes. Unknown or
    /// overflowing size is reported rather than guessed, so admission can
    /// reject it conservatively instead of under-provisioning.
    pub fn for_tensor(
        descriptor: &TensorDescriptor,
        placement: MemoryPlacement,
        owner: MemoryAllocationOwner,
    ) -> Result<Self, crate::TensorError> {
        let size_bytes = descriptor
            .estimated_byte_size()
            .map_err(|error| crate::TensorError::size_unknown(error.to_string()))?;
        Ok(Self::new(
            MemoryAllocationClass::Tensor,
            size_bytes,
            placement,
            owner,
        ))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryAllocation {
    pub id: MemoryAllocationId,
    pub request: MemoryAllocationRequest,
    pub state: MemoryAllocationState,
    pub arena: Option<MemoryArenaId>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryArenaGrowthPolicy {
    Fixed,
    GrowOnDemand { increment_bytes: u64 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryArenaShrinkPolicy {
    Never,
    ReleaseReusable,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MemoryArenaOwner {
    Runtime,
    Provider(ProviderBinding),
    Device(DeviceBinding),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryArena {
    pub id: MemoryArenaId,
    pub class: MemoryAllocationClass,
    pub placement: MemoryPlacement,
    pub capacity_bytes: u64,
    pub used_bytes: u64,
    pub growth: MemoryArenaGrowthPolicy,
    pub shrink: MemoryArenaShrinkPolicy,
    pub owner: MemoryArenaOwner,
    pub pressure: MemoryPressureLevel,
    pub diagnostics: Vec<String>,
}

impl MemoryArena {
    pub fn new(
        id: MemoryArenaId,
        class: MemoryAllocationClass,
        placement: MemoryPlacement,
        capacity_bytes: u64,
        owner: MemoryArenaOwner,
    ) -> Self {
        Self {
            id,
            class,
            placement,
            capacity_bytes,
            used_bytes: 0,
            growth: MemoryArenaGrowthPolicy::Fixed,
            shrink: MemoryArenaShrinkPolicy::Never,
            owner,
            pressure: MemoryPressureLevel::Low,
            diagnostics: Vec::new(),
        }
    }

    pub const fn with_growth(mut self, growth: MemoryArenaGrowthPolicy) -> Self {
        self.growth = growth;
        self
    }

    pub const fn with_shrink(mut self, shrink: MemoryArenaShrinkPolicy) -> Self {
        self.shrink = shrink;
        self
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DeviceMemoryPoolId(String);

impl DeviceMemoryPoolId {
    pub fn new(value: impl Into<String>) -> Result<Self, MemoryError> {
        let value = value.into();
        validate_memory_pool_identity(&value, "device memory pool id")?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for DeviceMemoryPoolId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MemoryPoolClass {
    Weights,
    KvCache,
    Workspace,
    Transient,
    Persistent,
    Transfer,
    Shared,
    Custom(String),
}

impl MemoryPoolClass {
    pub fn custom(value: impl Into<String>) -> Result<Self, MemoryError> {
        let value = value.into();
        validate_memory_pool_identity(&value, "memory pool class")?;
        Ok(Self::Custom(value))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceMemoryPoolState {
    Initializing,
    Ready,
    Pressure,
    Critical,
    Draining,
    Failed,
    Closed,
}

impl DeviceMemoryPoolState {
    pub const fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Initializing, Self::Ready)
                | (Self::Initializing, Self::Failed)
                | (Self::Ready, Self::Pressure)
                | (Self::Ready, Self::Critical)
                | (Self::Ready, Self::Draining)
                | (Self::Ready, Self::Failed)
                | (Self::Pressure, Self::Ready)
                | (Self::Pressure, Self::Critical)
                | (Self::Pressure, Self::Draining)
                | (Self::Pressure, Self::Failed)
                | (Self::Critical, Self::Pressure)
                | (Self::Critical, Self::Draining)
                | (Self::Critical, Self::Failed)
                | (Self::Draining, Self::Closed)
                | (Self::Draining, Self::Failed)
                | (Self::Failed, Self::Closed)
        )
    }

    pub const fn accepts_new_leases(self) -> bool {
        matches!(self, Self::Ready | Self::Pressure)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PoolCapacity {
    pub configured_limit_bytes: u64,
    pub reserved_bytes: u64,
    pub committed_bytes: u64,
    pub leased_bytes: u64,
    pub reclaimable_bytes: u64,
    pub pending_reclaim_bytes: u64,
    pub borrowed_bytes: u64,
}

impl PoolCapacity {
    pub const fn new(configured_limit_bytes: u64) -> Self {
        Self {
            configured_limit_bytes,
            reserved_bytes: 0,
            committed_bytes: 0,
            leased_bytes: 0,
            reclaimable_bytes: 0,
            pending_reclaim_bytes: 0,
            borrowed_bytes: 0,
        }
    }

    pub fn validate(&self) -> Result<(), MemoryError> {
        let active = self
            .leased_bytes
            .checked_add(self.pending_reclaim_bytes)
            .ok_or(MemoryError::SizeOverflow)?;
        if self.reserved_bytes > self.configured_limit_bytes {
            return Err(MemoryError::AllocationDenied {
                reason: "pool reservation exceeds configured limit".into(),
            });
        }
        if self.committed_bytes > self.configured_limit_bytes {
            return Err(MemoryError::AllocationDenied {
                reason: "pool commitment exceeds configured limit".into(),
            });
        }
        if active > self.committed_bytes {
            return Err(MemoryError::AllocationDenied {
                reason: "leased and pending reclaim bytes exceed committed bytes".into(),
            });
        }
        if self.reclaimable_bytes > self.leased_bytes {
            return Err(MemoryError::AllocationDenied {
                reason: "reclaimable bytes exceed leased bytes".into(),
            });
        }
        Ok(())
    }

    pub const fn free_committed_bytes(&self) -> u64 {
        self.committed_bytes
            .saturating_sub(self.leased_bytes)
            .saturating_sub(self.pending_reclaim_bytes)
    }

    pub const fn uncommitted_bytes(&self) -> u64 {
        self.configured_limit_bytes
            .saturating_sub(self.committed_bytes)
    }

    pub const fn immediately_available_bytes(&self) -> u64 {
        self.free_committed_bytes()
            .saturating_add(self.uncommitted_bytes())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryPoolReservationKind {
    Hard,
    Soft,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MemoryPoolReservationScope {
    ModelInstance(String),
    Session(String),
    Device(DeviceBinding),
    PoolClass(MemoryPoolClass),
    WorkloadClass(String),
    Deployment(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryPoolReservation {
    pub kind: MemoryPoolReservationKind,
    pub scope: MemoryPoolReservationScope,
    pub bytes: u64,
    pub borrowed_bytes: u64,
}

impl MemoryPoolReservation {
    pub const fn hard(scope: MemoryPoolReservationScope, bytes: u64) -> Self {
        Self {
            kind: MemoryPoolReservationKind::Hard,
            scope,
            bytes,
            borrowed_bytes: 0,
        }
    }

    pub const fn soft(scope: MemoryPoolReservationScope, bytes: u64) -> Self {
        Self {
            kind: MemoryPoolReservationKind::Soft,
            scope,
            bytes,
            borrowed_bytes: 0,
        }
    }

    pub fn borrow(&mut self, bytes: u64) -> Result<(), MemoryError> {
        if self.kind == MemoryPoolReservationKind::Hard {
            return Err(MemoryError::AllocationDenied {
                reason: "hard reservation cannot be borrowed".into(),
            });
        }
        let borrowed = self
            .borrowed_bytes
            .checked_add(bytes)
            .ok_or(MemoryError::SizeOverflow)?;
        if borrowed > self.bytes {
            return Err(MemoryError::AllocationDenied {
                reason: "borrow exceeds soft reservation".into(),
            });
        }
        self.borrowed_bytes = borrowed;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PoolWatermarks {
    pub low_percent: u8,
    pub high_percent: u8,
    pub critical_percent: Option<u8>,
}

impl PoolWatermarks {
    pub fn new(low_percent: u8, high_percent: u8) -> Result<Self, MemoryError> {
        Self::with_critical(low_percent, high_percent, None)
    }

    pub fn with_critical(
        low_percent: u8,
        high_percent: u8,
        critical_percent: Option<u8>,
    ) -> Result<Self, MemoryError> {
        if low_percent >= high_percent || high_percent > 100 {
            return Err(MemoryError::AllocationDenied {
                reason: "pool low watermark must be below high watermark".into(),
            });
        }
        if let Some(critical) = critical_percent
            && (critical < high_percent || critical > 100)
        {
            return Err(MemoryError::AllocationDenied {
                reason: "pool critical watermark must be at or above high watermark".into(),
            });
        }
        Ok(Self {
            low_percent,
            high_percent,
            critical_percent,
        })
    }

    pub fn state_for_capacity(&self, capacity: &PoolCapacity) -> DeviceMemoryPoolState {
        if capacity.configured_limit_bytes == 0 {
            return DeviceMemoryPoolState::Critical;
        }
        let ratio = capacity.leased_bytes.saturating_mul(100) / capacity.configured_limit_bytes;
        if self
            .critical_percent
            .is_some_and(|critical| ratio >= u64::from(critical))
        {
            DeviceMemoryPoolState::Critical
        } else if ratio >= u64::from(self.high_percent) {
            DeviceMemoryPoolState::Pressure
        } else {
            DeviceMemoryPoolState::Ready
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceMemoryPool {
    pub id: DeviceMemoryPoolId,
    pub class: MemoryPoolClass,
    pub provider: ProviderBinding,
    pub device: DeviceBinding,
    pub memory_domain: MemoryDomain,
    pub capacity: PoolCapacity,
    pub reservations: Vec<MemoryPoolReservation>,
    pub watermarks: Option<PoolWatermarks>,
    pub state: DeviceMemoryPoolState,
    pub policy_version: u64,
}

impl DeviceMemoryPool {
    pub fn new(
        id: DeviceMemoryPoolId,
        class: MemoryPoolClass,
        provider: ProviderBinding,
        device: DeviceBinding,
        memory_domain: MemoryDomain,
        capacity: PoolCapacity,
    ) -> Result<Self, MemoryError> {
        capacity.validate()?;
        if !memory_domain.is_device_resident() {
            return Err(MemoryError::UnsupportedPlacement(
                MemoryPlacement::HostOrdinary,
            ));
        }
        Ok(Self {
            id,
            class,
            provider,
            device,
            memory_domain,
            capacity,
            reservations: Vec::new(),
            watermarks: None,
            state: DeviceMemoryPoolState::Initializing,
            policy_version: 1,
        })
    }

    pub fn with_watermarks(mut self, watermarks: PoolWatermarks) -> Self {
        self.watermarks = Some(watermarks);
        self
    }

    pub fn add_reservation(
        &mut self,
        reservation: MemoryPoolReservation,
    ) -> Result<(), MemoryError> {
        let reserved = self
            .capacity
            .reserved_bytes
            .checked_add(reservation.bytes)
            .ok_or(MemoryError::SizeOverflow)?;
        if reserved > self.capacity.configured_limit_bytes {
            return Err(MemoryError::AllocationDenied {
                reason: "pool reservation exceeds configured limit".into(),
            });
        }
        self.capacity.reserved_bytes = reserved;
        self.reservations.push(reservation);
        self.policy_version = self.policy_version.saturating_add(1);
        Ok(())
    }

    pub fn transition_to(&mut self, next: DeviceMemoryPoolState) -> Result<(), MemoryError> {
        if !self.state.can_transition_to(next) {
            return Err(MemoryError::AllocationDenied {
                reason: "invalid device memory pool state transition".into(),
            });
        }
        self.state = next;
        Ok(())
    }

    pub fn refresh_pressure_state(&mut self) -> Result<DeviceMemoryPoolState, MemoryError> {
        self.capacity.validate()?;
        let next = self
            .watermarks
            .map(|watermarks| watermarks.state_for_capacity(&self.capacity))
            .unwrap_or(DeviceMemoryPoolState::Ready);
        if self.state != next && self.state.can_transition_to(next) {
            self.state = next;
        }
        Ok(self.state)
    }

    pub fn can_lease(&self, bytes: u64) -> bool {
        self.state.accepts_new_leases() && bytes <= self.capacity.immediately_available_bytes()
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum AllocationClass {
    ModelWeight,
    AdapterWeight,
    KvPage,
    PersistentCache,
    ExecutionWorkspace,
    Intermediate,
    TransferStaging,
    Output,
    Custom(String),
}

impl AllocationClass {
    pub fn custom(value: impl Into<String>) -> Result<Self, MemoryError> {
        let value = value.into();
        validate_memory_pool_identity(&value, "allocation class")?;
        Ok(Self::Custom(value))
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum AllocationLifetimeClass {
    ModelInstance,
    Session,
    ExecutionPlan,
    BatchStep,
    Operator,
    Temporary,
    CacheEntry,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AllocationMutability {
    Immutable,
    Mutable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AllocationReclaimability {
    NonReclaimable,
    Reclaimable,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AllocationRequest {
    pub bytes: u64,
    pub alignment_bytes: u64,
    pub allocation_class: AllocationClass,
    pub memory_domain: MemoryDomain,
    pub lifetime_class: AllocationLifetimeClass,
    pub residency_requirement: ResourceAffinity,
    pub mutability: AllocationMutability,
    pub reclaimability: AllocationReclaimability,
}

impl AllocationRequest {
    pub fn new(
        bytes: u64,
        alignment_bytes: u64,
        allocation_class: AllocationClass,
        memory_domain: MemoryDomain,
        lifetime_class: AllocationLifetimeClass,
        residency_requirement: ResourceAffinity,
    ) -> Result<Self, MemoryError> {
        let request = Self {
            bytes,
            alignment_bytes,
            allocation_class,
            memory_domain,
            lifetime_class,
            residency_requirement,
            mutability: AllocationMutability::Mutable,
            reclaimability: AllocationReclaimability::NonReclaimable,
        };
        request.validate()?;
        Ok(request)
    }

    pub const fn with_mutability(mut self, mutability: AllocationMutability) -> Self {
        self.mutability = mutability;
        self
    }

    pub const fn with_reclaimability(mut self, reclaimability: AllocationReclaimability) -> Self {
        self.reclaimability = reclaimability;
        self
    }

    pub fn validate(&self) -> Result<(), MemoryError> {
        if self.bytes == 0 {
            return Err(MemoryError::AllocationDenied {
                reason: "allocation request size must be non-zero".into(),
            });
        }
        if self.alignment_bytes == 0 || !self.alignment_bytes.is_power_of_two() {
            return Err(MemoryError::AllocationDenied {
                reason: "allocation alignment must be a non-zero power of two".into(),
            });
        }
        self.bytes
            .checked_add(self.alignment_bytes - 1)
            .ok_or(MemoryError::SizeOverflow)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AllocationLeaseId(u64);

impl AllocationLeaseId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn value(self) -> u64 {
        self.0
    }
}

impl fmt::Display for AllocationLeaseId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "allocation-lease:{}", self.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AllocationBlockId(u64);

impl AllocationBlockId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn value(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AllocationLeaseState {
    Active,
    PendingReclaim,
    Reusable,
    Released,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AllocationLease {
    pub id: AllocationLeaseId,
    pub pool: DeviceMemoryPoolId,
    pub block: AllocationBlockId,
    pub offset: u64,
    pub length: u64,
    pub alignment_bytes: u64,
    pub generation: u64,
    pub state: AllocationLeaseState,
    pub completion: Option<CompletionTokenId>,
}

impl AllocationLease {
    pub fn new(
        id: AllocationLeaseId,
        pool: DeviceMemoryPoolId,
        block: AllocationBlockId,
        offset: u64,
        length: u64,
        alignment_bytes: u64,
        generation: u64,
    ) -> Result<Self, MemoryError> {
        if length == 0 {
            return Err(MemoryError::AllocationDenied {
                reason: "allocation lease length must be non-zero".into(),
            });
        }
        if alignment_bytes == 0 || !alignment_bytes.is_power_of_two() {
            return Err(MemoryError::AllocationDenied {
                reason: "allocation lease alignment must be a non-zero power of two".into(),
            });
        }
        if !offset.is_multiple_of(alignment_bytes) {
            return Err(MemoryError::AllocationDenied {
                reason: "allocation lease offset violates alignment".into(),
            });
        }
        offset
            .checked_add(length)
            .ok_or(MemoryError::SizeOverflow)?;
        Ok(Self {
            id,
            pool,
            block,
            offset,
            length,
            alignment_bytes,
            generation,
            state: AllocationLeaseState::Active,
            completion: None,
        })
    }

    pub const fn with_completion(mut self, completion: CompletionTokenId) -> Self {
        self.completion = Some(completion);
        self
    }

    pub const fn release_after_completion(mut self) -> Self {
        self.state = AllocationLeaseState::PendingReclaim;
        self
    }

    pub const fn mark_reusable(mut self) -> Self {
        self.state = AllocationLeaseState::Reusable;
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AllocationRegion {
    pub lease: AllocationLeaseId,
    pub resource: Option<TensorResourceId>,
    pub range: Range<u64>,
    pub alignment_bytes: u64,
    pub lifetime: AllocationLifetimeClass,
}

impl AllocationRegion {
    pub fn overlaps(&self, other: &Self) -> bool {
        self.range.start < other.range.end && other.range.start < self.range.end
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AllocationBlock {
    pub id: AllocationBlockId,
    pub pool: DeviceMemoryPoolId,
    pub capacity_bytes: u64,
    regions: Vec<AllocationRegion>,
    pub generation: u64,
}

impl AllocationBlock {
    pub fn new(id: AllocationBlockId, pool: DeviceMemoryPoolId, capacity_bytes: u64) -> Self {
        Self {
            id,
            pool,
            capacity_bytes,
            regions: Vec::new(),
            generation: 1,
        }
    }

    pub fn regions(&self) -> &[AllocationRegion] {
        &self.regions
    }

    pub fn add_region(&mut self, region: AllocationRegion) -> Result<(), MemoryError> {
        if region.range.start >= region.range.end || region.range.end > self.capacity_bytes {
            return Err(MemoryError::AllocationDenied {
                reason: "allocation region is out of block bounds".into(),
            });
        }
        if region.alignment_bytes == 0 || !region.alignment_bytes.is_power_of_two() {
            return Err(MemoryError::AllocationDenied {
                reason: "allocation region alignment must be a non-zero power of two".into(),
            });
        }
        if !region.range.start.is_multiple_of(region.alignment_bytes) {
            return Err(MemoryError::AllocationDenied {
                reason: "allocation region violates alignment".into(),
            });
        }
        if self
            .regions
            .iter()
            .any(|existing| existing.overlaps(&region))
        {
            return Err(MemoryError::AllocationDenied {
                reason: "allocation region overlaps an existing region".into(),
            });
        }
        self.regions.push(region);
        Ok(())
    }

    pub fn largest_free_region_bytes(&self) -> u64 {
        let mut ranges = self
            .regions
            .iter()
            .map(|region| region.range.clone())
            .collect::<Vec<_>>();
        ranges.sort_by_key(|range| range.start);

        let mut cursor = 0_u64;
        let mut largest = 0_u64;
        for range in ranges {
            largest = largest.max(range.start.saturating_sub(cursor));
            cursor = cursor.max(range.end);
        }
        largest.max(self.capacity_bytes.saturating_sub(cursor))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryArenaRole {
    Persistent,
    Transient,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArenaAllocationStrategy {
    ProviderPrivate,
    Slab,
    BucketedSizeClass,
    LinearBump,
    Custom(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceMemoryArena {
    pub id: MemoryArenaId,
    pub pool: DeviceMemoryPoolId,
    pub role: MemoryArenaRole,
    pub strategy: ArenaAllocationStrategy,
    pub capacity_bytes: u64,
    pub used_bytes: u64,
}

impl DeviceMemoryArena {
    pub fn new(
        id: MemoryArenaId,
        pool: DeviceMemoryPoolId,
        role: MemoryArenaRole,
        capacity_bytes: u64,
    ) -> Self {
        Self {
            id,
            pool,
            role,
            strategy: ArenaAllocationStrategy::ProviderPrivate,
            capacity_bytes,
            used_bytes: 0,
        }
    }

    pub fn with_strategy(mut self, strategy: ArenaAllocationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    pub fn reserve(&mut self, bytes: u64) -> Result<(), MemoryError> {
        let requested = self
            .used_bytes
            .checked_add(bytes)
            .ok_or(MemoryError::SizeOverflow)?;
        if requested > self.capacity_bytes {
            return Err(MemoryError::OutOfMemory {
                required: requested,
                available: self.capacity_bytes.saturating_sub(self.used_bytes),
            });
        }
        self.used_bytes = requested;
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkspaceRequirement {
    pub class: AllocationClass,
    pub bytes: u64,
    pub alignment_bytes: u64,
    pub memory_domain: MemoryDomain,
}

impl WorkspaceRequirement {
    pub fn new(
        bytes: u64,
        alignment_bytes: u64,
        memory_domain: MemoryDomain,
    ) -> Result<Self, MemoryError> {
        AllocationRequest::new(
            bytes,
            alignment_bytes,
            AllocationClass::ExecutionWorkspace,
            memory_domain.clone(),
            AllocationLifetimeClass::Operator,
            ResourceAffinity::new(crate::FallbackClass::Transparent),
        )?;
        Ok(Self {
            class: AllocationClass::ExecutionWorkspace,
            bytes,
            alignment_bytes,
            memory_domain,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkspaceReuseGroup {
    pub id: String,
    pub completion_barriers: BTreeSet<CompletionTokenId>,
    pub slots: BTreeSet<AllocationSlotId>,
}

impl WorkspaceReuseGroup {
    pub fn new(id: impl Into<String>) -> Result<Self, MemoryError> {
        let id = id.into();
        validate_memory_pool_identity(&id, "workspace reuse group")?;
        Ok(Self {
            id,
            completion_barriers: BTreeSet::new(),
            slots: BTreeSet::new(),
        })
    }

    pub fn with_slot(mut self, slot: AllocationSlotId) -> Self {
        self.slots.insert(slot);
        self
    }

    pub fn with_barrier(mut self, barrier: CompletionTokenId) -> Self {
        self.completion_barriers.insert(barrier);
        self
    }

    pub fn barriers_satisfied(&self, completed: &BTreeSet<CompletionTokenId>) -> bool {
        self.completion_barriers
            .iter()
            .all(|barrier| completed.contains(barrier))
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AllocationPlanId(String);

impl AllocationPlanId {
    pub fn new(value: impl Into<String>) -> Result<Self, MemoryError> {
        let value = value.into();
        validate_memory_pool_identity(&value, "allocation plan id")?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for AllocationPlanId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AllocationPlanGeneration(u64);

impl AllocationPlanGeneration {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn value(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AllocationSlotId(u64);

impl AllocationSlotId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn value(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AllocationPlanScope {
    pub graph_fingerprint: String,
    pub workload_envelope: String,
    pub kernel_workspace_fingerprint: String,
    pub memory_domain_requirements: BTreeSet<MemoryDomain>,
    pub pool_policy_version: u64,
    pub allocation_policy_version: u64,
}

impl AllocationPlanScope {
    pub fn new(
        graph_fingerprint: impl Into<String>,
        workload_envelope: impl Into<String>,
        kernel_workspace_fingerprint: impl Into<String>,
    ) -> Result<Self, MemoryError> {
        let graph_fingerprint = graph_fingerprint.into();
        let workload_envelope = workload_envelope.into();
        let kernel_workspace_fingerprint = kernel_workspace_fingerprint.into();
        validate_memory_pool_identity(&graph_fingerprint, "allocation graph fingerprint")?;
        validate_memory_pool_identity(&workload_envelope, "allocation workload envelope")?;
        validate_memory_pool_identity(
            &kernel_workspace_fingerprint,
            "kernel workspace fingerprint",
        )?;
        Ok(Self {
            graph_fingerprint,
            workload_envelope,
            kernel_workspace_fingerprint,
            memory_domain_requirements: BTreeSet::new(),
            pool_policy_version: 1,
            allocation_policy_version: 1,
        })
    }

    pub fn with_memory_domain(mut self, domain: MemoryDomain) -> Self {
        self.memory_domain_requirements.insert(domain);
        self
    }

    pub const fn with_policy_versions(
        mut self,
        pool_policy_version: u64,
        allocation_policy_version: u64,
    ) -> Self {
        self.pool_policy_version = pool_policy_version;
        self.allocation_policy_version = allocation_policy_version;
        self
    }

    pub fn cache_key(&self) -> String {
        format!(
            "{}|{}|{}|pool:{}|alloc:{}|domains:{}",
            self.graph_fingerprint,
            self.workload_envelope,
            self.kernel_workspace_fingerprint,
            self.pool_policy_version,
            self.allocation_policy_version,
            self.memory_domain_requirements.len()
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceMovability {
    Movable,
    TemporarilyPinned,
    PermanentlyNonMovable,
}

impl ResourceMovability {
    pub const fn allows_relocation(self) -> bool {
        matches!(self, Self::Movable)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AllocationSlot {
    pub id: AllocationSlotId,
    pub minimum_bytes: u64,
    pub alignment_bytes: u64,
    pub pool_class: MemoryPoolClass,
    pub lifetime: AllocationLifetimeClass,
    pub reuse_group: Option<String>,
    pub stable: bool,
    pub movability: ResourceMovability,
}

impl AllocationSlot {
    pub fn new(
        id: AllocationSlotId,
        minimum_bytes: u64,
        alignment_bytes: u64,
        pool_class: MemoryPoolClass,
        lifetime: AllocationLifetimeClass,
    ) -> Result<Self, MemoryError> {
        if minimum_bytes == 0 {
            return Err(MemoryError::AllocationDenied {
                reason: "allocation slot minimum bytes must be non-zero".into(),
            });
        }
        if alignment_bytes == 0 || !alignment_bytes.is_power_of_two() {
            return Err(MemoryError::AllocationDenied {
                reason: "allocation slot alignment must be a non-zero power of two".into(),
            });
        }
        minimum_bytes
            .checked_add(alignment_bytes - 1)
            .ok_or(MemoryError::SizeOverflow)?;
        Ok(Self {
            id,
            minimum_bytes,
            alignment_bytes,
            pool_class,
            lifetime,
            reuse_group: None,
            stable: false,
            movability: ResourceMovability::Movable,
        })
    }

    pub fn with_reuse_group(mut self, group: impl Into<String>) -> Result<Self, MemoryError> {
        let group = group.into();
        validate_memory_pool_identity(&group, "allocation slot reuse group")?;
        self.reuse_group = Some(group);
        Ok(self)
    }

    pub const fn stable(mut self) -> Self {
        self.stable = true;
        self.movability = ResourceMovability::PermanentlyNonMovable;
        self
    }

    pub const fn with_movability(mut self, movability: ResourceMovability) -> Self {
        self.movability = movability;
        self
    }

    pub fn compatible_with(&self, request: &AllocationRequest) -> bool {
        self.minimum_bytes >= request.bytes
            && self.alignment_bytes >= request.alignment_bytes
            && self.alignment_bytes.is_multiple_of(request.alignment_bytes)
            && self.lifetime == request.lifetime_class
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceLifetimeInterval {
    pub resource: TensorResourceId,
    pub first_step: u64,
    pub last_step: u64,
    pub execution_streams: BTreeSet<String>,
    pub completion: Option<CompletionTokenId>,
}

impl ResourceLifetimeInterval {
    pub fn new(
        resource: TensorResourceId,
        first_step: u64,
        last_step: u64,
    ) -> Result<Self, MemoryError> {
        if first_step > last_step {
            return Err(MemoryError::AllocationDenied {
                reason: "resource lifetime interval start must not exceed end".into(),
            });
        }
        Ok(Self {
            resource,
            first_step,
            last_step,
            execution_streams: BTreeSet::new(),
            completion: None,
        })
    }

    pub fn with_stream(mut self, stream: impl Into<String>) -> Result<Self, MemoryError> {
        let stream = stream.into();
        validate_memory_pool_identity(&stream, "execution stream identity")?;
        self.execution_streams.insert(stream);
        Ok(self)
    }

    pub const fn with_completion(mut self, completion: CompletionTokenId) -> Self {
        self.completion = Some(completion);
        self
    }

    pub fn overlaps(&self, other: &Self) -> bool {
        self.first_step <= other.last_step && other.first_step <= self.last_step
    }

    pub fn async_overlap_unknown(&self, other: &Self) -> bool {
        if self.execution_streams.is_empty() || other.execution_streams.is_empty() {
            return false;
        }
        self.execution_streams != other.execution_streams
            && (self.completion.is_none() || other.completion.is_none())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AllocationPlanGuard {
    PoolAvailable(DeviceMemoryPoolId),
    ReservationAvailable(MemoryPoolClass, u64),
    ProviderCompatible(ProviderBinding),
    DeviceCompatible(DeviceBinding),
    WorkspaceAvailable(u64),
    AlignmentSupported(u64),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AllocationPlanState {
    Building,
    Ready,
    Stale,
    Invalid,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AllocationPlan {
    pub id: AllocationPlanId,
    pub generation: AllocationPlanGeneration,
    pub scope: AllocationPlanScope,
    pub pool_bindings: BTreeMap<MemoryPoolClass, DeviceMemoryPoolId>,
    pub slots: Vec<AllocationSlot>,
    pub lifetime_intervals: BTreeMap<TensorResourceId, ResourceLifetimeInterval>,
    pub reuse_groups: Vec<WorkspaceReuseGroup>,
    pub reservation_requirements: Vec<MemoryPoolReservation>,
    pub guards: Vec<AllocationPlanGuard>,
    pub state: AllocationPlanState,
}

impl AllocationPlan {
    pub fn new(
        id: AllocationPlanId,
        generation: AllocationPlanGeneration,
        scope: AllocationPlanScope,
    ) -> Self {
        Self {
            id,
            generation,
            scope,
            pool_bindings: BTreeMap::new(),
            slots: Vec::new(),
            lifetime_intervals: BTreeMap::new(),
            reuse_groups: Vec::new(),
            reservation_requirements: Vec::new(),
            guards: Vec::new(),
            state: AllocationPlanState::Building,
        }
    }

    pub fn bind_pool(mut self, class: MemoryPoolClass, pool: DeviceMemoryPoolId) -> Self {
        self.pool_bindings.insert(class, pool);
        self
    }

    pub fn add_slot(&mut self, slot: AllocationSlot) -> Result<(), MemoryError> {
        if self.slots.iter().any(|existing| existing.id == slot.id) {
            return Err(MemoryError::AllocationDenied {
                reason: "allocation slot id is duplicated".into(),
            });
        }
        self.slots.push(slot);
        Ok(())
    }

    pub fn add_lifetime_interval(
        &mut self,
        interval: ResourceLifetimeInterval,
    ) -> Result<(), MemoryError> {
        if self.lifetime_intervals.contains_key(&interval.resource) {
            return Err(MemoryError::AllocationDenied {
                reason: "resource lifetime interval is duplicated".into(),
            });
        }
        self.lifetime_intervals
            .insert(interval.resource.clone(), interval);
        Ok(())
    }

    pub fn add_reuse_group(&mut self, group: WorkspaceReuseGroup) {
        self.reuse_groups.push(group);
    }

    pub fn add_guard(&mut self, guard: AllocationPlanGuard) {
        self.guards.push(guard);
    }

    pub fn mark_ready(&mut self) -> Result<(), MemoryError> {
        self.validate()?;
        self.state = AllocationPlanState::Ready;
        Ok(())
    }

    pub fn mark_stale(&mut self) {
        if self.state == AllocationPlanState::Ready {
            self.state = AllocationPlanState::Stale;
        }
    }

    pub fn hard_invalidate(&mut self) {
        self.state = AllocationPlanState::Invalid;
    }

    pub fn validate(&self) -> Result<(), MemoryError> {
        if self.slots.is_empty() {
            return Err(MemoryError::AllocationDenied {
                reason: "allocation plan must contain at least one slot".into(),
            });
        }
        for slot in &self.slots {
            if !self.pool_bindings.contains_key(&slot.pool_class) {
                return Err(MemoryError::AllocationDenied {
                    reason: "allocation slot references unbound pool class".into(),
                });
            }
        }
        Ok(())
    }

    pub fn can_temporally_reuse(
        &self,
        left: &TensorResourceId,
        right: &TensorResourceId,
        completed: &BTreeSet<CompletionTokenId>,
    ) -> Result<bool, MemoryError> {
        let left =
            self.lifetime_intervals
                .get(left)
                .ok_or_else(|| MemoryError::AllocationDenied {
                    reason: "left resource lifetime is unknown".into(),
                })?;
        let right =
            self.lifetime_intervals
                .get(right)
                .ok_or_else(|| MemoryError::AllocationDenied {
                    reason: "right resource lifetime is unknown".into(),
                })?;
        if left.overlaps(right) || left.async_overlap_unknown(right) {
            return Ok(false);
        }
        let completions_ok = [left.completion, right.completion]
            .into_iter()
            .flatten()
            .all(|completion| completed.contains(&completion));
        Ok(completions_ok)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AllocationPlanner {
    pub pool_policy_version: u64,
    pub allocation_policy_version: u64,
}

impl AllocationPlanner {
    pub fn plan(
        &self,
        id: AllocationPlanId,
        scope: AllocationPlanScope,
        pools: impl IntoIterator<Item = DeviceMemoryPool>,
        requests: impl IntoIterator<Item = AllocationRequest>,
        workspace: impl IntoIterator<Item = WorkspaceRequirement>,
    ) -> Result<AllocationPlan, MemoryError> {
        let mut plan = AllocationPlan::new(id, AllocationPlanGeneration::new(1), scope);
        let pools = pools.into_iter().collect::<Vec<_>>();
        for pool in &pools {
            plan = plan.bind_pool(pool.class.clone(), pool.id.clone());
            plan.add_guard(AllocationPlanGuard::PoolAvailable(pool.id.clone()));
            plan.add_guard(AllocationPlanGuard::ProviderCompatible(
                pool.provider.clone(),
            ));
            plan.add_guard(AllocationPlanGuard::DeviceCompatible(pool.device.clone()));
        }
        let mut next_slot = 1_u64;
        for request in requests {
            let pool_class = pool_class_for_allocation_class(&request.allocation_class);
            if !pools
                .iter()
                .any(|pool| pool.class == pool_class && pool.can_lease(request.bytes))
            {
                return Err(MemoryError::AllocationDenied {
                    reason: "no compatible pool can satisfy allocation request".into(),
                });
            }
            plan.add_slot(AllocationSlot::new(
                AllocationSlotId::new(next_slot),
                request.bytes,
                request.alignment_bytes,
                pool_class,
                request.lifetime_class,
            )?)?;
            next_slot = next_slot.saturating_add(1);
        }
        for requirement in workspace {
            let slot = AllocationSlot::new(
                AllocationSlotId::new(next_slot),
                requirement.bytes,
                requirement.alignment_bytes,
                MemoryPoolClass::Workspace,
                AllocationLifetimeClass::BatchStep,
            )?
            .with_reuse_group("workspace")?;
            plan.add_guard(AllocationPlanGuard::WorkspaceAvailable(requirement.bytes));
            plan.add_slot(slot)?;
            next_slot = next_slot.saturating_add(1);
        }
        plan.validate()?;
        Ok(plan)
    }
}

fn pool_class_for_allocation_class(class: &AllocationClass) -> MemoryPoolClass {
    match class {
        AllocationClass::ModelWeight | AllocationClass::AdapterWeight => MemoryPoolClass::Weights,
        AllocationClass::KvPage => MemoryPoolClass::KvCache,
        AllocationClass::PersistentCache => MemoryPoolClass::Persistent,
        AllocationClass::ExecutionWorkspace => MemoryPoolClass::Workspace,
        AllocationClass::Intermediate | AllocationClass::Output => MemoryPoolClass::Transient,
        AllocationClass::TransferStaging => MemoryPoolClass::Transfer,
        AllocationClass::Custom(value) => MemoryPoolClass::Custom(value.clone()),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FragmentationReport {
    pub free_bytes: u64,
    pub largest_free_region_bytes: u64,
    pub requested_bytes: u64,
    pub committed_bytes: u64,
}

impl FragmentationReport {
    pub const fn is_fragmented_failure(&self) -> bool {
        self.free_bytes >= self.requested_bytes
            && self.largest_free_region_bytes < self.requested_bytes
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompactionDisposition {
    Relocate,
    SkipPinned,
    SkipInFlight,
    SkipMapped,
    Unsupported,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompactionCandidate {
    pub resource: TensorResourceId,
    pub lease: AllocationLeaseId,
    pub movability: ResourceMovability,
    pub in_flight: bool,
    pub mapped: bool,
}

impl CompactionCandidate {
    pub fn disposition(&self) -> CompactionDisposition {
        if self.in_flight {
            CompactionDisposition::SkipInFlight
        } else if self.mapped {
            CompactionDisposition::SkipMapped
        } else if !self.movability.allows_relocation() {
            CompactionDisposition::SkipPinned
        } else {
            CompactionDisposition::Relocate
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RelocationOperation {
    pub resource: TensorResourceId,
    pub old_lease: AllocationLeaseId,
    pub new_lease: AllocationLeaseId,
    pub completion: CompletionTokenId,
    pub requires_plan_revalidation: bool,
}

impl RelocationOperation {
    pub fn new(
        candidate: &CompactionCandidate,
        new_lease: AllocationLeaseId,
        completion: CompletionTokenId,
    ) -> Result<Self, MemoryError> {
        if candidate.disposition() != CompactionDisposition::Relocate {
            return Err(MemoryError::EvictionDenied {
                reason: "resource cannot be relocated safely".into(),
            });
        }
        Ok(Self {
            resource: candidate.resource.clone(),
            old_lease: candidate.lease,
            new_lease,
            completion,
            requires_plan_revalidation: true,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AddressStabilityRequirement {
    None,
    PreparedKernel,
    PreparedSegment,
}

impl AddressStabilityRequirement {
    pub const fn pins_slot(self) -> bool {
        !matches!(self, Self::None)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedPlanMemoryBinding {
    pub allocation_plan: AllocationPlanId,
    pub generation: AllocationPlanGeneration,
    pub required_slots: BTreeSet<AllocationSlotId>,
    pub reservation_required: bool,
    pub capacity_validated: bool,
}

impl PreparedPlanMemoryBinding {
    pub fn validate_ready(&self) -> Result<(), MemoryError> {
        if self.reservation_required && !self.capacity_validated {
            return Err(MemoryError::AllocationDenied {
                reason: "prepared plan memory reservation is not satisfied".into(),
            });
        }
        if self.required_slots.is_empty() {
            return Err(MemoryError::AllocationDenied {
                reason: "prepared plan has no logical resource slots".into(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PoolOvercommitPolicy {
    #[default]
    Disabled,
    Enabled {
        max_bytes: u64,
        max_ratio_percent: u16,
    },
}

impl PoolOvercommitPolicy {
    pub const fn permits(self, base_capacity: u64, requested_total: u64) -> bool {
        match self {
            Self::Disabled => requested_total <= base_capacity,
            Self::Enabled {
                max_bytes,
                max_ratio_percent,
            } => {
                requested_total <= base_capacity.saturating_add(max_bytes)
                    && requested_total.saturating_mul(100)
                        <= base_capacity.saturating_mul(max_ratio_percent as u64)
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryAdmissionProjection {
    pub persistent_weight_bytes: u64,
    pub adapter_bytes: u64,
    pub mandatory_workspace_bytes: u64,
    pub minimum_kv_bytes: u64,
    pub pinned_bytes: u64,
    pub provider_prepared_graph_bytes: u64,
    pub session_initial_kv_pages: u64,
    pub session_max_kv_pages: Option<u64>,
    pub kv_page_bytes: u64,
}

impl MemoryAdmissionProjection {
    pub fn required_bytes(&self) -> Result<u64, MemoryError> {
        let kv_pages = self
            .session_max_kv_pages
            .unwrap_or(self.session_initial_kv_pages);
        [
            self.persistent_weight_bytes,
            self.adapter_bytes,
            self.mandatory_workspace_bytes,
            self.minimum_kv_bytes,
            self.pinned_bytes,
            self.provider_prepared_graph_bytes,
            kv_pages
                .checked_mul(self.kv_page_bytes)
                .ok_or(MemoryError::SizeOverflow)?,
        ]
        .into_iter()
        .try_fold(0_u64, |total, value| {
            total.checked_add(value).ok_or(MemoryError::SizeOverflow)
        })
    }

    pub fn admit(&self, available_bytes: u64) -> Result<(), MemoryError> {
        let required = self.required_bytes()?;
        if required > available_bytes {
            return Err(MemoryError::OutOfMemory {
                required,
                available: available_bytes,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvPageLeaseState {
    Active,
    PendingReclaim,
    Free,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KvPageOwner {
    Session(String),
    Sequence(String),
    PrefixCache(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvPageLease {
    pub page_index: u64,
    pub owner: KvPageOwner,
    pub state: KvPageLeaseState,
    pub completion: Option<CompletionTokenId>,
    pub shared_references: BTreeSet<KvPageOwner>,
}

impl KvPageLease {
    pub fn can_recycle(&self, completed: &BTreeSet<CompletionTokenId>) -> bool {
        self.state == KvPageLeaseState::PendingReclaim
            && self.shared_references.is_empty()
            && self
                .completion
                .is_none_or(|completion| completed.contains(&completion))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvPagePool {
    pub pool: DeviceMemoryPoolId,
    pub page_size_bytes: u64,
    pub total_pages: u64,
    pub free_pages: u64,
    pub leased_pages: u64,
    pub pending_reclaim_pages: u64,
    leases: BTreeMap<u64, KvPageLease>,
}

impl KvPagePool {
    pub fn new(
        pool: DeviceMemoryPoolId,
        page_size_bytes: u64,
        total_pages: u64,
    ) -> Result<Self, MemoryError> {
        if page_size_bytes == 0 || total_pages == 0 {
            return Err(MemoryError::AllocationDenied {
                reason: "KV page pool requires non-zero page size and count".into(),
            });
        }
        Ok(Self {
            pool,
            page_size_bytes,
            total_pages,
            free_pages: total_pages,
            leased_pages: 0,
            pending_reclaim_pages: 0,
            leases: BTreeMap::new(),
        })
    }

    pub fn lease_page(&mut self, owner: KvPageOwner) -> Result<KvPageLease, MemoryError> {
        if self.free_pages == 0 {
            return Err(MemoryError::AllocationDenied {
                reason: "memory-kv-page-exhausted".into(),
            });
        }
        let page_index = (0..self.total_pages)
            .find(|index| !self.leases.contains_key(index))
            .ok_or_else(|| MemoryError::AllocationDenied {
                reason: "memory-kv-page-exhausted".into(),
            })?;
        let lease = KvPageLease {
            page_index,
            owner,
            state: KvPageLeaseState::Active,
            completion: None,
            shared_references: BTreeSet::new(),
        };
        self.free_pages -= 1;
        self.leased_pages += 1;
        self.leases.insert(page_index, lease.clone());
        Ok(lease)
    }

    pub fn retain_for_prefix(
        &mut self,
        page_index: u64,
        prefix: KvPageOwner,
    ) -> Result<(), MemoryError> {
        let lease =
            self.leases
                .get_mut(&page_index)
                .ok_or_else(|| MemoryError::AllocationDenied {
                    reason: "KV page lease not found".into(),
                })?;
        lease.shared_references.insert(prefix);
        Ok(())
    }

    pub fn release_page(
        &mut self,
        page_index: u64,
        completion: Option<CompletionTokenId>,
    ) -> Result<(), MemoryError> {
        let lease =
            self.leases
                .get_mut(&page_index)
                .ok_or_else(|| MemoryError::AllocationDenied {
                    reason: "KV page lease not found".into(),
                })?;
        lease.state = KvPageLeaseState::PendingReclaim;
        lease.completion = completion;
        self.leased_pages = self.leased_pages.saturating_sub(1);
        self.pending_reclaim_pages += 1;
        Ok(())
    }

    pub fn recycle_completed(
        &mut self,
        completed: &BTreeSet<CompletionTokenId>,
    ) -> Result<u64, MemoryError> {
        let recyclable = self
            .leases
            .iter()
            .filter_map(|(page, lease)| lease.can_recycle(completed).then_some(*page))
            .collect::<Vec<_>>();
        let count = recyclable.len() as u64;
        for page in recyclable {
            self.leases.remove(&page);
        }
        self.pending_reclaim_pages = self.pending_reclaim_pages.saturating_sub(count);
        self.free_pages = self.free_pages.saturating_add(count);
        Ok(count)
    }

    pub fn grow(&mut self, additional_pages: u64) -> Result<(), MemoryError> {
        self.total_pages = self
            .total_pages
            .checked_add(additional_pages)
            .ok_or(MemoryError::SizeOverflow)?;
        self.free_pages = self
            .free_pages
            .checked_add(additional_pages)
            .ok_or(MemoryError::SizeOverflow)?;
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClassIsolationPolicy {
    pub protected_classes: BTreeSet<MemoryPoolClass>,
    pub borrow_soft_reservations: bool,
}

impl ClassIsolationPolicy {
    pub fn protects(class: MemoryPoolClass) -> Self {
        let mut protected_classes = BTreeSet::new();
        protected_classes.insert(class);
        Self {
            protected_classes,
            borrow_soft_reservations: false,
        }
    }

    pub fn permits_borrow(
        &self,
        reservation: &MemoryPoolReservation,
        requester: &MemoryPoolClass,
    ) -> bool {
        reservation.kind == MemoryPoolReservationKind::Soft
            && self.borrow_soft_reservations
            && !self.protected_classes.contains(requester)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReclaimPriority {
    OptionalReplica,
    Cache,
    Workspace,
    NeverActive,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReclaimCandidate {
    pub resource: TensorResourceId,
    pub bytes: u64,
    pub reclaimability: AllocationReclaimability,
    pub priority: ReclaimPriority,
    pub completion: Option<CompletionTokenId>,
    pub mapped: bool,
    pub pinned: bool,
    pub semantic_aliases: BTreeSet<TensorResourceId>,
}

impl ReclaimCandidate {
    pub fn can_reclaim(&self, completed: &BTreeSet<CompletionTokenId>) -> bool {
        self.reclaimability == AllocationReclaimability::Reclaimable
            && !self.mapped
            && !self.pinned
            && self.semantic_aliases.is_empty()
            && self
                .completion
                .is_none_or(|completion| completed.contains(&completion))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AsyncNativeFreeState {
    LogicallyReleased,
    PendingNativeReclaim,
    PhysicallyReusable,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderPoolCapability {
    pub provider: ProviderBinding,
    pub block_allocation: bool,
    pub async_native_free: bool,
    pub address_stability: bool,
    pub movable_allocations: bool,
    pub minimum_alignment: u64,
    pub preferred_granularity: u64,
    pub grow_shrink: bool,
}

impl ProviderPoolCapability {
    pub fn validate(&self) -> Result<(), MemoryError> {
        if self.minimum_alignment == 0
            || !self.minimum_alignment.is_power_of_two()
            || self.preferred_granularity == 0
        {
            return Err(MemoryError::AllocationDenied {
                reason: "provider pool capability alignment/granularity is invalid".into(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceMemoryCapacitySnapshot {
    pub device: DeviceBinding,
    pub total_bytes: u64,
    pub available_estimate_bytes: u64,
    pub pressure: MemoryPressureLevel,
    pub allocation_granularity_bytes: u64,
}

impl DeviceMemoryCapacitySnapshot {
    pub fn validate_metadata_only(&self) -> Result<(), MemoryError> {
        if self.allocation_granularity_bytes == 0
            || !self.allocation_granularity_bytes.is_power_of_two()
        {
            return Err(MemoryError::AllocationDenied {
                reason: "device allocation granularity must be a power of two".into(),
            });
        }
        if self.available_estimate_bytes > self.total_bytes {
            return Err(MemoryError::AllocationDenied {
                reason: "device available estimate exceeds total memory".into(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PoolGrowthRequest {
    pub additional_bytes: u64,
    pub device_available_bytes: u64,
    pub policy_limit_bytes: u64,
}

impl PoolGrowthRequest {
    pub fn validate(&self, current_capacity: u64) -> Result<u64, MemoryError> {
        let desired = current_capacity
            .checked_add(self.additional_bytes)
            .ok_or(MemoryError::SizeOverflow)?;
        if self.additional_bytes > self.device_available_bytes || desired > self.policy_limit_bytes
        {
            return Err(MemoryError::OutOfMemory {
                required: desired,
                available: self.device_available_bytes.min(self.policy_limit_bytes),
            });
        }
        Ok(desired)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PoolShrinkRequest {
    pub release_bytes: u64,
    pub live_bytes: u64,
    pub pending_reclaim_bytes: u64,
}

impl PoolShrinkRequest {
    pub fn releasable_bytes(&self, capacity: u64) -> u64 {
        capacity
            .saturating_sub(self.live_bytes)
            .saturating_sub(self.pending_reclaim_bytes)
            .min(self.release_bytes)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryOomCategory {
    PoolCapacityExceeded,
    DeviceCapacityExceeded,
    ReservationConflict,
    Fragmentation,
    AlignmentUnsatisfied,
    PinnedCapacityExhausted,
    KvPageExhausted,
    WorkspaceExhausted,
    ProviderAllocationFailed,
}

impl MemoryOomCategory {
    pub const fn code(self) -> &'static str {
        match self {
            Self::PoolCapacityExceeded => "pool-capacity-exceeded",
            Self::DeviceCapacityExceeded => "device-capacity-exceeded",
            Self::ReservationConflict => "reservation-conflict",
            Self::Fragmentation => "fragmentation",
            Self::AlignmentUnsatisfied => "alignment-unsatisfied",
            Self::PinnedCapacityExhausted => "pinned-capacity-exhausted",
            Self::KvPageExhausted => "kv-page-exhausted",
            Self::WorkspaceExhausted => "workspace-capacity-exceeded",
            Self::ProviderAllocationFailed => "provider-allocation-failed",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllocationRetryPolicy {
    pub max_retries: u8,
}

impl AllocationRetryPolicy {
    pub const fn can_retry(self, attempts: u8) -> bool {
        attempts < self.max_retries
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OomFallbackAction {
    TrimOptionalCaches,
    DropOptionalReplicas,
    SelectLowerWorkspaceKernel,
    SelectAlternatePlan(AllocationPlanId),
    SelectAlternateDevice(DeviceBinding),
    Spill,
    RejectAdmission,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelMemoryEligibility {
    pub kernel_id: String,
    pub required_workspace_bytes: u64,
    pub compatible_pool_available_bytes: u64,
}

impl KernelMemoryEligibility {
    pub fn is_eligible(&self) -> bool {
        self.required_workspace_bytes <= self.compatible_pool_available_bytes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AutotuningMemoryBudget {
    pub max_bytes: u64,
    pub protect_inference_reservations: bool,
    pub deny_under_critical_pressure: bool,
}

impl AutotuningMemoryBudget {
    pub fn admit(&self, request_bytes: u64, pressure: DeviceMemoryPoolState) -> bool {
        request_bytes <= self.max_bytes
            && !(self.deny_under_critical_pressure && pressure == DeviceMemoryPoolState::Critical)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryPerformanceFeedback {
    pub allocation_latency_micros: u64,
    pub fragmentation: Option<FragmentationReport>,
    pub pressure: MemoryPressureLevel,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AllocationPlanCache {
    entries: BTreeMap<String, AllocationPlan>,
}

impl AllocationPlanCache {
    pub fn insert(&mut self, plan: AllocationPlan) {
        self.entries.insert(plan.scope.cache_key(), plan);
    }

    pub fn lookup(&self, scope: &AllocationPlanScope) -> Option<&AllocationPlan> {
        self.entries.get(&scope.cache_key()).filter(|plan| {
            matches!(
                plan.state,
                AllocationPlanState::Ready | AllocationPlanState::Stale
            )
        })
    }

    pub fn revalidate(
        &self,
        plan: &AllocationPlan,
        pools: &[DeviceMemoryPool],
        workspace_requirements: &[WorkspaceRequirement],
    ) -> Result<(), MemoryError> {
        plan.validate()?;
        for guard in &plan.guards {
            match guard {
                AllocationPlanGuard::PoolAvailable(pool_id) => {
                    if !pools.iter().any(|pool| &pool.id == pool_id) {
                        return Err(MemoryError::AllocationDenied {
                            reason: "allocation plan required pool is unavailable".into(),
                        });
                    }
                }
                AllocationPlanGuard::ReservationAvailable(class, bytes) => {
                    if !pools
                        .iter()
                        .any(|pool| &pool.class == class && pool.capacity.reserved_bytes >= *bytes)
                    {
                        return Err(MemoryError::AllocationDenied {
                            reason: "allocation plan reservation is unavailable".into(),
                        });
                    }
                }
                AllocationPlanGuard::ProviderCompatible(provider) => {
                    if !pools.iter().any(|pool| &pool.provider == provider) {
                        return Err(MemoryError::AllocationDenied {
                            reason: "allocation plan provider is incompatible".into(),
                        });
                    }
                }
                AllocationPlanGuard::DeviceCompatible(device) => {
                    if !pools.iter().any(|pool| &pool.device == device) {
                        return Err(MemoryError::AllocationDenied {
                            reason: "allocation plan device is incompatible".into(),
                        });
                    }
                }
                AllocationPlanGuard::WorkspaceAvailable(bytes) => {
                    if workspace_requirements.iter().all(|req| req.bytes < *bytes) {
                        return Err(MemoryError::AllocationDenied {
                            reason: "allocation plan workspace requirement is stale".into(),
                        });
                    }
                }
                AllocationPlanGuard::AlignmentSupported(alignment) => {
                    if plan
                        .slots
                        .iter()
                        .all(|slot| slot.alignment_bytes < *alignment)
                    {
                        return Err(MemoryError::AllocationDenied {
                            reason: "allocation plan alignment is unsupported".into(),
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryBoundaryActor {
    WasmComponent,
    InferenceRequest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryBoundaryRequest {
    pub actor: MemoryBoundaryActor,
    pub create_pool: bool,
    pub choose_allocator_strategy: bool,
    pub expose_native_handle: bool,
    pub choose_native_pool: bool,
}

impl MemoryBoundaryRequest {
    pub fn validate(&self) -> Result<(), MemoryError> {
        if self.create_pool
            || self.choose_allocator_strategy
            || self.expose_native_handle
            || self.choose_native_pool
        {
            return Err(MemoryError::NativeHandleForbidden {
                reason: "memory pool and native allocator authority is runtime policy only".into(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryPoolErrorCode {
    PoolNotFound,
    PoolNotReady,
    PoolDraining,
    PoolFailed,
    PoolCapacityExceeded,
    PoolCriticalPressure,
    PoolReservationConflict,
    PoolBorrowDenied,
    AllocationInvalid,
    AllocationSizeOverflow,
    AllocationAlignmentInvalid,
    AllocationAlignmentUnsatisfied,
    AllocationFragmented,
    AllocationProviderFailed,
    AllocationNoCompatiblePool,
    AllocationOvercommitDenied,
    AllocationPlanInvalid,
    AllocationPlanStale,
    AllocationPlanIncompatible,
    AllocationPlanBuildFailed,
    AllocationPlanCapacityInsufficient,
    AllocationLeaseInvalid,
    AllocationLeaseInUse,
    AllocationReleasePending,
    CompactionDenied,
    CompactionResourcePinned,
    CompactionResourceInFlight,
    RelocationFailed,
    KvPageExhausted,
    WorkspaceExhausted,
    ReclamationFailed,
    ReclamationInsufficient,
    OomRetryExhausted,
    InternalPoolError,
}

impl MemoryPoolErrorCode {
    pub const fn id(self) -> &'static str {
        match self {
            Self::PoolNotFound => "memory-pool-not-found",
            Self::PoolNotReady => "memory-pool-not-ready",
            Self::PoolDraining => "memory-pool-draining",
            Self::PoolFailed => "memory-pool-failed",
            Self::PoolCapacityExceeded => "memory-pool-capacity-exceeded",
            Self::PoolCriticalPressure => "memory-pool-critical-pressure",
            Self::PoolReservationConflict => "memory-pool-reservation-conflict",
            Self::PoolBorrowDenied => "memory-pool-borrow-denied",
            Self::AllocationInvalid => "memory-allocation-invalid",
            Self::AllocationSizeOverflow => "memory-allocation-size-overflow",
            Self::AllocationAlignmentInvalid => "memory-allocation-alignment-invalid",
            Self::AllocationAlignmentUnsatisfied => "memory-allocation-alignment-unsatisfied",
            Self::AllocationFragmented => "memory-allocation-fragmented",
            Self::AllocationProviderFailed => "memory-allocation-provider-failed",
            Self::AllocationNoCompatiblePool => "memory-allocation-no-compatible-pool",
            Self::AllocationOvercommitDenied => "memory-allocation-overcommit-denied",
            Self::AllocationPlanInvalid => "memory-allocation-plan-invalid",
            Self::AllocationPlanStale => "memory-allocation-plan-stale",
            Self::AllocationPlanIncompatible => "memory-allocation-plan-incompatible",
            Self::AllocationPlanBuildFailed => "memory-allocation-plan-build-failed",
            Self::AllocationPlanCapacityInsufficient => {
                "memory-allocation-plan-capacity-insufficient"
            }
            Self::AllocationLeaseInvalid => "memory-allocation-lease-invalid",
            Self::AllocationLeaseInUse => "memory-allocation-lease-in-use",
            Self::AllocationReleasePending => "memory-allocation-release-pending",
            Self::CompactionDenied => "memory-compaction-denied",
            Self::CompactionResourcePinned => "memory-compaction-resource-pinned",
            Self::CompactionResourceInFlight => "memory-compaction-resource-in-flight",
            Self::RelocationFailed => "memory-relocation-failed",
            Self::KvPageExhausted => "memory-kv-page-exhausted",
            Self::WorkspaceExhausted => "memory-workspace-exhausted",
            Self::ReclamationFailed => "memory-reclamation-failed",
            Self::ReclamationInsufficient => "memory-reclamation-insufficient",
            Self::OomRetryExhausted => "memory-oom-retry-exhausted",
            Self::InternalPoolError => "internal-memory-pool-error",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryPoolObservationKind {
    PoolCreated,
    PoolGrown,
    PoolShrunk,
    PoolPressure,
    PoolCritical,
    PoolDraining,
    AllocationRequested,
    AllocationLeased,
    AllocationReused,
    AllocationReleased,
    AllocationReclaimPending,
    AllocationPlanBuilt,
    AllocationPlanCacheHit,
    AllocationPlanStale,
    FragmentationDetected,
    ReclamationStarted,
    ReclamationCompleted,
    CompactionStarted,
    CompactionCompleted,
    KvPageLeased,
    KvPageReclaimed,
    OomDetected,
    OomFallbackSelected,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryPoolObservation {
    pub kind: MemoryPoolObservationKind,
    pub pool: Option<DeviceMemoryPoolId>,
    pub allocation_class: Option<AllocationClass>,
    pub message: String,
    pub capacity: Option<PoolCapacity>,
    pub allocation_latency_micros: Option<u64>,
}

impl MemoryPoolObservation {
    pub fn redacted(
        kind: MemoryPoolObservationKind,
        pool: Option<DeviceMemoryPoolId>,
        message: impl AsRef<str>,
    ) -> Self {
        Self {
            kind,
            pool,
            allocation_class: None,
            message: redact_memory_diagnostic(message.as_ref()),
            capacity: None,
            allocation_latency_micros: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryPoolConformanceReport {
    pub memory_manager_policy_authority: bool,
    pub provider_native_realization: bool,
    pub device_metadata_only: bool,
    pub native_pointer_redaction: bool,
    pub temporal_reuse_safe: bool,
    pub async_reuse_safe: bool,
    pub alignment_enforced: bool,
    pub reservations_isolated: bool,
    pub watermark_reclamation: bool,
    pub pending_reclaim_accounted: bool,
    pub fragmentation_classified: bool,
    pub compaction_safe: bool,
    pub address_pinning: bool,
    pub plan_reservation_readiness: bool,
    pub kv_page_lifetime: bool,
    pub class_isolation: bool,
    pub oom_policy: bool,
    pub cache_revalidation: bool,
}

impl MemoryPoolConformanceReport {
    pub const fn conformant(&self) -> bool {
        self.memory_manager_policy_authority
            && self.provider_native_realization
            && self.device_metadata_only
            && self.native_pointer_redaction
            && self.temporal_reuse_safe
            && self.async_reuse_safe
            && self.alignment_enforced
            && self.reservations_isolated
            && self.watermark_reclamation
            && self.pending_reclaim_accounted
            && self.fragmentation_classified
            && self.compaction_safe
            && self.address_pinning
            && self.plan_reservation_readiness
            && self.kv_page_lifetime
            && self.class_isolation
            && self.oom_policy
            && self.cache_revalidation
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PendingMemoryAllocation {
    pub allocation: MemoryAllocation,
    pub queued_at_millis: u64,
    pub deadline_millis: Option<u64>,
    pub diagnostic_reason: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorResidency {
    pub tensor: TensorResourceId,
    pub allocation: Option<MemoryAllocationId>,
    pub placement: MemoryPlacement,
    pub affinity: ResourceAffinity,
    pub provider_owned: bool,
    pub staged: bool,
    pub eviction_eligible: bool,
    pub size_bytes_estimate: Option<u64>,
}

impl TensorResidency {
    pub fn new(
        tensor: TensorResourceId,
        placement: MemoryPlacement,
        affinity: ResourceAffinity,
    ) -> Self {
        let provider_owned = matches!(placement, MemoryPlacement::ProviderOwnedOpaque(_));
        let staged = matches!(placement, MemoryPlacement::StagedTemporary(_));
        Self {
            tensor,
            allocation: None,
            placement,
            affinity,
            provider_owned,
            staged,
            eviction_eligible: false,
            size_bytes_estimate: None,
        }
    }

    pub const fn with_allocation(mut self, allocation: MemoryAllocationId) -> Self {
        self.allocation = Some(allocation);
        self
    }

    pub const fn with_eviction_eligible(mut self, eviction_eligible: bool) -> Self {
        self.eviction_eligible = eviction_eligible;
        self
    }

    pub const fn with_size_estimate(mut self, size_bytes: u64) -> Self {
        self.size_bytes_estimate = Some(size_bytes);
        self
    }

    /// Portable memory-class classification derived from `placement`.
    pub fn memory_class(&self) -> crate::TensorMemoryClass {
        crate::TensorMemoryClass::from(&self.placement)
    }

    /// Whether this residency's storage is reachable from host code.
    pub fn is_host_visible(&self) -> bool {
        self.memory_class().is_host_accessible()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryPressureLevel {
    Unknown,
    Low,
    Moderate,
    High,
    Saturated,
}

impl From<ProviderPressureLevel> for MemoryPressureLevel {
    fn from(value: ProviderPressureLevel) -> Self {
        match value {
            ProviderPressureLevel::Unknown => Self::Unknown,
            ProviderPressureLevel::Low => Self::Low,
            ProviderPressureLevel::Moderate => Self::Moderate,
            ProviderPressureLevel::High => Self::High,
            ProviderPressureLevel::Saturated => Self::Saturated,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryPressureSnapshot {
    pub runtime: MemoryPressureLevel,
    pub provider: Option<(ProviderBinding, MemoryPressureLevel)>,
    pub device: Option<(DeviceBinding, MemoryPressureLevel)>,
    pub allocation_class: Option<MemoryAllocationClass>,
    pub arena: Option<(MemoryArenaId, MemoryPressureLevel)>,
    pub cache: Option<MemoryPressureLevel>,
    pub kv_cache: Option<MemoryPressureLevel>,
    pub used_bytes: u64,
    pub limit_bytes: Option<u64>,
}

impl Default for MemoryPressureSnapshot {
    fn default() -> Self {
        Self {
            runtime: MemoryPressureLevel::Low,
            provider: None,
            device: None,
            allocation_class: None,
            arena: None,
            cache: None,
            kv_cache: None,
            used_bytes: 0,
            limit_bytes: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MemoryAdmissionDecision {
    Admit { reason: String },
    Queue { reason: String },
    Reject { reason: String },
    RetryLater { reason: String },
}

impl MemoryAdmissionDecision {
    pub const fn is_admitted(&self) -> bool {
        matches!(self, Self::Admit { .. })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryAdmissionRequest {
    pub allocation: MemoryAllocationRequest,
    pub pressure: MemoryPressureSnapshot,
    pub queue_allowed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryFeasibility {
    pub feasible: bool,
    pub reason: String,
    pub required_bytes: u64,
    pub available_bytes: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ZeroCopyFeasibility {
    pub feasible: bool,
    pub reason: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StagingFeasibility {
    pub feasible: bool,
    pub reason: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryObservationKind {
    AllocationRequested,
    AllocationAdmitted,
    AllocationQueued,
    AllocationCompleted,
    AllocationFailed,
    AllocationReleased,
    CacheHit,
    CacheMiss,
    CacheEviction,
    ArenaPressure,
    PendingQueueDelay,
    PinnedMemoryUsage,
    ZeroCopyAccepted,
    ZeroCopyRejected,
    ResourceResidencyRecorded,
    ResourceMapped,
    ResourceUnmapped,
    ResourceViewCreated,
    ResourceTransferStarted,
    ResourceTransferCompleted,
    TransferElided,
    PeerAccessUsed,
    ResourceEvictionStarted,
    ResourceEvicted,
    ResourceSpilled,
    ResidencyGuardFailed,
    StagingInserted,
    StagingDenied,
    PressureChanged,
    ZeroCopySelected,
    ZeroCopyUnavailable,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryObservation {
    pub kind: MemoryObservationKind,
    pub allocation: Option<MemoryAllocationId>,
    pub message: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MemoryError {
    AllocationDenied {
        reason: String,
    },
    AllocationPending {
        reason: String,
    },
    AllocationTimeout {
        reason: String,
    },
    AllocationCancelled {
        reason: String,
    },
    OutOfMemory {
        required: u64,
        available: u64,
    },
    PressureSaturated,
    UnsupportedMemoryClass(MemoryAllocationClass),
    UnsupportedPlacement(MemoryPlacement),
    UnsupportedStorageDType(DTypeDescriptor),
    UnsupportedComputeDType(DTypeDescriptor),
    ZeroCopyUnavailable {
        reason: String,
    },
    ResidencyUnavailable {
        reason: String,
    },
    ResidencyInvalid {
        reason: String,
    },
    MappingRangeInvalid {
        reason: String,
    },
    MappingNotReady {
        reason: String,
    },
    InvalidMappingHandle(ResourceMappingId),
    MappingConflict {
        reason: String,
    },
    ViewInvalid {
        reason: String,
    },
    ViewOutOfBounds {
        reason: String,
    },
    ViewOverflow {
        reason: String,
    },
    ZeroCopyPolicyDenied {
        reason: String,
    },
    ZeroCopyProviderIncompatible {
        reason: String,
    },
    ZeroCopyDeviceIncompatible {
        reason: String,
    },
    ZeroCopyLayoutIncompatible {
        reason: String,
    },
    ZeroCopyAlignmentIncompatible {
        reason: String,
    },
    ZeroCopyCoherencyUnsupported {
        reason: String,
    },
    TransferDenied {
        reason: String,
    },
    TransferHostStagingDenied {
        reason: String,
    },
    PeerAccessUnsupported {
        reason: String,
    },
    NativeHandleForbidden {
        reason: String,
    },
    ImportDenied {
        reason: String,
    },
    ExportDenied {
        reason: String,
    },
    EvictionDenied {
        reason: String,
    },
    SpillDenied {
        reason: String,
    },
    StagingForbidden,
    StagingDenied {
        reason: String,
    },
    PinnedMemoryUnavailable,
    ProviderAllocationFailed {
        provider: ProviderBinding,
        reason: String,
    },
    DeviceMemoryUnavailable {
        device: DeviceBinding,
        reason: String,
    },
    BrowserMemoryLimitExceeded {
        required: u64,
        limit: u64,
    },
    InvalidAllocationHandle(MemoryAllocationId),
    InvalidArenaHandle(MemoryArenaId),
    ResourceAffinityConflict(AffinityError),
    SizeOverflow,
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AllocationDenied { reason } => write!(f, "memory allocation denied: {reason}"),
            Self::AllocationPending { reason } => write!(f, "memory allocation pending: {reason}"),
            Self::AllocationTimeout { reason } => {
                write!(f, "memory allocation timed out: {reason}")
            }
            Self::AllocationCancelled { reason } => {
                write!(f, "memory allocation cancelled: {reason}")
            }
            Self::OutOfMemory {
                required,
                available,
            } => write!(
                f,
                "out of memory: required {required} bytes, available {available} bytes"
            ),
            Self::PressureSaturated => write!(f, "memory pressure is saturated"),
            Self::UnsupportedMemoryClass(class) => {
                write!(f, "unsupported memory class {class:?}")
            }
            Self::UnsupportedPlacement(placement) => {
                write!(f, "unsupported memory placement {placement:?}")
            }
            Self::UnsupportedStorageDType(dtype) => {
                write!(f, "unsupported storage dtype {dtype:?}")
            }
            Self::UnsupportedComputeDType(dtype) => {
                write!(f, "unsupported compute dtype {dtype:?}")
            }
            Self::ZeroCopyUnavailable { reason } => write!(f, "zero-copy unavailable: {reason}"),
            Self::ResidencyUnavailable { reason } => {
                write!(f, "resource residency unavailable: {reason}")
            }
            Self::ResidencyInvalid { reason } => {
                write!(f, "resource residency invalid: {reason}")
            }
            Self::MappingRangeInvalid { reason } => {
                write!(f, "resource mapping range invalid: {reason}")
            }
            Self::MappingNotReady { reason } => {
                write!(f, "resource mapping not ready: {reason}")
            }
            Self::InvalidMappingHandle(id) => write!(f, "invalid resource mapping handle {id}"),
            Self::MappingConflict { reason } => {
                write!(f, "resource mapping conflict: {reason}")
            }
            Self::ViewInvalid { reason } => write!(f, "resource view invalid: {reason}"),
            Self::ViewOutOfBounds { reason } => {
                write!(f, "resource view out of bounds: {reason}")
            }
            Self::ViewOverflow { reason } => write!(f, "resource view overflow: {reason}"),
            Self::ZeroCopyPolicyDenied { reason } => write!(f, "zero-copy policy denied: {reason}"),
            Self::ZeroCopyProviderIncompatible { reason } => {
                write!(f, "zero-copy provider incompatible: {reason}")
            }
            Self::ZeroCopyDeviceIncompatible { reason } => {
                write!(f, "zero-copy device incompatible: {reason}")
            }
            Self::ZeroCopyLayoutIncompatible { reason } => {
                write!(f, "zero-copy layout incompatible: {reason}")
            }
            Self::ZeroCopyAlignmentIncompatible { reason } => {
                write!(f, "zero-copy alignment incompatible: {reason}")
            }
            Self::ZeroCopyCoherencyUnsupported { reason } => {
                write!(f, "zero-copy coherency unsupported: {reason}")
            }
            Self::TransferDenied { reason } => write!(f, "resource transfer denied: {reason}"),
            Self::TransferHostStagingDenied { reason } => {
                write!(f, "resource transfer host staging denied: {reason}")
            }
            Self::PeerAccessUnsupported { reason } => {
                write!(f, "resource peer access unsupported: {reason}")
            }
            Self::NativeHandleForbidden { reason } => {
                write!(f, "native memory handle forbidden: {reason}")
            }
            Self::ImportDenied { reason } => write!(f, "resource import denied: {reason}"),
            Self::ExportDenied { reason } => write!(f, "resource export denied: {reason}"),
            Self::EvictionDenied { reason } => write!(f, "resource eviction denied: {reason}"),
            Self::SpillDenied { reason } => write!(f, "resource spill denied: {reason}"),
            Self::StagingForbidden => write!(f, "host staging is forbidden"),
            Self::StagingDenied { reason } => write!(f, "host staging denied: {reason}"),
            Self::PinnedMemoryUnavailable => write!(f, "pinned host memory is unavailable"),
            Self::ProviderAllocationFailed { provider, reason } => {
                write!(f, "provider '{provider}' allocation failed: {reason}")
            }
            Self::DeviceMemoryUnavailable { device, reason } => {
                write!(f, "device '{device}' memory unavailable: {reason}")
            }
            Self::BrowserMemoryLimitExceeded { required, limit } => write!(
                f,
                "browser memory limit exceeded: required {required} bytes, limit {limit} bytes"
            ),
            Self::InvalidAllocationHandle(id) => write!(f, "invalid memory allocation handle {id}"),
            Self::InvalidArenaHandle(id) => write!(f, "invalid memory arena handle {id}"),
            Self::ResourceAffinityConflict(error) => write!(f, "{error}"),
            Self::SizeOverflow => write!(f, "memory size overflows u64"),
        }
    }
}

impl Error for MemoryError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryManagerConfig {
    pub max_runtime_bytes: Option<u64>,
    pub max_cached_bytes: u64,
    pub max_pinned_host_bytes: u64,
    pub max_pending_allocations: usize,
    pub allow_pending_allocations: bool,
    pub allow_browser_linear_memory: bool,
    pub allow_native_pinned_memory: bool,
}

impl Default for MemoryManagerConfig {
    fn default() -> Self {
        Self {
            max_runtime_bytes: None,
            max_cached_bytes: 0,
            max_pinned_host_bytes: 0,
            max_pending_allocations: 1024,
            allow_pending_allocations: true,
            allow_browser_linear_memory: cfg!(target_arch = "wasm32"),
            allow_native_pinned_memory: !cfg!(target_arch = "wasm32"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryManager {
    config: MemoryManagerConfig,
    next_allocation_id: u64,
    next_arena_id: u64,
    next_mapping_id: u64,
    next_view_id: u64,
    allocations: BTreeMap<MemoryAllocationId, MemoryAllocation>,
    arenas: BTreeMap<MemoryArenaId, MemoryArena>,
    pending: BTreeMap<MemoryAllocationId, PendingMemoryAllocation>,
    tensor_residency: BTreeMap<TensorResourceId, TensorResidency>,
    resource_residency: BTreeMap<TensorResourceId, ResidencySet>,
    resource_mappings: BTreeMap<ResourceMappingId, ResourceMapping>,
    resource_views: BTreeMap<ResourceViewId, ResourceView>,
    residency_pins: BTreeMap<TensorResourceId, ResidencyPin>,
    residency_observations: Vec<ResidencyObservation>,
    observations: Vec<MemoryObservation>,
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new(MemoryManagerConfig::default())
    }
}

impl MemoryManager {
    pub fn new(config: MemoryManagerConfig) -> Self {
        Self {
            config,
            next_allocation_id: 1,
            next_arena_id: 1,
            next_mapping_id: 1,
            next_view_id: 1,
            allocations: BTreeMap::new(),
            arenas: BTreeMap::new(),
            pending: BTreeMap::new(),
            tensor_residency: BTreeMap::new(),
            resource_residency: BTreeMap::new(),
            resource_mappings: BTreeMap::new(),
            resource_views: BTreeMap::new(),
            residency_pins: BTreeMap::new(),
            residency_observations: Vec::new(),
            observations: Vec::new(),
        }
    }

    pub fn config(&self) -> &MemoryManagerConfig {
        &self.config
    }

    pub fn allocations(&self) -> impl Iterator<Item = &MemoryAllocation> {
        self.allocations.values()
    }

    pub fn arenas(&self) -> impl Iterator<Item = &MemoryArena> {
        self.arenas.values()
    }

    pub fn arena_mut(&mut self, id: MemoryArenaId) -> Option<&mut MemoryArena> {
        self.arenas.get_mut(&id)
    }

    pub fn pending_allocations(&self) -> impl Iterator<Item = &PendingMemoryAllocation> {
        self.pending.values()
    }

    pub fn observations(&self) -> &[MemoryObservation] {
        &self.observations
    }

    pub fn resource_residency(&self, resource: &TensorResourceId) -> Option<&ResidencySet> {
        self.resource_residency.get(resource)
    }

    pub fn resource_mappings(&self) -> impl Iterator<Item = &ResourceMapping> {
        self.resource_mappings.values()
    }

    pub fn resource_views(&self) -> impl Iterator<Item = &ResourceView> {
        self.resource_views.values()
    }

    pub fn residency_observations(&self) -> &[ResidencyObservation] {
        &self.residency_observations
    }

    pub fn allocate(
        &mut self,
        request: MemoryAllocationRequest,
    ) -> Result<MemoryAllocation, MemoryError> {
        self.observe(
            MemoryObservationKind::AllocationRequested,
            None,
            "memory allocation requested",
        );
        self.validate_request(&request)?;
        let pressure = self.current_pressure_for(&request);
        match self.admit(MemoryAdmissionRequest {
            allocation: request.clone(),
            pressure,
            queue_allowed: false,
        }) {
            MemoryAdmissionDecision::Admit { .. } => {
                self.observe(
                    MemoryObservationKind::AllocationAdmitted,
                    None,
                    "memory allocation admitted",
                );
            }
            MemoryAdmissionDecision::Queue { reason }
            | MemoryAdmissionDecision::RetryLater { reason } => {
                self.observe(
                    MemoryObservationKind::AllocationQueued,
                    None,
                    "memory allocation queued",
                );
                return Err(MemoryError::AllocationPending { reason });
            }
            MemoryAdmissionDecision::Reject { reason } => {
                self.observe(
                    MemoryObservationKind::AllocationFailed,
                    None,
                    "memory allocation rejected",
                );
                return Err(MemoryError::AllocationDenied { reason });
            }
        }

        if let Some(reusable_id) = self.find_reusable_allocation(&request) {
            let allocation = self
                .allocations
                .get_mut(&reusable_id)
                .expect("known allocation");
            allocation.state = MemoryAllocationState::Active;
            let allocation = allocation.clone();
            self.observe(
                MemoryObservationKind::CacheHit,
                Some(reusable_id),
                "memory allocation reused from cache",
            );
            self.observe(
                MemoryObservationKind::AllocationCompleted,
                Some(reusable_id),
                "memory allocation completed",
            );
            return Ok(allocation);
        }
        self.observe(
            MemoryObservationKind::CacheMiss,
            None,
            "no compatible reusable allocation found",
        );

        let id = MemoryAllocationId::new(self.next_allocation_id);
        self.next_allocation_id = self.next_allocation_id.saturating_add(1);
        let allocation = MemoryAllocation {
            id,
            request,
            state: MemoryAllocationState::Active,
            arena: None,
        };
        self.allocations.insert(id, allocation.clone());
        self.observe(
            MemoryObservationKind::AllocationCompleted,
            Some(id),
            "memory allocation completed",
        );
        Ok(allocation)
    }

    pub fn release(&mut self, id: MemoryAllocationId) -> Result<(), MemoryError> {
        let allocation = self
            .allocations
            .get_mut(&id)
            .ok_or(MemoryError::InvalidAllocationHandle(id))?;
        if allocation.state == MemoryAllocationState::Released {
            return Err(MemoryError::InvalidAllocationHandle(id));
        }
        allocation.state = if self.config.max_cached_bytes > 0 {
            MemoryAllocationState::Reusable
        } else {
            MemoryAllocationState::Released
        };
        self.enforce_cache_limit();
        self.observe(
            MemoryObservationKind::AllocationReleased,
            Some(id),
            "memory allocation released",
        );
        Ok(())
    }

    pub fn feasibility(&self, request: &MemoryAllocationRequest) -> MemoryFeasibility {
        if let Err(error) = self.validate_request(request) {
            return MemoryFeasibility {
                feasible: false,
                reason: error.to_string(),
                required_bytes: request.size_bytes,
                available_bytes: self.available_bytes(),
            };
        }
        if let Some(available) = self.available_bytes()
            && request.size_bytes > available
        {
            return MemoryFeasibility {
                feasible: false,
                reason: format!(
                    "request requires {} bytes but only {available} bytes are available",
                    request.size_bytes
                ),
                required_bytes: request.size_bytes,
                available_bytes: Some(available),
            };
        }
        MemoryFeasibility {
            feasible: true,
            reason: "memory request is feasible".into(),
            required_bytes: request.size_bytes,
            available_bytes: self.available_bytes(),
        }
    }

    pub fn admit(&self, request: MemoryAdmissionRequest) -> MemoryAdmissionDecision {
        if matches!(request.pressure.runtime, MemoryPressureLevel::Saturated) {
            if request.queue_allowed && self.config.allow_pending_allocations {
                return MemoryAdmissionDecision::Queue {
                    reason: "runtime memory pressure is saturated".into(),
                };
            }
            return MemoryAdmissionDecision::Reject {
                reason: "runtime memory pressure is saturated".into(),
            };
        }
        if let Some((provider, pressure)) = &request.pressure.provider
            && matches!(pressure, MemoryPressureLevel::Saturated)
        {
            return MemoryAdmissionDecision::Reject {
                reason: format!("provider '{provider}' memory pressure is saturated"),
            };
        }
        if let Some((device, pressure)) = &request.pressure.device
            && matches!(pressure, MemoryPressureLevel::Saturated)
        {
            return MemoryAdmissionDecision::Reject {
                reason: format!("device '{device}' memory pressure is saturated"),
            };
        }
        if matches!(request.pressure.runtime, MemoryPressureLevel::High) && request.queue_allowed {
            return MemoryAdmissionDecision::RetryLater {
                reason: "runtime memory pressure is high".into(),
            };
        }
        let feasibility = self.feasibility(&request.allocation);
        if feasibility.feasible {
            MemoryAdmissionDecision::Admit {
                reason: feasibility.reason,
            }
        } else if request.queue_allowed && self.config.allow_pending_allocations {
            MemoryAdmissionDecision::Queue {
                reason: feasibility.reason,
            }
        } else {
            MemoryAdmissionDecision::Reject {
                reason: feasibility.reason,
            }
        }
    }

    /// Evaluate admission for a Tensor Resource, deriving its size from the
    /// descriptor via [`MemoryAllocationRequest::for_tensor`] instead of
    /// requiring the caller to compute it. A descriptor whose size cannot be
    /// determined (e.g. an underspecified packed/quantized layout) is
    /// rejected rather than conservatively admitted, so admission never
    /// under-reserves for an unknown-sized tensor.
    pub fn admit_tensor(
        &self,
        descriptor: &TensorDescriptor,
        placement: MemoryPlacement,
        owner: MemoryAllocationOwner,
        pressure: MemoryPressureSnapshot,
    ) -> MemoryAdmissionDecision {
        let allocation = match MemoryAllocationRequest::for_tensor(descriptor, placement, owner) {
            Ok(allocation) => allocation,
            Err(error) => {
                return MemoryAdmissionDecision::Reject {
                    reason: error.to_string(),
                };
            }
        };
        self.admit(MemoryAdmissionRequest {
            allocation,
            pressure,
            queue_allowed: self.config.allow_pending_allocations,
        })
    }

    pub fn record_tensor_residency(
        &mut self,
        residency: TensorResidency,
    ) -> Result<(), MemoryError> {
        if let Some(allocation) = residency.allocation
            && !self.allocations.contains_key(&allocation)
        {
            return Err(MemoryError::InvalidAllocationHandle(allocation));
        }
        self.tensor_residency
            .insert(residency.tensor.clone(), residency);
        Ok(())
    }

    pub fn record_resource_residency(
        &mut self,
        residency: ResourceResidency,
    ) -> Result<(), MemoryError> {
        if let Some(allocation) = residency.allocation
            && !self.allocations.contains_key(&allocation)
        {
            return Err(MemoryError::InvalidAllocationHandle(allocation));
        }
        let resource = residency.resource.clone();
        self.resource_residency
            .entry(resource.clone())
            .or_insert_with(|| ResidencySet::new(resource.clone()))
            .add(residency)?;
        self.observe(
            MemoryObservationKind::ResourceResidencyRecorded,
            None,
            "resource residency recorded",
        );
        Ok(())
    }

    pub fn readable_residency(
        &self,
        resource: &TensorResourceId,
        domain: &MemoryDomain,
    ) -> Result<&ResourceResidency, MemoryError> {
        self.resource_residency
            .get(resource)
            .ok_or_else(|| MemoryError::ResidencyUnavailable {
                reason: "resource residency is unknown".into(),
            })?
            .readable_for(domain)
    }

    pub fn map_resource(
        &mut self,
        resource: TensorResourceId,
        access: ResourceMappingAccess,
        range: Range<u64>,
        mapped_domain: MemoryDomain,
    ) -> Result<ResourceMapping, MemoryError> {
        let residency = self
            .resource_residency
            .get(&resource)
            .and_then(ResidencySet::authoritative)
            .ok_or_else(|| MemoryError::ResidencyUnavailable {
                reason: "resource has no authoritative current residency".into(),
            })?;
        if !residency.state.is_readable() && matches!(access, ResourceMappingAccess::Read) {
            return Err(MemoryError::MappingNotReady {
                reason: "resource writes are not complete for read mapping".into(),
            });
        }
        let id = ResourceMappingId::new(self.next_mapping_id);
        self.next_mapping_id = self.next_mapping_id.saturating_add(1);
        let mapping = ResourceMapping::new(id, resource, access, range, mapped_domain)?.activate();
        if self
            .resource_mappings
            .values()
            .filter(|existing| existing.state == ResourceMappingState::Active)
            .any(|existing| mapping.overlaps(existing) && mapping.conflicts_with(existing))
        {
            return Err(MemoryError::MappingConflict {
                reason: "active mapping has overlapping write access".into(),
            });
        }
        self.resource_mappings.insert(id, mapping.clone());
        self.observe(
            MemoryObservationKind::ResourceMapped,
            None,
            "resource mapped through logical mapping",
        );
        Ok(mapping)
    }

    pub fn release_mapping(&mut self, id: ResourceMappingId) -> Result<(), MemoryError> {
        let visibility_event = {
            let mapping = self
                .resource_mappings
                .get_mut(&id)
                .ok_or(MemoryError::InvalidMappingHandle(id))?;
            mapping.state = ResourceMappingState::Releasing;
            let visibility_event = (mapping.visibility_transition_required
                && matches!(
                    mapping.access,
                    ResourceMappingAccess::Write | ResourceMappingAccess::ReadWrite
                ))
            .then(|| (mapping.resource.clone(), mapping.mapped_domain.clone()));
            mapping.state = ResourceMappingState::Released;
            visibility_event
        };
        if let Some((resource, domain)) = visibility_event {
            self.observe_residency(
                MemoryObservationKind::ResourceUnmapped,
                Some(resource),
                Some(domain),
                "non-coherent mapping release performed visibility transition",
            );
        }
        self.observe(
            MemoryObservationKind::ResourceUnmapped,
            None,
            "resource mapping released",
        );
        Ok(())
    }

    pub fn create_resource_view(
        &mut self,
        parent: TensorResourceId,
        offset_elements: u64,
        shape: ShapeDescriptor,
        strides_elements: impl Into<Vec<i64>>,
        layout: impl Into<String>,
        parent_elements: u64,
    ) -> Result<ResourceView, MemoryError> {
        let id = ResourceViewId::new(self.next_view_id);
        self.next_view_id = self.next_view_id.saturating_add(1);
        let view = ResourceView::new(id, parent, offset_elements, shape, strides_elements, layout)?;
        view.validate_bounds(parent_elements)?;
        self.resource_views.insert(id, view.clone());
        self.observe(
            MemoryObservationKind::ResourceViewCreated,
            None,
            "resource view created without materialization",
        );
        Ok(view)
    }

    pub fn plan_materialization(
        &self,
        view: &ResourceView,
        result: TensorResourceId,
        supports_strided: bool,
    ) -> Option<ResourceMaterialization> {
        view.requires_materialization(supports_strided)
            .then(|| ResourceMaterialization {
                source_view: view.id,
                result,
                reason: "consumer requires materialized contiguous layout".into(),
            })
    }

    pub fn validate_residency_requirement(
        &self,
        resource: &TensorResourceId,
        requirement: &ResidencyRequirement,
    ) -> Result<(), MemoryError> {
        let set = self.resource_residency.get(resource).ok_or_else(|| {
            MemoryError::ResidencyUnavailable {
                reason: "resource residency is unknown".into(),
            }
        })?;
        if set
            .current_replicas()
            .any(|residency| requirement.satisfied_by(residency))
        {
            Ok(())
        } else {
            Err(MemoryError::ResidencyUnavailable {
                reason: "hard residency requirement is not satisfied".into(),
            })
        }
    }

    pub fn validate_zero_copy_eligibility(
        &mut self,
        eligibility: ZeroCopyEligibility,
    ) -> Result<(), MemoryError> {
        if eligibility.is_eligible() {
            self.observe(
                MemoryObservationKind::ZeroCopySelected,
                None,
                "zero-copy selected",
            );
            return Ok(());
        }
        self.observe(
            MemoryObservationKind::ZeroCopyUnavailable,
            None,
            "zero-copy unavailable",
        );
        if !eligibility.provider_compatible {
            return Err(MemoryError::ZeroCopyProviderIncompatible {
                reason: "provider differs or lacks sharing capability".into(),
            });
        }
        if !eligibility.device_compatible {
            return Err(MemoryError::ZeroCopyDeviceIncompatible {
                reason: "device differs or lacks peer access".into(),
            });
        }
        if !eligibility.layout_compatible {
            return Err(MemoryError::ZeroCopyLayoutIncompatible {
                reason: "layout is incompatible".into(),
            });
        }
        if !eligibility.alignment_compatible {
            return Err(MemoryError::ZeroCopyAlignmentIncompatible {
                reason: "alignment is incompatible".into(),
            });
        }
        if !eligibility.coherency_compatible {
            return Err(MemoryError::ZeroCopyCoherencyUnsupported {
                reason: "coherency requirements are unsupported".into(),
            });
        }
        Err(MemoryError::ZeroCopyUnavailable {
            reason: "zero-copy eligibility gates failed".into(),
        })
    }

    pub fn movement_for_residency(
        &mut self,
        resource: &TensorResourceId,
        destination: MemoryDomain,
        host_staging_policy: HostStagingPolicy,
    ) -> Result<Option<ResourceMovement>, MemoryError> {
        if self.readable_residency(resource, &destination).is_ok() {
            self.observe(
                MemoryObservationKind::TransferElided,
                None,
                "resource movement elided because residency already matches",
            );
            return Ok(None);
        }
        let source = self
            .resource_residency
            .get(resource)
            .and_then(ResidencySet::authoritative)
            .ok_or_else(|| MemoryError::ResidencyUnavailable {
                reason: "resource has no authoritative source residency".into(),
            })?;
        let movement = ResourceMovement::new(
            resource.clone(),
            &source.memory_domain,
            destination,
            host_staging_policy,
        );
        movement.validate()?;
        self.observe(
            MemoryObservationKind::ResourceTransferStarted,
            None,
            "explicit resource movement planned",
        );
        Ok(Some(movement))
    }

    pub fn complete_movement(
        &mut self,
        movement: ResourceMovement,
        completion: CompletionTokenId,
    ) -> Result<ResourceResidency, MemoryError> {
        movement.validate()?;
        let residency =
            ResourceResidency::new(movement.source.clone(), movement.destination_domain)
                .with_state(ResourceResidencyState::Resident)
                .with_device_if_some(movement.destination_device)
                .with_provider_if_some(movement.destination_provider);
        self.record_resource_residency(residency.clone())?;
        self.observe(
            MemoryObservationKind::ResourceTransferCompleted,
            None,
            format!("resource movement completed by {completion}"),
        );
        Ok(residency)
    }

    pub fn pin_residency(&mut self, pin: ResidencyPin) -> Result<(), MemoryError> {
        let active = self.active_bytes();
        if active.saturating_add(pin.bounded_bytes)
            > self.config.max_runtime_bytes.unwrap_or(u64::MAX)
        {
            return Err(MemoryError::EvictionDenied {
                reason: "residency pin exceeds configured runtime memory limit".into(),
            });
        }
        self.residency_pins.insert(pin.resource.clone(), pin);
        Ok(())
    }

    pub fn pressure_action(
        &self,
        resource: TensorResourceId,
        source_domain: &MemoryDomain,
        target_domain: MemoryDomain,
        spill: bool,
        host_staging_policy: HostStagingPolicy,
    ) -> Result<MemoryPressureAction, MemoryError> {
        let movement = ResourceMovement::new(
            resource.clone(),
            source_domain,
            target_domain,
            host_staging_policy,
        );
        movement.validate()?;
        if self.residency_pins.contains_key(&resource) {
            return Ok(MemoryPressureAction::RejectAdmission(
                "resource is residency-pinned".into(),
            ));
        }
        if spill {
            Ok(MemoryPressureAction::Spill(movement))
        } else {
            Ok(MemoryPressureAction::Evict(movement))
        }
    }

    pub fn validate_peer_zero_copy(
        &mut self,
        capability: &MemoryCapabilityDescriptor,
        source: &DeviceBinding,
        target: &DeviceBinding,
        mode: PeerAccessMode,
    ) -> Result<(), MemoryError> {
        if capability.allows_peer_access(source, target, mode) {
            self.observe(
                MemoryObservationKind::PeerAccessUsed,
                None,
                "peer access used through explicit capability",
            );
            Ok(())
        } else {
            Err(MemoryError::PeerAccessUnsupported {
                reason: "peer access is not advertised for device pair".into(),
            })
        }
    }

    pub fn validate_resource_import(
        &self,
        descriptor: &ResourceImportDescriptor,
        capability: &MemoryCapabilityDescriptor,
    ) -> Result<(), MemoryError> {
        if descriptor.size_bytes == 0 {
            return Err(MemoryError::ImportDenied {
                reason: "import size must be non-zero".into(),
            });
        }
        if descriptor.alignment_bytes == 0 || !descriptor.alignment_bytes.is_power_of_two() {
            return Err(MemoryError::ImportDenied {
                reason: "import alignment must be a power of two".into(),
            });
        }
        if contains_native_handle_marker(&descriptor.lifetime_description) {
            return Err(MemoryError::NativeHandleForbidden {
                reason: "native handle marker is not portable import identity".into(),
            });
        }
        if descriptor.device.is_some()
            && !capability
                .memory_domains
                .iter()
                .any(MemoryDomain::is_device_resident)
        {
            return Err(MemoryError::ImportDenied {
                reason: "provider does not advertise compatible device memory domain".into(),
            });
        }
        Ok(())
    }

    pub fn validate_resource_export(
        &self,
        policy: &ResourceExportPolicy,
    ) -> Result<(), MemoryError> {
        if !policy.allowed {
            return Err(MemoryError::ExportDenied {
                reason: "resource export is denied by policy".into(),
            });
        }
        if policy.exposes_native_handle {
            return Err(MemoryError::NativeHandleForbidden {
                reason: "native handles are not exported through generic resource contract".into(),
            });
        }
        Ok(())
    }

    pub fn create_arena(
        &mut self,
        class: MemoryAllocationClass,
        placement: MemoryPlacement,
        capacity_bytes: u64,
        owner: MemoryArenaOwner,
    ) -> Result<MemoryArena, MemoryError> {
        let id = MemoryArenaId::new(self.next_arena_id);
        self.next_arena_id = self.next_arena_id.saturating_add(1);
        let mut arena = MemoryArena::new(id, class, placement, capacity_bytes, owner);
        arena.pressure = pressure_for_usage(0, Some(capacity_bytes));
        self.arenas.insert(id, arena.clone());
        Ok(arena)
    }

    pub fn reserve_in_arena(
        &mut self,
        arena: MemoryArenaId,
        size_bytes: u64,
    ) -> Result<(), MemoryError> {
        let arena = self
            .arenas
            .get_mut(&arena)
            .ok_or(MemoryError::InvalidArenaHandle(arena))?;
        let requested = arena
            .used_bytes
            .checked_add(size_bytes)
            .ok_or(MemoryError::SizeOverflow)?;
        if requested > arena.capacity_bytes {
            match arena.growth {
                MemoryArenaGrowthPolicy::Fixed => {
                    return Err(MemoryError::OutOfMemory {
                        required: requested,
                        available: arena.capacity_bytes.saturating_sub(arena.used_bytes),
                    });
                }
                MemoryArenaGrowthPolicy::GrowOnDemand { increment_bytes } => {
                    while requested > arena.capacity_bytes {
                        arena.capacity_bytes = arena
                            .capacity_bytes
                            .checked_add(increment_bytes)
                            .ok_or(MemoryError::SizeOverflow)?;
                    }
                    arena.diagnostics.push("arena grew on demand".into());
                }
            }
        }
        arena.used_bytes = requested;
        arena.pressure = pressure_for_usage(arena.used_bytes, Some(arena.capacity_bytes));
        self.observe(
            MemoryObservationKind::ArenaPressure,
            None,
            "arena pressure changed",
        );
        Ok(())
    }

    pub fn shrink_arena(&mut self, arena: MemoryArenaId) -> Result<(), MemoryError> {
        let arena = self
            .arenas
            .get_mut(&arena)
            .ok_or(MemoryError::InvalidArenaHandle(arena))?;
        if arena.shrink == MemoryArenaShrinkPolicy::ReleaseReusable {
            arena.capacity_bytes = arena.used_bytes;
            arena.diagnostics.push("arena shrank to used bytes".into());
        }
        arena.pressure = pressure_for_usage(arena.used_bytes, Some(arena.capacity_bytes));
        Ok(())
    }

    pub fn submit_pending_allocation(
        &mut self,
        request: MemoryAllocationRequest,
        queued_at_millis: u64,
    ) -> Result<PendingMemoryAllocation, MemoryError> {
        if !self.config.allow_pending_allocations {
            return Err(MemoryError::AllocationDenied {
                reason: "pending allocations are disabled".into(),
            });
        }
        if self.pending.len() >= self.config.max_pending_allocations {
            return Err(MemoryError::AllocationDenied {
                reason: "pending allocation queue is full".into(),
            });
        }
        self.validate_request(&request)?;
        let id = MemoryAllocationId::new(self.next_allocation_id);
        self.next_allocation_id = self.next_allocation_id.saturating_add(1);
        let allocation = MemoryAllocation {
            id,
            request: request.clone(),
            state: MemoryAllocationState::Pending,
            arena: None,
        };
        let pending = PendingMemoryAllocation {
            allocation,
            queued_at_millis,
            deadline_millis: request.deadline_millis,
            diagnostic_reason: "waiting for memory admission".into(),
        };
        self.pending.insert(id, pending.clone());
        self.observe(
            MemoryObservationKind::AllocationQueued,
            Some(id),
            "memory allocation queued",
        );
        Ok(pending)
    }

    pub fn cancel_pending_allocation(&mut self, id: MemoryAllocationId) -> Result<(), MemoryError> {
        let mut pending = self
            .pending
            .remove(&id)
            .ok_or(MemoryError::InvalidAllocationHandle(id))?;
        pending.allocation.state = MemoryAllocationState::Cancelled;
        self.allocations.insert(id, pending.allocation);
        Err(MemoryError::AllocationCancelled {
            reason: "pending allocation was cancelled".into(),
        })
    }

    pub fn expire_pending_allocations(&mut self, now_millis: u64) -> Vec<MemoryError> {
        let expired = self
            .pending
            .iter()
            .filter_map(|(id, pending)| {
                pending
                    .deadline_millis
                    .is_some_and(|deadline| now_millis >= deadline)
                    .then_some(*id)
            })
            .collect::<Vec<_>>();
        let mut errors = Vec::new();
        for id in expired {
            if let Some(mut pending) = self.pending.remove(&id) {
                pending.allocation.state = MemoryAllocationState::Failed;
                self.allocations.insert(id, pending.allocation);
                self.observe(
                    MemoryObservationKind::AllocationFailed,
                    Some(id),
                    "pending memory allocation timed out",
                );
                errors.push(MemoryError::AllocationTimeout {
                    reason: "pending allocation deadline expired".into(),
                });
            }
        }
        errors
    }

    pub fn retry_pending_allocations(&mut self) -> Vec<Result<MemoryAllocation, MemoryError>> {
        let ids = self.pending.keys().copied().collect::<Vec<_>>();
        let mut results = Vec::new();
        for id in ids {
            let Some(pending) = self.pending.remove(&id) else {
                continue;
            };
            results.push(self.allocate(pending.allocation.request));
            self.observe(
                MemoryObservationKind::PendingQueueDelay,
                Some(id),
                "pending memory allocation retried",
            );
        }
        results
    }

    pub fn tensor_residency(&self, tensor: &TensorResourceId) -> Option<&TensorResidency> {
        self.tensor_residency.get(tensor)
    }

    pub fn staging_feasibility(
        &self,
        policy: HostStagingPolicy,
        size_bytes: u64,
    ) -> StagingFeasibility {
        if policy == HostStagingPolicy::Forbid {
            return StagingFeasibility {
                feasible: false,
                reason: "host staging is forbidden by compute request".into(),
            };
        }
        let request = MemoryAllocationRequest::new(
            MemoryAllocationClass::TransferStaging,
            size_bytes,
            MemoryPlacement::HostOrdinary,
            MemoryAllocationOwner::Runtime,
        );
        let feasibility = self.feasibility(&request);
        StagingFeasibility {
            feasible: feasibility.feasible,
            reason: feasibility.reason,
        }
    }

    pub fn observed_staging_feasibility(
        &mut self,
        policy: HostStagingPolicy,
        size_bytes: u64,
    ) -> StagingFeasibility {
        let result = self.staging_feasibility(policy, size_bytes);
        self.observe(
            if result.feasible {
                MemoryObservationKind::StagingInserted
            } else {
                MemoryObservationKind::StagingDenied
            },
            None,
            result.reason.clone(),
        );
        result
    }

    pub fn zero_copy_feasibility(
        &self,
        source: &TensorResidency,
        target: &MemoryPlacement,
        dtype: Option<&MemoryDTypeRelation>,
    ) -> ZeroCopyFeasibility {
        if source.placement == *target {
            return ZeroCopyFeasibility {
                feasible: true,
                reason: "source and target placement match".into(),
            };
        }
        if source.is_host_visible()
            && matches!(
                target,
                MemoryPlacement::HostOrdinary
                    | MemoryPlacement::HostPinned
                    | MemoryPlacement::UnifiedShared
            )
        {
            return ZeroCopyFeasibility {
                feasible: true,
                reason: "source residency is host-visible without copy".into(),
            };
        }
        if let Some(dtype) = dtype
            && dtype.storage_dtype != dtype.compute_dtype
        {
            return ZeroCopyFeasibility {
                feasible: false,
                reason: "storage dtype differs from compute dtype".into(),
            };
        }
        ZeroCopyFeasibility {
            feasible: false,
            reason: "source and target placement are incompatible".into(),
        }
    }

    pub fn observed_zero_copy_feasibility(
        &mut self,
        source: &TensorResidency,
        target: &MemoryPlacement,
        dtype: Option<&MemoryDTypeRelation>,
    ) -> ZeroCopyFeasibility {
        let result = self.zero_copy_feasibility(source, target, dtype);
        self.observe(
            if result.feasible {
                MemoryObservationKind::ZeroCopyAccepted
            } else {
                MemoryObservationKind::ZeroCopyRejected
            },
            source.allocation,
            result.reason.clone(),
        );
        result
    }

    pub fn zero_copy_for_residency(
        &self,
        resource: &TensorResourceId,
        target: &MemoryDomain,
    ) -> ZeroCopyFeasibility {
        match self.readable_residency(resource, target) {
            Ok(_) => ZeroCopyFeasibility {
                feasible: true,
                reason: "current readable residency satisfies target domain".into(),
            },
            Err(error) => ZeroCopyFeasibility {
                feasible: false,
                reason: error.to_string(),
            },
        }
    }

    pub fn pressure_snapshot(&self) -> MemoryPressureSnapshot {
        let used_bytes = self.active_bytes();
        let mut snapshot = MemoryPressureSnapshot {
            runtime: pressure_for_usage(used_bytes, self.config.max_runtime_bytes),
            used_bytes,
            limit_bytes: self.config.max_runtime_bytes,
            cache: Some(pressure_for_usage(
                self.cached_bytes(),
                Some(self.config.max_cached_bytes),
            )),
            kv_cache: Some(self.pressure_for_class(MemoryAllocationClass::KvCache)),
            ..MemoryPressureSnapshot::default()
        };
        if let Some((id, arena)) = self.arenas.iter().next() {
            snapshot.arena = Some((*id, arena.pressure));
        }
        snapshot
    }

    pub fn pressure_for_provider_status(
        provider: &ProviderStatusSnapshot,
    ) -> MemoryPressureSnapshot {
        let mut snapshot = MemoryPressureSnapshot {
            runtime: MemoryPressureLevel::from(provider.pressure),
            provider: Some((
                provider.provider.clone(),
                MemoryPressureLevel::from(provider.pressure),
            )),
            ..MemoryPressureSnapshot::default()
        };
        if matches!(provider.admission, ProviderAdmissionDecision::Reject) {
            snapshot.runtime = MemoryPressureLevel::Saturated;
        }
        snapshot
    }

    pub fn pressure_for_device_metadata(
        device: &DeviceMetadata,
        used_bytes: u64,
        availability: DeviceAvailability,
    ) -> MemoryPressureSnapshot {
        let pressure = if !availability.accepts_new_work_by_default() {
            MemoryPressureLevel::Saturated
        } else {
            pressure_for_usage(used_bytes, Some(device.memory_capacity))
        };
        MemoryPressureSnapshot {
            runtime: pressure,
            device: Some((DeviceBinding::new(device.id.clone()), pressure)),
            used_bytes,
            limit_bytes: Some(device.memory_capacity),
            ..MemoryPressureSnapshot::default()
        }
    }

    pub fn allocation_request_for_tensor(
        descriptor: &TensorDescriptor,
        placement: MemoryPlacement,
        owner: MemoryAllocationOwner,
        affinity: ResourceAffinity,
    ) -> Result<MemoryAllocationRequest, MemoryError> {
        let size_bytes = descriptor
            .byte_size()
            .map_err(|_| MemoryError::SizeOverflow)?;
        Ok(MemoryAllocationRequest::new(
            MemoryAllocationClass::Tensor,
            size_bytes,
            placement,
            owner,
        )
        .with_affinity(affinity))
    }

    fn validate_request(&self, request: &MemoryAllocationRequest) -> Result<(), MemoryError> {
        if request.alignment_bytes == 0 || !request.alignment_bytes.is_power_of_two() {
            return Err(MemoryError::AllocationDenied {
                reason: "allocation alignment must be a non-zero power of two".into(),
            });
        }
        if request.placement.requires_pinned_host() && !self.config.allow_native_pinned_memory {
            return Err(MemoryError::PinnedMemoryUnavailable);
        }
        if request.placement.requires_pinned_host()
            && self.pinned_bytes().saturating_add(request.size_bytes)
                > self.config.max_pinned_host_bytes
            && self.config.max_pinned_host_bytes > 0
        {
            return Err(MemoryError::PinnedMemoryUnavailable);
        }
        if request.placement.is_browser_only() && !self.config.allow_browser_linear_memory {
            return Err(MemoryError::UnsupportedPlacement(
                MemoryPlacement::BrowserLinearMemory,
            ));
        }
        if let Some(limit) = self.config.max_runtime_bytes
            && request.size_bytes > limit
        {
            return Err(MemoryError::OutOfMemory {
                required: request.size_bytes,
                available: limit,
            });
        }
        Ok(())
    }

    fn find_reusable_allocation(
        &self,
        request: &MemoryAllocationRequest,
    ) -> Option<MemoryAllocationId> {
        self.allocations
            .iter()
            .find(|(_, allocation)| {
                allocation.state == MemoryAllocationState::Reusable
                    && allocation.request.class == request.class
                    && allocation.request.size_bytes >= request.size_bytes
                    && allocation.request.alignment_bytes >= request.alignment_bytes
                    && allocation.request.placement == request.placement
            })
            .map(|(id, _)| *id)
    }

    fn enforce_cache_limit(&mut self) {
        if self.config.max_cached_bytes == 0 {
            return;
        }
        while self.cached_bytes() > self.config.max_cached_bytes {
            let Some(id) = self
                .allocations
                .iter()
                .find(|(_, allocation)| allocation.state == MemoryAllocationState::Reusable)
                .map(|(id, _)| *id)
            else {
                break;
            };
            if let Some(allocation) = self.allocations.get_mut(&id) {
                allocation.state = MemoryAllocationState::Released;
            }
            self.observe(
                MemoryObservationKind::CacheEviction,
                Some(id),
                "cached memory allocation evicted",
            );
        }
    }

    fn current_pressure_for(&self, request: &MemoryAllocationRequest) -> MemoryPressureSnapshot {
        let used_bytes = self.active_bytes();
        MemoryPressureSnapshot {
            runtime: pressure_for_usage(used_bytes, self.config.max_runtime_bytes),
            allocation_class: Some(request.class),
            used_bytes,
            limit_bytes: self.config.max_runtime_bytes,
            ..MemoryPressureSnapshot::default()
        }
    }

    fn available_bytes(&self) -> Option<u64> {
        let limit = self.config.max_runtime_bytes?;
        let used = self.active_bytes();
        Some(limit.saturating_sub(used))
    }

    fn active_bytes(&self) -> u64 {
        self.bytes_for_state(MemoryAllocationState::Active)
    }

    fn cached_bytes(&self) -> u64 {
        self.bytes_for_state(MemoryAllocationState::Reusable)
    }

    fn pinned_bytes(&self) -> u64 {
        self.allocations
            .values()
            .filter(|allocation| {
                allocation.state == MemoryAllocationState::Active
                    && allocation.request.placement.requires_pinned_host()
            })
            .map(|allocation| allocation.request.size_bytes)
            .sum()
    }

    fn bytes_for_state(&self, state: MemoryAllocationState) -> u64 {
        self.allocations
            .values()
            .filter(|allocation| allocation.state == state)
            .map(|allocation| allocation.request.size_bytes)
            .sum()
    }

    fn pressure_for_class(&self, class: MemoryAllocationClass) -> MemoryPressureLevel {
        let used = self
            .allocations
            .values()
            .filter(|allocation| {
                allocation.state == MemoryAllocationState::Active
                    && allocation.request.class == class
            })
            .map(|allocation| allocation.request.size_bytes)
            .sum::<u64>();
        pressure_for_usage(used, self.config.max_runtime_bytes)
    }

    fn observe(
        &mut self,
        kind: MemoryObservationKind,
        allocation: Option<MemoryAllocationId>,
        message: impl Into<String>,
    ) {
        self.observations.push(MemoryObservation {
            kind,
            allocation,
            message: message.into(),
        });
    }

    fn observe_residency(
        &mut self,
        kind: MemoryObservationKind,
        resource: Option<TensorResourceId>,
        domain: Option<MemoryDomain>,
        message: impl AsRef<str>,
    ) {
        self.residency_observations
            .push(ResidencyObservation::redacted(
                kind, resource, domain, message,
            ));
    }
}

fn pressure_for_usage(used_bytes: u64, limit_bytes: Option<u64>) -> MemoryPressureLevel {
    let Some(limit_bytes) = limit_bytes else {
        return MemoryPressureLevel::Low;
    };
    if limit_bytes == 0 {
        return MemoryPressureLevel::Unknown;
    }
    let ratio = used_bytes.saturating_mul(100) / limit_bytes;
    match ratio {
        0..=49 => MemoryPressureLevel::Low,
        50..=74 => MemoryPressureLevel::Moderate,
        75..=94 => MemoryPressureLevel::High,
        _ => MemoryPressureLevel::Saturated,
    }
}

fn validate_memory_pool_identity(value: &str, label: &str) -> Result<(), MemoryError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(MemoryError::AllocationDenied {
            reason: format!("{label} must be non-empty"),
        });
    }
    if trimmed != value || trimmed.len() > 128 {
        return Err(MemoryError::AllocationDenied {
            reason: format!("{label} must be a stable logical identifier"),
        });
    }
    if contains_native_handle_marker(value) {
        return Err(MemoryError::NativeHandleForbidden {
            reason: format!("{label} must not encode native handles or pointers"),
        });
    }
    Ok(())
}

fn redact_memory_diagnostic(value: &str) -> String {
    if contains_native_handle_marker(value) {
        "[redacted]".into()
    } else {
        value.into()
    }
}

fn contains_native_handle_marker(value: &str) -> bool {
    let lowered = value.to_ascii_lowercase();
    [
        "0x",
        "ptr",
        "pointer",
        "dma-buf",
        "dmabuf",
        "fd=",
        "nt handle",
        "iosurface",
        "cuda ipc",
        "vulkan external",
        "metal object",
    ]
    .iter()
    .any(|marker| lowered.contains(marker))
}

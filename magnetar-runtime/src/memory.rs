//! Runtime-owned memory management contracts.
//!
//! Compute describes portable tensors and operations. The memory manager owns
//! the Runtime interpretation of allocation classes, placement, residency,
//! staging feasibility, zero-copy feasibility, pressure, and memory admission.

use crate::{
    AffinityError, DTypeDescriptor, DeviceAvailability, DeviceBinding, DeviceMetadata,
    HostStagingPolicy, ProviderAdmissionDecision, ProviderBinding, ProviderPressureLevel,
    ProviderStatusSnapshot, ResourceAffinity, TensorDescriptor, TensorResourceId,
};
use std::{collections::BTreeMap, error::Error, fmt};

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
        }
    }

    pub const fn with_allocation(mut self, allocation: MemoryAllocationId) -> Self {
        self.allocation = Some(allocation);
        self
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
    StagingInserted,
    StagingDenied,
    PressureChanged,
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
    allocations: BTreeMap<MemoryAllocationId, MemoryAllocation>,
    arenas: BTreeMap<MemoryArenaId, MemoryArena>,
    pending: BTreeMap<MemoryAllocationId, PendingMemoryAllocation>,
    tensor_residency: BTreeMap<TensorResourceId, TensorResidency>,
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
            allocations: BTreeMap::new(),
            arenas: BTreeMap::new(),
            pending: BTreeMap::new(),
            tensor_residency: BTreeMap::new(),
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

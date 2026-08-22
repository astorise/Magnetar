use crate::compute::redact_backend_diagnostic;
use crate::*;
use std::{collections::BTreeMap, error::Error, fmt};
/// Process-local identity of a Runtime execution context.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ExecutionContextId(u64);
impl ExecutionContextId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}
impl fmt::Display for ExecutionContextId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Process-local identity for resources resolved as one affinity cohort.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AffinityGroupId(u64);
impl AffinityGroupId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}
impl fmt::Display for AffinityGroupId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Stable name of the Provider that owns a live resource in one Runtime.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProviderBinding(String);
impl ProviderBinding {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl fmt::Display for ProviderBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Globally unique Device selected for a device-resident resource.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DeviceBinding(DeviceId);
impl DeviceBinding {
    pub fn new(id: DeviceId) -> Self {
        Self(id)
    }
    pub fn id(&self) -> &DeviceId {
        &self.0
    }
}
impl fmt::Display for DeviceBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Exact Capability implementation that created or constrains a live resource.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CapabilityBinding {
    id: CapabilityId,
    version: CapabilityVersion,
}
impl CapabilityBinding {
    pub fn new(id: CapabilityId, version: CapabilityVersion) -> Self {
        Self { id, version }
    }
    pub fn id(&self) -> &CapabilityId {
        &self.id
    }
    pub const fn version(&self) -> CapabilityVersion {
        self.version
    }
}
impl fmt::Display for CapabilityBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}", self.id, self.version)
    }
}

/// Canonical content fingerprint attached under a semantic artifact role.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ArtifactBinding {
    role: String,
    fingerprint: String,
}
impl ArtifactBinding {
    pub fn new(role: impl Into<String>, fingerprint: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            fingerprint: fingerprint.into(),
        }
    }
    pub fn role(&self) -> &str {
        &self.role
    }
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }
}

/// Recovery classification for state associated with an affinity.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum FallbackClass {
    Transparent,
    Restartable,
    ProviderPinned,
}

/// Logical point at which resolution or re-resolution is being considered.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ExecutionPhase {
    #[default]
    BeforeResourceCreation,
    AfterResourceCreation,
    AfterSubmittedWork,
    AfterObservableOutput,
}

/// Stable portable health states reported by Providers, Devices and Capabilities.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum HealthState {
    Unknown,
    Initializing,
    #[default]
    Available,
    Degraded,
    Saturated,
    Draining,
    Unavailable,
    Interrupted,
}
impl HealthState {
    pub const fn priority(self) -> u8 {
        match self {
            Self::Available => 0,
            Self::Degraded => 1,
            Self::Saturated => 2,
            Self::Draining => 3,
            Self::Initializing => 4,
            Self::Unknown => 5,
            Self::Unavailable => 6,
            Self::Interrupted => 7,
        }
    }
    pub const fn accepts_new_work_by_default(self) -> bool {
        matches!(self, Self::Available | Self::Degraded)
    }
}
impl Ord for HealthState {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority().cmp(&other.priority())
    }
}
impl PartialOrd for HealthState {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Stable health category reported by the host-facing Provider wrapper.
pub type ProviderHealth = HealthState;

/// Stable availability category for a candidate Device.
pub type DeviceAvailability = HealthState;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum HealthScope {
    Provider,
    Device,
    Capability,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct HealthTimestamp(u64);
impl HealthTimestamp {
    pub const fn unix_millis(value: u64) -> Self {
        Self(value)
    }
    pub const fn as_unix_millis(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct HealthTimeToLive(u64);
impl HealthTimeToLive {
    pub const fn millis(value: u64) -> Self {
        Self(value)
    }
    pub const fn as_millis(self) -> u64 {
        self.0
    }
    pub const fn is_expired_at(self, timestamp: HealthTimestamp, now: HealthTimestamp) -> bool {
        now.as_unix_millis()
            .saturating_sub(timestamp.as_unix_millis())
            > self.0
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HealthCapacityHints {
    pub queue_depth: Option<u32>,
    pub available_memory_bytes: Option<u64>,
    pub memory_pressure: Option<u8>,
    pub active_operations: Option<u32>,
    pub maximum_accepted_operations: Option<u32>,
    pub recommended_admission_limit: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HealthDiagnostic {
    pub scope: HealthScope,
    pub state: HealthState,
    pub code: Option<String>,
    pub message: Option<String>,
    pub trace_id: Option<String>,
}
impl HealthDiagnostic {
    pub fn new(scope: HealthScope, state: HealthState) -> Self {
        Self {
            scope,
            state,
            code: None,
            message: None,
            trace_id: None,
        }
    }
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }
    pub fn with_message(mut self, message: impl AsRef<str>) -> Self {
        self.message = Some(redact_backend_diagnostic(message.as_ref()));
        self
    }
    pub fn with_trace_id(mut self, trace_id: impl Into<String>) -> Self {
        self.trace_id = Some(trace_id.into());
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderHealthReport {
    pub provider: ProviderBinding,
    pub state: ProviderHealth,
    pub timestamp: Option<HealthTimestamp>,
    pub time_to_live: Option<HealthTimeToLive>,
    pub diagnostics: Vec<HealthDiagnostic>,
    pub capacity: HealthCapacityHints,
    pub devices: Vec<DeviceHealth>,
    pub capabilities: Vec<CapabilityHealth>,
}
impl ProviderHealthReport {
    pub fn new(provider: ProviderBinding, state: ProviderHealth) -> Self {
        Self {
            provider,
            state,
            timestamp: None,
            time_to_live: None,
            diagnostics: Vec::new(),
            capacity: HealthCapacityHints::default(),
            devices: Vec::new(),
            capabilities: Vec::new(),
        }
    }
    pub fn is_stale_at(&self, now: HealthTimestamp) -> bool {
        match (self.timestamp, self.time_to_live) {
            (Some(timestamp), Some(ttl)) => ttl.is_expired_at(timestamp, now),
            _ => false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceHealth {
    pub provider: ProviderBinding,
    pub device: DeviceBinding,
    pub state: DeviceAvailability,
    pub timestamp: Option<HealthTimestamp>,
    pub time_to_live: Option<HealthTimeToLive>,
    pub diagnostics: Vec<HealthDiagnostic>,
    pub capacity: HealthCapacityHints,
}
impl DeviceHealth {
    pub fn new(
        provider: ProviderBinding,
        device: DeviceBinding,
        state: DeviceAvailability,
    ) -> Self {
        Self {
            provider,
            device,
            state,
            timestamp: None,
            time_to_live: None,
            diagnostics: Vec::new(),
            capacity: HealthCapacityHints::default(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapabilityHealth {
    pub provider: ProviderBinding,
    pub capability: CapabilityBinding,
    pub state: HealthState,
    pub timestamp: Option<HealthTimestamp>,
    pub time_to_live: Option<HealthTimeToLive>,
    pub diagnostics: Vec<HealthDiagnostic>,
}
impl CapabilityHealth {
    pub fn new(
        provider: ProviderBinding,
        capability: CapabilityBinding,
        state: HealthState,
    ) -> Self {
        Self {
            provider,
            capability,
            state,
            timestamp: None,
            time_to_live: None,
            diagnostics: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HealthReport {
    Provider(ProviderHealthReport),
    Device(DeviceHealth),
    Capability(CapabilityHealth),
}

/// Immutable ownership and compatibility facts carried by one opaque resource.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceAffinity {
    provider: Option<ProviderBinding>,
    device: Option<DeviceBinding>,
    capabilities: BTreeMap<CapabilityId, CapabilityBinding>,
    artifacts: BTreeMap<String, ArtifactBinding>,
    execution_context: Option<ExecutionContextId>,
    group: Option<AffinityGroupId>,
    fallback: FallbackClass,
}
impl ResourceAffinity {
    pub fn new(fallback: FallbackClass) -> Self {
        Self {
            provider: None,
            device: None,
            capabilities: BTreeMap::new(),
            artifacts: BTreeMap::new(),
            execution_context: None,
            group: None,
            fallback,
        }
    }
    pub fn with_provider(mut self, binding: ProviderBinding) -> Self {
        self.provider = Some(binding);
        self
    }
    pub fn with_device(mut self, binding: DeviceBinding) -> Self {
        self.device = Some(binding);
        self
    }
    pub fn with_capability(mut self, binding: CapabilityBinding) -> Self {
        self.capabilities.insert(binding.id.clone(), binding);
        self
    }
    pub fn with_artifact(mut self, binding: ArtifactBinding) -> Self {
        self.artifacts.insert(binding.role.clone(), binding);
        self
    }
    pub fn with_execution_context(mut self, id: ExecutionContextId) -> Self {
        self.execution_context = Some(id);
        self
    }
    pub fn with_group(mut self, id: AffinityGroupId) -> Self {
        self.group = Some(id);
        self
    }
    pub fn with_fallback(mut self, fallback: FallbackClass) -> Self {
        self.fallback = fallback;
        self
    }
    pub fn provider(&self) -> Option<&ProviderBinding> {
        self.provider.as_ref()
    }
    pub fn device(&self) -> Option<&DeviceBinding> {
        self.device.as_ref()
    }
    pub fn capability(&self, id: &CapabilityId) -> Option<&CapabilityBinding> {
        self.capabilities.get(id)
    }
    pub fn capabilities(&self) -> impl Iterator<Item = &CapabilityBinding> {
        self.capabilities.values()
    }
    pub fn artifact(&self, role: &str) -> Option<&ArtifactBinding> {
        self.artifacts.get(role)
    }
    pub fn artifacts(&self) -> impl Iterator<Item = &ArtifactBinding> {
        self.artifacts.values()
    }
    pub const fn execution_context(&self) -> Option<ExecutionContextId> {
        self.execution_context
    }
    pub const fn group(&self) -> Option<AffinityGroupId> {
        self.group
    }
    pub const fn fallback(&self) -> FallbackClass {
        self.fallback
    }
    pub fn validate_with(&self, other: &Self) -> Result<(), AffinityError> {
        AffinityConstraints::try_from_affinities([self, other]).map(|_| ())
    }
}

/// A conflict-checked aggregation of all affinities consumed by one call.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AffinityConstraints {
    affinity: ResourceAffinity,
}
impl AffinityConstraints {
    pub fn new(fallback: FallbackClass) -> Self {
        Self {
            affinity: ResourceAffinity::new(fallback),
        }
    }
    pub fn try_from_affinities<'a>(
        affinities: impl IntoIterator<Item = &'a ResourceAffinity>,
    ) -> Result<Self, AffinityError> {
        let mut constraints = Self::new(FallbackClass::Transparent);
        for affinity in affinities {
            constraints.merge(affinity)?;
        }
        Ok(constraints)
    }
    pub fn affinity(&self) -> &ResourceAffinity {
        &self.affinity
    }
    pub fn into_affinity(self) -> ResourceAffinity {
        self.affinity
    }
    pub fn require_fallback(&mut self, fallback: FallbackClass) {
        self.affinity.fallback = self.affinity.fallback.max(fallback);
    }
    pub(crate) fn merge(&mut self, incoming: &ResourceAffinity) -> Result<(), AffinityError> {
        if let Some(found) = &incoming.provider {
            match &self.affinity.provider {
                Some(expected) if expected != found => {
                    return Err(AffinityError::ProviderMismatch {
                        expected: expected.clone(),
                        found: found.clone(),
                    });
                }
                None => self.affinity.provider = Some(found.clone()),
                _ => {}
            }
        }
        if let Some(found) = &incoming.device {
            match &self.affinity.device {
                Some(expected) if expected != found => {
                    return Err(AffinityError::DeviceMismatch {
                        expected: expected.clone(),
                        found: found.clone(),
                    });
                }
                None => self.affinity.device = Some(found.clone()),
                _ => {}
            }
        }
        if let Some(found) = incoming.execution_context {
            match self.affinity.execution_context {
                Some(expected) if expected != found => {
                    return Err(AffinityError::ExecutionContextMismatch { expected, found });
                }
                None => self.affinity.execution_context = Some(found),
                _ => {}
            }
        }
        if let Some(found) = incoming.group {
            match self.affinity.group {
                Some(expected) if expected != found => {
                    return Err(AffinityError::AffinityGroupMismatch { expected, found });
                }
                None => self.affinity.group = Some(found),
                _ => {}
            }
        }
        for (id, found) in &incoming.capabilities {
            match self.affinity.capabilities.get(id) {
                Some(expected) if expected != found => {
                    return Err(AffinityError::CapabilityMismatch {
                        id: id.clone(),
                        expected: expected.version,
                        found: found.version,
                    });
                }
                None => {
                    self.affinity.capabilities.insert(id.clone(), found.clone());
                }
                _ => {}
            }
        }
        for (role, found) in &incoming.artifacts {
            match self.affinity.artifacts.get(role) {
                Some(expected) if expected != found => {
                    return Err(AffinityError::ArtifactMismatch {
                        role: role.clone(),
                        expected: expected.fingerprint.clone(),
                        found: found.fingerprint.clone(),
                    });
                }
                None => {
                    self.affinity.artifacts.insert(role.clone(), found.clone());
                }
                _ => {}
            }
        }
        self.affinity.fallback = self.affinity.fallback.max(incoming.fallback);
        Ok(())
    }
}

/// Structured validation and constrained-resolution failures for affinities.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AffinityError {
    ProviderMismatch {
        expected: ProviderBinding,
        found: ProviderBinding,
    },
    DeviceMismatch {
        expected: DeviceBinding,
        found: DeviceBinding,
    },
    CapabilityMismatch {
        id: CapabilityId,
        expected: CapabilityVersion,
        found: CapabilityVersion,
    },
    ArtifactMismatch {
        role: String,
        expected: String,
        found: String,
    },
    ExecutionContextMismatch {
        expected: ExecutionContextId,
        found: ExecutionContextId,
    },
    AffinityGroupMismatch {
        expected: AffinityGroupId,
        found: AffinityGroupId,
    },
    BoundProviderUnavailable(ProviderBinding),
    BoundDeviceUnavailable(DeviceBinding),
    DeviceProviderMismatch {
        device: DeviceBinding,
        provider: ProviderBinding,
        owner: ProviderBinding,
    },
    ProviderDoesNotImplementCapability {
        provider: ProviderBinding,
        capability: CapabilityBinding,
    },
    NoCompatibleProvider(CapabilityBinding),
    PolicyRejectedProvider {
        capability: CapabilityBinding,
        policy: ResolutionPolicyId,
    },
    RuntimeNotInitialized,
}
impl fmt::Display for AffinityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProviderMismatch { expected, found } => write!(
                f,
                "resource provider mismatch: expected '{expected}', found '{found}'"
            ),
            Self::DeviceMismatch { expected, found } => write!(
                f,
                "resource device mismatch: expected '{expected}', found '{found}'"
            ),
            Self::CapabilityMismatch {
                id,
                expected,
                found,
            } => write!(
                f,
                "resource capability mismatch for '{id}': expected {expected}, found {found}"
            ),
            Self::ArtifactMismatch {
                role,
                expected,
                found,
            } => write!(
                f,
                "resource artifact mismatch for role '{role}': expected '{expected}', found '{found}'"
            ),
            Self::ExecutionContextMismatch { expected, found } => write!(
                f,
                "resource execution-context mismatch: expected {expected}, found {found}"
            ),
            Self::AffinityGroupMismatch { expected, found } => write!(
                f,
                "resource affinity-group mismatch: expected {expected}, found {found}"
            ),
            Self::BoundProviderUnavailable(provider) => {
                write!(f, "bound provider '{provider}' is unavailable")
            }
            Self::BoundDeviceUnavailable(device) => {
                write!(f, "bound device '{device}' is unavailable")
            }
            Self::DeviceProviderMismatch {
                device,
                provider,
                owner,
            } => write!(
                f,
                "bound device '{device}' belongs to provider '{owner}', not '{provider}'"
            ),
            Self::ProviderDoesNotImplementCapability {
                provider,
                capability,
            } => write!(
                f,
                "bound provider '{provider}' does not implement compatible capability '{capability}'"
            ),
            Self::NoCompatibleProvider(capability) => {
                write!(
                    f,
                    "no Provider implements compatible capability '{capability}'"
                )
            }
            Self::PolicyRejectedProvider { capability, policy } => write!(
                f,
                "resolution policy '{policy}' rejected every Provider for capability '{capability}'"
            ),
            Self::RuntimeNotInitialized => {
                write!(f, "runtime is not initialized for affinity resolution")
            }
        }
    }
}
impl Error for AffinityError {}

impl From<AffinityError> for ComputeError {
    fn from(error: AffinityError) -> Self {
        let message = error.to_string();
        match error {
            AffinityError::ProviderMismatch { expected, found } => ComputeError::new(
                ComputeErrorCode::ProviderPinnedResource,
                ComputeErrorPhase::AffinityValidation,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(
                ComputeDiagnostic::new()
                    .with_provider(found)
                    .with_rejected_candidate(expected),
            )
            .with_recovery_hint(RecoveryHint::ProviderPinned),
            AffinityError::DeviceMismatch { expected: _, found } => ComputeError::new(
                ComputeErrorCode::DeviceBoundResource,
                ComputeErrorPhase::AffinityValidation,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_device(found))
            .with_recovery_hint(RecoveryHint::ExplicitTransferRequired),
            AffinityError::CapabilityMismatch {
                id,
                expected,
                found: _,
            } => ComputeError::new(
                ComputeErrorCode::CapabilityVersionMismatch,
                ComputeErrorPhase::AffinityValidation,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(
                ComputeDiagnostic::new().with_capability(CapabilityBinding::new(id, expected)),
            ),
            AffinityError::ArtifactMismatch { .. } => ComputeError::new(
                ComputeErrorCode::ArtifactFingerprintMismatch,
                ComputeErrorPhase::AffinityValidation,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::ProviderPinned),
            AffinityError::ExecutionContextMismatch { .. } => ComputeError::new(
                ComputeErrorCode::ProviderPinnedResource,
                ComputeErrorPhase::Interruption,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::ProviderPinned),
            AffinityError::AffinityGroupMismatch { .. } => ComputeError::new(
                ComputeErrorCode::AffinityGroupMismatch,
                ComputeErrorPhase::AffinityValidation,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::ExplicitTransferRequired),
            AffinityError::BoundProviderUnavailable(provider) => ComputeError::new(
                ComputeErrorCode::ProviderUnavailable,
                ComputeErrorPhase::Interruption,
                ComputeErrorSeverity::Recoverable,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_provider(provider))
            .with_recovery_hint(RecoveryHint::ProviderPinned),
            AffinityError::BoundDeviceUnavailable(device) => ComputeError::new(
                ComputeErrorCode::DeviceUnavailable,
                ComputeErrorPhase::Interruption,
                ComputeErrorSeverity::Recoverable,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_device(device))
            .with_recovery_hint(RecoveryHint::ProviderPinned),
            AffinityError::DeviceProviderMismatch {
                device,
                provider,
                owner,
            } => ComputeError::new(
                ComputeErrorCode::DeviceBoundResource,
                ComputeErrorPhase::AffinityValidation,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(
                ComputeDiagnostic::new()
                    .with_device(device)
                    .with_provider(owner)
                    .with_rejected_candidate(provider),
            ),
            AffinityError::ProviderDoesNotImplementCapability {
                provider,
                capability,
            } => ComputeError::new(
                ComputeErrorCode::UnsupportedOperation,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(
                ComputeDiagnostic::new()
                    .with_provider(provider)
                    .with_capability(capability),
            ),
            AffinityError::NoCompatibleProvider(capability) => ComputeError::new(
                ComputeErrorCode::NoCompatibleProvider,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_capability(capability)),
            AffinityError::PolicyRejectedProvider {
                capability,
                policy: _,
            } => ComputeError::new(
                ComputeErrorCode::PolicyRejectedProvider,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_capability(capability)),
            AffinityError::RuntimeNotInitialized => ComputeError::new(
                ComputeErrorCode::ProviderUnavailable,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Recoverable,
                message,
            )
            .with_recovery_hint(RecoveryHint::RetryBeforeState),
        }
    }
}

/// Host-side opaque value paired with immutable Resource Affinity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AffinityResource<T> {
    value: T,
    affinity: ResourceAffinity,
}
impl<T> AffinityResource<T> {
    pub fn new(value: T, affinity: ResourceAffinity) -> Self {
        Self { value, affinity }
    }
    pub fn value(&self) -> &T {
        &self.value
    }
    pub fn affinity(&self) -> &ResourceAffinity {
        &self.affinity
    }
    pub fn into_parts(self) -> (T, ResourceAffinity) {
        (self.value, self.affinity)
    }
}

/// One coherent Provider and Capability selection plus affinity for its output.
pub struct AffinityResolution<'a> {
    pub(crate) provider: &'a dyn Provider,
    pub(crate) capability: &'a Capability,
    pub(crate) affinity: ResourceAffinity,
    pub(crate) decision: ResolutionDecision,
}
impl<'a> AffinityResolution<'a> {
    pub fn provider(&self) -> &'a dyn Provider {
        self.provider
    }
    pub fn capability(&self) -> &'a Capability {
        self.capability
    }
    pub fn affinity(&self) -> &ResourceAffinity {
        &self.affinity
    }
    pub fn decision(&self) -> &ResolutionDecision {
        &self.decision
    }
    pub fn into_affinity(self) -> ResourceAffinity {
        self.affinity
    }
}

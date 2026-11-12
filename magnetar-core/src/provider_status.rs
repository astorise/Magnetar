//! Health, lifecycle, readiness, pressure and admission status of Providers,
//! Devices and Capabilities.
//!
//! Core layer (`quality/layers.toml`). Re-exported from [`crate::affinity`],
//! where these types were first defined, so existing paths are unchanged.

use crate::binding::*;
use crate::capability::*;
use crate::compute_types::*;
use crate::device::*;
use crate::provider_abi::*;
use crate::resolution::*;
use std::path::PathBuf;
use std::{error::Error, fmt};

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

/// Runtime-managed Provider lifecycle, separate from readiness and health.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProviderLifecycleState {
    Registered,
    Loading,
    Initializing,
    #[default]
    Ready,
    Draining,
    Stopped,
    Failed,
    Removed,
}

impl ProviderLifecycleState {
    pub const fn accepts_new_work_by_default(self) -> bool {
        matches!(self, Self::Ready)
    }
    pub const fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Registered, Self::Loading)
                | (Self::Loading, Self::Initializing)
                | (Self::Loading, Self::Failed)
                | (Self::Initializing, Self::Ready)
                | (Self::Initializing, Self::Failed)
                | (Self::Ready, Self::Draining)
                | (Self::Ready, Self::Failed)
                | (Self::Draining, Self::Stopped)
                | (Self::Draining, Self::Failed)
                | (Self::Stopped, Self::Removed)
                | (Self::Failed, Self::Removed)
        )
    }
}

/// Provider internal functional state, separate from execution admission.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProviderHealthState {
    Unknown,
    #[default]
    Healthy,
    Degraded,
    Unhealthy,
    Failed,
}

impl ProviderHealthState {
    pub const fn accepts_new_work_by_default(self) -> bool {
        matches!(self, Self::Healthy | Self::Degraded)
    }
}

/// Whether a Provider should receive work in the current scope.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProviderReadinessState {
    NotReady,
    #[default]
    Ready,
    ReadOnly,
    Draining,
}

impl ProviderReadinessState {
    pub const fn accepts_new_work_by_default(self) -> bool {
        matches!(self, Self::Ready)
    }
}

/// Current Provider load/capacity pressure.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProviderPressureLevel {
    Unknown,
    #[default]
    Low,
    Moderate,
    High,
    Saturated,
}

/// Status-derived admission guidance for one scope.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProviderAdmissionDecision {
    #[default]
    Admit,
    PreferNot,
    Reject,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProviderStatusSeverity {
    Info,
    Warning,
    Recoverable,
    Terminal,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProviderStatusReason {
    None,
    Warming,
    Degraded,
    HighPressure,
    Saturated,
    Draining,
    Stale,
    DeviceUnavailable,
    CapabilityUnavailable,
    Interrupted,
    Administrative,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProviderInterruptionReason {
    DeviceReset,
    DriverLoss,
    DeviceRemoved,
    AllocatorFailure,
    OutOfMemoryRecovery,
    ThermalThrottling,
    AdministrativeDrain,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProviderStatusScope {
    Provider,
    Device,
    Capability,
    OperationFamily,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderAdmission {
    pub scope: ProviderStatusScope,
    pub decision: ProviderAdmissionDecision,
    pub reason: Option<String>,
}

impl ProviderAdmission {
    pub fn new(scope: ProviderStatusScope, decision: ProviderAdmissionDecision) -> Self {
        Self {
            scope,
            decision,
            reason: None,
        }
    }
    pub fn with_reason(mut self, reason: impl AsRef<str>) -> Self {
        self.reason = Some(redact_backend_diagnostic(reason.as_ref()));
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceStatus {
    pub provider: ProviderBinding,
    pub device: DeviceBinding,
    pub health: ProviderHealthState,
    pub readiness: ProviderReadinessState,
    pub pressure: ProviderPressureLevel,
    pub availability: DeviceAvailability,
    pub interruption: Option<ProviderInterruptionReason>,
    pub capacity: HealthCapacityHints,
}

impl DeviceStatus {
    pub fn from_health(report: DeviceHealth) -> Self {
        let health = ProviderHealthState::from(report.state);
        let readiness = ProviderReadinessState::from(report.state);
        let pressure = ProviderPressureLevel::from(report.state);
        Self {
            provider: report.provider,
            device: report.device,
            health,
            readiness,
            pressure,
            availability: report.state,
            interruption: matches!(report.state, HealthState::Interrupted)
                .then_some(ProviderInterruptionReason::DeviceReset),
            capacity: report.capacity,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapabilityStatus {
    pub provider: ProviderBinding,
    pub capability: CapabilityBinding,
    pub health: ProviderHealthState,
    pub readiness: ProviderReadinessState,
    pub pressure: ProviderPressureLevel,
}

impl CapabilityStatus {
    pub fn from_health(report: CapabilityHealth) -> Self {
        Self {
            provider: report.provider,
            capability: report.capability,
            health: ProviderHealthState::from(report.state),
            readiness: ProviderReadinessState::from(report.state),
            pressure: ProviderPressureLevel::from(report.state),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperationFamilyStatus {
    pub provider: ProviderBinding,
    pub family: ComputeOperationFamily,
    pub supported: bool,
    pub health: ProviderHealthState,
    pub readiness: ProviderReadinessState,
    pub pressure: ProviderPressureLevel,
}

impl OperationFamilyStatus {
    pub fn unsupported(provider: ProviderBinding, family: ComputeOperationFamily) -> Self {
        Self {
            provider,
            family,
            supported: false,
            health: ProviderHealthState::Healthy,
            readiness: ProviderReadinessState::NotReady,
            pressure: ProviderPressureLevel::Low,
        }
    }
    pub fn available(provider: ProviderBinding, family: ComputeOperationFamily) -> Self {
        Self {
            provider,
            family,
            supported: true,
            health: ProviderHealthState::Healthy,
            readiness: ProviderReadinessState::Ready,
            pressure: ProviderPressureLevel::Low,
        }
    }
}

/// Immutable Provider status captured for one Runtime decision point.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderStatusSnapshot {
    pub provider: ProviderBinding,
    pub lifecycle: ProviderLifecycleState,
    pub health: ProviderHealthState,
    pub readiness: ProviderReadinessState,
    pub pressure: ProviderPressureLevel,
    pub admission: ProviderAdmissionDecision,
    pub health_reason: ProviderStatusReason,
    pub readiness_reason: ProviderStatusReason,
    pub severity: ProviderStatusSeverity,
    pub interruption: Option<ProviderInterruptionReason>,
    pub timestamp: Option<HealthTimestamp>,
    pub time_to_live: Option<HealthTimeToLive>,
    pub diagnostics: Vec<HealthDiagnostic>,
    pub capacity: HealthCapacityHints,
    pub devices: Vec<DeviceStatus>,
    pub capabilities: Vec<CapabilityStatus>,
    pub operation_families: Vec<OperationFamilyStatus>,
    pub in_flight_operations: u32,
}

impl ProviderStatusSnapshot {
    pub fn from_health_report(report: ProviderHealthReport) -> Self {
        let health = ProviderHealthState::from(report.state);
        let readiness = ProviderReadinessState::from(report.state);
        let pressure = ProviderPressureLevel::from(report.state);
        let lifecycle = ProviderLifecycleState::from(report.state);
        let admission = provider_admission_from_dimensions(lifecycle, health, readiness, pressure);
        Self {
            provider: report.provider,
            lifecycle,
            health,
            readiness,
            pressure,
            admission,
            health_reason: ProviderStatusReason::from_health_state(report.state),
            readiness_reason: ProviderStatusReason::from_health_state(report.state),
            severity: ProviderStatusSeverity::from_health_state(report.state),
            interruption: matches!(report.state, HealthState::Interrupted)
                .then_some(ProviderInterruptionReason::DriverLoss),
            timestamp: report.timestamp,
            time_to_live: report.time_to_live,
            diagnostics: report.diagnostics,
            capacity: report.capacity,
            devices: report
                .devices
                .into_iter()
                .map(DeviceStatus::from_health)
                .collect(),
            capabilities: report
                .capabilities
                .into_iter()
                .map(CapabilityStatus::from_health)
                .collect(),
            operation_families: Vec::new(),
            in_flight_operations: 0,
        }
    }
    pub fn is_stale_at(&self, now: HealthTimestamp) -> bool {
        match (self.timestamp, self.time_to_live) {
            (Some(timestamp), Some(ttl)) => ttl.is_expired_at(timestamp, now),
            _ => false,
        }
    }
    pub const fn accepts_new_work_by_default(&self) -> bool {
        self.lifecycle.accepts_new_work_by_default()
            && self.health.accepts_new_work_by_default()
            && self.readiness.accepts_new_work_by_default()
            && !matches!(self.pressure, ProviderPressureLevel::Saturated)
            && matches!(self.admission, ProviderAdmissionDecision::Admit)
    }
    pub const fn provider_health_compat(&self) -> ProviderHealth {
        match (self.health, self.readiness, self.pressure, self.lifecycle) {
            (ProviderHealthState::Unknown, _, _, _) => HealthState::Unknown,
            (ProviderHealthState::Failed, _, _, _) => HealthState::Interrupted,
            (ProviderHealthState::Unhealthy, _, _, _) => HealthState::Unavailable,
            (_, ProviderReadinessState::Draining, _, _)
            | (_, _, _, ProviderLifecycleState::Draining) => HealthState::Draining,
            (_, ProviderReadinessState::NotReady, _, _) => HealthState::Initializing,
            (_, _, ProviderPressureLevel::Saturated, _) => HealthState::Saturated,
            (ProviderHealthState::Degraded, _, _, _) => HealthState::Degraded,
            _ => HealthState::Available,
        }
    }
    pub const fn is_drain_complete(&self) -> bool {
        matches!(self.lifecycle, ProviderLifecycleState::Draining) && self.in_flight_operations == 0
    }
    pub fn with_operation_family_status(mut self, status: OperationFamilyStatus) -> Self {
        self.operation_families.push(status);
        self
    }
    pub fn operation_family_status(
        &self,
        family: ComputeOperationFamily,
    ) -> Option<&OperationFamilyStatus> {
        self.operation_families
            .iter()
            .find(|status| status.family == family)
    }
    pub fn operation_family_or_capability_status(
        &self,
        family: ComputeOperationFamily,
    ) -> ProviderReadinessState {
        self.operation_family_status(family)
            .map(|status| status.readiness)
            .unwrap_or(self.readiness)
    }
    pub const fn pinned_work_allowed_during_drain(&self) -> bool {
        matches!(self.lifecycle, ProviderLifecycleState::Draining)
            && matches!(
                self.readiness,
                ProviderReadinessState::Ready | ProviderReadinessState::Draining
            )
            && !matches!(
                self.health,
                ProviderHealthState::Failed | ProviderHealthState::Unhealthy
            )
    }
}

impl ProviderStatusReason {
    pub const fn from_health_state(state: HealthState) -> Self {
        match state {
            HealthState::Unknown => Self::Stale,
            HealthState::Initializing => Self::Warming,
            HealthState::Available => Self::None,
            HealthState::Degraded => Self::Degraded,
            HealthState::Saturated => Self::Saturated,
            HealthState::Draining => Self::Draining,
            HealthState::Unavailable => Self::DeviceUnavailable,
            HealthState::Interrupted => Self::Interrupted,
        }
    }
}

impl ProviderStatusSeverity {
    pub const fn from_health_state(state: HealthState) -> Self {
        match state {
            HealthState::Available => Self::Info,
            HealthState::Degraded | HealthState::Saturated | HealthState::Draining => Self::Warning,
            HealthState::Unknown | HealthState::Initializing => Self::Recoverable,
            HealthState::Unavailable | HealthState::Interrupted => Self::Terminal,
        }
    }
}

pub const fn provider_admission_from_dimensions(
    lifecycle: ProviderLifecycleState,
    health: ProviderHealthState,
    readiness: ProviderReadinessState,
    pressure: ProviderPressureLevel,
) -> ProviderAdmissionDecision {
    if !lifecycle.accepts_new_work_by_default()
        || !health.accepts_new_work_by_default()
        || !readiness.accepts_new_work_by_default()
        || matches!(pressure, ProviderPressureLevel::Saturated)
    {
        ProviderAdmissionDecision::Reject
    } else if matches!(health, ProviderHealthState::Degraded)
        || matches!(pressure, ProviderPressureLevel::High)
    {
        ProviderAdmissionDecision::PreferNot
    } else {
        ProviderAdmissionDecision::Admit
    }
}

impl From<HealthState> for ProviderLifecycleState {
    fn from(value: HealthState) -> Self {
        match value {
            HealthState::Initializing => Self::Initializing,
            HealthState::Draining => Self::Draining,
            HealthState::Unavailable => Self::Stopped,
            HealthState::Interrupted => Self::Failed,
            HealthState::Unknown
            | HealthState::Available
            | HealthState::Degraded
            | HealthState::Saturated => Self::Ready,
        }
    }
}

impl From<HealthState> for ProviderHealthState {
    fn from(value: HealthState) -> Self {
        match value {
            HealthState::Unknown => Self::Unknown,
            HealthState::Available
            | HealthState::Initializing
            | HealthState::Saturated
            | HealthState::Draining => Self::Healthy,
            HealthState::Degraded => Self::Degraded,
            HealthState::Unavailable => Self::Unhealthy,
            HealthState::Interrupted => Self::Failed,
        }
    }
}

impl From<HealthState> for ProviderReadinessState {
    fn from(value: HealthState) -> Self {
        match value {
            HealthState::Available | HealthState::Degraded => Self::Ready,
            HealthState::Draining => Self::Draining,
            HealthState::Unknown
            | HealthState::Initializing
            | HealthState::Saturated
            | HealthState::Unavailable
            | HealthState::Interrupted => Self::NotReady,
        }
    }
}

impl From<HealthState> for ProviderPressureLevel {
    fn from(value: HealthState) -> Self {
        match value {
            HealthState::Unknown => Self::Unknown,
            HealthState::Saturated => Self::Saturated,
            HealthState::Degraded | HealthState::Draining => Self::Moderate,
            HealthState::Available | HealthState::Initializing => Self::Low,
            HealthState::Unavailable | HealthState::Interrupted => Self::High,
        }
    }
}

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
    pub estimated_queue_delay_millis: Option<u64>,
    pub utilization_percent: Option<u8>,
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

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProviderExecutionPhase {
    Prepare,
    Submit,
    Observe,
    Cancel,
    Complete,
    Release,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProviderExecutionErrorCode {
    ProviderHealthUnknown,
    ProviderInitializing,
    ProviderDegradedRejected,
    ProviderSaturated,
    ProviderDraining,
    ProviderUnavailable,
    ProviderInterrupted,
    DeviceHealthUnknown,
    DeviceSaturated,
    DeviceUnavailable,
    StaleHealthReport,
    CapabilityUnavailable,
    InvalidExecutionPlan,
    IncompatibleResourceAffinity,
    MemoryPlanRejected,
    UnsupportedOperation,
    UnsupportedDType,
    UnsupportedLayout,
    DataMovementFailed,
    MaterializationFailed,
    SubmissionFailed,
    ExecutionFailed,
    ExecutionInterrupted,
    CancellationUnsupported,
    CancellationFailed,
    ResourceExhausted,
    OutOfMemory,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderExecutionDiagnostic {
    pub provider: ProviderBinding,
    pub device: Option<DeviceBinding>,
    pub phase: ProviderExecutionPhase,
    pub stable_reason: Option<ProviderExecutionErrorCode>,
    pub detail: Option<String>,
    pub trace_id: Option<String>,
}

impl ProviderExecutionDiagnostic {
    pub fn new(provider: ProviderBinding, phase: ProviderExecutionPhase) -> Self {
        Self {
            provider,
            device: None,
            phase,
            stable_reason: None,
            detail: None,
            trace_id: None,
        }
    }
    pub fn with_device(mut self, device: Option<DeviceBinding>) -> Self {
        self.device = device;
        self
    }
    pub fn with_reason(mut self, reason: ProviderExecutionErrorCode) -> Self {
        self.stable_reason = Some(reason);
        self
    }
    pub fn with_detail(mut self, detail: impl AsRef<str>) -> Self {
        self.detail = Some(redact_backend_diagnostic(detail.as_ref()));
        self
    }
    pub fn with_trace_id(mut self, trace_id: impl Into<String>) -> Self {
        self.trace_id = Some(trace_id.into());
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderExecutionError {
    pub code: ProviderExecutionErrorCode,
    pub phase: ProviderExecutionPhase,
    pub provider: ProviderBinding,
    pub device: Option<DeviceBinding>,
    pub message: String,
    pub diagnostics: Vec<ProviderExecutionDiagnostic>,
}

impl ProviderExecutionError {
    pub fn new(
        code: ProviderExecutionErrorCode,
        phase: ProviderExecutionPhase,
        provider: ProviderBinding,
        device: Option<DeviceBinding>,
        message: impl Into<String>,
    ) -> Self {
        let diagnostic = ProviderExecutionDiagnostic::new(provider.clone(), phase)
            .with_device(device.clone())
            .with_reason(code);
        Self {
            code,
            phase,
            provider,
            device,
            message: message.into(),
            diagnostics: vec![diagnostic],
        }
    }
    pub fn with_diagnostic(mut self, diagnostic: ProviderExecutionDiagnostic) -> Self {
        self.diagnostics.push(diagnostic);
        self
    }
    pub fn scheduler_code(&self) -> SchedulerErrorCode {
        match self.code {
            ProviderExecutionErrorCode::ProviderHealthUnknown => {
                SchedulerErrorCode::ProviderHealthUnknown
            }
            ProviderExecutionErrorCode::ProviderInitializing => {
                SchedulerErrorCode::ProviderInitializing
            }
            ProviderExecutionErrorCode::ProviderDegradedRejected => {
                SchedulerErrorCode::ProviderDegradedRejected
            }
            ProviderExecutionErrorCode::ProviderSaturated => SchedulerErrorCode::ProviderSaturated,
            ProviderExecutionErrorCode::ProviderDraining => SchedulerErrorCode::ProviderDraining,
            ProviderExecutionErrorCode::ProviderUnavailable => {
                SchedulerErrorCode::ProviderUnavailable
            }
            ProviderExecutionErrorCode::ProviderInterrupted => {
                SchedulerErrorCode::ProviderInterrupted
            }
            ProviderExecutionErrorCode::DeviceHealthUnknown => {
                SchedulerErrorCode::DeviceHealthUnknown
            }
            ProviderExecutionErrorCode::DeviceSaturated => SchedulerErrorCode::DeviceSaturated,
            ProviderExecutionErrorCode::DeviceUnavailable => SchedulerErrorCode::DeviceUnavailable,
            ProviderExecutionErrorCode::StaleHealthReport => SchedulerErrorCode::StaleHealthReport,
            ProviderExecutionErrorCode::InvalidExecutionPlan => {
                SchedulerErrorCode::InvalidExecutionPlan
            }
            ProviderExecutionErrorCode::IncompatibleResourceAffinity => {
                SchedulerErrorCode::ResourceAffinityConflict
            }
            ProviderExecutionErrorCode::MemoryPlanRejected
            | ProviderExecutionErrorCode::OutOfMemory
            | ProviderExecutionErrorCode::ResourceExhausted => {
                SchedulerErrorCode::MemoryPlanInvalid
            }
            ProviderExecutionErrorCode::CancellationUnsupported => {
                SchedulerErrorCode::CancellationUnsupported
            }
            ProviderExecutionErrorCode::CancellationFailed => {
                SchedulerErrorCode::CancellationFailed
            }
            ProviderExecutionErrorCode::ExecutionInterrupted => {
                SchedulerErrorCode::ExecutionInterrupted
            }
            ProviderExecutionErrorCode::CapabilityUnavailable
            | ProviderExecutionErrorCode::UnsupportedOperation
            | ProviderExecutionErrorCode::UnsupportedDType
            | ProviderExecutionErrorCode::UnsupportedLayout
            | ProviderExecutionErrorCode::DataMovementFailed
            | ProviderExecutionErrorCode::MaterializationFailed
            | ProviderExecutionErrorCode::SubmissionFailed
            | ProviderExecutionErrorCode::ExecutionFailed => SchedulerErrorCode::ExecutionFailed,
        }
    }
}

impl fmt::Display for ProviderExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "provider execution error {:?} during {:?} for provider '{}': {}",
            self.code, self.phase, self.provider, self.message
        )
    }
}

impl Error for ProviderExecutionError {}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SchedulerErrorCode {
    InvalidExecutionPlan,
    QueueCapacityExceeded,
    ProviderHealthUnknown,
    ProviderInitializing,
    ProviderDegradedRejected,
    ProviderSaturated,
    ProviderDraining,
    ProviderUnavailable,
    ProviderInterrupted,
    DeviceHealthUnknown,
    DeviceSaturated,
    DeviceUnavailable,
    StaleHealthReport,
    ResourceAffinityConflict,
    MemoryPlanInvalid,
    SubmissionFailed,
    CancellationUnsupported,
    CancellationFailed,
    ExecutionFailed,
    ExecutionInterrupted,
    OperationTimeout,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ScheduledOperationId(u64);

impl ScheduledOperationId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for ScheduledOperationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Debug)]
pub enum ProviderError {
    ProviderAlreadyRegistered(String),
    DeviceAlreadyRegistered(DeviceId),
    DeviceProviderMismatch {
        device: DeviceId,
        expected: String,
        found: String,
    },
    InvalidDevice(String),
    IncompatibleApiVersion {
        provider: String,
        expected: u32,
        found: u32,
    },
    InvalidCapability(String),
    InvalidCapabilityVersion(String),
    ConflictingCapability {
        id: CapabilityId,
        version: CapabilityVersion,
    },
    MissingCapabilityDependency {
        capability: CapabilityId,
        dependency: CapabilityId,
        required: CapabilityVersion,
    },
    NoCompatibleProvider(CapabilityBinding),
    PolicyRejectedProvider {
        capability: CapabilityBinding,
        policy: ResolutionPolicyId,
    },
    Discovery {
        path: PathBuf,
        source: std::io::Error,
    },
    Load {
        path: PathBuf,
        message: String,
    },
    ProviderPathDenied {
        path: PathBuf,
    },
    UnsupportedDynamicAbi {
        path: PathBuf,
        expected_symbol: &'static str,
        expected_descriptor: ProviderAbiDescriptor,
    },
    InvalidAbiDescriptor(String),
    UnsupportedAbiVersion {
        expected: ProviderAbiVersion,
        found: ProviderAbiVersion,
    },
    InvalidMetadata(String),
    InvalidAdvertisement(String),
    InvalidDeviceMetadata(String),
    InitializationFailure(String),
    ProviderNotReady(String),
    ProviderDraining(String),
    ProviderSaturated(String),
    ExecutionRejected(String),
    ExecutionFailed(String),
    CancellationUnsupported(String),
    ResourceInvalid(String),
    PanicOrUnwindViolation(String),
    InternalProviderError(String),
    CancellationFailed {
        operation: ScheduledOperationId,
        reason: String,
    },
    Registration(String),
    Lifecycle(String),
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProviderAlreadyRegistered(x) => write!(f, "provider '{x}' is already registered"),
            Self::DeviceAlreadyRegistered(id) => write!(f, "device '{id}' is already registered"),
            Self::DeviceProviderMismatch {
                device,
                expected,
                found,
            } => write!(
                f,
                "device '{device}' is owned by provider '{found}', not registering provider '{expected}'"
            ),
            Self::InvalidDevice(message) => write!(f, "invalid device: {message}"),
            Self::IncompatibleApiVersion {
                provider,
                expected,
                found,
            } => write!(
                f,
                "provider '{provider}' targets API {found}, but Magnetar supports API {expected}"
            ),
            Self::InvalidCapability(message) => write!(f, "invalid capability: {message}"),
            Self::InvalidCapabilityVersion(version) => {
                write!(f, "invalid capability semantic version '{version}'")
            }
            Self::ConflictingCapability { id, version } => {
                write!(
                    f,
                    "capability '{id}@{version}' has a conflicting definition"
                )
            }
            Self::MissingCapabilityDependency {
                capability,
                dependency,
                required,
            } => write!(
                f,
                "capability '{capability}' requires unavailable dependency '{dependency}@{required}'"
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
            Self::Discovery { path, source } => write!(
                f,
                "could not discover providers in '{}': {source}",
                path.display()
            ),
            Self::Load { path, message } => {
                write!(f, "could not load provider '{}': {message}", path.display())
            }
            Self::ProviderPathDenied { path } => write!(
                f,
                "provider library '{}' is denied by loading policy",
                path.display()
            ),
            Self::UnsupportedDynamicAbi {
                path,
                expected_symbol,
                expected_descriptor,
            } => write!(
                f,
                "provider library '{}' does not implement stable Provider ABI symbol '{expected_symbol}' v{}.{}; Rust trait-object factories are not supported as the stable dynamic ABI",
                path.display(),
                expected_descriptor.abi_version.major,
                expected_descriptor.abi_version.minor
            ),
            Self::InvalidAbiDescriptor(message) => {
                write!(f, "invalid Provider ABI descriptor: {message}")
            }
            Self::UnsupportedAbiVersion { expected, found } => write!(
                f,
                "unsupported Provider ABI version {}.{}, expected {}.{}",
                found.major, found.minor, expected.major, expected.minor
            ),
            Self::InvalidMetadata(message) => write!(f, "invalid Provider metadata: {message}"),
            Self::InvalidAdvertisement(message) => {
                write!(f, "invalid Provider Capability advertisement: {message}")
            }
            Self::InvalidDeviceMetadata(message) => {
                write!(f, "invalid Provider Device metadata: {message}")
            }
            Self::InitializationFailure(message) => {
                write!(f, "provider initialization failed: {message}")
            }
            Self::ProviderNotReady(message) => write!(f, "provider is not ready: {message}"),
            Self::ProviderDraining(message) => write!(f, "provider is draining: {message}"),
            Self::ProviderSaturated(message) => write!(f, "provider is saturated: {message}"),
            Self::ExecutionRejected(message) => write!(f, "provider rejected execution: {message}"),
            Self::ExecutionFailed(message) => write!(f, "provider execution failed: {message}"),
            Self::CancellationUnsupported(message) => {
                write!(f, "provider cancellation is unsupported: {message}")
            }
            Self::ResourceInvalid(message) => write!(f, "provider resource is invalid: {message}"),
            Self::PanicOrUnwindViolation(message) => {
                write!(f, "provider violated panic/unwind ABI boundary: {message}")
            }
            Self::InternalProviderError(message) => {
                write!(f, "internal provider error: {message}")
            }
            Self::CancellationFailed { operation, reason } => {
                write!(
                    f,
                    "provider cancellation failed for operation '{operation}': {reason}"
                )
            }
            Self::Registration(x) => write!(f, "provider registration failed: {x}"),
            Self::Lifecycle(x) => write!(f, "provider lifecycle operation failed: {x}"),
        }
    }
}

impl Error for ProviderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        if let Self::Discovery { source, .. } = self {
            Some(source)
        } else {
            None
        }
    }
}

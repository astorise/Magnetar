use crate::affinity::{
    CapabilityBinding, DeviceAvailability, DeviceBinding, ExecutionPhase, FallbackClass,
    HealthState, ProviderAdmissionDecision, ProviderBinding, ProviderHealth, ProviderHealthState,
    ProviderLifecycleState, ProviderPressureLevel, ProviderReadinessState, ProviderStatusReason,
    ProviderStatusSnapshot, ResourceAffinity,
};
use crate::capability::{CapabilityId, CapabilityVersion};
use std::fmt;
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ResolutionPolicyId(String);
impl ResolutionPolicyId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl fmt::Display for ResolutionPolicyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Built-in runtime policy families. Preference placeholders currently use the
/// deterministic ordering after applying their eligibility gates.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum BuiltInResolutionPolicy {
    #[default]
    Deterministic,
    Priority,
    Availability,
    PerformancePreferred,
    EnergyPreferred,
    MemoryConstrained,
}
impl BuiltInResolutionPolicy {
    pub fn id(self) -> ResolutionPolicyId {
        let id = match self {
            Self::Deterministic => "magnetar:policy/deterministic",
            Self::Priority => "magnetar:policy/priority",
            Self::Availability => "magnetar:policy/availability",
            Self::PerformancePreferred => "magnetar:policy/performance-preferred",
            Self::EnergyPreferred => "magnetar:policy/energy-preferred",
            Self::MemoryConstrained => "magnetar:policy/memory-constrained",
        };
        ResolutionPolicyId::new(id)
    }
}

/// A stable, inspectable candidate considered by a [`ResolutionPolicy`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolutionCandidate {
    pub provider: ProviderBinding,
    pub capability: CapabilityBinding,
    pub device: Option<DeviceBinding>,
    pub provider_health: ProviderHealth,
    pub provider_status: ProviderStatusSnapshot,
    pub capability_health: Option<HealthState>,
    pub device_availability: DeviceAvailability,
    pub affinity_compatible: bool,
    pub priority: i32,
}
impl ResolutionCandidate {
    fn sort_key(&self) -> (&ProviderBinding, &CapabilityBinding, Option<&DeviceBinding>) {
        (&self.provider, &self.capability, self.device.as_ref())
    }
}

/// Policy input assembled by the Runtime before execution begins.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolutionContext {
    pub requested_capability: CapabilityId,
    pub requested_version: CapabilityVersion,
    pub candidates: Vec<ResolutionCandidate>,
    pub affinity: Option<ResourceAffinity>,
    pub fallback: FallbackClass,
    pub execution_phase: ExecutionPhase,
    pub replayable_input: bool,
}

/// Stable reason a candidate was rejected before or by policy selection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResolutionRejectionReason {
    IncompatibleCapability,
    ProviderHealthUnknown,
    ProviderInitializing,
    ProviderSaturated,
    ProviderDraining,
    ProviderStatusStale,
    ProviderUnavailable,
    ProviderInterrupted,
    DeviceHealthUnknown,
    DeviceSaturated,
    DeviceUnavailable,
    DeviceInterrupted,
    CapabilityUnavailable,
    AffinityIncompatible,
    FallbackNotAllowed,
    PolicyRejected,
}

/// Stable rejection record; backend-specific strings are intentionally absent.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolutionCandidateRejection {
    pub provider: ProviderBinding,
    pub capability: CapabilityBinding,
    pub reason: ResolutionRejectionReason,
}

/// Stable decision reason emitted by policy evaluation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResolutionDecisionReason {
    SelectedDeterministically,
    SelectedByPriority,
    SelectedByAvailability,
    SelectedByPreferencePlaceholder,
    PreservedAffinity,
    NoCompatibleProvider,
    PolicyRejectedAllCandidates,
}

/// Inspectable result of applying a Resolution Policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolutionDecision {
    pub policy_id: ResolutionPolicyId,
    pub selected_provider: Option<ProviderBinding>,
    pub selected_device: Option<DeviceBinding>,
    pub selected_capability: Option<CapabilityBinding>,
    pub selected_provider_status: Option<ProviderStatusSnapshot>,
    pub reason: ResolutionDecisionReason,
    pub rejected_candidates: Vec<ResolutionCandidateRejection>,
}

/// Selects one candidate from a deterministic context.
pub trait ResolutionPolicy {
    fn id(&self) -> ResolutionPolicyId;
    fn decide(&self, context: &ResolutionContext) -> ResolutionDecision;
}
impl ResolutionPolicy for BuiltInResolutionPolicy {
    fn id(&self) -> ResolutionPolicyId {
        (*self).id()
    }

    fn decide(&self, context: &ResolutionContext) -> ResolutionDecision {
        let mut rejected_candidates = Vec::new();
        let mut eligible = Vec::new();
        for candidate in &context.candidates {
            if !candidate
                .capability
                .version()
                .is_compatible_with(context.requested_version)
            {
                rejected_candidates.push(rejection(
                    candidate,
                    ResolutionRejectionReason::IncompatibleCapability,
                ));
            } else if let Some(reason) = provider_status_rejection(&candidate.provider_status) {
                rejected_candidates.push(rejection(candidate, reason));
            } else if let Some(reason) = candidate
                .capability_health
                .and_then(capability_health_rejection)
            {
                rejected_candidates.push(rejection(candidate, reason));
            } else if let Some(reason) = device_health_rejection(candidate.device_availability) {
                rejected_candidates.push(rejection(candidate, reason));
            } else if !candidate.affinity_compatible {
                rejected_candidates.push(rejection(
                    candidate,
                    ResolutionRejectionReason::AffinityIncompatible,
                ));
            } else if context.execution_phase >= ExecutionPhase::AfterObservableOutput
                || (context.fallback >= FallbackClass::Restartable && !context.replayable_input)
            {
                rejected_candidates.push(rejection(
                    candidate,
                    ResolutionRejectionReason::FallbackNotAllowed,
                ));
            } else {
                eligible.push(candidate);
            }
        }

        eligible.sort_by(|left, right| match self {
            Self::Priority => right
                .priority
                .cmp(&left.priority)
                .then_with(|| left.sort_key().cmp(&right.sort_key())),
            Self::Availability => left
                .provider_health
                .cmp(&right.provider_health)
                .then_with(|| left.device_availability.cmp(&right.device_availability))
                .then_with(|| left.sort_key().cmp(&right.sort_key())),
            _ => left.sort_key().cmp(&right.sort_key()),
        });

        let reason = match self {
            Self::Deterministic => ResolutionDecisionReason::SelectedDeterministically,
            Self::Priority => ResolutionDecisionReason::SelectedByPriority,
            Self::Availability => ResolutionDecisionReason::SelectedByAvailability,
            Self::PerformancePreferred | Self::EnergyPreferred | Self::MemoryConstrained => {
                ResolutionDecisionReason::SelectedByPreferencePlaceholder
            }
        };
        let selected = eligible.first();
        ResolutionDecision {
            policy_id: self.id(),
            selected_provider: selected.map(|candidate| candidate.provider.clone()),
            selected_device: selected.and_then(|candidate| candidate.device.clone()),
            selected_capability: selected.map(|candidate| candidate.capability.clone()),
            selected_provider_status: selected.map(|candidate| candidate.provider_status.clone()),
            reason: if selected.is_some() {
                reason
            } else if context.candidates.is_empty() {
                ResolutionDecisionReason::NoCompatibleProvider
            } else {
                ResolutionDecisionReason::PolicyRejectedAllCandidates
            },
            rejected_candidates,
        }
    }
}

fn rejection(
    candidate: &ResolutionCandidate,
    reason: ResolutionRejectionReason,
) -> ResolutionCandidateRejection {
    ResolutionCandidateRejection {
        provider: candidate.provider.clone(),
        capability: candidate.capability.clone(),
        reason,
    }
}

fn provider_health_rejection(health: ProviderHealth) -> Option<ResolutionRejectionReason> {
    match health {
        HealthState::Unknown => Some(ResolutionRejectionReason::ProviderHealthUnknown),
        HealthState::Initializing => Some(ResolutionRejectionReason::ProviderInitializing),
        HealthState::Available | HealthState::Degraded => None,
        HealthState::Saturated => Some(ResolutionRejectionReason::ProviderSaturated),
        HealthState::Draining => Some(ResolutionRejectionReason::ProviderDraining),
        HealthState::Unavailable => Some(ResolutionRejectionReason::ProviderUnavailable),
        HealthState::Interrupted => Some(ResolutionRejectionReason::ProviderInterrupted),
    }
}

fn provider_status_rejection(status: &ProviderStatusSnapshot) -> Option<ResolutionRejectionReason> {
    if matches!(status.health_reason, ProviderStatusReason::Stale) {
        return Some(ResolutionRejectionReason::ProviderStatusStale);
    }
    if matches!(status.lifecycle, ProviderLifecycleState::Failed) {
        return Some(ResolutionRejectionReason::ProviderUnavailable);
    }
    if matches!(
        status.health,
        ProviderHealthState::Unknown | ProviderHealthState::Unhealthy | ProviderHealthState::Failed
    ) {
        return provider_health_rejection(status.provider_health_compat());
    }
    if matches!(status.readiness, ProviderReadinessState::NotReady) {
        return Some(ResolutionRejectionReason::ProviderInitializing);
    }
    if matches!(status.readiness, ProviderReadinessState::Draining)
        || matches!(status.lifecycle, ProviderLifecycleState::Draining)
    {
        return Some(ResolutionRejectionReason::ProviderDraining);
    }
    if matches!(status.pressure, ProviderPressureLevel::Saturated)
        || matches!(status.admission, ProviderAdmissionDecision::Reject)
    {
        return Some(ResolutionRejectionReason::ProviderSaturated);
    }
    None
}

fn device_health_rejection(health: DeviceAvailability) -> Option<ResolutionRejectionReason> {
    match health {
        HealthState::Unknown => Some(ResolutionRejectionReason::DeviceHealthUnknown),
        HealthState::Initializing => Some(ResolutionRejectionReason::DeviceHealthUnknown),
        HealthState::Available | HealthState::Degraded | HealthState::Draining => None,
        HealthState::Saturated => Some(ResolutionRejectionReason::DeviceSaturated),
        HealthState::Unavailable => Some(ResolutionRejectionReason::DeviceUnavailable),
        HealthState::Interrupted => Some(ResolutionRejectionReason::DeviceInterrupted),
    }
}

fn capability_health_rejection(health: HealthState) -> Option<ResolutionRejectionReason> {
    match health {
        HealthState::Available | HealthState::Degraded => None,
        HealthState::Unknown
        | HealthState::Initializing
        | HealthState::Saturated
        | HealthState::Draining
        | HealthState::Unavailable
        | HealthState::Interrupted => Some(ResolutionRejectionReason::CapabilityUnavailable),
    }
}

pub use crate::binding::*;
use crate::capability::*;
use crate::compute::*;
use crate::provider::*;
pub use crate::provider_status::*;
use crate::resolution::*;

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

#[cfg(test)]
mod tests;

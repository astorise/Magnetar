//! Runtime-owned Kernel Registry.
//!
//! The registry stores validated Kernel advertisements and produces metadata
//! candidates for Runtime selection. It never exposes native function pointers,
//! Provider handles, or direct client authority over Provider selection.

use crate::affinity::*;
use crate::compute::*;
use crate::execution_graph::*;
use crate::kernel::*;
use crate::model_instance::*;
use crate::operator::*;
use std::collections::{BTreeMap, BTreeSet};
use std::{error::Error, fmt};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelRegistrationAuthority {
    Provider,
    RuntimeTestFixture,
    Client,
    Component,
}

impl KernelRegistrationAuthority {
    pub const fn may_register_kernel(self) -> bool {
        matches!(self, Self::Provider | Self::RuntimeTestFixture)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelCandidateRejection {
    OperatorMismatch,
    OperatorVersionUnsupported,
    DTypeUnsupported,
    LayoutUnsupported,
    ShapeUnsupported,
    MemoryClassUnsupported,
    ExecutionModeUnsupported,
    ResourceAffinityConflict,
    ProviderUnavailable,
    ProviderNotReady,
    ProviderSaturated,
    DeviceUnavailable,
    DeviceIncompatible,
    ProviderFeatureMissing,
    DeviceFeatureMissing,
    WorkspaceUnavailable,
    BatchingUnsupported,
    AdapterUnsupported,
    KvCacheUnsupported,
    PrefixCacheUnsupported,
    ConformanceMissing,
    ConformanceFailed,
    PolicyDenied,
    StaleRegistryEntry,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelRegistryError {
    RegistryUnavailable,
    AdvertisementInvalid(String),
    RegistrationDenied(KernelRegistrationAuthority),
    CandidateNotFound { operator: OperatorId },
    CandidateIncompatible { reason: KernelCandidateRejection },
    SelectionFailed(String),
    PolicyDenied(String),
    ConformanceRequired,
    ConformanceMissing { kernel: String },
    ConformanceFailed { kernel: String },
    ProviderUnavailable { provider: ProviderBinding },
    ProviderNotReady { provider: ProviderBinding },
    ProviderSaturated { provider: ProviderBinding },
    DeviceUnavailable { device: DeviceBinding },
    DeviceIncompatible { device: DeviceBinding },
    MemoryInfeasible(String),
    WorkspaceUnavailable,
    ResourceAffinityConflict(String),
    BrowserFeatureUnsupported(String),
    Internal(String),
}

impl KernelRegistryError {
    pub const fn code(&self) -> &'static str {
        match self {
            Self::RegistryUnavailable => "kernel-registry-unavailable",
            Self::AdvertisementInvalid(_) => "kernel-advertisement-invalid",
            Self::RegistrationDenied(_) => "kernel-registration-denied",
            Self::CandidateNotFound { .. } => "kernel-candidate-not-found",
            Self::CandidateIncompatible { .. } => "kernel-candidate-incompatible",
            Self::SelectionFailed(_) => "kernel-selection-failed",
            Self::PolicyDenied(_) => "kernel-policy-denied",
            Self::ConformanceRequired => "kernel-conformance-required",
            Self::ConformanceMissing { .. } => "kernel-conformance-missing",
            Self::ConformanceFailed { .. } => "kernel-conformance-failed",
            Self::ProviderUnavailable { .. } => "kernel-provider-unavailable",
            Self::ProviderNotReady { .. } => "kernel-provider-not-ready",
            Self::ProviderSaturated { .. } => "kernel-provider-saturated",
            Self::DeviceUnavailable { .. } => "kernel-device-unavailable",
            Self::DeviceIncompatible { .. } => "kernel-device-incompatible",
            Self::MemoryInfeasible(_) => "kernel-memory-infeasible",
            Self::WorkspaceUnavailable => "kernel-workspace-unavailable",
            Self::ResourceAffinityConflict(_) => "kernel-resource-affinity-conflict",
            Self::BrowserFeatureUnsupported(_) => "kernel-browser-feature-unsupported",
            Self::Internal(_) => "internal-kernel-registry",
        }
    }
}

impl fmt::Display for KernelRegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RegistryUnavailable => write!(f, "Kernel Registry is unavailable"),
            Self::AdvertisementInvalid(reason) => {
                write!(f, "Kernel advertisement is invalid: {reason}")
            }
            Self::RegistrationDenied(authority) => {
                write!(f, "Kernel registration denied for {authority:?}")
            }
            Self::CandidateNotFound { operator } => {
                write!(f, "no Kernel candidate found for Operator {operator}")
            }
            Self::CandidateIncompatible { reason } => {
                write!(f, "Kernel candidate is incompatible: {reason:?}")
            }
            Self::SelectionFailed(reason) => write!(f, "Kernel selection failed: {reason}"),
            Self::PolicyDenied(reason) => write!(f, "Kernel policy denied selection: {reason}"),
            Self::ConformanceRequired => write!(f, "Kernel conformance is required"),
            Self::ConformanceMissing { kernel } => {
                write!(f, "Kernel conformance is missing for {kernel}")
            }
            Self::ConformanceFailed { kernel } => {
                write!(f, "Kernel conformance failed for {kernel}")
            }
            Self::ProviderUnavailable { provider } => {
                write!(f, "Kernel Provider unavailable: {provider}")
            }
            Self::ProviderNotReady { provider } => {
                write!(f, "Kernel Provider not ready: {provider}")
            }
            Self::ProviderSaturated { provider } => {
                write!(f, "Kernel Provider saturated: {provider}")
            }
            Self::DeviceUnavailable { device } => {
                write!(f, "Kernel Device unavailable: {device}")
            }
            Self::DeviceIncompatible { device } => {
                write!(f, "Kernel Device incompatible: {device}")
            }
            Self::MemoryInfeasible(reason) => write!(f, "Kernel memory infeasible: {reason}"),
            Self::WorkspaceUnavailable => write!(f, "Kernel workspace unavailable"),
            Self::ResourceAffinityConflict(reason) => {
                write!(f, "Kernel Resource Affinity conflict: {reason}")
            }
            Self::BrowserFeatureUnsupported(feature) => {
                write!(f, "Kernel browser feature unsupported: {feature}")
            }
            Self::Internal(reason) => write!(f, "internal Kernel Registry error: {reason}"),
        }
    }
}

impl Error for KernelRegistryError {}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelRegistryEntry {
    pub advertisement: KernelAdvertisement,
    pub authority: KernelRegistrationAuthority,
    pub active: bool,
    pub invalidation_reason: Option<String>,
}

impl KernelRegistryEntry {
    pub fn new(advertisement: KernelAdvertisement, authority: KernelRegistrationAuthority) -> Self {
        Self {
            advertisement,
            authority,
            active: true,
            invalidation_reason: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelSelectionRequest {
    pub request_id: String,
    pub operator: OperatorId,
    pub operator_version: u32,
    pub graph_plan: Option<ExecutionGraphId>,
    pub model_instance: Option<ModelInstanceId>,
    pub inputs: Vec<KernelResource>,
    pub outputs: Vec<KernelResource>,
    pub dtype_requirements: BTreeSet<ComputeDType>,
    pub layout_requirements: BTreeSet<TensorLayoutKind>,
    pub memory_class_requirements: BTreeSet<KernelMemoryClass>,
    pub affinity: ResourceAffinity,
    pub deterministic_required: bool,
    pub precision: ComputePrecision,
    pub execution_mode: Option<KernelExecutionMode>,
    pub batching: Option<KernelBatchMetadata>,
    pub kv_cache: Option<KernelKvCacheMetadata>,
    pub prefix_cache: Option<KernelPrefixCacheMetadata>,
    pub adapter_methods: BTreeSet<String>,
    pub deadline_millis: Option<u64>,
    pub policy: BTreeMap<String, String>,
    pub observability_correlation: Option<String>,
    pub require_conformance: bool,
    pub browser_target: bool,
}

impl KernelSelectionRequest {
    pub fn new(
        request_id: impl Into<String>,
        operator: OperatorId,
        affinity: ResourceAffinity,
    ) -> Self {
        let operator_version = operator.version();
        Self {
            request_id: request_id.into(),
            operator,
            operator_version,
            graph_plan: None,
            model_instance: None,
            inputs: Vec::new(),
            outputs: Vec::new(),
            dtype_requirements: BTreeSet::new(),
            layout_requirements: BTreeSet::new(),
            memory_class_requirements: BTreeSet::new(),
            affinity,
            deterministic_required: false,
            precision: ComputePrecision::Default,
            execution_mode: None,
            batching: None,
            kv_cache: None,
            prefix_cache: None,
            adapter_methods: BTreeSet::new(),
            deadline_millis: None,
            policy: BTreeMap::new(),
            observability_correlation: None,
            require_conformance: false,
            browser_target: false,
        }
    }

    pub fn with_input(mut self, input: KernelResource) -> Self {
        self.inputs.push(input);
        self
    }

    pub fn with_output(mut self, output: KernelResource) -> Self {
        self.outputs.push(output);
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelCandidate {
    pub kernel: KernelId,
    pub provider: ProviderBinding,
    pub device: Option<DeviceBinding>,
    pub operator: OperatorId,
    pub compatible: bool,
    pub dtype_compatible: bool,
    pub layout_compatible: bool,
    pub shape_compatible: bool,
    pub memory_compatible: bool,
    pub workspace_feasible: bool,
    pub affinity_compatible: bool,
    pub deterministic_compatible: bool,
    pub precision_compatible: bool,
    pub provider_ready: bool,
    pub device_ready: bool,
    pub provider_status: Option<ProviderStatusSnapshot>,
    pub device_status: Option<DeviceStatus>,
    pub pressure_score: u32,
    pub conformance_status: Option<String>,
    pub estimated_cost: u64,
    pub fallback_rank: u32,
    pub rejection_reason: Option<KernelCandidateRejection>,
}

impl KernelCandidate {
    fn rejected(advertisement: &KernelAdvertisement, reason: KernelCandidateRejection) -> Self {
        Self {
            kernel: advertisement.id.clone(),
            provider: advertisement.id.provider.clone(),
            device: advertisement.devices.iter().next().cloned(),
            operator: advertisement.implemented_operator.clone(),
            compatible: false,
            dtype_compatible: true,
            layout_compatible: true,
            shape_compatible: true,
            memory_compatible: true,
            workspace_feasible: true,
            affinity_compatible: true,
            deterministic_compatible: true,
            precision_compatible: true,
            provider_ready: true,
            device_ready: true,
            provider_status: None,
            device_status: None,
            pressure_score: 0,
            conformance_status: advertisement.id.conformance_profile.clone(),
            estimated_cost: 0,
            fallback_rank: u32::MAX,
            rejection_reason: Some(reason),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelSelectionResult {
    pub request_id: String,
    pub selected: Option<KernelCandidate>,
    pub candidates: Vec<KernelCandidate>,
    pub fallback_chain: Vec<KernelCandidate>,
    pub observations: Vec<KernelObservation>,
}

#[derive(Clone, Debug, Default)]
pub struct KernelRegistry {
    entries: BTreeMap<String, KernelRegistryEntry>,
    provider_statuses: BTreeMap<ProviderBinding, ProviderStatusSnapshot>,
    device_statuses: BTreeMap<DeviceBinding, DeviceStatus>,
    provider_features: BTreeMap<ProviderBinding, BTreeSet<String>>,
    device_features: BTreeMap<DeviceBinding, BTreeSet<String>>,
    revoked_conformance_profiles: BTreeSet<String>,
    policy_generation: u64,
    observations: Vec<KernelObservation>,
}

impl KernelRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn entries(&self) -> impl Iterator<Item = &KernelRegistryEntry> {
        self.entries.values()
    }

    pub fn observations(&self) -> &[KernelObservation] {
        &self.observations
    }

    pub fn set_provider_status(&mut self, status: ProviderStatusSnapshot) {
        let provider = status.provider.clone();
        let reason = match status.provider_health_compat() {
            HealthState::Draining => Some("provider draining"),
            HealthState::Initializing => Some("provider not ready"),
            HealthState::Saturated => Some("provider saturated"),
            HealthState::Unavailable | HealthState::Interrupted => Some("provider unavailable"),
            HealthState::Unknown => Some("provider status stale"),
            HealthState::Available | HealthState::Degraded => None,
        };
        self.provider_statuses.insert(provider.clone(), status);
        if let Some(reason) = reason {
            self.invalidate_provider(&provider, reason);
        }
    }

    pub fn set_device_status(&mut self, status: DeviceStatus) {
        let device = status.device.clone();
        let reason = match status.availability {
            HealthState::Available | HealthState::Degraded | HealthState::Draining => None,
            HealthState::Saturated => Some("device pressure saturated"),
            HealthState::Unknown | HealthState::Initializing => Some("device not ready"),
            HealthState::Unavailable => Some("device unavailable"),
            HealthState::Interrupted => Some("device lost"),
        };
        self.device_statuses.insert(device.clone(), status);
        if let Some(reason) = reason {
            self.invalidate_device(&device, reason);
        }
    }

    pub fn set_provider_features(
        &mut self,
        provider: ProviderBinding,
        features: impl IntoIterator<Item = impl Into<String>>,
    ) {
        self.provider_features
            .insert(provider, features.into_iter().map(Into::into).collect());
    }

    pub fn set_device_features(
        &mut self,
        device: DeviceBinding,
        features: impl IntoIterator<Item = impl Into<String>>,
    ) {
        self.device_features
            .insert(device, features.into_iter().map(Into::into).collect());
    }

    pub fn advertisement(&self, kernel: &KernelId) -> Option<&KernelAdvertisement> {
        self.entries
            .get(&kernel.stable_key())
            .map(|entry| &entry.advertisement)
    }

    pub fn active_advertisement(&self, kernel: &KernelId) -> Option<&KernelAdvertisement> {
        self.entries
            .get(&kernel.stable_key())
            .filter(|entry| entry.active)
            .map(|entry| &entry.advertisement)
    }

    pub fn register_provider_advertisement(
        &mut self,
        advertisement: KernelAdvertisement,
    ) -> Result<(), KernelRegistryError> {
        self.register_advertisement(advertisement, KernelRegistrationAuthority::Provider)
    }

    pub fn register_fixture_advertisement(
        &mut self,
        advertisement: KernelAdvertisement,
    ) -> Result<(), KernelRegistryError> {
        self.register_advertisement(
            advertisement,
            KernelRegistrationAuthority::RuntimeTestFixture,
        )
    }

    pub fn register_advertisement(
        &mut self,
        advertisement: KernelAdvertisement,
        authority: KernelRegistrationAuthority,
    ) -> Result<(), KernelRegistryError> {
        self.observations.push(
            KernelObservation::new(KernelObservationKind::KernelAdvertisementReceived)
                .with_kernel(&advertisement.id),
        );
        if !authority.may_register_kernel() {
            self.observations.push(
                KernelObservation::new(KernelObservationKind::KernelAdvertisementRejected)
                    .with_kernel(&advertisement.id)
                    .with_redacted_metadata("error", "kernel-registration-denied"),
            );
            return Err(KernelRegistryError::RegistrationDenied(authority));
        }
        validate_kernel_advertisement(&advertisement)?;
        let key = advertisement.id.stable_key();
        self.entries.insert(
            key,
            KernelRegistryEntry::new(advertisement.clone(), authority),
        );
        self.observations.push(
            KernelObservation::new(KernelObservationKind::KernelAdvertisementAccepted)
                .with_kernel(&advertisement.id),
        );
        self.observations.push(
            KernelObservation::new(KernelObservationKind::KernelRegistryUpdated)
                .with_kernel(&advertisement.id),
        );
        Ok(())
    }

    pub fn invalidate_provider(&mut self, provider: &ProviderBinding, reason: impl Into<String>) {
        let reason = reason.into();
        for entry in self.entries.values_mut() {
            if &entry.advertisement.id.provider == provider {
                entry.active = false;
                entry.invalidation_reason = Some(reason.clone());
                self.observations.push(
                    KernelObservation::new(KernelObservationKind::KernelRegistryInvalidated)
                        .with_kernel(&entry.advertisement.id)
                        .with_redacted_metadata("reason", reason.clone()),
                );
            }
        }
    }

    pub fn invalidate_device(&mut self, device: &DeviceBinding, reason: impl Into<String>) {
        let reason = reason.into();
        for entry in self.entries.values_mut() {
            if entry.advertisement.devices.contains(device) {
                entry.active = false;
                entry.invalidation_reason = Some(reason.clone());
                self.observations.push(
                    KernelObservation::new(KernelObservationKind::KernelRegistryInvalidated)
                        .with_kernel(&entry.advertisement.id)
                        .with_redacted_metadata("reason", reason.clone()),
                );
            }
        }
    }

    pub fn revoke_conformance_profile(&mut self, profile: impl Into<String>) {
        let profile = profile.into();
        self.revoked_conformance_profiles.insert(profile.clone());
        for entry in self.entries.values_mut() {
            if entry.advertisement.id.conformance_profile.as_deref() == Some(profile.as_str()) {
                entry.active = false;
                entry.invalidation_reason = Some("kernel conformance revoked".into());
                self.observations.push(
                    KernelObservation::new(KernelObservationKind::KernelRegistryInvalidated)
                        .with_kernel(&entry.advertisement.id)
                        .with_redacted_metadata("reason", "kernel conformance revoked"),
                );
            }
        }
    }

    pub fn apply_policy_change(&mut self, generation: u64) {
        self.policy_generation = generation;
        for entry in self.entries.values_mut() {
            entry.active = false;
            entry.invalidation_reason = Some("policy changed".into());
            self.observations.push(
                KernelObservation::new(KernelObservationKind::KernelRegistryInvalidated)
                    .with_kernel(&entry.advertisement.id)
                    .with_redacted_metadata("reason", "policy changed"),
            );
        }
    }

    pub fn candidates(&self, request: &KernelSelectionRequest) -> Vec<KernelCandidate> {
        self.observations_for_lookup(request);
        self.entries
            .values()
            .filter(|entry| {
                entry.advertisement.implemented_operator.namespace() == request.operator.namespace()
                    && entry.advertisement.implemented_operator.name() == request.operator.name()
            })
            .map(|entry| self.candidate_for_entry(entry, request))
            .collect()
    }

    pub fn select(
        &self,
        request: &KernelSelectionRequest,
    ) -> Result<KernelSelectionResult, KernelRegistryError> {
        let mut candidates = self.candidates(request);
        if candidates.is_empty() {
            return Err(KernelRegistryError::CandidateNotFound {
                operator: request.operator.clone(),
            });
        }
        candidates.sort_by_key(|candidate| {
            (
                !candidate.compatible,
                candidate.fallback_rank,
                candidate.pressure_score,
                candidate.estimated_cost,
                candidate.kernel.stable_key(),
            )
        });
        let selected = candidates
            .iter()
            .find(|candidate| candidate.compatible)
            .cloned();
        let Some(selected_candidate) = selected.clone() else {
            let reason = candidates
                .first()
                .and_then(|candidate| candidate.rejection_reason.clone())
                .unwrap_or(KernelCandidateRejection::PolicyDenied);
            return Err(KernelRegistryError::CandidateIncompatible { reason });
        };
        let fallback_chain = candidates
            .iter()
            .filter(|candidate| {
                candidate.compatible && candidate.kernel != selected_candidate.kernel
            })
            .cloned()
            .collect::<Vec<_>>();
        let mut observations = vec![
            KernelObservation::new(KernelObservationKind::KernelCandidateLookup)
                .with_redacted_metadata("request_id", request.request_id.clone()),
        ];
        observations.extend(candidates.iter().filter_map(|candidate| {
            candidate.rejection_reason.as_ref().map(|reason| {
                KernelObservation::new(KernelObservationKind::KernelCandidateRejected)
                    .with_kernel(&candidate.kernel)
                    .with_redacted_metadata("reason", format!("{reason:?}"))
            })
        }));
        observations.extend(
            candidates
                .iter()
                .filter(|candidate| candidate.compatible)
                .map(|candidate| {
                    KernelObservation::new(KernelObservationKind::KernelCandidateRanked)
                        .with_kernel(&candidate.kernel)
                        .with_redacted_metadata("rank", candidate.fallback_rank.to_string())
                }),
        );
        observations.push(
            KernelObservation::new(KernelObservationKind::KernelSelected)
                .with_kernel(&selected_candidate.kernel)
                .with_redacted_metadata("request_id", request.request_id.clone()),
        );
        Ok(KernelSelectionResult {
            request_id: request.request_id.clone(),
            selected,
            candidates,
            fallback_chain,
            observations,
        })
    }

    fn observations_for_lookup(&self, _request: &KernelSelectionRequest) {}

    fn candidate_for_entry(
        &self,
        entry: &KernelRegistryEntry,
        request: &KernelSelectionRequest,
    ) -> KernelCandidate {
        let advertisement = &entry.advertisement;
        if !entry.active {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::StaleRegistryEntry,
            );
        }
        if !advertisement
            .id
            .operator_versions
            .contains(request.operator_version)
        {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::OperatorVersionUnsupported,
            );
        }
        if request.browser_target && !advertisement.browser_compatible {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::ProviderUnavailable,
            );
        }
        if let Some(profile) = advertisement.id.conformance_profile.as_ref()
            && self.revoked_conformance_profiles.contains(profile)
        {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::ConformanceFailed,
            );
        }
        if request.require_conformance && advertisement.id.conformance_profile.is_none() {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::ConformanceMissing,
            );
        }
        if let Some(mode) = request.execution_mode
            && !advertisement.execution_modes.contains(&mode)
        {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::ExecutionModeUnsupported,
            );
        }
        if let Some(status) = self.provider_statuses.get(&advertisement.id.provider) {
            if matches!(status.provider_health_compat(), HealthState::Saturated) {
                return KernelCandidate::rejected(
                    advertisement,
                    KernelCandidateRejection::ProviderSaturated,
                );
            }
            if !status.accepts_new_work_by_default() {
                return KernelCandidate::rejected(
                    advertisement,
                    KernelCandidateRejection::ProviderNotReady,
                );
            }
        }
        for device in &advertisement.devices {
            if let Some(status) = self.device_statuses.get(device)
                && !matches!(
                    status.availability,
                    HealthState::Available | HealthState::Degraded | HealthState::Draining
                )
            {
                return KernelCandidate::rejected(
                    advertisement,
                    KernelCandidateRejection::DeviceUnavailable,
                );
            }
        }
        if !advertisement.required_provider_features.is_empty() {
            let features = self
                .provider_features
                .get(&advertisement.id.provider)
                .cloned()
                .unwrap_or_default();
            if !advertisement
                .required_provider_features
                .iter()
                .all(|feature| features.contains(feature))
            {
                return KernelCandidate::rejected(
                    advertisement,
                    KernelCandidateRejection::ProviderFeatureMissing,
                );
            }
        }
        if !advertisement.required_device_features.is_empty() {
            let device_features_match = advertisement.devices.iter().any(|device| {
                let features = self
                    .device_features
                    .get(device)
                    .cloned()
                    .unwrap_or_default();
                advertisement
                    .required_device_features
                    .iter()
                    .all(|feature| features.contains(feature))
            });
            if !device_features_match {
                return KernelCandidate::rejected(
                    advertisement,
                    KernelCandidateRejection::DeviceFeatureMissing,
                );
            }
        }
        let dtype_compatible = request.dtype_requirements.iter().all(|required| {
            advertisement
                .supported_dtypes
                .values()
                .any(|supported| supported.is_empty() || supported.contains(required))
        });
        if !dtype_compatible {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::DTypeUnsupported,
            );
        }
        let layout_compatible = request.layout_requirements.iter().all(|required| {
            advertisement.supported_layouts.is_empty()
                || advertisement.supported_layouts.contains(required)
        });
        if !layout_compatible {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::LayoutUnsupported,
            );
        }
        let memory_compatible = request.memory_class_requirements.iter().all(|required| {
            advertisement.memory_classes.is_empty()
                || advertisement.memory_classes.contains(required)
        });
        if !memory_compatible {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::MemoryClassUnsupported,
            );
        }
        if !shape_compatible(advertisement, request) {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::ShapeUnsupported,
            );
        }
        if advertisement.workspace.required
            && advertisement.workspace.size_bytes_upper_bound == Some(0)
        {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::WorkspaceUnavailable,
            );
        }
        if !batching_compatible(advertisement, request) {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::BatchingUnsupported,
            );
        }
        if !adapter_compatible(advertisement, request) {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::AdapterUnsupported,
            );
        }
        if !kv_cache_compatible(advertisement, request) {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::KvCacheUnsupported,
            );
        }
        if !prefix_cache_compatible(advertisement, request) {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::PrefixCacheUnsupported,
            );
        }
        if request.deterministic_required && !advertisement.determinism.deterministic {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::PolicyDenied,
            );
        }
        if request.precision == ComputePrecision::Exact && advertisement.precision.approximate_math
        {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::PolicyDenied,
            );
        }
        for input in &request.inputs {
            if let Err(_error) =
                validate_affinity_compatibility(&request.affinity, &input.resource.affinity)
            {
                return KernelCandidate::rejected(
                    advertisement,
                    KernelCandidateRejection::ResourceAffinityConflict,
                );
            }
        }
        for output in &request.outputs {
            if let Err(_error) =
                validate_affinity_compatibility(&request.affinity, &output.resource.affinity)
            {
                return KernelCandidate::rejected(
                    advertisement,
                    KernelCandidateRejection::ResourceAffinityConflict,
                );
            }
        }
        let provider_status = self
            .provider_statuses
            .get(&advertisement.id.provider)
            .cloned();
        let device = advertisement.devices.iter().next().cloned();
        let device_status = device
            .as_ref()
            .and_then(|device| self.device_statuses.get(device))
            .cloned();
        let provider_pressure = provider_status
            .as_ref()
            .map(|status| pressure_score(status.pressure))
            .unwrap_or(0);
        let device_pressure = device_status
            .as_ref()
            .map(|status| pressure_score(status.pressure))
            .unwrap_or(0);
        KernelCandidate {
            kernel: advertisement.id.clone(),
            provider: advertisement.id.provider.clone(),
            device,
            operator: advertisement.implemented_operator.clone(),
            compatible: true,
            dtype_compatible: true,
            layout_compatible: true,
            shape_compatible: true,
            memory_compatible: true,
            workspace_feasible: true,
            affinity_compatible: true,
            deterministic_compatible: true,
            precision_compatible: true,
            provider_ready: provider_status
                .as_ref()
                .map(ProviderStatusSnapshot::accepts_new_work_by_default)
                .unwrap_or(true),
            device_ready: device_status
                .as_ref()
                .map(|status| {
                    matches!(
                        status.availability,
                        HealthState::Available | HealthState::Degraded | HealthState::Draining
                    )
                })
                .unwrap_or(true),
            provider_status,
            device_status,
            pressure_score: provider_pressure
                + device_pressure
                + advertisement
                    .performance_hints
                    .get("pressure")
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(0),
            conformance_status: advertisement.id.conformance_profile.clone(),
            estimated_cost: advertisement
                .performance_hints
                .get("estimated-cost")
                .or_else(|| advertisement.performance_hints.get("expected-latency"))
                .or_else(|| {
                    advertisement
                        .performance_hints
                        .get("expected-throughput-cost")
                })
                .and_then(|value| value.parse().ok())
                .unwrap_or(0),
            fallback_rank: advertisement
                .performance_hints
                .get("fallback-rank")
                .and_then(|value| value.parse().ok())
                .unwrap_or(self.policy_generation as u32),
            rejection_reason: None,
        }
    }
}

pub fn validate_kernel_advertisement(
    advertisement: &KernelAdvertisement,
) -> Result<(), KernelRegistryError> {
    if advertisement.id.provider.as_str().trim().is_empty() {
        return Err(KernelRegistryError::AdvertisementInvalid(
            "Provider identity must not be empty".into(),
        ));
    }
    if advertisement.id.name.trim().is_empty() {
        return Err(KernelRegistryError::AdvertisementInvalid(
            "Kernel identity must not be empty".into(),
        ));
    }
    if advertisement.implemented_operator != advertisement.id.operator {
        return Err(KernelRegistryError::AdvertisementInvalid(
            "implemented Operator must match Kernel identity".into(),
        ));
    }
    if !advertisement
        .id
        .operator_versions
        .contains(advertisement.implemented_operator.version())
    {
        return Err(KernelRegistryError::AdvertisementInvalid(
            "Kernel Operator version range must include implemented Operator".into(),
        ));
    }
    if advertisement.execution_modes.is_empty() {
        return Err(KernelRegistryError::AdvertisementInvalid(
            "execution mode metadata must not be empty".into(),
        ));
    }
    if advertisement
        .required_provider_features
        .iter()
        .any(|feature| feature.trim().is_empty())
    {
        return Err(KernelRegistryError::AdvertisementInvalid(
            "required Provider features must not be empty".into(),
        ));
    }
    if advertisement
        .required_device_features
        .iter()
        .any(|feature| feature.trim().is_empty())
    {
        return Err(KernelRegistryError::AdvertisementInvalid(
            "required Device features must not be empty".into(),
        ));
    }
    if advertisement.browser_compatible
        && advertisement
            .execution_modes
            .contains(&KernelExecutionMode::GraphCaptured)
    {
        return Err(KernelRegistryError::AdvertisementInvalid(
            "browser-compatible Kernel cannot require graph-captured native execution".into(),
        ));
    }
    Ok(())
}

fn pressure_score(pressure: ProviderPressureLevel) -> u32 {
    match pressure {
        ProviderPressureLevel::Unknown => 10,
        ProviderPressureLevel::Low => 0,
        ProviderPressureLevel::Moderate => 25,
        ProviderPressureLevel::High => 75,
        ProviderPressureLevel::Saturated => 100,
    }
}

fn shape_compatible(advertisement: &KernelAdvertisement, request: &KernelSelectionRequest) -> bool {
    request
        .inputs
        .iter()
        .chain(&request.outputs)
        .all(|resource| {
            let shape = &resource.resource.descriptor.shape;
            if let Some(rank) = advertisement.shape.rank
                && shape.rank() != rank
            {
                return false;
            }
            for (index, expected) in &advertisement.shape.static_dimensions {
                if shape.dimensions.get(*index) != Some(expected) {
                    return false;
                }
            }
            if let Some(alignment) = advertisement.shape.alignment
                && shape
                    .dimensions
                    .iter()
                    .any(|dimension| dimension % alignment != 0)
            {
                return false;
            }
            if let Some(max) = advertisement.shape.max_total_elements {
                return shape.element_count().is_ok_and(|count| count <= max);
            }
            true
        })
}

fn batching_compatible(
    advertisement: &KernelAdvertisement,
    request: &KernelSelectionRequest,
) -> bool {
    let Some(required) = request.batching.as_ref() else {
        return true;
    };
    let Some(supported) = advertisement.batching.as_ref() else {
        return false;
    };
    (!required.supports_ragged_batches || supported.supports_ragged_batches)
        && (!required.per_operation_output_mapping || supported.per_operation_output_mapping)
        && required
            .max_batch_size
            .zip(supported.max_batch_size)
            .is_none_or(|(required, supported)| required <= supported)
        && required
            .max_active_sequences
            .zip(supported.max_active_sequences)
            .is_none_or(|(required, supported)| required <= supported)
        && required
            .max_total_tokens
            .zip(supported.max_total_tokens)
            .is_none_or(|(required, supported)| required <= supported)
}

fn adapter_compatible(
    advertisement: &KernelAdvertisement,
    request: &KernelSelectionRequest,
) -> bool {
    if request.adapter_methods.is_empty() {
        return true;
    }
    let Some(adapter) = advertisement.adapter.as_ref() else {
        return false;
    };
    request
        .adapter_methods
        .iter()
        .all(|method| adapter.methods.contains(method))
}

fn kv_cache_compatible(
    advertisement: &KernelAdvertisement,
    request: &KernelSelectionRequest,
) -> bool {
    let Some(required) = request.kv_cache.as_ref() else {
        return true;
    };
    let Some(supported) = advertisement.kv_cache.as_ref() else {
        return false;
    };
    (!required.paged_cache || supported.paged_cache)
        && (!required.append || supported.append)
        && (!required.read || supported.read)
        && required
            .layouts
            .iter()
            .all(|layout| supported.layouts.contains(layout))
        && required
            .dtypes
            .iter()
            .all(|dtype| supported.dtypes.contains(dtype))
        && required
            .memory_classes
            .iter()
            .all(|class| supported.memory_classes.contains(class))
}

fn prefix_cache_compatible(
    advertisement: &KernelAdvertisement,
    request: &KernelSelectionRequest,
) -> bool {
    let Some(required) = request.prefix_cache.as_ref() else {
        return true;
    };
    let Some(supported) = advertisement.prefix_cache.as_ref() else {
        return false;
    };
    (!required.supports_adjusted_sequence_length || supported.supports_adjusted_sequence_length)
        && (!required.supports_adjusted_context_length
            || supported.supports_adjusted_context_length)
        && (!required.supports_reused_prefix_boundary || supported.supports_reused_prefix_boundary)
}

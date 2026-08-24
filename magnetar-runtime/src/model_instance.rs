//! Runtime-owned Model Instance lifecycle contracts.
//!
//! A Model Instance is the Runtime-owned loaded inference context created from
//! successful model loading. It is distinct from the immutable Model Artifact,
//! from residency records, from sessions, from Provider resources, and from KV
//! cache state. The public contract exposes stable metadata only; raw model
//! weights, native handles, Device handles, Provider handles, memory pointers,
//! prompts, and cache contents stay behind Runtime-owned boundaries.

use crate::{
    AdapterSetId, CorrelationId, DeviceBinding, GenerationModelReference, InferenceSessionId,
    KvCacheId, LoadedModelContext, MemoryAllocationId, MemoryPressureLevel,
    ModelArchitectureImplementation, ModelArtifactId, ModelDType, ModelResidencyId,
    PrefixCacheEntryId, ProviderAdmissionDecision, ProviderBinding, ProviderHealthState,
    ProviderPressureLevel, ProviderReadinessState, ResourceAffinity, TokenizerId,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModelInstanceId(String);

impl ModelInstanceId {
    pub fn new(value: impl Into<String>) -> Result<Self, ModelInstanceError> {
        let value = value.into();
        validate_instance_identity(&value)?;
        Ok(Self(value))
    }

    pub(crate) fn runtime_issued(sequence: u64) -> Self {
        Self(format!("model-instance-{sequence}"))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ModelInstanceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ModelInstanceLifecycleState {
    Creating,
    Loading,
    Warming,
    Ready,
    Active,
    Idle,
    Draining,
    Suspended,
    Reloading,
    Unloading,
    Unloaded,
    Failed,
    Invalid,
    Removed,
}

impl ModelInstanceLifecycleState {
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Unloaded | Self::Failed | Self::Invalid | Self::Removed
        )
    }

    pub const fn allows_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Creating, Self::Loading)
                | (Self::Creating, Self::Failed)
                | (Self::Creating, Self::Invalid)
                | (Self::Loading, Self::Warming)
                | (Self::Loading, Self::Ready)
                | (Self::Loading, Self::Failed)
                | (Self::Warming, Self::Ready)
                | (Self::Warming, Self::Failed)
                | (Self::Ready, Self::Active)
                | (Self::Ready, Self::Idle)
                | (Self::Ready, Self::Draining)
                | (Self::Ready, Self::Suspended)
                | (Self::Ready, Self::Reloading)
                | (Self::Ready, Self::Unloading)
                | (Self::Ready, Self::Failed)
                | (Self::Ready, Self::Invalid)
                | (Self::Active, Self::Idle)
                | (Self::Active, Self::Draining)
                | (Self::Active, Self::Failed)
                | (Self::Idle, Self::Active)
                | (Self::Idle, Self::Draining)
                | (Self::Idle, Self::Suspended)
                | (Self::Idle, Self::Reloading)
                | (Self::Idle, Self::Unloading)
                | (Self::Idle, Self::Failed)
                | (Self::Idle, Self::Invalid)
                | (Self::Draining, Self::Unloading)
                | (Self::Draining, Self::Reloading)
                | (Self::Draining, Self::Failed)
                | (Self::Suspended, Self::Loading)
                | (Self::Suspended, Self::Reloading)
                | (Self::Suspended, Self::Unloading)
                | (Self::Suspended, Self::Failed)
                | (Self::Reloading, Self::Loading)
                | (Self::Reloading, Self::Ready)
                | (Self::Reloading, Self::Failed)
                | (Self::Unloading, Self::Unloaded)
                | (Self::Unloading, Self::Failed)
                | (Self::Unloaded, Self::Removed)
                | (Self::Failed, Self::Unloading)
                | (Self::Failed, Self::Invalid)
                | (Self::Invalid, Self::Unloading)
                | (Self::Invalid, Self::Removed)
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ModelInstanceReadiness {
    NotReady,
    Ready,
    ReadOnly,
    Draining,
    Suspended,
    Failed,
}

impl ModelInstanceReadiness {
    pub const fn accepts_generation(self) -> bool {
        matches!(self, Self::Ready)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelInstanceWarmupPolicy {
    Disabled,
    ValidateMetadataOnly,
    ProviderInitialization,
    Full,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelInstanceUnloadPolicy {
    DrainActiveUse,
    RejectActiveUse,
    ForceInvalidate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelInstanceSharingPolicy {
    Private,
    RuntimeLocal,
    TenantIsolated,
    PolicyControlled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelInstanceMutationKind {
    AdapterMerge,
    ProviderPreparation,
    QuantizationTransform,
    ResidencyRelocation,
    Reload,
    Warmup,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelInstanceSuspensionReason {
    MemoryPressure,
    ProviderPressure,
    DevicePressure,
    AdministrativePolicy,
    BrowserLifecycle,
    TemporaryResourceLoss,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelInstancePolicy {
    pub warmup: ModelInstanceWarmupPolicy,
    pub unload: ModelInstanceUnloadPolicy,
    pub sharing: ModelInstanceSharingPolicy,
    pub implicit_loading_allowed: bool,
    pub suspension_allowed: bool,
    pub raw_handle_exposure_allowed: bool,
    pub tenant_isolation_required: bool,
    pub browser_linear_memory_limit_bytes: Option<u64>,
}

impl Default for ModelInstancePolicy {
    fn default() -> Self {
        Self {
            warmup: ModelInstanceWarmupPolicy::ValidateMetadataOnly,
            unload: ModelInstanceUnloadPolicy::DrainActiveUse,
            sharing: ModelInstanceSharingPolicy::RuntimeLocal,
            implicit_loading_allowed: false,
            suspension_allowed: true,
            raw_handle_exposure_allowed: false,
            tenant_isolation_required: false,
            browser_linear_memory_limit_bytes: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelInstanceCreationChecks {
    pub artifact_identity_valid: bool,
    pub artifact_trusted: bool,
    pub architecture_available: bool,
    pub residency_plan_valid: bool,
    pub memory_admitted: bool,
    pub provider_device_compatible: bool,
    pub tokenizer_compatible: bool,
    pub runtime_policy_allows: bool,
    pub browser_native_supported: bool,
}

impl Default for ModelInstanceCreationChecks {
    fn default() -> Self {
        Self {
            artifact_identity_valid: true,
            artifact_trusted: true,
            architecture_available: true,
            residency_plan_valid: true,
            memory_admitted: true,
            provider_device_compatible: true,
            tokenizer_compatible: true,
            runtime_policy_allows: true,
            browser_native_supported: true,
        }
    }
}

impl ModelInstanceCreationChecks {
    pub fn validate(&self) -> Result<(), ModelInstanceError> {
        if !self.artifact_identity_valid {
            return Err(ModelInstanceError::ModelInstancePolicyDenied);
        }
        if !self.artifact_trusted {
            return Err(ModelInstanceError::ModelInstancePolicyDenied);
        }
        if !self.architecture_available {
            return Err(ModelInstanceError::ModelInstanceNotReady);
        }
        if !self.residency_plan_valid {
            return Err(ModelInstanceError::ModelInstanceResidencyMissing);
        }
        if !self.memory_admitted {
            return Err(ModelInstanceError::ModelInstanceMemoryPressure);
        }
        if !self.provider_device_compatible {
            return Err(ModelInstanceError::ModelInstanceProviderNotReady);
        }
        if !self.tokenizer_compatible {
            return Err(ModelInstanceError::ModelInstancePolicyDenied);
        }
        if !self.runtime_policy_allows {
            return Err(ModelInstanceError::ModelInstancePolicyDenied);
        }
        if !self.browser_native_supported {
            return Err(ModelInstanceError::ModelInstanceBrowserFeatureUnsupported);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelInstanceReadinessChecks {
    pub residency_available: bool,
    pub provider_ready: bool,
    pub device_ready: bool,
    pub adapter_ready: bool,
    pub memory_pressure: MemoryPressureLevel,
    pub runtime_policy_allows: bool,
    pub browser_supported: bool,
}

impl Default for ModelInstanceReadinessChecks {
    fn default() -> Self {
        Self {
            residency_available: true,
            provider_ready: true,
            device_ready: true,
            adapter_ready: true,
            memory_pressure: MemoryPressureLevel::Low,
            runtime_policy_allows: true,
            browser_supported: true,
        }
    }
}

impl ModelInstanceReadinessChecks {
    pub fn readiness(&self) -> ModelInstanceReadiness {
        if !self.provider_ready
            || !self.device_ready
            || !self.adapter_ready
            || !self.runtime_policy_allows
        {
            return ModelInstanceReadiness::Failed;
        }
        if !self.residency_available
            || matches!(
                self.memory_pressure,
                MemoryPressureLevel::High | MemoryPressureLevel::Saturated
            )
        {
            return ModelInstanceReadiness::Suspended;
        }
        if !self.browser_supported {
            return ModelInstanceReadiness::Failed;
        }
        ModelInstanceReadiness::Ready
    }

    pub fn validate(&self) -> Result<(), ModelInstanceError> {
        if !self.residency_available {
            return Err(ModelInstanceError::ModelInstanceResidencyMissing);
        }
        if !self.provider_ready {
            return Err(ModelInstanceError::ModelInstanceProviderNotReady);
        }
        if !self.device_ready {
            return Err(ModelInstanceError::ModelInstanceDeviceUnavailable);
        }
        if !self.adapter_ready {
            return Err(ModelInstanceError::ModelInstanceAdapterIncompatible);
        }
        if matches!(
            self.memory_pressure,
            MemoryPressureLevel::High | MemoryPressureLevel::Saturated
        ) {
            return Err(ModelInstanceError::ModelInstanceMemoryPressure);
        }
        if !self.runtime_policy_allows {
            return Err(ModelInstanceError::ModelInstancePolicyDenied);
        }
        if !self.browser_supported {
            return Err(ModelInstanceError::ModelInstanceBrowserFeatureUnsupported);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelInstanceWarmupStep {
    ProviderInitialization,
    KernelPreparationPlaceholder,
    OperatorGraphPreparationPlaceholder,
    ShapePlanPreparationPlaceholder,
    TokenizerModelMetadataValidation,
    SmallTestExecutionPlaceholder,
    MemoryResidencyVerification,
    AdapterReadinessVerification,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelInstanceWarmupPlan {
    pub policy: ModelInstanceWarmupPolicy,
    pub steps: Vec<ModelInstanceWarmupStep>,
}

impl ModelInstanceWarmupPlan {
    pub fn for_policy(policy: ModelInstanceWarmupPolicy) -> Self {
        let steps = match policy {
            ModelInstanceWarmupPolicy::Disabled => Vec::new(),
            ModelInstanceWarmupPolicy::ValidateMetadataOnly => {
                vec![ModelInstanceWarmupStep::TokenizerModelMetadataValidation]
            }
            ModelInstanceWarmupPolicy::ProviderInitialization => vec![
                ModelInstanceWarmupStep::ProviderInitialization,
                ModelInstanceWarmupStep::TokenizerModelMetadataValidation,
                ModelInstanceWarmupStep::MemoryResidencyVerification,
            ],
            ModelInstanceWarmupPolicy::Full => vec![
                ModelInstanceWarmupStep::ProviderInitialization,
                ModelInstanceWarmupStep::KernelPreparationPlaceholder,
                ModelInstanceWarmupStep::OperatorGraphPreparationPlaceholder,
                ModelInstanceWarmupStep::ShapePlanPreparationPlaceholder,
                ModelInstanceWarmupStep::TokenizerModelMetadataValidation,
                ModelInstanceWarmupStep::SmallTestExecutionPlaceholder,
                ModelInstanceWarmupStep::MemoryResidencyVerification,
                ModelInstanceWarmupStep::AdapterReadinessVerification,
            ],
        };
        Self { policy, steps }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelInstanceUsage {
    pub active_operation_count: usize,
    pub active_session_count: usize,
    pub queued_operation_count: usize,
    pub total_request_count: u64,
    pub input_token_count: u64,
    pub output_token_count: u64,
    pub last_used_millis: Option<u64>,
    pub residency_bytes: u64,
    pub kv_cache_dependencies: BTreeSet<KvCacheId>,
    pub prefix_cache_dependencies: BTreeSet<PrefixCacheEntryId>,
    pub adapter_dependencies: BTreeSet<AdapterSetId>,
    pub failure_count: u64,
}

impl ModelInstanceUsage {
    pub const fn has_active_use(&self) -> bool {
        self.active_operation_count > 0 || self.active_session_count > 0
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelInstanceAdapterState {
    pub active_adapter_set: Option<AdapterSetId>,
    pub activation_scope: Option<String>,
    pub merged: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderModelResource {
    pub provider: ProviderBinding,
    pub handle_kind: String,
    pub release_required: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelInstanceResourceBindings {
    pub memory_allocations: BTreeSet<MemoryAllocationId>,
    pub released_memory_allocations: BTreeSet<MemoryAllocationId>,
    pub released_provider_resources: BTreeSet<ProviderBinding>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelInstancePlacement {
    pub provider: Option<ProviderBinding>,
    pub device: Option<DeviceBinding>,
    pub affinity: ResourceAffinity,
    pub provider_resource: Option<ProviderModelResource>,
}

impl ModelInstancePlacement {
    pub fn new(affinity: ResourceAffinity) -> Self {
        Self {
            provider: affinity.provider().cloned(),
            device: affinity.device().cloned(),
            affinity,
            provider_resource: None,
        }
    }

    pub const fn exposes_raw_handles(&self) -> bool {
        false
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelInstanceDefinition {
    pub artifact: ModelArtifactId,
    pub architecture: ModelArchitectureImplementation,
    pub residencies: BTreeSet<ModelResidencyId>,
    pub tokenizer: Option<TokenizerId>,
    pub placement: ModelInstancePlacement,
    pub policy: ModelInstancePolicy,
    pub adapter_state: ModelInstanceAdapterState,
    pub associated_sessions: BTreeSet<InferenceSessionId>,
    pub usage: ModelInstanceUsage,
    pub compute_dtype: Option<ModelDType>,
    pub mutation_version: u64,
    pub tenant: Option<String>,
    pub owner: Option<String>,
    pub resource_bindings: ModelInstanceResourceBindings,
}

impl ModelInstanceDefinition {
    pub fn from_loaded_context(
        context: &LoadedModelContext,
        architecture: ModelArchitectureImplementation,
        affinity: ResourceAffinity,
    ) -> Self {
        Self {
            artifact: context.artifact.clone(),
            architecture,
            residencies: BTreeSet::from([context.residency]),
            tokenizer: None,
            placement: ModelInstancePlacement::new(affinity),
            policy: ModelInstancePolicy::default(),
            adapter_state: ModelInstanceAdapterState::default(),
            associated_sessions: BTreeSet::new(),
            usage: ModelInstanceUsage {
                residency_bytes: context.plan.expected_resident_bytes,
                ..ModelInstanceUsage::default()
            },
            compute_dtype: context.plan.target_compute_dtype,
            mutation_version: 0,
            tenant: None,
            owner: None,
            resource_bindings: ModelInstanceResourceBindings::default(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelInstanceSharingContext {
    pub owner: Option<String>,
    pub tenant: Option<String>,
    pub adapter_state: ModelInstanceAdapterState,
    pub kv_cache_private: bool,
    pub prefix_cache_private: bool,
    pub affinity: ResourceAffinity,
}

impl ModelInstanceSharingContext {
    pub fn from_definition(definition: &ModelInstanceDefinition) -> Self {
        Self {
            owner: definition.owner.clone(),
            tenant: definition.tenant.clone(),
            adapter_state: definition.adapter_state.clone(),
            kv_cache_private: false,
            prefix_cache_private: false,
            affinity: definition.placement.affinity.clone(),
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelInstanceInvalidationReport {
    pub kv_caches: BTreeSet<KvCacheId>,
    pub prefix_entries: BTreeSet<PrefixCacheEntryId>,
    pub adapters_released: BTreeSet<AdapterSetId>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelInstanceUnloadReport {
    pub invalidated: ModelInstanceInvalidationReport,
    pub released_memory_allocations: BTreeSet<MemoryAllocationId>,
    pub released_provider_resources: BTreeSet<ProviderBinding>,
    pub dangling_session_references: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelInstanceReloadRequest {
    pub replacement: ModelInstanceDefinition,
    pub migrate_sessions: bool,
    pub allow_active_semantic_mutation: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelInstance {
    pub id: ModelInstanceId,
    pub lifecycle: ModelInstanceLifecycleState,
    pub readiness: ModelInstanceReadiness,
    pub definition: ModelInstanceDefinition,
    pub last_error: Option<ModelInstanceError>,
}

impl ModelInstance {
    pub fn new(id: ModelInstanceId, definition: ModelInstanceDefinition) -> Self {
        Self {
            id,
            lifecycle: ModelInstanceLifecycleState::Creating,
            readiness: ModelInstanceReadiness::NotReady,
            definition,
            last_error: None,
        }
    }

    pub fn transition_to(
        &mut self,
        next: ModelInstanceLifecycleState,
    ) -> Result<(), ModelInstanceError> {
        if !self.lifecycle.allows_transition_to(next) {
            return Err(ModelInstanceError::InvalidLifecycleTransition {
                from: self.lifecycle,
                to: next,
            });
        }
        self.lifecycle = next;
        self.readiness = readiness_for_lifecycle(next);
        Ok(())
    }

    pub fn mark_ready(&mut self) -> Result<(), ModelInstanceError> {
        match self.lifecycle {
            ModelInstanceLifecycleState::Creating => {
                self.transition_to(ModelInstanceLifecycleState::Loading)?;
                self.transition_to(ModelInstanceLifecycleState::Ready)
            }
            ModelInstanceLifecycleState::Loading | ModelInstanceLifecycleState::Warming => {
                self.transition_to(ModelInstanceLifecycleState::Ready)
            }
            _ if self.readiness.accepts_generation() => Ok(()),
            _ => Err(ModelInstanceError::ModelInstanceNotReady),
        }
    }

    pub fn validate_creation(
        &self,
        checks: &ModelInstanceCreationChecks,
    ) -> Result<(), ModelInstanceError> {
        checks.validate()
    }

    pub fn validate_readiness(
        &mut self,
        checks: &ModelInstanceReadinessChecks,
    ) -> Result<(), ModelInstanceError> {
        let result = checks.validate();
        self.readiness = checks.readiness();
        if result.is_err() {
            self.last_error = result.clone().err();
        }
        result
    }

    pub fn warmup(
        &mut self,
        plan: &ModelInstanceWarmupPlan,
        checks: &ModelInstanceReadinessChecks,
    ) -> Result<(), ModelInstanceError> {
        if plan.policy == ModelInstanceWarmupPolicy::Disabled {
            return self.validate_readiness(checks);
        }
        if matches!(
            self.lifecycle,
            ModelInstanceLifecycleState::Creating | ModelInstanceLifecycleState::Loading
        ) {
            if self.lifecycle == ModelInstanceLifecycleState::Creating {
                self.transition_to(ModelInstanceLifecycleState::Loading)?;
            }
            self.transition_to(ModelInstanceLifecycleState::Warming)?;
        }
        match self.validate_readiness(checks) {
            Ok(()) => self.transition_to(ModelInstanceLifecycleState::Ready),
            Err(error) => {
                self.lifecycle = ModelInstanceLifecycleState::Failed;
                self.readiness = ModelInstanceReadiness::Failed;
                self.last_error = Some(error.clone());
                Err(match error {
                    ModelInstanceError::ModelInstanceProviderNotReady => {
                        ModelInstanceError::ModelInstanceProviderNotReady
                    }
                    ModelInstanceError::ModelInstanceAdapterIncompatible => {
                        ModelInstanceError::ModelInstanceAdapterIncompatible
                    }
                    _ => ModelInstanceError::ModelInstanceWarmupFailed,
                })
            }
        }
    }

    pub fn acquire_usage(&mut self, now_millis: u64) -> Result<(), ModelInstanceError> {
        if !self.readiness.accepts_generation() {
            return Err(readiness_error(self.lifecycle, self.readiness));
        }
        self.definition.usage.active_operation_count = self
            .definition
            .usage
            .active_operation_count
            .saturating_add(1);
        self.definition.usage.total_request_count =
            self.definition.usage.total_request_count.saturating_add(1);
        self.definition.usage.last_used_millis = Some(now_millis);
        if self.lifecycle == ModelInstanceLifecycleState::Ready
            || self.lifecycle == ModelInstanceLifecycleState::Idle
        {
            self.lifecycle = ModelInstanceLifecycleState::Active;
        }
        Ok(())
    }

    pub fn release_usage(&mut self) -> Result<(), ModelInstanceError> {
        if self.definition.usage.active_operation_count == 0 {
            return Err(ModelInstanceError::InternalModelInstance {
                reason: "no active model instance operation to release".into(),
            });
        }
        self.definition.usage.active_operation_count -= 1;
        if self.definition.usage.active_operation_count == 0
            && self.lifecycle == ModelInstanceLifecycleState::Active
        {
            self.lifecycle = ModelInstanceLifecycleState::Idle;
            self.readiness = ModelInstanceReadiness::Ready;
        }
        Ok(())
    }

    pub fn can_unload(&self, policy: ModelInstanceUnloadPolicy) -> bool {
        match policy {
            ModelInstanceUnloadPolicy::ForceInvalidate => true,
            ModelInstanceUnloadPolicy::RejectActiveUse
            | ModelInstanceUnloadPolicy::DrainActiveUse => !self.definition.usage.has_active_use(),
        }
    }

    pub fn record_mutation(&mut self, _kind: ModelInstanceMutationKind) {
        self.definition.mutation_version = self.definition.mutation_version.saturating_add(1);
    }

    pub fn activate_adapters(
        &mut self,
        adapter_set: AdapterSetId,
        scope: impl Into<String>,
        merged: bool,
    ) -> ModelInstanceInvalidationReport {
        self.definition.adapter_state.active_adapter_set = Some(adapter_set.clone());
        self.definition.adapter_state.activation_scope = Some(scope.into());
        self.definition.adapter_state.merged = merged;
        self.definition
            .usage
            .adapter_dependencies
            .insert(adapter_set);
        self.record_mutation(ModelInstanceMutationKind::AdapterMerge);
        self.invalidate_cache_dependencies()
    }

    pub fn invalidate_cache_dependencies(&mut self) -> ModelInstanceInvalidationReport {
        ModelInstanceInvalidationReport {
            kv_caches: std::mem::take(&mut self.definition.usage.kv_cache_dependencies),
            prefix_entries: std::mem::take(&mut self.definition.usage.prefix_cache_dependencies),
            adapters_released: BTreeSet::new(),
        }
    }

    pub fn can_share_with(&self, other: &ModelInstanceSharingContext) -> bool {
        match self.definition.policy.sharing {
            ModelInstanceSharingPolicy::Private => false,
            ModelInstanceSharingPolicy::RuntimeLocal => {
                !other.kv_cache_private
                    && !other.prefix_cache_private
                    && self.definition.adapter_state == other.adapter_state
                    && self
                        .definition
                        .placement
                        .affinity
                        .validate_with(&other.affinity)
                        .is_ok()
            }
            ModelInstanceSharingPolicy::TenantIsolated => {
                self.definition.tenant.is_some()
                    && self.definition.tenant == other.tenant
                    && self.definition.adapter_state == other.adapter_state
            }
            ModelInstanceSharingPolicy::PolicyControlled => {
                self.definition.owner == other.owner
                    && self.definition.adapter_state == other.adapter_state
                    && !other.kv_cache_private
                    && !other.prefix_cache_private
            }
        }
    }

    pub fn suspend(
        &mut self,
        _reason: ModelInstanceSuspensionReason,
    ) -> Result<(), ModelInstanceError> {
        if !self.definition.policy.suspension_allowed {
            return Err(ModelInstanceError::ModelInstancePolicyDenied);
        }
        if self.definition.usage.has_active_use() {
            return Err(ModelInstanceError::ModelInstanceActive);
        }
        if matches!(
            self.lifecycle,
            ModelInstanceLifecycleState::Ready | ModelInstanceLifecycleState::Idle
        ) {
            self.transition_to(ModelInstanceLifecycleState::Suspended)
        } else {
            Err(readiness_error(self.lifecycle, self.readiness))
        }
    }

    pub fn resume(&mut self) -> Result<(), ModelInstanceError> {
        if self.lifecycle == ModelInstanceLifecycleState::Suspended {
            self.transition_to(ModelInstanceLifecycleState::Loading)?;
            self.transition_to(ModelInstanceLifecycleState::Ready)
        } else {
            Err(ModelInstanceError::ModelInstanceNotReady)
        }
    }

    pub fn drain(&mut self) -> Result<(), ModelInstanceError> {
        if matches!(
            self.lifecycle,
            ModelInstanceLifecycleState::Ready
                | ModelInstanceLifecycleState::Idle
                | ModelInstanceLifecycleState::Active
        ) {
            self.transition_to(ModelInstanceLifecycleState::Draining)
        } else {
            Err(readiness_error(self.lifecycle, self.readiness))
        }
    }

    pub fn fail(&mut self, error: ModelInstanceError) {
        self.lifecycle = ModelInstanceLifecycleState::Failed;
        self.readiness = ModelInstanceReadiness::Failed;
        self.definition.usage.failure_count = self.definition.usage.failure_count.saturating_add(1);
        self.last_error = Some(error);
    }

    pub fn invalidate(&mut self, error: ModelInstanceError) {
        self.lifecycle = ModelInstanceLifecycleState::Invalid;
        self.readiness = ModelInstanceReadiness::Failed;
        self.last_error = Some(error);
    }

    pub fn provider_status_changed(
        &mut self,
        health: ProviderHealthState,
        readiness: ProviderReadinessState,
        pressure: ProviderPressureLevel,
        admission: ProviderAdmissionDecision,
    ) -> Result<(), ModelInstanceError> {
        if matches!(
            health,
            ProviderHealthState::Unhealthy | ProviderHealthState::Failed
        ) {
            self.fail(ModelInstanceError::ModelInstanceProviderFailed);
            return Err(ModelInstanceError::ModelInstanceProviderFailed);
        }
        if matches!(
            readiness,
            ProviderReadinessState::NotReady | ProviderReadinessState::Draining
        ) {
            if readiness == ProviderReadinessState::Draining {
                self.drain()?;
                return Err(ModelInstanceError::ModelInstanceDraining);
            }
            self.readiness = ModelInstanceReadiness::NotReady;
            return Err(ModelInstanceError::ModelInstanceProviderNotReady);
        }
        if matches!(
            pressure,
            ProviderPressureLevel::High | ProviderPressureLevel::Saturated
        ) || admission == ProviderAdmissionDecision::Reject
        {
            if self.definition.policy.suspension_allowed && !self.definition.usage.has_active_use()
            {
                self.suspend(ModelInstanceSuspensionReason::ProviderPressure)?;
            }
            return Err(ModelInstanceError::ModelInstanceProviderNotReady);
        }
        Ok(())
    }

    pub fn device_unavailable(&mut self, lost: bool) -> Result<(), ModelInstanceError> {
        if lost {
            self.suspend(ModelInstanceSuspensionReason::TemporaryResourceLoss)?;
            return Err(ModelInstanceError::ModelInstanceDeviceLost);
        }
        self.suspend(ModelInstanceSuspensionReason::DevicePressure)?;
        Err(ModelInstanceError::ModelInstanceDeviceUnavailable)
    }

    pub fn browser_supported(&self) -> Result<(), ModelInstanceError> {
        if self
            .definition
            .policy
            .browser_linear_memory_limit_bytes
            .is_some_and(|limit| self.definition.usage.residency_bytes > limit)
        {
            return Err(ModelInstanceError::ModelInstanceBrowserFeatureUnsupported);
        }
        Ok(())
    }

    pub fn status(&self) -> ModelInstanceStatus {
        ModelInstanceStatus {
            id: self.id.clone(),
            artifact: self.definition.artifact.clone(),
            lifecycle: self.lifecycle,
            readiness: self.readiness,
            active_operation_count: self.definition.usage.active_operation_count,
            active_session_count: self.definition.usage.active_session_count,
            queued_operation_count: self.definition.usage.queued_operation_count,
            total_request_count: self.definition.usage.total_request_count,
            residency_bytes: self.definition.usage.residency_bytes,
            mutation_version: self.definition.mutation_version,
            raw_weights_available: false,
            raw_provider_handle_available: false,
            raw_device_handle_available: false,
            raw_memory_pointer_available: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelInstanceStatus {
    pub id: ModelInstanceId,
    pub artifact: ModelArtifactId,
    pub lifecycle: ModelInstanceLifecycleState,
    pub readiness: ModelInstanceReadiness,
    pub active_operation_count: usize,
    pub active_session_count: usize,
    pub queued_operation_count: usize,
    pub total_request_count: u64,
    pub residency_bytes: u64,
    pub mutation_version: u64,
    pub raw_weights_available: bool,
    pub raw_provider_handle_available: bool,
    pub raw_device_handle_available: bool,
    pub raw_memory_pointer_available: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelInstanceObservationKind {
    CreationRequested,
    Created,
    Loading,
    Warming,
    Ready,
    Active,
    Idle,
    Draining,
    Suspended,
    Reloading,
    Unloading,
    Unloaded,
    Failed,
    Invalidated,
    Removed,
    UsageAcquired,
    UsageReleased,
    SharingDenied,
    CacheInvalidation,
    MemoryPressure,
    ProviderPressure,
    DeviceUnavailable,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelInstanceObservation {
    pub kind: ModelInstanceObservationKind,
    pub instance: Option<ModelInstanceId>,
    pub message: String,
    pub correlation_id: Option<CorrelationId>,
    pub raw_weights_available: bool,
    pub raw_prompt_available: bool,
    pub raw_cache_available: bool,
    pub raw_provider_handle_available: bool,
    pub raw_device_handle_available: bool,
}

impl ModelInstanceObservation {
    pub fn redacted(
        kind: ModelInstanceObservationKind,
        instance: Option<ModelInstanceId>,
        message: impl Into<String>,
        correlation_id: Option<CorrelationId>,
    ) -> Self {
        Self {
            kind,
            instance,
            message: message.into(),
            correlation_id,
            raw_weights_available: false,
            raw_prompt_available: false,
            raw_cache_available: false,
            raw_provider_handle_available: false,
            raw_device_handle_available: false,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelInstanceManager {
    next_id: u64,
    instances: BTreeMap<ModelInstanceId, ModelInstance>,
    observations: Vec<ModelInstanceObservation>,
}

impl ModelInstanceManager {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            instances: BTreeMap::new(),
            observations: Vec::new(),
        }
    }

    pub fn instances(&self) -> impl Iterator<Item = &ModelInstance> {
        self.instances.values()
    }

    pub fn observations(&self) -> &[ModelInstanceObservation] {
        &self.observations
    }

    pub fn instance(&self, id: &ModelInstanceId) -> Result<&ModelInstance, ModelInstanceError> {
        self.instances
            .get(id)
            .ok_or(ModelInstanceError::ModelInstanceNotFound)
    }

    pub fn instance_mut(
        &mut self,
        id: &ModelInstanceId,
    ) -> Result<&mut ModelInstance, ModelInstanceError> {
        self.instances
            .get_mut(id)
            .ok_or(ModelInstanceError::ModelInstanceNotFound)
    }

    pub fn create(
        &mut self,
        definition: ModelInstanceDefinition,
    ) -> Result<ModelInstanceId, ModelInstanceError> {
        definition
            .policy
            .browser_linear_memory_limit_bytes
            .map(|limit| definition.usage.residency_bytes <= limit)
            .unwrap_or(true)
            .then_some(())
            .ok_or(ModelInstanceError::ModelInstanceBrowserFeatureUnsupported)?;
        if definition.residencies.is_empty() {
            return Err(ModelInstanceError::ModelInstanceResidencyMissing);
        }
        let id = ModelInstanceId::runtime_issued(self.next_id);
        self.next_id = self.next_id.saturating_add(1);
        self.observe(
            ModelInstanceObservationKind::CreationRequested,
            Some(id.clone()),
            "model instance creation requested",
            None,
        );
        let mut instance = ModelInstance::new(id.clone(), definition);
        instance.transition_to(ModelInstanceLifecycleState::Loading)?;
        instance.mark_ready()?;
        self.instances.insert(id.clone(), instance);
        self.observe(
            ModelInstanceObservationKind::Created,
            Some(id.clone()),
            "model instance created",
            None,
        );
        self.observe(
            ModelInstanceObservationKind::Ready,
            Some(id.clone()),
            "model instance ready",
            None,
        );
        Ok(id)
    }

    pub fn create_checked(
        &mut self,
        definition: ModelInstanceDefinition,
        checks: &ModelInstanceCreationChecks,
    ) -> Result<ModelInstanceId, ModelInstanceError> {
        checks.validate()?;
        self.create(definition)
    }

    pub fn warmup(
        &mut self,
        id: &ModelInstanceId,
        plan: &ModelInstanceWarmupPlan,
        checks: &ModelInstanceReadinessChecks,
    ) -> Result<(), ModelInstanceError> {
        let result = self.instance_mut(id)?.warmup(plan, checks);
        self.observe(
            if result.is_ok() {
                ModelInstanceObservationKind::Ready
            } else {
                ModelInstanceObservationKind::Failed
            },
            Some(id.clone()),
            if result.is_ok() {
                "model instance warmup completed"
            } else {
                "model instance warmup failed"
            },
            None,
        );
        result
    }

    pub fn acquire_usage(
        &mut self,
        id: &ModelInstanceId,
        now_millis: u64,
    ) -> Result<(), ModelInstanceError> {
        self.instance_mut(id)?.acquire_usage(now_millis)?;
        self.observe(
            ModelInstanceObservationKind::UsageAcquired,
            Some(id.clone()),
            "model instance usage acquired",
            None,
        );
        Ok(())
    }

    pub fn release_usage(&mut self, id: &ModelInstanceId) -> Result<(), ModelInstanceError> {
        self.instance_mut(id)?.release_usage()?;
        self.observe(
            ModelInstanceObservationKind::UsageReleased,
            Some(id.clone()),
            "model instance usage released",
            None,
        );
        Ok(())
    }

    pub fn generation_reference(
        &self,
        id: &ModelInstanceId,
    ) -> Result<GenerationModelReference, ModelInstanceError> {
        let instance = self.instance(id)?;
        if !instance.readiness.accepts_generation() {
            return Err(readiness_error(instance.lifecycle, instance.readiness));
        }
        Ok(GenerationModelReference::ModelInstance(id.clone()))
    }

    pub fn unload(
        &mut self,
        id: &ModelInstanceId,
        policy: ModelInstanceUnloadPolicy,
    ) -> Result<ModelInstanceUnloadReport, ModelInstanceError> {
        if !self.instance(id)?.can_unload(policy) {
            return Err(ModelInstanceError::ModelInstanceActive);
        }
        let report = self.prepare_unload_report(id)?;
        let instance = self.instance_mut(id)?;
        if matches!(
            instance.lifecycle,
            ModelInstanceLifecycleState::Ready
                | ModelInstanceLifecycleState::Idle
                | ModelInstanceLifecycleState::Suspended
                | ModelInstanceLifecycleState::Failed
                | ModelInstanceLifecycleState::Invalid
        ) {
            if matches!(
                instance.lifecycle,
                ModelInstanceLifecycleState::Ready | ModelInstanceLifecycleState::Idle
            ) {
                instance.transition_to(ModelInstanceLifecycleState::Draining)?;
            }
            instance.transition_to(ModelInstanceLifecycleState::Unloading)?;
            instance.transition_to(ModelInstanceLifecycleState::Unloaded)?;
            self.observe(
                ModelInstanceObservationKind::Unloaded,
                Some(id.clone()),
                "model instance unloaded",
                None,
            );
            Ok(report)
        } else {
            Err(readiness_error(instance.lifecycle, instance.readiness))
        }
    }

    pub fn reload(
        &mut self,
        id: &ModelInstanceId,
        request: ModelInstanceReloadRequest,
    ) -> Result<ModelInstanceId, ModelInstanceError> {
        if !request.allow_active_semantic_mutation
            && self.instance(id)?.definition.usage.has_active_use()
        {
            return Err(ModelInstanceError::ModelInstanceActive);
        }
        self.unload(id, ModelInstanceUnloadPolicy::DrainActiveUse)?;
        let replacement = self.create(request.replacement)?;
        self.observe(
            ModelInstanceObservationKind::Reloading,
            Some(id.clone()),
            "model instance reload created replacement",
            None,
        );
        Ok(replacement)
    }

    pub fn activate_adapters(
        &mut self,
        id: &ModelInstanceId,
        adapter_set: AdapterSetId,
        scope: impl Into<String>,
        merged: bool,
    ) -> Result<ModelInstanceInvalidationReport, ModelInstanceError> {
        let report = self
            .instance_mut(id)?
            .activate_adapters(adapter_set, scope, merged);
        self.observe(
            ModelInstanceObservationKind::CacheInvalidation,
            Some(id.clone()),
            "model instance adapter change invalidated dependent caches",
            None,
        );
        Ok(report)
    }

    pub fn invalidate_for_mutation(
        &mut self,
        id: &ModelInstanceId,
        kind: ModelInstanceMutationKind,
    ) -> Result<ModelInstanceInvalidationReport, ModelInstanceError> {
        let instance = self.instance_mut(id)?;
        instance.record_mutation(kind);
        let report = instance.invalidate_cache_dependencies();
        self.observe(
            ModelInstanceObservationKind::CacheInvalidation,
            Some(id.clone()),
            "model instance semantic mutation invalidated dependent caches",
            None,
        );
        Ok(report)
    }

    pub fn fail_instance(
        &mut self,
        id: &ModelInstanceId,
        error: ModelInstanceError,
    ) -> Result<(), ModelInstanceError> {
        self.instance_mut(id)?.fail(error);
        self.observe(
            ModelInstanceObservationKind::Failed,
            Some(id.clone()),
            "model instance failed",
            None,
        );
        Ok(())
    }

    pub fn invalidate_instance(
        &mut self,
        id: &ModelInstanceId,
        error: ModelInstanceError,
    ) -> Result<(), ModelInstanceError> {
        self.instance_mut(id)?.invalidate(error);
        self.observe(
            ModelInstanceObservationKind::Invalidated,
            Some(id.clone()),
            "model instance invalidated",
            None,
        );
        Ok(())
    }

    fn prepare_unload_report(
        &mut self,
        id: &ModelInstanceId,
    ) -> Result<ModelInstanceUnloadReport, ModelInstanceError> {
        let instance = self.instance_mut(id)?;
        let invalidated = ModelInstanceInvalidationReport {
            kv_caches: std::mem::take(&mut instance.definition.usage.kv_cache_dependencies),
            prefix_entries: std::mem::take(
                &mut instance.definition.usage.prefix_cache_dependencies,
            ),
            adapters_released: std::mem::take(&mut instance.definition.usage.adapter_dependencies),
        };
        let released_memory_allocations =
            std::mem::take(&mut instance.definition.resource_bindings.memory_allocations);
        instance
            .definition
            .resource_bindings
            .released_memory_allocations
            .extend(released_memory_allocations.iter().copied());
        let released_provider_resources = instance
            .definition
            .placement
            .provider_resource
            .take()
            .map(|resource| BTreeSet::from([resource.provider]))
            .unwrap_or_default();
        instance
            .definition
            .resource_bindings
            .released_provider_resources
            .extend(released_provider_resources.iter().cloned());
        instance.definition.associated_sessions.clear();
        Ok(ModelInstanceUnloadReport {
            invalidated,
            released_memory_allocations,
            released_provider_resources,
            dangling_session_references: false,
        })
    }

    pub fn mark_memory_pressure(
        &mut self,
        id: &ModelInstanceId,
        pressure: MemoryPressureLevel,
    ) -> Result<(), ModelInstanceError> {
        let instance = self.instance_mut(id)?;
        if pressure != MemoryPressureLevel::Low && instance.definition.policy.suspension_allowed {
            if matches!(
                instance.lifecycle,
                ModelInstanceLifecycleState::Ready | ModelInstanceLifecycleState::Idle
            ) {
                instance.transition_to(ModelInstanceLifecycleState::Suspended)?;
            }
            self.observe(
                ModelInstanceObservationKind::MemoryPressure,
                Some(id.clone()),
                "model instance memory pressure observed",
                None,
            );
        }
        Ok(())
    }

    fn observe(
        &mut self,
        kind: ModelInstanceObservationKind,
        instance: Option<ModelInstanceId>,
        message: impl Into<String>,
        correlation_id: Option<CorrelationId>,
    ) {
        self.observations.push(ModelInstanceObservation::redacted(
            kind,
            instance,
            message,
            correlation_id,
        ));
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelInstanceError {
    ModelInstanceNotFound,
    ModelInstanceNotReady,
    ModelInstanceLoading,
    ModelInstanceWarming,
    ModelInstanceDraining,
    ModelInstanceSuspended,
    ModelInstanceUnloading,
    ModelInstanceUnloaded,
    ModelInstanceFailed,
    ModelInstanceInvalid,
    ModelInstanceRemoved,
    ModelInstanceActive,
    ModelInstanceBusy,
    ModelInstanceSharingDenied,
    ModelInstancePolicyDenied,
    ModelInstanceReloadRequired,
    ModelInstanceReloadFailed,
    ModelInstanceUnloadFailed,
    ModelInstanceWarmupFailed,
    ModelInstanceProviderUnavailable,
    ModelInstanceProviderNotReady,
    ModelInstanceProviderFailed,
    ModelInstanceDeviceUnavailable,
    ModelInstanceDeviceLost,
    ModelInstanceMemoryPressure,
    ModelInstanceResidencyMissing,
    ModelInstanceAdapterIncompatible,
    ModelInstanceKvCacheInvalidated,
    ModelInstancePrefixCacheInvalidated,
    ModelInstanceBrowserFeatureUnsupported,
    InvalidLifecycleTransition {
        from: ModelInstanceLifecycleState,
        to: ModelInstanceLifecycleState,
    },
    InternalModelInstance {
        reason: String,
    },
}

impl fmt::Display for ModelInstanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelInstanceNotFound => f.write_str("model instance not found"),
            Self::ModelInstanceNotReady => f.write_str("model instance not ready"),
            Self::ModelInstanceLoading => f.write_str("model instance loading"),
            Self::ModelInstanceWarming => f.write_str("model instance warming"),
            Self::ModelInstanceDraining => f.write_str("model instance draining"),
            Self::ModelInstanceSuspended => f.write_str("model instance suspended"),
            Self::ModelInstanceUnloading => f.write_str("model instance unloading"),
            Self::ModelInstanceUnloaded => f.write_str("model instance unloaded"),
            Self::ModelInstanceFailed => f.write_str("model instance failed"),
            Self::ModelInstanceInvalid => f.write_str("model instance invalid"),
            Self::ModelInstanceRemoved => f.write_str("model instance removed"),
            Self::ModelInstanceActive => f.write_str("model instance active"),
            Self::ModelInstanceBusy => f.write_str("model instance busy"),
            Self::ModelInstanceSharingDenied => f.write_str("model instance sharing denied"),
            Self::ModelInstancePolicyDenied => f.write_str("model instance policy denied"),
            Self::ModelInstanceReloadRequired => f.write_str("model instance reload required"),
            Self::ModelInstanceReloadFailed => f.write_str("model instance reload failed"),
            Self::ModelInstanceUnloadFailed => f.write_str("model instance unload failed"),
            Self::ModelInstanceWarmupFailed => f.write_str("model instance warmup failed"),
            Self::ModelInstanceProviderUnavailable => {
                f.write_str("model instance Provider unavailable")
            }
            Self::ModelInstanceProviderNotReady => f.write_str("model instance Provider not ready"),
            Self::ModelInstanceProviderFailed => f.write_str("model instance Provider failed"),
            Self::ModelInstanceDeviceUnavailable => {
                f.write_str("model instance Device unavailable")
            }
            Self::ModelInstanceDeviceLost => f.write_str("model instance Device lost"),
            Self::ModelInstanceMemoryPressure => f.write_str("model instance memory pressure"),
            Self::ModelInstanceResidencyMissing => f.write_str("model instance residency missing"),
            Self::ModelInstanceAdapterIncompatible => {
                f.write_str("model instance adapter incompatible")
            }
            Self::ModelInstanceKvCacheInvalidated => {
                f.write_str("model instance KV cache invalidated")
            }
            Self::ModelInstancePrefixCacheInvalidated => {
                f.write_str("model instance Prefix Cache invalidated")
            }
            Self::ModelInstanceBrowserFeatureUnsupported => {
                f.write_str("model instance browser feature unsupported")
            }
            Self::InvalidLifecycleTransition { from, to } => {
                write!(
                    f,
                    "invalid model instance transition from {from:?} to {to:?}"
                )
            }
            Self::InternalModelInstance { reason } => {
                write!(f, "internal model instance: {reason}")
            }
        }
    }
}

impl Error for ModelInstanceError {}

pub fn readiness_for_lifecycle(lifecycle: ModelInstanceLifecycleState) -> ModelInstanceReadiness {
    match lifecycle {
        ModelInstanceLifecycleState::Ready
        | ModelInstanceLifecycleState::Active
        | ModelInstanceLifecycleState::Idle => ModelInstanceReadiness::Ready,
        ModelInstanceLifecycleState::Draining => ModelInstanceReadiness::Draining,
        ModelInstanceLifecycleState::Suspended => ModelInstanceReadiness::Suspended,
        ModelInstanceLifecycleState::Failed
        | ModelInstanceLifecycleState::Invalid
        | ModelInstanceLifecycleState::Removed => ModelInstanceReadiness::Failed,
        _ => ModelInstanceReadiness::NotReady,
    }
}

pub fn readiness_error(
    lifecycle: ModelInstanceLifecycleState,
    readiness: ModelInstanceReadiness,
) -> ModelInstanceError {
    match lifecycle {
        ModelInstanceLifecycleState::Loading => ModelInstanceError::ModelInstanceLoading,
        ModelInstanceLifecycleState::Warming => ModelInstanceError::ModelInstanceWarming,
        ModelInstanceLifecycleState::Draining => ModelInstanceError::ModelInstanceDraining,
        ModelInstanceLifecycleState::Suspended => ModelInstanceError::ModelInstanceSuspended,
        ModelInstanceLifecycleState::Unloading => ModelInstanceError::ModelInstanceUnloading,
        ModelInstanceLifecycleState::Unloaded => ModelInstanceError::ModelInstanceUnloaded,
        ModelInstanceLifecycleState::Failed => ModelInstanceError::ModelInstanceFailed,
        ModelInstanceLifecycleState::Invalid => ModelInstanceError::ModelInstanceInvalid,
        ModelInstanceLifecycleState::Removed => ModelInstanceError::ModelInstanceRemoved,
        _ if readiness == ModelInstanceReadiness::Failed => ModelInstanceError::ModelInstanceFailed,
        _ => ModelInstanceError::ModelInstanceNotReady,
    }
}

fn validate_instance_identity(value: &str) -> Result<(), ModelInstanceError> {
    let lower = value.to_ascii_lowercase();
    if value.trim().is_empty()
        || value.contains('/')
        || value.contains('\\')
        || value.contains(':')
        || value.contains("0x")
        || lower.contains("provider")
        || lower.contains("device")
        || lower.contains("ptr")
        || lower.contains("weight")
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    {
        return Err(ModelInstanceError::ModelInstancePolicyDenied);
    }
    Ok(())
}

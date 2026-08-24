//! Runtime-owned Kernel Dispatch planning and revalidation.
//!
//! Dispatch accepts Kernel Candidates produced by the Runtime-owned registry and
//! turns them into Provider-bound Kernel Invocations. It carries only stable
//! metadata and never exposes raw Provider handles or function pointers.

use crate::*;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelDispatchPlanId(String);

impl KernelDispatchPlanId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for KernelDispatchPlanId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelDispatchLifecycleState {
    Planned,
    Ready,
    Submitted,
    Running,
    Completed,
    Failed,
    CancelRequested,
    Cancelled,
    TimedOut,
    FallbackPending,
    FallbackRunning,
    Released,
}

impl KernelDispatchLifecycleState {
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Cancelled | Self::TimedOut | Self::Released
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelFallbackStep {
    AlternateKernel(KernelId),
    SameProviderDifferentDevice(DeviceBinding),
    AlternateProvider(ProviderBinding),
    ExplicitDTypeConversion {
        from: ComputeDType,
        to: ComputeDType,
    },
    ExplicitLayoutConversion {
        from: TensorLayoutKind,
        to: TensorLayoutKind,
    },
    ExplicitDataMovement(ResourceAffinity),
    HostExecution,
    Rejection(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelDispatchRevalidationContext {
    pub provider_status: Option<ProviderStatusSnapshot>,
    pub device_status: Option<DeviceStatus>,
    pub memory_reservation_valid: bool,
    pub operation_active: bool,
    pub session_active: bool,
    pub model_instance_ready: bool,
    pub batching_valid: bool,
    pub adapter_valid: bool,
    pub kv_cache_valid: bool,
    pub prefix_cache_valid: bool,
    pub cancellation_requested: bool,
    pub policy_allows_dispatch: bool,
}

impl Default for KernelDispatchRevalidationContext {
    fn default() -> Self {
        Self {
            provider_status: None,
            device_status: None,
            memory_reservation_valid: true,
            operation_active: true,
            session_active: true,
            model_instance_ready: true,
            batching_valid: true,
            adapter_valid: true,
            kv_cache_valid: true,
            prefix_cache_valid: true,
            cancellation_requested: false,
            policy_allows_dispatch: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelDispatchPlan {
    pub id: KernelDispatchPlanId,
    pub selected_kernel: KernelId,
    pub provider: ProviderBinding,
    pub device: Option<DeviceBinding>,
    pub invocation: KernelInvocation,
    pub input_bindings: Vec<KernelResource>,
    pub output_bindings: Vec<KernelResource>,
    pub workspace_reservation: Option<MemoryAllocationId>,
    pub movement_steps: Vec<ResourceAffinity>,
    pub conversion_steps: Vec<String>,
    pub execution_mode: KernelExecutionMode,
    pub cancellation: KernelCancellationSupport,
    pub deadline_millis: Option<u64>,
    pub fallback_chain: Vec<KernelFallbackStep>,
    pub observability_correlation: Option<String>,
    pub cleanup_behavior: BTreeMap<String, String>,
    pub expected_result_metadata: BTreeMap<String, String>,
    pub device_metadata: Option<DeviceMetadata>,
    pub lifecycle: KernelDispatchLifecycleState,
}

impl KernelDispatchPlan {
    pub fn from_selection(
        id: KernelDispatchPlanId,
        request: &KernelSelectionRequest,
        candidate: &KernelCandidate,
        advertisement: &KernelAdvertisement,
        invocation_id: KernelInvocationId,
    ) -> Result<Self, KernelDispatchError> {
        if !candidate.compatible {
            return Err(KernelDispatchError::PlanInvalid(
                "selected candidate is incompatible".into(),
            ));
        }
        if candidate.kernel != advertisement.id {
            return Err(KernelDispatchError::PlanInvalid(
                "candidate Kernel does not match advertisement".into(),
            ));
        }
        let mut invocation = KernelInvocation::new(
            invocation_id,
            request.operator.clone(),
            candidate.kernel.clone(),
            candidate.provider.clone(),
            request.affinity.clone(),
        );
        invocation.inputs = request.inputs.clone();
        invocation.outputs = request.outputs.clone();
        invocation.device = candidate.device.clone();
        invocation.execution_mode = request.execution_mode.unwrap_or_else(|| {
            advertisement
                .execution_modes
                .iter()
                .next()
                .copied()
                .unwrap_or(KernelExecutionMode::Synchronous)
        });
        invocation.deadline_millis = request.deadline_millis;
        invocation.observability_correlation = request.observability_correlation.clone();
        invocation.policy = request.policy.clone();
        invocation.deterministic_required = request.deterministic_required;
        invocation.precision = request.precision;
        let fallback_chain = selection_fallback_chain(candidate, request);
        let movement_steps = fallback_chain
            .iter()
            .filter_map(|step| match step {
                KernelFallbackStep::ExplicitDataMovement(affinity) => Some(affinity.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let conversion_steps = fallback_chain
            .iter()
            .filter_map(|step| match step {
                KernelFallbackStep::ExplicitDTypeConversion { from, to } => {
                    Some(format!("dtype:{from:?}->{to:?}"))
                }
                KernelFallbackStep::ExplicitLayoutConversion { from, to } => {
                    Some(format!("layout:{from:?}->{to:?}"))
                }
                _ => None,
            })
            .collect::<Vec<_>>();

        Ok(Self {
            id,
            selected_kernel: candidate.kernel.clone(),
            provider: candidate.provider.clone(),
            device: candidate.device.clone(),
            input_bindings: request.inputs.clone(),
            output_bindings: request.outputs.clone(),
            workspace_reservation: invocation.workspace,
            execution_mode: invocation.execution_mode,
            cancellation: advertisement.cancellation,
            deadline_millis: request.deadline_millis,
            fallback_chain,
            observability_correlation: request.observability_correlation.clone(),
            cleanup_behavior: BTreeMap::new(),
            expected_result_metadata: BTreeMap::new(),
            device_metadata: None,
            lifecycle: KernelDispatchLifecycleState::Planned,
            invocation,
            movement_steps,
            conversion_steps,
        })
    }

    pub fn without_raw_handles(&self) -> bool {
        let text = format!("{self:?}");
        !text.contains("0x") && !text.contains("function pointer") && !text.contains("raw handle")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelDispatchError {
    PlanInvalid(String),
    DispatchStale(String),
    DispatchRejected(String),
    DispatchFailed(String),
    FallbackUnavailable,
    FallbackFailed(String),
    CancellationUnsupported,
    Cancelled,
    Timeout,
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

impl KernelDispatchError {
    pub const fn code(&self) -> &'static str {
        match self {
            Self::PlanInvalid(_) => "kernel-dispatch-plan-invalid",
            Self::DispatchStale(_) => "kernel-dispatch-stale",
            Self::DispatchRejected(_) => "kernel-dispatch-rejected",
            Self::DispatchFailed(_) => "kernel-dispatch-failed",
            Self::FallbackUnavailable => "kernel-fallback-unavailable",
            Self::FallbackFailed(_) => "kernel-fallback-failed",
            Self::CancellationUnsupported => "kernel-cancellation-unsupported",
            Self::Cancelled => "kernel-cancelled",
            Self::Timeout => "kernel-timeout",
            Self::ProviderUnavailable { .. } => "kernel-provider-unavailable",
            Self::ProviderNotReady { .. } => "kernel-provider-not-ready",
            Self::ProviderSaturated { .. } => "kernel-provider-saturated",
            Self::DeviceUnavailable { .. } => "kernel-device-unavailable",
            Self::DeviceIncompatible { .. } => "kernel-device-incompatible",
            Self::MemoryInfeasible(_) => "kernel-memory-infeasible",
            Self::WorkspaceUnavailable => "kernel-workspace-unavailable",
            Self::ResourceAffinityConflict(_) => "kernel-resource-affinity-conflict",
            Self::BrowserFeatureUnsupported(_) => "kernel-browser-feature-unsupported",
            Self::Internal(_) => "internal-kernel-dispatch",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelDispatchResult {
    pub selected_kernel: KernelId,
    pub provider: ProviderBinding,
    pub device: Option<DeviceBinding>,
    pub status: KernelResultStatus,
    pub output_readiness: BTreeMap<String, bool>,
    pub updated_resources: Vec<TensorResourceDescriptor>,
    pub timing_micros: Option<u64>,
    pub fallback_used: Option<KernelFallbackStep>,
    pub cancellation_result: Option<KernelDispatchLifecycleState>,
    pub determinism: Option<KernelDeterminism>,
    pub precision: Option<KernelPrecisionMetadata>,
    pub provider_diagnostics: BTreeMap<String, String>,
    pub device_diagnostics: BTreeMap<String, String>,
    pub error: Option<KernelDispatchError>,
}

impl KernelDispatchResult {
    pub fn from_kernel_result(plan: &KernelDispatchPlan, result: KernelResult) -> Self {
        Self {
            selected_kernel: plan.selected_kernel.clone(),
            provider: plan.provider.clone(),
            device: plan.device.clone(),
            status: result.status,
            output_readiness: result.output_readiness,
            updated_resources: result.updated_resources,
            timing_micros: result.timing_micros,
            fallback_used: None,
            cancellation_result: match result.status {
                KernelResultStatus::Cancelled => Some(KernelDispatchLifecycleState::Cancelled),
                KernelResultStatus::TimedOut => Some(KernelDispatchLifecycleState::TimedOut),
                KernelResultStatus::Succeeded | KernelResultStatus::Failed => None,
            },
            determinism: result.determinism,
            precision: result.precision,
            provider_diagnostics: result.provider_diagnostics,
            device_diagnostics: result.device_diagnostics,
            error: result.error.map(map_kernel_error),
        }
    }

    pub fn failure(plan: &KernelDispatchPlan, error: KernelDispatchError) -> Self {
        Self {
            selected_kernel: plan.selected_kernel.clone(),
            provider: plan.provider.clone(),
            device: plan.device.clone(),
            status: KernelResultStatus::Failed,
            output_readiness: BTreeMap::new(),
            updated_resources: Vec::new(),
            timing_micros: None,
            fallback_used: None,
            cancellation_result: None,
            determinism: None,
            precision: None,
            provider_diagnostics: BTreeMap::new(),
            device_diagnostics: BTreeMap::new(),
            error: Some(error),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct KernelDispatcher {
    observations: Vec<KernelObservation>,
}

impl KernelDispatcher {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn observations(&self) -> &[KernelObservation] {
        &self.observations
    }

    pub fn revalidate(
        &mut self,
        registry: &KernelRegistry,
        plan: &mut KernelDispatchPlan,
    ) -> Result<(), KernelDispatchError> {
        self.revalidate_with_context(
            registry,
            plan,
            &KernelDispatchRevalidationContext::default(),
        )
    }

    pub fn revalidate_with_context(
        &mut self,
        registry: &KernelRegistry,
        plan: &mut KernelDispatchPlan,
        context: &KernelDispatchRevalidationContext,
    ) -> Result<(), KernelDispatchError> {
        self.observations.push(
            KernelObservation::new(KernelObservationKind::KernelDispatchPlanRevalidated)
                .with_kernel(&plan.selected_kernel),
        );
        let Some(advertisement) = registry.active_advertisement(&plan.selected_kernel) else {
            plan.lifecycle = KernelDispatchLifecycleState::Failed;
            return Err(KernelDispatchError::DispatchStale(
                "selected Kernel is no longer active".into(),
            ));
        };
        if advertisement.id.provider != plan.provider {
            plan.lifecycle = KernelDispatchLifecycleState::Failed;
            let error =
                KernelDispatchError::DispatchStale("selected Kernel Provider changed".into());
            self.observe_failure(plan, &error);
            return Err(error);
        }
        if let Some(status) = context.provider_status.as_ref() {
            match status.provider_health_compat() {
                HealthState::Available | HealthState::Degraded => {}
                HealthState::Saturated => {
                    let error = KernelDispatchError::ProviderSaturated {
                        provider: plan.provider.clone(),
                    };
                    self.observe_failure(plan, &error);
                    return Err(error);
                }
                HealthState::Draining | HealthState::Initializing | HealthState::Unknown => {
                    let error = KernelDispatchError::ProviderNotReady {
                        provider: plan.provider.clone(),
                    };
                    self.observe_failure(plan, &error);
                    return Err(error);
                }
                HealthState::Unavailable | HealthState::Interrupted => {
                    let error = KernelDispatchError::ProviderUnavailable {
                        provider: plan.provider.clone(),
                    };
                    self.observe_failure(plan, &error);
                    return Err(error);
                }
            }
            if !matches!(status.admission, ProviderAdmissionDecision::Admit) {
                let error = KernelDispatchError::DispatchRejected(
                    "Provider admission rejected dispatch".into(),
                );
                self.observe_failure(plan, &error);
                return Err(error);
            }
        }
        if let Some(status) = context.device_status.as_ref() {
            plan.device_metadata = Some(DeviceMetadata {
                id: status.device.id().clone(),
                name: status.device.to_string(),
                device_type: DeviceType::Other,
                vendor: String::new(),
                architecture: String::new(),
                memory_capacity: status.capacity.available_memory_bytes.unwrap_or(0),
                compute_units: 0,
                execution_capabilities: Default::default(),
                provider: status.provider.to_string(),
            });
            if !matches!(
                status.availability,
                HealthState::Available | HealthState::Degraded | HealthState::Draining
            ) {
                let error = KernelDispatchError::DeviceUnavailable {
                    device: status.device.clone(),
                };
                self.observe_failure(plan, &error);
                return Err(error);
            }
        }
        if !advertisement
            .execution_modes
            .contains(&plan.invocation.execution_mode)
        {
            plan.lifecycle = KernelDispatchLifecycleState::Failed;
            let error = KernelDispatchError::DispatchRejected(
                "execution mode is no longer supported".into(),
            );
            self.observe_failure(plan, &error);
            return Err(error);
        }
        for output in &plan.output_bindings {
            if let Err(error) = validate_affinity_compatibility(
                &plan.invocation.affinity,
                &output.resource.affinity,
            ) {
                plan.lifecycle = KernelDispatchLifecycleState::Failed;
                let error = KernelDispatchError::ResourceAffinityConflict(error.to_string());
                self.observe_failure(plan, &error);
                self.observations.push(
                    KernelObservation::new(KernelObservationKind::KernelResourceAffinityConflict)
                        .with_kernel(&plan.selected_kernel),
                );
                return Err(error);
            }
        }
        if !context.memory_reservation_valid {
            let error = KernelDispatchError::WorkspaceUnavailable;
            self.observe_failure(plan, &error);
            self.observations.push(
                KernelObservation::new(KernelObservationKind::KernelMemoryFeasibilityFailed)
                    .with_kernel(&plan.selected_kernel),
            );
            return Err(error);
        }
        if !context.operation_active {
            let error =
                KernelDispatchError::DispatchStale("operation lifecycle is inactive".into());
            self.observe_failure(plan, &error);
            return Err(error);
        }
        if !context.session_active {
            let error = KernelDispatchError::DispatchStale("session lifecycle is inactive".into());
            self.observe_failure(plan, &error);
            return Err(error);
        }
        if !context.model_instance_ready {
            let error =
                KernelDispatchError::DispatchStale("Model Instance lifecycle is not ready".into());
            self.observe_failure(plan, &error);
            return Err(error);
        }
        if !context.batching_valid {
            let error = KernelDispatchError::DispatchRejected("batch metadata is invalid".into());
            self.observe_failure(plan, &error);
            return Err(error);
        }
        if !context.adapter_valid {
            let error = KernelDispatchError::DispatchStale("adapter state changed".into());
            self.observe_failure(plan, &error);
            return Err(error);
        }
        if !context.kv_cache_valid {
            let error = KernelDispatchError::DispatchStale("KV cache state is invalid".into());
            self.observe_failure(plan, &error);
            return Err(error);
        }
        if !context.prefix_cache_valid {
            let error = KernelDispatchError::DispatchStale("Prefix Cache policy changed".into());
            self.observe_failure(plan, &error);
            return Err(error);
        }
        if context.cancellation_requested {
            let error = KernelDispatchError::Cancelled;
            self.observe_failure(plan, &error);
            return Err(error);
        }
        if !context.policy_allows_dispatch {
            let error =
                KernelDispatchError::DispatchRejected("Runtime policy denied dispatch".into());
            self.observe_failure(plan, &error);
            return Err(error);
        }
        plan.lifecycle = KernelDispatchLifecycleState::Ready;
        Ok(())
    }

    pub fn submit_metadata_only(
        &mut self,
        registry: &KernelRegistry,
        plan: &mut KernelDispatchPlan,
    ) -> Result<KernelDispatchResult, KernelDispatchError> {
        self.revalidate(registry, plan)?;
        plan.lifecycle = KernelDispatchLifecycleState::Submitted;
        self.observations.push(
            KernelObservation::new(KernelObservationKind::KernelDispatchSubmitted)
                .with_kernel(&plan.selected_kernel),
        );
        plan.lifecycle = KernelDispatchLifecycleState::Running;
        self.observations.push(
            KernelObservation::new(KernelObservationKind::KernelDispatchRunning)
                .with_kernel(&plan.selected_kernel),
        );
        plan.lifecycle = KernelDispatchLifecycleState::Completed;
        let result = KernelResult::success(plan.invocation.id.clone());
        self.observations.push(
            KernelObservation::new(KernelObservationKind::KernelDispatchCompleted)
                .with_kernel(&plan.selected_kernel),
        );
        Ok(KernelDispatchResult::from_kernel_result(plan, result))
    }

    pub fn record_fallback_considered(&mut self, plan: &KernelDispatchPlan) {
        self.observations.push(
            KernelObservation::new(KernelObservationKind::KernelFallbackConsidered)
                .with_kernel(&plan.selected_kernel),
        );
    }

    pub fn record_fallback_selected(&mut self, plan: &KernelDispatchPlan) {
        self.observations.push(
            KernelObservation::new(KernelObservationKind::KernelFallbackSelected)
                .with_kernel(&plan.selected_kernel),
        );
    }

    pub fn record_fallback_failed(&mut self, plan: &KernelDispatchPlan) {
        self.observations.push(
            KernelObservation::new(KernelObservationKind::KernelFallbackFailed)
                .with_kernel(&plan.selected_kernel),
        );
    }

    pub fn record_conformance_gating(&mut self, plan: &KernelDispatchPlan) {
        self.observations.push(
            KernelObservation::new(KernelObservationKind::KernelConformanceGatingApplied)
                .with_kernel(&plan.selected_kernel),
        );
    }

    fn observe_failure(&mut self, plan: &KernelDispatchPlan, error: &KernelDispatchError) {
        self.observations.push(
            KernelObservation::new(KernelObservationKind::KernelDispatchFailed)
                .with_kernel(&plan.selected_kernel)
                .with_redacted_metadata("error", error.code()),
        );
    }
}

fn map_kernel_error(error: KernelError) -> KernelDispatchError {
    match error {
        KernelError::KernelProviderUnavailable { provider } => {
            KernelDispatchError::ProviderUnavailable {
                provider: ProviderBinding::new(provider),
            }
        }
        KernelError::KernelProviderNotReady { provider } => KernelDispatchError::ProviderNotReady {
            provider: ProviderBinding::new(provider),
        },
        KernelError::KernelProviderSaturated { provider } => {
            KernelDispatchError::ProviderSaturated {
                provider: ProviderBinding::new(provider),
            }
        }
        KernelError::KernelDeviceUnsupported { device } => {
            KernelDispatchError::DeviceIncompatible {
                device: DeviceBinding::new(DeviceId::new(device)),
            }
        }
        KernelError::KernelWorkspaceUnavailable => KernelDispatchError::WorkspaceUnavailable,
        KernelError::KernelResourceAffinityConflict { reason } => {
            KernelDispatchError::ResourceAffinityConflict(reason)
        }
        KernelError::KernelCancellationUnsupported => KernelDispatchError::CancellationUnsupported,
        KernelError::KernelCancelled => KernelDispatchError::Cancelled,
        KernelError::KernelTimeout => KernelDispatchError::Timeout,
        KernelError::KernelBrowserFeatureUnsupported { feature } => {
            KernelDispatchError::BrowserFeatureUnsupported(feature)
        }
        other => KernelDispatchError::DispatchFailed(other.to_string()),
    }
}

pub fn kernel_dispatch_error_from_provider_execution(
    error: ProviderExecutionError,
) -> KernelDispatchError {
    match error.code {
        ProviderExecutionErrorCode::ProviderUnavailable
        | ProviderExecutionErrorCode::ProviderInterrupted => {
            KernelDispatchError::ProviderUnavailable {
                provider: error.provider,
            }
        }
        ProviderExecutionErrorCode::ProviderInitializing
        | ProviderExecutionErrorCode::ProviderDraining
        | ProviderExecutionErrorCode::ProviderHealthUnknown
        | ProviderExecutionErrorCode::StaleHealthReport => KernelDispatchError::ProviderNotReady {
            provider: error.provider,
        },
        ProviderExecutionErrorCode::ProviderSaturated => KernelDispatchError::ProviderSaturated {
            provider: error.provider,
        },
        ProviderExecutionErrorCode::DeviceUnavailable
        | ProviderExecutionErrorCode::DeviceHealthUnknown => {
            KernelDispatchError::DeviceUnavailable {
                device: error
                    .device
                    .unwrap_or_else(|| DeviceBinding::new(DeviceId::new("unknown-device"))),
            }
        }
        ProviderExecutionErrorCode::DeviceSaturated
        | ProviderExecutionErrorCode::IncompatibleResourceAffinity => {
            KernelDispatchError::DeviceIncompatible {
                device: error
                    .device
                    .unwrap_or_else(|| DeviceBinding::new(DeviceId::new("unknown-device"))),
            }
        }
        ProviderExecutionErrorCode::MemoryPlanRejected
        | ProviderExecutionErrorCode::ResourceExhausted
        | ProviderExecutionErrorCode::OutOfMemory => {
            KernelDispatchError::MemoryInfeasible(error.message)
        }
        ProviderExecutionErrorCode::UnsupportedDType
        | ProviderExecutionErrorCode::UnsupportedLayout
        | ProviderExecutionErrorCode::DataMovementFailed
        | ProviderExecutionErrorCode::MaterializationFailed => {
            KernelDispatchError::DispatchRejected(error.message)
        }
        ProviderExecutionErrorCode::CancellationUnsupported => {
            KernelDispatchError::CancellationUnsupported
        }
        ProviderExecutionErrorCode::CancellationFailed => {
            KernelDispatchError::DispatchFailed(error.message)
        }
        ProviderExecutionErrorCode::InvalidExecutionPlan => {
            KernelDispatchError::PlanInvalid(error.message)
        }
        ProviderExecutionErrorCode::SubmissionFailed
        | ProviderExecutionErrorCode::ExecutionFailed
        | ProviderExecutionErrorCode::ExecutionInterrupted
        | ProviderExecutionErrorCode::UnsupportedOperation
        | ProviderExecutionErrorCode::CapabilityUnavailable
        | ProviderExecutionErrorCode::ProviderDegradedRejected => {
            KernelDispatchError::DispatchFailed(error.message)
        }
    }
}

fn selection_fallback_chain(
    candidate: &KernelCandidate,
    request: &KernelSelectionRequest,
) -> Vec<KernelFallbackStep> {
    let mut steps = Vec::new();
    steps.push(KernelFallbackStep::AlternateKernel(
        candidate.kernel.clone(),
    ));
    if let Some(device) = candidate.device.clone() {
        steps.push(KernelFallbackStep::SameProviderDifferentDevice(device));
    }
    steps.push(KernelFallbackStep::AlternateProvider(
        candidate.provider.clone(),
    ));
    if let Some(dtype) = request.dtype_requirements.iter().next().copied() {
        steps.push(KernelFallbackStep::ExplicitDTypeConversion {
            from: dtype,
            to: dtype,
        });
    }
    if let Some(layout) = request.layout_requirements.iter().next().copied() {
        steps.push(KernelFallbackStep::ExplicitLayoutConversion {
            from: layout,
            to: layout,
        });
    }
    steps.push(KernelFallbackStep::ExplicitDataMovement(
        request.affinity.clone(),
    ));
    steps.push(KernelFallbackStep::HostExecution);
    steps.push(KernelFallbackStep::Rejection(
        "no compatible fallback".into(),
    ));
    steps
}

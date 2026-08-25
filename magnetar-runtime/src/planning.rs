use crate::affinity::{
    AffinityError, AffinityGroupId, CapabilityBinding, DeviceBinding, ExecutionContextId,
    FallbackClass, ProviderBinding, ResourceAffinity,
};
use crate::capability::CapabilityId;
use crate::compute::{
    COMPUTE_CAPABILITY_ID, COMPUTE_CAPABILITY_VERSION, ComputeDType, ComputeDiagnostic,
    ComputeGraph, ComputeGraphId, ComputeInputId, ComputeLayout, ComputeNodeId, ComputeOperationId,
    ComputeOutputId, ComputePrecision, ComputeValidationError, ComputeValueRef, TensorDescriptor,
    TensorResourceId, effective_compute_advertisement,
};
use crate::observability::{RuntimeEvent, TraceId};
use crate::provider::ProviderMetadata;
use crate::resolution::{
    ResolutionDecision, ResolutionDecisionReason, ResolutionPolicyId, ResolutionRejectionReason,
};
use crate::scheduler::{
    ProviderExecutionId, ScheduledOperationId, runtime_events_for_execution_plan,
};
use std::{error::Error, fmt};
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ExecutionPlanId(String);
impl ExecutionPlanId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl fmt::Display for ExecutionPlanId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputeExecutionPhase {
    Validation,
    Resolution,
    Planning,
    DataMovement,
    Materialization,
    MemoryAllocation,
    ProviderSubmission,
    Execution,
    Completion,
    Cancellation,
    Interruption,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputeExecutionClassification {
    Transparent,
    Restartable,
    ProviderPinned,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutionStepKind {
    ValidateGraph,
    ResolveProvider,
    ResolveDevice,
    ValidateCapabilityVersion,
    ValidateOperationSchema,
    ValidateDType,
    ValidateLayout,
    ValidatePrecisionPolicy,
    ValidateDeterminism,
    BindInputResource,
    BindOutputResource,
    PreserveProviderPinnedAffinity,
    PreserveDeviceBoundAffinity,
    PreserveAffinityGroup,
    RejectIncompatibleResourceChain,
    Upload,
    Download,
    Copy,
    Transfer,
    Materialize,
    AllocateMemory,
    ValidateMemory,
    SubmitToProvider,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionStep {
    pub id: String,
    pub phase: ComputeExecutionPhase,
    pub kind: ExecutionStepKind,
    pub provider: ProviderBinding,
    pub device: Option<DeviceBinding>,
    pub dependencies: Vec<String>,
}
impl ExecutionStep {
    pub fn new(
        id: impl Into<String>,
        phase: ComputeExecutionPhase,
        kind: ExecutionStepKind,
        provider: ProviderBinding,
    ) -> Self {
        Self {
            id: id.into(),
            phase,
            kind,
            provider,
            device: None,
            dependencies: Vec::new(),
        }
    }
    pub fn with_device(mut self, device: Option<DeviceBinding>) -> Self {
        self.device = device;
        self
    }
    pub fn depends_on(mut self, dependency: impl Into<String>) -> Self {
        self.dependencies.push(dependency.into());
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionInput {
    pub id: ComputeInputId,
    pub descriptor: TensorDescriptor,
    pub resource: Option<TensorResourceId>,
    pub affinity: ResourceAffinity,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionOutput {
    pub id: ComputeOutputId,
    pub descriptor: TensorDescriptor,
    pub affinity: ResourceAffinity,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutionConstraint {
    ResolutionPolicy(ResolutionPolicyId),
    CapabilityVersion(CapabilityBinding),
    Provider(ProviderBinding),
    Device(DeviceBinding),
    ResourceAffinity(ResourceAffinity),
    AffinityGroup(AffinityGroupId),
    OperationSchema(ComputeOperationId),
    DType(ComputeDType),
    Layout(ComputeLayout),
    PrecisionPolicy(ComputePrecision),
    DeterministicBehavior,
    MemoryRequirement(String),
    ExplicitTransferRequired(TensorResourceId),
    ExplicitTransferRequirement(String),
    ExplicitMaterializationRequired(String),
    NoHiddenCpuStaging,
    NoImplicitProviderMigration,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutionDiagnostic {
    SelectedProvider(ProviderBinding),
    SelectedDevice(DeviceBinding),
    SelectedCapability(CapabilityBinding),
    ResolutionDecision(ResolutionDecision),
    RejectedProviderCandidate {
        provider: ProviderBinding,
        reason: ResolutionRejectionReason,
    },
    Memory(MemoryPlanningDiagnostic),
    TransferRequired {
        resource: TensorResourceId,
        from: ResourceAffinity,
        to: ResourceAffinity,
    },
    MaterializationRequired {
        source: String,
    },
    PolicyDecisionReason(ResolutionDecisionReason),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeExecutionPlan {
    pub id: ExecutionPlanId,
    pub trace_id: TraceId,
    pub graph: ComputeGraphId,
    pub provider: ProviderBinding,
    pub device: Option<DeviceBinding>,
    pub capability: CapabilityBinding,
    pub policy: ResolutionPolicyId,
    pub classification: ComputeExecutionClassification,
    pub inputs: Vec<ExecutionInput>,
    pub outputs: Vec<ExecutionOutput>,
    pub constraints: Vec<ExecutionConstraint>,
    pub steps: Vec<ExecutionStep>,
    pub memory_plan: MemoryPlan,
    pub diagnostics: Vec<ExecutionDiagnostic>,
    pub(crate) validated: bool,
}
impl ComputeExecutionPlan {
    pub fn is_validated(&self) -> bool {
        self.validated
    }
    pub fn trace_id(&self) -> &TraceId {
        &self.trace_id
    }
    pub fn observations(&self) -> Vec<RuntimeEvent> {
        runtime_events_for_execution_plan(self)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ComputePlanningError {
    PlanningFailed {
        reason: String,
    },
    NoCompatibleProvider {
        capability: CapabilityBinding,
    },
    NoCompatibleDevice {
        provider: ProviderBinding,
    },
    PolicyRejectedProvider {
        capability: CapabilityBinding,
        policy: ResolutionPolicyId,
    },
    UnsupportedOperation(ComputeOperationId),
    UnsupportedDType(ComputeDType),
    UnsupportedLayout(ComputeLayout),
    UnsupportedPrecisionPolicy(ComputePrecision),
    IncompatibleResourceAffinity(AffinityError),
    UnresolvedAffinityGroup(AffinityGroupId),
    MemoryPlanFailed(MemoryPlanningError),
    DataMovementRequired {
        resource: TensorResourceId,
    },
    UnsupportedTransfer {
        reason: String,
    },
    MaterializationRequired {
        source: String,
    },
    ProviderUnavailable(ProviderBinding),
    DeviceUnavailable(DeviceBinding),
    InvalidExecutionPlan {
        reason: String,
    },
}
impl fmt::Display for ComputePlanningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PlanningFailed { reason } => {
                write!(f, "compute execution planning failed: {reason}")
            }
            Self::NoCompatibleProvider { capability } => {
                write!(f, "no compatible provider for capability '{capability}'")
            }
            Self::NoCompatibleDevice { provider } => {
                write!(f, "no compatible device for provider '{provider}'")
            }
            Self::PolicyRejectedProvider { capability, policy } => write!(
                f,
                "resolution policy '{policy}' rejected all providers for capability '{capability}'"
            ),
            Self::UnsupportedOperation(operation) => {
                write!(f, "unsupported operation schema '{operation}'")
            }
            Self::UnsupportedDType(dtype) => write!(f, "unsupported dtype {dtype:?}"),
            Self::UnsupportedLayout(layout) => write!(f, "unsupported layout {layout:?}"),
            Self::UnsupportedPrecisionPolicy(precision) => {
                write!(f, "unsupported precision policy {precision:?}")
            }
            Self::IncompatibleResourceAffinity(error) => {
                write!(f, "incompatible execution resource affinity: {error}")
            }
            Self::UnresolvedAffinityGroup(group) => {
                write!(f, "unresolved affinity group '{group}'")
            }
            Self::MemoryPlanFailed(error) => write!(f, "{error}"),
            Self::DataMovementRequired { resource } => {
                write!(
                    f,
                    "explicit data movement required for resource '{resource}'"
                )
            }
            Self::UnsupportedTransfer { reason } => {
                write!(f, "unsupported execution transfer: {reason}")
            }
            Self::MaterializationRequired { source } => {
                write!(f, "explicit materialization required for '{source}'")
            }
            Self::ProviderUnavailable(provider) => {
                write!(f, "provider '{provider}' is unavailable")
            }
            Self::DeviceUnavailable(device) => write!(f, "device '{device}' is unavailable"),
            Self::InvalidExecutionPlan { reason } => {
                write!(f, "invalid compute execution plan: {reason}")
            }
        }
    }
}
impl Error for ComputePlanningError {}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MemoryRegionKind {
    GraphInput,
    GraphOutput,
    Intermediate,
    Temporary,
    Materialization,
    Transfer,
    HostStaging,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryRequirement {
    pub id: String,
    pub region: MemoryRegionKind,
    pub byte_size: u64,
    pub affinity: ResourceAffinity,
    pub reusable: bool,
}
impl MemoryRequirement {
    pub fn new(
        id: impl Into<String>,
        region: MemoryRegionKind,
        byte_size: u64,
        affinity: ResourceAffinity,
    ) -> Self {
        Self {
            id: id.into(),
            region,
            byte_size,
            affinity,
            reusable: false,
        }
    }
    pub const fn reusable(mut self) -> Self {
        self.reusable = true;
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorLifetime {
    pub id: String,
    pub first_step: usize,
    pub last_step: usize,
    pub byte_size: u64,
    pub affinity: ResourceAffinity,
}
impl TensorLifetime {
    pub fn overlaps(&self, other: &Self) -> bool {
        self.first_step <= other.last_step && other.first_step <= self.last_step
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BufferLifetime {
    pub id: String,
    pub source: String,
    pub first_step: usize,
    pub last_step: usize,
    pub byte_size: u64,
    pub affinity: ResourceAffinity,
    pub reuses: Option<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MemoryPressureReport {
    pub estimated_required_bytes: u64,
    pub estimated_peak_bytes: u64,
    pub selected_provider: Option<ProviderBinding>,
    pub selected_device: Option<DeviceBinding>,
    pub rejected_device_limit: Option<u64>,
    pub materialization_cost_bytes: u64,
    pub transfer_buffer_cost_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MemoryPlanningDecision {
    Allocate { requirement: String },
    Reuse { requirement: String, buffer: String },
    PreservePinnedResource { resource: TensorResourceId },
    RequireMaterialization { requirement: String },
    RequireTransfer { requirement: String },
    AccountHostStaging { requirement: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MemoryPlanningDiagnostic {
    EstimatedRequirement {
        requirement: String,
        bytes: u64,
    },
    PeakBytes {
        bytes: u64,
    },
    ProviderLimit {
        provider: ProviderBinding,
        max_bytes: u64,
    },
    DeviceLimit {
        device: DeviceBinding,
        max_bytes: u64,
    },
    MaterializationCost {
        bytes: u64,
    },
    TransferCost {
        bytes: u64,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryPlan {
    pub provider: ProviderBinding,
    pub graph: Option<ComputeGraphId>,
    pub requirements: Vec<MemoryRequirement>,
    pub tensor_lifetimes: Vec<TensorLifetime>,
    pub buffer_lifetimes: Vec<BufferLifetime>,
    pub pressure: MemoryPressureReport,
    pub decisions: Vec<MemoryPlanningDecision>,
    pub diagnostics: Vec<MemoryPlanningDiagnostic>,
    pub output_affinity: ResourceAffinity,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MemoryPlanningError {
    MemoryPlanningFailed {
        reason: String,
        report: MemoryPressureReport,
    },
    OutOfMemory {
        required: u64,
        limit: u64,
        report: MemoryPressureReport,
    },
    ResourceExhausted {
        reason: String,
        report: MemoryPressureReport,
    },
    SizeOverflow {
        reason: String,
        report: MemoryPressureReport,
    },
    IncompatibleResourceAffinity(AffinityError),
    UnsupportedLayout {
        layout: ComputeLayout,
        report: MemoryPressureReport,
    },
    MaterializationRequired {
        reason: String,
        report: MemoryPressureReport,
    },
    TransferRequired {
        reason: String,
        report: MemoryPressureReport,
    },
    ProviderMemoryLimitExceeded {
        provider: ProviderBinding,
        required: u64,
        limit: u64,
        report: MemoryPressureReport,
    },
    DeviceMemoryLimitExceeded {
        device: DeviceBinding,
        required: u64,
        limit: u64,
        report: MemoryPressureReport,
    },
}
impl fmt::Display for MemoryPlanningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MemoryPlanningFailed { reason, .. } => {
                write!(f, "memory planning failed: {reason}")
            }
            Self::OutOfMemory {
                required, limit, ..
            } => write!(
                f,
                "memory planning requires {required} bytes but only {limit} bytes are available"
            ),
            Self::ResourceExhausted { reason, .. } => write!(f, "resource exhausted: {reason}"),
            Self::SizeOverflow { reason, .. } => write!(f, "memory size overflow: {reason}"),
            Self::IncompatibleResourceAffinity(error) => {
                write!(f, "incompatible memory resource affinity: {error}")
            }
            Self::UnsupportedLayout { layout, .. } => {
                write!(f, "memory planning does not support layout {layout:?}")
            }
            Self::MaterializationRequired { reason, .. } => {
                write!(f, "memory planning requires materialization: {reason}")
            }
            Self::TransferRequired { reason, .. } => {
                write!(f, "memory planning requires explicit transfer: {reason}")
            }
            Self::ProviderMemoryLimitExceeded {
                provider,
                required,
                limit,
                ..
            } => write!(
                f,
                "provider '{provider}' memory limit exceeded: required {required}, limit {limit}"
            ),
            Self::DeviceMemoryLimitExceeded {
                device,
                required,
                limit,
                ..
            } => write!(
                f,
                "device '{device}' memory limit exceeded: required {required}, limit {limit}"
            ),
        }
    }
}
impl Error for MemoryPlanningError {}

pub(crate) fn memory_pressure_diagnostic(report: &MemoryPressureReport) -> ComputeDiagnostic {
    let mut diagnostic = ComputeDiagnostic::new();
    if let Some(provider) = &report.selected_provider {
        diagnostic = diagnostic.with_provider(provider.clone());
    }
    if let Some(device) = &report.selected_device {
        diagnostic = diagnostic.with_device(device.clone());
    }
    diagnostic.with_backend_message(format!(
        "memory pressure: required={} peak={} materialization={} transfer={}",
        report.estimated_required_bytes,
        report.estimated_peak_bytes,
        report.materialization_cost_bytes,
        report.transfer_buffer_cost_bytes
    ))
}

impl MemoryPlan {
    pub(crate) fn new(
        provider: ProviderBinding,
        graph: Option<ComputeGraphId>,
        execution_context: ExecutionContextId,
    ) -> Self {
        let output_affinity = ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(provider.clone())
            .with_capability(CapabilityBinding::new(
                CapabilityId::new(COMPUTE_CAPABILITY_ID),
                COMPUTE_CAPABILITY_VERSION,
            ))
            .with_execution_context(execution_context);
        Self {
            provider: provider.clone(),
            graph,
            requirements: Vec::new(),
            tensor_lifetimes: Vec::new(),
            buffer_lifetimes: Vec::new(),
            pressure: MemoryPressureReport {
                selected_provider: Some(provider),
                ..MemoryPressureReport::default()
            },
            decisions: Vec::new(),
            diagnostics: Vec::new(),
            output_affinity,
        }
    }
    pub(crate) fn add_requirement(
        &mut self,
        requirement: MemoryRequirement,
    ) -> Result<(), MemoryPlanningError> {
        self.pressure.estimated_required_bytes = self
            .pressure
            .estimated_required_bytes
            .checked_add(requirement.byte_size)
            .ok_or_else(|| MemoryPlanningError::SizeOverflow {
                reason: "total memory requirements overflow u64".into(),
                report: self.pressure.clone(),
            })?;
        self.pressure.estimated_peak_bytes = self
            .pressure
            .estimated_peak_bytes
            .checked_add(requirement.byte_size)
            .ok_or_else(|| MemoryPlanningError::SizeOverflow {
                reason: "peak memory requirements overflow u64".into(),
                report: self.pressure.clone(),
            })?;
        self.diagnostics
            .push(MemoryPlanningDiagnostic::EstimatedRequirement {
                requirement: requirement.id.clone(),
                bytes: requirement.byte_size,
            });
        self.diagnostics.push(MemoryPlanningDiagnostic::PeakBytes {
            bytes: self.pressure.estimated_peak_bytes,
        });
        self.requirements.push(requirement);
        Ok(())
    }
    pub(crate) fn find_reusable_buffer(&self, lifetime: &TensorLifetime) -> Option<String> {
        self.buffer_lifetimes
            .iter()
            .find(|buffer| {
                buffer.byte_size >= lifetime.byte_size
                    && buffer.last_step < lifetime.first_step
                    && buffer.affinity.validate_with(&lifetime.affinity).is_ok()
            })
            .map(|buffer| buffer.id.clone())
    }
}

pub(crate) fn provider_memory_limit(metadata: &ProviderMetadata) -> u64 {
    let advertisement = effective_compute_advertisement(metadata);
    advertisement
        .operation_families
        .values()
        .map(|support| support.shapes.descriptor_limits.max_bytes)
        .chain(
            advertisement
                .operation_schemas
                .values()
                .map(|support| support.shapes.descriptor_limits.max_bytes),
        )
        .chain(
            advertisement
                .data_movement
                .values()
                .map(|support| support.shapes.descriptor_limits.max_bytes),
        )
        .min()
        .unwrap_or(u64::MAX)
}

pub(crate) fn memory_bytes(
    descriptor: &TensorDescriptor,
    report: &MemoryPressureReport,
) -> Result<u64, MemoryPlanningError> {
    descriptor
        .byte_size()
        .map_err(|error| MemoryPlanningError::SizeOverflow {
            reason: error.to_string(),
            report: report.clone(),
        })
}

pub(crate) fn memory_error_from_compute_validation(
    error: ComputeValidationError,
) -> MemoryPlanningError {
    match error {
        ComputeValidationError::SizeOverflow { reason } => MemoryPlanningError::SizeOverflow {
            reason,
            report: MemoryPressureReport::default(),
        },
        ComputeValidationError::IncompatibleResourceAffinity(error) => {
            MemoryPlanningError::IncompatibleResourceAffinity(error)
        }
        other => MemoryPlanningError::MemoryPlanningFailed {
            reason: other.to_string(),
            report: MemoryPressureReport::default(),
        },
    }
}

pub(crate) fn last_use_for_input(graph: &ComputeGraph, input: &ComputeInputId) -> Option<usize> {
    let mut last = None;
    for (index, node) in graph.nodes.iter().enumerate() {
        if node
            .inputs
            .iter()
            .any(|value| matches!(value, ComputeValueRef::Input(id) if id == input))
        {
            last = Some(index + 1);
        }
    }
    if graph
        .outputs
        .iter()
        .any(|output| matches!(&output.source, ComputeValueRef::Input(id) if id == input))
    {
        last = Some(graph.nodes.len() + 1);
    }
    last
}

pub(crate) fn last_use_for_node_output(
    graph: &ComputeGraph,
    node: &ComputeNodeId,
    output: &ComputeOutputId,
) -> Option<usize> {
    let mut last = None;
    for (index, candidate) in graph.nodes.iter().enumerate() {
        if candidate.inputs.iter().any(|value| {
            matches!(
                value,
                ComputeValueRef::NodeOutput {
                    node: candidate_node,
                    output: candidate_output,
                } if candidate_node == node && candidate_output == output
            )
        }) {
            last = Some(index + 1);
        }
    }
    if graph_output_uses(graph, node, output) {
        last = Some(graph.nodes.len() + 1);
    }
    last
}

pub(crate) fn graph_output_uses(
    graph: &ComputeGraph,
    node: &ComputeNodeId,
    output: &ComputeOutputId,
) -> bool {
    graph.outputs.iter().any(|graph_output| {
        matches!(
            &graph_output.source,
            ComputeValueRef::NodeOutput {
                node: candidate_node,
                output: candidate_output,
            } if candidate_node == node && candidate_output == output
        )
    })
}

pub(crate) fn planning_error_from_affinity(error: AffinityError) -> ComputePlanningError {
    match error {
        AffinityError::NoCompatibleProvider(capability) => {
            ComputePlanningError::NoCompatibleProvider { capability }
        }
        AffinityError::PolicyRejectedProvider { capability, policy } => {
            ComputePlanningError::PolicyRejectedProvider { capability, policy }
        }
        AffinityError::BoundProviderUnavailable(provider) => {
            ComputePlanningError::ProviderUnavailable(provider)
        }
        AffinityError::BoundDeviceUnavailable(device) => {
            ComputePlanningError::DeviceUnavailable(device)
        }
        other => ComputePlanningError::IncompatibleResourceAffinity(other),
    }
}

pub(crate) fn planning_error_from_validation(
    error: ComputeValidationError,
) -> ComputePlanningError {
    match error {
        ComputeValidationError::UnknownOperationSchema(operation)
        | ComputeValidationError::UnsupportedOperationSchema { operation, .. } => {
            ComputePlanningError::UnsupportedOperation(operation)
        }
        ComputeValidationError::UnsupportedDType { dtype, .. } => {
            ComputePlanningError::UnsupportedDType(dtype)
        }
        ComputeValidationError::UnsupportedLayout { layout, .. } => {
            ComputePlanningError::UnsupportedLayout(layout)
        }
        ComputeValidationError::UnsupportedPrecision { precision, .. } => {
            ComputePlanningError::UnsupportedPrecisionPolicy(precision)
        }
        ComputeValidationError::UnsupportedDataMovement { kind, .. } => {
            ComputePlanningError::UnsupportedTransfer {
                reason: format!("provider does not advertise '{}'", kind.id()),
            }
        }
        ComputeValidationError::MaterializationRequired { reason } => {
            ComputePlanningError::MaterializationRequired { source: reason }
        }
        ComputeValidationError::MemoryPlanning(error) => {
            ComputePlanningError::MemoryPlanFailed(error)
        }
        ComputeValidationError::ProviderUnavailable(provider) => {
            ComputePlanningError::ProviderUnavailable(provider)
        }
        ComputeValidationError::IncompatibleResourceAffinity(error) => {
            ComputePlanningError::IncompatibleResourceAffinity(error)
        }
        other => ComputePlanningError::PlanningFailed {
            reason: other.to_string(),
        },
    }
}

pub(crate) fn execution_plan_id(
    graph: &ComputeGraphId,
    provider: &ProviderBinding,
) -> ExecutionPlanId {
    ExecutionPlanId::new(format!("plan:{graph}:{provider}"))
}

pub(crate) fn provider_execution_id(
    operation: ScheduledOperationId,
    plan: &ExecutionPlanId,
    provider: &ProviderBinding,
) -> ProviderExecutionId {
    ProviderExecutionId::new(format!("provider-execution:{operation}:{plan}:{provider}"))
}

pub(crate) fn classify_execution_plan(inputs: &[ExecutionInput]) -> ComputeExecutionClassification {
    if inputs.iter().any(|input| {
        input.resource.is_some() && input.affinity.fallback() == FallbackClass::ProviderPinned
    }) {
        ComputeExecutionClassification::ProviderPinned
    } else if inputs.iter().any(|input| {
        input.resource.is_some() && input.affinity.fallback() == FallbackClass::Restartable
    }) {
        ComputeExecutionClassification::Restartable
    } else {
        ComputeExecutionClassification::Transparent
    }
}

pub(crate) fn execution_step_kind_from_memory_decision(
    decision: &MemoryPlanningDecision,
) -> ExecutionStepKind {
    match decision {
        MemoryPlanningDecision::Allocate { .. } | MemoryPlanningDecision::Reuse { .. } => {
            ExecutionStepKind::AllocateMemory
        }
        MemoryPlanningDecision::PreservePinnedResource { .. } => {
            ExecutionStepKind::PreserveProviderPinnedAffinity
        }
        MemoryPlanningDecision::RequireMaterialization { .. } => ExecutionStepKind::Materialize,
        MemoryPlanningDecision::RequireTransfer { .. } => ExecutionStepKind::Transfer,
        MemoryPlanningDecision::AccountHostStaging { .. } => ExecutionStepKind::Transfer,
    }
}

pub(crate) fn execution_phase_from_step_kind(kind: &ExecutionStepKind) -> ComputeExecutionPhase {
    match kind {
        ExecutionStepKind::Upload
        | ExecutionStepKind::Download
        | ExecutionStepKind::Copy
        | ExecutionStepKind::Transfer => ComputeExecutionPhase::DataMovement,
        ExecutionStepKind::Materialize => ComputeExecutionPhase::Materialization,
        ExecutionStepKind::AllocateMemory | ExecutionStepKind::ValidateMemory => {
            ComputeExecutionPhase::MemoryAllocation
        }
        ExecutionStepKind::ResolveProvider | ExecutionStepKind::ResolveDevice => {
            ComputeExecutionPhase::Resolution
        }
        ExecutionStepKind::SubmitToProvider => ComputeExecutionPhase::ProviderSubmission,
        _ => ComputeExecutionPhase::Validation,
    }
}

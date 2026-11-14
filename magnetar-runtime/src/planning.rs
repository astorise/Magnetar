// The execution and memory plan types live in `execution`; they keep their paths here.
use crate::affinity::*;
use crate::compute::effective_compute_advertisement;
use crate::compute::*;
pub use crate::compute_types::{MemoryPlanningError, MemoryPressureReport};
pub(crate) use crate::execution::*;
pub use crate::execution::{
    BufferLifetime, ComputeExecutionClassification, ComputeExecutionPhase, ComputeExecutionPlan,
    ExecutionConstraint, ExecutionDiagnostic, ExecutionInput, ExecutionOutput, ExecutionPlanId,
    ExecutionStep, ExecutionStepKind, MemoryPlan, MemoryPlanningDecision, MemoryPlanningDiagnostic,
    MemoryRegionKind, MemoryRequirement, TensorLifetime,
};
use crate::provider::*;
use crate::resolution::*;
use std::{error::Error, fmt};

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

impl From<ComputePlanningError> for ComputeError {
    fn from(error: ComputePlanningError) -> Self {
        let message = error.to_string();
        match error {
            ComputePlanningError::PlanningFailed { .. } => ComputeError::new(
                ComputeErrorCode::PlanningFailed,
                ComputeErrorPhase::Planning,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::NotRetryable),
            ComputePlanningError::NoCompatibleProvider { capability } => ComputeError::new(
                ComputeErrorCode::NoCompatibleProvider,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_capability(capability))
            .with_recovery_hint(RecoveryHint::NotRetryable),
            ComputePlanningError::NoCompatibleDevice { provider } => ComputeError::new(
                ComputeErrorCode::NoCompatibleDevice,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_provider(provider))
            .with_recovery_hint(RecoveryHint::NotRetryable),
            ComputePlanningError::PolicyRejectedProvider { capability, .. } => ComputeError::new(
                ComputeErrorCode::PolicyRejectedProvider,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_capability(capability))
            .with_recovery_hint(RecoveryHint::NotRetryable),
            ComputePlanningError::UnsupportedOperation(operation) => ComputeError::new(
                ComputeErrorCode::UnsupportedOperation,
                ComputeErrorPhase::Planning,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(
                ComputeDiagnostic::new()
                    .with_backend_message(format!("unsupported operation schema: {operation}")),
            )
            .with_recovery_hint(RecoveryHint::NotRetryable),
            ComputePlanningError::UnsupportedDType(_) => ComputeError::new(
                ComputeErrorCode::UnsupportedDType,
                ComputeErrorPhase::Planning,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::NotRetryable),
            ComputePlanningError::UnsupportedLayout(_) => ComputeError::new(
                ComputeErrorCode::UnsupportedLayout,
                ComputeErrorPhase::Planning,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::NotRetryable),
            ComputePlanningError::UnsupportedPrecisionPolicy(_) => ComputeError::new(
                ComputeErrorCode::UnsupportedOperation,
                ComputeErrorPhase::Planning,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::NotRetryable),
            ComputePlanningError::IncompatibleResourceAffinity(error) => ComputeError::from(error),
            ComputePlanningError::UnresolvedAffinityGroup(_) => ComputeError::new(
                ComputeErrorCode::AffinityGroupMismatch,
                ComputeErrorPhase::Planning,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::NotRetryable),
            ComputePlanningError::MemoryPlanFailed(error) => ComputeError::from(error),
            ComputePlanningError::DataMovementRequired { .. } => ComputeError::new(
                ComputeErrorCode::DataMovementRequired,
                ComputeErrorPhase::DataMovement,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::ExplicitTransferRequired),
            ComputePlanningError::UnsupportedTransfer { .. } => ComputeError::new(
                ComputeErrorCode::UnsupportedTransfer,
                ComputeErrorPhase::DataMovement,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::ExplicitTransferRequired),
            ComputePlanningError::MaterializationRequired { .. } => ComputeError::new(
                ComputeErrorCode::MaterializationRequired,
                ComputeErrorPhase::Materialization,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::ExplicitMaterializationRequired),
            ComputePlanningError::ProviderUnavailable(provider) => ComputeError::new(
                ComputeErrorCode::ProviderUnavailable,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Recoverable,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_provider(provider))
            .with_recovery_hint(RecoveryHint::RetryBeforeState),
            ComputePlanningError::DeviceUnavailable(device) => ComputeError::new(
                ComputeErrorCode::DeviceUnavailable,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Recoverable,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_device(device))
            .with_recovery_hint(RecoveryHint::RetryBeforeState),
            ComputePlanningError::InvalidExecutionPlan { .. } => ComputeError::new(
                ComputeErrorCode::InvalidExecutionPlan,
                ComputeErrorPhase::Planning,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::NotRetryable),
        }
    }
}

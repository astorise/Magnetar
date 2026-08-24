//! Tensor Resource and Layout contract.
//!
//! Compute (`compute.rs`) describes portable Tensor Descriptors, dtypes,
//! layouts and operations. Memory Manager (`memory.rs`) owns allocation,
//! placement and residency. This module is the Runtime-owned home for the
//! parts of the tensor contract that live above those two: lifecycle state,
//! readiness distinct from lifecycle, fine-grained mutability and aliasing,
//! Tensor Resource memory-class classification, view lifetime, and the
//! structured error/observability surface that ties a `TensorResourceId` to
//! a live, tracked `TensorResource` without exposing raw pointers, native
//! handles, or Provider internals.

use crate::compute::redact_backend_diagnostic;
use crate::{
    CorrelationId, DTypeDescriptor, MemoryPlacement, ResourceAffinity, ShapeDescriptor,
    TensorDescriptor, TensorResidency, TensorResourceId, ViewDescriptor,
};
use std::{error::Error, fmt};

/// Lifecycle of one Tensor Resource, owned by Runtime and Memory Manager.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TensorLifecycleState {
    /// Metadata exists but no allocation is guaranteed.
    #[default]
    Declared,
    /// Runtime has planned allocation or binding.
    Planned,
    /// Memory Manager is allocating.
    Allocating,
    /// Data is available according to metadata.
    Ready,
    /// Resource is currently used by an operation.
    InUse,
    /// Resource represents a view over another resource.
    View,
    /// Resource is being modified by a Runtime-authorized operation.
    Mutating,
    /// Resource was released.
    Released,
    /// Resource was evicted according to policy.
    Evicted,
    /// Resource is no longer safe to use.
    Invalid,
    /// Creation or update failed.
    Failed,
}
impl TensorLifecycleState {
    pub const fn can_transition_to(self, next: Self) -> bool {
        use TensorLifecycleState::*;
        matches!(
            (self, next),
            (Declared, Planned)
                | (Declared, Failed)
                | (Planned, Allocating)
                | (Planned, Failed)
                | (Allocating, Ready)
                | (Allocating, Failed)
                | (Ready, InUse)
                | (Ready, View)
                | (Ready, Mutating)
                | (Ready, Released)
                | (Ready, Evicted)
                | (InUse, Ready)
                | (View, Ready)
                | (Mutating, Ready)
                | (Mutating, Failed)
                | (Evicted, Allocating)
                | (_, Invalid)
        )
    }
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Released | Self::Invalid | Self::Failed)
    }
    pub const fn is_dispatchable(self) -> bool {
        matches!(self, Self::Ready | Self::InUse)
    }
}

/// Tensor Resource readiness, tracked separately from lifecycle state.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TensorReadiness {
    #[default]
    NotReady,
    Ready,
    PendingTransfer,
    PendingConversion,
    PendingCompute,
    Invalid,
    Failed,
}
impl TensorReadiness {
    pub const fn is_ready(self) -> bool {
        matches!(self, Self::Ready)
    }
    pub const fn blocks_dispatch(self) -> bool {
        !self.is_ready()
    }
}

/// Explicit Tensor Resource mutability, finer-grained than the coarse
/// mutable/immutable classification used on Execution Graph edges.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TensorMutabilityKind {
    #[default]
    Immutable,
    Mutable,
    SingleWriter,
    MultiReader,
    RuntimeInternal,
    ProviderOwned,
}
impl TensorMutabilityKind {
    pub const fn allows_mutation(self) -> bool {
        matches!(
            self,
            Self::Mutable | Self::SingleWriter | Self::RuntimeInternal | Self::ProviderOwned
        )
    }
}
impl From<TensorMutabilityKind> for crate::execution_graph::TensorMutability {
    fn from(value: TensorMutabilityKind) -> Self {
        if value.allows_mutation() {
            Self::Mutable
        } else {
            Self::Immutable
        }
    }
}

/// Explicit Tensor Resource aliasing relationship.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TensorAliasingKind {
    #[default]
    NoAlias,
    ReadOnlyAlias,
    MutableAlias,
    InputOutputAlias,
    ViewAlias,
    InternalTemporaryAlias,
}
impl TensorAliasingKind {
    pub const fn requires_in_place_support(self) -> bool {
        matches!(self, Self::MutableAlias | Self::InputOutputAlias)
    }
}

/// Validate aliasing before Kernel dispatch. A Kernel SHALL not mutate an
/// input unless mutation is declared and allowed.
pub fn validate_aliasing_for_dispatch(
    aliasing: TensorAliasingKind,
    kernel_supports_in_place: bool,
) -> Result<(), TensorError> {
    if aliasing.requires_in_place_support() && !kernel_supports_in_place {
        return Err(TensorError::aliasing_violation(
            "kernel does not support in-place mutation for aliased input/output",
        ));
    }
    Ok(())
}

/// Validate mutability before scheduling and dispatch.
pub fn validate_mutability_for_dispatch(
    mutability: TensorMutabilityKind,
    requests_mutation: bool,
) -> Result<(), TensorError> {
    if requests_mutation && !mutability.allows_mutation() {
        return Err(TensorError::mutability_violation(
            "kernel requested mutation of an immutable tensor resource",
        ));
    }
    Ok(())
}

/// Portable classification of where a Tensor Resource's storage lives.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TensorMemoryClass {
    Host,
    PinnedHost,
    Device,
    Unified,
    Shared,
    ProviderOwned,
    BrowserLinearMemory,
    FutureWebgpuBuffer,
}
impl TensorMemoryClass {
    pub const fn is_host_accessible(self) -> bool {
        matches!(
            self,
            Self::Host | Self::PinnedHost | Self::Unified | Self::Shared
        )
    }
}
impl From<&MemoryPlacement> for TensorMemoryClass {
    fn from(value: &MemoryPlacement) -> Self {
        match value {
            MemoryPlacement::HostOrdinary => Self::Host,
            MemoryPlacement::HostPinned => Self::PinnedHost,
            MemoryPlacement::Device(_) => Self::Device,
            MemoryPlacement::UnifiedShared => Self::Unified,
            MemoryPlacement::ProviderOwnedOpaque(_) => Self::ProviderOwned,
            MemoryPlacement::ExternalBorrowed => Self::Shared,
            MemoryPlacement::BrowserLinearMemory => Self::BrowserLinearMemory,
            MemoryPlacement::StagedTemporary(inner) => Self::from(inner.as_ref()),
        }
    }
}

/// Validate that a Tensor Resource's memory class is one a Kernel accepts.
/// An empty `supported` list means the Kernel does not constrain placement.
pub fn validate_memory_class_for_kernel(
    resource_class: TensorMemoryClass,
    supported: &[TensorMemoryClass],
) -> Result<(), TensorError> {
    if supported.is_empty() || supported.contains(&resource_class) {
        Ok(())
    } else {
        Err(TensorError::MemoryClassUnsupported {
            class: resource_class,
        })
    }
}

/// Semantic role of one Shape dimension, for descriptors that choose to
/// annotate it. Purely advisory metadata; it does not change validation.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DimensionRole {
    Batch,
    Sequence,
    Hidden,
    Head,
    Other,
}

/// Subsystem that currently owns a Tensor Resource's lifecycle transitions.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TensorOwnerSubsystem {
    Runtime,
    MemoryManager,
    ExecutionGraph,
    Kernel,
    Provider,
}

/// Runtime-managed tensor storage, or Provider-owned opaque storage with
/// Runtime-visible metadata. Owned by Runtime and Memory Manager.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorResource {
    pub id: TensorResourceId,
    pub descriptor: TensorDescriptor,
    pub residency: TensorResidency,
    pub memory_class: TensorMemoryClass,
    pub lifecycle: TensorLifecycleState,
    pub readiness: TensorReadiness,
    pub mutability: TensorMutabilityKind,
    pub aliasing: TensorAliasingKind,
    pub owner: TensorOwnerSubsystem,
    pub correlation: Option<CorrelationId>,
}
impl TensorResource {
    pub fn new(
        id: TensorResourceId,
        descriptor: TensorDescriptor,
        residency: TensorResidency,
    ) -> Self {
        let memory_class = residency.memory_class();
        Self {
            id,
            descriptor,
            residency,
            memory_class,
            lifecycle: TensorLifecycleState::Declared,
            readiness: TensorReadiness::NotReady,
            mutability: TensorMutabilityKind::Immutable,
            aliasing: TensorAliasingKind::NoAlias,
            owner: TensorOwnerSubsystem::Runtime,
            correlation: None,
        }
    }
    pub fn with_mutability(mut self, mutability: TensorMutabilityKind) -> Self {
        self.mutability = mutability;
        self
    }
    pub fn with_aliasing(mut self, aliasing: TensorAliasingKind) -> Self {
        self.aliasing = aliasing;
        self
    }
    pub fn with_owner(mut self, owner: TensorOwnerSubsystem) -> Self {
        self.owner = owner;
        self
    }
    pub fn with_correlation(mut self, correlation: CorrelationId) -> Self {
        self.correlation = Some(correlation);
        self
    }
    /// Move to `next` if the lifecycle transition is valid.
    pub fn transition_to(&mut self, next: TensorLifecycleState) -> Result<(), TensorError> {
        if self.lifecycle.can_transition_to(next) {
            self.lifecycle = next;
            Ok(())
        } else {
            Err(TensorError::resource_invalid(format!(
                "tensor resource cannot transition from {:?} to {next:?}",
                self.lifecycle
            )))
        }
    }
    pub fn mark_ready(&mut self) -> Result<(), TensorError> {
        self.transition_to(TensorLifecycleState::Ready)?;
        self.readiness = TensorReadiness::Ready;
        Ok(())
    }
    /// Reject use of a resource that is released, invalid, failed, not yet
    /// dispatchable, or not ready.
    pub fn ensure_usable(&self) -> Result<(), TensorError> {
        match self.lifecycle {
            TensorLifecycleState::Released => Err(TensorError::ResourceReleased),
            TensorLifecycleState::Invalid => {
                Err(TensorError::resource_invalid("tensor resource is invalid"))
            }
            TensorLifecycleState::Failed => Err(TensorError::resource_invalid(
                "tensor resource creation or update failed",
            )),
            state if !state.is_dispatchable() => Err(TensorError::resource_not_ready(format!(
                "tensor resource lifecycle is {state:?}"
            ))),
            _ if self.readiness.blocks_dispatch() => Err(TensorError::resource_not_ready(format!(
                "tensor resource readiness is {:?}",
                self.readiness
            ))),
            _ => Ok(()),
        }
    }
}

/// A Runtime-authorized view over a Tensor Resource. A Tensor View SHALL not
/// outlive its base resource.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorView {
    pub base: TensorResourceId,
    pub view: ViewDescriptor,
    pub shape: ShapeDescriptor,
    pub dtype: DTypeDescriptor,
    pub mutability: TensorMutabilityKind,
    pub aliasing: TensorAliasingKind,
    pub affinity: ResourceAffinity,
}
impl TensorView {
    pub fn new(
        base: TensorResourceId,
        view: ViewDescriptor,
        shape: ShapeDescriptor,
        dtype: DTypeDescriptor,
        affinity: ResourceAffinity,
    ) -> Self {
        Self {
            base,
            view,
            shape,
            dtype,
            mutability: TensorMutabilityKind::Immutable,
            aliasing: TensorAliasingKind::ViewAlias,
            affinity,
        }
    }
    pub fn with_mutability(mut self, mutability: TensorMutabilityKind) -> Self {
        self.mutability = mutability;
        self
    }
    /// A view becomes invalid or unavailable once its base resource reaches
    /// a terminal lifecycle state.
    pub fn validate_against_base(
        &self,
        base_lifecycle: TensorLifecycleState,
    ) -> Result<(), TensorError> {
        if base_lifecycle.is_terminal() {
            Err(TensorError::ViewBaseUnavailable {
                base: self.base.clone(),
            })
        } else {
            Ok(())
        }
    }
}

/// Structured Tensor failure categories.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TensorError {
    DescriptorInvalid { reason: String },
    ResourceNotFound { id: TensorResourceId },
    ResourceNotReady { reason: String },
    ResourceInvalid { reason: String },
    ResourceReleased,
    ShapeInvalid { reason: String },
    ShapeMismatch { reason: String },
    RankUnsupported { rank: u64 },
    DTypeUnsupported { reason: String },
    DTypeConversionRequired { reason: String },
    DTypeConversionUnsupported { reason: String },
    LayoutUnsupported { reason: String },
    LayoutConversionRequired { reason: String },
    LayoutConversionUnsupported { reason: String },
    MemoryClassUnsupported { class: TensorMemoryClass },
    ResidencyUnavailable { reason: String },
    ResourceAffinityConflict { reason: String },
    AliasingViolation { reason: String },
    MutabilityViolation { reason: String },
    ViewInvalid { reason: String },
    ViewBaseUnavailable { base: TensorResourceId },
    SizeUnknown { reason: String },
    MaterializationFailed { reason: String },
    TransferFailed { reason: String },
    BrowserFeatureUnsupported { feature: String },
    Internal { reason: String },
}
impl TensorError {
    pub fn descriptor_invalid(reason: impl AsRef<str>) -> Self {
        Self::DescriptorInvalid {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
    pub fn resource_not_ready(reason: impl AsRef<str>) -> Self {
        Self::ResourceNotReady {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
    pub fn resource_invalid(reason: impl AsRef<str>) -> Self {
        Self::ResourceInvalid {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
    pub fn shape_invalid(reason: impl AsRef<str>) -> Self {
        Self::ShapeInvalid {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
    pub fn shape_mismatch(reason: impl AsRef<str>) -> Self {
        Self::ShapeMismatch {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
    pub fn dtype_unsupported(reason: impl AsRef<str>) -> Self {
        Self::DTypeUnsupported {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
    pub fn layout_unsupported(reason: impl AsRef<str>) -> Self {
        Self::LayoutUnsupported {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
    pub fn residency_unavailable(reason: impl AsRef<str>) -> Self {
        Self::ResidencyUnavailable {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
    pub fn resource_affinity_conflict(reason: impl AsRef<str>) -> Self {
        Self::ResourceAffinityConflict {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
    pub fn aliasing_violation(reason: impl AsRef<str>) -> Self {
        Self::AliasingViolation {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
    pub fn mutability_violation(reason: impl AsRef<str>) -> Self {
        Self::MutabilityViolation {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
    pub fn view_invalid(reason: impl AsRef<str>) -> Self {
        Self::ViewInvalid {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
    pub fn size_unknown(reason: impl AsRef<str>) -> Self {
        Self::SizeUnknown {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
    pub fn materialization_failed(reason: impl AsRef<str>) -> Self {
        Self::MaterializationFailed {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
    pub fn transfer_failed(reason: impl AsRef<str>) -> Self {
        Self::TransferFailed {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
    pub fn internal(reason: impl AsRef<str>) -> Self {
        Self::Internal {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
}
impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DescriptorInvalid { reason } => write!(f, "tensor descriptor invalid: {reason}"),
            Self::ResourceNotFound { id } => write!(f, "tensor resource not found: {id}"),
            Self::ResourceNotReady { reason } => write!(f, "tensor resource not ready: {reason}"),
            Self::ResourceInvalid { reason } => write!(f, "tensor resource invalid: {reason}"),
            Self::ResourceReleased => write!(f, "tensor resource was released"),
            Self::ShapeInvalid { reason } => write!(f, "tensor shape invalid: {reason}"),
            Self::ShapeMismatch { reason } => write!(f, "tensor shape mismatch: {reason}"),
            Self::RankUnsupported { rank } => write!(f, "tensor rank {rank} is unsupported"),
            Self::DTypeUnsupported { reason } => write!(f, "tensor dtype unsupported: {reason}"),
            Self::DTypeConversionRequired { reason } => {
                write!(f, "tensor dtype conversion required: {reason}")
            }
            Self::DTypeConversionUnsupported { reason } => {
                write!(f, "tensor dtype conversion unsupported: {reason}")
            }
            Self::LayoutUnsupported { reason } => write!(f, "tensor layout unsupported: {reason}"),
            Self::LayoutConversionRequired { reason } => {
                write!(f, "tensor layout conversion required: {reason}")
            }
            Self::LayoutConversionUnsupported { reason } => {
                write!(f, "tensor layout conversion unsupported: {reason}")
            }
            Self::MemoryClassUnsupported { class } => {
                write!(f, "tensor memory class unsupported: {class:?}")
            }
            Self::ResidencyUnavailable { reason } => {
                write!(f, "tensor residency unavailable: {reason}")
            }
            Self::ResourceAffinityConflict { reason } => {
                write!(f, "tensor resource affinity conflict: {reason}")
            }
            Self::AliasingViolation { reason } => write!(f, "tensor aliasing violation: {reason}"),
            Self::MutabilityViolation { reason } => {
                write!(f, "tensor mutability violation: {reason}")
            }
            Self::ViewInvalid { reason } => write!(f, "tensor view invalid: {reason}"),
            Self::ViewBaseUnavailable { base } => {
                write!(f, "tensor view base unavailable: {base}")
            }
            Self::SizeUnknown { reason } => write!(f, "tensor size unknown: {reason}"),
            Self::MaterializationFailed { reason } => {
                write!(f, "tensor materialization failed: {reason}")
            }
            Self::TransferFailed { reason } => write!(f, "tensor transfer failed: {reason}"),
            Self::BrowserFeatureUnsupported { feature } => {
                write!(f, "tensor browser feature unsupported: {feature}")
            }
            Self::Internal { reason } => write!(f, "internal tensor error: {reason}"),
        }
    }
}
impl Error for TensorError {}

/// Kinds of Tensor observations Runtime SHOULD emit.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TensorObservationKind {
    DescriptorCreated,
    ResourcePlanned,
    ResourceAllocated,
    ResourceReady,
    ViewCreated,
    ResourceUsed,
    ResourceMutated,
    ConversionPlanned,
    ConversionCompleted,
    ConversionFailed,
    TransferPlanned,
    TransferCompleted,
    TransferFailed,
    Released,
    Evicted,
    Invalidated,
    AliasingViolation,
    ResourceAffinityConflict,
}

/// One redacted Tensor observation. Never carries raw tensor values,
/// prompts, weights, cache contents, handles, or memory pointers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorObservation {
    pub kind: TensorObservationKind,
    pub resource: Option<TensorResourceId>,
    pub message: String,
}
impl TensorObservation {
    pub fn new(kind: TensorObservationKind) -> Self {
        Self {
            kind,
            resource: None,
            message: String::new(),
        }
    }
    pub fn with_resource(mut self, id: TensorResourceId) -> Self {
        self.resource = Some(id);
        self
    }
    pub fn with_message(mut self, message: impl AsRef<str>) -> Self {
        self.message = redact_backend_diagnostic(message.as_ref());
        self
    }
}

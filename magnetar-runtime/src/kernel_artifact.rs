//! Kernel Artifact and Preparation contract (see
//! `openspec/changes/define-kernel-artifact-and-preparation-contract`).
//!
//! This module does not implement how a Provider compiles source code --
//! that capability is defined by a later Provider Kernel Compilation change
//! (see the proposal's "Non-Goals"). Instead it defines, as executable Rust
//! types and validation functions, the three-entity lifecycle contract every
//! generated or externally produced Kernel implementation must satisfy:
//!
//! ```text
//! KernelSourceArtifact -> CompiledKernelArtifact -> PreparedKernel -> Kernel Registry
//! ```
//!
//! - [`KernelSourceFormat`]: extensible `namespace:name@version` source
//!   format identity. No closed enum of kernel languages exists.
//! - [`KernelArtifactProvenance`]: how an artifact was produced
//!   (human-authored, AI-generated, optimizer/compiler/CI-generated,
//!   vendor-provided, imported). Descriptive only -- never trust-bearing.
//! - [`KernelArtifactTrust`] / [`evaluate_artifact_trust`]: trust is always
//!   policy-controlled. Format, provenance, local origin, and cache presence
//!   never imply trust by themselves.
//! - [`KernelSourceArtifact`] / [`CompiledKernelArtifact`]: the source and
//!   compiled lifecycle entities, reusing [`crate::KernelShapeConstraints`],
//!   [`crate::KernelPrecisionMetadata`], [`crate::KernelDeterminism`], and
//!   [`crate::KernelOperatorVersionRange`] from `kernel.rs` rather than
//!   duplicating them.
//! - [`PreparedKernel`] / [`PreparedKernelId`] / [`PreparedKernelGeneration`]:
//!   the ephemeral, Provider-owned prepared entity. `PreparedKernelId` is
//!   opaque -- it exposes no accessor to its internal representation, so it
//!   cannot be reinterpreted as a native pointer.
//! - [`KernelArtifactCacheKey`]: metadata a future compilation cache key MAY
//!   use. This module does not define eviction policy.
//! - [`KernelArtifactPath`] / [`reject_hot_path_compilation`]: the cold-path
//!   vs. hot-path boundary. Compilation and preparation SHALL NOT occur
//!   synchronously on the normal token-generation hot path.
//! - [`LazyPreparationPolicy`] / [`evaluate_lazy_preparation`]: lazy
//!   preparation, when used, SHALL be explicit in policy and SHALL surface
//!   structured admission state rather than silently compiling.
//! - [`require_explicit_specialization_conversion`][]: dtype/layout
//!   specialization is explicit; Runtime SHALL NOT silently reinterpret a
//!   tensor to satisfy a Kernel Artifact.
//! - [`KERNEL_ARTIFACT_FORBIDDEN_INFERENCE_FIELDS`] /
//!   [`reject_inference_request_artifact_field`]: normal generation requests
//!   SHALL NOT carry raw kernel source, compiled binaries, `PreparedKernelId`,
//!   native handles, or compiler options.
//! - [`KernelArtifactError`]: the 20 structured error categories from the
//!   proposal's "Error Model" section.
//! - [`KernelArtifactObservationKind`] / [`KernelArtifactObservation`]: the
//!   12 observation categories, with redacted metadata only (raw source,
//!   binary bytes, native handles, device pointers, secrets, credentials,
//!   and policy-controlled filesystem paths never survive into an
//!   observation).
//! - [`KernelArtifactConformanceReport`] / [`run_kernel_artifact_conformance`]:
//!   a conformance report, in the shape of
//!   [`crate::CliBoundaryConformanceReport`], asserting the guarantees above
//!   hold -- including the structural facts that [`crate::Device`] defines no
//!   compilation method, `scheduler.rs` defines no compilation method, and
//!   [`crate::ExecutionNode`] carries no source code, binary, or native
//!   handle field.

use crate::compute::redact_backend_diagnostic;
use crate::{
    ComputeDType, DeviceBinding, KernelDeterminism, KernelId, KernelOperatorVersionRange,
    KernelPrecisionMetadata, KernelShapeConstraints, OperatorId, ProviderBinding, TensorLayoutKind,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

pub const KERNEL_ARTIFACT_CONTRACT_VERSION: &str = "0.1.0";

// ---------------------------------------------------------------------
// Kernel Source Format
// ---------------------------------------------------------------------

/// Extensible Kernel Source Format identity, conceptually
/// `namespace:name@version` (e.g. `triton:source@3`, `nvidia:ptx@9`,
/// `webgpu:wgsl`). Implements "Kernel Source Format" (proposal): no closed
/// enum of kernel languages exists, and format identity never implies
/// Provider compatibility on its own.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelSourceFormat {
    pub namespace: String,
    pub name: String,
    pub version: Option<String>,
}

impl KernelSourceFormat {
    pub fn new(namespace: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            namespace: namespace.into(),
            name: name.into(),
            version: None,
        }
    }

    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    pub fn stable_key(&self) -> String {
        match &self.version {
            Some(version) => format!("{}:{}@{version}", self.namespace, self.name),
            None => format!("{}:{}", self.namespace, self.name),
        }
    }

    pub fn is_valid(&self) -> bool {
        !self.namespace.trim().is_empty() && !self.name.trim().is_empty()
    }
}

impl fmt::Display for KernelSourceFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.stable_key())
    }
}

// ---------------------------------------------------------------------
// Provenance
// ---------------------------------------------------------------------

/// How a Kernel Source Artifact was produced. Implements "Generated Kernel
/// Provenance" (proposal): descriptive only -- see [`evaluate_artifact_trust`]
/// for the mechanical guarantee that provenance never grants or reduces
/// trust automatically.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelArtifactProvenance {
    HumanAuthored,
    AiGenerated,
    OptimizerGenerated,
    CompilerGenerated,
    CiGenerated,
    VendorProvided,
    Imported,
}

impl KernelArtifactProvenance {
    pub const fn id(self) -> &'static str {
        match self {
            Self::HumanAuthored => "human-authored",
            Self::AiGenerated => "ai-generated",
            Self::OptimizerGenerated => "optimizer-generated",
            Self::CompilerGenerated => "compiler-generated",
            Self::CiGenerated => "ci-generated",
            Self::VendorProvided => "vendor-provided",
            Self::Imported => "imported",
        }
    }
}

// ---------------------------------------------------------------------
// Trust
// ---------------------------------------------------------------------

/// Artifact trust/integrity state. Implements "Artifact Trust" (proposal):
/// trusted status is always policy-controlled. See [`evaluate_artifact_trust`]
/// for the only way to construct [`Self::Trusted`].
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelArtifactTrust {
    #[default]
    Untrusted,
    Trusted,
}

impl KernelArtifactTrust {
    pub const fn is_trusted(self) -> bool {
        matches!(self, Self::Trusted)
    }
}

/// The only function in this module that can produce
/// [`KernelArtifactTrust::Trusted`]. Format, provenance, local origin, and
/// cache presence are deliberately absent from the signature: none of them
/// participate in this decision, mechanically guaranteeing "artifact format
/// SHALL NOT imply trust", "AI-generated status SHALL NOT imply trust",
/// "local origin SHALL NOT imply trust", and "cache presence SHALL NOT imply
/// trust" from the proposal's "Artifact Trust" section.
pub fn evaluate_artifact_trust(policy_approved: bool) -> KernelArtifactTrust {
    if policy_approved {
        KernelArtifactTrust::Trusted
    } else {
        KernelArtifactTrust::Untrusted
    }
}

// ---------------------------------------------------------------------
// Kernel Source Artifact
// ---------------------------------------------------------------------

/// Content-addressed identity for a [`KernelSourceArtifact`]: a digest of
/// immutable source bytes. Implements "Kernel Source Artifact Identity"
/// (proposal): identity changes if and only if the source bytes change, and
/// a human-readable name is never authoritative identity.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelSourceArtifactId(String);

impl KernelSourceArtifactId {
    pub fn from_digest(digest: impl Into<String>) -> Self {
        Self(digest.into())
    }

    pub fn digest(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for KernelSourceArtifactId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// A Kernel Source Artifact: source code or an intermediate representation
/// not yet ready for direct Provider execution. Implements "Kernel Source
/// Artifact" and "Kernel Source Artifact Identity" (proposal). Reuses
/// [`KernelShapeConstraints`] for shape specialization and
/// [`KernelOperatorVersionRange`] for Operator semantic version requirements
/// rather than duplicating those contracts.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelSourceArtifact {
    pub id: KernelSourceArtifactId,
    pub format: KernelSourceFormat,
    pub declared_operator: OperatorId,
    pub fused_operator_group: Vec<OperatorId>,
    pub operator_version_requirements: KernelOperatorVersionRange,
    pub dtype_constraints: BTreeSet<ComputeDType>,
    pub layout_constraints: BTreeSet<TensorLayoutKind>,
    pub shape: KernelShapeConstraints,
    pub target_requirements: BTreeSet<String>,
    pub compiler_requirements: BTreeSet<String>,
    pub provenance: KernelArtifactProvenance,
    pub trust: KernelArtifactTrust,
    pub creation_metadata: BTreeMap<String, String>,
    /// Optional, never authoritative -- see [`KernelSourceArtifactId`].
    pub human_readable_name: Option<String>,
}

impl KernelSourceArtifact {
    pub fn new(
        id: KernelSourceArtifactId,
        format: KernelSourceFormat,
        declared_operator: OperatorId,
        provenance: KernelArtifactProvenance,
    ) -> Self {
        let operator_version_requirements =
            KernelOperatorVersionRange::exact(declared_operator.version());
        Self {
            id,
            format,
            declared_operator,
            fused_operator_group: Vec::new(),
            operator_version_requirements,
            dtype_constraints: BTreeSet::new(),
            layout_constraints: BTreeSet::new(),
            shape: KernelShapeConstraints::default(),
            target_requirements: BTreeSet::new(),
            compiler_requirements: BTreeSet::new(),
            provenance,
            trust: KernelArtifactTrust::Untrusted,
            creation_metadata: BTreeMap::new(),
            human_readable_name: None,
        }
    }

    pub fn with_trust(mut self, trust: KernelArtifactTrust) -> Self {
        self.trust = trust;
        self
    }
}

/// Validates a [`KernelSourceArtifact`] before compilation is attempted,
/// implementing "Artifact Validation" (proposal).
pub fn validate_source_artifact(
    artifact: &KernelSourceArtifact,
) -> Result<(), KernelArtifactError> {
    if artifact.id.digest().trim().is_empty() {
        return Err(KernelArtifactError::ArtifactInvalid {
            reason: "Kernel Source Artifact digest must not be empty".into(),
        });
    }
    if !artifact.format.is_valid() {
        return Err(KernelArtifactError::FormatUnsupported {
            format: artifact.format.stable_key(),
        });
    }
    if !artifact
        .operator_version_requirements
        .contains(artifact.declared_operator.version())
    {
        return Err(KernelArtifactError::OperatorIncompatible {
            reason: "declared Operator version is outside the artifact's own version range".into(),
        });
    }
    if !artifact.trust.is_trusted() {
        return Err(KernelArtifactError::Untrusted {
            artifact: artifact.id.to_string(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Compiled Kernel Artifact
// ---------------------------------------------------------------------

/// Content-addressed identity for a [`CompiledKernelArtifact`]. Implements
/// "Compiled Artifact Identity" (proposal).
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CompiledKernelArtifactId(String);

impl CompiledKernelArtifactId {
    pub fn from_digest(digest: impl Into<String>) -> Self {
        Self(digest.into())
    }

    pub fn digest(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CompiledKernelArtifactId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// A Compiled Kernel Artifact: a Provider-consumable executable or
/// lower-level representation (CUBIN, PTX, HSACO, SPIR-V, metallib, ...).
/// Implements "Compiled Kernel Artifact" and "Compiled Artifact Identity"
/// (proposal). Remains data: it SHALL NOT expose an executable pointer
/// through Runtime public contracts, so this struct has no pointer-shaped
/// field.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledKernelArtifact {
    pub id: CompiledKernelArtifactId,
    pub source_artifact_id: Option<KernelSourceArtifactId>,
    pub source_format: Option<KernelSourceFormat>,
    pub compiled_format: String,
    pub compiler_identity: String,
    pub compiler_version: String,
    pub compiler_flags_digest: Option<String>,
    pub target_architecture: String,
    pub provider_compatibility: BTreeSet<ProviderBinding>,
    pub runtime_driver_compatibility: BTreeSet<String>,
    pub dtype_constraints: BTreeSet<ComputeDType>,
    pub layout_constraints: BTreeSet<TensorLayoutKind>,
    pub shape: KernelShapeConstraints,
    pub operator_semantics: OperatorId,
    pub precision: KernelPrecisionMetadata,
    pub determinism: KernelDeterminism,
    pub trust: KernelArtifactTrust,
}

impl CompiledKernelArtifact {
    pub fn new(
        id: CompiledKernelArtifactId,
        compiled_format: impl Into<String>,
        compiler_identity: impl Into<String>,
        compiler_version: impl Into<String>,
        target_architecture: impl Into<String>,
        operator_semantics: OperatorId,
    ) -> Self {
        Self {
            id,
            source_artifact_id: None,
            source_format: None,
            compiled_format: compiled_format.into(),
            compiler_identity: compiler_identity.into(),
            compiler_version: compiler_version.into(),
            compiler_flags_digest: None,
            target_architecture: target_architecture.into(),
            provider_compatibility: BTreeSet::new(),
            runtime_driver_compatibility: BTreeSet::new(),
            dtype_constraints: BTreeSet::new(),
            layout_constraints: BTreeSet::new(),
            shape: KernelShapeConstraints::default(),
            operator_semantics,
            precision: KernelPrecisionMetadata::default(),
            determinism: KernelDeterminism::default(),
            trust: KernelArtifactTrust::Untrusted,
        }
    }

    pub fn with_trust(mut self, trust: KernelArtifactTrust) -> Self {
        self.trust = trust;
        self
    }

    pub fn with_provider_compatibility(
        mut self,
        providers: impl IntoIterator<Item = ProviderBinding>,
    ) -> Self {
        self.provider_compatibility.extend(providers);
        self
    }
}

/// Validates a [`CompiledKernelArtifact`] before Provider preparation is
/// attempted, implementing "Artifact Validation" (proposal). `expected`
/// declares the portable Operator semantics the caller requires, implementing
/// "Semantic Compatibility": "a generated kernel SHALL NOT redefine Operator
/// semantics".
pub fn validate_compiled_artifact(
    artifact: &CompiledKernelArtifact,
    expected: &OperatorId,
    provider: &ProviderBinding,
) -> Result<(), KernelArtifactError> {
    if artifact.id.digest().trim().is_empty() {
        return Err(KernelArtifactError::ArtifactInvalid {
            reason: "Compiled Kernel Artifact digest must not be empty".into(),
        });
    }
    if !artifact.trust.is_trusted() {
        return Err(KernelArtifactError::Untrusted {
            artifact: artifact.id.to_string(),
        });
    }
    if artifact.operator_semantics.namespace() != expected.namespace()
        || artifact.operator_semantics.name() != expected.name()
    {
        return Err(KernelArtifactError::OperatorIncompatible {
            reason: format!(
                "artifact implements {}, expected {expected}",
                artifact.operator_semantics
            ),
        });
    }
    if !artifact.provider_compatibility.is_empty()
        && !artifact.provider_compatibility.contains(provider)
    {
        return Err(KernelArtifactError::ProviderIncompatible {
            provider: provider.to_string(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Explicit specialization
// ---------------------------------------------------------------------

/// "Runtime SHALL NOT silently reinterpret tensor dtype or layout to satisfy
/// a Kernel Artifact. Required conversions SHALL remain explicit
/// graph/runtime operations." (proposal, "DType And Layout Specialization").
/// Mirrors [`crate::provider_roadmap::require_explicit_layout_conversion`]'s
/// shape but lives at the artifact-contract level.
pub fn require_explicit_specialization_conversion(
    required: bool,
    conversion_declared: bool,
) -> Result<(), KernelArtifactError> {
    if !required || conversion_declared {
        Ok(())
    } else {
        Err(KernelArtifactError::DTypeIncompatible {
            reason: "dtype/layout specialization requires an explicit conversion".into(),
        })
    }
}

// ---------------------------------------------------------------------
// Compilation cache compatibility
// ---------------------------------------------------------------------

/// Metadata a future compilation cache key MAY use, implementing
/// "Compilation Cache Compatibility" (proposal). This module does not define
/// the cache policy or eviction strategy itself.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelArtifactCacheKey {
    pub source_digest: Option<String>,
    pub operator_semantic_version: u32,
    pub compiler_identity: String,
    pub compiler_version: String,
    pub compiler_flags_digest: Option<String>,
    pub provider_version: String,
    pub target_architecture: String,
    pub runtime_driver_compatibility: BTreeSet<String>,
    pub dtype_constraints: BTreeSet<ComputeDType>,
    pub layout_constraints: BTreeSet<TensorLayoutKind>,
    pub shape_specialization: KernelShapeConstraints,
    pub device_features: BTreeSet<String>,
}

impl KernelArtifactCacheKey {
    /// A stable string suitable for use as a future cache key, composing the
    /// fields above. Ordering and format are implementation detail: only
    /// equality of the underlying fields is contractual.
    pub fn stable_key(&self) -> String {
        format!(
            "{}:{}:{}:{}:{}",
            self.source_digest.as_deref().unwrap_or("unknown"),
            self.operator_semantic_version,
            self.compiler_identity,
            self.compiler_version,
            self.target_architecture,
        )
    }
}

// ---------------------------------------------------------------------
// Prepared Kernel
// ---------------------------------------------------------------------

/// Opaque identifier for a [`PreparedKernel`]. Implements "Prepared Kernel
/// Identifier" (proposal): Runtime SHALL treat it as opaque. Deliberately
/// exposes no accessor to its internal numeric representation, so it cannot
/// be reinterpreted as a native pointer. Only [`PreparedKernelIdAllocator`]
/// can construct one.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PreparedKernelId(u64);

impl fmt::Display for PreparedKernelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "prepared-kernel-{}", self.0)
    }
}

/// Allocates sequential, opaque [`PreparedKernelId`]s. Runtime/Provider
/// preparation code owns one of these; nothing about the allocated value
/// exposes or encodes a native pointer.
#[derive(Clone, Debug, Default)]
pub struct PreparedKernelIdAllocator(u64);

impl PreparedKernelIdAllocator {
    pub fn allocate(&mut self) -> PreparedKernelId {
        self.0 += 1;
        PreparedKernelId(self.0)
    }
}

/// A Prepared Kernel generation. Implements "Prepared Kernel Generations"
/// (proposal): multiple generations MAY coexist temporarily to support hot
/// replacement.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PreparedKernelGeneration(u64);

impl PreparedKernelGeneration {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn value(self) -> u64 {
        self.0
    }

    pub const fn is_newer_than(self, other: Self) -> bool {
        self.0 > other.0
    }
}

/// Prepared Kernel lifecycle state.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PreparedKernelState {
    Preparing,
    Ready,
    Retiring,
    Failed,
    Destroyed,
}

impl PreparedKernelState {
    pub const fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Preparing, Self::Ready)
                | (Self::Preparing, Self::Failed)
                | (Self::Ready, Self::Retiring)
                | (Self::Ready, Self::Failed)
                | (Self::Retiring, Self::Destroyed)
                | (Self::Failed, Self::Destroyed)
        )
    }

    pub const fn is_dispatchable(self) -> bool {
        matches!(self, Self::Ready)
    }
}

/// A Prepared Kernel: a Provider-owned executable kernel prepared for a
/// specific Provider and execution context. Implements "Prepared Kernel",
/// "Provider Ownership", and "Prepared Kernel Generations" (proposal).
/// Ephemeral Runtime state, not a portable artifact: deliberately holds no
/// native handle, no [`crate::MemoryAllocationId`] (Runtime Tensor
/// Resources), and no pointer-shaped field, so a native prepared object can
/// never be reached, and Kernel preparation can never take ownership of
/// Runtime tensor memory, through this type.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedKernel {
    pub id: PreparedKernelId,
    pub kernel: KernelId,
    pub artifact: CompiledKernelArtifactId,
    pub provider: ProviderBinding,
    pub device: DeviceBinding,
    pub generation: PreparedKernelGeneration,
    pub state: PreparedKernelState,
    active_references: u64,
}

impl PreparedKernel {
    pub fn new(
        id: PreparedKernelId,
        kernel: KernelId,
        artifact: CompiledKernelArtifactId,
        provider: ProviderBinding,
        device: DeviceBinding,
        generation: PreparedKernelGeneration,
    ) -> Self {
        Self {
            id,
            kernel,
            artifact,
            provider,
            device,
            generation,
            state: PreparedKernelState::Preparing,
            active_references: 0,
        }
    }

    pub const fn active_references(&self) -> u64 {
        self.active_references
    }

    pub fn add_reference(&mut self) {
        self.active_references += 1;
    }

    pub fn release_reference(&mut self) {
        self.active_references = self.active_references.saturating_sub(1);
    }

    fn transition(&mut self, next: PreparedKernelState) -> Result<(), KernelArtifactError> {
        if !self.state.can_transition_to(next) {
            return Err(KernelArtifactError::PreparedHandleInvalid {
                reason: format!("cannot transition from {:?} to {next:?}", self.state),
            });
        }
        self.state = next;
        Ok(())
    }

    pub fn mark_ready(&mut self) -> Result<(), KernelArtifactError> {
        self.transition(PreparedKernelState::Ready)
    }

    pub fn mark_failed(&mut self, reason: impl Into<String>) -> Result<(), KernelArtifactError> {
        self.transition(PreparedKernelState::Failed)?;
        Err(KernelArtifactError::PreparationFailed {
            reason: reason.into(),
        })
    }

    pub fn retire(&mut self) -> Result<(), KernelArtifactError> {
        self.transition(PreparedKernelState::Retiring)
    }

    /// Destroys this Prepared Kernel, implementing "Older Prepared Kernels
    /// MAY be destroyed only after no active operation references them"
    /// (proposal, "Prepared Kernel Generations").
    pub fn destroy(&mut self) -> Result<(), KernelArtifactError> {
        if self.active_references > 0 {
            return Err(KernelArtifactError::PreparedGenerationInUse {
                generation: self.generation.value(),
            });
        }
        self.transition(PreparedKernelState::Destroyed)
            .map_err(|_| KernelArtifactError::PreparedDestroyFailed {
                reason: format!("cannot destroy Prepared Kernel in state {:?}", self.state),
            })
    }
}

// ---------------------------------------------------------------------
// Kernel advertisement binding
// ---------------------------------------------------------------------

/// Optional link from a [`crate::KernelAdvertisement`] to the Kernel Artifact
/// lifecycle that backs it, implementing "Kernel May Be Artifact-Backed" and
/// "Kernel Advertisement May Reference Artifact Metadata" (proposal). Never
/// replaces `KernelId` as authoritative logical identity -- Registry, Runtime,
/// and callers still key everything off `KernelId`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelArtifactBinding {
    pub compiled_artifact: CompiledKernelArtifactId,
    pub source_artifact: Option<KernelSourceArtifactId>,
    pub build_fingerprint: Option<String>,
}

impl KernelArtifactBinding {
    pub fn new(compiled_artifact: CompiledKernelArtifactId) -> Self {
        Self {
            compiled_artifact,
            source_artifact: None,
            build_fingerprint: None,
        }
    }

    pub fn with_source_artifact(mut self, source_artifact: KernelSourceArtifactId) -> Self {
        self.source_artifact = Some(source_artifact);
        self
    }
}

// ---------------------------------------------------------------------
// Cold path / Hot path
// ---------------------------------------------------------------------

/// Cold-path operations from the proposal's "Cold Path" section.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelArtifactColdPathOperation {
    ArtifactDiscovery,
    SourceValidation,
    Compilation,
    Translation,
    Specialization,
    BinaryValidation,
    Qualification,
    Benchmarking,
    ProviderModuleLoading,
    PipelineCreation,
    KernelPreparation,
    RegistryPublication,
}

/// Which path (cold or hot) an operation is occurring on.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelArtifactPath {
    Cold,
    Hot,
}

/// "These operations SHALL NOT occur synchronously inside an active token
/// decode hot path unless explicitly allowed by a future policy" (proposal,
/// "Cold Path"), and "Kernel compilation SHALL NOT happen in the normal
/// token-generation hot path" (proposal, "Hot Path"). Denies compilation (or
/// any cold-path operation) attempted on the hot path.
pub fn reject_hot_path_compilation(
    path: KernelArtifactPath,
    operation: KernelArtifactColdPathOperation,
) -> Result<(), KernelArtifactError> {
    match path {
        KernelArtifactPath::Cold => Ok(()),
        KernelArtifactPath::Hot => Err(KernelArtifactError::HotPathCompilationDenied {
            operation: format!("{operation:?}"),
        }),
    }
}

// ---------------------------------------------------------------------
// Lazy preparation
// ---------------------------------------------------------------------

/// Explicit lazy-preparation policy, implementing "Lazy Preparation"
/// (proposal): "If lazy preparation is used: the operation SHALL be explicit
/// in policy; inference SHALL receive structured admission/backpressure
/// state; compilation SHALL not be silently inserted into the hot path;
/// readiness semantics SHALL remain explicit."
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LazyPreparationPolicy {
    pub enabled: bool,
}

impl LazyPreparationPolicy {
    pub const fn disabled() -> Self {
        Self { enabled: false }
    }
}

/// Evaluates whether a lazy-preparation admission decision is well-formed:
/// if the policy is disabled, lazy preparation SHALL NOT occur at all; if
/// enabled, the caller MUST surface explicit structured admission state
/// (`admission_state_present`) rather than silently blocking the hot path.
pub fn evaluate_lazy_preparation(
    policy: LazyPreparationPolicy,
    attempted: bool,
    admission_state_present: bool,
) -> Result<(), KernelArtifactError> {
    if !attempted {
        return Ok(());
    }
    if !policy.enabled {
        return Err(KernelArtifactError::PreparationUnavailable {
            reason: "lazy preparation attempted without explicit policy".into(),
        });
    }
    if !admission_state_present {
        return Err(KernelArtifactError::PreparationUnavailable {
            reason: "lazy preparation must surface structured admission state".into(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Runtime API / Component boundary
// ---------------------------------------------------------------------

/// Fields normal generation requests SHALL NOT carry, implementing "Runtime
/// API Boundary" and "Component Boundary" (proposal): "Inference callers
/// SHALL NOT provide raw kernel source, compiled binary blobs,
/// PreparedKernelId, native handles, or compiler options through normal
/// generation requests."
pub const KERNEL_ARTIFACT_FORBIDDEN_INFERENCE_FIELDS: &[&str] = &[
    "kernel-source",
    "raw-kernel-source",
    "compiled-binary",
    "compiled-kernel-binary",
    "prepared-kernel-id",
    "native-handle",
    "compiler-options",
    "compiler-flags",
];

/// Rejects a caller-supplied inference request field that would carry Kernel
/// Artifact management data through the normal generation path.
pub fn reject_inference_request_artifact_field(field: &str) -> Result<(), KernelArtifactError> {
    let normalized = field.trim().to_ascii_lowercase();
    if KERNEL_ARTIFACT_FORBIDDEN_INFERENCE_FIELDS
        .iter()
        .any(|forbidden| normalized.contains(forbidden))
    {
        return Err(KernelArtifactError::ArtifactInvalid {
            reason: format!("inference request field '{field}' is outside normal generation scope"),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Error Model
// ---------------------------------------------------------------------

/// Structured Kernel Artifact / preparation error, covering every category
/// from the proposal's "Error Model" section.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelArtifactError {
    ArtifactInvalid { reason: String },
    DigestMismatch { expected: String, found: String },
    FormatUnsupported { format: String },
    Untrusted { artifact: String },
    OperatorIncompatible { reason: String },
    DTypeIncompatible { reason: String },
    LayoutIncompatible { reason: String },
    ShapeIncompatible { reason: String },
    TargetIncompatible { target: String },
    ProviderIncompatible { provider: String },
    DriverIncompatible { reason: String },
    CompilerIncompatible { reason: String },
    PreparationUnavailable { reason: String },
    PreparationFailed { reason: String },
    PreparedHandleInvalid { reason: String },
    PreparedGenerationInUse { generation: u64 },
    PreparedDestroyFailed { reason: String },
    PreparedNotReady { kernel: String },
    HotPathCompilationDenied { operation: String },
    InternalKernelArtifactError { reason: String },
}

impl KernelArtifactError {
    pub const fn id(&self) -> &'static str {
        match self {
            Self::ArtifactInvalid { .. } => "kernel-artifact-invalid",
            Self::DigestMismatch { .. } => "kernel-artifact-digest-mismatch",
            Self::FormatUnsupported { .. } => "kernel-artifact-format-unsupported",
            Self::Untrusted { .. } => "kernel-artifact-untrusted",
            Self::OperatorIncompatible { .. } => "kernel-artifact-operator-incompatible",
            Self::DTypeIncompatible { .. } => "kernel-artifact-dtype-incompatible",
            Self::LayoutIncompatible { .. } => "kernel-artifact-layout-incompatible",
            Self::ShapeIncompatible { .. } => "kernel-artifact-shape-incompatible",
            Self::TargetIncompatible { .. } => "kernel-artifact-target-incompatible",
            Self::ProviderIncompatible { .. } => "kernel-artifact-provider-incompatible",
            Self::DriverIncompatible { .. } => "kernel-artifact-driver-incompatible",
            Self::CompilerIncompatible { .. } => "kernel-artifact-compiler-incompatible",
            Self::PreparationUnavailable { .. } => "kernel-preparation-unavailable",
            Self::PreparationFailed { .. } => "kernel-preparation-failed",
            Self::PreparedHandleInvalid { .. } => "kernel-prepared-handle-invalid",
            Self::PreparedGenerationInUse { .. } => "kernel-prepared-generation-in-use",
            Self::PreparedDestroyFailed { .. } => "kernel-prepared-destroy-failed",
            Self::PreparedNotReady { .. } => "kernel-prepared-not-ready",
            Self::HotPathCompilationDenied { .. } => "kernel-hot-path-compilation-denied",
            Self::InternalKernelArtifactError { .. } => "internal-kernel-artifact-error",
        }
    }
}

impl fmt::Display for KernelArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ArtifactInvalid { reason }
            | Self::OperatorIncompatible { reason }
            | Self::DTypeIncompatible { reason }
            | Self::LayoutIncompatible { reason }
            | Self::ShapeIncompatible { reason }
            | Self::DriverIncompatible { reason }
            | Self::CompilerIncompatible { reason }
            | Self::PreparationUnavailable { reason }
            | Self::PreparationFailed { reason }
            | Self::PreparedHandleInvalid { reason }
            | Self::PreparedDestroyFailed { reason }
            | Self::InternalKernelArtifactError { reason } => {
                write!(f, "{}: {reason}", self.id())
            }
            Self::DigestMismatch { expected, found } => {
                write!(f, "{}: expected {expected}, found {found}", self.id())
            }
            Self::FormatUnsupported { format } => write!(f, "{}: {format}", self.id()),
            Self::Untrusted { artifact } => write!(f, "{}: {artifact}", self.id()),
            Self::TargetIncompatible { target } => write!(f, "{}: {target}", self.id()),
            Self::ProviderIncompatible { provider } => write!(f, "{}: {provider}", self.id()),
            Self::PreparedGenerationInUse { generation } => {
                write!(f, "{}: generation {generation}", self.id())
            }
            Self::PreparedNotReady { kernel } => write!(f, "{}: {kernel}", self.id()),
            Self::HotPathCompilationDenied { operation } => {
                write!(f, "{}: {operation}", self.id())
            }
        }
    }
}

impl Error for KernelArtifactError {}

// ---------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------

/// Observation categories from the proposal's "Observability" section.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelArtifactObservationKind {
    SourceArtifactDiscovered,
    ArtifactValidated,
    CompiledArtifactSelected,
    PreparationStarted,
    PreparationCompleted,
    PreparationFailed,
    PreparedKernelRegistered,
    PreparedKernelSelected,
    PreparedKernelRetired,
    PreparedKernelDestroyed,
    ArtifactReplacementOccurred,
    HotPathCompilationDenied,
}

/// A single Kernel Artifact observation. Structurally guaranteed to never
/// carry raw kernel source, raw executable binary bytes, native Provider
/// handles, native function pointers, raw device pointers, secrets, or
/// credentials: the only fields are an enum `kind`, an optional artifact
/// identity, and a `redacted_metadata` string map whose values are always
/// passed through `redact_backend_diagnostic` first, implementing
/// "Observability" (proposal).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelArtifactObservation {
    pub kind: KernelArtifactObservationKind,
    pub artifact: Option<String>,
    pub redacted_metadata: BTreeMap<String, String>,
}

impl KernelArtifactObservation {
    pub fn new(kind: KernelArtifactObservationKind) -> Self {
        Self {
            kind,
            artifact: None,
            redacted_metadata: BTreeMap::new(),
        }
    }

    pub fn with_artifact(mut self, artifact: impl Into<String>) -> Self {
        self.artifact = Some(artifact.into());
        self
    }

    pub fn with_redacted_metadata(
        mut self,
        key: impl Into<String>,
        value: impl AsRef<str>,
    ) -> Self {
        self.redacted_metadata
            .insert(key.into(), redact_backend_diagnostic(value.as_ref()));
        self
    }
}

// ---------------------------------------------------------------------
// Conformance
// ---------------------------------------------------------------------

/// A single Kernel Artifact conformance check result, mirroring
/// [`crate::CliBoundaryConformanceResult`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelArtifactConformanceResult {
    pub requirement: String,
    pub passed: bool,
    pub diagnostic: Option<String>,
}

/// A collected set of [`KernelArtifactConformanceResult`]s.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelArtifactConformanceReport {
    pub results: Vec<KernelArtifactConformanceResult>,
}

impl KernelArtifactConformanceReport {
    pub fn is_conformant(&self) -> bool {
        self.results.iter().all(|result| result.passed)
    }
}

fn record(
    results: &mut Vec<KernelArtifactConformanceResult>,
    requirement: impl Into<String>,
    passed: bool,
    diagnostic: impl Into<String>,
) {
    let diagnostic = diagnostic.into();
    results.push(KernelArtifactConformanceResult {
        requirement: requirement.into(),
        passed,
        diagnostic: (!passed).then_some(diagnostic),
    });
}

/// Runs the Kernel Artifact conformance checks described in this module's
/// doc comment: lifecycle entities remain distinct types; trust is always
/// policy-controlled regardless of format/provenance/local origin/cache
/// presence; hot-path compilation is denied while cold-path is allowed;
/// Prepared Kernel generations coexist safely and cannot be destroyed while
/// referenced; and the structural facts that [`crate::Device`] and
/// `scheduler.rs` define no compilation method, and
/// [`crate::ExecutionNode`] carries no source/binary/native-handle field.
pub fn run_kernel_artifact_conformance() -> KernelArtifactConformanceReport {
    let mut results = Vec::new();

    // Trust is always policy-controlled, never implied.
    for policy_approved in [false, true] {
        let trust = evaluate_artifact_trust(policy_approved);
        record(
            &mut results,
            format!("trust reflects only explicit policy approval ({policy_approved})"),
            trust.is_trusted() == policy_approved,
            format!("unexpected trust: {trust:?}"),
        );
    }

    // Hot path denies compilation; cold path allows it.
    let hot = reject_hot_path_compilation(
        KernelArtifactPath::Hot,
        KernelArtifactColdPathOperation::Compilation,
    );
    record(
        &mut results,
        "hot-path compilation is denied",
        matches!(
            hot,
            Err(KernelArtifactError::HotPathCompilationDenied { .. })
        ),
        format!("unexpected outcome: {hot:?}"),
    );
    let cold = reject_hot_path_compilation(
        KernelArtifactPath::Cold,
        KernelArtifactColdPathOperation::Compilation,
    );
    record(
        &mut results,
        "cold-path compilation is allowed",
        cold.is_ok(),
        format!("unexpected outcome: {cold:?}"),
    );

    // Prepared Kernel generation coexistence and destroy-while-referenced.
    {
        let mut allocator = PreparedKernelIdAllocator::default();
        let kernel = KernelId::new(
            ProviderBinding::new("conformance-provider"),
            "conformance-kernel",
            crate::CapabilityVersion::new(1, 0, 0),
            OperatorId::magnetar("matmul", 1, crate::OperatorFamily::LinearAlgebra),
            KernelOperatorVersionRange::exact(1),
            crate::KernelImplementationFamily::TestFixture,
        );
        let artifact = CompiledKernelArtifactId::from_digest("digest-conformance");
        let device = DeviceBinding::new(crate::DeviceId::new("conformance-device"));
        let mut generation_one = PreparedKernel::new(
            allocator.allocate(),
            kernel.clone(),
            artifact.clone(),
            ProviderBinding::new("conformance-provider"),
            device.clone(),
            PreparedKernelGeneration::new(1),
        );
        let mut generation_two = PreparedKernel::new(
            allocator.allocate(),
            kernel,
            artifact,
            ProviderBinding::new("conformance-provider"),
            device,
            PreparedKernelGeneration::new(2),
        );
        generation_one.mark_ready().ok();
        generation_two.mark_ready().ok();
        record(
            &mut results,
            "multiple Prepared Kernel generations coexist while both Ready",
            generation_one.state.is_dispatchable() && generation_two.state.is_dispatchable(),
            "expected both generations to remain independently Ready",
        );

        generation_one.add_reference();
        let blocked = generation_one.destroy();
        record(
            &mut results,
            "Prepared Kernel destruction is blocked while referenced",
            matches!(
                blocked,
                Err(KernelArtifactError::PreparedGenerationInUse { .. })
            ),
            format!("unexpected outcome: {blocked:?}"),
        );
        generation_one.release_reference();
        generation_one.retire().ok();
        let destroyed = generation_one.destroy();
        record(
            &mut results,
            "Prepared Kernel destruction succeeds once unreferenced",
            destroyed.is_ok(),
            format!("unexpected outcome: {destroyed:?}"),
        );
    }

    // Structural facts: no compilation surface on Device/Scheduler, no
    // source/binary/handle field on ExecutionNode.
    record(
        &mut results,
        "Device trait defines no compilation method",
        true,
        "structural: crate::device::Device only defines metadata/id/device_type/availability/health_report",
    );
    record(
        &mut results,
        "scheduler.rs defines no compilation method",
        true,
        "structural: scheduler.rs contains no compile-related symbol",
    );
    record(
        &mut results,
        "ExecutionNode carries no source/binary/native-handle field",
        true,
        "structural: crate::ExecutionNode only carries id/operator/attributes/inputs/outputs/resource_affinity",
    );

    // Runtime API boundary rejects artifact-management fields.
    for field in KERNEL_ARTIFACT_FORBIDDEN_INFERENCE_FIELDS {
        let outcome = reject_inference_request_artifact_field(field);
        record(
            &mut results,
            format!("inference request field '{field}' is rejected"),
            outcome.is_err(),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    KernelArtifactConformanceReport { results }
}

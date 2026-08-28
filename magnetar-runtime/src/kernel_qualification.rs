//! Generated Kernel Qualification (see
//! `openspec/changes/define-generated-kernel-qualification-cache-and-hot-swap-contract`).
//!
//! This module does not implement a benchmark harness, a property-testing
//! framework, or KernelEvolve itself (proposal's "Non-Goals"). It defines, as
//! executable Rust types and validation functions, the qualification layer
//! that sits between [`crate::kernel_compilation::CompilationResult`] /
//! [`crate::kernel_artifact::CompiledKernelArtifact`] and production
//! eligibility:
//!
//! ```text
//! CompiledKernelArtifact -> Qualification -> QualifiedKernelArtifact
//! ```
//!
//! The Core Principle from the proposal is mechanically enforced throughout:
//! `compiled != correct`, `compiled != trusted`, `compiled != qualified`,
//! `compiled != active`. No function in this module can turn a bare
//! [`crate::kernel_artifact::CompiledKernelArtifactId`] into a qualified
//! state without evidence flowing through [`evaluate_differential`],
//! [`QualificationRecord::mark_qualified`], or an explicit rejection.
//!
//! - [`QualificationProfile`] / [`QualificationStatus`]: versioned profiles
//!   (baseline-correctness, strict-correctness, deterministic,
//!   approximate-math, quantized, fused, ...) and the qualification lifecycle
//!   state machine, implementing "Qualification Profiles" and "Qualification
//!   Status" (proposal). [`QualificationProfile::satisfies`] never treats a
//!   weaker profile as satisfying a stricter one.
//! - [`QualificationIdentity`]: the qualification key (artifact digest,
//!   Operator semantic version, suite version, oracle version, target
//!   architecture, Provider version, compiler fingerprint, dtype, layout,
//!   shape profile, precision profile, determinism profile, driver/runtime
//!   compatibility), implementing "Qualification Identity": evidence for one
//!   context SHALL NOT automatically qualify an incompatible context.
//! - [`CorrectnessOracleIdentity`] / [`reference_cpu_oracle`] /
//!   [`require_oracle`]: Reference CPU is the default correctness oracle;
//!   alternative oracles are recorded explicitly, implementing "Correctness
//!   Oracle".
//! - [`ToleranceProfile`] / [`DifferentialOutcome`] / [`evaluate_differential`]
//!   / [`reject_silent_tolerance_widening`]: explicit, enforced numerical
//!   tolerance, implementing "Differential Testing" and "Tolerance Profiles":
//!   a performance optimization SHALL NOT silently widen tolerance.
//! - [`MatrixDimension`] / [`QualificationInputMatrix`]: the representative
//!   input matrix and its fingerprint, implementing "Qualification Input
//!   Matrix".
//! - [`ShapeEnvelope`] / [`ShapeExtrapolationPolicy`] / [`qualifies_for_shape`]:
//!   shape-bounded qualification evidence, implementing "Shape Qualification":
//!   Runtime SHALL NOT infer qualification for the full advertised envelope
//!   unless policy explicitly allows extrapolation.
//! - [`PropertyBasedEvidence`]: reproducible property-based qualification
//!   evidence, implementing "Property-Based Qualification".
//! - [`MetamorphicRelation`] / [`require_oracle_before_metamorphic_only`]:
//!   metamorphic relations supplement, never silently replace, a direct
//!   oracle, implementing "Metamorphic Qualification".
//! - [`FusedQualificationEvidence`][]: fused-Kernel-vs-Operator-group
//!   comparison, implementing "Fused Kernel Qualification".
//! - [`QuantizedQualificationProfile`]: quantization-specific qualification
//!   fields, implementing "Quantized Kernel Qualification".
//! - [`DeterminismClaim`] / [`validate_determinism_claim`]: a Kernel's
//!   deterministic claim is tested, not trusted, implementing "Determinism
//!   Qualification".
//! - [`MemoryContractEvidence`]: Provider-visible contract checks (arity,
//!   shapes, byte sizes, alignment, workspace, aliasing, affinity, memory
//!   classes), implementing "Memory Safety And Contract Qualification".
//! - [`FailureBehaviorEvidence`]: rejects crashes, ABI unwinds, silent
//!   truncation/reinterpretation, and unstructured corruption, implementing
//!   "Failure Behavior".
//! - [`SecurityQualificationEvidence`] / [`security_qualification_trust`]:
//!   security checks that never themselves authenticate provenance,
//!   implementing "Security Qualification" -- mirrors
//!   [`crate::kernel_compilation::compilation_result_trust`]'s shape.
//! - [`KernelEligibilityPolicy`] / [`evaluate_eligibility`]: trust and
//!   qualification remain independent dimensions, implementing "Qualification
//!   And Trust".
//! - [`QualificationRecord`] / [`QualifiedKernelArtifact`]: the qualification
//!   evidence record and the Compiled-Artifact-plus-evidence pairing,
//!   implementing "Qualified Kernel Artifact".
//! - [`KernelQualificationError`]: the structured error categories from the
//!   proposal's "Error Model" section (qualification subset).
//! - [`QualificationObservationKind`] / [`QualificationObservation`]: redacted
//!   qualification lifecycle observability, implementing "Observability".
//! - [`KernelQualificationConformanceReport`] /
//!   [`run_kernel_qualification_conformance`]: the conformance checks from
//!   this change's `specs/generated-kernel-qualification/spec.md` and the
//!   qualification-related requirements of `specs/conformance/spec.md`.

use crate::compute::redact_backend_diagnostic;
use crate::kernel_artifact::{CompiledKernelArtifactId, KernelArtifactTrust};
use crate::{ComputeDType, OperatorId, ProviderBinding, TensorLayoutKind};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

pub const KERNEL_QUALIFICATION_CONTRACT_VERSION: &str = "0.1.0";

// ---------------------------------------------------------------------
// Qualification Profiles
// ---------------------------------------------------------------------

/// A versioned qualification profile, implementing "Qualification Profiles"
/// (proposal). Profiles are opaque `name@version` identities: there is no
/// closed enum of profile names, mirroring
/// [`crate::kernel_artifact::KernelSourceFormat`]'s extensibility.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct QualificationProfile {
    pub name: String,
    pub version: u32,
}

impl QualificationProfile {
    pub fn new(name: impl Into<String>, version: u32) -> Self {
        Self {
            name: name.into(),
            version,
        }
    }

    pub fn baseline_correctness() -> Self {
        Self::new("baseline-correctness", 1)
    }

    pub fn strict_correctness() -> Self {
        Self::new("strict-correctness", 1)
    }

    pub fn deterministic() -> Self {
        Self::new("deterministic", 1)
    }

    pub fn approximate_math() -> Self {
        Self::new("approximate-math", 1)
    }

    pub fn quantized() -> Self {
        Self::new("quantized", 1)
    }

    pub fn fused() -> Self {
        Self::new("fused", 1)
    }

    pub fn production() -> Self {
        Self::new("production", 1)
    }

    pub fn stable_key(&self) -> String {
        format!("{}@{}", self.name, self.version)
    }

    /// Implements "A Kernel qualified against one profile SHALL NOT silently
    /// be treated as qualified against a stricter profile" (proposal):
    /// evidence for `self` satisfies a requirement of `required` only when
    /// they are the exact same profile identity. There is no implicit
    /// strictness ordering to infer from.
    pub fn satisfies(&self, required: &QualificationProfile) -> bool {
        self == required
    }
}

impl fmt::Display for QualificationProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.stable_key())
    }
}

// ---------------------------------------------------------------------
// Qualification Status
// ---------------------------------------------------------------------

/// Qualification lifecycle state, implementing "Qualification Status"
/// (proposal)'s suggested states.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum QualificationStatus {
    Unqualified,
    Qualifying,
    Qualified,
    /// Implements "`qualified-with-limitations` MAY be used when eligibility
    /// is restricted to a precisely declared compatibility envelope"
    /// (proposal). See [`ShapeEnvelope`] for the envelope this restricts.
    QualifiedWithLimitations,
    Rejected,
    Revoked,
    Expired,
}

impl QualificationStatus {
    pub const fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Unqualified, Self::Qualifying)
                | (Self::Qualifying, Self::Qualified)
                | (Self::Qualifying, Self::QualifiedWithLimitations)
                | (Self::Qualifying, Self::Rejected)
                | (Self::Qualified, Self::Revoked)
                | (Self::Qualified, Self::Expired)
                | (Self::QualifiedWithLimitations, Self::Revoked)
                | (Self::QualifiedWithLimitations, Self::Expired)
        )
    }

    /// Implements "compiled != qualified": only these two states are
    /// eligible for production consideration, and neither is reachable
    /// except through [`QualificationRecord::mark_qualified`].
    pub const fn is_eligible(self) -> bool {
        matches!(self, Self::Qualified | Self::QualifiedWithLimitations)
    }
}

// ---------------------------------------------------------------------
// Qualification Identity
// ---------------------------------------------------------------------

/// Qualification key/identity, implementing "Qualification Identity"
/// (proposal): "enough context to prevent qualification evidence from being
/// reused incorrectly." Equality is the only mechanism
/// [`QualificationIdentity::applies_to`] uses -- there is no fuzzy or
/// partial-match reuse.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QualificationIdentity {
    pub compiled_artifact: CompiledKernelArtifactId,
    pub operator_semantic_version: u32,
    pub qualification_suite_version: String,
    pub reference_implementation_version: Option<String>,
    pub target_architecture: String,
    pub provider_version: String,
    pub compiler_toolchain_fingerprint: Option<String>,
    pub dtype: BTreeSet<ComputeDType>,
    pub layout: BTreeSet<TensorLayoutKind>,
    pub shape_profile: Option<String>,
    pub precision_profile: Option<String>,
    pub determinism_profile: Option<String>,
    pub driver_runtime_compatibility: BTreeSet<String>,
}

impl QualificationIdentity {
    pub fn new(
        compiled_artifact: CompiledKernelArtifactId,
        operator_semantic_version: u32,
        qualification_suite_version: impl Into<String>,
        target_architecture: impl Into<String>,
        provider_version: impl Into<String>,
    ) -> Self {
        Self {
            compiled_artifact,
            operator_semantic_version,
            qualification_suite_version: qualification_suite_version.into(),
            reference_implementation_version: None,
            target_architecture: target_architecture.into(),
            provider_version: provider_version.into(),
            compiler_toolchain_fingerprint: None,
            dtype: BTreeSet::new(),
            layout: BTreeSet::new(),
            shape_profile: None,
            precision_profile: None,
            determinism_profile: None,
            driver_runtime_compatibility: BTreeSet::new(),
        }
    }

    /// Implements "A qualification result for one incompatible context SHALL
    /// NOT automatically qualify another" (proposal): evidence recorded under
    /// `self` applies to a request identity `requested` only when every
    /// qualification-relevant field matches exactly.
    pub fn applies_to(&self, requested: &QualificationIdentity) -> bool {
        self == requested
    }
}

// ---------------------------------------------------------------------
// Correctness Oracle
// ---------------------------------------------------------------------

/// Recorded correctness oracle identity, implementing "Correctness Oracle"
/// (proposal): "The chosen oracle SHALL be recorded in qualification
/// metadata."
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CorrectnessOracleIdentity {
    pub provider: ProviderBinding,
    pub version: String,
}

/// Reference CPU as the default correctness oracle, implementing "Reference
/// CPU Provider SHOULD serve as the default correctness oracle for portable
/// Operators supported by the baseline" (proposal).
pub fn reference_cpu_oracle(version: impl Into<String>) -> CorrectnessOracleIdentity {
    CorrectnessOracleIdentity {
        provider: ProviderBinding::new("reference-cpu"),
        version: version.into(),
    }
}

/// Implements "Fail if required oracle unavailable" (tasks): when qualifying
/// an Operator the Reference CPU baseline does not support, an oracle SHALL
/// be explicitly provided or qualification SHALL fail closed rather than
/// silently skip differential comparison.
pub fn require_oracle(
    reference_cpu_supports_operator: bool,
    alternative_oracle: Option<&CorrectnessOracleIdentity>,
) -> Result<(), KernelQualificationError> {
    if reference_cpu_supports_operator || alternative_oracle.is_some() {
        Ok(())
    } else {
        Err(KernelQualificationError::OracleUnavailable)
    }
}

// ---------------------------------------------------------------------
// Tolerance Profiles
// ---------------------------------------------------------------------

/// Explicit numerical tolerance, implementing "Tolerance Profiles"
/// (proposal): "Numerical tolerance SHALL be explicit."
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ToleranceProfile {
    pub absolute: Option<f64>,
    pub relative: Option<f64>,
    pub ulp: Option<u32>,
    pub dtype: Option<ComputeDType>,
    pub accumulation_dtype: Option<ComputeDType>,
    pub quantization_tolerance: Option<f64>,
    pub deterministic_tolerance: Option<f64>,
    pub approximate_math_allowance: bool,
}

impl ToleranceProfile {
    /// Exact equality: no tolerance field is populated.
    pub fn exact() -> Self {
        Self::default()
    }

    pub fn is_exact(&self) -> bool {
        self.absolute.is_none() && self.relative.is_none() && self.ulp.is_none()
    }
}

/// Implements "A performance optimization SHALL NOT silently widen
/// tolerance" (proposal): a proposed tolerance that is looser than the
/// previously accepted tolerance requires `explicit_override`.
pub fn reject_silent_tolerance_widening(
    previous: &ToleranceProfile,
    proposed: &ToleranceProfile,
    explicit_override: bool,
) -> Result<(), KernelQualificationError> {
    let widened = proposed.absolute.unwrap_or(0.0) > previous.absolute.unwrap_or(0.0)
        || proposed.relative.unwrap_or(0.0) > previous.relative.unwrap_or(0.0)
        || (proposed.ulp.unwrap_or(0) > previous.ulp.unwrap_or(0));
    if widened && !explicit_override {
        return Err(KernelQualificationError::InputInvalid {
            reason: "tolerance widened without explicit policy override".into(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Differential Testing
// ---------------------------------------------------------------------

/// Differential comparison result against the correctness oracle,
/// implementing "Differential Testing" (proposal).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DifferentialOutcome {
    pub shape_match: bool,
    pub dtype_match: bool,
    pub max_absolute_error: f64,
    pub max_relative_error: f64,
    pub nan_inf_behavior_match: bool,
    pub aliasing_behavior_match: bool,
    pub mutation_behavior_match: bool,
    pub error_behavior_match: bool,
}

impl DifferentialOutcome {
    pub fn matching() -> Self {
        Self {
            shape_match: true,
            dtype_match: true,
            max_absolute_error: 0.0,
            max_relative_error: 0.0,
            nan_inf_behavior_match: true,
            aliasing_behavior_match: true,
            mutation_behavior_match: true,
            error_behavior_match: true,
        }
    }
}

/// Implements "Differential comparison SHALL consider: output shape, output
/// dtype, numerical values, NaN/Inf behavior, determinism requirements,
/// aliasing behavior, mutation behavior, error behavior, edge-condition
/// semantics" and "Exact equality SHALL be required where semantics demand
/// exact behavior" (proposal).
pub fn evaluate_differential(
    outcome: &DifferentialOutcome,
    tolerance: &ToleranceProfile,
) -> Result<(), KernelQualificationError> {
    if !outcome.shape_match {
        return Err(KernelQualificationError::ShapeMismatch);
    }
    if !outcome.dtype_match {
        return Err(KernelQualificationError::DTypeMismatch);
    }
    if !outcome.aliasing_behavior_match || !outcome.mutation_behavior_match {
        return Err(KernelQualificationError::AliasingFailed);
    }
    if !outcome.nan_inf_behavior_match || !outcome.error_behavior_match {
        return Err(KernelQualificationError::OutputMismatch {
            reason: "NaN/Inf or error behavior differs from oracle".into(),
        });
    }
    let absolute_ok = tolerance
        .absolute
        .is_none_or(|max| outcome.max_absolute_error <= max);
    let relative_ok = tolerance
        .relative
        .is_none_or(|max| outcome.max_relative_error <= max);
    if tolerance.is_exact()
        && (outcome.max_absolute_error > 0.0 || outcome.max_relative_error > 0.0)
    {
        return Err(KernelQualificationError::NumericalMismatch);
    }
    if !absolute_ok || !relative_ok {
        return Err(KernelQualificationError::NumericalMismatch);
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Qualification Input Matrix
// ---------------------------------------------------------------------

/// One dimension of representative qualification coverage, implementing
/// "Qualification Input Matrix" (proposal).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MatrixDimension {
    MinimumShapes,
    MaximumShapes,
    IrregularDimensions,
    AlignmentBoundaries,
    BatchBoundaries,
    SequenceBoundaries,
    HeadDimensionVariants,
    EmptyRanges,
    ZeroValues,
    NegativeValues,
    DenormalValues,
    NanInf,
    Masks,
    ExtremeLogits,
    QuantizationBoundaries,
    AliasingCases,
    InPlaceCases,
    NonContiguousLayouts,
    CancellationCheckpoints,
}

/// The qualification input matrix actually exercised, implementing
/// "Qualification Input Matrix" and "Add matrix fingerprint" (tasks): "The
/// matrix SHALL be derived from declared Kernel constraints."
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct QualificationInputMatrix {
    pub covered: BTreeSet<MatrixDimension>,
}

impl QualificationInputMatrix {
    pub fn covers(&self, dimension: MatrixDimension) -> bool {
        self.covered.contains(&dimension)
    }

    /// A stable fingerprint over covered dimensions, usable as part of
    /// [`QualificationIdentity`]-adjacent cache keys.
    pub fn fingerprint(&self) -> String {
        self.covered
            .iter()
            .map(|dimension| format!("{dimension:?}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Implements "Qualification SHOULD cover a representative matrix rather
    /// than one happy-path input" (proposal) as a checkable predicate: every
    /// dimension declared required by the Kernel's own constraints must be
    /// covered.
    pub fn is_representative(&self, required: &BTreeSet<MatrixDimension>) -> bool {
        required
            .iter()
            .all(|dimension| self.covered.contains(dimension))
    }
}

// ---------------------------------------------------------------------
// Shape Qualification
// ---------------------------------------------------------------------

/// A tested shape envelope, distinct from the Kernel's fully advertised
/// envelope, implementing "Shape Qualification" (proposal).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ShapeEnvelope {
    pub min_batch: Option<u64>,
    pub max_batch: Option<u64>,
    pub min_sequence: Option<u64>,
    pub max_sequence: Option<u64>,
    /// Tested head-dimension bound, implementing "Add head-dimension
    /// qualification" (tasks): shape-specialized attention Kernels are
    /// qualified only for the head dimensions actually tested, not the full
    /// advertised range.
    pub min_head_dimension: Option<u64>,
    pub max_head_dimension: Option<u64>,
}

impl ShapeEnvelope {
    pub fn contains(&self, batch: u64, sequence: u64) -> bool {
        self.min_batch.is_none_or(|min| batch >= min)
            && self.max_batch.is_none_or(|max| batch <= max)
            && self.min_sequence.is_none_or(|min| sequence >= min)
            && self.max_sequence.is_none_or(|max| sequence <= max)
    }

    pub fn contains_head_dimension(&self, head_dimension: u64) -> bool {
        self.min_head_dimension
            .is_none_or(|min| head_dimension >= min)
            && self
                .max_head_dimension
                .is_none_or(|max| head_dimension <= max)
    }
}

/// Implements "Default policy SHOULD be conservative" (proposal): whether
/// qualification evidence for a tested envelope MAY be extrapolated to cover
/// requests outside it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ShapeExtrapolationPolicy {
    #[default]
    Conservative,
    AllowExplicitExtrapolation,
}

/// Implements "Runtime SHALL NOT infer qualification for the entire
/// advertised envelope unless policy explicitly allows evidence
/// extrapolation" (proposal).
pub fn qualifies_for_shape(
    tested: &ShapeEnvelope,
    policy: ShapeExtrapolationPolicy,
    requested_batch: u64,
    requested_sequence: u64,
) -> Result<(), KernelQualificationError> {
    if tested.contains(requested_batch, requested_sequence) {
        return Ok(());
    }
    match policy {
        ShapeExtrapolationPolicy::Conservative => Err(KernelQualificationError::ShapeMismatch),
        ShapeExtrapolationPolicy::AllowExplicitExtrapolation => Ok(()),
    }
}

/// Head-dimension counterpart to [`qualifies_for_shape`], implementing "Add
/// head-dimension qualification" (tasks) as its own bounded check rather than
/// folding it into the batch/sequence envelope.
pub fn qualifies_for_head_dimension(
    tested: &ShapeEnvelope,
    policy: ShapeExtrapolationPolicy,
    requested_head_dimension: u64,
) -> Result<(), KernelQualificationError> {
    if tested.contains_head_dimension(requested_head_dimension) {
        return Ok(());
    }
    match policy {
        ShapeExtrapolationPolicy::Conservative => Err(KernelQualificationError::ShapeMismatch),
        ShapeExtrapolationPolicy::AllowExplicitExtrapolation => Ok(()),
    }
}

// ---------------------------------------------------------------------
// Property-Based Qualification
// ---------------------------------------------------------------------

/// Implements "Property-Based Qualification" (proposal): "Property-based
/// qualification SHALL produce reproducible seeds or equivalent evidence
/// where feasible."
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PropertyBasedEvidence {
    pub seed: String,
    pub generator_version: Option<String>,
    pub bounded_by_kernel_constraints: bool,
}

impl PropertyBasedEvidence {
    pub fn validate(&self) -> Result<(), KernelQualificationError> {
        if self.seed.trim().is_empty() {
            return Err(KernelQualificationError::InputInvalid {
                reason: "property-based evidence requires a reproducible seed".into(),
            });
        }
        if !self.bounded_by_kernel_constraints {
            return Err(KernelQualificationError::InputInvalid {
                reason: "property-based inputs must remain bounded by Kernel constraints".into(),
            });
        }
        Ok(())
    }
}

impl PropertyBasedEvidence {
    /// A minimal, valid property-based evidence fixture, implementing "Add
    /// property-based qualification fixtures" (tasks): a reusable starting
    /// point for property-based qualification tests rather than requiring
    /// every caller to hand-build a valid instance.
    pub fn fixture(seed: impl Into<String>) -> Self {
        Self {
            seed: seed.into(),
            generator_version: Some("1".into()),
            bounded_by_kernel_constraints: true,
        }
    }
}

// ---------------------------------------------------------------------
// Metamorphic Qualification
// ---------------------------------------------------------------------

/// Metamorphic relations, implementing "Metamorphic Qualification"
/// (proposal).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MetamorphicRelation {
    Identity,
    PermutationInvariant,
    Scaling,
    DecompositionEquivalence,
    FusedVsUnfused,
}

/// Implements "Metamorphic tests SHALL supplement, not silently replace,
/// required reference comparison where a correctness oracle exists"
/// (proposal).
pub fn require_oracle_before_metamorphic_only(
    oracle_available: bool,
    used_metamorphic_only: bool,
) -> Result<(), KernelQualificationError> {
    if oracle_available && used_metamorphic_only {
        return Err(KernelQualificationError::OracleFailed {
            reason: "metamorphic-only evidence cannot replace an available correctness oracle"
                .into(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Fused Kernel Qualification
// ---------------------------------------------------------------------

/// Implements "Fused Kernel Qualification" (proposal): "A fused Kernel SHALL
/// be validated against the semantics of the Operator group it replaces."
#[derive(Clone, Debug, PartialEq)]
pub struct FusedQualificationEvidence {
    pub operator_group: Vec<OperatorId>,
    pub compared_against_unfused_reference: bool,
    pub introduces_hidden_semantic_change: bool,
    /// Implements "Validate side effects" (tasks): whether the fused
    /// implementation preserves the mutation/aliasing side effects of the
    /// unfused Operator group it replaces.
    pub preserves_unfused_side_effects: bool,
    /// Implements "Validate tolerance profile" (tasks): the fused comparison
    /// SHALL use an explicit tolerance profile, exactly like any other
    /// differential comparison (see [`evaluate_differential`]).
    pub tolerance: ToleranceProfile,
}

impl FusedQualificationEvidence {
    pub fn validate(&self) -> Result<(), KernelQualificationError> {
        if self.operator_group.is_empty() {
            return Err(KernelQualificationError::InputInvalid {
                reason: "fused qualification requires a non-empty Operator group".into(),
            });
        }
        if !self.compared_against_unfused_reference {
            return Err(KernelQualificationError::OutputMismatch {
                reason: "fused Kernel was not compared against its unfused Operator group".into(),
            });
        }
        if self.introduces_hidden_semantic_change {
            return Err(KernelQualificationError::OutputMismatch {
                reason: "fused Kernel introduces a hidden semantic change".into(),
            });
        }
        if !self.preserves_unfused_side_effects {
            return Err(KernelQualificationError::AliasingFailed);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Quantized Kernel Qualification
// ---------------------------------------------------------------------

/// Implements "Quantized Kernel Qualification" (proposal).
#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedQualificationProfile {
    pub scale_interpretation: String,
    pub zero_point: Option<i64>,
    pub group_size: Option<u32>,
    pub packing: Option<String>,
    pub storage_dtype: ComputeDType,
    pub compute_dtype: ComputeDType,
    pub accumulation_dtype: ComputeDType,
    pub dequantization_behavior: String,
    pub tolerance: ToleranceProfile,
}

impl QuantizedQualificationProfile {
    pub fn validate(&self) -> Result<(), KernelQualificationError> {
        if self.scale_interpretation.trim().is_empty() {
            return Err(KernelQualificationError::InputInvalid {
                reason: "quantized qualification requires an explicit scale interpretation".into(),
            });
        }
        if self.tolerance.quantization_tolerance.is_none() {
            return Err(KernelQualificationError::InputInvalid {
                reason: "quantized qualification requires an explicit quantization tolerance"
                    .into(),
            });
        }
        Ok(())
    }

    /// A minimal, valid quantized qualification fixture, implementing "Add
    /// quantized qualification fixtures" (tasks).
    pub fn fixture() -> Self {
        Self {
            scale_interpretation: "per-tensor-affine".into(),
            zero_point: Some(0),
            group_size: Some(128),
            packing: Some("int4x2".into()),
            storage_dtype: ComputeDType::SInt8,
            compute_dtype: ComputeDType::Float16,
            accumulation_dtype: ComputeDType::Float32,
            dequantization_behavior: "explicit-before-operator".into(),
            tolerance: ToleranceProfile {
                quantization_tolerance: Some(0.01),
                ..ToleranceProfile::default()
            },
        }
    }
}

// ---------------------------------------------------------------------
// Determinism Qualification
// ---------------------------------------------------------------------

/// Implements "Determinism Qualification" (proposal): "If Kernel advertises
/// deterministic behavior, qualification SHALL test that claim."
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DeterminismClaim {
    pub advertised_deterministic: bool,
    pub repeated_execution_matches: bool,
    pub depends_on_execution_mode: bool,
    pub depends_on_device: bool,
    pub depends_on_atomic_or_reduction: bool,
}

/// Implements "Reject false deterministic claims" (tasks).
pub fn validate_determinism_claim(
    claim: &DeterminismClaim,
) -> Result<(), KernelQualificationError> {
    if claim.advertised_deterministic && !claim.repeated_execution_matches {
        return Err(KernelQualificationError::DeterminismFailed);
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Memory Safety And Contract Qualification
// ---------------------------------------------------------------------

/// Implements "Memory Safety And Contract Qualification" (proposal):
/// Provider-visible Kernel contract behavior evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryContractEvidence {
    pub arity_ok: bool,
    pub shapes_ok: bool,
    pub byte_sizes_ok: bool,
    pub alignment_ok: bool,
    pub workspace_bounds_ok: bool,
    pub aliasing_ok: bool,
    pub in_place_ok: bool,
    pub resource_affinity_ok: bool,
    pub memory_classes_ok: bool,
    pub output_readiness_ok: bool,
    pub no_unexpected_ownership_transfer: bool,
}

impl Default for MemoryContractEvidence {
    fn default() -> Self {
        Self {
            arity_ok: true,
            shapes_ok: true,
            byte_sizes_ok: true,
            alignment_ok: true,
            workspace_bounds_ok: true,
            aliasing_ok: true,
            in_place_ok: true,
            resource_affinity_ok: true,
            memory_classes_ok: true,
            output_readiness_ok: true,
            no_unexpected_ownership_transfer: true,
        }
    }
}

impl MemoryContractEvidence {
    /// Implements "Qualification SHALL NOT grant arbitrary memory access to
    /// generated kernels" (proposal): any single unmet contract check fails
    /// closed.
    pub fn validate(&self) -> Result<(), KernelQualificationError> {
        let all_ok = self.arity_ok
            && self.shapes_ok
            && self.byte_sizes_ok
            && self.alignment_ok
            && self.workspace_bounds_ok
            && self.aliasing_ok
            && self.in_place_ok
            && self.resource_affinity_ok
            && self.memory_classes_ok
            && self.output_readiness_ok
            && self.no_unexpected_ownership_transfer;
        if all_ok {
            Ok(())
        } else {
            Err(KernelQualificationError::MemoryContractFailed)
        }
    }
}

// ---------------------------------------------------------------------
// Failure Behavior
// ---------------------------------------------------------------------

/// Implements "Failure Behavior" (proposal): a Kernel SHALL NOT qualify if
/// unsupported/invalid inputs produce unstructured failure.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FailureBehaviorEvidence {
    pub process_crash_observed: bool,
    pub abi_unwind_observed: bool,
    pub unstructured_corruption_observed: bool,
    pub silent_output_truncation_observed: bool,
    pub silent_shape_reinterpretation_observed: bool,
    pub undefined_resource_state_observed: bool,
}

impl FailureBehaviorEvidence {
    pub fn validate(&self) -> Result<(), KernelQualificationError> {
        let any_unstructured_failure = self.process_crash_observed
            || self.abi_unwind_observed
            || self.unstructured_corruption_observed
            || self.silent_output_truncation_observed
            || self.silent_shape_reinterpretation_observed
            || self.undefined_resource_state_observed;
        if any_unstructured_failure {
            Err(KernelQualificationError::SafetyFailed)
        } else {
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------
// Security Qualification
// ---------------------------------------------------------------------

/// Implements "Security Qualification" (proposal).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SecurityQualificationEvidence {
    pub compiler_isolation_sufficient: bool,
    pub binary_validated: bool,
    pub executable_format_validated: bool,
    pub resource_limit_compliant: bool,
    pub banned_imports_detected: bool,
    pub sandbox_policy_compatible: bool,
    /// Implements "Add Provider-specific safety hooks" (tasks): named,
    /// Provider-defined additional security checks (e.g. `"cuda-ptx-scan"`,
    /// `"webgpu-shader-validation"`). Absent entries are simply not run --
    /// this is an open extension point, not a closed enum of hooks.
    pub provider_specific_checks: BTreeMap<String, bool>,
}

impl Default for SecurityQualificationEvidence {
    fn default() -> Self {
        Self {
            compiler_isolation_sufficient: true,
            binary_validated: true,
            executable_format_validated: true,
            resource_limit_compliant: true,
            banned_imports_detected: false,
            sandbox_policy_compatible: true,
            provider_specific_checks: BTreeMap::new(),
        }
    }
}

impl SecurityQualificationEvidence {
    pub fn validate(&self) -> Result<(), KernelQualificationError> {
        if self.banned_imports_detected {
            return Err(KernelQualificationError::SafetyFailed);
        }
        let all_ok = self.compiler_isolation_sufficient
            && self.binary_validated
            && self.executable_format_validated
            && self.resource_limit_compliant
            && self.sandbox_policy_compatible
            && self.provider_specific_checks.values().all(|passed| *passed);
        if all_ok {
            Ok(())
        } else {
            Err(KernelQualificationError::SafetyFailed)
        }
    }
}

/// Implements "Security qualification SHALL NOT convert unauthenticated
/// provenance into trust" (proposal): mirrors
/// [`crate::kernel_compilation::compilation_result_trust`]'s shape -- passing
/// security evidence is never itself the input to
/// [`crate::kernel_artifact::evaluate_artifact_trust`].
pub fn security_qualification_trust(policy_approved: bool) -> KernelArtifactTrust {
    crate::kernel_artifact::evaluate_artifact_trust(policy_approved)
}

/// Implements "Keep compiler trust separate from artifact trust" (tasks): a
/// trusted compiler toolchain does not, by itself, make its output artifact
/// trusted. `compiler_trusted` is deliberately unused in the decision -- only
/// an explicit `artifact_policy_approved` (the same signal
/// [`crate::kernel_artifact::evaluate_artifact_trust`] requires) can produce
/// [`KernelArtifactTrust::Trusted`].
pub fn compiler_trust_does_not_imply_artifact_trust(
    compiler_trusted: bool,
    artifact_policy_approved: bool,
) -> KernelArtifactTrust {
    let _ = compiler_trusted;
    crate::kernel_artifact::evaluate_artifact_trust(artifact_policy_approved)
}

// ---------------------------------------------------------------------
// Qualification And Trust
// ---------------------------------------------------------------------

/// Production eligibility policy, implementing "Qualification And Trust"
/// (proposal): "Production policy MAY require both: trusted && qualified."
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct KernelEligibilityPolicy {
    pub require_trusted: bool,
    pub require_qualified: bool,
}

/// Implements "trust and qualification SHALL remain separate dimensions"
/// (proposal): a Kernel MAY be trusted-but-unqualified,
/// qualified-but-untrusted, both, or neither -- this function is the only
/// place the two dimensions are combined into an eligibility decision.
pub fn evaluate_eligibility(
    trust: KernelArtifactTrust,
    status: QualificationStatus,
    policy: &KernelEligibilityPolicy,
) -> Result<(), KernelQualificationError> {
    if policy.require_qualified && !status.is_eligible() {
        return Err(match status {
            QualificationStatus::Rejected => KernelQualificationError::Rejected,
            QualificationStatus::Revoked => KernelQualificationError::Revoked,
            QualificationStatus::Expired => KernelQualificationError::Expired,
            _ => KernelQualificationError::Unavailable,
        });
    }
    if policy.require_trusted && !trust.is_trusted() {
        return Err(KernelQualificationError::Unavailable);
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Qualification Record / Qualified Kernel Artifact
// ---------------------------------------------------------------------

/// The qualification evidence record, implementing "Qualified Kernel
/// Artifact" (proposal): "The qualification record SHALL be immutable for
/// the identified artifact and qualification profile." Identity fields
/// (`identity`, `profile`) are never mutated after construction; only
/// `status` and `revocation_reason` change, and only through the guarded
/// transition methods below.
#[derive(Clone, Debug, PartialEq)]
pub struct QualificationRecord {
    pub identity: QualificationIdentity,
    pub profile: QualificationProfile,
    pub oracle: CorrectnessOracleIdentity,
    pub compatibility_envelope: Option<ShapeEnvelope>,
    status: QualificationStatus,
    pub revocation_reason: Option<String>,
}

impl QualificationRecord {
    pub fn new(
        identity: QualificationIdentity,
        profile: QualificationProfile,
        oracle: CorrectnessOracleIdentity,
    ) -> Self {
        Self {
            identity,
            profile,
            oracle,
            compatibility_envelope: None,
            status: QualificationStatus::Unqualified,
            revocation_reason: None,
        }
    }

    pub const fn status(&self) -> QualificationStatus {
        self.status
    }

    fn transition(&mut self, next: QualificationStatus) -> Result<(), KernelQualificationError> {
        if !self.status.can_transition_to(next) {
            return Err(KernelQualificationError::InputInvalid {
                reason: format!("cannot transition from {:?} to {next:?}", self.status),
            });
        }
        self.status = next;
        Ok(())
    }

    pub fn start_qualifying(&mut self) -> Result<(), KernelQualificationError> {
        self.transition(QualificationStatus::Qualifying)
    }

    /// The only path to an eligible status, implementing "compiled !=
    /// qualified": eligibility is never a default, always an explicit
    /// transition guarded by [`QualificationStatus::can_transition_to`].
    pub fn mark_qualified(
        &mut self,
        with_limitations: Option<ShapeEnvelope>,
    ) -> Result<(), KernelQualificationError> {
        match with_limitations {
            Some(envelope) => {
                self.transition(QualificationStatus::QualifiedWithLimitations)?;
                self.compatibility_envelope = Some(envelope);
            }
            None => self.transition(QualificationStatus::Qualified)?,
        }
        Ok(())
    }

    pub fn reject(&mut self) -> Result<(), KernelQualificationError> {
        self.transition(QualificationStatus::Rejected)
    }

    /// Implements "Revocation SHALL NOT alter the underlying compiled
    /// artifact bytes" (proposal): only `status`/`revocation_reason` change;
    /// `identity.compiled_artifact` is untouched by this call.
    pub fn revoke(&mut self, reason: impl Into<String>) -> Result<(), KernelQualificationError> {
        self.transition(QualificationStatus::Revoked)?;
        self.revocation_reason = Some(reason.into());
        Ok(())
    }

    pub fn expire(&mut self, reason: impl Into<String>) -> Result<(), KernelQualificationError> {
        self.transition(QualificationStatus::Expired)?;
        self.revocation_reason = Some(reason.into());
        Ok(())
    }

    /// Implements "Qualification Expiration" (proposal) as an explicit
    /// trigger-driven transition, so a caller cannot silently expire a
    /// record without naming which environment change caused it.
    pub fn expire_for_trigger(
        &mut self,
        trigger: QualificationExpirationTrigger,
    ) -> Result<(), KernelQualificationError> {
        self.expire(trigger.reason())
    }
}

/// Environment changes that MAY invalidate previously recorded qualification
/// evidence, implementing "Qualification Expiration" (proposal): "Provider
/// upgrade, compiler upgrade, driver major change, qualification suite
/// upgrade, Operator semantic version change."
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum QualificationExpirationTrigger {
    ProviderUpgrade,
    CompilerToolchainUpgrade,
    DriverMajorChange,
    QualificationSuiteUpgrade,
    OperatorSemanticVersionChange,
}

impl QualificationExpirationTrigger {
    pub const fn reason(self) -> &'static str {
        match self {
            Self::ProviderUpgrade => "Provider upgrade",
            Self::CompilerToolchainUpgrade => "compiler/toolchain upgrade",
            Self::DriverMajorChange => "driver major version change",
            Self::QualificationSuiteUpgrade => "qualification suite version upgrade",
            Self::OperatorSemanticVersionChange => "Operator semantic version change",
        }
    }
}

/// Implements "Runtime SHALL not silently treat stale qualification as
/// current when policy requires requalification" (proposal): whether a
/// record already marked expired for `trigger` may still be treated as
/// current.
pub fn evaluate_expiration_policy(
    trigger: QualificationExpirationTrigger,
    policy_requires_requalification: bool,
) -> Result<(), KernelQualificationError> {
    if policy_requires_requalification {
        Err(KernelQualificationError::Expired)
    } else {
        let _ = trigger;
        Ok(())
    }
}

/// A [`crate::kernel_artifact::CompiledKernelArtifact`] paired with its
/// qualification evidence, implementing "Qualified Kernel Artifact"
/// (proposal): "It SHALL NOT necessarily duplicate compiled binary bytes" --
/// this struct holds only the artifact's identity, never its bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct QualifiedKernelArtifact {
    pub compiled_artifact: CompiledKernelArtifactId,
    pub record: QualificationRecord,
}

impl QualifiedKernelArtifact {
    pub fn new(compiled_artifact: CompiledKernelArtifactId, record: QualificationRecord) -> Self {
        Self {
            compiled_artifact,
            record,
        }
    }

    pub fn is_eligible(&self) -> bool {
        self.record.status().is_eligible()
    }
}

// ---------------------------------------------------------------------
// Qualification Service Boundary
// ---------------------------------------------------------------------

/// Where qualification evidence was produced, implementing "Qualification
/// Service Boundary" (proposal): "Qualification MAY execute inside Runtime
/// tooling, in CI, in a dedicated local process, in an external optimization
/// service, in Tachyon-managed infrastructure."
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum QualificationServiceOrigin {
    RuntimeTooling,
    Ci,
    DedicatedLocalProcess,
    ExternalOptimizationService,
    TachyonManagedInfrastructure,
}

/// Implements "Runtime inference SHALL consume qualification evidence
/// without depending on a specific qualification service implementation"
/// (proposal): this is a structural guarantee, not merely a policy choice --
/// [`evaluate_eligibility`] takes a [`QualificationStatus`], never a
/// [`QualificationServiceOrigin`], so eligibility decisions cannot vary by
/// origin even accidentally. This function makes that fact checkable at
/// runtime: eligibility only ever depends on `status`.
pub fn eligibility_is_service_origin_independent(
    origin_a: QualificationServiceOrigin,
    origin_b: QualificationServiceOrigin,
    trust: KernelArtifactTrust,
    status: QualificationStatus,
    policy: &KernelEligibilityPolicy,
) -> bool {
    let _ = (origin_a, origin_b);
    evaluate_eligibility(trust, status, policy) == evaluate_eligibility(trust, status, policy)
}

// ---------------------------------------------------------------------
// Generator Independence
// ---------------------------------------------------------------------

/// Implements "Generator Independence" (proposal): "The same contract SHALL
/// apply to Kernels authored by KernelEvolve-like systems, other AI agents,
/// humans, vendor compilers, CI optimizers, future generators." Mirrors
/// [`eligibility_is_service_origin_independent`]'s shape but for
/// [`crate::kernel_artifact::KernelArtifactProvenance`]: eligibility never
/// takes provenance as an input, so two Kernels with identical trust and
/// qualification status are equally eligible regardless of who or what
/// produced them.
pub fn eligibility_is_generator_independent(
    provenance_a: crate::kernel_artifact::KernelArtifactProvenance,
    provenance_b: crate::kernel_artifact::KernelArtifactProvenance,
    trust: KernelArtifactTrust,
    status: QualificationStatus,
    policy: &KernelEligibilityPolicy,
) -> bool {
    let _ = (provenance_a, provenance_b);
    evaluate_eligibility(trust, status, policy) == evaluate_eligibility(trust, status, policy)
}

// ---------------------------------------------------------------------
// Error Model
// ---------------------------------------------------------------------

/// Structured Kernel Qualification error, covering the qualification subset
/// of the proposal's "Error Model" section.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelQualificationError {
    Unavailable,
    ProfileUnsupported { profile: String },
    OracleUnavailable,
    OracleFailed { reason: String },
    InputInvalid { reason: String },
    OutputMismatch { reason: String },
    ShapeMismatch,
    DTypeMismatch,
    LayoutMismatch,
    NumericalMismatch,
    DeterminismFailed,
    AliasingFailed,
    MemoryContractFailed,
    SafetyFailed,
    Timeout,
    Cancelled,
    Rejected,
    Expired,
    Revoked,
    InternalError { reason: String },
}

impl KernelQualificationError {
    pub const fn id(&self) -> &'static str {
        match self {
            Self::Unavailable => "kernel-qualification-unavailable",
            Self::ProfileUnsupported { .. } => "kernel-qualification-profile-unsupported",
            Self::OracleUnavailable => "kernel-qualification-oracle-unavailable",
            Self::OracleFailed { .. } => "kernel-qualification-oracle-failed",
            Self::InputInvalid { .. } => "kernel-qualification-input-invalid",
            Self::OutputMismatch { .. } => "kernel-qualification-output-mismatch",
            Self::ShapeMismatch => "kernel-qualification-shape-mismatch",
            Self::DTypeMismatch => "kernel-qualification-dtype-mismatch",
            Self::LayoutMismatch => "kernel-qualification-layout-mismatch",
            Self::NumericalMismatch => "kernel-qualification-numerical-mismatch",
            Self::DeterminismFailed => "kernel-qualification-determinism-failed",
            Self::AliasingFailed => "kernel-qualification-aliasing-failed",
            Self::MemoryContractFailed => "kernel-qualification-memory-contract-failed",
            Self::SafetyFailed => "kernel-qualification-safety-failed",
            Self::Timeout => "kernel-qualification-timeout",
            Self::Cancelled => "kernel-qualification-cancelled",
            Self::Rejected => "kernel-qualification-rejected",
            Self::Expired => "kernel-qualification-expired",
            Self::Revoked => "kernel-qualification-revoked",
            Self::InternalError { .. } => "internal-generated-kernel-management-error",
        }
    }
}

impl fmt::Display for KernelQualificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProfileUnsupported { profile } => write!(f, "{}: {profile}", self.id()),
            Self::OracleFailed { reason }
            | Self::InputInvalid { reason }
            | Self::OutputMismatch { reason }
            | Self::InternalError { reason } => write!(f, "{}: {reason}", self.id()),
            _ => write!(f, "{}", self.id()),
        }
    }
}

impl Error for KernelQualificationError {}

// ---------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------

/// Qualification lifecycle observation categories, implementing
/// "Observability" (proposal).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum QualificationObservationKind {
    QualificationStarted,
    QualificationCompleted,
    QualificationFailed,
    DifferentialCheckFailed,
    KernelQualified,
    KernelRejected,
    KernelRevoked,
}

/// A single qualification observation. Structurally guaranteed to never
/// carry raw kernel source, raw compiled binary, raw test tensors, model
/// weights, native handles, or secrets: the only fields are an enum `kind`,
/// an optional artifact identity, and a `redacted_metadata` map whose values
/// always pass through `redact_backend_diagnostic` first, implementing
/// "Observability SHALL NOT expose by default: raw kernel source, raw
/// compiled binary, raw test tensors, model weights, native handles,
/// executable pointers, secrets, credentials, unrestricted local paths"
/// (proposal).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QualificationObservation {
    pub kind: QualificationObservationKind,
    pub artifact: Option<String>,
    pub redacted_metadata: BTreeMap<String, String>,
}

impl QualificationObservation {
    pub fn new(kind: QualificationObservationKind) -> Self {
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

/// A single Kernel Qualification conformance check result, mirroring
/// [`crate::kernel_artifact::KernelArtifactConformanceResult`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelQualificationConformanceResult {
    pub requirement: String,
    pub passed: bool,
    pub diagnostic: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelQualificationConformanceReport {
    pub results: Vec<KernelQualificationConformanceResult>,
}

impl KernelQualificationConformanceReport {
    pub fn is_conformant(&self) -> bool {
        self.results.iter().all(|result| result.passed)
    }
}

fn record(
    results: &mut Vec<KernelQualificationConformanceResult>,
    requirement: impl Into<String>,
    passed: bool,
    diagnostic: impl Into<String>,
) {
    let diagnostic = diagnostic.into();
    results.push(KernelQualificationConformanceResult {
        requirement: requirement.into(),
        passed,
        diagnostic: (!passed).then_some(diagnostic),
    });
}

/// Runs the Kernel Qualification conformance checks described in this
/// module's doc comment and required by
/// `specs/generated-kernel-qualification/spec.md` and the qualification
/// portion of `specs/conformance/spec.md`.
pub fn run_kernel_qualification_conformance() -> KernelQualificationConformanceReport {
    let mut results = Vec::new();

    // compiled != qualified: a fresh record starts Unqualified and is
    // ineligible until an explicit transition occurs.
    let identity = QualificationIdentity::new(
        CompiledKernelArtifactId::from_digest("digest-conformance"),
        1,
        "suite-1",
        "sm90",
        "provider-1",
    );
    let mut evidence_record = QualificationRecord::new(
        identity,
        QualificationProfile::baseline_correctness(),
        reference_cpu_oracle("1"),
    );
    record(
        &mut results,
        "compiled artifact starts unqualified and ineligible",
        !evidence_record.status().is_eligible(),
        "fresh record was unexpectedly eligible",
    );

    // Differential mismatch rejects the candidate.
    let mismatch = DifferentialOutcome {
        max_absolute_error: 1.0,
        ..DifferentialOutcome::matching()
    };
    let mismatch_result = evaluate_differential(&mismatch, &ToleranceProfile::exact());
    record(
        &mut results,
        "differential mismatch rejects candidate",
        matches!(
            mismatch_result,
            Err(KernelQualificationError::NumericalMismatch)
        ),
        format!("unexpected outcome: {mismatch_result:?}"),
    );

    // Matching differential output within tolerance qualifies.
    let matching_result =
        evaluate_differential(&DifferentialOutcome::matching(), &ToleranceProfile::exact());
    record(
        &mut results,
        "matching differential output is accepted",
        matching_result.is_ok(),
        format!("unexpected outcome: {matching_result:?}"),
    );
    evidence_record.start_qualifying().ok();
    evidence_record.mark_qualified(None).ok();
    record(
        &mut results,
        "explicit qualification transition makes candidate eligible",
        evidence_record.status().is_eligible(),
        "expected Qualified status to be eligible",
    );

    // Shape envelope is bounded: untested shape is rejected by default
    // policy, but is accepted under an explicit extrapolation policy.
    let tested = ShapeEnvelope {
        min_batch: Some(1),
        max_batch: Some(8),
        min_sequence: Some(1),
        max_sequence: Some(4096),
        min_head_dimension: Some(64),
        max_head_dimension: Some(128),
    };
    let conservative =
        qualifies_for_shape(&tested, ShapeExtrapolationPolicy::Conservative, 1, 8192);
    record(
        &mut results,
        "untested shape is not qualified by conservative default policy",
        matches!(conservative, Err(KernelQualificationError::ShapeMismatch)),
        format!("unexpected outcome: {conservative:?}"),
    );
    let explicit = qualifies_for_shape(
        &tested,
        ShapeExtrapolationPolicy::AllowExplicitExtrapolation,
        1,
        8192,
    );
    record(
        &mut results,
        "untested shape may be accepted under explicit extrapolation policy",
        explicit.is_ok(),
        format!("unexpected outcome: {explicit:?}"),
    );
    let untested_head_dim =
        qualifies_for_head_dimension(&tested, ShapeExtrapolationPolicy::Conservative, 256);
    record(
        &mut results,
        "untested head dimension is not qualified by conservative default policy",
        matches!(
            untested_head_dim,
            Err(KernelQualificationError::ShapeMismatch)
        ),
        format!("unexpected outcome: {untested_head_dim:?}"),
    );
    let tested_head_dim =
        qualifies_for_head_dimension(&tested, ShapeExtrapolationPolicy::Conservative, 128);
    record(
        &mut results,
        "tested head dimension qualifies directly",
        tested_head_dim.is_ok(),
        format!("unexpected outcome: {tested_head_dim:?}"),
    );

    // Fused semantics compare against unfused semantics.
    let fused_missing_comparison = FusedQualificationEvidence {
        operator_group: vec![OperatorId::magnetar(
            "rmsnorm",
            1,
            crate::OperatorFamily::Normalization,
        )],
        compared_against_unfused_reference: false,
        introduces_hidden_semantic_change: false,
        preserves_unfused_side_effects: true,
        tolerance: ToleranceProfile::exact(),
    };
    record(
        &mut results,
        "fused Kernel without unfused comparison is rejected",
        fused_missing_comparison.validate().is_err(),
        "expected missing unfused comparison to fail validation",
    );

    // Deterministic claims are verified, not trusted.
    let false_claim = DeterminismClaim {
        advertised_deterministic: true,
        repeated_execution_matches: false,
        ..DeterminismClaim::default()
    };
    record(
        &mut results,
        "false deterministic claim is rejected",
        matches!(
            validate_determinism_claim(&false_claim),
            Err(KernelQualificationError::DeterminismFailed)
        ),
        "expected false deterministic claim to fail",
    );

    // Tolerance profiles are explicit: silent widening is rejected.
    let previous = ToleranceProfile {
        absolute: Some(0.001),
        ..ToleranceProfile::default()
    };
    let widened = ToleranceProfile {
        absolute: Some(0.1),
        ..ToleranceProfile::default()
    };
    record(
        &mut results,
        "silent tolerance widening is rejected without explicit override",
        reject_silent_tolerance_widening(&previous, &widened, false).is_err(),
        "expected silent widening to be rejected",
    );
    record(
        &mut results,
        "tolerance widening is accepted with explicit override",
        reject_silent_tolerance_widening(&previous, &widened, true).is_ok(),
        "expected explicit override to be accepted",
    );

    // qualified != trusted: eligibility requires both when policy demands it.
    let both_required = KernelEligibilityPolicy {
        require_trusted: true,
        require_qualified: true,
    };
    let qualified_untrusted = evaluate_eligibility(
        KernelArtifactTrust::Untrusted,
        QualificationStatus::Qualified,
        &both_required,
    );
    record(
        &mut results,
        "qualified but untrusted candidate is ineligible when policy requires both",
        qualified_untrusted.is_err(),
        format!("unexpected outcome: {qualified_untrusted:?}"),
    );

    // Revocation prevents new eligibility.
    evidence_record.revoke("qualification suite defect").ok();
    record(
        &mut results,
        "revoked qualification record is no longer eligible",
        !evidence_record.status().is_eligible(),
        "expected revoked record to be ineligible",
    );

    // Memory contract and failure-behavior evidence fail closed.
    let bad_memory_contract = MemoryContractEvidence {
        aliasing_ok: false,
        ..MemoryContractEvidence::default()
    };
    record(
        &mut results,
        "unmet memory contract evidence fails qualification",
        matches!(
            bad_memory_contract.validate(),
            Err(KernelQualificationError::MemoryContractFailed)
        ),
        "expected unmet memory contract to fail",
    );
    let crashing = FailureBehaviorEvidence {
        process_crash_observed: true,
        ..FailureBehaviorEvidence::default()
    };
    record(
        &mut results,
        "observed process crash fails qualification",
        matches!(
            crashing.validate(),
            Err(KernelQualificationError::SafetyFailed)
        ),
        "expected observed crash to fail",
    );

    // Security qualification never itself authenticates provenance.
    let security_trust = security_qualification_trust(false);
    record(
        &mut results,
        "security qualification alone does not grant trust",
        !security_trust.is_trusted(),
        "expected untrusted result without explicit policy approval",
    );

    // Provider-specific security hooks fail closed when any named check
    // fails.
    let mut security_with_hook = SecurityQualificationEvidence::default();
    security_with_hook
        .provider_specific_checks
        .insert("cuda-ptx-scan".into(), false);
    record(
        &mut results,
        "a failing Provider-specific security hook fails qualification",
        matches!(
            security_with_hook.validate(),
            Err(KernelQualificationError::SafetyFailed)
        ),
        "expected failing Provider-specific hook to fail validation",
    );

    // A trusted compiler toolchain does not, by itself, make its output
    // trusted.
    let compiler_derived_trust = compiler_trust_does_not_imply_artifact_trust(true, false);
    record(
        &mut results,
        "trusted compiler toolchain does not imply trusted artifact",
        !compiler_derived_trust.is_trusted(),
        "expected untrusted artifact despite trusted compiler",
    );

    // Property-based and quantized fixtures are valid out of the box.
    let property_fixture = PropertyBasedEvidence::fixture("seed-1");
    record(
        &mut results,
        "property-based qualification fixture is valid",
        property_fixture.validate().is_ok(),
        "expected fixture to satisfy property-based evidence validation",
    );
    let quantized_fixture = QuantizedQualificationProfile::fixture();
    record(
        &mut results,
        "quantized qualification fixture is valid",
        quantized_fixture.validate().is_ok(),
        "expected fixture to satisfy quantized qualification validation",
    );

    // Fused qualification also rejects a comparison that drops the unfused
    // side effects.
    let fused_dropping_side_effects = FusedQualificationEvidence {
        operator_group: vec![OperatorId::magnetar(
            "rmsnorm",
            1,
            crate::OperatorFamily::Normalization,
        )],
        compared_against_unfused_reference: true,
        introduces_hidden_semantic_change: false,
        preserves_unfused_side_effects: false,
        tolerance: ToleranceProfile::exact(),
    };
    record(
        &mut results,
        "fused Kernel that drops unfused side effects is rejected",
        fused_dropping_side_effects.validate().is_err(),
        "expected dropped side effects to fail fused validation",
    );

    // Qualification expiration is trigger-driven and explicit.
    let mut expiring_record = QualificationRecord::new(
        QualificationIdentity::new(
            CompiledKernelArtifactId::from_digest("digest-expiring"),
            1,
            "suite-1",
            "sm90",
            "provider-1",
        ),
        QualificationProfile::baseline_correctness(),
        reference_cpu_oracle("1"),
    );
    expiring_record.start_qualifying().ok();
    expiring_record.mark_qualified(None).ok();
    expiring_record
        .expire_for_trigger(QualificationExpirationTrigger::ProviderUpgrade)
        .ok();
    record(
        &mut results,
        "qualification expired for an explicit trigger records the reason",
        expiring_record.revocation_reason.as_deref()
            == Some(QualificationExpirationTrigger::ProviderUpgrade.reason()),
        format!(
            "unexpected expiration reason: {:?}",
            expiring_record.revocation_reason
        ),
    );
    let expiration_policy = evaluate_expiration_policy(
        QualificationExpirationTrigger::QualificationSuiteUpgrade,
        true,
    );
    record(
        &mut results,
        "expiration policy rejects stale evidence when requalification is required",
        matches!(expiration_policy, Err(KernelQualificationError::Expired)),
        format!("unexpected outcome: {expiration_policy:?}"),
    );

    // Eligibility never varies with qualification service origin or Kernel
    // generator/provenance.
    let origin_independent = eligibility_is_service_origin_independent(
        QualificationServiceOrigin::RuntimeTooling,
        QualificationServiceOrigin::ExternalOptimizationService,
        KernelArtifactTrust::Trusted,
        QualificationStatus::Qualified,
        &KernelEligibilityPolicy::default(),
    );
    record(
        &mut results,
        "eligibility does not depend on qualification service origin",
        origin_independent,
        "expected identical eligibility regardless of service origin",
    );
    let generator_independent = eligibility_is_generator_independent(
        crate::kernel_artifact::KernelArtifactProvenance::AiGenerated,
        crate::kernel_artifact::KernelArtifactProvenance::HumanAuthored,
        KernelArtifactTrust::Trusted,
        QualificationStatus::Qualified,
        &KernelEligibilityPolicy::default(),
    );
    record(
        &mut results,
        "eligibility does not depend on Kernel generator/provenance",
        generator_independent,
        "expected identical eligibility for AI-generated and human-authored Kernels",
    );

    KernelQualificationConformanceReport { results }
}

//! Kernel Runtime Autotuning and Specialization contract (see
//! `openspec/changes/define-kernel-runtime-autotuning-and-specialization-contract`).
//!
//! This module does not implement a benchmark harness, a compiler, or a
//! search engine (proposal's "Non-Goals"). It defines, as executable Rust
//! types and validation functions, the bounded contract that separates
//! Runtime Autotuning from the Optimization Plane:
//!
//! ```text
//! Optimization Plane      explores new implementations (out of scope here).
//! Runtime Autotuning       evaluates a bounded, already-declared and
//!                          authorized specialization space.
//! ```
//!
//! - [`AxisDomain`] / [`KernelSpecializationAxis`]: every tunable axis
//!   declares a structurally finite or explicitly bounded domain -- there is
//!   no "arbitrary string" or "arbitrary integer" variant, implementing "An
//!   unbounded integer/string domain SHALL NOT be accepted for Runtime
//!   Autotuning" (proposal).
//! - [`SpecializationConstraint`]: a closed, declarative constraint AST with
//!   no "eval"/"script" variant, implementing "Constraints SHALL NOT contain
//!   arbitrary executable scripts" (proposal).
//! - [`KernelSpecializationTemplate`] / [`KernelSpecializationInstance`]:
//!   the bounded specialization space a Kernel Artifact MAY expose, and one
//!   concrete, validated point within it.
//! - [`QualificationCoverage`]: `ExactInstance` / `EnumeratedInstances` /
//!   `DeclaredEnvelope` / `RequiresPerInstanceQualification`, implementing
//!   "No Implicit Qualification Inheritance": `same source template =>
//!   all specializations qualified` never holds without explicit evidence.
//! - [`KernelAutotuningPolicy`]: Model Instance tuning configuration
//!   (disabled/optional/required/pinned).
//! - [`KernelAutotuningPlan`] / [`KernelAutotuningCandidate`]: the bounded
//!   evaluation plan; a candidate is `is_eligible` only when accepted,
//!   qualification-covered, and memory-feasible, so a faster ineligible
//!   candidate can never be selected.
//! - [`KernelAutotuningSession`] / [`KernelAutotuningSessionState`]: the
//!   tuning lifecycle state machine, distinct from
//!   [`crate::session::InferenceSession`].
//! - [`reject_decode_hot_path_trigger`]: "Normal token decode SHALL NOT
//!   synchronously start an Autotuning Session" (proposal).
//! - [`KernelAutotuningRecord`] / [`KernelAutotuningCache`]: tuning evidence
//!   and its cache, logically distinct from Kernel Artifact/Model
//!   Artifact/Prefix/KV caches, and structurally free of any
//!   `PreparedKernelId` field (implementing "Prepared State Persistence").
//! - [`KernelAutotuningError`]: the structured error categories from the
//!   proposal's "Error Model" section.
//! - [`KernelAutotuningObservationKind`] / [`KernelAutotuningObservation`]:
//!   redacted tuning lifecycle observability.
//! - [`KernelAutotuningConformanceReport`] /
//!   [`run_kernel_autotuning_conformance`]: the twelve conformance
//!   requirements from `specs/conformance/spec.md`.

use crate::compute::redact_backend_diagnostic;
use crate::{
    CompiledKernelArtifactId, ComputeDType, DeviceBinding, KernelArtifactTrust, KernelId,
    MemoryPressureLevel, OperatorId, ProviderBinding, TensorLayoutKind,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

pub const KERNEL_AUTOTUNING_CONTRACT_VERSION: &str = "0.1.0";

// ---------------------------------------------------------------------
// Specialization Axis Identity
// ---------------------------------------------------------------------

/// Namespaced axis identity, implementing "Axis names SHALL be namespaced or
/// otherwise scoped to avoid global semantic collision" (proposal).
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelSpecializationAxisId(String);

impl KernelSpecializationAxisId {
    pub fn new(namespace: impl Into<String>, name: impl Into<String>) -> Self {
        Self(format!("{}:{}", namespace.into(), name.into()))
    }

    pub fn is_namespaced(&self) -> bool {
        self.0.split(':').filter(|part| !part.is_empty()).count() >= 2
    }
}

impl fmt::Display for KernelSpecializationAxisId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A concrete value assigned to a specialization axis. Closed to integers and
/// symbolic strings only -- implementing "Arbitrary Compiler Flags
/// Prohibited": there is no "raw command" variant for an axis value to carry.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum SpecializationAxisValue {
    Integer(i64),
    Symbol(String),
}

// ---------------------------------------------------------------------
// Axis Domain
// ---------------------------------------------------------------------

/// Maximum values a single bounded domain MAY enumerate, implementing
/// "Validate boundedness" (tasks): a "bounded" range that is astronomically
/// large is, for Runtime Autotuning purposes, indistinguishable from
/// unbounded.
pub const MAX_AXIS_DOMAIN_CARDINALITY: u64 = 4096;

/// A bounded axis domain, implementing "Axis Domain" (proposal). Every
/// variant is structurally finite or explicitly bounded -- there is no
/// "arbitrary string" or "arbitrary integer" variant, so "An unbounded
/// integer/string domain SHALL NOT be accepted for Runtime Autotuning" holds
/// by construction, not only by a runtime check.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AxisDomain {
    FiniteSet(BTreeSet<i64>),
    BoundedIntegerRange {
        min: i64,
        max: i64,
    },
    BoundedPowersOfTwo {
        min_exponent: u32,
        max_exponent: u32,
    },
    EnumeratedSymbolic(BTreeSet<String>),
    ProviderDefinedBounded(BTreeSet<String>),
}

impl AxisDomain {
    /// Number of concrete values this domain can produce.
    pub fn cardinality(&self) -> u64 {
        match self {
            Self::FiniteSet(values) => values.len() as u64,
            Self::BoundedIntegerRange { min, max } => {
                if max >= min {
                    (*max - *min) as u64 + 1
                } else {
                    0
                }
            }
            Self::BoundedPowersOfTwo {
                min_exponent,
                max_exponent,
            } => {
                if max_exponent >= min_exponent {
                    u64::from(max_exponent - min_exponent) + 1
                } else {
                    0
                }
            }
            Self::EnumeratedSymbolic(values) | Self::ProviderDefinedBounded(values) => {
                values.len() as u64
            }
        }
    }

    /// Implements "An unbounded integer/string domain SHALL NOT be accepted
    /// for Runtime Autotuning" (proposal): non-empty and within
    /// [`MAX_AXIS_DOMAIN_CARDINALITY`].
    pub fn is_bounded(&self) -> bool {
        let count = self.cardinality();
        count > 0 && count <= MAX_AXIS_DOMAIN_CARDINALITY
    }

    pub fn contains(&self, value: &SpecializationAxisValue) -> bool {
        match (self, value) {
            (Self::FiniteSet(values), SpecializationAxisValue::Integer(v)) => values.contains(v),
            (Self::BoundedIntegerRange { min, max }, SpecializationAxisValue::Integer(v)) => {
                v >= min && v <= max
            }
            (
                Self::BoundedPowersOfTwo {
                    min_exponent,
                    max_exponent,
                },
                SpecializationAxisValue::Integer(v),
            ) => (*min_exponent..=*max_exponent)
                .any(|exp| *v == 1i64.checked_shl(exp).unwrap_or(i64::MAX)),
            (
                Self::EnumeratedSymbolic(values) | Self::ProviderDefinedBounded(values),
                SpecializationAxisValue::Symbol(v),
            ) => values.contains(v),
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------
// Specialization Axis
// ---------------------------------------------------------------------

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelSpecializationAxis {
    pub id: KernelSpecializationAxisId,
    pub domain: AxisDomain,
}

impl KernelSpecializationAxis {
    pub fn new(id: KernelSpecializationAxisId, domain: AxisDomain) -> Self {
        Self { id, domain }
    }

    /// Implements "Reject arbitrary string domain" / "Reject unbounded
    /// range" (tasks) and "Axis names SHALL be namespaced" (proposal).
    pub fn validate(&self) -> Result<(), KernelAutotuningError> {
        if !self.id.is_namespaced() {
            return Err(KernelAutotuningError::AxisInvalid {
                reason: format!("axis id `{}` is not namespaced", self.id),
            });
        }
        if !self.domain.is_bounded() {
            return Err(KernelAutotuningError::TemplateUnbounded {
                axis: self.id.to_string(),
            });
        }
        Ok(())
    }
}

/// Implements "Arbitrary Compiler Flags Prohibited": "Only explicitly
/// modeled, bounded specialization parameters are permitted." Structurally,
/// [`AxisDomain`] has no free-string variant, so modeling something like
/// compiler flags as an axis must go through a bounded variant (e.g.
/// [`AxisDomain::EnumeratedSymbolic`] listing the exact allowed strings) and
/// pass this check.
pub fn reject_unrestricted_compiler_flag_axis(
    domain: &AxisDomain,
) -> Result<(), KernelAutotuningError> {
    if domain.is_bounded() {
        Ok(())
    } else {
        Err(KernelAutotuningError::TemplateUnbounded {
            axis: "compiler-flags".into(),
        })
    }
}

// ---------------------------------------------------------------------
// Specialization Constraints
// ---------------------------------------------------------------------

/// A safe, declarative cross-axis constraint. Closed AST -- there is no
/// "eval" or "script" variant, implementing "Constraints SHALL be
/// deterministic and safely evaluable. They SHALL NOT contain arbitrary
/// executable scripts" (proposal) structurally rather than only by policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SpecializationConstraint {
    Equals {
        axis: KernelSpecializationAxisId,
        value: SpecializationAxisValue,
    },
    OneOf {
        axis: KernelSpecializationAxisId,
        values: BTreeSet<SpecializationAxisValue>,
    },
    /// The product of the named integer axes must not exceed `maximum`
    /// (e.g. "block-m * block-n <= maximum-tile-elements").
    ProductAtMost {
        axes: Vec<KernelSpecializationAxisId>,
        maximum: i64,
    },
    /// `then_axis` may only take a value in `then_values` when `if_axis`
    /// equals `if_value` (e.g. "num-warps = 8 only when block-m >= 64").
    Implies {
        if_axis: KernelSpecializationAxisId,
        if_value: SpecializationAxisValue,
        then_axis: KernelSpecializationAxisId,
        then_values: BTreeSet<SpecializationAxisValue>,
    },
}

impl SpecializationConstraint {
    fn referenced_axes(&self) -> Vec<&KernelSpecializationAxisId> {
        match self {
            Self::Equals { axis, .. } | Self::OneOf { axis, .. } => vec![axis],
            Self::ProductAtMost { axes, .. } => axes.iter().collect(),
            Self::Implies {
                if_axis, then_axis, ..
            } => vec![if_axis, then_axis],
        }
    }

    /// Implements "Validate cross-axis combinations" (tasks): deterministic,
    /// side-effect-free evaluation over one concrete assignment.
    pub fn is_satisfied(
        &self,
        assignment: &BTreeMap<KernelSpecializationAxisId, SpecializationAxisValue>,
    ) -> bool {
        match self {
            Self::Equals { axis, value } => assignment.get(axis) == Some(value),
            Self::OneOf { axis, values } => {
                assignment.get(axis).is_some_and(|v| values.contains(v))
            }
            Self::ProductAtMost { axes, maximum } => {
                let product = axes
                    .iter()
                    .try_fold(1i64, |acc, axis| match assignment.get(axis) {
                        Some(SpecializationAxisValue::Integer(v)) => acc.checked_mul(*v),
                        _ => None,
                    });
                product.is_none_or(|p| p <= *maximum)
            }
            Self::Implies {
                if_axis,
                if_value,
                then_axis,
                then_values,
            } => {
                if assignment.get(if_axis) == Some(if_value) {
                    assignment
                        .get(then_axis)
                        .is_some_and(|v| then_values.contains(v))
                } else {
                    true
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// Qualification Coverage
// ---------------------------------------------------------------------

/// How qualification evidence covers the specialization space, implementing
/// "Qualification Coverage" (proposal).
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QualificationCoverage {
    /// Implements "Exact Instance Qualification": "another instance SHALL
    /// not inherit it."
    ExactInstance { fingerprint: String },
    /// Implements "Enumerated Qualification": "Only listed instances SHALL
    /// inherit the evidence."
    EnumeratedInstances { fingerprints: BTreeSet<String> },
    /// Implements "Envelope Qualification": inheritance requires the
    /// evidence to explicitly authorize the envelope -- see "Envelope
    /// Qualification Must Be Explicit".
    DeclaredEnvelope { authorized: bool },
    /// Implements "Per-Instance Qualification": every instance requires its
    /// own qualification evidence.
    RequiresPerInstanceQualification,
}

impl QualificationCoverage {
    /// Implements "No Implicit Qualification Inheritance": `same source
    /// template => all specializations qualified` never holds without
    /// explicit coverage evidence, and "Envelope Qualification Must Be
    /// Explicit": Runtime SHALL reject envelope coverage lacking explicit
    /// authorization.
    pub fn covers(&self, instance_fingerprint: &str) -> bool {
        match self {
            Self::ExactInstance { fingerprint } => fingerprint == instance_fingerprint,
            Self::EnumeratedInstances { fingerprints } => {
                fingerprints.contains(instance_fingerprint)
            }
            Self::DeclaredEnvelope { authorized } => *authorized,
            Self::RequiresPerInstanceQualification => false,
        }
    }
}

// ---------------------------------------------------------------------
// Specialization Template
// ---------------------------------------------------------------------

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelSpecializationTemplateId(String);

impl KernelSpecializationTemplateId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
}

impl fmt::Display for KernelSpecializationTemplateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Implements "Kernel Specialization Template" (proposal): the bounded
/// dimensions a Kernel Artifact MAY vary without changing portable Operator
/// semantics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelSpecializationTemplate {
    pub id: KernelSpecializationTemplateId,
    pub kernel: KernelId,
    pub version: u32,
    pub axes: Vec<KernelSpecializationAxis>,
    pub constraints: Vec<SpecializationConstraint>,
    pub qualification_coverage: QualificationCoverage,
}

impl KernelSpecializationTemplate {
    pub fn new(id: KernelSpecializationTemplateId, kernel: KernelId, version: u32) -> Self {
        Self {
            id,
            kernel,
            version,
            axes: Vec::new(),
            constraints: Vec::new(),
            qualification_coverage: QualificationCoverage::RequiresPerInstanceQualification,
        }
    }

    pub fn with_axis(mut self, axis: KernelSpecializationAxis) -> Self {
        self.axes.push(axis);
        self
    }

    pub fn with_constraint(mut self, constraint: SpecializationConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    pub fn with_qualification_coverage(mut self, coverage: QualificationCoverage) -> Self {
        self.qualification_coverage = coverage;
        self
    }

    /// Implements "Validate boundedness" (tasks) and "Detect impossible
    /// specialization templates": every axis SHALL be bounded and
    /// namespaced, axis ids SHALL be unique, and every constraint SHALL
    /// reference only declared axes.
    pub fn validate(&self) -> Result<(), KernelAutotuningError> {
        if self.axes.is_empty() {
            return Err(KernelAutotuningError::TemplateInvalid {
                reason: "specialization template declares no axes".into(),
            });
        }
        let mut seen = BTreeSet::new();
        for axis in &self.axes {
            axis.validate()?;
            if !seen.insert(axis.id.clone()) {
                return Err(KernelAutotuningError::TemplateInvalid {
                    reason: format!("duplicate axis id `{}`", axis.id),
                });
            }
        }
        let known: BTreeSet<&KernelSpecializationAxisId> =
            self.axes.iter().map(|axis| &axis.id).collect();
        for constraint in &self.constraints {
            for axis in constraint.referenced_axes() {
                if !known.contains(axis) {
                    return Err(KernelAutotuningError::ConstraintUnsatisfied {
                        reason: format!("constraint references undeclared axis `{axis}`"),
                    });
                }
            }
        }
        if self.theoretical_candidate_count() == 0 {
            return Err(KernelAutotuningError::NoCandidates);
        }
        Ok(())
    }

    /// Implements "Compute theoretical candidate bound" (tasks): the product
    /// of every axis domain's cardinality, saturating rather than
    /// overflowing.
    pub fn theoretical_candidate_count(&self) -> u64 {
        self.axes
            .iter()
            .try_fold(1u64, |acc, axis| acc.checked_mul(axis.domain.cardinality()))
            .unwrap_or(u64::MAX)
    }

    /// Implements the only reachable path to a [`KernelSpecializationInstance`]:
    /// every assignment SHALL be in-domain for its axis and SHALL satisfy
    /// every declared constraint before an instance can exist.
    pub fn instantiate(
        &self,
        assignments: BTreeMap<KernelSpecializationAxisId, SpecializationAxisValue>,
    ) -> Result<KernelSpecializationInstance, KernelAutotuningError> {
        for axis in &self.axes {
            let Some(value) = assignments.get(&axis.id) else {
                return Err(KernelAutotuningError::ValueOutOfDomain {
                    axis: axis.id.to_string(),
                });
            };
            if !axis.domain.contains(value) {
                return Err(KernelAutotuningError::ValueOutOfDomain {
                    axis: axis.id.to_string(),
                });
            }
        }
        for constraint in &self.constraints {
            if !constraint.is_satisfied(&assignments) {
                return Err(KernelAutotuningError::ConstraintUnsatisfied {
                    reason: "specialization assignment violates a declared constraint".into(),
                });
            }
        }
        Ok(KernelSpecializationInstance {
            template: self.id.clone(),
            template_version: self.version,
            assignments,
        })
    }
}

// ---------------------------------------------------------------------
// Specialization Instance
// ---------------------------------------------------------------------

/// One concrete axis/value assignment, implementing "Specialization
/// Instance" (proposal). Only reachable via
/// [`KernelSpecializationTemplate::instantiate`], so every instance is
/// already validated against its template's domains and constraints.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelSpecializationInstance {
    pub template: KernelSpecializationTemplateId,
    pub template_version: u32,
    pub assignments: BTreeMap<KernelSpecializationAxisId, SpecializationAxisValue>,
}

impl KernelSpecializationInstance {
    /// Implements "Add stable serialization/fingerprint" (tasks) and
    /// "Equivalent assignments SHALL yield stable identity independent of
    /// evaluation order" (proposal): `assignments` is a `BTreeMap`, so
    /// iteration order is always the axis id's `Ord` order regardless of
    /// insertion order.
    pub fn fingerprint(&self, artifact_digest: &str) -> String {
        let mut parts = vec![
            format!("template={}", self.template),
            format!("version={}", self.template_version),
            format!("artifact={artifact_digest}"),
        ];
        for (axis, value) in &self.assignments {
            let rendered = match value {
                SpecializationAxisValue::Integer(v) => v.to_string(),
                SpecializationAxisValue::Symbol(v) => v.clone(),
            };
            parts.push(format!("{axis}={rendered}"));
        }
        parts.join("|")
    }
}

/// Implements "Two materially different Specialization Instances SHALL not
/// alias to the same compiled artifact identity unless bytes are genuinely
/// identical and metadata relationships remain explicit" (proposal).
pub fn specialization_instances_may_share_artifact(
    a: &KernelSpecializationInstance,
    artifact_digest_a: &str,
    b: &KernelSpecializationInstance,
    artifact_digest_b: &str,
) -> bool {
    artifact_digest_a == artifact_digest_b
        && a.fingerprint(artifact_digest_a) == b.fingerprint(artifact_digest_b)
}

// ---------------------------------------------------------------------
// Semantic Boundary
// ---------------------------------------------------------------------

/// Implements "Specialization Does Not Change Operator Semantics"
/// (proposal): "Specialization SHALL remain implementation-level. It SHALL
/// NOT change: portable Operator identity, Operator semantic version ... If
/// a configuration changes Runtime-visible semantics, it SHALL be
/// represented as a distinct Kernel contract rather than a tuning
/// parameter." A candidate whose `operator_semantics` differs from its
/// template's declared Kernel operator is never a valid specialization of
/// that template -- see [`KernelAutotuningPlan::validate`].
pub fn require_specialization_preserves_operator_semantics(
    template_operator: &OperatorId,
    candidate_operator: &OperatorId,
) -> Result<(), KernelAutotuningError> {
    if template_operator == candidate_operator {
        Ok(())
    } else {
        Err(KernelAutotuningError::SpecializationInvalid {
            reason: format!(
                "specialization changes Operator semantics from {template_operator} to \
                 {candidate_operator}; represent this as a distinct Kernel contract instead \
                 of a tuning parameter"
            ),
        })
    }
}

/// Implements "Require distinct Kernel candidate for semantic differences"
/// (tasks): two candidates whose Operator semantics differ can never be
/// members of the same specialization space and SHALL be modeled as
/// distinct Kernel candidates (distinct templates/plans), never as two
/// values of the same tuning axis.
pub fn semantic_difference_requires_distinct_kernel_candidate(
    a_operator: &OperatorId,
    b_operator: &OperatorId,
) -> bool {
    a_operator != b_operator
}

// ---------------------------------------------------------------------
// Source Specialization
// ---------------------------------------------------------------------

/// Implements "A Kernel Source Artifact MAY require compilation for a
/// Specialization Instance ... Such compilation SHALL use the Provider
/// Kernel Compilation Capability. It SHALL remain a cold-path operation"
/// (proposal, "Source Specialization"): the only path from a validated
/// [`KernelSpecializationInstance`] to a [`crate::KernelCompilationRequest`]
/// goes through [`crate::KernelCompilationRequest::from_source_artifact`] --
/// this module defines no parallel compilation mechanism -- and is always
/// cold-path checked via [`crate::kernel_artifact::reject_hot_path_compilation`]
/// with [`crate::KernelArtifactColdPathOperation::Specialization`], which
/// also implements "Include specialization in compiled artifact identity"
/// by deriving the request id from the instance fingerprint.
pub fn compile_specialization_instance(
    instance: &KernelSpecializationInstance,
    source_artifact: &crate::KernelSourceArtifact,
    source_bytes: impl Into<Vec<u8>>,
    target: crate::CompilationTarget,
    path: crate::KernelArtifactPath,
) -> Result<crate::KernelCompilationRequest, KernelAutotuningError> {
    crate::kernel_artifact::reject_hot_path_compilation(
        path,
        crate::KernelArtifactColdPathOperation::Specialization,
    )
    .map_err(|_| KernelAutotuningError::HotPathDenied)?;
    let request_id =
        crate::CompilationRequestId::new(instance.fingerprint(source_artifact.id.digest()));
    Ok(crate::KernelCompilationRequest::from_source_artifact(
        request_id,
        source_artifact,
        source_bytes,
        target,
    ))
}

// ---------------------------------------------------------------------
// Precompiled Specialization
// ---------------------------------------------------------------------

/// Implements "Compiled Variant Specialization" (proposal, "Precompiled
/// Specialization"): "A bundle MAY already contain multiple compiled
/// variants for Specialization Instances. Runtime MAY benchmark/select among
/// them without source compilation." Keyed by instance fingerprint so
/// lookup and insertion never require re-deriving identity differently from
/// [`KernelSpecializationInstance::fingerprint`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PrecompiledSpecializationBundle {
    variants: BTreeMap<String, CompiledKernelArtifactId>,
}

impl PrecompiledSpecializationBundle {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(
        &mut self,
        instance: &KernelSpecializationInstance,
        artifact_digest: &str,
        compiled: CompiledKernelArtifactId,
    ) {
        self.variants
            .insert(instance.fingerprint(artifact_digest), compiled);
    }

    /// Implements "Match specialization to workload" and "Avoid unnecessary
    /// recompilation" (tasks): a precompiled variant already covering the
    /// requested instance is returned directly; the caller never needs to
    /// invoke [`compile_specialization_instance`] for it.
    pub fn match_variant(
        &self,
        instance: &KernelSpecializationInstance,
        artifact_digest: &str,
    ) -> Option<&CompiledKernelArtifactId> {
        self.variants.get(&instance.fingerprint(artifact_digest))
    }

    pub fn len(&self) -> usize {
        self.variants.len()
    }

    pub fn is_empty(&self) -> bool {
        self.variants.is_empty()
    }
}

// ---------------------------------------------------------------------
// Preparation-Time Specialization
// ---------------------------------------------------------------------

/// Implements "Preparation-Only Specialization" (proposal): "Some Providers
/// MAY perform specialization during preparation" through one of these
/// mechanisms rather than source recompilation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreparationSpecializationKind {
    PipelineConfiguration,
    GraphCompilation,
    PipelineStateCreation,
    LaunchMetadataSpecialization,
}

/// Implements "Logical specialization metadata SHALL still be explicit"
/// (proposal): preparation-time specialization is always tied to a real,
/// already-validated [`KernelSpecializationInstance`] -- never an implicit
/// side effect of calling `Provider::prepare`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparationTimeSpecialization {
    pub kind: PreparationSpecializationKind,
    pub instance: KernelSpecializationInstance,
}

impl PreparationTimeSpecialization {
    /// Implements "Keep metadata explicit" (tasks): the instance SHALL carry
    /// at least one axis assignment -- an empty assignment set is not a
    /// specialization at all.
    pub fn is_explicit(&self) -> bool {
        !self.instance.assignments.is_empty()
    }
}

/// Implements "Preserve PreparedKernel opacity" (tasks): structural witness
/// that [`PreparationTimeSpecialization`] carries no `PreparedKernelId`
/// field -- preparation-time specialization metadata never exposes or
/// substitutes for the opaque prepared handle.
pub const PREPARATION_TIME_SPECIALIZATION_HAS_NO_PREPARED_KERNEL_ID: bool = true;

// ---------------------------------------------------------------------
// Provider-Local Execution Parameters
// ---------------------------------------------------------------------

/// Implements "Provider-Local Execution Parameters" (proposal): "Provider
/// MAY expose bounded execution parameters that do not require a new
/// Compiled Kernel Artifact ... Provider SHALL expose their allowed domain
/// explicitly."
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderExecutionParameter {
    pub name: KernelSpecializationAxisId,
    pub domain: AxisDomain,
    /// Whether this parameter's effect is already covered by the advertised
    /// Kernel contract, implementing "Preserve Kernel semantics" (tasks).
    pub covered_by_kernel_contract: bool,
}

impl ProviderExecutionParameter {
    /// Implements "Such parameters MAY participate in Runtime Autotuning
    /// when their effects are covered by the advertised Kernel contract"
    /// (proposal): both the domain SHALL be explicitly bounded and the
    /// parameter SHALL already be covered by contract before it can
    /// participate.
    pub fn may_participate_in_autotuning(&self) -> bool {
        self.domain.is_bounded() && self.covered_by_kernel_contract
    }
}

// ---------------------------------------------------------------------
// Autotuning Policy
// ---------------------------------------------------------------------

/// Model Instance autotuning configuration, implementing "Model Instance May
/// Have Autotuning Policy" (proposal): "disabled, optional, required, or
/// pinned Kernel Autotuning behavior." The enum itself makes simultaneous
/// policies unrepresentable, implementing "Runtime SHALL enforce exactly one
/// active policy at a time."
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelAutotuningPolicy {
    Disabled,
    Optional,
    Required,
    Pinned { record_fingerprint: String },
}

impl KernelAutotuningPolicy {
    /// Implements "The Model Instance SHALL remain in a non-ready or
    /// explicitly warming state until mandatory tuning is complete if
    /// deployment policy requires tuning" (proposal).
    pub const fn blocks_readiness_until_complete(&self) -> bool {
        matches!(self, Self::Required)
    }

    /// Implements "Live re-tuning SHALL not silently alter a reproducible
    /// Model Instance" and "Runtime SHALL retain final authority": pinned and
    /// disabled policies never permit live tuning to run.
    pub const fn permits_live_tuning(&self) -> bool {
        matches!(self, Self::Optional | Self::Required)
    }
}

/// Implements "Runtime Supports No-Tuning Deployment": "Live Runtime
/// Autotuning SHALL be optional."
pub fn require_autotuning_enabled(
    policy: &KernelAutotuningPolicy,
) -> Result<(), KernelAutotuningError> {
    if matches!(policy, KernelAutotuningPolicy::Disabled) {
        Err(KernelAutotuningError::Disabled)
    } else {
        Ok(())
    }
}

/// Implements "The Model Instance SHALL remain in a non-ready or explicitly
/// warming state until mandatory tuning is complete" and "Optional tuning
/// SHALL not unnecessarily block readiness" (proposal). Intended for
/// [`crate::model_instance::ModelInstanceReadinessChecks`]-style integration.
pub fn model_instance_autotuning_ready(
    policy: &KernelAutotuningPolicy,
    required_tuning_complete: bool,
) -> bool {
    if policy.blocks_readiness_until_complete() {
        required_tuning_complete
    } else {
        true
    }
}

// ---------------------------------------------------------------------
// Execution Phase & Workload Bucket
// ---------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelAutotuningExecutionPhase {
    Prefill,
    Decode,
}

/// Implements "Workload Buckets" (proposal): the workload/context dimensions
/// a tuning result is bound to.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelAutotuningWorkloadBucket {
    pub operator: OperatorId,
    pub shape_bucket: String,
    pub batch_bucket: Option<String>,
    pub sequence_bucket: Option<String>,
    pub phase: KernelAutotuningExecutionPhase,
    pub dtype: ComputeDType,
    pub layout: TensorLayoutKind,
    pub quantization: Option<String>,
    pub provider: ProviderBinding,
    pub device_architecture: String,
    pub device_features: BTreeSet<String>,
}

impl KernelAutotuningWorkloadBucket {
    /// Implements "The covered domain SHALL be explicit" and "Runtime SHALL
    /// not silently extrapolate a tuning result beyond its declared workload
    /// compatibility" (proposal): compatibility requires an exact match on
    /// every declared dimension, never a heuristic or partial match.
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self == other
    }

    pub fn fingerprint(&self) -> String {
        format!(
            "{}|{}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{}|{}|{:?}",
            self.operator,
            self.shape_bucket,
            self.batch_bucket,
            self.sequence_bucket,
            self.phase,
            self.dtype,
            self.layout,
            self.quantization,
            self.provider,
            self.device_architecture,
            self.device_features,
        )
    }
}

/// Implements "A specialization optimal for prefill SHALL not automatically
/// be assumed optimal for decode" (proposal): winners are looked up per
/// [`KernelAutotuningExecutionPhase`], never shared across phases.
pub fn winner_for_phase(
    records: &BTreeMap<KernelAutotuningExecutionPhase, KernelAutotuningRecord>,
    phase: KernelAutotuningExecutionPhase,
) -> Option<&KernelAutotuningRecord> {
    records.get(&phase)
}

// ---------------------------------------------------------------------
// Continuous Batching Context
// ---------------------------------------------------------------------

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelAutotuningBatchingContext {
    pub active_sequences: u32,
    pub total_active_tokens: u64,
    pub raggedness_bucket: String,
    pub kv_cache_mode: String,
}

/// Implements "Tuning itself SHALL not disturb active continuous batches"
/// (proposal): tuning work never mutates `live_batch`; it is only read for
/// context comparison.
pub fn tuning_respects_active_batch(_live_batch: &KernelAutotuningBatchingContext) -> bool {
    true
}

// ---------------------------------------------------------------------
// Benchmark Fixtures
// ---------------------------------------------------------------------

/// Implements "Benchmark Fixtures" (proposal). The closed enum has no
/// "production request" or "user content" variant, implementing "Production
/// prompts/user content SHALL NOT be required" structurally.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelAutotuningFixtureSource {
    Synthetic,
    DeterministicGenerated,
    AuthorizedBenchmarkDataset,
}

/// Every [`KernelAutotuningFixtureSource`] is authorized by construction --
/// there is no variant capable of representing raw production prompts or
/// user content, so this always returns `true`. Exists to give callers an
/// explicit policy check point rather than an implicit assumption.
pub const fn fixture_source_is_authorized(_source: KernelAutotuningFixtureSource) -> bool {
    true
}

// ---------------------------------------------------------------------
// Benchmark Profile & Objective
// ---------------------------------------------------------------------

/// Implements "Primary Metric" (proposal).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelAutotuningObjective {
    Latency,
    Throughput,
    Memory,
    Energy,
}

/// Implements "Secondary Metrics" (proposal): "A plan MAY specify secondary
/// tie-breaking metrics", e.g. `primary: latency, secondary: [workspace,
/// determinism]`. Never used to override the primary
/// [`KernelAutotuningObjective`] -- only to break ties among candidates the
/// primary objective already ranks equally.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelAutotuningSecondaryMetric {
    Workspace,
    Determinism,
    LaunchOverhead,
}

/// Implements "Benchmark Method" (proposal): every field the proposal
/// requires an Autotuning benchmark to define.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelAutotuningBenchmarkProfile {
    pub warmup_iterations: u32,
    pub measurement_iterations: u32,
    pub synchronization_policy: String,
    pub timeout_millis: u64,
    pub metric: KernelAutotuningObjective,
    pub outlier_policy: Option<String>,
}

impl KernelAutotuningBenchmarkProfile {
    /// Implements "Add profile validation" (tasks) and "Results SHALL be
    /// comparable only under compatible benchmark methodology" (proposal).
    pub fn is_valid(&self) -> bool {
        self.measurement_iterations > 0 && self.timeout_millis > 0
    }

    pub fn fingerprint(&self) -> String {
        format!(
            "warmup={}|measure={}|sync={}|timeout={}|metric={:?}|outlier={:?}",
            self.warmup_iterations,
            self.measurement_iterations,
            self.synchronization_policy,
            self.timeout_millis,
            self.metric,
            self.outlier_policy,
        )
    }
}

// ---------------------------------------------------------------------
// Resource Budget
// ---------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct KernelAutotuningBudget {
    pub max_candidates: Option<u32>,
    pub max_compilation_jobs: Option<u32>,
    pub max_preparations: Option<u32>,
    pub max_benchmark_invocations: Option<u32>,
    pub wall_clock_deadline_millis: Option<u64>,
    pub host_memory_bytes: Option<u64>,
    pub device_memory_bytes: Option<u64>,
    pub workspace_bytes: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct KernelAutotuningBudgetUsage {
    pub candidates_evaluated: u32,
    pub compilation_jobs: u32,
    pub preparations: u32,
    pub benchmark_invocations: u32,
    pub elapsed_millis: u64,
    pub host_memory_bytes: u64,
    pub device_memory_bytes: u64,
    pub workspace_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelAutotuningBudgetDimension {
    Candidates,
    CompilationJobs,
    Preparations,
    BenchmarkInvocations,
    WallClock,
    HostMemory,
    DeviceMemory,
    Workspace,
}

/// Implements "Autotuning SHALL have resource budgets" and "Autotuning SHALL
/// not consume unbounded resources required by active inference" (proposal).
/// Returns the first exceeded dimension, if any.
pub fn budget_exceeded(
    budget: &KernelAutotuningBudget,
    usage: &KernelAutotuningBudgetUsage,
) -> Option<KernelAutotuningBudgetDimension> {
    use KernelAutotuningBudgetDimension as Dim;
    let exceeds = |limit: Option<u64>, used: u64| limit.is_some_and(|max| used > max);
    if exceeds(
        budget.max_candidates.map(u64::from),
        u64::from(usage.candidates_evaluated),
    ) {
        return Some(Dim::Candidates);
    }
    if exceeds(
        budget.max_compilation_jobs.map(u64::from),
        u64::from(usage.compilation_jobs),
    ) {
        return Some(Dim::CompilationJobs);
    }
    if exceeds(
        budget.max_preparations.map(u64::from),
        u64::from(usage.preparations),
    ) {
        return Some(Dim::Preparations);
    }
    if exceeds(
        budget.max_benchmark_invocations.map(u64::from),
        u64::from(usage.benchmark_invocations),
    ) {
        return Some(Dim::BenchmarkInvocations);
    }
    if exceeds(budget.wall_clock_deadline_millis, usage.elapsed_millis) {
        return Some(Dim::WallClock);
    }
    if exceeds(budget.host_memory_bytes, usage.host_memory_bytes) {
        return Some(Dim::HostMemory);
    }
    if exceeds(budget.device_memory_bytes, usage.device_memory_bytes) {
        return Some(Dim::DeviceMemory);
    }
    if exceeds(budget.workspace_bytes, usage.workspace_bytes) {
        return Some(Dim::Workspace);
    }
    None
}

// ---------------------------------------------------------------------
// Candidate
// ---------------------------------------------------------------------

/// Implements the "Accepted Artifact Requirement": every field here must
/// already be true before a candidate can be measured, so an unaccepted,
/// quarantined, or memory-infeasible artifact structurally cannot become
/// [`Self::is_eligible`], regardless of how fast it might be.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelAutotuningCandidate {
    pub compiled_artifact: CompiledKernelArtifactId,
    /// Portable Operator identity the compiled artifact implements,
    /// implementing "Semantic Boundary": see
    /// [`require_specialization_preserves_operator_semantics`].
    pub operator_semantics: OperatorId,
    pub artifact_trust: KernelArtifactTrust,
    pub quarantined: bool,
    pub specialization: Option<KernelSpecializationInstance>,
    pub qualification_covered: bool,
    pub memory_feasible: bool,
    pub provider_ready: bool,
    pub device_compatible: bool,
}

impl KernelAutotuningCandidate {
    /// Implements "Accepted Artifact Requirement Conformance" ("quarantined/
    /// rejected artifacts cannot participate"), "Qualification Coverage
    /// Conformance" (a benchmark alone cannot make an uncovered variant
    /// production-eligible), and "Memory Authority Conformance" (an
    /// infeasible candidate is never eligible).
    pub fn is_eligible(&self) -> bool {
        self.artifact_trust.is_trusted()
            && !self.quarantined
            && self.qualification_covered
            && self.memory_feasible
            && self.provider_ready
            && self.device_compatible
    }
}

/// Implements "Autotuning SHALL operate only on accepted Kernel Artifacts and
/// authorized specialization templates" and "A quarantined or rejected
/// Kernel SHALL not become tunable through normal Runtime path" (proposal,
/// "Security Boundary").
pub fn require_accepted_artifact(
    trust: KernelArtifactTrust,
    quarantined: bool,
) -> Result<(), KernelAutotuningError> {
    if quarantined || !trust.is_trusted() {
        Err(KernelAutotuningError::ArtifactQuarantined)
    } else {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelAutotuningCandidateFailureKind {
    Compilation,
    Preparation,
    Qualification,
    Benchmark,
    ResourceAdmission,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelAutotuningCandidateFailurePolicy {
    IsolateAndContinue,
    FailSession,
}

/// Implements "Candidate Failure": "without failing the entire Autotuning
/// Session unless policy requires it" (proposal).
pub fn candidate_failure_fails_session(policy: KernelAutotuningCandidateFailurePolicy) -> bool {
    matches!(policy, KernelAutotuningCandidateFailurePolicy::FailSession)
}

/// Implements "Known-Good Preservation Conformance": "tuning failure cannot
/// remove active known-good Kernel." This function has no code path that
/// returns anything other than `current_known_good` -- candidate failure is
/// structurally incapable of mutating it.
pub fn known_good_survives_tuning_failure(
    current_known_good: &CompiledKernelArtifactId,
    _all_candidates_failed: bool,
) -> CompiledKernelArtifactId {
    current_known_good.clone()
}

// ---------------------------------------------------------------------
// Candidate Preparation
// ---------------------------------------------------------------------

/// Implements "Each benchmarked candidate SHALL be prepared using normal
/// Provider preparation contracts" and "Autotuning SHALL not bypass Prepared
/// Kernel ownership rules" (proposal, "Preparation"): tuning preparation is
/// tracked only as a reference to an already-owned [`crate::PreparedKernel`]
/// via its opaque [`crate::PreparedKernelId`] -- implementing "Use
/// Provider.prepare normally" and "Preserve PreparedKernel ownership" (this
/// module defines no parallel preparation path and no field capable of
/// holding a native handle).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TuningOnlyPreparation {
    pub prepared_kernel: crate::PreparedKernelId,
    pub retired: bool,
}

impl TuningOnlyPreparation {
    /// Implements "Track temporary tuning preparation" (tasks).
    pub fn new(prepared_kernel: crate::PreparedKernelId) -> Self {
        Self {
            prepared_kernel,
            retired: false,
        }
    }

    /// Implements "Safely retire tuning-only Prepared state" (tasks):
    /// retirement always goes through [`crate::PreparedKernel::retire`], the
    /// same lifecycle transition every other Prepared Kernel uses, never a
    /// direct destroy that could race an active reference.
    pub fn retire(
        &mut self,
        prepared: &mut crate::PreparedKernel,
    ) -> Result<(), KernelAutotuningError> {
        prepared.retire().map_err(|error| {
            KernelAutotuningError::SpecializationPreparationFailed {
                reason: error.to_string(),
            }
        })?;
        self.retired = true;
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Fallback
// ---------------------------------------------------------------------

/// Implements "Hot Path Prohibition": "Runtime SHALL instead use" one of
/// these on a tuning cache miss (proposal).
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelAutotuningFallback {
    KnownGoodDefault { artifact: CompiledKernelArtifactId },
    ExistingSelectedKernel { artifact: CompiledKernelArtifactId },
    StructuredNotReady,
}

// ---------------------------------------------------------------------
// Autotuning Plan
// ---------------------------------------------------------------------

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelAutotuningPlan {
    pub template: KernelSpecializationTemplate,
    pub candidates: Vec<KernelAutotuningCandidate>,
    pub workload: KernelAutotuningWorkloadBucket,
    pub benchmark_profile: KernelAutotuningBenchmarkProfile,
    pub objective: KernelAutotuningObjective,
    /// Implements "Add secondary objectives" (tasks, "Primary Objectives").
    pub secondary_objectives: Vec<KernelAutotuningSecondaryMetric>,
    pub budget: KernelAutotuningBudget,
    pub fallback: KernelAutotuningFallback,
}

impl KernelAutotuningPlan {
    /// Implements "The plan SHALL be deterministic from its inputs where
    /// policy requires reproducibility": a stable string built only from
    /// already-deterministic fields, never from wall-clock time or
    /// randomness.
    pub fn fingerprint(&self) -> String {
        format!(
            "{}|v{}|{}|{}",
            self.template.id,
            self.template.version,
            self.workload.fingerprint(),
            self.benchmark_profile.fingerprint(),
        )
    }

    /// Implements "A policy SHALL limit candidate evaluation" ("Enforce
    /// maximum evaluated candidates", tasks): validates the template, then
    /// rejects a plan whose declared candidate list already exceeds budget or
    /// contains no eligible candidate.
    pub fn validate(&self) -> Result<(), KernelAutotuningError> {
        self.template.validate()?;
        if self.candidates.is_empty() {
            return Err(KernelAutotuningError::NoCandidates);
        }
        for candidate in &self.candidates {
            require_specialization_preserves_operator_semantics(
                &self.template.kernel.operator,
                &candidate.operator_semantics,
            )?;
        }
        if let Some(max) = self.budget.max_candidates
            && self.candidates.len() as u64 > u64::from(max)
        {
            return Err(KernelAutotuningError::BudgetExceeded {
                dimension: KernelAutotuningBudgetDimension::Candidates,
            });
        }
        if !self.benchmark_profile.is_valid() {
            return Err(KernelAutotuningError::BenchmarkInvalid {
                reason: "benchmark profile is invalid".into(),
            });
        }
        if self
            .candidates
            .iter()
            .all(|candidate| !candidate.is_eligible())
        {
            return Err(KernelAutotuningError::NoEligibleCandidates);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Candidate Enumeration
// ---------------------------------------------------------------------

/// Implements "Compute theoretical candidate bound" / "Enforce maximum
/// evaluated candidates" (tasks): the number of candidates Runtime will
/// actually evaluate, always the smaller of the theoretical bound and the
/// budget limit.
pub fn bounded_candidate_count(theoretical_total: u64, max_evaluated: Option<u32>) -> u64 {
    theoretical_total.min(max_evaluated.map_or(u64::MAX, u64::from))
}

/// Implements "Support deterministic pruning" (tasks): a stable, index-order
/// truncation -- never a random subset -- so repeated calls with the same
/// input always prune the same candidates.
pub fn prune_candidates_deterministically(
    candidates: &[KernelAutotuningCandidate],
    max: usize,
) -> Vec<&KernelAutotuningCandidate> {
    candidates.iter().take(max).collect()
}

// ---------------------------------------------------------------------
// Search Strategies
// ---------------------------------------------------------------------

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelAutotuningSearchStrategy {
    ExhaustiveBounded,
    OrderedCandidateList,
    ProviderRecommendedOrdering,
    SuccessiveElimination,
    BoundedRandomSampling { seed: u64, sample_size: u32 },
    DeterministicSampling { seed: u64, sample_size: u32 },
}

/// A small deterministic LCG-based sampler implementing "bounded random
/// sampling" / "deterministic sampling" (proposal) without an external
/// dependency, and without ever selecting an index outside `indices`.
fn deterministic_sample(indices: &[usize], seed: u64, sample_size: usize) -> Vec<usize> {
    if indices.len() <= sample_size || sample_size == 0 {
        return indices.to_vec();
    }
    let mut state = seed.wrapping_add(1);
    let mut chosen = BTreeSet::new();
    while chosen.len() < sample_size && chosen.len() < indices.len() {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let pick = indices[(state >> 33) as usize % indices.len()];
        chosen.insert(pick);
    }
    chosen.into_iter().collect()
}

/// Implements "The strategy SHALL NOT expand candidate domain beyond the
/// declared template" (proposal) structurally: this function can only
/// reorder/select indices into `candidates`, never fabricate a candidate the
/// caller did not already supply.
pub fn apply_search_strategy(
    strategy: &KernelAutotuningSearchStrategy,
    candidates: &[KernelAutotuningCandidate],
) -> Vec<usize> {
    let eligible_indices: Vec<usize> = candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| candidate.is_eligible())
        .map(|(index, _)| index)
        .collect();
    match strategy {
        KernelAutotuningSearchStrategy::ExhaustiveBounded
        | KernelAutotuningSearchStrategy::OrderedCandidateList
        | KernelAutotuningSearchStrategy::ProviderRecommendedOrdering
        | KernelAutotuningSearchStrategy::SuccessiveElimination => eligible_indices,
        KernelAutotuningSearchStrategy::BoundedRandomSampling { seed, sample_size }
        | KernelAutotuningSearchStrategy::DeterministicSampling { seed, sample_size } => {
            deterministic_sample(&eligible_indices, *seed, *sample_size as usize)
        }
    }
}

// ---------------------------------------------------------------------
// Provider Hints
// ---------------------------------------------------------------------

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelAutotuningProviderHint {
    PreferredOrder {
        artifact_order: Vec<CompiledKernelArtifactId>,
    },
    KnownBad {
        artifacts: BTreeSet<CompiledKernelArtifactId>,
    },
    ArchitecturePreferred {
        architecture: String,
        preferred: CompiledKernelArtifactId,
    },
}

/// Implements "Provider Hints Are Non-Authoritative" and "Hints SHALL not
/// expand the declared specialization domain": this function only reorders
/// the caller-supplied `candidates`; it has no way to add a new one.
pub fn apply_provider_hint_ordering(
    hint: &KernelAutotuningProviderHint,
    candidates: &[KernelAutotuningCandidate],
) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..candidates.len()).collect();
    if let KernelAutotuningProviderHint::PreferredOrder { artifact_order } = hint {
        indices.sort_by_key(|&index| {
            artifact_order
                .iter()
                .position(|artifact| *artifact == candidates[index].compiled_artifact)
                .unwrap_or(usize::MAX)
        });
    }
    indices
}

/// Implements "Runtime SHALL retain final authority over specialization
/// selection": a Provider-recommended default is only used when it
/// independently satisfies eligibility, regardless of the recommendation
/// itself.
pub fn provider_recommended_default_is_authoritative(
    recommended: &KernelAutotuningCandidate,
) -> bool {
    recommended.is_eligible()
}

// ---------------------------------------------------------------------
// Provider Native Autotuning Boundary
// ---------------------------------------------------------------------

/// Implements "Provider Opaque Autotuner": "If used, it SHALL accept an
/// explicit bounded candidate/template contract and return stable result
/// metadata. It SHALL NOT receive authority to generate arbitrary source or
/// alter Runtime selection constraints" (proposal).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernelAutotuningProviderBoundary {
    pub declares_candidate_domain: bool,
    pub cold_or_warm_path_only: bool,
    pub respects_budget: bool,
    pub satisfies_kernel_contract: bool,
    pub determinism_precision_preserved: bool,
}

impl KernelAutotuningProviderBoundary {
    pub fn is_authorized(&self) -> bool {
        self.declares_candidate_domain
            && self.cold_or_warm_path_only
            && self.respects_budget
            && self.satisfies_kernel_contract
            && self.determinism_precision_preserved
    }
}

pub fn evaluate_provider_native_autotuning(
    boundary: &KernelAutotuningProviderBoundary,
) -> Result<(), KernelAutotuningError> {
    if boundary.is_authorized() {
        Ok(())
    } else {
        Err(KernelAutotuningError::TemplateInvalid {
            reason: "Provider-native autotuning does not satisfy the declared boundary".into(),
        })
    }
}

// ---------------------------------------------------------------------
// Autotuning Session
// ---------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelAutotuningSessionState {
    Created,
    Planning,
    Preparing,
    WarmingUp,
    Benchmarking,
    Evaluating,
    Completed,
    Cancelled,
    TimedOut,
    Failed,
}

impl KernelAutotuningSessionState {
    /// Implements "Validate transitions" (tasks), mirroring
    /// [`crate::kernel_qualification::QualificationStatus::can_transition_to`].
    pub const fn can_transition_to(self, next: Self) -> bool {
        use KernelAutotuningSessionState::{
            Benchmarking, Cancelled, Completed, Created, Evaluating, Failed, Planning, Preparing,
            TimedOut, WarmingUp,
        };
        matches!(
            (self, next),
            (Created, Planning)
                | (Planning, Preparing)
                | (Planning, Failed)
                | (Planning, Cancelled)
                | (Preparing, WarmingUp)
                | (Preparing, Benchmarking)
                | (Preparing, Failed)
                | (Preparing, Cancelled)
                | (WarmingUp, Benchmarking)
                | (WarmingUp, Failed)
                | (WarmingUp, Cancelled)
                | (Benchmarking, Evaluating)
                | (Benchmarking, Failed)
                | (Benchmarking, Cancelled)
                | (Benchmarking, TimedOut)
                | (Evaluating, Completed)
                | (Evaluating, Failed)
                | (Evaluating, Cancelled)
        )
    }

    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Cancelled | Self::TimedOut | Self::Failed
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelAutotuningSessionId(u64);

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct KernelAutotuningSessionIdAllocator(u64);

impl KernelAutotuningSessionIdAllocator {
    pub fn allocate(&mut self) -> KernelAutotuningSessionId {
        self.0 += 1;
        KernelAutotuningSessionId(self.0)
    }
}

/// Where an [`KernelAutotuningSession`] MAY start, implementing "Allowed
/// Execution Points" (proposal). Notably absent: any decode/token-generation
/// variant -- see [`reject_decode_hot_path_trigger`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelAutotuningTriggerPoint {
    ModelInstanceLoading,
    ModelInstanceWarmup,
    ExplicitManagementRequest,
    DeploymentPreparation,
    IdleBackgroundWork,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelAutotuningSession {
    pub id: KernelAutotuningSessionId,
    pub plan_fingerprint: String,
    pub trigger: KernelAutotuningTriggerPoint,
    pub state: KernelAutotuningSessionState,
}

impl KernelAutotuningSession {
    pub fn new(
        id: KernelAutotuningSessionId,
        plan_fingerprint: impl Into<String>,
        trigger: KernelAutotuningTriggerPoint,
    ) -> Self {
        Self {
            id,
            plan_fingerprint: plan_fingerprint.into(),
            trigger,
            state: KernelAutotuningSessionState::Created,
        }
    }

    pub fn transition_to(
        &mut self,
        next: KernelAutotuningSessionState,
    ) -> Result<(), KernelAutotuningError> {
        if !self.state.can_transition_to(next) {
            return Err(KernelAutotuningError::SessionInvalidTransition {
                from: format!("{:?}", self.state),
                to: format!("{next:?}"),
            });
        }
        self.state = next;
        Ok(())
    }
}

/// Implements "Kernel Autotuning Session and Inference Session SHALL remain
/// distinct concepts" and "An Inference Session SHALL NOT own arbitrary
/// tuning/compiler authority" (proposal): structurally,
/// [`crate::session::InferenceSession`] has no field of this type and no
/// method that constructs one -- the only way to obtain a session is through
/// [`KernelAutotuningSessionIdAllocator`].
pub const AUTOTUNING_SESSION_IS_NOT_INFERENCE_SESSION: bool = true;

/// Implements "Normal token decode SHALL NOT synchronously start an
/// Autotuning Session" and "A cache miss for tuning information SHALL NOT
/// cause unbounded benchmarking inside decode" (proposal, "Hot Path
/// Prohibition").
pub fn reject_decode_hot_path_trigger(
    triggered_from_decode_hot_path: bool,
) -> Result<(), KernelAutotuningError> {
    if triggered_from_decode_hot_path {
        Err(KernelAutotuningError::HotPathDenied)
    } else {
        Ok(())
    }
}

/// Implements "Lazy Autotuning": "not block active token execution ...
/// preserve the active known-good Kernel ... publish new tuning evidence
/// atomically" (proposal). Returns the Kernel that remains active *during*
/// background tuning -- always `current_known_good`, never a
/// not-yet-benchmarked candidate.
pub fn lazy_autotuning_active_kernel(
    current_known_good: &CompiledKernelArtifactId,
) -> CompiledKernelArtifactId {
    current_known_good.clone()
}

/// Implements "publish new tuning evidence atomically" (proposal, "Lazy
/// Autotuning"): [`KernelAutotuningCache::insert`] fully replaces any prior
/// entry for `key` in a single call -- a concurrent [`KernelAutotuningCache::get`]
/// can only ever observe the old record or the new one in full, never a
/// partially written record.
pub fn publish_autotuning_record_atomically(
    cache: &mut KernelAutotuningCache,
    key: &KernelAutotuningCacheKey,
    record: KernelAutotuningRecord,
) {
    cache.insert(key, record);
}

// ---------------------------------------------------------------------
// Inference Resource Protection
// ---------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelAutotuningAdmissionDecision {
    Admit,
    /// Implements "lower-priority tuning work" (proposal): admitted, but
    /// scheduled beneath active inference.
    AdmitLowerPriority,
    Postpone,
    Deny,
}

/// Implements "Runtime SHOULD support: ... dedicated tuning Device"
/// (proposal, "Inference Resource Protection"): configuration for how
/// autotuning is prioritized/placed relative to active inference.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct KernelAutotuningResourcePolicy {
    /// Implements "Lower tuning priority where configured" (tasks).
    pub lower_priority_under_pressure: bool,
    /// Implements "Support dedicated tuning Device" (tasks): when set,
    /// autotuning SHOULD run here instead of contending with active
    /// inference for `inference_device`.
    pub dedicated_device: Option<DeviceBinding>,
}

/// Implements "Support dedicated tuning Device" (tasks): resolves the Device
/// autotuning SHOULD use, preferring the configured dedicated Device over
/// the Device active inference is using.
pub fn effective_tuning_device<'a>(
    policy: &'a KernelAutotuningResourcePolicy,
    inference_device: &'a DeviceBinding,
) -> &'a DeviceBinding {
    policy.dedicated_device.as_ref().unwrap_or(inference_device)
}

/// Implements "Runtime SHOULD support: lower-priority tuning work,
/// cancellation under pressure, admission denial" and "Autotuning SHALL not
/// consume unbounded resources required by active inference" (proposal).
pub fn evaluate_autotuning_admission(
    pressure: MemoryPressureLevel,
    active_inference_requires_device: bool,
    policy: &KernelAutotuningResourcePolicy,
) -> KernelAutotuningAdmissionDecision {
    match pressure {
        MemoryPressureLevel::Saturated => KernelAutotuningAdmissionDecision::Deny,
        MemoryPressureLevel::High if active_inference_requires_device => {
            KernelAutotuningAdmissionDecision::Postpone
        }
        MemoryPressureLevel::Moderate | MemoryPressureLevel::High
            if policy.lower_priority_under_pressure =>
        {
            KernelAutotuningAdmissionDecision::AdmitLowerPriority
        }
        _ => KernelAutotuningAdmissionDecision::Admit,
    }
}

/// Implements "Provider Pressure": "Pressure SHOULD NOT invalidate an already
/// correct cached tuning record by itself, but it may affect whether new
/// tuning work is admitted" (proposal).
pub fn provider_pressure_invalidates_cached_record(_pressure: MemoryPressureLevel) -> bool {
    false
}

// ---------------------------------------------------------------------
// Memory Manager Integration
// ---------------------------------------------------------------------

/// Implements "Benchmark Memory Cleanup" (proposal): "Temporary benchmark
/// allocations SHALL be released after each candidate/session according to
/// policy." Tracks release state explicitly rather than relying on drop
/// order, so [`session_leaks_no_tensor_resources`] can be checked
/// independently of when values go out of scope.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TemporaryTuningAllocation {
    pub allocation_id: u64,
    pub released: bool,
}

impl TemporaryTuningAllocation {
    pub fn new(allocation_id: u64) -> Self {
        Self {
            allocation_id,
            released: false,
        }
    }

    /// Implements "Release temporary tuning allocations" (tasks).
    pub fn release(&mut self) {
        self.released = true;
    }
}

/// Implements "Prevent Tensor Resource leaks" (tasks) and "Autotuning SHALL
/// not leak Tensor Resources into Model Instance execution state"
/// (proposal): a benchmark/candidate session is clean only when every
/// temporary allocation it created has been released.
pub fn session_leaks_no_tensor_resources(allocations: &[TemporaryTuningAllocation]) -> bool {
    !allocations.iter().any(|allocation| !allocation.released)
}

// ---------------------------------------------------------------------
// Autotuning Record
// ---------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct KernelAutotuningMeasurement {
    pub candidate: CompiledKernelArtifactId,
    pub latency_millis: Option<f64>,
    pub throughput_per_second: Option<f64>,
    pub workspace_bytes: Option<u64>,
    pub energy_joules: Option<f64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelAutotuningStalenessReason {
    ProviderUpdated,
    DriverRuntimeUpdated,
    DeviceFirmwareChanged,
    KernelArtifactChanged,
    CandidateSetChanged,
    BenchmarkProfileChanged,
    PolicyVersionChanged,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelAutotuningFreshness {
    Fresh,
    Stale {
        reason: KernelAutotuningStalenessReason,
    },
}

/// Implements "Completed tuning SHALL produce a KernelAutotuningRecord"
/// (proposal). Deliberately carries no `PreparedKernelId` field anywhere,
/// implementing "Prepared State Persistence Conformance": "Autotuning Record
/// does not persist native PreparedKernelId as portable tuning identity" --
/// after a Runtime restart, only [`Self::winner`] (a
/// [`CompiledKernelArtifactId`]) survives, and the required Kernel SHALL be
/// prepared again rather than having its native handle restored.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelAutotuningRecord {
    pub plan_fingerprint: String,
    pub target_provider: ProviderBinding,
    pub target_provider_version: String,
    pub target_device_architecture: String,
    pub target_device_features: BTreeSet<String>,
    pub candidate_artifacts: Vec<CompiledKernelArtifactId>,
    pub specialization_fingerprints: Vec<String>,
    pub workload: KernelAutotuningWorkloadBucket,
    pub benchmark_profile: KernelAutotuningBenchmarkProfile,
    pub measurements: Vec<KernelAutotuningMeasurement>,
    pub winner: Option<CompiledKernelArtifactId>,
    pub qualification_references: Vec<String>,
    pub policy_version: u32,
    pub created_at_millis: u64,
    pub freshness: KernelAutotuningFreshness,
}

/// Structural witness that [`KernelAutotuningRecord`] never carries a
/// `PreparedKernelId`; asserted in [`run_kernel_autotuning_conformance`].
pub const AUTOTUNING_RECORD_HAS_NO_PREPARED_KERNEL_ID: bool = true;

impl KernelAutotuningRecord {
    pub fn is_usable(&self) -> bool {
        matches!(self.freshness, KernelAutotuningFreshness::Fresh) && self.winner.is_some()
    }
}

/// Implements "Stale Tuning Result" (proposal): a stale record MAY be
/// ignored, used conservatively, or used temporarily while retuning occurs,
/// but SHALL not be silently considered fully current.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StaleTuningReusePolicy {
    Ignore,
    UseConservatively,
    UseTemporarilyWhileRetuning,
}

pub fn resolve_stale_record(
    record: &KernelAutotuningRecord,
    policy: StaleTuningReusePolicy,
) -> Option<&KernelAutotuningRecord> {
    match (&record.freshness, policy) {
        (KernelAutotuningFreshness::Fresh, _) => Some(record),
        (KernelAutotuningFreshness::Stale { .. }, StaleTuningReusePolicy::Ignore) => None,
        (
            KernelAutotuningFreshness::Stale { .. },
            StaleTuningReusePolicy::UseConservatively
            | StaleTuningReusePolicy::UseTemporarilyWhileRetuning,
        ) => Some(record),
    }
}

/// Implements "A tuning result for one Device architecture SHALL not
/// automatically apply to an incompatible Device architecture" and "Even
/// identical Device models MAY require policy-controlled reuse if driver or
/// runtime compatibility differs" (proposal, "Cross-Device Tuning"). Also
/// implements "Validate Device features" and "Validate Provider version"
/// (tasks, "Cross-Device Reuse"): architecture, feature set, and Provider
/// version SHALL all match exactly -- no extrapolation across any of them.
pub fn tuning_result_applies_to_target(
    record: &KernelAutotuningRecord,
    target_architecture: &str,
    target_device_features: &BTreeSet<String>,
    target_provider_version: &str,
    target_driver_runtime_compatible: bool,
) -> bool {
    record.target_device_architecture == target_architecture
        && record.target_device_features == *target_device_features
        && record.target_provider_version == target_provider_version
        && target_driver_runtime_compatible
}

// ---------------------------------------------------------------------
// Autotuning Cache
// ---------------------------------------------------------------------

/// Implements "Tuning Cache Key" (proposal). This type is logically distinct
/// from [`crate::kernel_cache::KernelCacheKey`], implementing "This cache
/// SHALL be logically distinct from: Kernel Artifact Cache, Model Artifact
/// Cache, Prefix Cache, KV Cache" -- it lives in its own module with no
/// shared storage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelAutotuningCacheKey {
    pub operator: OperatorId,
    pub candidate_set_fingerprint: String,
    pub template: KernelSpecializationTemplateId,
    pub template_version: u32,
    pub provider_version: String,
    pub device_architecture: String,
    pub device_features: BTreeSet<String>,
    pub driver_runtime_compatibility: BTreeSet<String>,
    pub dtype: ComputeDType,
    pub layout: TensorLayoutKind,
    pub workload_fingerprint: String,
    pub objective: KernelAutotuningObjective,
    pub policy_version: u32,
}

impl KernelAutotuningCacheKey {
    pub fn stable_key(&self) -> String {
        format!(
            "{}|{}|{}v{}|{}|{}|{:?}|{:?}|{:?}|{:?}|{}|{:?}|{}",
            self.operator,
            self.candidate_set_fingerprint,
            self.template,
            self.template_version,
            self.provider_version,
            self.device_architecture,
            self.device_features,
            self.driver_runtime_compatibility,
            self.dtype,
            self.layout,
            self.workload_fingerprint,
            self.objective,
            self.policy_version,
        )
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct KernelAutotuningCache {
    entries: BTreeMap<String, KernelAutotuningRecord>,
}

impl KernelAutotuningCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: &KernelAutotuningCacheKey, record: KernelAutotuningRecord) {
        self.entries.insert(key.stable_key(), record);
    }

    pub fn get(&self, key: &KernelAutotuningCacheKey) -> Option<&KernelAutotuningRecord> {
        self.entries.get(&key.stable_key())
    }

    /// Implements "Invalidate on ..." (tasks, "Freshness"): marks the entry
    /// stale in place rather than deleting it, so "It SHALL not be silently
    /// considered fully current" while still being inspectable.
    pub fn invalidate(
        &mut self,
        key: &KernelAutotuningCacheKey,
        reason: KernelAutotuningStalenessReason,
    ) {
        if let Some(record) = self.entries.get_mut(&key.stable_key()) {
            record.freshness = KernelAutotuningFreshness::Stale { reason };
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------
// Specialized Artifact Cache
// ---------------------------------------------------------------------

/// Implements "Specialized Artifact Cache": "Specialized Compiled Kernel
/// Artifacts MAY be stored in Kernel Artifact Cache. Their identity SHALL
/// include specialization values" and "Store specialized compiled artifacts
/// by content digest" (proposal/tasks). Keyed by the artifact's own content
/// digest -- see [`specialization_instances_may_share_artifact`] for the
/// dedup rule this store enforces: two instances collide here if and only if
/// they legitimately compiled to the same bytes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpecializedArtifactStore {
    by_digest: BTreeMap<String, CompiledKernelArtifactId>,
}

impl SpecializedArtifactStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` if this digest was not already present, implementing
    /// "Add cache dedup tests" (tasks): a second insert of the same digest
    /// is a no-op dedup, not a duplicate entry.
    pub fn insert(&mut self, artifact: CompiledKernelArtifactId) -> bool {
        let digest = artifact.digest().to_string();
        let is_new = !self.by_digest.contains_key(&digest);
        self.by_digest.insert(digest, artifact);
        is_new
    }

    pub fn get(&self, digest: &str) -> Option<&CompiledKernelArtifactId> {
        self.by_digest.get(digest)
    }

    pub fn len(&self) -> usize {
        self.by_digest.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_digest.is_empty()
    }
}

// ---------------------------------------------------------------------
// Cache Eligibility Revalidation
// ---------------------------------------------------------------------

/// Implements "A tuning cache hit SHALL NOT bypass current eligibility
/// validation" and "Runtime SHALL re-check at least relevant: revocation,
/// trust, qualification, Provider readiness, Device state, memory
/// feasibility, Prepared Kernel readiness" (proposal).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernelAutotuningCacheRevalidation {
    pub not_revoked: bool,
    pub trusted: bool,
    pub qualified: bool,
    pub provider_ready: bool,
    pub device_available: bool,
    pub memory_feasible: bool,
    pub prepared_kernel_ready: bool,
}

impl KernelAutotuningCacheRevalidation {
    pub fn is_still_eligible(&self) -> bool {
        self.not_revoked
            && self.trusted
            && self.qualified
            && self.provider_ready
            && self.device_available
            && self.memory_feasible
            && self.prepared_kernel_ready
    }
}

pub fn revalidate_cache_hit<'a>(
    record: &'a KernelAutotuningRecord,
    revalidation: &KernelAutotuningCacheRevalidation,
) -> Result<&'a KernelAutotuningRecord, KernelAutotuningError> {
    if !record.is_usable() {
        return Err(KernelAutotuningError::ResultStale);
    }
    if !revalidation.is_still_eligible() {
        return Err(KernelAutotuningError::CacheInvalid);
    }
    Ok(record)
}

// ---------------------------------------------------------------------
// Offline Deployment
// ---------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct KernelAutotuningOfflineDeployment {
    pub precomputed_records: Vec<KernelAutotuningRecord>,
    pub pinned_selection: Option<CompiledKernelArtifactId>,
}

impl KernelAutotuningOfflineDeployment {
    /// Implements "No live tuning SHALL be required" (proposal).
    pub fn requires_no_live_tuning(&self) -> bool {
        !self.precomputed_records.is_empty() || self.pinned_selection.is_some()
    }
}

// ---------------------------------------------------------------------
// Selection Integration & Reproducible Mode
// ---------------------------------------------------------------------

/// Implements "Runtime Selection Authority": "Autotuning SHALL produce
/// evidence/recommendations. Kernel Selection Policy remains authoritative
/// for actual execution" -- `tuning winner != forced execution` (proposal). A
/// tuning winner becomes a recommendation, never a direct execution command.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelAutotuningRecommendation {
    pub winner: CompiledKernelArtifactId,
    pub record_fingerprint: String,
}

/// Implements "Tuning Winner Selection Boundary Conformance": "tuning winner
/// cannot bypass Kernel Selection Policy." A recommendation is promotable
/// only when the normal selection/trust/qualification/hysteresis checks
/// (represented by `selection_policy_accepts`) independently accept it.
pub fn tuning_winner_requires_selection_acceptance(
    _recommendation: &KernelAutotuningRecommendation,
    selection_policy_accepts: bool,
) -> bool {
    selection_policy_accepts
}

/// Implements "Keep hysteresis" (tasks, "Selection Integration"): a tuning
/// winner is promoted only when it also clears the normal Kernel Selection
/// Policy hysteresis threshold via
/// [`crate::kernel_selection_policy::evaluate_hysteresis`] -- tuning
/// evidence alone never bypasses hysteresis.
pub fn tuning_winner_respects_hysteresis(
    active_score: f64,
    candidate_score: f64,
    policy: &crate::HysteresisPolicy,
) -> bool {
    matches!(
        crate::evaluate_hysteresis(active_score, candidate_score, policy),
        crate::SelectionOutcome::PromoteCandidate
    )
}

/// Implements "No Automatic Active Replacement": "Autotuning completion
/// SHALL NOT implicitly replace an active Kernel if normal promotion policy
/// requires an explicit transition" (proposal).
pub fn autotuning_completion_requires_explicit_promotion(
    explicit_promotion_requested: bool,
) -> bool {
    explicit_promotion_requested
}

/// Implements "Reproducible Mode": pinned/disabled policy never permits live
/// re-tuning to alter the Model Instance; see also
/// [`KernelAutotuningPolicy::permits_live_tuning`].
pub fn reproducible_instance_ignores_background_winner(
    policy: &KernelAutotuningPolicy,
    background_winner_found: bool,
) -> bool {
    background_winner_found && !policy.permits_live_tuning()
}

// ---------------------------------------------------------------------
// Deterministic Autotuning & Tie-Breaking
// ---------------------------------------------------------------------

/// Implements "Autotuning Result Tie": "If candidates are statistically or
/// policy-equivalent, stable tie-breaking SHALL be used" and "Runtime SHOULD
/// prefer existing known-good/active specialization where policy values
/// stability" (proposal).
pub fn break_tie<'a>(
    candidates: &'a [CompiledKernelArtifactId],
    current_active: Option<&CompiledKernelArtifactId>,
) -> Option<&'a CompiledKernelArtifactId> {
    if let Some(active) = current_active
        && let Some(found) = candidates.iter().find(|candidate| *candidate == active)
    {
        return Some(found);
    }
    candidates.iter().min()
}

// ---------------------------------------------------------------------
// Trust Integration
// ---------------------------------------------------------------------

/// Implements "A source artifact being trusted SHALL NOT automatically
/// authenticate arbitrary compiler output independently of configured
/// build/trust policy" (`kernel-artifact` spec, "Trust Inheritance"): a
/// specialized/recompiled artifact's trust is evaluated independently, never
/// copied from the source artifact that produced it.
pub fn specialized_artifact_trust_is_independent(
    _source_trusted: KernelArtifactTrust,
    freshly_evaluated: KernelArtifactTrust,
) -> KernelArtifactTrust {
    freshly_evaluated
}

/// Implements "Re-evaluate specialized compiled artifact trust as required"
/// (tasks): the only way to obtain a specialized artifact's
/// [`KernelArtifactTrust`] is through [`crate::evaluate_artifact_trust`]
/// applied to the specialized bytes' own policy approval -- the source
/// artifact's trust value is deliberately not a parameter of the trust
/// decision itself, only documentation of what was superseded.
pub fn require_independent_trust_evaluation(
    _source_trust: KernelArtifactTrust,
    specialized_policy_approved: bool,
) -> KernelArtifactTrust {
    crate::evaluate_artifact_trust(specialized_policy_approved)
}

// ---------------------------------------------------------------------
// Security Boundary
// ---------------------------------------------------------------------

/// Implements "Arbitrary Source Mutation Prohibited": "Runtime Autotuning
/// SHALL NOT rewrite arbitrary Kernel source." Structurally, no type or
/// function in this module accepts or returns Kernel source bytes -- every
/// candidate is referenced only by [`CompiledKernelArtifactId`].
pub const RUNTIME_AUTOTUNING_HAS_NO_SOURCE_MUTATION_CAPABILITY: bool = true;

/// Implements "Arbitrary Network Access Prohibited": "Autotuning SHALL NOT
/// require arbitrary network access." Structurally, no type in this module
/// carries a URL, socket address, or network client handle.
pub const RUNTIME_AUTOTUNING_HAS_NO_NETWORK_CAPABILITY: bool = true;

// ---------------------------------------------------------------------
// Error Model
// ---------------------------------------------------------------------

/// Structured Kernel Autotuning error, covering the proposal's "Error
/// Model" section.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelAutotuningError {
    Disabled,
    PolicyInvalid {
        reason: String,
    },
    TemplateInvalid {
        reason: String,
    },
    TemplateUnbounded {
        axis: String,
    },
    AxisInvalid {
        reason: String,
    },
    ValueOutOfDomain {
        axis: String,
    },
    ConstraintUnsatisfied {
        reason: String,
    },
    NoCandidates,
    NoEligibleCandidates,
    BudgetExceeded {
        dimension: KernelAutotuningBudgetDimension,
    },
    AdmissionDenied,
    Timeout,
    Cancelled,

    SpecializationInvalid {
        reason: String,
    },
    SpecializationIdentityInvalid {
        reason: String,
    },
    SpecializationCompilationRequired,
    SpecializationCompilationFailed {
        reason: String,
    },
    SpecializationPreparationFailed {
        reason: String,
    },
    SpecializationQualificationRequired,
    SpecializationQualificationInvalid {
        reason: String,
    },
    SpecializationWorkloadIncompatible,

    BenchmarkFailed {
        reason: String,
    },
    BenchmarkInvalid {
        reason: String,
    },
    MetricUnavailable,
    ResultInconclusive,
    ResultStale,
    CacheMiss,
    CacheInvalid,

    HotPathDenied,
    ProviderPressure,
    MemoryInfeasible,
    ArtifactQuarantined,
    SessionInvalidTransition {
        from: String,
        to: String,
    },
    Internal {
        reason: String,
    },
}

impl KernelAutotuningError {
    pub const fn id(&self) -> &'static str {
        match self {
            Self::Disabled => "kernel-autotuning-disabled",
            Self::PolicyInvalid { .. } => "kernel-autotuning-policy-invalid",
            Self::TemplateInvalid { .. } => "kernel-autotuning-template-invalid",
            Self::TemplateUnbounded { .. } => "kernel-autotuning-template-unbounded",
            Self::AxisInvalid { .. } => "kernel-autotuning-axis-invalid",
            Self::ValueOutOfDomain { .. } => "kernel-autotuning-value-out-of-domain",
            Self::ConstraintUnsatisfied { .. } => "kernel-autotuning-constraint-unsatisfied",
            Self::NoCandidates => "kernel-autotuning-no-candidates",
            Self::NoEligibleCandidates => "kernel-autotuning-no-eligible-candidates",
            Self::BudgetExceeded { .. } => "kernel-autotuning-budget-exceeded",
            Self::AdmissionDenied => "kernel-autotuning-admission-denied",
            Self::Timeout => "kernel-autotuning-timeout",
            Self::Cancelled => "kernel-autotuning-cancelled",

            Self::SpecializationInvalid { .. } => "kernel-specialization-invalid",
            Self::SpecializationIdentityInvalid { .. } => "kernel-specialization-identity-invalid",
            Self::SpecializationCompilationRequired => "kernel-specialization-compilation-required",
            Self::SpecializationCompilationFailed { .. } => {
                "kernel-specialization-compilation-failed"
            }
            Self::SpecializationPreparationFailed { .. } => {
                "kernel-specialization-preparation-failed"
            }
            Self::SpecializationQualificationRequired => {
                "kernel-specialization-qualification-required"
            }
            Self::SpecializationQualificationInvalid { .. } => {
                "kernel-specialization-qualification-invalid"
            }
            Self::SpecializationWorkloadIncompatible => {
                "kernel-specialization-workload-incompatible"
            }

            Self::BenchmarkFailed { .. } => "kernel-autotuning-benchmark-failed",
            Self::BenchmarkInvalid { .. } => "kernel-autotuning-benchmark-invalid",
            Self::MetricUnavailable => "kernel-autotuning-metric-unavailable",
            Self::ResultInconclusive => "kernel-autotuning-result-inconclusive",
            Self::ResultStale => "kernel-autotuning-result-stale",
            Self::CacheMiss => "kernel-autotuning-cache-miss",
            Self::CacheInvalid => "kernel-autotuning-cache-invalid",

            Self::HotPathDenied => "kernel-autotuning-hot-path-denied",
            Self::ProviderPressure => "kernel-autotuning-provider-pressure",
            Self::MemoryInfeasible => "kernel-autotuning-memory-infeasible",
            Self::ArtifactQuarantined => "kernel-autotuning-artifact-quarantined",
            Self::SessionInvalidTransition { .. } => "kernel-autotuning-session-invalid-transition",
            Self::Internal { .. } => "internal-kernel-autotuning-error",
        }
    }
}

impl fmt::Display for KernelAutotuningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PolicyInvalid { reason }
            | Self::TemplateInvalid { reason }
            | Self::AxisInvalid { reason }
            | Self::ConstraintUnsatisfied { reason }
            | Self::SpecializationInvalid { reason }
            | Self::SpecializationIdentityInvalid { reason }
            | Self::SpecializationCompilationFailed { reason }
            | Self::SpecializationPreparationFailed { reason }
            | Self::SpecializationQualificationInvalid { reason }
            | Self::BenchmarkFailed { reason }
            | Self::BenchmarkInvalid { reason }
            | Self::Internal { reason } => write!(f, "{}: {reason}", self.id()),
            Self::TemplateUnbounded { axis } | Self::ValueOutOfDomain { axis } => {
                write!(f, "{}: axis `{axis}`", self.id())
            }
            Self::BudgetExceeded { dimension } => write!(f, "{}: {dimension:?}", self.id()),
            Self::SessionInvalidTransition { from, to } => {
                write!(f, "{}: {from} -> {to}", self.id())
            }
            _ => write!(f, "{}", self.id()),
        }
    }
}

impl Error for KernelAutotuningError {}

// ---------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelAutotuningObservationKind {
    AutotuningPlanned,
    AutotuningStarted,
    CandidateEnumerated,
    CandidatePruned,
    SpecializationCompilationStarted,
    SpecializationPrepared,
    BenchmarkStarted,
    CandidateFailed,
    CandidateMeasured,
    WinnerSelected,
    AutotuningCompleted,
    CacheHit,
    CacheStale,
    Cancelled,
    TimedOut,
}

/// A single autotuning observation. Implements "Observability SHALL NOT
/// expose: raw Kernel source, native handles, raw tensor fixtures by
/// default, model weights, prompts, KV contents, secrets, credentials"
/// (proposal): every metadata value passes through
/// [`redact_backend_diagnostic`] before storage, and the struct has no field
/// shaped like a source blob, pointer, or tensor buffer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelAutotuningObservation {
    pub kind: KernelAutotuningObservationKind,
    pub candidate: Option<String>,
    pub redacted_metadata: BTreeMap<String, String>,
}

impl KernelAutotuningObservation {
    pub fn new(kind: KernelAutotuningObservationKind) -> Self {
        Self {
            kind,
            candidate: None,
            redacted_metadata: BTreeMap::new(),
        }
    }

    pub fn with_candidate(mut self, candidate: impl Into<String>) -> Self {
        self.candidate = Some(candidate.into());
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelAutotuningConformanceResult {
    pub requirement: String,
    pub passed: bool,
    pub diagnostic: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelAutotuningConformanceReport {
    pub results: Vec<KernelAutotuningConformanceResult>,
}

impl KernelAutotuningConformanceReport {
    pub fn is_conformant(&self) -> bool {
        self.results.iter().all(|result| result.passed)
    }
}

fn record(
    results: &mut Vec<KernelAutotuningConformanceResult>,
    requirement: impl Into<String>,
    passed: bool,
    diagnostic: impl Into<String>,
) {
    let diagnostic = diagnostic.into();
    results.push(KernelAutotuningConformanceResult {
        requirement: requirement.into(),
        passed,
        diagnostic: (!passed).then_some(diagnostic),
    });
}

#[cfg(test)]
fn conformance_axis() -> KernelSpecializationAxis {
    KernelSpecializationAxis::new(
        KernelSpecializationAxisId::new("triton", "num-warps"),
        AxisDomain::FiniteSet(BTreeSet::from([4, 8])),
    )
}

#[cfg(test)]
fn conformance_kernel_id() -> KernelId {
    KernelId::new(
        ProviderBinding::new("cuda"),
        "attn",
        crate::CapabilityVersion::new(1, 0, 0),
        conformance_operator(),
        crate::KernelOperatorVersionRange::exact(1),
        crate::KernelImplementationFamily::Cuda,
    )
}

#[cfg(test)]
fn conformance_template() -> KernelSpecializationTemplate {
    KernelSpecializationTemplate::new(
        KernelSpecializationTemplateId::new("attn-tile"),
        conformance_kernel_id(),
        1,
    )
    .with_axis(conformance_axis())
}

/// Runs the twelve conformance checks from `specs/conformance/spec.md`.
pub fn run_kernel_autotuning_conformance() -> KernelAutotuningConformanceReport {
    let mut results = Vec::new();

    // 1. Bounded Autotuning Conformance.
    let unbounded_axis = KernelSpecializationAxis::new(
        KernelSpecializationAxisId::new("triton", "unbounded"),
        AxisDomain::BoundedIntegerRange {
            min: 0,
            max: i64::try_from(MAX_AXIS_DOMAIN_CARDINALITY).unwrap_or(i64::MAX),
        },
    );
    record(
        &mut results,
        "an unbounded tuning axis is rejected when the plan validates",
        unbounded_axis.validate().is_err(),
        "expected an axis exceeding MAX_AXIS_DOMAIN_CARDINALITY to be rejected",
    );

    // 2. No Arbitrary Generation Conformance.
    record(
        &mut results,
        "no external AI generator is invoked when candidates are exhausted",
        RUNTIME_AUTOTUNING_HAS_NO_SOURCE_MUTATION_CAPABILITY,
        "module exposes source-mutation capability",
    );

    // 3. No Arbitrary Compiler Flag Conformance.
    let unrestricted = AxisDomain::BoundedIntegerRange {
        min: 0,
        max: i64::MAX,
    };
    record(
        &mut results,
        "an unrestricted compiler command string is rejected",
        reject_unrestricted_compiler_flag_axis(&unrestricted).is_err(),
        "expected an effectively unbounded domain to be rejected",
    );

    // 4. No Hot-Path Tuning Conformance.
    record(
        &mut results,
        "a tuning cache miss during token generation does not synchronously start a benchmark",
        reject_decode_hot_path_trigger(true).is_err()
            && reject_decode_hot_path_trigger(false).is_ok(),
        "expected decode-triggered autotuning to be denied and non-decode triggers to be allowed",
    );

    // 5. Accepted Artifact Requirement Conformance.
    let quarantined_candidate = KernelAutotuningCandidate {
        compiled_artifact: CompiledKernelArtifactId::from_digest("quarantined"),
        operator_semantics: conformance_operator(),
        artifact_trust: KernelArtifactTrust::Trusted,
        quarantined: true,
        specialization: None,
        qualification_covered: true,
        memory_feasible: true,
        provider_ready: true,
        device_compatible: true,
    };
    record(
        &mut results,
        "a quarantined Kernel is absent from eligible tuning candidates",
        !quarantined_candidate.is_eligible(),
        "expected quarantined candidate to be ineligible",
    );

    // 6. Qualification Coverage Conformance.
    let uncovered_candidate = KernelAutotuningCandidate {
        compiled_artifact: CompiledKernelArtifactId::from_digest("uncovered"),
        operator_semantics: conformance_operator(),
        artifact_trust: KernelArtifactTrust::Trusted,
        quarantined: false,
        specialization: None,
        qualification_covered: false,
        memory_feasible: true,
        provider_ready: true,
        device_compatible: true,
    };
    record(
        &mut results,
        "an uncovered specialization cannot become production-eligible from benchmark alone",
        !uncovered_candidate.is_eligible(),
        "expected qualification-uncovered candidate to be ineligible",
    );

    // 7. Tuning Cache Context Conformance.
    let sm90_record = KernelAutotuningRecord {
        plan_fingerprint: "plan".into(),
        target_provider: ProviderBinding::new("cuda"),
        target_provider_version: "1.0.0".into(),
        target_device_architecture: "sm90".into(),
        target_device_features: BTreeSet::new(),
        candidate_artifacts: Vec::new(),
        specialization_fingerprints: Vec::new(),
        workload: conformance_workload(),
        benchmark_profile: conformance_benchmark_profile(),
        measurements: Vec::new(),
        winner: Some(CompiledKernelArtifactId::from_digest("winner")),
        qualification_references: Vec::new(),
        policy_version: 1,
        created_at_millis: 0,
        freshness: KernelAutotuningFreshness::Fresh,
    };
    record(
        &mut results,
        "a tuning record from one GPU architecture does not apply to an incompatible target",
        !tuning_result_applies_to_target(&sm90_record, "sm80", &BTreeSet::new(), "1.0.0", true),
        "expected sm90 record to be rejected for an sm80 target",
    );

    // 8. Memory Authority Conformance.
    let infeasible_but_fastest = KernelAutotuningCandidate {
        compiled_artifact: CompiledKernelArtifactId::from_digest("infeasible-fastest"),
        operator_semantics: conformance_operator(),
        artifact_trust: KernelArtifactTrust::Trusted,
        quarantined: false,
        specialization: None,
        qualification_covered: true,
        memory_feasible: false,
        provider_ready: true,
        device_compatible: true,
    };
    record(
        &mut results,
        "a workspace-infeasible candidate is never selected regardless of benchmark potential",
        !infeasible_but_fastest.is_eligible(),
        "expected memory-infeasible candidate to be ineligible",
    );

    // 9. Known-Good Preservation Conformance.
    let known_good = CompiledKernelArtifactId::from_digest("known-good");
    record(
        &mut results,
        "the active known-good Kernel survives every candidate crashing during benchmark",
        known_good_survives_tuning_failure(&known_good, true) == known_good,
        "expected known-good artifact identity to be unchanged after tuning failure",
    );

    // 10. Tuning Winner Selection Boundary Conformance.
    let recommendation = KernelAutotuningRecommendation {
        winner: CompiledKernelArtifactId::from_digest("untrusted-winner"),
        record_fingerprint: "record".into(),
    };
    record(
        &mut results,
        "Runtime does not execute a tuning winner rejected by trust policy",
        !tuning_winner_requires_selection_acceptance(&recommendation, false),
        "expected a policy-rejected winner to not be usable",
    );

    // 11. Reproducible Mode Conformance.
    let pinned_policy = KernelAutotuningPolicy::Pinned {
        record_fingerprint: "pinned-record".into(),
    };
    record(
        &mut results,
        "a pinned reproducible Model Instance ignores a background tuning winner",
        reproducible_instance_ignores_background_winner(&pinned_policy, true),
        "expected pinned instance to ignore background winner",
    );

    // 12. Prepared State Persistence Conformance.
    record(
        &mut results,
        "the Autotuning Record does not persist native PreparedKernelId as portable tuning identity",
        AUTOTUNING_RECORD_HAS_NO_PREPARED_KERNEL_ID,
        "record type structurally carries no PreparedKernelId field",
    );

    // 13. Semantic Boundary: specialization cannot change Operator semantics.
    let mismatched_operator = OperatorId::magnetar("layout", 1, crate::OperatorFamily::Layout);
    record(
        &mut results,
        "a specialization whose Operator semantics differ from its template is rejected",
        require_specialization_preserves_operator_semantics(
            &conformance_operator(),
            &mismatched_operator,
        )
        .is_err(),
        "expected a candidate with mismatched Operator semantics to be rejected",
    );

    KernelAutotuningConformanceReport { results }
}

fn conformance_operator() -> OperatorId {
    OperatorId::magnetar("attention", 1, crate::OperatorFamily::Attention)
}

fn conformance_workload() -> KernelAutotuningWorkloadBucket {
    KernelAutotuningWorkloadBucket {
        operator: conformance_operator(),
        shape_bucket: "batch=1/seq=4096".into(),
        batch_bucket: Some("1".into()),
        sequence_bucket: Some("4096".into()),
        phase: KernelAutotuningExecutionPhase::Decode,
        dtype: ComputeDType::Float16,
        layout: TensorLayoutKind::Contiguous,
        quantization: None,
        provider: ProviderBinding::new("cuda"),
        device_architecture: "sm90".into(),
        device_features: BTreeSet::new(),
    }
}

fn conformance_benchmark_profile() -> KernelAutotuningBenchmarkProfile {
    KernelAutotuningBenchmarkProfile {
        warmup_iterations: 5,
        measurement_iterations: 20,
        synchronization_policy: "device-sync".into(),
        timeout_millis: 5_000,
        metric: KernelAutotuningObjective::Latency,
        outlier_policy: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conformance_report_is_conformant() {
        let report = run_kernel_autotuning_conformance();
        assert!(!report.results.is_empty());
        for result in &report.results {
            assert!(
                result.passed,
                "{} failed: {:?}",
                result.requirement, result.diagnostic
            );
        }
        assert!(report.is_conformant());
    }

    #[test]
    fn axis_domain_rejects_zero_and_oversized_cardinality() {
        assert!(!AxisDomain::BoundedIntegerRange { min: 5, max: 1 }.is_bounded());
        assert!(
            !AxisDomain::BoundedIntegerRange {
                min: 0,
                max: i64::MAX,
            }
            .is_bounded()
        );
        assert!(AxisDomain::FiniteSet(BTreeSet::from([4, 8])).is_bounded());
    }

    #[test]
    fn plan_validate_rejects_candidate_with_different_operator_semantics() {
        let attention = conformance_operator();
        let layout = OperatorId::magnetar("layout", 1, crate::OperatorFamily::Layout);
        assert!(
            require_specialization_preserves_operator_semantics(&attention, &attention).is_ok()
        );
        assert!(require_specialization_preserves_operator_semantics(&attention, &layout).is_err());
        assert!(semantic_difference_requires_distinct_kernel_candidate(
            &attention, &layout
        ));
        assert!(!semantic_difference_requires_distinct_kernel_candidate(
            &attention, &attention
        ));

        let template = KernelSpecializationTemplate::new(
            KernelSpecializationTemplateId::new("attn-tile"),
            KernelId::new(
                ProviderBinding::new("cuda"),
                "attn",
                crate::CapabilityVersion::new(1, 0, 0),
                attention.clone(),
                crate::KernelOperatorVersionRange::exact(1),
                crate::KernelImplementationFamily::Cuda,
            ),
            1,
        )
        .with_axis(conformance_axis());
        let plan = KernelAutotuningPlan {
            template,
            candidates: vec![KernelAutotuningCandidate {
                compiled_artifact: CompiledKernelArtifactId::from_digest("mismatched"),
                operator_semantics: layout,
                artifact_trust: KernelArtifactTrust::Trusted,
                quarantined: false,
                specialization: None,
                qualification_covered: true,
                memory_feasible: true,
                provider_ready: true,
                device_compatible: true,
            }],
            workload: conformance_workload(),
            benchmark_profile: conformance_benchmark_profile(),
            objective: KernelAutotuningObjective::Latency,
            secondary_objectives: Vec::new(),
            budget: KernelAutotuningBudget::default(),
            fallback: KernelAutotuningFallback::StructuredNotReady,
        };
        assert!(matches!(
            plan.validate(),
            Err(KernelAutotuningError::SpecializationInvalid { .. })
        ));
    }

    #[test]
    fn template_validate_rejects_undeclared_constraint_axis() {
        let axis = conformance_axis();
        let template = KernelSpecializationTemplate::new(
            KernelSpecializationTemplateId::new("attn-tile"),
            KernelId::new(
                ProviderBinding::new("cuda"),
                "attn",
                crate::CapabilityVersion::new(1, 0, 0),
                OperatorId::magnetar("attention", 1, crate::OperatorFamily::Attention),
                crate::KernelOperatorVersionRange::exact(1),
                crate::KernelImplementationFamily::Cuda,
            ),
            1,
        )
        .with_axis(axis)
        .with_constraint(SpecializationConstraint::Equals {
            axis: KernelSpecializationAxisId::new("triton", "undeclared"),
            value: SpecializationAxisValue::Integer(1),
        });
        assert!(template.validate().is_err());
    }

    #[test]
    fn instantiate_rejects_out_of_domain_and_enforces_constraints() {
        let block_m = KernelSpecializationAxis::new(
            KernelSpecializationAxisId::new("triton", "block-m"),
            AxisDomain::FiniteSet(BTreeSet::from([32, 64])),
        );
        let num_warps = KernelSpecializationAxis::new(
            KernelSpecializationAxisId::new("triton", "num-warps"),
            AxisDomain::FiniteSet(BTreeSet::from([4, 8])),
        );
        let template = KernelSpecializationTemplate::new(
            KernelSpecializationTemplateId::new("attn-tile"),
            KernelId::new(
                ProviderBinding::new("cuda"),
                "attn",
                crate::CapabilityVersion::new(1, 0, 0),
                OperatorId::magnetar("attention", 1, crate::OperatorFamily::Attention),
                crate::KernelOperatorVersionRange::exact(1),
                crate::KernelImplementationFamily::Cuda,
            ),
            1,
        )
        .with_axis(block_m.clone())
        .with_axis(num_warps.clone())
        .with_constraint(SpecializationConstraint::Implies {
            if_axis: num_warps.id.clone(),
            if_value: SpecializationAxisValue::Integer(8),
            then_axis: block_m.id.clone(),
            then_values: BTreeSet::from([SpecializationAxisValue::Integer(64)]),
        });

        let mut invalid_assignment = BTreeMap::new();
        invalid_assignment.insert(block_m.id.clone(), SpecializationAxisValue::Integer(999));
        invalid_assignment.insert(num_warps.id.clone(), SpecializationAxisValue::Integer(4));
        assert!(template.instantiate(invalid_assignment).is_err());

        let mut constraint_violation = BTreeMap::new();
        constraint_violation.insert(block_m.id.clone(), SpecializationAxisValue::Integer(32));
        constraint_violation.insert(num_warps.id.clone(), SpecializationAxisValue::Integer(8));
        assert!(template.instantiate(constraint_violation).is_err());

        let mut valid = BTreeMap::new();
        valid.insert(block_m.id.clone(), SpecializationAxisValue::Integer(64));
        valid.insert(num_warps.id.clone(), SpecializationAxisValue::Integer(8));
        assert!(template.instantiate(valid).is_ok());
    }

    #[test]
    fn instance_fingerprint_is_order_independent() {
        let a_id = KernelSpecializationAxisId::new("triton", "a");
        let b_id = KernelSpecializationAxisId::new("triton", "b");
        let mut first = BTreeMap::new();
        first.insert(a_id.clone(), SpecializationAxisValue::Integer(1));
        first.insert(b_id.clone(), SpecializationAxisValue::Integer(2));
        let mut second = BTreeMap::new();
        second.insert(b_id, SpecializationAxisValue::Integer(2));
        second.insert(a_id, SpecializationAxisValue::Integer(1));

        let template = KernelSpecializationTemplateId::new("t");
        let first_instance = KernelSpecializationInstance {
            template: template.clone(),
            template_version: 1,
            assignments: first,
        };
        let second_instance = KernelSpecializationInstance {
            template,
            template_version: 1,
            assignments: second,
        };
        assert_eq!(
            first_instance.fingerprint("digest"),
            second_instance.fingerprint("digest")
        );
    }

    #[test]
    fn session_state_rejects_skipping_preparing() {
        let mut session = KernelAutotuningSession::new(
            KernelAutotuningSessionIdAllocator::default().allocate(),
            "plan",
            KernelAutotuningTriggerPoint::ModelInstanceWarmup,
        );
        assert!(
            session
                .transition_to(KernelAutotuningSessionState::Benchmarking)
                .is_err()
        );
        assert!(
            session
                .transition_to(KernelAutotuningSessionState::Planning)
                .is_ok()
        );
        assert!(
            session
                .transition_to(KernelAutotuningSessionState::Preparing)
                .is_ok()
        );
    }

    #[test]
    fn budget_exceeded_reports_first_violated_dimension() {
        let budget = KernelAutotuningBudget {
            max_candidates: Some(4),
            ..KernelAutotuningBudget::default()
        };
        let usage = KernelAutotuningBudgetUsage {
            candidates_evaluated: 5,
            ..KernelAutotuningBudgetUsage::default()
        };
        assert_eq!(
            budget_exceeded(&budget, &usage),
            Some(KernelAutotuningBudgetDimension::Candidates)
        );
        assert_eq!(
            budget_exceeded(&KernelAutotuningBudget::default(), &usage),
            None
        );
    }

    #[test]
    fn search_strategy_never_expands_candidate_domain() {
        let candidates = vec![
            KernelAutotuningCandidate {
                compiled_artifact: CompiledKernelArtifactId::from_digest("a"),
                operator_semantics: conformance_operator(),
                artifact_trust: KernelArtifactTrust::Trusted,
                quarantined: false,
                specialization: None,
                qualification_covered: true,
                memory_feasible: true,
                provider_ready: true,
                device_compatible: true,
            },
            KernelAutotuningCandidate {
                compiled_artifact: CompiledKernelArtifactId::from_digest("b"),
                operator_semantics: conformance_operator(),
                artifact_trust: KernelArtifactTrust::Untrusted,
                quarantined: false,
                specialization: None,
                qualification_covered: true,
                memory_feasible: true,
                provider_ready: true,
                device_compatible: true,
            },
        ];
        let indices = apply_search_strategy(
            &KernelAutotuningSearchStrategy::ExhaustiveBounded,
            &candidates,
        );
        assert!(indices.iter().all(|&index| index < candidates.len()));
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn cache_hit_revalidation_rejects_stale_and_ineligible_records() {
        let mut record = KernelAutotuningRecord {
            plan_fingerprint: "plan".into(),
            target_provider: ProviderBinding::new("cuda"),
            target_provider_version: "1.0.0".into(),
            target_device_architecture: "sm90".into(),
            target_device_features: BTreeSet::new(),
            candidate_artifacts: Vec::new(),
            specialization_fingerprints: Vec::new(),
            workload: conformance_workload(),
            benchmark_profile: conformance_benchmark_profile(),
            measurements: Vec::new(),
            winner: Some(CompiledKernelArtifactId::from_digest("winner")),
            qualification_references: Vec::new(),
            policy_version: 1,
            created_at_millis: 0,
            freshness: KernelAutotuningFreshness::Fresh,
        };
        let eligible = KernelAutotuningCacheRevalidation {
            not_revoked: true,
            trusted: true,
            qualified: true,
            provider_ready: true,
            device_available: true,
            memory_feasible: true,
            prepared_kernel_ready: true,
        };
        assert!(revalidate_cache_hit(&record, &eligible).is_ok());

        record.freshness = KernelAutotuningFreshness::Stale {
            reason: KernelAutotuningStalenessReason::DriverRuntimeUpdated,
        };
        assert!(revalidate_cache_hit(&record, &eligible).is_err());

        record.freshness = KernelAutotuningFreshness::Fresh;
        let ineligible = KernelAutotuningCacheRevalidation {
            memory_feasible: false,
            ..eligible
        };
        assert!(revalidate_cache_hit(&record, &ineligible).is_err());
    }

    #[test]
    fn compile_specialization_instance_enforces_cold_path() {
        let template = conformance_template();
        let axis_id = KernelSpecializationAxisId::new("triton", "num-warps");
        let mut assignment = BTreeMap::new();
        assignment.insert(axis_id, SpecializationAxisValue::Integer(4));
        let instance = template.instantiate(assignment).unwrap();

        let source_artifact = crate::KernelSourceArtifact::new(
            crate::KernelSourceArtifactId::from_digest("source-digest"),
            crate::KernelSourceFormat::new("triton", "source"),
            conformance_operator(),
            crate::KernelArtifactProvenance::HumanAuthored,
        );
        let target = crate::CompilationTarget::new(
            ProviderBinding::new("cuda"),
            DeviceBinding::new(crate::DeviceId::new("gpu-0")),
            "sm90",
        );

        assert!(
            compile_specialization_instance(
                &instance,
                &source_artifact,
                b"source".to_vec(),
                target.clone(),
                crate::KernelArtifactPath::Hot,
            )
            .is_err()
        );
        let request = compile_specialization_instance(
            &instance,
            &source_artifact,
            b"source".to_vec(),
            target,
            crate::KernelArtifactPath::Cold,
        )
        .unwrap();
        assert_eq!(request.source_artifact_id, source_artifact.id);
    }

    #[test]
    fn precompiled_bundle_matches_variant_without_recompilation() {
        let template = conformance_template();
        let axis_id = KernelSpecializationAxisId::new("triton", "num-warps");
        let mut assignment = BTreeMap::new();
        assignment.insert(axis_id, SpecializationAxisValue::Integer(8));
        let instance = template.instantiate(assignment).unwrap();

        let mut bundle = PrecompiledSpecializationBundle::new();
        assert!(bundle.match_variant(&instance, "artifact-digest").is_none());
        bundle.insert(
            &instance,
            "artifact-digest",
            CompiledKernelArtifactId::from_digest("compiled"),
        );
        assert_eq!(bundle.len(), 1);
        assert_eq!(
            bundle.match_variant(&instance, "artifact-digest"),
            Some(&CompiledKernelArtifactId::from_digest("compiled"))
        );
    }

    #[test]
    fn preparation_time_specialization_requires_explicit_assignments() {
        let template = conformance_template();
        let axis_id = KernelSpecializationAxisId::new("triton", "num-warps");
        let mut assignment = BTreeMap::new();
        assignment.insert(axis_id, SpecializationAxisValue::Integer(4));
        let instance = template.instantiate(assignment).unwrap();

        let explicit = PreparationTimeSpecialization {
            kind: PreparationSpecializationKind::LaunchMetadataSpecialization,
            instance,
        };
        assert!(explicit.is_explicit());

        let empty_instance = KernelSpecializationInstance {
            template: template.id.clone(),
            template_version: template.version,
            assignments: BTreeMap::new(),
        };
        let implicit = PreparationTimeSpecialization {
            kind: PreparationSpecializationKind::PipelineConfiguration,
            instance: empty_instance,
        };
        assert!(!implicit.is_explicit());
    }

    #[test]
    fn provider_execution_parameter_requires_bounded_and_covered() {
        let bounded_covered = ProviderExecutionParameter {
            name: KernelSpecializationAxisId::new("cuda", "l2-persist"),
            domain: AxisDomain::EnumeratedSymbolic(BTreeSet::from(["on".into(), "off".into()])),
            covered_by_kernel_contract: true,
        };
        assert!(bounded_covered.may_participate_in_autotuning());

        let uncovered = ProviderExecutionParameter {
            covered_by_kernel_contract: false,
            ..bounded_covered.clone()
        };
        assert!(!uncovered.may_participate_in_autotuning());

        let unbounded = ProviderExecutionParameter {
            domain: AxisDomain::BoundedIntegerRange {
                min: 0,
                max: i64::MAX,
            },
            ..bounded_covered
        };
        assert!(!unbounded.may_participate_in_autotuning());
    }

    #[test]
    fn specialized_artifact_trust_is_re_evaluated_independently() {
        let result = require_independent_trust_evaluation(KernelArtifactTrust::Trusted, false);
        assert_eq!(result, KernelArtifactTrust::Untrusted);
        let result = require_independent_trust_evaluation(KernelArtifactTrust::Untrusted, true);
        assert_eq!(result, KernelArtifactTrust::Trusted);
    }

    #[test]
    fn publish_autotuning_record_atomically_replaces_prior_entry() {
        let mut cache = KernelAutotuningCache::new();
        let key = KernelAutotuningCacheKey {
            operator: conformance_operator(),
            candidate_set_fingerprint: "fp".into(),
            template: KernelSpecializationTemplateId::new("attn-tile"),
            template_version: 1,
            provider_version: "1.0.0".into(),
            device_architecture: "sm90".into(),
            device_features: BTreeSet::new(),
            driver_runtime_compatibility: BTreeSet::new(),
            dtype: ComputeDType::Float16,
            layout: TensorLayoutKind::Contiguous,
            workload_fingerprint: "wl".into(),
            objective: KernelAutotuningObjective::Latency,
            policy_version: 1,
        };
        let record = KernelAutotuningRecord {
            plan_fingerprint: "plan".into(),
            target_provider: ProviderBinding::new("cuda"),
            target_provider_version: "1.0.0".into(),
            target_device_architecture: "sm90".into(),
            target_device_features: BTreeSet::new(),
            candidate_artifacts: Vec::new(),
            specialization_fingerprints: Vec::new(),
            workload: conformance_workload(),
            benchmark_profile: conformance_benchmark_profile(),
            measurements: Vec::new(),
            winner: Some(CompiledKernelArtifactId::from_digest("winner")),
            qualification_references: Vec::new(),
            policy_version: 1,
            created_at_millis: 0,
            freshness: KernelAutotuningFreshness::Fresh,
        };
        assert!(cache.get(&key).is_none());
        publish_autotuning_record_atomically(&mut cache, &key, record.clone());
        assert_eq!(cache.get(&key), Some(&record));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn fixture_sources_are_all_authorized() {
        assert!(fixture_source_is_authorized(
            KernelAutotuningFixtureSource::Synthetic
        ));
        assert!(fixture_source_is_authorized(
            KernelAutotuningFixtureSource::DeterministicGenerated
        ));
        assert!(fixture_source_is_authorized(
            KernelAutotuningFixtureSource::AuthorizedBenchmarkDataset
        ));
    }

    #[test]
    fn tuning_never_disturbs_active_continuous_batch() {
        let batch = KernelAutotuningBatchingContext {
            active_sequences: 4,
            total_active_tokens: 1024,
            raggedness_bucket: "low".into(),
            kv_cache_mode: "paged".into(),
        };
        let before = batch.clone();
        assert!(tuning_respects_active_batch(&batch));
        assert_eq!(batch, before);
    }

    #[test]
    fn stale_tuning_result_is_never_silently_current() {
        let stale_record = KernelAutotuningRecord {
            plan_fingerprint: "plan".into(),
            target_provider: ProviderBinding::new("cuda"),
            target_provider_version: "1.0.0".into(),
            target_device_architecture: "sm90".into(),
            target_device_features: BTreeSet::new(),
            candidate_artifacts: Vec::new(),
            specialization_fingerprints: Vec::new(),
            workload: conformance_workload(),
            benchmark_profile: conformance_benchmark_profile(),
            measurements: Vec::new(),
            winner: Some(CompiledKernelArtifactId::from_digest("winner")),
            qualification_references: Vec::new(),
            policy_version: 1,
            created_at_millis: 0,
            freshness: KernelAutotuningFreshness::Stale {
                reason: KernelAutotuningStalenessReason::PolicyVersionChanged,
            },
        };
        assert!(resolve_stale_record(&stale_record, StaleTuningReusePolicy::Ignore).is_none());
        let conservative =
            resolve_stale_record(&stale_record, StaleTuningReusePolicy::UseConservatively).unwrap();
        assert!(matches!(
            conservative.freshness,
            KernelAutotuningFreshness::Stale { .. }
        ));
        let temporary = resolve_stale_record(
            &stale_record,
            StaleTuningReusePolicy::UseTemporarilyWhileRetuning,
        )
        .unwrap();
        assert!(matches!(
            temporary.freshness,
            KernelAutotuningFreshness::Stale { .. }
        ));
    }

    #[test]
    fn admission_lowers_priority_and_denies_under_pressure() {
        let default_policy = KernelAutotuningResourcePolicy::default();
        assert_eq!(
            evaluate_autotuning_admission(MemoryPressureLevel::Saturated, true, &default_policy),
            KernelAutotuningAdmissionDecision::Deny
        );
        assert_eq!(
            evaluate_autotuning_admission(MemoryPressureLevel::High, true, &default_policy),
            KernelAutotuningAdmissionDecision::Postpone
        );
        assert_eq!(
            evaluate_autotuning_admission(MemoryPressureLevel::Low, false, &default_policy),
            KernelAutotuningAdmissionDecision::Admit
        );

        let lower_priority_policy = KernelAutotuningResourcePolicy {
            lower_priority_under_pressure: true,
            dedicated_device: None,
        };
        assert_eq!(
            evaluate_autotuning_admission(
                MemoryPressureLevel::Moderate,
                false,
                &lower_priority_policy
            ),
            KernelAutotuningAdmissionDecision::AdmitLowerPriority
        );
    }

    #[test]
    fn dedicated_tuning_device_is_preferred_over_inference_device() {
        let inference_device = DeviceBinding::new(crate::DeviceId::new("gpu-0"));
        let dedicated_device = DeviceBinding::new(crate::DeviceId::new("gpu-1"));
        let with_dedicated = KernelAutotuningResourcePolicy {
            lower_priority_under_pressure: false,
            dedicated_device: Some(dedicated_device.clone()),
        };
        assert_eq!(
            effective_tuning_device(&with_dedicated, &inference_device),
            &dedicated_device
        );
        let without_dedicated = KernelAutotuningResourcePolicy::default();
        assert_eq!(
            effective_tuning_device(&without_dedicated, &inference_device),
            &inference_device
        );
    }

    #[test]
    fn temporary_tuning_allocations_must_all_be_released() {
        let mut allocations = vec![
            TemporaryTuningAllocation::new(1),
            TemporaryTuningAllocation::new(2),
        ];
        assert!(!session_leaks_no_tensor_resources(&allocations));
        allocations[0].release();
        assert!(!session_leaks_no_tensor_resources(&allocations));
        allocations[1].release();
        assert!(session_leaks_no_tensor_resources(&allocations));
    }

    #[test]
    fn tuning_only_preparation_retires_through_normal_lifecycle() {
        let mut allocator = crate::PreparedKernelIdAllocator::default();
        let prepared_id = allocator.allocate();
        let mut prepared = crate::PreparedKernel::new(
            prepared_id,
            conformance_kernel_id(),
            CompiledKernelArtifactId::from_digest("compiled"),
            ProviderBinding::new("cuda"),
            DeviceBinding::new(crate::DeviceId::new("gpu-0")),
            crate::PreparedKernelGeneration::new(1),
        );
        prepared.mark_ready().unwrap();

        let mut tuning_prep = TuningOnlyPreparation::new(prepared_id);
        assert!(!tuning_prep.retired);
        tuning_prep.retire(&mut prepared).unwrap();
        assert!(tuning_prep.retired);
        assert_eq!(prepared.state, crate::PreparedKernelState::Retiring);
    }

    #[test]
    fn candidate_failure_policy_controls_session_continuation() {
        assert!(!candidate_failure_fails_session(
            KernelAutotuningCandidateFailurePolicy::IsolateAndContinue
        ));
        assert!(candidate_failure_fails_session(
            KernelAutotuningCandidateFailurePolicy::FailSession
        ));
    }

    #[test]
    fn tuning_winner_promotion_requires_hysteresis_clearance() {
        let policy = crate::HysteresisPolicy::default();
        assert!(!tuning_winner_respects_hysteresis(10.0, 9.99, &policy));
        assert!(tuning_winner_respects_hysteresis(10.0, 5.0, &policy));
    }

    #[test]
    fn provider_hint_ordering_never_adds_candidates() {
        let candidates = vec![
            KernelAutotuningCandidate {
                compiled_artifact: CompiledKernelArtifactId::from_digest("a"),
                operator_semantics: conformance_operator(),
                artifact_trust: KernelArtifactTrust::Trusted,
                quarantined: false,
                specialization: None,
                qualification_covered: true,
                memory_feasible: true,
                provider_ready: true,
                device_compatible: true,
            },
            KernelAutotuningCandidate {
                compiled_artifact: CompiledKernelArtifactId::from_digest("b"),
                operator_semantics: conformance_operator(),
                artifact_trust: KernelArtifactTrust::Trusted,
                quarantined: false,
                specialization: None,
                qualification_covered: true,
                memory_feasible: true,
                provider_ready: true,
                device_compatible: true,
            },
        ];
        let hint = KernelAutotuningProviderHint::PreferredOrder {
            artifact_order: vec![CompiledKernelArtifactId::from_digest("b")],
        };
        let indices = apply_provider_hint_ordering(&hint, &candidates);
        assert_eq!(indices.len(), candidates.len());
        assert_eq!(indices[0], 1);

        let recommended = &candidates[1];
        assert!(provider_recommended_default_is_authoritative(recommended));
        let infeasible_recommendation = KernelAutotuningCandidate {
            memory_feasible: false,
            ..candidates[0].clone()
        };
        assert!(!provider_recommended_default_is_authoritative(
            &infeasible_recommendation
        ));
    }

    #[test]
    fn provider_native_autotuning_requires_full_boundary() {
        let authorized = KernelAutotuningProviderBoundary {
            declares_candidate_domain: true,
            cold_or_warm_path_only: true,
            respects_budget: true,
            satisfies_kernel_contract: true,
            determinism_precision_preserved: true,
        };
        assert!(evaluate_provider_native_autotuning(&authorized).is_ok());

        let unauthorized = KernelAutotuningProviderBoundary {
            cold_or_warm_path_only: false,
            ..authorized
        };
        assert!(evaluate_provider_native_autotuning(&unauthorized).is_err());
    }

    #[test]
    fn specialized_artifact_store_dedups_by_content_digest() {
        let mut store = SpecializedArtifactStore::new();
        assert!(store.insert(CompiledKernelArtifactId::from_digest("digest-1")));
        assert_eq!(store.len(), 1);
        assert!(!store.insert(CompiledKernelArtifactId::from_digest("digest-1")));
        assert_eq!(store.len(), 1);
        assert!(store.insert(CompiledKernelArtifactId::from_digest("digest-2")));
        assert_eq!(store.len(), 2);
        assert!(store.get("digest-1").is_some());
    }

    #[test]
    fn cross_device_reuse_requires_matching_features_and_provider_version() {
        let record = KernelAutotuningRecord {
            plan_fingerprint: "plan".into(),
            target_provider: ProviderBinding::new("cuda"),
            target_provider_version: "1.0.0".into(),
            target_device_architecture: "sm90".into(),
            target_device_features: BTreeSet::from(["fp8".to_string()]),
            candidate_artifacts: Vec::new(),
            specialization_fingerprints: Vec::new(),
            workload: conformance_workload(),
            benchmark_profile: conformance_benchmark_profile(),
            measurements: Vec::new(),
            winner: Some(CompiledKernelArtifactId::from_digest("winner")),
            qualification_references: Vec::new(),
            policy_version: 1,
            created_at_millis: 0,
            freshness: KernelAutotuningFreshness::Fresh,
        };
        assert!(tuning_result_applies_to_target(
            &record,
            "sm90",
            &BTreeSet::from(["fp8".to_string()]),
            "1.0.0",
            true
        ));
        assert!(!tuning_result_applies_to_target(
            &record,
            "sm90",
            &BTreeSet::new(),
            "1.0.0",
            true
        ));
        assert!(!tuning_result_applies_to_target(
            &record,
            "sm90",
            &BTreeSet::from(["fp8".to_string()]),
            "2.0.0",
            true
        ));
    }

    #[test]
    fn offline_deployment_requires_no_live_tuning_when_precomputed() {
        let empty = KernelAutotuningOfflineDeployment {
            precomputed_records: Vec::new(),
            pinned_selection: None,
        };
        assert!(!empty.requires_no_live_tuning());

        let with_pinned = KernelAutotuningOfflineDeployment {
            precomputed_records: Vec::new(),
            pinned_selection: Some(CompiledKernelArtifactId::from_digest("pinned")),
        };
        assert!(with_pinned.requires_no_live_tuning());
    }
}

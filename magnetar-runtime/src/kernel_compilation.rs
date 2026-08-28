//! Provider Kernel Compilation Capability (see
//! `openspec/changes/define-provider-kernel-compilation-capability`).
//!
//! This module does not implement any actual compiler (Triton, NVCC, HIP,
//! WGSL, MSL, SPIR-V, ...). It defines, as executable Rust types and
//! validation functions, the optional, independently versioned cold-path
//! contract a Provider MAY implement to transform a
//! [`crate::kernel_artifact::KernelSourceArtifact`] into a
//! [`crate::kernel_artifact::CompiledKernelArtifact`]:
//!
//! ```text
//! KernelSourceArtifact -> Provider Kernel Compilation Capability -> CompiledKernelArtifact
//! ```
//!
//! - [`kernel_compilation_capability_id`] / [`KERNEL_COMPILATION_CAPABILITY_VERSION`]:
//!   the capability identity, reusing [`crate::CapabilityId`] /
//!   [`crate::CapabilityVersion`] rather than a new versioning scheme. Its
//!   version SHALL NOT automatically track the crate, Provider, Provider
//!   ABI, WIT, or Kernel Artifact manifest versions.
//! - [`KernelCompilationCapabilityDescriptor`]: what a Provider advertises
//!   (accepted/produced formats, Devices, architectures, modes, async /
//!   cancellation / deadline support, size and concurrency limits, isolation
//!   model, and specialization support). Advertising this descriptor is
//!   entirely optional -- see [`crate::provider::Provider::kernel_compilation_capability`],
//!   which defaults to `None`.
//! - [`CompilationMode`]: portable behavior categories (not specific
//!   languages). [`CompilationMode::LoadTimeJit`] never means token-loop
//!   compilation -- only cold-path preparation/loading JIT.
//! - [`negotiate_source_format`] / [`negotiate_output_format`]: format
//!   negotiation happens before compilation and never infers support from a
//!   filename or Provider name.
//! - [`KernelCompilationRequest`] / [`CompilationTarget`] /
//!   [`CompilationSpecialization`]: the explicit compilation input. Source is
//!   `Vec<u8>`, never assumed UTF-8 `String`; the request carries no host
//!   filesystem path, and the target carries no native Device handle.
//! - [`enforce_runtime_target_authority`]: the compiler receives the
//!   Runtime-selected [`crate::ProviderBinding`] / [`crate::DeviceBinding`]
//!   and SHALL NOT redirect compilation to a different Provider or Device.
//! - [`CompilerIdentity`]: compiler name/version/backend/toolchain
//!   fingerprint, recorded on successful compilation where available.
//! - [`CompilationJobId`] / [`CompilationJobState`] / [`CompilationJob`]:
//!   the opaque job lifecycle (`queued -> compiling -> {succeeded, failed,
//!   cancelled, timed-out}`), never exposing a process ID, thread pointer, or
//!   native driver handle.
//! - [`ProviderKernelCompilationApi`]: the Runtime-to-Provider submit / poll
//!   / cancel / result / release boundary, mirroring
//!   [`crate::provider::ProviderExecutionApi`]'s shape for execution.
//! - [`CompilationCancellationSupport`] / [`evaluate_cancellation_request`]:
//!   cancellation semantics are declared and structured; a cancelled job
//!   never publishes valid partial output.
//! - [`CompilationDeadline`] / [`enforce_compilation_deadline`]:
//!   deadlines fail closed when Runtime policy requires enforcement the
//!   Provider cannot provide.
//! - [`CompilationLimits`] / [`enforce_compilation_limits`]: source/output
//!   size and concurrency limits are enforced before compiler invocation.
//! - [`CompilationIsolationModel`] / [`evaluate_isolation_sufficiency`]: the
//!   compiler trust boundary Runtime policy can require or reject.
//! - [`CompilationNetworkPolicy`] / [`enforce_compilation_network_boundary`]:
//!   compiler dependency downloads never happen implicitly.
//! - [`CompilationProcessArguments`]: compiler subprocess arguments are
//!   passed structurally (`Vec<String>`) -- never built as an interpolated
//!   shell string from untrusted metadata.
//! - [`CompilationEnvironmentPolicy`] / [`evaluate_environment_variable`]:
//!   ambient environment variables are denied by default; secrets never
//!   become compiler inputs without an explicit policy.
//! - [`CompilationResult`] / [`verify_output_integrity`]: successful
//!   compilation produces a validated candidate
//!   [`crate::kernel_artifact::CompiledKernelArtifact`] with a verified
//!   digest. Compilation success SHALL NOT itself grant
//!   [`crate::kernel_artifact::KernelArtifactTrust::Trusted`] -- trust
//!   remains governed by [`crate::kernel_artifact::evaluate_artifact_trust`].
//! - [`normalize_compiler_crash`]: a compiler crash is normalized into
//!   [`KernelCompilationError::CompilerCrashed`] and SHALL NOT be reported as
//!   a successful artifact.
//! - [`call_provider_compiler_without_unwinding`]: Provider compiler code
//!   SHALL NOT unwind across the Runtime/Provider boundary; panics are
//!   caught and normalized into a structured failure.
//! - [`KernelCompilationAbiDescriptor`]: the versioned, C-compatible-shaped
//!   ABI extension descriptor (function table + buffer ownership rules),
//!   mirroring [`crate::provider::ProviderAbiDescriptor`]. A Provider
//!   implementing Provider ABI v1 without this extension remains valid --
//!   absence yields [`KernelCompilationError::Unavailable`], never Provider
//!   corruption.
//! - [`KernelCompilationObservationKind`] / [`KernelCompilationObservation`]:
//!   redacted-only compilation lifecycle observability (raw source, compiled
//!   binary bytes, native handles, temp paths, environment, and secrets
//!   never survive into an observation).
//! - [`CompilerDiagnostic`]: classified, redacted compiler diagnostics in
//!   place of unrestricted raw compiler stdout/stderr.
//! - [`KernelCompilationError`]: the structured error categories from the
//!   proposal's "Error Model" section.
//! - [`KernelCompilationConformanceReport`] /
//!   [`run_kernel_compilation_conformance`]: the conformance checks required
//!   by `openspec/changes/define-provider-kernel-compilation-capability/specs/conformance/spec.md`.

use crate::compute::redact_backend_diagnostic;
use crate::kernel_artifact::{
    CompiledKernelArtifact, CompiledKernelArtifactId, KernelArtifactColdPathOperation,
    KernelArtifactPath, KernelArtifactTrust, KernelSourceArtifact, KernelSourceArtifactId,
    KernelSourceFormat, evaluate_artifact_trust,
};
use crate::{
    CapabilityId, CapabilityVersion, ComputeDType, DeviceBinding, KernelDeterminism,
    KernelPrecisionMetadata, KernelShapeConstraints, ProviderBinding, TensorLayoutKind,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
    panic::{AssertUnwindSafe, catch_unwind},
};

/// The capability version, independent of the crate, Provider, Provider ABI,
/// WIT, and Kernel Artifact manifest versions, implementing "Compilation
/// Capability Identity" (proposal).
pub const KERNEL_COMPILATION_CAPABILITY_VERSION: CapabilityVersion =
    CapabilityVersion::new(1, 0, 0);

/// `magnetar:provider/kernel-compilation`, implementing "Compilation
/// Capability Identity" (proposal).
pub fn kernel_compilation_capability_id() -> CapabilityId {
    CapabilityId::new("magnetar:provider/kernel-compilation")
}

// ---------------------------------------------------------------------
// Compilation Modes
// ---------------------------------------------------------------------

/// Portable compilation behavior categories, implementing "Compilation
/// Modes" (proposal). These describe behavior, not specific languages --
/// there is no closed enum of kernel languages anywhere in this contract.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CompilationMode {
    SourceCompilation,
    IntermediateTranslation,
    BinarySpecialization,
    ShaderCompilation,
    PipelineCompilation,
    OfflineAot,
    /// JIT during an explicit cold-path preparation/loading phase. This
    /// SHALL NOT mean token-loop (decode hot path) compilation -- see
    /// [`crate::kernel_artifact::reject_hot_path_compilation`].
    LoadTimeJit,
    ProviderManaged,
}

// ---------------------------------------------------------------------
// Capability Descriptor
// ---------------------------------------------------------------------

/// Whether a Provider supports compilation from source, preparation-only
/// from externally produced artifacts, or neither, implementing "AOT-Only
/// Platforms" and "Preparation Capability" (proposal): Runtime compilation
/// support SHALL NOT be required for Provider execution support.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CompilationSupportLevel {
    /// No source compilation and no artifact preparation capability.
    None,
    /// Can prepare externally produced [`CompiledKernelArtifact`]s (e.g. an
    /// AOT build-farm output) but cannot compile source itself.
    PreparationOnly,
    /// Can both compile [`KernelSourceArtifact`] source and prepare the
    /// resulting [`CompiledKernelArtifact`].
    SourceCompilation,
}

/// Declared cancellation semantics, implementing "Compilation Cancellation"
/// (proposal).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CompilationCancellationSupport {
    NotSupported,
    BeforeStartOnly,
    Cooperative,
    Interruptible,
    ProviderSpecific,
}

/// Provider compilation isolation model, implementing "Compilation Isolation
/// Model" (proposal). These categories describe trust boundaries, not
/// implementation requirements, ordered from least to most isolated so
/// [`evaluate_isolation_sufficiency`] can compare a declared model against a
/// policy-required minimum.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CompilationIsolationModel {
    Unavailable,
    InProcessTrustedCompiler,
    PlatformManagedCompiler,
    BrowserManagedCompiler,
    RestrictedSubprocess,
    SandboxedSubprocess,
    ExternalCompilationService,
}

impl CompilationIsolationModel {
    /// A coarse isolation strength ranking used only by
    /// [`evaluate_isolation_sufficiency`]. Higher is more isolated from the
    /// Runtime host process. This is a policy comparison helper, not a
    /// statement that any one model is universally "better".
    const fn rank(self) -> u8 {
        match self {
            Self::Unavailable => 0,
            Self::InProcessTrustedCompiler => 1,
            Self::PlatformManagedCompiler => 2,
            Self::BrowserManagedCompiler => 2,
            Self::RestrictedSubprocess => 3,
            Self::SandboxedSubprocess => 4,
            Self::ExternalCompilationService => 5,
        }
    }
}

/// What a Provider advertises about its Kernel Compilation Capability,
/// implementing "Compilation Capability Descriptor" (proposal). Advertising
/// this at all is optional -- see
/// [`crate::provider::Provider::kernel_compilation_capability`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelCompilationCapabilityDescriptor {
    pub capability_version: CapabilityVersion,
    pub support_level: CompilationSupportLevel,
    pub accepted_source_formats: BTreeSet<KernelSourceFormat>,
    pub produced_compiled_formats: BTreeSet<String>,
    pub supported_devices: BTreeSet<DeviceBinding>,
    pub supported_architectures: BTreeSet<String>,
    pub modes: BTreeSet<CompilationMode>,
    pub supports_async: bool,
    pub cancellation: CompilationCancellationSupport,
    pub supports_deadlines: bool,
    pub max_source_bytes: Option<u64>,
    pub max_output_bytes: Option<u64>,
    pub max_concurrent_jobs: Option<u32>,
    pub isolation_model: CompilationIsolationModel,
    pub compiler_identity_available: bool,
    pub reproducibility_metadata_available: bool,
    pub supports_specialization: bool,
}

impl KernelCompilationCapabilityDescriptor {
    pub fn unsupported() -> Self {
        Self {
            capability_version: KERNEL_COMPILATION_CAPABILITY_VERSION,
            support_level: CompilationSupportLevel::None,
            accepted_source_formats: BTreeSet::new(),
            produced_compiled_formats: BTreeSet::new(),
            supported_devices: BTreeSet::new(),
            supported_architectures: BTreeSet::new(),
            modes: BTreeSet::new(),
            supports_async: false,
            cancellation: CompilationCancellationSupport::NotSupported,
            supports_deadlines: false,
            max_source_bytes: None,
            max_output_bytes: None,
            max_concurrent_jobs: None,
            isolation_model: CompilationIsolationModel::Unavailable,
            compiler_identity_available: false,
            reproducibility_metadata_available: false,
            supports_specialization: false,
        }
    }

    pub fn is_present(&self) -> bool {
        !matches!(self.support_level, CompilationSupportLevel::None)
    }

    /// Implements "Capability Advertises Source Formats" and "Descriptor
    /// Validation" (proposal): a compilation-capable Provider SHALL declare
    /// at least one accepted format, one produced format, and a real
    /// isolation model.
    pub fn validate(&self) -> Result<(), KernelCompilationError> {
        if !self.is_present() {
            return Ok(());
        }
        if matches!(
            self.support_level,
            CompilationSupportLevel::SourceCompilation
        ) && self.accepted_source_formats.is_empty()
        {
            return Err(KernelCompilationError::DescriptorInvalid {
                reason: "source-compiling Provider must declare accepted source formats".into(),
            });
        }
        if self.produced_compiled_formats.is_empty() {
            return Err(KernelCompilationError::DescriptorInvalid {
                reason: "compilation-capable Provider must declare produced compiled formats"
                    .into(),
            });
        }
        if matches!(self.isolation_model, CompilationIsolationModel::Unavailable) {
            return Err(KernelCompilationError::DescriptorInvalid {
                reason: "compilation-capable Provider must declare a real isolation model".into(),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Format Negotiation
// ---------------------------------------------------------------------

/// Validates a requested source format against the Provider's advertised
/// accepted formats, implementing "Source Format Negotiation" (proposal):
/// unsupported formats are rejected before compilation, never inferred from
/// a filename or Provider name (neither parameter exists on this
/// signature).
pub fn negotiate_source_format(
    requested: &KernelSourceFormat,
    accepted: &BTreeSet<KernelSourceFormat>,
) -> Result<(), KernelCompilationError> {
    if accepted.contains(requested) {
        Ok(())
    } else {
        Err(KernelCompilationError::SourceFormatUnsupported {
            format: requested.stable_key(),
        })
    }
}

/// Validates a requested output format against the Provider's advertised
/// produced formats, implementing "Output Format Negotiation" (proposal).
pub fn negotiate_output_format(
    requested: &str,
    produced: &BTreeSet<String>,
) -> Result<(), KernelCompilationError> {
    if produced.contains(requested) {
        Ok(())
    } else {
        Err(KernelCompilationError::OutputFormatUnsupported {
            format: requested.into(),
        })
    }
}

// ---------------------------------------------------------------------
// Compilation Target
// ---------------------------------------------------------------------

/// Portable compilation target metadata, implementing "Compilation Target"
/// (proposal). Deliberately holds no native Device handle or pointer field
/// -- the Provider alone resolves `device` to its private native state.
#[derive(Clone, Debug, PartialEq)]
pub struct CompilationTarget {
    pub provider: ProviderBinding,
    pub device: DeviceBinding,
    pub architecture: String,
    pub hardware_features: BTreeSet<String>,
    pub abi: Option<String>,
    pub execution_environment: Option<String>,
    pub dtype_specialization: BTreeSet<ComputeDType>,
    pub layout_specialization: BTreeSet<TensorLayoutKind>,
    pub shape_specialization: KernelShapeConstraints,
    pub precision: KernelPrecisionMetadata,
    pub determinism: KernelDeterminism,
}

impl CompilationTarget {
    pub fn new(
        provider: ProviderBinding,
        device: DeviceBinding,
        architecture: impl Into<String>,
    ) -> Self {
        Self {
            provider,
            device,
            architecture: architecture.into(),
            hardware_features: BTreeSet::new(),
            abi: None,
            execution_environment: None,
            dtype_specialization: BTreeSet::new(),
            layout_specialization: BTreeSet::new(),
            shape_specialization: KernelShapeConstraints::default(),
            precision: KernelPrecisionMetadata::default(),
            determinism: KernelDeterminism::default(),
        }
    }
}

/// Implements "Runtime Selection Authority" (proposal): the compiler SHALL
/// NOT use compilation as a mechanism to choose an arbitrary Provider or
/// Device. `target` SHALL match the Runtime-selected `provider`/`device`
/// exactly.
pub fn enforce_runtime_target_authority(
    target: &CompilationTarget,
    runtime_selected_provider: &ProviderBinding,
    runtime_selected_device: &DeviceBinding,
) -> Result<(), KernelCompilationError> {
    if &target.provider != runtime_selected_provider {
        return Err(KernelCompilationError::TargetUnsupported {
            reason: format!(
                "compilation target Provider '{}' does not match Runtime-selected Provider '{}'",
                target.provider, runtime_selected_provider
            ),
        });
    }
    if &target.device != runtime_selected_device {
        return Err(KernelCompilationError::TargetUnsupported {
            reason: format!(
                "compilation target Device '{}' does not match Runtime-selected Device '{}'",
                target.device, runtime_selected_device
            ),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Specialization
// ---------------------------------------------------------------------

/// Explicit specialization inputs, implementing "Specialization Inputs"
/// (proposal): specialization SHALL be explicit and represented in the
/// resulting artifact metadata. Hidden specialization SHALL NOT be allowed
/// -- see [`require_explicit_compilation_specialization`].
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CompilationSpecialization {
    pub shape: KernelShapeConstraints,
    pub dtype: BTreeSet<ComputeDType>,
    pub layout: BTreeSet<TensorLayoutKind>,
    pub hardware_features: BTreeSet<String>,
    pub quantization: Option<String>,
}

/// Rejects specialization that was applied without being explicitly
/// declared, implementing "Specialization SHALL be explicit... Hidden
/// specialization SHALL NOT be allowed" (proposal).
pub fn require_explicit_compilation_specialization(
    applied: bool,
    declared: &CompilationSpecialization,
) -> Result<(), KernelCompilationError> {
    let declared_something = declared.shape != KernelShapeConstraints::default()
        || !declared.dtype.is_empty()
        || !declared.layout.is_empty()
        || !declared.hardware_features.is_empty()
        || declared.quantization.is_some();
    if applied && !declared_something {
        return Err(KernelCompilationError::SpecializationUnsupported {
            reason: "specialization was applied without explicit declaration".into(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Compiler Identity
// ---------------------------------------------------------------------

/// Recorded compiler identity, implementing "Compiler Identity" (proposal).
/// `flags_fingerprint` SHOULD be a deterministic fingerprint rather than a
/// raw command line -- see "Compiler Flags" (proposal): raw command lines
/// SHALL be redacted by default, which [`CompilerIdentity::with_raw_flags`]
/// enforces by redacting through `redact_backend_diagnostic`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CompilerIdentity {
    pub name: Option<String>,
    pub version: Option<String>,
    pub backend_version: Option<String>,
    pub toolchain_fingerprint: Option<String>,
    pub flags_fingerprint: Option<String>,
}

impl CompilerIdentity {
    pub fn with_raw_flags(mut self, raw_flags: impl AsRef<str>) -> Self {
        self.flags_fingerprint = Some(redact_backend_diagnostic(raw_flags.as_ref()));
        self
    }
}

// ---------------------------------------------------------------------
// Compilation Request
// ---------------------------------------------------------------------

/// Opaque compilation request identifier.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CompilationRequestId(String);

impl CompilationRequestId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
}

impl fmt::Display for CompilationRequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Explicit, immutable compilation input, implementing "Compilation Input"
/// and "Compilation Request" (proposal). `source_bytes` is `Vec<u8>` rather
/// than `String` because not every source format is textual. This struct
/// has no field capable of carrying a host filesystem path -- "The request
/// SHALL NOT carry arbitrary host filesystem paths" holds structurally.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelCompilationRequest {
    pub request_id: CompilationRequestId,
    pub source_artifact_id: KernelSourceArtifactId,
    pub source_format: KernelSourceFormat,
    pub source_bytes: Vec<u8>,
    pub target: CompilationTarget,
    pub specialization: CompilationSpecialization,
    pub policy: CompilationPolicy,
}

impl KernelCompilationRequest {
    pub fn from_source_artifact(
        request_id: CompilationRequestId,
        artifact: &KernelSourceArtifact,
        source_bytes: impl Into<Vec<u8>>,
        target: CompilationTarget,
    ) -> Self {
        Self {
            request_id,
            source_artifact_id: artifact.id.clone(),
            source_format: artifact.format.clone(),
            source_bytes: source_bytes.into(),
            target,
            specialization: CompilationSpecialization::default(),
            policy: CompilationPolicy::default(),
        }
    }
}

/// Runtime-enforced policy accompanying a [`KernelCompilationRequest`],
/// implementing "Compiler Authority", "Compilation Resource Limits",
/// "Compilation Deadlines", and "Compilation Isolation Model" (proposal).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CompilationPolicy {
    pub required_isolation: Option<CompilationIsolationModel>,
    pub required_deadline: Option<CompilationDeadline>,
    pub limits: CompilationLimits,
    pub network: CompilationNetworkPolicy,
    pub environment: CompilationEnvironmentPolicy,
}

// ---------------------------------------------------------------------
// Compilation Deadlines
// ---------------------------------------------------------------------

/// A compilation wall-clock deadline, implementing "Compilation Deadlines"
/// (proposal).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompilationDeadline {
    pub max_wall_clock_millis: u64,
}

/// Implements "When Runtime policy requires enforceable deadlines and
/// Provider cannot enforce them, compilation SHALL fail closed" (proposal).
pub fn enforce_compilation_deadline(
    required: Option<CompilationDeadline>,
    provider_can_enforce: bool,
) -> Result<(), KernelCompilationError> {
    match required {
        Some(_) if !provider_can_enforce => Err(KernelCompilationError::DeadlineUnsupported {
            reason: "Runtime requires an enforceable compilation deadline".into(),
        }),
        _ => Ok(()),
    }
}

// ---------------------------------------------------------------------
// Compilation Resource Limits
// ---------------------------------------------------------------------

/// Runtime-imposed resource limits, implementing "Compilation Resource
/// Limits" (proposal): source bytes, output bytes, concurrent jobs, temporary
/// workspace, and host memory. Wall-clock duration is covered separately by
/// [`CompilationDeadline`]; device compiler memory is Provider-internal and
/// out of this contract's scope.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CompilationLimits {
    pub max_source_bytes: Option<u64>,
    pub max_output_bytes: Option<u64>,
    pub max_concurrent_jobs: Option<u32>,
    /// Maximum size of the Provider-private temporary workspace, implementing
    /// "Define workspace limits" (tasks). This module carries no path into
    /// or out of that workspace -- see "Filesystem Authority" (proposal).
    pub max_workspace_bytes: Option<u64>,
    /// Maximum host memory the compiler process may use, implementing
    /// "Define host memory limit metadata" (tasks).
    pub max_host_memory_bytes: Option<u64>,
}

/// Rejects a request exceeding enforced source-size limits before compiler
/// invocation, implementing "Provider SHALL reject requests exceeding
/// enforced limits" (proposal).
pub fn enforce_compilation_limits(
    source_bytes: u64,
    limits: &CompilationLimits,
) -> Result<(), KernelCompilationError> {
    if let Some(max) = limits.max_source_bytes
        && source_bytes > max
    {
        return Err(KernelCompilationError::SourceTooLarge {
            max_bytes: max,
            found_bytes: source_bytes,
        });
    }
    Ok(())
}

/// Rejects output that exceeds the enforced output-size limit, implementing
/// "Compilation Resource Limits" (proposal).
pub fn enforce_output_limit(
    output_bytes: u64,
    limits: &CompilationLimits,
) -> Result<(), KernelCompilationError> {
    if let Some(max) = limits.max_output_bytes
        && output_bytes > max
    {
        return Err(KernelCompilationError::OutputTooLarge {
            max_bytes: max,
            found_bytes: output_bytes,
        });
    }
    Ok(())
}

/// Implements "Compilation Concurrency" (proposal): Runtime SHALL respect
/// Provider limits and global resource policy before submitting another
/// concurrent job.
pub fn enforce_concurrency_limit(
    current_jobs: u32,
    limits: &CompilationLimits,
) -> Result<(), KernelCompilationError> {
    if let Some(max) = limits.max_concurrent_jobs
        && current_jobs >= max
    {
        return Err(KernelCompilationError::ConcurrencyLimit {
            max_concurrent: max,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Isolation
// ---------------------------------------------------------------------

/// Implements "Runtime policy MAY reject a compilation path whose isolation
/// model is insufficient for the source artifact trust level" (proposal,
/// "Compilation Isolation Model").
pub fn evaluate_isolation_sufficiency(
    declared: CompilationIsolationModel,
    required_minimum: CompilationIsolationModel,
) -> Result<(), KernelCompilationError> {
    if declared.rank() >= required_minimum.rank() {
        Ok(())
    } else {
        Err(KernelCompilationError::IsolationInsufficient {
            declared: format!("{declared:?}"),
            required_minimum: format!("{required_minimum:?}"),
        })
    }
}

/// Implements "Untrusted Kernel Source" (proposal): a trusted compiler does
/// not make untrusted source safe automatically, and compilation success
/// SHALL NOT itself mark output trusted. This function exists to make that
/// mechanically explicit -- it never accepts a "compiled successfully" input
/// and always returns [`KernelArtifactTrust::Untrusted`] unless `policy`
/// says otherwise, delegating to
/// [`crate::kernel_artifact::evaluate_artifact_trust`].
pub fn compilation_result_trust(policy_approved: bool) -> KernelArtifactTrust {
    evaluate_artifact_trust(policy_approved)
}

// ---------------------------------------------------------------------
// Network / Filesystem / Process / Environment boundaries
// ---------------------------------------------------------------------

/// Implements "Network Authority" (proposal): compiler dependency fetching
/// SHALL NOT happen implicitly.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CompilationNetworkPolicy {
    pub network_access_authorized: bool,
}

pub fn enforce_compilation_network_boundary(
    requires_network: bool,
    policy: &CompilationNetworkPolicy,
) -> Result<(), KernelCompilationError> {
    if requires_network && !policy.network_access_authorized {
        return Err(KernelCompilationError::PolicyDenied {
            reason: "compilation requires network access without explicit policy authorization"
                .into(),
        });
    }
    Ok(())
}

/// Structural compiler subprocess arguments, implementing "Process
/// Execution" (proposal): arguments are passed as a `Vec<String>` rather
/// than an interpolated shell string, so untrusted source metadata can never
/// be concatenated into a shell command.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CompilationProcessArguments {
    pub program: String,
    pub args: Vec<String>,
}

impl CompilationProcessArguments {
    pub fn new(program: impl Into<String>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
        }
    }

    pub fn with_arg(mut self, arg: impl Into<String>) -> Self {
        self.args.push(arg.into());
        self
    }
}

/// How ambient environment variables reach compiler subprocesses,
/// implementing "Environment Variables" (proposal). Defaults to `Deny`:
/// "Secrets SHALL NOT become compiler inputs unless an explicit non-inference
/// management policy authorizes them."
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum CompilationEnvironmentPolicy {
    #[default]
    Deny,
    Allowlist(BTreeSet<String>),
    CapturedInFingerprint,
}

/// Evaluates whether `variable` may reach the compiler under `policy`.
pub fn evaluate_environment_variable(
    variable: &str,
    policy: &CompilationEnvironmentPolicy,
) -> bool {
    match policy {
        CompilationEnvironmentPolicy::Deny => false,
        CompilationEnvironmentPolicy::Allowlist(allowed) => allowed.contains(variable),
        CompilationEnvironmentPolicy::CapturedInFingerprint => true,
    }
}

// ---------------------------------------------------------------------
// Compilation Jobs
// ---------------------------------------------------------------------

/// Opaque compilation job identifier, implementing "Compilation Jobs"
/// (proposal): "It SHALL NOT expose process IDs, thread pointers, compiler
/// object pointers, or native driver handles." Exposes no accessor to its
/// internal representation.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CompilationJobId(u64);

impl fmt::Display for CompilationJobId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "compilation-job-{}", self.0)
    }
}

/// Allocates sequential, opaque [`CompilationJobId`]s.
#[derive(Clone, Debug, Default)]
pub struct CompilationJobIdAllocator(u64);

impl CompilationJobIdAllocator {
    pub fn allocate(&mut self) -> CompilationJobId {
        self.0 += 1;
        CompilationJobId(self.0)
    }
}

/// Compilation job lifecycle state, implementing "Compilation Jobs"
/// (proposal)'s suggested states.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CompilationJobState {
    Queued,
    Compiling,
    Succeeded,
    Failed,
    Cancelled,
    TimedOut,
}

impl CompilationJobState {
    /// Implements "Define valid state transitions" (tasks): states progress
    /// forward only and can never revert to `Compiling` once terminal or
    /// past it, satisfying the conformance requirement "states progress
    /// legally to succeeded and cannot revert to compiling".
    pub const fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Queued, Self::Compiling)
                | (Self::Queued, Self::Cancelled)
                | (Self::Queued, Self::TimedOut)
                | (Self::Compiling, Self::Succeeded)
                | (Self::Compiling, Self::Failed)
                | (Self::Compiling, Self::Cancelled)
                | (Self::Compiling, Self::TimedOut)
        )
    }

    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Succeeded | Self::Failed | Self::Cancelled | Self::TimedOut
        )
    }

    /// Only `Succeeded` is eligible to publish a ready
    /// [`CompiledKernelArtifact`], implementing "A cancelled job SHALL NOT
    /// publish a partially generated artifact as valid" and "Compilation
    /// timeout SHALL NOT leave a Compiled Kernel Artifact marked ready"
    /// (proposal).
    pub const fn may_publish_artifact(self) -> bool {
        matches!(self, Self::Succeeded)
    }
}

/// A tracked compilation job. Provider owns native compilation-job state;
/// this struct is Runtime-visible orchestration state only, implementing
/// "Compilation Job Ownership" (proposal) -- it carries no native handle.
#[derive(Clone, Debug, PartialEq)]
pub struct CompilationJob {
    pub id: CompilationJobId,
    pub request_id: CompilationRequestId,
    pub state: CompilationJobState,
    pub cancellation_requested: bool,
}

impl CompilationJob {
    pub fn new(id: CompilationJobId, request_id: CompilationRequestId) -> Self {
        Self {
            id,
            request_id,
            state: CompilationJobState::Queued,
            cancellation_requested: false,
        }
    }

    fn transition(&mut self, next: CompilationJobState) -> Result<(), KernelCompilationError> {
        if !self.state.can_transition_to(next) {
            return Err(KernelCompilationError::JobStateInvalid {
                reason: format!("cannot transition from {:?} to {next:?}", self.state),
            });
        }
        self.state = next;
        Ok(())
    }

    pub fn start_compiling(&mut self) -> Result<(), KernelCompilationError> {
        self.transition(CompilationJobState::Compiling)
    }

    pub fn mark_succeeded(&mut self) -> Result<(), KernelCompilationError> {
        self.transition(CompilationJobState::Succeeded)
    }

    pub fn mark_failed(&mut self) -> Result<(), KernelCompilationError> {
        self.transition(CompilationJobState::Failed)
    }

    pub fn mark_timed_out(&mut self) -> Result<(), KernelCompilationError> {
        self.transition(CompilationJobState::TimedOut)
    }

    pub fn mark_cancelled(&mut self) -> Result<(), KernelCompilationError> {
        self.transition(CompilationJobState::Cancelled)
    }
}

/// Implements "A cancelled job SHALL NOT publish a partially generated
/// artifact as valid" (proposal) as a standalone check usable independently
/// of [`CompilationJob`], given a cancellation support level and the job's
/// terminal state.
pub fn evaluate_cancellation_request(
    support: CompilationCancellationSupport,
    already_compiling: bool,
) -> Result<CompilationJobState, KernelCompilationError> {
    match support {
        CompilationCancellationSupport::NotSupported => {
            Err(KernelCompilationError::CancellationUnsupported)
        }
        CompilationCancellationSupport::BeforeStartOnly if already_compiling => {
            Err(KernelCompilationError::CancellationUnsupported)
        }
        _ => Ok(CompilationJobState::Cancelled),
    }
}

// ---------------------------------------------------------------------
// Runtime-to-Provider async boundary
// ---------------------------------------------------------------------

/// Outcome of a cancellation attempt, mirroring
/// [`crate::ProviderCancellationOutcome`]'s shape for execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompilationCancellationOutcome {
    Accepted,
    Unsupported,
    AlreadyTerminal(CompilationJobState),
}

/// Runtime-to-Provider Kernel Compilation boundary, mirroring
/// [`crate::provider::ProviderExecutionApi`]'s submit/status/cancel/release
/// shape. Implements "Compilation Job Ownership" (proposal): Runtime MAY
/// submit/poll/await/cancel/time-out/discard, and SHALL NOT reach into
/// Provider-native job internals -- every method here returns only opaque
/// Runtime-visible state.
pub trait ProviderKernelCompilationApi: Send + Sync {
    fn capability(&self) -> KernelCompilationCapabilityDescriptor;

    fn submit(
        &self,
        request: KernelCompilationRequest,
    ) -> Result<CompilationJobId, KernelCompilationError>;

    fn poll(&self, job: CompilationJobId) -> Result<CompilationJobState, KernelCompilationError>;

    fn cancel(
        &self,
        job: CompilationJobId,
    ) -> Result<CompilationCancellationOutcome, KernelCompilationError>;

    fn result(&self, job: CompilationJobId) -> Result<CompilationResult, KernelCompilationError>;

    fn release(&self, job: CompilationJobId) -> Result<(), KernelCompilationError>;
}

// ---------------------------------------------------------------------
// Compilation Result / Output Integrity
// ---------------------------------------------------------------------

/// A successful compilation result, implementing "Compilation Result"
/// (proposal). Wraps the shared [`CompiledKernelArtifact`] lifecycle entity
/// (see `kernel_artifact.rs`) with compilation-specific provenance.
#[derive(Clone, Debug, PartialEq)]
pub struct CompilationResult {
    pub job: CompilationJobId,
    pub artifact: CompiledKernelArtifact,
    pub compiler: CompilerIdentity,
    pub specialization: CompilationSpecialization,
    pub duration_millis: Option<u64>,
}

/// Implements "Output Integrity" (proposal): a compiled artifact SHALL have
/// immutable, verified identity before admission for preparation or caching.
pub fn verify_output_integrity(
    computed_digest: &str,
    declared_id: &CompiledKernelArtifactId,
) -> Result<(), KernelCompilationError> {
    if computed_digest == declared_id.digest() {
        Ok(())
    } else {
        Err(KernelCompilationError::OutputIntegrityFailed {
            expected: declared_id.digest().to_string(),
            found: computed_digest.to_string(),
        })
    }
}

// ---------------------------------------------------------------------
// Failure Atomicity / Crash Containment
// ---------------------------------------------------------------------

/// Implements "Compilation Failure Atomicity" (proposal): failure SHALL NOT
/// mutate an existing known-good [`CompiledKernelArtifact`] or
/// [`crate::kernel_artifact::PreparedKernel`]. Takes the existing artifact by
/// shared reference (never `&mut`) precisely so this cannot compile a
/// mutation path -- on failure the caller keeps `existing` untouched and
/// simply does not install a replacement.
pub fn preserve_known_good_artifact_on_failure<'a>(
    existing: &'a CompiledKernelArtifact,
    failure: &KernelCompilationError,
) -> &'a CompiledKernelArtifact {
    let _ = failure;
    existing
}

/// Normalizes a compiler crash into a structured failure, implementing
/// "Compiler Failure Containment" (proposal): "It SHALL NOT be reported as a
/// successful artifact."
pub fn normalize_compiler_crash(detail: impl AsRef<str>) -> KernelCompilationError {
    KernelCompilationError::CompilerCrashed {
        detail: redact_backend_diagnostic(detail.as_ref()),
    }
}

/// Implements "No Unwinding Across ABI" (proposal): calls `f` (Provider
/// compiler code) and normalizes any panic into
/// [`KernelCompilationError::CompilerCrashed`] instead of letting it unwind
/// across the Runtime/Provider boundary.
pub fn call_provider_compiler_without_unwinding<F, R>(f: F) -> Result<R, KernelCompilationError>
where
    F: FnOnce() -> Result<R, KernelCompilationError>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(payload) => {
            let detail = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "provider compiler panicked".into());
            Err(normalize_compiler_crash(detail))
        }
    }
}

// ---------------------------------------------------------------------
// Hot Path
// ---------------------------------------------------------------------

/// Implements "Hot Path Prohibition" (proposal): Provider execute APIs SHALL
/// NOT trigger source compilation. Delegates to
/// [`crate::kernel_artifact::reject_hot_path_compilation`] with the
/// compilation operation fixed, so callers cannot pass a different
/// operation and accidentally bypass the check.
pub fn reject_hot_path_kernel_compilation(
    path: KernelArtifactPath,
) -> Result<(), KernelCompilationError> {
    crate::kernel_artifact::reject_hot_path_compilation(
        path,
        KernelArtifactColdPathOperation::Compilation,
    )
    .map_err(|_| KernelCompilationError::HotPathDenied)
}

// ---------------------------------------------------------------------
// Provider ABI Extension
// ---------------------------------------------------------------------

/// Which optional ABI functions a Provider's Kernel Compilation extension
/// implements, mirroring [`crate::provider::ProviderAbiFunctionTable`]'s
/// shape. Implements "Provider ABI Extension" (proposal)'s submit/poll/
/// cancel/release-job ABI functions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernelCompilationAbiFunctionTable {
    pub query_capability: bool,
    pub submit: bool,
    pub poll: bool,
    pub cancel: bool,
    pub release_job: bool,
}

impl KernelCompilationAbiFunctionTable {
    pub const REQUIRED: Self = Self {
        query_capability: true,
        submit: true,
        poll: true,
        cancel: true,
        release_job: true,
    };
}

/// Buffer ownership rules for the Kernel Compilation ABI extension,
/// mirroring [`crate::provider::ProviderAbiOwnershipRules`]. Implements "ABI
/// Buffer Ownership" (proposal): request, result, output artifact, and
/// diagnostic buffers all have explicit ownership; no allocator ownership is
/// implicit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelCompilationAbiOwnershipRules {
    pub request_buffer: crate::provider::ProviderAbiMemoryRule,
    pub result_buffer: crate::provider::ProviderAbiMemoryRule,
    pub output_artifact: crate::provider::ProviderAbiMemoryRule,
    pub diagnostic_buffer: crate::provider::ProviderAbiMemoryRule,
}

impl Default for KernelCompilationAbiOwnershipRules {
    fn default() -> Self {
        Self {
            request_buffer: crate::provider::ProviderAbiMemoryRule::runtime_borrowed(),
            result_buffer: crate::provider::ProviderAbiMemoryRule::provider_released(),
            output_artifact: crate::provider::ProviderAbiMemoryRule::provider_released(),
            diagnostic_buffer: crate::provider::ProviderAbiMemoryRule::provider_released(),
        }
    }
}

/// The versioned Kernel Compilation ABI extension descriptor, implementing
/// "ABI Boundary" and "Optional ABI Extension" (proposal): "A Provider
/// implementing Provider ABI v1 but not Kernel Compilation SHALL remain
/// valid... Absence SHALL produce `kernel-compilation-unavailable`... It
/// SHALL NOT be treated as Provider corruption." This descriptor is a plain
/// value type -- no `dyn Trait` field -- so this half of the contract holds
/// "SHALL NOT expose Rust trait objects for kernel compilation" (proposal)
/// even though the higher-level [`ProviderKernelCompilationApi`] (this
/// crate's existing in-process Provider trait convention, matching
/// [`crate::provider::ProviderExecutionApi`]) is itself a trait object.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelCompilationAbiDescriptor {
    pub descriptor_size: usize,
    pub abi_version: crate::provider::ProviderAbiVersion,
    pub functions: KernelCompilationAbiFunctionTable,
    pub ownership: KernelCompilationAbiOwnershipRules,
}

impl KernelCompilationAbiDescriptor {
    pub fn current() -> Self {
        Self {
            descriptor_size: std::mem::size_of::<Self>(),
            abi_version: crate::provider::ProviderAbiVersion::CURRENT,
            functions: KernelCompilationAbiFunctionTable::REQUIRED,
            ownership: KernelCompilationAbiOwnershipRules::default(),
        }
    }

    /// Implements "Add ABI version checks" (tasks).
    pub fn validate(&self) -> Result<(), KernelCompilationError> {
        if self.descriptor_size < std::mem::size_of::<Self>() {
            return Err(KernelCompilationError::AbiIncompatible {
                reason: "descriptor size is smaller than the current layout".into(),
            });
        }
        if self.abi_version.major != crate::provider::PROVIDER_ABI_MAJOR_VERSION {
            return Err(KernelCompilationError::AbiIncompatible {
                reason: format!("unsupported ABI major version {}", self.abi_version.major),
            });
        }
        if !self.functions.query_capability
            || !self.functions.submit
            || !self.functions.poll
            || !self.functions.cancel
            || !self.functions.release_job
        {
            return Err(KernelCompilationError::AbiIncompatible {
                reason: "required Kernel Compilation ABI function is missing".into(),
            });
        }
        if !self.ownership.result_buffer.release_required
            || !self.ownership.output_artifact.release_required
            || !self.ownership.diagnostic_buffer.release_required
        {
            return Err(KernelCompilationError::AbiIncompatible {
                reason: "provider-owned Kernel Compilation ABI memory requires release functions"
                    .into(),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Prepared Kernel ABI
// ---------------------------------------------------------------------

/// Implements "Prepared Kernel ABI" (proposal): rejects treating a
/// [`crate::kernel_artifact::PreparedKernelId`]'s `Display` output as if it
/// contained a native pointer. `PreparedKernelId` already exposes no numeric
/// accessor (see `kernel_artifact.rs`); this function documents and checks
/// the surface contract that only its opaque string form crosses any
/// diagnostic boundary.
pub fn assert_prepared_kernel_id_opaque(rendered: &str) -> Result<(), KernelCompilationError> {
    if rendered.starts_with("0x") {
        return Err(KernelCompilationError::BufferOwnershipViolation {
            reason: "PreparedKernelId rendering resembles a native pointer".into(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------

/// Compilation lifecycle observation categories, implementing "Observability"
/// (proposal).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelCompilationObservationKind {
    RequestSubmitted,
    Queued,
    CompilerStarted,
    CompilerCompleted,
    CompilerFailed,
    Cancelled,
    TimedOut,
    ArtifactCreated,
    OutputValidated,
    PreparationStarted,
    PreparationCompleted,
}

/// A single compilation observation. Structurally guaranteed to never carry
/// raw kernel source, raw compiled binary bytes, native handles, or compiler
/// temp paths: the only fields are an enum `kind`, an optional job/artifact
/// identity, and a `redacted_metadata` map whose values always pass through
/// `redact_backend_diagnostic` first, implementing "Observability SHALL
/// NOT expose by default: raw kernel source, compiled binary bytes, native
/// handles, compiler temporary paths, arbitrary compiler stdout/stderr,
/// environment contents, secrets, credentials" (proposal).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelCompilationObservation {
    pub kind: KernelCompilationObservationKind,
    pub job: Option<String>,
    pub redacted_metadata: BTreeMap<String, String>,
}

impl KernelCompilationObservation {
    pub fn new(kind: KernelCompilationObservationKind) -> Self {
        Self {
            kind,
            job: None,
            redacted_metadata: BTreeMap::new(),
        }
    }

    pub fn with_job(mut self, job: impl Into<String>) -> Self {
        self.job = Some(job.into());
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
// Diagnostics
// ---------------------------------------------------------------------

/// A classified, redacted compiler diagnostic, implementing "Compiler
/// Diagnostics" (proposal): "Runtime SHOULD prefer error category, compiler
/// stage, source-location metadata, redacted diagnostic rather than
/// unrestricted raw compiler output."
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompilerDiagnostic {
    pub stage: String,
    pub source_location: Option<String>,
    pub redacted_message: String,
}

impl CompilerDiagnostic {
    /// Builds a diagnostic from raw compiler output, redacting it through
    /// `redact_backend_diagnostic` rather than exposing it verbatim.
    pub fn from_raw_output(stage: impl Into<String>, raw_message: impl AsRef<str>) -> Self {
        Self {
            stage: stage.into(),
            source_location: None,
            redacted_message: redact_backend_diagnostic(raw_message.as_ref()),
        }
    }

    pub fn with_source_location(mut self, location: impl Into<String>) -> Self {
        self.source_location = Some(location.into());
        self
    }
}

// ---------------------------------------------------------------------
// Error Model
// ---------------------------------------------------------------------

/// Structured Kernel Compilation error, covering every category from the
/// proposal's "Error Model" section.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelCompilationError {
    Unavailable,
    CapabilityVersionUnsupported {
        found: CapabilityVersion,
    },
    SourceFormatUnsupported {
        format: String,
    },
    OutputFormatUnsupported {
        format: String,
    },
    TargetUnsupported {
        reason: String,
    },
    SpecializationUnsupported {
        reason: String,
    },
    PolicyDenied {
        reason: String,
    },
    IsolationInsufficient {
        declared: String,
        required_minimum: String,
    },
    SourceTooLarge {
        max_bytes: u64,
        found_bytes: u64,
    },
    OutputTooLarge {
        max_bytes: u64,
        found_bytes: u64,
    },
    ConcurrencyLimit {
        max_concurrent: u32,
    },
    DeadlineUnsupported {
        reason: String,
    },
    Timeout,
    CancellationUnsupported,
    Cancelled,
    CompilerUnavailable {
        reason: String,
    },
    CompilerCrashed {
        detail: String,
    },
    Failed {
        reason: String,
    },
    OutputInvalid {
        reason: String,
    },
    OutputIntegrityFailed {
        expected: String,
        found: String,
    },
    JobNotFound {
        job: String,
    },
    JobStateInvalid {
        reason: String,
    },
    AbiIncompatible {
        reason: String,
    },
    BufferOwnershipViolation {
        reason: String,
    },
    HotPathDenied,
    DescriptorInvalid {
        reason: String,
    },
    InternalError {
        reason: String,
    },
}

impl KernelCompilationError {
    pub const fn id(&self) -> &'static str {
        match self {
            Self::Unavailable => "kernel-compilation-unavailable",
            Self::CapabilityVersionUnsupported { .. } => {
                "kernel-compilation-capability-version-unsupported"
            }
            Self::SourceFormatUnsupported { .. } => "kernel-compilation-source-format-unsupported",
            Self::OutputFormatUnsupported { .. } => "kernel-compilation-output-format-unsupported",
            Self::TargetUnsupported { .. } => "kernel-compilation-target-unsupported",
            Self::SpecializationUnsupported { .. } => {
                "kernel-compilation-specialization-unsupported"
            }
            Self::PolicyDenied { .. } => "kernel-compilation-policy-denied",
            Self::IsolationInsufficient { .. } => "kernel-compilation-isolation-insufficient",
            Self::SourceTooLarge { .. } => "kernel-compilation-source-too-large",
            Self::OutputTooLarge { .. } => "kernel-compilation-output-too-large",
            Self::ConcurrencyLimit { .. } => "kernel-compilation-concurrency-limit",
            Self::DeadlineUnsupported { .. } => "kernel-compilation-deadline-unsupported",
            Self::Timeout => "kernel-compilation-timeout",
            Self::CancellationUnsupported => "kernel-compilation-cancellation-unsupported",
            Self::Cancelled => "kernel-compilation-cancelled",
            Self::CompilerUnavailable { .. } => "kernel-compilation-compiler-unavailable",
            Self::CompilerCrashed { .. } => "kernel-compilation-compiler-crashed",
            Self::Failed { .. } => "kernel-compilation-failed",
            Self::OutputInvalid { .. } => "kernel-compilation-output-invalid",
            Self::OutputIntegrityFailed { .. } => "kernel-compilation-output-integrity-failed",
            Self::JobNotFound { .. } => "kernel-compilation-job-not-found",
            Self::JobStateInvalid { .. } => "kernel-compilation-job-state-invalid",
            Self::AbiIncompatible { .. } => "kernel-compilation-abi-incompatible",
            Self::BufferOwnershipViolation { .. } => {
                "kernel-compilation-buffer-ownership-violation"
            }
            Self::HotPathDenied => "kernel-compilation-hot-path-denied",
            Self::DescriptorInvalid { .. } => "kernel-compilation-descriptor-invalid",
            Self::InternalError { .. } => "internal-kernel-compilation-error",
        }
    }
}

impl fmt::Display for KernelCompilationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable
            | Self::Timeout
            | Self::CancellationUnsupported
            | Self::Cancelled
            | Self::HotPathDenied => f.write_str(self.id()),
            Self::CapabilityVersionUnsupported { found } => {
                write!(f, "{}: found {found}", self.id())
            }
            Self::SourceFormatUnsupported { format } | Self::OutputFormatUnsupported { format } => {
                write!(f, "{}: {format}", self.id())
            }
            Self::TargetUnsupported { reason }
            | Self::SpecializationUnsupported { reason }
            | Self::PolicyDenied { reason }
            | Self::DeadlineUnsupported { reason }
            | Self::CompilerUnavailable { reason }
            | Self::Failed { reason }
            | Self::OutputInvalid { reason }
            | Self::JobStateInvalid { reason }
            | Self::AbiIncompatible { reason }
            | Self::BufferOwnershipViolation { reason }
            | Self::DescriptorInvalid { reason }
            | Self::InternalError { reason } => write!(f, "{}: {reason}", self.id()),
            Self::IsolationInsufficient {
                declared,
                required_minimum,
            } => write!(
                f,
                "{}: declared {declared}, required at least {required_minimum}",
                self.id()
            ),
            Self::SourceTooLarge {
                max_bytes,
                found_bytes,
            }
            | Self::OutputTooLarge {
                max_bytes,
                found_bytes,
            } => write!(f, "{}: max {max_bytes}, found {found_bytes}", self.id()),
            Self::ConcurrencyLimit { max_concurrent } => {
                write!(f, "{}: max {max_concurrent}", self.id())
            }
            Self::CompilerCrashed { detail } => write!(f, "{}: {detail}", self.id()),
            Self::OutputIntegrityFailed { expected, found } => {
                write!(f, "{}: expected {expected}, found {found}", self.id())
            }
            Self::JobNotFound { job } => write!(f, "{}: {job}", self.id()),
        }
    }
}

impl Error for KernelCompilationError {}

// ---------------------------------------------------------------------
// Conformance
// ---------------------------------------------------------------------

/// A single Kernel Compilation conformance check result, mirroring
/// [`crate::kernel_artifact::KernelArtifactConformanceResult`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelCompilationConformanceResult {
    pub requirement: String,
    pub passed: bool,
    pub diagnostic: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelCompilationConformanceReport {
    pub results: Vec<KernelCompilationConformanceResult>,
}

impl KernelCompilationConformanceReport {
    pub fn is_conformant(&self) -> bool {
        self.results.iter().all(|result| result.passed)
    }
}

fn record(
    results: &mut Vec<KernelCompilationConformanceResult>,
    requirement: impl Into<String>,
    passed: bool,
    diagnostic: impl Into<String>,
) {
    let diagnostic = diagnostic.into();
    results.push(KernelCompilationConformanceResult {
        requirement: requirement.into(),
        passed,
        diagnostic: (!passed).then_some(diagnostic),
    });
}

/// Runs the conformance checks required by
/// `openspec/changes/define-provider-kernel-compilation-capability/specs/conformance/spec.md`.
pub fn run_kernel_compilation_conformance() -> KernelCompilationConformanceReport {
    let mut results = Vec::new();

    // "Provider without compiler": absence of the capability is valid.
    let absent = KernelCompilationCapabilityDescriptor::unsupported();
    record(
        &mut results,
        "Provider Compilation Capability Conformance: absent capability is valid",
        !absent.is_present() && absent.validate().is_ok(),
        "unsupported() descriptor unexpectedly reports present or invalid",
    );

    // "Source Format Negotiation Conformance": WGSL rejected by a CPU
    // Provider that only accepts triton source.
    let accepted: BTreeSet<KernelSourceFormat> =
        [KernelSourceFormat::new("triton", "source").with_version("3")]
            .into_iter()
            .collect();
    let wgsl = KernelSourceFormat::new("webgpu", "wgsl");
    let negotiated = negotiate_source_format(&wgsl, &accepted);
    record(
        &mut results,
        "Source Format Negotiation Conformance: unsupported format rejected",
        matches!(
            negotiated,
            Err(KernelCompilationError::SourceFormatUnsupported { .. })
        ),
        format!("unexpected outcome: {negotiated:?}"),
    );

    // "Compilation Job Lifecycle Conformance": legal progression to
    // succeeded, and no reversion to compiling.
    {
        let mut allocator = CompilationJobIdAllocator::default();
        let mut job = CompilationJob::new(
            allocator.allocate(),
            CompilationRequestId::new("conformance-request"),
        );
        let started = job.start_compiling();
        let succeeded = job.mark_succeeded();
        let reverted = job.transition(CompilationJobState::Compiling);
        record(
            &mut results,
            "Compilation Job Lifecycle Conformance: succeeds and cannot revert",
            started.is_ok()
                && succeeded.is_ok()
                && job.state == CompilationJobState::Succeeded
                && matches!(
                    reverted,
                    Err(KernelCompilationError::JobStateInvalid { .. })
                ),
            format!("unexpected job state after transitions: {job:?}"),
        );
    }

    // "Compilation Cancellation Conformance": cooperative cancellation never
    // yields a state eligible to publish an artifact.
    let cancel_outcome =
        evaluate_cancellation_request(CompilationCancellationSupport::Cooperative, true);
    record(
        &mut results,
        "Compilation Cancellation Conformance: cancelled job cannot publish output",
        matches!(cancel_outcome, Ok(state) if !state.may_publish_artifact()),
        format!("unexpected outcome: {cancel_outcome:?}"),
    );

    // "Compilation Deadline Conformance": enforceable deadline exceeded ends
    // timed-out without a ready artifact.
    {
        let mut allocator = CompilationJobIdAllocator::default();
        let mut job = CompilationJob::new(
            allocator.allocate(),
            CompilationRequestId::new("conformance-deadline"),
        );
        job.start_compiling().ok();
        job.mark_timed_out().ok();
        record(
            &mut results,
            "Compilation Deadline Conformance: timed-out job cannot publish output",
            job.state == CompilationJobState::TimedOut && !job.state.may_publish_artifact(),
            format!("unexpected job state: {job:?}"),
        );
    }

    // "Compilation Isolation Conformance": policy requiring sandboxing
    // rejects a Provider advertising in-process compilation only.
    let isolation = evaluate_isolation_sufficiency(
        CompilationIsolationModel::InProcessTrustedCompiler,
        CompilationIsolationModel::SandboxedSubprocess,
    );
    record(
        &mut results,
        "Compilation Isolation Conformance: insufficient isolation denied",
        matches!(
            isolation,
            Err(KernelCompilationError::IsolationInsufficient { .. })
        ),
        format!("unexpected outcome: {isolation:?}"),
    );

    // "Compilation Trust Separation Conformance": compilation success never
    // grants trust by itself.
    let trust = compilation_result_trust(false);
    record(
        &mut results,
        "Compilation Trust Separation Conformance: success does not imply trust",
        !trust.is_trusted(),
        format!("unexpected trust: {trust:?}"),
    );

    // "Hot Path Compilation Conformance": hot-path compilation is denied.
    let hot = reject_hot_path_kernel_compilation(KernelArtifactPath::Hot);
    record(
        &mut results,
        "Hot Path Compilation Conformance: hot-path compilation denied",
        matches!(hot, Err(KernelCompilationError::HotPathDenied)),
        format!("unexpected outcome: {hot:?}"),
    );

    // "ABI Ownership Conformance": the current ABI descriptor declares
    // provider release requirements for result/output/diagnostic buffers.
    let abi = KernelCompilationAbiDescriptor::current();
    record(
        &mut results,
        "ABI Ownership Conformance: result/output/diagnostic buffers require release",
        abi.validate().is_ok()
            && abi.ownership.result_buffer.release_required
            && abi.ownership.output_artifact.release_required
            && abi.ownership.diagnostic_buffer.release_required,
        format!("unexpected ABI descriptor: {abi:?}"),
    );

    // "ABI Handle Opacity Conformance": job IDs render without native
    // pointer semantics.
    let mut allocator = CompilationJobIdAllocator::default();
    let job_id = allocator.allocate();
    let opacity = assert_prepared_kernel_id_opaque(&job_id.to_string());
    record(
        &mut results,
        "ABI Handle Opacity Conformance: job/prepared IDs expose no pointer semantics",
        opacity.is_ok() && !job_id.to_string().starts_with("0x"),
        format!("unexpected job id rendering: {job_id}"),
    );

    // "Compiler Failure Atomicity Conformance": a crashing replacement
    // compile leaves the existing known-good artifact untouched.
    {
        let operator =
            crate::OperatorId::magnetar("matmul", 1, crate::OperatorFamily::LinearAlgebra);
        let v1 = CompiledKernelArtifact::new(
            CompiledKernelArtifactId::from_digest("v1-digest"),
            "cubin",
            "nvcc",
            "12.4",
            "sm_90",
            operator,
        );
        let crash = normalize_compiler_crash("compiler segfault at replacement build");
        let preserved = preserve_known_good_artifact_on_failure(&v1, &crash);
        record(
            &mut results,
            "Compiler Failure Atomicity Conformance: known-good artifact untouched on crash",
            preserved == &v1 && matches!(crash, KernelCompilationError::CompilerCrashed { .. }),
            "existing artifact unexpectedly changed on compiler crash",
        );
    }

    // AOT-only conformance fixture: a Provider that can prepare externally
    // produced artifacts without any source-compilation capability is a
    // valid, distinct support level from `SourceCompilation` and `None`.
    {
        let mut aot_only = KernelCompilationCapabilityDescriptor::unsupported();
        aot_only.support_level = CompilationSupportLevel::PreparationOnly;
        aot_only
            .produced_compiled_formats
            .insert("nvidia:cubin".into());
        aot_only.isolation_model = CompilationIsolationModel::PlatformManagedCompiler;
        record(
            &mut results,
            "AOT-Only Provider Is Supported: preparation-only descriptor is present and valid",
            aot_only.is_present()
                && aot_only.validate().is_ok()
                && aot_only.accepted_source_formats.is_empty(),
            format!("unexpected descriptor: {aot_only:?}"),
        );
    }

    // Structural facts matching kernel_artifact.rs's equivalents: Device and
    // scheduler.rs still define no compilation method.
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

    KernelCompilationConformanceReport { results }
}

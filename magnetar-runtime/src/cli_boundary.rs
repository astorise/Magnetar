//! Runtime-side formalization of the `magnetar-cli` inference boundary (see
//! `openspec/changes/define-magnetar-cli-inference-boundary`).
//!
//! `magnetar-cli` is the first-party client/workspace/tool/agent runtime
//! built around Magnetar inference: it owns workspace and file access, Git,
//! network, secrets, tool execution, shell/process execution, agent
//! orchestration, chat/session UX, and CLI configuration. `magnetar-runtime`
//! remains the inference runtime and SHALL NOT absorb those responsibilities
//! (see [`crate::inference_api`]'s module doc comment for the symmetric
//! Runtime-side boundary and [`crate::inference_api::FORBIDDEN_INFERENCE_API_SCOPES`]
//! for the capability names this module rejects).
//!
//! This module does not implement a new boundary mechanism. It composes the
//! already Runtime-owned [`crate::inference_api::validate_inference_scope`]
//! check and `crate::compute::redact_backend_diagnostic` redaction utility
//! behind:
//!
//! - [`CliBoundaryError`]: the CLI-side structured error model from the
//!   change's proposal "Error Model" section, wrapping
//!   [`crate::inference_api::InferenceApiError`] so Runtime structured error
//!   categories survive CLI display/wrapping instead of being flattened into
//!   an opaque string (the "CLI Preserves Runtime Structured Errors"
//!   requirement in `specs/conformance/spec.md`),
//! - [`reject_cli_owned_authority`]: an executable, regression-proof check
//!   that Runtime rejects capability names owned by `magnetar-cli`
//!   (workspace, filesystem, Git, network tool, shell, process, secrets,
//!   tool-call, agent-orchestration, ...), implementing "Runtime Does Not
//!   Execute CLI-Owned Capabilities" (`specs/conformance/spec.md`) and "CLI
//!   Authority Is Not Runtime Authority" (`specs/cli-boundary/spec.md`), and
//! - [`run_cli_boundary_conformance`]: a small conformance report, in the
//!   spirit of [`crate::conformance::ProviderConformanceReport`], that
//!   asserts the above plus Runtime's existing diagnostic redaction
//!   guarantee holds for CLI-facing diagnostics.
//!
//! `magnetar-cli` itself (a separate crate) is the only intended caller of
//! [`CliBoundaryError`]; Runtime other than this module has no CLI-specific
//! code.

use crate::compute::redact_backend_diagnostic;
use crate::inference_api::{
    FORBIDDEN_INFERENCE_API_SCOPES, InferenceApiError, validate_inference_scope,
};
use std::{error::Error, fmt};

// ---------------------------------------------------------------------
// Error Model
// ---------------------------------------------------------------------

/// Structured `magnetar-cli` boundary error, unifying the CLI-owned error
/// categories from the change proposal's "Error Model" section behind a
/// stable, caller-facing enum. `CliRuntimeRequestFailed` wraps
/// [`InferenceApiError`] so a Runtime structured error category is never
/// flattened into an opaque string when the CLI displays or re-raises it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CliBoundaryError {
    CliCommandInvalid { reason: String },
    CliPromptInputInvalid { reason: String },
    CliFileReadFailed { reason: String },
    CliWorkspaceAccessDenied { reason: String },
    CliGitFailed { reason: String },
    CliNetworkDenied { reason: String },
    CliSecretUnavailable { reason: String },
    CliToolFailed { reason: String },
    CliShellDenied { reason: String },
    CliModelAliasNotFound { alias: String },
    CliModelReferenceInvalid { reason: String },
    CliRuntimeUnavailable { reason: String },
    CliRuntimeRequestFailed(InferenceApiError),
    CliStreamInterrupted { reason: String },
    CliCancellationRequested,
    CliDiagnosticsRedacted,
    CliBoundaryViolation { capability: String },
    InternalCliError { reason: String },
}

impl fmt::Display for CliBoundaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CliCommandInvalid { reason } => write!(f, "cli command invalid: {reason}"),
            Self::CliPromptInputInvalid { reason } => {
                write!(f, "cli prompt input invalid: {reason}")
            }
            Self::CliFileReadFailed { reason } => write!(f, "cli file read failed: {reason}"),
            Self::CliWorkspaceAccessDenied { reason } => {
                write!(f, "cli workspace access denied: {reason}")
            }
            Self::CliGitFailed { reason } => write!(f, "cli git failed: {reason}"),
            Self::CliNetworkDenied { reason } => write!(f, "cli network denied: {reason}"),
            Self::CliSecretUnavailable { reason } => {
                write!(f, "cli secret unavailable: {reason}")
            }
            Self::CliToolFailed { reason } => write!(f, "cli tool failed: {reason}"),
            Self::CliShellDenied { reason } => write!(f, "cli shell denied: {reason}"),
            Self::CliModelAliasNotFound { alias } => {
                write!(f, "cli model alias not found: {alias}")
            }
            Self::CliModelReferenceInvalid { reason } => {
                write!(f, "cli model reference invalid: {reason}")
            }
            Self::CliRuntimeUnavailable { reason } => {
                write!(f, "cli runtime unavailable: {reason}")
            }
            Self::CliRuntimeRequestFailed(inner) => write!(f, "runtime request failed: {inner}"),
            Self::CliStreamInterrupted { reason } => {
                write!(f, "cli stream interrupted: {reason}")
            }
            Self::CliCancellationRequested => f.write_str("cli cancellation requested"),
            Self::CliDiagnosticsRedacted => f.write_str("cli diagnostics redacted"),
            Self::CliBoundaryViolation { capability } => {
                write!(f, "cli boundary violation: capability '{capability}'")
            }
            Self::InternalCliError { reason } => write!(f, "internal cli error: {reason}"),
        }
    }
}

impl Error for CliBoundaryError {}

impl From<InferenceApiError> for CliBoundaryError {
    fn from(error: InferenceApiError) -> Self {
        Self::CliRuntimeRequestFailed(error)
    }
}

impl CliBoundaryError {
    /// Returns the preserved Runtime structured error category for
    /// [`Self::CliRuntimeRequestFailed`], or `None` for every other variant.
    /// Lets callers inspect/match the Runtime error category without
    /// unwrapping the whole enum, satisfying "CLI Preserves Runtime
    /// Structured Errors" mechanically rather than by convention alone.
    pub fn runtime_category(&self) -> Option<&InferenceApiError> {
        match self {
            Self::CliRuntimeRequestFailed(inner) => Some(inner),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------
// CLI-owned authority boundary
// ---------------------------------------------------------------------

/// Rejects a capability name that names a `magnetar-cli`-owned
/// responsibility (workspace, filesystem, Git, network tool, shell,
/// process, secrets, tool-call, agent-orchestration, ...) by delegating to
/// [`validate_inference_scope`] and mapping its `PolicyDenied` outcome into
/// [`CliBoundaryError::CliBoundaryViolation`]. Implements "Runtime Does Not
/// Execute CLI-Owned Capabilities" (`specs/conformance/spec.md`) and "CLI
/// Authority Is Not Runtime Authority" (`specs/cli-boundary/spec.md`) as an
/// executable, regression-proof check.
pub fn reject_cli_owned_authority(capability: &str) -> Result<(), CliBoundaryError> {
    validate_inference_scope(capability).map_err(|_| CliBoundaryError::CliBoundaryViolation {
        capability: capability.to_string(),
    })
}

// ---------------------------------------------------------------------
// Conformance
// ---------------------------------------------------------------------

/// A single CLI boundary conformance check result, mirroring
/// [`crate::conformance::ProviderConformanceReport`]'s pass/fail/diagnostic
/// shape but standalone (no Provider dependency).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CliBoundaryConformanceResult {
    pub requirement: String,
    pub passed: bool,
    pub diagnostic: Option<String>,
}

/// A collected set of [`CliBoundaryConformanceResult`]s.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CliBoundaryConformanceReport {
    pub results: Vec<CliBoundaryConformanceResult>,
}

impl CliBoundaryConformanceReport {
    pub fn is_conformant(&self) -> bool {
        self.results.iter().all(|result| result.passed)
    }
}

/// Capability names covered by [`run_cli_boundary_conformance`]'s
/// CLI-owned-authority-rejection check. Every entry here also appears in
/// [`FORBIDDEN_INFERENCE_API_SCOPES`] -- this list intentionally does not
/// invent new forbidden capability names, it just exercises the existing
/// ones through the `magnetar-cli`-facing entry point.
const CLI_OWNED_AUTHORITY_CAPABILITIES: &[&str] = &[
    "workspace",
    "workspace-filesystem",
    "filesystem",
    "git",
    "network-tool",
    "shell",
    "process",
    "process-execution",
    "secret",
    "secrets",
    "tool-call",
    "agent-orchestration",
    "external-service",
    "source-editing",
    // Kernel Optimization Orchestration boundary: "kernel optimize" and
    // similar future tooling commands belong to CLI/tooling authority, not
    // Runtime Inference API authority (see
    // `crate::kernel_optimization_orchestration`).
    "kernel-optimization-orchestration",
    "generator-credential",
];

/// Runs the CLI boundary conformance checks described in this module's doc
/// comment: Runtime rejects every CLI-owned authority capability, a wrapped
/// [`InferenceApiError`] round-trips through [`CliBoundaryError::runtime_category`],
/// and a synthetic handle-like diagnostic is redacted by
/// `redact_backend_diagnostic`.
pub fn run_cli_boundary_conformance() -> CliBoundaryConformanceReport {
    let mut results = Vec::new();

    for capability in CLI_OWNED_AUTHORITY_CAPABILITIES {
        debug_assert!(
            FORBIDDEN_INFERENCE_API_SCOPES
                .iter()
                .any(|forbidden| capability.contains(forbidden)),
            "conformance capability '{capability}' must overlap FORBIDDEN_INFERENCE_API_SCOPES"
        );
        let outcome = reject_cli_owned_authority(capability);
        let passed = matches!(outcome, Err(CliBoundaryError::CliBoundaryViolation { .. }));
        results.push(CliBoundaryConformanceResult {
            requirement: format!("Runtime rejects CLI-owned authority capability '{capability}'"),
            passed,
            diagnostic: (!passed).then(|| format!("unexpected outcome: {outcome:?}")),
        });
    }

    {
        let source = InferenceApiError::ModelLoadingFailed {
            reason: "example".into(),
        };
        let wrapped = CliBoundaryError::from(source.clone());
        let passed = wrapped.runtime_category() == Some(&source);
        results.push(CliBoundaryConformanceResult {
            requirement: "CliBoundaryError preserves the wrapped Runtime error category".into(),
            passed,
            diagnostic: (!passed).then(|| "runtime_category() did not round-trip".to_string()),
        });
    }

    {
        let diagnostic = "provider handle=0xdeadbeef failed";
        let redacted = redact_backend_diagnostic(diagnostic);
        let passed = !redacted.contains("0xdeadbeef");
        results.push(CliBoundaryConformanceResult {
            requirement: "CLI-facing diagnostics inherit Runtime's handle redaction".into(),
            passed,
            diagnostic: (!passed).then(|| format!("diagnostic leaked a handle: {redacted}")),
        });
    }

    CliBoundaryConformanceReport { results }
}

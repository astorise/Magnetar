//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::compute::ComputeDType;

use crate::affinity::{DeviceBinding, ProviderBinding};
use crate::kernel_artifact::CompiledKernelArtifactId;
use crate::kernel_artifact::{KernelArtifactPath, KernelSourceFormat};
#[test]
fn output_format_negotiation_requires_explicit_declaration() {
    let produced: std::collections::BTreeSet<String> =
        ["nvidia:ptx@9".to_string()].into_iter().collect();
    assert!(negotiate_output_format("nvidia:ptx@9", &produced).is_ok());
    assert!(matches!(
        negotiate_output_format("nvidia:cubin", &produced),
        Err(KernelCompilationError::OutputFormatUnsupported { .. })
    ));
}

#[test]
fn compilation_job_lifecycle_progresses_legally_and_never_reverts() {
    let mut allocator = CompilationJobIdAllocator::default();
    let mut job = CompilationJob::new(allocator.allocate(), CompilationRequestId::new("req-1"));
    assert_eq!(job.state, CompilationJobState::Queued);
    assert!(job.start_compiling().is_ok());
    assert_eq!(job.state, CompilationJobState::Compiling);
    assert!(job.mark_succeeded().is_ok());
    assert_eq!(job.state, CompilationJobState::Succeeded);
    assert!(job.state.may_publish_artifact());

    let mut cancelled_before_start =
        CompilationJob::new(allocator.allocate(), CompilationRequestId::new("req-2"));
    assert!(matches!(
        cancelled_before_start.mark_succeeded(),
        Err(KernelCompilationError::JobStateInvalid { .. })
    ));
    assert!(cancelled_before_start.mark_cancelled().is_ok());
    assert!(!cancelled_before_start.state.may_publish_artifact());
}

#[test]
fn cancellation_support_levels_are_respected() {
    assert!(matches!(
        evaluate_cancellation_request(CompilationCancellationSupport::NotSupported, false),
        Err(KernelCompilationError::CancellationUnsupported)
    ));
    assert!(matches!(
        evaluate_cancellation_request(CompilationCancellationSupport::BeforeStartOnly, true),
        Err(KernelCompilationError::CancellationUnsupported)
    ));
    assert!(matches!(
        evaluate_cancellation_request(CompilationCancellationSupport::BeforeStartOnly, false),
        Ok(CompilationJobState::Cancelled)
    ));
    assert!(matches!(
        evaluate_cancellation_request(CompilationCancellationSupport::Cooperative, true),
        Ok(CompilationJobState::Cancelled)
    ));
}

#[test]
fn compilation_deadline_fails_closed_when_unenforceable() {
    let required = CompilationDeadline {
        max_wall_clock_millis: 5_000,
    };
    assert!(matches!(
        enforce_compilation_deadline(Some(required), false),
        Err(KernelCompilationError::DeadlineUnsupported { .. })
    ));
    assert!(enforce_compilation_deadline(Some(required), true).is_ok());
    assert!(enforce_compilation_deadline(None, false).is_ok());
}

#[test]
fn compilation_limits_are_enforced_before_compiler_invocation() {
    let limits = CompilationLimits {
        max_source_bytes: Some(1024),
        max_output_bytes: Some(2048),
        max_concurrent_jobs: Some(2),
        max_workspace_bytes: Some(1 << 20),
        max_host_memory_bytes: Some(1 << 30),
    };
    assert!(enforce_compilation_limits(512, &limits).is_ok());
    assert!(matches!(
        enforce_compilation_limits(2048, &limits),
        Err(KernelCompilationError::SourceTooLarge { .. })
    ));
    assert!(matches!(
        enforce_output_limit(4096, &limits),
        Err(KernelCompilationError::OutputTooLarge { .. })
    ));
    assert!(enforce_concurrency_limit(1, &limits).is_ok());
    assert!(matches!(
        enforce_concurrency_limit(2, &limits),
        Err(KernelCompilationError::ConcurrencyLimit { .. })
    ));
}

#[test]
fn isolation_sufficiency_rejects_weaker_than_required_model() {
    assert!(matches!(
        evaluate_isolation_sufficiency(
            CompilationIsolationModel::InProcessTrustedCompiler,
            CompilationIsolationModel::SandboxedSubprocess,
        ),
        Err(KernelCompilationError::IsolationInsufficient { .. })
    ));
    assert!(
        evaluate_isolation_sufficiency(
            CompilationIsolationModel::SandboxedSubprocess,
            CompilationIsolationModel::SandboxedSubprocess,
        )
        .is_ok()
    );
    assert!(
        evaluate_isolation_sufficiency(
            CompilationIsolationModel::ExternalCompilationService,
            CompilationIsolationModel::SandboxedSubprocess,
        )
        .is_ok()
    );
}

#[test]
fn environment_variables_are_denied_by_default() {
    let deny = CompilationEnvironmentPolicy::Deny;
    assert!(!evaluate_environment_variable("SECRET_TOKEN", &deny));

    let mut allowed = std::collections::BTreeSet::new();
    allowed.insert("CUDA_HOME".to_string());
    let allowlist = CompilationEnvironmentPolicy::Allowlist(allowed);
    assert!(evaluate_environment_variable("CUDA_HOME", &allowlist));
    assert!(!evaluate_environment_variable("SECRET_TOKEN", &allowlist));
}

#[test]
fn explicit_specialization_is_required_when_applied() {
    let empty = CompilationSpecialization::default();
    assert!(matches!(
        require_explicit_compilation_specialization(true, &empty),
        Err(KernelCompilationError::SpecializationUnsupported { .. })
    ));
    assert!(require_explicit_compilation_specialization(false, &empty).is_ok());

    let mut declared = CompilationSpecialization::default();
    declared.dtype.insert(ComputeDType::Float16);
    assert!(require_explicit_compilation_specialization(true, &declared).is_ok());
}

#[test]
fn provider_compiler_panic_is_caught_and_never_unwinds_across_boundary() {
    let result: Result<(), KernelCompilationError> =
        call_provider_compiler_without_unwinding(|| -> Result<(), KernelCompilationError> {
            panic!("simulated compiler panic");
        });
    assert!(matches!(
        result,
        Err(KernelCompilationError::CompilerCrashed { .. })
    ));

    let ok: Result<u32, KernelCompilationError> =
        call_provider_compiler_without_unwinding(|| Ok(42));
    assert_eq!(ok, Ok(42));
}

#[test]
fn compilation_job_ids_and_prepared_kernel_ids_expose_no_pointer_semantics() {
    let mut allocator = CompilationJobIdAllocator::default();
    let job_id = allocator.allocate();
    assert!(assert_prepared_kernel_id_opaque(&job_id.to_string()).is_ok());
    assert!(matches!(
        assert_prepared_kernel_id_opaque("0xdeadbeef"),
        Err(KernelCompilationError::BufferOwnershipViolation { .. })
    ));
}

#[test]
fn compiler_diagnostics_redact_raw_output() {
    let diagnostic = CompilerDiagnostic::from_raw_output(
        "parse",
        "error in C:\\tmp\\kernel.triton at 0xffffabcd",
    )
    .with_source_location("kernel.triton:12:5");
    assert_eq!(diagnostic.redacted_message, "[redacted backend diagnostic]");
    assert_eq!(
        diagnostic.source_location.as_deref(),
        Some("kernel.triton:12:5")
    );
}

#[test]
fn compilation_process_arguments_are_structural_never_shell_strings() {
    let untrusted_metadata = "kernel; rm -rf / #";
    let invocation = CompilationProcessArguments::new("nvcc")
        .with_arg("--compile")
        .with_arg(untrusted_metadata);
    assert_eq!(invocation.program, "nvcc");
    // The untrusted value is preserved as a single argument element -- never
    // interpolated into (and re-parsed out of) a shell command string.
    assert_eq!(invocation.args.len(), 2);
    assert_eq!(invocation.args[1], untrusted_metadata);
}

#[test]
fn compilation_observation_never_carries_raw_source_or_native_handles() {
    let observation =
        KernelCompilationObservation::new(KernelCompilationObservationKind::CompilerCompleted)
            .with_job("compilation-job-1")
            .with_redacted_metadata("compiler", "nvcc 12.4")
            .with_redacted_metadata("temp_path", "C:\\tmp\\build\\0xdeadbeef");
    assert_eq!(
        observation.kind,
        KernelCompilationObservationKind::CompilerCompleted
    );
    assert_eq!(observation.job.as_deref(), Some("compilation-job-1"));
    assert_eq!(
        observation.redacted_metadata.get("temp_path").unwrap(),
        "[redacted backend diagnostic]"
    );
    assert_eq!(
        observation.redacted_metadata.get("compiler").unwrap(),
        "nvcc 12.4"
    );
}

#[test]
fn compilation_capability_absence_is_valid_and_optional() {
    let descriptor = KernelCompilationCapabilityDescriptor::unsupported();
    assert!(!descriptor.is_present());
    assert!(descriptor.validate().is_ok());
}

#[test]
fn network_boundary_denies_implicit_dependency_downloads() {
    let policy = CompilationNetworkPolicy::default();
    assert!(matches!(
        enforce_compilation_network_boundary(true, &policy),
        Err(KernelCompilationError::PolicyDenied { .. })
    ));
    let authorized = CompilationNetworkPolicy {
        network_access_authorized: true,
    };
    assert!(enforce_compilation_network_boundary(true, &authorized).is_ok());
    assert!(enforce_compilation_network_boundary(false, &policy).is_ok());
}

#[test]
fn preparation_only_provider_is_a_valid_distinct_support_level() {
    let mut descriptor = KernelCompilationCapabilityDescriptor::unsupported();
    descriptor.support_level = CompilationSupportLevel::PreparationOnly;
    descriptor
        .produced_compiled_formats
        .insert("nvidia:cubin".into());
    descriptor.isolation_model = CompilationIsolationModel::PlatformManagedCompiler;
    assert!(descriptor.is_present());
    assert!(descriptor.validate().is_ok());
    assert!(descriptor.accepted_source_formats.is_empty());
}

#[test]
fn compilation_success_never_grants_trust_by_itself() {
    assert!(!compilation_result_trust(false).is_trusted());
    assert!(compilation_result_trust(true).is_trusted());
}

#[test]
fn compiler_crash_is_normalized_and_redacted_never_a_success() {
    let error = normalize_compiler_crash("segfault at C:\\temp\\compiler\\work\\0xdeadbeef");
    match &error {
        KernelCompilationError::CompilerCrashed { detail } => {
            assert_eq!(detail, "[redacted backend diagnostic]");
        }
        other => panic!("unexpected error: {other:?}"),
    }
    assert_eq!(error.id(), "kernel-compilation-compiler-crashed");
}

#[test]
fn compiler_flags_are_redacted_by_default() {
    let identity = CompilerIdentity::default().with_raw_flags("-I C:\\vendor\\include -DSECRET=1");
    assert_eq!(
        identity.flags_fingerprint.as_deref(),
        Some("[redacted backend diagnostic]")
    );
}

#[test]
fn source_format_negotiation_rejects_unsupported_before_compilation() {
    let accepted: std::collections::BTreeSet<KernelSourceFormat> =
        [KernelSourceFormat::new("triton", "source").with_version("3")]
            .into_iter()
            .collect();
    let wgsl = KernelSourceFormat::new("webgpu", "wgsl");
    assert!(matches!(
        negotiate_source_format(&wgsl, &accepted),
        Err(KernelCompilationError::SourceFormatUnsupported { .. })
    ));
    let triton = KernelSourceFormat::new("triton", "source").with_version("3");
    assert!(negotiate_source_format(&triton, &accepted).is_ok());
}

#[test]
fn runtime_target_authority_rejects_provider_and_device_redirection() {
    let selected_provider = ProviderBinding::new("cuda-provider");
    let selected_device = DeviceBinding::new(crate::DeviceId::new("cuda-0"));
    let target =
        CompilationTarget::new(selected_provider.clone(), selected_device.clone(), "sm_90");
    assert!(
        enforce_runtime_target_authority(&target, &selected_provider, &selected_device).is_ok()
    );

    let other_provider = ProviderBinding::new("metal-provider");
    assert!(matches!(
        enforce_runtime_target_authority(&target, &other_provider, &selected_device),
        Err(KernelCompilationError::TargetUnsupported { .. })
    ));

    let other_device = DeviceBinding::new(crate::DeviceId::new("cuda-1"));
    assert!(matches!(
        enforce_runtime_target_authority(&target, &selected_provider, &other_device),
        Err(KernelCompilationError::TargetUnsupported { .. })
    ));
}

#[test]
fn hot_path_denies_kernel_compilation_cold_path_allows_it() {
    assert!(matches!(
        reject_hot_path_kernel_compilation(KernelArtifactPath::Hot),
        Err(KernelCompilationError::HotPathDenied)
    ));
    assert!(reject_hot_path_kernel_compilation(KernelArtifactPath::Cold).is_ok());
}

#[test]
fn platform_managed_compilation_mode_preserves_cold_hot_boundary() {
    let mut descriptor = KernelCompilationCapabilityDescriptor::unsupported();
    descriptor.support_level = CompilationSupportLevel::SourceCompilation;
    descriptor.modes.insert(CompilationMode::ProviderManaged);
    descriptor.isolation_model = CompilationIsolationModel::PlatformManagedCompiler;
    descriptor
        .accepted_source_formats
        .insert(KernelSourceFormat::new("apple", "msl"));
    descriptor
        .produced_compiled_formats
        .insert("apple:metallib".into());
    assert!(descriptor.validate().is_ok());
    // Even a platform that logically combines compile+prepare internally
    // still denies compilation on the decode hot path.
    assert!(matches!(
        reject_hot_path_kernel_compilation(KernelArtifactPath::Hot),
        Err(KernelCompilationError::HotPathDenied)
    ));
}

#[test]
fn kernel_compilation_conformance_report_is_conformant() {
    let report = run_kernel_compilation_conformance();
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
fn output_integrity_rejects_digest_mismatch() {
    let id = CompiledKernelArtifactId::from_digest("expected-digest");
    assert!(verify_output_integrity("expected-digest", &id).is_ok());
    assert!(matches!(
        verify_output_integrity("different-digest", &id),
        Err(KernelCompilationError::OutputIntegrityFailed { .. })
    ));
}

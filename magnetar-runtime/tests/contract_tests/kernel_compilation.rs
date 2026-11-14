use magnetar_runtime::{
    CompilationIsolationModel, CompilationJobIdAllocator, CompilationResult,
    CompilationSpecialization, CompilationSupportLevel, CompiledKernelArtifact,
    CompiledKernelArtifactId, CompilerIdentity, KernelCompilationCapabilityDescriptor,
    KernelCompilationError, KernelSourceFormat, OperatorFamily, OperatorId,
    normalize_compiler_crash, preserve_known_good_artifact_on_failure,
};

#[test]
fn compilation_capability_descriptor_validation_requires_declared_formats_and_isolation() {
    let mut descriptor = KernelCompilationCapabilityDescriptor::unsupported();
    descriptor.support_level = CompilationSupportLevel::SourceCompilation;
    assert!(matches!(
        descriptor.validate(),
        Err(KernelCompilationError::DescriptorInvalid { .. })
    ));

    descriptor
        .accepted_source_formats
        .insert(KernelSourceFormat::new("triton", "source").with_version("3"));
    assert!(matches!(
        descriptor.validate(),
        Err(KernelCompilationError::DescriptorInvalid { .. })
    ));

    descriptor
        .produced_compiled_formats
        .insert("nvidia:ptx@9".into());
    assert!(matches!(
        descriptor.validate(),
        Err(KernelCompilationError::DescriptorInvalid { .. })
    ));

    descriptor.isolation_model = CompilationIsolationModel::SandboxedSubprocess;
    assert!(descriptor.validate().is_ok());
}

#[test]
fn failed_compilation_never_mutates_existing_known_good_artifact() {
    let operator = OperatorId::magnetar("matmul", 1, OperatorFamily::LinearAlgebra);
    let v1 = CompiledKernelArtifact::new(
        CompiledKernelArtifactId::from_digest("v1-digest"),
        "cubin",
        "nvcc",
        "12.4",
        "sm_90",
        operator,
    );
    let crash = normalize_compiler_crash("replacement compile crashed");
    let preserved = preserve_known_good_artifact_on_failure(&v1, &crash);
    assert_eq!(preserved, &v1);
}

#[test]
fn compilation_result_wraps_compiled_artifact_with_compilation_provenance() {
    let operator = OperatorId::magnetar("matmul", 1, OperatorFamily::LinearAlgebra);
    let artifact = CompiledKernelArtifact::new(
        CompiledKernelArtifactId::from_digest("digest-result"),
        "ptx",
        "triton",
        "3.1",
        "sm_90",
        operator,
    );
    let mut allocator = CompilationJobIdAllocator::default();
    let result = CompilationResult {
        job: allocator.allocate(),
        artifact: artifact.clone(),
        compiler: CompilerIdentity {
            name: Some("triton".into()),
            version: Some("3.1".into()),
            ..CompilerIdentity::default()
        },
        specialization: CompilationSpecialization::default(),
        duration_millis: Some(1200),
    };
    assert_eq!(result.artifact, artifact);
    assert_eq!(result.compiler.name.as_deref(), Some("triton"));
    assert_eq!(result.duration_millis, Some(1200));
}

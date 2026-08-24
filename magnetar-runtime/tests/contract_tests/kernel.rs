use magnetar_runtime::{
    CapabilityVersion, ComputeDType, ComputePrecision, DTypeDescriptor, FallbackClass,
    KernelAdvertisement, KernelError, KernelErrorCode, KernelId, KernelImplementationFamily,
    KernelInvocation, KernelInvocationId, KernelMemoryClass, KernelObservation,
    KernelObservationKind, KernelOperatorVersionRange, KernelPrecisionMetadata, KernelResource,
    KernelShapeConstraints, KernelWorkspaceRequirements, MemoryAllocationId, OperatorFamily,
    OperatorId, ProviderBinding, ResourceAffinity, ShapeDescriptor, TensorDescriptor,
    TensorLayoutKind, TensorResourceDescriptor, TensorResourceId, initial_operator_catalog,
};

fn matmul_kernel() -> (KernelAdvertisement, magnetar_runtime::OperatorSpec) {
    let operator = OperatorId::magnetar("matmul", 1, OperatorFamily::LinearAlgebra);
    let id = KernelId::new(
        ProviderBinding::new("cuda"),
        "cuda.matmul.tensorcore.fp16",
        CapabilityVersion::new(1, 0, 0),
        operator.clone(),
        KernelOperatorVersionRange::exact(1),
        KernelImplementationFamily::Cuda,
    )
    .with_features(["tensorcore"])
    .with_conformance_profile("kernel-standard-v1");
    let mut advertisement = KernelAdvertisement::new(id)
        .with_dtypes(magnetar_runtime::TensorRole::Input, [ComputeDType::Float16])
        .with_dtypes(
            magnetar_runtime::TensorRole::Output,
            [ComputeDType::Float16],
        )
        .with_layouts([TensorLayoutKind::Contiguous])
        .with_memory_classes([KernelMemoryClass::Device]);
    advertisement.shape = KernelShapeConstraints {
        rank: Some(2),
        alignment: Some(8),
        max_total_elements: Some(4096),
        ..KernelShapeConstraints::default()
    };
    advertisement.workspace =
        KernelWorkspaceRequirements::required(1024, KernelMemoryClass::Device, 256);
    let spec = initial_operator_catalog().get(&operator).unwrap().clone();
    (advertisement, spec)
}

fn resource(id: &str, shape: [u64; 2], dtype: ComputeDType) -> KernelResource {
    let tensor = TensorDescriptor::materialized(
        ShapeDescriptor::new(shape),
        DTypeDescriptor::portable(dtype),
    );
    KernelResource::new(
        TensorResourceDescriptor::new(
            TensorResourceId::new(id),
            tensor,
            ResourceAffinity::new(FallbackClass::ProviderPinned)
                .with_provider(ProviderBinding::new("cuda")),
        ),
        KernelMemoryClass::Device,
    )
}

fn valid_invocation(advertisement: &KernelAdvertisement) -> KernelInvocation {
    KernelInvocation::new(
        KernelInvocationId::new("invocation:1"),
        advertisement.implemented_operator.clone(),
        advertisement.id.clone(),
        ProviderBinding::new("cuda"),
        ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(ProviderBinding::new("cuda")),
    )
    .with_input(resource("a", [8, 16], ComputeDType::Float16))
    .with_input(resource("b", [16, 8], ComputeDType::Float16))
    .with_output(resource("out", [8, 8], ComputeDType::Float16))
    .with_workspace(MemoryAllocationId::new(1))
}

#[test]
fn kernel_identity_is_stable_metadata_not_raw_function_pointer() {
    let (advertisement, _) = matmul_kernel();

    assert!(
        advertisement
            .id
            .stable_key()
            .contains("cuda.matmul.tensorcore.fp16")
    );
    assert!(!advertisement.id.stable_key().contains("0x"));
    assert_eq!(advertisement.id.provider.as_str(), "cuda");
    assert_eq!(advertisement.id.operator.name(), "matmul");
}

#[test]
fn kernel_advertisement_validates_operator_dtype_layout_shape_workspace_and_affinity() {
    let (advertisement, operator) = matmul_kernel();
    let invocation = valid_invocation(&advertisement);

    advertisement
        .validate_invocation(&operator, &invocation)
        .expect("valid kernel invocation should pass contract validation");
}

#[test]
fn kernel_rejects_operator_version_mismatch() {
    let (mut advertisement, operator) = matmul_kernel();
    advertisement.id.operator_versions = KernelOperatorVersionRange { min: 2, max: 2 };
    let invocation = valid_invocation(&advertisement);

    let error = advertisement
        .validate_invocation(&operator, &invocation)
        .expect_err("operator version mismatch must fail");
    assert_eq!(error.code(), KernelErrorCode::KernelVersionUnsupported);
}

#[test]
fn kernel_rejects_unsupported_dtype_shape_layout_memory_and_workspace() {
    let (advertisement, operator) = matmul_kernel();

    let mut bad_dtype = valid_invocation(&advertisement);
    bad_dtype.inputs[0] = resource("bad-dtype", [8, 16], ComputeDType::Float32);
    assert_eq!(
        advertisement
            .validate_invocation(&operator, &bad_dtype)
            .unwrap_err()
            .code(),
        KernelErrorCode::KernelDTypeUnsupported
    );

    let mut bad_shape = valid_invocation(&advertisement);
    bad_shape.inputs[0] = resource("bad-shape", [7, 16], ComputeDType::Float16);
    assert_eq!(
        advertisement
            .validate_invocation(&operator, &bad_shape)
            .unwrap_err()
            .code(),
        KernelErrorCode::KernelShapeUnsupported
    );

    let mut bad_memory = valid_invocation(&advertisement);
    bad_memory.outputs[0].memory_class = KernelMemoryClass::Host;
    assert_eq!(
        advertisement
            .validate_invocation(&operator, &bad_memory)
            .unwrap_err()
            .code(),
        KernelErrorCode::KernelMemoryClassUnsupported
    );

    let mut missing_workspace = valid_invocation(&advertisement);
    missing_workspace.workspace = None;
    assert_eq!(
        advertisement
            .validate_invocation(&operator, &missing_workspace)
            .unwrap_err()
            .code(),
        KernelErrorCode::KernelWorkspaceUnavailable
    );
}

#[test]
fn kernel_rejects_determinism_and_precision_policy_mismatches() {
    let (mut advertisement, operator) = matmul_kernel();
    let mut invocation = valid_invocation(&advertisement);
    invocation.deterministic_required = true;
    advertisement.determinism.deterministic = false;
    assert_eq!(
        advertisement
            .validate_invocation(&operator, &invocation)
            .unwrap_err()
            .code(),
        KernelErrorCode::KernelDeterminismUnsupported
    );

    advertisement.determinism.deterministic = true;
    advertisement.precision = KernelPrecisionMetadata {
        approximate_math: true,
        ..KernelPrecisionMetadata::default()
    };
    invocation.deterministic_required = false;
    invocation.precision = ComputePrecision::Exact;
    assert_eq!(
        advertisement
            .validate_invocation(&operator, &invocation)
            .unwrap_err()
            .code(),
        KernelErrorCode::KernelPrecisionUnsupported
    );
}

#[test]
fn kernel_errors_and_observations_are_structured_and_redacted() {
    let error = KernelError::KernelProviderSaturated {
        provider: "cuda".into(),
    };
    assert_eq!(error.code(), KernelErrorCode::KernelProviderSaturated);
    assert!(!error.to_string().contains("0x"));

    let (advertisement, _) = matmul_kernel();
    let observation = KernelObservation::new(KernelObservationKind::KernelDispatchFailed)
        .with_kernel(&advertisement.id)
        .with_redacted_metadata("error", error.code().id());
    assert_eq!(
        observation
            .redacted_metadata
            .get("error")
            .map(String::as_str),
        Some("kernel-provider-saturated")
    );
    assert!(
        observation
            .redacted_metadata
            .values()
            .all(|value| !value.contains("prompt") && !value.contains("0x"))
    );
}

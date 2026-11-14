use magnetar_runtime::{
    AdmissionState, ComputeDType, DTypeDescriptor, MemoryAdmissionDecision, MemoryAllocationClass,
    MemoryAllocationOwner, MemoryDTypeRelation, MemoryManager, ReferenceCpuExecutor,
    TensorResourceId, TensorValueAdmissionError,
};

#[test]
fn memory_manager_distinguishes_storage_and_compute_dtype_costs() {
    let dtype = MemoryDTypeRelation::new(
        DTypeDescriptor::portable(ComputeDType::SInt8),
        DTypeDescriptor::portable(ComputeDType::BrainFloat16),
    );

    assert_eq!(dtype.storage_size_bytes(8).unwrap(), 8);
    assert_eq!(dtype.compute_workspace_bytes(8).unwrap(), 16);
}

#[test]
fn reference_cpu_copy_tensor_admitted_rejects_a_missing_source() {
    let executor = ReferenceCpuExecutor::new();
    let mut memory = MemoryManager::default();
    let error = executor
        .copy_tensor_admitted(
            &mut memory,
            &TensorResourceId::new("does-not-exist"),
            TensorResourceId::new("copy-destination-3"),
            MemoryAllocationClass::Tensor,
            MemoryAllocationOwner::Session("test-session".into()),
        )
        .expect_err("copying a nonexistent source must fail structurally");
    assert!(matches!(error, TensorValueAdmissionError::Memory(_)));
}

#[test]
fn inference_api_admission_state_reports_structured_backpressure() {
    assert_eq!(
        AdmissionState::from(&MemoryAdmissionDecision::Admit {
            reason: "ok".into()
        }),
        AdmissionState::Accepted
    );
    assert!(matches!(
        AdmissionState::from(&MemoryAdmissionDecision::Queue {
            reason: "busy".into()
        }),
        AdmissionState::Queued { .. }
    ));
    assert!(matches!(
        AdmissionState::from(&MemoryAdmissionDecision::Reject {
            reason: "no memory".into()
        }),
        AdmissionState::Rejected { .. }
    ));
    assert!(matches!(
        AdmissionState::from(&MemoryAdmissionDecision::RetryLater {
            reason: "pressure".into()
        }),
        AdmissionState::Delayed { .. }
    ));
}

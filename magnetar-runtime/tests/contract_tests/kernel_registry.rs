use magnetar_runtime::{KernelError, KernelErrorCode};

#[test]
fn kernel_out_of_device_memory_error_round_trips_code_id_and_display() {
    let error = KernelError::KernelOutOfDeviceMemory {
        reason: "requested 8GiB, 2GiB free".into(),
    };
    assert_eq!(error.code(), KernelErrorCode::KernelOutOfDeviceMemory);
    assert_eq!(error.id(), "kernel-out-of-device-memory");
    assert_ne!(error.code(), KernelErrorCode::KernelExecutionFailed);
    let rendered = error.to_string();
    assert!(rendered.contains("out of device memory"));
    assert!(rendered.contains("8GiB"));
}

use magnetar_runtime::{
    CompletionScope, CompletionToken, CompletionTokenId, CompletionTokenState, DependencyReadiness,
    DeviceId, ExecutionDependency, ExecutionPriorityHint, ExecutionStream, ExecutionStreamClass,
    ExecutionStreamId, ExecutionStreamState, ExecutionSubmission, ExecutionSubmissionTarget,
    ExecutionSynchronizationError, MemoryAllocationId, MemoryReuseFence,
    PreparedExecutionSegmentId, PreparedKernelIdAllocator, ProviderBinding,
    ProviderCancellationLevel, ProviderSynchronizationCapability, ResourceAccessScope,
    ResourceReadiness, SynchronizationObservation, SynchronizationObservationKind,
    TensorResourceId, cancellation_preserves_physical_lifetime, cross_stream_ordered_by_dependency,
    same_stream_ordered,
};

#[test]
fn execution_stream_identity_class_lifecycle_and_priority_are_logical() {
    let stream = ExecutionStream::new(
        ExecutionStreamId::new(7, 2),
        ExecutionStreamClass::compute(),
        ProviderBinding::new("gpu"),
    )
    .unwrap()
    .with_priority(ExecutionPriorityHint::LatencySensitive);

    assert_eq!(stream.id.value(), 7);
    assert_eq!(stream.id.generation(), 2);
    assert_eq!(stream.class.as_str(), "magnetar:execution/compute");
    assert_eq!(stream.priority, ExecutionPriorityHint::LatencySensitive);
    assert_eq!(
        stream.ensure_submittable(),
        Err(ExecutionSynchronizationError::StreamNotReady)
    );

    assert!(
        ExecutionStreamClass::transfer()
            .as_str()
            .ends_with("/transfer")
    );
    assert!(
        ExecutionStreamClass::control()
            .as_str()
            .ends_with("/control")
    );
    assert!(matches!(
        ExecutionStreamClass::new("CUDA stream"),
        Err(ExecutionSynchronizationError::NativeSynchronizationLeak)
    ));
}

#[test]
fn stream_state_transitions_support_drain_before_close() {
    let mut stream = ready_stream(1);

    stream
        .transition_to(ExecutionStreamState::Draining)
        .unwrap();
    assert_eq!(
        stream.ensure_submittable(),
        Err(ExecutionSynchronizationError::StreamDraining)
    );
    stream.transition_to(ExecutionStreamState::Closed).unwrap();
    assert_eq!(stream.state, ExecutionStreamState::Closed);
    assert!(matches!(
        stream.transition_to(ExecutionStreamState::Ready),
        Err(ExecutionSynchronizationError::InvalidStreamState)
    ));
}

#[test]
fn completion_token_terminal_state_is_set_once_and_aba_safe() {
    let stream = ready_stream(1);
    let reused_number_old = CompletionTokenId::new(9, 1);
    let reused_number_new = CompletionTokenId::new(9, 2);
    assert_ne!(reused_number_old, reused_number_new);

    let mut token = CompletionToken::pending(reused_number_old, &stream, CompletionScope::Transfer);
    token
        .transition_to(CompletionTokenState::Completed)
        .unwrap();
    assert_eq!(
        token.transition_to(CompletionTokenState::Failed),
        Err(ExecutionSynchronizationError::CompletionAlreadyTerminal)
    );
}

#[test]
fn completion_scopes_cover_kernel_transfer_segment_and_grouped_work() {
    let stream = ready_stream(1);
    let mut allocator = PreparedKernelIdAllocator::default();
    let kernel = CompletionToken::pending(
        CompletionTokenId::new(1, 1),
        &stream,
        CompletionScope::Kernel(allocator.allocate()),
    );
    let transfer = CompletionToken::pending(
        CompletionTokenId::new(2, 1),
        &stream,
        CompletionScope::Transfer,
    );
    let segment = CompletionToken::pending(
        CompletionTokenId::new(3, 1),
        &stream,
        CompletionScope::PreparedSegment(PreparedExecutionSegmentId::new("segment").unwrap()),
    );
    let grouped = CompletionToken::pending(
        CompletionTokenId::new(4, 1),
        &stream,
        CompletionScope::GroupedSubmission,
    );

    assert!(matches!(kernel.scope, CompletionScope::Kernel(_)));
    assert!(matches!(transfer.scope, CompletionScope::Transfer));
    assert!(matches!(segment.scope, CompletionScope::PreparedSegment(_)));
    assert!(matches!(grouped.scope, CompletionScope::GroupedSubmission));
}

#[test]
fn dependencies_block_until_predecessors_complete_and_propagate_failure() {
    let stream = ready_stream(1);
    let pending = CompletionToken::pending(
        CompletionTokenId::new(1, 1),
        &stream,
        CompletionScope::Transfer,
    );
    let dependency = ExecutionDependency::new(stream.provider.clone(), [pending.id]).unwrap();

    assert_eq!(
        dependency
            .validate_against_tokens([pending.clone()])
            .unwrap(),
        DependencyReadiness::Pending(vec![pending.id])
    );

    let completed = CompletionToken::completed(pending.id, &stream, CompletionScope::Transfer);
    assert!(
        dependency
            .validate_against_tokens([completed])
            .unwrap()
            .is_ready()
    );

    let mut failed = pending;
    failed.mark_failed("launch failed").unwrap();
    assert_eq!(
        dependency.validate_against_tokens([failed]),
        Err(ExecutionSynchronizationError::DependencyFailed)
    );
}

#[test]
fn same_stream_ordering_and_cross_stream_dependency_are_explicit() {
    let stream_a = ready_stream(1);
    let stream_b = ready_stream(2);
    let producer = CompletionTokenId::new(44, 1);
    let first = ExecutionSubmission::new(stream_a.clone(), ExecutionSubmissionTarget::Transfer);
    let mut second =
        ExecutionSubmission::new(stream_a.clone(), ExecutionSubmissionTarget::Transfer);
    assert!(same_stream_ordered(&first, &second).unwrap());

    second.stream = stream_b;
    assert!(!same_stream_ordered(&first, &second).unwrap());
    assert!(!cross_stream_ordered_by_dependency(producer, &second));

    second
        .dependencies
        .push(ExecutionDependency::new(stream_a.provider.clone(), [producer]).unwrap());
    assert!(cross_stream_ordered_by_dependency(producer, &second));
}

#[test]
fn submission_validation_rejects_unready_stream_and_device_mismatch() {
    let not_ready = ExecutionStream::new(
        ExecutionStreamId::new(1, 1),
        ExecutionStreamClass::compute(),
        ProviderBinding::new("gpu"),
    )
    .unwrap();
    let submission = ExecutionSubmission::new(not_ready, ExecutionSubmissionTarget::Transfer);
    assert_eq!(
        submission.validate(),
        Err(ExecutionSynchronizationError::StreamNotReady)
    );

    let stream = ready_stream(1).with_device(magnetar_runtime::DeviceBinding::new(DeviceId::new(
        "device-a",
    )));
    let dependency =
        ExecutionDependency::new(stream.provider.clone(), [CompletionTokenId::new(1, 1)])
            .unwrap()
            .with_device(magnetar_runtime::DeviceBinding::new(DeviceId::new(
                "device-b",
            )));
    let mut submission = ExecutionSubmission::new(stream, ExecutionSubmissionTarget::Transfer);
    submission.dependencies.push(dependency);
    assert_eq!(
        submission.validate(),
        Err(ExecutionSynchronizationError::DeviceMismatch)
    );
}

#[test]
fn resource_readiness_and_memory_reuse_wait_for_completion() {
    let writer = CompletionTokenId::new(22, 1);
    let mut readiness = ResourceReadiness::pending_write(TensorResourceId::new("tensor"), writer);
    assert!(readiness.blocks_host_read());
    assert!(readiness.blocks_device_consumer());
    readiness.mark_completed_by(writer).unwrap();
    assert!(!readiness.blocks_host_read());
    assert!(!readiness.blocks_device_consumer());

    let mut fence = MemoryReuseFence::new(MemoryAllocationId::new(1)).retain(writer);
    assert!(!fence.is_reusable());
    fence.release(writer);
    assert!(fence.is_reusable());

    let ready =
        ResourceReadiness::ready(TensorResourceId::new("host"), ResourceAccessScope::HostRead);
    assert!(!ready.blocks_host_read());
}

#[test]
fn cancellation_is_separate_from_physical_completion() {
    let stream = ready_stream(1);
    let mut token = CompletionToken::pending(
        CompletionTokenId::new(1, 1),
        &stream,
        CompletionScope::Transfer,
    );
    assert!(cancellation_preserves_physical_lifetime(&token));
    token
        .transition_to(CompletionTokenState::Cancelled)
        .unwrap();
    assert!(!cancellation_preserves_physical_lifetime(&token));
}

#[test]
fn provider_synchronization_capabilities_include_sync_baseline_and_async_profile() {
    let sync = ProviderSynchronizationCapability::synchronous_baseline();
    assert!(!sync.asynchronous_submission);
    assert!(sync.ordered_streams);
    assert_eq!(sync.cancellation, ProviderCancellationLevel::NotSupported);

    let async_capable = ProviderSynchronizationCapability::async_capable();
    assert!(async_capable.asynchronous_submission);
    assert!(async_capable.cross_stream_dependencies);
    assert!(async_capable.device_side_dependencies);
    assert!(async_capable.transfer_overlap);
}

#[test]
fn synchronization_observability_redacts_native_handles() {
    let observation = SynchronizationObservation::new(
        SynchronizationObservationKind::DependencyWait,
        "waiting on CUstream handle=0xdeadbeef",
    );

    assert_eq!(observation.message, "[redacted]");
}

fn ready_stream(value: u64) -> ExecutionStream {
    let mut stream = ExecutionStream::new(
        ExecutionStreamId::new(value, 1),
        ExecutionStreamClass::compute(),
        ProviderBinding::new("gpu"),
    )
    .unwrap();
    stream.transition_to(ExecutionStreamState::Ready).unwrap();
    stream
}

use magnetar_runtime::{
    CapabilityVersion, ComputeDType, DTypeDescriptor, DeviceBinding, DeviceId, DeviceStatus,
    FallbackClass, HealthState, KernelAdapterMetadata, KernelAdvertisement, KernelBatchMetadata,
    KernelCandidateRejection, KernelDispatchLifecycleState, KernelDispatchPlan,
    KernelDispatchPlanId, KernelDispatchRevalidationContext, KernelDispatcher, KernelExecutionMode,
    KernelFallbackStep, KernelId, KernelImplementationFamily, KernelInvocationId,
    KernelKvCacheMetadata, KernelMemoryClass, KernelOperatorVersionRange,
    KernelPrefixCacheMetadata, KernelRegistrationAuthority, KernelRegistry, KernelRegistryError,
    KernelSelectionRequest, OperatorFamily, OperatorId, Provider, ProviderBinding, ProviderError,
    ProviderExecutionError, ProviderExecutionErrorCode, ProviderExecutionPhase, ProviderMetadata,
    ProviderRegistry, ProviderStatusSnapshot, ResourceAffinity, Runtime, ShapeDescriptor,
    TensorDescriptor, TensorLayoutKind, TensorResourceDescriptor, TensorResourceId,
    kernel_dispatch_error_from_provider_execution,
};
use std::{collections::BTreeSet, sync::Arc};

fn matmul_advertisement(provider: &str) -> KernelAdvertisement {
    let operator = OperatorId::magnetar("matmul", 1, OperatorFamily::LinearAlgebra);
    let id = KernelId::new(
        ProviderBinding::new(provider),
        format!("{provider}.matmul.fp16"),
        CapabilityVersion::new(1, 0, 0),
        operator,
        KernelOperatorVersionRange::exact(1),
        KernelImplementationFamily::TestFixture,
    )
    .with_conformance_profile("kernel-standard-v1");
    let mut advertisement = KernelAdvertisement::new(id)
        .with_dtypes(magnetar_runtime::TensorRole::Input, [ComputeDType::Float16])
        .with_dtypes(
            magnetar_runtime::TensorRole::Output,
            [ComputeDType::Float16],
        )
        .with_layouts([TensorLayoutKind::Contiguous])
        .with_memory_classes([KernelMemoryClass::Device])
        .with_devices([DeviceBinding::new(DeviceId::new(format!(
            "{provider}:device0"
        )))]);
    advertisement.batching = Some(KernelBatchMetadata {
        max_batch_size: Some(8),
        max_active_sequences: Some(8),
        max_total_tokens: Some(1024),
        supports_ragged_batches: true,
        supports_paged_kv_cache: true,
        per_operation_output_mapping: true,
        batch_slot_compatible: true,
    });
    advertisement.adapter = Some(KernelAdapterMetadata {
        methods: BTreeSet::from(["lora".into()]),
        max_rank: Some(16),
        dtypes: BTreeSet::from([ComputeDType::Float16]),
        merge_strategy: Some("runtime".into()),
        target_modules: BTreeSet::from(["q_proj".into()]),
    });
    advertisement.kv_cache = Some(KernelKvCacheMetadata {
        layouts: BTreeSet::from(["paged".into()]),
        paged_cache: true,
        append: true,
        read: true,
        dtypes: BTreeSet::from([ComputeDType::Float16]),
        memory_classes: BTreeSet::from([KernelMemoryClass::Device]),
        affinity: None,
    });
    advertisement.prefix_cache = Some(KernelPrefixCacheMetadata {
        supports_adjusted_sequence_length: true,
        supports_adjusted_context_length: true,
        supports_reused_prefix_boundary: true,
    });
    advertisement
}

fn resource(id: &str, provider: &str, dtype: ComputeDType) -> magnetar_runtime::KernelResource {
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([8, 8]),
        DTypeDescriptor::portable(dtype),
    );
    magnetar_runtime::KernelResource::new(
        TensorResourceDescriptor::new(
            TensorResourceId::new(id),
            descriptor,
            ResourceAffinity::new(FallbackClass::ProviderPinned)
                .with_provider(ProviderBinding::new(provider)),
        ),
        KernelMemoryClass::Device,
    )
}

fn selection_request(provider: &str) -> KernelSelectionRequest {
    let operator = OperatorId::magnetar("matmul", 1, OperatorFamily::LinearAlgebra);
    let mut request = KernelSelectionRequest::new(
        "selection:1",
        operator,
        ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(ProviderBinding::new(provider)),
    )
    .with_input(resource("input", provider, ComputeDType::Float16))
    .with_output(resource("output", provider, ComputeDType::Float16));
    request.dtype_requirements.insert(ComputeDType::Float16);
    request
        .layout_requirements
        .insert(TensorLayoutKind::Contiguous);
    request
        .memory_class_requirements
        .insert(KernelMemoryClass::Device);
    request.execution_mode = Some(KernelExecutionMode::Synchronous);
    request.require_conformance = true;
    request
}

fn unavailable_provider_status(provider: &str) -> ProviderStatusSnapshot {
    ProviderStatusSnapshot::from_health_report(magnetar_runtime::ProviderHealthReport::new(
        ProviderBinding::new(provider),
        HealthState::Initializing,
    ))
}

fn unavailable_device_status(provider: &str) -> DeviceStatus {
    DeviceStatus::from_health(magnetar_runtime::DeviceHealth::new(
        ProviderBinding::new(provider),
        DeviceBinding::new(DeviceId::new(format!("{provider}:device0"))),
        HealthState::Unavailable,
    ))
}

#[test]
fn registry_accepts_provider_and_fixture_advertisements_only() {
    let mut registry = KernelRegistry::new();
    registry
        .register_provider_advertisement(matmul_advertisement("provider-a"))
        .expect("Provider advertisements are trusted registry input");
    registry
        .register_fixture_advertisement(matmul_advertisement("fixture"))
        .expect("Runtime fixtures may seed test kernels");

    let error = registry
        .register_advertisement(
            matmul_advertisement("client"),
            KernelRegistrationAuthority::Client,
        )
        .expect_err("clients must not register Kernels directly");
    assert_eq!(
        error,
        KernelRegistryError::RegistrationDenied(KernelRegistrationAuthority::Client)
    );
}

#[test]
fn registry_rejects_invalid_advertisements_before_insertion() {
    let mut registry = KernelRegistry::new();
    let mut advertisement = matmul_advertisement("provider-a");
    advertisement.implemented_operator =
        OperatorId::magnetar("attention", 1, OperatorFamily::Attention);

    let error = registry
        .register_provider_advertisement(advertisement)
        .expect_err("mismatched Operator metadata must fail validation");
    assert_eq!(error.code(), "kernel-advertisement-invalid");
    assert_eq!(registry.entries().count(), 0);
}

#[test]
fn selection_filters_candidates_by_metadata_and_policy() {
    let mut registry = KernelRegistry::new();
    registry
        .register_provider_advertisement(matmul_advertisement("provider-a"))
        .unwrap();

    let mut request = selection_request("provider-a");
    let result = registry
        .select(&request)
        .expect("matching Kernel should be selected");
    assert_eq!(result.selected.unwrap().provider.as_str(), "provider-a");

    request.dtype_requirements.clear();
    request
        .dtype_requirements
        .insert(ComputeDType::BrainFloat16);
    let error = registry
        .select(&request)
        .expect_err("unsupported dtype must reject selection");
    assert_eq!(
        error,
        KernelRegistryError::CandidateIncompatible {
            reason: KernelCandidateRejection::DTypeUnsupported
        }
    );
}

#[test]
fn selection_filters_provider_device_shape_workspace_conformance_and_policy() {
    let mut registry = KernelRegistry::new();
    registry
        .register_provider_advertisement(matmul_advertisement("provider-a"))
        .unwrap();
    registry.set_provider_status(unavailable_provider_status("provider-a"));
    let error = registry
        .select(&selection_request("provider-a"))
        .expect_err("not-ready Provider must be rejected");
    assert_eq!(
        error,
        KernelRegistryError::CandidateIncompatible {
            reason: KernelCandidateRejection::StaleRegistryEntry
        }
    );

    let mut registry = KernelRegistry::new();
    registry
        .register_provider_advertisement(matmul_advertisement("provider-a"))
        .unwrap();
    registry.set_device_status(unavailable_device_status("provider-a"));
    let error = registry
        .select(&selection_request("provider-a"))
        .expect_err("unavailable Device must be rejected");
    assert_eq!(
        error,
        KernelRegistryError::CandidateIncompatible {
            reason: KernelCandidateRejection::StaleRegistryEntry
        }
    );

    let mut registry = KernelRegistry::new();
    let mut advertisement = matmul_advertisement("provider-a");
    advertisement.shape.rank = Some(3);
    registry
        .register_provider_advertisement(advertisement)
        .unwrap();
    let error = registry
        .select(&selection_request("provider-a"))
        .expect_err("shape mismatch must be rejected");
    assert_eq!(
        error,
        KernelRegistryError::CandidateIncompatible {
            reason: KernelCandidateRejection::ShapeUnsupported
        }
    );

    let mut registry = KernelRegistry::new();
    let mut advertisement = matmul_advertisement("provider-a");
    advertisement.workspace =
        magnetar_runtime::KernelWorkspaceRequirements::required(0, KernelMemoryClass::Device, 1);
    registry
        .register_provider_advertisement(advertisement)
        .unwrap();
    let error = registry
        .select(&selection_request("provider-a"))
        .expect_err("zero-sized required workspace models infeasible workspace");
    assert_eq!(
        error,
        KernelRegistryError::CandidateIncompatible {
            reason: KernelCandidateRejection::WorkspaceUnavailable
        }
    );

    let mut registry = KernelRegistry::new();
    let mut advertisement = matmul_advertisement("provider-a");
    advertisement.id.conformance_profile = None;
    registry
        .register_provider_advertisement(advertisement)
        .unwrap();
    let error = registry
        .select(&selection_request("provider-a"))
        .expect_err("missing conformance must be rejected when required");
    assert_eq!(
        error,
        KernelRegistryError::CandidateIncompatible {
            reason: KernelCandidateRejection::ConformanceMissing
        }
    );

    let mut registry = KernelRegistry::new();
    let mut advertisement = matmul_advertisement("provider-a");
    advertisement.determinism.deterministic = false;
    registry
        .register_provider_advertisement(advertisement)
        .unwrap();
    let mut request = selection_request("provider-a");
    request.deterministic_required = true;
    let error = registry
        .select(&request)
        .expect_err("determinism policy must reject nondeterministic Kernel");
    assert_eq!(
        error,
        KernelRegistryError::CandidateIncompatible {
            reason: KernelCandidateRejection::PolicyDenied
        }
    );
}

#[test]
fn selection_validates_required_features_batch_adapter_kv_and_prefix_cache() {
    let mut advertisement = matmul_advertisement("provider-a");
    advertisement
        .required_provider_features
        .insert("tensorcore".into());
    advertisement.required_device_features.insert("fp16".into());

    let mut registry = KernelRegistry::new();
    registry
        .register_provider_advertisement(advertisement)
        .unwrap();
    let error = registry
        .select(&selection_request("provider-a"))
        .expect_err("missing Provider feature must reject candidate");
    assert_eq!(
        error,
        KernelRegistryError::CandidateIncompatible {
            reason: KernelCandidateRejection::ProviderFeatureMissing
        }
    );

    registry.set_provider_features(ProviderBinding::new("provider-a"), ["tensorcore"]);
    let error = registry
        .select(&selection_request("provider-a"))
        .expect_err("missing Device feature must reject candidate");
    assert_eq!(
        error,
        KernelRegistryError::CandidateIncompatible {
            reason: KernelCandidateRejection::DeviceFeatureMissing
        }
    );

    registry.set_device_features(
        DeviceBinding::new(DeviceId::new("provider-a:device0")),
        ["fp16"],
    );
    let mut request = selection_request("provider-a");
    request.batching = Some(KernelBatchMetadata {
        max_batch_size: Some(16),
        max_active_sequences: Some(16),
        max_total_tokens: Some(2048),
        supports_ragged_batches: true,
        supports_paged_kv_cache: true,
        per_operation_output_mapping: true,
        batch_slot_compatible: true,
    });
    assert_eq!(
        registry.select(&request).unwrap_err(),
        KernelRegistryError::CandidateIncompatible {
            reason: KernelCandidateRejection::BatchingUnsupported
        }
    );

    let mut request = selection_request("provider-a");
    request.adapter_methods.insert("qlora".into());
    assert_eq!(
        registry.select(&request).unwrap_err(),
        KernelRegistryError::CandidateIncompatible {
            reason: KernelCandidateRejection::AdapterUnsupported
        }
    );

    let mut request = selection_request("provider-a");
    request.kv_cache = Some(KernelKvCacheMetadata {
        layouts: BTreeSet::from(["blocked".into()]),
        paged_cache: true,
        append: true,
        read: true,
        dtypes: BTreeSet::from([ComputeDType::Float16]),
        memory_classes: BTreeSet::from([KernelMemoryClass::Device]),
        affinity: None,
    });
    assert_eq!(
        registry.select(&request).unwrap_err(),
        KernelRegistryError::CandidateIncompatible {
            reason: KernelCandidateRejection::KvCacheUnsupported
        }
    );

    let mut request = selection_request("provider-a");
    request.prefix_cache = Some(KernelPrefixCacheMetadata {
        supports_adjusted_sequence_length: true,
        supports_adjusted_context_length: true,
        supports_reused_prefix_boundary: true,
    });
    assert!(registry.select(&request).is_ok());
}

#[test]
fn selection_ranks_policy_and_emits_observations() {
    let mut slow = matmul_advertisement("slow-provider");
    slow.performance_hints
        .insert("estimated-cost".into(), "100".into());
    slow.performance_hints
        .insert("fallback-rank".into(), "10".into());
    let mut fast = matmul_advertisement("fast-provider");
    fast.performance_hints
        .insert("estimated-cost".into(), "1".into());
    fast.performance_hints
        .insert("fallback-rank".into(), "0".into());

    let mut registry = KernelRegistry::new();
    registry.register_provider_advertisement(slow).unwrap();
    registry.register_provider_advertisement(fast).unwrap();
    let mut request = selection_request("fast-provider");
    request.affinity = ResourceAffinity::new(FallbackClass::Transparent);
    request.outputs[0].resource.affinity = ResourceAffinity::new(FallbackClass::Transparent);
    let result = registry.select(&request).unwrap();
    assert_eq!(result.selected.unwrap().provider.as_str(), "fast-provider");
    assert!(
        result
            .observations
            .iter()
            .any(|observation| observation.kind
                == magnetar_runtime::KernelObservationKind::KernelCandidateLookup)
    );
    assert!(
        result
            .observations
            .iter()
            .any(|observation| observation.kind
                == magnetar_runtime::KernelObservationKind::KernelCandidateRanked)
    );
}

#[test]
fn dispatch_plan_is_runtime_created_metadata_and_revalidates_fail_closed() {
    let mut registry = KernelRegistry::new();
    let advertisement = matmul_advertisement("provider-a");
    registry
        .register_provider_advertisement(advertisement.clone())
        .unwrap();
    let request = selection_request("provider-a");
    let selection = registry.select(&request).unwrap();
    let candidate = selection.selected.as_ref().unwrap();
    let mut plan = KernelDispatchPlan::from_selection(
        KernelDispatchPlanId::new("dispatch-plan:1"),
        &request,
        candidate,
        &advertisement,
        KernelInvocationId::new("invocation:1"),
    )
    .expect("selection should become a Runtime dispatch plan");
    assert!(plan.without_raw_handles());
    assert_eq!(plan.provider.as_str(), "provider-a");

    let mut dispatcher = KernelDispatcher::new();
    dispatcher
        .revalidate(&registry, &mut plan)
        .expect("active registry entry should revalidate");
    assert_eq!(plan.lifecycle, KernelDispatchLifecycleState::Ready);

    registry.invalidate_provider(&ProviderBinding::new("provider-a"), "provider failed");
    let error = dispatcher
        .revalidate(&registry, &mut plan)
        .expect_err("stale Kernel must fail closed");
    assert_eq!(error.code(), "kernel-dispatch-stale");
}

#[test]
fn dispatch_revalidates_lifecycle_memory_adapter_kv_prefix_and_observes_failures() {
    let mut registry = KernelRegistry::new();
    let advertisement = matmul_advertisement("provider-a");
    registry
        .register_provider_advertisement(advertisement.clone())
        .unwrap();
    let request = selection_request("provider-a");
    let selection = registry.select(&request).unwrap();
    let candidate = selection.selected.as_ref().unwrap();
    let mut plan = KernelDispatchPlan::from_selection(
        KernelDispatchPlanId::new("dispatch-plan:2"),
        &request,
        candidate,
        &advertisement,
        KernelInvocationId::new("invocation:2"),
    )
    .unwrap();
    assert!(
        plan.fallback_chain
            .iter()
            .any(|step| matches!(step, KernelFallbackStep::ExplicitDataMovement(_)))
    );
    assert!(!plan.conversion_steps.is_empty());

    let mut dispatcher = KernelDispatcher::new();
    let context = KernelDispatchRevalidationContext {
        provider_status: Some(unavailable_provider_status("provider-a")),
        ..KernelDispatchRevalidationContext::default()
    };
    let error = dispatcher
        .revalidate_with_context(&registry, &mut plan, &context)
        .expect_err("Provider status must be rechecked");
    assert_eq!(error.code(), "kernel-provider-not-ready");

    let mut plan = KernelDispatchPlan::from_selection(
        KernelDispatchPlanId::new("dispatch-plan:3"),
        &request,
        candidate,
        &advertisement,
        KernelInvocationId::new("invocation:3"),
    )
    .unwrap();
    let context = KernelDispatchRevalidationContext {
        memory_reservation_valid: false,
        ..KernelDispatchRevalidationContext::default()
    };
    assert_eq!(
        dispatcher
            .revalidate_with_context(&registry, &mut plan, &context)
            .unwrap_err()
            .code(),
        "kernel-workspace-unavailable"
    );

    for context in [
        KernelDispatchRevalidationContext {
            operation_active: false,
            ..KernelDispatchRevalidationContext::default()
        },
        KernelDispatchRevalidationContext {
            session_active: false,
            ..KernelDispatchRevalidationContext::default()
        },
        KernelDispatchRevalidationContext {
            model_instance_ready: false,
            ..KernelDispatchRevalidationContext::default()
        },
        KernelDispatchRevalidationContext {
            batching_valid: false,
            ..KernelDispatchRevalidationContext::default()
        },
        KernelDispatchRevalidationContext {
            adapter_valid: false,
            ..KernelDispatchRevalidationContext::default()
        },
        KernelDispatchRevalidationContext {
            kv_cache_valid: false,
            ..KernelDispatchRevalidationContext::default()
        },
        KernelDispatchRevalidationContext {
            prefix_cache_valid: false,
            ..KernelDispatchRevalidationContext::default()
        },
    ] {
        let mut plan = KernelDispatchPlan::from_selection(
            KernelDispatchPlanId::new("dispatch-plan:loop"),
            &request,
            candidate,
            &advertisement,
            KernelInvocationId::new("invocation:loop"),
        )
        .unwrap();
        assert!(
            dispatcher
                .revalidate_with_context(&registry, &mut plan, &context)
                .is_err()
        );
    }

    dispatcher.record_fallback_considered(&plan);
    dispatcher.record_fallback_selected(&plan);
    dispatcher.record_fallback_failed(&plan);
    dispatcher.record_conformance_gating(&plan);
    assert!(dispatcher.observations().iter().any(|observation| {
        observation.kind == magnetar_runtime::KernelObservationKind::KernelDispatchFailed
    }));
    assert!(dispatcher.observations().iter().any(|observation| {
        observation.kind == magnetar_runtime::KernelObservationKind::KernelFallbackConsidered
    }));
}

#[test]
fn provider_dispatch_errors_map_to_kernel_dispatch_errors() {
    let error = ProviderExecutionError::new(
        ProviderExecutionErrorCode::ExecutionFailed,
        ProviderExecutionPhase::Complete,
        ProviderBinding::new("provider-a"),
        None,
        "native failure 0x1234",
    );
    let mapped = kernel_dispatch_error_from_provider_execution(error);
    assert_eq!(mapped.code(), "kernel-dispatch-failed");

    let error = ProviderExecutionError::new(
        ProviderExecutionErrorCode::OutOfMemory,
        ProviderExecutionPhase::Submit,
        ProviderBinding::new("provider-a"),
        Some(DeviceBinding::new(DeviceId::new("provider-a:device0"))),
        "oom",
    );
    let mapped = kernel_dispatch_error_from_provider_execution(error);
    assert_eq!(mapped.code(), "kernel-memory-infeasible");
}

struct KernelAdvertisingProvider {
    advertisement: KernelAdvertisement,
}

impl Provider for KernelAdvertisingProvider {
    fn metadata(&self) -> ProviderMetadata {
        ProviderMetadata::new(
            self.advertisement.id.provider.as_str(),
            "1.0.0",
            "test",
            "Kernel advertising test Provider",
        )
    }

    fn register(&self, _registry: &mut ProviderRegistry) -> Result<(), ProviderError> {
        Ok(())
    }

    fn kernel_advertisements(&self) -> Vec<KernelAdvertisement> {
        vec![self.advertisement.clone()]
    }
}

#[test]
fn runtime_owns_registry_and_indexes_provider_kernel_advertisements() {
    let provider = Arc::new(KernelAdvertisingProvider {
        advertisement: matmul_advertisement("provider-a"),
    });
    let runtime = Runtime::builder()
        .register_provider(provider)
        .build()
        .expect("Provider should register");

    let request = selection_request("provider-a");
    let result = runtime
        .kernel_registry()
        .select(&request)
        .expect("Runtime-owned registry should select Provider Kernel");
    assert_eq!(result.selected.unwrap().provider.as_str(), "provider-a");
}

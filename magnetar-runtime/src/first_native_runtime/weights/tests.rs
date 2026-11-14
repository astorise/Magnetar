//! Unit tests for the parent module.
//!
//! Shared mock executors and fixtures live in `first_native_runtime/tests.rs`.

use super::super::tests::*;
use super::*;
use std::sync::atomic::Ordering;

/// P0 fix regression: `WeightMaterializationTransaction::stage_weight`
/// propagates a real Provider write failure as
/// `InferenceApiError::ProviderTensorWriteFailed` instead of ignoring a
/// bare `()`, and rolls back the Memory Manager allocation it had already
/// admitted for that weight before propagating -- the same rollback shape
/// this transaction already applies to a residency-registration failure.
#[test]
fn stage_weight_propagates_and_rolls_back_on_provider_write_failure() {
    let fixture = e2e_fixture().expect("fixture builds");
    let mut runtime = build_runtime_trusting_fixture(&fixture);
    let (instance, _memory) =
        load_fixture_instance(&fixture, &mut runtime).expect("instance loads");

    let executor = Arc::new(FailableProviderExecutionApi::new());
    executor.fail_write.store(true, Ordering::SeqCst);
    let mut transaction = WeightMaterializationTransaction {
        provider_binding: ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        executor,
        staged: Vec::new(),
    };
    let tensor = HostTensor::new([4, 8], vec![0.0; 32]).unwrap();
    let error = transaction
        .stage_weight(
            &mut runtime,
            &instance,
            "test",
            "transformer.wte.weight",
            &tensor,
        )
        .unwrap_err();
    assert!(matches!(
        error,
        InferenceApiError::ProviderTensorWriteFailed { .. }
    ));
    // `MemoryManager::allocations()` is a permanent ledger -- a released
    // allocation stays present with `state: Released`, it does not
    // disappear -- so "rolled back, not leaked" means the allocation this
    // attempt admitted (the highest-numbered one, ids being assigned
    // sequentially) ended in `Released` state, not `Active`.
    let this_attempts_allocation = runtime
        .memory()
        .allocations()
        .max_by_key(|allocation| allocation.id)
        .expect("at least one allocation exists");
    assert_eq!(
        this_attempts_allocation.state,
        MemoryAllocationState::Released,
        "the allocation admitted just before the failed write must be rolled back, not leaked"
    );
}

/// P0 fix regression: `Runtime::unload_model_instance` propagates a real
/// Provider release failure as a structured `ModelInstanceError` instead
/// of ignoring a bare `bool`.
#[test]
fn unload_model_instance_propagates_provider_release_failure() {
    let fixture = e2e_fixture().expect("fixture builds");
    let executor = Arc::new(FailableProviderExecutionApi::new());
    executor.fail_release.store(true, Ordering::SeqCst);
    let mut runtime = Runtime::builder()
        .register_provider(Arc::new(TestFailableProvider {
            metadata: ProviderMetadata::new(
                REFERENCE_CPU_PROVIDER_NAME,
                "test",
                "test",
                "test-only Provider whose release_tensor always fails",
            ),
            executor,
        }))
        .trust_store(
            ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone()),
        )
        .build()
        .unwrap();
    // `load_fixture_instance` materializes the fixture's weights internally
    // (via `bind_qwen_fixture_weights`) and reaches `Ready` -- this
    // succeeds because only `fail_release`, not `fail_write`, is set.
    let (instance, _memory) =
        load_fixture_instance(&fixture, &mut runtime).expect("instance loads and reaches Ready");
    assert_eq!(
        runtime.model_instance(&instance).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Ready
    );

    let error = runtime
        .unload_model_instance(&instance, ModelInstanceUnloadPolicy::DrainActiveUse)
        .unwrap_err();
    assert!(matches!(
        error,
        ModelInstanceError::InternalModelInstance { .. }
    ));
}

/// `generalize-first-native-provider-dispatch` P0 fix regression: a Model
/// Instance whose placement binds a non-Reference-CPU registered Provider
/// materializes its weights through *that* Provider's storage, not
/// Reference CPU's -- `WeightMaterializationTransaction::begin()` used to
/// hardcode Reference CPU unconditionally, which would have silently
/// written this weight into the wrong Provider (or failed to find it later
/// under the right one).
#[test]
fn weight_materialization_uses_the_model_instances_bound_provider() {
    let fixture = e2e_fixture().expect("fixture builds");
    let mock_provider = ProviderBinding::new("magnetar:provider/mock-kernel");
    let mut runtime = Runtime::builder()
        .register_provider(Arc::new(ReferenceCpuProvider::new()))
        .register_provider(Arc::new(MockKernelProvider::new()))
        .trust_store(
            ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone()),
        )
        .build()
        .expect("both providers register cleanly");

    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(fixture.architecture_implementation.clone());
    let mut request = ModelLoadingRequest::new(
        ModelLoadingRequestId::new("provider-dispatch-weight-test"),
        fixture.manifest.id.clone(),
    );
    request.quantization_policy = ModelQuantizationPolicy::RejectUnsupported;
    let loaded = load_model(
        &mut coordinator,
        &mut runtime,
        ModelLoadingApiRequest::new(request),
        &fixture.manifest,
    )
    .expect("model loads");
    let instance = create_model_instance(
        &mut runtime,
        &loaded,
        fixture.architecture_implementation.clone(),
        ResourceAffinity::new(FallbackClass::Transparent).with_provider(mock_provider.clone()),
    )
    .expect("instance binds to the mock provider (loading resolved no conflicting binding)");

    let weight_name = "test.arbitrary.weight";
    let tensor = HostTensor::new([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    materialize_model_instance_weights(
        &mut runtime,
        &instance,
        "test",
        &BTreeMap::from([(weight_name.to_string(), tensor.clone())]),
    )
    .expect("weight materializes through the bound (mock) provider");

    let resource_id = runtime
        .model_instance(&instance)
        .unwrap()
        .definition()
        .resource_bindings
        .weights
        .get(weight_name)
        .cloned()
        .expect("weight binding recorded");

    let mock_api = resolve_kernel_execution_provider(&runtime, &mock_provider)
        .expect("mock provider resolves");
    assert_eq!(
        mock_api.read_tensor(&resource_id),
        Some(tensor),
        "weight must be readable from the bound mock Provider's storage"
    );

    let cpu_api = resolve_kernel_execution_provider(
        &runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )
    .expect("reference CPU provider resolves");
    assert!(
        cpu_api.read_tensor(&resource_id).is_none(),
        "weight must not have been written to Reference CPU's storage"
    );
}

#[test]
fn e2e_materialize_model_instance_weights_rejects_content_digest_mismatch() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_materialize_model_instance_weights_rejects_content_digest_mismatch(&fixture)
        .expect("materialize_model_instance_weights rejects tampered tensor content");
}

#[test]
fn e2e_materialize_model_instance_weights_accepts_matching_content() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_materialize_model_instance_weights_accepts_matching_content(&fixture)
        .expect("materialize_model_instance_weights accepts real, untampered tensor content");
}

#[test]
fn e2e_weight_materialization_failure_never_reaches_ready() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_weight_materialization_failure_never_reaches_ready(&fixture).expect(
        "a Model Instance never reports Ready when weight materialization fails, and rolls \
         back every weight staged in the failed attempt",
    );
}

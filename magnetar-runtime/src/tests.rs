use crate::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};
struct TestProvider {
    metadata: ProviderMetadata,
    initialized: AtomicBool,
    shut_down: AtomicBool,
    fail_initialization: bool,
    health: ProviderHealth,
    capability_health: BTreeMap<CapabilityId, HealthState>,
    devices: Vec<Arc<dyn Device>>,
    execution_api: Option<Arc<dyn ProviderExecutionApi>>,
}
impl TestProvider {
    fn new(name: &str) -> Self {
        Self {
            metadata: ProviderMetadata::new(name, "1", "test", "test"),
            initialized: AtomicBool::new(false),
            shut_down: AtomicBool::new(false),
            fail_initialization: false,
            health: ProviderHealth::Available,
            capability_health: BTreeMap::new(),
            devices: Vec::new(),
            execution_api: None,
        }
    }
}

fn capability(name: &str, version: CapabilityVersion) -> Capability {
    Capability::new(
        CapabilityId::new(name),
        version,
        CapabilityDescriptor::new("test capability")
            .with_contract(WitInterface::new(name, version.to_string())),
    )
}
fn capability_binding(name: &str, version: CapabilityVersion) -> CapabilityBinding {
    CapabilityBinding::new(CapabilityId::new(name), version)
}
fn provider_with_capabilities(
    name: &str,
    capabilities: impl IntoIterator<Item = Capability>,
) -> TestProvider {
    let mut provider = TestProvider::new(name);
    provider.metadata.capabilities.extend(capabilities);
    provider
}
impl Provider for TestProvider {
    fn metadata(&self) -> ProviderMetadata {
        self.metadata.clone()
    }
    fn register(&self, _registry: &mut ProviderRegistry) -> Result<(), ProviderError> {
        Ok(())
    }
    fn health(&self) -> ProviderHealth {
        self.health
    }
    fn capability_health(&self, capability: &CapabilityBinding) -> Option<CapabilityHealth> {
        Some(CapabilityHealth::new(
            ProviderBinding::new(&self.metadata.name),
            capability.clone(),
            self.capability_health
                .get(capability.id())
                .copied()
                .unwrap_or(self.health),
        ))
    }
    fn initialize(&self) -> Result<(), ProviderError> {
        if self.fail_initialization {
            return Err(ProviderError::Lifecycle("unavailable".into()));
        }
        self.initialized.store(true, Ordering::SeqCst);
        Ok(())
    }
    fn devices(&self) -> Vec<Arc<dyn Device>> {
        self.devices.clone()
    }
    fn shutdown(&self) -> Result<(), ProviderError> {
        self.shut_down.store(true, Ordering::SeqCst);
        Ok(())
    }
    fn execution_api(&self) -> Option<&dyn ProviderExecutionApi> {
        self.execution_api.as_deref()
    }
}
struct TestProviderExecutionApi {
    submitted: Mutex<Vec<ProviderExecutionRequest>>,
    released: AtomicBool,
    cancel_outcome: Mutex<ProviderCancellationOutcome>,
    outputs: Mutex<Vec<TensorResourceDescriptor>>,
}
impl TestProviderExecutionApi {
    fn new() -> Self {
        Self {
            submitted: Mutex::new(Vec::new()),
            released: AtomicBool::new(false),
            cancel_outcome: Mutex::new(ProviderCancellationOutcome::Unsupported),
            outputs: Mutex::new(Vec::new()),
        }
    }
}
impl ProviderExecutionApi for TestProviderExecutionApi {
    fn submit(
        &self,
        request: ProviderExecutionRequest,
    ) -> Result<ProviderExecutionHandle, ProviderExecutionError> {
        let handle = ProviderExecutionHandle::new(
            request.operation,
            request.plan.id.clone(),
            request.provider.clone(),
            request.device.clone(),
        );
        self.submitted.lock().unwrap().push(request);
        Ok(handle)
    }
    fn status(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderExecutionStatus, ProviderExecutionError> {
        let mut status = ProviderExecutionStatus::new(handle.clone(), SchedulingState::Running);
        status.progress = Some(
            ProviderExecutionProgress::new(1, 3).with_message("provider execution is running"),
        );
        Ok(status)
    }
    fn cancel(
        &self,
        _handle: &ProviderExecutionHandle,
    ) -> Result<ProviderCancellationOutcome, ProviderExecutionError> {
        Ok(*self.cancel_outcome.lock().unwrap())
    }
    fn complete(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderExecutionResult, ProviderExecutionError> {
        Ok(ProviderExecutionResult::completed(
            handle.clone(),
            self.outputs.lock().unwrap().clone(),
        ))
    }
    fn release(&self, _handle: ProviderExecutionHandle) -> Result<(), ProviderExecutionError> {
        self.released.store(true, Ordering::SeqCst);
        Ok(())
    }
}
#[test]
fn load_valid_provider() {
    let p = Arc::new(TestProvider::new("valid"));
    let mut m = ProviderLoader::new();
    m.register_provider(p.clone()).unwrap();
    assert!(m.provider("valid").is_some());
    assert!(p.initialized.load(Ordering::SeqCst));
}
#[test]
fn runtime_enumerates_provider_devices_with_metadata() {
    let mut provider = TestProvider::new("cuda");
    let mut metadata = DeviceMetadata::new(
        DeviceId::new("cuda:gpu:0"),
        "NVIDIA Test GPU",
        DeviceType::Gpu,
        "cuda",
    );
    metadata.vendor = "NVIDIA".into();
    metadata.architecture = "Ada".into();
    metadata.memory_capacity = 24 * 1024 * 1024 * 1024;
    metadata.compute_units = 128;
    metadata
        .execution_capabilities
        .insert(CapabilityId::new("magnetar:compute/run"));
    provider
        .devices
        .push(Arc::new(DeviceDescriptor::new(metadata)));

    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let devices = runtime.devices().collect::<Vec<_>>();
    assert_eq!(devices.len(), 1);
    assert_eq!(devices[0].id().as_str(), "cuda:gpu:0");
    assert_eq!(devices[0].metadata().vendor, "NVIDIA");
    assert_eq!(
        runtime
            .providers()
            .registry()
            .provider_for_device(&DeviceId::new("cuda:gpu:0")),
        Some("cuda")
    );
}
#[test]
fn device_registration_rejects_duplicate_ids_and_mismatched_owners() {
    let device = |id: &str, provider: &str| {
        Arc::new(DeviceDescriptor::new(DeviceMetadata::new(
            DeviceId::new(id),
            "test",
            DeviceType::Gpu,
            provider,
        ))) as Arc<dyn Device>
    };
    let mut registry = ProviderRegistry::default();
    registry
        .register_devices("cuda", [device("gpu:0", "cuda")])
        .unwrap();
    assert!(matches!(
        registry.register_devices("other", [device("gpu:0", "other")]),
        Err(ProviderError::DeviceAlreadyRegistered(_))
    ));
    assert!(matches!(
        registry.register_devices("cuda", [device("gpu:2", "cuda"), device("gpu:0", "cuda")]),
        Err(ProviderError::DeviceAlreadyRegistered(_))
    ));
    assert!(registry.device(&DeviceId::new("gpu:2")).is_none());
    assert!(matches!(
        registry.register_devices("cuda", [device("gpu:1", "other")]),
        Err(ProviderError::DeviceProviderMismatch { .. })
    ));
}
#[test]
fn registers_capabilities_and_resolves_fallbacks_by_name() {
    let capability = capability("magnetar:runtime/execute", CapabilityVersion::new(1, 0, 0));
    let mut primary = TestProvider::new("a-primary");
    primary.metadata.capabilities.insert(capability.clone());
    let mut fallback = TestProvider::new("z-fallback");
    fallback.metadata.capabilities.insert(capability.clone());
    let mut loader = ProviderLoader::new();
    loader.register_provider(Arc::new(fallback)).unwrap();
    loader.register_provider(Arc::new(primary)).unwrap();

    assert!(loader.registry().has_capability(&capability));
    let names = loader
        .resolve_providers(&capability)
        .into_iter()
        .map(|provider| provider.metadata().name)
        .collect::<Vec<_>>();
    assert_eq!(names, ["a-primary", "z-fallback"]);
}
#[test]
fn semantic_versions_select_the_latest_compatible_capability() {
    let mut registry = ProviderRegistry::default();
    registry
        .register_capability(capability(
            "magnetar:compute/run",
            CapabilityVersion::new(1, 0, 0),
        ))
        .unwrap();
    registry
        .register_capability(capability(
            "magnetar:compute/run",
            CapabilityVersion::new(1, 2, 0),
        ))
        .unwrap();
    registry
        .register_capability(capability(
            "magnetar:compute/run",
            CapabilityVersion::new(2, 0, 0),
        ))
        .unwrap();

    let id = CapabilityId::new("magnetar:compute/run");
    assert_eq!(
        registry
            .resolve_capability(&id, CapabilityVersion::new(1, 1, 0))
            .unwrap()
            .version,
        CapabilityVersion::new(1, 2, 0)
    );
    assert!(
        registry
            .resolve_capability(&id, CapabilityVersion::new(3, 0, 0))
            .is_none()
    );
    assert!(!CapabilityVersion::new(0, 1, 1).is_compatible_with(CapabilityVersion::new(0, 1, 0)));
}
#[test]
fn capability_validation_rejects_invalid_and_conflicting_definitions() {
    let mut registry = ProviderRegistry::default();
    let invalid = Capability::new(
        CapabilityId::new("magnetar:invalid"),
        CapabilityVersion::new(1, 0, 0),
        CapabilityDescriptor::new("missing contract"),
    );
    assert!(matches!(
        registry.register_capability(invalid),
        Err(ProviderError::InvalidCapability(_))
    ));

    let original = capability("magnetar:compute/run", CapabilityVersion::new(1, 0, 0));
    registry.register_capability(original).unwrap();
    let conflicting = Capability::new(
        CapabilityId::new("magnetar:compute/run"),
        CapabilityVersion::new(1, 0, 0),
        CapabilityDescriptor::new("different")
            .with_contract(WitInterface::new("magnetar:compute/other", "1.0.0")),
    );
    assert!(matches!(
        registry.register_capability(conflicting),
        Err(ProviderError::ConflictingCapability { .. })
    ));
}
#[test]
fn capability_dependencies_must_resolve_compatibly() {
    let mut registry = ProviderRegistry::default();
    let dependent = Capability::new(
        CapabilityId::new("magnetar:app/run"),
        CapabilityVersion::new(1, 0, 0),
        CapabilityDescriptor::new("dependent")
            .with_contract(WitInterface::new("magnetar:app/run", "1.0.0"))
            .with_dependency(
                CapabilityId::new("magnetar:compute/run"),
                CapabilityVersion::new(1, 1, 0),
            ),
    );
    registry.register_capability(dependent).unwrap();
    assert!(matches!(
        registry.validate_dependencies(),
        Err(ProviderError::MissingCapabilityDependency { .. })
    ));
    registry
        .register_capability(capability(
            "magnetar:compute/run",
            CapabilityVersion::new(1, 2, 0),
        ))
        .unwrap();
    registry.validate_dependencies().unwrap();
}
#[test]
fn component_import_uses_a_semantic_capability_version() {
    let mut provider = TestProvider::new("compute");
    provider.metadata.capabilities.insert(capability(
        "magnetar:compute/run",
        CapabilityVersion::new(1, 1, 0),
    ));
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    assert_eq!(
        runtime
            .resolve_component_import(&WitInterface::new("magnetar:compute/run", "1.0.0"))
            .unwrap()
            .len(),
        1
    );
}
#[test]
fn compute_capability_has_the_canonical_wit_contract() {
    let compute = compute_capability();
    assert_eq!(compute.id.as_str(), COMPUTE_CAPABILITY_ID);
    assert_eq!(compute.version, COMPUTE_CAPABILITY_VERSION);
    assert_eq!(COMPUTE_WIT_PACKAGE, "magnetar:compute");
    assert_eq!(
        compute.descriptor.contracts,
        BTreeSet::from([WitInterface::new(COMPUTE_WIT_INTERFACE, "2.0.0")])
    );
}
#[test]
fn compute_v1_import_is_not_satisfied_by_compute_v2_provider() {
    let provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();

    assert!(matches!(
        runtime.resolve_component_import(&WitInterface::new(COMPUTE_WIT_INTERFACE, "1.1.0")),
        Err(ProviderError::NoCompatibleProvider(capability))
            if capability.id().as_str() == COMPUTE_CAPABILITY_ID
                && capability.version() == CapabilityVersion::new(1, 1, 0)
    ));
}
#[test]
fn compute_operation_catalog_defines_stable_family_metadata() {
    let families = ComputeOperationFamily::ALL
        .into_iter()
        .map(|family| family.id())
        .collect::<BTreeSet<_>>();

    assert_eq!(families.len(), 11);
    assert!(families.contains("descriptor-and-view"));
    assert!(families.contains("construction-and-allocation"));
    assert!(families.contains("data-movement-and-conversion"));
    assert!(families.contains("elementwise"));
    assert!(families.contains("comparison-and-selection"));
    assert!(families.contains("reduction"));
    assert!(families.contains("linear-algebra"));
    assert!(families.contains("convolution-and-spatial-transform"));
    assert!(families.contains("indexing-and-update"));
    assert!(families.contains("random-generation"));
    assert!(families.contains("synchronization-and-completion"));

    let metadata = ComputeOperationFamily::LinearAlgebra.metadata();
    assert_eq!(metadata.family, ComputeOperationFamily::LinearAlgebra);
    assert_eq!(metadata.scope, "matrix and batched matrix operations");
    assert!(metadata.examples.contains(&"matmul"));
    assert_eq!(
        ComputeOperationFamily::from_id("elementwise"),
        Some(ComputeOperationFamily::Elementwise)
    );
    assert_eq!(ComputeOperationFamily::from_id("autograd"), None);
}
#[test]
fn compute_wit_defines_the_stabilized_run_surface() {
    let wit = include_str!("../wit/compute.wit");
    assert!(wit.contains("package magnetar:compute@2.0.0;"));
    assert!(wit.contains("resource tensor"));
    assert!(wit.contains("resource graph"));
    assert!(wit.contains("resource operation"));
    assert!(wit.contains("enum operation-family"));
    assert!(wit.contains("type operation-id = string"));
    assert!(wit.contains("record operation-schema"));
    assert!(wit.contains("record operation-schema-support"));
    assert!(wit.contains("variant operation-attribute"));
    assert!(wit.contains("record operation-input-rule"));
    assert!(wit.contains("record operation-output-rule"));
    assert!(wit.contains("record operation-descriptor"));
    assert!(wit.contains("schema-id: option<operation-id>"));
    assert!(wit.contains("attributes: list<tuple<string, operation-attribute>>"));
    assert!(wit.contains("record graph-descriptor"));
    assert!(wit.contains("record graph-node"));
    assert!(wit.contains("variant graph-value-ref"));
    assert!(wit.contains("record shape-descriptor"));
    assert!(wit.contains("variant dtype-descriptor"));
    assert!(wit.contains("variant layout-descriptor"));
    assert!(wit.contains("record view-descriptor"));
    assert!(wit.contains("record tensor-resource-descriptor"));
    assert!(wit.contains("enum data-movement-kind"));
    assert!(wit.contains("record host-buffer-descriptor"));
    assert!(wit.contains("record data-movement-support"));
    assert!(wit.contains("record data-movement-descriptor"));
    assert!(wit.contains("enum placement-intent"));
    assert!(wit.contains("preserve-source-affinity"));
    assert!(wit.contains("runtime-selected"));
    assert!(wit.contains("host-accessible"));
    assert!(wit.contains("enum host-staging-policy"));
    assert!(wit.contains("forbid"));
    assert!(wit.contains("permit"));
    assert!(wit.contains("placement: placement-intent"));
    assert!(wit.contains("host-staging: host-staging-policy"));
    assert!(wit.contains("world compute-consumer"));
    assert!(wit.contains("import run"));
    assert!(!wit.contains("target-provider"));
    assert!(!wit.contains("target-device"));
    assert!(!wit.contains("target-affinity-group"));
    assert!(wit.contains("invalid-shape"));
    assert!(wit.contains("size-overflow"));
    assert!(wit.contains("unsupported-operation-family"));
    assert!(wit.contains("invalid-tensor-descriptor"));
    assert!(wit.contains("invalid-dtype"));
    assert!(wit.contains("invalid-operation-attribute"));
    assert!(wit.contains("invalid-operation-arity"));
    assert!(wit.contains("invalid-output-descriptor"));
    assert!(wit.contains("unsupported-dtype"));
    assert!(wit.contains("unsupported-layout"));
    assert!(wit.contains("unsupported-data-movement"));
    assert!(wit.contains("no-compatible-provider"));
    assert!(wit.contains("policy-rejected-provider"));
    assert!(wit.contains("provider-unavailable"));
    assert!(wit.contains("device-unavailable"));
    assert!(wit.contains("provider-pinned-resource"));
    assert!(wit.contains("device-bound-resource"));
    assert!(wit.contains("artifact-fingerprint-mismatch"));
    assert!(wit.contains("affinity-group-mismatch"));
    assert!(wit.contains("execution-interrupted"));
    assert!(wit.contains("execution-cancelled"));
    assert!(wit.contains("invalid-host-buffer"));
    assert!(wit.contains("invalid-transfer"));
    assert!(wit.contains("unsupported-conversion"));
    assert!(wit.contains("materialization-required"));
    assert!(wit.contains("enum compute-error-phase"));
    assert!(wit.contains("enum compute-error-severity"));
    assert!(wit.contains("record compute-diagnostic"));
    assert!(wit.contains("enum recovery-hint"));
    assert!(wit.contains("diagnostics: list<compute-diagnostic>"));
    assert!(wit.contains("recovery-hints: list<recovery-hint>"));
    assert!(wit.contains("submit: func("));
    assert!(wit.contains("result<operation, compute-error>"));
    assert!(!wit.contains("BackendStorage"));
    assert!(!wit.contains("Tensor`"));
    assert!(!wit.contains("autograd"));
    assert!(!wit.contains("training"));
    assert!(!wit.contains("kernel-name"));
    assert!(!wit.contains("queue"));
    assert!(!wit.contains("custom-operation"));
}
#[test]
fn compute_error_model_maps_validation_resolution_affinity_and_execution_failures() {
    let invalid_shape = ComputeError::from(ComputeValidationError::InvalidShape {
        reason: "rank exceeds limit".into(),
    });
    assert_eq!(invalid_shape.code, ComputeErrorCode::InvalidShape);
    assert_eq!(invalid_shape.phase, ComputeErrorPhase::Validation);
    assert_eq!(invalid_shape.severity, ComputeErrorSeverity::Terminal);
    assert!(
        invalid_shape
            .recovery_hints
            .contains(&RecoveryHint::NotRetryable)
    );

    let materialization = ComputeError::from(ComputeValidationError::MaterializationRequired {
        reason: "view must be materialized".into(),
    });
    assert_eq!(
        materialization.code,
        ComputeErrorCode::MaterializationRequired
    );
    assert!(
        materialization
            .recovery_hints
            .contains(&RecoveryHint::ExplicitMaterializationRequired)
    );

    let affinity = ComputeError::from(AffinityError::BoundProviderUnavailable(
        ProviderBinding::new("provider-a"),
    ));
    assert_eq!(affinity.code, ComputeErrorCode::ProviderUnavailable);
    assert_eq!(affinity.phase, ComputeErrorPhase::Interruption);
    assert!(
        affinity
            .recovery_hints
            .contains(&RecoveryHint::ProviderPinned)
    );

    let policy = ComputeError::from(ProviderError::PolicyRejectedProvider {
        capability: CapabilityBinding::new(
            CapabilityId::new(COMPUTE_CAPABILITY_ID),
            COMPUTE_CAPABILITY_VERSION,
        ),
        policy: BuiltInResolutionPolicy::Availability.id(),
    });
    assert_eq!(policy.code, ComputeErrorCode::PolicyRejectedProvider);
    assert_eq!(policy.phase, ComputeErrorPhase::Resolution);
    assert_eq!(
        policy.diagnostics[0]
            .capability
            .as_ref()
            .unwrap()
            .id()
            .as_str(),
        COMPUTE_CAPABILITY_ID
    );

    let execution = ComputeError::from(ProviderError::Lifecycle("device lost".into()));
    assert_eq!(execution.code, ComputeErrorCode::ExecutionInterrupted);
    assert_eq!(execution.phase, ComputeErrorPhase::Interruption);
    assert!(
        execution
            .recovery_hints
            .contains(&RecoveryHint::RetryBeforeState)
    );
}
#[test]
fn compute_diagnostics_are_optional_redacted_and_non_contractual() {
    let diagnostic = ComputeDiagnostic::new()
        .with_provider(ProviderBinding::new("provider-a"))
        .with_device(DeviceBinding::new(DeviceId::new("gpu:0")))
        .with_operation_family(ComputeOperationFamily::LinearAlgebra)
        .with_backend_message("native handle=0xdeadbeef at C:\\secret\\tensor.bin")
        .with_debug_trace_id("trace-42");

    assert_eq!(
        diagnostic.provider.as_ref().map(ProviderBinding::as_str),
        Some("provider-a")
    );
    assert_eq!(
        diagnostic.backend_message.as_deref(),
        Some("[redacted backend diagnostic]")
    );

    let error = ComputeError::new(
        ComputeErrorCode::ExecutionFailed,
        ComputeErrorPhase::Execution,
        ComputeErrorSeverity::Terminal,
        "provider execution failed",
    )
    .with_diagnostic(diagnostic)
    .with_recovery_hint(RecoveryHint::RestartableWithReplay);

    assert_eq!(error.code, ComputeErrorCode::ExecutionFailed);
    assert_eq!(error.diagnostics.len(), 1);
    assert!(
        error
            .recovery_hints
            .contains(&RecoveryHint::RestartableWithReplay)
    );
}
#[test]
fn compute_operation_validation_uses_provider_advertisements() {
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32, ComputeDType::Float64])
            .with_layouts([ComputeLayout::Dense])
            .with_precision_modes([ComputePrecision::Default, ComputePrecision::Reduced]),
    );
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::LinearAlgebra,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense])
            .with_precision_modes([ComputePrecision::Default]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();

    runtime
        .validate_compute_operations(
            "portable-compute",
            &[
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                    .with_dtype(ComputeDType::Float32)
                    .with_layout(ComputeLayout::Dense)
                    .with_precision(ComputePrecision::Reduced),
                ComputeOperationDescriptor::new(ComputeOperationFamily::LinearAlgebra)
                    .with_dtype(ComputeDType::Float32)
                    .with_layout(ComputeLayout::Dense),
            ],
        )
        .unwrap();

    assert!(matches!(
        runtime.validate_compute_operations(
            "portable-compute",
            &[ComputeOperationDescriptor::new(
                ComputeOperationFamily::Reduction
            )],
        ),
        Err(ComputeValidationError::UnsupportedOperationFamily { .. })
    ));
    assert!(matches!(
        runtime.validate_compute_operations(
            "portable-compute",
            &[
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                    .with_dtype(ComputeDType::UInt8)
            ],
        ),
        Err(ComputeValidationError::UnsupportedDType { .. })
    ));
    assert!(matches!(
        runtime.validate_compute_operations(
            "portable-compute",
            &[
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                    .with_layout(ComputeLayout::Strided)
            ],
        ),
        Err(ComputeValidationError::UnsupportedLayout { .. })
    ));
}
#[test]
fn compute_operation_schemas_define_initial_portable_operations() {
    let schemas = initial_compute_operation_schemas();

    for id in [
        "tensor.reshape",
        "tensor.transpose",
        "tensor.permute",
        "tensor.slice",
        "tensor.broadcast",
        "tensor.squeeze",
        "tensor.unsqueeze",
        "elementwise.unary.relu",
        "elementwise.binary.add",
        "comparison.eq",
        "selection.where",
        "reduction.sum",
        "linalg.matmul",
        "linalg.batched-matmul",
        "tensor.gather",
        "tensor.index-select",
        "tensor.scatter",
        "tensor.scatter-add",
        "tensor.concat",
        "random.uniform",
        "random.normal",
    ] {
        assert!(schemas.contains_key(&ComputeOperationId::new(id)), "{id}");
    }
    assert!(!schemas.contains_key(&ComputeOperationId::new("convolution.conv2d")));
    assert!(!schemas.contains_key(&ComputeOperationId::new("pooling.max")));
    assert!(!schemas.contains_key(&ComputeOperationId::new("attention.flash")));
    assert!(!schemas.contains_key(&ComputeOperationId::new("quantized.matmul")));
    assert!(!schemas.contains_key(&ComputeOperationId::new("custom.kernel")));
    assert!(!schemas.contains_key(&ComputeOperationId::new("autograd.backward")));

    let scatter = schemas
        .get(&ComputeOperationId::new("tensor.scatter"))
        .unwrap();
    assert!(scatter.provider_specific_semantics);
}
#[test]
fn provider_compute_advertisement_drives_operation_validation() {
    let schemas = initial_compute_operation_schemas();
    let add = schemas
        .get(&ComputeOperationId::new("elementwise.binary.add"))
        .unwrap();
    let mut provider = provider_with_capabilities("advertised-compute", [compute_capability()]);
    provider.metadata.compute_advertisement = ProviderComputeAdvertisement::new()
        .with_capability(
            ComputeCapabilitySupport::default()
                .with_versions([COMPUTE_CAPABILITY_VERSION])
                .with_operation_catalog_revision("initial")
                .with_operation_schema_revision("initial"),
        )
        .with_operation_schema(OperationSchemaSupport::from_operation_support(
            add.id.clone(),
            add.family,
            ComputeOperationSupport::new()
                .with_dtypes([ComputeDType::Float32])
                .with_layouts([ComputeLayout::Dense])
                .with_precision_modes([ComputePrecision::Default]),
        ));
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let tensor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );

    runtime
        .validate_compute_operations(
            "advertised-compute",
            &[ComputeOperationDescriptor::from_schema(add)
                .with_tensor(tensor.clone())
                .with_tensor(tensor.clone())
                .with_tensor(tensor)],
        )
        .unwrap();
}
#[test]
fn provider_compute_advertisement_reports_version_and_schema_rejections() {
    let schemas = initial_compute_operation_schemas();
    let add = schemas
        .get(&ComputeOperationId::new("elementwise.binary.add"))
        .unwrap();
    let mut incompatible = TestProvider::new("incompatible-compute");
    incompatible.metadata.compute_advertisement = ProviderComputeAdvertisement::new()
        .with_capability(
            ComputeCapabilitySupport::default().with_versions([CapabilityVersion::new(0, 9, 0)]),
        );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(incompatible))
        .build()
        .unwrap();

    assert!(matches!(
        runtime.validate_compute_operations(
            "incompatible-compute",
            &[ComputeOperationDescriptor::from_schema(add)]
        ),
        Err(ComputeValidationError::UnsupportedAdvertisement { .. })
    ));

    let mut unsupported = provider_with_capabilities("unsupported-schema", [compute_capability()]);
    unsupported.metadata.compute_advertisement = ProviderComputeAdvertisement::new()
        .with_capability(
            ComputeCapabilitySupport::default().with_versions([COMPUTE_CAPABILITY_VERSION]),
        )
        .with_operation_family(OperationFamilySupport::from_operation_support(
            ComputeOperationFamily::Elementwise,
            ComputeOperationSupport::new().with_dtypes([ComputeDType::Float32]),
        ))
        .with_unsupported_operation_schema(add.id.clone());
    let runtime = Runtime::builder()
        .register_provider(Arc::new(unsupported))
        .build()
        .unwrap();

    assert!(matches!(
        runtime.validate_compute_operations(
            "unsupported-schema",
            &[ComputeOperationDescriptor::from_schema(add)]
        ),
        Err(ComputeValidationError::UnsupportedOperationSchema { .. })
    ));
}
#[test]
fn compute_operation_schema_validation_checks_attributes_and_shapes() {
    let schemas = initial_compute_operation_schemas();
    let add = schemas
        .get(&ComputeOperationId::new("elementwise.binary.add"))
        .unwrap();
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_operation_schema_support.insert(
        add.id.clone(),
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let lhs = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 1]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let rhs = TensorDescriptor::materialized(
        ShapeDescriptor::new([1, 3]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let output = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 3]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );

    runtime
        .validate_compute_operations(
            "portable-compute",
            &[ComputeOperationDescriptor::from_schema(add)
                .with_tensor(lhs.clone())
                .with_tensor(rhs.clone())
                .with_tensor(output.clone())],
        )
        .unwrap();

    let bad_output = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    assert!(matches!(
        runtime.validate_compute_operations(
            "portable-compute",
            &[ComputeOperationDescriptor::from_schema(add)
                .with_tensor(lhs.clone())
                .with_tensor(rhs.clone())
                .with_tensor(bad_output)]
        ),
        Err(ComputeValidationError::InvalidOutputDescriptor { .. })
    ));

    assert!(matches!(
        runtime.validate_compute_operations(
            "portable-compute",
            &[ComputeOperationDescriptor::from_schema(add)
                .with_attribute("unknown", ComputeOperationAttribute::Boolean(true))
                .with_tensor(lhs)
                .with_tensor(rhs)
                .with_tensor(output)]
        ),
        Err(ComputeValidationError::InvalidOperationAttribute { .. })
    ));
}
#[test]
fn compute_operation_schema_validation_checks_reduction_matmul_and_random_rules() {
    let schemas = initial_compute_operation_schemas();
    let sum = schemas
        .get(&ComputeOperationId::new("reduction.sum"))
        .unwrap();
    let matmul = schemas
        .get(&ComputeOperationId::new("linalg.matmul"))
        .unwrap();
    let random = schemas
        .get(&ComputeOperationId::new("random.uniform"))
        .unwrap();
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    for schema in [sum, matmul, random] {
        provider.metadata.compute_operation_schema_support.insert(
            schema.id.clone(),
            ComputeOperationSupport::new()
                .with_dtypes([ComputeDType::Float32, ComputeDType::SInt64])
                .with_layouts([ComputeLayout::Dense]),
        );
    }
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let f32_2x3 = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 3]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );

    runtime
        .validate_compute_operations(
            "portable-compute",
            &[ComputeOperationDescriptor::from_schema(sum)
                .with_attribute("axes", ComputeOperationAttribute::Axes(vec![1]))
                .with_attribute("keep-dimensions", ComputeOperationAttribute::Boolean(true))
                .with_tensor(f32_2x3.clone())
                .with_tensor(TensorDescriptor::materialized(
                    ShapeDescriptor::new([2, 1]),
                    DTypeDescriptor::portable(ComputeDType::Float32),
                ))],
        )
        .unwrap();
    assert!(matches!(
        runtime.validate_compute_operations(
            "portable-compute",
            &[ComputeOperationDescriptor::from_schema(sum)
                .with_attribute("axes", ComputeOperationAttribute::Axes(vec![2]))
                .with_tensor(f32_2x3.clone())
                .with_tensor(f32_2x3.clone())]
        ),
        Err(ComputeValidationError::InvalidShape { .. })
    ));

    runtime
        .validate_compute_operations(
            "portable-compute",
            &[ComputeOperationDescriptor::from_schema(matmul)
                .with_tensor(f32_2x3.clone())
                .with_tensor(TensorDescriptor::materialized(
                    ShapeDescriptor::new([3, 4]),
                    DTypeDescriptor::portable(ComputeDType::Float32),
                ))
                .with_tensor(TensorDescriptor::materialized(
                    ShapeDescriptor::new([2, 4]),
                    DTypeDescriptor::portable(ComputeDType::Float32),
                ))],
        )
        .unwrap();

    runtime
        .validate_compute_operations(
            "portable-compute",
            &[ComputeOperationDescriptor::from_schema(random)
                .with_attribute(
                    "shape",
                    ComputeOperationAttribute::Shape(ShapeDescriptor::new([2, 2])),
                )
                .with_attribute(
                    "dtype",
                    ComputeOperationAttribute::DType(ComputeDType::Float32),
                )
                .with_attribute("seed", ComputeOperationAttribute::Integer(42))
                .with_tensor(TensorDescriptor::materialized(
                    ShapeDescriptor::new([2, 2]),
                    DTypeDescriptor::portable(ComputeDType::Float32),
                ))],
        )
        .unwrap();
}
#[test]
fn tensor_descriptors_validate_shape_dtype_layout_and_provider_support() {
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::DescriptorAndView,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense, ComputeLayout::Strided])
            .with_descriptor_limits(TensorDescriptorLimits {
                max_rank: 4,
                max_dimension: 1024,
                max_elements: 4096,
                max_bytes: 16_384,
                allow_zero_sized: false,
            }),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let tensor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 3, 4]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );

    runtime
        .validate_compute_operations(
            "portable-compute",
            &[
                ComputeOperationDescriptor::new(ComputeOperationFamily::DescriptorAndView)
                    .with_tensor(tensor.clone()),
            ],
        )
        .unwrap();

    let zero_dim = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 0]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    assert!(matches!(
        runtime.validate_compute_operations(
            "portable-compute",
            &[
                ComputeOperationDescriptor::new(ComputeOperationFamily::DescriptorAndView)
                    .with_tensor(zero_dim)
            ]
        ),
        Err(ComputeValidationError::InvalidShape { .. })
    ));

    let unsupported_dtype = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 3]),
        DTypeDescriptor::portable(ComputeDType::Float64),
    );
    assert!(matches!(
        runtime.validate_compute_operations(
            "portable-compute",
            &[
                ComputeOperationDescriptor::new(ComputeOperationFamily::DescriptorAndView)
                    .with_tensor(unsupported_dtype)
            ]
        ),
        Err(ComputeValidationError::UnsupportedDType { .. })
    ));

    let unsupported_layout = TensorDescriptor::new(
        ShapeDescriptor::new([2, 3]),
        DTypeDescriptor::portable(ComputeDType::Float32),
        LayoutDescriptor::ProviderOpaque {
            layout_id: "native-blocked".into(),
        },
    );
    assert!(matches!(
        runtime.validate_compute_operations(
            "portable-compute",
            &[
                ComputeOperationDescriptor::new(ComputeOperationFamily::DescriptorAndView)
                    .with_tensor(unsupported_layout)
            ]
        ),
        Err(ComputeValidationError::UnsupportedLayout { .. })
    ));

    let overflowing = TensorDescriptor::materialized(
        ShapeDescriptor::new([64, 65]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    assert!(matches!(
        runtime.validate_compute_operations(
            "portable-compute",
            &[
                ComputeOperationDescriptor::new(ComputeOperationFamily::DescriptorAndView)
                    .with_tensor(overflowing)
            ]
        ),
        Err(ComputeValidationError::SizeOverflow { .. })
    ));
}
#[test]
fn tensor_views_and_resources_preserve_affinity_and_materialization_boundaries() {
    let provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let source = TensorResourceId::new("tensor-1");
    let descriptor = TensorDescriptor::new(
        ShapeDescriptor::new([4, 4]),
        DTypeDescriptor::portable(ComputeDType::Float32),
        LayoutDescriptor::Strided {
            strides_elements: vec![4, 1],
            offset_elements: 0,
        },
    )
    .with_view(ViewDescriptor::from_resource(source.clone(), 4, [4, 1]));
    let resource = TensorResourceDescriptor::new(
        source,
        descriptor,
        ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(ProviderBinding::new("portable-compute"))
            .with_capability(CapabilityBinding::new(
                CapabilityId::new(COMPUTE_CAPABILITY_ID),
                COMPUTE_CAPABILITY_VERSION,
            )),
    );

    runtime
        .validate_compute_tensor_resources("portable-compute", &[resource])
        .unwrap();

    let foreign = TensorResourceDescriptor::new(
        TensorResourceId::new("tensor-2"),
        TensorDescriptor::materialized(
            ShapeDescriptor::new([4, 4]),
            DTypeDescriptor::portable(ComputeDType::Float32),
        ),
        ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(ProviderBinding::new("other-provider")),
    );
    assert!(matches!(
        runtime.validate_compute_tensor_resources("portable-compute", &[foreign]),
        Err(ComputeValidationError::IncompatibleResourceAffinity(_))
    ));
}
#[test]
fn compute_data_movement_validates_host_buffers_affinity_and_provider_support() {
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_data_movement_support.insert(
        ComputeDataMovementKind::Upload,
        ComputeDataMovementSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense])
            .with_host_encodings([HostBufferEncoding::LittleEndian]),
    );
    provider.metadata.compute_data_movement_support.insert(
        ComputeDataMovementKind::Transfer,
        ComputeDataMovementSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    provider.metadata.compute_data_movement_support.insert(
        ComputeDataMovementKind::Materialize,
        ComputeDataMovementSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense, ComputeLayout::Strided]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );

    let upload = ComputeDataMovementDescriptor::upload(
        HostBufferDescriptor::new(16, HostBufferEncoding::LittleEndian),
        descriptor.clone(),
    );
    runtime
        .validate_compute_data_movement("portable-compute", std::slice::from_ref(&upload))
        .unwrap();
    let uploaded = runtime
        .wrap_compute_data_movement_output(
            "portable-compute",
            &upload,
            TensorResourceId::new("uploaded"),
        )
        .unwrap();
    assert_eq!(
        uploaded.affinity.provider().map(ProviderBinding::as_str),
        Some("portable-compute")
    );

    let invalid_upload = ComputeDataMovementDescriptor::upload(
        HostBufferDescriptor::new(8, HostBufferEncoding::LittleEndian),
        descriptor.clone(),
    );
    assert!(matches!(
        runtime.validate_compute_data_movement("portable-compute", &[invalid_upload]),
        Err(ComputeValidationError::InvalidHostBuffer { .. })
    ));

    let foreign = TensorResourceDescriptor::new(
        TensorResourceId::new("foreign"),
        descriptor.clone(),
        ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(ProviderBinding::new("other-provider")),
    );
    let transfer = ComputeDataMovementDescriptor::transfer(foreign, descriptor.clone());
    runtime
        .validate_compute_data_movement("portable-compute", &[transfer])
        .unwrap();

    let materialized = TensorResourceDescriptor::new(
        TensorResourceId::new("materialized"),
        descriptor.clone(),
        ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(ProviderBinding::new("portable-compute")),
    );
    let invalid_materialize = ComputeDataMovementDescriptor::materialize(materialized, descriptor);
    assert!(matches!(
        runtime.validate_compute_data_movement("portable-compute", &[invalid_materialize]),
        Err(ComputeValidationError::MaterializationRequired { .. })
    ));
}
#[test]
fn memory_planning_accounts_for_explicit_host_staged_transfers() {
    let mut provider = provider_with_capabilities("movement-compute", [compute_capability()]);
    provider.metadata.compute_data_movement_support.insert(
        ComputeDataMovementKind::Transfer,
        ComputeDataMovementSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense])
            .with_host_staging(),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let source = TensorResourceDescriptor::new(
        TensorResourceId::new("source"),
        descriptor.clone(),
        ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(ProviderBinding::new("other-provider")),
    );
    let movement =
        ComputeDataMovementDescriptor::transfer(source, descriptor).permit_host_staging();

    let plan = runtime
        .plan_compute_data_movement_memory("movement-compute", &[movement])
        .unwrap();

    assert_eq!(plan.pressure.transfer_buffer_cost_bytes, 16);
    assert!(
        plan.decisions.iter().any(|decision| {
            matches!(decision, MemoryPlanningDecision::AccountHostStaging { .. })
        })
    );
}
#[test]
fn provider_compute_advertisement_drives_data_movement_validation() {
    let mut provider = provider_with_capabilities("advertised-movement", [compute_capability()]);
    provider.metadata.compute_advertisement = ProviderComputeAdvertisement::new()
        .with_capability(
            ComputeCapabilitySupport::default().with_versions([COMPUTE_CAPABILITY_VERSION]),
        )
        .with_data_movement(DataMovementSupport::from_compute_support(
            ComputeDataMovementKind::Upload,
            ComputeDataMovementSupport::new()
                .with_dtypes([ComputeDType::Float32])
                .with_layouts([ComputeLayout::Dense])
                .with_host_encodings([HostBufferEncoding::LittleEndian]),
        ));
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );

    runtime
        .validate_compute_data_movement(
            "advertised-movement",
            &[ComputeDataMovementDescriptor::upload(
                HostBufferDescriptor::new(16, HostBufferEncoding::LittleEndian),
                descriptor,
            )],
        )
        .unwrap();
}
#[test]
fn compute_operation_requests_reject_unknown_family_ids() {
    let provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();

    assert!(matches!(
        runtime.validate_compute_operation_requests(
            "portable-compute",
            &[ComputeOperationRequest::new("backend-kernel-name")]
        ),
        Err(ComputeValidationError::UnknownOperationFamily(_))
    ));
}
#[test]
fn compute_graph_validation_checks_references_provider_support_and_submission_affinity() {
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let graph = ComputeGraph::new(ComputeGraphId::new("graph-1"))
        .with_input(ComputeInput::new(
            ComputeInputId::new("x"),
            ComputeInputValue::TensorDescriptor(descriptor.clone()),
        ))
        .with_node(
            ComputeNode::new(
                ComputeNodeId::new("add"),
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                    .with_dtype(ComputeDType::Float32)
                    .with_layout(ComputeLayout::Dense),
            )
            .with_input(ComputeValueRef::Input(ComputeInputId::new("x")))
            .with_output(ComputeNodeOutput::new(
                ComputeOutputId::new("y"),
                descriptor,
            )),
        )
        .with_output(ComputeOutput::new(
            ComputeOutputId::new("result"),
            ComputeValueRef::NodeOutput {
                node: ComputeNodeId::new("add"),
                output: ComputeOutputId::new("y"),
            },
        ));

    let report = runtime
        .validate_compute_graph("portable-compute", &graph)
        .unwrap();
    assert_eq!(report.node_count, 1);
    assert_eq!(report.input_count, 1);
    assert_eq!(report.output_count, 1);

    let submission = runtime
        .submit_validated_compute_graph("portable-compute", &graph)
        .unwrap();
    assert_eq!(submission.graph, ComputeGraphId::new("graph-1"));
    assert_eq!(submission.state(), ComputeSubmissionState::Pending);
    assert_eq!(
        submission.affinity.provider().map(ProviderBinding::as_str),
        Some("portable-compute")
    );
}
#[test]
fn compute_execution_planning_selects_provider_device_and_validates_plan() {
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let mut device = DeviceMetadata::new(
        DeviceId::new("gpu:0"),
        "GPU 0",
        DeviceType::Gpu,
        "portable-compute",
    );
    device.memory_capacity = 1_048_576;
    provider
        .devices
        .push(Arc::new(DeviceDescriptor::new(device)));
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let graph = ComputeGraph::new(ComputeGraphId::new("planned-graph"))
        .with_input(ComputeInput::new(
            ComputeInputId::new("x"),
            ComputeInputValue::TensorDescriptor(descriptor.clone()),
        ))
        .with_node(
            ComputeNode::new(
                ComputeNodeId::new("add"),
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                    .with_dtype(ComputeDType::Float32)
                    .with_layout(ComputeLayout::Dense),
            )
            .with_input(ComputeValueRef::Input(ComputeInputId::new("x")))
            .with_output(ComputeNodeOutput::new(
                ComputeOutputId::new("y"),
                descriptor,
            )),
        )
        .with_output(ComputeOutput::new(
            ComputeOutputId::new("result"),
            ComputeValueRef::NodeOutput {
                node: ComputeNodeId::new("add"),
                output: ComputeOutputId::new("y"),
            },
        ));

    let plan = runtime.plan_compute_execution(&graph).unwrap();

    assert!(plan.is_validated());
    assert_eq!(plan.provider.as_str(), "portable-compute");
    assert_eq!(
        plan.device.as_ref().map(|device| device.id().as_str()),
        Some("gpu:0")
    );
    assert_eq!(plan.policy, BuiltInResolutionPolicy::Deterministic.id());
    assert_eq!(
        plan.classification,
        ComputeExecutionClassification::Transparent
    );
    assert!(
        plan.steps
            .iter()
            .any(|step| step.kind == ExecutionStepKind::SubmitToProvider)
    );
    assert!(
        plan.constraints
            .iter()
            .any(|constraint| matches!(constraint, ExecutionConstraint::NoHiddenCpuStaging))
    );
    assert_eq!(
        plan.memory_plan.graph,
        Some(ComputeGraphId::new("planned-graph"))
    );
}
#[test]
fn compute_execution_planning_preserves_provider_pinned_resources() {
    let mut provider_a = provider_with_capabilities("provider-a", [compute_capability()]);
    provider_a.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let mut provider_b = provider_with_capabilities("provider-b", [compute_capability()]);
    provider_b.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider_b))
        .register_provider(Arc::new(provider_a))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let resource = TensorResourceDescriptor::new(
        TensorResourceId::new("pinned"),
        descriptor.clone(),
        ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(ProviderBinding::new("provider-b"))
            .with_capability(CapabilityBinding::new(
                CapabilityId::new(COMPUTE_CAPABILITY_ID),
                COMPUTE_CAPABILITY_VERSION,
            )),
    );
    let graph = ComputeGraph::new(ComputeGraphId::new("pinned-graph"))
        .with_input(ComputeInput::new(
            ComputeInputId::new("x"),
            ComputeInputValue::TensorResource(resource),
        ))
        .with_node(
            ComputeNode::new(
                ComputeNodeId::new("add"),
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                    .with_dtype(ComputeDType::Float32)
                    .with_layout(ComputeLayout::Dense),
            )
            .with_input(ComputeValueRef::Input(ComputeInputId::new("x")))
            .with_output(ComputeNodeOutput::new(
                ComputeOutputId::new("y"),
                descriptor,
            )),
        );

    let plan = runtime.plan_compute_execution(&graph).unwrap();

    assert_eq!(plan.provider.as_str(), "provider-b");
    assert_eq!(
        plan.classification,
        ComputeExecutionClassification::ProviderPinned
    );
    assert!(
        plan.steps
            .iter()
            .any(|step| step.kind == ExecutionStepKind::PreserveProviderPinnedAffinity)
    );
}
#[test]
fn scheduler_accepts_validated_plans_and_runs_fifo() {
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let first_graph = ComputeGraph::new(ComputeGraphId::new("first")).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                .with_dtype(ComputeDType::Float32)
                .with_layout(ComputeLayout::Dense),
        )
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor.clone(),
        )),
    );
    let second_graph = ComputeGraph::new(ComputeGraphId::new("second")).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                .with_dtype(ComputeDType::Float32)
                .with_layout(ComputeLayout::Dense),
        )
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor,
        )),
    );
    let first_plan = runtime.plan_compute_execution(&first_graph).unwrap();
    let second_plan = runtime.plan_compute_execution(&second_graph).unwrap();
    let mut scheduler = runtime.scheduler(2);

    let first = runtime
        .schedule_compute_execution(&mut scheduler, first_plan)
        .unwrap();
    let second = runtime
        .schedule_compute_execution(&mut scheduler, second_plan)
        .unwrap();

    assert_eq!(scheduler.policy(), SchedulingPolicy::Fifo);
    assert_eq!(scheduler.submit_next(&runtime).unwrap(), Some(first));
    assert_eq!(
        scheduler.operation(first).unwrap().state(),
        SchedulingState::Running
    );
    assert_eq!(scheduler.submit_next(&runtime).unwrap(), Some(second));
}

#[test]
fn runtime_observability_correlates_plan_scheduler_and_metrics() {
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let graph = ComputeGraph::new(ComputeGraphId::new("observable")).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                .with_dtype(ComputeDType::Float32)
                .with_layout(ComputeLayout::Dense),
        )
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor,
        )),
    );
    let plan = runtime.plan_compute_execution(&graph).unwrap();
    let trace = plan.trace_id().clone();

    let plan_events = plan.observations();
    assert!(plan_events.iter().all(|event| event.trace_id == trace));
    assert!(plan_events.iter().any(|event| {
        event.kind == RuntimeEventKind::CapabilityResolution
            && event.provider.as_ref().map(ProviderBinding::as_str) == Some("portable-compute")
    }));
    assert!(plan_events.iter().any(|event| {
        event.kind == RuntimeEventKind::ExecutionPlanning
            && event.execution_plan.as_ref() == Some(&plan.id)
    }));

    let metrics = runtime_metrics_for_execution_plan(&plan);
    assert!(
        metrics
            .iter()
            .any(|metric| metric.kind == RuntimeMetricKind::MemoryUsageEstimate)
    );
    assert!(metrics.iter().all(|metric| {
        metric.trace_id.as_ref() == Some(&trace)
            && metric.provider.as_ref().map(ProviderBinding::as_str) == Some("portable-compute")
    }));

    let mut scheduler = runtime.scheduler(1);
    let operation = runtime
        .schedule_compute_execution(&mut scheduler, plan)
        .unwrap();
    scheduler.submit_next(&runtime).unwrap();
    scheduler.complete(operation).unwrap();

    assert!(
        scheduler
            .observations()
            .iter()
            .all(|event| event.trace_id == trace)
    );
    assert!(
        scheduler
            .observations()
            .iter()
            .any(|event| event.kind == RuntimeEventKind::ExecutionStarted)
    );
    assert!(
        scheduler
            .observations()
            .iter()
            .any(|event| event.kind == RuntimeEventKind::ExecutionCompleted)
    );
}

#[test]
fn runtime_diagnostics_redact_native_details_and_exporters_are_components() {
    let diagnostic = RuntimeDiagnostic::new(
        RuntimeDiagnosticCode::ExecutionFailed,
        "backend handle=0xdeadbeef at C:\\native\\queue",
    )
    .with_trace(TraceId::new("trace:failure"))
    .with_provider(ProviderBinding::new("provider-a"));
    assert_eq!(diagnostic.message, "[redacted backend diagnostic]");

    let component = ComponentMetadata::new("otel-exporter", "1", "exports observations");
    let mut exporter =
        ObservabilityExporterDescriptor::new(component, ObservabilitySink::OpenTelemetry);
    exporter
        .accepted_events
        .insert(RuntimeEventKind::ExecutionCompleted);

    assert_eq!(
        exporter.input_contract,
        WitInterface::new("magnetar:runtime/observability", "1.0.0")
    );
    assert!(
        exporter
            .accepted_events
            .contains(&RuntimeEventKind::ExecutionCompleted)
    );
}

#[test]
fn provider_execution_api_submits_validated_scheduled_work() {
    let api = Arc::new(TestProviderExecutionApi::new());
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.execution_api = Some(api.clone());
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let graph = ComputeGraph::new(ComputeGraphId::new("provider-execution")).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                .with_dtype(ComputeDType::Float32)
                .with_layout(ComputeLayout::Dense),
        )
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor.clone(),
        )),
    );
    let plan = runtime.plan_compute_execution(&graph).unwrap();
    let mut scheduler = runtime.scheduler(1);
    let operation_id = runtime
        .schedule_compute_execution(&mut scheduler, plan)
        .unwrap();
    let operation = scheduler.operation(operation_id).unwrap();

    let request = runtime.prepare_provider_execution(operation).unwrap();
    let handle = runtime.submit_provider_execution(request).unwrap();
    let status = runtime.observe_provider_execution(&handle).unwrap();

    assert_eq!(handle.operation, operation_id);
    assert_eq!(handle.provider.as_str(), "portable-compute");
    assert!(handle.id.as_str().contains("provider-execution:"));
    assert_eq!(status.state, SchedulingState::Running);
    assert_eq!(
        status
            .progress
            .as_ref()
            .map(|progress| progress.total_steps),
        Some(3)
    );
    let submitted = api.submitted.lock().unwrap();
    assert_eq!(submitted.len(), 1);
    assert!(submitted[0].plan.is_validated());
    assert!(
        submitted[0].constraints.iter().any(|constraint| matches!(
            constraint,
            ExecutionConstraint::NoImplicitProviderMigration
        ))
    );
}
#[test]
fn provider_execution_completion_preserves_output_affinity_and_releases() {
    let api = Arc::new(TestProviderExecutionApi::new());
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.execution_api = Some(api.clone());
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let graph = ComputeGraph::new(ComputeGraphId::new("provider-complete")).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                .with_dtype(ComputeDType::Float32)
                .with_layout(ComputeLayout::Dense),
        )
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor.clone(),
        )),
    );
    let plan = runtime.plan_compute_execution(&graph).unwrap();
    let mut scheduler = runtime.scheduler(1);
    let operation_id = runtime
        .schedule_compute_execution(&mut scheduler, plan)
        .unwrap();
    let request = runtime
        .prepare_provider_execution(scheduler.operation(operation_id).unwrap())
        .unwrap();
    let handle = runtime.submit_provider_execution(request).unwrap();
    api.outputs
        .lock()
        .unwrap()
        .push(TensorResourceDescriptor::new(
            TensorResourceId::new("out-resource"),
            descriptor,
            ResourceAffinity::new(FallbackClass::ProviderPinned)
                .with_provider(ProviderBinding::new("portable-compute"))
                .with_capability(CapabilityBinding::new(
                    CapabilityId::new(COMPUTE_CAPABILITY_ID),
                    COMPUTE_CAPABILITY_VERSION,
                )),
        ));

    let result = runtime.complete_provider_execution(&handle).unwrap();
    runtime.release_provider_execution(handle).unwrap();

    assert_eq!(result.state, SchedulingState::Completed);
    assert_eq!(result.outputs.len(), 1);
    assert_eq!(
        result.outputs[0]
            .affinity
            .provider()
            .map(ProviderBinding::as_str),
        Some("portable-compute")
    );
    assert!(api.released.load(Ordering::SeqCst));
}
#[test]
fn provider_execution_rejects_mismatched_provider_request_and_maps_cancellation() {
    let api = Arc::new(TestProviderExecutionApi::new());
    *api.cancel_outcome.lock().unwrap() = ProviderCancellationOutcome::Accepted;
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.execution_api = Some(api);
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let graph = ComputeGraph::new(ComputeGraphId::new("provider-cancel")).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                .with_dtype(ComputeDType::Float32)
                .with_layout(ComputeLayout::Dense),
        )
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor,
        )),
    );
    let plan = runtime.plan_compute_execution(&graph).unwrap();
    let mut scheduler = runtime.scheduler(1);
    let operation_id = runtime
        .schedule_compute_execution(&mut scheduler, plan)
        .unwrap();
    let mut request = runtime
        .prepare_provider_execution(scheduler.operation(operation_id).unwrap())
        .unwrap();
    request.provider = ProviderBinding::new("other-provider");

    assert!(matches!(
        runtime.submit_provider_execution(request),
        Err(ProviderExecutionError {
            code: ProviderExecutionErrorCode::InvalidExecutionPlan,
            phase: ProviderExecutionPhase::Submit,
            ..
        })
    ));

    let request = runtime
        .prepare_provider_execution(scheduler.operation(operation_id).unwrap())
        .unwrap();
    let handle = runtime.submit_provider_execution(request).unwrap();
    assert_eq!(
        runtime.cancel_provider_execution(&handle).unwrap(),
        ProviderCancellationOutcome::Accepted
    );
}
#[test]
fn provider_execution_diagnostics_redact_native_details() {
    let diagnostic = ProviderExecutionDiagnostic::new(
        ProviderBinding::new("provider"),
        ProviderExecutionPhase::Submit,
    )
    .with_detail("backend handle=0xdeadbeef");

    assert_eq!(
        diagnostic.detail.as_deref(),
        Some("[redacted backend diagnostic]")
    );
}
#[test]
fn scheduler_rejects_over_capacity_and_cancels_queued_work() {
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let graph = ComputeGraph::new(ComputeGraphId::new("queued")).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                .with_dtype(ComputeDType::Float32)
                .with_layout(ComputeLayout::Dense),
        )
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor,
        )),
    );
    let plan = runtime.plan_compute_execution(&graph).unwrap();
    let mut scheduler = runtime.scheduler(1);
    let operation = scheduler.schedule(&runtime, plan.clone()).unwrap();

    assert!(matches!(
        scheduler.schedule(&runtime, plan),
        Err(SchedulerError::QueueCapacityExceeded { capacity: 1 })
    ));

    scheduler.cancel(operation).unwrap();
    assert_eq!(
        scheduler.operation(operation).unwrap().state(),
        SchedulingState::Cancelled
    );
    assert_eq!(
        scheduler.result(operation).unwrap().state,
        SchedulingState::Cancelled
    );
    assert_eq!(scheduler.submit_next(&runtime).unwrap(), None);
}
#[test]
fn scheduler_completion_exposes_terminal_result_without_native_handles() {
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let graph = ComputeGraph::new(ComputeGraphId::new("completed"))
        .with_node(
            ComputeNode::new(
                ComputeNodeId::new("node"),
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                    .with_dtype(ComputeDType::Float32)
                    .with_layout(ComputeLayout::Dense),
            )
            .with_output(ComputeNodeOutput::new(
                ComputeOutputId::new("out"),
                descriptor,
            )),
        )
        .with_output(ComputeOutput::new(
            ComputeOutputId::new("result"),
            ComputeValueRef::NodeOutput {
                node: ComputeNodeId::new("node"),
                output: ComputeOutputId::new("out"),
            },
        ));
    let plan = runtime.plan_compute_execution(&graph).unwrap();
    let mut scheduler = runtime.scheduler(1);
    let operation = scheduler.schedule(&runtime, plan).unwrap();

    scheduler.submit_next(&runtime).unwrap();
    scheduler.complete(operation).unwrap();

    let result = scheduler.result(operation).unwrap();
    assert_eq!(result.state, SchedulingState::Completed);
    assert_eq!(result.outputs.len(), 1);
    assert!(result.error.is_none());
}
#[test]
fn scheduler_interrupts_when_provider_is_unavailable_before_submission() {
    let mut healthy = provider_with_capabilities("portable-compute", [compute_capability()]);
    healthy.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let planning_runtime = Runtime::builder()
        .register_provider(Arc::new(healthy))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let graph = ComputeGraph::new(ComputeGraphId::new("interrupted")).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                .with_dtype(ComputeDType::Float32)
                .with_layout(ComputeLayout::Dense),
        )
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor,
        )),
    );
    let plan = planning_runtime.plan_compute_execution(&graph).unwrap();
    let mut unavailable = provider_with_capabilities("portable-compute", [compute_capability()]);
    unavailable.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    unavailable.health = ProviderHealth::Unavailable;
    let submission_runtime = Runtime::builder()
        .register_provider(Arc::new(unavailable))
        .build()
        .unwrap();
    let mut scheduler = submission_runtime.scheduler(1);
    let operation = scheduler.schedule(&submission_runtime, plan).unwrap();

    assert!(matches!(
        scheduler.submit_next(&submission_runtime),
        Err(SchedulerError::ProviderUnavailable(provider))
            if provider.as_str() == "portable-compute"
    ));
    assert_eq!(
        scheduler.operation(operation).unwrap().state(),
        SchedulingState::Interrupted
    );
}
#[test]
fn compute_graph_validation_rejects_missing_and_future_references() {
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let missing_input = ComputeGraph::new(ComputeGraphId::new("bad-input")).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise),
        )
        .with_input(ComputeValueRef::Input(ComputeInputId::new("missing")))
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor.clone(),
        )),
    );
    assert!(matches!(
        runtime.validate_compute_graph("portable-compute", &missing_input),
        Err(ComputeValidationError::MissingInput { .. })
    ));

    let future_reference = ComputeGraph::new(ComputeGraphId::new("cycle"))
        .with_node(
            ComputeNode::new(
                ComputeNodeId::new("first"),
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise),
            )
            .with_input(ComputeValueRef::NodeOutput {
                node: ComputeNodeId::new("second"),
                output: ComputeOutputId::new("out"),
            })
            .with_output(ComputeNodeOutput::new(
                ComputeOutputId::new("out"),
                descriptor.clone(),
            )),
        )
        .with_node(
            ComputeNode::new(
                ComputeNodeId::new("second"),
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise),
            )
            .with_output(ComputeNodeOutput::new(
                ComputeOutputId::new("out"),
                descriptor,
            )),
        );
    assert!(matches!(
        runtime.validate_compute_graph("portable-compute", &future_reference),
        Err(ComputeValidationError::CyclicGraph { .. })
    ));
}
#[test]
fn memory_planning_tracks_lifetimes_reuse_and_output_affinity() {
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let graph = ComputeGraph::new(ComputeGraphId::new("memory-graph"))
        .with_input(ComputeInput::new(
            ComputeInputId::new("x"),
            ComputeInputValue::TensorDescriptor(descriptor.clone()),
        ))
        .with_node(
            ComputeNode::new(
                ComputeNodeId::new("temp"),
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                    .with_dtype(ComputeDType::Float32)
                    .with_layout(ComputeLayout::Dense),
            )
            .with_input(ComputeValueRef::Input(ComputeInputId::new("x")))
            .with_output(ComputeNodeOutput::new(
                ComputeOutputId::new("tmp"),
                descriptor.clone(),
            )),
        )
        .with_node(
            ComputeNode::new(
                ComputeNodeId::new("result-node"),
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                    .with_dtype(ComputeDType::Float32)
                    .with_layout(ComputeLayout::Dense),
            )
            .with_input(ComputeValueRef::Input(ComputeInputId::new("x")))
            .with_output(ComputeNodeOutput::new(
                ComputeOutputId::new("y"),
                descriptor,
            )),
        )
        .with_output(ComputeOutput::new(
            ComputeOutputId::new("result"),
            ComputeValueRef::NodeOutput {
                node: ComputeNodeId::new("result-node"),
                output: ComputeOutputId::new("y"),
            },
        ));

    let plan = runtime
        .plan_compute_graph_memory("portable-compute", &graph)
        .unwrap();

    assert!(
        plan.decisions
            .iter()
            .any(|decision| matches!(decision, MemoryPlanningDecision::Reuse { .. }))
    );
    assert_eq!(
        plan.output_affinity.provider().map(ProviderBinding::as_str),
        Some("portable-compute")
    );
    assert!(
        plan.requirements
            .iter()
            .any(|requirement| requirement.region == MemoryRegionKind::Intermediate)
    );
}
#[test]
fn memory_planning_rejects_provider_and_device_memory_limits() {
    let mut limited_provider =
        provider_with_capabilities("limited-compute", [compute_capability()]);
    limited_provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense])
            .with_descriptor_limits(TensorDescriptorLimits {
                max_bytes: 8,
                ..TensorDescriptorLimits::default()
            }),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(limited_provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let graph = ComputeGraph::new(ComputeGraphId::new("too-large")).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                .with_dtype(ComputeDType::Float32)
                .with_layout(ComputeLayout::Dense),
        )
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor.clone(),
        )),
    );
    assert!(matches!(
        runtime.plan_compute_graph_memory("limited-compute", &graph),
        Err(MemoryPlanningError::ProviderMemoryLimitExceeded { .. })
    ));

    let mut device_provider = provider_with_capabilities("device-limited", [compute_capability()]);
    device_provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let mut device = DeviceMetadata::new(
        DeviceId::new("gpu:tiny"),
        "Tiny GPU",
        DeviceType::Gpu,
        "device-limited",
    );
    device.memory_capacity = 8;
    device_provider
        .devices
        .push(Arc::new(DeviceDescriptor::new(device)));
    let runtime = Runtime::builder()
        .register_provider(Arc::new(device_provider))
        .build()
        .unwrap();
    assert!(matches!(
        runtime.plan_compute_graph_memory("device-limited", &graph),
        Err(MemoryPlanningError::DeviceMemoryLimitExceeded { .. })
    ));
}
#[test]
fn compute_providers_register_and_resolve_compatibly() {
    let mut provider = TestProvider::new("portable-compute");
    provider.metadata.capabilities.insert(compute_capability());
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();

    assert_eq!(
        runtime
            .resolve_component_import(&WitInterface::new(
                COMPUTE_WIT_INTERFACE,
                COMPUTE_CAPABILITY_VERSION.to_string(),
            ))
            .unwrap()
            .len(),
        1
    );
}
#[test]
fn resolution_policy_records_selected_provider_capability_and_reason() {
    let compute = compute_capability();
    let provider_a = provider_with_capabilities("provider-a", [compute.clone()]);
    let provider_b = provider_with_capabilities("provider-b", [compute.clone()]);
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider_b))
        .register_provider(Arc::new(provider_a))
        .build()
        .unwrap();

    let resolution = runtime
        .resolve_with_affinity(&compute, &[], FallbackClass::Transparent)
        .unwrap();
    let decision = resolution.decision();

    assert_eq!(resolution.provider().metadata().name, "provider-a");
    assert_eq!(
        decision
            .selected_provider
            .as_ref()
            .map(ProviderBinding::as_str),
        Some("provider-a")
    );
    assert_eq!(
        decision.selected_capability,
        Some(CapabilityBinding::new(compute.id.clone(), compute.version))
    );
    assert_eq!(
        decision.reason,
        ResolutionDecisionReason::SelectedDeterministically
    );
}
#[test]
fn policy_rejection_is_structured_when_all_candidates_are_unhealthy() {
    let compute = compute_capability();
    let mut provider = provider_with_capabilities("provider-a", [compute.clone()]);
    provider.health = ProviderHealth::Unavailable;
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();

    assert!(matches!(
        runtime.resolve_component_import(&WitInterface::new(
            COMPUTE_WIT_INTERFACE,
            COMPUTE_CAPABILITY_VERSION.to_string(),
        )),
        Err(ProviderError::PolicyRejectedProvider { capability, policy })
            if capability.id() == &compute.id
                && capability.version() == compute.version
                && policy == BuiltInResolutionPolicy::Deterministic.id()
    ));
}
#[test]
fn availability_policy_prefers_healthy_candidates() {
    let compute = compute_capability();
    let mut degraded = provider_with_capabilities("a-degraded", [compute.clone()]);
    degraded.health = ProviderHealth::Degraded;
    let healthy = provider_with_capabilities("z-healthy", [compute.clone()]);
    let runtime = Runtime::builder()
        .config(RuntimeConfig {
            resolution_policy: BuiltInResolutionPolicy::Availability,
        })
        .register_provider(Arc::new(degraded))
        .register_provider(Arc::new(healthy))
        .build()
        .unwrap();

    let providers = runtime.resolve_component_import(&WitInterface::new(
        COMPUTE_WIT_INTERFACE,
        COMPUTE_CAPABILITY_VERSION.to_string(),
    ));

    assert_eq!(providers.unwrap()[0].metadata().name, "z-healthy");
}
#[test]
fn capability_health_rejects_unavailable_implementation() {
    let compute = compute_capability();
    let mut provider = provider_with_capabilities("provider-a", [compute.clone()]);
    provider
        .capability_health
        .insert(compute.id.clone(), HealthState::Unavailable);
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();

    assert!(matches!(
        runtime.resolve_component_import(&WitInterface::new(
            COMPUTE_WIT_INTERFACE,
            COMPUTE_CAPABILITY_VERSION.to_string(),
        )),
        Err(ProviderError::PolicyRejectedProvider { .. })
    ));

    let candidates = runtime
        .providers()
        .candidates_for_capability(&compute)
        .unwrap();
    let context = ResolutionContext {
        requested_capability: compute.id.clone(),
        requested_version: compute.version,
        candidates,
        affinity: None,
        fallback: FallbackClass::Transparent,
        execution_phase: ExecutionPhase::BeforeResourceCreation,
        replayable_input: true,
    };
    let decision = BuiltInResolutionPolicy::Deterministic.decide(&context);
    assert_eq!(
        decision.rejected_candidates[0].reason,
        ResolutionRejectionReason::CapabilityUnavailable
    );
}
#[test]
fn health_reports_redact_diagnostics_and_track_freshness() {
    let provider = ProviderBinding::new("provider-a");
    let mut report = ProviderHealthReport::new(provider.clone(), HealthState::Saturated);
    report.timestamp = Some(HealthTimestamp::unix_millis(1_000));
    report.time_to_live = Some(HealthTimeToLive::millis(250));
    report.capacity.queue_depth = Some(8);
    report.diagnostics.push(
        HealthDiagnostic::new(HealthScope::Provider, HealthState::Saturated)
            .with_code("queue-pressure")
            .with_message("cuda stream /tmp/secret token=abc is saturated")
            .with_trace_id("trace-1"),
    );

    assert!(report.is_stale_at(HealthTimestamp::unix_millis(1_251)));
    assert_eq!(report.capacity.queue_depth, Some(8));
    let message = report.diagnostics[0].message.as_deref().unwrap();
    assert!(!message.contains("/tmp/secret"));
    assert!(!message.contains("token=abc"));
    assert_eq!(message, "[redacted backend diagnostic]");
    assert_eq!(report.diagnostics[0].trace_id.as_deref(), Some("trace-1"));

    let device_health = DeviceHealth::new(
        provider.clone(),
        DeviceBinding::new(DeviceId::new("gpu:0")),
        HealthState::Available,
    );
    let capability_health = CapabilityHealth::new(
        provider,
        CapabilityBinding::new(compute_capability().id, COMPUTE_CAPABILITY_VERSION),
        HealthState::Degraded,
    );
    assert!(matches!(
        HealthReport::Device(device_health),
        HealthReport::Device(_)
    ));
    assert!(matches!(
        HealthReport::Capability(capability_health),
        HealthReport::Capability(_)
    ));
}
#[test]
fn phase_aware_resolution_rejects_restart_after_observable_output() {
    let compute = compute_capability();
    let provider = provider_with_capabilities("provider-a", [compute.clone()]);
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();

    assert!(matches!(
        runtime.resolve_with_affinity_at_phase(
            &compute,
            &[],
            FallbackClass::Restartable,
            ExecutionPhase::AfterObservableOutput,
            true,
        ),
        Err(AffinityError::PolicyRejectedProvider { .. })
    ));
}
#[test]
fn builder_isolates_failed_provider_initialization() {
    let mut failed = TestProvider::new("failed");
    failed.fail_initialization = true;
    let runtime = Runtime::builder()
        .register_provider(Arc::new(failed))
        .register_provider(Arc::new(TestProvider::new("available")))
        .build()
        .unwrap();
    assert!(runtime.providers().provider("failed").is_none());
    assert!(runtime.providers().provider("available").is_some());
}
#[test]
fn reject_incompatible() {
    let mut p = TestProvider::new("old");
    p.metadata.api_version += 1;
    assert!(matches!(
        ProviderLoader::new().register_provider(Arc::new(p)),
        Err(ProviderError::IncompatibleApiVersion { .. })
    ));
}
#[test]
fn reject_duplicate() {
    let mut m = ProviderLoader::new();
    m.register_provider(Arc::new(TestProvider::new("same")))
        .unwrap();
    assert!(matches!(
        m.register_provider(Arc::new(TestProvider::new("same"))),
        Err(ProviderError::ProviderAlreadyRegistered(_))
    ));
}
#[test]
fn provider_shutdown_releases_registered_provider() {
    let p = TestProvider::new("provider");
    let p = Arc::new(p);
    let mut m = ProviderLoader::new();
    m.register_provider(p.clone()).unwrap();
    assert!(m.provider("provider").is_some());
    m.shutdown().unwrap();
    assert!(p.shut_down.load(Ordering::SeqCst));
}

#[test]
fn runtime_source_does_not_restore_legacy_backend_or_plugin_surface() {
    let source = include_str!("lib.rs");
    let forbidden = [
        concat!("trait ", "Backend"),
        concat!("register_", "backend"),
        concat!("preferred_", "backend"),
        concat!("backend_", "names"),
        concat!("trait ", "Plugin"),
        concat!("Plugin", "Registry"),
    ];

    for term in forbidden {
        assert!(
            !source.contains(term),
            "legacy architecture surface remains in runtime source: {term}"
        );
    }
}

#[test]
fn execution_context_default_allocates_unique_ids() {
    let first = ExecutionContext::default();
    let second = ExecutionContext::default();
    assert_ne!(first.id(), second.id());
    assert_ne!(first.id(), ExecutionContextId::default());
}

#[test]
fn affinity_constraints_preserve_compatible_facts_and_fallback_precedence() {
    let capability_a = capability_binding("magnetar:compute/run", CapabilityVersion::new(1, 1, 0));
    let capability_b = capability_binding("magnetar:tokenize/run", CapabilityVersion::new(1, 0, 0));
    let provider = ProviderBinding::new("provider-a");
    let device = DeviceBinding::new(DeviceId::new("gpu:0"));
    let context = ExecutionContextId::new(42);
    let group = AffinityGroupId::new(7);

    let model = ResourceAffinity::new(FallbackClass::Transparent)
        .with_provider(provider.clone())
        .with_device(device.clone())
        .with_capability(capability_a.clone())
        .with_artifact(ArtifactBinding::new("model", "sha256:model"))
        .with_artifact(ArtifactBinding::new("bundle", "sha256:bundle"))
        .with_execution_context(context)
        .with_group(group);
    let tokenizer = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_provider(provider)
        .with_device(device)
        .with_capability(capability_b.clone())
        .with_artifact(ArtifactBinding::new("tokenizer", "sha256:tokenizer"))
        .with_artifact(ArtifactBinding::new("bundle", "sha256:bundle"))
        .with_execution_context(context)
        .with_group(group);

    let constraints = AffinityConstraints::try_from_affinities([&model, &tokenizer]).unwrap();
    let aggregate = constraints.affinity();
    assert_eq!(aggregate.capability(capability_a.id()), Some(&capability_a));
    assert_eq!(aggregate.capability(capability_b.id()), Some(&capability_b));
    assert_eq!(
        aggregate.artifact("model").unwrap().fingerprint(),
        "sha256:model"
    );
    assert_eq!(
        aggregate.artifact("tokenizer").unwrap().fingerprint(),
        "sha256:tokenizer"
    );
    assert_eq!(
        aggregate.artifact("bundle").unwrap().fingerprint(),
        "sha256:bundle"
    );
    assert_eq!(aggregate.fallback(), FallbackClass::ProviderPinned);
}

#[test]
fn affinity_constraints_report_each_binding_conflict() {
    let base = ResourceAffinity::new(FallbackClass::Transparent)
        .with_provider(ProviderBinding::new("provider-a"))
        .with_device(DeviceBinding::new(DeviceId::new("gpu:0")))
        .with_capability(capability_binding(
            "magnetar:compute/run",
            CapabilityVersion::new(1, 1, 0),
        ))
        .with_artifact(ArtifactBinding::new("bundle", "sha256:a"))
        .with_execution_context(ExecutionContextId::new(1))
        .with_group(AffinityGroupId::new(1));

    let provider_conflict = base
        .clone()
        .with_provider(ProviderBinding::new("provider-b"));
    assert!(matches!(
        base.validate_with(&provider_conflict),
        Err(AffinityError::ProviderMismatch { .. })
    ));

    let device_conflict = base
        .clone()
        .with_device(DeviceBinding::new(DeviceId::new("gpu:1")));
    assert!(matches!(
        base.validate_with(&device_conflict),
        Err(AffinityError::DeviceMismatch { .. })
    ));

    let capability_conflict = base.clone().with_capability(capability_binding(
        "magnetar:compute/run",
        CapabilityVersion::new(1, 2, 0),
    ));
    assert!(matches!(
        base.validate_with(&capability_conflict),
        Err(AffinityError::CapabilityMismatch { .. })
    ));

    let artifact_conflict = base
        .clone()
        .with_artifact(ArtifactBinding::new("bundle", "sha256:b"));
    assert!(matches!(
        base.validate_with(&artifact_conflict),
        Err(AffinityError::ArtifactMismatch { .. })
    ));

    let context_conflict = base
        .clone()
        .with_execution_context(ExecutionContextId::new(2));
    assert!(matches!(
        base.validate_with(&context_conflict),
        Err(AffinityError::ExecutionContextMismatch { .. })
    ));

    let group_conflict = base.clone().with_group(AffinityGroupId::new(2));
    assert!(matches!(
        base.validate_with(&group_conflict),
        Err(AffinityError::AffinityGroupMismatch { .. })
    ));
}

#[test]
fn affinity_resource_keeps_value_and_affinity_together() {
    let affinity = ResourceAffinity::new(FallbackClass::Restartable)
        .with_provider(ProviderBinding::new("provider-a"));
    let resource = AffinityResource::new("native-handle", affinity.clone());

    assert_eq!(resource.value(), &"native-handle");
    assert_eq!(resource.affinity(), &affinity);
    assert_eq!(resource.into_parts(), ("native-handle", affinity));
}

#[test]
fn affinity_resolution_uses_provider_local_compatible_version() {
    let requested = capability("magnetar:compute/run", CapabilityVersion::new(1, 1, 0));
    let provider_a = provider_with_capabilities(
        "provider-a",
        [capability(
            "magnetar:compute/run",
            CapabilityVersion::new(1, 1, 0),
        )],
    );
    let provider_b = provider_with_capabilities(
        "provider-b",
        [capability(
            "magnetar:compute/run",
            CapabilityVersion::new(1, 2, 0),
        )],
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider_a))
        .register_provider(Arc::new(provider_b))
        .build()
        .unwrap();
    let dependency = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_provider(ProviderBinding::new("provider-a"));

    let resolution = runtime
        .resolve_with_affinity(&requested, &[&dependency], FallbackClass::ProviderPinned)
        .unwrap();
    assert_eq!(resolution.provider().metadata().name, "provider-a");
    assert_eq!(
        resolution.capability().version,
        CapabilityVersion::new(1, 1, 0)
    );
    assert_eq!(
        resolution
            .affinity()
            .provider()
            .map(ProviderBinding::as_str),
        Some("provider-a")
    );
}

#[test]
fn affinity_resolution_preserves_exact_live_capability_version() {
    let requested = capability("magnetar:compute/run", CapabilityVersion::new(1, 1, 0));
    let provider = provider_with_capabilities(
        "provider-a",
        [
            capability("magnetar:compute/run", CapabilityVersion::new(1, 1, 0)),
            capability("magnetar:compute/run", CapabilityVersion::new(1, 2, 0)),
        ],
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let dependency = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_provider(ProviderBinding::new("provider-a"))
        .with_capability(capability_binding(
            "magnetar:compute/run",
            CapabilityVersion::new(1, 1, 0),
        ));

    let resolution = runtime
        .resolve_with_affinity(&requested, &[&dependency], FallbackClass::ProviderPinned)
        .unwrap();
    assert_eq!(
        resolution.capability().version,
        CapabilityVersion::new(1, 1, 0)
    );
}

#[test]
fn affinity_resolution_requires_selected_provider_to_implement_all_bound_capabilities() {
    let compute = capability("magnetar:compute/run", CapabilityVersion::new(1, 1, 0));
    let provider = provider_with_capabilities("provider-a", [compute.clone()]);
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let dependency = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_provider(ProviderBinding::new("provider-a"))
        .with_capability(capability_binding(
            "magnetar:tokenize/run",
            CapabilityVersion::new(1, 0, 0),
        ));

    assert!(matches!(
        runtime.resolve_with_affinity(&compute, &[&dependency], FallbackClass::ProviderPinned),
        Err(AffinityError::ProviderDoesNotImplementCapability { .. })
    ));
}

#[test]
fn affinity_resolution_reconciles_devices_with_provider_ownership() {
    let compute = compute_capability();
    let device_id = DeviceId::new("gpu:0");
    let mut provider = provider_with_capabilities("provider-a", [compute.clone()]);
    provider
        .devices
        .push(Arc::new(DeviceDescriptor::new(DeviceMetadata::new(
            device_id.clone(),
            "test gpu",
            DeviceType::Gpu,
            "provider-a",
        ))));
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let dependency = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_device(DeviceBinding::new(device_id.clone()));

    let resolution = runtime
        .resolve_with_affinity(&compute, &[&dependency], FallbackClass::ProviderPinned)
        .unwrap();
    assert_eq!(
        resolution.affinity().device().map(DeviceBinding::id),
        Some(&device_id)
    );
    assert_eq!(
        resolution
            .affinity()
            .provider()
            .map(ProviderBinding::as_str),
        Some("provider-a")
    );

    let mismatched = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_provider(ProviderBinding::new("other"))
        .with_device(DeviceBinding::new(device_id));
    assert!(matches!(
        runtime.resolve_with_affinity(&compute, &[&mismatched], FallbackClass::ProviderPinned),
        Err(AffinityError::DeviceProviderMismatch { .. })
    ));

    let missing = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_device(DeviceBinding::new(DeviceId::new("missing")));
    assert!(matches!(
        runtime.resolve_with_affinity(&compute, &[&missing], FallbackClass::ProviderPinned),
        Err(AffinityError::BoundDeviceUnavailable(_))
    ));
}

#[test]
fn affinity_resolution_reports_unavailable_bound_provider_without_fallback() {
    let compute = compute_capability();
    let mut fallback = provider_with_capabilities("fallback", [compute.clone()]);
    fallback.metadata.capabilities.insert(compute.clone());
    let runtime = Runtime::builder()
        .register_provider(Arc::new(fallback))
        .build()
        .unwrap();
    let dependency = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_provider(ProviderBinding::new("missing"));

    assert!(matches!(
        runtime.resolve_with_affinity(&compute, &[&dependency], FallbackClass::ProviderPinned),
        Err(AffinityError::BoundProviderUnavailable(provider)) if provider.as_str() == "missing"
    ));
}

#[test]
fn affinity_resolution_rejects_foreign_context_and_preserves_groups() {
    let compute = compute_capability();
    let first = Runtime::builder()
        .register_provider(Arc::new(provider_with_capabilities(
            "provider-a",
            [compute.clone()],
        )))
        .build()
        .unwrap();
    let second = Runtime::builder()
        .register_provider(Arc::new(provider_with_capabilities(
            "provider-a",
            [compute.clone()],
        )))
        .build()
        .unwrap();

    let ungrouped = first
        .resolve_with_affinity(&compute, &[], FallbackClass::ProviderPinned)
        .unwrap()
        .into_affinity();
    assert_eq!(ungrouped.group(), None);

    let grouped = first
        .resolve_with_affinity(&compute, &[&ungrouped], FallbackClass::ProviderPinned)
        .unwrap()
        .into_affinity();
    assert!(grouped.group().is_some());

    let inherited = first
        .resolve_with_affinity(&compute, &[&grouped], FallbackClass::ProviderPinned)
        .unwrap()
        .into_affinity();
    assert_eq!(inherited.group(), grouped.group());

    assert!(matches!(
        second.resolve_with_affinity(&compute, &[&grouped], FallbackClass::ProviderPinned),
        Err(AffinityError::ExecutionContextMismatch { .. })
    ));
}

#[test]
fn affinity_resolution_rejects_shutdown_runtime() {
    let compute = compute_capability();
    let mut runtime = Runtime::builder()
        .register_provider(Arc::new(provider_with_capabilities(
            "provider-a",
            [compute.clone()],
        )))
        .build()
        .unwrap();
    runtime.shutdown();

    assert!(matches!(
        runtime.resolve_with_affinity(&compute, &[], FallbackClass::Transparent),
        Err(AffinityError::RuntimeNotInitialized)
    ));
}

#[test]
fn component_runtime_instantiates_without_generic_start_or_stop() {
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("component", "1", "test component"),
            "component.wasm",
        ))
        .unwrap();

    let instance = manager.instantiate_component("component").unwrap();
    assert_eq!(
        manager.definition_state("component"),
        Some(ComponentDefinitionState::Prepared)
    );
    assert_eq!(
        manager.instance_state(instance),
        Some(ComponentInstanceState::Ready)
    );

    manager.shutdown();
    assert_eq!(
        manager.instance_state(instance),
        None,
        "shutdown removes Runtime-owned Component instances"
    );
}

#[test]
fn component_imports_are_authorized_and_linked_explicitly() {
    let interface = WitInterface::new("magnetar:runtime/run", "1.0.0");
    let metadata =
        ComponentMetadata::new("consumer", "1", "test component").with_import(interface.clone());
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(metadata, "consumer.wasm"))
        .unwrap();

    assert!(matches!(
        manager.instantiate_component("consumer"),
        Err(ComponentError::UnauthorizedImport { .. })
    ));

    manager.authorize_interface(interface.clone());
    assert!(matches!(
        manager.instantiate_component("consumer"),
        Err(ComponentError::UnresolvedImport { .. })
    ));

    manager.provide_interface(interface);
    let instance = manager.instantiate_component("consumer").unwrap();
    assert_eq!(
        manager.instance_state(instance),
        Some(ComponentInstanceState::Ready)
    );
}

#[test]
fn component_exports_do_not_automatically_satisfy_imports() {
    let interface = WitInterface::new("example:service/api", "1.0.0");
    let producer =
        ComponentMetadata::new("producer", "1", "producer").with_export(interface.clone());
    let consumer = ComponentMetadata::new("consumer", "1", "consumer").with_import(interface);
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(producer, "producer.wasm"))
        .unwrap();
    manager
        .register_component(ComponentDescriptor::new(consumer, "consumer.wasm"))
        .unwrap();

    assert!(matches!(
        manager.instantiate_component("consumer"),
        Err(ComponentError::UnauthorizedImport { .. })
    ));
}

#[test]
fn component_definition_can_create_multiple_isolated_instances() {
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("component", "1", "test component"),
            "component.wasm",
        ))
        .unwrap();

    let first = manager.instantiate_component("component").unwrap();
    let second = manager.instantiate_component("component").unwrap();
    assert_ne!(first, second);
    assert_eq!(
        manager.instance_state(first),
        Some(ComponentInstanceState::Ready)
    );
    assert_eq!(
        manager.instance_state(second),
        Some(ComponentInstanceState::Ready)
    );
}

#[test]
fn component_invocation_after_destruction_fails() {
    let interface = WitInterface::new("example:app/run", "1.0.0");
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("component", "1", "test component")
                .with_export(interface.clone()),
            "component.wasm",
        ))
        .unwrap();
    let instance = manager.instantiate_component("component").unwrap();
    manager.destroy_instance(instance).unwrap();

    assert!(matches!(
        manager.invoke(ComponentInvocation::new(instance, interface, "run")),
        Err(ComponentError::InstanceNotFound(_))
    ));
}

#[test]
fn component_engine_normalizes_traps_interruptions_and_limit_failures() {
    let interface = WitInterface::new("example:app/run", "1.0.0");
    let mut trapping_engine = MockComponentEngine::new();
    trapping_engine.trap_on_invoke = Some(ComponentTrapKind::Trap);
    let mut manager = ComponentManager::with_engine(Box::new(trapping_engine));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("component", "1", "test component")
                .with_export(interface.clone()),
            "component.wasm",
        ))
        .unwrap();
    let instance = manager.instantiate_component("component").unwrap();
    assert!(matches!(
        manager.invoke(ComponentInvocation::new(instance, interface, "run")),
        Err(ComponentError::Trap {
            kind: ComponentTrapKind::Trap,
            ..
        })
    ));

    let mut manager = ComponentManager::with_engine(Box::new(
        MockComponentEngine::new().without_resource_limits(),
    ));
    manager.set_resource_limits(ComponentResourceLimits {
        require_memory_limit: true,
        max_memory_bytes: Some(1024),
        ..ComponentResourceLimits::default()
    });
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("limited", "1", "test component"),
            "limited.wasm",
        ))
        .unwrap();
    assert!(matches!(
        manager.instantiate_component("limited"),
        Err(ComponentError::ResourceLimitUnsupported { .. })
    ));
}

#[test]
fn component_discovery_returns_only_wasm_artifacts() {
    let directory =
        std::env::temp_dir().join(format!("magnetar-components-{}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    fs::write(directory.join("valid.wasm"), []).unwrap();
    fs::write(directory.join("ignored.txt"), []).unwrap();

    let discovered = ComponentManager::discover([&directory]).unwrap();
    fs::remove_dir_all(&directory).unwrap();
    assert_eq!(discovered, vec![directory.join("valid.wasm")]);
}

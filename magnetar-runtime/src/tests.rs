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
    status_snapshot: Option<ProviderStatusSnapshot>,
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
            status_snapshot: None,
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
fn simple_elementwise_compute_graph(name: &str) -> ComputeGraph {
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    ComputeGraph::new(ComputeGraphId::new(name)).with_node(
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
    fn status_snapshot(&self) -> ProviderStatusSnapshot {
        self.status_snapshot
            .clone()
            .unwrap_or_else(|| ProviderStatusSnapshot::from_health_report(self.health_report()))
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
fn provider_status_snapshot_separates_health_readiness_pressure_and_admission() {
    let provider = ProviderBinding::new("provider-a");
    let mut snapshot = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        provider.clone(),
        HealthState::Available,
    ));
    snapshot.health = ProviderHealthState::Healthy;
    snapshot.readiness = ProviderReadinessState::NotReady;
    snapshot.pressure = ProviderPressureLevel::Low;
    snapshot.admission = provider_admission_from_dimensions(
        snapshot.lifecycle,
        snapshot.health,
        snapshot.readiness,
        snapshot.pressure,
    );
    snapshot.timestamp = Some(HealthTimestamp::unix_millis(10));
    snapshot.time_to_live = Some(HealthTimeToLive::millis(5));

    assert_eq!(snapshot.provider, provider);
    assert_eq!(snapshot.health, ProviderHealthState::Healthy);
    assert_eq!(snapshot.readiness, ProviderReadinessState::NotReady);
    assert_eq!(snapshot.pressure, ProviderPressureLevel::Low);
    assert_eq!(snapshot.admission, ProviderAdmissionDecision::Reject);
    assert!(!snapshot.accepts_new_work_by_default());
    assert!(snapshot.is_stale_at(HealthTimestamp::unix_millis(16)));
}

#[test]
fn provider_lifecycle_transitions_and_drain_completion_are_explicit() {
    assert!(ProviderLifecycleState::Registered.can_transition_to(ProviderLifecycleState::Loading));
    assert!(
        ProviderLifecycleState::Loading.can_transition_to(ProviderLifecycleState::Initializing)
    );
    assert!(ProviderLifecycleState::Initializing.can_transition_to(ProviderLifecycleState::Ready));
    assert!(ProviderLifecycleState::Ready.can_transition_to(ProviderLifecycleState::Draining));
    assert!(ProviderLifecycleState::Draining.can_transition_to(ProviderLifecycleState::Stopped));
    assert!(!ProviderLifecycleState::Ready.can_transition_to(ProviderLifecycleState::Removed));

    let mut draining = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        ProviderBinding::new("provider-a"),
        HealthState::Draining,
    ));
    draining.lifecycle = ProviderLifecycleState::Draining;
    draining.readiness = ProviderReadinessState::Draining;
    draining.in_flight_operations = 1;
    assert!(!draining.is_drain_complete());
    assert!(draining.pinned_work_allowed_during_drain());
    draining.in_flight_operations = 0;
    assert!(draining.is_drain_complete());
}

#[test]
fn operation_family_status_falls_back_to_capability_status_when_absent() {
    let provider = ProviderBinding::new("provider-a");
    let mut snapshot = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        provider.clone(),
        HealthState::Available,
    ));
    assert_eq!(
        snapshot.operation_family_or_capability_status(ComputeOperationFamily::LinearAlgebra),
        ProviderReadinessState::Ready
    );

    let unsupported =
        OperationFamilyStatus::unsupported(provider.clone(), ComputeOperationFamily::LinearAlgebra);
    snapshot = snapshot.with_operation_family_status(unsupported);
    assert_eq!(
        snapshot.operation_family_or_capability_status(ComputeOperationFamily::LinearAlgebra),
        ProviderReadinessState::NotReady
    );

    let mut saturated =
        OperationFamilyStatus::available(provider, ComputeOperationFamily::Elementwise);
    saturated.pressure = ProviderPressureLevel::Saturated;
    saturated.readiness = ProviderReadinessState::NotReady;
    snapshot = snapshot.with_operation_family_status(saturated);
    assert_eq!(
        snapshot
            .operation_family_status(ComputeOperationFamily::Elementwise)
            .unwrap()
            .pressure,
        ProviderPressureLevel::Saturated
    );
}

#[test]
fn provider_status_maps_interruption_to_health_readiness_and_admission() {
    let snapshot = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        ProviderBinding::new("provider-a"),
        HealthState::Interrupted,
    ));

    assert_eq!(snapshot.lifecycle, ProviderLifecycleState::Failed);
    assert_eq!(snapshot.health, ProviderHealthState::Failed);
    assert_eq!(snapshot.readiness, ProviderReadinessState::NotReady);
    assert_eq!(snapshot.admission, ProviderAdmissionDecision::Reject);
    assert_eq!(snapshot.health_reason, ProviderStatusReason::Interrupted);
    assert!(matches!(
        snapshot.interruption,
        Some(ProviderInterruptionReason::DriverLoss)
    ));
}

#[test]
fn provider_status_observations_cover_all_status_dimensions() {
    let mut report =
        ProviderHealthReport::new(ProviderBinding::new("provider-a"), HealthState::Draining);
    report.timestamp = Some(HealthTimestamp::unix_millis(10));
    report.time_to_live = Some(HealthTimeToLive::millis(5));
    report.devices.push(DeviceHealth::new(
        ProviderBinding::new("provider-a"),
        DeviceBinding::new(DeviceId::new("gpu:0")),
        HealthState::Available,
    ));
    report.capabilities.push(CapabilityHealth::new(
        ProviderBinding::new("provider-a"),
        CapabilityBinding::new(compute_capability().id, COMPUTE_CAPABILITY_VERSION),
        HealthState::Available,
    ));
    let mut snapshot = ProviderStatusSnapshot::from_health_report(report);
    snapshot.lifecycle = ProviderLifecycleState::Draining;
    snapshot.readiness = ProviderReadinessState::Draining;
    snapshot.in_flight_operations = 0;

    let events =
        runtime_events_for_provider_status(&snapshot, Some(HealthTimestamp::unix_millis(16)));
    let kinds = events
        .iter()
        .map(|event| event.kind.clone())
        .collect::<BTreeSet<_>>();

    assert!(kinds.contains(&RuntimeEventKind::ProviderLifecycleChanged));
    assert!(kinds.contains(&RuntimeEventKind::ProviderHealthChanged));
    assert!(kinds.contains(&RuntimeEventKind::ProviderReadinessChanged));
    assert!(kinds.contains(&RuntimeEventKind::ProviderPressureChanged));
    assert!(kinds.contains(&RuntimeEventKind::ProviderAdmissionChanged));
    assert!(kinds.contains(&RuntimeEventKind::ProviderStatusStale));
    assert!(kinds.contains(&RuntimeEventKind::ProviderDrainStarted));
    assert!(kinds.contains(&RuntimeEventKind::ProviderDrainCompleted));
    assert!(kinds.contains(&RuntimeEventKind::DeviceStatusChanged));
    assert!(kinds.contains(&RuntimeEventKind::CapabilityStatusChanged));
}

#[test]
fn resolution_rejects_healthy_provider_that_is_not_ready() {
    let compute = compute_capability();
    let mut provider = provider_with_capabilities("provider-a", [compute.clone()]);
    let mut snapshot = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        ProviderBinding::new("provider-a"),
        HealthState::Available,
    ));
    snapshot.health = ProviderHealthState::Healthy;
    snapshot.readiness = ProviderReadinessState::NotReady;
    snapshot.admission = provider_admission_from_dimensions(
        snapshot.lifecycle,
        snapshot.health,
        snapshot.readiness,
        snapshot.pressure,
    );
    provider.status_snapshot = Some(snapshot);
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();

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

    assert_eq!(decision.selected_provider, None);
    assert_eq!(
        decision.rejected_candidates[0].reason,
        ResolutionRejectionReason::ProviderInitializing
    );
}

#[test]
fn resolution_records_selected_provider_status_and_stale_rejection_reason() {
    let compute = compute_capability();
    let ready_provider = provider_with_capabilities("provider-a", [compute.clone()]);
    let mut stale_provider = provider_with_capabilities("provider-b", [compute.clone()]);
    let mut stale = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        ProviderBinding::new("provider-b"),
        HealthState::Available,
    ));
    stale.health_reason = ProviderStatusReason::Stale;
    stale_provider.status_snapshot = Some(stale);
    let runtime = Runtime::builder()
        .register_provider(Arc::new(stale_provider))
        .register_provider(Arc::new(ready_provider))
        .build()
        .unwrap();

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
        decision
            .selected_provider_status
            .as_ref()
            .map(|status| status.provider.as_str()),
        Some("provider-a")
    );
    assert!(decision.rejected_candidates.iter().any(|candidate| {
        candidate.provider.as_str() == "provider-b"
            && candidate.reason == ResolutionRejectionReason::ProviderStatusStale
    }));
}

#[test]
fn scheduler_checks_refined_provider_status_before_submission() {
    let compute = compute_capability();
    let mut planning_provider = provider_with_capabilities("provider-a", [compute]);
    planning_provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    planning_provider.execution_api = Some(Arc::new(TestProviderExecutionApi::new()));
    let planning_runtime = Runtime::builder()
        .register_provider(Arc::new(planning_provider))
        .build()
        .unwrap();
    let graph = simple_elementwise_compute_graph("status-change");
    let plan = planning_runtime.plan_compute_execution(&graph).unwrap();

    let mut submission_provider = provider_with_capabilities("provider-a", [compute_capability()]);
    submission_provider
        .metadata
        .compute_operation_support
        .insert(
            ComputeOperationFamily::Elementwise,
            ComputeOperationSupport::new()
                .with_dtypes([ComputeDType::Float32])
                .with_layouts([ComputeLayout::Dense]),
        );
    submission_provider.execution_api = Some(Arc::new(TestProviderExecutionApi::new()));
    let mut snapshot = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        ProviderBinding::new("provider-a"),
        HealthState::Available,
    ));
    snapshot.health = ProviderHealthState::Healthy;
    snapshot.readiness = ProviderReadinessState::Ready;
    snapshot.pressure = ProviderPressureLevel::Saturated;
    snapshot.admission = provider_admission_from_dimensions(
        snapshot.lifecycle,
        snapshot.health,
        snapshot.readiness,
        snapshot.pressure,
    );
    submission_provider.status_snapshot = Some(snapshot);
    let submission_runtime = Runtime::builder()
        .register_provider(Arc::new(submission_provider))
        .build()
        .unwrap();
    let mut scheduler = submission_runtime.scheduler(1);
    let operation = scheduler.schedule(&submission_runtime, plan).unwrap();

    assert!(matches!(
        scheduler.submit_next(&submission_runtime),
        Err(SchedulerError::ProviderSaturated(provider)) if provider.as_str() == "provider-a"
    ));
    assert_eq!(
        scheduler.operation(operation).unwrap().state(),
        SchedulingState::Interrupted
    );
}

#[test]
fn scheduler_and_provider_execution_reject_stale_provider_status() {
    let mut provider = provider_with_capabilities("provider-a", [compute_capability()]);
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    provider.execution_api = Some(Arc::new(TestProviderExecutionApi::new()));
    let planning_runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let plan = planning_runtime
        .plan_compute_execution(&simple_elementwise_compute_graph("stale-status"))
        .unwrap();

    let mut stale_provider = provider_with_capabilities("provider-a", [compute_capability()]);
    stale_provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    stale_provider.execution_api = Some(Arc::new(TestProviderExecutionApi::new()));
    let mut stale = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        ProviderBinding::new("provider-a"),
        HealthState::Available,
    ));
    stale.health_reason = ProviderStatusReason::Stale;
    stale_provider.status_snapshot = Some(stale);
    let runtime = Runtime::builder()
        .register_provider(Arc::new(stale_provider))
        .build()
        .unwrap();
    let mut scheduler = runtime.scheduler(1);
    let operation = scheduler.schedule(&runtime, plan).unwrap();

    assert!(matches!(
        scheduler.submit_next(&runtime),
        Err(SchedulerError::StaleHealthReport(provider)) if provider.as_str() == "provider-a"
    ));
    assert!(matches!(
        runtime.prepare_provider_execution(scheduler.operation(operation).unwrap()),
        Err(error) if error.code == ProviderExecutionErrorCode::StaleHealthReport
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
fn dynamic_provider_loading_denies_paths_by_default() {
    let mut loader = ProviderLoader::new();
    let path = std::path::PathBuf::from("target/test-provider.dll");

    let result = unsafe { loader.load_dynamic(&path) };

    assert!(matches!(
        result,
        Err(ProviderError::ProviderPathDenied { path: denied }) if denied == path
    ));
}

#[test]
fn dynamic_provider_loading_rejects_legacy_trait_object_factory_contract() {
    let mut loader = ProviderLoader::new();
    let root = std::path::PathBuf::from("target");
    let path = root.join("test-provider.dll");
    let policy = ProviderLoadingPolicy::development([root]);

    let result = unsafe { loader.load_dynamic_with_policy(&path, &policy) };

    assert!(matches!(
        result,
        Err(ProviderError::UnsupportedDynamicAbi {
            expected_symbol: PROVIDER_ABI_FACTORY_SYMBOL_V1,
            expected_descriptor,
            ..
        }) if expected_descriptor.abi_version == ProviderAbiVersion::CURRENT
    ));
    assert!(
        !include_str!("provider.rs").contains("Box<dyn Provider>"),
        "stable dynamic loading must not use Rust trait-object factories"
    );
}

#[test]
fn provider_abi_descriptor_validates_version_layout_functions_and_ownership() {
    let descriptor = ProviderAbiDescriptor::current();
    descriptor.validate().unwrap();
    assert_eq!(descriptor.abi_version, ProviderAbiVersion::CURRENT);
    assert!(descriptor.features.contains(&ProviderAbiFeature::Execution));
    assert_eq!(
        descriptor.threading,
        ProviderAbiThreadingModel::RuntimeSynchronized
    );
    assert_eq!(
        descriptor.execution_behavior,
        ProviderAbiExecutionBehavior::Blocking
    );
    assert_eq!(
        descriptor.unload_policy,
        ProviderAbiUnloadPolicy::NeverUnload
    );
    assert!(descriptor.ownership.strings.release_required);
    assert!(!descriptor.ownership.runtime_buffers.release_required);

    let mut too_small = descriptor.clone();
    too_small.descriptor_size = 1;
    assert!(matches!(
        too_small.validate(),
        Err(ProviderError::InvalidAbiDescriptor(_))
    ));

    let mut unsupported_major = descriptor.clone();
    unsupported_major.abi_version = ProviderAbiVersion::new(PROVIDER_ABI_MAJOR_VERSION + 1, 0);
    assert!(matches!(
        unsupported_major.validate(),
        Err(ProviderError::UnsupportedAbiVersion { .. })
    ));

    let mut unsupported_minor = descriptor.clone();
    unsupported_minor.abi_version =
        ProviderAbiVersion::new(PROVIDER_ABI_MAJOR_VERSION, PROVIDER_ABI_MINOR_VERSION + 1);
    assert!(matches!(
        unsupported_minor.validate(),
        Err(ProviderError::UnsupportedAbiVersion { .. })
    ));

    let mut missing_status = descriptor.clone();
    missing_status.functions.status = false;
    assert!(matches!(
        missing_status.validate(),
        Err(ProviderError::InvalidAbiDescriptor(_))
    ));

    let mut cross_allocator = descriptor;
    cross_allocator.ownership.error_messages = ProviderAbiMemoryRule::runtime_borrowed();
    assert!(matches!(
        cross_allocator.validate(),
        Err(ProviderError::InvalidAbiDescriptor(_))
    ));
}

#[test]
fn provider_abi_handles_lifecycle_and_errors_are_internal_runtime_contracts() {
    let instance = ProviderAbiHandleDescriptor::new(
        ProviderAbiHandleKind::ProviderInstance,
        ProviderAbiHandle::new(7),
    );
    let resource = ProviderAbiHandleDescriptor::new(
        ProviderAbiHandleKind::ProviderResource,
        ProviderAbiHandle::new(8),
    );
    let operation = ProviderAbiHandleDescriptor::new(
        ProviderAbiHandleKind::Operation,
        ProviderAbiHandle::new(9),
    );

    assert!(instance.destroy_required);
    assert_eq!(resource.handle.as_u64(), 8);
    assert_eq!(operation.kind, ProviderAbiHandleKind::Operation);
    assert!(
        ProviderAbiLoadingLifecycle::Discovered
            .can_transition_to(ProviderAbiLoadingLifecycle::LibraryLoaded)
    );
    assert!(
        !ProviderAbiLoadingLifecycle::Discovered
            .can_transition_to(ProviderAbiLoadingLifecycle::Registered)
    );
    assert!(
        ProviderAbiLoadingLifecycle::Failed
            .can_transition_to(ProviderAbiLoadingLifecycle::Destroyed)
    );

    let categories = [
        ProviderAbiErrorCode::InvalidAbiDescriptor,
        ProviderAbiErrorCode::UnsupportedAbiVersion,
        ProviderAbiErrorCode::InvalidMetadata,
        ProviderAbiErrorCode::InvalidAdvertisement,
        ProviderAbiErrorCode::InvalidDeviceMetadata,
        ProviderAbiErrorCode::InitializationFailure,
        ProviderAbiErrorCode::ProviderNotReady,
        ProviderAbiErrorCode::ProviderDraining,
        ProviderAbiErrorCode::ProviderSaturated,
        ProviderAbiErrorCode::ExecutionRejected,
        ProviderAbiErrorCode::ExecutionFailed,
        ProviderAbiErrorCode::CancellationUnsupported,
        ProviderAbiErrorCode::CancellationFailed,
        ProviderAbiErrorCode::ResourceInvalid,
        ProviderAbiErrorCode::InternalProviderError,
        ProviderAbiErrorCode::PanicOrUnwindViolation,
    ];
    assert_eq!(categories.len(), 16);

    let compute_error = ComputeError::from(ProviderError::PanicOrUnwindViolation(
        "panic crossed boundary".into(),
    ));
    assert_eq!(compute_error.code, ComputeErrorCode::ProviderUnavailable);
    assert_eq!(compute_error.phase, ComputeErrorPhase::Resolution);
}

#[test]
fn provider_loading_policy_is_explicit_for_dynamic_and_development_modes() {
    let root = std::path::PathBuf::from("target/providers");
    let provider = root.join("provider.dll");
    let outside = std::path::PathBuf::from("target/other/provider.dll");
    let dynamic = ProviderLoadingPolicy::dynamic_library([root.clone()]);
    let development = ProviderLoadingPolicy::development([root.clone()]);

    assert_eq!(dynamic.mode, ProviderLoadingMode::DynamicLibrary);
    assert!(!dynamic.development_mode);
    assert!(dynamic.allows(&provider));
    assert!(!dynamic.allows(&outside));
    assert_eq!(development.mode, ProviderLoadingMode::DevelopmentProvider);
    assert!(development.development_mode);
    assert!(development.allows(&provider));
    assert!(!ProviderLoadingPolicy::default().allows(&provider));
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
fn public_component_api_does_not_expose_wasmtime_native_types() {
    let public_sources = [include_str!("lib.rs"), include_str!("component.rs")];
    let forbidden = [
        "wasmtime::Engine",
        "wasmtime::Config",
        "wasmtime::Store",
        "wasmtime::component::Component",
        "wasmtime::component::Linker",
        "wasmtime::component::Instance",
        "wasmtime::Trap",
        "wasmtime::Error",
    ];

    for source in public_sources {
        for term in forbidden {
            assert!(
                !source.contains(term),
                "public Component API exposes concrete engine type: {term}"
            );
        }
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
fn component_artifact_reference_prepares_future_artifact_model_without_trust_policy() {
    let descriptor = ComponentDescriptor::new(
        ComponentMetadata::new("component", "1", "test component"),
        "component.wasm",
    );

    assert!(matches!(
        descriptor.artifact_reference(),
        ComponentArtifactReference::LocalPath(path) if path == std::path::Path::new("component.wasm")
    ));
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
fn component_import_version_must_match_authorized_interface() {
    let authorized = WitInterface::new("magnetar:runtime/run", "1.0.0");
    let requested = WitInterface::new("magnetar:runtime/run", "2.0.0");
    let metadata = ComponentMetadata::new("consumer", "1", "test component").with_import(requested);
    let mut manager = ComponentManager::new();
    manager.provide_interface(authorized);
    manager
        .register_component(ComponentDescriptor::new(metadata, "consumer.wasm"))
        .unwrap();

    assert!(matches!(
        manager.instantiate_component("consumer"),
        Err(ComponentError::UnauthorizedImport { .. })
    ));
}

#[test]
fn component_wasi_imports_fail_closed_without_authorization() {
    let filesystem = WitInterface::new("wasi:filesystem/types", "0.2.0");
    let environment = WitInterface::new("wasi:cli/environment", "0.2.0");
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("wasi-consumer", "1", "test component")
                .with_import(filesystem)
                .with_import(environment),
            "wasi-consumer.wasm",
        ))
        .unwrap();

    assert!(matches!(
        manager.instantiate_component("wasi-consumer"),
        Err(ComponentError::UnauthorizedImport { .. })
    ));
}

#[test]
fn component_ambient_network_process_and_secret_imports_fail_closed() {
    let interfaces = [
        WitInterface::new("wasi:sockets/tcp", "0.2.0"),
        WitInterface::new("wasi:cli/run", "0.2.0"),
        WitInterface::new("magnetar:secrets/read", "1.0.0"),
    ];
    for (index, interface) in interfaces.into_iter().enumerate() {
        let name = format!("authority-{index}");
        let mut manager = ComponentManager::new();
        manager
            .register_component(ComponentDescriptor::new(
                ComponentMetadata::new(&name, "1", "test component").with_import(interface),
                format!("{name}.wasm"),
            ))
            .unwrap();

        assert!(matches!(
            manager.instantiate_component(&name),
            Err(ComponentError::UnauthorizedImport { .. })
        ));
    }
}

#[test]
fn component_link_plan_is_runtime_owned_and_immutable_to_callers() {
    let interface = WitInterface::new("magnetar:runtime/run", "1.0.0");
    let metadata =
        ComponentMetadata::new("consumer", "1", "test component").with_import(interface.clone());
    let mut manager = ComponentManager::new();
    manager.provide_interface(interface.clone());
    manager
        .register_component(ComponentDescriptor::new(metadata, "consumer.wasm"))
        .unwrap();

    let plan = manager.link_plan("consumer").unwrap();
    let links = plan.links().collect::<Vec<_>>();
    assert_eq!(links.len(), 1);
    assert_eq!(links[0].0, &interface);
    assert!(matches!(
        links[0].1,
        ComponentEndpoint::Capability { interface: linked } if linked == &interface
    ));
    assert_eq!(plan.endpoint(&interface), Some(links[0].1));
}

#[test]
fn component_link_plan_rejects_forbidden_external_interfaces_even_if_provided() {
    for interface in [
        WitInterface::new("wasi:filesystem/types", "0.2.0"),
        WitInterface::new("wasi:sockets/tcp", "0.2.0"),
        WitInterface::new("magnetar:workspace/read", "1.0.0"),
        WitInterface::new("magnetar:git/status", "1.0.0"),
        WitInterface::new("magnetar:process/run", "1.0.0"),
        WitInterface::new("magnetar:secrets/read", "1.0.0"),
    ] {
        let mut manager = ComponentManager::new();
        manager.provide_interface(interface.clone());
        manager
            .register_component(ComponentDescriptor::new(
                ComponentMetadata::new("external", "1", "external component")
                    .with_import(interface),
                "external.wasm",
            ))
            .unwrap();

        assert!(matches!(
            manager.link_plan("external"),
            Err(ComponentError::UnauthorizedImport { .. })
        ));
    }
}

#[test]
fn component_authority_requirements_map_to_inference_runtime_endpoints() {
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "compute-capability".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::Capability { interface }
            if interface == WitInterface::new("magnetar:compute/run", "2.0.0")
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "model-artifact-read".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::InferenceArtifactRegistry {
            kind: InferenceArtifactKind::Model
        }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "tokenizer-artifact-read".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::InferenceArtifactRegistry {
            kind: InferenceArtifactKind::Tokenizer
        }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "prompt-template-read".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::InferenceArtifactRegistry {
            kind: InferenceArtifactKind::PromptTemplate
        }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "adapter-artifact-read".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::InferenceArtifactRegistry {
            kind: InferenceArtifactKind::Adapter
        }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "quantization-artifact-read".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::InferenceArtifactRegistry {
            kind: InferenceArtifactKind::Quantization
        }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "kv-cache-access".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::InferenceCacheService {
            kind: InferenceCacheKind::Kv
        }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "prefix-cache-access".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::InferenceCacheService {
            kind: InferenceCacheKind::Prefix
        }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "observability-emit".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::Observability
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "runtime-diagnostics".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::RuntimeDiagnostics
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "generation-capability".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::PendingRuntimeService { .. }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "sampling-capability".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::PendingRuntimeService { .. }
    ));
}

#[test]
fn inference_artifact_registry_uses_identities_not_paths_and_scopes_sessions() {
    let mut manager = ComponentManager::new();
    let digest = ComponentDigest::sha256(b"model");
    let session = InferenceSessionId::new("session-a").unwrap();
    manager
        .register_inference_artifact(
            InferenceArtifactReference::new(InferenceArtifactKind::Model, "qwen-model", digest)
                .unwrap()
                .with_session(session.clone()),
        )
        .unwrap();

    let artifact = manager
        .resolve_inference_artifact(InferenceArtifactKind::Model, "qwen-model", Some(&session))
        .unwrap();
    assert_eq!(artifact.id, "qwen-model");
    assert!(matches!(
        manager.resolve_inference_artifact(InferenceArtifactKind::Model, "../qwen-model", None),
        Err(ComponentError::ArtifactRejected { .. })
    ));
    assert!(matches!(
        manager.resolve_inference_artifact(
            InferenceArtifactKind::Model,
            "qwen-model",
            Some(&InferenceSessionId::new("session-b").unwrap())
        ),
        Err(ComponentError::ArtifactRejected { .. })
    ));
}

#[test]
fn inference_artifact_registry_handles_tokenizer_template_adapter_and_quantization() {
    let mut registry = InferenceArtifactRegistry::default();
    for kind in [
        InferenceArtifactKind::Tokenizer,
        InferenceArtifactKind::PromptTemplate,
        InferenceArtifactKind::Adapter,
        InferenceArtifactKind::Quantization,
    ] {
        let id = format!("{kind:?}").to_ascii_lowercase();
        registry
            .register(
                InferenceArtifactReference::new(kind, &id, ComponentDigest::sha256(id.as_bytes()))
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(registry.resolve(kind, &id, None).unwrap().kind, kind);
    }
    assert!(matches!(
        registry.resolve(InferenceArtifactKind::Tokenizer, "C:\\tokenizer.json", None),
        Err(ComponentError::ArtifactRejected { .. })
    ));
}

#[test]
fn inference_cache_registry_scopes_access_to_session_and_model() {
    let mut registry = InferenceCacheRegistry::default();
    let session = InferenceSessionId::new("session-a").unwrap();
    let authorized =
        InferenceCacheScope::new(InferenceCacheKind::Kv, session.clone(), "qwen-model").unwrap();
    registry.authorize(authorized.clone());

    registry.authorize_access(&authorized).unwrap();
    assert!(matches!(
        registry.authorize_access(
            &InferenceCacheScope::new(InferenceCacheKind::Kv, session, "other-model").unwrap()
        ),
        Err(ComponentError::ArtifactRejected { .. })
    ));
    assert!(matches!(
        registry.authorize_access(
            &InferenceCacheScope::new(
                InferenceCacheKind::Prefix,
                InferenceSessionId::new("session-b").unwrap(),
                "qwen-model"
            )
            .unwrap()
        ),
        Err(ComponentError::ArtifactRejected { .. })
    ));
}

#[test]
fn prepared_component_contract_must_match_declared_imports() {
    let declared = WitInterface::new("example:declared/api", "1.0.0");
    let undeclared = WitInterface::new("example:undeclared/api", "1.0.0");
    let mut contract = ComponentContract::default();
    contract.imports.insert(ComponentImportRequirement::new(
        undeclared,
        ComponentInterfaceShape::Interface,
    ));
    let mut engine = MockComponentEngine::new();
    engine.prepared_contract = Some(contract);
    let mut manager = ComponentManager::with_engine(Box::new(engine));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("consumer", "1", "test component").with_import(declared),
            "consumer.wasm",
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("consumer"),
        Err(ComponentError::ContractValidationFailed { .. })
    ));
}

#[test]
fn prepared_component_contract_must_include_declared_exports() {
    let exported = WitInterface::new("example:export/api", "1.0.0");
    let mut engine = MockComponentEngine::new();
    engine.prepared_contract = Some(ComponentContract::default());
    let mut manager = ComponentManager::with_engine(Box::new(engine));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("producer", "1", "test component").with_export(exported),
            "producer.wasm",
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("producer"),
        Err(ComponentError::ContractValidationFailed { .. })
    ));
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
fn component_manager_enforces_instance_and_invocation_limits() {
    let mut manager = ComponentManager::new();
    manager.set_resource_limits(ComponentResourceLimits {
        max_instances: Some(1),
        ..ComponentResourceLimits::default()
    });
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("component", "1", "test component"),
            "component.wasm",
        ))
        .unwrap();

    manager.instantiate_component("component").unwrap();
    assert!(matches!(
        manager.instantiate_component("component"),
        Err(ComponentError::ResourceLimitExceeded {
            limit: "instances",
            ..
        })
    ));

    let interface = WitInterface::new("example:app/run", "1.0.0");
    let mut manager = ComponentManager::new();
    manager.set_resource_limits(ComponentResourceLimits {
        max_concurrent_invocations: Some(0),
        ..ComponentResourceLimits::default()
    });
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("callable", "1", "test component")
                .with_export(interface.clone()),
            "callable.wasm",
        ))
        .unwrap();
    let instance = manager.instantiate_component("callable").unwrap();
    assert!(matches!(
        manager.invoke(ComponentInvocation::new(instance, interface, "run")),
        Err(ComponentError::ResourceLimitExceeded {
            limit: "concurrent invocations",
            ..
        })
    ));
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
fn component_shutdown_prevents_new_lifecycle_operations() {
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
    manager.shutdown();

    assert!(matches!(
        manager.invoke(ComponentInvocation::new(instance, interface, "run")),
        Err(ComponentError::RuntimeShutdown)
    ));
    assert!(matches!(
        manager.instantiate_component("component"),
        Err(ComponentError::RuntimeShutdown)
    ));
    assert!(matches!(
        manager.register_component(ComponentDescriptor::new(
            ComponentMetadata::new("other", "1", "test component"),
            "other.wasm",
        )),
        Err(ComponentError::RuntimeShutdown)
    ));
}

#[test]
fn component_observations_are_non_authoritative_and_redacted() {
    let interface = WitInterface::new("example:app/run", "1.0.0");
    let mut engine = MockComponentEngine::new();
    engine.trap_on_invoke = Some(ComponentTrapKind::Trap);
    let mut manager = ComponentManager::with_engine(Box::new(engine));
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
        Err(ComponentError::Trap { .. })
    ));
    assert!(
        manager
            .observations()
            .iter()
            .any(
                |observation| observation.kind == ComponentObservationKind::Trap
                    && observation.instance == Some(instance)
                    && observation.message.contains("[redacted component trap]")
            )
    );
    assert!(
        !manager
            .observations()
            .iter()
            .any(|observation| observation.message.contains("wasmtime::"))
    );
    assert!(!manager.observations().iter().any(|observation| {
        observation.message.contains("Provider")
            || observation.message.contains("Device")
            || observation.message.contains("Store")
    }));
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

fn temp_component_artifact_dir(label: &str) -> std::path::PathBuf {
    let directory = std::env::temp_dir().join(format!(
        "magnetar-component-artifact-{label}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&directory);
    fs::create_dir_all(&directory).unwrap();
    directory
}

fn manifest_yaml(digest: &str, runtime_version: &str) -> String {
    format!(
        r#"schema: magnetar-component-artifact
schema_version: 1
artifact:
  kind: component
  digest:
    algorithm: sha256
    value: "{digest}"
component:
  name: "magnetar.examples.hello"
  version: "0.1.0"
  description: "Minimal Magnetar Component fixture"
  role: "test-fixture"
runtime:
  magnetar:
    min_version: "{runtime_version}"
wit:
  imports:
    - package: "magnetar:test"
      interface: "echo"
      version: "1.0.0"
  exports:
    - package: "magnetar:test"
      interface: "run"
      version: "1.0.0"
capabilities:
  requires:
    - id: "magnetar:test/echo"
      version: "1.0.0"
authority:
  requires: []
publisher:
  id: "local-dev"
  name: "Local Development"
source:
  kind: "local"
  uri: "./fixtures/hello.component.wasm"
signatures: []
"#
    )
}

fn tokenizer_manifest_yaml(digest: &str) -> String {
    format!(
        r#"schema: magnetar-component-artifact
schema_version: 1
artifact:
  kind: component
  digest:
    algorithm: sha256
    value: "{digest}"
component:
  name: "magnetar.examples.tokenizer"
  version: "0.1.0"
  description: "Tokenizer Component fixture"
  role: "tokenizer"
runtime:
  magnetar:
    min_version: "0.1.0"
wit:
  imports:
    - package: "magnetar:compute"
      interface: "run"
      version: "2.0.0"
  exports:
    - package: "magnetar:tokenizer"
      interface: "tokenize"
      version: "1.0.0"
capabilities:
  requires:
    - id: "magnetar:compute/run"
      version: "2.0.0"
authority:
  requires:
    - tokenizer-artifact-read
    - compute-capability
    - observability-emit
publisher:
  id: "local-dev"
  name: "Local Development"
source:
  kind: "local"
  uri: "./fixtures/tokenizer.component.wasm"
signatures: []
"#
    )
}

fn manifest_yaml_with_authority(digest: &str, authorities: &[&str]) -> String {
    let requires = authorities
        .iter()
        .map(|authority| format!("    - {authority}"))
        .collect::<Vec<_>>()
        .join("\n");
    manifest_yaml(digest, MAGNETAR_RUNTIME_VERSION).replace(
        "authority:\n  requires: []",
        &format!("authority:\n  requires:\n{requires}"),
    )
}

fn trust_store_yaml(digest: &str) -> String {
    format!(
        r#"schema: magnetar-component-trust
schema_version: 1
trusted_digests:
  - "{digest}"
rejected_digests: []
revoked_digests: []
trusted_publishers: []
trusted_sources: []
development:
  allow_unsigned_local: false
"#
    )
}

fn component_artifact_package(
    bytes: &[u8],
    source_kind: ComponentDistributionSourceKind,
) -> ComponentArtifactPackage {
    let digest = ComponentDigest::sha256(bytes);
    ComponentArtifactPackage::new(
        bytes.to_vec(),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION).into_bytes(),
        digest,
        ComponentDistributionSource::new(source_kind, "test-source"),
    )
}

#[derive(Clone)]
struct TestComponentDistributionSource {
    package: ComponentArtifactPackage,
    candidates: Vec<ComponentDigest>,
}

impl ComponentDistributionSourceProvider for TestComponentDistributionSource {
    fn resolve(
        &self,
        component: &str,
        _version_requirement: Option<&str>,
    ) -> Result<Vec<ComponentDigest>, ComponentError> {
        if component == "magnetar.examples.hello" {
            Ok(self.candidates.clone())
        } else {
            Ok(Vec::new())
        }
    }

    fn fetch(&self, digest: &ComponentDigest) -> Result<ComponentArtifactPackage, ComponentError> {
        if self.package.declared_digest == *digest {
            Ok(self.package.clone())
        } else {
            Err(ComponentError::Distribution {
                category: ComponentDistributionErrorCategory::ArtifactNotFound,
                message: "digest not found".into(),
            })
        }
    }
}

#[test]
fn pushed_component_package_is_validated_before_preparation() {
    let bytes = b"component-bytes";
    let digest = ComponentDigest::sha256(bytes);
    let package =
        component_artifact_package(bytes, ComponentDistributionSourceKind::ClientProvided);
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value.clone()));

    manager.prepare_pushed_package(package).unwrap();

    let definition = manager.definition("magnetar.examples.hello").unwrap();
    assert_eq!(definition.artifact_digest, Some(digest));
    assert!(matches!(
        definition.trust_decision,
        Some(ComponentTrustDecision {
            status: ComponentTrustStatus::Trusted,
            ..
        })
    ));
    assert!(manager.observations().iter().any(|observation| {
        observation.kind == ComponentObservationKind::Distribution
            && observation.message.contains("client-provided")
    }));
}

#[test]
fn pushed_component_package_rejects_source_digest_mismatch() {
    let mut package =
        component_artifact_package(b"component-bytes", ComponentDistributionSourceKind::Tachyon);
    package.declared_digest = ComponentDigest::sha256(b"different");
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_source("tachyon"));

    assert!(matches!(
        manager.prepare_pushed_package(package),
        Err(ComponentError::Distribution {
            category: ComponentDistributionErrorCategory::DigestMismatch,
            ..
        })
    ));
}

#[test]
fn distribution_source_identity_does_not_imply_trust() {
    let package =
        component_artifact_package(b"component-bytes", ComponentDistributionSourceKind::Tachyon);
    let mut manager = ComponentManager::new();

    assert!(matches!(
        manager.prepare_pushed_package(package),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Unknown,
            ..
        })
    ));
}

#[test]
fn trusted_distribution_source_still_rejects_forbidden_authority() {
    let bytes = b"component-bytes";
    let digest = ComponentDigest::sha256(bytes);
    let manifest = manifest_yaml_with_authority(&digest.value, &["filesystem"])
        .replace("  kind: \"local\"", "  kind: \"tachyon\"");
    let package = ComponentArtifactPackage::new(
        bytes.to_vec(),
        manifest.into_bytes(),
        digest,
        ComponentDistributionSource::new(ComponentDistributionSourceKind::Tachyon, "tachyon"),
    );
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_source("tachyon"));

    assert!(matches!(
        manager.prepare_pushed_package(package),
        Err(ComponentError::Manifest { message, .. })
            if message == "authority kind is outside Magnetar inference scope"
    ));
}

#[test]
fn pulled_component_package_resolves_fetches_and_validates_locally() {
    let bytes = b"component-bytes";
    let digest = ComponentDigest::sha256(bytes);
    let package =
        component_artifact_package(bytes, ComponentDistributionSourceKind::LocalDirectory);
    let source = TestComponentDistributionSource {
        package,
        candidates: vec![digest.clone()],
    };
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value.clone()));

    manager
        .prepare_pulled_package(&source, "magnetar.examples.hello", Some(">=0.1.0,<1.0.0"))
        .unwrap();

    assert_eq!(
        manager
            .definition("magnetar.examples.hello")
            .and_then(|definition| definition.artifact_digest.clone()),
        Some(digest)
    );
    assert!(manager.observations().iter().any(|observation| {
        observation.kind == ComponentObservationKind::Distribution
            && observation.message.contains("candidate digest")
    }));
}

#[test]
fn pulled_component_package_rejects_empty_candidate_list() {
    let package = component_artifact_package(
        b"component-bytes",
        ComponentDistributionSourceKind::LocalCache,
    );
    let source = TestComponentDistributionSource {
        package,
        candidates: Vec::new(),
    };
    let mut manager = ComponentManager::new();

    assert!(matches!(
        manager.prepare_pulled_package(&source, "magnetar.examples.hello", None),
        Err(ComponentError::Distribution {
            category: ComponentDistributionErrorCategory::ArtifactNotFound,
            ..
        })
    ));
}

#[test]
fn local_distribution_does_not_require_tachyon_or_network() {
    let bytes = b"component-bytes";
    let digest = ComponentDigest::sha256(bytes);
    let package =
        component_artifact_package(bytes, ComponentDistributionSourceKind::LocalDirectory);
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));

    manager.prepare_pushed_package(package).unwrap();
    assert_eq!(
        manager.definition_state("magnetar.examples.hello"),
        Some(ComponentDefinitionState::Prepared)
    );
}

#[test]
fn component_artifact_accepts_target_tokenizer_manifest_authorities() {
    let directory = temp_component_artifact_dir("tokenizer-authority");
    let artifact = directory.join("tokenizer.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("tokenizer.component.wasm.magnetar-component.yaml"),
        tokenizer_manifest_yaml(&digest.value),
    )
    .unwrap();

    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.tokenizer", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:compute/run", "2.0.0"))
                .with_export(WitInterface::new("magnetar:tokenizer/tokenize", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    manager
        .prepare_component("magnetar.examples.tokenizer")
        .unwrap();
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_rejects_broad_authority_kinds() {
    for authority in [
        "filesystem",
        "network",
        "secrets",
        "git",
        "workspace",
        "process",
    ] {
        let directory = temp_component_artifact_dir(authority);
        let artifact = directory.join("hello.component.wasm");
        let bytes = b"component-bytes";
        fs::write(&artifact, bytes).unwrap();
        let digest = ComponentDigest::sha256(bytes);
        fs::write(
            directory.join("hello.component.wasm.magnetar-component.yaml"),
            manifest_yaml_with_authority(&digest.value, &[authority]),
        )
        .unwrap();
        let mut manager = ComponentManager::new();
        manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
        manager
            .register_component(ComponentDescriptor::new(
                ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                    .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                    .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
                &artifact,
            ))
            .unwrap();

        assert!(matches!(
            manager.prepare_component("magnetar.examples.hello"),
            Err(ComponentError::Manifest { message, .. })
                if message == "authority kind is outside Magnetar inference scope"
        ));
        fs::remove_dir_all(directory).unwrap();
    }
}

#[test]
fn component_artifact_rejects_unknown_authority_kinds() {
    let directory = temp_component_artifact_dir("unknown-authority");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml_with_authority(&digest.value, &["workspace-admin"]),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::Manifest { message, .. }) if message == "unsupported authority kind"
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_accepts_model_artifact_read_authority_when_trusted() {
    let directory = temp_component_artifact_dir("model-artifact-authority");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml_with_authority(&digest.value, &["model-artifact-read"]),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    manager
        .prepare_component("magnetar.examples.hello")
        .unwrap();
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_trust_overrides_do_not_allow_forbidden_authority() {
    for (label, trust_store, source_kind) in [
        ("trusted-digest", ComponentTrustStore::default(), "local"),
        (
            "trusted-publisher",
            ComponentTrustStore::default().trust_publisher("local-dev"),
            "local",
        ),
        (
            "trusted-tachyon-source",
            ComponentTrustStore::default().trust_source("tachyon"),
            "tachyon",
        ),
        (
            "trusted-local-source",
            ComponentTrustStore::default().trust_source("local"),
            "local",
        ),
        (
            "development-mode",
            ComponentTrustStore::default().allow_unsigned_local_development(true),
            "local",
        ),
    ] {
        let directory = temp_component_artifact_dir(label);
        let artifact = directory.join("hello.component.wasm");
        let bytes = b"component-bytes";
        fs::write(&artifact, bytes).unwrap();
        let digest = ComponentDigest::sha256(bytes);
        let mut trust_store = trust_store;
        if label == "trusted-digest" {
            trust_store = trust_store.trust_digest(digest.value.clone());
        }
        let manifest = manifest_yaml_with_authority(&digest.value, &["filesystem"])
            .replace("  kind: \"local\"", &format!("  kind: \"{source_kind}\""));
        fs::write(
            directory.join("hello.component.wasm.magnetar-component.yaml"),
            manifest,
        )
        .unwrap();
        let mut manager = ComponentManager::new();
        manager.set_trust_store(trust_store);
        manager
            .register_component(ComponentDescriptor::new(
                ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                    .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                    .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
                &artifact,
            ))
            .unwrap();

        assert!(matches!(
            manager.prepare_component("magnetar.examples.hello"),
            Err(ComponentError::Manifest { message, .. })
                if message == "authority kind is outside Magnetar inference scope"
        ));
        fs::remove_dir_all(directory).unwrap();
    }
}

#[test]
fn component_authority_rejection_is_observed_with_reason_before_prepare() {
    let directory = temp_component_artifact_dir("authority-diagnostic");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml_with_authority(&digest.value, &["filesystem"]),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::Manifest { .. })
    ));
    assert_eq!(
        manager.definition_state("magnetar.examples.hello"),
        Some(ComponentDefinitionState::Failed)
    );
    assert!(manager.observations().iter().any(|observation| {
        observation.kind == ComponentObservationKind::Validation
            && observation.message.contains("component authority rejected")
            && observation
                .message
                .contains("authority kind is outside Magnetar inference scope")
    }));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_validation_observations_redact_paths_and_secrets() {
    let directory = temp_component_artifact_dir("redacted-diagnostic");
    let artifact = directory.join("hello.component.wasm");
    fs::write(&artifact, b"component-bytes").unwrap();
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture"),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ManifestMissing { .. })
    ));
    let messages = manager
        .observations()
        .iter()
        .map(|observation| observation.message.as_str())
        .collect::<Vec<_>>();
    assert!(
        messages
            .iter()
            .any(|message| message.contains("[redacted]"))
    );
    assert!(!messages.iter().any(|message| {
        message.contains(directory.to_string_lossy().as_ref())
            || message.to_ascii_lowercase().contains("secret")
    }));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_pipeline_requires_external_trust_policy_before_prepare() {
    let directory = temp_component_artifact_dir("trusted");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    let manifest = directory.join("hello.component.wasm.magnetar-component.yaml");
    fs::write(
        &manifest,
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();

    let import = WitInterface::new("magnetar:test/echo", "1.0.0");
    let export = WitInterface::new("magnetar:test/run", "1.0.0");
    let metadata = ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
        .with_import(import)
        .with_export(export);
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value.clone()));
    manager
        .register_component(ComponentDescriptor::new(metadata, &artifact))
        .unwrap();

    manager
        .prepare_component("magnetar.examples.hello")
        .unwrap();
    let definition = manager.definition("magnetar.examples.hello").unwrap();
    assert_eq!(definition.artifact_digest, Some(digest));
    assert!(matches!(
        definition.trust_decision,
        Some(ComponentTrustDecision {
            status: ComponentTrustStatus::Trusted,
            ..
        })
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_rejects_local_wasm_without_manifest() {
    let directory = temp_component_artifact_dir("missing-manifest");
    let artifact = directory.join("unknown.component.wasm");
    fs::write(&artifact, b"component-bytes").unwrap();
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("unknown", "0.1.0", "fixture"),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("unknown"),
        Err(ComponentError::ManifestMissing { .. })
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_rejects_digest_mismatch_before_prepare() {
    let directory = temp_component_artifact_dir("digest-mismatch");
    let artifact = directory.join("hello.component.wasm");
    fs::write(&artifact, b"component-bytes").unwrap();
    let digest = ComponentDigest::sha256(b"different-bytes");
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Rejected,
            ..
        })
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_rejects_manifest_wit_that_differs_from_actual_contract() {
    let directory = temp_component_artifact_dir("wit-mismatch");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();

    let mut engine = MockComponentEngine::new();
    let mut actual = ComponentContract::default();
    actual.imports.insert(ComponentImportRequirement::new(
        WitInterface::new("magnetar:test/other", "1.0.0"),
        ComponentInterfaceShape::Interface,
    ));
    engine.prepared_contract = Some(actual);
    let mut manager = ComponentManager::with_engine(Box::new(engine));
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture"),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ContractValidationFailed { .. })
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_trust_store_revocation_overrides_digest_allowlist() {
    let directory = temp_component_artifact_dir("revoked");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(
        ComponentTrustStore::default()
            .trust_digest(digest.value.clone())
            .revoke_digest(digest.value),
    );
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Revoked,
            ..
        })
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_trust_store_loads_minimal_yaml_format() {
    let directory = temp_component_artifact_dir("trust-store");
    let digest = ComponentDigest::sha256(b"component-bytes");
    let trust_store = directory.join("trust.yaml");
    fs::write(&trust_store, trust_store_yaml(&digest.value)).unwrap();

    let loaded = ComponentTrustStore::load_yaml(&trust_store).unwrap();
    assert!(loaded.trusted_digests.contains(&digest.value));
    assert!(!loaded.allow_unsigned_local_development);
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_manifest_may_declare_optional_wit_import_metadata() {
    let directory = temp_component_artifact_dir("optional-import");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    let manifest = manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION).replace(
        "  exports:",
        "    - package: \"magnetar:optional\"\n      interface: \"telemetry\"\n      version: \"1.0.0\"\n      optional: true\n  exports:",
    );
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest,
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    manager
        .prepare_component("magnetar.examples.hello")
        .unwrap();
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_rejects_runtime_max_version_incompatibility() {
    let directory = temp_component_artifact_dir("runtime-max");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    let manifest = manifest_yaml(&digest.value, "0.0.1").replace(
        "    min_version: \"0.0.1\"",
        "    min_version: \"0.0.1\"\n    max_version: \"0.0.1\"",
    );
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest,
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Rejected,
            ..
        })
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_rejects_incompatible_capability_versions() {
    for (label, capability_block) in [
        (
            "cap-major",
            "    - id: \"magnetar:test/echo\"\n      version: \"2.0.0\"",
        ),
        (
            "cap-range",
            "    - id: \"magnetar:test/echo\"\n      version: \"1.0.0\"\n      max_version: \"0.9.0\"",
        ),
    ] {
        let directory = temp_component_artifact_dir(label);
        let artifact = directory.join("hello.component.wasm");
        let bytes = b"component-bytes";
        fs::write(&artifact, bytes).unwrap();
        let digest = ComponentDigest::sha256(bytes);
        let manifest = manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION).replace(
            "    - id: \"magnetar:test/echo\"\n      version: \"1.0.0\"",
            capability_block,
        );
        fs::write(
            directory.join("hello.component.wasm.magnetar-component.yaml"),
            manifest,
        )
        .unwrap();
        let mut manager = ComponentManager::new();
        manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
        manager
            .register_component(ComponentDescriptor::new(
                ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                    .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                    .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
                &artifact,
            ))
            .unwrap();

        assert!(matches!(
            manager.prepare_component("magnetar.examples.hello"),
            Err(ComponentError::ArtifactRejected {
                status: ComponentTrustStatus::Rejected,
                ..
            })
        ));
        fs::remove_dir_all(directory).unwrap();
    }
}

#[test]
fn component_publisher_trust_is_only_policy_driven() {
    let directory = temp_component_artifact_dir("publisher-policy");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();
    let descriptor = || {
        ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        )
    };
    let mut untrusted = ComponentManager::new();
    untrusted.register_component(descriptor()).unwrap();
    assert!(matches!(
        untrusted.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Unknown,
            ..
        })
    ));

    let mut trusted = ComponentManager::new();
    trusted.set_trust_store(ComponentTrustStore::default().trust_publisher("local-dev"));
    trusted.register_component(descriptor()).unwrap();
    trusted
        .prepare_component("magnetar.examples.hello")
        .unwrap();
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_signature_metadata_is_recorded_but_not_trusted_by_itself() {
    let directory = temp_component_artifact_dir("signature");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    let signed_manifest = manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION).replace(
        "signatures: []",
        &format!(
            "signatures:\n  - algorithm: \"test\"\n    digest: \"{}\"",
            digest.value
        ),
    );
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        signed_manifest,
    )
    .unwrap();
    let descriptor = || {
        ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        )
    };
    let mut no_trust = ComponentManager::new();
    no_trust.register_component(descriptor()).unwrap();
    assert!(matches!(
        no_trust.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Unknown,
            ..
        })
    ));

    let mut trusted = ComponentManager::new();
    trusted.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value.clone()));
    trusted.register_component(descriptor()).unwrap();
    trusted
        .prepare_component("magnetar.examples.hello")
        .unwrap();

    let bad_manifest = manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION).replace(
        "signatures: []",
        "signatures:\n  - algorithm: \"test\"\n    digest: \"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"",
    );
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        bad_manifest,
    )
    .unwrap();
    let mut mismatch = ComponentManager::new();
    mismatch.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    mismatch.register_component(descriptor()).unwrap();
    assert!(matches!(
        mismatch.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Rejected,
            ..
        })
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_development_mode_is_explicit_and_still_validates_artifact() {
    let directory = temp_component_artifact_dir("development");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().allow_unsigned_local_development(true));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    manager
        .prepare_component("magnetar.examples.hello")
        .unwrap();
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_cache_is_digest_keyed_and_non_authoritative() {
    let bytes = b"component-bytes".to_vec();
    let mut cache = ComponentArtifactCache::default();
    let digest = cache.insert(bytes.clone());
    assert!(cache.contains_untrusted(&digest));
    assert_eq!(cache.get_verified(&digest).unwrap(), Some(bytes.as_slice()));

    let wrong_digest = ComponentDigest::sha256(b"wrong");
    cache.insert_unchecked_for_test(wrong_digest.clone(), bytes);
    assert!(matches!(
        cache.get_verified(&wrong_digest),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Rejected,
            ..
        })
    ));
}

#[test]
fn component_quarantine_prevents_preparation_and_preserves_diagnostic_status() {
    let directory = temp_component_artifact_dir("quarantine");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().quarantine_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Quarantined,
            ..
        })
    ));
    assert!(
        manager
            .observations()
            .iter()
            .any(|observation| observation.message.contains("Quarantined"))
    );
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_validation_emits_structured_observations() {
    let directory = temp_component_artifact_dir("observations");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    manager
        .prepare_component("magnetar.examples.hello")
        .unwrap();
    let messages = manager
        .observations()
        .iter()
        .map(|observation| observation.message.as_str())
        .collect::<Vec<_>>();
    assert!(
        messages
            .iter()
            .any(|message| message.contains("discovered"))
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("digest computed"))
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("manifest loaded"))
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("WIT declarations match"))
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("compatibility validated"))
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("trust decision"))
    );
    fs::remove_dir_all(directory).unwrap();
}
